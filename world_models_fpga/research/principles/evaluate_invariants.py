"""
Phase 5: Mathematical Invariance, Equivariance & Embodiment Perturbation Evaluator (Repaired)

Fixes:
  1. Eliminates all hardcoded scores (0.085, 0.742, 0.815, 0.785, 0.942).
  2. Separates strict mathematical Invariance (||f(g.x) - f(x)||) from Equivariance
     (||f(g.x) - T_g f(x)|| where T_g is fit on train pairs and evaluated on held-out pairs)
     across all 7 counterfactual interventions E1..E7.
  3. Executes genuine MuJoCo closed-loop rollouts for embodiment perturbations
     (floor_friction_scale = 0.4, action_magnitude_scaling = 0.7x, camera_pitch = +10 deg)
     and multi-horizon rollout evaluations across K in {1, 3, 5, 8}.
"""

from __future__ import annotations
import sys as _hsys
from pathlib import Path as _HPath
_hsys.path.insert(0, str(_HPath(__file__).resolve().parents[2]))
from research.datasets.dataset_config import SPLITS_DIR as _ACTIVE_SPLITS_DIR, CKPT_DIR as _ACTIVE_CKPT_DIR  # noqa: E402

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

from models.baseline_policy import BaselineNavigationPolicy
from models.h_jepa import HJEPAModel
from models.le_world_model import LeWorldModel
from research.benchmarks.evaluator import BenchmarkEvaluator
from research.benchmarks.model_wrappers import load_all_agents
from research.datasets.generation.capability_tasks import generate_task_specification


def _compute_invariance_and_equivariance(z0: np.ndarray, zi: np.ndarray) -> dict:
    """
    Given paired representations z0 (N, D) and zi (N, D):
      - Raw Invariance Score: 1 / (1 + E[||zi - z0||_2] / E[||z0 - mean(z0)||_2])
      - Cosine Invariance: E[cos(z0, zi)]
      - Equivariance R^2 (held-out pairs): fits orthogonal/linear map T_g on first half of pairs
        (z0_train @ T_g ≈ zi_train) and evaluates R^2 and cosine similarity on second half!
    """
    spread = float(np.mean(np.linalg.norm(z0 - np.mean(z0, axis=0, keepdims=True), axis=-1)) + 1e-8)
    raw_l2 = float(np.mean(np.linalg.norm(zi - z0, axis=-1)))
    norm_l2 = raw_l2 / spread
    invariance_score = float(1.0 / (1.0 + norm_l2))

    cos_inv = float(
        np.mean(
            np.sum(z0 * zi, axis=-1)
            / (np.linalg.norm(z0, axis=-1) * np.linalg.norm(zi, axis=-1) + 1e-8)
        )
    )

    n = len(z0)
    mid = max(2, n // 2)
    X_tr, Y_tr = z0[:mid], zi[:mid]
    X_te, Y_te = z0[mid:], zi[mid:]

    # Ridge-regularized linear equivariance operator T_g
    reg = 1.0 * np.eye(z0.shape[1], dtype=np.float32)
    T_g = np.linalg.solve(X_tr.T @ X_tr + reg, X_tr.T @ Y_tr)
    Y_pred = X_te @ T_g

    eq_err = float(np.mean(np.linalg.norm(Y_te - Y_pred, axis=-1)) / spread)
    equivariance_score = float(1.0 / (1.0 + eq_err))
    eq_cos = float(
        np.mean(
            np.sum(Y_te * Y_pred, axis=-1)
            / (np.linalg.norm(Y_te, axis=-1) * np.linalg.norm(Y_pred, axis=-1) + 1e-8)
        )
    )
    return {
        "invariance_score": invariance_score,
        "cosine_invariance": cos_inv,
        "normalized_l2_drift": norm_l2,
        "equivariance_score": equivariance_score,
        "equivariance_heldout_cosine": eq_cos,
    }


def evaluate_invariants():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v2_splits = str(_ACTIVE_SPLITS_DIR)
    legacy_splits = str(_ACTIVE_SPLITS_DIR)
    splits_dir = v2_splits if os.path.exists(os.path.join(v2_splits, "counterfactual_suite.npz")) else legacy_splits

    v2_ckpt = str(_ACTIVE_CKPT_DIR)
    legacy_ckpt = str(_ACTIVE_CKPT_DIR)
    ckpt_dir = v2_ckpt if os.path.exists(os.path.join(v2_ckpt, "lewm_best.pt")) else legacy_ckpt

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(out_dir, exist_ok=True)

    print("=================================================================")
    print("  PHASE 5: INVARIANCE, EQUIVARIANCE & EMBODIMENT EVALUATION")
    print("=================================================================")

    cf_d = np.load(os.path.join(splits_dir, "counterfactual_suite.npz"))
    cf_rgb = cf_d["rgb"]
    cf_actions = cf_d["actions"]
    with open(os.path.join(splits_dir, "counterfactual_suite_metadata.json")) as f:
        cf_meta = json.load(f)["episodes"]
    te_d = np.load(os.path.join(splits_dir, "held_out_test_benchmark.npz"))

    b_model = BaselineNavigationPolicy().to(device)
    b_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "baseline_best.pt"), map_location=device, weights_only=False), strict=False)
    b_model.eval()

    l_model = LeWorldModel().to(device)
    l_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "lewm_best.pt"), map_location=device, weights_only=False), strict=False)
    l_model.eval()

    h_model = HJEPAModel().to(device)
    h_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "hjepa_best.pt"), map_location=device, weights_only=False), strict=False)
    h_model.eval()

    models_dict = {"Baseline": b_model, "LeWM": l_model, "H-JEPA": h_model}

    # Group counterfactual pairs by experiment_type
    pairs_by_exp: dict[str, list[tuple[dict, dict]]] = {}
    for i in range(0, len(cf_meta) - 1, 2):
        ep0, epi = cf_meta[i], cf_meta[i + 1]
        exp_t = ep0.get("experiment_type") or ep0["episode_id"].split("_p")[0]
        pairs_by_exp.setdefault(exp_t, []).append((ep0, epi))

    invariant_results: dict = {
        m: {"transformation_invariance": {}, "embodiment_stability": {}, "horizon_generalization": {}}
        for m in models_dict
    }

    # 1. Transformation Invariance vs Equivariance across E1..E7
    for exp_name, pairs in pairs_by_exp.items():
        print(f"\n--- Evaluating Invariance vs Equivariance: {exp_name} ---")
        f0_list, fi_list, a0_list = [], [], []
        for ep0, epi in pairs:
            s0, si = int(ep0["start_index"]), int(epi["start_index"])
            L = min(int(ep0["length"]), int(epi["length"]), 8)
            for t in range(L):
                f0_list.append(cf_rgb[s0 + t])
                fi_list.append(cf_rgb[si + t])
                a0_list.append(cf_actions[s0 + t])

        img0 = torch.from_numpy(np.array(f0_list)).permute(0, 3, 1, 2).float().to(device) / 255.0
        imgi = torch.from_numpy(np.array(fi_list)).permute(0, 3, 1, 2).float().to(device) / 255.0
        act0 = torch.from_numpy(np.array(a0_list)).float().to(device)

        with torch.no_grad():
            for m_name, mdl in models_dict.items():
                z0 = (mdl.encoder(img0) if m_name == "Baseline" else mdl.encode(img0)).cpu().numpy()
                zi = (mdl.encoder(imgi) if m_name == "Baseline" else mdl.encode(imgi)).cpu().numpy()
                metrics = _compute_invariance_and_equivariance(z0, zi)

                if m_name == "LeWM":
                    ns0, _ = mdl.predict_next(torch.from_numpy(z0).to(device), act0)
                    nsi, _ = mdl.predict_next(torch.from_numpy(zi).to(device), act0)
                    dyn_m = _compute_invariance_and_equivariance(ns0.cpu().numpy() - z0, nsi.cpu().numpy() - zi)
                    metrics["dynamics_residual_invariance"] = dyn_m["invariance_score"]
                    metrics["dynamics_residual_equivariance"] = dyn_m["equivariance_score"]
                elif m_name == "H-JEPA":
                    S0, _ = mdl.macro_predictor(torch.from_numpy(z0).to(device), act0)
                    Si, _ = mdl.macro_predictor(torch.from_numpy(zi).to(device), act0)
                    dyn_m = _compute_invariance_and_equivariance(S0.cpu().numpy() - z0, Si.cpu().numpy() - zi)
                    metrics["macro_subgoal_invariance"] = dyn_m["invariance_score"]
                    metrics["macro_subgoal_equivariance"] = dyn_m["equivariance_score"]

                invariant_results[m_name]["transformation_invariance"][exp_name] = metrics
                print(
                    f"  [{m_name:8s}] Inv: {metrics['invariance_score']:.3f} (Cos: {metrics['cosine_invariance']:.3f}) | "
                    f"Equiv: {metrics['equivariance_score']:.3f} (HeldoutCos: {metrics['equivariance_heldout_cosine']:.3f})"
                )

    # 2. Genuine Closed-Loop MuJoCo Embodiment Perturbations
    print("\n--- Evaluating Closed-Loop Embodiment Perturbations in MuJoCo ---")
    evaluator = BenchmarkEvaluator()
    agents = load_all_agents(ckpt_dir=ckpt_dir, device=device, track="structured")
    base_specs = [generate_task_specification("spatial_reasoning", "easy", seed_idx=6200 + i) for i in range(10)]

    for m_name in ("Baseline", "LeWM", "H-JEPA"):
        ag = agents[m_name]
        # Baseline nominal friction (1.0)
        nom_recs = [evaluator.evaluate_episode(ag, sp) for sp in base_specs]
        nom_succ = float(np.mean([1.0 if r["success"] else 0.0 for r in nom_recs]))

        # Perturbation 1: Low floor friction (floor_friction_scale = 0.4)
        fric_specs = []
        for sp in base_specs:
            sp_f = dict(sp)
            sp_f["floor_friction_scale"] = 0.4
            fric_specs.append(sp_f)
        fric_recs = [evaluator.evaluate_episode(ag, sp) for sp in fric_specs]
        fric_succ = float(np.mean([1.0 if r["success"] else 0.0 for r in fric_recs]))

        # Perturbation 2: Camera pitch offset (+10 deg)
        cam_specs = []
        for sp in base_specs:
            sp_c = dict(sp)
            sp_c["camera_pitch_offset"] = 10.0
            cam_specs.append(sp_c)
        cam_recs = [evaluator.evaluate_episode(ag, sp) for sp in cam_specs]
        cam_succ = float(np.mean([1.0 if r["success"] else 0.0 for r in cam_recs]))

        # Perturbation 3: Action gain scaling (0.7x)
        orig_act = ag.act

        def _scaled_act(*args, **kwargs):
            a, info = orig_act(*args, **kwargs)
            return (a * 0.7).astype(np.float32), info

        ag.act = _scaled_act
        try:
            scale_recs = [evaluator.evaluate_episode(ag, sp) for sp in base_specs]
        finally:
            ag.act = orig_act
        scale_succ = float(np.mean([1.0 if r["success"] else 0.0 for r in scale_recs]))

        invariant_results[m_name]["embodiment_stability"] = {
            "nominal_success_rate": nom_succ,
            "camera_pitch_10deg": cam_succ,
            "action_magnitude_scaling_0.7x": scale_succ,
            "surface_friction_perturbation": fric_succ,
        }
        print(
            f"  [{m_name:8s}] Nominal: {nom_succ*100:5.1f}% | Friction(0.4x): {fric_succ*100:5.1f}% | "
            f"ActionScale(0.7x): {scale_succ*100:5.1f}% | CamPitch(+10d): {cam_succ*100:5.1f}%"
        )

    evaluator.close()

    # 3. Multi-Horizon Rollout Stability across K in {1, 3, 5, 8} on held_out_test_benchmark
    print("\n--- Evaluating Multi-Horizon Latent Rollout Stability (K = 1, 3, 5, 8) ---")
    ep_starts = np.flatnonzero(te_d["episode_starts"])
    ep_ends = list(ep_starts[1:]) + [len(te_d["rgb"])]

    for H in (1, 3, 5, 8):
        valid_t = []
        for s_idx, e_idx in zip(ep_starts, ep_ends):
            for t in range(int(s_idx), int(e_idx) - H):
                valid_t.append(t)
        valid_t = valid_t[:400]

        img_0 = torch.from_numpy(te_d["rgb"][valid_t]).permute(0, 3, 1, 2).float().to(device) / 255.0
        img_H = torch.from_numpy(te_d["rgb"][[t + H for t in valid_t]]).permute(0, 3, 1, 2).float().to(device) / 255.0
        act_slices = np.asarray([te_d["actions"][t : t + H] for t in valid_t], dtype=np.float32)

        with torch.no_grad():
            # Baseline has no latent transition predictor; measure static latent similarity E[cos(z_t, z_{t+H})]
            z0_b = b_model.encoder(img_0)
            zH_b = b_model.encoder(img_H)
            cos_b = float(F.cosine_similarity(z0_b, zH_b, dim=-1).mean().item())
            invariant_results["Baseline"]["horizon_generalization"][f"horizon_K{H}"] = cos_b

            # LeWM autoregressive H-step rollout
            s_l = l_model.encode(img_0)
            sH_l = l_model.encode(img_H)
            for step_i in range(H):
                a_step = torch.from_numpy(act_slices[:, step_i, :]).float().to(device)
                s_l, _ = l_model.predict_next(s_l, a_step)
            cos_l = float(F.cosine_similarity(s_l, sH_l, dim=-1).mean().item())
            invariant_results["LeWM"]["horizon_generalization"][f"horizon_K{H}"] = cos_l

            # H-JEPA: uses MacroPredictor for K >= 5 (plus micro steps for remainder)
            s_h = h_model.encode(img_0)
            sH_h = h_model.encode(img_H)
            rem = H
            cur_step = 0
            while rem >= h_model.stride_k:
                mac_a = torch.from_numpy(act_slices[:, cur_step : cur_step + h_model.stride_k, :].reshape(len(valid_t), -1)).float().to(device)
                s_h, _ = h_model.macro_predictor(s_h, mac_a)
                cur_step += h_model.stride_k
                rem -= h_model.stride_k
            while rem > 0:
                a_step = torch.from_numpy(act_slices[:, cur_step, :]).float().to(device)
                s_h, _ = h_model.micro_predictor.forward_transition(s_h, a_step)
                cur_step += 1
                rem -= 1
            cos_h = float(F.cosine_similarity(s_h, sH_h, dim=-1).mean().item())
            invariant_results["H-JEPA"]["horizon_generalization"][f"horizon_K{H}"] = cos_h

        print(f"  [Horizon K={H}] Baseline (static): {cos_b:.4f} | LeWM (rollout): {cos_l:.4f} | H-JEPA (hier): {cos_h:.4f}")

    out_path = os.path.join(out_dir, "invariant_evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(invariant_results, f, indent=2)
    print(f"\n[SAVED INVARIANT EVALUATION RESULTS TO {out_path}]")
    return invariant_results


if __name__ == "__main__":
    evaluate_invariants()
