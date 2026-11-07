"""
Phase 4: Causal Subspace & SAE Feature Ablation Engine (Repaired)

Executes both:
  1. Open-loop latent/action/collision ablations on held-out test transitions
     (comparing targeted top-8 subspaces against matched 8-dim random subspace controls).
  2. Closed-loop MuJoCo behavioral ablations (via forward hooks on the visual encoder during
     closed-loop navigation episodes, measuring task success drop and collision increase).
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

from models.baseline_policy import BaselineNavigationPolicy
from models.h_jepa import HJEPAModel
from models.le_world_model import LeWorldModel
from research.benchmarks.evaluator import BenchmarkEvaluator
from research.benchmarks.model_wrappers import BaselineAgent, HJEPAAgent, LeWMAgent
from research.datasets.generation.capability_tasks import generate_task_specification
from research.representations.sae_model import SparseAutoencoder


def _select_correlated_dims(latents: np.ndarray, target_vec: np.ndarray, k: int = 8) -> np.ndarray:
    if target_vec.ndim > 1:
        target_vec = np.linalg.norm(target_vec, axis=-1)
    t_std = np.std(target_vec) + 1e-8
    t_c = (target_vec - np.mean(target_vec)) / t_std
    corrs = np.zeros(latents.shape[1], dtype=np.float32)
    for d in range(latents.shape[1]):
        l_d = latents[:, d]
        corrs[d] = abs(float(np.mean(((l_d - np.mean(l_d)) / (np.std(l_d) + 1e-8)) * t_c)))
    return np.argsort(-corrs)[:k]


def _run_closed_loop_with_latent_mask(evaluator: BenchmarkEvaluator, agent, specs: list[dict], dims_to_ablate: np.ndarray, mean_vec: torch.Tensor) -> dict:
    """Installs a temporary forward hook on agent.model.encoder that mean-ablates `dims_to_ablate`."""
    idx_tensor = torch.as_tensor(dims_to_ablate, dtype=torch.long, device=agent.device)
    mean_dev = mean_vec.to(agent.device)

    def _hook(_module, _inp, out):
        patched = out.clone()
        patched[:, idx_tensor] = mean_dev[idx_tensor]
        return patched

    handle = agent.model.encoder.register_forward_hook(_hook)
    try:
        records = [evaluator.evaluate_episode(agent, sp) for sp in specs]
    finally:
        handle.remove()

    return {
        "success_rate": float(np.mean([1.0 if r["success"] else 0.0 for r in records])),
        "collision_rate": float(np.mean([0.0 if r["collision_free"] else 1.0 for r in records])),
        "mean_goal_error_m": float(np.mean([r["final_goal_error"] for r in records])),
    }


def run_ablation_experiments():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v2_splits = str(_ACTIVE_SPLITS_DIR)
    legacy_splits = str(_ACTIVE_SPLITS_DIR)
    splits_dir = v2_splits if os.path.exists(os.path.join(v2_splits, "held_out_test_benchmark.npz")) else legacy_splits

    v2_ckpt = str(_ACTIVE_CKPT_DIR)
    legacy_ckpt = str(_ACTIVE_CKPT_DIR)
    ckpt_dir = v2_ckpt if os.path.exists(os.path.join(v2_ckpt, "lewm_best.pt")) else legacy_ckpt

    sae_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../representations/results"))
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(out_dir, exist_ok=True)

    print("=================================================================")
    print("  PHASE 4: CAUSAL SUBSPACE & CLOSED-LOOP ABLATION EXPERIMENTS")
    print("=================================================================")

    te_d = np.load(os.path.join(splits_dir, "held_out_test_benchmark.npz"))
    N = min(1200, len(te_d["rgb"]))

    imgs = torch.from_numpy(te_d["rgb"][:N]).permute(0, 3, 1, 2).float().to(device) / 255.0
    poses = te_d["robot_pose"][:N]
    goals = te_d["grounded_goals"][:N]
    acts_gt = torch.from_numpy(te_d["actions"][:N]).float().to(device)
    cols_gt = te_d["collisions"][:N].astype(np.float32)

    dx = goals[:, 0] - poses[:, 0]
    dy = goals[:, 1] - poses[:, 1]
    th = poses[:, 2]
    c, s = np.cos(th), np.sin(th)
    dth = goals[:, 2] - poses[:, 2]
    g_vecs = torch.from_numpy(np.stack([c * dx + s * dy, -s * dx + c * dy, np.sin(dth), np.cos(dth)], axis=-1).astype(np.float32)).to(device)

    b_ckpt = os.path.join(ckpt_dir, "baseline_best.pt")
    l_ckpt = os.path.join(ckpt_dir, "lewm_best.pt")
    h_ckpt = os.path.join(ckpt_dir, "hjepa_best.pt")

    b_model = BaselineNavigationPolicy().to(device)
    b_model.load_state_dict(torch.load(b_ckpt, map_location=device, weights_only=False), strict=False)
    b_model.eval()

    l_model = LeWorldModel().to(device)
    l_model.load_state_dict(torch.load(l_ckpt, map_location=device, weights_only=False), strict=False)
    l_model.eval()

    h_model = HJEPAModel().to(device)
    h_model.load_state_dict(torch.load(h_ckpt, map_location=device, weights_only=False), strict=False)
    h_model.eval()

    models_dict = {
        "Baseline": b_model,
        "LeWM": l_model,
        "H-JEPA": h_model,
    }
    agent_wrappers = {
        "Baseline": BaselineAgent(b_ckpt, device=device, track="structured"),
        "LeWM": LeWMAgent(l_ckpt, device=device, track="structured"),
        "H-JEPA": HJEPAAgent(h_ckpt, device=device, track="structured"),
    }

    closed_loop_specs = (
        [generate_task_specification("obstacle_avoidance", "difficult", seed_idx=5200 + i) for i in range(6)]
        + [generate_task_specification("spatial_reasoning", "difficult", seed_idx=5300 + i) for i in range(6)]
    )
    evaluator = BenchmarkEvaluator()
    rng = np.random.default_rng(42)

    ablation_results: dict = {}

    for m_name, mdl in models_dict.items():
        print(f"\n--- Ablating Model: {m_name} ---")
        with torch.no_grad():
            s_clean = mdl.encoder(imgs) if m_name == "Baseline" else mdl.encode(imgs)
            s_np = s_clean.cpu().numpy()
            s_mean = torch.mean(s_clean, dim=0)

            # Identify top-8 correlated dimensions for each functional variable
            nav_dims = _select_correlated_dims(s_np, g_vecs[:, :2].cpu().numpy(), k=8)
            frame_dims = _select_correlated_dims(s_np, np.stack([np.sin(th), np.cos(th)], axis=-1), k=8)
            col_dims = _select_correlated_dims(s_np, cols_gt, k=8)
            rand_dims = rng.choice(32, size=8, replace=False)

            def eval_open_loop(s_mod):
                if m_name == "Baseline":
                    act_p, col_p = mdl.forward_with_latent(s_mod, g_vecs)
                    return act_p, col_p.squeeze(-1), s_mod
                elif m_name == "LeWM":
                    ns, col_log = mdl.predict_next(s_mod, acts_gt)
                    act_p = mdl.predict_reactive_action(s_mod, g_vecs)
                    return act_p, torch.sigmoid(col_log).squeeze(-1), ns
                else:
                    S_mac, col_log = mdl.macro_predictor(s_mod, acts_gt)
                    act_p = mdl.micro_predictor.select_action(s_mod, S_mac)
                    return act_p, torch.sigmoid(col_log).squeeze(-1), S_mac

            act_0, col_0, ns_0 = eval_open_loop(s_clean)

            subspaces = {
                "goal_navigation_subspace": nav_dims,
                "egocentric_frame_subspace": frame_dims,
                "collision_avoidance_subspace": col_dims,
                "random_control_subspace": rand_dims,
            }

            m_abl = {"open_loop_subspace_ablations": {}, "closed_loop_behavioral_ablations": {}, "sae_top_feature_ablations": {}}

            for sub_name, dims in subspaces.items():
                s_zero = s_clean.clone()
                s_zero[:, dims] = 0.0
                act_z, col_z, ns_z = eval_open_loop(s_zero)

                s_m = s_clean.clone()
                s_m[:, dims] = s_mean[dims]
                act_m, col_m, ns_m = eval_open_loop(s_m)

                act_mse_zero = float(torch.mean((act_z - act_0) ** 2).item())
                act_mse_mean = float(torch.mean((act_m - act_0) ** 2).item())
                col_shift = float(torch.mean(torch.abs(col_z - col_0)).item())
                lat_shift = float(torch.mean((ns_z - ns_0) ** 2).item())

                m_abl["open_loop_subspace_ablations"][sub_name] = {
                    "dimensions": dims.tolist(),
                    "zero_ablation_action_mse_shift": act_mse_zero,
                    "mean_ablation_action_mse_shift": act_mse_mean,
                    "collision_prob_shift": col_shift,
                    "next_latent_mse_shift": lat_shift,
                }
                print(
                    f"  [Open-Loop   | {sub_name:28s}] ActMSE: {act_mse_mean:.4f} | "
                    f"ColShift: {col_shift:.4f} | LatShift: {lat_shift:.4f}"
                )

        # Closed-loop MuJoCo behavioral ablations (intact vs targeted subspace vs random-8 control)
        ag = agent_wrappers[m_name]
        intact_recs = [evaluator.evaluate_episode(ag, sp) for sp in closed_loop_specs]
        intact_cl = {
            "success_rate": float(np.mean([1.0 if r["success"] else 0.0 for r in intact_recs])),
            "collision_rate": float(np.mean([0.0 if r["collision_free"] else 1.0 for r in intact_recs])),
            "mean_goal_error_m": float(np.mean([r["final_goal_error"] for r in intact_recs])),
        }
        m_abl["closed_loop_behavioral_ablations"]["intact_baseline"] = intact_cl

        for sub_name in ("goal_navigation_subspace", "collision_avoidance_subspace", "random_control_subspace"):
            dims = subspaces[sub_name]
            cl_res = _run_closed_loop_with_latent_mask(evaluator, ag, closed_loop_specs, dims, s_mean)
            cl_res["success_rate_drop"] = intact_cl["success_rate"] - cl_res["success_rate"]
            cl_res["collision_rate_increase"] = cl_res["collision_rate"] - intact_cl["collision_rate"]
            m_abl["closed_loop_behavioral_ablations"][sub_name] = cl_res
            print(
                f"  [Closed-Loop | {sub_name:28s}] Success: {cl_res['success_rate']*100:5.1f}% "
                f"(drop={cl_res['success_rate_drop']*100:+5.1f}%) | Col: {cl_res['collision_rate']*100:5.1f}%"
            )

        # SAE Feature Clamping (if SAE checkpoint exists)
        sae_path = os.path.join(sae_dir, f"sae_{m_name}.pt")
        if os.path.exists(sae_path):
            with torch.no_grad():
                sae = SparseAutoencoder(act_dim=32, dict_dim=128).to(device)
                sae.load_state_dict(torch.load(sae_path, map_location=device, weights_only=False))
                sae.eval()
                _, f_acts = sae(s_clean)
                top_feats = torch.argsort(torch.var(f_acts, dim=0), descending=True)[:8]
                f_abl = f_acts.clone()
                f_abl[:, top_feats] = 0.0
                s_sae_abl = sae.W_dec(f_abl) + sae.b_dec
                act_sae, col_sae, ns_sae = eval_open_loop(s_sae_abl)

                f_rand = f_acts.clone()
                rand_feats = torch.tensor(rng.choice(128, size=8, replace=False), dtype=torch.long, device=device)
                f_rand[:, rand_feats] = 0.0
                s_sae_rand = sae.W_dec(f_rand) + sae.b_dec
                act_sae_r, col_sae_r, ns_sae_r = eval_open_loop(s_sae_rand)

                m_abl["sae_top_feature_ablations"] = {
                    "ablated_feature_ids": top_feats.cpu().tolist(),
                    "action_mse_shift": float(torch.mean((act_sae - act_0) ** 2).item()),
                    "random8_feature_control_action_mse_shift": float(torch.mean((act_sae_r - act_0) ** 2).item()),
                    "collision_prob_shift": float(torch.mean(torch.abs(col_sae - col_0)).item()),
                    "next_latent_mse_shift": float(torch.mean((ns_sae - ns_0) ** 2).item()),
                }

        ablation_results[m_name] = m_abl

    evaluator.close()

    out_path = os.path.join(out_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\n[SAVED ABLATION RESULTS TO {out_path}]")
    return ablation_results


if __name__ == "__main__":
    run_ablation_experiments()
