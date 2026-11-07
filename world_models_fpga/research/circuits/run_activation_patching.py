"""
Phase 4: Causal Activation Patching Engine (Repaired)

Fixes:
  1. Looks up counterfactual pairs by `experiment_type` in `counterfactual_suite_metadata.json`
     (fixing the legacy indexing bug that mapped E5_obstacle_insertion to E3 episodes 100..149).
  2. Separates full-latent upper-bound sanity checks from genuine k=8 targeted subspace
     interventions, and compares every targeted k=8 subspace patch against a matched k=8
     random subspace control with 95% confidence intervals.
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


def _ci95(arr: np.ndarray) -> float:
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)))


def compute_patching_effect(out_clean: np.ndarray, out_corrupted: np.ndarray, out_patched: np.ndarray) -> tuple[float, float]:
    diff_total = np.linalg.norm(out_clean - out_corrupted, axis=-1)
    diff_rem = np.linalg.norm(out_clean - out_patched, axis=-1)
    valid = diff_total > 1e-4
    if np.sum(valid) == 0:
        return 0.0, 0.0
    effects = np.clip(1.0 - (diff_rem[valid] / (diff_total[valid] + 1e-8)), 0.0, 1.0)
    return float(np.mean(effects)), _ci95(effects)


def _find_top_k_dims(s_donor: torch.Tensor, s_rec: torch.Tensor, k: int = 8) -> torch.Tensor:
    mean_diff = torch.mean(torch.abs(s_donor - s_rec), dim=0)
    return torch.topk(mean_diff, k=k).indices


def run_activation_patching():
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
    print("  PHASE 4: CAUSAL ACTIVATION PATCHING EXPERIMENTS (REPAIRED)")
    print(f"  Splits: {splits_dir} | Checkpoints: {ckpt_dir}")
    print("=================================================================")

    cf_d = np.load(os.path.join(splits_dir, "counterfactual_suite.npz"))
    cf_rgb = cf_d["rgb"]
    cf_actions = cf_d["actions"]
    cf_pose = cf_d["robot_pose"]
    cf_goals = cf_d["grounded_goals"]
    with open(os.path.join(splits_dir, "counterfactual_suite_metadata.json")) as f:
        cf_meta = json.load(f)["episodes"]

    b_model = BaselineNavigationPolicy().to(device)
    b_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "baseline_best.pt"), map_location=device, weights_only=False), strict=False)
    b_model.eval()

    l_model = LeWorldModel().to(device)
    l_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "lewm_best.pt"), map_location=device, weights_only=False), strict=False)
    l_model.eval()

    h_model = HJEPAModel().to(device)
    h_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "hjepa_best.pt"), map_location=device, weights_only=False), strict=False)
    h_model.eval()

    # Group matched (E0, Ei) episode pairs by experiment_type from metadata
    pairs_by_exp: dict[str, list[tuple[dict, dict]]] = {}
    for i in range(0, len(cf_meta) - 1, 2):
        ep0, epi = cf_meta[i], cf_meta[i + 1]
        exp_t = ep0.get("experiment_type") or ep0["episode_id"].split("_p")[0]
        pairs_by_exp.setdefault(exp_t, []).append((ep0, epi))

    patching_configs = [
        ("E1_position_shift", "target_localization_circuit"),
        ("E2_color_shift", "appearance_invariance_circuit"),
        ("E3_robot_rotation", "egocentric_frame_transform_circuit"),
        ("E5_intermediate_obstacle", "obstacle_collision_avoidance_circuit"),
        ("E6_shape_mutation", "shape_invariance_circuit"),
        ("E7_camera_viewpoint", "viewpoint_pitch_calibration_circuit"),
    ]

    patching_results: dict = {
        "Baseline": {},
        "LeWM": {},
        "H-JEPA": {},
    }

    rng = np.random.default_rng(42)

    for exp_name, circuit_label in patching_configs:
        print(f"\n--- Patching Intervention: {exp_name} ({circuit_label}) ---", flush=True)
        pairs = pairs_by_exp.get(exp_name, [])
        donor_frames, rec_frames, donor_goals, rec_goals, donor_acts = [], [], [], [], []

        for ep0, epi in pairs:
            s0, l0 = int(ep0["start_index"]), int(ep0["length"])
            si, li = int(epi["start_index"]), int(epi["length"])
            L = min(l0, li, 12)
            for t in range(L):
                i0, ii = s0 + t, si + t
                donor_frames.append(cf_rgb[i0])
                rec_frames.append(cf_rgb[ii])
                donor_acts.append(cf_actions[i0])

                for pose, goal, dst_list in (
                    (cf_pose[i0], cf_goals[i0], donor_goals),
                    (cf_pose[ii], cf_goals[ii], rec_goals),
                ):
                    dx, dy = float(goal[0] - pose[0]), float(goal[1] - pose[1])
                    th = float(pose[2])
                    c, s = np.cos(th), np.sin(th)
                    dth = float(goal[2] - pose[2])
                    dst_list.append(np.array([c * dx + s * dy, -s * dx + c * dy, np.sin(dth), np.cos(dth)], dtype=np.float32))

        img_donor = torch.from_numpy(np.array(donor_frames)).permute(0, 3, 1, 2).float().to(device) / 255.0
        img_rec = torch.from_numpy(np.array(rec_frames)).permute(0, 3, 1, 2).float().to(device) / 255.0
        g_donor = torch.from_numpy(np.array(donor_goals)).float().to(device)
        g_rec = torch.from_numpy(np.array(rec_goals)).float().to(device)
        a_donor = torch.from_numpy(np.array(donor_acts)).float().to(device)

        with torch.no_grad():
            # 1. Baseline Policy
            z_d_b = b_model.encoder(img_donor)
            z_r_b = b_model.encoder(img_rec)
            act_d_b, _ = b_model.forward_with_latent(z_d_b, g_donor)
            act_r_b, _ = b_model.forward_with_latent(z_r_b, g_rec)

            top8_b = _find_top_k_dims(z_d_b, z_r_b, k=8)
            rand8_b = torch.tensor(rng.choice(32, size=8, replace=False), dtype=torch.long, device=device)

            z_patch_top8_b = z_r_b.clone()
            z_patch_top8_b[:, top8_b] = z_d_b[:, top8_b]
            act_p_top8_b, _ = b_model.forward_with_latent(z_patch_top8_b, g_rec)

            z_patch_rand8_b = z_r_b.clone()
            z_patch_rand8_b[:, rand8_b] = z_d_b[:, rand8_b]
            act_p_rand8_b, _ = b_model.forward_with_latent(z_patch_rand8_b, g_rec)

            act_p_goal_b, _ = b_model.forward_with_latent(z_r_b, g_donor)

            eff_top8_b, ci_top8_b = compute_patching_effect(act_d_b.cpu().numpy(), act_r_b.cpu().numpy(), act_p_top8_b.cpu().numpy())
            eff_rand8_b, ci_rand8_b = compute_patching_effect(act_d_b.cpu().numpy(), act_r_b.cpu().numpy(), act_p_rand8_b.cpu().numpy())
            eff_goal_b, ci_goal_b = compute_patching_effect(act_d_b.cpu().numpy(), act_r_b.cpu().numpy(), act_p_goal_b.cpu().numpy())

            patching_results["Baseline"][exp_name] = {
                "target_circuit": circuit_label,
                "top8_visual_subspace_effect": eff_top8_b,
                "top8_visual_subspace_ci95": ci_top8_b,
                "random8_subspace_control_effect": eff_rand8_b,
                "random8_subspace_control_ci95": ci_rand8_b,
                "subspace_selectivity_gain": eff_top8_b - eff_rand8_b,
                "goal_conditioning_patch_effect": eff_goal_b,
                "goal_conditioning_patch_ci95": ci_goal_b,
                "top8_dimensions": top8_b.cpu().tolist(),
            }

            # 2. LeWorldModel (LeWM)
            s_d_l = l_model.encode(img_donor)
            s_r_l = l_model.encode(img_rec)
            ns_d_l, col_d_l = l_model.predict_next(s_d_l, a_donor)
            ns_r_l, col_r_l = l_model.predict_next(s_r_l, a_donor)
            disp_d_l = l_model.predict_ego_displacement(s_d_l, ns_d_l)
            disp_r_l = l_model.predict_ego_displacement(s_r_l, ns_r_l)

            top8_l = _find_top_k_dims(s_d_l, s_r_l, k=8)
            rand8_l = torch.tensor(rng.choice(32, size=8, replace=False), dtype=torch.long, device=device)

            s_patch_top8_l = s_r_l.clone()
            s_patch_top8_l[:, top8_l] = s_d_l[:, top8_l]
            ns_p_top8_l, col_p_top8_l = l_model.predict_next(s_patch_top8_l, a_donor)
            disp_p_top8_l = l_model.predict_ego_displacement(s_patch_top8_l, ns_p_top8_l)

            s_patch_rand8_l = s_r_l.clone()
            s_patch_rand8_l[:, rand8_l] = s_d_l[:, rand8_l]
            ns_p_rand8_l, col_p_rand8_l = l_model.predict_next(s_patch_rand8_l, a_donor)
            disp_p_rand8_l = l_model.predict_ego_displacement(s_patch_rand8_l, ns_p_rand8_l)

            eff_top8_l, ci_top8_l = compute_patching_effect(ns_d_l.cpu().numpy(), ns_r_l.cpu().numpy(), ns_p_top8_l.cpu().numpy())
            eff_rand8_l, ci_rand8_l = compute_patching_effect(ns_d_l.cpu().numpy(), ns_r_l.cpu().numpy(), ns_p_rand8_l.cpu().numpy())
            eff_disp_l, _ = compute_patching_effect(disp_d_l.cpu().numpy(), disp_r_l.cpu().numpy(), disp_p_top8_l.cpu().numpy())
            eff_col_l, ci_col_l = compute_patching_effect(
                torch.sigmoid(col_d_l).cpu().numpy(),
                torch.sigmoid(col_r_l).cpu().numpy(),
                torch.sigmoid(col_p_top8_l).cpu().numpy(),
            )
            eff_col_rand_l, _ = compute_patching_effect(
                torch.sigmoid(col_d_l).cpu().numpy(),
                torch.sigmoid(col_r_l).cpu().numpy(),
                torch.sigmoid(col_p_rand8_l).cpu().numpy(),
            )

            patching_results["LeWM"][exp_name] = {
                "target_circuit": circuit_label,
                "top8_subspace_effect": eff_top8_l,
                "top8_subspace_ci95": ci_top8_l,
                "random8_control_effect": eff_rand8_l,
                "random8_control_ci95": ci_rand8_l,
                "subspace_selectivity_gain": eff_top8_l - eff_rand8_l,
                "ego_displacement_patch_effect": eff_disp_l,
                "collision_head_top8_patch_effect": eff_col_l,
                "collision_head_top8_ci95": ci_col_l,
                "collision_head_random8_control": eff_col_rand_l,
                "top8_dimensions": top8_l.cpu().tolist(),
            }

            # 3. H-JEPA (Hierarchical Macro + Micro)
            s_d_h = h_model.encode(img_donor)
            s_r_h = h_model.encode(img_rec)
            S_d_macro, col_d_m = h_model.macro_predictor(s_d_h, a_donor)
            S_r_macro, col_r_m = h_model.macro_predictor(s_r_h, a_donor)
            act_d_h = h_model.micro_predictor.select_action(s_d_h, S_d_macro)
            act_r_h = h_model.micro_predictor.select_action(s_r_h, S_r_macro)

            top8_h = _find_top_k_dims(s_d_h, s_r_h, k=8)
            rand8_h = torch.tensor(rng.choice(32, size=8, replace=False), dtype=torch.long, device=device)

            # Patch top-8 dimensions of macro subgoal vs random-8 control
            S_patch_top8 = S_r_macro.clone()
            S_patch_top8[:, top8_h] = S_d_macro[:, top8_h]
            act_p_macro_top8 = h_model.micro_predictor.select_action(s_r_h, S_patch_top8)

            S_patch_rand8 = S_r_macro.clone()
            S_patch_rand8[:, rand8_h] = S_d_macro[:, rand8_h]
            act_p_macro_rand8 = h_model.micro_predictor.select_action(s_r_h, S_patch_rand8)

            # Full macro subgoal patch (keeping micro state s_r_h unpatched) vs Full micro state patch (keeping macro subgoal S_r_macro unpatched)
            act_p_macro_only = h_model.micro_predictor.select_action(s_r_h, S_d_macro)
            act_p_micro_only = h_model.micro_predictor.select_action(s_d_h, S_r_macro)

            eff_macro_top8, ci_macro_top8 = compute_patching_effect(act_d_h.cpu().numpy(), act_r_h.cpu().numpy(), act_p_macro_top8.cpu().numpy())
            eff_macro_rand8, ci_macro_rand8 = compute_patching_effect(act_d_h.cpu().numpy(), act_r_h.cpu().numpy(), act_p_macro_rand8.cpu().numpy())
            eff_macro_branch, ci_macro_branch = compute_patching_effect(act_d_h.cpu().numpy(), act_r_h.cpu().numpy(), act_p_macro_only.cpu().numpy())
            eff_micro_branch, ci_micro_branch = compute_patching_effect(act_d_h.cpu().numpy(), act_r_h.cpu().numpy(), act_p_micro_only.cpu().numpy())

            patching_results["H-JEPA"][exp_name] = {
                "target_circuit": circuit_label,
                "macro_top8_subspace_effect": eff_macro_top8,
                "macro_top8_subspace_ci95": ci_macro_top8,
                "macro_random8_control_effect": eff_macro_rand8,
                "macro_random8_control_ci95": ci_macro_rand8,
                "subspace_selectivity_gain": eff_macro_top8 - eff_macro_rand8,
                "macro_subgoal_branch_patch_effect": eff_macro_branch,
                "macro_subgoal_branch_ci95": ci_macro_branch,
                "micro_state_branch_patch_effect": eff_micro_branch,
                "micro_state_branch_ci95": ci_micro_branch,
                "top8_dimensions": top8_h.cpu().tolist(),
            }

            print(
                f"  [Baseline] Top-8: {eff_top8_b:.3f} (+-{ci_top8_b:.3f}) vs Rand-8: {eff_rand8_b:.3f} | GoalPatch: {eff_goal_b:.3f}",
                flush=True
            )
            print(
                f"  [LeWM]     Top-8: {eff_top8_l:.3f} (+-{ci_top8_l:.3f}) vs Rand-8: {eff_rand8_l:.3f} | ColPatch: {eff_col_l:.3f}",
                flush=True
            )
            print(
                f"  [H-JEPA]   MacroTop-8: {eff_macro_top8:.3f} vs Rand-8: {eff_macro_rand8:.3f} | "
                f"MacroBranch: {eff_macro_branch:.3f} | MicroBranch: {eff_micro_branch:.3f}",
                flush=True
            )

    out_path = os.path.join(out_dir, "activation_patching_results.json")
    with open(out_path, "w") as f:
        json.dump(patching_results, f, indent=2)
    print(f"\n[SAVED ACTIVATION PATCHING RESULTS TO {out_path}]")
    return patching_results


if __name__ == "__main__":
    run_activation_patching()
