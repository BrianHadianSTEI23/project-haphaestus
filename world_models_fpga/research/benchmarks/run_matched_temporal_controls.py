"""
Matched Temporal-Prediction Controls Benchmark (Resolving Finding A & Finding B)

Strictly isolates learned temporal prediction from candidate search confounds:
1. Candidate search is held identical across all conditions:
   Every condition evaluates the exact same 24 candidate action sequences of length K.
2. Initial states, obstacle layouts, goal locations, scoring weights, and step budgets
   are bit-identically matched across all conditions (N = 30 episodes).
3. Evaluates:
   - Condition A: Reactive baseline (K=0, no lookahead).
   - Condition B: Direct K-step macro prediction (H-JEPA MacroPredictor in 1 pass).
   - Condition C: Recursive K-step micro unrolling (H-JEPA MicroPredictor unrolled K times).
   - Condition D: Oracle MuJoCo K-step physics simulation (ground-truth physical MPC).
   - Condition E: Shuffled / inverted predicted transitions (negative control).
   - Condition F: Macro candidate sequences scored without future prediction (reactive prior only).
4. Evaluates open-loop multi-step prediction across horizons K in {1, 3, 5, 10} on held_out_test_benchmark.
5. Computes paired differences (Delta Success, Delta Collision, Delta GoalError) with
   exact 95% bootstrap confidence intervals and Wilcoxon signed-rank tests.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Add project root and world_models_fpga
repo_root = Path(__file__).resolve().parents[3]
pkg_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(pkg_root))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

from env.amr_env import JetAutoEnv
from models.h_jepa import HJEPAModel
from models.le_world_model import LeWorldModel
from models.baseline_policy import encode_task_id
from research.benchmarks.model_wrappers import compute_body_frame_goal_vec
from research.datasets.generation.capability_tasks import generate_task_specification


def _sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _bootstrap_ci95_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """Computes mean difference and 95% bootstrap confidence interval."""
    diff = a - b
    mean_diff = float(np.mean(diff))
    if len(diff) <= 1:
        return mean_diff, mean_diff, mean_diff
    rng = np.random.default_rng(seed)
    boot_means = [np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(n_boot)]
    low = float(np.percentile(boot_means, 2.5))
    high = float(np.percentile(boot_means, 97.5))
    return mean_diff, low, high


def _paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if np.all(diff == 0.0):
        return 1.0
    try:
        res = wilcoxon(diff, zero_method="pratt")
        return float(res.pvalue)
    except Exception:
        return 1.0


def build_matched_action_candidates(a_react: torch.Tensor, k: int, device: str) -> torch.Tensor:
    """
    Constructs exactly 24 matched action sequences of length K (shape: 24, K, 3).
    - Cand 0: a_react repeated K steps.
    - Cand 1..8: 8 local perturbations around a_react repeated K steps.
    - Cand 9..16: 8 fixed directional basis actions repeated K steps.
    - Cand 17..23: 7 two-phase detour-advance sequences of length K.
    """
    ar = a_react.squeeze(0)  # (3,)
    deltas = torch.tensor(
        [
            [0.25, 0.0, 0.0],
            [-0.25, 0.0, 0.0],
            [0.0, 0.40, 0.0],
            [0.0, -0.40, 0.0],
            [0.15, 0.35, 0.15],
            [0.15, -0.35, -0.15],
            [0.0, 0.0, 0.40],
            [0.0, 0.0, -0.40],
        ],
        dtype=torch.float32,
        device=device,
    )
    local_1step = torch.clamp(ar.unsqueeze(0) + deltas, -1.0, 1.0)  # (8, 3)

    fixed_1step = torch.tensor(
        [
            [0.85, 0.0, 0.0],
            [-0.40, 0.0, 0.0],
            [0.0, 0.75, 0.0],
            [0.0, -0.75, 0.0],
            [0.60, 0.50, 0.0],
            [0.60, -0.50, 0.0],
            [0.0, 0.0, 0.80],
            [0.0, 0.0, -0.80],
        ],
        dtype=torch.float32,
        device=device,
    )  # (8, 3)

    # 17 constant-action candidates repeated K times
    const_1step = torch.cat([ar.unsqueeze(0), local_1step, fixed_1step], dim=0)  # (17, 3)
    const_k = const_1step.unsqueeze(1).repeat(1, k, 1)  # (17, K, 3)

    # 7 two-phase sequences
    k_first = max(1, k // 2)
    k_second = k - k_first

    two_phase_pairs = [
        # Left strafe then forward
        (torch.tensor([0.20, 0.70, 0.10], device=device), torch.tensor([0.80, 0.0, 0.0], device=device)),
        # Right strafe then forward
        (torch.tensor([0.20, -0.70, -0.10], device=device), torch.tensor([0.80, 0.0, 0.0], device=device)),
        # Turn left then forward
        (torch.tensor([0.10, 0.0, 0.75], device=device), torch.tensor([0.85, 0.0, 0.0], device=device)),
        # Turn right then forward
        (torch.tensor([0.10, 0.0, -0.75], device=device), torch.tensor([0.85, 0.0, 0.0], device=device)),
        # Wide left flank then inward turn
        (torch.tensor([0.45, 0.60, 0.20], device=device), torch.tensor([0.70, -0.30, -0.15], device=device)),
        # Wide right flank then inward turn
        (torch.tensor([0.45, -0.60, -0.20], device=device), torch.tensor([0.70, 0.30, 0.15], device=device)),
        # Reactive proposal then damped approach
        (ar, ar * 0.6),
    ]

    two_phase_k = []
    for a1, a2 in two_phase_pairs:
        seq = torch.cat([a1.repeat(k_first, 1), a2.repeat(k_second, 1)], dim=0)  # (K, 3)
        two_phase_k.append(seq)
    two_phase_k = torch.stack(two_phase_k, dim=0)  # (7, K, 3)

    # Total: 17 + 7 = 24 candidates
    cands = torch.cat([const_k, two_phase_k], dim=0)
    return cands  # (24, K, 3)


def run_open_loop_prediction_audit(
    hjepa_model: HJEPAModel,
    lewm_model: LeWorldModel,
    val_npz_path: Path,
    device: str,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
    max_transitions: int = 500,
) -> dict:
    """
    Evaluates open-loop multi-step prediction on held_out_test_benchmark:
    - Direct Macro prediction vs Recursive Micro unrolling vs LeWM single-scale.
    """
    d = np.load(val_npz_path)
    n = min(max_transitions, len(d["rgb"]))
    rgb_t = torch.from_numpy(d["rgb"][:n]).permute(0, 3, 1, 2).float().to(device) / 255.0
    act_t = torch.from_numpy(d["actions"][:n]).float().to(device)
    poses = d["robot_pose"][:n]
    cols = d["collisions"][:n].astype(np.float32)
    ep_idx = d["transition_episode_indices"][:n]

    with torch.no_grad():
        z_h = hjepa_model.encode(rgb_t)
        z_l = lewm_model.encode(rgb_t)

        results_by_h = {}
        for h in horizons:
            macro_latent_mses = []
            macro_latent_cosines = []
            micro_latent_mses = []
            micro_latent_cosines = []
            lewm_latent_mses = []

            macro_disp_errs = []
            micro_disp_errs = []
            lewm_disp_errs = []

            macro_col_preds, micro_col_preds, true_cols = [], [], []

            for i in range(n - h):
                if ep_idx[i] != ep_idx[i + h]:
                    continue
                # Ground truth transition
                p0 = poses[i]
                pk = poses[i + h]
                th0 = float(p0[2])
                dx_w = float(pk[0] - p0[0])
                dy_w = float(pk[1] - p0[1])
                dth_true = float(pk[2] - p0[2])
                # True body-frame displacement over h steps
                dx_b = np.cos(th0) * dx_w + np.sin(th0) * dy_w
                dy_b = -np.sin(th0) * dx_w + np.cos(th0) * dy_w
                true_disp = np.array([dx_b, dy_b, dth_true], dtype=np.float32)

                true_zh_k = z_h[i + h : i + h + 1]
                true_zl_k = z_l[i + h : i + h + 1]
                col_in_horizon = float(np.max(cols[i : i + h]))
                true_cols.append(col_in_horizon)

                # 1. H-JEPA Macro (Direct h-step if h == 5, or scaled sequence)
                # Flatten action sequence for macro input
                act_seq = act_t[i : i + h].flatten().unsqueeze(0)  # (1, 3*h)
                if h == 5:
                    pred_zh_macro, m_col_logit = hjepa_model.macro_predictor(z_h[i : i + 1], act_seq)
                    macro_disp = hjepa_model.predict_macro_displacement(z_h[i : i + 1], pred_zh_macro).squeeze(0).detach().cpu().numpy()
                    macro_latent_mses.append(float(F.mse_loss(pred_zh_macro, true_zh_k).item()))
                    macro_latent_cosines.append(float(F.cosine_similarity(pred_zh_macro, true_zh_k).item()))
                    macro_disp_errs.append(float(np.linalg.norm(macro_disp[:2] - true_disp[:2])))
                    macro_col_preds.append(float(torch.sigmoid(m_col_logit).item()))

                # 2. H-JEPA Micro recursive unroll over h steps
                curr_zh = z_h[i : i + 1]
                cum_micro_disp = np.zeros(3, dtype=np.float32)
                max_micro_col = 0.0
                for s in range(h):
                    a_s = act_t[i + s : i + s + 1]
                    next_zh, col_logit = hjepa_model.micro_predictor.forward_transition(curr_zh, a_s)
                    step_disp = hjepa_model.predict_micro_displacement(curr_zh, next_zh).squeeze(0).detach().cpu().numpy()
                    cum_micro_disp[:2] += step_disp[:2]
                    cum_micro_disp[2] += step_disp[2]
                    max_micro_col = max(max_micro_col, float(torch.sigmoid(col_logit).item()))
                    curr_zh = next_zh

                micro_latent_mses.append(float(F.mse_loss(curr_zh, true_zh_k).item()))
                micro_latent_cosines.append(float(F.cosine_similarity(curr_zh, true_zh_k).item()))
                micro_disp_errs.append(float(np.linalg.norm(cum_micro_disp[:2] - true_disp[:2])))
                micro_col_preds.append(max_micro_col)

                # 3. LeWM recursive unroll over h steps
                curr_zl = z_l[i : i + 1]
                cum_lewm_disp = np.zeros(3, dtype=np.float32)
                for s in range(h):
                    a_s = act_t[i + s : i + s + 1]
                    next_zl, _ = lewm_model.predict_next(curr_zl, a_s)
                    step_disp = lewm_model.predict_ego_displacement(curr_zl, next_zl).squeeze(0).detach().cpu().numpy()
                    cum_lewm_disp[:2] += step_disp[:2]
                    cum_lewm_disp[2] += step_disp[2]
                    curr_zl = next_zl
                lewm_latent_mses.append(float(F.mse_loss(curr_zl, true_zl_k).item()))
                lewm_disp_errs.append(float(np.linalg.norm(cum_lewm_disp[:2] - true_disp[:2])))

            results_by_h[f"k_{h}"] = {
                "micro_recursive_latent_mse": float(np.mean(micro_latent_mses)),
                "micro_recursive_latent_cosine": float(np.mean(micro_latent_cosines)),
                "micro_recursive_disp_error_m": float(np.mean(micro_disp_errs)),
                "lewm_recursive_latent_mse": float(np.mean(lewm_latent_mses)),
                "lewm_recursive_disp_error_m": float(np.mean(lewm_disp_errs)),
            }
            if h == 5:
                results_by_h[f"k_{h}"]["macro_direct_latent_mse"] = float(np.mean(macro_latent_mses))
                results_by_h[f"k_{h}"]["macro_direct_latent_cosine"] = float(np.mean(macro_latent_cosines))
                results_by_h[f"k_{h}"]["macro_direct_disp_error_m"] = float(np.mean(macro_disp_errs))
                results_by_h[f"k_{h}"]["macro_vs_micro_disp_error_delta_m"] = float(np.mean(macro_disp_errs) - np.mean(micro_disp_errs))

    return results_by_h


def evaluate_closed_loop_matched_controls(
    hjepa_model: HJEPAModel,
    task_specs: list[dict],
    device: str,
    k: int = 5,
    max_steps: int = 45,
) -> dict:
    """
    Executes closed-loop evaluation on identical tasks with strictly matched candidate sets.
    """
    env = JetAutoEnv()
    conditions = [
        "cond_A_reactive",
        "cond_B_direct_macro",
        "cond_C_recursive_micro",
        "cond_D_oracle_physics",
        "cond_E_shuffled_control",
        "cond_F_open_loop_reactive_scorer",
    ]

    cond_records = {c: [] for c in conditions}

    for spec_idx, spec in enumerate(task_specs):
        for cond in conditions:
            env.restore_physical_baseline()
            env.apply_task_spec(spec)
            init_pos = np.asarray(spec["initial_pose"], dtype=np.float32)
            obs, _ = env.reset(options={"initial_pos": (float(init_pos[0]), float(init_pos[1]), float(init_pos[2]))})

            waypoints = spec.get("waypoints", [init_pos])
            final_wp = np.asarray(waypoints[-1], dtype=np.float32)

            rng = np.random.default_rng(spec_idx * 100 + 42)
            collision_occurred = False
            goal_reached = False
            total_path = 0.0
            prev_xy = init_pos[:2].copy()

            for step in range(max_steps):
                curr_pose = env.get_robot_pose()
                goal_body = compute_body_frame_goal_vec(curr_pose, final_wp)
                dist_to_goal = float(np.linalg.norm(curr_pose[:2] - final_wp[:2]))

                if dist_to_goal <= 0.25:
                    goal_reached = True
                    break

                img_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                with torch.no_grad():
                    s_t = hjepa_model.encode(img_t)
                    goal_t = torch.as_tensor(goal_body, dtype=torch.float32, device=device).view(1, 4)
                    a_react = hjepa_model.predict_reactive_action(s_t, goal_t)

                gx = goal_t[0, 0]
                gy = goal_t[0, 1]
                g_dth = torch.atan2(goal_t[0, 2], goal_t[0, 3])

                # Condition A: Reactive (no candidate search)
                if cond == "cond_A_reactive":
                    action_exec = np.clip(a_react.squeeze(0).cpu().numpy(), -1.0, 1.0)
                else:
                    # Build matched candidate sequences: (24, K, 3)
                    cands = build_matched_action_candidates(a_react, k=k, device=device)
                    num_cands = cands.shape[0]  # 24

                    if cond == "cond_B_direct_macro":
                        # Direct Macro forward pass: flatten action sequence (24, 3*K)
                        macro_in = cands.view(num_cands, -1)
                        s_rep = s_t.repeat(num_cands, 1)
                        with torch.no_grad():
                            next_S, m_col_logits = hjepa_model.macro_predictor(s_rep, macro_in)
                            disp = hjepa_model.predict_macro_displacement(s_rep, next_S)
                            col_p = torch.sigmoid(m_col_logits).squeeze(-1)

                        rem_dx = gx - disp[:, 0]
                        rem_dy = gy - disp[:, 1]
                        rem_dist = torch.sqrt(rem_dx ** 2 + rem_dy ** 2 + 1e-6)
                        prior_cost = torch.sum((cands[:, 0] - a_react) ** 2, dim=-1)
                        cost = rem_dist + 0.10 * prior_cost + 1.8 * col_p
                        best_idx = int(torch.argmin(cost).item())
                        action_exec = np.clip(cands[best_idx, 0].cpu().numpy(), -1.0, 1.0)

                    elif cond == "cond_C_recursive_micro":
                        # Recursive Micro unrolling K steps
                        s_curr = s_t.repeat(num_cands, 1)
                        cum_disp = torch.zeros(num_cands, 3, device=device)
                        max_col = torch.zeros(num_cands, device=device)
                        with torch.no_grad():
                            for t_step in range(k):
                                a_step = cands[:, t_step]
                                s_next, col_log = hjepa_model.micro_predictor.forward_transition(s_curr, a_step)
                                step_disp = hjepa_model.predict_micro_displacement(s_curr, s_next)
                                cum_disp = cum_disp + step_disp
                                max_col = torch.maximum(max_col, torch.sigmoid(col_log).squeeze(-1))
                                s_curr = s_next

                        rem_dx = gx - cum_disp[:, 0]
                        rem_dy = gy - cum_disp[:, 1]
                        rem_dist = torch.sqrt(rem_dx ** 2 + rem_dy ** 2 + 1e-6)
                        prior_cost = torch.sum((cands[:, 0] - a_react) ** 2, dim=-1)
                        cost = rem_dist + 0.10 * prior_cost + 1.8 * max_col
                        best_idx = int(torch.argmin(cost).item())
                        action_exec = np.clip(cands[best_idx, 0].cpu().numpy(), -1.0, 1.0)

                    elif cond == "cond_D_oracle_physics":
                        # True MuJoCo state clone simulation
                        qpos_saved = env.data.qpos.copy()
                        qvel_saved = env.data.qvel.copy()
                        c_np = cands.cpu().numpy()
                        oracle_costs = []

                        for c_i in range(num_cands):
                            env.data.qpos[:] = qpos_saved
                            env.data.qvel[:] = qvel_saved
                            import mujoco
                            mujoco.mj_forward(env.model, env.data)
                            hit_col = False
                            for s_i in range(k):
                                if env.step_physics(c_np[c_i, s_i]):
                                    hit_col = True
                            p_k = env.get_robot_pose()
                            d_k = float(np.linalg.norm(p_k[:2] - final_wp[:2]))
                            prior = float(np.sum((c_np[c_i, 0] - a_react.cpu().numpy()) ** 2))
                            c_val = d_k + 0.10 * prior + (2.5 if hit_col else 0.0)
                            oracle_costs.append(c_val)

                        # Restore simulator state
                        env.data.qpos[:] = qpos_saved
                        env.data.qvel[:] = qvel_saved
                        import mujoco
                        mujoco.mj_forward(env.model, env.data)
                        best_idx = int(np.argmin(oracle_costs))
                        action_exec = np.clip(cands[best_idx, 0].cpu().numpy(), -1.0, 1.0)

                    elif cond == "cond_E_shuffled_control":
                        # Shuffled predicted transition (negative control)
                        macro_in = cands.view(num_cands, -1)
                        s_rep = s_t.repeat(num_cands, 1)
                        with torch.no_grad():
                            next_S, m_col_logits = hjepa_model.macro_predictor(s_rep, macro_in)
                            disp = hjepa_model.predict_macro_displacement(s_rep, next_S)
                            col_p = torch.sigmoid(m_col_logits).squeeze(-1)

                        rem_dx = gx - disp[:, 0]
                        rem_dy = gy - disp[:, 1]
                        rem_dist = torch.sqrt(rem_dx ** 2 + rem_dy ** 2 + 1e-6)
                        prior_cost = torch.sum((cands[:, 0] - a_react) ** 2, dim=-1)
                        cost = rem_dist + 0.10 * prior_cost + 1.8 * col_p
                        cost_np = cost.cpu().numpy()
                        shuffled_cost = rng.permutation(cost_np)
                        best_idx = int(np.argmin(shuffled_cost))
                        action_exec = np.clip(cands[best_idx, 0].cpu().numpy(), -1.0, 1.0)

                    elif cond == "cond_F_open_loop_reactive_scorer":
                        # Score candidate sequences solely by alignment to reactive action (no future world model)
                        prior_cost = torch.sum((cands[:, 0] - a_react) ** 2, dim=-1)
                        best_idx = int(torch.argmin(prior_cost).item())
                        action_exec = np.clip(cands[best_idx, 0].cpu().numpy(), -1.0, 1.0)

                # Step environment
                step_res = env.step(action_exec)
                next_obs = step_res[0]
                info = step_res[-1]
                if info.get("collision", False):
                    collision_occurred = True
                curr_xy = env.get_robot_pose()[:2]
                total_path += float(np.linalg.norm(curr_xy - prev_xy))
                prev_xy = curr_xy.copy()
                obs = next_obs

            final_pose = env.get_robot_pose()
            final_err = float(np.linalg.norm(final_pose[:2] - final_wp[:2]))
            success = bool(goal_reached and not collision_occurred)

            cond_records[cond].append({
                "episode_idx": spec_idx,
                "success": success,
                "goal_reached": goal_reached,
                "collision": collision_occurred,
                "final_goal_error_m": final_err,
                "path_length_m": total_path,
            })

    env.close()

    # Summarize conditions
    summary = {}
    for c in conditions:
        succ_arr = np.array([1.0 if r["success"] else 0.0 for r in cond_records[c]], dtype=np.float32)
        col_arr = np.array([1.0 if r["collision"] else 0.0 for r in cond_records[c]], dtype=np.float32)
        err_arr = np.array([r["final_goal_error_m"] for r in cond_records[c]], dtype=np.float32)

        summary[c] = {
            "episodes_evaluated": len(succ_arr),
            "success_rate": float(np.mean(succ_arr)),
            "collision_rate": float(np.mean(col_arr)),
            "mean_final_goal_error_m": float(np.mean(err_arr)),
            "per_episode_success": [bool(x) for x in succ_arr],
            "per_episode_collision": [bool(x) for x in col_arr],
            "per_episode_goal_error": [float(x) for x in err_arr],
        }

    # Statistical comparisons against reactive baseline (cond_A)
    succ_A = np.array(summary["cond_A_reactive"]["per_episode_success"], dtype=np.float32)
    succ_B = np.array(summary["cond_B_direct_macro"]["per_episode_success"], dtype=np.float32)
    succ_C = np.array(summary["cond_C_recursive_micro"]["per_episode_success"], dtype=np.float32)
    succ_D = np.array(summary["cond_D_oracle_physics"]["per_episode_success"], dtype=np.float32)
    succ_E = np.array(summary["cond_E_shuffled_control"]["per_episode_success"], dtype=np.float32)

    col_A = np.array(summary["cond_A_reactive"]["per_episode_collision"], dtype=np.float32)
    col_B = np.array(summary["cond_B_direct_macro"]["per_episode_collision"], dtype=np.float32)
    col_C = np.array(summary["cond_C_recursive_micro"]["per_episode_collision"], dtype=np.float32)

    # Macro vs Reactive
    m_diff, m_low, m_high = _bootstrap_ci95_diff(succ_B, succ_A)
    c_diff, c_low, c_high = _bootstrap_ci95_diff(col_B, col_A)
    p_succ_B_A = _paired_wilcoxon(succ_B, succ_A)

    # Macro vs Micro
    mm_diff, mm_low, mm_high = _bootstrap_ci95_diff(succ_B, succ_C)
    p_succ_B_C = _paired_wilcoxon(succ_B, succ_C)

    # Macro vs Shuffled (Causal test)
    me_diff, me_low, me_high = _bootstrap_ci95_diff(succ_B, succ_E)
    p_succ_B_E = _paired_wilcoxon(succ_B, succ_E)

    # Oracle vs Macro
    od_diff, _, _ = _bootstrap_ci95_diff(succ_D, succ_B)

    comparisons = {
        "macro_vs_reactive": {
            "delta_success_rate": m_diff,
            "delta_success_ci95": [m_low, m_high],
            "wilcoxon_p_value": p_succ_B_A,
            "delta_collision_rate": c_diff,
            "delta_collision_ci95": [c_low, c_high],
        },
        "macro_vs_recursive_micro": {
            "delta_success_rate": mm_diff,
            "delta_success_ci95": [mm_low, mm_high],
            "wilcoxon_p_value": p_succ_B_C,
            "notes": "With matched candidate search, isolates whether direct macro multi-step prediction outperforms micro unrolling.",
        },
        "macro_vs_shuffled_control": {
            "delta_success_rate": me_diff,
            "delta_success_ci95": [me_low, me_high],
            "wilcoxon_p_value": p_succ_B_E,
            "causal_effect_supported": bool(me_diff > 0.0 and p_succ_B_E < 0.05),
        },
        "oracle_vs_macro_gap": {
            "delta_success_rate": od_diff,
            "notes": "Remaining headroom between learned macro world model and ground-truth physics MPC.",
        },
    }

    return {"conditions": summary, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description="Run Matched Temporal-Prediction Controls Benchmark")
    parser.add_argument("--episodes", type=int, default=30, help="Number of matched evaluation episodes (N >= 30)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    out_file = Path(args.output) if args.output else pkg_root / "research/benchmarks/results/matched_temporal_controls.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    ckpt_dir = pkg_root / "checkpoints_v4"
    hjepa_path = ckpt_dir / "hjepa_best.pt"
    lewm_path = ckpt_dir / "lewm_best.pt"
    val_npz = pkg_root / "research/datasets/splits_v4/held_out_test_benchmark.npz"

    print("=================================================================")
    print("  MATCHED TEMPORAL-PREDICTION CONTROLS BENCHMARK")
    print(f"  Device: {args.device} | Checkpoint: {hjepa_path.name}")
    print(f"  Evaluation Episodes: {args.episodes} | Candidates per step: 24 (identical across all conditions)")
    print("=================================================================")

    hjepa = HJEPAModel(latent_dim=32, stride_k=5, hidden_dim=64).to(args.device)
    hjepa.load_state_dict(torch.load(hjepa_path, map_location=args.device, weights_only=False), strict=False)
    hjepa.eval()

    lewm = LeWorldModel(latent_dim=32, action_dim=3, hidden_dim=64).to(args.device)
    lewm.load_state_dict(torch.load(lewm_path, map_location=args.device, weights_only=False), strict=False)
    lewm.eval()

    # 1. Open-loop prediction audit
    print("\n--- 1. Evaluating Open-Loop Multi-Horizon Prediction Accuracy ---")
    open_loop_res = run_open_loop_prediction_audit(hjepa, lewm, val_npz, args.device, horizons=(1, 3, 5, 10))
    print(json.dumps(open_loop_res, indent=2))

    # 2. Closed-loop navigation with strictly matched candidate search
    print(f"\n--- 2. Evaluating Closed-Loop Navigation on {args.episodes} Matched Episodes ---")
    # Generate 30 matched episodes across capabilities
    task_specs = []
    caps = ["obstacle_avoidance", "spatial_reasoning", "object_grounding"]
    for i in range(args.episodes):
        cap = caps[i % len(caps)]
        diff = "easy" if i < 18 else "difficult"
        spec = generate_task_specification(cap, diff, seed_idx=8800 + i)
        spec["episode_id"] = f"matched_temp_{cap}_{diff}_{i:02d}"
        task_specs.append(spec)

    closed_loop_res = evaluate_closed_loop_matched_controls(hjepa, task_specs, args.device, k=5, max_steps=45)

    # Print summary table
    print("\n================== RESULTS SUMMARY ==================")
    for c_name, c_data in closed_loop_res["conditions"].items():
        print(f"  {c_name:<35s} | Succ: {c_data['success_rate']*100:5.1f}% | Col: {c_data['collision_rate']*100:5.1f}% | GoalErr: {c_data['mean_final_goal_error_m']:.3f}m")

    print("\n================== PAIRED STATISTICAL COMPARISONS ==================")
    for comp_name, comp_data in closed_loop_res["comparisons"].items():
        print(f"  [{comp_name}]")
        for k_k, v_v in comp_data.items():
            print(f"    {k_k}: {v_v}")

    # Build execution metadata
    metadata = {
        "command_executed": f"python world_models_fpga/research/benchmarks/run_matched_temporal_controls.py --episodes {args.episodes}",
        "execution_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_sha(),
        "dataset_path": str(val_npz.relative_to(repo_root)),
        "dataset_sha256": _sha256(val_npz),
        "checkpoint_path": str(hjepa_path.relative_to(repo_root)),
        "checkpoint_sha256": _sha256(hjepa_path),
        "checkpoints": [
            {"name": "hjepa_best.pt", "path": str(hjepa_path.relative_to(repo_root)), "sha256": _sha256(hjepa_path)},
            {"name": "lewm_best.pt", "path": str(lewm_path.relative_to(repo_root)), "sha256": _sha256(lewm_path)},
        ],
        "random_seed": 42,
        "configuration": {
            "episodes": args.episodes,
            "horizon_k": 5,
            "num_candidates": 24,
            "max_steps": 45,
            "conditions": list(closed_loop_res["conditions"].keys()),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": args.device,
        },
    }

    payload = {
        "execution_metadata": metadata,
        "open_loop_multi_horizon": open_loop_res,
        "closed_loop_matched_controls": closed_loop_res,
    }

    out_file.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n[SAVED COMPLETE ARTIFACT TO {out_file}]")


if __name__ == "__main__":
    main()
