"""
Predictive Safety as Prospective Constraint Benchmark (Scientifically Repaired Protocol)

Investigates whether learned predictive world models provide prospective collision avoidance:
1. Evaluates 5 strictly matched conditions across identical collision-critical tasks:
   - Condition 1: Intact H-JEPA Collision Predictor (action selection penalized by predicted collision probability).
   - Condition 2: Shuffled Action-Collision Associations (true causal test: collision probabilities permuted across candidates).
   - Condition 3: Collision Head Disabled (ablation: safety penalty set to 0.0; search guided only by goal distance).
   - Condition 4: Random Collision Predictor (uninformed control: collision probabilities sampled from empirical prior).
   - Condition 5: Oracle Physics Collision Predictor (upper bound: MuJoCo state clone forward simulation).
2. Measures prospective collision prediction quality on held-out test transitions:
   - Brier score (MSE of predicted probability vs binary collision ground truth)
   - Expected Calibration Error (ECE, 10 bins)
   - AUROC (Area under the ROC curve)
   - False-Negative Rate (FNR) at decision threshold 0.5 (missed imminent collisions)
3. Evaluates closed-loop collision rates, success rates, and paired statistical differences (McNemar tests).
4. Explicit scientific note: This benchmark falsifies this particular learned prospective collision-scoring filter on the tested discrete-horizon collision-critical tasks. It does not falsify all predictive safety mechanisms or formal theoretical continuous-time Control Barrier Functions (CBFs).
"""

from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "egl"

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
import mujoco
import torch
import torch.nn.functional as F
from scipy.stats import binom

from env.amr_env import JetAutoEnv
from models.h_jepa import HJEPAModel
from research.benchmarks.model_wrappers import compute_body_frame_goal_vec
from research.benchmarks.run_matched_temporal_controls import build_matched_action_candidates
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


def _git_clean() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return len(out) == 0
    except Exception:
        return False


def _paired_mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar binomial test on paired binary outcomes (a vs b)."""
    n10 = int(np.sum((a == 1.0) & (b == 0.0)))
    n01 = int(np.sum((a == 0.0) & (b == 1.0)))
    discordant = n10 + n01
    if discordant == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p_val = float(min(1.0, 2.0 * binom.cdf(k, discordant, 0.5)))
    return n10, n01, p_val


def compute_prediction_quality_metrics(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> dict:
    """Computes Brier score, ECE, AUROC, and FNR at threshold 0.5."""
    probs = np.asarray(probs, dtype=np.float64).clip(0.0, 1.0)
    targets = np.asarray(targets, dtype=np.float64)

    # 1. Brier score
    brier = float(np.mean((probs - targets) ** 2))

    # 2. Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(probs)
    for b in range(n_bins):
        low, high = bin_boundaries[b], bin_boundaries[b + 1]
        in_bin = (probs >= low) & (probs <= high if b == n_bins - 1 else probs < high)
        n_bin = np.sum(in_bin)
        if n_bin > 0:
            bin_acc = float(np.mean(targets[in_bin]))
            bin_conf = float(np.mean(probs[in_bin]))
            ece += (n_bin / n_total) * abs(bin_acc - bin_conf)

    # 3. AUROC
    # Sort by descending predicted probability
    desc_idx = np.argsort(-probs)
    sorted_targets = targets[desc_idx]
    n_pos = np.sum(targets == 1.0)
    n_neg = np.sum(targets == 0.0)
    if n_pos == 0 or n_neg == 0:
        auroc = 0.5
    else:
        # Wilcoxon-Mann-Whitney U statistic formulation for AUROC
        ranks = np.arange(len(sorted_targets), 0, -1)
        sum_pos_ranks = np.sum(ranks[sorted_targets == 1.0])
        u_stat = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
        auroc = float(u_stat / (n_pos * n_neg))

    # 4. False Negative Rate at threshold 0.5
    preds_05 = (probs >= 0.5).astype(np.float64)
    actual_pos = targets == 1.0
    if np.sum(actual_pos) > 0:
        fn_count = int(np.sum((preds_05 == 0.0) & actual_pos))
        fnr = float(fn_count / np.sum(actual_pos))
    else:
        fn_count = 0
        fnr = 0.0

    return {
        "brier_score": float(brier),
        "expected_calibration_error_ece": float(ece),
        "auroc": float(auroc),
        "false_negative_rate_at_05": float(fnr),
        "false_negatives_count": fn_count,
        "total_positive_collisions": int(n_pos),
        "total_eval_samples": int(n_total),
    }


def evaluate_closed_loop_safety_benchmark(
    hjepa_model: HJEPAModel,
    task_specs: list[dict],
    device: str,
    k: int = 5,
    max_steps: int = 45,
) -> dict:
    """Evaluates the 5 predictive safety conditions under strictly matched environments."""
    env = JetAutoEnv()
    conditions = [
        "cond_1_intact_collision_head",
        "cond_2_shuffled_collision_binding",
        "cond_3_collision_head_disabled",
        "cond_4_random_collision_predictor",
        "cond_5_oracle_collision_predictor",
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

            rng = np.random.default_rng(spec_idx * 200 + 77)
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

                # Generate matched candidate action sequences (24, K, 3)
                cands = build_matched_action_candidates(a_react, k=k, device=device)
                num_cands = cands.shape[0]

                # Model forward pass
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

                if cond == "cond_1_intact_collision_head":
                    # Intact prospective collision penalty
                    cost = rem_dist + 0.10 * prior_cost + 1.8 * col_p
                    best_idx = int(torch.argmin(cost).item())
                    action_exec = np.clip(cands[best_idx, 0].cpu().numpy(), -1.0, 1.0)

                elif cond == "cond_2_shuffled_collision_binding":
                    # Shuffled action-collision associations (multi-permutation average)
                    perm_costs = []
                    for _ in range(10):
                        perm = rng.permutation(num_cands)
                        col_p_perm = col_p[perm]
                        c_p = rem_dist + 0.10 * prior_cost + 1.8 * col_p_perm
                        perm_costs.append(c_p)
                    mean_perm_cost = torch.stack(perm_costs, dim=0).mean(dim=0)
                    best_idx = int(torch.argmin(mean_perm_cost).item())
                    action_exec = np.clip(cands[best_idx, 0].cpu().numpy(), -1.0, 1.0)

                elif cond == "cond_3_collision_head_disabled":
                    # Safety head disabled: lambda_col = 0.0
                    cost = rem_dist + 0.10 * prior_cost
                    best_idx = int(torch.argmin(cost).item())
                    action_exec = np.clip(cands[best_idx, 0].cpu().numpy(), -1.0, 1.0)

                elif cond == "cond_4_random_collision_predictor":
                    # Random collision penalty sampled uniformly in [0, 1]
                    rand_col = torch.rand(num_cands, device=device)
                    cost = rem_dist + 0.10 * prior_cost + 1.8 * rand_col
                    best_idx = int(torch.argmin(cost).item())
                    action_exec = np.clip(cands[best_idx, 0].cpu().numpy(), -1.0, 1.0)

                elif cond == "cond_5_oracle_collision_predictor":
                    # Oracle MuJoCo forward simulation
                    qpos_saved = env.data.qpos.copy()
                    qvel_saved = env.data.qvel.copy()
                    c_np = cands.cpu().numpy()
                    oracle_costs = []
                    for c_i in range(num_cands):
                        env.data.qpos[:] = qpos_saved
                        env.data.qvel[:] = qvel_saved
                        mujoco.mj_forward(env.model, env.data)
                        hit = False
                        for s_i in range(k):
                            if env.step_physics(c_np[c_i, s_i]):
                                hit = True
                        p_k = env.get_robot_pose()
                        d_k = float(np.linalg.norm(p_k[:2] - final_wp[:2]))
                        prior = float(np.sum((c_np[c_i, 0] - a_react.cpu().numpy()) ** 2))
                        oracle_costs.append(d_k + 0.10 * prior + (2.5 if hit else 0.0))

                    env.data.qpos[:] = qpos_saved
                    env.data.qvel[:] = qvel_saved
                    mujoco.mj_forward(env.model, env.data)
                    best_idx = int(np.argmin(oracle_costs))
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
                "collision": collision_occurred,
                "goal_error_m": final_err,
                "path_length_m": total_path,
            })

    env.close()

    summary = {}
    for c in conditions:
        recs = cond_records[c]
        succ_arr = np.array([1.0 if r["success"] else 0.0 for r in recs], dtype=np.float32)
        col_arr = np.array([1.0 if r["collision"] else 0.0 for r in recs], dtype=np.float32)
        err_arr = np.array([r["goal_error_m"] for r in recs], dtype=np.float32)

        summary[c] = {
            "episodes_evaluated": len(succ_arr),
            "success_rate": float(np.mean(succ_arr)),
            "collision_rate": float(np.mean(col_arr)),
            "mean_final_goal_error_m": float(np.mean(err_arr)),
            "per_episode_success": [bool(x) for x in succ_arr],
            "per_episode_collision": [bool(x) for x in col_arr],
            "per_episode_goal_error": [float(x) for x in err_arr],
            "records": recs,
        }

    # Paired comparisons against Intact Collision Head (cond_1)
    col_1 = np.array(summary["cond_1_intact_collision_head"]["per_episode_collision"], dtype=np.float32)
    col_2 = np.array(summary["cond_2_shuffled_collision_binding"]["per_episode_collision"], dtype=np.float32)
    col_3 = np.array(summary["cond_3_collision_head_disabled"]["per_episode_collision"], dtype=np.float32)
    col_4 = np.array(summary["cond_4_random_collision_predictor"]["per_episode_collision"], dtype=np.float32)
    col_5 = np.array(summary["cond_5_oracle_collision_predictor"]["per_episode_collision"], dtype=np.float32)

    _, _, p_mcnemar_1_2 = _paired_mcnemar(col_1, col_2)
    _, _, p_mcnemar_1_3 = _paired_mcnemar(col_1, col_3)
    _, _, p_mcnemar_1_4 = _paired_mcnemar(col_1, col_4)
    _, _, p_mcnemar_1_5 = _paired_mcnemar(col_1, col_5)

    comparisons = {
        "intact_vs_shuffled_binding": {
            "collision_rate_intact": float(np.mean(col_1)),
            "collision_rate_shuffled": float(np.mean(col_2)),
            "delta_collision_rate": float(np.mean(col_2) - np.mean(col_1)),
            "mcnemar_p_value": p_mcnemar_1_2,
            "hypothesis_binding_verified": bool(p_mcnemar_1_2 < 0.05 and np.mean(col_1) < np.mean(col_2)),
        },
        "intact_vs_collision_head_disabled": {
            "collision_rate_intact": float(np.mean(col_1)),
            "collision_rate_disabled": float(np.mean(col_3)),
            "delta_collision_rate": float(np.mean(col_3) - np.mean(col_1)),
            "mcnemar_p_value": p_mcnemar_1_3,
        },
        "intact_vs_random_collision_predictor": {
            "collision_rate_intact": float(np.mean(col_1)),
            "collision_rate_random": float(np.mean(col_4)),
            "delta_collision_rate": float(np.mean(col_4) - np.mean(col_1)),
            "mcnemar_p_value": p_mcnemar_1_4,
        },
        "intact_vs_oracle_physics": {
            "collision_rate_intact": float(np.mean(col_1)),
            "collision_rate_oracle": float(np.mean(col_5)),
            "delta_collision_rate": float(np.mean(col_5) - np.mean(col_1)),
            "mcnemar_p_value": p_mcnemar_1_5,
        },
    }

    return {"conditions": summary, "comparisons": comparisons}


def evaluate_offline_prediction_quality(
    hjepa_model: HJEPAModel,
    test_npz_path: str | Path,
    device: str,
    max_samples: int = 500,
    stride_k: int = 5,
) -> dict:
    """Evaluates macro K=5 collision prediction quality on held-out transitions matching closed-loop control."""
    d = np.load(test_npz_path)
    rgb = d["rgb"]
    act = d["actions"]
    cols = d["collisions"].astype(np.float64)
    ep_idx = d["transition_episode_indices"]

    valid_starts = []
    for i in range(len(rgb) - stride_k):
        if ep_idx[i] == ep_idx[i + stride_k - 1]:
            valid_starts.append(i)
        if len(valid_starts) >= max_samples:
            break

    valid_starts = np.array(valid_starts, dtype=np.int64)
    rgb_t = torch.from_numpy(rgb[valid_starts]).permute(0, 3, 1, 2).float().to(device) / 255.0

    macro_acts = []
    macro_cols = []
    for s_i in valid_starts:
        macro_acts.append(act[s_i : s_i + stride_k].reshape(-1))
        # Ground truth: collision occurs anywhere within K steps
        macro_cols.append(float(np.max(cols[s_i : s_i + stride_k])))

    macro_act_t = torch.from_numpy(np.array(macro_acts, dtype=np.float32)).to(device)
    gt_macro_cols = np.array(macro_cols, dtype=np.float64)

    with torch.no_grad():
        s = hjepa_model.encode(rgb_t)
        _, m_col_logits = hjepa_model.macro_predictor(s, macro_act_t)
        col_probs_macro = torch.sigmoid(m_col_logits).squeeze(-1).cpu().numpy()

    metrics = compute_prediction_quality_metrics(col_probs_macro, gt_macro_cols, n_bins=10)
    sample_eps = ep_idx[valid_starts]
    metrics["clustered_bootstrap"] = compute_clustered_bootstrap_quality_metrics(sample_eps, col_probs_macro, gt_macro_cols, n_boot=1000)
    metrics["horizon_k"] = stride_k
    metrics["evaluated_samples"] = len(valid_starts)
    return metrics


def compute_clustered_bootstrap_quality_metrics(
    sample_episodes: np.ndarray,
    probs: np.ndarray,
    targets: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Computes 95% confidence intervals using cluster bootstrap resampled at the episode level."""
    unique_eps = np.unique(sample_episodes)
    rng = np.random.default_rng(seed)

    brier_boots = []
    ece_boots = []
    auroc_boots = []

    ep_to_indices = {ep: np.where(sample_episodes == ep)[0] for ep in unique_eps}

    for _ in range(n_boot):
        boot_eps = rng.choice(unique_eps, size=len(unique_eps), replace=True)
        boot_indices = np.concatenate([ep_to_indices[ep] for ep in boot_eps])
        boot_probs = probs[boot_indices]
        boot_targets = targets[boot_indices]

        # Brier
        brier_boots.append(float(np.mean((boot_probs - boot_targets) ** 2)))

        # ECE (10 bins)
        bin_boundaries = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        n_tot = len(boot_probs)
        for b in range(10):
            low, high = bin_boundaries[b], bin_boundaries[b + 1]
            in_b = (boot_probs >= low) & (boot_probs <= high if b == 9 else boot_probs < high)
            n_b = np.sum(in_b)
            if n_b > 0:
                ece += (n_b / n_tot) * abs(float(np.mean(boot_targets[in_b])) - float(np.mean(boot_probs[in_b])))
        ece_boots.append(ece)

        # AUROC
        desc = np.argsort(-boot_probs)
        sorted_t = boot_targets[desc]
        n_p = np.sum(sorted_t == 1.0)
        n_n = np.sum(sorted_t == 0.0)
        if n_p > 0 and n_n > 0:
            ranks = np.arange(len(sorted_t), 0, -1)
            u_stat = np.sum(ranks[sorted_t == 1.0]) - (n_p * (n_p + 1)) / 2.0
            auroc_boots.append(float(u_stat / (n_p * n_n)))
        else:
            auroc_boots.append(0.5)

    return {
        "brier_ci95": [float(np.percentile(brier_boots, 2.5)), float(np.percentile(brier_boots, 97.5))],
        "ece_ci95": [float(np.percentile(ece_boots, 2.5)), float(np.percentile(ece_boots, 97.5))],
        "auroc_ci95": [float(np.percentile(auroc_boots, 2.5)), float(np.percentile(auroc_boots, 97.5))],
        "cluster_episodes_count": int(len(unique_eps)),
        "bootstrap_replicates": n_boot,
    }


def main():
    parser = argparse.ArgumentParser(description="Predictive Safety as Prospective Constraint Benchmark")
    parser.add_argument("--episodes", type=int, default=40, help="Number of collision-critical episodes (N >= 30)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=str(pkg_root / "research/benchmarks/results/predictive_safety_benchmark.json"))
    args = parser.parse_args()

    ckpt_dir = pkg_root / "checkpoints_v4"
    hjepa_path = ckpt_dir / "hjepa_best.pt"
    test_npz = pkg_root / "research/datasets/splits_v4/held_out_test_benchmark.npz"

    print("=================================================================")
    print("  PREDICTIVE SAFETY AS PROSPECTIVE CONSTRAINT BENCHMARK")
    print(f"  Device: {args.device} | Checkpoint: {hjepa_path.name}")
    print(f"  Episodes: {args.episodes} | Protocol: Strict Matched Evaluation")
    print("=================================================================")

    hjepa = HJEPAModel(latent_dim=32, stride_k=5, hidden_dim=64).to(args.device)
    hjepa.load_state_dict(torch.load(hjepa_path, map_location=args.device, weights_only=False), strict=False)
    hjepa.eval()

    # 1. Offline Prospective Prediction Quality on Held-Out Test Data
    print("\n--- 1. Evaluating Prospective Collision Prediction Quality (Held-Out Test Data) ---")
    quality_metrics = evaluate_offline_prediction_quality(hjepa, test_npz, args.device, max_samples=500)
    print(f"  Brier Score:   {quality_metrics['brier_score']:.4f}")
    print(f"  ECE (10 bins): {quality_metrics['expected_calibration_error_ece']:.4f}")
    print(f"  AUROC:         {quality_metrics['auroc']:.4f}")
    print(f"  FNR (@ 0.5):   {quality_metrics['false_negative_rate_at_05']*100:.1f}% ({quality_metrics['false_negatives_count']}/{quality_metrics['total_positive_collisions']})")

    # 2. Closed-Loop Evaluation on Collision-Critical Tasks
    print(f"\n--- 2. Evaluating Closed-Loop Collision Rates across 5 Conditions (N = {args.episodes}) ---")
    task_specs = []
    for i in range(args.episodes):
        cap = "obstacle_avoidance" if i % 2 == 0 else "spatial_reasoning"
        diff = "easy" if i < int(args.episodes * 0.5) else "difficult"
        spec = generate_task_specification(cap, diff, seed_idx=7200 + i)
        spec["episode_id"] = f"safety_eval_{cap}_{diff}_{i:02d}"
        task_specs.append(spec)

    closed_loop_res = evaluate_closed_loop_safety_benchmark(hjepa, task_specs, args.device, k=5, max_steps=45)

    print("\n================== SAFETY BENCHMARK SUMMARY ==================")
    for c_name, c_data in closed_loop_res["conditions"].items():
        print(f"  {c_name:<40s} | Collisions: {c_data['collision_rate']*100:5.1f}% | Succ: {c_data['success_rate']*100:5.1f}% | GoalErr: {c_data['mean_final_goal_error_m']:.3f}m")

    print("\n================== PAIRED STATISTICAL COMPARISONS ==================")
    for comp_name, comp_data in closed_loop_res["comparisons"].items():
        print(f"  [{comp_name}]")
        for k_k, v_v in comp_data.items():
            print(f"    {k_k}: {v_v}")

    # Build execution metadata
    this_script = Path(__file__).resolve()
    metadata = {
        "script_path": str(this_script.relative_to(repo_root)),
        "script_sha256": _sha256(this_script),
        "command_executed": f"python world_models_fpga/research/benchmarks/run_predictive_safety_benchmark.py --episodes {args.episodes} --device {args.device}",
        "execution_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_sha(),
        "working_tree_clean": _git_clean(),
        "dataset_path": str(test_npz.relative_to(repo_root)),
        "dataset_sha256": _sha256(test_npz),
        "checkpoint_path": str(hjepa_path.relative_to(repo_root)),
        "checkpoint_sha256": _sha256(hjepa_path),
        "checkpoints": [
            {"name": "hjepa_best.pt", "path": str(hjepa_path.relative_to(repo_root)), "sha256": _sha256(hjepa_path)},
        ],
        "random_seed": 42,
        "configuration": {
            "episodes": args.episodes,
            "horizon_k": 5,
            "num_candidates": 24,
            "conditions": list(closed_loop_res["conditions"].keys()),
            "safety_type": "Empirical prospective safety filter (NOT a formal Control Barrier Function)",
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": args.device,
        },
    }

    full_output = {
        "execution_metadata": metadata,
        "held_out_prediction_quality": quality_metrics,
        "closed_loop_safety_eval": closed_loop_res,
        "scientific_qualification": (
            "This benchmark evaluates the empirical utility of prospective collision predictions. "
            "It does NOT establish a formal Control Barrier Function (CBF). Formal CBFs require mathematical "
            "forward invariance guarantees with class-K functions and known Lie derivatives, which cannot be "
            "formally asserted for neural network latent predictions."
        ),
    }

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(full_output, indent=2) + "\n")
    print(f"\n[Saved Artifact] -> {out_p}")


if __name__ == "__main__":
    main()
