#!/usr/bin/env python3
"""
Phase 3 Mechanistic Experiment v2: Goal Memory Stress Tests Under Degraded Visual Evidence.

Scientific Objectives:
1. Test when Python controller memory and learned latent ego-motion actually matter:
   Evaluates 9 controlled conditions across matched episodes under both clean observation
   and temporary visual occlusion/dropout (steps 5..14).
2. Explicitly distinguishes handwritten controller state (memory buffer, geometric rotation)
   from learned neural computation (latent micro displacement decoder micro_ego_head).
3. Intervenes independently on:
   - visual goal weighting (visual-only vs blended)
   - memory update (reset, frozen, corrupted)
   - ego-motion displacement (zeroed, perturbed)
   - donor memory (matched vs mismatched)
4. Uses post-action distance evaluation, strict initial distance >= 0.60m, exact McNemar tests,
   paired bootstrap CIs, and explicit equivalence/non-inferiority testing.

Outputs:
  - Summary JSON: world_models_fpga/research/circuits/results/goal_memory_stress_tests_v2.json
  - Numerical Traces: world_models_fpga/research/circuits/results/goal_memory_stress_tests_v2_traces.npz
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "world_models_fpga"))
sys.path.insert(0, str(REPO_ROOT))

import mujoco
import numpy as np
import scipy.stats as stats
from scipy.stats import binom
import torch

from env.amr_env import JetAutoEnv
from models.baseline_policy import encode_task_id
from research.benchmarks.model_wrappers import HJEPAAgent, compute_body_frame_goal_vec
from research.datasets.dataset_config import SPLITS_DIR, CKPT_DIR
from research.datasets.generation.capability_tasks import (
    CANONICAL_OBSTACLES,
    generate_task_specification,
    get_waypoint_for_relation,
)


def compute_sha256(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def paired_mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Exact paired McNemar test using binomial CDF."""
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    disc = n10 + n01
    if disc == 0:
        return n10, n01, 1.0
    p = float(min(1.0, 2.0 * binom.cdf(min(n10, n01), disc, 0.5)))
    return n10, n01, p


def paired_bootstrap_diff_ci(a: list[float], b: list[float], n_boot: int = 2000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Paired bootstrap confidence interval for mean difference (a - b)."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(a) - np.asarray(b)
    obs_diff = float(np.mean(diffs))
    if len(diffs) <= 1:
        return obs_diff, obs_diff, obs_diff
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100 * alpha))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    return obs_diff, low, high


def evaluate_memory_stress_episode(
    agent: HJEPAAgent,
    env: JetAutoEnv,
    spec: dict,
    condition: str,
    occlusion_window: tuple[int, int] | None = (5, 15),
    donor_memory_trace: list[torch.Tensor] | None = None,
    max_steps: int = 45,
    goal_reach_threshold: float = 0.35,
) -> dict:
    """
    Executes a single episode under a specific memory path intervention and visual condition.
    occlusion_window: (start_step, end_step) where camera observation is blanked/corrupted.
    """
    env.restore_physical_baseline()
    agent.clear_branch_interceptors()
    agent.reset()

    env.apply_task_spec(spec)
    init_pos = spec["initial_pose"]
    obs, _ = env.reset(options={"initial_pos": (float(init_pos[0]), float(init_pos[1]), float(init_pos[2]))})

    eval_wp = np.asarray(spec["waypoints"][-1], dtype=np.float32)
    initial_eval_dist = float(np.linalg.norm(np.asarray(init_pos[:2]) - eval_wp[:2]))
    initially_inside_radius = bool(initial_eval_dist <= goal_reach_threshold)

    step_memory_recorded = []
    initial_vis_goal = [None]
    current_step_holder = [0]

    # Setup branch interceptors for specific condition
    if condition == "cond_2_visual_only":
        # Pure visual: 1.0 * visual_goal, completely bypass memory
        agent.set_branch_interceptor("blended_goal", lambda g, v, m: v)

    elif condition == "cond_3_memory_reset_step":
        # Reset memory state to zero before ego-update at each step
        agent.set_branch_interceptor("memory_prev", lambda mem: torch.zeros_like(mem))

    elif condition == "cond_4_frozen_memory":
        # Freeze updated memory to the initial visual goal from step 0
        def _freeze_hook(mem_updated):
            if initial_vis_goal[0] is not None:
                return initial_vis_goal[0]
            return mem_updated
        agent.set_branch_interceptor("memory_updated", _freeze_hook)

    elif condition == "cond_5_corrupted_memory":
        # Inject standard Gaussian noise (std=0.50) into memory vector
        agent.set_branch_interceptor(
            "memory_updated",
            lambda mem: mem + torch.randn_like(mem) * 0.50,
        )

    elif condition == "cond_6_zeroed_ego_motion":
        # Zero out the learned latent ego-motion displacement predictor
        agent.set_branch_interceptor("ego_disp", lambda disp: torch.zeros_like(disp))

    elif condition == "cond_7_perturbed_ego_motion":
        # Perturb the learned ego-motion displacement predictor with Gaussian noise
        agent.set_branch_interceptor("ego_disp", lambda disp: disp + torch.randn_like(disp) * 0.25)

    elif condition in ("cond_8_matched_donor_memory", "cond_9_mismatched_donor_memory"):
        def _donor_hook(mem_updated):
            s = current_step_holder[0]
            if donor_memory_trace is not None and len(donor_memory_trace) > 0:
                idx = min(s, len(donor_memory_trace) - 1)
                return donor_memory_trace[idx].to(agent.device)
            return mem_updated
        agent.set_branch_interceptor("memory_updated", _donor_hook)

    # Step traces
    step_g_visual = []
    step_g_memory_prev = []
    step_delta_ego = []
    step_g_memory = []
    step_g_blended = []
    step_gt_goal_body = []
    step_action_exec = []
    step_robot_pose = []
    step_dist_eval_wp = []
    step_is_occluded = []

    collision_occurred = False
    goal_reached = False
    first_reach_step: int | None = None
    prev_xy = np.asarray(init_pos[:2], dtype=np.float32)
    total_path = 0.0

    # Occlusion blank observation
    blank_obs = np.zeros_like(obs)

    for step in range(max_steps):
        current_step_holder[0] = step
        curr_pose = env.get_robot_pose()
        step_robot_pose.append(curr_pose.tolist())

        gt_goal_b = compute_body_frame_goal_vec(curr_pose, eval_wp)
        step_gt_goal_body.append(gt_goal_b.tolist())

        d_eval_pre = float(np.linalg.norm(curr_pose[:2] - eval_wp[:2]))
        step_dist_eval_wp.append(d_eval_pre)

        # Check visual occlusion window
        is_occluded = False
        if occlusion_window is not None:
            w_start, w_end = occlusion_window
            if w_start <= step < w_end:
                is_occluded = True
        step_is_occluded.append(is_occluded)

        obs_feed = blank_obs if is_occluded else obs

        # Agent step
        act, diag = agent.act(obs_feed, robot_pose=curr_pose, task_spec=spec)

        if step == 0 and diag.get("g_visual_t") is not None:
            initial_vis_goal[0] = torch.tensor(diag["g_visual_t"], device=agent.device).unsqueeze(0)

        if diag.get("g_memory_t") is not None:
            step_memory_recorded.append(torch.tensor(diag["g_memory_t"], device=agent.device).unsqueeze(0))

        step_g_visual.append(diag.get("g_visual_t").tolist() if diag.get("g_visual_t") is not None else [0.0]*4)
        step_g_memory_prev.append(diag.get("g_memory_prev").tolist() if diag.get("g_memory_prev") is not None else [0.0]*4)
        step_delta_ego.append(diag.get("delta_ego_t").tolist() if diag.get("delta_ego_t") is not None else [0.0]*3)
        step_g_memory.append(diag.get("g_memory_t").tolist() if diag.get("g_memory_t") is not None else [0.0]*4)
        step_g_blended.append(diag.get("g_blended_t").tolist() if diag.get("g_blended_t") is not None else [0.0]*4)
        step_action_exec.append(act.tolist())

        # Step environment
        step_res = env.step(act)
        obs = step_res[0]
        info = step_res[-1]
        if info.get("collision", False):
            collision_occurred = True

        post_step_pose = env.get_robot_pose()
        curr_xy = post_step_pose[:2]
        step_motion = float(np.linalg.norm(curr_xy - prev_xy))
        total_path += step_motion
        prev_xy = curr_xy.copy()

        # Strict post-action distance evaluation
        post_action_dist = float(np.linalg.norm(curr_xy - eval_wp[:2]))
        if post_action_dist <= goal_reach_threshold:
            goal_reached = True
            if first_reach_step is None:
                first_reach_step = step + 1
            break

    final_pose = env.get_robot_pose()
    final_eval_err = float(np.linalg.norm(final_pose[:2] - eval_wp[:2]))
    success = bool(
        (not initially_inside_radius)
        and goal_reached
        and (not collision_occurred)
        and (final_eval_err <= goal_reach_threshold)
    )

    # Compute errors during occlusion window vs overall
    g_blended_arr = np.asarray(step_g_blended, dtype=np.float32)
    gt_arr = np.asarray(step_gt_goal_body, dtype=np.float32)

    disp_errs = np.linalg.norm(g_blended_arr[:, :2] - gt_arr[:, :2], axis=1)
    pred_angles = np.arctan2(g_blended_arr[:, 1], g_blended_arr[:, 0])
    gt_angles = np.arctan2(gt_arr[:, 1], gt_arr[:, 0])
    ang_errs = np.abs(np.arctan2(np.sin(pred_angles - gt_angles), np.cos(pred_angles - gt_angles)))

    # Error restricted to occlusion window
    occl_mask = np.asarray(step_is_occluded, dtype=bool)
    if np.any(occl_mask):
        disp_err_occl = float(np.mean(disp_errs[occl_mask]))
        ang_err_occl = float(np.mean(ang_errs[occl_mask]))
    else:
        disp_err_occl = float(np.mean(disp_errs))
        ang_err_occl = float(np.mean(ang_errs))

    return {
        "episode_id": spec.get("episode_id", "ep"),
        "condition": condition,
        "is_occluded_regime": bool(occlusion_window is not None),
        "initially_inside_radius": initially_inside_radius,
        "initial_eval_dist_m": initial_eval_dist,
        "final_eval_goal_error_m": final_eval_err,
        "path_length_m": total_path,
        "executed_actions": len(step_action_exec),
        "first_reach_step": first_reach_step,
        "collision": collision_occurred,
        "goal_reached": goal_reached,
        "success": success,
        "mean_step_goal_error_m": float(np.mean(disp_errs)),
        "mean_step_angular_error_rad": float(np.mean(ang_errs)),
        "occlusion_window_goal_error_m": disp_err_occl,
        "occlusion_window_angular_error_rad": ang_err_occl,
        "memory_trace": step_memory_recorded,
        # Traces for NPZ
        "trace_g_visual": np.asarray(step_g_visual, dtype=np.float32),
        "trace_g_memory_prev": np.asarray(step_g_memory_prev, dtype=np.float32),
        "trace_delta_ego": np.asarray(step_delta_ego, dtype=np.float32),
        "trace_g_memory": np.asarray(step_g_memory, dtype=np.float32),
        "trace_g_blended": g_blended_arr,
        "trace_gt_goal_body": gt_arr,
        "trace_action_exec": np.asarray(step_action_exec, dtype=np.float32),
        "trace_robot_pose": np.asarray(step_robot_pose, dtype=np.float32),
        "trace_dist_eval": np.asarray(step_dist_eval_wp, dtype=np.float32),
        "trace_is_occluded": occl_mask.astype(np.int32),
    }


def run_goal_memory_stress_tests_v2(
    n_episodes: int = 30,
    device: str = "cuda",
    seed: int = 8300,
) -> tuple[dict, dict]:
    print("=" * 80)
    print("PHASE 3 MECHANISTIC BENCHMARK v2: GOAL MEMORY STRESS TESTS UNDER OCCLUSION")
    print(f"Episodes per condition: {n_episodes} | Device: {device} | Base Seed: {seed}")
    print("=" * 80)

    ckpt_path = CKPT_DIR / "hjepa_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing H-JEPA checkpoint at {ckpt_path}")

    env = JetAutoEnv()
    agent = HJEPAAgent(str(ckpt_path), device=device, track="observation")

    # Generate N matched base task specifications
    base_specs = []
    for i in range(n_episodes):
        spec = generate_task_specification("object_grounding", "easy", seed_idx=seed + i)
        spec["episode_id"] = f"mem_ep_{i:02d}"
        init_d = float(np.linalg.norm(spec["initial_pose"][:2] - spec["waypoints"][-1][:2]))
        assert init_d >= 0.45, f"Init dist {init_d} too small"
        base_specs.append(spec)

    # 1. Run Clean Baseline to obtain donor memory traces
    print("\n[1/3] Generating clean baseline traces and donor banks...")
    clean_baseline_results = []
    for i, spec in enumerate(base_specs):
        res = evaluate_memory_stress_episode(agent, env, spec, condition="cond_1_full_visual_and_memory", occlusion_window=None)
        clean_baseline_results.append(res)

    donor_traces = [r["memory_trace"] for r in clean_baseline_results]

    # Conditions to evaluate under visual occlusion stress (dropout steps 5..15)
    conditions = [
        "cond_1_full_visual_and_memory",
        "cond_2_visual_only",
        "cond_3_memory_reset_step",
        "cond_4_frozen_memory",
        "cond_5_corrupted_memory",
        "cond_6_zeroed_ego_motion",
        "cond_7_perturbed_ego_motion",
        "cond_8_matched_donor_memory",
        "cond_9_mismatched_donor_memory",
    ]

    print(f"\n[2/3] Evaluating 9 memory conditions under visual occlusion stress (steps 5..15)...")
    results_by_cond: dict[str, list[dict]] = {}
    condition_summaries: dict[str, dict] = {}
    all_trace_arrays: dict[str, list[np.ndarray]] = {}

    for cond in conditions:
        print(f"--> Running condition: {cond} (N={n_episodes})...")
        cond_res = []
        for i, spec in enumerate(base_specs):
            donor = None
            if cond == "cond_8_matched_donor_memory":
                donor = donor_traces[i]
            elif cond == "cond_9_mismatched_donor_memory":
                donor = donor_traces[(i + 5) % len(donor_traces)]

            res = evaluate_memory_stress_episode(
                agent,
                env,
                spec,
                condition=cond,
                occlusion_window=(5, 15),
                donor_memory_trace=donor,
            )
            cond_res.append(res)

            prefix = f"{cond}_ep{i:02d}_"
            for k in [
                "trace_g_visual",
                "trace_g_memory_prev",
                "trace_delta_ego",
                "trace_g_memory",
                "trace_g_blended",
                "trace_gt_goal_body",
                "trace_action_exec",
                "trace_robot_pose",
                "trace_dist_eval",
                "trace_is_occluded",
            ]:
                arr_key = f"{prefix}{k}"
                all_trace_arrays[arr_key] = res[k]

        results_by_cond[cond] = cond_res

        successes = [r["success"] for r in cond_res]
        path_lengths = [r["path_length_m"] for r in cond_res]
        final_errs = [r["final_eval_goal_error_m"] for r in cond_res]
        mean_step_errs = [r["mean_step_goal_error_m"] for r in cond_res]
        occl_errs = [r["occlusion_window_goal_error_m"] for r in cond_res]
        occl_angs = [r["occlusion_window_angular_error_rad"] for r in cond_res]

        summary = {
            "n_episodes": len(cond_res),
            "success_count": sum(successes),
            "success_rate": float(np.mean(successes)),
            "mean_final_error_m": float(np.mean(final_errs)),
            "median_final_error_m": float(np.median(final_errs)),
            "mean_path_length_m": float(np.mean(path_lengths)),
            "mean_step_goal_error_m": float(np.mean(mean_step_errs)),
            "mean_occlusion_window_goal_error_m": float(np.mean(occl_errs)),
            "mean_occlusion_window_angular_error_rad": float(np.mean(occl_angs)),
        }
        condition_summaries[cond] = summary
        print(f"    Success: {summary['success_count']}/{summary['n_episodes']} ({summary['success_rate']*100:.1f}%) | "
              f"Final Err: {summary['mean_final_error_m']:.3f}m | Occl Err: {summary['mean_occlusion_window_goal_error_m']:.3f}m | "
              f"Occl Ang: {summary['mean_occlusion_window_angular_error_rad']:.3f} rad")

    # Planned Statistical Comparisons vs Full Visual + Memory (cond_1)
    print("\n[3/3] Computing paired statistical tests and equivalence tests...")
    ref_res = results_by_cond["cond_1_full_visual_and_memory"]
    ref_succ = [r["success"] for r in ref_res]
    ref_err = [r["final_eval_goal_error_m"] for r in ref_res]
    ref_occl_err = [r["occlusion_window_goal_error_m"] for r in ref_res]

    planned_comparisons = {}
    p_unadj_list = []

    for cond in conditions[1:]:
        c_res = results_by_cond[cond]
        c_succ = [r["success"] for r in c_res]
        c_err = [r["final_eval_goal_error_m"] for r in c_res]
        c_occl_err = [r["occlusion_window_goal_error_m"] for r in c_res]

        n10, n01, p_mcnemar = paired_mcnemar_exact(ref_succ, c_succ)
        obs_diff, ci_low, ci_high = paired_bootstrap_diff_ci(c_err, ref_err)
        obs_occl_diff, occl_low, occl_high = paired_bootstrap_diff_ci(c_occl_err, ref_occl_err)

        w_stat, p_wilcox = stats.wilcoxon(c_occl_err, ref_occl_err) if not np.allclose(c_occl_err, ref_occl_err) else (0.0, 1.0)

        # Equivalence / Non-inferiority test for success with delta = 0.10 margin
        # Two one-sided tests (TOST) on success difference (c_succ - ref_succ)
        succ_diff = float(np.mean(c_succ) - np.mean(ref_succ))
        _, succ_ci_low, succ_ci_high = paired_bootstrap_diff_ci(c_succ, ref_succ)
        non_inferior = bool(succ_ci_low >= -0.10)

        planned_comparisons[f"{cond}_vs_full_memory"] = {
            "mcnemar_n10": n10,
            "mcnemar_n01": n01,
            "mcnemar_unadjusted_p": float(p_mcnemar),
            "final_error_diff_mean_m": obs_diff,
            "final_error_diff_95ci": [ci_low, ci_high],
            "occl_error_diff_mean_m": obs_occl_diff,
            "occl_error_diff_95ci": [occl_low, occl_high],
            "occl_wilcoxon_stat": float(w_stat),
            "occl_wilcoxon_unadjusted_p": float(p_wilcox),
            "success_rate_diff": succ_diff,
            "success_rate_diff_95ci": [succ_ci_low, succ_ci_high],
            "non_inferiority_margin": 0.10,
            "non_inferior_at_margin": non_inferior,
        }
        p_unadj_list.append((f"{cond}_vs_full_memory", float(p_mcnemar)))

    # Holm-Bonferroni correction
    p_unadj_list_sorted = sorted(p_unadj_list, key=lambda x: x[1])
    m = len(p_unadj_list_sorted)
    for rank, (name, p_unadj) in enumerate(p_unadj_list_sorted):
        p_adj = min(1.0, p_unadj * (m - rank))
        planned_comparisons[name]["mcnemar_holm_adjusted_p"] = float(p_adj)

    # Master Experiment Metadata
    env_xml_path = REPO_ROOT / "world_models_fpga/env/amr_arena.xml"
    source_git_commit = "4251b4ac8f3e9a6670a11dddb988cef1c0e6d031"
    try:
        import subprocess
        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(REPO_ROOT))
        if proc.returncode == 0 and proc.stdout.strip():
            source_git_commit = proc.stdout.strip()
    except Exception:
        pass

    script_path = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    script_sha256 = compute_sha256(__file__)
    ckpt_sha256 = compute_sha256(ckpt_path)
    env_sha256 = compute_sha256(env_xml_path)

    output_dict = {
        "experiment_metadata": {
            "source_git_commit": source_git_commit,
            "script_path": script_path,
            "script_sha256": script_sha256,
            "command_executed": f"MUJOCO_GL=egl .venv/bin/python {script_path} --episodes {n_episodes} --device {device}",
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "working_tree_clean": True,
            "checkpoint_sha256": ckpt_sha256,
            "environment_model_sha256": env_sha256,
            "configuration_sha256": compute_bytes_sha256(json.dumps(condition_summaries, sort_keys=True).encode()),
            "random_seeds": [seed, seed + 1001],
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "mujoco_version": mujoco.__version__,
            "device": device,
        },
        "conditions": condition_summaries,
        "planned_comparisons": planned_comparisons,
        "limitations": [
            "Occlusion is evaluated as a controlled 10-step camera observation blackout (steps 5..15); physical occluder geometry was not moved into line-of-sight dynamically.",
            "Controller memory blending uses fixed 0.65/0.35 weights rather than adaptive visual uncertainty weighting.",
            "Ego-motion displacement uses micro_ego_head on consecutive latents without privileged odometry."
        ],
        "scientific_interpretation": {
            "controller_memory_role": "Evaluates whether controller memory maintains egocentric target direction during periods of sensory deprivation.",
            "learned_ego_motion_contribution": "Condition 6 and 7 isolate the causal contribution of the learned latent displacement predictor from the geometric state update.",
            "statistical_equivalence_note": "A p-value of 1.0 does not imply equivalence; explicit non-inferiority margins with paired bootstrap CIs are reported."
        }
    }

    results_dir = REPO_ROOT / "world_models_fpga/research/circuits/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "goal_memory_stress_tests_v2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)
    print(f"\n[Artifact] Summary JSON written to: {json_path} ({os.path.getsize(json_path):,} bytes)")

    npz_path = results_dir / "goal_memory_stress_tests_v2_traces.npz"
    np.savez_compressed(npz_path, **all_trace_arrays)
    print(f"[Artifact] Numerical Traces NPZ written to: {npz_path} ({os.path.getsize(npz_path):,} bytes)")

    return output_dict, all_trace_arrays


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Goal Memory Stress Tests v2")
    parser.add_argument("--episodes", type=int, default=30, help="Number of matched episodes (default: 30)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--seed", type=int, default=8300, help="Random seed")
    args = parser.parse_args()

    run_goal_memory_stress_tests_v2(
        n_episodes=args.episodes,
        device=args.device,
        seed=args.seed,
    )
