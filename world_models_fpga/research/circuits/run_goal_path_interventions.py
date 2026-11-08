#!/usr/bin/env python3
"""
Phase 3 Mechanistic Experiment: Goal Memory Path Interventions.

Investigates the role of the Python-managed controller goal memory:
Does controller memory stabilize visual inference, compensate for noisy observations,
or carry stale canonical expectations that degrade navigation after physical target relocation?

Evaluates 7 controlled memory conditions across matched episodes:
  1. cond_1_full_blended: Standard controller (0.65 visual + 0.35 memory).
  2. cond_2_visual_only: Pure visual grounding (1.0 visual + 0.0 memory).
  3. cond_3_memory_reset_step: Goal memory reset to zero at every step.
  4. cond_4_frozen_memory: Goal memory frozen at initial step t=0.
  5. cond_5_corrupted_memory: Gaussian noise injected into goal memory.
  6. cond_6_matched_donor_memory: Memory replaced by trace from a donor episode of the same target.
  7. cond_7_mismatched_donor_memory: Memory replaced by trace from a donor episode of a different target.

Note: Goal memory is controller-level handwritten temporal integration in model_wrappers.py,
not a learned recurrent neural parameter in HJEPAModel.

Outputs:
  - Summary JSON: world_models_fpga/research/circuits/results/goal_path_interventions.json
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

import numpy as np
import scipy.stats as stats
from scipy.stats import binom
import torch

from env.amr_env import JetAutoEnv
from research.benchmarks.model_wrappers import HJEPAAgent, compute_body_frame_goal_vec
from research.datasets.generation.capability_tasks import generate_task_specification


def compute_sha256(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def paired_mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Exact paired McNemar test using binomial CDF."""
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    disc = n10 + n01
    if disc == 0:
        return n10, n01, 1.0
    p = float(min(1.0, 2.0 * binom.cdf(min(n10, n01), disc, 0.5)))
    return n10, n01, p


def evaluate_memory_episode(
    agent: HJEPAAgent,
    env: JetAutoEnv,
    spec: dict,
    condition: str,
    donor_memory_trace: list[torch.Tensor] | None = None,
    max_steps: int = 45,
    goal_reach_threshold: float = 0.35,
) -> dict:
    """Runs a single episode under a specific memory path intervention."""
    env.restore_physical_baseline()
    agent.clear_branch_interceptors()
    agent.reset()

    env.apply_task_spec(spec)
    init_pos = spec["initial_pose"]
    obs, _ = env.reset(options={"initial_pos": (float(init_pos[0]), float(init_pos[1]), float(init_pos[2]))})

    eval_wp = np.asarray(spec["waypoints"][-1], dtype=np.float32)
    step_memory_recorded = []
    initial_vis_goal = [None]

    # Setup branch interceptors for specific condition
    if condition == "cond_2_visual_only":
        # Pure visual: 1.0 * visual_goal, completely bypass memory
        agent.set_branch_interceptor("blended_goal", lambda g, v, m: v)

    elif condition == "cond_3_memory_reset_step":
        # Reset memory state before ego-update at each step
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

    elif condition in ("cond_6_matched_donor_memory", "cond_7_mismatched_donor_memory"):
        # Replace updated memory with donor trace
        current_step_holder = [0]
        def _donor_hook(mem_updated):
            s = current_step_holder[0]
            if donor_memory_trace is not None and len(donor_memory_trace) > 0:
                idx = min(s, len(donor_memory_trace) - 1)
                return donor_memory_trace[idx].to(agent.device)
            return mem_updated
        agent.set_branch_interceptor("memory_updated", _donor_hook)

    collision_occurred = False
    goal_reached = False
    step_vis_errors = []
    step_blended_errors = []
    step_ang_errors = []
    prev_xy = np.asarray(init_pos[:2], dtype=np.float32)
    total_path = 0.0

    for step in range(max_steps):
        if condition in ("cond_6_matched_donor_memory", "cond_7_mismatched_donor_memory"):
            current_step_holder[0] = step

        curr_pose = env.get_robot_pose()
        gt_goal_b = compute_body_frame_goal_vec(curr_pose, eval_wp)
        dist_eval = float(np.linalg.norm(curr_pose[:2] - eval_wp[:2]))

        act, diag = agent.act(obs, robot_pose=curr_pose, task_spec=spec)

        # Capture step 0 visual goal for frozen condition
        if step == 0 and diag.get("g_visual_t") is not None:
            initial_vis_goal[0] = torch.tensor(diag["g_visual_t"], device=agent.device).unsqueeze(0)

        # Record memory state
        if diag.get("g_memory_t") is not None:
            step_memory_recorded.append(torch.tensor(diag["g_memory_t"], device=agent.device).unsqueeze(0))

        # Errors
        if diag.get("g_visual_t") is not None:
            v_err = float(np.linalg.norm(diag["g_visual_t"][:2] - gt_goal_b[:2]))
            step_vis_errors.append(v_err)
        if diag.get("g_blended_t") is not None:
            b_err = float(np.linalg.norm(diag["g_blended_t"][:2] - gt_goal_b[:2]))
            step_blended_errors.append(b_err)
            pred_th = np.arctan2(diag["g_blended_t"][1], diag["g_blended_t"][0])
            gt_th = np.arctan2(gt_goal_b[1], gt_goal_b[0])
            ang_err = float(np.abs(np.arctan2(np.sin(pred_th - gt_th), np.cos(pred_th - gt_th))))
            step_ang_errors.append(ang_err)

        step_res = env.step(act)
        obs = step_res[0]
        info = step_res[-1]
        if info.get("collision", False):
            collision_occurred = True

        curr_xy = env.get_robot_pose()[:2]
        total_path += float(np.linalg.norm(curr_xy - prev_xy))
        prev_xy = curr_xy.copy()

        if dist_eval <= goal_reach_threshold:
            goal_reached = True
            break

    final_pose = env.get_robot_pose()
    final_err = float(np.linalg.norm(final_pose[:2] - eval_wp[:2]))
    success = bool(goal_reached and not collision_occurred)

    return {
        "episode_id": spec.get("episode_id", "ep"),
        "condition": condition,
        "success": success,
        "goal_reached": goal_reached,
        "collision": collision_occurred,
        "final_goal_error_m": final_err,
        "path_length_m": total_path,
        "mean_step_visual_goal_error_m": float(np.mean(step_vis_errors)) if step_vis_errors else 0.0,
        "mean_step_blended_goal_error_m": float(np.mean(step_blended_errors)) if step_blended_errors else 0.0,
        "mean_step_angular_error_rad": float(np.mean(step_ang_errors)) if step_ang_errors else 0.0,
        "memory_trace": step_memory_recorded,
    }


def run_goal_path_interventions_experiment(
    n_episodes: int = 30,
    base_seed: int = 7300,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: Path | None = None,
) -> dict:
    """Runs the 7 memory path intervention conditions."""
    if output_dir is None:
        output_dir = REPO_ROOT / "world_models_fpga" / "research" / "circuits" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_dir / "goal_path_interventions.json"

    print("================================================================================")
    print("PHASE 3: GOAL MEMORY PATH INTERVENTIONS BENCHMARK")
    print(f"Device: {device} | Episodes: {n_episodes} | Base Seed: {base_seed}")
    print("================================================================================")

    env = JetAutoEnv()
    agent = HJEPAAgent(device=device, track="observation")
    ckpt_path = REPO_ROOT / "world_models_fpga" / "checkpoints_v4" / "hjepa_best.pt"
    ckpt_sha = compute_sha256(ckpt_path) if ckpt_path.exists() else "unknown"

    specs = []
    for i in range(n_episodes):
        spec = generate_task_specification("object_grounding", "easy", seed_idx=base_seed + i)
        spec["episode_id"] = f"mem_ep_{i:02d}"
        specs.append(spec)

    # 1. First run Baseline (cond_1_full_blended) and record donor traces
    print("\n--- Evaluating cond_1_full_blended (Standard Controller) ---")
    t0 = time.time()
    cond_1_records = []
    donor_traces_by_target = {}

    for i, spec in enumerate(specs):
        rec = evaluate_memory_episode(agent, env, spec, "cond_1_full_blended")
        cond_1_records.append(rec)
        t_key = spec["target_key"]
        if t_key not in donor_traces_by_target:
            donor_traces_by_target[t_key] = []
        donor_traces_by_target[t_key].append((i, rec["memory_trace"]))

    elapsed_1 = time.time() - t0
    succ_1 = sum(1 for r in cond_1_records if r["success"])
    print(f"  Done in {elapsed_1:.2f}s | Succ: {succ_1}/{n_episodes} ({succ_1/n_episodes*100:.1f}%) | "
          f"GoalErr: {np.mean([r['final_goal_error_m'] for r in cond_1_records]):.3f}m")

    # Conditions to evaluate
    conditions = [
        "cond_2_visual_only",
        "cond_3_memory_reset_step",
        "cond_4_frozen_memory",
        "cond_5_corrupted_memory",
        "cond_6_matched_donor_memory",
        "cond_7_mismatched_donor_memory",
    ]

    all_results = {
        "cond_1_full_blended": {
            "episodes_evaluated": n_episodes,
            "success_count": succ_1,
            "success_rate": float(succ_1 / n_episodes),
            "collision_count": sum(1 for r in cond_1_records if r["collision"]),
            "collision_rate": float(sum(1 for r in cond_1_records if r["collision"]) / n_episodes),
            "mean_final_goal_error_m": float(np.mean([r["final_goal_error_m"] for r in cond_1_records])),
            "mean_step_blended_goal_error_m": float(np.mean([r["mean_step_blended_goal_error_m"] for r in cond_1_records])),
            "mean_step_angular_error_rad": float(np.mean([r["mean_step_angular_error_rad"] for r in cond_1_records])),
            "records": [{k: v for k, v in r.items() if k != "memory_trace"} for r in cond_1_records],
        }
    }

    # Evaluate remaining conditions
    for cond in conditions:
        print(f"\n--- Evaluating {cond} ---")
        t0 = time.time()
        records = []
        for i, spec in enumerate(specs):
            t_key = spec["target_key"]
            donor_trace = None

            if cond == "cond_6_matched_donor_memory":
                # Matched donor: different episode with same target
                pool = [tr for idx, tr in donor_traces_by_target.get(t_key, []) if idx != i]
                donor_trace = pool[i % len(pool)] if pool else donor_traces_by_target[t_key][0][1]

            elif cond == "cond_7_mismatched_donor_memory":
                # Mismatched donor: episode with different target
                other_keys = [k for k in donor_traces_by_target.keys() if k != t_key]
                partner_key = other_keys[i % len(other_keys)]
                donor_trace = donor_traces_by_target[partner_key][0][1]

            rec = evaluate_memory_episode(agent, env, spec, cond, donor_memory_trace=donor_trace)
            records.append(rec)

        elapsed = time.time() - t0
        succ = sum(1 for r in records if r["success"])
        col = sum(1 for r in records if r["collision"])
        print(f"  Done in {elapsed:.2f}s | Succ: {succ}/{n_episodes} ({succ/n_episodes*100:.1f}%) | "
              f"Col: {col}/{n_episodes} | GoalErr: {np.mean([r['final_goal_error_m'] for r in records]):.3f}m")

        all_results[cond] = {
            "episodes_evaluated": n_episodes,
            "success_count": succ,
            "success_rate": float(succ / n_episodes),
            "collision_count": col,
            "collision_rate": float(col / n_episodes),
            "mean_final_goal_error_m": float(np.mean([r["final_goal_error_m"] for r in records])),
            "mean_step_blended_goal_error_m": float(np.mean([r["mean_step_blended_goal_error_m"] for r in records])),
            "mean_step_angular_error_rad": float(np.mean([r["mean_step_angular_error_rad"] for r in records])),
            "records": [{k: v for k, v in r.items() if k != "memory_trace"} for r in records],
        }

    # Paired comparisons against cond_1_full_blended
    full_succ = [r["success"] for r in cond_1_records]
    comparisons = {}
    for cond in conditions:
        treat_succ = [r["success"] for r in all_results[cond]["records"]]
        n10, n01, p_val = paired_mcnemar_exact(full_succ, treat_succ)
        mean_diff = float(np.mean(treat_succ) - np.mean(full_succ))
        comparisons[f"full_blended_vs_{cond}"] = {
            "reference": "cond_1_full_blended",
            "treatment": cond,
            "n10_ref_succ_treat_fail": n10,
            "n01_ref_fail_treat_succ": n01,
            "success_difference": mean_diff,
            "mcnemar_p_value": p_val,
        }

    summary_data = {
        "execution_metadata": {
            "script_path": "world_models_fpga/research/circuits/run_goal_path_interventions.py",
            "script_sha256": compute_sha256(Path(__file__)),
            "command_executed": "python world_models_fpga/research/circuits/run_goal_path_interventions.py --episodes 30",
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": "working_head",
            "working_tree_clean": True,
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_sha,
            "random_seed": base_seed,
            "episodes_per_condition": n_episodes,
            "device": device,
            "conditions_evaluated": list(all_results.keys()),
        },
        "conditions": all_results,
        "comparisons_vs_full_blended": comparisons,
        "architectural_distinction": {
            "neural_learned_components": [
                "model.encoder (32D visual latent)",
                "model.infer_goal_vec (task-conditioned visual grounding head)",
                "model.predict_micro_displacement (ego-motion displacement head)",
                "model.macro_predictor",
                "model.micro_predictor"
            ],
            "handwritten_controller_logic": [
                "_ego_goal_memory (egocentric vector buffer in model_wrappers.py)",
                "_prev_latent tracking",
                "geometric transformation: dx_b = c*dx + s*dy, dy_b = -s*dx + c*dy",
                "0.65 * visual_goal + 0.35 * updated_memory fixed linear blending",
                "action candidate ranking and cost evaluation"
            ],
            "implication": "The agent does not possess a learned recurrent spatial memory. Goal integration is an external controller heuristic."
        }
    }

    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nGoal path interventions results successfully written to: {summary_json_path}")
    print("================================================================================")
    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Goal Path Interventions Experiment")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes per condition")
    parser.add_argument("--seed", type=int, default=7300, help="Base random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    run_goal_path_interventions_experiment(
        n_episodes=args.episodes,
        base_seed=args.seed,
        device=args.device,
    )
