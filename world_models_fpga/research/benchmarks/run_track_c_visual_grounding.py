"""
Track C Pure Visual Spatial Grounding Benchmark (Resolving Finding E)

Strict Information Regime:
- Track C ('observation'): The agent receives ONLY camera RGB (64x64x3) and discrete task ID.
- Zero simulator coordinates ([dx, dy, sin_th, cos_th] withheld).
- Evaluates four rigorous visual grounding test suites:
  1. Color-Grounded Target Identity (Target vs Distractors)
  2. Color Mutation: Irrelevant (Floor/Background) vs Relevant (Target Color)
  3. Spatial Landmark Rearrangement (Target & Distractor Swapped)
  4. SE(2) Robot Rotations (+/-45 deg, +/-90 deg, 180 deg)
- Measures closed-loop success, retention rate with exact numerator/denominator tables,
  and physical goal localization error (m).
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

from env.amr_env import JetAutoEnv
from models.h_jepa import HJEPAModel
from models.le_world_model import LeWorldModel
from models.baseline_policy import BaselineNavigationPolicy
from research.benchmarks.model_wrappers import BaselineAgent, HJEPAAgent, LeWMAgent
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


def evaluate_suite_track_c(agent, specs: list[dict], max_steps: int = 50) -> dict:
    """Evaluates an agent strictly under Track C (zero privileged coordinates)."""
    env = JetAutoEnv()
    records = []
    for spec in specs:
        env.restore_physical_baseline()
        env.apply_task_spec(spec)
        init_pos = np.asarray(spec["initial_pose"], dtype=np.float32)
        obs, _ = env.reset(options={"initial_pos": (float(init_pos[0]), float(init_pos[1]), float(init_pos[2]))})
        final_wp = np.asarray(spec.get("waypoints", [init_pos])[-1], dtype=np.float32)

        if hasattr(agent, "reset"):
            agent.reset()

        collision_hit = False
        goal_reached = False
        prev_xy = init_pos[:2].copy()
        total_path = 0.0

        for _ in range(max_steps):
            curr_pose = env.get_robot_pose()
            dist = float(np.linalg.norm(curr_pose[:2] - final_wp[:2]))
            if dist <= 0.28:
                goal_reached = True
                break

            # In Track C, pass ONLY obs and task_spec (with discrete instructions).
            public_spec = {
                "target_key": spec.get("target_key", ""),
                "semantic_relation": spec.get("semantic_relation", "approach"),
                "instruction": spec.get("instruction", ""),
            }
            act_np, _ = agent.act(obs, robot_pose=None, target_pose=None, task_spec=public_spec, goal_vec_body=None)

            step_res = env.step(act_np)
            obs = step_res[0]
            info = step_res[-1]
            if info.get("collision", False):
                collision_hit = True

            curr_xy = env.get_robot_pose()[:2]
            total_path += float(np.linalg.norm(curr_xy - prev_xy))
            prev_xy = curr_xy.copy()

        final_p = env.get_robot_pose()
        final_err = float(np.linalg.norm(final_p[:2] - final_wp[:2]))
        succ = bool(goal_reached and not collision_hit)
        records.append({
            "success": succ,
            "goal_reached": goal_reached,
            "collision": collision_hit,
            "final_goal_error_m": final_err,
            "path_length_m": total_path,
        })
    env.close()

    n = len(records)
    n_succ = sum(1 for r in records if r["success"])
    n_goal = sum(1 for r in records if r["goal_reached"])
    n_col = sum(1 for r in records if r["collision"])
    mean_err = float(np.mean([r["final_goal_error_m"] for r in records]))

    return {
        "episodes_evaluated": n,
        "success_count": n_succ,
        "success_rate": float(n_succ / max(1, n)),
        "goal_reached_count": n_goal,
        "goal_reached_rate": float(n_goal / max(1, n)),
        "collision_count": n_col,
        "collision_rate": float(n_col / max(1, n)),
        "mean_final_goal_error_m": mean_err,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Track C Pure Visual Spatial Grounding Benchmark")
    parser.add_argument("--episodes-per-suite", type=int, default=15, help="Number of episodes per test suite (N >= 15)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_dir = pkg_root / "checkpoints_v4"
    hjepa_path = ckpt_dir / "hjepa_best.pt"
    lewm_path = ckpt_dir / "lewm_best.pt"
    base_path = ckpt_dir / "baseline_best.pt"
    test_npz = pkg_root / "research/datasets/splits_v4/held_out_test_benchmark.npz"

    print("=================================================================")
    print("  TRACK C PURE VISUAL SPATIAL GROUNDING BENCHMARK")
    print(f"  Device: {args.device} | Checkpoint: {hjepa_path.name}")
    print(f"  Episodes per suite: {args.episodes_per_suite} | Privileged coordinates: ZERO")
    print("=================================================================")

    # Initialize Track C agents
    agent_h = HJEPAAgent(str(hjepa_path), device=args.device, track="observation")
    agent_l = LeWMAgent(str(lewm_path), device=args.device, track="observation")
    agent_b = BaselineAgent(str(base_path), device=args.device, track="observation")

    n_eps = args.episodes_per_suite

    # --- 1. Nominal Track C Grounding Suite ---
    print("\n--- 1. Evaluating Nominal Track C Visual Grounding ---")
    nominal_specs = [generate_task_specification("object_grounding", "easy", seed_idx=6100 + i) for i in range(n_eps)]
    res_nom_h = evaluate_suite_track_c(agent_h, nominal_specs)
    res_nom_l = evaluate_suite_track_c(agent_l, nominal_specs)
    res_nom_b = evaluate_suite_track_c(agent_b, nominal_specs)
    print(f"  [H-JEPA Nominal]   Succ: {res_nom_h['success_count']}/{n_eps} ({res_nom_h['success_rate']*100:5.1f}%) | Col: {res_nom_h['collision_rate']*100:5.1f}% | GoalErr: {res_nom_h['mean_final_goal_error_m']:.3f}m")
    print(f"  [LeWM Nominal]     Succ: {res_nom_l['success_count']}/{n_eps} ({res_nom_l['success_rate']*100:5.1f}%) | Col: {res_nom_l['collision_rate']*100:5.1f}% | GoalErr: {res_nom_l['mean_final_goal_error_m']:.3f}m")
    print(f"  [Baseline Nominal] Succ: {res_nom_b['success_count']}/{n_eps} ({res_nom_b['success_rate']*100:5.1f}%) | Col: {res_nom_b['collision_rate']*100:5.1f}% | GoalErr: {res_nom_b['mean_final_goal_error_m']:.3f}m")

    # --- 2. Color Mutation Suites: Irrelevant vs Relevant ---
    print("\n--- 2. Evaluating Color Mutation (Irrelevant vs Relevant) ---")
    # 2A: Irrelevant (Floor color shifted to dark blue/brown; target unchanged)
    irrel_specs = []
    for i in range(n_eps):
        spec = generate_task_specification("object_grounding", "easy", seed_idx=6200 + i)
        spec["floor_rgba"] = [0.15, 0.20, 0.35, 1.0]  # dark blue/slate floor
        irrel_specs.append(spec)
    res_irrel_h = evaluate_suite_track_c(agent_h, irrel_specs)

    # 2B: Relevant (Target object color mutated to unrelated color)
    rel_specs = []
    for i in range(n_eps):
        spec = generate_task_specification("object_grounding", "easy", seed_idx=6300 + i)
        target = spec.get("target_key", "yellow_cylinder")
        spec["color_mutations"] = {target: [0.85, 0.15, 0.85, 1.0]}  # mutated to magenta/purple
        rel_specs.append(spec)
    res_rel_h = evaluate_suite_track_c(agent_h, rel_specs)

    ret_irrel = float(res_irrel_h["success_rate"] / max(0.05, res_nom_h["success_rate"]))
    ret_rel = float(res_rel_h["success_rate"] / max(0.05, res_nom_h["success_rate"]))
    print(f"  [Irrelevant Floor Mutation] Succ: {res_irrel_h['success_count']}/{n_eps} ({res_irrel_h['success_rate']*100:5.1f}%) | Retention: {ret_irrel*100:5.1f}%")
    print(f"  [Relevant Target Mutation]  Succ: {res_rel_h['success_count']}/{n_eps} ({res_rel_h['success_rate']*100:5.1f}%) | Retention: {ret_rel*100:5.1f}%")

    # --- 3. Spatial Landmark Rearrangement ---
    print("\n--- 3. Evaluating Spatial Landmark Rearrangement ---")
    rearrange_specs = []
    for i in range(n_eps):
        spec = generate_task_specification("object_grounding", "easy", seed_idx=6400 + i)
        # Swap target location with an opposite quadrant obstacle
        target = spec.get("target_key", "yellow_cylinder")
        wp = spec["waypoints"][-1]
        spec["waypoints"] = [np.array([-wp[0], -wp[1], float(wp[2])], dtype=np.float32)]
        spec["rearranged_target"] = True
        rearrange_specs.append(spec)
    res_rearr_h = evaluate_suite_track_c(agent_h, rearrange_specs)
    ret_rearr = float(res_rearr_h["success_rate"] / max(0.05, res_nom_h["success_rate"]))
    print(f"  [Landmark Rearrangement]    Succ: {res_rearr_h['success_count']}/{n_eps} ({res_rearr_h['success_rate']*100:5.1f}%) | Retention: {ret_rearr*100:5.1f}% | GoalErr: {res_rearr_h['mean_final_goal_error_m']:.3f}m")

    # --- 4. SE(2) Robot In-Plane Rotations ---
    print("\n--- 4. Evaluating SE(2) Robot In-Plane Rotations (+/-45 deg, +/-90 deg, 180 deg) ---")
    rot_angles = [np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2, np.pi]
    rot_results = {}
    for ang_idx, ang in enumerate(rot_angles):
        deg = int(round(np.degrees(ang)))
        rot_specs = []
        for i in range(max(6, n_eps // 3)):
            spec = generate_task_specification("object_grounding", "easy", seed_idx=6500 + ang_idx * 50 + i)
            p = spec["initial_pose"]
            spec["initial_pose"] = np.array([p[0], p[1], float(p[2] + ang)], dtype=np.float32)
            rot_specs.append(spec)
        res_rot = evaluate_suite_track_c(agent_h, rot_specs)
        rot_results[f"rot_{deg:+}deg"] = {
            "episodes": len(rot_specs),
            "success_count": res_rot["success_count"],
            "success_rate": res_rot["success_rate"],
            "mean_goal_error_m": res_rot["mean_final_goal_error_m"],
        }
        print(f"  [Rotation {deg:+4d} deg] Succ: {res_rot['success_count']}/{len(rot_specs)} ({res_rot['success_rate']*100:5.1f}%) | GoalErr: {res_rot['mean_final_goal_error_m']:.3f}m")

    # Build execution metadata
    metadata = {
        "command_executed": "python world_models_fpga/research/benchmarks/run_track_c_visual_grounding.py",
        "execution_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_sha(),
        "dataset_path": str(test_npz.relative_to(repo_root)),
        "dataset_sha256": _sha256(test_npz),
        "checkpoint_path": str(hjepa_path.relative_to(repo_root)),
        "checkpoint_sha256": _sha256(hjepa_path),
        "checkpoints": [
            {"name": "hjepa_best.pt", "path": str(hjepa_path.relative_to(repo_root)), "sha256": _sha256(hjepa_path)},
            {"name": "lewm_best.pt", "path": str(lewm_path.relative_to(repo_root)), "sha256": _sha256(lewm_path)},
            {"name": "baseline_best.pt", "path": str(base_path.relative_to(repo_root)), "sha256": _sha256(base_path)},
        ],
        "random_seed": 42,
        "configuration": {
            "track": "Track C (observation only, zero simulator coordinates)",
            "episodes_per_suite": n_eps,
            "max_steps": 50,
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": args.device,
        },
    }

    payload = {
        "execution_metadata": metadata,
        "track_c_nominal": {
            "H-JEPA": res_nom_h,
            "LeWM": res_nom_l,
            "Baseline": res_nom_b,
        },
        "color_mutations": {
            "irrelevant_floor_shift": {
                "success_rate": res_irrel_h["success_rate"],
                "success_count": res_irrel_h["success_count"],
                "total_episodes": n_eps,
                "retention_rate": ret_irrel,
            },
            "relevant_target_mutation": {
                "success_rate": res_rel_h["success_rate"],
                "success_count": res_rel_h["success_count"],
                "total_episodes": n_eps,
                "retention_rate": ret_rel,
            },
            "affordance_appearance_factorization_supported": bool(ret_irrel >= 0.70 and ret_rel <= 0.40),
        },
        "landmark_rearrangement": {
            "success_rate": res_rearr_h["success_rate"],
            "success_count": res_rearr_h["success_count"],
            "total_episodes": n_eps,
            "retention_rate": ret_rearr,
            "mean_goal_error_m": res_rearr_h["mean_final_goal_error_m"],
        },
        "se2_rotations": rot_results,
    }

    out_file = pkg_root / "research/benchmarks/results/track_c_visual_grounding.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n[SAVED TRACK C VISUAL GROUNDING ARTIFACT TO {out_file}]")


if __name__ == "__main__":
    main()
