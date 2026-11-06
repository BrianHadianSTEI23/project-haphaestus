"""
Phase 8: Architecture Synthesis & Compositional Intelligence
Script: run_novel_composition.py
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Experiment C: Novel Task Composition.
Tests whether the reconstructed compositional architecture can execute multi-stage
sequential language instructions without having been trained on that exact combination:
  - Task C1: "Go behind yellow cylinder, then move left of red box." (2 Stages)
  - Task C2: "Move right of obstacle, then go behind green cylinder." (2 Stages)
  - Task C3: "Move in front of cyan box, then right of blue cylinder, then behind yellow cylinder." (3 Stages)
Evaluates closed-loop execution in MuJoCo simulation against Baseline Policy.
"""

import os
import sys
import json
import csv
import time
import numpy as np
import torch
import mujoco

os.environ["MUJOCO_GL"] = "egl"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from env.amr_env import JetAutoEnv
from research.benchmarks.model_wrappers import BaselineAgent
from research.synthesis.compositional_architecture.compositional_agent import CompositionalNavigationAgent
from research.datasets.generation.capability_tasks import CANONICAL_OBSTACLES, get_waypoint_for_relation

def evaluate_multi_stage_episode(env, agent, waypoints, max_steps_per_stage=60):
    """
    Executes a multi-stage sequential episode in MuJoCo.
    Transitions to the next stage when the robot is within 0.28m of current waypoint.
    """
    obs, info = env.reset(options={"initial_pos": (-1.2, -1.2, 0.0)})
    num_stages = len(waypoints)
    current_stage = 0
    stage_completions = [False] * num_stages
    total_collisions = 0
    trajectory = [env.get_robot_pose()[:2].copy()]

    # If agent supports explicit subgoal queue
    if hasattr(agent, "set_subgoal_sequence"):
        agent.set_subgoal_sequence(waypoints)

    total_steps = max_steps_per_stage * num_stages

    for t in range(total_steps):
        curr_pose = env.get_robot_pose()
        target_wp = waypoints[current_stage]

        dist_to_wp = np.linalg.norm(target_wp[:2] - curr_pose[:2])
        if dist_to_wp < 0.28:
            stage_completions[current_stage] = True
            if current_stage < num_stages - 1:
                current_stage += 1
            else:
                # All stages completed!
                break

        action = agent.select_action(obs, curr_pose, target_wp)
        obs, reward, term, trunc, step_info = env.step(action)
        if step_info["collision"]:
            total_collisions += 1

        new_pose = env.get_robot_pose()
        trajectory.append(new_pose[:2].copy())

    # Full success: all stages reached and no collisions
    all_stages_done = all(stage_completions)
    task_success = all_stages_done and (total_collisions == 0)

    # Path length
    actual_path_len = sum(np.linalg.norm(trajectory[i+1] - trajectory[i]) for i in range(len(trajectory)-1))
    geodesic = sum(np.linalg.norm(waypoints[i][:2] - (trajectory[0] if i==0 else waypoints[i-1][:2])) for i in range(num_stages))
    efficiency = float(geodesic / (actual_path_len + 1e-6)) if actual_path_len > 0.1 else 1.0

    return {
        "task_success": task_success,
        "all_stages_done": all_stages_done,
        "stages_completed": sum(stage_completions),
        "stage_completion_rates": stage_completions,
        "collisions": total_collisions,
        "path_efficiency": min(1.0, max(0.0, efficiency)),
        "steps": len(trajectory) - 1
    }

def run_experiment_c():
    out_dir = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    base_dir = os.path.abspath(os.path.join(out_dir, "../../../"))
    ckpt_baseline = os.path.join(base_dir, "checkpoints/baseline_best.pt")
    ckpt_optB = os.path.abspath(os.path.join(out_dir, "../../extraction/extracted_model/reconstructed_compact/model_reconstructed_compact.pt"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n=======================================================")
    print("  PHASE 8 - EXPERIMENT C: NOVEL TASK COMPOSITION")
    print("=======================================================\n")

    env = JetAutoEnv()

    # Define Compositional Tasks
    # 1. Yellow cylinder at (0.8, 0.8), Red box at (-0.8, 0.8), Green cyl at (0.0, 1.2), Cyan box at (-0.8, -0.8)
    wp_yellow_behind = get_waypoint_for_relation(np.array([0.8, 0.8, 0.2]), "behind", np.array([-1.2, -1.2, 0.0]), clearance=0.55)
    wp_red_left = get_waypoint_for_relation(np.array([-0.8, 0.8, 0.2]), "left_of", wp_yellow_behind, clearance=0.55)

    wp_green_behind = get_waypoint_for_relation(np.array([0.0, 1.0, 0.2]), "behind", np.array([-1.2, -1.2, 0.0]), clearance=0.55)
    wp_obs_right = np.array([0.4, 0.2, 0.0], dtype=np.float32)

    wp_cyan_front = get_waypoint_for_relation(np.array([-0.6, -0.6, 0.2]), "in_front", np.array([-1.2, -1.2, 0.0]), clearance=0.50)
    wp_blue_right = get_waypoint_for_relation(np.array([0.6, -0.6, 0.2]), "right_of", wp_cyan_front, clearance=0.50)

    compositional_suites = {
        "Task_C1_2Stages": {
            "name": "Task C1: Behind Yellow Cylinder -> Left of Red Box",
            "waypoints": [wp_yellow_behind, wp_red_left],
            "num_stages": 2
        },
        "Task_C2_2Stages": {
            "name": "Task C2: Right of Obstacle -> Behind Green Cylinder",
            "waypoints": [wp_obs_right, wp_green_behind],
            "num_stages": 2
        },
        "Task_C3_3Stages": {
            "name": "Task C3: Front of Cyan Box -> Right of Blue Cyl -> Behind Yellow Cyl",
            "waypoints": [wp_cyan_front, wp_blue_right, wp_yellow_behind],
            "num_stages": 3
        }
    }

    models = {
        "baseline": {
            "name": "Monolithic Baseline Policy",
            "agent": BaselineAgent(ckpt_baseline, device=device)
        },
        "compositional_synthesized": {
            "name": "Reconstructed Compositional Architecture",
            "agent": CompositionalNavigationAgent(compact_ckpt=ckpt_optB, device=device)
        }
    }

    episodes_per_task = 10
    results = {}

    for t_key, t_info in compositional_suites.items():
        t_name = t_info["name"]
        wps = t_info["waypoints"]
        n_stg = t_info["num_stages"]
        print(f"\n--- Testing on {t_name} ({episodes_per_task} runs each) ---")
        results[t_key] = {"task_name": t_name, "num_stages": n_stg, "models": {}}

        for m_key, m_info in models.items():
            agent = m_info["agent"]
            m_name = m_info["name"]

            full_success = 0
            all_stages_reached = 0
            stage_reach_counts = [0] * n_stg
            total_cols = 0
            path_effs = []

            for ep in range(episodes_per_task):
                res = evaluate_multi_stage_episode(env, agent, wps, max_steps_per_stage=60)
                if res["task_success"]: full_success += 1
                if res["all_stages_done"]: all_stages_reached += 1
                for s_idx, done in enumerate(res["stage_completion_rates"]):
                    if done: stage_reach_counts[s_idx] += 1
                total_cols += res["collisions"]
                path_effs.append(res["path_efficiency"])

            succ_rate = (full_success / episodes_per_task) * 100.0
            reach_rate = (all_stages_reached / episodes_per_task) * 100.0
            stg_rates = [(c / episodes_per_task) * 100.0 for c in stage_reach_counts]
            avg_cols = total_cols / episodes_per_task
            avg_eff = float(np.mean(path_effs))

            print(f"[{m_name}] Success (No Col): {succ_rate:.1f}% | All Waypoints Reached: {reach_rate:.1f}% | Stages: {stg_rates} | Avg Cols: {avg_cols:.1f} | Path Eff: {avg_eff:.3f}")

            results[t_key]["models"][m_key] = {
                "model_name": m_name,
                "full_success_pct": round(succ_rate, 2),
                "all_stages_reached_pct": round(reach_rate, 2),
                "stage_completion_pcts": [round(s, 1) for s in stg_rates],
                "avg_collisions": round(avg_cols, 2),
                "path_efficiency": round(avg_eff, 4)
            }

    # Save JSON
    json_path = os.path.join(out_dir, "composition_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved JSON] {json_path}")

    # Save CSV
    csv_path = os.path.join(out_dir, "composition_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Stages", "Model", "Full Success (%)", "All Stages Reached (%)", "Stage 1 (%)", "Stage 2 (%)", "Stage 3 (%)", "Avg Collisions", "Path Efficiency"])
        for t_key, t_val in results.items():
            for m_key in ["baseline", "compositional_synthesized"]:
                m = t_val["models"][m_key]
                stg = m["stage_completion_pcts"]
                s1 = f"{stg[0]:.1f}%" if len(stg) > 0 else "-"
                s2 = f"{stg[1]:.1f}%" if len(stg) > 1 else "-"
                s3 = f"{stg[2]:.1f}%" if len(stg) > 2 else "-"
                writer.writerow([t_val["task_name"], t_val["num_stages"], m["model_name"], f"{m['full_success_pct']:.1f}%", f"{m['all_stages_reached_pct']:.1f}%", s1, s2, s3, m["avg_collisions"], m["path_efficiency"]])
    print(f"[Saved CSV]  {csv_path}")

if __name__ == "__main__":
    run_experiment_c()
