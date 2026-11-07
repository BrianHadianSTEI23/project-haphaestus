"""
Phase 8: Architecture Synthesis & Compositional Intelligence
Script: run_circuit_recombination.py
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Experiment A: Circuit Recombination.
Combines individually validated circuits (Spatial Grounding 001, Safety Barrier 002,
Macro Planning 003, Egocentric Steering 004, Action Foresight 005) into a modular
recombined pipeline not trained end-to-end in this configuration.
Evaluates in closed-loop MuJoCo simulation against monolithic models.
"""

import sys as _hsys
from pathlib import Path as _HPath
_hsys.path.insert(0, str(_HPath(__file__).resolve().parents[3]))
from research.datasets.dataset_config import SPLITS_DIR as _ACTIVE_SPLITS_DIR, CKPT_DIR as _ACTIVE_CKPT_DIR  # noqa: E402
import os
import sys
import json
import csv
import time
import numpy as np
import torch

os.environ["MUJOCO_GL"] = "egl"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from research.benchmarks.evaluator import BenchmarkEvaluator
from research.benchmarks.model_wrappers import BaselineAgent, HJEPAAgent
from research.synthesis.compositional_architecture.compositional_agent import CompositionalNavigationAgent
from research.datasets.generation.capability_tasks import generate_task_specification

def run_experiment_a():
    out_dir = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    ckpt_dir = str(_ACTIVE_CKPT_DIR)
    optB_ckpt = os.path.abspath(os.path.join(out_dir, "../../extraction/extracted_model/reconstructed_compact/model_reconstructed_compact.pt"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n=======================================================")
    print("  PHASE 8 - EXPERIMENT A: CIRCUIT RECOMBINATION")
    print("=======================================================\n")

    # Evaluator
    evaluator = BenchmarkEvaluator()

    # Agents to compare
    agents = {
        "baseline": {
            "name": "Monolithic Baseline Policy",
            "agent": BaselineAgent(os.path.join(ckpt_dir, "baseline_best.pt"), device=device)
        },
        "hjepa_monolithic": {
            "name": "Monolithic H-JEPA (Reference)",
            "agent": HJEPAAgent(os.path.join(ckpt_dir, "hjepa_best.pt"), device=device)
        },
        "recombined_compositional": {
            "name": "Recombined Compositional Architecture",
            "agent": CompositionalNavigationAgent(compact_ckpt=optB_ckpt, device=device)
        }
    }

    num_episodes = 20
    test_tasks = []
    # Generate tasks requiring simultaneous spatial grounding and obstacle avoidance
    for i in range(num_episodes):
        diff = "difficult" if i % 2 == 0 else "easy"
        spec = generate_task_specification("spatial_reasoning", diff, seed_idx=5000 + i)
        spec["episode_id"] = f"recomb_task_{i:02d}"
        test_tasks.append(spec)

    results = {}

    for ag_id, ag_info in agents.items():
        agent = ag_info["agent"]
        name = ag_info["name"]
        print(f"Evaluating {name} on {num_episodes} closed-loop episodes...")

        success_count = 0
        collision_count = 0
        path_efficiencies = []
        final_errors = []
        latencies = []

        for spec in test_tasks:
            metrics = evaluator.evaluate_episode(agent, spec, max_steps=40)
            if metrics["task_success"]:
                success_count += 1
            collision_count += metrics["collision_count"]
            path_efficiencies.append(metrics["path_efficiency"])
            final_errors.append(metrics["final_distance"])
            latencies.append(metrics["mean_latency_us"])

        succ_rate = (success_count / num_episodes) * 100.0
        col_rate = (collision_count / num_episodes)
        mean_eff = float(np.mean(path_efficiencies))
        mean_err = float(np.mean(final_errors))
        lat_mean = float(np.mean(latencies))

        print(f"[{name}] Success: {succ_rate:.1f}% | Collisions/Ep: {col_rate:.2f} | Path Eff: {mean_eff:.3f} | Goal Err: {mean_err:.3f}m | Latency: {lat_mean:.1f}us")

        results[ag_id] = {
            "name": name,
            "success_rate_pct": round(succ_rate, 2),
            "collisions_per_episode": round(col_rate, 2),
            "path_efficiency": round(mean_eff, 4),
            "goal_error_m": round(mean_err, 4),
            "mean_latency_us": round(lat_mean, 1)
        }

    # Save JSON
    json_path = os.path.join(out_dir, "recombination_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved JSON] {json_path}")

    # Save CSV
    csv_path = os.path.join(out_dir, "recombination_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model Configuration", "Success Rate (%)", "Collisions/Episode", "Path Efficiency", "Goal Error (m)", "Inference Latency (us)"])
        for ag_id in ["baseline", "hjepa_monolithic", "recombined_compositional"]:
            r = results[ag_id]
            writer.writerow([r["name"], f"{r['success_rate_pct']:.1f}%", r["collisions_per_episode"], r["path_efficiency"], r["goal_error_m"], f"{r['mean_latency_us']:.1f}"])
    print(f"[Saved CSV]  {csv_path}")

if __name__ == "__main__":
    run_experiment_a()
