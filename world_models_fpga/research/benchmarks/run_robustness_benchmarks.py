"""
Phase 2: Robustness and Dynamic Disturbance Recovery Benchmarks
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Evaluates closed-loop recovery when subjected to unexpected kinetic disturbances mid-trajectory:
  - Step t=10 kinetic impulse injection (Delta vy = 0.8 m/s lateral push)
  - Measures recovery rate, collision rate, and path divergence
"""

import os
import sys
import json
import csv
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from research.datasets.generation.capability_tasks import generate_task_specification
from research.benchmarks.model_wrappers import BaselineAgent, LeWMAgent, HJEPAAgent
from research.benchmarks.evaluator import BenchmarkEvaluator

def run_robustness_benchmarks(episodes=20, output_dir=None):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "robustness"))
    os.makedirs(output_dir, exist_ok=True)

    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints"))
    evaluator = BenchmarkEvaluator()

    agents = [
        BaselineAgent(os.path.join(ckpt_dir, "baseline_best.pt")),
        LeWMAgent(os.path.join(ckpt_dir, "lewm_best.pt")),
        HJEPAAgent(os.path.join(ckpt_dir, "hjepa_best.pt"))
    ]

    print("\n=======================================================")
    print("  PHASE 2: ROBUSTNESS & DYNAMIC RECOVERY BENCHMARKS")
    print(f"  Episodes: {episodes} per model | Disturbance: Lateral Impulse at Step 10")
    print(f"  Output Directory: {output_dir}")
    print("=======================================================\n")

    # Generate perturbation evaluation tasks
    task_specs = [
        generate_task_specification("spatial_reasoning", "easy", seed_idx=9000 + i)
        for i in range(episodes)
    ]

    perturbation = {
        "step": 10,
        "impulse": [0.0, 0.8, 0.5] # Lateral + yaw disturbance
    }

    all_results = {}

    for agent in agents:
        print(f"Evaluating Disturbance Recovery for: {agent.name.upper()}...")
        t0 = time.time()
        res = evaluator.evaluate_suite(agent, task_specs, perturbation=perturbation)
        t1 = time.time()

        print(f"  Recovery Rate: {res['recovery_rate']:5.1f}% | "
              f"Collision Rate: {res['collision_rate']:5.1f}% | "
              f"Final Goal Error: {res['goal_localization_error']:.3f}m | "
              f"Time: {t1 - t0:.2f} s")

        all_results[agent.model_id] = res

    # Save JSON and CSV
    json_path = os.path.join(output_dir, "robustness_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    csv_path = os.path.join(output_dir, "robustness_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Recovery Rate (%)", "Collision Rate (%)", "Goal Localization Error (m)", "Path Efficiency"])
        for m_id, m_data in all_results.items():
            writer.writerow([
                m_data["model_name"],
                f"{m_data['recovery_rate']:.1f}",
                f"{m_data['collision_rate']:.1f}",
                f"{m_data['goal_localization_error']:.3f}",
                f"{m_data['path_efficiency']:.3f}"
            ])

    print(f"\nSaved robustness benchmark results to:")
    print(f"  -> {json_path}")
    print(f"  -> {csv_path}")

    evaluator.close()
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per model")
    args = parser.parse_args()
    run_robustness_benchmarks(episodes=args.episodes)
