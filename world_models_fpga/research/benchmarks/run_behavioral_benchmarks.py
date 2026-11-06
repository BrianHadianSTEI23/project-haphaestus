"""
Phase 2: Behavioral Benchmarks across 6 Capability Categories
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Evaluates Baseline Policy, LeWorldModel (LeWM), and H-JEPA under identical seeds:
  - Object Grounding
  - Spatial Reasoning
  - Obstacle Avoidance
  - Action-Conditioned Prediction
  - Temporal Planning
  - Generalization
Saves structured JSON and CSV results into research/benchmarks/behavioral/
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

from research.datasets.generation.capability_tasks import (
    CAPABILITIES, DIFFICULTY_TIERS, generate_task_specification
)
from research.benchmarks.model_wrappers import BaselineAgent, LeWMAgent, HJEPAAgent
from research.benchmarks.evaluator import BenchmarkEvaluator

def run_behavioral_benchmarks(episodes_per_cap=15, output_dir=None):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "behavioral"))
    os.makedirs(output_dir, exist_ok=True)

    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints"))
    evaluator = BenchmarkEvaluator()

    agents = [
        BaselineAgent(os.path.join(ckpt_dir, "baseline_best.pt")),
        LeWMAgent(os.path.join(ckpt_dir, "lewm_best.pt")),
        HJEPAAgent(os.path.join(ckpt_dir, "hjepa_best.pt"))
    ]

    print("\n=======================================================")
    print("  PHASE 2: BEHAVIORAL BASELINE BENCHMARKS")
    print(f"  Episodes per Capability: {episodes_per_cap} (Total {episodes_per_cap * len(CAPABILITIES)} per model)")
    print(f"  Output Directory: {output_dir}")
    print("=======================================================\n")

    # Generate benchmark task specs with fixed evaluation seeds
    test_specs_by_cap = {}
    for cap in CAPABILITIES:
        specs = []
        for i in range(episodes_per_cap):
            diff = DIFFICULTY_TIERS[i % len(DIFFICULTY_TIERS)]
            spec = generate_task_specification(cap, diff, seed_idx=5000 + i)
            spec["episode_id"] = f"eval_{cap}_{diff}_{i:03d}"
            specs.append(spec)
        test_specs_by_cap[cap] = specs

    all_results = {}

    for agent in agents:
        print(f"\nEvaluating Agent: {agent.name.upper()}...")
        agent_cap_results = {}
        t0 = time.time()

        for cap in CAPABILITIES:
            specs = test_specs_by_cap[cap]
            suite_res = evaluator.evaluate_suite(agent, specs)
            agent_cap_results[cap] = suite_res
            print(f"  [{cap.upper()}] Success: {suite_res['success_rate']:5.1f}% | "
                  f"Collision: {suite_res['collision_rate']:5.1f}% | "
                  f"Path Eff: {suite_res['path_efficiency']:.2f} | "
                  f"Goal Err: {suite_res['goal_localization_error']:.3f}m | "
                  f"Latency: {suite_res['mean_latency_us']:.0f} us")

        t1 = time.time()
        print(f"Completed {agent.name} evaluation in {t1 - t0:.2f} s.")
        all_results[agent.model_id] = {
            "model_name": agent.name,
            "capabilities": agent_cap_results
        }

    # Save JSON summary
    json_path = os.path.join(output_dir, "behavioral_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save CSV comparison
    csv_path = os.path.join(output_dir, "behavioral_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Capability", "Success Rate (%)", "Collision Rate (%)",
            "Path Efficiency", "Goal Error (m)", "Mean Latency (us)", "P99 Latency (us)"
        ])
        for m_id, m_data in all_results.items():
            for cap, c_data in m_data["capabilities"].items():
                writer.writerow([
                    m_data["model_name"], cap,
                    f"{c_data['success_rate']:.1f}",
                    f"{c_data['collision_rate']:.1f}",
                    f"{c_data['path_efficiency']:.3f}",
                    f"{c_data['goal_localization_error']:.3f}",
                    f"{c_data['mean_latency_us']:.1f}",
                    f"{c_data['p99_latency_us']:.1f}"
                ])

    print(f"\nSaved behavioral benchmark results to:")
    print(f"  -> {json_path}")
    print(f"  -> {csv_path}")

    evaluator.close()
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=15, help="Episodes per capability")
    args = parser.parse_args()
    run_behavioral_benchmarks(episodes_per_cap=args.episodes)
