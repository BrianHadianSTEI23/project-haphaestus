"""
Phase 2: Generalization Gap Analysis
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Evaluates Baseline Policy, LeWorldModel (LeWM), and H-JEPA across:
  1. Canonical In-Distribution Arena Layout
  2. Held-out Environmental Configurations
  3. Permuted Quadrant Obstacle Layouts
  4. Unseen Robot Initial Headings (0, 90, 180, 270 deg)
Computes exact Generalization Transfer Gaps:
  Delta_Success = Success_canonical - Success_held_out
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

def run_generalization_benchmarks(episodes_per_condition=20, output_dir=None):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "generalization"))
    os.makedirs(output_dir, exist_ok=True)

    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints"))
    evaluator = BenchmarkEvaluator()

    agents = [
        BaselineAgent(os.path.join(ckpt_dir, "baseline_best.pt")),
        LeWMAgent(os.path.join(ckpt_dir, "lewm_best.pt")),
        HJEPAAgent(os.path.join(ckpt_dir, "hjepa_best.pt"))
    ]

    print("\n=======================================================")
    print("  PHASE 2: GENERALIZATION & TRANSFER BENCHMARKS")
    print(f"  Episodes per Condition: {episodes_per_condition}")
    print(f"  Output Directory: {output_dir}")
    print("=======================================================\n")

    # 1. Condition A: Canonical In-Distribution
    cond_canonical = [
        generate_task_specification("spatial_reasoning", "easy", seed_idx=6000 + i)
        for i in range(episodes_per_condition)
    ]

    # 2. Condition B: Held-out Permuted Layouts
    cond_permuted = [
        generate_task_specification("generalization", "difficult", seed_idx=7000 + i)
        for i in range(episodes_per_condition)
    ]

    # 3. Condition C: Unseen Starting Orientations (0, 90, 180, 270 deg)
    cond_orientations = []
    headings = [0.0, np.pi/2, np.pi, -np.pi/2]
    for i in range(episodes_per_condition):
        spec = generate_task_specification("spatial_reasoning", "difficult", seed_idx=8000 + i)
        spec["initial_pose"][2] = headings[i % 4]
        cond_orientations.append(spec)

    conditions = {
        "canonical": cond_canonical,
        "permuted_layout": cond_permuted,
        "unseen_orientations": cond_orientations
    }

    all_results = {}

    for agent in agents:
        print(f"\nEvaluating Generalization for: {agent.name.upper()}...")
        agent_cond_results = {}

        for c_name, c_specs in conditions.items():
            res = evaluator.evaluate_suite(agent, c_specs)
            agent_cond_results[c_name] = res
            print(f"  [{c_name.upper():18s}] Success: {res['success_rate']:5.1f}% | "
                  f"Collision: {res['collision_rate']:5.1f}% | "
                  f"Path Eff: {res['path_efficiency']:.2f} | "
                  f"Goal Err: {res['goal_localization_error']:.3f}m")

        # Compute Generalization Gap
        succ_canonical = agent_cond_results["canonical"]["success_rate"]
        succ_permuted = agent_cond_results["permuted_layout"]["success_rate"]
        succ_orient = agent_cond_results["unseen_orientations"]["success_rate"]

        gap_permuted = succ_canonical - succ_permuted
        gap_orient = succ_canonical - succ_orient

        agent_cond_results["generalization_gap_permuted"] = gap_permuted
        agent_cond_results["generalization_gap_orientation"] = gap_orient

        print(f"  >> Generalization Gap (Permuted Layout): {gap_permuted:+5.1f}%")
        print(f"  >> Generalization Gap (Orientations):    {gap_orient:+5.1f}%")

        all_results[agent.model_id] = {
            "model_name": agent.name,
            "conditions": agent_cond_results,
            "gap_permuted": gap_permuted,
            "gap_orientation": gap_orient
        }

    # Save JSON and CSV
    json_path = os.path.join(output_dir, "generalization_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    csv_path = os.path.join(output_dir, "generalization_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Canonical Success (%)", "Permuted Success (%)", "Permuted Gap (%)", "Orientation Gap (%)"])
        for m_id, m_data in all_results.items():
            writer.writerow([
                m_data["model_name"],
                f"{m_data['conditions']['canonical']['success_rate']:.1f}",
                f"{m_data['conditions']['permuted_layout']['success_rate']:.1f}",
                f"{m_data['gap_permuted']:.1f}",
                f"{m_data['gap_orientation']:.1f}"
            ])

    print(f"\nSaved generalization benchmark results to:")
    print(f"  -> {json_path}")
    print(f"  -> {csv_path}")

    evaluator.close()
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per condition")
    args = parser.parse_args()
    run_generalization_benchmarks(episodes_per_condition=args.episodes)
