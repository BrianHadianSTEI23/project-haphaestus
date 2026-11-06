"""
Generate Novel Independent Validation Dataset
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Generates a completely novel, independent validation suite (60 episodes, ~1,800 transitions)
using novel seeds (7000-7059) and randomized configurations to strictly validate
Option A and Option B models out-of-sample, ensuring zero overlap with training or distillation sets.
"""

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from research.datasets.generation.generate_benchmark_dataset import BenchmarkDatasetGenerator
from research.datasets.generation.capability_tasks import CAPABILITIES, DIFFICULTY_TIERS, generate_task_specification

def generate_independent_val_dataset(episodes_per_cap=10, output_dir=None):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../splits"))
    os.makedirs(output_dir, exist_ok=True)

    print("\n=======================================================")
    print("  GENERATING NOVEL INDEPENDENT VALIDATION SUITE")
    print(f"  Episodes: {len(CAPABILITIES) * episodes_per_cap} ({episodes_per_cap} per capability)")
    print(f"  Output: {output_dir}")
    print("=======================================================\n")

    generator = BenchmarkDatasetGenerator(seed=8888)
    episodes = []
    ep_counter = 0
    t0 = time.time()

    for cap_idx, capability in enumerate(CAPABILITIES):
        print(f"[{cap_idx+1}/6] Generating novel {capability.upper()} episodes...")
        for i in range(episodes_per_cap):
            diff = DIFFICULTY_TIERS[i % len(DIFFICULTY_TIERS)]
            # Novel seed offset 7000
            spec = generate_task_specification(capability, diff, seed_idx=7000 + ep_counter)
            spec["episode_id"] = f"indep_val_{capability}_{diff}_{ep_counter:04d}"

            transitions = generator.run_episode(spec)
            episodes.append((spec, transitions))
            ep_counter += 1

    t1 = time.time()
    print(f"\nGenerated {ep_counter} independent episodes in {t1 - t0:.2f} s.")

    # Save to independent_validation_suite.npz
    generator._save_split("independent_validation_suite", episodes, output_dir)
    print("Successfully created independent_validation_suite.npz!")

if __name__ == "__main__":
    generate_independent_val_dataset(episodes_per_cap=10)
