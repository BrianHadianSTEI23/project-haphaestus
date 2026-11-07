"""
Phase 2: Generalization, Counterfactual Invariance (E1-E7), and Disambiguation Benchmarks

Evaluates Oracle (Track A), Random, Baseline, LeWM, and H-JEPA across:
  1. Counterfactual matched pairs E1..E7 (measuring E0 vs Ei success, retention rate, and delta)
  2. Disambiguation probe suites DISAMBIG_01..04
  3. Canonical vs Permuted Layout vs Unseen Orientation generalization gaps
  4. Mid-trajectory kinetic disturbance recovery
"""

from __future__ import annotations
import sys as _hsys
from pathlib import Path as _HPath
_hsys.path.insert(0, str(_HPath(__file__).resolve().parents[2]))
from research.datasets.dataset_config import SPLITS_DIR as _ACTIVE_SPLITS_DIR, CKPT_DIR as _ACTIVE_CKPT_DIR  # noqa: E402

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from research.benchmarks.evaluator import BenchmarkEvaluator
from research.benchmarks.model_wrappers import load_all_agents
from research.datasets.counterfactual.counterfactual_generator import generate_counterfactual_pair
from research.datasets.counterfactual.disambiguation_tasks import generate_disambiguation_tasks
from research.datasets.generation.capability_tasks import generate_task_specification


def run_generalization_benchmarks(pairs_per_exp: int = 8, repeats_per_probe: int = 6, output_dir: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v2_ckpt = str(_ACTIVE_CKPT_DIR)
    legacy_ckpt = str(_ACTIVE_CKPT_DIR)
    ckpt_dir = v2_ckpt if os.path.exists(os.path.join(v2_ckpt, "lewm_best.pt")) else legacy_ckpt

    agents = load_all_agents(ckpt_dir=ckpt_dir, device=device, track="structured")
    evaluator = BenchmarkEvaluator()

    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    gen_dir = output_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "generalization"))
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    exp_types = [
        "E1_position_shift",
        "E2_color_shift",
        "E3_robot_rotation",
        "E4_robot_position",
        "E5_intermediate_obstacle",
        "E6_shape_mutation",
        "E7_camera_viewpoint",
    ]

    gen_summary = {
        name: {"counterfactuals": {}, "disambiguation": {}, "disturbance_recovery": {}, "conditions": {}}
        for name in agents
    }

    print("=================================================================")
    print("  PHASE 2: COUNTERFACTUAL & GENERALIZATION BENCHMARKS (E1 - E7)")
    print("=================================================================")

    # 1. Counterfactual matched pairs E1..E7
    for exp_type in exp_types:
        print(f"\n--- Counterfactual Experiment: {exp_type} ---")
        pairs = [generate_counterfactual_pair(exp_type, p) for p in range(pairs_per_exp)]
        for agent_name, agent in agents.items():
            e0_succ, ei_succ, both_succ, ei_err = [], [], [], []
            for e0_spec, ei_spec in pairs:
                r0 = evaluator.evaluate_episode(agent, e0_spec)
                ri = evaluator.evaluate_episode(agent, ei_spec)
                e0_succ.append(1.0 if r0["success"] else 0.0)
                ei_succ.append(1.0 if ri["success"] else 0.0)
                both_succ.append(1.0 if (r0["success"] and ri["success"]) else 0.0)
                ei_err.append(ri["final_goal_error"])

            s0 = float(np.mean(e0_succ))
            si = float(np.mean(ei_succ))
            retention = float(np.mean(both_succ) / max(s0, 1e-5))
            gen_summary[agent_name]["counterfactuals"][exp_type] = {
                "baseline_E0_success": s0,
                "mutated_Ei_success": si,
                "invariance_retention_rate": min(1.0, retention),
                "performance_drop_delta": s0 - si,
                "mutated_goal_error_m": float(np.mean(ei_err)),
            }
            print(
                f"  [{agent_name:8s}] E0: {s0*100:5.1f}% -> Ei: {si*100:5.1f}% | "
                f"Retention: {min(1.0, retention)*100:5.1f}% | GoalErr: {np.mean(ei_err):.3f}m"
            )

    # 2. Disambiguation Probes
    probe_specs = generate_disambiguation_tasks()
    for p_spec in probe_specs:
        pid = p_spec["probe_id"]
        print(f"\n--- Disambiguation Probe: {pid} ---")
        for agent_name, agent in agents.items():
            succs, errs = [], []
            for r in range(repeats_per_probe):
                spec = dict(p_spec)
                spec["initial_pose"] = p_spec["initial_pose"].copy()
                rng = np.random.default_rng(7000 + r)
                spec["initial_pose"][:2] += rng.uniform(-0.04, 0.04, size=2).astype(np.float32)
                res = evaluator.evaluate_episode(agent, spec)
                succs.append(1.0 if res["success"] else 0.0)
                errs.append(res["final_goal_error"])
            gen_summary[agent_name]["disambiguation"][pid] = {
                "success_rate": float(np.mean(succs)),
                "mean_goal_error_m": float(np.mean(errs)),
            }
            print(f"  [{agent_name:8s}] Success: {np.mean(succs)*100:5.1f}% | GoalErr: {np.mean(errs):.3f}m")

    # 3. Canonical vs Permuted Layout vs Unseen Orientations
    cond_canonical = [generate_task_specification("spatial_reasoning", "easy", seed_idx=6000 + i) for i in range(12)]
    cond_permuted = [generate_task_specification("generalization", "difficult", seed_idx=7000 + i) for i in range(12)]
    headings = [0.0, np.pi / 2, np.pi, -np.pi / 2]
    cond_orient = []
    for i in range(12):
        sp = generate_task_specification("spatial_reasoning", "difficult", seed_idx=8000 + i)
        sp["initial_pose"][2] = headings[i % 4]
        cond_orient.append(sp)

    for agent_name, agent in agents.items():
        c_res = evaluator.evaluate_suite(agent, cond_canonical)
        p_res = evaluator.evaluate_suite(agent, cond_permuted)
        o_res = evaluator.evaluate_suite(agent, cond_orient)
        gen_summary[agent_name]["conditions"] = {
            "canonical": c_res,
            "permuted_layout": p_res,
            "unseen_orientations": o_res,
            "generalization_gap_permuted": c_res["success_rate"] - p_res["success_rate"],
            "generalization_gap_orientation": c_res["success_rate"] - o_res["success_rate"],
        }

    # 4. Mid-Trajectory Disturbance Recovery
    disturb_tasks = [generate_task_specification("spatial_reasoning", "easy", seed_idx=8500 + i) for i in range(12)]
    for agent_name, agent in agents.items():
        d_succ, d_err = [], []
        for sp in disturb_tasks:
            res = evaluator.evaluate_episode(agent, sp, inject_disturbance=True)
            d_succ.append(1.0 if res["success"] else 0.0)
            d_err.append(res["final_goal_error"])
        gen_summary[agent_name]["disturbance_recovery"] = {
            "recovery_success_rate": float(np.mean(d_succ)),
            "post_disturbance_goal_error_m": float(np.mean(d_err)),
        }

    evaluator.close()

    out_path_1 = os.path.join(results_dir, "generalization_eval_results.json")
    out_path_2 = os.path.join(gen_dir, "generalization_results.json")
    with open(out_path_1, "w") as f:
        json.dump(gen_summary, f, indent=2)
    with open(out_path_2, "w") as f:
        json.dump(gen_summary, f, indent=2)
    print(f"\n[SAVED GENERALIZATION BENCHMARK RESULTS TO {out_path_1}]")
    return gen_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()
    run_generalization_benchmarks(pairs_per_exp=args.episodes)
