"""
Phase 2: Behavioral Intelligence Benchmark Runner (Repaired for Multi-Track & Causal Ablations)

Executes closed-loop MuJoCo evaluations separated by information regime:
  - Track A ('oracle'): Privileged simulator state upper bound (ExpertNavigator).
  - Track B ('structured'): Camera RGB + egocentric body-frame goal vector (across 3 training seeds).
  - Track C ('observation'): Camera RGB + discrete task instruction only (zero simulator coordinates).
  - Controlled Ablations:
      * prediction_used (full latent rollout)
      * reactive (no lookahead)
      * prediction_ignored (next_s replaced by s_t in scorer)
      * no_action_conditioning (action input zeroed in transition model)
      * no_collision_prediction (collision penalty disabled)
      * no_temporal_hierarchy (H-JEPA K=5 macro predictor bypassed)
"""

from __future__ import annotations
import sys as _hsys
from pathlib import Path as _HPath
_hsys.path.insert(0, str(_HPath(__file__).resolve().parents[2]))
from research.datasets.dataset_config import SPLITS_DIR as _ACTIVE_SPLITS_DIR, CKPT_DIR as _ACTIVE_CKPT_DIR  # noqa: E402

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from research.benchmarks.evaluator import BenchmarkEvaluator
from research.benchmarks.model_wrappers import (
    BaselineAgent,
    HJEPAAgent,
    LeWMAgent,
    OracleAgent,
    RandomAgent,
    load_all_agents,
)
from research.datasets.generation.capability_tasks import CAPABILITIES, DIFFICULTY_TIERS, generate_task_specification


def _ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


def _summarize_episodes(ep_records: list[dict]) -> dict:
    succ = [1.0 if r["success"] else 0.0 for r in ep_records]
    goal = [1.0 if r["goal_reached"] else 0.0 for r in ep_records]
    col = [0.0 if r["collision_free"] else 1.0 for r in ep_records]
    eff = [float(r["path_efficiency"]) for r in ep_records]
    err = [float(r["final_goal_error"]) for r in ep_records]
    lat_mean = [float(r["mean_latency_us"]) for r in ep_records]
    lat_med = [float(r["median_latency_us"]) for r in ep_records]
    lat_p99 = [float(r["p99_latency_us"]) for r in ep_records]
    return {
        "episodes_evaluated": len(ep_records),
        "success_rate": float(np.mean(succ)),
        "success_rate_ci95": _ci95(succ),
        "goal_reached_rate": float(np.mean(goal)),
        "collision_rate": float(np.mean(col)),
        "collision_rate_ci95": _ci95(col),
        "path_efficiency": float(np.mean(eff)),
        "path_efficiency_ci95": _ci95(eff),
        "mean_goal_error_m": float(np.mean(err)),
        "goal_error_ci95_m": _ci95(err),
        "latency_mean_us": float(np.mean(lat_mean)),
        "latency_median_us": float(np.median(lat_med)),
        "latency_p99_us": float(np.percentile(lat_p99, 99)),
    }


def run_all_behavioral_benchmarks(
    episodes_per_cap: int = 10,
    ckpt_dir: str | None = None,
    seeds: tuple[int, ...] = (42, 1337, 2026),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if ckpt_dir is None:
        v2_dir = str(_ACTIVE_CKPT_DIR)
        legacy_dir = str(_ACTIVE_CKPT_DIR)
        ckpt_dir = v2_dir if os.path.exists(os.path.join(v2_dir, "lewm_best.pt")) else legacy_dir

    evaluator = BenchmarkEvaluator()
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(out_dir, exist_ok=True)

    print("=================================================================")
    print("  PHASE 2: MULTI-TRACK BEHAVIORAL & ABLATION BENCHMARK SUITE")
    print(f"  Device: {device} | Checkpoints: {ckpt_dir}")
    print(f"  Capabilities: {len(CAPABILITIES)} | Episodes per capability: {episodes_per_cap}")
    print("=================================================================")

    # Build deterministic held-out evaluation task suite (seed_idx >= 5000)
    tasks_by_cap: dict[str, list[dict]] = {}
    for cap_idx, cap in enumerate(CAPABILITIES):
        specs = []
        for i in range(episodes_per_cap):
            diff = DIFFICULTY_TIERS[i % 4]  # easy, difficult, counterfactual, ood
            seed = 5000 + cap_idx * 100 + i
            spec = generate_task_specification(cap, diff, seed_idx=seed)
            spec["episode_id"] = f"eval_{cap}_{diff}_{i:02d}"
            specs.append(spec)
        tasks_by_cap[cap] = specs

    # 1. Primary Track B ('structured') + Track A ('oracle') evaluation on primary seed
    agents_primary = load_all_agents(ckpt_dir=ckpt_dir, device=device, track="structured", seed=seeds[0])
    results_summary: dict[str, dict] = {name: {"capabilities": {}, "overall": {}} for name in agents_primary}

    for cap in CAPABILITIES:
        print(f"\n--- Capability [Track A/B]: {cap.upper()} ---")
        for agent_name, agent in agents_primary.items():
            ep_records = [evaluator.evaluate_episode(agent, spec) for spec in tasks_by_cap[cap]]
            cap_summary = _summarize_episodes(ep_records)
            results_summary[agent_name]["capabilities"][cap] = cap_summary
            print(
                f"  [{agent_name:8s} | {agent.track:10s}] Success: {cap_summary['success_rate']*100:5.1f}% "
                f"(+-{cap_summary['success_rate_ci95']*100:4.1f}%) | Col: {cap_summary['collision_rate']*100:5.1f}% | "
                f"Eff: {cap_summary['path_efficiency']:.3f} | GoalErr: {cap_summary['mean_goal_error_m']:.3f}m"
            )

    for agent_name in agents_primary:
        caps_dict = results_summary[agent_name]["capabilities"]
        results_summary[agent_name]["overall"] = {
            "track": agents_primary[agent_name].track,
            "ablation_mode": agents_primary[agent_name].ablation_mode,
            "success_rate": float(np.mean([c["success_rate"] for c in caps_dict.values()])),
            "success_rate_ci95": float(np.mean([c["success_rate_ci95"] for c in caps_dict.values()])),
            "goal_reached_rate": float(np.mean([c["goal_reached_rate"] for c in caps_dict.values()])),
            "collision_rate": float(np.mean([c["collision_rate"] for c in caps_dict.values()])),
            "collision_rate_ci95": float(np.mean([c["collision_rate_ci95"] for c in caps_dict.values()])),
            "path_efficiency": float(np.mean([c["path_efficiency"] for c in caps_dict.values()])),
            "mean_goal_error_m": float(np.mean([c["mean_goal_error_m"] for c in caps_dict.values()])),
            "latency_mean_us": float(np.mean([c["latency_mean_us"] for c in caps_dict.values()])),
            "latency_median_us": float(np.median([c["latency_median_us"] for c in caps_dict.values()])),
            "latency_p99_us": float(np.max([c["latency_p99_us"] for c in caps_dict.values()])),
        }

    # 2. Multi-Seed Stability in Track B across seeds (42, 1337, 2026)
    print("\n--- Multi-Seed Track B Evaluation (Seeds: " + ", ".join(map(str, seeds)) + ") ---")
    multi_seed_summary: dict[str, dict] = {}
    all_eval_specs = [spec for cap in CAPABILITIES for spec in tasks_by_cap[cap][:5]]  # 30 representative episodes
    for model_key, cls, prefix in (
        ("Baseline", BaselineAgent, "baseline"),
        ("LeWM", LeWMAgent, "lewm"),
        ("H-JEPA", HJEPAAgent, "hjepa"),
    ):
        seed_succs, seed_cols, seed_effs, seed_errs = [], [], [], []
        per_seed_records = {}
        for s in seeds:
            ckpt = os.path.join(ckpt_dir, f"{prefix}_seed{s}_best.pt")
            if not os.path.exists(ckpt):
                ckpt = os.path.join(ckpt_dir, f"{prefix}_best.pt")
            ag = cls(ckpt, device=device, track="structured")
            eps = [evaluator.evaluate_episode(ag, sp) for sp in all_eval_specs]
            summ = _summarize_episodes(eps)
            per_seed_records[f"seed_{s}"] = summ
            seed_succs.append(summ["success_rate"])
            seed_cols.append(summ["collision_rate"])
            seed_effs.append(summ["path_efficiency"])
            seed_errs.append(summ["mean_goal_error_m"])
        multi_seed_summary[model_key] = {
            "seeds": list(seeds),
            "per_seed": per_seed_records,
            "mean_success_rate": float(np.mean(seed_succs)),
            "std_success_rate": float(np.std(seed_succs, ddof=1)) if len(seed_succs) > 1 else 0.0,
            "mean_collision_rate": float(np.mean(seed_cols)),
            "std_collision_rate": float(np.std(seed_cols, ddof=1)) if len(seed_cols) > 1 else 0.0,
            "mean_path_efficiency": float(np.mean(seed_effs)),
            "mean_goal_error_m": float(np.mean(seed_errs)),
        }
        print(
            f"  [{model_key:8s}] Multi-Seed Success: {multi_seed_summary[model_key]['mean_success_rate']*100:5.1f}% "
            f"(std={multi_seed_summary[model_key]['std_success_rate']*100:4.1f}%) | "
            f"Col: {multi_seed_summary[model_key]['mean_collision_rate']*100:5.1f}%"
        )

    # 3. Track C ('observation' only — zero simulator coordinates)
    print("\n--- Track C ('observation' + discrete task ID only, zero simulator coordinates) ---")
    track_c_summary: dict[str, dict] = {}
    for model_key, cls, prefix in (
        ("Baseline", BaselineAgent, "baseline"),
        ("LeWM", LeWMAgent, "lewm"),
        ("H-JEPA", HJEPAAgent, "hjepa"),
    ):
        ckpt = os.path.join(ckpt_dir, f"{prefix}_best.pt")
        ag = cls(ckpt, device=device, track="observation")
        eps = [evaluator.evaluate_episode(ag, sp) for sp in all_eval_specs]
        track_c_summary[model_key] = _summarize_episodes(eps)
        print(
            f"  [{model_key:8s} | Track C] Success: {track_c_summary[model_key]['success_rate']*100:5.1f}% | "
            f"GoalReached: {track_c_summary[model_key]['goal_reached_rate']*100:5.1f}% | "
            f"Col: {track_c_summary[model_key]['collision_rate']*100:5.1f}% | "
            f"GoalErr: {track_c_summary[model_key]['mean_goal_error_m']:.3f}m"
        )

    # 4. Controlled Causal Ablations on LeWM and H-JEPA (testing H1, H3)
    print("\n--- Controlled World-Model Ablations (Track B) ---")
    ablation_specs = (
        tasks_by_cap["obstacle_avoidance"]
        + tasks_by_cap["spatial_reasoning"]
        + tasks_by_cap["temporal_planning"]
    )
    ablation_summary: dict[str, dict] = {"LeWM": {}, "H-JEPA": {}}
    lewm_ckpt = os.path.join(ckpt_dir, "lewm_best.pt")
    hjepa_ckpt = os.path.join(ckpt_dir, "hjepa_best.pt")

    for mode in (
        "prediction_used",
        "reactive",
        "prediction_ignored",
        "no_action_conditioning",
        "no_collision_prediction",
    ):
        ag_lewm = LeWMAgent(lewm_ckpt, device=device, track="structured", ablation_mode=mode)
        ablation_summary["LeWM"][mode] = _summarize_episodes(
            [evaluator.evaluate_episode(ag_lewm, sp) for sp in ablation_specs]
        )
        print(
            f"  [LeWM   | {mode:24s}] Success: {ablation_summary['LeWM'][mode]['success_rate']*100:5.1f}% | "
            f"Col: {ablation_summary['LeWM'][mode]['collision_rate']*100:5.1f}% | "
            f"GoalErr: {ablation_summary['LeWM'][mode]['mean_goal_error_m']:.3f}m"
        )

    for mode in (
        "prediction_used",
        "reactive",
        "prediction_ignored",
        "no_action_conditioning",
        "no_collision_prediction",
        "no_temporal_hierarchy",
    ):
        ag_hjepa = HJEPAAgent(hjepa_ckpt, device=device, track="structured", ablation_mode=mode)
        ablation_summary["H-JEPA"][mode] = _summarize_episodes(
            [evaluator.evaluate_episode(ag_hjepa, sp) for sp in ablation_specs]
        )
        print(
            f"  [H-JEPA | {mode:24s}] Success: {ablation_summary['H-JEPA'][mode]['success_rate']*100:5.1f}% | "
            f"Col: {ablation_summary['H-JEPA'][mode]['collision_rate']*100:5.1f}% | "
            f"GoalErr: {ablation_summary['H-JEPA'][mode]['mean_goal_error_m']:.3f}m"
        )

    evaluator.close()

    results_summary["_tracks_and_ablations"] = {
        "track_a_oracle": results_summary["Oracle"]["overall"],
        "track_b_structured_multi_seed": multi_seed_summary,
        "track_c_observation_only": track_c_summary,
        "controlled_ablations": ablation_summary,
    }

    out_path = os.path.join(out_dir, "behavioral_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n[SAVED BEHAVIORAL BENCHMARK RESULTS TO {out_path}]")
    return results_summary


if __name__ == "__main__":
    run_all_behavioral_benchmarks(episodes_per_cap=10)
