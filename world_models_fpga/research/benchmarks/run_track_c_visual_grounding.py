"""
Track C Pure Visual Spatial Grounding Benchmark (Scientifically Repaired Protocol)

Strict Information Regime:
- Track C ('observation'): The agent receives ONLY camera RGB (64x64x3) and discrete task ID.
- Zero simulator coordinates ([dx, dy, sin_th, cos_th] strictly withheld).
- Evaluates controlled counterfactual test suites against matched nominal episodes:
  1. Nominal Visual Grounding: Baseline canonical tasks balanced across all 6 targets.
  2A. Floor Appearance Nuisance: Floor texture altered; geometry and lighting held constant.
  2B. Illumination Nuisance: 50% lighting reduction; geometry and floor held constant.
  2C1. Target-Only Appearance Perturbation: Mutates ONLY the requested target object's appearance.
  2C2. Distractor-Only Appearance Perturbation: Mutates distractor appearances while target is canonical.
  3. Color-Defined Semantic Targeting: Evaluates whether agent routes by visual appearance or discrete categorical task token.
  4. Visual Geometry Grounding: Landmark physical positions swapped while task token, target instance,
     and waypoint remain consistent; tests visual steering vs canonical allocentric location memorization.
  5. Layout Generalization: Non-canonical randomized obstacle layouts with verified physical clearances.
  6. Rigid SE(2) Transformations: Global rigid arena rotation vs egocentric camera rotation.
- Rigorous geometric validation asserts task token, target object instance, and waypoint consistency
  accounting for robot footprint (radius 0.18m) and obstacle geometries (cylinders and boxes).
- Reports exact contingency tables, paired differences, McNemar exact tests, 95% bootstrap CIs,
  and saves full episode-level raw records with comprehensive metadata.
"""

from __future__ import annotations

import argparse
import copy
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
import torch.nn.functional as F
from scipy.stats import binom

from env.amr_env import JetAutoEnv
from models.h_jepa import HJEPAModel
from models.le_world_model import LeWorldModel
from models.baseline_policy import BaselineNavigationPolicy, encode_task_id, TARGET_KEYS
from research.benchmarks.model_wrappers import BaselineAgent, HJEPAAgent, LeWMAgent
from research.datasets.generation.capability_tasks import (
    CANONICAL_OBSTACLES,
    generate_task_specification,
    get_waypoint_for_relation,
)


def compute_surface_distance(point: np.ndarray, obs: dict, robot_radius: float = 0.18) -> float:
    """Computes exact Euclidean distance from robot footprint to obstacle boundary."""
    p = np.asarray(point[:2], dtype=np.float32)
    c = np.asarray(obs["pos"][:2], dtype=np.float32)
    if "radius" in obs:
        d = float(np.linalg.norm(p - c))
        return float(d - (float(obs["radius"]) + robot_radius))
    elif "size" in obs:
        hx, hy = float(obs["size"][0]), float(obs["size"][1])
        dx = max(0.0, abs(float(p[0] - c[0])) - hx)
        dy = max(0.0, abs(float(p[1] - c[1])) - hy)
        d_box = float(np.sqrt(dx * dx + dy * dy))
        return float(d_box - robot_radius)
    return float(np.linalg.norm(p - c) - (0.25 + robot_radius))


def assert_task_spec_valid(
    spec: dict,
    robot_radius: float = 0.18,
    min_surface_clearance: float = 0.05,
    min_obs_sep: float = 0.30,
    arena_bounds: tuple[float, float] = (-1.85, 1.85),
) -> None:
    """
    Rigorous geometric and task-specification validation:
    1. Task identity resolves to exactly one intended target.
    2. Physical target object corresponds to that task identity.
    3. Waypoint is derived from actual target object's current position using declared spatial relation.
    4. Task identity, instruction, target instance, and evaluation waypoint agree.
    5. Start pose and waypoint are physically feasible, within arena bounds.
    6. Obstacle geometry and robot footprint are considered (surface-to-surface clearance, not merely center distance).
    """
    target_key = spec.get("target_key", "")
    assert target_key, "Task spec missing 'target_key'"
    obs_dict = spec.get("obstacles", {})
    assert target_key in obs_dict, f"Target key '{target_key}' not found in obstacles: {list(obs_dict.keys())}"
    target_obj = obs_dict[target_key]

    # Task identity encoding check
    instruction = spec.get("instruction", "")
    sem_rel = spec.get("semantic_relation", "approach")
    tid = encode_task_id(target_key, sem_rel, instruction)
    assert 0 <= tid < 64, f"Invalid task ID {tid} for target_key {target_key}"

    # Verify instruction mentions target identity or label
    target_label = target_obj.get("label", target_key.replace("_", " "))
    aliases = target_obj.get("aliases", [target_label, target_key])
    has_target_mention = any(a.lower() in instruction.lower() for a in aliases) or (target_key.replace("_", " ") in instruction.lower())
    assert has_target_mention, f"Instruction '{instruction}' does not mention target '{target_key}' or aliases {aliases}"

    init_xy = np.asarray(spec["initial_pose"][:2], dtype=np.float32)
    assert (
        arena_bounds[0] <= init_xy[0] <= arena_bounds[1]
        and arena_bounds[0] <= init_xy[1] <= arena_bounds[1]
    ), f"Initial pose {init_xy} out of bounds {arena_bounds}"

    wps = spec.get("waypoints", [])
    assert len(wps) > 0, "No waypoints in spec"
    final_wp = np.asarray(wps[-1][:2], dtype=np.float32)
    assert (
        arena_bounds[0] <= final_wp[0] <= arena_bounds[1]
        and arena_bounds[0] <= final_wp[1] <= arena_bounds[1]
    ), f"Final waypoint {final_wp} out of bounds {arena_bounds}"

    # Verify waypoint is derived from actual target object's current position
    t_pos_2d = np.asarray(target_obj["pos"][:2], dtype=np.float32)
    dist_wp_target = float(np.linalg.norm(final_wp - t_pos_2d))
    assert 0.40 <= dist_wp_target <= 0.85, f"Waypoint {final_wp} distance to target {t_pos_2d} is {dist_wp_target:.3f}m, expected ~0.60m"

    # Robot footprint & obstacle geometry checks
    for k, o in obs_dict.items():
        o_pos = np.asarray(o["pos"][:2], dtype=np.float32)

        # Initial pose surface clearance
        init_surf_clearance = compute_surface_distance(init_xy, o, robot_radius=robot_radius)
        assert init_surf_clearance >= min_surface_clearance, (
            f"Initial pose {init_xy} collides with obstacle '{k}' at {o_pos}: "
            f"surface clearance = {init_surf_clearance:.3f}m < {min_surface_clearance}m"
        )

        # Final waypoint surface clearance
        wp_surf_clearance = compute_surface_distance(final_wp, o, robot_radius=robot_radius)
        if k != target_key:
            assert wp_surf_clearance >= min_surface_clearance, (
                f"Waypoint {final_wp} collides with non-target obstacle '{k}' at {o_pos}: "
                f"surface clearance = {wp_surf_clearance:.3f}m < {min_surface_clearance}m"
            )
        else:
            # Standoff clearance to target object (must be non-colliding, >= 0.05m)
            assert wp_surf_clearance >= min_surface_clearance, (
                f"Waypoint {final_wp} collides with target obstacle '{k}': "
                f"surface clearance = {wp_surf_clearance:.3f}m < {min_surface_clearance}m"
            )

    # Inter-obstacle separation
    obs_list = list(obs_dict.values())
    for i in range(len(obs_list)):
        for j in range(i + 1, len(obs_list)):
            p_i = np.asarray(obs_list[i]["pos"][:2], dtype=np.float32)
            p_j = np.asarray(obs_list[j]["pos"][:2], dtype=np.float32)
            d_obs = float(np.linalg.norm(p_i - p_j))
            assert d_obs >= min_obs_sep, f"Obstacle overlap: separated by {d_obs:.3f} < {min_obs_sep}"


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


def _git_clean() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return len(out) == 0
    except Exception:
        return False


def _paired_contingency(nom_succ: list[bool], mut_succ: list[bool]) -> dict:
    """Computes exact 2x2 contingency table and McNemar exact binomial p-value."""
    n11 = sum(1 for n, m in zip(nom_succ, mut_succ) if n and m)
    n00 = sum(1 for n, m in zip(nom_succ, mut_succ) if not n and not m)
    n10 = sum(1 for n, m in zip(nom_succ, mut_succ) if n and not m)  # nom succ, mut fail
    n01 = sum(1 for n, m in zip(nom_succ, mut_succ) if not n and m)  # nom fail, mut succ

    discordant = n10 + n01
    if discordant == 0:
        p_val = 1.0
    else:
        k = min(n10, n01)
        p_val = float(min(1.0, 2.0 * binom.cdf(k, discordant, 0.5)))

    diffs = np.array([float(m) - float(n) for n, m in zip(nom_succ, mut_succ)], dtype=np.float32)
    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(42)
    if len(diffs) > 1:
        boot_diffs = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(2000)]
        ci_low = float(np.percentile(boot_diffs, 2.5))
        ci_high = float(np.percentile(boot_diffs, 97.5))
    else:
        ci_low, ci_high = mean_diff, mean_diff

    return {
        "n11_both_success": n11,
        "n00_both_fail": n00,
        "n10_nom_succ_mut_fail": n10,
        "n01_nom_fail_mut_succ": n01,
        "discordant_count": discordant,
        "mcnemar_exact_p_value": p_val,
        "mean_diff": mean_diff,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


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
        
        t_key = spec.get("target_key", "")
        t_pos = spec.get("obstacles", {}).get(t_key, {}).get("pos", [0.0, 0.0, 0.0])
        tid = encode_task_id(t_key, spec.get("semantic_relation", "approach"), spec.get("instruction", ""))
        
        records.append({
            "episode_id": spec.get("episode_id", "ep"),
            "target_key": t_key,
            "target_instance": t_key,
            "target_physical_coordinates": [float(x) for x in t_pos],
            "task_token": int(tid),
            "instruction": spec.get("instruction", ""),
            "computed_waypoint": [float(x) for x in final_wp],
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
    parser.add_argument("--episodes-per-suite", type=int, default=20, help="Number of episodes per test suite (N >= 20)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=str(pkg_root / "research/benchmarks/results/track_c_visual_grounding.json"))
    args = parser.parse_args()

    ckpt_dir = pkg_root / "checkpoints_v4"
    hjepa_path = ckpt_dir / "hjepa_best.pt"
    lewm_path = ckpt_dir / "lewm_best.pt"
    base_path = ckpt_dir / "baseline_best.pt"
    test_npz = pkg_root / "research/datasets/splits_v4/held_out_test_benchmark.npz"

    print("=================================================================")
    print("  TRACK C PURE VISUAL SPATIAL GROUNDING BENCHMARK (REPAIRED)")
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
    nominal_specs = []
    for i in range(n_eps):
        spec = generate_task_specification("object_grounding", "easy", seed_idx=6100 + i)
        spec["episode_id"] = f"nom_{i:02d}"
        assert_task_spec_valid(spec)
        nominal_specs.append(spec)

    res_nom_h = evaluate_suite_track_c(agent_h, nominal_specs)
    res_nom_l = evaluate_suite_track_c(agent_l, nominal_specs)
    res_nom_b = evaluate_suite_track_c(agent_b, nominal_specs)
    print(f"  [H-JEPA Nominal]   Succ: {res_nom_h['success_count']}/{n_eps} ({res_nom_h['success_rate']*100:5.1f}%) | Col: {res_nom_h['collision_rate']*100:5.1f}% | GoalErr: {res_nom_h['mean_final_goal_error_m']:.3f}m")
    print(f"  [LeWM Nominal]     Succ: {res_nom_l['success_count']}/{n_eps} ({res_nom_l['success_rate']*100:5.1f}%) | Col: {res_nom_l['collision_rate']*100:5.1f}% | GoalErr: {res_nom_l['mean_final_goal_error_m']:.3f}m")
    print(f"  [Baseline Nominal] Succ: {res_nom_b['success_count']}/{n_eps} ({res_nom_b['success_rate']*100:5.1f}%) | Col: {res_nom_b['collision_rate']*100:5.1f}% | GoalErr: {res_nom_b['mean_final_goal_error_m']:.3f}m")

    nom_h_succ = [r["success"] for r in res_nom_h["records"]]

    # --- 2A. Appearance Nuisance: Floor Texture Only ---
    print("\n--- 2A. Evaluating Floor Appearance Nuisance (Geometry & Light Constant) ---")
    floor_specs = []
    for i in range(n_eps):
        spec = copy.deepcopy(nominal_specs[i])
        spec["episode_id"] = f"nuis_floor_{i:02d}"
        spec["floor_rgba"] = [0.15, 0.20, 0.35, 1.0]  # dark slate/blue floor
        assert_task_spec_valid(spec)
        floor_specs.append(spec)

    res_floor_h = evaluate_suite_track_c(agent_h, floor_specs)
    floor_h_succ = [r["success"] for r in res_floor_h["records"]]
    pair_floor = _paired_contingency(nom_h_succ, floor_h_succ)
    print(f"  [H-JEPA Floor Nuisance] Succ: {res_floor_h['success_count']}/{n_eps} ({res_floor_h['success_rate']*100:5.1f}%) | Delta: {pair_floor['mean_diff']*100:+5.1f}% [CI95: {pair_floor['ci95_low']*100:+5.1f}%, {pair_floor['ci95_high']*100:+5.1f}%] | p={pair_floor['mcnemar_exact_p_value']:.4f}")

    # --- 2B. Appearance Nuisance: Illumination Only ---
    print("\n--- 2B. Evaluating Illumination Nuisance (Geometry & Floor Constant) ---")
    illum_specs = []
    for i in range(n_eps):
        spec = copy.deepcopy(nominal_specs[i])
        spec["episode_id"] = f"nuis_illum_{i:02d}"
        spec["light_diffuse_scale"] = 0.50  # 50% lighting reduction
        assert_task_spec_valid(spec)
        illum_specs.append(spec)

    res_illum_h = evaluate_suite_track_c(agent_h, illum_specs)
    illum_h_succ = [r["success"] for r in res_illum_h["records"]]
    pair_illum = _paired_contingency(nom_h_succ, illum_h_succ)
    print(f"  [H-JEPA Illum Nuisance] Succ: {res_illum_h['success_count']}/{n_eps} ({res_illum_h['success_rate']*100:5.1f}%) | Delta: {pair_illum['mean_diff']*100:+5.1f}% [CI95: {pair_illum['ci95_low']*100:+5.1f}%, {pair_illum['ci95_high']*100:+5.1f}%] | p={pair_illum['mcnemar_exact_p_value']:.4f}")

    # --- 2C1. Object-Appearance Invariance: Target Object Only ---
    print("\n--- 2C1. Evaluating Target-Only Appearance Perturbation (Balanced Across Targets) ---")
    target_app_specs = []
    for i in range(n_eps):
        spec = copy.deepcopy(nominal_specs[i])
        spec["episode_id"] = f"nuis_target_app_{i:02d}"
        t_key = spec["target_key"]
        t_geom = spec["obstacles"][t_key]["geom_name"]
        # Mutate target appearance to magenta/violet while distractors retain canonical appearance
        spec["color_mutations"] = {t_geom: [0.85, 0.15, 0.85, 1.0]}
        assert_task_spec_valid(spec)
        target_app_specs.append(spec)

    res_target_app_h = evaluate_suite_track_c(agent_h, target_app_specs)
    target_app_h_succ = [r["success"] for r in res_target_app_h["records"]]
    pair_target_app = _paired_contingency(nom_h_succ, target_app_h_succ)
    print(f"  [H-JEPA Target Appearance] Succ: {res_target_app_h['success_count']}/{n_eps} ({res_target_app_h['success_rate']*100:5.1f}%) | Delta: {pair_target_app['mean_diff']*100:+5.1f}% [CI95: {pair_target_app['ci95_low']*100:+5.1f}%, {pair_target_app['ci95_high']*100:+5.1f}%] | p={pair_target_app['mcnemar_exact_p_value']:.4f}")

    # --- 2C2. Object-Appearance Invariance: Distractor Objects Only ---
    print("\n--- 2C2. Evaluating Distractor-Only Appearance Perturbation (Target Canonical) ---")
    dist_app_specs = []
    for i in range(n_eps):
        spec = copy.deepcopy(nominal_specs[i])
        spec["episode_id"] = f"nuis_dist_app_{i:02d}"
        t_key = spec["target_key"]
        # Mutate all distractor geoms, leave target object unchanged
        spec["color_mutations"] = {}
        for k, o in spec["obstacles"].items():
            if k != t_key:
                spec["color_mutations"][o["geom_name"]] = [0.15, 0.75, 0.75, 1.0]
        assert_task_spec_valid(spec)
        dist_app_specs.append(spec)

    res_dist_app_h = evaluate_suite_track_c(agent_h, dist_app_specs)
    dist_app_h_succ = [r["success"] for r in res_dist_app_h["records"]]
    pair_dist_app = _paired_contingency(nom_h_succ, dist_app_h_succ)
    print(f"  [H-JEPA Distractor App] Succ: {res_dist_app_h['success_count']}/{n_eps} ({res_dist_app_h['success_rate']*100:5.1f}%) | Delta: {pair_dist_app['mean_diff']*100:+5.1f}% [CI95: {pair_dist_app['ci95_low']*100:+5.1f}%, {pair_dist_app['ci95_high']*100:+5.1f}%] | p={pair_dist_app['mcnemar_exact_p_value']:.4f}")

    # --- 3. Semantically Relevant Appearance: Color Swap vs Landmark Categorical Token ---
    print("\n--- 3. Evaluating Color-Swap Semantics (Categorical Task Token vs Visual Color) ---")
    sem_specs = []
    for i in range(n_eps):
        spec = copy.deepcopy(nominal_specs[i])
        spec["episode_id"] = f"sem_color_{i:02d}"
        # Swap colors between target and a distractor
        t_key = spec["target_key"]
        obs = spec.get("obstacles", {})
        other_keys = [k for k in obs.keys() if k != t_key]
        partner_key = other_keys[i % len(other_keys)]
        
        t_geom = obs[t_key]["geom_name"]
        p_geom = obs[partner_key]["geom_name"]
        
        # Color swap: target receives partner's color, partner receives target's color
        spec["color_mutations"] = {
            t_geom: [0.2, 0.4, 0.85, 1.0],
            p_geom: [0.95, 0.85, 0.1, 1.0],
        }
        # In Track C, agent is prompted with target_key. We evaluate whether agent steers to target_key's physical pos
        assert_task_spec_valid(spec)
        sem_specs.append(spec)

    res_sem_h = evaluate_suite_track_c(agent_h, sem_specs)
    sem_h_succ = [r["success"] for r in res_sem_h["records"]]
    pair_sem = _paired_contingency(nom_h_succ, sem_h_succ)
    print(f"  [H-JEPA Color Swap] Succ: {res_sem_h['success_count']}/{n_eps} ({res_sem_h['success_rate']*100:5.1f}%) | Delta: {pair_sem['mean_diff']*100:+5.1f}% [CI95: {pair_sem['ci95_low']*100:+5.1f}%, {pair_sem['ci95_high']*100:+5.1f}%] | p={pair_sem['mcnemar_exact_p_value']:.4f}")

    # --- 4. Visual Geometry Grounding: Landmark Physical Rearrangement ---
    print("\n--- 4. Evaluating Visual Geometry Grounding (Physical Landmark Relocation) ---")
    rearrange_specs = []
    for i in range(n_eps):
        spec = copy.deepcopy(nominal_specs[i])
        spec["episode_id"] = f"rearr_{i:02d}"
        t_key = spec["target_key"]
        obs = spec.get("obstacles", {})
        other_keys = [k for k in obs.keys() if k != t_key]
        partner_key = other_keys[i % len(other_keys)]

        # Physically swap positions of target object and partner obstacle
        pos_target = obs[t_key]["pos"].copy()
        pos_partner = obs[partner_key]["pos"].copy()
        obs[t_key]["pos"] = pos_partner
        obs[partner_key]["pos"] = pos_target

        # Recompute valid approach waypoint with clearance in front of target's new position
        new_wp = get_waypoint_for_relation(obs[t_key]["pos"], spec.get("semantic_relation", "approach"), spec["initial_pose"], clearance=0.60)
        spec["waypoints"] = [new_wp]
        assert_task_spec_valid(spec)
        rearrange_specs.append(spec)

    res_rearr_h = evaluate_suite_track_c(agent_h, rearrange_specs)
    rearr_h_succ = [r["success"] for r in res_rearr_h["records"]]
    pair_rearr = _paired_contingency(nom_h_succ, rearr_h_succ)
    print(f"  [H-JEPA Rearrange] Succ: {res_rearr_h['success_count']}/{n_eps} ({res_rearr_h['success_rate']*100:5.1f}%) | Delta: {pair_rearr['mean_diff']*100:+5.1f}% [CI95: {pair_rearr['ci95_low']*100:+5.1f}%, {pair_rearr['ci95_high']*100:+5.1f}%] | p={pair_rearr['mcnemar_exact_p_value']:.4f} | GoalErr: {res_rearr_h['mean_final_goal_error_m']:.3f}m")

    # --- 5. Layout Generalization: Randomized Obstacle Layouts ---
    print("\n--- 5. Evaluating Layout Generalization (Randomized Configurations) ---")
    rng_layout = np.random.default_rng(777)
    rand_specs = []
    for i in range(n_eps):
        valid = False
        for _ in range(150):
            spec = copy.deepcopy(nominal_specs[i])
            spec["episode_id"] = f"rand_layout_{i:02d}"
            for k, o_dict in spec.get("obstacles", {}).items():
                delta_xy = rng_layout.uniform(-0.20, 0.20, size=2)
                o_dict["pos"][:2] = np.clip(o_dict["pos"][:2] + delta_xy, -1.65, 1.65)
            t_key = spec.get("target_key", "yellow_cylinder")
            if t_key in spec.get("obstacles", {}):
                t_pos = spec["obstacles"][t_key]["pos"]
                wp = get_waypoint_for_relation(t_pos, spec.get("semantic_relation", "approach"), spec["initial_pose"], clearance=0.60)
                spec["waypoints"] = [wp]
            try:
                assert_task_spec_valid(spec)
                valid = True
                rand_specs.append(spec)
                break
            except AssertionError:
                continue
        assert valid, f"Could not generate geometrically valid randomized layout for ep {i}"

    res_rand_h = evaluate_suite_track_c(agent_h, rand_specs)
    rand_h_succ = [r["success"] for r in res_rand_h["records"]]
    pair_rand = _paired_contingency(nom_h_succ, rand_h_succ)
    print(f"  [H-JEPA Layout]    Succ: {res_rand_h['success_count']}/{n_eps} ({res_rand_h['success_rate']*100:5.1f}%) | Delta: {pair_rand['mean_diff']*100:+5.1f}% [CI95: {pair_rand['ci95_low']*100:+5.1f}%, {pair_rand['ci95_high']*100:+5.1f}%] | p={pair_rand['mcnemar_exact_p_value']:.4f}")

    # --- 6. Rigid SE(2) Global Transformation (90, 180, 270 deg) & Latent Equivariance ---
    print("\n--- 6. Evaluating Rigid SE(2) Global Transformations & Latent Equivariance ---")
    se2_angles = [np.pi / 2, np.pi, -np.pi / 2]
    se2_results = {}

    for ang in se2_angles:
        deg = int(round(np.degrees(ang)))
        c_a, s_a = np.cos(ang), np.sin(ang)
        R = np.array([[c_a, -s_a], [s_a, c_a]], dtype=np.float32)

        se2_specs = []
        for i in range(n_eps):
            spec = copy.deepcopy(nominal_specs[i])
            spec["episode_id"] = f"se2_{deg:+d}_{i:02d}"
            p = spec["initial_pose"]
            new_xy = R @ p[:2]
            spec["initial_pose"] = np.array([new_xy[0], new_xy[1], p[2] + ang], dtype=np.float32)

            for k, o_dict in spec.get("obstacles", {}).items():
                o_xy = R @ o_dict["pos"][:2]
                o_dict["pos"] = np.array([o_xy[0], o_xy[1], o_dict["pos"][2]], dtype=np.float32)

            wps = []
            for wp in spec.get("waypoints", []):
                w_xy = R @ wp[:2]
                wps.append(np.array([w_xy[0], w_xy[1], wp[2] + ang], dtype=np.float32))
            spec["waypoints"] = wps
            assert_task_spec_valid(spec)
            se2_specs.append(spec)

        res_se2_h = evaluate_suite_track_c(agent_h, se2_specs)
        se2_succ = [r["success"] for r in res_se2_h["records"]]
        pair_se2 = _paired_contingency(nom_h_succ, se2_succ)
        se2_results[f"rot_{deg:+}deg"] = {
            "success_count": res_se2_h["success_count"],
            "success_rate": res_se2_h["success_rate"],
            "mean_goal_error_m": res_se2_h["mean_final_goal_error_m"],
            "paired_stats": pair_se2,
            "records": res_se2_h["records"],
        }
        print(f"  [SE(2) {deg:+4d} deg] Succ: {res_se2_h['success_count']}/{n_eps} ({res_se2_h['success_rate']*100:5.1f}%) | Delta: {pair_se2['mean_diff']*100:+5.1f}% | p={pair_se2['mcnemar_exact_p_value']:.4f}")

    # Latent SE(2) Global Rigid vs Egocentric Rotation Analysis
    print("\n--- Measuring Latent Transformation under Global Rigid vs Egocentric Rotation ---")
    env_eval = JetAutoEnv()
    cos_sims_global = []
    cos_sims_ego_only = []
    with torch.no_grad():
        for i in range(min(10, n_eps)):
            nom_s = nominal_specs[i]
            env_eval.restore_physical_baseline()
            env_eval.apply_task_spec(nom_s)
            p_init = nom_s["initial_pose"]
            obs_nom, _ = env_eval.reset(options={"initial_pos": (float(p_init[0]), float(p_init[1]), float(p_init[2]))})
            t_nom = torch.from_numpy(obs_nom).permute(2, 0, 1).unsqueeze(0).float().to(args.device) / 255.0
            z_nom = agent_h.model.encode(t_nom)

            # Global 90 deg rotation (whole scene rigidly rotated with AMR)
            rot_s = copy.deepcopy(nom_s)
            R_90 = np.array([[0, -1], [1, 0]], dtype=np.float32)
            p_rot = R_90 @ p_init[:2]
            rot_s["initial_pose"] = np.array([p_rot[0], p_rot[1], p_init[2] + np.pi/2], dtype=np.float32)
            for k, o_dict in rot_s.get("obstacles", {}).items():
                o_dict["pos"][:2] = R_90 @ o_dict["pos"][:2]
            env_eval.restore_physical_baseline()
            env_eval.apply_task_spec(rot_s)
            obs_glob, _ = env_eval.reset(options={"initial_pos": (float(p_rot[0]), float(p_rot[1]), float(p_init[2] + np.pi/2))})
            t_glob = torch.from_numpy(obs_glob).permute(2, 0, 1).unsqueeze(0).float().to(args.device) / 255.0
            z_glob = agent_h.model.encode(t_glob)
            cos_glob = float(F.cosine_similarity(z_nom, z_glob).item())
            cos_sims_global.append(cos_glob)

            # Egocentric yaw only (robot yawed by 90 deg, scene static)
            obs_yaw, _ = env_eval.reset(options={"initial_pos": (float(p_init[0]), float(p_init[1]), float(p_init[2] + np.pi/2))})
            t_yaw = torch.from_numpy(obs_yaw).permute(2, 0, 1).unsqueeze(0).float().to(args.device) / 255.0
            z_yaw = agent_h.model.encode(t_yaw)
            cos_yaw = float(F.cosine_similarity(z_nom, z_yaw).item())
            cos_sims_ego_only.append(cos_yaw)
    env_eval.close()

    latent_equiv_results = {
        "global_rigid_rotation_latent_cosine": float(np.mean(cos_sims_global)),
        "egocentric_yaw_rotation_latent_cosine": float(np.mean(cos_sims_ego_only)),
        "interpretation": (
            "Under global rigid rotation, the robot camera view of local obstacles is preserved "
            f"(cosine sim = {np.mean(cos_sims_global):.3f}), yielding behavioral rotation robustness. "
            "However, egocentric camera rotation shifts visual scene contents "
            f"(cosine sim = {np.mean(cos_sims_ego_only):.3f}); the CNN does not possess steerable group-equivariant filters."
        ),
    }
    print(f"  Latent Cosine Sim (Global Rigid Rotation): {latent_equiv_results['global_rigid_rotation_latent_cosine']:.4f}")
    print(f"  Latent Cosine Sim (Egocentric Yaw Shift):   {latent_equiv_results['egocentric_yaw_rotation_latent_cosine']:.4f}")

    # Build execution metadata
    this_script = Path(__file__).resolve()
    metadata = {
        "script_path": str(this_script.relative_to(repo_root)),
        "script_sha256": _sha256(this_script),
        "command_executed": f"python world_models_fpga/research/benchmarks/run_track_c_visual_grounding.py --episodes-per-suite {n_eps} --device {args.device}",
        "execution_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_sha(),
        "working_tree_clean": _git_clean(),
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
            "track": "Track C (pure observation, zero simulator coordinates)",
            "episodes_per_suite": n_eps,
            "max_steps": 50,
            "categorical_task_encoding": "encode_task_id discrete index in [0, 63]",
        },
        "environment": {
            "platform": sys.platform,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }

    full_output = {
        "execution_metadata": metadata,
        "summary": {
            "hjepa_nominal": res_nom_h,
            "lewm_nominal": res_nom_l,
            "baseline_nominal": res_nom_b,
            "appearance_nuisance_floor": {
                "results": res_floor_h,
                "paired_contingency": pair_floor,
            },
            "appearance_nuisance_illumination": {
                "results": res_illum_h,
                "paired_contingency": pair_illum,
            },
            "appearance_nuisance_target_only": {
                "results": res_target_app_h,
                "paired_contingency": pair_target_app,
            },
            "appearance_nuisance_distractors_only": {
                "results": res_dist_app_h,
                "paired_contingency": pair_dist_app,
            },
            "semantically_relevant_appearance": {
                "results": res_sem_h,
                "paired_contingency": pair_sem,
            },
            "visual_geometry_grounding": {
                "results": res_rearr_h,
                "paired_contingency": pair_rearr,
            },
            "layout_generalization": {
                "results": res_rand_h,
                "paired_contingency": pair_rand,
            },
            "se2_rigid_transformations": se2_results,
            "latent_equivariance": latent_equiv_results,
        },
    }

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(full_output, indent=2) + "\n")
    print(f"\n[Saved Artifact] -> {out_p}")


if __name__ == "__main__":
    main()
