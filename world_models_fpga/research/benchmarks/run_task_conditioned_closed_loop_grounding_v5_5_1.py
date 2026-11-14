#!/usr/bin/env python3
"""Project Hephaestus: Phase 5.5.1 Geometry-Separated Task Selectivity & Feedback/Memory Decomposition.

Evaluates whether H-JEPA exhibits genuine task-conditioned physical object selection
and independently decomposes repeated visual feedback vs controller ego-memory across
an independent N=60 geometry-separated scene cohort (30 nominal + 30 illumination_shift)
evaluated under BOTH task assignments (A and B) across 7 conditions (840 episodes +
120 Oracle feasibility checks):

  2x2 Visual Feedback x Controller Memory Factorial:
    - F1_dynamic_visual_with_memory
    - F2_dynamic_visual_no_memory
    - F3_frozen_visual_with_memory   (anchors step-0 visual goal in world frame; keeps memory)
    - F4_frozen_visual_no_memory     (anchors step-0 visual goal in world frame; disables memory)

  Task-Object Visual Specificity & Planner Attribution:
    - M1_target_masked_rgb
    - M2_distractor_masked_rgb
    - O1_track_c_planner_goal_override (Track C pipeline intact; planner goal set to gt_t)
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "world_models_fpga"))
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy import stats
import torch

from env.amr_env import JetAutoEnv
from models.baseline_policy import encode_task_id
from research.benchmarks.model_wrappers import HJEPAAgent, OracleAgent, compute_body_frame_goal_vec
from research.benchmarks.run_task_conditioned_closed_loop_grounding_v5_5 import (
    GEOM_NAME_MAP,
    _opt_float_or_none,
    _safe_nanmean,
    _safe_nanmean_deg,
    apply_local_background_mask,
    body_goal_to_world_pose,
    build_area_matched_distractor_mask,
    compute_bytes_sha256,
    compute_file_sha256,
    extract_geom_mask_and_stats,
    get_git_info,
    holm_bonferroni_adjust,
    paired_bootstrap_diff_ci,
    paired_mcnemar_exact,
    render_segmentation_array,
    wilson_ci,
)
from research.datasets.dataset_config import CKPT_DIR
from research.datasets.generation.capability_tasks import CANONICAL_OBSTACLES


TARGET_KEYS: Tuple[str, ...] = (
    "red_box",
    "blue_cylinder",
    "green_cylinder",
    "yellow_cylinder",
    "purple_box",
    "cyan_cylinder",
)

V5_5_1_CONDITIONS: Tuple[str, ...] = (
    "F1_dynamic_visual_with_memory",
    "F2_dynamic_visual_no_memory",
    "F3_frozen_visual_with_memory",
    "F4_frozen_visual_no_memory",
    "M1_target_masked_rgb",
    "M2_distractor_masked_rgb",
    "O1_track_c_planner_goal_override",
)

MIN_WAYPOINT_SEPARATION_M: float = 0.90
MIN_TARGET_INIT_DIST_M: float = 0.60
MAX_TARGET_INIT_DIST_M: float = 1.20
DEFAULT_REACH_RADIUS_M: float = 0.35
MAX_JSONL_SHARD_BYTES: int = 4_200_000  # Strictly < 5 MB (5,242,880 B) GitHub UI threshold


def _to_json_clean(obj: Any) -> Any:
    """Recursively converts numpy values and NaNs into valid JSON primitives (NaN -> None)."""
    if isinstance(obj, np.ndarray):
        return _to_json_clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _to_json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_clean(v) for v in obj]
    return obj


def _paired_wilcoxon_p(vals_a: Sequence[float], vals_b: Sequence[float]) -> float:
    a = np.asarray(vals_a, dtype=np.float64)
    b = np.asarray(vals_b, dtype=np.float64)
    diff = a - b
    valid = ~np.isnan(diff)
    diff = diff[valid]
    if diff.size == 0 or np.allclose(diff, 0.0):
        return 1.0
    return float(stats.wilcoxon(diff, alternative="two-sided").pvalue)


def _one_sample_wilcoxon_zero_p(vals: Sequence[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0 or np.allclose(arr, 0.0):
        return 1.0
    return float(stats.wilcoxon(arr, alternative="two-sided").pvalue)


def _mean_bootstrap_ci(vals: Sequence[float], n_boot: int = 4000, seed: int = 5510) -> Tuple[float, float, float]:
    arr = np.asarray([float(v) for v in vals if v is not None and not np.isnan(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_means = arr[idx].mean(axis=1)
    return float(np.mean(arr)), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _check_corridor_clearance(
    robot_xy: np.ndarray,
    wp_1: np.ndarray,
    wp_2: np.ndarray,
    obstacles: Dict[str, Dict[str, Any]],
    active_keys: Tuple[str, str],
    min_obj_clearance: float = 0.50,
    min_wp_clearance: float = 0.45,
) -> bool:
    """Checks clearance of placed target/distractor and waypoints against other obstacles."""
    pos_1 = np.asarray(obstacles[active_keys[0]]["pos"][:2], dtype=np.float64)
    pos_2 = np.asarray(obstacles[active_keys[1]]["pos"][:2], dtype=np.float64)
    for k, obst in obstacles.items():
        if k in active_keys:
            continue
        op = np.asarray(obst["pos"][:2], dtype=np.float64)
        if np.linalg.norm(pos_1 - op) < min_obj_clearance or np.linalg.norm(pos_2 - op) < min_obj_clearance:
            return False
        if np.linalg.norm(wp_1[:2] - op) < min_wp_clearance or np.linalg.norm(wp_2[:2] - op) < min_wp_clearance:
            return False
        if np.linalg.norm(robot_xy[:2] - op) < min_wp_clearance:
            return False
        # Check midpoint along robot -> wp1 and robot -> wp2 corridors so robot won't hit stray obstacles
        mid_1 = 0.5 * (robot_xy[:2] + wp_1[:2])
        mid_2 = 0.5 * (robot_xy[:2] + wp_2[:2])
        if np.linalg.norm(mid_1 - op) < 0.38 or np.linalg.norm(mid_2 - op) < 0.38:
            return False
    return True


def make_assignment_spec(scene_record: Dict[str, Any], assignment: str) -> Dict[str, Any]:
    """Constructs the JetAutoEnv/agent task spec for assignment 'A' (obj_1) or 'B' (obj_2)."""
    if assignment not in ("A", "B"):
        raise ValueError(f"Unsupported assignment {assignment!r}; expected 'A' or 'B'")
    if assignment == "A":
        tgt_key = scene_record["obj_1_key"]
        dist_key = scene_record["obj_2_key"]
        eval_wp = list(scene_record["wp_1"])
        dist_wp = list(scene_record["wp_2"])
        assigned_side = scene_record["obj_1_side"]
        task_tok = int(encode_task_id(tgt_key, "approach", f"approach {tgt_key}"))
        swap_tok = int(encode_task_id(dist_key, "approach", f"approach {dist_key}"))
    else:
        tgt_key = scene_record["obj_2_key"]
        dist_key = scene_record["obj_1_key"]
        eval_wp = list(scene_record["wp_2"])
        dist_wp = list(scene_record["wp_1"])
        assigned_side = scene_record["obj_2_side"]
        task_tok = int(encode_task_id(tgt_key, "approach", f"approach {tgt_key}"))
        swap_tok = int(encode_task_id(dist_key, "approach", f"approach {dist_key}"))

    return {
        "scene_id": scene_record["scene_id"],
        "scene_idx": int(scene_record["scene_idx"]),
        "assignment": assignment,
        "context": scene_record["context"],
        "category_pair": scene_record["category_pair"],
        "target_key": tgt_key,
        "distractor_key": dist_key,
        "assigned_side": assigned_side,
        "semantic_relation": "approach",
        "task_token": task_tok,
        "swapped_task_token": swap_tok,
        "initial_pose": list(scene_record["initial_pose"]),
        "waypoints": [eval_wp],
        "distractor_waypoint": dist_wp,
        "obstacles": copy.deepcopy(scene_record["obstacles"]),
        "light_diffuse_scale": float(scene_record["light_diffuse_scale"]),
    }


def _oracle_solves_assignment(
    env: JetAutoEnv,
    oracle: OracleAgent,
    spec: Dict[str, Any],
    max_steps: int,
    goal_reach_threshold: float,
) -> Tuple[bool, Optional[int], float]:
    """Verifies physical solvability of a single task assignment using Track A OracleAgent."""
    env.restore_physical_baseline()
    env.apply_task_spec(spec)
    init_pos = np.asarray(spec["initial_pose"], dtype=np.float32)
    env.reset(options={"initial_pos": (float(init_pos[0]), float(init_pos[1]), float(init_pos[2]))})
    eval_wp = np.asarray(spec["waypoints"][-1], dtype=np.float32)
    oracle_wp = np.array(
        [eval_wp[0], eval_wp[1], float(np.arctan2(eval_wp[1] - init_pos[1], eval_wp[0] - init_pos[0]))],
        dtype=np.float32,
    )
    reached_step: Optional[int] = None
    collision = False
    for step in range(max_steps):
        pose = env.get_robot_pose()
        act, _ = oracle.act(np.zeros((64, 64, 3), dtype=np.uint8), robot_pose=pose, target_pose=oracle_wp, task_spec=spec)
        _, _, _, _, info = env.step(act)
        if info.get("collision", False):
            collision = True
        dist_now = float(np.linalg.norm(env.get_robot_pose()[:2] - eval_wp[:2]))
        if dist_now <= goal_reach_threshold:
            reached_step = step + 1
            break
    final_err = float(np.linalg.norm(env.get_robot_pose()[:2] - eval_wp[:2]))
    ok = bool((reached_step is not None) and (not collision) and (final_err <= goal_reach_threshold))
    return ok, reached_step, final_err


def build_geometry_separated_cohort(
    env: JetAutoEnv,
    oracle: OracleAgent,
    n_nominal: int = 30,
    n_illum: int = 30,
    min_wp_separation_m: float = MIN_WAYPOINT_SEPARATION_M,
    min_target_dist_m: float = MIN_TARGET_INIT_DIST_M,
    max_target_dist_m: float = MAX_TARGET_INIT_DIST_M,
    min_visible_pixels: int = 60,
    max_steps: int = 45,
    goal_reach_threshold: float = DEFAULT_REACH_RADIUS_M,
    seed: int = 5510,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generates N=60 independent geometry-separated scenes (30 nominal + 30 illum)."""
    rng = np.random.default_rng(seed)
    all_pairs = list(itertools.combinations(TARGET_KEYS, 2))  # 15 unique pairs

    def _build_context_schedule(n_scenes: int, context_name: str) -> List[Dict[str, Any]]:
        slots = []
        # Repeat/tile balanced 15 unordered pairs cleanly with alternating A/B role & left/right side
        for i in range(n_scenes):
            base_k1, base_k2 = all_pairs[i % len(all_pairs)]
            cycle = i // len(all_pairs)
            key_1, key_2 = (base_k1, base_k2) if (cycle % 2 == 0) else (base_k2, base_k1)
            side_sign = 1.0 if (i % 2 == 0) else -1.0  # +1: obj_1 on left (+bearing); -1: obj_1 on right
            slots.append(
                {
                    "context": context_name,
                    "obj_1_key": key_1,
                    "obj_2_key": key_2,
                    "side_sign": side_sign,
                    "category_pair": f"{min(key_1, key_2)}__{max(key_1, key_2)}",
                }
            )
        return slots

    schedule = _build_context_schedule(n_nominal, "nominal") + _build_context_schedule(n_illum, "illumination_shift")
    rejection_counts: Dict[str, int] = {}
    accepted_scenes: List[Dict[str, Any]] = []
    total_candidates = 0
    max_attempts_per_slot = 500

    for scene_idx, slot in enumerate(schedule):
        context = slot["context"]
        key_1 = slot["obj_1_key"]
        key_2 = slot["obj_2_key"]
        side_sign = float(slot["side_sign"])
        accepted_this = False

        for _attempt in range(max_attempts_per_slot):
            total_candidates += 1
            rx = float(rng.uniform(-1.05, 1.05))
            ry = float(rng.uniform(-1.05, 1.05))
            rth = float(rng.uniform(-np.pi, np.pi))
            robot_pose = np.array([rx, ry, rth], dtype=np.float32)

            b1 = side_sign * float(rng.uniform(0.39, 0.49))
            b2 = -side_sign * float(rng.uniform(0.39, 0.49))
            d1 = float(rng.uniform(1.42, 1.80))
            d2 = float(rng.uniform(1.42, 1.80))

            p1_x = float(rx + d1 * np.cos(rth + b1))
            p1_y = float(ry + d1 * np.sin(rth + b1))
            p2_x = float(rx + d2 * np.cos(rth + b2))
            p2_y = float(ry + d2 * np.sin(rth + b2))

            if not (-2.05 <= p1_x <= 2.05 and -2.05 <= p1_y <= 2.05 and -2.05 <= p2_x <= 2.05 and -2.05 <= p2_y <= 2.05):
                rejection_counts["arena_bounds"] = rejection_counts.get("arena_bounds", 0) + 1
                continue

            pos_1 = np.array([p1_x, p1_y, 0.35], dtype=np.float32)
            pos_2 = np.array([p2_x, p2_y, 0.35], dtype=np.float32)
            wp_1 = (pos_1[:2] - (pos_1[:2] - robot_pose[:2]) / d1 * 0.60).astype(np.float32)
            wp_2 = (pos_2[:2] - (pos_2[:2] - robot_pose[:2]) / d2 * 0.60).astype(np.float32)

            d_wp1 = float(np.linalg.norm(wp_1 - robot_pose[:2]))
            d_wp2 = float(np.linalg.norm(wp_2 - robot_pose[:2]))
            if not (min_target_dist_m <= d_wp1 <= max_target_dist_m and min_target_dist_m <= d_wp2 <= max_target_dist_m):
                rejection_counts["waypoint_dist_out_of_range"] = rejection_counts.get("waypoint_dist_out_of_range", 0) + 1
                continue

            wp_sep = float(np.linalg.norm(wp_1 - wp_2))
            if wp_sep < min_wp_separation_m:
                rejection_counts["waypoint_sep_below_threshold"] = rejection_counts.get("waypoint_sep_below_threshold", 0) + 1
                continue

            obj_sep = float(np.linalg.norm(pos_1[:2] - pos_2[:2]))
            if obj_sep < 0.70:
                rejection_counts["object_sep_too_small"] = rejection_counts.get("object_sep_too_small", 0) + 1
                continue

            obstacles = copy.deepcopy(CANONICAL_OBSTACLES)
            obstacles[key_1]["pos"] = pos_1.tolist()
            obstacles[key_2]["pos"] = pos_2.tolist()

            if not _check_corridor_clearance(robot_pose[:2], wp_1, wp_2, obstacles, (key_1, key_2)):
                rejection_counts["obstacle_clearance"] = rejection_counts.get("obstacle_clearance", 0) + 1
                continue

            if context == "nominal":
                light_scale = 1.0
            else:
                light_scale = float(rng.uniform(0.45, 0.72) if rng.uniform() < 0.5 else rng.uniform(1.25, 1.55))

            scene_rec = {
                "scene_id": f"SCENE_{scene_idx:03d}",
                "scene_idx": scene_idx,
                "context": context,
                "category_pair": slot["category_pair"],
                "obj_1_key": key_1,
                "obj_2_key": key_2,
                "obj_1_side": "left" if b1 > 0 else "right",
                "obj_2_side": "left" if b2 > 0 else "right",
                "bearing_1_rad": float(b1),
                "bearing_2_rad": float(b2),
                "initial_pose": [float(rx), float(ry), float(rth)],
                "obj_1_pos": [float(p1_x), float(p1_y), 0.35],
                "obj_2_pos": [float(p2_x), float(p2_y), 0.35],
                "wp_1": [float(wp_1[0]), float(wp_1[1]), 0.0],
                "wp_2": [float(wp_2[0]), float(wp_2[1]), 0.0],
                "init_dist_wp1_m": d_wp1,
                "init_dist_wp2_m": d_wp2,
                "waypoint_separation_m": wp_sep,
                "object_separation_m": obj_sep,
                "reach_bubble_gap_m": float(wp_sep - 2.0 * goal_reach_threshold),
                "obstacles": {k: {"pos": [float(v) for v in val["pos"]], "size": val.get("size")} if isinstance(val, dict) else val for k, val in obstacles.items()},
                "light_diffuse_scale": float(light_scale),
            }
            # Preserve exact structure from CANONICAL_OBSTACLES for apply_task_spec
            raw_obs = copy.deepcopy(CANONICAL_OBSTACLES)
            raw_obs[key_1]["pos"] = pos_1.tolist()
            raw_obs[key_2]["pos"] = pos_2.tolist()
            for ok in raw_obs:
                if isinstance(raw_obs[ok].get("pos"), np.ndarray):
                    raw_obs[ok]["pos"] = raw_obs[ok]["pos"].tolist()
            scene_rec["obstacles"] = raw_obs

            # Check initial visibility of BOTH objects
            spec_a = make_assignment_spec(scene_rec, "A")
            env.restore_physical_baseline()
            env.apply_task_spec(spec_a)
            env.reset(options={"initial_pos": (rx, ry, rth)})
            seg0 = render_segmentation_array(env)
            gid1 = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, GEOM_NAME_MAP[key_1])
            gid2 = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, GEOM_NAME_MAP[key_2])
            _, px1, _ = extract_geom_mask_and_stats(seg0, gid1)
            _, px2, _ = extract_geom_mask_and_stats(seg0, gid2)
            if px1 < min_visible_pixels or px2 < min_visible_pixels:
                rejection_counts["initial_visibility_failed"] = rejection_counts.get("initial_visibility_failed", 0) + 1
                continue

            # Check Track A Oracle reachability on BOTH Assignment A and Assignment B
            ok_a, step_a, err_a = _oracle_solves_assignment(env, oracle, spec_a, max_steps, goal_reach_threshold)
            if not ok_a:
                rejection_counts["oracle_unreachable_A"] = rejection_counts.get("oracle_unreachable_A", 0) + 1
                continue
            spec_b = make_assignment_spec(scene_rec, "B")
            ok_b, step_b, err_b = _oracle_solves_assignment(env, oracle, spec_b, max_steps, goal_reach_threshold)
            if not ok_b:
                rejection_counts["oracle_unreachable_B"] = rejection_counts.get("oracle_unreachable_B", 0) + 1
                continue

            scene_rec["init_vis_px_1"] = int(px1)
            scene_rec["init_vis_px_2"] = int(px2)
            scene_rec["oracle_A_reach_step"] = int(step_a)
            scene_rec["oracle_A_final_err_m"] = float(err_a)
            scene_rec["oracle_B_reach_step"] = int(step_b)
            scene_rec["oracle_B_final_err_m"] = float(err_b)
            accepted_scenes.append(scene_rec)
            accepted_this = True
            break

        if not accepted_this:
            raise RuntimeError(f"Failed to sample scene {scene_idx} ({context}, {key_1}, {key_2}) within {max_attempts_per_slot} trials.")

    cohort_report = {
        "n_accepted_scenes": len(accepted_scenes),
        "n_nominal_scenes": sum(1 for s in accepted_scenes if s["context"] == "nominal"),
        "n_illum_scenes": sum(1 for s in accepted_scenes if s["context"] == "illumination_shift"),
        "total_candidates_evaluated": total_candidates,
        "n_rejected_candidates": total_candidates - len(accepted_scenes),
        "acceptance_rate": float(len(accepted_scenes) / max(1, total_candidates)),
        "rejection_breakdown": rejection_counts,
        "min_waypoint_separation_m": float(min(s["waypoint_separation_m"] for s in accepted_scenes)),
        "max_waypoint_separation_m": float(max(s["waypoint_separation_m"] for s in accepted_scenes)),
        "mean_waypoint_separation_m": float(np.mean([s["waypoint_separation_m"] for s in accepted_scenes])),
        "min_init_target_dist_m": float(min(min(s["init_dist_wp1_m"], s["init_dist_wp2_m"]) for s in accepted_scenes)),
        "max_init_target_dist_m": float(max(max(s["init_dist_wp1_m"], s["init_dist_wp2_m"]) for s in accepted_scenes)),
        "min_init_visible_px": int(min(min(s["init_vis_px_1"], s["init_vis_px_2"]) for s in accepted_scenes)),
        "oracle_A_success_count": len(accepted_scenes),
        "oracle_B_success_count": len(accepted_scenes),
    }
    return accepted_scenes, cohort_report


def compute_axis_projections(
    vec_xy: np.ndarray,
    gt_assigned_xy: np.ndarray,
    gt_unassigned_xy: np.ndarray,
) -> Tuple[float, float]:
    """Computes signed projection relative to midpoint along (assigned - unassigned) in meters and normalized [-0.5, +0.5] units."""
    if np.any(np.isnan(vec_xy)):
        return float("nan"), float("nan")
    diff = np.asarray(gt_assigned_xy, dtype=np.float64) - np.asarray(gt_unassigned_xy, dtype=np.float64)
    norm = float(np.linalg.norm(diff))
    if norm <= 1e-8:
        return float("nan"), float("nan")
    mid = 0.5 * (np.asarray(gt_assigned_xy, dtype=np.float64) + np.asarray(gt_unassigned_xy, dtype=np.float64))
    delta = np.asarray(vec_xy, dtype=np.float64) - mid
    u = diff / norm
    proj_m = float(np.dot(delta, u))
    coord_norm = float(np.dot(delta, diff) / (norm * norm))
    return proj_m, coord_norm


def evaluate_v5_5_1_episode(
    agent: HJEPAAgent,
    env: JetAutoEnv,
    spec: Dict[str, Any],
    condition_name: str,
    max_steps: int = 45,
    goal_reach_threshold: float = DEFAULT_REACH_RADIUS_M,
    capture_rgb_keyframes: bool = False,
) -> Dict[str, Any]:
    """Runs one Phase 5.5.1 closed-loop episode under a pre-registered condition."""
    if condition_name not in V5_5_1_CONDITIONS:
        raise ValueError(f"Unsupported Phase 5.5.1 condition {condition_name!r}")

    env.restore_physical_baseline()
    agent.clear_branch_interceptors()
    agent.reset()

    original_track = agent.track
    original_disable_memory = bool(getattr(agent, "disable_memory", False))
    agent.track = "observation"
    agent.disable_memory = condition_name in ("F2_dynamic_visual_no_memory", "F4_frozen_visual_no_memory")

    l1_captured: List[Optional[np.ndarray]] = [None]
    l2_captured: List[Optional[np.ndarray]] = [None]
    hooks = []
    if hasattr(agent.model, "grounding_head") and len(agent.model.grounding_head) >= 4:
        def _hook_l1(_mod, _inp, out):
            l1_captured[0] = out.squeeze(0).detach().cpu().numpy().astype(np.float32)

        def _hook_l2(_mod, _inp, out):
            l2_captured[0] = out.squeeze(0).detach().cpu().numpy().astype(np.float32)

        hooks.append(agent.model.grounding_head[1].register_forward_hook(_hook_l1))
        hooks.append(agent.model.grounding_head[3].register_forward_hook(_hook_l2))

    try:
        env.apply_task_spec(spec)
        init_pos = np.asarray(spec["initial_pose"], dtype=np.float32)
        obs, _ = env.reset(options={"initial_pos": (float(init_pos[0]), float(init_pos[1]), float(init_pos[2]))})
        obs_unmasked_0_hash = compute_bytes_sha256(obs.tobytes())

        eval_wp = np.asarray(spec["waypoints"][-1], dtype=np.float32)
        dist_wp = np.asarray(spec["distractor_waypoint"], dtype=np.float32)
        target_key = spec["target_key"]
        distractor_key = spec["distractor_key"]
        gid_target = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, GEOM_NAME_MAP[target_key])
        gid_distractor = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, GEOM_NAME_MAP[distractor_key])

        initial_assigned_dist = float(np.linalg.norm(init_pos[:2] - eval_wp[:2]))
        initial_unassigned_dist = float(np.linalg.norm(init_pos[:2] - dist_wp[:2]))
        initially_inside_radius = bool(initial_assigned_dist <= goal_reach_threshold)

        current_pose_holder: List[np.ndarray] = [init_pos.copy()]
        frozen_world_vis_holder: List[Optional[np.ndarray]] = [None]
        current_gt_holder: List[Optional[np.ndarray]] = [None]

        nom_tid = int(spec["task_token"])
        agent.set_branch_interceptor("task_id", lambda _tid, nt=nom_tid: nt)

        if condition_name in ("F3_frozen_visual_with_memory", "F4_frozen_visual_no_memory"):
            def _freeze_visual_goal(vis_goal_t):
                if frozen_world_vis_holder[0] is None:
                    v0 = vis_goal_t.squeeze(0).detach().cpu().numpy()
                    frozen_world_vis_holder[0] = body_goal_to_world_pose(current_pose_holder[0], v0)
                v_body = compute_body_frame_goal_vec(current_pose_holder[0], frozen_world_vis_holder[0])
                return torch.as_tensor(v_body, dtype=vis_goal_t.dtype, device=vis_goal_t.device).view(1, 4)

            agent.set_branch_interceptor("visual_goal", _freeze_visual_goal)

        elif condition_name == "O1_track_c_planner_goal_override":
            def _override_planner_goal(_goal_t, _diag=None, holder=current_gt_holder):
                return torch.as_tensor(holder[0], dtype=torch.float32, device=agent.device).view(1, 4)

            agent.set_branch_interceptor("planner_goal", _override_planner_goal)

        # Accumulators
        tr_pose: List[List[float]] = []
        tr_latent_s: List[np.ndarray] = []
        tr_task_emb: List[np.ndarray] = []
        tr_ground_z: List[np.ndarray] = []
        tr_l1: List[np.ndarray] = []
        tr_l2: List[np.ndarray] = []
        tr_g_vis: List[np.ndarray] = []
        tr_g_mem: List[np.ndarray] = []
        tr_g_blend: List[np.ndarray] = []
        tr_g_planner: List[np.ndarray] = []
        tr_delta_ego: List[np.ndarray] = []
        tr_gt_assigned: List[np.ndarray] = []
        tr_gt_unassigned: List[np.ndarray] = []
        tr_a_react: List[np.ndarray] = []
        tr_a_exec: List[np.ndarray] = []
        tr_best_macro_idx: List[int] = []
        tr_pred_macro_disp: List[np.ndarray] = []
        tr_pred_micro_disp: List[np.ndarray] = []
        tr_col_risk: List[float] = []
        tr_dist_assigned: List[float] = []
        tr_dist_unassigned: List[float] = []
        tr_assigned_vis_px: List[int] = []
        tr_unassigned_vis_px: List[int] = []
        tr_masked_px: List[int] = []
        tr_fallback_mask: List[bool] = []
        tr_act_assigned_cos: List[float] = []
        tr_act_unassigned_cos: List[float] = []
        tr_goal_ang_err: List[float] = []
        tr_vis_ang_err: List[float] = []
        tr_planner_ang_err: List[float] = []
        tr_proj_vis_m: List[float] = []
        tr_proj_blend_m: List[float] = []
        tr_coord_vis: List[float] = []
        tr_coord_blend: List[float] = []
        tr_task_emb_exec: List[bool] = []
        tr_ground_exec: List[bool] = []
        tr_mem_exec: List[bool] = []
        tr_planner_sources: List[str] = []
        captured_frames: Dict[str, np.ndarray] = {}

        collision_occurred = False
        assigned_reached = False
        unassigned_reached = False
        first_assigned_step: Optional[int] = None
        first_unassigned_step: Optional[int] = None
        prev_xy = init_pos[:2].copy()
        total_path = 0.0

        def _vec_or_nan(val: Any, dim: int) -> np.ndarray:
            if val is None:
                return np.full(dim, np.nan, dtype=np.float32)
            return np.asarray(val, dtype=np.float32)

        for step in range(max_steps):
            curr_pose = env.get_robot_pose()
            current_pose_holder[0] = curr_pose
            tr_pose.append(curr_pose.tolist())

            gt_a = compute_body_frame_goal_vec(curr_pose, eval_wp)
            gt_u = compute_body_frame_goal_vec(curr_pose, dist_wp)
            current_gt_holder[0] = gt_a
            tr_gt_assigned.append(gt_a)
            tr_gt_unassigned.append(gt_u)
            tr_dist_assigned.append(float(np.linalg.norm(curr_pose[:2] - eval_wp[:2])))
            tr_dist_unassigned.append(float(np.linalg.norm(curr_pose[:2] - dist_wp[:2])))

            seg = render_segmentation_array(env)
            mask_t, px_t, _ = extract_geom_mask_and_stats(seg, gid_target)
            mask_d, px_d, _ = extract_geom_mask_and_stats(seg, gid_distractor)
            tr_assigned_vis_px.append(px_t)
            tr_unassigned_vis_px.append(px_d)

            obs_for_agent = obs
            masked_px = 0
            fallback_used = False
            if condition_name == "M1_target_masked_rgb":
                obs_for_agent = apply_local_background_mask(obs, mask_t)
                masked_px = int(mask_t.sum())
            elif condition_name == "M2_distractor_masked_rgb":
                matched_mask, fallback_used = build_area_matched_distractor_mask(mask_t, mask_d)
                obs_for_agent = apply_local_background_mask(obs, matched_mask)
                masked_px = int(matched_mask.sum())

            if capture_rgb_keyframes and step == 0:
                captured_frames["step_0_unmasked"] = obs.copy()
                captured_frames["step_0_agent_input"] = obs_for_agent.copy()

            tr_masked_px.append(masked_px)
            tr_fallback_mask.append(bool(fallback_used))
            l1_captured[0] = None
            l2_captured[0] = None

            act, diag = agent.act(obs_for_agent, robot_pose=curr_pose, task_spec=spec)

            tr_latent_s.append(np.asarray(diag["latent"], dtype=np.float32))
            tr_task_emb.append(_vec_or_nan(diag.get("task_embedding"), 16))
            tr_ground_z.append(_vec_or_nan(diag.get("grounding_latent"), 32))
            tr_l1.append(l1_captured[0] if l1_captured[0] is not None else np.full(64, np.nan, dtype=np.float32))
            tr_l2.append(l2_captured[0] if l2_captured[0] is not None else np.full(64, np.nan, dtype=np.float32))

            g_vis = _vec_or_nan(diag.get("g_visual_t"), 4)
            g_mem = _vec_or_nan(diag.get("g_memory_t"), 4)
            g_bld = _vec_or_nan(diag.get("g_blended_t"), 4)
            g_pln = _vec_or_nan(diag.get("planner_goal_t"), 4)
            d_ego = _vec_or_nan(diag.get("delta_ego_t"), 3)

            tr_g_vis.append(g_vis)
            tr_g_mem.append(g_mem)
            tr_g_blend.append(g_bld)
            tr_g_planner.append(g_pln)
            tr_delta_ego.append(d_ego)
            tr_task_emb_exec.append(bool(diag.get("task_embedding_executed", False)))
            tr_ground_exec.append(bool(diag.get("grounding_head_executed", False)))
            tr_mem_exec.append(bool(diag.get("memory_branch_executed", False)))
            tr_planner_sources.append(str(diag.get("planner_goal_source", "unknown")))
            tr_a_react.append(np.asarray(diag.get("a_react", act), dtype=np.float32))
            tr_a_exec.append(np.asarray(act, dtype=np.float32))
            tr_best_macro_idx.append(int(diag.get("best_macro_idx", -1)))
            tr_pred_macro_disp.append(_vec_or_nan(diag.get("pred_macro_disp"), 3))
            tr_pred_micro_disp.append(_vec_or_nan(diag.get("pred_micro_disp"), 3))
            tr_col_risk.append(float(diag.get("col_risk", np.nan)))

            ang_gt = float(np.arctan2(gt_a[1], gt_a[0]))

            def _ang_err_or_nan(vec: np.ndarray) -> float:
                if np.any(np.isnan(vec[:2])):
                    return float("nan")
                ang_v = float(np.arctan2(vec[1], vec[0]))
                return float(abs(np.arctan2(np.sin(ang_v - ang_gt), np.cos(ang_v - ang_gt))))

            tr_goal_ang_err.append(_ang_err_or_nan(g_bld))
            tr_vis_ang_err.append(_ang_err_or_nan(g_vis))
            tr_planner_ang_err.append(_ang_err_or_nan(g_pln))

            pv_m, cv_norm = compute_axis_projections(g_vis[:2], gt_a[:2], gt_u[:2])
            pb_m, cb_norm = compute_axis_projections(g_bld[:2], gt_a[:2], gt_u[:2])
            tr_proj_vis_m.append(pv_m)
            tr_coord_vis.append(cv_norm)
            tr_proj_blend_m.append(pb_m)
            tr_coord_blend.append(cb_norm)

            act_xy = np.asarray(act[:2], dtype=np.float64)
            na = float(np.linalg.norm(act_xy))
            gta_xy = np.asarray(gt_a[:2], dtype=np.float64)
            gtu_xy = np.asarray(gt_u[:2], dtype=np.float64)
            nga, ngu = float(np.linalg.norm(gta_xy)), float(np.linalg.norm(gtu_xy))
            tr_act_assigned_cos.append(float(np.dot(act_xy, gta_xy) / (na * nga)) if (na > 1e-8 and nga > 1e-8) else 0.0)
            tr_act_unassigned_cos.append(float(np.dot(act_xy, gtu_xy) / (na * ngu)) if (na > 1e-8 and ngu > 1e-8) else 0.0)

            step_res = env.step(act)
            obs = step_res[0]
            info = step_res[-1]
            if info.get("collision", False):
                collision_occurred = True

            post_pose = env.get_robot_pose()
            curr_xy = post_pose[:2]
            total_path += float(np.linalg.norm(curr_xy - prev_xy))
            prev_xy = curr_xy.copy()

            if capture_rgb_keyframes:
                captured_frames["last_obs"] = obs.copy()

            if float(np.linalg.norm(curr_xy - dist_wp[:2])) <= goal_reach_threshold:
                unassigned_reached = True
                if first_unassigned_step is None:
                    first_unassigned_step = step + 1

            if float(np.linalg.norm(curr_xy - eval_wp[:2])) <= goal_reach_threshold:
                assigned_reached = True
                if first_assigned_step is None:
                    first_assigned_step = step + 1
                break

        final_pose = env.get_robot_pose()
        final_assigned_err = float(np.linalg.norm(final_pose[:2] - eval_wp[:2]))
        final_unassigned_err = float(np.linalg.norm(final_pose[:2] - dist_wp[:2]))

        assigned_success = bool(
            (not initially_inside_radius)
            and assigned_reached
            and (not collision_occurred)
            and (final_assigned_err <= goal_reach_threshold)
        )
        if first_assigned_step is not None and (first_unassigned_step is None or first_assigned_step <= first_unassigned_step):
            first_reached_object = "assigned"
        elif first_unassigned_step is not None:
            first_reached_object = "unassigned"
        else:
            first_reached_object = "neither"

        ep_rec: Dict[str, Any] = {
            "scene_id": spec["scene_id"],
            "scene_idx": int(spec["scene_idx"]),
            "assignment": spec["assignment"],
            "episode_key": f"{condition_name}__{spec['scene_id']}__{spec['assignment']}",
            "context": spec["context"],
            "category_pair": spec["category_pair"],
            "condition": condition_name,
            "target_key": target_key,
            "distractor_key": distractor_key,
            "assigned_side": spec["assigned_side"],
            "obs_unmasked_0_sha256": obs_unmasked_0_hash,
            "initially_inside_radius": initially_inside_radius,
            "initial_assigned_dist_m": initial_assigned_dist,
            "initial_unassigned_dist_m": initial_unassigned_dist,
            "waypoint_separation_m": float(np.linalg.norm(eval_wp[:2] - dist_wp[:2])),
            "final_assigned_goal_error_m": final_assigned_err,
            "final_unassigned_goal_error_m": final_unassigned_err,
            "final_margin_unassigned_minus_assigned_m": float(final_unassigned_err - final_assigned_err),
            "closer_to_assigned": bool(final_assigned_err < final_unassigned_err),
            "closer_to_unassigned": bool(final_unassigned_err < final_assigned_err),
            "path_length_m": float(total_path),
            "executed_steps": len(tr_a_exec),
            "first_assigned_reach_step": first_assigned_step,
            "first_unassigned_reach_step": first_unassigned_step,
            "first_reached_object": first_reached_object,
            "assigned_reached": assigned_reached,
            "unassigned_reached": unassigned_reached,
            "collision": collision_occurred,
            "assigned_success": assigned_success,
            "step0_vis_goal_ang_err_deg": (None if np.isnan(tr_vis_ang_err[0]) else float(np.degrees(tr_vis_ang_err[0]))),
            "step0_blended_goal_ang_err_deg": (None if np.isnan(tr_goal_ang_err[0]) else float(np.degrees(tr_goal_ang_err[0]))),
            "step0_proj_vis_m": _opt_float_or_none(tr_proj_vis_m[0]),
            "step0_coord_vis_norm": _opt_float_or_none(tr_coord_vis[0]),
            "mean_step_vis_goal_ang_err_deg": _safe_nanmean_deg(tr_vis_ang_err),
            "mean_step_blended_goal_ang_err_deg": _safe_nanmean_deg(tr_goal_ang_err),
            "mean_step_planner_goal_ang_err_deg": _safe_nanmean_deg(tr_planner_ang_err),
            "mean_step_proj_vis_m": _safe_nanmean(tr_proj_vis_m),
            "mean_step_proj_blended_m": _safe_nanmean(tr_proj_blend_m),
            "mean_step_coord_vis_norm": _safe_nanmean(tr_coord_vis),
            "mean_step_coord_blended_norm": _safe_nanmean(tr_coord_blend),
            "mean_step_action_assigned_cos": float(np.mean(tr_act_assigned_cos)),
            "mean_step_action_unassigned_cos": float(np.mean(tr_act_unassigned_cos)),
            "task_embedding_executed": bool(any(tr_task_emb_exec)),
            "grounding_head_executed": bool(any(tr_ground_exec)),
            "memory_branch_enabled": bool(not agent.disable_memory),
            "memory_branch_executed": bool(any(tr_mem_exec)),
            "memory_branch_executed_steps": int(sum(1 for x in tr_mem_exec if x)),
            "planner_goal_source": tr_planner_sources[-1] if tr_planner_sources else "unknown",
            "mean_assigned_visible_pixels": float(np.mean(tr_assigned_vis_px)),
            "mean_unassigned_visible_pixels": float(np.mean(tr_unassigned_vis_px)),
            "mean_masked_pixels": float(np.mean(tr_masked_px)),
            "fallback_mask_steps": int(sum(1 for x in tr_fallback_mask if x)),
            "traces": {
                "pose": np.asarray(tr_pose, dtype=np.float32),
                "latent_s": np.asarray(tr_latent_s, dtype=np.float32),
                "task_embedding": np.asarray(tr_task_emb, dtype=np.float32),
                "grounding_latent": np.asarray(tr_ground_z, dtype=np.float32),
                "l1_act": np.asarray(tr_l1, dtype=np.float32),
                "l2_act": np.asarray(tr_l2, dtype=np.float32),
                "g_visual": np.asarray(tr_g_vis, dtype=np.float32),
                "g_memory": np.asarray(tr_g_mem, dtype=np.float32),
                "g_blended": np.asarray(tr_g_blend, dtype=np.float32),
                "planner_goal": np.asarray(tr_g_planner, dtype=np.float32),
                "delta_ego": np.asarray(tr_delta_ego, dtype=np.float32),
                "gt_assigned_body": np.asarray(tr_gt_assigned, dtype=np.float32),
                "gt_unassigned_body": np.asarray(tr_gt_unassigned, dtype=np.float32),
                "a_react": np.asarray(tr_a_react, dtype=np.float32),
                "action_exec": np.asarray(tr_a_exec, dtype=np.float32),
                "best_macro_idx": np.asarray(tr_best_macro_idx, dtype=np.int32),
                "pred_macro_disp": np.asarray(tr_pred_macro_disp, dtype=np.float32),
                "pred_micro_disp": np.asarray(tr_pred_micro_disp, dtype=np.float32),
                "col_risk": np.asarray(tr_col_risk, dtype=np.float32),
                "dist_assigned_wp": np.asarray(tr_dist_assigned, dtype=np.float32),
                "dist_unassigned_wp": np.asarray(tr_dist_unassigned, dtype=np.float32),
                "assigned_vis_px": np.asarray(tr_assigned_vis_px, dtype=np.int32),
                "unassigned_vis_px": np.asarray(tr_unassigned_vis_px, dtype=np.int32),
                "masked_px": np.asarray(tr_masked_px, dtype=np.int32),
                "fallback_mask": np.asarray(tr_fallback_mask, dtype=bool),
                "action_assigned_cos": np.asarray(tr_act_assigned_cos, dtype=np.float32),
                "action_unassigned_cos": np.asarray(tr_act_unassigned_cos, dtype=np.float32),
                "goal_ang_err_rad": np.asarray(tr_goal_ang_err, dtype=np.float32),
                "vis_ang_err_rad": np.asarray(tr_vis_ang_err, dtype=np.float32),
                "planner_goal_ang_err_rad": np.asarray(tr_planner_ang_err, dtype=np.float32),
                "proj_signed_vis_m": np.asarray(tr_proj_vis_m, dtype=np.float32),
                "proj_signed_blended_m": np.asarray(tr_proj_blend_m, dtype=np.float32),
                "coord_vis_norm": np.asarray(tr_coord_vis, dtype=np.float32),
                "coord_blended_norm": np.asarray(tr_coord_blend, dtype=np.float32),
                "task_embedding_executed": np.asarray(tr_task_emb_exec, dtype=bool),
                "grounding_head_executed": np.asarray(tr_ground_exec, dtype=bool),
                "memory_branch_executed": np.asarray(tr_mem_exec, dtype=bool),
            },
        }
        if capture_rgb_keyframes:
            ep_rec["keyframes"] = captured_frames
        return ep_rec
    finally:
        agent.track = original_track
        agent.disable_memory = original_disable_memory
        for h in hooks:
            h.remove()


def compute_scene_pair_crossover_record(
    ep_a: Dict[str, Any],
    ep_b: Dict[str, Any],
    g0_vis_a: np.ndarray,
    g0_vis_b: np.ndarray,
    g0_bld_a: np.ndarray,
    g0_bld_b: np.ndarray,
    gt0_a: np.ndarray,
    gt0_b: np.ndarray,
) -> Dict[str, Any]:
    """Computes scene-pair crossover & task-switch metrics combining Assignment A and Assignment B."""
    assert ep_a["scene_id"] == ep_b["scene_id"] and ep_a["condition"] == ep_b["condition"]
    d_gt = np.asarray(gt0_a[:2], dtype=np.float64) - np.asarray(gt0_b[:2], dtype=np.float64)
    n_gt = float(np.linalg.norm(d_gt))
    if n_gt > 1e-8 and not (np.any(np.isnan(g0_vis_a[:2])) or np.any(np.isnan(g0_vis_b[:2]))):
        dv = np.asarray(g0_vis_a[:2], dtype=np.float64) - np.asarray(g0_vis_b[:2], dtype=np.float64)
        db = np.asarray(g0_bld_a[:2], dtype=np.float64) - np.asarray(g0_bld_b[:2], dtype=np.float64)
        eta_vis_0 = float(np.dot(dv, d_gt) / (n_gt * n_gt))
        eta_bld_0 = float(np.dot(db, d_gt) / (n_gt * n_gt))
        cos_vis_0 = float(np.dot(dv, d_gt) / (float(np.linalg.norm(dv)) * n_gt)) if float(np.linalg.norm(dv)) > 1e-8 else 0.0
    else:
        eta_vis_0, eta_bld_0, cos_vis_0 = float("nan"), float("nan"), float("nan")

    succ_a = int(bool(ep_a["assigned_success"]))
    succ_b = int(bool(ep_b["assigned_success"]))
    unass_a = int(bool(ep_a["unassigned_reached"]))
    unass_b = int(bool(ep_b["unassigned_reached"]))
    fa = ep_a["first_reached_object"]
    fb = ep_b["first_reached_object"]
    dual_first_correct = bool(fa == "assigned" and fb == "assigned")
    same_object_collapse = bool((fa == "assigned" and fb == "unassigned") or (fa == "unassigned" and fb == "assigned"))

    return {
        "condition": ep_a["condition"],
        "scene_id": ep_a["scene_id"],
        "scene_idx": int(ep_a["scene_idx"]),
        "context": ep_a["context"],
        "category_pair": ep_a["category_pair"],
        "obj_1_key": ep_a["target_key"],
        "obj_2_key": ep_b["target_key"],
        "assigned_success_A": bool(succ_a),
        "assigned_success_B": bool(succ_b),
        "pair_mean_assigned_success": 0.5 * (succ_a + succ_b),
        "dual_assigned_success": bool(succ_a and succ_b),
        "pair_mean_unassigned_reached": 0.5 * (unass_a + unass_b),
        "first_reached_A": fa,
        "first_reached_B": fb,
        "dual_first_correct_selection": dual_first_correct,
        "same_object_collapse": same_object_collapse,
        "pair_selectivity_margin": 0.5 * ((succ_a - unass_a) + (succ_b - unass_b)),
        "pair_mean_final_assigned_err_m": 0.5 * (float(ep_a["final_assigned_goal_error_m"]) + float(ep_b["final_assigned_goal_error_m"])),
        "pair_mean_final_unassigned_err_m": 0.5 * (float(ep_a["final_unassigned_goal_error_m"]) + float(ep_b["final_unassigned_goal_error_m"])),
        "pair_mean_margin_m": 0.5 * (
            float(ep_a["final_margin_unassigned_minus_assigned_m"]) + float(ep_b["final_margin_unassigned_minus_assigned_m"])
        ),
        "task_switch_eta_vis_step0": _opt_float_or_none(eta_vis_0),
        "task_switch_eta_blended_step0": _opt_float_or_none(eta_bld_0),
        "task_switch_cos_vis_step0": _opt_float_or_none(cos_vis_0),
        "pair_mean_coord_vis_norm": _safe_nanmean([ep_a["mean_step_coord_vis_norm"], ep_b["mean_step_coord_vis_norm"]]),
    }


def _summarize_episode_subset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n_episodes": 0}
    ks = sum(1 for r in records if r["assigned_success"])
    ku = sum(1 for r in records if r["unassigned_reached"])
    fa = sum(1 for r in records if r["first_reached_object"] == "assigned")
    fu = sum(1 for r in records if r["first_reached_object"] == "unassigned")
    fn = sum(1 for r in records if r["first_reached_object"] == "neither")
    kc = sum(1 for r in records if r["collision"])
    return {
        "n_episodes": n,
        "assigned_success_count": ks,
        "assigned_success_rate": float(ks / n),
        "assigned_success_wilson_95ci": wilson_ci(ks, n),
        "unassigned_reached_count": ku,
        "unassigned_reached_rate": float(ku / n),
        "first_reached_assigned_count": fa,
        "first_reached_unassigned_count": fu,
        "first_reached_neither_count": fn,
        "collision_count": kc,
        "collision_rate": float(kc / n),
        "closer_to_assigned_count": sum(1 for r in records if r["closer_to_assigned"]),
        "mean_final_assigned_error_m": float(np.mean([r["final_assigned_goal_error_m"] for r in records])),
        "median_final_assigned_error_m": float(np.median([r["final_assigned_goal_error_m"] for r in records])),
        "mean_final_unassigned_error_m": float(np.mean([r["final_unassigned_goal_error_m"] for r in records])),
        "mean_margin_unassigned_minus_assigned_m": float(np.mean([r["final_margin_unassigned_minus_assigned_m"] for r in records])),
        "mean_path_length_m": float(np.mean([r["path_length_m"] for r in records])),
        "mean_executed_steps": float(np.mean([r["executed_steps"] for r in records])),
        "mean_step_vis_goal_ang_err_deg": _safe_nanmean([r["mean_step_vis_goal_ang_err_deg"] for r in records]),
        "mean_step_blended_goal_ang_err_deg": _safe_nanmean([r["mean_step_blended_goal_ang_err_deg"] for r in records]),
        "mean_step_planner_goal_ang_err_deg": _safe_nanmean([r["mean_step_planner_goal_ang_err_deg"] for r in records]),
        "mean_step_coord_vis_norm": _safe_nanmean([r["mean_step_coord_vis_norm"] for r in records]),
        "mean_step_action_assigned_cos": float(np.mean([r["mean_step_action_assigned_cos"] for r in records])),
    }


def _summarize_condition_full(
    ep_records: List[Dict[str, Any]],
    pair_records: List[Dict[str, Any]],
    seed: int = 5510,
) -> Dict[str, Any]:
    out = _summarize_episode_subset(ep_records)
    npairs = len(pair_records)
    pm_mean, pm_lo, pm_hi = _mean_bootstrap_ci([p["pair_mean_assigned_success"] for p in pair_records], seed=seed)
    sm_mean, sm_lo, sm_hi = _mean_bootstrap_ci([p["pair_selectivity_margin"] for p in pair_records], seed=seed + 1)
    eta_mean, eta_lo, eta_hi = _mean_bootstrap_ci(
        [p["task_switch_eta_vis_step0"] for p in pair_records if p["task_switch_eta_vis_step0"] is not None], seed=seed + 2
    )
    out["scene_pair_summary"] = {
        "n_scenes": npairs,
        "pair_mean_assigned_success": pm_mean,
        "pair_mean_assigned_success_95ci": [pm_lo, pm_hi],
        "dual_assigned_success_count": sum(1 for p in pair_records if p["dual_assigned_success"]),
        "dual_assigned_success_rate": float(np.mean([p["dual_assigned_success"] for p in pair_records])),
        "dual_first_correct_count": sum(1 for p in pair_records if p["dual_first_correct_selection"]),
        "dual_first_correct_rate": float(np.mean([p["dual_first_correct_selection"] for p in pair_records])),
        "same_object_collapse_count": sum(1 for p in pair_records if p["same_object_collapse"]),
        "same_object_collapse_rate": float(np.mean([p["same_object_collapse"] for p in pair_records])),
        "pair_selectivity_margin": sm_mean,
        "pair_selectivity_margin_95ci": [sm_lo, sm_hi],
        "task_switch_eta_vis_step0_mean": eta_mean,
        "task_switch_eta_vis_step0_95ci": [eta_lo, eta_hi],
        "pair_mean_coord_vis_norm": _safe_nanmean([p["pair_mean_coord_vis_norm"] for p in pair_records]),
    }
    out["stratified"] = {
        "by_context": {
            ctx: _summarize_episode_subset([r for r in ep_records if r["context"] == ctx])
            for ctx in ("nominal", "illumination_shift")
        },
        "by_assignment": {
            asg: _summarize_episode_subset([r for r in ep_records if r["assignment"] == asg])
            for asg in ("A", "B")
        },
        "by_assigned_side": {
            sd: _summarize_episode_subset([r for r in ep_records if r["assigned_side"] == sd])
            for sd in ("left", "right")
        },
        "by_target_category": {
            tk: _summarize_episode_subset([r for r in ep_records if r["target_key"] == tk])
            for tk in TARGET_KEYS
        },
        "by_category_pair": {
            cp: _summarize_episode_subset([r for r in ep_records if r["category_pair"] == cp])
            for cp in sorted({r["category_pair"] for r in ep_records})
        },
        "by_outcome": {
            "assigned_success": _summarize_episode_subset([r for r in ep_records if r["assigned_success"]]),
            "failure": _summarize_episode_subset([r for r in ep_records if not r["assigned_success"]]),
        },
    }
    return out


def compute_v5_5_1_statistics(
    episodes_by_cond: Dict[str, List[Dict[str, Any]]],
    pairs_by_cond: Dict[str, List[Dict[str, Any]]],
    seed: int = 5510,
) -> Dict[str, Any]:
    """Computes the 5 pre-registered Holm–Bonferroni hypothesis families at N=60 scene-pair level."""
    cond_summaries = {
        c: _summarize_condition_full(episodes_by_cond[c], pairs_by_cond[c], seed=seed + idx * 10)
        for idx, c in enumerate(V5_5_1_CONDITIONS)
    }

    def _pair_vec(cname: str, field: str) -> List[float]:
        return [float(p[field]) for p in sorted(pairs_by_cond[cname], key=lambda x: x["scene_idx"])]

    def _pair_contrast(vals_a: Sequence[float], vals_b: Sequence[float], label: str, s_off: int) -> Dict[str, Any]:
        diff_m, ci_lo, ci_hi = paired_bootstrap_diff_ci(vals_a, vals_b, n_boot=4000, seed=seed + s_off)
        return {
            "contrast": label,
            "n_scene_pairs": len(vals_a),
            "mean_a": float(np.mean(vals_a)),
            "mean_b": float(np.mean(vals_b)),
            "diff_mean": diff_m,
            "diff_95ci_pair_bootstrap": [ci_lo, ci_hi],
            "wilcoxon_two_sided_p": _paired_wilcoxon_p(vals_a, vals_b),
        }

    # H1: Task selectivity under F1 (scene-pair N=60)
    f1_pairs = sorted(pairs_by_cond["F1_dynamic_visual_with_memory"], key=lambda x: x["scene_idx"])
    h1_sel = _pair_contrast(
        [float(p["pair_mean_assigned_success"]) for p in f1_pairs],
        [float(p["pair_mean_unassigned_reached"]) for p in f1_pairs],
        "F1: pair_mean_assigned_success - pair_mean_unassigned_reached",
        100,
    )
    n_dual = sum(1 for p in f1_pairs if p["dual_first_correct_selection"])
    n_coll = sum(1 for p in f1_pairs if p["same_object_collapse"])
    p_h1_binom = float(stats.binomtest(n_dual, n=max(1, n_dual + n_coll), p=0.5).pvalue) if (n_dual + n_coll) > 0 else 1.0
    etas_f1 = [float(p["task_switch_eta_vis_step0"]) for p in f1_pairs if p["task_switch_eta_vis_step0"] is not None]
    eta_m, eta_lo, eta_hi = _mean_bootstrap_ci(etas_f1, seed=seed + 105)
    h1 = {
        **h1_sel,
        "dual_first_correct_count": n_dual,
        "same_object_collapse_count": n_coll,
        "dual_vs_collapse_binom_two_sided_p": p_h1_binom,
        "task_switch_eta_vis_step0_mean": eta_m,
        "task_switch_eta_vis_step0_95ci": [eta_lo, eta_hi],
        "task_switch_eta_wilcoxon_zero_p": _one_sample_wilcoxon_zero_p(etas_f1),
        "family_primary_p": h1_sel["wilcoxon_two_sided_p"],
    }

    # H2: Visual feedback main effect: 0.5*(F1+F2) - 0.5*(F3+F4)
    f1 = _pair_vec("F1_dynamic_visual_with_memory", "pair_mean_assigned_success")
    f2 = _pair_vec("F2_dynamic_visual_no_memory", "pair_mean_assigned_success")
    f3 = _pair_vec("F3_frozen_visual_with_memory", "pair_mean_assigned_success")
    f4 = _pair_vec("F4_frozen_visual_no_memory", "pair_mean_assigned_success")
    dyn_avg = [0.5 * (a + b) for a, b in zip(f1, f2)]
    frz_avg = [0.5 * (a + b) for a, b in zip(f3, f4)]
    h2_main = _pair_contrast(dyn_avg, frz_avg, "Visual Feedback Main Effect: 0.5*(F1+F2) - 0.5*(F3+F4)", 200)
    h2 = {
        **h2_main,
        "simple_effect_F1_minus_F3": _pair_contrast(f1, f3, "F1 - F3 (with memory)", 205),
        "simple_effect_F2_minus_F4": _pair_contrast(f2, f4, "F2 - F4 (no memory)", 210),
        "family_primary_p": h2_main["wilcoxon_two_sided_p"],
    }

    # H3: Controller memory main effect & interaction: 0.5*(F1+F3) - 0.5*(F2+F4)
    mem_avg = [0.5 * (a + b) for a, b in zip(f1, f3)]
    nom_avg = [0.5 * (a + b) for a, b in zip(f2, f4)]
    h3_main = _pair_contrast(mem_avg, nom_avg, "Controller Memory Main Effect: 0.5*(F1+F3) - 0.5*(F2+F4)", 300)
    inter_dyn = [a - b for a, b in zip(f1, f2)]
    inter_frz = [a - b for a, b in zip(f3, f4)]
    h3 = {
        **h3_main,
        "interaction_F1_minus_F2_vs_F3_minus_F4": _pair_contrast(inter_dyn, inter_frz, "Interaction (F1-F2) - (F3-F4)", 305),
        "simple_effect_F1_minus_F2": _pair_contrast(f1, f2, "F1 - F2 (dynamic visual)", 310),
        "simple_effect_F3_minus_F4": _pair_contrast(f3, f4, "F3 - F4 (frozen visual)", 315),
        "family_primary_p": h3_main["wilcoxon_two_sided_p"],
    }

    # H4: Masking specificity: M1 vs M2 (+ M1 vs F1, M2 vs F1)
    m1 = _pair_vec("M1_target_masked_rgb", "pair_mean_assigned_success")
    m2 = _pair_vec("M2_distractor_masked_rgb", "pair_mean_assigned_success")
    h4_m1_m2 = _pair_contrast(m1, m2, "M1_target_masked - M2_distractor_masked", 400)
    h4_m1_f1 = _pair_contrast(m1, f1, "M1_target_masked - F1", 405)
    h4_m2_f1 = _pair_contrast(m2, f1, "M2_distractor_masked - F1", 410)
    h4_within_p = holm_bonferroni_adjust([h4_m1_m2["wilcoxon_two_sided_p"], h4_m1_f1["wilcoxon_two_sided_p"], h4_m2_f1["wilcoxon_two_sided_p"]])
    h4_m1_m2["within_family_holm_p"] = h4_within_p[0]
    h4_m1_f1["within_family_holm_p"] = h4_within_p[1]
    h4_m2_f1["within_family_holm_p"] = h4_within_p[2]
    h4 = {
        **h4_m1_m2,
        "M1_minus_F1": h4_m1_f1,
        "M2_minus_F1": h4_m2_f1,
        "family_primary_p": h4_m1_m2["wilcoxon_two_sided_p"],
    }

    # H5: Planner goal override O1 vs F1
    o1 = _pair_vec("O1_track_c_planner_goal_override", "pair_mean_assigned_success")
    h5_o1_f1 = _pair_contrast(o1, f1, "O1_track_c_planner_goal_override - F1", 500)
    h5 = {**h5_o1_f1, "family_primary_p": h5_o1_f1["wilcoxon_two_sided_p"]}

    families = [
        ("H1_task_selectivity_F1", h1),
        ("H2_visual_feedback_main_effect", h2),
        ("H3_controller_memory_main_effect", h3),
        ("H4_object_masking_specificity", h4),
        ("H5_planner_goal_override_O1", h5),
    ]
    global_holm = holm_bonferroni_adjust([f[1]["family_primary_p"] for f in families])
    confirmatory_holm5 = {}
    for (fname, fdict), hp in zip(families, global_holm):
        fdict["holm_adjusted_5family_p"] = float(hp)
        confirmatory_holm5[fname] = fdict

    return {
        "conditions": cond_summaries,
        "confirmatory_hypothesis_families_holm5": confirmatory_holm5,
    }


def export_trace_shards_jsonl(
    episode_records: List[Dict[str, Any]],
    traces_by_key: Dict[str, Dict[str, np.ndarray]],
    shard_dir: Path,
    max_shard_bytes: int = MAX_JSONL_SHARD_BYTES,
) -> Dict[str, Any]:
    """Exports complete step-level traces as text JSONL shards strictly < max_shard_bytes (< 5 MB)."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    for old_f in shard_dir.glob("part_*.jsonl"):
        old_f.unlink()

    shard_files: List[Dict[str, Any]] = []
    shard_idx = 0
    curr_lines: List[bytes] = []
    curr_bytes = 0
    curr_eps: Dict[str, int] = {}
    total_rows = 0

    def _flush_shard() -> None:
        nonlocal shard_idx, curr_lines, curr_bytes, curr_eps
        if not curr_lines:
            return
        fname = f"part_{shard_idx:03d}.jsonl"
        fpath = shard_dir / fname
        payload = b"".join(curr_lines)
        fpath.write_bytes(payload)
        assert len(payload) < 5 * 1024 * 1024, f"Shard {fname} exceeded 5 MB limit: {len(payload)}"
        shard_files.append(
            {
                "filename": fname,
                "bytes": len(payload),
                "sha256": compute_bytes_sha256(payload),
                "n_rows": len(curr_lines),
                "episodes": [{"episode_key": ek, "n_steps_in_shard": cnt} for ek, cnt in curr_eps.items()],
            }
        )
        shard_idx += 1
        curr_lines = []
        curr_bytes = 0
        curr_eps = {}

    for ep in episode_records:
        ekey = ep["episode_key"]
        tr = traces_by_key[ekey]
        n_steps = int(len(tr["action_exec"]))
        for t in range(n_steps):
            row = _to_json_clean(
                {
                    "episode_key": ekey,
                    "condition": ep["condition"],
                    "scene_id": ep["scene_id"],
                    "assignment": ep["assignment"],
                    "context": ep["context"],
                    "target_key": ep["target_key"],
                    "distractor_key": ep["distractor_key"],
                    "t": t,
                    "pose": tr["pose"][t],
                    "latent_s": tr["latent_s"][t],
                    "task_embedding": tr["task_embedding"][t],
                    "grounding_latent": tr["grounding_latent"][t],
                    "l1_act": tr["l1_act"][t],
                    "l2_act": tr["l2_act"][t],
                    "g_visual": tr["g_visual"][t],
                    "g_memory": tr["g_memory"][t],
                    "g_blended": tr["g_blended"][t],
                    "planner_goal": tr["planner_goal"][t],
                    "delta_ego": tr["delta_ego"][t],
                    "gt_assigned_body": tr["gt_assigned_body"][t],
                    "gt_unassigned_body": tr["gt_unassigned_body"][t],
                    "a_react": tr["a_react"][t],
                    "action_exec": tr["action_exec"][t],
                    "best_macro_idx": tr["best_macro_idx"][t],
                    "pred_macro_disp": tr["pred_macro_disp"][t],
                    "pred_micro_disp": tr["pred_micro_disp"][t],
                    "col_risk": tr["col_risk"][t],
                    "dist_assigned_wp": tr["dist_assigned_wp"][t],
                    "dist_unassigned_wp": tr["dist_unassigned_wp"][t],
                    "assigned_vis_px": tr["assigned_vis_px"][t],
                    "unassigned_vis_px": tr["unassigned_vis_px"][t],
                    "masked_px": tr["masked_px"][t],
                    "fallback_mask": tr["fallback_mask"][t],
                    "action_assigned_cos": tr["action_assigned_cos"][t],
                    "action_unassigned_cos": tr["action_unassigned_cos"][t],
                    "goal_ang_err_rad": tr["goal_ang_err_rad"][t],
                    "vis_ang_err_rad": tr["vis_ang_err_rad"][t],
                    "planner_goal_ang_err_rad": tr["planner_goal_ang_err_rad"][t],
                    "proj_signed_vis_m": tr["proj_signed_vis_m"][t],
                    "proj_signed_blended_m": tr["proj_signed_blended_m"][t],
                    "coord_vis_norm": tr["coord_vis_norm"][t],
                    "coord_blended_norm": tr["coord_blended_norm"][t],
                    "task_embedding_executed": tr["task_embedding_executed"][t],
                    "grounding_head_executed": tr["grounding_head_executed"][t],
                    "memory_branch_executed": tr["memory_branch_executed"][t],
                }
            )
            line_bytes = (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
            if curr_lines and (curr_bytes + len(line_bytes) > max_shard_bytes):
                _flush_shard()
            curr_lines.append(line_bytes)
            curr_bytes += len(line_bytes)
            curr_eps[ekey] = curr_eps.get(ekey, 0) + 1
            total_rows += 1

    _flush_shard()
    manifest = {
        "n_shards": len(shard_files),
        "total_rows": total_rows,
        "total_episodes": len(episode_records),
        "max_shard_bytes_limit": max_shard_bytes,
        "max_observed_shard_bytes": max((s["bytes"] for s in shard_files), default=0),
        "shards": shard_files,
    }
    manifest_path = shard_dir / "shard_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _save_labeled_rgb(rgb: np.ndarray, title: str, out_path: Path) -> None:
    img = Image.fromarray(rgb.astype(np.uint8))
    scale = 4
    img_big = img.resize((img.width * scale, img.height * scale), resample=Image.NEAREST)
    canvas = Image.new("RGB", (img_big.width, img_big.height + 34), (24, 24, 28))
    canvas.paste(img_big, (0, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), title[:64], fill=(245, 245, 245))
    canvas.save(out_path)


def extract_representative_keyframes_payload(
    episodes: List[Dict[str, Any]],
    scene_pairs: List[Dict[str, Any]],
    scenes_by_id: Dict[str, Dict[str, Any]],
    agent: HJEPAAgent,
    env: JetAutoEnv,
    max_steps: int,
    goal_reach_threshold: float,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Captures raw RGB keyframes cleanly as a small raw-data artifact for downstream visualization."""
    f1_eps = [e for e in episodes if e["condition"] == "F1_dynamic_visual_with_memory"]
    f1_pairs = [p for p in scene_pairs if p["condition"] == "F1_dynamic_visual_with_memory"]

    selected: Dict[str, Optional[Dict[str, Any]]] = {
        "dual_success_scene": next((p for p in f1_pairs if p["dual_assigned_success"]), f1_pairs[0] if f1_pairs else None),
        "correct_selection_ep": next((e for e in f1_eps if e["first_reached_object"] == "assigned"), f1_eps[0] if f1_eps else None),
        "incorrect_selection_ep": next((e for e in f1_eps if e["first_reached_object"] == "unassigned"), None),
        "failure_ep": next((e for e in f1_eps if not e["assigned_success"]), None),
    }

    meta: Dict[str, Any] = {}
    raw_npz: Dict[str, np.ndarray] = {}
    for label in ("correct_selection_ep", "incorrect_selection_ep", "failure_ep"):
        ep = selected.get(label)
        if ep is None:
            continue
        spec = make_assignment_spec(scenes_by_id[ep["scene_id"]], ep["assignment"])
        rec = evaluate_v5_5_1_episode(
            agent, env, spec, ep["condition"], max_steps=max_steps, goal_reach_threshold=goal_reach_threshold, capture_rgb_keyframes=True
        )
        kf = rec.get("keyframes", {})
        keys_saved = {}
        for kname in ("step_0_unmasked", "last_obs"):
            if kname in kf:
                arr_key = f"{label}__{kname}"
                raw_npz[arr_key] = kf[kname].astype(np.uint8)
                keys_saved[kname] = arr_key
        meta[label] = {
            "scene_id": ep["scene_id"],
            "assignment": ep["assignment"],
            "condition": ep["condition"],
            "target_key": ep["target_key"],
            "first_reached_object": ep["first_reached_object"],
            "assigned_success": ep["assigned_success"],
            "npz_keys": keys_saved,
        }

    ts_pair = selected["dual_success_scene"] or (f1_pairs[0] if f1_pairs else None)
    if ts_pair is not None:
        sid = ts_pair["scene_id"]
        sc = scenes_by_id[sid]
        meta["task_switch_comparison"] = {
            "scene_id": sid,
            "waypoint_separation_m": float(sc["waypoint_separation_m"]),
            "wp_1": list(sc["wp_1"]),
            "wp_2": list(sc["wp_2"]),
            "obj_1_key": sc["obj_1_key"],
            "obj_2_key": sc["obj_2_key"],
        }
    return meta, raw_npz


def render_representative_figures_from_raw(
    rep_meta: Dict[str, Any],
    episodes: List[Dict[str, Any]],
    traces_npz: Any,
    keyframes_npz: Optional[Any],
    frames_dir: Path,
    goal_reach_threshold: float = DEFAULT_REACH_RADIUS_M,
) -> List[str]:
    """Renders representative RGB frames and task-switch trajectory panels strictly from saved raw artifacts."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    if keyframes_npz is not None:
        for label in ("correct_selection_ep", "incorrect_selection_ep", "failure_ep"):
            info = rep_meta.get(label)
            if not info:
                continue
            for kname, arr_key in info.get("npz_keys", {}).items():
                if arr_key in keyframes_npz:
                    fn = f"{label}__{info['scene_id']}_{info['assignment']}__{kname}.png"
                    outp = frames_dir / fn
                    _save_labeled_rgb(
                        keyframes_npz[arr_key],
                        f"{label}: {info['scene_id']} {info['assignment']} ({info['target_key']}) {kname}",
                        outp,
                    )
                    written.append(f"frames/{fn}")

    ts_info = rep_meta.get("task_switch_comparison")
    if ts_info is not None:
        sid = ts_info["scene_id"]
        ep_lookup = {(e["condition"], e["scene_id"], e["assignment"]): e for e in episodes}
        fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.6), dpi=160)
        plot_conds = [
            "F1_dynamic_visual_with_memory",
            "M1_target_masked_rgb",
            "M2_distractor_masked_rgb",
            "O1_track_c_planner_goal_override",
        ]
        for col_i, cname in enumerate(plot_conds):
            for row_i, asg in enumerate(("A", "B")):
                ax = axes[row_i, col_i]
                ep = ep_lookup[(cname, sid, asg)]
                ek = ep["episode_key"]
                pose = traces_npz[f"{ek}__pose"]
                eval_wp = ts_info["wp_1"] if asg == "A" else ts_info["wp_2"]
                dist_wp = ts_info["wp_2"] if asg == "A" else ts_info["wp_1"]
                ax.plot(pose[:, 0], pose[:, 1], "-o", markersize=2.5, linewidth=1.8, color="#2563eb", label="Robot")
                ax.scatter([pose[0, 0]], [pose[0, 1]], s=55, color="#111827", zorder=5, label="Start")
                c1 = plt.Circle((eval_wp[0], eval_wp[1]), goal_reach_threshold, color="#16a34a", fill=False, linestyle="--", linewidth=1.5)
                c2 = plt.Circle((dist_wp[0], dist_wp[1]), goal_reach_threshold, color="#dc2626", fill=False, linestyle=":", linewidth=1.5)
                ax.add_patch(c1)
                ax.add_patch(c2)
                ax.scatter([eval_wp[0]], [eval_wp[1]], s=65, color="#16a34a", marker="*", label=f"Tgt ({ep['target_key']})")
                ax.scatter([dist_wp[0]], [dist_wp[1]], s=50, color="#dc2626", marker="x", label=f"Dist ({ep['distractor_key']})")
                ax.set_aspect("equal", adjustable="datalim")
                ax.grid(alpha=0.3)
                ax.set_title(
                    f"{cname.split('_')[0]} | Task {asg}\n1st={ep['first_reached_object']} succ={ep['assigned_success']}",
                    fontsize=8.5,
                )
                if row_i == 0 and col_i == 0:
                    ax.legend(fontsize=6.5, loc="best")
        fig.suptitle(f"Task-Switch Geometry & Trajectories ({sid}, sep={ts_info['waypoint_separation_m']:.2f}m)", fontsize=11)
        fig.tight_layout()
        ts_name = f"task_switch_comparison_{sid}.png"
        fig.savefig(frames_dir / ts_name)
        plt.close(fig)
        written.append(f"frames/{ts_name}")
    return written


def generate_v5_5_1_figures(
    summary_path: Path,
    episode_jsonl_path: Path,
    scene_pairs_jsonl_path: Path,
    traces_npz_path: Path,
    figures_dir: Path,
    keyframes_npz_path: Optional[Path] = None,
) -> List[str]:
    """Regenerates Phase 5.5.1 summary & dynamical figures strictly from saved artifacts."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    episodes = [json.loads(line) for line in episode_jsonl_path.open("r", encoding="utf-8") if line.strip()]
    scene_pairs = [json.loads(line) for line in scene_pairs_jsonl_path.open("r", encoding="utf-8") if line.strip()]
    npz = np.load(traces_npz_path)

    written: List[str] = []
    short_labels = {
        "F1_dynamic_visual_with_memory": "F1\nDyn+Mem",
        "F2_dynamic_visual_no_memory": "F2\nDyn NoMem",
        "F3_frozen_visual_with_memory": "F3\nFrz+Mem",
        "F4_frozen_visual_no_memory": "F4\nFrz NoMem",
        "M1_target_masked_rgb": "M1\nTgtMask",
        "M2_distractor_masked_rgb": "M2\nDistMask",
        "O1_track_c_planner_goal_override": "O1\nPlnOvrd",
    }
    conds = [c for c in V5_5_1_CONDITIONS if c in summary["conditions"]]

    # Figure 1: Assigned vs Unassigned reach & Context stratification
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), dpi=180)
    x = np.arange(len(conds))
    w = 0.27
    ass_r = [summary["conditions"][c]["assigned_success_rate"] * 100.0 for c in conds]
    una_r = [summary["conditions"][c]["unassigned_reached_rate"] * 100.0 for c in conds]
    dual_r = [summary["conditions"][c]["scene_pair_summary"]["dual_first_correct_rate"] * 100.0 for c in conds]

    axes[0].bar(x - w, ass_r, width=w, color="#2563eb", label="Assigned Success (%)")
    axes[0].bar(x, una_r, width=w, color="#dc2626", label="Unassigned Reached (%)")
    axes[0].bar(x + w, dual_r, width=w, color="#0d9488", label="Dual-Task First-Correct (% pairs)")
    for i, val in enumerate(ass_r):
        axes[0].text(i - w, val + 1.5, f"{val:.0f}%", ha="center", va="bottom", fontsize=7.5)
    axes[0].set_ylim(0, 112)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([short_labels[c] for c in conds], fontsize=8)
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_title("Phase 5.5.1 Geometry-Separated Task Selection (N=120 eps / 60 scenes)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    nom_r = [summary["conditions"][c]["stratified"]["by_context"]["nominal"]["assigned_success_rate"] * 100.0 for c in conds]
    ill_r = [summary["conditions"][c]["stratified"]["by_context"]["illumination_shift"]["assigned_success_rate"] * 100.0 for c in conds]
    axes[1].bar(x - 0.18, nom_r, width=0.36, color="#3b82f6", label="Nominal (30 scenes / 60 eps)")
    axes[1].bar(x + 0.18, ill_r, width=0.36, color="#f97316", label="Illumination Shift (30 scenes / 60 eps)")
    axes[1].set_ylim(0, 112)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([short_labels[c] for c in conds], fontsize=8)
    axes[1].set_ylabel("Assigned Target Success (%)")
    axes[1].set_title("Assigned Success Stratified by Illumination Context")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = figures_dir / "fig1_v5_5_1_selection_and_context_rates.png"
    fig.savefig(p1)
    plt.close(fig)
    written.append(p1.name)

    # Figure 2: 2x2 Visual Feedback x Controller Memory Factorial
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=180)
    f_map = {c: summary["conditions"][c] for c in conds}
    if all(k in f_map for k in V5_5_1_CONDITIONS[:4]):
        xs_fb = [0, 1]
        mem_on_succ = [
            f_map["F1_dynamic_visual_with_memory"]["scene_pair_summary"]["pair_mean_assigned_success"] * 100.0,
            f_map["F3_frozen_visual_with_memory"]["scene_pair_summary"]["pair_mean_assigned_success"] * 100.0,
        ]
        mem_off_succ = [
            f_map["F2_dynamic_visual_no_memory"]["scene_pair_summary"]["pair_mean_assigned_success"] * 100.0,
            f_map["F4_frozen_visual_no_memory"]["scene_pair_summary"]["pair_mean_assigned_success"] * 100.0,
        ]
        axes[0].plot(xs_fb, mem_on_succ, "-o", linewidth=2.2, color="#2563eb", label="Memory Enabled (F1, F3)")
        axes[0].plot(xs_fb, mem_off_succ, "--s", linewidth=2.2, color="#f59e0b", label="Memory Disabled (F2, F4)")
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(["Dynamic Visual Feedback", "Frozen t=0 World Visual Goal"])
        axes[0].set_ylim(0, 105)
        axes[0].set_ylabel("Scene-Pair Mean Assigned Success (%)")
        axes[0].set_title("2x2 Factorial: Assigned Target Success")
        axes[0].legend(fontsize=8.5)
        axes[0].grid(alpha=0.3)

        mem_on_margin = [
            f_map["F1_dynamic_visual_with_memory"]["scene_pair_summary"]["pair_selectivity_margin"] * 100.0,
            f_map["F3_frozen_visual_with_memory"]["scene_pair_summary"]["pair_selectivity_margin"] * 100.0,
        ]
        mem_off_margin = [
            f_map["F2_dynamic_visual_no_memory"]["scene_pair_summary"]["pair_selectivity_margin"] * 100.0,
            f_map["F4_frozen_visual_no_memory"]["scene_pair_summary"]["pair_selectivity_margin"] * 100.0,
        ]
        axes[1].plot(xs_fb, mem_on_margin, "-o", linewidth=2.2, color="#2563eb", label="Memory Enabled (F1, F3)")
        axes[1].plot(xs_fb, mem_off_margin, "--s", linewidth=2.2, color="#f59e0b", label="Memory Disabled (F2, F4)")
        axes[1].axhline(0.0, color="#6b7280", linestyle=":", linewidth=1.0)
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(["Dynamic Visual Feedback", "Frozen t=0 World Visual Goal"])
        axes[1].set_ylabel("Selectivity Margin (Assigned - Unassigned) %")
        axes[1].set_title("2x2 Factorial: Net Selectivity Margin")
        axes[1].legend(fontsize=8.5)
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p2 = figures_dir / "fig2_v5_5_1_2x2_feedback_memory_factorial.png"
    fig.savefig(p2)
    plt.close(fig)
    written.append(p2.name)

    # Figure 3: Temporal evolution across steps
    max_t = 30
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=180)
    styles = {
        "F1_dynamic_visual_with_memory": ("F1 Dyn+Mem", "#2563eb", "-"),
        "F2_dynamic_visual_no_memory": ("F2 Dyn NoMem", "#0284c7", "--"),
        "F3_frozen_visual_with_memory": ("F3 Frz+Mem", "#f59e0b", "-"),
        "F4_frozen_visual_no_memory": ("F4 Frz NoMem", "#d97706", "--"),
        "M1_target_masked_rgb": ("M1 TgtMask", "#dc2626", "-"),
        "M2_distractor_masked_rgb": ("M2 DistMask", "#16a34a", "-."),
        "O1_track_c_planner_goal_override": ("O1 PlnOvrd", "#0d9488", ":"),
    }
    for cname in conds:
        eps_c = [e for e in episodes if e["condition"] == cname]
        if not eps_c:
            continue
        label, col, ls = styles[cname]
        d_mat = np.full((len(eps_c), max_t), np.nan)
        c_mat = np.full((len(eps_c), max_t), np.nan)
        a_mat = np.full((len(eps_c), max_t), np.nan)
        for i, ep in enumerate(eps_c):
            ek = ep["episode_key"]
            d_arr = npz[f"{ek}__dist_assigned_wp"]
            cv_arr = npz[f"{ek}__coord_vis_norm"]
            ang_arr = np.degrees(npz[f"{ek}__vis_ang_err_rad"])
            ln = min(max_t, len(d_arr))
            d_mat[i, :ln] = d_arr[:ln]
            c_mat[i, :ln] = cv_arr[:ln]
            a_mat[i, :ln] = ang_arr[:ln]
            if ln < max_t and ln > 0:
                d_mat[i, ln:] = d_arr[ln - 1]
        ts = np.arange(max_t)
        axes[0].plot(ts, np.nanmean(d_mat, axis=0), label=label, color=col, linestyle=ls, linewidth=1.8)
        min_act = min(10, len(eps_c))
        keep = np.sum(~np.isnan(c_mat), axis=0) >= min_act
        if np.any(keep):
            axes[1].plot(ts[keep], np.nanmean(c_mat[:, keep], axis=0), label=label, color=col, linestyle=ls, linewidth=1.8)
            axes[2].plot(ts[keep], np.nanmean(a_mat[:, keep], axis=0), label=label, color=col, linestyle=ls, linewidth=1.8)

    axes[0].axhline(0.35, color="#111827", linestyle=":", linewidth=1.0)
    axes[0].set_title("Distance to Assigned Waypoint")
    axes[0].set_xlabel("Control Step t")
    axes[0].set_ylabel("Distance (m)")
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0.5, color="#16a34a", linestyle=":", linewidth=1.0, label="+0.5 Assigned")
    axes[1].axhline(-0.5, color="#dc2626", linestyle=":", linewidth=1.0, label="-0.5 Unassigned")
    axes[1].axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.9)
    axes[1].set_title("Visual Goal Signed Projection on Axis")
    axes[1].set_xlabel("Control Step t")
    axes[1].set_ylabel("Normalized Coord (+0.5=Assigned)")
    axes[1].grid(alpha=0.3)

    axes[2].set_title("Visual Goal Angular Error")
    axes[2].set_xlabel("Control Step t")
    axes[2].set_ylabel("Angular Error (deg)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=7, loc="best")
    fig.tight_layout()
    p3 = figures_dir / "fig3_v5_5_1_temporal_dynamics.png"
    fig.savefig(p3)
    plt.close(fig)
    written.append(p3.name)

    kf_npz = np.load(keyframes_npz_path) if (keyframes_npz_path is not None and keyframes_npz_path.exists()) else None
    rep_meta = summary.get("representative_cases", {})
    if rep_meta:
        written.extend(
            render_representative_figures_from_raw(
                rep_meta,
                episodes,
                npz,
                kf_npz,
                figures_dir / "frames",
                goal_reach_threshold=float(summary.get("provenance", {}).get("goal_reach_threshold", DEFAULT_REACH_RADIUS_M)),
            )
        )

    return written


def run_phase5_5_1_experiment(
    device: str = "cuda",
    max_steps: int = 45,
    goal_reach_threshold: float = DEFAULT_REACH_RADIUS_M,
    n_nominal: int = 30,
    n_illum: int = 30,
    seed: int = 5510,
) -> Dict[str, Any]:
    """Executes the complete Phase 5.5.1 geometry-separated experiment and saves all outputs."""
    t0 = time.time()
    ckpt_path = CKPT_DIR / "hjepa_best.pt"
    env_xml_path = REPO_ROOT / "world_models_fpga/env/amr_arena.xml"

    env = JetAutoEnv()
    oracle = OracleAgent()
    agent = HJEPAAgent(str(ckpt_path), device=device, track="observation")

    print(f"Generating Phase 5.5.1 geometry-separated cohort ({n_nominal} nominal + {n_illum} illum_shift)...")
    scenes, cohort_report = build_geometry_separated_cohort(
        env=env,
        oracle=oracle,
        n_nominal=n_nominal,
        n_illum=n_illum,
        max_steps=max_steps,
        goal_reach_threshold=goal_reach_threshold,
        seed=seed,
    )
    print(
        f"Accepted {cohort_report['n_accepted_scenes']} scenes "
        f"(rejected={cohort_report['n_rejected_candidates']}, min_wp_sep={cohort_report['min_waypoint_separation_m']:.3f}m)"
    )

    out_dir = REPO_ROOT / "world_models_fpga/research/benchmarks/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes_path = out_dir / "task_conditioned_closed_loop_grounding_v5_5_1_scenes.jsonl"
    cohort_path = out_dir / "task_conditioned_closed_loop_grounding_v5_5_1_cohort_report.json"
    with scenes_path.open("w", encoding="utf-8") as f:
        for sc in scenes:
            f.write(json.dumps(_to_json_clean(sc)) + "\n")
    with cohort_path.open("w", encoding="utf-8") as f:
        json.dump(_to_json_clean(cohort_report), f, indent=2)

    episodes_by_cond: Dict[str, List[Dict[str, Any]]] = {c: [] for c in V5_5_1_CONDITIONS}
    pairs_by_cond: Dict[str, List[Dict[str, Any]]] = {c: [] for c in V5_5_1_CONDITIONS}
    all_episodes_flat: List[Dict[str, Any]] = []
    all_pairs_flat: List[Dict[str, Any]] = []
    traces_by_key: Dict[str, Dict[str, np.ndarray]] = {}
    npz_payload: Dict[str, np.ndarray] = {}

    for cname in V5_5_1_CONDITIONS:
        print(f"Running {cname} ({len(scenes)} scenes x 2 assignments = {2 * len(scenes)} episodes)...")
        for sc in scenes:
            ep_a = evaluate_v5_5_1_episode(
                agent, env, make_assignment_spec(sc, "A"), cname, max_steps=max_steps, goal_reach_threshold=goal_reach_threshold
            )
            tr_a = ep_a.pop("traces")
            ep_b = evaluate_v5_5_1_episode(
                agent, env, make_assignment_spec(sc, "B"), cname, max_steps=max_steps, goal_reach_threshold=goal_reach_threshold
            )
            tr_b = ep_b.pop("traces")

            for ep, tr in ((ep_a, tr_a), (ep_b, tr_b)):
                ek = ep["episode_key"]
                traces_by_key[ek] = tr
                for tk, tval in tr.items():
                    npz_payload[f"{ek}__{tk}"] = tval
                episodes_by_cond[cname].append(ep)
                all_episodes_flat.append(ep)

            pair_rec = compute_scene_pair_crossover_record(
                ep_a,
                ep_b,
                tr_a["g_visual"][0],
                tr_b["g_visual"][0],
                tr_a["g_blended"][0],
                tr_b["g_blended"][0],
                tr_a["gt_assigned_body"][0],
                tr_b["gt_assigned_body"][0],
            )
            pairs_by_cond[cname].append(pair_rec)
            all_pairs_flat.append(pair_rec)

        sr = float(np.mean([r["assigned_success"] for r in episodes_by_cond[cname]]))
        ur = float(np.mean([r["unassigned_reached"] for r in episodes_by_cond[cname]]))
        dc = float(np.mean([p["dual_first_correct_selection"] for p in pairs_by_cond[cname]]))
        print(f"  -> {cname}: assigned_success={sr*100:.1f}%, unassigned_reached={ur*100:.1f}%, dual_1st_correct={dc*100:.1f}%")

    stats_out = compute_v5_5_1_statistics(episodes_by_cond, pairs_by_cond, seed=seed)

    ep_jsonl_path = out_dir / "task_conditioned_closed_loop_grounding_v5_5_1_episodes.jsonl"
    pairs_jsonl_path = out_dir / "task_conditioned_closed_loop_grounding_v5_5_1_scene_pairs.jsonl"
    cond_csv_path = out_dir / "task_conditioned_closed_loop_grounding_v5_5_1_condition_table.csv"
    summary_path = out_dir / "task_conditioned_closed_loop_grounding_v5_5_1_summary.json"
    npz_path = out_dir / "task_conditioned_closed_loop_v5_5_1_traces.npz"
    shard_dir = out_dir / "task_conditioned_closed_loop_v5_5_1_trace_export"

    with ep_jsonl_path.open("w", encoding="utf-8") as f:
        for row in all_episodes_flat:
            f.write(json.dumps(_to_json_clean(row)) + "\n")
    with pairs_jsonl_path.open("w", encoding="utf-8") as f:
        for row in all_pairs_flat:
            f.write(json.dumps(_to_json_clean(row)) + "\n")

    np.savez_compressed(npz_path, **npz_payload)
    print(f"Exporting <5MB JSONL trace shards to {shard_dir}...")
    shard_manifest = export_trace_shards_jsonl(all_episodes_flat, traces_by_key, shard_dir)
    print(
        f"  -> Exported {shard_manifest['n_shards']} shards "
        f"(total_rows={shard_manifest['total_rows']}, max_shard_bytes={shard_manifest['max_observed_shard_bytes']})"
    )

    with cond_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "condition",
            "n_episodes",
            "n_scenes",
            "assigned_success_count",
            "assigned_success_rate",
            "assigned_success_ci_low",
            "assigned_success_ci_high",
            "unassigned_reached_rate",
            "first_reached_assigned_count",
            "first_reached_unassigned_count",
            "dual_first_correct_rate",
            "same_object_collapse_rate",
            "pair_selectivity_margin",
            "task_switch_eta_vis_step0_mean",
            "mean_final_assigned_error_m",
            "mean_final_unassigned_error_m",
            "nominal_assigned_success_rate",
            "illumination_shift_assigned_success_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cname in V5_5_1_CONDITIONS:
            cs = stats_out["conditions"][cname]
            sps = cs["scene_pair_summary"]
            writer.writerow(
                {
                    "condition": cname,
                    "n_episodes": cs["n_episodes"],
                    "n_scenes": sps["n_scenes"],
                    "assigned_success_count": cs["assigned_success_count"],
                    "assigned_success_rate": cs["assigned_success_rate"],
                    "assigned_success_ci_low": cs["assigned_success_wilson_95ci"][0],
                    "assigned_success_ci_high": cs["assigned_success_wilson_95ci"][1],
                    "unassigned_reached_rate": cs["unassigned_reached_rate"],
                    "first_reached_assigned_count": cs["first_reached_assigned_count"],
                    "first_reached_unassigned_count": cs["first_reached_unassigned_count"],
                    "dual_first_correct_rate": sps["dual_first_correct_rate"],
                    "same_object_collapse_rate": sps["same_object_collapse_rate"],
                    "pair_selectivity_margin": sps["pair_selectivity_margin"],
                    "task_switch_eta_vis_step0_mean": sps["task_switch_eta_vis_step0_mean"],
                    "mean_final_assigned_error_m": cs["mean_final_assigned_error_m"],
                    "mean_final_unassigned_error_m": cs["mean_final_unassigned_error_m"],
                    "nominal_assigned_success_rate": cs["stratified"]["by_context"]["nominal"]["assigned_success_rate"],
                    "illumination_shift_assigned_success_rate": cs["stratified"]["by_context"]["illumination_shift"]["assigned_success_rate"],
                }
            )

    # Extract lightweight raw RGB keyframes for offline Commit-3 reproducible visualization
    kf_path = out_dir / "task_conditioned_closed_loop_v5_5_1_representative_keyframes.npz"
    rep_meta, kf_payload = extract_representative_keyframes_payload(
        all_episodes_flat,
        all_pairs_flat,
        {s["scene_id"]: s for s in scenes},
        agent,
        env,
        max_steps=max_steps,
        goal_reach_threshold=goal_reach_threshold,
    )
    np.savez_compressed(kf_path, **kf_payload)
    env.close()

    git_commit, git_clean = get_git_info(REPO_ROOT)
    elapsed = time.time() - t0
    summary_out = {
        "experiment_name": "Phase 5.5.1 Geometry-Separated Task Selectivity and Feedback-Memory Decomposition",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_clean": git_clean,
        "elapsed_seconds": float(elapsed),
        "provenance": {
            "checkpoint_path": str(ckpt_path.relative_to(REPO_ROOT)),
            "checkpoint_sha256": compute_file_sha256(ckpt_path),
            "env_xml_sha256": compute_file_sha256(env_xml_path),
            "scenes_path": str(scenes_path.relative_to(REPO_ROOT)),
            "scenes_sha256": compute_file_sha256(scenes_path),
            "keyframes_npz_path": str(kf_path.relative_to(REPO_ROOT)),
            "keyframes_npz_sha256": compute_file_sha256(kf_path),
            "device": device,
            "max_steps": max_steps,
            "goal_reach_threshold": goal_reach_threshold,
            "min_waypoint_separation_m": MIN_WAYPOINT_SEPARATION_M,
            "seed": seed,
        },
        "cohort_report": cohort_report,
        **stats_out,
        "trace_export": {
            "npz_path": str(npz_path.relative_to(REPO_ROOT)),
            "npz_bytes": npz_path.stat().st_size,
            "jsonl_shard_dir": str(shard_dir.relative_to(REPO_ROOT)),
            "n_shards": shard_manifest["n_shards"],
            "max_observed_shard_bytes": shard_manifest["max_observed_shard_bytes"],
            "total_step_rows": shard_manifest["total_rows"],
        },
        "representative_cases": rep_meta,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_to_json_clean(summary_out), f, indent=2)

    return summary_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5.5.1 Geometry-Separated Mechanistic Experiment")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--goal-threshold", type=float, default=DEFAULT_REACH_RADIUS_M)
    parser.add_argument("--n-nominal", type=int, default=30)
    parser.add_argument("--n-illum", type=int, default=30)
    parser.add_argument("--seed", type=int, default=5510)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate Phase 5.5.1 summary figures strictly from saved summary/episodes/pairs/traces.",
    )
    args = parser.parse_args()

    results_dir = REPO_ROOT / "world_models_fpga/research/benchmarks/results"
    summary_path = results_dir / "task_conditioned_closed_loop_grounding_v5_5_1_summary.json"
    ep_jsonl_path = results_dir / "task_conditioned_closed_loop_grounding_v5_5_1_episodes.jsonl"
    pairs_jsonl_path = results_dir / "task_conditioned_closed_loop_grounding_v5_5_1_scene_pairs.jsonl"
    npz_path = results_dir / "task_conditioned_closed_loop_v5_5_1_traces.npz"
    kf_path = results_dir / "task_conditioned_closed_loop_v5_5_1_representative_keyframes.npz"
    figures_dir = REPO_ROOT / "research/review_package/phase5_5_1/figures"

    if args.plot_only:
        written = generate_v5_5_1_figures(
            summary_path, ep_jsonl_path, pairs_jsonl_path, npz_path, figures_dir, keyframes_npz_path=kf_path
        )
        print("Regenerated Phase 5.5.1 figures from saved artifacts:", written)
        return

    out = run_phase5_5_1_experiment(
        device=args.device,
        max_steps=args.max_steps,
        goal_reach_threshold=args.goal_threshold,
        n_nominal=args.n_nominal,
        n_illum=args.n_illum,
        seed=args.seed,
    )
    print(f"\nPhase 5.5.1 execution completed in {out['elapsed_seconds']:.1f}s.")
    for fname, fval in out["confirmatory_hypothesis_families_holm5"].items():
        print(
            f"  {fname}: diff={fval['diff_mean']:+.3f} "
            f"[{fval['diff_95ci_pair_bootstrap'][0]:+.3f}, {fval['diff_95ci_pair_bootstrap'][1]:+.3f}] "
            f"(p={fval['family_primary_p']:.4e}, Holm5_p={fval['holm_adjusted_5family_p']:.4e})"
        )


if __name__ == "__main__":
    main()
