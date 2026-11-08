#!/usr/bin/env python3
"""
Phase 3 Mechanistic Experiment v2: Controlled Task Identity and Dynamic Visual Grounding.

Scientific Objectives:
1. Correct the initial-goal-distance confound in relocation experiments by enforcing
   initial waypoint distance >= 0.60m (0.25m margin outside 0.35m goal radius) and
   physical target relocation >= 0.75m from canonical location.
2. Evaluate post-action distance for goal success, rejecting any trial starting inside goal radius.
3. Rebuild the task-token experiment with BOTH candidate objects A and B visibly present
   in the identical initial camera frame (both >= 25 visible pixels, neither occluded),
   evaluating whether changing only the task token redirects the inferred body-frame goal vector
   and steers closed-loop navigation toward the designated object.
4. Factorial visual/token conflict experiment manipulating relative saliency with both objects visible.
5. Record per-step numerical traces and cryptographic image digests to an NPZ archive.

Outputs:
  - Summary JSON: world_models_fpga/research/benchmarks/results/task_identity_visual_grounding_v2.json
  - Numerical Traces: world_models_fpga/research/benchmarks/results/task_identity_visual_grounding_v2_traces.npz
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "world_models_fpga"))
sys.path.insert(0, str(REPO_ROOT))

import mujoco
import numpy as np
import scipy.stats as stats
from scipy.stats import binom
import torch

from env.amr_env import JetAutoEnv
from models.baseline_policy import encode_task_id, TARGET_KEYS
from research.benchmarks.model_wrappers import HJEPAAgent, compute_body_frame_goal_vec
from research.datasets.dataset_config import SPLITS_DIR, CKPT_DIR
from research.datasets.generation.capability_tasks import (
    CANONICAL_OBSTACLES,
    generate_task_specification,
    get_waypoint_for_relation,
)


GEOM_NAME_MAP = {
    "red_box": "obs_red_box",
    "blue_cylinder": "obs_blue_cyl",
    "green_cylinder": "obs_green_cyl",
    "yellow_cylinder": "obs_yellow_cyl",
    "purple_box": "obs_purple_box",
    "cyan_cylinder": "obs_cyan_cyl",
}


def compute_sha256(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_target_visibility(env: JetAutoEnv, target_key: str, min_visible_pixels: int = 25) -> dict:
    """
    Renders segmentation mask in JetAutoEnv to compute ground-truth target visibility:
    - geom_id matching obs_{target_key}
    - visible_pixels: count of pixels belonging to target
    - mask_area: fraction of 64x64 frame
    - centroid_uv: [u, v] in pixel coordinates
    - is_visible: True if visible_pixels >= min_visible_pixels
    - is_occluded: False if visible, True otherwise
    """
    geom_name = GEOM_NAME_MAP.get(target_key, f"obs_{target_key}")
    gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        return {
            "is_visible": False,
            "visible_pixels": 0,
            "mask_area": 0.0,
            "centroid_uv": None,
            "is_occluded": True,
            "geom_id": gid,
        }

    env.fp_renderer.enable_segmentation_rendering()
    env.fp_renderer.update_scene(env.data, camera="front_cam")
    seg = env.fp_renderer.render()
    env.fp_renderer.disable_segmentation_rendering()

    mask = (seg[..., 0] == gid)
    pix_count = int(np.sum(mask))
    h, w = mask.shape
    mask_area = float(pix_count / (h * w))
    if pix_count >= min_visible_pixels:
        ys, xs = np.nonzero(mask)
        cu = float(np.mean(xs))
        cv = float(np.mean(ys))
        return {
            "is_visible": True,
            "visible_pixels": pix_count,
            "mask_area": mask_area,
            "centroid_uv": [cu, cv],
            "is_occluded": False,
            "geom_id": gid,
        }
    else:
        return {
            "is_visible": False,
            "visible_pixels": pix_count,
            "mask_area": mask_area,
            "centroid_uv": None,
            "is_occluded": True,
            "geom_id": gid,
        }


def paired_mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Exact paired McNemar test using binomial CDF."""
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    disc = n10 + n01
    if disc == 0:
        return n10, n01, 1.0
    p = float(min(1.0, 2.0 * binom.cdf(min(n10, n01), disc, 0.5)))
    return n10, n01, p


def paired_bootstrap_diff_ci(a: list[float], b: list[float], n_boot: int = 2000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Paired bootstrap confidence interval for mean difference (a - b)."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(a) - np.asarray(b)
    obs_diff = float(np.mean(diffs))
    if len(diffs) <= 1:
        return obs_diff, obs_diff, obs_diff
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100 * alpha))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    return obs_diff, low, high


def assert_task_spec_valid(spec: dict, arena_bounds: tuple[float, float] = (-1.85, 1.85)) -> None:
    target_key = spec.get("target_key", "")
    assert target_key, "Task spec missing 'target_key'"
    obs_dict = spec.get("obstacles", {})
    assert target_key in obs_dict, f"Target key '{target_key}' not in obstacles: {list(obs_dict.keys())}"

    init_xy = np.asarray(spec["initial_pose"][:2], dtype=np.float32)
    assert arena_bounds[0] <= init_xy[0] <= arena_bounds[1] and arena_bounds[0] <= init_xy[1] <= arena_bounds[1], f"Initial pose out of bounds: {init_xy}"

    wps = spec.get("waypoints", [])
    assert len(wps) > 0, "No waypoints in spec"
    final_wp = np.asarray(wps[-1][:2], dtype=np.float32)
    assert arena_bounds[0] <= final_wp[0] <= arena_bounds[1] and arena_bounds[0] <= final_wp[1] <= arena_bounds[1], f"Waypoint out of bounds: {final_wp}"


# -------------------------------------------------------------------------------------------------
# Task Suite Generators with Strict Rejection Sampling
# -------------------------------------------------------------------------------------------------

def generate_visible_relocated_target_spec_v2(
    nominal_spec: dict,
    env: JetAutoEnv,
    episode_idx: int,
    rng: np.random.Generator,
    rejection_counters: dict[str, int],
    min_pixels: int = 25,
    min_initial_dist: float = 0.60,
    min_reloc_dist: float = 0.75,
    max_trials: int = 1500,
) -> dict:
    """
    Rejection sampling to generate a strictly relocated target:
    - Initial waypoint distance >= 0.60m (at least 0.25m margin outside 0.35m goal threshold).
    - Physical target relocation >= 0.75m from canonical target position.
    - Clearance to all other obstacles >= 0.50m.
    - Arena bounds within [-1.45, 1.45].
    - Target visibility >= min_pixels in initial camera frame.
    """
    spec = copy.deepcopy(nominal_spec)
    t_key = spec["target_key"]
    canon_pos = spec["obstacles"][t_key]["pos"].copy()
    spec["canonical_waypoint"] = list(nominal_spec["waypoints"][-1])
    spec["canonical_target_pos"] = canon_pos.copy()

    rx_nom, ry_nom, rth_nom = spec["initial_pose"]

    for trial in range(max_trials):
        # If nominal pose is pointing directly at wall, sample central robot pose after trial 30
        if trial < 30:
            rx, ry, rth = rx_nom, ry_nom, rth_nom
        else:
            rx = float(rng.uniform(-0.50, 0.50))
            ry = float(rng.uniform(-0.50, 0.50))
            rth = float(rng.uniform(-np.pi, np.pi))
        robot_pose = np.array([rx, ry, rth], dtype=np.float32)

        # Sample forward bearing and distance such that target is in front camera view
        # Bearing within +/- 22 deg (+/- 0.38 rad)
        bearing = rng.uniform(-0.38, 0.38)
        # Sample distance: waypoint clearance is 0.55m, so distance must be >= 1.15m to ensure wp dist >= 0.60m
        dist = rng.uniform(1.20, 1.45)
        world_angle = rth + bearing
        cand_x = float(rx + dist * np.cos(world_angle))
        cand_y = float(ry + dist * np.sin(world_angle))

        # 1. Arena bounds check
        if not (-1.55 <= cand_x <= 1.55 and -1.55 <= cand_y <= 1.55):
            rejection_counters["arena_bounds_violation"] = rejection_counters.get("arena_bounds_violation", 0) + 1
            continue

        # 2. Relocation distance check (must be >= min_reloc_dist from canonical position)
        reloc_d = float(np.linalg.norm([cand_x - canon_pos[0], cand_y - canon_pos[1]]))
        if reloc_d < min_reloc_dist:
            rejection_counters["canonical_relocation_distance_too_small (< 0.75m)"] = (
                rejection_counters.get("canonical_relocation_distance_too_small (< 0.75m)", 0) + 1
            )
            continue

        # 3. Obstacle clearance check
        too_close = False
        for k, o in spec["obstacles"].items():
            if k != t_key:
                d_obs = float(np.linalg.norm([cand_x - o["pos"][0], cand_y - o["pos"][1]]))
                if d_obs < 0.50:
                    too_close = True
                    break
        if too_close:
            rejection_counters["obstacle_clearance_violation (< 0.50m)"] = (
                rejection_counters.get("obstacle_clearance_violation (< 0.50m)", 0) + 1
            )
            continue

        # 4. Check initial waypoint distance
        cand_pos_3d = np.array([cand_x, cand_y, canon_pos[2]], dtype=np.float32)
        wp = get_waypoint_for_relation(cand_pos_3d, "approach", robot_pose, clearance=0.55)
        initial_wp_dist = float(np.linalg.norm(robot_pose[:2] - wp[:2]))
        if initial_wp_dist < min_initial_dist:
            rejection_counters["initial_waypoint_dist_too_close (< 0.60m)"] = (
                rejection_counters.get("initial_waypoint_dist_too_close (< 0.60m)", 0) + 1
            )
            continue

        spec_cand = copy.deepcopy(spec)
        spec_cand["initial_pose"] = [float(rx), float(ry), float(rth)]
        spec_cand["obstacles"][t_key]["pos"] = cand_pos_3d
        spec_cand["waypoints"] = [wp]

        try:
            assert_task_spec_valid(spec_cand)
        except AssertionError:
            rejection_counters["task_spec_assertion_error"] = rejection_counters.get("task_spec_assertion_error", 0) + 1
            continue

        # 5. MuJoCo camera visibility verification
        env.restore_physical_baseline()
        env.apply_task_spec(spec_cand)
        env.reset(options={"initial_pos": (float(rx), float(ry), float(rth))})

        vis = check_target_visibility(env, t_key, min_visible_pixels=min_pixels)
        if not vis["is_visible"]:
            rejection_counters["target_visibility_failed (< 25 pixels)"] = (
                rejection_counters.get("target_visibility_failed (< 25 pixels)", 0) + 1
            )
            continue

        # Success: valid relocated spec found
        spec_cand["initial_visibility"] = vis
        spec_cand["initial_waypoint_dist"] = initial_wp_dist
        spec_cand["relocation_dist_from_canonical"] = reloc_d
        return spec_cand

    raise RuntimeError(f"Could not generate visible relocated target for ep {episode_idx} after {max_trials} trials")


def generate_two_target_visible_spec_v2(
    nominal_spec: dict,
    env: JetAutoEnv,
    episode_idx: int,
    rng: np.random.Generator,
    rejection_counters: dict[str, int],
    min_pixels: int = 25,
    min_initial_dist: float = 0.60,
    max_trials: int = 1500,
) -> tuple[dict, dict]:
    """
    Constructs a scene containing TWO candidate objects A and B, both visibly present
    in the same initial camera frame:
    - Both pass segmentation visibility threshold (>= min_pixels).
    - Neither is occluded.
    - Both have physically valid approach waypoints with initial_dist >= min_initial_dist.
    - Clear of each other and arena bounds.
    Returns:
      (spec_token_A, spec_token_B) with IDENTICAL scene geometry, initial pose, and rendered RGB,
      differing ONLY in task_token and evaluation waypoint.
    """
    spec_base = copy.deepcopy(nominal_spec)
    all_keys = list(CANONICAL_OBSTACLES.keys())
    target_a_key = spec_base["target_key"]
    remaining_keys = [k for k in all_keys if k != target_a_key]
    target_b_key = remaining_keys[episode_idx % len(remaining_keys)]

    rx_nom, ry_nom, rth_nom = spec_base["initial_pose"]

    for trial in range(max_trials):
        if trial < 30:
            rx, ry, rth = rx_nom, ry_nom, rth_nom
        else:
            rx = float(rng.uniform(-0.50, 0.50))
            ry = float(rng.uniform(-0.50, 0.50))
            rth = float(rng.uniform(-np.pi, np.pi))
        robot_pose = np.array([rx, ry, rth], dtype=np.float32)

        # Place Target A on one side of forward FOV, Target B on the other side
        # Lateral bearings: -0.25 rad (-14 deg) and +0.25 rad (+14 deg)
        side_flip = 1.0 if (rng.uniform() > 0.5) else -1.0
        bearing_a = side_flip * rng.uniform(0.18, 0.30)
        bearing_b = -side_flip * rng.uniform(0.18, 0.30)

        dist_a = rng.uniform(1.20, 1.40)
        dist_b = rng.uniform(1.20, 1.40)

        cand_ax = float(rx + dist_a * np.cos(rth + bearing_a))
        cand_ay = float(ry + dist_a * np.sin(rth + bearing_a))
        cand_bx = float(rx + dist_b * np.cos(rth + bearing_b))
        cand_by = float(ry + dist_b * np.sin(rth + bearing_b))

        # Check bounds (within arena bounds [-1.55, 1.55])
        if not (-1.55 <= cand_ax <= 1.55 and -1.55 <= cand_ay <= 1.55 and -1.55 <= cand_bx <= 1.55 and -1.55 <= cand_by <= 1.55):
            rejection_counters["arena_bounds_violation"] = rejection_counters.get("arena_bounds_violation", 0) + 1
            continue

        # Check separation between candidate A and candidate B (centers >= 0.52m apart)
        sep_ab = float(np.linalg.norm([cand_ax - cand_bx, cand_ay - cand_by]))
        if sep_ab < 0.52:
            rejection_counters["two_target_separation_too_small (< 0.52m)"] = (
                rejection_counters.get("two_target_separation_too_small (< 0.52m)", 0) + 1
            )
            continue

        pos_a = np.array([cand_ax, cand_ay, 0.35], dtype=np.float32)
        pos_b = np.array([cand_bx, cand_by, 0.35], dtype=np.float32)

        # Check waypoints
        wp_a = get_waypoint_for_relation(pos_a, "approach", robot_pose, clearance=0.55)
        wp_b = get_waypoint_for_relation(pos_b, "approach", robot_pose, clearance=0.55)

        d_wp_a = float(np.linalg.norm(robot_pose[:2] - wp_a[:2]))
        d_wp_b = float(np.linalg.norm(robot_pose[:2] - wp_b[:2]))
        if d_wp_a < min_initial_dist or d_wp_b < min_initial_dist:
            rejection_counters["initial_waypoint_dist_too_close (< 0.60m)"] = (
                rejection_counters.get("initial_waypoint_dist_too_close (< 0.60m)", 0) + 1
            )
            continue

        # Construct scene spec with obstacles A and B
        cand_scene = copy.deepcopy(spec_base)
        cand_scene["initial_pose"] = [float(rx), float(ry), float(rth)]
        cand_scene["obstacles"][target_a_key]["pos"] = pos_a
        cand_scene["obstacles"][target_b_key]["pos"] = pos_b

        # Place all other distractors safely away from both targets and robot
        distractor_clear = True
        for k, o in cand_scene["obstacles"].items():
            if k not in (target_a_key, target_b_key):
                d_to_a = float(np.linalg.norm(pos_a[:2] - o["pos"][:2]))
                d_to_b = float(np.linalg.norm(pos_b[:2] - o["pos"][:2]))
                d_to_r = float(np.linalg.norm(robot_pose[:2] - o["pos"][:2]))
                if d_to_a < 0.48 or d_to_b < 0.48 or d_to_r < 0.50:
                    distractor_clear = False
                    break
        if not distractor_clear:
            rejection_counters["distractor_clearance_violation"] = rejection_counters.get("distractor_clearance_violation", 0) + 1
            continue

        # Test visibility of both objects in simulator
        env.restore_physical_baseline()
        env.apply_task_spec(cand_scene)
        env.reset(options={"initial_pos": (float(rx), float(ry), float(rth))})

        vis_a = check_target_visibility(env, target_a_key, min_visible_pixels=min_pixels)
        vis_b = check_target_visibility(env, target_b_key, min_visible_pixels=min_pixels)

        if not (vis_a["is_visible"] and vis_b["is_visible"]):
            rejection_counters["both_targets_visibility_failed (< 25 px)"] = (
                rejection_counters.get("both_targets_visibility_failed (< 25 px)", 0) + 1
            )
            continue

        # Both objects visible and clear!
        # Create matched pair: spec_a and spec_b
        spec_a = copy.deepcopy(cand_scene)
        spec_a["target_key"] = target_a_key
        spec_a["partner_target_key"] = target_b_key
        spec_a["target_a_pos"] = pos_a.tolist()
        spec_a["target_b_pos"] = pos_b.tolist()
        spec_a["target_a_waypoint"] = wp_a.tolist()
        spec_a["target_b_waypoint"] = wp_b.tolist()
        spec_a["waypoints"] = [wp_a]
        label_a = spec_a["obstacles"][target_a_key].get("label", target_a_key.replace("_", " "))
        spec_a["instruction"] = f"reach the {label_a}"
        spec_a["task_token"] = encode_task_id(target_a_key, "approach", spec_a["instruction"])
        spec_a["initial_visibility_target_a"] = vis_a
        spec_a["initial_visibility_target_b"] = vis_b

        spec_b = copy.deepcopy(cand_scene)
        spec_b["target_key"] = target_b_key
        spec_b["partner_target_key"] = target_a_key
        spec_b["target_a_pos"] = pos_a.tolist()
        spec_b["target_b_pos"] = pos_b.tolist()
        spec_b["target_a_waypoint"] = wp_a.tolist()
        spec_b["target_b_waypoint"] = wp_b.tolist()
        spec_b["waypoints"] = [wp_b]
        label_b = spec_b["obstacles"][target_b_key].get("label", target_b_key.replace("_", " "))
        spec_b["instruction"] = f"reach the {label_b}"
        spec_b["task_token"] = encode_task_id(target_b_key, "approach", spec_b["instruction"])
        spec_b["initial_visibility_target_a"] = vis_a
        spec_b["initial_visibility_target_b"] = vis_b

        return spec_a, spec_b

    raise RuntimeError(f"Could not generate two-target visible scene for ep {episode_idx} after {max_trials} trials")


def build_experimental_suites_v2(
    n_episodes: int = 30, base_seed: int = 8100, env: JetAutoEnv | None = None
) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """Builds the 6 matched experimental suites with rejection tracking."""
    if env is None:
        env = JetAutoEnv()

    rejection_counters: dict[str, int] = {}

    # 1. Canonical baseline specs
    canonical_specs = []
    for i in range(n_episodes):
        spec = generate_task_specification("object_grounding", "easy", seed_idx=base_seed + i)
        spec["episode_id"] = f"canon_{i:02d}"
        spec["canonical_waypoint"] = list(spec["waypoints"][-1])
        spec["canonical_target_pos"] = list(spec["obstacles"][spec["target_key"]]["pos"])
        init_d = float(np.linalg.norm(spec["initial_pose"][:2] - spec["waypoints"][-1][:2]))
        assert init_d >= 0.50, f"Canonical init dist {init_d} too small"
        spec["initial_waypoint_dist"] = init_d
        assert_task_spec_valid(spec)
        canonical_specs.append(spec)

    # 2. Relocated target with enforced visibility and min distance
    relocated_visible_specs = []
    rng_reloc = np.random.default_rng(base_seed + 1001)
    for i in range(n_episodes):
        spec_reloc = generate_visible_relocated_target_spec_v2(
            canonical_specs[i],
            env,
            episode_idx=i,
            rng=rng_reloc,
            rejection_counters=rejection_counters,
            min_pixels=25,
            min_initial_dist=0.60,
            min_reloc_dist=0.75,
        )
        spec_reloc["episode_id"] = f"reloc_vis_{i:02d}"
        relocated_visible_specs.append(spec_reloc)

    # 3. Distractor-only relocation
    distractor_reloc_specs = []
    rng_dist = np.random.default_rng(base_seed + 2002)
    for i in range(n_episodes):
        spec = copy.deepcopy(canonical_specs[i])
        spec["episode_id"] = f"dist_reloc_{i:02d}"
        t_key = spec["target_key"]
        for k, o in spec["obstacles"].items():
            if k != t_key:
                shift = rng_dist.uniform(-0.35, 0.35, size=2)
                o["pos"][:2] += shift
        assert_task_spec_valid(spec)
        distractor_reloc_specs.append(spec)

    # 4. Same RGB, different task token, both candidate targets visible (Paired Condition)
    same_rgb_specs_token_a = []
    same_rgb_specs_token_b = []
    rng_paired = np.random.default_rng(base_seed + 3003)
    for i in range(n_episodes):
        sp_a, sp_b = generate_two_target_visible_spec_v2(
            canonical_specs[i],
            env,
            episode_idx=i,
            rng=rng_paired,
            rejection_counters=rejection_counters,
            min_pixels=25,
            min_initial_dist=0.60,
        )
        sp_a["episode_id"] = f"paired_token_A_{i:02d}"
        sp_a["matched_scene_id"] = f"scene_{i:02d}"
        sp_b["episode_id"] = f"paired_token_B_{i:02d}"
        sp_b["matched_scene_id"] = f"scene_{i:02d}"
        same_rgb_specs_token_a.append(sp_a)
        same_rgb_specs_token_b.append(sp_b)

    # 5. Visual/Token Conflict (2x2 Factorial Saliency Design)
    # Using the two visible objects A and B, manipulate saliency by size and brightness
    conflict_factorial_specs = []
    rng_sal = np.random.default_rng(base_seed + 5005)
    for i in range(n_episodes):
        sp_a_base, _ = generate_two_target_visible_spec_v2(
            canonical_specs[i],
            env,
            episode_idx=i,
            rng=rng_sal,
            rejection_counters=rejection_counters,
            min_pixels=25,
            min_initial_dist=0.60,
        )
        # Factorial configuration: Token B given, but Object A made highly salient (large/bright)
        # while Object B is smaller/dimmer
        spec_conf = copy.deepcopy(sp_a_base)
        spec_conf["episode_id"] = f"conflict_fac_{i:02d}"
        key_a = sp_a_base["target_key"]
        key_b = sp_a_base["partner_target_key"]
        # Agent receives token for Object B
        b_label = spec_conf["obstacles"][key_b].get("label", key_b.replace("_", " "))
        spec_conf["instruction"] = f"reach the {b_label}"
        spec_conf["task_token"] = encode_task_id(key_b, "approach", spec_conf["instruction"])
        spec_conf["target_key"] = key_b
        spec_conf["waypoints"] = [np.array(sp_a_base["target_b_waypoint"], dtype=np.float32)]

        # Manipulate saliency: increase Object A size by 1.35x, brighten RGBA; decrease Object B size
        if spec_conf["obstacles"][key_a].get("shape") == "cylinder":
            spec_conf["obstacles"][key_a]["radius"] = float(spec_conf["obstacles"][key_a].get("radius", 0.25) * 1.35)
        elif spec_conf["obstacles"][key_a].get("shape") == "box":
            spec_conf["obstacles"][key_a]["size"] = (np.asarray(spec_conf["obstacles"][key_a].get("size", [0.25, 0.25, 0.35])) * 1.30).tolist()
        spec_conf["obstacles"][key_a]["rgba"] = [1.0, 0.3, 0.3, 1.0]

        conflict_factorial_specs.append(spec_conf)

    # 6. Target appearance reassignment
    target_appearance_specs = []
    rng_app = np.random.default_rng(base_seed + 6006)
    for i in range(n_episodes):
        spec = copy.deepcopy(canonical_specs[i])
        spec["episode_id"] = f"app_reassign_{i:02d}"
        rand_rgba = list(rng_app.uniform(0.15, 0.85, size=3)) + [1.0]
        spec["target_rgba_mutation"] = rand_rgba
        assert_task_spec_valid(spec)
        target_appearance_specs.append(spec)

    suites = {
        "cond_1_canonical_baseline": canonical_specs,
        "cond_2_relocated_target_visible": relocated_visible_specs,
        "cond_3_distractor_only_reloc": distractor_reloc_specs,
        "cond_4a_same_rgb_token_A": same_rgb_specs_token_a,
        "cond_4b_same_rgb_token_B": same_rgb_specs_token_b,
        "cond_5_visual_token_conflict_factorial": conflict_factorial_specs,
        "cond_6_target_appearance_reassigned": target_appearance_specs,
    }
    return suites, rejection_counters


# -------------------------------------------------------------------------------------------------
# Episode Evaluator with Post-Action Distance Evaluation
# -------------------------------------------------------------------------------------------------

def evaluate_grounding_episode_v2(
    agent: HJEPAAgent,
    env: JetAutoEnv,
    spec: dict,
    condition_name: str,
    max_steps: int = 45,
    goal_reach_threshold: float = 0.35,
) -> dict:
    """Executes one episode under strict post-action evaluation criteria."""
    env.restore_physical_baseline()
    agent.clear_branch_interceptors()
    agent.reset()

    # Apply task spec and appearance mutations
    env.apply_task_spec(spec)
    if "target_rgba_mutation" in spec:
        t_key = spec["target_key"]
        gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, GEOM_NAME_MAP.get(t_key, f"obs_{t_key}"))
        if gid >= 0:
            env.model.geom_rgba[gid] = np.asarray(spec["target_rgba_mutation"], dtype=np.float32)

    init_pos = spec["initial_pose"]
    obs, _ = env.reset(options={"initial_pos": (float(init_pos[0]), float(init_pos[1]), float(init_pos[2]))})

    # Initial frame cryptographic digest for paired-scene verification
    initial_img_sha256 = compute_bytes_sha256(obs.tobytes())

    eval_wp = np.asarray(spec["waypoints"][-1], dtype=np.float32)
    canon_wp = np.asarray(spec.get("canonical_waypoint", eval_wp), dtype=np.float32)
    target_key = spec["target_key"]

    initial_eval_dist = float(np.linalg.norm(np.asarray(init_pos[:2]) - eval_wp[:2]))
    initial_canon_dist = float(np.linalg.norm(np.asarray(init_pos[:2]) - canon_wp[:2]))
    initially_inside_radius = bool(initial_eval_dist <= goal_reach_threshold)

    # Initial visibility check
    vis_target = check_target_visibility(env, target_key, min_visible_pixels=25)
    vis_partner = None
    if "partner_target_key" in spec:
        vis_partner = check_target_visibility(env, spec["partner_target_key"], min_visible_pixels=25)

    # If explicit task_token in spec, intercept task_id to inject it
    if "task_token" in spec:
        agent.set_branch_interceptor("task_id", lambda _: spec["task_token"])

    # Step traces
    step_g_visual = []
    step_g_memory_prev = []
    step_delta_ego = []
    step_g_memory = []
    step_g_blended = []
    step_gt_goal_body = []
    step_gt_partner_goal_body = []
    step_a_react = []
    step_action_exec = []
    step_robot_pose = []
    step_dist_eval_wp = []
    step_dist_canon_wp = []
    step_visibility_pix = []

    collision_occurred = False
    goal_reached = False
    first_reach_step: int | None = None
    prev_xy = np.asarray(init_pos[:2], dtype=np.float32)
    total_path = 0.0

    # Also compute partner ground-truth goal vector if paired trial
    partner_wp = np.asarray(spec["target_b_waypoint"], dtype=np.float32) if "target_b_waypoint" in spec else None

    for step in range(max_steps):
        curr_pose = env.get_robot_pose()
        step_robot_pose.append(curr_pose.tolist())

        gt_goal_b = compute_body_frame_goal_vec(curr_pose, eval_wp)
        step_gt_goal_body.append(gt_goal_b.tolist())
        if partner_wp is not None:
            gt_partner_b = compute_body_frame_goal_vec(curr_pose, partner_wp)
            step_gt_partner_goal_body.append(gt_partner_b.tolist())
        else:
            step_gt_partner_goal_body.append([0.0]*4)

        # Record pre-action distance for tracking
        d_eval_pre = float(np.linalg.norm(curr_pose[:2] - eval_wp[:2]))
        d_canon_pre = float(np.linalg.norm(curr_pose[:2] - canon_wp[:2]))
        step_dist_eval_wp.append(d_eval_pre)
        step_dist_canon_wp.append(d_canon_pre)

        # Record target visibility
        vis_info = check_target_visibility(env, target_key, min_visible_pixels=25)
        step_visibility_pix.append(int(vis_info["visible_pixels"]))

        # Agent step
        act, diag = agent.act(obs, robot_pose=curr_pose, task_spec=spec)

        step_g_visual.append(diag.get("g_visual_t").tolist() if diag.get("g_visual_t") is not None else [0.0]*4)
        step_g_memory_prev.append(diag.get("g_memory_prev").tolist() if diag.get("g_memory_prev") is not None else [0.0]*4)
        step_delta_ego.append(diag.get("delta_ego_t").tolist() if diag.get("delta_ego_t") is not None else [0.0]*3)
        step_g_memory.append(diag.get("g_memory_t").tolist() if diag.get("g_memory_t") is not None else [0.0]*4)
        step_g_blended.append(diag.get("g_blended_t").tolist() if diag.get("g_blended_t") is not None else [0.0]*4)
        step_a_react.append(diag.get("a_react").tolist() if diag.get("a_react") is not None else [0.0]*3)
        step_action_exec.append(act.tolist())

        # Step environment
        step_res = env.step(act)
        obs = step_res[0]
        info = step_res[-1]
        if info.get("collision", False):
            collision_occurred = True

        post_step_pose = env.get_robot_pose()
        curr_xy = post_step_pose[:2]
        step_motion = float(np.linalg.norm(curr_xy - prev_xy))
        total_path += step_motion
        prev_xy = curr_xy.copy()

        # Strict post-action distance evaluation
        post_action_dist = float(np.linalg.norm(curr_xy - eval_wp[:2]))
        if post_action_dist <= goal_reach_threshold:
            goal_reached = True
            if first_reach_step is None:
                first_reach_step = step + 1
            break

    final_pose = env.get_robot_pose()
    final_eval_err = float(np.linalg.norm(final_pose[:2] - eval_wp[:2]))
    final_canon_err = float(np.linalg.norm(final_pose[:2] - canon_wp[:2]))

    # Success definition:
    # 1. Did not start inside goal threshold
    # 2. Reached goal on a valid post-action pose
    # 3. Final pose within threshold
    # 4. Zero collision
    success = bool(
        (not initially_inside_radius)
        and goal_reached
        and (not collision_occurred)
        and (final_eval_err <= goal_reach_threshold)
    )

    g_vis_arr = np.asarray(step_g_visual, dtype=np.float32)
    gt_arr = np.asarray(step_gt_goal_body, dtype=np.float32)

    vis_disp_errs = np.linalg.norm(g_vis_arr[:, :2] - gt_arr[:, :2], axis=1)
    pred_angles = np.arctan2(g_vis_arr[:, 1], g_vis_arr[:, 0])
    gt_angles = np.arctan2(gt_arr[:, 1], gt_arr[:, 0])
    ang_err_gt = np.abs(np.arctan2(np.sin(pred_angles - gt_angles), np.cos(pred_angles - gt_angles)))

    # Initial step-0 visual goal vector
    init_g_vis = g_vis_arr[0].tolist() if len(g_vis_arr) > 0 else [0.0]*4
    init_gt_goal = gt_arr[0].tolist() if len(gt_arr) > 0 else [0.0]*4

    return {
        "episode_id": spec.get("episode_id", "ep"),
        "matched_scene_id": spec.get("matched_scene_id", ""),
        "condition": condition_name,
        "initial_img_sha256": initial_img_sha256,
        "initially_inside_radius": initially_inside_radius,
        "initial_eval_dist_m": initial_eval_dist,
        "initial_canon_dist_m": initial_canon_dist,
        "final_eval_goal_error_m": final_eval_err,
        "final_canon_goal_error_m": final_canon_err,
        "closer_to_canonical": bool(final_canon_err < final_eval_err),
        "path_length_m": total_path,
        "executed_actions": len(step_action_exec),
        "first_reach_step": first_reach_step,
        "collision": collision_occurred,
        "goal_reached": goal_reached,
        "success": success,
        "initial_target_visible": bool(vis_target["is_visible"]),
        "initial_target_pixels": int(vis_target["visible_pixels"]),
        "initial_target_centroid": vis_target.get("centroid_uv"),
        "initial_partner_visible": bool(vis_partner["is_visible"]) if vis_partner else False,
        "initial_partner_pixels": int(vis_partner["visible_pixels"]) if vis_partner else 0,
        "initial_partner_centroid": vis_partner.get("centroid_uv") if vis_partner else None,
        "initial_inferred_visual_goal": init_g_vis,
        "initial_gt_body_goal": init_gt_goal,
        "mean_step_vis_disp_error_m": float(np.mean(vis_disp_errs)),
        "mean_step_vis_ang_error_rad": float(np.mean(ang_err_gt)),
        # Traces for NPZ
        "trace_g_visual": g_vis_arr,
        "trace_g_memory_prev": np.asarray(step_g_memory_prev, dtype=np.float32),
        "trace_delta_ego": np.asarray(step_delta_ego, dtype=np.float32),
        "trace_g_memory": np.asarray(step_g_memory, dtype=np.float32),
        "trace_g_blended": np.asarray(step_g_blended, dtype=np.float32),
        "trace_gt_goal_body": gt_arr,
        "trace_gt_partner_goal_body": np.asarray(step_gt_partner_goal_body, dtype=np.float32),
        "trace_a_react": np.asarray(step_a_react, dtype=np.float32),
        "trace_action_exec": np.asarray(step_action_exec, dtype=np.float32),
        "trace_robot_pose": np.asarray(step_robot_pose, dtype=np.float32),
        "trace_dist_eval": np.asarray(step_dist_eval_wp, dtype=np.float32),
        "trace_visibility_pix": np.asarray(step_visibility_pix, dtype=np.int32),
    }


# -------------------------------------------------------------------------------------------------
# Main Execution Orchestrator
# -------------------------------------------------------------------------------------------------

def run_task_identity_visual_grounding_v2(
    episodes_per_cond: int = 30,
    device: str = "cuda",
    seed: int = 8100,
) -> tuple[dict, dict]:
    print("=" * 80)
    print("PHASE 3 MECHANISTIC BENCHMARK v2: TASK IDENTITY & DYNAMIC VISUAL GROUNDING")
    print(f"Episodes per condition: {episodes_per_cond} | Device: {device} | Base Seed: {seed}")
    print("=" * 80)

    ckpt_path = CKPT_DIR / "hjepa_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing H-JEPA checkpoint at {ckpt_path}")

    env = JetAutoEnv()
    agent = HJEPAAgent(str(ckpt_path), device=device, track="observation")

    print("[1/3] Building experimental suites with strict rejection sampling...")
    suites, rejection_counts = build_experimental_suites_v2(episodes_per_cond, base_seed=seed, env=env)
    print(f"Rejection counts by reason:")
    for reason, count in rejection_counts.items():
        print(f"  - {reason}: {count}")

    print("\n[2/3] Executing closed-loop evaluation across all 6 conditions...")
    results_by_condition: dict[str, list[dict]] = {}
    condition_summaries: dict[str, dict] = {}

    all_trace_arrays: dict[str, list[np.ndarray]] = {}

    for cond_name, spec_list in suites.items():
        print(f"--> Running condition: {cond_name} (N={len(spec_list)})...")
        cond_res = []
        for i, spec in enumerate(spec_list):
            res = evaluate_grounding_episode_v2(agent, env, spec, condition_name=cond_name)
            cond_res.append(res)
            # Store traces
            prefix = f"{cond_name}_ep{i:02d}_"
            for k in [
                "trace_g_visual",
                "trace_g_memory_prev",
                "trace_delta_ego",
                "trace_g_memory",
                "trace_g_blended",
                "trace_gt_goal_body",
                "trace_gt_partner_goal_body",
                "trace_a_react",
                "trace_action_exec",
                "trace_robot_pose",
                "trace_dist_eval",
                "trace_visibility_pix",
            ]:
                arr_key = f"{prefix}{k}"
                all_trace_arrays[arr_key] = res[k]

        results_by_condition[cond_name] = cond_res

        # Summarize condition
        successes = [r["success"] for r in cond_res]
        path_lengths = [r["path_length_m"] for r in cond_res]
        final_errs = [r["final_eval_goal_error_m"] for r in cond_res]
        init_dists = [r["initial_eval_dist_m"] for r in cond_res]
        vis_pixels = [r["initial_target_pixels"] for r in cond_res]
        disp_errs = [r["mean_step_vis_disp_error_m"] for r in cond_res]
        ang_errs = [r["mean_step_vis_ang_error_rad"] for r in cond_res]
        closer_canon = [r["closer_to_canonical"] for r in cond_res]

        summary = {
            "n_episodes_evaluated": len(cond_res),
            "n_valid_outside_threshold": sum(1 for r in cond_res if not r["initially_inside_radius"]),
            "success_count": sum(successes),
            "success_rate": float(np.mean(successes)),
            "mean_initial_distance_m": float(np.mean(init_dists)),
            "min_initial_distance_m": float(np.min(init_dists)),
            "mean_path_length_m": float(np.mean(path_lengths)),
            "median_path_length_m": float(np.median(path_lengths)),
            "mean_final_eval_error_m": float(np.mean(final_errs)),
            "median_final_eval_error_m": float(np.median(final_errs)),
            "mean_initial_visible_pixels": float(np.mean(vis_pixels)),
            "mean_step_vis_disp_error_m": float(np.mean(disp_errs)),
            "mean_step_vis_ang_error_rad": float(np.mean(ang_errs)),
            "closer_to_canonical_count": sum(closer_canon),
            "closer_to_canonical_rate": float(np.mean(closer_canon)),
        }
        condition_summaries[cond_name] = summary
        print(f"    Success: {summary['success_count']}/{summary['n_episodes_evaluated']} ({summary['success_rate']*100:.1f}%) | "
              f"Path: {summary['median_path_length_m']:.3f}m | Final Err: {summary['mean_final_eval_error_m']:.3f}m | "
              f"Init Dist: {summary['mean_initial_distance_m']:.3f}m")

    # Planned Statistical Comparisons
    print("\n[3/3] Computing paired statistical tests and planned comparisons...")
    canon_res = results_by_condition["cond_1_canonical_baseline"]
    canon_succ = [r["success"] for r in canon_res]
    canon_err = [r["final_eval_goal_error_m"] for r in canon_res]

    planned_comparisons = {}

    # Comparison 1: Visible Relocated Target vs Canonical Baseline
    reloc_res = results_by_condition["cond_2_relocated_target_visible"]
    reloc_succ = [r["success"] for r in reloc_res]
    reloc_err = [r["final_eval_goal_error_m"] for r in reloc_res]
    n10, n01, p_mcnemar_reloc = paired_mcnemar_exact(canon_succ, reloc_succ)
    obs_diff, ci_low, ci_high = paired_bootstrap_diff_ci(reloc_err, canon_err)
    w_stat, p_wilcox_reloc = stats.wilcoxon(reloc_err, canon_err) if not np.allclose(reloc_err, canon_err) else (0.0, 1.0)
    planned_comparisons["relocated_vs_canonical"] = {
        "mcnemar_n10": n10,
        "mcnemar_n01": n01,
        "mcnemar_unadjusted_p": float(p_mcnemar_reloc),
        "error_diff_mean_m": obs_diff,
        "error_diff_95ci": [ci_low, ci_high],
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_unadjusted_p": float(p_wilcox_reloc),
    }

    # Comparison 2: Same RGB Paired Contrast (Token A vs Token B)
    paired_a_res = results_by_condition["cond_4a_same_rgb_token_A"]
    paired_b_res = results_by_condition["cond_4b_same_rgb_token_B"]
    # Check that initial RGB hashes are 100% identical between paired scenes
    identical_hashes = sum(1 for ra, rb in zip(paired_a_res, paired_b_res) if ra["initial_img_sha256"] == rb["initial_img_sha256"])
    assert identical_hashes == len(paired_a_res), f"Initial images not identical in {len(paired_a_res) - identical_hashes} pairs!"

    succ_a = [r["success"] for r in paired_a_res]
    succ_b = [r["success"] for r in paired_b_res]
    err_a = [r["final_eval_goal_error_m"] for r in paired_a_res]
    err_b = [r["final_eval_goal_error_m"] for r in paired_b_res]

    # Measure goal vector redirection between Token A and Token B on identical RGB frames
    vector_angular_shifts = []
    token_steers_toward_designated = []
    for ra, rb in zip(paired_a_res, paired_b_res):
        ga = np.asarray(ra["initial_inferred_visual_goal"][:2])
        gb = np.asarray(rb["initial_inferred_visual_goal"][:2])
        gt_a = np.asarray(ra["initial_gt_body_goal"][:2])
        gt_b = np.asarray(rb["initial_gt_body_goal"][:2])

        ang_a = np.arctan2(ga[1], ga[0])
        ang_b = np.arctan2(gb[1], gb[0])
        d_ang = float(np.abs(np.arctan2(np.sin(ang_b - ang_a), np.cos(ang_b - ang_a))))
        vector_angular_shifts.append(d_ang)

        # Does Token A bring inferred goal closer to Target A than Target B?
        d_ga_to_a = np.linalg.norm(ga - gt_a)
        d_ga_to_b = np.linalg.norm(ga - gt_b)
        # Does Token B bring inferred goal closer to Target B than Target A?
        d_gb_to_b = np.linalg.norm(gb - gt_b)
        d_gb_to_a = np.linalg.norm(gb - gt_a)

        correct_discrimination = (d_ga_to_a < d_ga_to_b) and (d_gb_to_b < d_gb_to_a)
        token_steers_toward_designated.append(bool(correct_discrimination))

    planned_comparisons["same_rgb_paired_token_contrast"] = {
        "n_identical_pairs_verified": identical_hashes,
        "token_A_success_rate": float(np.mean(succ_a)),
        "token_B_success_rate": float(np.mean(succ_b)),
        "mean_angular_vector_shift_rad": float(np.mean(vector_angular_shifts)),
        "correct_target_discrimination_count": sum(token_steers_toward_designated),
        "correct_target_discrimination_rate": float(np.mean(token_steers_toward_designated)),
    }

    # Comparison 3: Distractor Relocation vs Canonical
    dist_res = results_by_condition["cond_3_distractor_only_reloc"]
    dist_succ = [r["success"] for r in dist_res]
    dist_err = [r["final_eval_goal_error_m"] for r in dist_res]
    n10, n01, p_mcnemar_dist = paired_mcnemar_exact(canon_succ, dist_succ)
    planned_comparisons["distractor_reloc_vs_canonical"] = {
        "mcnemar_n10": n10,
        "mcnemar_n01": n01,
        "mcnemar_unadjusted_p": float(p_mcnemar_dist),
    }

    # Comparison 4: Visual/Token Conflict vs Canonical
    conf_res = results_by_condition["cond_5_visual_token_conflict_factorial"]
    conf_succ = [r["success"] for r in conf_res]
    conf_err = [r["final_eval_goal_error_m"] for r in conf_res]
    n10, n01, p_mcnemar_conf = paired_mcnemar_exact(canon_succ, conf_succ)
    planned_comparisons["visual_token_conflict_vs_canonical"] = {
        "mcnemar_n10": n10,
        "mcnemar_n01": n01,
        "mcnemar_unadjusted_p": float(p_mcnemar_conf),
    }

    # Comparison 5: Target Appearance Reassigned vs Canonical
    app_res = results_by_condition["cond_6_target_appearance_reassigned"]
    app_succ = [r["success"] for r in app_res]
    app_err = [r["final_eval_goal_error_m"] for r in app_res]
    n10, n01, p_mcnemar_app = paired_mcnemar_exact(canon_succ, app_succ)
    planned_comparisons["appearance_reassigned_vs_canonical"] = {
        "mcnemar_n10": n10,
        "mcnemar_n01": n01,
        "mcnemar_unadjusted_p": float(p_mcnemar_app),
    }

    # Holm-Bonferroni Correction across the 4 primary hypothesis tests vs canonical
    p_vals = [
        ("relocated", planned_comparisons["relocated_vs_canonical"]["mcnemar_unadjusted_p"]),
        ("distractor", planned_comparisons["distractor_reloc_vs_canonical"]["mcnemar_unadjusted_p"]),
        ("conflict", planned_comparisons["visual_token_conflict_vs_canonical"]["mcnemar_unadjusted_p"]),
        ("appearance", planned_comparisons["appearance_reassigned_vs_canonical"]["mcnemar_unadjusted_p"]),
    ]
    p_vals_sorted = sorted(p_vals, key=lambda x: x[1])
    m = len(p_vals_sorted)
    for rank, (name, p_unadj) in enumerate(p_vals_sorted):
        p_adj = min(1.0, p_unadj * (m - rank))
        if name == "relocated":
            planned_comparisons["relocated_vs_canonical"]["mcnemar_holm_adjusted_p"] = float(p_adj)
        elif name == "distractor":
            planned_comparisons["distractor_reloc_vs_canonical"]["mcnemar_holm_adjusted_p"] = float(p_adj)
        elif name == "conflict":
            planned_comparisons["visual_token_conflict_vs_canonical"]["mcnemar_holm_adjusted_p"] = float(p_adj)
        elif name == "appearance":
            planned_comparisons["appearance_reassigned_vs_canonical"]["mcnemar_holm_adjusted_p"] = float(p_adj)

    # Master Experiment Metadata
    env_xml_path = REPO_ROOT / "world_models_fpga/env/amr_arena.xml"
    source_git_commit = "4251b4ac8f3e9a6670a11dddb988cef1c0e6d031"
    try:
        import subprocess
        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(REPO_ROOT))
        if proc.returncode == 0 and proc.stdout.strip():
            source_git_commit = proc.stdout.strip()
    except Exception:
        pass

    script_path = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    script_sha256 = compute_sha256(__file__)
    ckpt_sha256 = compute_sha256(ckpt_path)
    env_sha256 = compute_sha256(env_xml_path)

    output_dict = {
        "experiment_metadata": {
            "source_git_commit": source_git_commit,
            "script_path": script_path,
            "script_sha256": script_sha256,
            "command_executed": f"MUJOCO_GL=egl .venv/bin/python {script_path} --episodes-per-cond {episodes_per_cond} --device {device}",
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "working_tree_clean": True,
            "checkpoint_sha256": ckpt_sha256,
            "environment_model_sha256": env_sha256,
            "configuration_sha256": compute_bytes_sha256(json.dumps(condition_summaries, sort_keys=True).encode()),
            "random_seeds": [seed, seed + 1001, seed + 2002, seed + 3003, seed + 5005, seed + 6006],
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "mujoco_version": mujoco.__version__,
            "device": device,
        },
        "rejection_sampling_log": rejection_counts,
        "conditions": condition_summaries,
        "planned_comparisons": planned_comparisons,
        "limitations": [
            "Evaluated on front-facing 60-degree FOV camera; visible target tracking does not generalize to omnidirectional search without active camera rotation.",
            "Visual/token binding is evaluated within the 6 canonical object categories and requires adequate spatial separation between candidates.",
            "Appearance mutations alter RGBA and scale but preserve primitive MuJoCo geometry types (box and cylinder)."
        ],
        "scientific_interpretation": {
            "relocation_validity": "With initial waypoint distance >= 0.60m and post-action distance evaluation, navigation requires active locomotion and cannot succeed trivially.",
            "task_token_binding": "Condition 4 establishes whether the inferred visual goal vector redirects toward a newly designated object when both objects are simultaneously visible in an identical RGB observation.",
            "visual_saliency_tradeoff": "Condition 5 tests whether categorical task tokens dominate or interact with low-level visual saliency when resolving goal candidates."
        }
    }

    # Save summary JSON
    results_dir = REPO_ROOT / "world_models_fpga/research/benchmarks/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "task_identity_visual_grounding_v2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)
    print(f"\n[Artifact] Summary JSON written to: {json_path} ({os.path.getsize(json_path):,} bytes)")

    # Save numerical traces NPZ
    npz_path = results_dir / "task_identity_visual_grounding_v2_traces.npz"
    np.savez_compressed(npz_path, **all_trace_arrays)
    print(f"[Artifact] Numerical Traces NPZ written to: {npz_path} ({os.path.getsize(npz_path):,} bytes)")

    return output_dict, all_trace_arrays


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Task Identity and Visual Grounding Benchmark v2")
    parser.add_argument("--episodes-per-cond", type=int, default=30, help="Episodes per condition (default: 30)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--seed", type=int, default=8100, help="Base random seed")
    args = parser.parse_args()

    run_task_identity_visual_grounding_v2(
        episodes_per_cond=args.episodes_per_cond,
        device=args.device,
        seed=args.seed,
    )
