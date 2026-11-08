"""
Project Hephaestus: Phase 5.3 Valid Norm-Matched Causal Mediation & Necessity Revalidation Benchmark
Script: world_models_fpga/research/circuits/run_task_conditioned_circuits_v5_3.py

Scientific Objective:
Determine whether the task-sensitive hidden-unit subsets identified by H-JEPA exhibit genuine
causal selectivity beyond activation perturbation magnitude, or if their apparent effects arise
simply because selected units have larger task-induced activation differences.

Methodological Upgrades in Phase 5.3:
1. Strict Selection Filter:
   - Valid subsets strictly require rel_error = |N_rand - N_cand| / N_cand <= 0.05.
   - If >= 100 valid subsets exist, retain the 100 closest within tolerance.
   - NEVER substitute controls outside 5% tolerance.
   - If < 100 valid subsets exist after adaptive sampling up to 200,000 draws, assign
     INSUFFICIENT_COMMON_SUPPORT, retain None, and do not calculate confirmatory p-values.
2. Overlap Recording and Hypergeometric Stratification:
   - For every random control, record overlap O = |U_cand ∩ U_rand| and fraction O / k.
   - Compute hypergeometric overlap distribution weights w_o = C(k, o) * C(D - k, k - o) / C(D, k).
   - Compute overlap-standardized mean transfer efficiency across evaluable strata.
   - Disjoint controls (O = 0) evaluated as a secondary reference. At k = 32, evaluated as a
     single deterministic complement reference (C(32, 32) = 1).
3. Matched-Control Necessity Distribution:
   - Apply condition-averaged ablation (h_i -> (h_i^A + h_i^B) / 2) to BOTH the candidate subset
     AND the eligible norm-matched random controls.
   - Directly test whether candidate units exhibit greater causal necessity (contrast drop ratio)
     than activation-norm-matched random subsets.
4. Bidirectional Sufficiency:
   - A -> B and B -> A transplantation across 60 independent scene pairs (120 scenes).
   - Evaluate projected transfer efficiency eta, directional cosine, orthogonal residual,
     translation magnitude, and true heading transfer (atan2).
5. Statistical Rigor:
   - Statistical unit: Independent scene pair (N = 60). Average Scene A and Scene B, and average
     A -> B and B -> A.
   - Primary test family: 2 layers x 5 k values = 10 planned tests.
   - Paired Wilcoxon and sign-flip permutation tests (100,000 resamples).
   - Holm-Bonferroni correction across all 10 planned tests. Clustered bootstrap 95% CIs.

Outputs:
  - Summary JSON: world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_3_summary.json
  - Pairs JSONL: world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_3_pairs.jsonl
  - Activations JSONL: world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_3_activations.jsonl
  - Interventions JSONL (gzipped): world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_3_interventions.jsonl.gz
  - Control Audit JSONL: world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_3_control_audit.jsonl
  - Schema Markdown: world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_3_schema.md
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "world_models_fpga"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F

from env.amr_env import JetAutoEnv
from models.baseline_policy import encode_task_id, TARGET_KEYS
from research.benchmarks.model_wrappers import HJEPAAgent, compute_body_frame_goal_vec
from research.datasets.generation.capability_tasks import CANONICAL_OBSTACLES

GEOM_NAME_MAP = {
    "red_box": "obs_red_box",
    "blue_cylinder": "obs_blue_cyl",
    "green_cylinder": "obs_green_cyl",
    "yellow_cylinder": "obs_yellow_cyl",
    "purple_box": "obs_purple_box",
    "cyan_cylinder": "obs_cyan_cyl",
}


def check_target_visibility(env: JetAutoEnv, target_key: str, min_visible_pixels: int = 100) -> dict:
    """Renders segmentation mask in JetAutoEnv to compute ground-truth target visibility."""
    import mujoco
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


def generate_dual_visible_scene_pair(
    pair_idx: int,
    env: JetAutoEnv,
    rng: np.random.Generator,
    rejection_counters: dict[str, int],
    light_scale: float = 1.0,
    min_pixels: int = 100,
    min_initial_dist: float = 0.60,
    max_trials: int = 3000,
) -> dict | None:
    """
    Generates an independent scene pair with guaranteed dual visibility and strict physical clearance.
    Scene A: Object A at Pos 1, Object B at Pos 2.
    Scene B: Object A at Pos 2, Object B at Pos 1.
    """
    all_keys = list(TARGET_KEYS)
    key_a = all_keys[pair_idx % len(all_keys)]
    other_keys = [k for k in all_keys if k != key_a]
    key_b = other_keys[(pair_idx // len(all_keys)) % len(other_keys)]

    for trial in range(max_trials):
        rx = float(rng.uniform(-0.45, 0.45))
        ry = float(rng.uniform(-0.45, 0.45))
        rth = float(rng.uniform(-np.pi, np.pi))
        robot_pose = np.array([rx, ry, rth], dtype=np.float32)

        side_flip = 1.0 if (rng.uniform() > 0.5) else -1.0
        bearing_1 = side_flip * float(rng.uniform(0.12, 0.28))
        bearing_2 = -side_flip * float(rng.uniform(0.12, 0.28))

        dist_1 = float(rng.uniform(1.05, 1.35))
        dist_2 = float(rng.uniform(1.05, 1.35))

        p1_x = float(rx + dist_1 * np.cos(rth + bearing_1))
        p1_y = float(ry + dist_1 * np.sin(rth + bearing_1))
        p2_x = float(rx + dist_2 * np.cos(rth + bearing_2))
        p2_y = float(ry + dist_2 * np.sin(rth + bearing_2))

        if not (-1.55 <= p1_x <= 1.55 and -1.55 <= p1_y <= 1.55 and -1.55 <= p2_x <= 1.55 and -1.55 <= p2_y <= 1.55):
            rejection_counters["arena_bounds"] = rejection_counters.get("arena_bounds", 0) + 1
            continue

        sep = float(np.linalg.norm([p1_x - p2_x, p1_y - p2_y]))
        if sep < 0.52:
            rejection_counters["target_separation"] = rejection_counters.get("target_separation", 0) + 1
            continue

        pos_1 = np.array([p1_x, p1_y, 0.35], dtype=np.float32)
        pos_2 = np.array([p2_x, p2_y, 0.35], dtype=np.float32)

        d_p1 = np.linalg.norm(pos_1[:2] - robot_pose[:2])
        d_p2 = np.linalg.norm(pos_2[:2] - robot_pose[:2])
        if d_p1 < 0.70 or d_p2 < 0.70:
            rejection_counters["robot_clearance"] = rejection_counters.get("robot_clearance", 0) + 1
            continue

        wp_1 = pos_1[:2] - (pos_1[:2] - robot_pose[:2]) / d_p1 * 0.60
        wp_2 = pos_2[:2] - (pos_2[:2] - robot_pose[:2]) / d_p2 * 0.60

        d_wp1 = float(np.linalg.norm(wp_1 - robot_pose[:2]))
        d_wp2 = float(np.linalg.norm(wp_2 - robot_pose[:2]))
        if d_wp1 < min_initial_dist or d_wp2 < min_initial_dist:
            rejection_counters["waypoint_dist"] = rejection_counters.get("waypoint_dist", 0) + 1
            continue

        obstacles_scene_a = copy.deepcopy(CANONICAL_OBSTACLES)
        obstacles_scene_a[key_a]["pos"] = pos_1
        obstacles_scene_a[key_b]["pos"] = pos_2

        collision_with_other = False
        for k, obst in obstacles_scene_a.items():
            if k not in (key_a, key_b):
                if np.linalg.norm(pos_1[:2] - obst["pos"][:2]) < 0.50 or np.linalg.norm(pos_2[:2] - obst["pos"][:2]) < 0.50:
                    collision_with_other = True
                    break
        if collision_with_other:
            rejection_counters["obstacle_clearance"] = rejection_counters.get("obstacle_clearance", 0) + 1
            continue

        spec_a = {
            "target_key": key_a,
            "semantic_relation": "approach",
            "task_id": encode_task_id(key_a, "approach"),
            "initial_pose": robot_pose,
            "waypoints": [wp_1],
            "obstacles": obstacles_scene_a,
            "distractor_key": key_b,
            "light_diffuse_scale": light_scale,
        }

        env.restore_physical_baseline()
        env.apply_task_spec(spec_a)
        obs_a, _ = env.reset(options={"initial_pos": (rx, ry, rth)})

        vis_a_scene_a = check_target_visibility(env, key_a, min_visible_pixels=min_pixels)
        vis_b_scene_a = check_target_visibility(env, key_b, min_visible_pixels=min_pixels)

        if not (vis_a_scene_a["is_visible"] and vis_b_scene_a["is_visible"]):
            rejection_counters["scene_a_visibility"] = rejection_counters.get("scene_a_visibility", 0) + 1
            continue

        obstacles_scene_b = copy.deepcopy(CANONICAL_OBSTACLES)
        obstacles_scene_b[key_a]["pos"] = pos_2
        obstacles_scene_b[key_b]["pos"] = pos_1

        spec_b = {
            "target_key": key_a,
            "semantic_relation": "approach",
            "task_id": encode_task_id(key_a, "approach"),
            "initial_pose": robot_pose,
            "waypoints": [wp_2],
            "obstacles": obstacles_scene_b,
            "distractor_key": key_b,
            "light_diffuse_scale": light_scale,
        }

        env.restore_physical_baseline()
        env.apply_task_spec(spec_b)
        obs_b, _ = env.reset(options={"initial_pos": (rx, ry, rth)})

        vis_a_scene_b = check_target_visibility(env, key_a, min_visible_pixels=min_pixels)
        vis_b_scene_b = check_target_visibility(env, key_b, min_visible_pixels=min_pixels)

        if not (vis_a_scene_b["is_visible"] and vis_b_scene_b["is_visible"]):
            rejection_counters["scene_b_visibility"] = rejection_counters.get("scene_b_visibility", 0) + 1
            continue

        obs_a_hash = hashlib.sha256(obs_a.tobytes()).hexdigest()
        obs_b_hash = hashlib.sha256(obs_b.tobytes()).hexdigest()

        return {
            "pair_idx": pair_idx,
            "pair_id": f"PAIR_{pair_idx:03d}",
            "context": "nominal" if light_scale >= 0.99 else "illumination_shift",
            "light_diffuse_scale": light_scale,
            "key_a": key_a,
            "key_b": key_b,
            "task_id_a": encode_task_id(key_a, "approach"),
            "task_id_b": encode_task_id(key_b, "approach"),
            "robot_pose": [float(v) for v in robot_pose],
            "pos_1": [float(v) for v in pos_1],
            "pos_2": [float(v) for v in pos_2],
            "wp_1": [float(v) for v in wp_1],
            "wp_2": [float(v) for v in wp_2],
            "vis_scene_a": {"vis_a": vis_a_scene_a, "vis_b": vis_b_scene_a},
            "vis_scene_b": {"vis_a": vis_a_scene_b, "vis_b": vis_b_scene_b},
            "obs_a_hash": obs_a_hash,
            "obs_b_hash": obs_b_hash,
            "obs_a": obs_a,
            "obs_b": obs_b,
        }

    return None


def compute_file_sha256(filepath: str | Path) -> str:
    p = Path(filepath)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=REPO_ROOT
        ).decode("ascii").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, cwd=REPO_ROOT
        ).decode("ascii").strip()
        is_clean = len(status) == 0
        return commit, is_clean
    except Exception:
        return "unknown", False


def wrap_angle(theta: float | np.ndarray) -> float | np.ndarray:
    """Wraps angle into [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def paired_bootstrap_ci(
    data: list[float] | np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> list[float]:
    arr = np.array(data, dtype=np.float64)
    if len(arr) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    boot_means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    low = float(np.percentile(boot_means, 100 * (alpha / 2)))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return [low, high]


def sign_flip_permutation_test(
    diffs: np.ndarray,
    n_perm: int = 100000,
    seed: int = 42,
) -> float:
    diffs_arr = np.array(diffs, dtype=np.float64)
    n = len(diffs_arr)
    if n == 0 or np.all(diffs_arr == 0):
        return 1.0
    obs_stat = float(np.mean(diffs_arr))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_stats = np.mean(signs * diffs_arr, axis=1)
    p_val = float((1 + np.sum(perm_stats >= obs_stat)) / (n_perm + 1))
    return p_val


def holm_bonferroni_correction(p_values: list[float] | np.ndarray) -> list[float]:
    p = np.array(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        return []
    sorted_indices = np.argsort(p)
    sorted_p = p[sorted_indices]
    adj_p_sorted = np.zeros(m, dtype=np.float64)

    for i in range(m):
        rank = i + 1
        adj_p_sorted[i] = min(1.0, (m - rank + 1) * sorted_p[i])

    for i in range(1, m):
        adj_p_sorted[i] = max(adj_p_sorted[i], adj_p_sorted[i - 1])

    adjusted_p = np.zeros(m, dtype=np.float64)
    adjusted_p[sorted_indices] = adj_p_sorted
    return [float(v) for v in adjusted_p]


def compute_hypergeometric_pmf(k: int, o: int, D: int = 64) -> float:
    """Hypergeometric probability P(O = o) for overlap between two uniform k-subsets from D units."""
    if o < max(0, 2 * k - D) or o > min(k, k):
        return 0.0
    num = math.comb(k, o) * math.comb(D - k, k - o)
    den = math.comb(D, k)
    return float(num / den)


def sample_norm_matched_controls_strict(
    delta_h: np.ndarray,
    cand_units: list[int] | np.ndarray,
    k: int,
    norm_tolerance: float = 0.05,
    required_controls: int = 100,
    initial_draws: int = 5000,
    chunk_draws: int = 20000,
    max_draws: int = 200000,
    rng: np.random.Generator | None = None,
    overlap_constraint: int | None = None,
    disjoint_only: bool = False,
    pool_units: list[int] | np.ndarray | None = None,
) -> dict:
    """
    Strict activation-norm-matching sampler.
    Returns:
      {
        "status": "VALID_COMMON_SUPPORT" | "INSUFFICIENT_COMMON_SUPPORT",
        "total_drawn": int,
        "n_valid": int,
        "selected_subsets": list[list[int]] | None,
        "selected_rel_errors": list[float] | None,
        "selected_norms": list[float] | None,
        "selected_overlaps": list[int] | None,
        "min_rel_error": float,
        "median_rel_error": float,
        "max_rel_error": float,
      }
    """
    if rng is None:
        rng = np.random.default_rng(42)

    cand_set = set(cand_units)
    cand_arr = np.array(cand_units, dtype=np.int64)
    cand_norm = float(np.linalg.norm(delta_h[cand_arr]))

    D = len(delta_h)
    comp_units = np.array([u for u in range(D) if u not in cand_set], dtype=np.int64)

    # Special case: disjoint k=32 in D=64 has exactly 1 subset (C(32, 32) = 1)
    if disjoint_only and k == 32:
        sub = comp_units
        norm_sub = float(np.linalg.norm(delta_h[sub]))
        rel_err = abs(norm_sub - cand_norm) / max(1e-8, cand_norm)
        is_val = rel_err <= norm_tolerance
        return {
            "status": "DETERMINISTIC_REFERENCE",
            "total_drawn": 1,
            "n_valid": 1 if is_val else 0,
            "selected_subsets": [sub.tolist()],
            "selected_rel_errors": [float(rel_err)],
            "selected_norms": [float(norm_sub)],
            "selected_overlaps": [0],
            "min_rel_error": float(rel_err),
            "median_rel_error": float(rel_err),
            "max_rel_error": float(rel_err),
        }

    total_drawn = 0
    valid_subsets = []
    valid_rel_errors = []
    valid_norms = []
    valid_overlaps = []
    all_rel_errors = []

    cur_chunk = initial_draws

    while total_drawn < max_draws:
        chunk_size = min(cur_chunk, max_draws - total_drawn)
        if chunk_size <= 0:
            break

        # Generate chunk of subsets
        if disjoint_only:
            # Sample strictly from complement
            if len(comp_units) < k:
                break
            batch_subsets = np.zeros((chunk_size, k), dtype=np.int64)
            # argpartition sampling from comp_units
            rand_matrix = rng.random((chunk_size, len(comp_units)))
            idx_in_comp = rand_matrix.argpartition(k, axis=1)[:, :k]
            batch_subsets = comp_units[idx_in_comp]
        elif overlap_constraint is not None:
            # Exact overlap o from cand_arr, k - o from comp_units
            o = overlap_constraint
            n_comp = k - o
            if o > len(cand_arr) or n_comp > len(comp_units) or o < 0 or n_comp < 0:
                break
            # Sample o from cand_arr
            rand_cand = rng.random((chunk_size, len(cand_arr)))
            idx_cand = rand_cand.argpartition(o, axis=1)[:, :o] if o > 0 else np.empty((chunk_size, 0), dtype=np.int64)
            sub_cand = cand_arr[idx_cand] if o > 0 else np.empty((chunk_size, 0), dtype=np.int64)

            # Sample n_comp from comp_units
            rand_comp = rng.random((chunk_size, len(comp_units)))
            idx_comp = rand_comp.argpartition(n_comp, axis=1)[:, :n_comp] if n_comp > 0 else np.empty((chunk_size, 0), dtype=np.int64)
            sub_comp = comp_units[idx_comp] if n_comp > 0 else np.empty((chunk_size, 0), dtype=np.int64)

            if o > 0 and n_comp > 0:
                batch_subsets = np.hstack([sub_cand, sub_comp])
            elif o > 0:
                batch_subsets = sub_cand
            else:
                batch_subsets = sub_comp
        else:
            # Uniform random sampling from all D units
            rand_matrix = rng.random((chunk_size, D))
            batch_subsets = rand_matrix.argpartition(k, axis=1)[:, :k]

        # Vectorized norm calculation
        norms = np.linalg.norm(delta_h[batch_subsets], axis=1)
        rel_errs = np.abs(norms - cand_norm) / max(1e-8, cand_norm)

        # Overlaps
        if overlap_constraint is not None:
            overlaps = np.full(chunk_size, overlap_constraint, dtype=np.int64)
        elif disjoint_only:
            overlaps = np.zeros(chunk_size, dtype=np.int64)
        else:
            # Overlap with cand_set: sum boolean membership
            in_cand = np.isin(batch_subsets, cand_arr)
            overlaps = np.sum(in_cand, axis=1)

        # Record all rel_errors (subsample for memory if needed)
        if len(all_rel_errors) < 50000:
            all_rel_errors.extend(rel_errs[: min(chunk_size, 50000 - len(all_rel_errors))])

        valid_mask = rel_errs <= norm_tolerance
        valid_indices = np.where(valid_mask)[0]

        for vi in valid_indices:
            valid_subsets.append(batch_subsets[vi].tolist())
            valid_rel_errors.append(float(rel_errs[vi]))
            valid_norms.append(float(norms[vi]))
            valid_overlaps.append(int(overlaps[vi]))

        total_drawn += chunk_size
        if len(valid_subsets) >= required_controls:
            break

        cur_chunk = chunk_draws

    n_valid = len(valid_subsets)
    all_err_arr = np.array(all_rel_errors if len(all_rel_errors) > 0 else [1.0])
    min_err = float(np.min(all_err_arr))
    med_err = float(np.median(all_err_arr))
    max_err = float(np.max(all_err_arr))

    if n_valid >= required_controls:
        # Sort valid subsets by relative error and retain required_controls closest
        sort_order = np.argsort(valid_rel_errors)
        retained_idx = sort_order[:required_controls]
        return {
            "status": "VALID_COMMON_SUPPORT",
            "total_drawn": total_drawn,
            "n_valid": n_valid,
            "selected_subsets": [valid_subsets[i] for i in retained_idx],
            "selected_rel_errors": [valid_rel_errors[i] for i in retained_idx],
            "selected_norms": [valid_norms[i] for i in retained_idx],
            "selected_overlaps": [valid_overlaps[i] for i in retained_idx],
            "min_rel_error": min_err,
            "median_rel_error": med_err,
            "max_rel_error": max_err,
        }
    else:
        # STRICT REQUIREMENT: Never replace with subsets outside tolerance!
        return {
            "status": "INSUFFICIENT_COMMON_SUPPORT",
            "total_drawn": total_drawn,
            "n_valid": n_valid,
            "selected_subsets": None,
            "selected_rel_errors": None,
            "selected_norms": None,
            "selected_overlaps": None,
            "min_rel_error": min_err,
            "median_rel_error": med_err,
            "max_rel_error": max_err,
        }


def extract_9_stages(model: torch.nn.Module, z: torch.Tensor, t_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Extracts intermediate representations from H-JEPA grounding head."""
    e = model.task_embedding(t_tensor)
    x_in = torch.cat([z, e], dim=-1)

    fc1 = model.grounding_head[0]
    relu1 = model.grounding_head[1]
    fc2 = model.grounding_head[2]
    relu2 = model.grounding_head[3]
    fc3 = model.grounding_head[4]

    l1_pre = fc1(x_in)
    l1_post = relu1(l1_pre)
    l2_pre = fc2(l1_post)
    l2_post = relu2(l2_pre)
    raw = fc3(l2_post)

    xy = raw[:, :2]
    sin_th = raw[:, 2:3]
    cos_th = raw[:, 3:4]
    norm_th = torch.sqrt(sin_th**2 + cos_th**2 + 1e-8)
    sin_norm = sin_th / norm_th
    cos_norm = cos_th / norm_th
    postprocessed_goal = torch.cat([xy, sin_norm, cos_norm], dim=-1)

    return {
        "visual_latent_z_32d": z,
        "task_embedding_e_16d": e,
        "concatenated_input_48d": x_in,
        "layer1_pre_relu_64d": l1_pre,
        "layer1_post_relu_64d": l1_post,
        "layer2_pre_relu_64d": l2_pre,
        "layer2_post_relu_64d": l2_post,
        "raw_goal_output_4d": raw,
        "postprocessed_goal_4d": postprocessed_goal,
    }


def forward_grounding_from_layer(
    model: torch.nn.Module,
    h_layer: torch.Tensor,
    start_layer: int,
) -> torch.Tensor:
    """Forward passes activations from intermediate grounding head layers."""
    fc2 = model.grounding_head[2]
    relu2 = model.grounding_head[3]
    fc3 = model.grounding_head[4]

    if start_layer == 1:
        x = fc2(h_layer)
        x = relu2(x)
        raw = fc3(x)
    elif start_layer == 2:
        raw = fc3(h_layer)
    else:
        raise ValueError(f"Unsupported start_layer: {start_layer}")

    xy = raw[:, :2]
    sin_th = raw[:, 2:3]
    cos_th = raw[:, 3:4]
    norm_th = torch.sqrt(sin_th**2 + cos_th**2 + 1e-8)
    sin_norm = sin_th / norm_th
    cos_norm = cos_th / norm_th
    return torch.cat([xy, sin_norm, cos_norm], dim=-1)


def compute_spectral_analysis(A: np.ndarray) -> dict:
    N, D = A.shape
    if N < 2:
        return {"participation_ratio": 1.0, "dims_80_var": 1, "dims_90_var": 1, "dims_95_var": 1, "dims_99_var": 1, "top_10_eigenvals": []}
    A_c = A - np.mean(A, axis=0, keepdims=True)
    C = (A_c.T @ A_c) / (N - 1)
    evals = np.linalg.eigvalsh(C)
    evals = np.sort(evals)[::-1]
    evals = np.maximum(evals, 0.0)

    sum_evals = float(np.sum(evals))
    if sum_evals < 1e-12:
        pr = 1.0
        exp_var = np.zeros(D)
    else:
        sum_sq = float(np.sum(evals**2))
        pr = (sum_evals**2) / (sum_sq + 1e-12)
        exp_var = np.cumsum(evals) / sum_evals

    def threshold_dim(thresh: float) -> int:
        idx = np.where(exp_var >= thresh)[0]
        return int(idx[0] + 1) if len(idx) > 0 else D

    return {
        "participation_ratio": float(pr),
        "dims_80_var": threshold_dim(0.80),
        "dims_90_var": threshold_dim(0.90),
        "dims_95_var": threshold_dim(0.95),
        "dims_99_var": threshold_dim(0.99),
        "top_10_eigenvals": [float(v) for v in evals[:10]],
    }


def compute_representation_statistics_v5_3(
    act_dict: dict[str, np.ndarray],
    stage_name: str,
) -> dict:
    A = act_dict[stage_name]
    N, D = A.shape

    spectral = compute_spectral_analysis(A)
    min_val = float(np.min(A))
    is_non_negative = min_val >= -1e-6
    col_var = np.var(A, axis=0)
    silent_mask = col_var < 1e-6
    silent_count = int(np.sum(silent_mask))

    stats_out = {
        "dimensions": D,
        "num_samples": N,
        "is_non_negative": is_non_negative,
        "silent_units_count": silent_count,
        "silent_units_fraction": float(silent_count / D),
        "participation_ratio": spectral["participation_ratio"],
        "dims_80_var": spectral["dims_80_var"],
        "dims_90_var": spectral["dims_90_var"],
        "dims_95_var": spectral["dims_95_var"],
        "dims_99_var": spectral["dims_99_var"],
        "top_10_eigenvals": spectral["top_10_eigenvals"],
    }

    if is_non_negative:
        mean_sq_col = np.mean(A, axis=0) ** 2
        sq_mean_col = np.mean(A**2, axis=0) + 1e-12
        lifetime_sp = (1.0 - (mean_sq_col / sq_mean_col)) / (1.0 - (1.0 / max(2, N)))

        mean_sq_row = np.mean(A, axis=1) ** 2
        sq_mean_row = np.mean(A**2, axis=1) + 1e-12
        pop_sp = (1.0 - (mean_sq_row / sq_mean_row)) / (1.0 - (1.0 / max(2, D)))

        stats_out["lifetime_sparsity"] = {
            "mean": float(np.mean(lifetime_sp)),
            "std": float(np.std(lifetime_sp)),
            "median": float(np.median(lifetime_sp)),
        }
        stats_out["population_sparsity"] = {
            "mean": float(np.mean(pop_sp)),
            "std": float(np.std(pop_sp)),
            "median": float(np.median(pop_sp)),
        }
        stats_out["zero_fraction"] = float(np.mean(A < 1e-6))
        stats_out["excess_kurtosis"] = None
    else:
        A_pos = np.maximum(0.0, A)
        mean_sq_c = np.mean(A_pos, axis=0) ** 2
        sq_mean_c = np.mean(A_pos**2, axis=0) + 1e-12
        pos_lt = (1.0 - (mean_sq_c / sq_mean_c)) / (1.0 - (1.0 / max(2, N)))
        stats_out["positive_part_lifetime_sparsity"] = {
            "mean": float(np.mean(pos_lt)),
            "std": float(np.std(pos_lt)),
        }
        active_cols = ~silent_mask
        if np.any(active_cols):
            A_active = A[:, active_cols]
            kurt = stats.kurtosis(A_active, axis=0, fisher=True)
            stats_out["excess_kurtosis"] = {
                "mean": float(np.mean(kurt)),
                "std": float(np.std(kurt)),
                "median": float(np.median(kurt)),
            }
        else:
            stats_out["excess_kurtosis"] = {"mean": 0.0, "std": 0.0, "median": 0.0}
        stats_out["negative_fraction"] = float(np.mean(A < -1e-6))
        stats_out["zero_fraction"] = float(np.mean(np.abs(A) < 1e-6))

    return stats_out


def run_task_conditioned_circuits_v5_3(
    checkpoint_path: str | Path = "world_models_fpga/checkpoints_v4/hjepa_best.pt",
    specs_path: str | Path = "world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_2_scene_specs.jsonl",
    results_dir: str | Path = "world_models_fpga/research/circuits/results",
    device: str = "cuda",
    norm_tolerance: float = 0.05,
    required_controls: int = 100,
    initial_draws: int = 5000,
    chunk_draws: int = 20000,
    max_draws: int = 200000,
    random_seed: int = 2026,
    k_values: list[int] | None = None,
    quick_test_pairs: int | None = None,
) -> dict:
    start_time = time.time()
    if k_values is None:
        k_values = [2, 4, 8, 16, 32]

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    git_commit, git_clean = get_git_info()
    print("=" * 80)
    print("PROJECT HEPHAESTUS: PHASE 5.3 VALID NORM-MATCHED CAUSAL MEDIATION BENCHMARK")
    print("=" * 80)
    print(f"Git Commit: {git_commit} (Clean: {git_clean})")
    print(f"Device: {device} | Random Seed: {random_seed} | Tolerance: {norm_tolerance*100:.1f}%")
    print(f"Required Controls: {required_controls} | Initial Draws: {initial_draws} | Max Draws: {max_draws}")
    print(f"k Values: {k_values}")
    print(f"Hardware: {platform.processor()} | OS: {platform.system()} {platform.release()}")

    # 1. Load Checkpoint and Agent
    print(f"\n[1/6] Loading H-JEPA Agent from {checkpoint_path}...")
    agent = HJEPAAgent(str(checkpoint_path), device=device)
    model = agent.model
    model.eval()
    ckpt_sha256 = compute_file_sha256(checkpoint_path)
    print(f"Loaded H-JEPA checkpoint (SHA-256: {ckpt_sha256[:16]}...)")

    # 2. Candidate Unit Rankings (Frozen Discovery Split from Phase 5.2)
    # Layer 1 and Layer 2 both-contexts rankings
    l1_ranking = [
        40, 39, 58, 36, 54, 25, 27, 57, 2, 50, 46, 6, 4, 7, 56, 18,
        32, 12, 42, 33, 1, 52, 8, 63, 15, 3, 26, 60, 44, 21, 17, 62,
        51, 43, 30, 22, 16, 41, 49, 48, 35, 10, 0, 23, 61, 53, 11, 13,
        34, 38, 19, 9, 47, 45, 5, 24, 20, 37, 55, 59, 28, 31, 29, 14
    ]
    l2_ranking = [
        13, 61, 4, 14, 50, 8, 56, 19, 1, 31, 25, 60, 62, 21, 15, 55,
        57, 29, 34, 24, 18, 9, 35, 3, 40, 16, 28, 22, 51, 54, 36, 27,
        38, 10, 26, 37, 0, 2, 7, 39, 52, 48, 47, 43, 58, 23, 63, 59,
        45, 6, 41, 46, 11, 53, 49, 42, 32, 33, 44, 30, 17, 20, 12, 5
    ]
    frozen_rankings = {
        "layer1_both_contexts_ranking": l1_ranking,
        "layer2_both_contexts_ranking": l2_ranking,
    }
    ranking_hash = hashlib.sha256(json.dumps(frozen_rankings, sort_keys=True).encode()).hexdigest()
    print(f"Frozen candidate unit rankings loaded (Hash: {ranking_hash[:16]}...)")

    # 3. Load Phase 5.2 Scene Specifications and Regenerate Observations
    print(f"\n[2/6] Loading Phase 5.2 Scene Specifications from {specs_path}...")
    specs_sha256 = compute_file_sha256(specs_path)
    with open(specs_path) as f:
        all_specs = [json.loads(line) for line in f]

    if quick_test_pairs is not None:
        all_specs = all_specs[:quick_test_pairs]

    print(f"Loaded {len(all_specs)} scene pairs. Regenerating observations & verifying bit-for-bit SHA-256 hashes...")
    env = JetAutoEnv()
    rng_scenes = np.random.default_rng(random_seed)
    rejection_counters = {}
    verified_specs = []
    hash_mismatches = 0

    n_nom = min(30, len(all_specs))
    n_illum = max(0, len(all_specs) - 30)

    # 1. Nominal scenes
    for p_idx in range(n_nom):
        sp = generate_dual_visible_scene_pair(
            pair_idx=p_idx,
            env=env,
            rng=rng_scenes,
            rejection_counters=rejection_counters,
            light_scale=1.0,
            min_pixels=100,
            min_initial_dist=0.60,
        )
        if sp["obs_a_hash"] != all_specs[p_idx]["obs_a_hash"] or sp["obs_b_hash"] != all_specs[p_idx]["obs_b_hash"]:
            hash_mismatches += 1
            print(f"WARNING: Hash mismatch at pair {sp['pair_id']}! A: {sp['obs_a_hash'] == all_specs[p_idx]['obs_a_hash']}, B: {sp['obs_b_hash'] == all_specs[p_idx]['obs_b_hash']}")
        verified_specs.append(sp)

    # 2. Illumination shifted scenes
    for p_idx in range(30, 30 + n_illum):
        sp = generate_dual_visible_scene_pair(
            pair_idx=p_idx,
            env=env,
            rng=rng_scenes,
            rejection_counters=rejection_counters,
            light_scale=0.60,
            min_pixels=100,
            min_initial_dist=0.60,
        )
        if sp["obs_a_hash"] != all_specs[p_idx]["obs_a_hash"] or sp["obs_b_hash"] != all_specs[p_idx]["obs_b_hash"]:
            hash_mismatches += 1
            print(f"WARNING: Hash mismatch at pair {sp['pair_id']}! A: {sp['obs_a_hash'] == all_specs[p_idx]['obs_a_hash']}, B: {sp['obs_b_hash'] == all_specs[p_idx]['obs_b_hash']}")
        verified_specs.append(sp)

    env.close()
    print(f"Observation regeneration complete. 120/120 scenes verified (Hash mismatches: {hash_mismatches}).")
    assert hash_mismatches == 0, "Regenerated observation hashes did not match Phase 5.2 specs!"

    # 4. Extract Internal Representations Across All 9 Stages & Save Activations JSONL
    print(f"\n[3/6] Forward passing models and extracting internal representations...")
    stage_names = [
        "visual_latent_z_32d",
        "task_embedding_e_16d",
        "concatenated_input_48d",
        "layer1_pre_relu_64d",
        "layer1_post_relu_64d",
        "layer2_pre_relu_64d",
        "layer2_post_relu_64d",
        "raw_goal_output_4d",
        "postprocessed_goal_4d",
    ]
    eval_stage_arrays = {st: [] for st in stage_names}
    processed_pairs = []

    activations_path = out_dir / "task_conditioned_circuits_v5_3_activations.jsonl"
    with open(activations_path, "w") as f_act:
        for p_idx, sp in enumerate(verified_specs):
            key_a = sp["key_a"]
            key_b = sp["key_b"]
            t_id_a = sp["task_id_a"]
            t_id_b = sp["task_id_b"]

            t_tensor_a = torch.tensor([t_id_a], dtype=torch.long, device=device)
            t_tensor_b = torch.tensor([t_id_b], dtype=torch.long, device=device)

            with torch.no_grad():
                obs_a_t = torch.tensor(sp["obs_a"], dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
                z_a = model.encode(obs_a_t)

                obs_b_t = torch.tensor(sp["obs_b"], dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
                z_b = model.encode(obs_b_t)

                cond_aa = extract_9_stages(model, z_a, t_tensor_a)
                cond_ab = extract_9_stages(model, z_a, t_tensor_b)
                cond_ba = extract_9_stages(model, z_b, t_tensor_a)
                cond_bb = extract_9_stages(model, z_b, t_tensor_b)

            # Collect for representation statistics
            for c in [cond_aa, cond_ab, cond_ba, cond_bb]:
                for st in stage_names:
                    eval_stage_arrays[st].append(c[st].squeeze(0).cpu().numpy())

            # Ground truth body goals
            gt_a_scene_a = compute_body_frame_goal_vec(sp["robot_pose"], sp["wp_1"])
            gt_b_scene_a = compute_body_frame_goal_vec(sp["robot_pose"], sp["wp_2"])
            gt_a_scene_b = compute_body_frame_goal_vec(sp["robot_pose"], sp["wp_2"])
            gt_b_scene_b = compute_body_frame_goal_vec(sp["robot_pose"], sp["wp_1"])

            pair_entry = {
                "pair_idx": p_idx,
                "pair_id": sp["pair_id"],
                "context": sp["context"],
                "light_diffuse_scale": sp["light_diffuse_scale"],
                "key_a": key_a,
                "key_b": key_b,
                "task_id_a": t_id_a,
                "task_id_b": t_id_b,
                "robot_pose": sp["robot_pose"],
                "pos_1": sp["pos_1"],
                "pos_2": sp["pos_2"],
                "wp_1": sp["wp_1"],
                "wp_2": sp["wp_2"],
                "obs_a_hash": sp["obs_a_hash"],
                "obs_b_hash": sp["obs_b_hash"],
                "gt_goals": {
                    "gt_a_scene_a": [float(v) for v in gt_a_scene_a],
                    "gt_b_scene_a": [float(v) for v in gt_b_scene_a],
                    "gt_a_scene_b": [float(v) for v in gt_a_scene_b],
                    "gt_b_scene_b": [float(v) for v in gt_b_scene_b],
                },
                "model_outputs": {
                    "g_aa": [float(v) for v in cond_aa["postprocessed_goal_4d"].squeeze(0).cpu().numpy()],
                    "g_ab": [float(v) for v in cond_ab["postprocessed_goal_4d"].squeeze(0).cpu().numpy()],
                    "g_ba": [float(v) for v in cond_ba["postprocessed_goal_4d"].squeeze(0).cpu().numpy()],
                    "g_bb": [float(v) for v in cond_bb["postprocessed_goal_4d"].squeeze(0).cpu().numpy()],
                },
                "activations": {
                    "cond_aa": {st: cond_aa[st] for st in ["layer1_post_relu_64d", "layer2_post_relu_64d"]},
                    "cond_ab": {st: cond_ab[st] for st in ["layer1_post_relu_64d", "layer2_post_relu_64d"]},
                    "cond_ba": {st: cond_ba[st] for st in ["layer1_post_relu_64d", "layer2_post_relu_64d"]},
                    "cond_bb": {st: cond_bb[st] for st in ["layer1_post_relu_64d", "layer2_post_relu_64d"]},
                },
            }
            processed_pairs.append(pair_entry)

            # Write activation records for Scene A and Scene B
            f_act.write(json.dumps({
                "pair_id": sp["pair_id"],
                "scene_id": "scene_a",
                "context": sp["context"],
                "checkpoint_sha256": ckpt_sha256,
                "ranking_version": ranking_hash,
                "l1_act_task_a": [float(v) for v in cond_aa["layer1_post_relu_64d"].squeeze(0).cpu().numpy()],
                "l1_act_task_b": [float(v) for v in cond_ab["layer1_post_relu_64d"].squeeze(0).cpu().numpy()],
                "l2_act_task_a": [float(v) for v in cond_aa["layer2_post_relu_64d"].squeeze(0).cpu().numpy()],
                "l2_act_task_b": [float(v) for v in cond_ab["layer2_post_relu_64d"].squeeze(0).cpu().numpy()],
                "output_task_a": [float(v) for v in cond_aa["postprocessed_goal_4d"].squeeze(0).cpu().numpy()],
                "output_task_b": [float(v) for v in cond_ab["postprocessed_goal_4d"].squeeze(0).cpu().numpy()],
                "gt_goal_a": [float(v) for v in gt_a_scene_a],
                "gt_goal_b": [float(v) for v in gt_b_scene_a],
            }) + "\n")

            f_act.write(json.dumps({
                "pair_id": sp["pair_id"],
                "scene_id": "scene_b",
                "context": sp["context"],
                "checkpoint_sha256": ckpt_sha256,
                "ranking_version": ranking_hash,
                "l1_act_task_a": [float(v) for v in cond_ba["layer1_post_relu_64d"].squeeze(0).cpu().numpy()],
                "l1_act_task_b": [float(v) for v in cond_bb["layer1_post_relu_64d"].squeeze(0).cpu().numpy()],
                "l2_act_task_a": [float(v) for v in cond_ba["layer2_post_relu_64d"].squeeze(0).cpu().numpy()],
                "l2_act_task_b": [float(v) for v in cond_bb["layer2_post_relu_64d"].squeeze(0).cpu().numpy()],
                "output_task_a": [float(v) for v in cond_ba["postprocessed_goal_4d"].squeeze(0).cpu().numpy()],
                "output_task_b": [float(v) for v in cond_bb["postprocessed_goal_4d"].squeeze(0).cpu().numpy()],
                "gt_goal_a": [float(v) for v in gt_a_scene_b],
                "gt_goal_b": [float(v) for v in gt_b_scene_b],
            }) + "\n")

    print(f"Saved raw activations to {activations_path}")

    # Compute Representation Statistics
    print("\nComputing comprehensive representation statistics across all 9 stages...")
    eval_stage_numpy = {st: np.array(eval_stage_arrays[st], dtype=np.float32) for st in stage_names}
    stage_representation_stats = {}
    for st in stage_names:
        stats_st = compute_representation_statistics_v5_3(eval_stage_numpy, st)
        stage_representation_stats[st] = stats_st
        print(f"  Stage {st:25s}: PR = {stats_st['participation_ratio']:.2f}/{stats_st['dimensions']} | 80% Dims = {stats_st['dims_80_var']} | Silent = {stats_st['silent_units_count']}")

    # 5. Causal Interventions: Necessity & Sufficiency with Valid Norm-Matched Controls
    print("\n[4/6] Executing Causal Interventions (Sufficiency & Matched Necessity)...")
    layer_configs = [
        {"layer_key": "layer1_post_relu_64d", "start_layer": 1, "ranked_units": l1_ranking},
        {"layer_key": "layer2_post_relu_64d", "start_layer": 2, "ranked_units": l2_ranking},
    ]

    interventions_gz_path = out_dir / "task_conditioned_circuits_v5_3_interventions.jsonl.gz"
    control_audit_path = out_dir / "task_conditioned_circuits_v5_3_control_audit.jsonl"
    pairs_path = out_dir / "task_conditioned_circuits_v5_3_pairs.jsonl"

    f_gz = gzip.open(interventions_gz_path, "wt", encoding="utf-8")
    f_audit = open(control_audit_path, "w")
    f_pairs = open(pairs_path, "w")

    pair_summary_records = []
    total_interventions_recorded = 0

    master_rng = np.random.default_rng(random_seed)

    for p in processed_pairs:
        pair_id = p["pair_id"]
        pair_idx = p["pair_idx"]
        context = p["context"]
        acts = p["activations"]

        pair_metrics = {
            "pair_id": pair_id,
            "pair_idx": pair_idx,
            "context": context,
            "results": {},
        }

        scenes_data = [
            ("scene_a", 0, acts["cond_aa"], acts["cond_ab"]),
            ("scene_b", 1, acts["cond_ba"], acts["cond_bb"]),
        ]

        for scene_id, scene_num, act_a, act_b in scenes_data:
            for l_num, l_cfg in enumerate(layer_configs):
                l_key = l_cfg["layer_key"]
                start_l = l_cfg["start_layer"]
                ranked_u = l_cfg["ranked_units"]

                h_a = act_a[l_key]  # [1, 64]
                h_b = act_b[l_key]  # [1, 64]

                with torch.no_grad():
                    g_a = forward_grounding_from_layer(model, h_a, start_l).squeeze(0)  # [4]
                    g_b = forward_grounding_from_layer(model, h_b, start_l).squeeze(0)  # [4]

                g_a_xy = g_a[:2]
                g_b_xy = g_b[:2]
                delta_g_full_xy = g_b_xy - g_a_xy
                full_contrast_norm = float(torch.norm(delta_g_full_xy).item())

                delta_h = (h_b - h_a).squeeze(0).cpu().numpy()  # [64]

                theta_a = float(torch.atan2(g_a[2], g_a[3]).item())
                theta_b = float(torch.atan2(g_b[2], g_b[3]).item())
                delta_theta_full = wrap_angle(theta_b - theta_a)

                bearing_a = float(torch.atan2(g_a_xy[1], g_a_xy[0]).item())
                bearing_b = float(torch.atan2(g_b_xy[1], g_b_xy[0]).item())

                for k in k_values:
                    cand_k = ranked_u[:k]
                    cand_norm = float(np.linalg.norm(delta_h[cand_k]))

                    # Setting-specific deterministic seed
                    setting_seed = int((random_seed * 1000003 + pair_idx * 10007 + scene_num * 1009 + l_num * 101 + k * 17) & 0xFFFFFFFF)
                    rng_setting = np.random.default_rng(setting_seed)

                    # A. Strict Norm-Matched Random Controls (Unstratified general pool)
                    norm_matched_res = sample_norm_matched_controls_strict(
                        delta_h=delta_h,
                        cand_units=cand_k,
                        k=k,
                        norm_tolerance=norm_tolerance,
                        required_controls=required_controls,
                        initial_draws=initial_draws,
                        chunk_draws=chunk_draws,
                        max_draws=max_draws,
                        rng=rng_setting,
                    )

                    # Audit record for strict norm-matching
                    f_audit.write(json.dumps({
                        "pair_id": pair_id,
                        "scene_id": scene_id,
                        "layer": l_key,
                        "k": k,
                        "stratum": "general_norm_matched",
                        "random_seed": setting_seed,
                        "attempted_draws": norm_matched_res["total_drawn"],
                        "valid_norm_matches": norm_matched_res["n_valid"],
                        "retained_count": len(norm_matched_res["selected_subsets"]) if norm_matched_res["selected_subsets"] else 0,
                        "min_rel_error": norm_matched_res["min_rel_error"],
                        "median_rel_error": norm_matched_res["median_rel_error"],
                        "max_rel_error": norm_matched_res["max_rel_error"],
                        "status": norm_matched_res["status"],
                        "eligible_confirmatory": norm_matched_res["status"] == "VALID_COMMON_SUPPORT",
                    }) + "\n")

                    # B. Unconstrained Dimension-Matched Random Controls (100 uniform draws)
                    rng_dim = np.random.default_rng(setting_seed + 1)
                    dim_matched_subsets = [rng_dim.choice(64, size=k, replace=False).tolist() for _ in range(required_controls)]
                    dim_norms = [float(np.linalg.norm(delta_h[sub])) for sub in dim_matched_subsets]
                    dim_rel_errors = [float(abs(n - cand_norm) / max(1e-8, cand_norm)) for n in dim_norms]
                    cand_set_k = set(cand_k)
                    dim_overlaps = [int(len(set(sub).intersection(cand_set_k))) for sub in dim_matched_subsets]

                    # C. Overlap Stratified Controls
                    # Identify confirmatory strata with hypergeometric probability >= 0.01
                    stratified_results = {}
                    possible_strata = []
                    for o_val in range(k + 1):
                        p_hyp = compute_hypergeometric_pmf(k, o_val, D=64)
                        if p_hyp >= 0.01:
                            possible_strata.append((o_val, p_hyp))

                    stratified_subsets_by_stratum = {}
                    for o_val, p_hyp in possible_strata:
                        rng_strat = np.random.default_rng(setting_seed + 100 + o_val * 7)
                        res_strat = sample_norm_matched_controls_strict(
                            delta_h=delta_h,
                            cand_units=cand_k,
                            k=k,
                            norm_tolerance=norm_tolerance,
                            required_controls=required_controls,
                            initial_draws=min(initial_draws, 20000),
                            chunk_draws=chunk_draws,
                            max_draws=50000,
                            rng=rng_strat,
                            overlap_constraint=o_val,
                        )
                        f_audit.write(json.dumps({
                            "pair_id": pair_id,
                            "scene_id": scene_id,
                            "layer": l_key,
                            "k": k,
                            "stratum": f"overlap_{o_val}",
                            "hypergeometric_prob": p_hyp,
                            "random_seed": setting_seed + 100 + o_val * 7,
                            "attempted_draws": res_strat["total_drawn"],
                            "valid_norm_matches": res_strat["n_valid"],
                            "retained_count": len(res_strat["selected_subsets"]) if res_strat["selected_subsets"] else 0,
                            "min_rel_error": res_strat["min_rel_error"],
                            "median_rel_error": res_strat["median_rel_error"],
                            "max_rel_error": res_strat["max_rel_error"],
                            "status": res_strat["status"],
                            "eligible_confirmatory": res_strat["status"] == "VALID_COMMON_SUPPORT",
                        }) + "\n")
                        if res_strat["status"] == "VALID_COMMON_SUPPORT":
                            stratified_subsets_by_stratum[o_val] = {
                                "prob": p_hyp,
                                "subsets": res_strat["selected_subsets"],
                                "norms": res_strat["selected_norms"],
                                "errors": res_strat["selected_rel_errors"],
                            }

                    # D. Disjoint Controls (O = 0)
                    rng_disjoint = np.random.default_rng(setting_seed + 999)
                    disjoint_res = sample_norm_matched_controls_strict(
                        delta_h=delta_h,
                        cand_units=cand_k,
                        k=k,
                        norm_tolerance=norm_tolerance,
                        required_controls=min(required_controls, 100 if k < 32 else 1),
                        initial_draws=initial_draws,
                        chunk_draws=chunk_draws,
                        max_draws=50000 if k < 32 else 1,
                        rng=rng_disjoint,
                        disjoint_only=True,
                    )
                    f_audit.write(json.dumps({
                        "pair_id": pair_id,
                        "scene_id": scene_id,
                        "layer": l_key,
                        "k": k,
                        "stratum": "disjoint_o_0",
                        "attempted_draws": disjoint_res["total_drawn"],
                        "valid_norm_matches": disjoint_res["n_valid"],
                        "retained_count": len(disjoint_res["selected_subsets"]) if disjoint_res["selected_subsets"] else 0,
                        "status": disjoint_res["status"],
                        "min_rel_error": disjoint_res["min_rel_error"],
                        "median_rel_error": disjoint_res["median_rel_error"],
                        "max_rel_error": disjoint_res["max_rel_error"],
                        "eligible_confirmatory": disjoint_res["status"] in ["VALID_COMMON_SUPPORT", "DETERMINISTIC_REFERENCE"],
                    }) + "\n")

                    # =========================================================================
                    # 1. Evaluate CAUSAL NECESSITY on Candidate Units
                    # =========================================================================
                    # Condition-averaged ablation: h_i -> (h_i^A + h_i^B) / 2
                    h_a_mabl = h_a.clone()
                    h_b_mabl = h_b.clone()
                    mean_val = 0.5 * (h_a[:, cand_k] + h_b[:, cand_k])
                    h_a_mabl[:, cand_k] = mean_val
                    h_b_mabl[:, cand_k] = mean_val

                    # Zero ablation (separate sensitivity analysis)
                    h_a_zabl = h_a.clone()
                    h_b_zabl = h_b.clone()
                    h_a_zabl[:, cand_k] = 0.0
                    h_b_zabl[:, cand_k] = 0.0

                    with torch.no_grad():
                        g_a_mabl = forward_grounding_from_layer(model, h_a_mabl, start_l).squeeze(0)
                        g_b_mabl = forward_grounding_from_layer(model, h_b_mabl, start_l).squeeze(0)
                        g_a_zabl = forward_grounding_from_layer(model, h_a_zabl, start_l).squeeze(0)
                        g_b_zabl = forward_grounding_from_layer(model, h_b_zabl, start_l).squeeze(0)

                    delta_g_mabl_xy = g_b_mabl[:2] - g_a_mabl[:2]
                    mabl_contrast_norm = float(torch.norm(delta_g_mabl_xy).item())
                    cand_retention = mabl_contrast_norm / max(1e-8, full_contrast_norm)
                    cand_drop = 1.0 - cand_retention

                    dot_mabl = float(torch.dot(delta_g_mabl_xy, delta_g_full_xy).item())
                    cos_mabl = dot_mabl / max(1e-8, mabl_contrast_norm * full_contrast_norm)

                    bearing_a_mabl = float(torch.atan2(g_a_mabl[1], g_a_mabl[0]).item())
                    bearing_b_mabl = float(torch.atan2(g_b_mabl[1], g_b_mabl[0]).item())
                    cand_delta_bearing = 0.5 * (abs(wrap_angle(bearing_a_mabl - bearing_a)) + abs(wrap_angle(bearing_b_mabl - bearing_b)))

                    theta_a_mabl = float(torch.atan2(g_a_mabl[2], g_a_mabl[3]).item())
                    theta_b_mabl = float(torch.atan2(g_b_mabl[2], g_b_mabl[3]).item())
                    cand_delta_heading = 0.5 * (abs(wrap_angle(theta_a_mabl - theta_a)) + abs(wrap_angle(theta_b_mabl - theta_b)))

                    delta_g_zabl_xy = g_b_zabl[:2] - g_a_zabl[:2]
                    cand_zero_retention = float(torch.norm(delta_g_zabl_xy).item()) / max(1e-8, full_contrast_norm)

                    # =========================================================================
                    # 2. Evaluate CAUSAL NECESSITY on Norm-Matched Random Controls (Batch)
                    # =========================================================================
                    norm_ctrl_drops = None
                    norm_ctrl_retentions = None
                    norm_ctrl_cosines = None
                    if norm_matched_res["status"] == "VALID_COMMON_SUPPORT":
                        n_ctrl = len(norm_matched_res["selected_subsets"])
                        batch_h_a_mabl = h_a.repeat(n_ctrl, 1)
                        batch_h_b_mabl = h_b.repeat(n_ctrl, 1)
                        for d_idx, sub_u in enumerate(norm_matched_res["selected_subsets"]):
                            m_val = 0.5 * (h_a[0, sub_u] + h_b[0, sub_u])
                            batch_h_a_mabl[d_idx, sub_u] = m_val
                            batch_h_b_mabl[d_idx, sub_u] = m_val

                        with torch.no_grad():
                            batch_g_a_mabl = forward_grounding_from_layer(model, batch_h_a_mabl, start_l)
                            batch_g_b_mabl = forward_grounding_from_layer(model, batch_h_b_mabl, start_l)

                        batch_delta_mabl_xy = batch_g_b_mabl[:, :2] - batch_g_a_mabl[:, :2]
                        batch_mabl_norms = torch.norm(batch_delta_mabl_xy, dim=-1)
                        norm_ctrl_retentions = (batch_mabl_norms / max(1e-8, full_contrast_norm)).cpu().numpy()
                        norm_ctrl_drops = (1.0 - norm_ctrl_retentions)
                        batch_dots = torch.sum(batch_delta_mabl_xy * delta_g_full_xy, dim=-1)
                        norm_ctrl_cosines = (batch_dots / (batch_mabl_norms * full_contrast_norm + 1e-8)).cpu().numpy()

                    # =========================================================================
                    # 3. Evaluate CAUSAL SUFFICIENCY in Both Directions (A -> B and B -> A)
                    # =========================================================================
                    for dir_name, h_recip, h_donor, g_recip, g_target, delta_target_xy, th_recip, th_target in [
                        ("A_to_B", h_a, h_b, g_a, g_b, delta_g_full_xy, theta_a, delta_theta_full),
                        ("B_to_A", h_b, h_a, g_b, g_a, -delta_g_full_xy, theta_b, -delta_theta_full),
                    ]:
                        target_norm_sq = float(torch.dot(delta_target_xy, delta_target_xy).item()) + 1e-12
                        target_norm = float(torch.norm(delta_target_xy).item()) + 1e-8

                        # a) Candidate sufficiency pass
                        h_trans = h_recip.clone()
                        h_trans[:, cand_k] = h_donor[:, cand_k]
                        with torch.no_grad():
                            g_trans = forward_grounding_from_layer(model, h_trans, start_l).squeeze(0)

                        delta_trans_xy = g_trans[:2] - g_recip[:2]
                        cand_transfer_mag = float(torch.norm(delta_trans_xy).item())
                        cand_eta = float(torch.dot(delta_trans_xy, delta_target_xy).item()) / target_norm_sq
                        cand_cos = float(torch.dot(delta_trans_xy, delta_target_xy).item()) / (cand_transfer_mag * target_norm + 1e-8)
                        cand_r_perp = float(torch.norm(delta_trans_xy - cand_eta * delta_target_xy).item())

                        th_trans = float(torch.atan2(g_trans[2], g_trans[3]).item())
                        delta_th_trans = wrap_angle(th_trans - th_recip)
                        cand_th_eff = float(delta_th_trans / (th_target + 1e-8)) if abs(th_target) > 1e-3 else 1.0

                        # Record candidate intervention
                        f_gz.write(json.dumps({
                            "pair_id": pair_id,
                            "scene_id": scene_id,
                            "context": context,
                            "layer": l_key,
                            "direction": dir_name,
                            "k": k,
                            "condition": "candidate",
                            "draw_id": 0,
                            "units": [int(u) for u in cand_k],
                            "candidate_overlap_count": k,
                            "candidate_overlap_fraction": 1.0,
                            "candidate_perturbation_norm": cand_norm,
                            "control_perturbation_norm": cand_norm,
                            "relative_norm_mismatch": 0.0,
                            "is_within_tolerance": True,
                            "matching_stratum": "candidate",
                            "baseline_output": [float(v) for v in g_recip.cpu().numpy()],
                            "transplanted_output": [float(v) for v in g_trans.cpu().numpy()],
                            "target_contrast_xy": [float(v) for v in delta_target_xy.cpu().numpy()],
                            "transferred_xy": [float(v) for v in delta_trans_xy.cpu().numpy()],
                            "projected_transfer_efficiency": cand_eta,
                            "transfer_magnitude_m": cand_transfer_mag,
                            "directional_cosine": cand_cos,
                            "orthogonal_residual_m": cand_r_perp,
                            "heading_transfer_eff": cand_th_eff,
                            "necessity_drop_ratio": cand_drop,
                            "necessity_retention_ratio": cand_retention,
                            "necessity_cos": cos_mabl,
                        }) + "\n")
                        total_interventions_recorded += 1

                        # b) Strict Norm-Matched Controls Forward Pass
                        batch_norm_etas = None
                        batch_norm_cos = None
                        batch_norm_mags_np = None
                        batch_r_perps = None
                        batch_th_diffs = None

                        if norm_matched_res["status"] == "VALID_COMMON_SUPPORT":
                            norm_subsets = norm_matched_res["selected_subsets"]
                            batch_norm_trans = h_recip.repeat(len(norm_subsets), 1)
                            for d_idx, sub_u in enumerate(norm_subsets):
                                batch_norm_trans[d_idx, sub_u] = h_donor[0, sub_u]

                            with torch.no_grad():
                                batch_g_norm_trans = forward_grounding_from_layer(model, batch_norm_trans, start_l)

                            batch_delta_norm_xy = batch_g_norm_trans[:, :2] - g_recip[:2]
                            batch_norm_mags = torch.norm(batch_delta_norm_xy, dim=-1)
                            batch_norm_dots = torch.sum(batch_delta_norm_xy * delta_target_xy, dim=-1)
                            batch_norm_etas = (batch_norm_dots / target_norm_sq).cpu().numpy()
                            batch_norm_cos = (batch_norm_dots / (batch_norm_mags * target_norm + 1e-8)).cpu().numpy()
                            batch_norm_mags_np = batch_norm_mags.cpu().numpy()

                            batch_r_perps = torch.norm(
                                batch_delta_norm_xy - (batch_norm_dots / target_norm_sq).unsqueeze(-1) * delta_target_xy,
                                dim=-1
                            ).cpu().numpy()

                            batch_th_trans = torch.atan2(batch_g_norm_trans[:, 2], batch_g_norm_trans[:, 3])
                            batch_th_diffs = wrap_angle((batch_th_trans - th_recip).cpu().numpy())

                            for d_idx, sub_u in enumerate(norm_subsets):
                                f_gz.write(json.dumps({
                                    "pair_id": pair_id,
                                    "scene_id": scene_id,
                                    "context": context,
                                    "layer": l_key,
                                    "direction": dir_name,
                                    "k": k,
                                    "condition": "norm_matched_random",
                                    "draw_id": d_idx,
                                    "units": [int(u) for u in sub_u],
                                    "candidate_overlap_count": int(norm_matched_res["selected_overlaps"][d_idx]),
                                    "candidate_overlap_fraction": float(norm_matched_res["selected_overlaps"][d_idx] / k),
                                    "candidate_perturbation_norm": cand_norm,
                                    "control_perturbation_norm": float(norm_matched_res["selected_norms"][d_idx]),
                                    "relative_norm_mismatch": float(norm_matched_res["selected_rel_errors"][d_idx]),
                                    "is_within_tolerance": True,
                                    "matching_stratum": "general_norm_matched",
                                    "baseline_output": [float(v) for v in g_recip.cpu().numpy()],
                                    "transplanted_output": [float(v) for v in batch_g_norm_trans[d_idx].cpu().numpy()],
                                    "target_contrast_xy": [float(v) for v in delta_target_xy.cpu().numpy()],
                                    "transferred_xy": [float(v) for v in batch_delta_norm_xy[d_idx].cpu().numpy()],
                                    "projected_transfer_efficiency": float(batch_norm_etas[d_idx]),
                                    "transfer_magnitude_m": float(batch_norm_mags_np[d_idx]),
                                    "directional_cosine": float(batch_norm_cos[d_idx]),
                                    "orthogonal_residual_m": float(batch_r_perps[d_idx]),
                                    "heading_transfer_eff": float(batch_th_diffs[d_idx] / (th_target + 1e-8)) if abs(th_target) > 1e-3 else 1.0,
                                    "necessity_drop_ratio": float(norm_ctrl_drops[d_idx]) if norm_ctrl_drops is not None else None,
                                    "necessity_retention_ratio": float(norm_ctrl_retentions[d_idx]) if norm_ctrl_retentions is not None else None,
                                    "necessity_cos": float(norm_ctrl_cosines[d_idx]) if norm_ctrl_cosines is not None else None,
                                }) + "\n")
                                total_interventions_recorded += 1

                        # c) Unconstrained Dimension-Matched Controls Forward Pass
                        batch_dim_trans = h_recip.repeat(len(dim_matched_subsets), 1)
                        for d_idx, sub_u in enumerate(dim_matched_subsets):
                            batch_dim_trans[d_idx, sub_u] = h_donor[0, sub_u]

                        with torch.no_grad():
                            batch_g_dim_trans = forward_grounding_from_layer(model, batch_dim_trans, start_l)

                        batch_delta_dim_xy = batch_g_dim_trans[:, :2] - g_recip[:2]
                        batch_dim_mags = torch.norm(batch_delta_dim_xy, dim=-1)
                        batch_dim_dots = torch.sum(batch_delta_dim_xy * delta_target_xy, dim=-1)
                        batch_dim_etas = (batch_dim_dots / target_norm_sq).cpu().numpy()
                        batch_dim_cos = (batch_dim_dots / (batch_dim_mags * target_norm + 1e-8)).cpu().numpy()
                        batch_dim_mags_np = batch_dim_mags.cpu().numpy()

                        batch_dim_r_perps = torch.norm(
                            batch_delta_dim_xy - (batch_dim_dots / target_norm_sq).unsqueeze(-1) * delta_target_xy,
                            dim=-1
                        ).cpu().numpy()

                        batch_dim_th_trans = torch.atan2(batch_g_dim_trans[:, 2], batch_g_dim_trans[:, 3])
                        batch_dim_th_diffs = wrap_angle((batch_dim_th_trans - th_recip).cpu().numpy())

                        for d_idx, sub_u in enumerate(dim_matched_subsets):
                            f_gz.write(json.dumps({
                                "pair_id": pair_id,
                                "scene_id": scene_id,
                                "context": context,
                                "layer": l_key,
                                "direction": dir_name,
                                "k": k,
                                "condition": "dimension_matched_random",
                                "draw_id": d_idx,
                                "units": [int(u) for u in sub_u],
                                "candidate_overlap_count": int(dim_overlaps[d_idx]),
                                "candidate_overlap_fraction": float(dim_overlaps[d_idx] / k),
                                "candidate_perturbation_norm": cand_norm,
                                "control_perturbation_norm": float(dim_norms[d_idx]),
                                "relative_norm_mismatch": float(dim_rel_errors[d_idx]),
                                "is_within_tolerance": bool(dim_rel_errors[d_idx] <= norm_tolerance),
                                "matching_stratum": "dimension_matched_unconstrained",
                                "baseline_output": [float(v) for v in g_recip.cpu().numpy()],
                                "transplanted_output": [float(v) for v in batch_g_dim_trans[d_idx].cpu().numpy()],
                                "target_contrast_xy": [float(v) for v in delta_target_xy.cpu().numpy()],
                                "transferred_xy": [float(v) for v in batch_delta_dim_xy[d_idx].cpu().numpy()],
                                "projected_transfer_efficiency": float(batch_dim_etas[d_idx]),
                                "transfer_magnitude_m": float(batch_dim_mags_np[d_idx]),
                                "directional_cosine": float(batch_dim_cos[d_idx]),
                                "orthogonal_residual_m": float(batch_dim_r_perps[d_idx]),
                                "heading_transfer_eff": float(batch_dim_th_diffs[d_idx] / (th_target + 1e-8)) if abs(th_target) > 1e-3 else 1.0,
                            }) + "\n")
                            total_interventions_recorded += 1

                        # d) Overlap-Standardized Transfer Efficiency Computation
                        overlap_std_eta = None
                        strat_etas_by_o = {}
                        if len(stratified_subsets_by_stratum) > 0:
                            sum_weights = 0.0
                            sum_weighted_eta = 0.0
                            for o_val, strat_info in stratified_subsets_by_stratum.items():
                                sub_list = strat_info["subsets"]
                                batch_strat_trans = h_recip.repeat(len(sub_list), 1)
                                for d_idx, sub_u in enumerate(sub_list):
                                    batch_strat_trans[d_idx, sub_u] = h_donor[0, sub_u]
                                with torch.no_grad():
                                    batch_g_strat = forward_grounding_from_layer(model, batch_strat_trans, start_l)
                                batch_delta_strat_xy = batch_g_strat[:, :2] - g_recip[:2]
                                batch_dots_strat = torch.sum(batch_delta_strat_xy * delta_target_xy, dim=-1)
                                batch_etas_strat = (batch_dots_strat / target_norm_sq).cpu().numpy()
                                m_eta_o = float(np.mean(batch_etas_strat))
                                strat_etas_by_o[str(o_val)] = m_eta_o
                                sum_weighted_eta += strat_info["prob"] * m_eta_o
                                sum_weights += strat_info["prob"]

                            if sum_weights > 0:
                                overlap_std_eta = float(sum_weighted_eta / sum_weights)

                        # e) Disjoint Transfer Efficiency Computation
                        disjoint_eta = None
                        if disjoint_res["status"] in ["VALID_COMMON_SUPPORT", "DETERMINISTIC_REFERENCE"]:
                            disjoint_subsets = disjoint_res["selected_subsets"]
                            batch_disjoint_trans = h_recip.repeat(len(disjoint_subsets), 1)
                            for d_idx, sub_u in enumerate(disjoint_subsets):
                                batch_disjoint_trans[d_idx, sub_u] = h_donor[0, sub_u]
                            with torch.no_grad():
                                batch_g_disjoint = forward_grounding_from_layer(model, batch_disjoint_trans, start_l)
                            batch_delta_disjoint_xy = batch_g_disjoint[:, :2] - g_recip[:2]
                            batch_dots_disjoint = torch.sum(batch_delta_disjoint_xy * delta_target_xy, dim=-1)
                            disjoint_etas = (batch_dots_disjoint / target_norm_sq).cpu().numpy()
                            disjoint_eta = float(np.mean(disjoint_etas))

                        # Store summary for this setting
                        setting_key = f"{scene_id}__{l_key}__{dir_name}__k_{k}"
                        pair_metrics["results"][setting_key] = {
                            "candidate_eta": cand_eta,
                            "norm_matched_eta_mean": float(np.mean(batch_norm_etas)) if batch_norm_etas is not None else None,
                            "dim_matched_eta_mean": float(np.mean(batch_dim_etas)),
                            "overlap_standardized_eta": overlap_std_eta,
                            "disjoint_eta": disjoint_eta,
                            "cand_norm": cand_norm,
                            "norm_matched_norm_mean": float(np.mean(norm_matched_res["selected_norms"])) if norm_matched_res["selected_norms"] else None,
                            "dim_matched_norm_mean": float(np.mean(dim_norms)),
                            "norm_matched_status": norm_matched_res["status"],
                            "n_within_5pct": norm_matched_res["n_valid"],
                            "total_drawn": norm_matched_res["total_drawn"],
                            "necessity_drop_ratio": cand_drop,
                            "necessity_retention_ratio": cand_retention,
                            "necessity_cos": cos_mabl,
                            "norm_matched_necessity_drop_mean": float(np.mean(norm_ctrl_drops)) if norm_ctrl_drops is not None else None,
                            "norm_matched_necessity_retention_mean": float(np.mean(norm_ctrl_retentions)) if norm_ctrl_retentions is not None else None,
                            "delta_bearing": cand_delta_bearing,
                            "delta_heading": cand_delta_heading,
                        }

        pair_summary_records.append(pair_metrics)
        f_pairs.write(json.dumps(pair_metrics) + "\n")

    f_gz.close()
    f_audit.close()
    f_pairs.close()

    print(f"Executed interventions. Total individual intervention records recorded: {total_interventions_recorded}")

    # =========================================================================
    # 6. Statistical Analysis Across Independent Scene Pairs (N = 60)
    # =========================================================================
    print("\n[5/6] Computing Statistical Tests Across Independent Scene Pairs (N = 60)...")
    # Unit of analysis: Independent scene pair. Average Scene A and Scene B, and average A->B and B->A.
    statistical_results = {}
    primary_test_keys = []
    p_vals_raw_sufficiency = []

    for l_cfg in layer_configs:
        l_key = l_cfg["layer_key"]
        statistical_results[l_key] = {}

        for k in k_values:
            cand_etas = []
            norm_rand_etas = []
            dim_rand_etas = []
            overlap_std_etas = []
            disjoint_etas = []

            cand_drops = []
            norm_ctrl_drops = []

            valid_norm_pair_count = 0

            for pm in pair_summary_records:
                # Directions: A_to_B and B_to_A
                keys_scene_a_fwd = f"scene_a__{l_key}__A_to_B__k_{k}"
                keys_scene_a_rev = f"scene_a__{l_key}__B_to_A__k_{k}"
                keys_scene_b_fwd = f"scene_b__{l_key}__A_to_B__k_{k}"
                keys_scene_b_rev = f"scene_b__{l_key}__B_to_A__k_{k}"

                res_af = pm["results"][keys_scene_a_fwd]
                res_ar = pm["results"][keys_scene_a_rev]
                res_bf = pm["results"][keys_scene_b_fwd]
                res_br = pm["results"][keys_scene_b_rev]

                # Pair average candidate eta
                c_eta = 0.25 * (res_af["candidate_eta"] + res_ar["candidate_eta"] + res_bf["candidate_eta"] + res_br["candidate_eta"])
                cand_etas.append(c_eta)

                # Pair average dimension-matched eta
                dr_eta = 0.25 * (res_af["dim_matched_eta_mean"] + res_ar["dim_matched_eta_mean"] + res_bf["dim_matched_eta_mean"] + res_br["dim_matched_eta_mean"])
                dim_rand_etas.append(dr_eta)

                # Overlap-standardized eta
                if all(r["overlap_standardized_eta"] is not None for r in [res_af, res_ar, res_bf, res_br]):
                    ov_eta = 0.25 * (res_af["overlap_standardized_eta"] + res_ar["overlap_standardized_eta"] + res_bf["overlap_standardized_eta"] + res_br["overlap_standardized_eta"])
                    overlap_std_etas.append(ov_eta)

                # Disjoint eta
                if all(r["disjoint_eta"] is not None for r in [res_af, res_ar, res_bf, res_br]):
                    dj_eta = 0.25 * (res_af["disjoint_eta"] + res_ar["disjoint_eta"] + res_bf["disjoint_eta"] + res_br["disjoint_eta"])
                    disjoint_etas.append(dj_eta)

                # Necessity drops
                c_drop = 0.5 * (res_af["necessity_drop_ratio"] + res_bf["necessity_drop_ratio"])
                cand_drops.append(c_drop)

                # Norm-matched controls: check if ALL 4 settings had VALID_COMMON_SUPPORT
                has_valid_norm = all(
                    r["norm_matched_status"] == "VALID_COMMON_SUPPORT" and r["norm_matched_eta_mean"] is not None
                    for r in [res_af, res_ar, res_bf, res_br]
                )
                if has_valid_norm:
                    valid_norm_pair_count += 1
                    nr_eta = 0.25 * (res_af["norm_matched_eta_mean"] + res_ar["norm_matched_eta_mean"] + res_bf["norm_matched_eta_mean"] + res_br["norm_matched_eta_mean"])
                    norm_rand_etas.append(nr_eta)

                    if all(r["norm_matched_necessity_drop_mean"] is not None for r in [res_af, res_bf]):
                        n_drop = 0.5 * (res_af["norm_matched_necessity_drop_mean"] + res_bf["norm_matched_necessity_drop_mean"])
                        norm_ctrl_drops.append(n_drop)

            cand_etas = np.array(cand_etas)
            dim_rand_etas = np.array(dim_rand_etas)
            cand_drops = np.array(cand_drops)

            n_pairs = len(pair_summary_records)
            coverage_pct = float(valid_norm_pair_count / max(1, n_pairs))

            # Candidate vs Dimension-Matched Random
            diff_dim = cand_etas - dim_rand_etas
            mean_diff_dim = float(np.mean(diff_dim))
            ci_dim = paired_bootstrap_ci(diff_dim, seed=random_seed + k)
            p_dim = float(stats.wilcoxon(diff_dim, alternative="greater").pvalue) if np.any(diff_dim != 0) else 1.0

            # Candidate vs Strict Norm-Matched Controls
            is_common_support_complete = (valid_norm_pair_count == n_pairs)
            if is_common_support_complete and len(norm_rand_etas) == n_pairs:
                norm_rand_etas_arr = np.array(norm_rand_etas)
                diff_norm = cand_etas - norm_rand_etas_arr
                mean_diff_norm = float(np.mean(diff_norm))
                ci_norm = paired_bootstrap_ci(diff_norm, seed=random_seed + k * 10)
                p_norm_wilcoxon = float(stats.wilcoxon(diff_norm, alternative="greater").pvalue) if np.any(diff_norm != 0) else 1.0
                p_norm_perm = sign_flip_permutation_test(diff_norm, n_perm=100000, seed=random_seed + k * 20)
                status_sufficiency = "CONFIRMATORY_EVALUATED"

                # Necessity comparison
                if len(norm_ctrl_drops) == n_pairs:
                    diff_drop = cand_drops - np.array(norm_ctrl_drops)
                    mean_diff_drop = float(np.mean(diff_drop))
                    ci_drop = paired_bootstrap_ci(diff_drop, seed=random_seed + k * 30)
                    p_drop_wilcoxon = float(stats.wilcoxon(diff_drop, alternative="greater").pvalue) if np.any(diff_drop != 0) else 1.0
                else:
                    mean_diff_drop = None
                    ci_drop = None
                    p_drop_wilcoxon = None
            else:
                # STRICT REQUIREMENT: When common support is incomplete, do NOT compute confirmatory p-values!
                diff_norm = None
                mean_diff_norm = None
                ci_norm = None
                p_norm_wilcoxon = None
                p_norm_perm = None
                status_sufficiency = "INSUFFICIENT_COMMON_SUPPORT"
                mean_diff_drop = None
                ci_drop = None
                p_drop_wilcoxon = None

            primary_test_keys.append((l_key, f"k_{k}"))
            p_vals_raw_sufficiency.append(p_norm_wilcoxon if p_norm_wilcoxon is not None else 1.0)

            statistical_results[l_key][f"k_{k}"] = {
                "k": k,
                "n_independent_pairs": n_pairs,
                "valid_norm_matched_pairs": valid_norm_pair_count,
                "match_coverage_pct": coverage_pct,
                "common_support_status": status_sufficiency,
                "candidate_efficiency_mean": float(np.mean(cand_etas)),
                "candidate_efficiency_ci": paired_bootstrap_ci(cand_etas, seed=random_seed),
                "dimension_matched_efficiency_mean": float(np.mean(dim_rand_etas)),
                "dimension_matched_efficiency_ci": paired_bootstrap_ci(dim_rand_etas, seed=random_seed),
                "norm_matched_efficiency_mean": float(np.mean(norm_rand_etas)) if len(norm_rand_etas) > 0 else None,
                "overlap_standardized_efficiency_mean": float(np.mean(overlap_std_etas)) if len(overlap_std_etas) > 0 else None,
                "disjoint_efficiency_mean": float(np.mean(disjoint_etas)) if len(disjoint_etas) > 0 else None,
                # Primary Outcome: Candidate vs Strict Norm-Matched Controls
                "primary_sufficiency_cand_vs_norm_matched": {
                    "status": status_sufficiency,
                    "mean_paired_difference": mean_diff_norm,
                    "bootstrap_95_ci": ci_norm,
                    "wilcoxon_one_sided_p_value": p_norm_wilcoxon,
                    "permutation_sign_flip_p_value": p_norm_perm,
                },
                # Causal Necessity: Candidate Drop vs Norm-Matched Drop
                "causal_necessity_cand_vs_norm_matched": {
                    "candidate_drop_mean": float(np.mean(cand_drops)),
                    "norm_matched_drop_mean": float(np.mean(norm_ctrl_drops)) if len(norm_ctrl_drops) > 0 else None,
                    "mean_drop_difference": mean_diff_drop,
                    "drop_difference_ci": ci_drop,
                    "wilcoxon_one_sided_p_value": p_drop_wilcoxon,
                },
                # Secondary Comparison: Candidate vs Unconstrained Dimension-Matched Random
                "cand_vs_dimension_matched": {
                    "mean_paired_difference": mean_diff_dim,
                    "bootstrap_95_ci": ci_dim,
                    "wilcoxon_one_sided_p_value": p_dim,
                },
            }

    # Holm-Bonferroni correction across the 10 planned primary tests
    p_adjusted = holm_bonferroni_correction(p_vals_raw_sufficiency)

    for idx, (l_key, k_str) in enumerate(primary_test_keys):
        entry = statistical_results[l_key][k_str]["primary_sufficiency_cand_vs_norm_matched"]
        if entry["status"] == "CONFIRMATORY_EVALUATED":
            entry["holm_bonferroni_adjusted_p_value"] = float(p_adjusted[idx])
        else:
            entry["holm_bonferroni_adjusted_p_value"] = None

    # Print Primary Findings Table
    print("\n" + "=" * 80)
    print("PHASE 5.3 PRIMARY STATISTICAL RESULTS: STRICT ACTIVATION-NORM MATCHING (N = 60)")
    print("=" * 80)
    for l_key in ["layer1_post_relu_64d", "layer2_post_relu_64d"]:
        print(f"\n--- {l_key} ---")
        for k in k_values:
            res = statistical_results[l_key][f"k_{k}"]
            cov = res["match_coverage_pct"] * 100
            c_eff = res["candidate_efficiency_mean"]
            d_eff = res["dimension_matched_efficiency_mean"]
            n_eff = res["norm_matched_efficiency_mean"]
            ov_eff = res["overlap_standardized_efficiency_mean"]
            status = res["common_support_status"]
            cand_vs_norm = res["primary_sufficiency_cand_vs_norm_matched"]
            diff = cand_vs_norm["mean_paired_difference"]
            p_adj = cand_vs_norm["holm_bonferroni_adjusted_p_value"]

            c_drop = res["causal_necessity_cand_vs_norm_matched"]["candidate_drop_mean"]
            n_drop = res["causal_necessity_cand_vs_norm_matched"]["norm_matched_drop_mean"]

            diff_str = f"{diff:+.3f}" if diff is not None else "N/A"
            p_adj_str = f"{p_adj:.4e}" if p_adj is not None else "N/A (unsupported)"
            n_eff_str = f"{n_eff:.3f}" if n_eff is not None else "N/A"
            ov_eff_str = f"{ov_eff:.3f}" if ov_eff is not None else "N/A"
            n_drop_str = f"{n_drop:.3f}" if n_drop is not None else "N/A"

            print(f"  k={k:2d} | Coverage={cov:5.1f}% | Cand={c_eff:.3f} | DimRand={d_eff:.3f} | NormRand={n_eff_str} | OverlapStd={ov_eff_str} | Diff={diff_str} | p_adj={p_adj_str}")
            print(f"       Necessity: CandDrop={c_drop:.3f} | NormRandDrop={n_drop_str} | Status: {status}")

    # Determine Mechanistic Verdict Based Strictly on Evidence
    # Check evaluated tests
    evaluated_tests = [
        res["primary_sufficiency_cand_vs_norm_matched"]
        for l in statistical_results.values()
        for res in l.values()
    ]
    insufficient_count = sum(1 for t in evaluated_tests if t["status"] == "INSUFFICIENT_COMMON_SUPPORT")
    sig_count = sum(1 for t in evaluated_tests if t.get("holm_bonferroni_adjusted_p_value") is not None and t["holm_bonferroni_adjusted_p_value"] < 0.05)

    if insufficient_count > 0:
        mechanistic_verdict = (
            f"Unidentifiable Under Current Representation ({insufficient_count}/10 settings exhibit INSUFFICIENT_COMMON_SUPPORT). "
            "For multiple (layer, k) configurations, the candidate task-sensitive units occupy an activation-perturbation "
            "magnitude region that random subsets cannot match within 5% tolerance even after up to 200,000 draws. "
            "Under strict norm-matching without nearest-outside fallback, candidate advantage cannot be identified as "
            "functional selectivity versus perturbation-magnitude modulation in those settings."
        )
    elif sig_count > 0:
        mechanistic_verdict = (
            f"Selective Subspace Supported ({sig_count}/10 settings significant after strict norm matching). "
            "Candidate task-sensitive units maintain significant causal sufficiency and necessity advantages "
            "over strictly activation-norm-matched controls (100% common support across all 60 pairs)."
        )
    else:
        mechanistic_verdict = (
            "Activation-Magnitude Dominated (0/10 settings show significant candidate advantage over norm-matched controls). "
            "While candidate units outperform unconstrained random subsets, this advantage completely disappears when "
            "activation perturbation magnitude is strictly controlled within 5% tolerance. Apparent selectivity is an artefact."
        )

    print(f"\nMechanistic Verdict: {mechanistic_verdict}")

    elapsed_time = time.time() - start_time

    # 7. Construct Summary JSON and Save
    summary_data = {
        "experiment_name": "Phase 5.3 Valid Norm-Matched Causal Mediation & Necessity Revalidation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_clean": git_clean,
        "random_seed": random_seed,
        "device": device,
        "elapsed_seconds": elapsed_time,
        "provenance_hashes": {
            "checkpoint_sha256": ckpt_sha256,
            "scene_specs_sha256": specs_sha256,
            "candidate_rankings_sha256": ranking_hash,
            "script_sha256": compute_file_sha256(__file__),
        },
        "dataset_composition": {
            "total_independent_pairs": len(verified_specs),
            "nominal_pairs": sum(1 for sp in verified_specs if sp["context"] == "nominal"),
            "illumination_shift_pairs": sum(1 for sp in verified_specs if sp["context"] == "illumination_shift"),
            "total_scenes": len(verified_specs) * 2,
            "total_interventions_recorded": total_interventions_recorded,
        },
        "frozen_candidate_rankings": frozen_rankings,
        "representation_statistics": stage_representation_stats,
        "statistical_results": statistical_results,
        "mechanistic_conclusion": {
            "verdict": mechanistic_verdict,
            "insufficient_common_support_count": insufficient_count,
            "significant_primary_tests_count": sig_count,
            "total_primary_tests": len(evaluated_tests),
        },
    }

    summary_path = out_dir / "task_conditioned_circuits_v5_3_summary.json"
    with open(summary_path, "w") as f_sum:
        json.dump(summary_data, f_sum, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # 8. Generate Schema Markdown
    schema_path = out_dir / "task_conditioned_circuits_v5_3_schema.md"
    generate_schema_documentation(schema_path)
    print(f"Generated schema documentation at {schema_path}")

    print("\nBenchmark execution successfully completed.")
    return summary_data


def generate_schema_documentation(filepath: Path) -> None:
    schema_md = """# Phase 5.3 Data Schema Documentation

## 1. Primary Output Files

- `task_conditioned_circuits_v5_3_summary.json`: Top-level experimental configuration, provenance hashes, aggregate statistics across 60 independent pairs, Holm-Bonferroni adjusted p-values, match coverage table, representation profiling across all 9 stages, and machine-readable mechanistic verdict.
- `task_conditioned_circuits_v5_3_pairs.jsonl`: One record per independent scene pair (N = 60), containing paired sufficiency metrics, necessity drops, contrast retentions, heading/bearing shifts, and common-support eligibility status.
- `task_conditioned_circuits_v5_3_activations.jsonl`: One record per independent pair and scene (120 records total), containing raw Layer 1 and Layer 2 post-ReLU 64D activations under Task A and Task B, baseline outputs, and ground-truth goal vectors.
- `task_conditioned_circuits_v5_3_interventions.jsonl.gz`: Gzipped records of all individual causal interventions across candidate, strict norm-matched random, unconstrained dimension-matched random, overlap-stratified, and disjoint controls.
- `task_conditioned_circuits_v5_3_control_audit.jsonl`: Granular audit records for every setting (pair × scene × layer × k × stratum), detailing random seed, attempted draws, valid norm matches within 5%, retained count, min/median/max relative error, and common-support eligibility status.

## 2. Statistical Unit & Conventions

- **Statistical Unit**: Independent scene pair ($N = 60$). Each pair features two visual scenes (Scene A and Scene B) with target and distractor positions swapped.
- **Pair Aggregation**: Scene A and Scene B are averaged. Transplantation directions ($A \\to B$ and $B \\to A$) are averaged relative to their respective contrast directions $\\Delta \\mathbf{g}^*$.
- **Strict Norm-Matching Filter**:
  $$\\text{rel\\_error} = \\frac{|\\|\\Delta \\mathbf{h}[U_{\\text{ctrl}}]\\| - \\|\\Delta \\mathbf{h}[U_{\\text{cand}}]\\||}{\\max(10^{-8}, \\|\\Delta \\mathbf{h}[U_{\\text{cand}}]\\|)} \\le 0.05$$
  If $< 100$ valid subsets exist after adaptive sampling up to 200,000 draws, the setting is marked `INSUFFICIENT_COMMON_SUPPORT`, retained controls are set to `None`, and no confirmatory p-value is computed.
- **Overlap Recording**:
  $$O = |U_{\\text{cand}} \\cap U_{\\text{rand}}|$$
  $$P(O = o) = \\frac{\\binom{k}{o} \\binom{D - k}{k - o}}{\\binom{D}{k}}, \\quad D = 64$$
- **Projected Transfer Efficiency**:
  $$\\eta = \\frac{(\\mathbf{g}_{\\text{trans}} - \\mathbf{g}_{\\text{recip}})_{:2} \\cdot \\Delta \\mathbf{g}^*}{\\|\\Delta \\mathbf{g}^*\\|^2}$$
- **Condition-Averaged Necessity Ablation**:
  $$h_i^A, h_i^B \\longrightarrow \\frac{h_i^A + h_i^B}{2} \\quad \\text{for } i \\in U$$
  $$\\text{retention\\_ratio} = \\frac{\\|\\Delta \\tilde{\\mathbf{g}}_{\\text{xy}}\\|}{\\|\\Delta \\mathbf{g}^*_{\\text{xy}}\\|}, \\quad \\text{drop\\_ratio} = 1 - \\text{retention\\_ratio}$$
"""
    with open(filepath, "w") as f:
        f.write(schema_md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 5.3 Valid Norm-Matched Causal Mediation Benchmark")
    parser.add_argument("--checkpoint", type=str, default="world_models_fpga/checkpoints_v4/hjepa_best.pt")
    parser.add_argument("--specs", type=str, default="world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_2_scene_specs.jsonl")
    parser.add_argument("--results-dir", type=str, default="world_models_fpga/research/circuits/results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--required-controls", type=int, default=100)
    parser.add_argument("--initial-draws", type=int, default=5000)
    parser.add_argument("--chunk-draws", type=int, default=20000)
    parser.add_argument("--max-draws", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--quick-pairs", type=int, default=None, help="Debug only: test with fewer pairs")

    args = parser.parse_args()

    run_task_conditioned_circuits_v5_3(
        checkpoint_path=args.checkpoint,
        specs_path=args.specs,
        results_dir=args.results_dir,
        device=args.device,
        norm_tolerance=args.tolerance,
        required_controls=args.required_controls,
        initial_draws=args.initial_draws,
        chunk_draws=args.chunk_draws,
        max_draws=args.max_draws,
        random_seed=args.seed,
        quick_test_pairs=args.quick_pairs,
    )
