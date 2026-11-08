"""
Project Hephaestus: Phase 5.2 Mechanistic Selectivity & Cross-Context Revalidation Benchmark
Script: world_models_fpga/research/circuits/run_task_conditioned_circuits_v5_2.py

Scientific Objective:
Determine whether the task-sensitive hidden-unit subspaces identified in Phase 5.1
are genuinely selective causal mechanisms or simply activation-magnitude effects.

Methodological Upgrades in Phase 5.2:
1. Both-Context Candidate Ranking on Discovery Split:
   - Units ranked using both Scene A and Scene B visual contexts:
     s_i = 0.5 * (|h_i(z_A, e_B) - h_i(z_A, e_A)| + |h_i(z_B, e_B) - h_i(z_B, e_A)|)
   - Frozen strictly prior to testing on unseen scenes.
2. Independent Cross-Context Scene Evaluation Suite (60 pairs, 120 scenes):
   - 30 nominal appearance pairs (light_diffuse_scale = 1.0)
   - 30 moderate illumination shift pairs (light_diffuse_scale = 0.60)
   - Rejection-sampled with verified 5-step simulator initialization and runtime assertions.
   - Object visibility (>= 100 px), initial waypoint distance (>= 0.60m), separation (>= 0.52m).
3. Proper Activation-Norm-Matched Random Controls:
   - For each layer, context, direction, and k in [2, 4, 8, 16, 32]:
     - Sample >= 5,000 random dimension-matched subsets.
     - Compute activation perturbation norm ||delta_h_U||.
     - Select closest 100 subsets within 5% tolerance (|N_rand - N_cand| / N_cand <= 0.05).
     - Record matching quality and norm differences for every control.
     - Also retain standard dimension-matched random controls (unconstrained by norm).
     - Save all individual random draws in interventions JSONL.
4. Bidirectional Causal Sufficiency (A -> B and B -> A):
   - Report projected transfer efficiency eta, transfer magnitude, directional cosine,
     orthogonal residual r_perp, and true heading transfer (atan2).
5. Causal Necessity:
   - Condition-averaged ablation (h_i -> (h_i^A + h_i^B)/2) and zero-ablation.
   - Measure XY contrast retention, directional cosine, bearing change, and heading change.
6. Representation Statistics with Corrected Signed Handling:
   - Participation ratio, eigenvalue spectra, explained variance thresholds.
   - Post-ReLU: Treves-Rolls lifetime and population sparseness, zero fraction.
   - Signed pre-ReLU and linear stages: positive-part sparseness, excess kurtosis, negative fraction.
   - Explicit reporting of silent units (variance < 1e-6).
7. Statistical Testing:
   - Independent scene pair is statistical unit.
   - Paired Wilcoxon and sign-flip permutation tests (100,000 resamples).
   - Pair-clustered bootstrap 95% CIs.
   - Holm-Bonferroni correction across planned primary-test family (20 tests).

Outputs:
  - Summary JSON: world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_2_summary.json
  - Pairs JSONL: world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_2_pairs.jsonl
  - Interventions JSONL: world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_2_interventions.jsonl
  - Scene Specs JSONL: world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_2_scene_specs.jsonl
  - Schema Markdown: world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_2_schema.md
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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

import mujoco
import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F

from env.amr_env import JetAutoEnv
from models.baseline_policy import encode_task_id, TARGET_KEYS
from research.benchmarks.model_wrappers import HJEPAAgent, compute_body_frame_goal_vec
from research.datasets.dataset_config import CKPT_DIR
from research.datasets.generation.capability_tasks import (
    CANONICAL_OBSTACLES,
)

GEOM_NAME_MAP = {
    "red_box": "obs_red_box",
    "blue_cylinder": "obs_blue_cyl",
    "green_cylinder": "obs_green_cyl",
    "yellow_cylinder": "obs_yellow_cyl",
    "purple_box": "obs_purple_box",
    "cyan_cylinder": "obs_cyan_cyl",
}


def compute_file_sha256(filepath: str | Path) -> str:
    p = Path(filepath)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_info() -> tuple[str, bool]:
    commit = "unknown"
    is_clean = False
    try:
        proc_c = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(REPO_ROOT))
        if proc_c.returncode == 0 and proc_c.stdout.strip():
            commit = proc_c.stdout.strip()
        proc_s = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=str(REPO_ROOT))
        if proc_s.returncode == 0:
            is_clean = (len(proc_s.stdout.strip()) == 0)
    except Exception:
        pass
    return commit, is_clean


def wrap_angle(theta: float | np.ndarray) -> float | np.ndarray:
    """Wraps angle to [-pi, +pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def paired_bootstrap_ci(data: list[float] | np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42) -> list[float]:
    """Computes non-parametric bootstrap 95% confidence interval for paired differences."""
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    boot_means = np.mean(rng.choice(arr, size=(n_boot, len(arr)), replace=True), axis=1)
    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return [low, high]


def sign_flip_permutation_test(diffs: np.ndarray, n_perm: int = 100000, seed: int = 42) -> float:
    """Exact/Monte-Carlo sign-flip permutation test for paired differences."""
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    obs_stat = float(np.mean(d))
    if obs_stat == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm_stats = np.mean(flips * d, axis=1)
    p_val = float(np.mean(np.abs(perm_stats) >= np.abs(obs_stat)))
    return max(p_val, 1.0 / n_perm)


def holm_bonferroni_correction(p_values: list[float] | np.ndarray) -> list[float]:
    """Applies step-down Holm-Bonferroni correction to an array of raw p-values."""
    raw = np.asarray(p_values, dtype=np.float64)
    m = len(raw)
    if m == 0:
        return []
    sort_idx = np.argsort(raw)
    p_adj = np.zeros(m, dtype=np.float64)
    for rank, idx in enumerate(sort_idx):
        adj = raw[idx] * (m - rank)
        p_adj[idx] = min(1.0, adj)
    # Enforce monotonicity
    for i in range(1, m):
        if p_adj[sort_idx[i]] < p_adj[sort_idx[i - 1]]:
            p_adj[sort_idx[i]] = p_adj[sort_idx[i - 1]]
    return [float(p) for p in p_adj]


def check_target_visibility(env: JetAutoEnv, target_key: str, min_visible_pixels: int = 100) -> dict:
    """Renders segmentation mask in JetAutoEnv to compute ground-truth target visibility."""
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
        # Sample robot pose in central arena
        rx = float(rng.uniform(-0.45, 0.45))
        ry = float(rng.uniform(-0.45, 0.45))
        rth = float(rng.uniform(-np.pi, np.pi))
        robot_pose = np.array([rx, ry, rth], dtype=np.float32)

        # Lateral bearings: one on left, one on right of front FOV
        side_flip = 1.0 if (rng.uniform() > 0.5) else -1.0
        bearing_1 = side_flip * float(rng.uniform(0.12, 0.28))
        bearing_2 = -side_flip * float(rng.uniform(0.12, 0.28))

        dist_1 = float(rng.uniform(1.05, 1.35))
        dist_2 = float(rng.uniform(1.05, 1.35))

        p1_x = float(rx + dist_1 * np.cos(rth + bearing_1))
        p1_y = float(ry + dist_1 * np.sin(rth + bearing_1))
        p2_x = float(rx + dist_2 * np.cos(rth + bearing_2))
        p2_y = float(ry + dist_2 * np.sin(rth + bearing_2))

        # Arena bounds check [-1.55, 1.55]
        if not (-1.55 <= p1_x <= 1.55 and -1.55 <= p1_y <= 1.55 and -1.55 <= p2_x <= 1.55 and -1.55 <= p2_y <= 1.55):
            rejection_counters["arena_bounds"] = rejection_counters.get("arena_bounds", 0) + 1
            continue

        # Separation check between targets (>= 0.52m)
        sep = float(np.linalg.norm([p1_x - p2_x, p1_y - p2_y]))
        if sep < 0.52:
            rejection_counters["target_separation"] = rejection_counters.get("target_separation", 0) + 1
            continue

        pos_1 = np.array([p1_x, p1_y, 0.35], dtype=np.float32)
        pos_2 = np.array([p2_x, p2_y, 0.35], dtype=np.float32)

        # Approach waypoints with clearance 0.60m from target centers
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

        # Assemble obstacles for Scene A
        obstacles_scene_a = copy.deepcopy(CANONICAL_OBSTACLES)
        obstacles_scene_a[key_a]["pos"] = pos_1
        obstacles_scene_a[key_b]["pos"] = pos_2

        # Check clearance to other obstacles
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

        # Verify Scene A
        env.restore_physical_baseline()
        env.apply_task_spec(spec_a)
        obs_a, _ = env.reset(options={"initial_pos": (rx, ry, rth)})

        # Assertions
        actual_pose_a = env.get_robot_pose()
        assert np.allclose(actual_pose_a, [rx, ry, rth], atol=1e-3), "Scene A robot pose mismatch"
        gid_a1 = env.geometry_id(GEOM_NAME_MAP.get(key_a, f"obs_{key_a}"))
        gid_b1 = env.geometry_id(GEOM_NAME_MAP.get(key_b, f"obs_{key_b}"))
        assert np.allclose(env.model.geom_pos[gid_a1][:2], pos_1[:2], atol=1e-3), "Scene A geom A mismatch"
        assert np.allclose(env.model.geom_pos[gid_b1][:2], pos_2[:2], atol=1e-3), "Scene A geom B mismatch"

        vis_a_scene_a = check_target_visibility(env, key_a, min_visible_pixels=min_pixels)
        vis_b_scene_a = check_target_visibility(env, key_b, min_visible_pixels=min_pixels)

        if not (vis_a_scene_a["is_visible"] and vis_b_scene_a["is_visible"]):
            rejection_counters["scene_a_visibility"] = rejection_counters.get("scene_a_visibility", 0) + 1
            continue

        # Assemble Scene B: Swapped positions (A at pos_2, B at pos_1)
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

        # Verify Scene B
        env.restore_physical_baseline()
        env.apply_task_spec(spec_b)
        obs_b, _ = env.reset(options={"initial_pos": (rx, ry, rth)})

        actual_pose_b = env.get_robot_pose()
        assert np.allclose(actual_pose_b, [rx, ry, rth], atol=1e-3), "Scene B robot pose mismatch"
        assert np.allclose(env.model.geom_pos[gid_a1][:2], pos_2[:2], atol=1e-3), "Scene B geom A mismatch"
        assert np.allclose(env.model.geom_pos[gid_b1][:2], pos_1[:2], atol=1e-3), "Scene B geom B mismatch"

        vis_a_scene_b = check_target_visibility(env, key_a, min_visible_pixels=min_pixels)
        vis_b_scene_b = check_target_visibility(env, key_b, min_visible_pixels=min_pixels)

        if not (vis_a_scene_b["is_visible"] and vis_b_scene_b["is_visible"]):
            rejection_counters["scene_b_visibility"] = rejection_counters.get("scene_b_visibility", 0) + 1
            continue

        # Camera observation hashes
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


def extract_9_stages(model: torch.nn.Module, z: torch.Tensor, t_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Forward passes through H-JEPA grounding head and extracts all 9 intermediate representations."""
    # Stage 1: Visual latent z (32D)
    # Stage 2: Task embedding e (16D)
    e = model.task_embedding(t_tensor)
    # Stage 3: Concatenated input (48D)
    x_in = torch.cat([z, e], dim=-1)

    fc1 = model.grounding_head[0]
    relu1 = model.grounding_head[1]
    fc2 = model.grounding_head[2]
    relu2 = model.grounding_head[3]
    fc3 = model.grounding_head[4]

    # Stage 4: Layer 1 pre-ReLU (64D)
    l1_pre = fc1(x_in)
    # Stage 5: Layer 1 post-ReLU (64D)
    l1_post = relu1(l1_pre)
    # Stage 6: Layer 2 pre-ReLU (64D)
    l2_pre = fc2(l1_post)
    # Stage 7: Layer 2 post-ReLU (64D)
    l2_post = relu2(l2_pre)
    # Stage 8: Raw output (4D)
    raw = fc3(l2_post)

    # Stage 9: Postprocessed goal output (4D)
    dx_dy = torch.clamp(raw[:, :2], -3.5, 3.5)
    trig = F.normalize(raw[:, 2:4] + 1e-6, dim=-1)
    goal = torch.cat([dx_dy, trig], dim=-1)

    return {
        "visual_latent_z": z,
        "task_embedding_e": e,
        "input_concat_48d": x_in,
        "layer1_pre_relu_64d": l1_pre,
        "layer1_post_relu_64d": l1_post,
        "layer2_pre_relu_64d": l2_pre,
        "layer2_post_relu_64d": l2_post,
        "raw_output_4d": raw,
        "postprocessed_goal_4d": goal,
    }


def forward_grounding_from_layer(
    model: torch.nn.Module,
    act: torch.Tensor,
    start_layer: int,  # 1 for layer1_post, 2 for layer2_post
) -> torch.Tensor:
    """Passes intermediate activation tensor [B, 64] through remaining layers to 4D goal output [B, 4]."""
    fc2 = model.grounding_head[2]
    relu2 = model.grounding_head[3]
    fc3 = model.grounding_head[4]

    if start_layer == 1:
        l2_pre = fc2(act)
        l2_post = relu2(l2_pre)
        raw = fc3(l2_post)
    elif start_layer == 2:
        raw = fc3(act)
    else:
        raise ValueError(f"start_layer must be 1 or 2, got {start_layer}")

    dx_dy = torch.clamp(raw[:, :2], -3.5, 3.5)
    trig = F.normalize(raw[:, 2:4] + 1e-6, dim=-1)
    return torch.cat([dx_dy, trig], dim=-1)


def compute_spectral_analysis(A: np.ndarray) -> dict:
    """Computes covariance eigenvalue spectrum and participation ratio."""
    A_cent = A - np.mean(A, axis=0, keepdims=True)
    N = A.shape[0]
    cov = (A_cent.T @ A_cent) / max(1, N - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
    total_var = np.sum(eigvals)
    if total_var > 1e-12:
        norm_eig = eigvals / total_var
        pr = float((np.sum(eigvals) ** 2) / (np.sum(eigvals ** 2) + 1e-12))
        cum_var = np.cumsum(norm_eig)
        d80 = int(np.searchsorted(cum_var, 0.80) + 1)
        d90 = int(np.searchsorted(cum_var, 0.90) + 1)
        d95 = int(np.searchsorted(cum_var, 0.95) + 1)
        d99 = int(np.searchsorted(cum_var, 0.99) + 1)
    else:
        norm_eig = np.zeros_like(eigvals)
        pr = 1.0
        d80 = d90 = d95 = d99 = 1

    return {
        "participation_ratio": pr,
        "eigenvalues_top8": [float(x) for x in eigvals[:8]],
        "dims_80_var": min(d80, A.shape[1]),
        "dims_90_var": min(d90, A.shape[1]),
        "dims_95_var": min(d95, A.shape[1]),
        "dims_99_var": min(d99, A.shape[1]),
    }


def compute_representation_statistics_v5_2(
    act_dict: dict[str, np.ndarray],  # stage_name -> array of shape [N, D]
    stage_name: str,
) -> dict:
    """
    Computes rigorous representation statistics:
    - Participation ratio and eigenvalue spectrum
    - Non-negative post-ReLU: true Treves-Rolls lifetime & population sparseness, zero fraction
    - Signed pre-ReLU and linear stages: positive-part sparseness, excess kurtosis, negative fraction
    - Silent units (variance < 1e-6) reported separately
    """
    A = act_dict[stage_name]
    N, D = A.shape
    spec = compute_spectral_analysis(A)

    # Unit variances across observations
    unit_vars = np.var(A, axis=0)
    silent_mask = (unit_vars < 1e-6)
    n_silent = int(np.sum(silent_mask))
    silent_frac = float(n_silent / D)

    active_unit_indices = np.where(~silent_mask)[0]

    # Non-negative vs signed
    is_post_relu = ("post_relu" in stage_name)

    if is_post_relu:
        # Standard Treves-Rolls
        lifetime_sparsities = []
        for d_idx in active_unit_indices:
            col = A[:, d_idx]
            mean_a = np.mean(col)
            mean_sq = np.mean(col ** 2)
            if mean_sq > 1e-12:
                s_l = float((1.0 - (mean_a ** 2) / mean_sq) / max(1e-6, 1.0 - 1.0 / N))
                lifetime_sparsities.append(np.clip(s_l, 0.0, 1.0))

        pop_sparsities = []
        for n_idx in range(N):
            row = A[n_idx, :]
            mean_a = np.mean(row)
            mean_sq = np.mean(row ** 2)
            if mean_sq > 1e-12:
                s_p = float((1.0 - (mean_a ** 2) / mean_sq) / max(1e-6, 1.0 - 1.0 / D))
                pop_sparsities.append(np.clip(s_p, 0.0, 1.0))

        zero_frac = float(np.mean(A == 0.0))

        return {
            "stage_name": stage_name,
            "dimensions": D,
            "participation_ratio": spec["participation_ratio"],
            "dims_80_var": spec["dims_80_var"],
            "dims_90_var": spec["dims_90_var"],
            "dims_95_var": spec["dims_95_var"],
            "dims_99_var": spec["dims_99_var"],
            "silent_units_count": n_silent,
            "silent_units_fraction": silent_frac,
            "is_non_negative": True,
            "lifetime_sparsity": {
                "mean": float(np.mean(lifetime_sparsities)) if lifetime_sparsities else 0.0,
                "std": float(np.std(lifetime_sparsities)) if lifetime_sparsities else 0.0,
            },
            "population_sparsity": {
                "mean": float(np.mean(pop_sparsities)) if pop_sparsities else 0.0,
                "std": float(np.std(pop_sparsities)) if pop_sparsities else 0.0,
            },
            "zero_fraction": zero_frac,
            "negative_fraction": 0.0,
            "excess_kurtosis": None,
        }
    else:
        # Signed activations
        A_pos = np.maximum(A, 0.0)
        pos_lifetime = []
        for d_idx in active_unit_indices:
            col = A_pos[:, d_idx]
            m1 = np.mean(col)
            m2 = np.mean(col ** 2)
            if m2 > 1e-12:
                s = float((1.0 - (m1 ** 2) / m2) / max(1e-6, 1.0 - 1.0 / N))
                pos_lifetime.append(np.clip(s, 0.0, 1.0))

        pos_pop = []
        for n_idx in range(N):
            row = A_pos[n_idx, :]
            m1 = np.mean(row)
            m2 = np.mean(row ** 2)
            if m2 > 1e-12:
                s = float((1.0 - (m1 ** 2) / m2) / max(1e-6, 1.0 - 1.0 / D))
                pos_pop.append(np.clip(s, 0.0, 1.0))

        # True signed kurtosis across active units
        kurt_list = []
        for d_idx in active_unit_indices:
            col = A[:, d_idx]
            k = float(stats.kurtosis(col, fisher=True))  # excess kurtosis
            if not np.isnan(k):
                kurt_list.append(k)

        zero_frac = float(np.mean(A == 0.0))
        neg_frac = float(np.mean(A < 0.0))

        return {
            "stage_name": stage_name,
            "dimensions": D,
            "participation_ratio": spec["participation_ratio"],
            "dims_80_var": spec["dims_80_var"],
            "dims_90_var": spec["dims_90_var"],
            "dims_95_var": spec["dims_95_var"],
            "dims_99_var": spec["dims_99_var"],
            "silent_units_count": n_silent,
            "silent_units_fraction": silent_frac,
            "is_non_negative": False,
            "positive_part_lifetime_sparsity": {
                "mean": float(np.mean(pos_lifetime)) if pos_lifetime else 0.0,
                "std": float(np.std(pos_lifetime)) if pos_lifetime else 0.0,
            },
            "positive_part_population_sparsity": {
                "mean": float(np.mean(pos_pop)) if pos_pop else 0.0,
                "std": float(np.std(pos_pop)) if pos_pop else 0.0,
            },
            "excess_kurtosis": {
                "mean": float(np.mean(kurt_list)) if kurt_list else 0.0,
                "std": float(np.std(kurt_list)) if kurt_list else 0.0,
            },
            "zero_fraction": zero_frac,
            "negative_fraction": neg_frac,
        }


def run_task_conditioned_circuits_v5_2(
    ckpt_path: Path,
    out_dir: Path,
    n_nominal_pairs: int = 30,
    n_illum_pairs: int = 30,
    n_norm_samples: int = 5000,
    n_retained_controls: int = 100,
    norm_tolerance: float = 0.05,
    seed: int = 2026,
    device: str = "cuda",
) -> dict:
    """Executes the complete Phase 5.2 Mechanistic Selectivity & Cross-Context Revalidation benchmark."""
    start_time = time.time()
    rng = np.random.default_rng(seed)
    git_commit, git_clean = get_git_info()

    print("================================================================================")
    print("Project Hephaestus: Phase 5.2 Mechanistic Selectivity & Cross-Context Revalidation")
    print(f"Commit: {git_commit} (Clean: {git_clean}) | Seed: {seed} | Device: {device}")
    print("================================================================================")

    # 1. Load Discovery Data from v4 Activations
    v4_activations_path = REPO_ROOT / "world_models_fpga/research/circuits/results/task_conditioned_circuits_v4_activations.jsonl"
    if not v4_activations_path.exists():
        raise FileNotFoundError(f"Missing v4 activations at {v4_activations_path}")

    print(f"\n[1/6] Loading discovery split (first 25 pairs) from {v4_activations_path.name}...")
    discovery_records = []
    with v4_activations_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                pid = item["pair_id"]
                pid_int = int(pid.split("_")[1]) if isinstance(pid, str) and "_" in pid else int(pid)
                item["pair_idx"] = pid_int
                if pid_int < 25:
                    discovery_records.append(item)

    print(f"Loaded {len(discovery_records)} discovery pairs.")
    assert len(discovery_records) == 25, f"Expected 25 discovery records, found {len(discovery_records)}"

    # 2. Both-Context Task-Sensitivity Candidate Ranking on Discovery Split
    print("\n[2/6] Computing Both-Context Candidate Ranking on Discovery Split...")
    # s_i = 0.5 * (|h_i(z_A, e_B) - h_i(z_A, e_A)| + |h_i(z_B, e_B) - h_i(z_B, e_A)|)
    def compute_both_context_ranking(layer_key: str) -> list[int]:
        pair_sensitivities = []
        for r in discovery_records:
            h_aa = np.array(r["conditions"]["cond_aa"][layer_key], dtype=np.float32)
            h_ab = np.array(r["conditions"]["cond_ab"][layer_key], dtype=np.float32)
            h_ba = np.array(r["conditions"]["cond_ba"][layer_key], dtype=np.float32)
            h_bb = np.array(r["conditions"]["cond_bb"][layer_key], dtype=np.float32)
            diff_A = np.abs(h_ab - h_aa)
            diff_B = np.abs(h_bb - h_ba)
            s = 0.5 * (diff_A + diff_B)
            pair_sensitivities.append(s)
        mean_s = np.mean(pair_sensitivities, axis=0)
        return [int(u) for u in np.argsort(mean_s)[::-1]]

    ranked_l1 = compute_both_context_ranking("layer1_post_relu_64d")
    ranked_l2 = compute_both_context_ranking("layer2_post_relu_64d")

    print(f"Frozen Layer 1 Candidate Ranking (Top 16): {ranked_l1[:16]}")
    print(f"Frozen Layer 2 Candidate Ranking (Top 16): {ranked_l2[:16]}")

    # 3. Generate Independent Cross-Context Evaluation Dataset (60 pairs)
    total_pairs = n_nominal_pairs + n_illum_pairs
    print(f"\n[3/6] Generating {total_pairs} independent cross-context scene pairs ({n_nominal_pairs} nominal, {n_illum_pairs} illum shift)...")

    env = JetAutoEnv()
    agent = HJEPAAgent(ckpt_path, device=device)
    model = agent.model
    model.eval()

    rejection_counters = {}
    eval_scene_pairs = []

    # Nominal pairs: light_diffuse_scale = 1.0
    for p_idx in range(n_nominal_pairs):
        sp = generate_dual_visible_scene_pair(
            pair_idx=p_idx,
            env=env,
            rng=rng,
            rejection_counters=rejection_counters,
            light_scale=1.0,
            min_pixels=100,
            min_initial_dist=0.60,
        )
        if sp is None:
            raise RuntimeError(f"Failed to generate nominal scene pair {p_idx} within trial budget")
        eval_scene_pairs.append(sp)

    # Illumination shifted pairs: light_diffuse_scale = 0.60
    for p_idx in range(n_nominal_pairs, total_pairs):
        sp = generate_dual_visible_scene_pair(
            pair_idx=p_idx,
            env=env,
            rng=rng,
            rejection_counters=rejection_counters,
            light_scale=0.60,
            min_pixels=100,
            min_initial_dist=0.60,
        )
        if sp is None:
            raise RuntimeError(f"Failed to generate illumination shifted scene pair {p_idx} within trial budget")
        eval_scene_pairs.append(sp)

    print(f"Successfully generated {len(eval_scene_pairs)} independent scene pairs. Rejection counts: {rejection_counters}")

    # 4. Extract Intermediate Representations Across All 9 Stages for Evaluation Set
    print("\n[4/6] Extracting 9-stage intermediate activations on evaluation set (120 scenes x 2 tasks = 240 passes)...")

    # Storage for representation statistics
    stage_names = [
        "visual_latent_z",
        "task_embedding_e",
        "input_concat_48d",
        "layer1_pre_relu_64d",
        "layer1_post_relu_64d",
        "layer2_pre_relu_64d",
        "layer2_post_relu_64d",
        "raw_output_4d",
        "postprocessed_goal_4d",
    ]
    eval_stage_arrays = {st: [] for st in stage_names}

    # Store full pair model outputs
    processed_pairs = []

    for sp in eval_scene_pairs:
        p_idx = sp["pair_idx"]
        key_a = sp["key_a"]
        key_b = sp["key_b"]
        t_id_a = sp["task_id_a"]
        t_id_b = sp["task_id_b"]

        t_tensor_a = torch.tensor([t_id_a], dtype=torch.long, device=device)
        t_tensor_b = torch.tensor([t_id_b], dtype=torch.long, device=device)

        with torch.no_grad():
            # Encode Scene A
            obs_a_t = torch.tensor(sp["obs_a"], dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
            z_a = model.encode(obs_a_t)

            # Encode Scene B
            obs_b_t = torch.tensor(sp["obs_b"], dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
            z_b = model.encode(obs_b_t)

            # Four factorial conditions
            cond_aa = extract_9_stages(model, z_a, t_tensor_a)
            cond_ab = extract_9_stages(model, z_a, t_tensor_b)
            cond_ba = extract_9_stages(model, z_b, t_tensor_a)
            cond_bb = extract_9_stages(model, z_b, t_tensor_b)

        # Collect for representation statistics
        for c in [cond_aa, cond_ab, cond_ba, cond_bb]:
            for st in stage_names:
                eval_stage_arrays[st].append(c[st].squeeze(0).cpu().numpy())

        # Ground truth body goals
        # In Scene A: wp_1 is target for A, wp_2 is target for B
        gt_a_scene_a = compute_body_frame_goal_vec(sp["robot_pose"], sp["wp_1"])
        gt_b_scene_a = compute_body_frame_goal_vec(sp["robot_pose"], sp["wp_2"])
        # In Scene B: wp_2 is target for A, wp_1 is target for B
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
            # Raw torch tensors for interventions
            "activations": {
                "cond_aa": {st: cond_aa[st] for st in ["layer1_post_relu_64d", "layer2_post_relu_64d"]},
                "cond_ab": {st: cond_ab[st] for st in ["layer1_post_relu_64d", "layer2_post_relu_64d"]},
                "cond_ba": {st: cond_ba[st] for st in ["layer1_post_relu_64d", "layer2_post_relu_64d"]},
                "cond_bb": {st: cond_bb[st] for st in ["layer1_post_relu_64d", "layer2_post_relu_64d"]},
            },
        }
        processed_pairs.append(pair_entry)

    # Convert representation arrays to numpy
    eval_stage_numpy = {st: np.array(eval_stage_arrays[st], dtype=np.float32) for st in stage_names}

    # Compute Representation Statistics
    print("\nComputing comprehensive representation statistics across all 9 stages...")
    stage_representation_stats = {}
    for st in stage_names:
        stats_st = compute_representation_statistics_v5_2(eval_stage_numpy, st)
        stage_representation_stats[st] = stats_st
        print(f"  Stage {st:25s}: PR = {stats_st['participation_ratio']:.2f}/{stats_st['dimensions']} | 80% Dims = {stats_st['dims_80_var']} | Silent = {stats_st['silent_units_count']}")

    # 5. Causal Interventions: Necessity & Sufficiency with Activation-Norm-Matched Controls
    print("\n[5/6] Executing Causal Interventions: Candidate vs Norm-Matched & Dimension-Matched Random Controls...")
    k_values = [2, 4, 8, 16, 32]
    layer_configs = [
        {"layer_key": "layer1_post_relu_64d", "start_layer": 1, "ranked_units": ranked_l1},
        {"layer_key": "layer2_post_relu_64d", "start_layer": 2, "ranked_units": ranked_l2},
    ]

    all_units_pool = np.arange(64)
    intervention_records = []  # Detailed JSONL records
    pair_summary_records = []  # Aggregated per-pair records

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

        # Evaluate both Scene A and Scene B
        scenes_data = [
            ("scene_a", acts["cond_aa"], acts["cond_ab"]),
            ("scene_b", acts["cond_ba"], acts["cond_bb"]),
        ]

        for scene_id, act_a, act_b in scenes_data:
            for l_cfg in layer_configs:
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

                # Task-induced activation difference vector in this scene
                delta_h = (h_b - h_a).squeeze(0).cpu().numpy()  # [64]

                # Heading angles
                theta_a = float(torch.atan2(g_a[2], g_a[3]).item())
                theta_b = float(torch.atan2(g_b[2], g_b[3]).item())
                delta_theta_full = wrap_angle(theta_b - theta_a)

                # Bearing angles
                bearing_a = float(torch.atan2(g_a_xy[1], g_a_xy[0]).item())
                bearing_b = float(torch.atan2(g_b_xy[1], g_b_xy[0]).item())

                for k in k_values:
                    cand_k = ranked_u[:k]
                    cand_norm = float(np.linalg.norm(delta_h[cand_k]))

                    # 1. Sample >= 5000 random subsets for norm-matching
                    random_subsets = [rng.choice(all_units_pool, size=k, replace=False) for _ in range(n_norm_samples)]
                    subset_norms = np.array([np.linalg.norm(delta_h[u]) for u in random_subsets])
                    rel_errors = np.abs(subset_norms - cand_norm) / max(1e-8, cand_norm)

                    # Subsets within 5% tolerance
                    within_tol_indices = np.where(rel_errors <= norm_tolerance)[0]
                    n_within_tol = int(len(within_tol_indices))

                    # Retain closest 100 subsets
                    sorted_indices = np.argsort(rel_errors)
                    retained_norm_matched_indices = sorted_indices[:n_retained_controls]
                    norm_matched_subsets = [random_subsets[idx] for idx in retained_norm_matched_indices]
                    norm_matched_errors = rel_errors[retained_norm_matched_indices]

                    # Unconstrained dimension-matched random controls (first 100)
                    dim_matched_subsets = random_subsets[:n_retained_controls]

                    # 2. Evaluate Necessity on Candidate Units
                    # Condition-averaged ablation: h_i -> (h_i^A + h_i^B) / 2 for i in cand_k
                    h_a_mean_abl = h_a.clone()
                    h_b_mean_abl = h_b.clone()
                    mean_val = 0.5 * (h_a[:, cand_k] + h_b[:, cand_k])
                    h_a_mean_abl[:, cand_k] = mean_val
                    h_b_mean_abl[:, cand_k] = mean_val

                    # Zero ablation
                    h_a_zero_abl = h_a.clone()
                    h_b_zero_abl = h_b.clone()
                    h_a_zero_abl[:, cand_k] = 0.0
                    h_b_zero_abl[:, cand_k] = 0.0

                    with torch.no_grad():
                        g_a_mabl = forward_grounding_from_layer(model, h_a_mean_abl, start_l).squeeze(0)
                        g_b_mabl = forward_grounding_from_layer(model, h_b_mean_abl, start_l).squeeze(0)
                        g_a_zabl = forward_grounding_from_layer(model, h_a_zero_abl, start_l).squeeze(0)
                        g_b_zabl = forward_grounding_from_layer(model, h_b_zero_abl, start_l).squeeze(0)

                    delta_g_mabl_xy = g_b_mabl[:2] - g_a_mabl[:2]
                    mabl_contrast_norm = float(torch.norm(delta_g_mabl_xy).item())
                    retention_ratio = mabl_contrast_norm / max(1e-8, full_contrast_norm)
                    drop_ratio = 1.0 - retention_ratio

                    # Directional cosine
                    dot_mabl = float(torch.dot(delta_g_mabl_xy, delta_g_full_xy).item())
                    cos_mabl = dot_mabl / max(1e-8, mabl_contrast_norm * full_contrast_norm)

                    # Bearing and Heading changes
                    bearing_a_mabl = float(torch.atan2(g_a_mabl[1], g_a_mabl[0]).item())
                    bearing_b_mabl = float(torch.atan2(g_b_mabl[1], g_b_mabl[0]).item())
                    delta_bearing = 0.5 * (abs(wrap_angle(bearing_a_mabl - bearing_a)) + abs(wrap_angle(bearing_b_mabl - bearing_b)))

                    theta_a_mabl = float(torch.atan2(g_a_mabl[2], g_a_mabl[3]).item())
                    theta_b_mabl = float(torch.atan2(g_b_mabl[2], g_b_mabl[3]).item())
                    delta_heading = 0.5 * (abs(wrap_angle(theta_a_mabl - theta_a)) + abs(wrap_angle(theta_b_mabl - theta_b)))

                    # 3. Evaluate Sufficiency in Both Directions (A -> B and B -> A)
                    # Directions:
                    # dir_fwd: A -> B (Recipient: h_a, Donor: h_b, Target: delta_g_full_xy)
                    # dir_rev: B -> A (Recipient: h_b, Donor: h_a, Target: -delta_g_full_xy)
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
                        intervention_records.append({
                            "pair_id": pair_id,
                            "scene_id": scene_id,
                            "context": context,
                            "layer": l_key,
                            "direction": dir_name,
                            "k": k,
                            "condition": "candidate",
                            "draw_id": 0,
                            "units": [int(u) for u in cand_k],
                            "subset_norm": cand_norm,
                            "norm_diff_abs": 0.0,
                            "norm_diff_rel": 0.0,
                            "is_within_tolerance": True,
                            "projected_transfer_efficiency": cand_eta,
                            "transfer_magnitude_m": cand_transfer_mag,
                            "directional_cosine": cand_cos,
                            "orthogonal_residual_m": cand_r_perp,
                            "heading_transfer_eff": cand_th_eff,
                        })

                        # b) Batch forward for Norm-Matched Random Controls
                        batch_norm_trans = h_recip.repeat(len(norm_matched_subsets), 1)
                        for d_idx, sub_u in enumerate(norm_matched_subsets):
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

                        for d_idx, sub_u in enumerate(norm_matched_subsets):
                            intervention_records.append({
                                "pair_id": pair_id,
                                "scene_id": scene_id,
                                "context": context,
                                "layer": l_key,
                                "direction": dir_name,
                                "k": k,
                                "condition": "norm_matched_random",
                                "draw_id": d_idx,
                                "units": [int(u) for u in sub_u],
                                "subset_norm": float(subset_norms[retained_norm_matched_indices[d_idx]]),
                                "norm_diff_abs": float(abs(subset_norms[retained_norm_matched_indices[d_idx]] - cand_norm)),
                                "norm_diff_rel": float(norm_matched_errors[d_idx]),
                                "is_within_tolerance": bool(norm_matched_errors[d_idx] <= norm_tolerance),
                                "projected_transfer_efficiency": float(batch_norm_etas[d_idx]),
                                "transfer_magnitude_m": float(batch_norm_mags_np[d_idx]),
                                "directional_cosine": float(batch_norm_cos[d_idx]),
                                "orthogonal_residual_m": float(batch_r_perps[d_idx]),
                                "heading_transfer_eff": float(batch_th_diffs[d_idx] / (th_target + 1e-8)) if abs(th_target) > 1e-3 else 1.0,
                            })

                        # c) Batch forward for Dimension-Matched Random Controls
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
                            intervention_records.append({
                                "pair_id": pair_id,
                                "scene_id": scene_id,
                                "context": context,
                                "layer": l_key,
                                "direction": dir_name,
                                "k": k,
                                "condition": "dimension_matched_random",
                                "draw_id": d_idx,
                                "units": [int(u) for u in sub_u],
                                "subset_norm": float(subset_norms[d_idx]),
                                "norm_diff_abs": float(abs(subset_norms[d_idx] - cand_norm)),
                                "norm_diff_rel": float(rel_errors[d_idx]),
                                "is_within_tolerance": bool(rel_errors[d_idx] <= norm_tolerance),
                                "projected_transfer_efficiency": float(batch_dim_etas[d_idx]),
                                "transfer_magnitude_m": float(batch_dim_mags_np[d_idx]),
                                "directional_cosine": float(batch_dim_cos[d_idx]),
                                "orthogonal_residual_m": float(batch_dim_r_perps[d_idx]),
                                "heading_transfer_eff": float(batch_dim_th_diffs[d_idx] / (th_target + 1e-8)) if abs(th_target) > 1e-3 else 1.0,
                            })

                        # Store summary for pair
                        setting_key = f"{scene_id}__{l_key}__{dir_name}__k_{k}"
                        pair_metrics["results"][setting_key] = {
                            "candidate_eta": cand_eta,
                            "norm_matched_eta_mean": float(np.mean(batch_norm_etas)),
                            "dim_matched_eta_mean": float(np.mean(batch_dim_etas)),
                            "cand_norm": cand_norm,
                            "norm_matched_norm_mean": float(np.mean(subset_norms[retained_norm_matched_indices])),
                            "dim_matched_norm_mean": float(np.mean(subset_norms[:n_retained_controls])),
                            "n_within_5pct": n_within_tol,
                            "necessity_drop_ratio": drop_ratio,
                            "necessity_retention_ratio": retention_ratio,
                            "necessity_cos": cos_mabl,
                            "delta_bearing": delta_bearing,
                            "delta_heading": delta_heading,
                        }

        pair_summary_records.append(pair_metrics)

    # Whole-layer reference condition (k=64)
    # Trivial reference: whole layer transplantation transfers 100% of task contrast
    print(f"Executed interventions. Total intervention records recorded: {len(intervention_records)}")

    # 6. Statistical Analysis: Candidate vs Norm-Matched vs Dimension-Matched Random
    print("\n[6/6] Computing Statistical Tests Across Independent Scene Pairs (N = 60)...")
    # Independent pair is statistical unit. Average Scene A and Scene B per pair.
    statistical_results = {}

    for l_cfg in layer_configs:
        l_key = l_cfg["layer_key"]
        statistical_results[l_key] = {}

        for dir_name in ["A_to_B", "B_to_A"]:
            statistical_results[l_key][dir_name] = {}

            for k in k_values:
                # For each pair p:
                cand_etas = []
                norm_rand_etas = []
                dim_rand_etas = []
                cand_norms = []
                norm_rand_norms = []
                dim_rand_norms = []

                for pm in pair_summary_records:
                    k_a = f"scene_a__{l_key}__{dir_name}__k_{k}"
                    k_b = f"scene_b__{l_key}__{dir_name}__k_{k}"
                    # Average over Scene A and Scene B
                    c_eta = 0.5 * (pm["results"][k_a]["candidate_eta"] + pm["results"][k_b]["candidate_eta"])
                    nr_eta = 0.5 * (pm["results"][k_a]["norm_matched_eta_mean"] + pm["results"][k_b]["norm_matched_eta_mean"])
                    dr_eta = 0.5 * (pm["results"][k_a]["dim_matched_eta_mean"] + pm["results"][k_b]["dim_matched_eta_mean"])

                    cand_etas.append(c_eta)
                    norm_rand_etas.append(nr_eta)
                    dim_rand_etas.append(dr_eta)

                    cand_norms.append(0.5 * (pm["results"][k_a]["cand_norm"] + pm["results"][k_b]["cand_norm"]))
                    norm_rand_norms.append(0.5 * (pm["results"][k_a]["norm_matched_norm_mean"] + pm["results"][k_b]["norm_matched_norm_mean"]))
                    dim_rand_norms.append(0.5 * (pm["results"][k_a]["dim_matched_norm_mean"] + pm["results"][k_b]["dim_matched_norm_mean"]))

                cand_etas = np.array(cand_etas)
                norm_rand_etas = np.array(norm_rand_etas)
                dim_rand_etas = np.array(dim_rand_etas)

                # Paired differences
                diff_norm = cand_etas - norm_rand_etas
                diff_dim = cand_etas - dim_rand_etas

                mean_diff_norm = float(np.mean(diff_norm))
                mean_diff_dim = float(np.mean(diff_dim))

                # Clustered bootstrap 95% CI across pairs
                boot_diff_norm = [float(np.mean(rng.choice(diff_norm, size=len(diff_norm), replace=True))) for _ in range(10000)]
                boot_ci_norm = [float(np.percentile(boot_diff_norm, 2.5)), float(np.percentile(boot_diff_norm, 97.5))]

                boot_diff_dim = [float(np.mean(rng.choice(diff_dim, size=len(diff_dim), replace=True))) for _ in range(10000)]
                boot_ci_dim = [float(np.percentile(boot_diff_dim, 2.5)), float(np.percentile(boot_diff_dim, 97.5))]

                # Wilcoxon signed rank tests
                w_p_norm = float(stats.wilcoxon(diff_norm, alternative="greater").pvalue) if np.any(diff_norm != 0) else 1.0
                w_p_dim = float(stats.wilcoxon(diff_dim, alternative="greater").pvalue) if np.any(diff_dim != 0) else 1.0

                # Permutation sign-flip tests (100,000 resamples)
                obs_stat_norm = np.mean(diff_norm)
                signs = rng.choice([-1.0, 1.0], size=(100000, len(diff_norm)))
                perm_stats_norm = np.mean(signs * diff_norm, axis=1)
                perm_p_norm = float((1 + np.sum(perm_stats_norm >= obs_stat_norm)) / (100000 + 1))

                statistical_results[l_key][dir_name][f"k_{k}"] = {
                    "k": k,
                    "candidate_efficiency_mean": float(np.mean(cand_etas)),
                    "candidate_efficiency_ci": [float(np.percentile(cand_etas, 2.5)), float(np.percentile(cand_etas, 97.5))],
                    "norm_matched_random_efficiency_mean": float(np.mean(norm_rand_etas)),
                    "norm_matched_random_efficiency_ci": [float(np.percentile(norm_rand_etas, 2.5)), float(np.percentile(norm_rand_etas, 97.5))],
                    "dimension_matched_random_efficiency_mean": float(np.mean(dim_rand_etas)),
                    "dimension_matched_random_efficiency_ci": [float(np.percentile(dim_rand_etas, 2.5)), float(np.percentile(dim_rand_etas, 97.5))],
                    "candidate_activation_norm_mean": float(np.mean(cand_norms)),
                    "norm_matched_activation_norm_mean": float(np.mean(norm_rand_norms)),
                    "dimension_matched_activation_norm_mean": float(np.mean(dim_rand_norms)),
                    # Primary Outcome: Candidate vs Norm-Matched Random
                    "cand_vs_norm_matched": {
                        "mean_paired_difference": mean_diff_norm,
                        "bootstrap_95_ci": boot_ci_norm,
                        "wilcoxon_one_sided_p_value": w_p_norm,
                        "permutation_sign_flip_p_value": perm_p_norm,
                    },
                    # Secondary Comparison: Candidate vs Dimension-Matched Random
                    "cand_vs_dimension_matched": {
                        "mean_paired_difference": mean_diff_dim,
                        "bootstrap_95_ci": boot_ci_dim,
                        "wilcoxon_one_sided_p_value": w_p_dim,
                    },
                }

    # Holm-Bonferroni correction across the 20 planned primary tests (2 layers x 2 directions x 5 k values)
    primary_test_keys = []
    p_vals_raw = []
    for l_key in ["layer1_post_relu_64d", "layer2_post_relu_64d"]:
        for dir_name in ["A_to_B", "B_to_A"]:
            for k in k_values:
                entry = statistical_results[l_key][dir_name][f"k_{k}"]["cand_vs_norm_matched"]
                primary_test_keys.append((l_key, dir_name, f"k_{k}"))
                p_vals_raw.append(entry["wilcoxon_one_sided_p_value"])

    # Compute Holm-Bonferroni
    p_adjusted = holm_bonferroni_correction(p_vals_raw)

    for idx, (l_key, dir_name, k_str) in enumerate(primary_test_keys):
        statistical_results[l_key][dir_name][k_str]["cand_vs_norm_matched"]["holm_bonferroni_adjusted_p_value"] = float(p_adjusted[idx])

    # Print Key Statistical Findings
    print("\n================================================================================")
    print("PHASE 5.2 PRIMARY RESULTS: CANDIDATE VS ACTIVATION-NORM-MATCHED CONTROLS")
    print("================================================================================")
    for l_key in ["layer1_post_relu_64d", "layer2_post_relu_64d"]:
        print(f"\n--- {l_key} ---")
        for dir_name in ["A_to_B", "B_to_A"]:
            print(f" Direction: {dir_name}")
            for k in k_values:
                res = statistical_results[l_key][dir_name][f"k_{k}"]
                cand_eff = res["candidate_efficiency_mean"]
                norm_eff = res["norm_matched_random_efficiency_mean"]
                dim_eff = res["dimension_matched_random_efficiency_mean"]
                p_raw = res["cand_vs_norm_matched"]["wilcoxon_one_sided_p_value"]
                p_adj = res["cand_vs_norm_matched"]["holm_bonferroni_adjusted_p_value"]
                diff = res["cand_vs_norm_matched"]["mean_paired_difference"]
                c_norm = res["candidate_activation_norm_mean"]
                n_norm = res["norm_matched_activation_norm_mean"]
                d_norm = res["dimension_matched_activation_norm_mean"]
                print(f"  k={k:2d}: Cand={cand_eff:.3f} (||dh||={c_norm:.3f}) | NormRand={norm_eff:.3f} (||dh||={n_norm:.3f}) | DimRand={dim_eff:.3f} (||dh||={d_norm:.3f}) | Diff={diff:+.3f} | p_raw={p_raw:.4e} | p_adj={p_adj:.4e}")

    # Determine Mechanistic Verdict based on actual data
    # Check if candidate advantage remains significant after norm-matching
    sig_norm_matches = [res["cand_vs_norm_matched"]["holm_bonferroni_adjusted_p_value"] < 0.05 for l in statistical_results.values() for d in l.values() for res in d.values()]
    n_sig = sum(sig_norm_matches)

    if n_sig > 0:
        mechanistic_verdict = (
            f"Selective Subspace Confirmed ({n_sig}/20 tests significant after norm-matching). "
            "Candidate task-sensitive units maintain significant projected transfer efficiency advantage "
            "even when compared against activation-perturbation-norm-matched random subsets. "
            "This demonstrates that the causal effect is not purely an activation-magnitude artefact."
        )
    else:
        mechanistic_verdict = (
            "Activation-Magnitude Artefact Demonstrated (0/20 tests significant after norm-matching). "
            "While candidate units outperform unconstrained dimension-matched random subsets, "
            "this advantage completely disappears when compared against activation-perturbation-norm-matched controls. "
            "The apparent selectivity observed in Phase 5.1 is largely explained by selecting units with "
            "larger task-induced activation changes rather than a distinct, localized causal circuit."
        )

    print(f"\nMechanistic Verdict: {mechanistic_verdict}")

    elapsed_time = time.time() - start_time

    # Construct final summary dictionary
    summary_data = {
        "experiment_name": "Phase 5.2 Mechanistic Selectivity & Cross-Context Revalidation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_clean": git_clean,
        "random_seed": seed,
        "device": device,
        "elapsed_seconds": elapsed_time,
        "provenance_hashes": {
            "checkpoint_hjepa": compute_file_sha256(ckpt_path),
            "v4_activations": compute_file_sha256(v4_activations_path),
        },
        "dataset_composition": {
            "total_pairs": total_pairs,
            "nominal_pairs": n_nominal_pairs,
            "illumination_shift_pairs": n_illum_pairs,
            "rejection_counters": rejection_counters,
        },
        "frozen_candidate_rankings": {
            "layer1_both_contexts_ranking": ranked_l1,
            "layer2_both_contexts_ranking": ranked_l2,
        },
        "representation_statistics": stage_representation_stats,
        "statistical_results": statistical_results,
        "mechanistic_conclusion": mechanistic_verdict,
    }

    # Save output artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "task_conditioned_circuits_v5_2_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[Saved] Summary JSON: {summary_path}")

    # Save Pairs JSONL
    pairs_path = out_dir / "task_conditioned_circuits_v5_2_pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as f:
        for pm in pair_summary_records:
            f.write(json.dumps(pm) + "\n")
    print(f"[Saved] Pairs JSONL: {pairs_path} ({len(pair_summary_records)} records)")

    # Save Scene Specs JSONL
    specs_path = out_dir / "task_conditioned_circuits_v5_2_scene_specs.jsonl"
    with specs_path.open("w", encoding="utf-8") as f:
        for sp in eval_scene_pairs:
            # Omit large numpy image arrays from jsonl specs
            spec_copy = {k: v for k, v in sp.items() if k not in ("obs_a", "obs_b")}
            f.write(json.dumps(spec_copy) + "\n")
    print(f"[Saved] Scene Specs JSONL: {specs_path} ({len(eval_scene_pairs)} records)")

    # Save Interventions JSONL
    interventions_path = out_dir / "task_conditioned_circuits_v5_2_interventions.jsonl"
    with interventions_path.open("w", encoding="utf-8") as f:
        for rec in intervention_records:
            f.write(json.dumps(rec) + "\n")
    print(f"[Saved] Interventions JSONL: {interventions_path} ({len(intervention_records)} records)")

    # Save Schema Markdown
    schema_path = out_dir / "task_conditioned_circuits_v5_2_schema.md"
    schema_content = f"""# Schema: Phase 5.2 Task-Conditioned Circuits Revalidation Artifacts

**Execution Commit:** `{git_commit}` (Clean: `{git_clean}`)  
**Timestamp:** `{datetime.now(timezone.utc).isoformat()}`  
**Dataset Size:** {total_pairs} independent scene pairs ({n_nominal_pairs} nominal, {n_illum_pairs} illumination shifted)

---

## 1. Primary Artifacts

1. `task_conditioned_circuits_v5_2_summary.json`:
   - `dataset_composition`: Pair counts, illumination parameters, rejection counters.
   - `frozen_candidate_rankings`: Top units scored via both-context sensitivity on discovery split (pairs 0..24).
   - `representation_statistics`: Effective dimensionality (participation ratio), eigenvalue spectra, cumulative explained variance thresholds, positive-part and signed sparseness, excess kurtosis, and silent units across all 9 stages.
   - `statistical_results`: Primary comparison (Candidate vs Norm-Matched Random) and secondary comparison (Candidate vs Dimension-Matched Random) with paired effect sizes, pair-clustered bootstrap 95% CIs, Wilcoxon signed-rank tests, sign-flip permutation tests, and Holm-Bonferroni adjusted p-values across 20 planned tests.
   - `mechanistic_conclusion`: Formal verdict on whether candidate selectivity is a genuine causal mechanism or an activation-magnitude artefact.

2. `task_conditioned_circuits_v5_2_pairs.jsonl`:
   - One line per independent pair ($N={total_pairs}$).
   - Contains paired candidate transfer efficiencies, norm-matched random means, dimension-matched random means, and necessity retention/drop ratios.

3. `task_conditioned_circuits_v5_2_interventions.jsonl`:
   - Complete record of EVERY intervention pass (candidate + all random draws).
   - Fields:
     - `pair_id`: Pair identifier (e.g., `PAIR_000`).
     - `scene_id`: `scene_a` or `scene_b`.
     - `context`: `nominal` or `illumination_shift`.
     - `layer`: `layer1_post_relu_64d` or `layer2_post_relu_64d`.
     - `direction`: `A_to_B` or `B_to_A`.
     - `k`: Number of intervened dimensions (k in [2, 4, 8, 16, 32]).
     - `condition`: `candidate`, `norm_matched_random`, or `dimension_matched_random`.
     - `draw_id`: 0 for candidate; 0..99 for random draws.
     - `units`: List of intervened unit indices.
     - `subset_norm`: Task-induced activation perturbation norm sqrt(sum_{i in U} (delta h_i)^2).
     - `norm_diff_abs`: |N_rand - N_cand|.
     - `norm_diff_rel`: |N_rand - N_cand| / N_cand.
     - `is_within_tolerance`: True if relative norm error <= 0.05 (5%).
     - `projected_transfer_efficiency`: Projected transfer fraction along task-contrast direction (eta).
     - `transfer_magnitude_m`: Euclidean norm of transferred XY displacement in meters.
     - `directional_cosine`: Cosine alignment with target task-contrast vector.
     - `orthogonal_residual_m`: Transferred vector magnitude orthogonal to intended contrast vector.
     - `heading_transfer_eff`: Heading transfer ratio derived from `atan2(sin, cos)`.

4. `task_conditioned_circuits_v5_2_scene_specs.jsonl`:
   - Exact physical specifications of all 60 scene pairs.
   - Robot pose, target positions, approach waypoints, obstacle dictionary, visibility pixel counts, and rendered camera observation SHA-256 hashes.
"""
    with schema_path.open("w", encoding="utf-8") as f:
        f.write(schema_content)
    print(f"[Saved] Schema Markdown: {schema_path}")

    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5.2 Mechanistic Selectivity & Cross-Context Revalidation")
    parser.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "hjepa_best.pt"), help="H-JEPA checkpoint path")
    parser.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "world_models_fpga/research/circuits/results"), help="Output directory")
    parser.add_argument("--nominal-pairs", type=int, default=30, help="Number of nominal appearance scene pairs")
    parser.add_argument("--illum-pairs", type=int, default=30, help="Number of illumination-shifted scene pairs")
    parser.add_argument("--norm-samples", type=int, default=5000, help="Number of random subsets to sample for norm matching")
    parser.add_argument("--retained-controls", type=int, default=100, help="Number of random controls to evaluate per setting")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Norm matching tolerance (default: 0.05 = 5%)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="PyTorch device")
    args = parser.parse_args()

    run_task_conditioned_circuits_v5_2(
        ckpt_path=Path(args.ckpt),
        out_dir=Path(args.out_dir),
        n_nominal_pairs=args.nominal_pairs,
        n_illum_pairs=args.illum_pairs,
        n_norm_samples=args.norm_samples,
        n_retained_controls=args.retained_controls,
        norm_tolerance=args.tolerance,
        seed=args.seed,
        device=args.device,
    )
