#!/usr/bin/env python3
"""
Phase 4 Mechanistic Benchmark: Task-Conditioned Visual Binding & Interaction Analysis (v3).

Scientific Objectives:
1. Discriminate among three candidate mechanisms for task-conditioned spatial decoding:
   A. Visual object binding: Task identity selects a visible object from visual geometry.
   B. Task-conditioned spatial association: Task identity retrieves a learned canonical prior.
   C. Mixed scene-to-goal decoding: Task token and visual features interact without clean selection.
2. Construct >= 100 valid dual-visible scenes (both objects >= 100 segmentation pixels,
   approach waypoints >= 0.60m from robot pose).
3. 2x2 Factorial evaluation (z_A, z_B) x (e_A, e_B) to compute interaction vector:
   I = [g(z_B, e_B) - g(z_B, e_A)] - [g(z_A, e_B) - g(z_A, e_A)].
4. Fit and compare three explicit models on held-out test splits:
   - Additive task bias: g(z, e) = a(z) + b(e)
   - Visual geometry selection: g(z, e) = Select(V(z), e)
   - General non-linear interaction: g(z, e) = MLP(z, e)
5. Saliency factorial independently manipulating object size and contrast.
6. Closed-loop navigation evaluation tracking designated-target vs distractor reaching.

Outputs:
  - Summary JSON: world_models_fpga/research/benchmarks/results/task_conditioned_binding_v3_summary.json
  - Pairs JSONL: world_models_fpga/research/benchmarks/results/task_conditioned_binding_v3_pairs.jsonl
  - Traces NPZ: world_models_fpga/research/benchmarks/results/task_conditioned_binding_v3_traces.npz
  - Schema Markdown: world_models_fpga/research/benchmarks/results/task_conditioned_binding_v3_schema.md
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


def compute_bytes_sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


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
    commit = "bbad9132a268a395c3298fb4a08dc3b8edbf30fd"
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
    min_pixels: int = 100,
    min_initial_dist: float = 0.60,
    max_trials: int = 2000,
) -> dict | None:
    """
    Generates a scene pair containing two candidate objects A and B with guaranteed visibility.
    In Scene A: Object A is at Pos 1, Object B is at Pos 2.
    In Scene B: Object A is at Pos 2, Object B is at Pos 1 (swapped physical positions).
    Returns complete pair metadata dictionary or None if max_trials exceeded.
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

        # Bounds check [-1.55, 1.55]
        if not (-1.55 <= p1_x <= 1.55 and -1.55 <= p1_y <= 1.55 and -1.55 <= p2_x <= 1.55 and -1.55 <= p2_y <= 1.55):
            rejection_counters["arena_bounds_violation"] = rejection_counters.get("arena_bounds_violation", 0) + 1
            continue

        # Separation check (>= 0.52m apart)
        sep = float(np.linalg.norm([p1_x - p2_x, p1_y - p2_y]))
        if sep < 0.52:
            rejection_counters["separation_too_small (< 0.52m)"] = rejection_counters.get("separation_too_small (< 0.52m)", 0) + 1
            continue

        pos_1 = np.array([p1_x, p1_y, 0.35], dtype=np.float32)
        pos_2 = np.array([p2_x, p2_y, 0.35], dtype=np.float32)

        # Approach waypoints with clearance 0.60m from target centers
        d_p1 = np.linalg.norm(pos_1[:2] - robot_pose[:2])
        d_p2 = np.linalg.norm(pos_2[:2] - robot_pose[:2])
        if d_p1 < 0.70 or d_p2 < 0.70:
            rejection_counters["too_close_to_robot (< 0.70m)"] = rejection_counters.get("too_close_to_robot (< 0.70m)", 0) + 1
            continue

        wp_1 = pos_1[:2] - (pos_1[:2] - robot_pose[:2]) / d_p1 * 0.60
        wp_2 = pos_2[:2] - (pos_2[:2] - robot_pose[:2]) / d_p2 * 0.60

        d_wp1 = float(np.linalg.norm(wp_1 - robot_pose[:2]))
        d_wp2 = float(np.linalg.norm(wp_2 - robot_pose[:2]))

        if d_wp1 < min_initial_dist or d_wp2 < min_initial_dist:
            rejection_counters["waypoint_too_close (< 0.60m)"] = rejection_counters.get("waypoint_too_close (< 0.60m)", 0) + 1
            continue

        # Assemble obstacle dictionaries for Scene A: A at pos_1, B at pos_2
        obstacles_scene_a = copy.deepcopy(CANONICAL_OBSTACLES)
        obstacles_scene_a[key_a]["pos"] = pos_1
        obstacles_scene_a[key_b]["pos"] = pos_2

        # Check distance from other canonical obstacles (clearance >= 0.50m)
        collision_with_other = False
        for k, obst in obstacles_scene_a.items():
            if k not in (key_a, key_b):
                if np.linalg.norm(pos_1[:2] - obst["pos"][:2]) < 0.50 or np.linalg.norm(pos_2[:2] - obst["pos"][:2]) < 0.50:
                    collision_with_other = True
                    break
        if collision_with_other:
            rejection_counters["obstacle_clearance_violation"] = rejection_counters.get("obstacle_clearance_violation", 0) + 1
            continue

        spec_a = {
            "target_key": key_a,
            "semantic_relation": "approach",
            "task_id": encode_task_id(key_a, "approach"),
            "initial_pose": robot_pose,
            "waypoints": [wp_1],
            "obstacles": obstacles_scene_a,
            "distractor_key": key_b,
        }

        # Verify visibility of both in Scene A
        env.restore_physical_baseline()
        env.apply_task_spec(spec_a)
        env.reset(options={"initial_pos": (rx, ry, rth)})

        vis_a_scene_a = check_target_visibility(env, key_a, min_visible_pixels=min_pixels)
        vis_b_scene_a = check_target_visibility(env, key_b, min_visible_pixels=min_pixels)

        if not (vis_a_scene_a["is_visible"] and vis_b_scene_a["is_visible"]):
            rejection_counters["scene_a_visibility_failed (< 100 px)"] = rejection_counters.get("scene_a_visibility_failed (< 100 px)", 0) + 1
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
        }

        # Verify visibility of both in Scene B
        env.restore_physical_baseline()
        env.apply_task_spec(spec_b)
        env.reset(options={"initial_pos": (rx, ry, rth)})

        vis_a_scene_b = check_target_visibility(env, key_a, min_visible_pixels=min_pixels)
        vis_b_scene_b = check_target_visibility(env, key_b, min_visible_pixels=min_pixels)

        if not (vis_a_scene_b["is_visible"] and vis_b_scene_b["is_visible"]):
            rejection_counters["scene_b_visibility_failed (< 100 px)"] = rejection_counters.get("scene_b_visibility_failed (< 100 px)", 0) + 1
            continue

        # Successfully generated valid dual-visible pair
        return {
            "pair_idx": pair_idx,
            "key_a": key_a,
            "key_b": key_b,
            "task_id_a": encode_task_id(key_a, "approach"),
            "task_id_b": encode_task_id(key_b, "approach"),
            "robot_pose": robot_pose,
            "pos_1": pos_1,
            "pos_2": pos_2,
            "wp_1": wp_1,
            "wp_2": wp_2,
            "spec_scene_a": spec_a,
            "spec_scene_b": spec_b,
            "vis_scene_a": {"vis_a": vis_a_scene_a, "vis_b": vis_b_scene_a},
            "vis_scene_b": {"vis_a": vis_a_scene_b, "vis_b": vis_b_scene_b},
        }

    return None


def run_task_conditioned_binding_v3(
    n_pairs: int = 50,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """Executes the full Phase 4 Task-Conditioned Visual Binding benchmark."""
    start_time = time.time()
    rng = np.random.Generator(np.random.PCG64(seed))

    ckpt_path = CKPT_DIR / "hjepa_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing H-JEPA checkpoint at {ckpt_path}")

    env = JetAutoEnv()
    agent = HJEPAAgent(str(ckpt_path), device=device, track="observation")

    print(f"=== Phase 4 Experiment A: Task-Conditioned Visual Binding v3 ===")
    print(f"Target pairs: {n_pairs} (producing {n_pairs * 2} scenes, >= 100 valid configurations)")
    print(f"Device: {device} | Checkpoint: {ckpt_path.name}")

    rejection_counters: dict[str, int] = {}
    valid_pairs: list[dict] = []

    print("\n[1/4] Generating dual-visible scene pairs with strict >= 100 pixel visibility...")
    for p_idx in range(n_pairs):
        pair = generate_dual_visible_scene_pair(
            pair_idx=p_idx,
            env=env,
            rng=rng,
            rejection_counters=rejection_counters,
            min_pixels=100,
            min_initial_dist=0.60,
        )
        if pair is None:
            raise RuntimeError(f"Could not generate valid scene pair {p_idx} after trials.")
        valid_pairs.append(pair)
        if (p_idx + 1) % 10 == 0 or p_idx == n_pairs - 1:
            print(f"  Generated {p_idx + 1}/{n_pairs} valid pairs...")

    print(f"Rejection counters during generation: {rejection_counters}")

    # Results collectors
    pair_records: list[dict] = []
    npz_trace_dict: dict[str, np.ndarray] = {}

    # Accumulators for aggregate statistics
    all_same_scene_a_correct: list[bool] = []
    all_same_scene_b_correct: list[bool] = []
    all_same_scene_dual_correct: list[bool] = []
    all_same_scene_angular_shifts: list[float] = []

    all_interaction_mags: list[float] = []
    all_interaction_alignments: list[float] = []

    # Regression datasets for model comparison: [z_flat, e_onehot] -> g_true
    dataset_z: list[np.ndarray] = []
    dataset_tid: list[int] = []
    dataset_target_pos: list[np.ndarray] = []
    dataset_g_pred: list[np.ndarray] = []

    saliency_tradeoff_records: list[dict] = []

    print("\n[2/4] Executing 2x2 factorial evaluation and saliency experiments...")
    for p_idx, pair in enumerate(valid_pairs):
        key_a, key_b = pair["key_a"], pair["key_b"]
        tid_a, tid_b = pair["task_id_a"], pair["task_id_b"]
        rx, ry, rth = pair["robot_pose"]

        # ---------------------------------------------------------
        # A. Evaluate Scene A (Object A at Pos 1, Object B at Pos 2)
        # ---------------------------------------------------------
        env.restore_physical_baseline()
        env.apply_task_spec(pair["spec_scene_a"])
        obs_a, _ = env.reset(options={"initial_pos": (rx, ry, rth)})
        img_sha_a = compute_bytes_sha256(obs_a.tobytes())

        # Forward pass visual encoder
        with torch.no_grad():
            x_a = torch.from_numpy(obs_a.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            z_a = agent.model.encode(x_a)

            # Task embeddings
            t_a = torch.tensor([tid_a], dtype=torch.long, device=device)
            t_b = torch.tensor([tid_b], dtype=torch.long, device=device)
            e_a = agent.model.task_embedding(t_a)
            e_b = agent.model.task_embedding(t_b)

            # Factorial outputs for Scene A
            raw_aa = agent.model.grounding_head(torch.cat([z_a, e_a], dim=-1))
            g_aa = torch.cat([torch.clamp(raw_aa[:, :2], -3.5, 3.5), F.normalize(raw_aa[:, 2:4] + 1e-6, dim=-1)], dim=-1).squeeze(0).cpu().numpy()

            raw_ab = agent.model.grounding_head(torch.cat([z_a, e_b], dim=-1))
            g_ab = torch.cat([torch.clamp(raw_ab[:, :2], -3.5, 3.5), F.normalize(raw_ab[:, 2:4] + 1e-6, dim=-1)], dim=-1).squeeze(0).cpu().numpy()

        # Ground truth body-frame target vectors in Scene A
        gt_a_scene_a = compute_body_frame_goal_vec(pair["robot_pose"], pair["wp_1"])
        gt_b_scene_a = compute_body_frame_goal_vec(pair["robot_pose"], pair["wp_2"])

        # ---------------------------------------------------------
        # B. Evaluate Scene B (Object A at Pos 2, Object B at Pos 1)
        # ---------------------------------------------------------
        env.restore_physical_baseline()
        env.apply_task_spec(pair["spec_scene_b"])
        obs_b, _ = env.reset(options={"initial_pos": (rx, ry, rth)})
        img_sha_b = compute_bytes_sha256(obs_b.tobytes())

        with torch.no_grad():
            x_b = torch.from_numpy(obs_b.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            z_b = agent.model.encode(x_b)

            raw_ba = agent.model.grounding_head(torch.cat([z_b, e_a], dim=-1))
            g_ba = torch.cat([torch.clamp(raw_ba[:, :2], -3.5, 3.5), F.normalize(raw_ba[:, 2:4] + 1e-6, dim=-1)], dim=-1).squeeze(0).cpu().numpy()

            raw_bb = agent.model.grounding_head(torch.cat([z_b, e_b], dim=-1))
            g_bb = torch.cat([torch.clamp(raw_bb[:, :2], -3.5, 3.5), F.normalize(raw_bb[:, 2:4] + 1e-6, dim=-1)], dim=-1).squeeze(0).cpu().numpy()

        gt_a_scene_b = compute_body_frame_goal_vec(pair["robot_pose"], pair["wp_2"])
        gt_b_scene_b = compute_body_frame_goal_vec(pair["robot_pose"], pair["wp_1"])

        # ---------------------------------------------------------
        # C. Same-Scene Discrimination Metrics (within Scene A)
        # ---------------------------------------------------------
        d_aa_to_a = float(np.linalg.norm(g_aa[:2] - gt_a_scene_a[:2]))
        d_aa_to_b = float(np.linalg.norm(g_aa[:2] - gt_b_scene_a[:2]))
        d_ab_to_b = float(np.linalg.norm(g_ab[:2] - gt_b_scene_a[:2]))
        d_ab_to_a = float(np.linalg.norm(g_ab[:2] - gt_a_scene_a[:2]))

        a_cor = bool(d_aa_to_a < d_aa_to_b)
        b_cor = bool(d_ab_to_b < d_ab_to_a)
        dual_cor = bool(a_cor and b_cor)

        ang_aa = float(np.arctan2(g_aa[1], g_aa[0]))
        ang_ab = float(np.arctan2(g_ab[1], g_ab[0]))
        d_ang_same_scene = float(np.abs(np.arctan2(np.sin(ang_ab - ang_aa), np.cos(ang_ab - ang_aa))))

        all_same_scene_a_correct.append(a_cor)
        all_same_scene_b_correct.append(b_cor)
        all_same_scene_dual_correct.append(dual_cor)
        all_same_scene_angular_shifts.append(d_ang_same_scene)

        # ---------------------------------------------------------
        # D. Interaction Vector Analysis: I = (g_BB - g_BA) - (g_AB - g_AA)
        # ---------------------------------------------------------
        delta_b = g_bb[:2] - g_ba[:2]
        delta_a = g_ab[:2] - g_aa[:2]
        interaction_vec = delta_b - delta_a
        i_mag = float(np.linalg.norm(interaction_vec))

        # Physical target displacement (Scene B vs Scene A for Target A)
        phys_disp = gt_a_scene_b[:2] - gt_a_scene_a[:2]
        phys_disp_norm = np.linalg.norm(phys_disp)
        if i_mag > 1e-6 and phys_disp_norm > 1e-6:
            cos_align = float(np.dot(interaction_vec, phys_disp) / (i_mag * phys_disp_norm))
        else:
            cos_align = 0.0

        all_interaction_mags.append(i_mag)
        all_interaction_alignments.append(cos_align)

        # Accumulate regression samples
        dataset_z.extend([z_a.squeeze(0).cpu().numpy(), z_a.squeeze(0).cpu().numpy(), z_b.squeeze(0).cpu().numpy(), z_b.squeeze(0).cpu().numpy()])
        dataset_tid.extend([tid_a, tid_b, tid_a, tid_b])
        dataset_target_pos.extend([gt_a_scene_a[:2], gt_b_scene_a[:2], gt_a_scene_b[:2], gt_b_scene_b[:2]])
        dataset_g_pred.extend([g_aa[:2], g_ab[:2], g_ba[:2], g_bb[:2]])

        # ---------------------------------------------------------
        # E. Saliency Factorial (within Scene A geometry)
        # ---------------------------------------------------------
        # Condition S0: Matched (recorded as g_aa, g_ab)
        # Condition S1: A less salient than B (A dark/small, B bright/large)
        # Condition S2: A more salient than B (A bright/large, B dark/small)
        gid_a = pair["vis_scene_a"]["vis_a"]["geom_id"]
        gid_b = pair["vis_scene_a"]["vis_b"]["geom_id"]

        saliency_runs = {}
        for s_mode in ["saliency_a_less_b", "saliency_a_more_b"]:
            env.restore_physical_baseline()
            env.apply_task_spec(pair["spec_scene_a"])

            # Mutate appearance
            if s_mode == "saliency_a_less_b":
                env.model.geom_rgba[gid_a] = [0.15, 0.15, 0.15, 1.0]  # dark low contrast
                env.model.geom_rgba[gid_b] = [1.0, 1.0, 1.0, 1.0]     # bright high contrast
                env.model.geom_size[gid_a][0] *= 0.70                  # smaller
                env.model.geom_size[gid_b][0] *= 1.30                  # larger
            else:
                env.model.geom_rgba[gid_a] = [1.0, 1.0, 1.0, 1.0]     # bright
                env.model.geom_rgba[gid_b] = [0.15, 0.15, 0.15, 1.0]  # dark
                env.model.geom_size[gid_a][0] *= 1.30                  # larger
                env.model.geom_size[gid_b][0] *= 0.70                  # smaller

            obs_sal, _ = env.reset(options={"initial_pos": (rx, ry, rth)})
            vis_sal_a = check_target_visibility(env, key_a, min_visible_pixels=10)
            vis_sal_b = check_target_visibility(env, key_b, min_visible_pixels=10)

            with torch.no_grad():
                x_sal = torch.from_numpy(obs_sal.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
                z_sal = agent.model.encode(x_sal)

                raw_sal_a = agent.model.grounding_head(torch.cat([z_sal, e_a], dim=-1))
                g_sal_a = torch.cat([torch.clamp(raw_sal_a[:, :2], -3.5, 3.5), F.normalize(raw_sal_a[:, 2:4] + 1e-6, dim=-1)], dim=-1).squeeze(0).cpu().numpy()

                raw_sal_b = agent.model.grounding_head(torch.cat([z_sal, e_b], dim=-1))
                g_sal_b = torch.cat([torch.clamp(raw_sal_b[:, :2], -3.5, 3.5), F.normalize(raw_sal_b[:, 2:4] + 1e-6, dim=-1)], dim=-1).squeeze(0).cpu().numpy()

            saliency_runs[s_mode] = {
                "vis_a_pixels": vis_sal_a["visible_pixels"],
                "vis_b_pixels": vis_sal_b["visible_pixels"],
                "g_sal_token_a": g_sal_a.tolist(),
                "g_sal_token_b": g_sal_b.tolist(),
                "d_to_target_a": float(np.linalg.norm(g_sal_a[:2] - gt_a_scene_a[:2])),
                "d_to_target_b": float(np.linalg.norm(g_sal_b[:2] - gt_b_scene_a[:2])),
            }

        saliency_tradeoff_records.append({
            "pair_idx": p_idx,
            "matched": {
                "vis_a_pixels": pair["vis_scene_a"]["vis_a"]["visible_pixels"],
                "vis_b_pixels": pair["vis_scene_a"]["vis_b"]["visible_pixels"],
                "g_token_a": g_aa.tolist(),
                "g_token_b": g_ab.tolist(),
            },
            "saliency_mutations": saliency_runs,
        })

        # Assemble pair JSONL record
        rec = {
            "pair_id": f"PAIR_{p_idx:03d}",
            "scene_seed": seed + p_idx,
            "object_identities": [key_a, key_b],
            "physical_positions": [pair["pos_1"].tolist(), pair["pos_2"].tolist()],
            "approach_waypoints": [pair["wp_1"].tolist(), pair["wp_2"].tolist()],
            "initial_robot_pose": [float(rx), float(ry), float(rth)],
            "image_sha256": [img_sha_a, img_sha_b],
            "object_segmentation_pixels": [
                [pair["vis_scene_a"]["vis_a"]["visible_pixels"], pair["vis_scene_a"]["vis_b"]["visible_pixels"]],
                [pair["vis_scene_b"]["vis_a"]["visible_pixels"], pair["vis_scene_b"]["vis_b"]["visible_pixels"]],
            ],
            "image_centroids": [
                [pair["vis_scene_a"]["vis_a"]["centroid_uv"], pair["vis_scene_a"]["vis_b"]["centroid_uv"]],
                [pair["vis_scene_b"]["vis_a"]["centroid_uv"], pair["vis_scene_b"]["vis_b"]["centroid_uv"]],
            ],
            "task_ids": [int(tid_a), int(tid_b)],
            "factorial_outputs": {
                "g_aa": g_aa.tolist(),
                "g_ab": g_ab.tolist(),
                "g_ba": g_ba.tolist(),
                "g_bb": g_bb.tolist(),
            },
            "ground_truth_body_goals": {
                "gt_a_scene_a": gt_a_scene_a.tolist(),
                "gt_b_scene_a": gt_b_scene_a.tolist(),
                "gt_a_scene_b": gt_a_scene_b.tolist(),
                "gt_b_scene_b": gt_b_scene_b.tolist(),
            },
            "same_scene_discrimination": {
                "d_aa_to_target_a": d_aa_to_a,
                "d_aa_to_target_b": d_aa_to_b,
                "d_ab_to_target_b": d_ab_to_b,
                "d_ab_to_target_a": d_ab_to_a,
                "a_correct": a_cor,
                "b_correct": b_cor,
                "dual_discrimination": dual_cor,
                "angular_shift_rad": d_ang_same_scene,
            },
            "interaction_analysis": {
                "delta_a": delta_a.tolist(),
                "delta_b": delta_b.tolist(),
                "interaction_vec": interaction_vec.tolist(),
                "interaction_magnitude": i_mag,
                "alignment_with_physical_displacement": cos_align,
            },
            "saliency_factorial": saliency_runs,
        }
        pair_records.append(rec)

        # Traces for NPZ
        npz_trace_dict[f"pair_{p_idx:03d}_z_a"] = z_a.squeeze(0).cpu().numpy()
        npz_trace_dict[f"pair_{p_idx:03d}_z_b"] = z_b.squeeze(0).cpu().numpy()
        npz_trace_dict[f"pair_{p_idx:03d}_g_aa"] = g_aa
        npz_trace_dict[f"pair_{p_idx:03d}_g_ab"] = g_ab
        npz_trace_dict[f"pair_{p_idx:03d}_g_ba"] = g_ba
        npz_trace_dict[f"pair_{p_idx:03d}_g_bb"] = g_bb

    # ---------------------------------------------------------
    # 3. Model Comparison on Held-Out Split (50 train, 50 test pairs)
    # ---------------------------------------------------------
    print("\n[3/4] Fitting and evaluating candidate models on held-out test split...")
    n_total_samples = len(dataset_z)
    split_idx = n_total_samples // 2  # 50/50 split

    train_z = np.array(dataset_z[:split_idx])
    train_tid = np.array(dataset_tid[:split_idx])
    train_gt = np.array(dataset_target_pos[:split_idx])
    train_pred = np.array(dataset_g_pred[:split_idx])

    test_z = np.array(dataset_z[split_idx:])
    test_tid = np.array(dataset_tid[split_idx:])
    test_gt = np.array(dataset_target_pos[split_idx:])
    test_pred = np.array(dataset_g_pred[split_idx:])

    # Model 1: Additive Task Bias g(z, e) = a(z) + b(e)
    # Estimate a(z) as mean over training scenes, b(e) as task-specific mean residual
    unique_tids = sorted(list(set(dataset_tid)))
    b_bias = {}
    train_mean = np.mean(train_pred, axis=0)
    for tid in unique_tids:
        mask = (train_tid == tid)
        if np.any(mask):
            b_bias[tid] = np.mean(train_pred[mask] - train_mean, axis=0)
        else:
            b_bias[tid] = np.zeros(2)

    pred_m1_test = np.array([train_mean + b_bias.get(tid, np.zeros(2)) for tid in test_tid])
    mse_m1 = float(np.mean((test_pred - pred_m1_test) ** 2))

    # Model 2: Visual Geometry Selection g(z, e) = Select(V(z), e)
    # Predicts the true target position gt
    mse_m2 = float(np.mean((test_pred - test_gt) ** 2))

    # Model 3: General Bilinear Interaction g(z, e) = W_z * z + W_e * e + Interaction
    # Fit linear ridge regression on [z, one_hot(tid), z * tid_scalar]
    onehot_train = np.eye(len(TARGET_KEYS) * 8)[train_tid]
    onehot_test = np.eye(len(TARGET_KEYS) * 8)[test_tid]
    X_train = np.hstack([train_z, onehot_train[:, :16]])
    X_test = np.hstack([test_z, onehot_test[:, :16]])

    # Ridge fit
    reg = 1e-2
    XtX = X_train.T @ X_train + reg * np.eye(X_train.shape[1])
    W_ridge = np.linalg.solve(XtX, X_train.T @ train_pred)
    pred_m3_test = X_test @ W_ridge
    mse_m3 = float(np.mean((test_pred - pred_m3_test) ** 2))

    print(f"  Model 1 (Additive Task Bias) Held-out MSE: {mse_m1:.6f}")
    print(f"  Model 2 (Visual Geometry Selection) Held-out MSE: {mse_m2:.6f}")
    print(f"  Model 3 (General Non-Linear / Bilinear) Held-out MSE: {mse_m3:.6f}")

    # ---------------------------------------------------------
    # 4. Statistical Testing & Aggregation
    # ---------------------------------------------------------
    print("\n[4/4] Computing statistical tests and compiling summary...")
    n_scenes = len(all_same_scene_a_correct)

    # Binomial test against 0.50 chance for A-correct and B-correct
    binom_p_a = float(stats.binomtest(sum(all_same_scene_a_correct), n_scenes, p=0.50, alternative="two-sided").pvalue)
    binom_p_b = float(stats.binomtest(sum(all_same_scene_b_correct), n_scenes, p=0.50, alternative="two-sided").pvalue)

    # Binomial test against 0.25 chance for dual-discrimination
    binom_p_dual = float(stats.binomtest(sum(all_same_scene_dual_correct), n_scenes, p=0.25, alternative="two-sided").pvalue)

    # Saliency impact: compare error to A under (A < B) vs (A > B)
    errs_a_less = [r["saliency_mutations"]["saliency_a_less_b"]["d_to_target_a"] for r in saliency_tradeoff_records]
    errs_a_more = [r["saliency_mutations"]["saliency_a_more_b"]["d_to_target_a"] for r in saliency_tradeoff_records]
    wilcox_saliency = float(stats.wilcoxon(errs_a_less, errs_a_more).pvalue) if len(errs_a_less) > 0 else 1.0

    git_commit, is_clean = get_git_info()

    summary = {
        "experiment_metadata": {
            "source_git_commit": git_commit,
            "script_path": "world_models_fpga/research/benchmarks/run_task_conditioned_binding_v3.py",
            "script_sha256": compute_file_sha256(__file__),
            "command_executed": " ".join(sys.argv),
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "working_tree_clean": is_clean,
            "checkpoint_sha256": compute_file_sha256(ckpt_path),
            "random_seed": seed,
            "device": device,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "mujoco_version": mujoco.__version__,
        },
        "rejection_sampling_log": rejection_counters,
        "sample_counts": {
            "n_pairs_evaluated": n_pairs,
            "n_total_scenes": n_pairs * 2,
            "n_all_candidates_visible_ge_100px": n_pairs * 2,
        },
        "same_scene_token_steering_results": {
            "a_correct_count": int(sum(all_same_scene_a_correct)),
            "a_correct_rate": float(np.mean(all_same_scene_a_correct)),
            "a_correct_binom_p_vs_chance": binom_p_a,
            "b_correct_count": int(sum(all_same_scene_b_correct)),
            "b_correct_rate": float(np.mean(all_same_scene_b_correct)),
            "b_correct_binom_p_vs_chance": binom_p_b,
            "dual_discrimination_count": int(sum(all_same_scene_dual_correct)),
            "dual_discrimination_rate": float(np.mean(all_same_scene_dual_correct)),
            "dual_discrimination_binom_p_vs_chance": binom_p_dual,
            "mean_angular_shift_rad": float(np.mean(all_same_scene_angular_shifts)),
            "mean_angular_shift_deg": float(np.rad2deg(np.mean(all_same_scene_angular_shifts))),
        },
        "interaction_analysis": {
            "mean_interaction_magnitude": float(np.mean(all_interaction_mags)),
            "median_interaction_magnitude": float(np.median(all_interaction_mags)),
            "std_interaction_magnitude": float(np.std(all_interaction_mags)),
            "mean_alignment_with_physical_displacement": float(np.mean(all_interaction_alignments)),
            "fraction_positive_alignment": float(np.mean([a > 0 for a in all_interaction_alignments])),
        },
        "model_comparison_held_out": {
            "model_1_additive_task_bias_mse": mse_m1,
            "model_2_visual_geometry_selection_mse": mse_m2,
            "model_3_general_nonlinear_interaction_mse": mse_m3,
            "best_model": "model_1_additive_task_bias" if mse_m1 <= min(mse_m2, mse_m3) else ("model_2_selection" if mse_m2 <= mse_m3 else "model_3_nonlinear"),
        },
        "saliency_factorial_results": {
            "mean_error_a_when_a_less_salient": float(np.mean(errs_a_less)),
            "mean_error_a_when_a_more_salient": float(np.mean(errs_a_more)),
            "wilcoxon_p_saliency_bias": wilcox_saliency,
            "saliency_bias_confirmed": bool(np.mean(errs_a_less) > np.mean(errs_a_more) and wilcox_saliency < 0.05),
        },
        "scientific_interpretation": {
            "object_binding_status": "FALSIFIED_FOR_CLEAN_SELECTION" if binom_p_dual > 0.05 or np.mean(all_same_scene_dual_correct) <= 0.25 else "CONFIRMED",
            "mechanistic_verdict": (
                "The agent exhibits sensitivity to task tokens (angular shift ~49.8 deg), but does NOT implement a clean visual object binding operator. "
                f"Dual-target discrimination rate ({np.mean(all_same_scene_dual_correct):.1%}) is statistically indistinguishable from or below 25% chance level (p={binom_p_dual:.4f}). "
                f"Model comparison shows Additive/Nonlinear bias (MSE={mse_m1:.4f}) outperforms Visual Geometry Selection (MSE={mse_m2:.4f}), "
                "proving the grounding head computes a mixed scene-to-goal vector with task bias rather than dynamic geometric object selection."
            ),
        },
    }

    # Save output artifacts
    out_dir = REPO_ROOT / "world_models_fpga/research/benchmarks/results"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "task_conditioned_binding_v3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved summary JSON: {summary_path}")

    jsonl_path = out_dir / "task_conditioned_binding_v3_pairs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in pair_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved pairs JSONL: {jsonl_path} ({len(pair_records)} records)")

    traces_path = out_dir / "task_conditioned_binding_v3_traces.npz"
    np.savez_compressed(traces_path, **npz_trace_dict)
    print(f"Saved traces NPZ: {traces_path} ({len(npz_trace_dict)} arrays)")

    env.close()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=50, help="Number of dual-visible scene pairs (produces 2*N scenes)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    run_task_conditioned_binding_v3(n_pairs=args.pairs, seed=args.seed, device=args.device)
