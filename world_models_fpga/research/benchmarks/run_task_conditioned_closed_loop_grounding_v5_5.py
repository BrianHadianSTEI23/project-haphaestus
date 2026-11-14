#!/usr/bin/env python3
"""Project Hephaestus: Phase 5.5 Closed-Loop Mechanistic Grounding Experiment.

Resolves how H-JEPA can exhibit closed-loop navigation success when Phase 5.4
finds that its static t=0 task-conditioned output contrast is poorly aligned
with ground-truth waypoint contrast.

Evaluates 6 matched primary conditions (C1-C6) plus 1 auxiliary controller-memory
ablation (A1) across the exact N=60 dual-visible Phase 5.3/5.4 scenes
(30 nominal + 30 illumination_shift) with full per-timestep neural, physical,
and planner telemetry.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
from scipy import stats
import torch

from env.amr_env import JetAutoEnv
from models.baseline_policy import encode_task_id
from research.benchmarks.model_wrappers import HJEPAAgent, OracleAgent, compute_body_frame_goal_vec
from research.datasets.dataset_config import CKPT_DIR
from research.datasets.generation.capability_tasks import CANONICAL_OBSTACLES


GEOM_NAME_MAP: Dict[str, str] = {
    "red_box": "obs_red_box",
    "blue_cylinder": "obs_blue_cyl",
    "green_cylinder": "obs_green_cyl",
    "yellow_cylinder": "obs_yellow_cyl",
    "purple_box": "obs_purple_box",
    "cyan_cylinder": "obs_cyan_cyl",
}

PRIMARY_CONDITIONS: Tuple[str, ...] = (
    "C1_full_closed_loop",
    "C2_frozen_world_goal",
    "C3_target_masked_rgb",
    "C4_distractor_masked_rgb",
    "C5_swapped_task_token",
    "C6_oracle_waypoint",
)
AUXILIARY_CONDITIONS: Tuple[str, ...] = ("A1_visual_only_no_memory",)
ALL_CONDITIONS: Tuple[str, ...] = PRIMARY_CONDITIONS + AUXILIARY_CONDITIONS


def compute_file_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "missing"
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_git_info(repo_root: Path = REPO_ROOT) -> Tuple[str, bool]:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
        porcelain = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
        return commit, len(porcelain) == 0
    except Exception:
        return "unknown", False


def body_goal_to_world_pose(robot_pose: Sequence[float] | np.ndarray, goal_vec_body: Sequence[float] | np.ndarray) -> np.ndarray:
    """Inverse of compute_body_frame_goal_vec: converts body-frame goal to fixed world-space pose."""
    rx, ry, rth = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    gx_b, gy_b = float(goal_vec_body[0]), float(goal_vec_body[1])
    sin_dth = float(goal_vec_body[2]) if len(goal_vec_body) > 2 else 0.0
    cos_dth = float(goal_vec_body[3]) if len(goal_vec_body) > 3 else 1.0
    c0, s0 = float(np.cos(rth)), float(np.sin(rth))
    wx = rx + c0 * gx_b - s0 * gy_b
    wy = ry + s0 * gx_b + c0 * gy_b
    dth = float(np.arctan2(sin_dth, cos_dth))
    wth = float(np.arctan2(np.sin(rth + dth), np.cos(rth + dth)))
    return np.array([wx, wy, wth], dtype=np.float32)


def render_segmentation_array(env: JetAutoEnv) -> np.ndarray:
    """Renders exact MuJoCo front_cam segmentation ID map (H, W, 2)."""
    env.fp_renderer.enable_segmentation_rendering()
    env.fp_renderer.update_scene(env.data, camera="front_cam")
    seg = env.fp_renderer.render()
    env.fp_renderer.disable_segmentation_rendering()
    return seg


def extract_geom_mask_and_stats(seg: np.ndarray, geom_id: int) -> Tuple[np.ndarray, int, Optional[List[float]]]:
    """Returns (boolean mask HxW, pixel count, centroid [u, v] or None)."""
    if geom_id < 0:
        h, w = seg.shape[:2]
        return np.zeros((h, w), dtype=bool), 0, None
    mask = (seg[..., 0] == geom_id)
    pix = int(mask.sum())
    if pix == 0:
        return mask, 0, None
    ys, xs = np.nonzero(mask)
    return mask, pix, [float(np.mean(xs)), float(np.mean(ys))]


def _dilate_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Pure-NumPy morphological square dilation of a 2D boolean mask."""
    if not mask.any():
        return mask.copy()
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y_src_start = max(0, -dy)
            y_src_end = min(h, h - dy)
            x_src_start = max(0, -dx)
            x_src_end = min(w, w - dx)
            y_dst_start = max(0, dy)
            y_dst_end = min(h, h + dy)
            x_dst_start = max(0, dx)
            x_dst_end = min(w, w + dx)
            out[y_dst_start:y_dst_end, x_dst_start:x_dst_end] |= mask[y_src_start:y_src_end, x_src_start:x_src_end]
    return out


def apply_local_background_mask(obs_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fills masked pixels with the median RGB of a 2-px local perimeter ring (fallback: unmasked median)."""
    if not mask.any():
        return obs_rgb.copy()
    out = obs_rgb.copy()
    ring = _dilate_mask(mask, radius=2) & (~mask)
    if ring.any():
        fill_rgb = np.median(obs_rgb[ring], axis=0)
    elif (~mask).any():
        fill_rgb = np.median(obs_rgb[~mask], axis=0)
    else:
        fill_rgb = np.array([128.0, 128.0, 128.0], dtype=np.float32)
    out[mask] = np.clip(np.round(fill_rgb), 0, 255).astype(np.uint8)
    return out


def build_area_matched_distractor_mask(
    mask_target: np.ndarray,
    mask_distractor: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """Constructs a mask centered on distractor matched in pixel area to mask_target.

    Strictly disjoint from mask_target. Returns (mask_applied, fallback_symmetric_used).
    """
    h, w = mask_target.shape
    desired_area = int(mask_target.sum())
    if desired_area == 0:
        return np.zeros((h, w), dtype=bool), False

    allowed = ~mask_target
    max_possible = int(allowed.sum())
    desired_area = min(desired_area, max_possible)
    if desired_area == 0:
        return np.zeros((h, w), dtype=bool), False

    ys_grid, xs_grid = np.mgrid[0:h, 0:w]
    dist_allowed = mask_distractor & allowed
    cur_area = int(dist_allowed.sum())

    if cur_area > 0:
        ys_d, xs_d = np.nonzero(dist_allowed)
        cy, cx = float(np.mean(ys_d)), float(np.mean(xs_d))
        if cur_area >= desired_area:
            d2 = (ys_d - cy) ** 2 + (xs_d - cx) ** 2
            order = np.argsort(d2, kind="mergesort")[:desired_area]
            out = np.zeros((h, w), dtype=bool)
            out[ys_d[order], xs_d[order]] = True
            return out, False
        # Expand outside current distractor mask into nearest allowed background pixels
        out = dist_allowed.copy()
        need = desired_area - cur_area
        cand_mask = allowed & (~dist_allowed)
        yc, xc = np.nonzero(cand_mask)
        d2 = (yc - cy) ** 2 + (xc - cx) ** 2
        order = np.argsort(d2, kind="mergesort")[:need]
        out[yc[order], xc[order]] = True
        return out, False

    # Distractor exited camera FOV: place area-matched mask at mirrored target x-location (disjoint from target)
    yt, xt = np.nonzero(mask_target)
    cy = float(np.mean(yt))
    cx = float((w - 1) - np.mean(xt))
    yc, xc = np.nonzero(allowed)
    d2 = (yc - cy) ** 2 + (xc - cx) ** 2
    order = np.argsort(d2, kind="mergesort")[:desired_area]
    out = np.zeros((h, w), dtype=bool)
    out[yc[order], xc[order]] = True
    return out, True


def paired_mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> Tuple[int, int, float]:
    """Exact two-sided binomial McNemar test on matched binary outcomes (a vs b)."""
    n10 = sum(1 for ai, bi in zip(a, b) if bool(ai) and not bool(bi))
    n01 = sum(1 for ai, bi in zip(a, b) if (not bool(ai)) and bool(bi))
    n = n10 + n01
    if n == 0:
        return 0, 0, 1.0
    p = float(stats.binomtest(min(n10, n01), n=n, p=0.5, alternative="two-sided").pvalue)
    return int(n10), int(n01), p


def paired_bootstrap_diff_ci(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 5500,
) -> Tuple[float, float, float]:
    """Returns (mean_diff, ci_low, ci_high) for paired difference (a - b)."""
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    obs = float(np.mean(diffs)) if diffs.size else 0.0
    if diffs.size <= 1:
        return obs, obs, obs
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot_means = diffs[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return (
        obs,
        float(np.percentile(boot_means, 100.0 * alpha)),
        float(np.percentile(boot_means, 100.0 * (1.0 - alpha))),
    )


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> List[float]:
    if n <= 0:
        return [0.0, 0.0]
    p = k / float(n)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def holm_bonferroni_adjust(p_values: Sequence[float]) -> List[float]:
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [1.0] * m
    run_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        val = min(1.0, float(p_val) * (m - rank))
        run_max = max(run_max, val)
        adj[orig_idx] = float(run_max)
    return adj


def load_phase5_matched_cohort(specs_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Loads exact N=60 dual-visible scenes (scene_a of PAIR_000..PAIR_059) from v5.2/5.3 specs."""
    specs: List[Dict[str, Any]] = []
    with specs_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            obstacles = copy.deepcopy(CANONICAL_OBSTACLES)
            obstacles[raw["key_a"]]["pos"] = np.asarray(raw["pos_1"], dtype=np.float32)
            obstacles[raw["key_b"]]["pos"] = np.asarray(raw["pos_2"], dtype=np.float32)
            spec = {
                "pair_id": raw["pair_id"],
                "pair_idx": int(raw["pair_idx"]),
                "episode_id": raw["pair_id"],
                "context": raw["context"],
                "light_diffuse_scale": float(raw["light_diffuse_scale"]),
                "target_key": raw["key_a"],
                "distractor_key": raw["key_b"],
                "semantic_relation": "approach",
                "task_token": int(raw["task_id_a"]),
                "swapped_task_token": int(raw["task_id_b"]),
                "instruction": f"approach the {raw['key_a'].replace('_', ' ')}",
                "initial_pose": [float(v) for v in raw["robot_pose"]],
                "waypoints": [np.asarray(raw["wp_1"], dtype=np.float32)],
                "distractor_waypoint": np.asarray(raw["wp_2"], dtype=np.float32),
                "obstacles": obstacles,
                "expected_obs_a_hash": raw["obs_a_hash"],
                "vis_scene_a": raw["vis_scene_a"],
            }
            specs.append(spec)
            if limit is not None and len(specs) >= limit:
                break
    return specs


def evaluate_episode_condition(
    agent: HJEPAAgent,
    env: JetAutoEnv,
    spec: Dict[str, Any],
    condition_name: str,
    max_steps: int = 45,
    goal_reach_threshold: float = 0.35,
) -> Dict[str, Any]:
    """Runs one closed-loop episode under a Phase 5.5 mechanistic condition with full telemetry."""
    if condition_name not in ALL_CONDITIONS:
        raise ValueError(f"Unsupported condition {condition_name!r}")

    env.restore_physical_baseline()
    agent.clear_branch_interceptors()
    agent.reset()

    original_track = agent.track
    if condition_name == "C6_oracle_waypoint":
        agent.track = "structured"
    else:
        agent.track = "observation"

    # Capture grounding_head internal L1/L2 post-ReLU activations during forward pass
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

        initial_eval_dist = float(np.linalg.norm(init_pos[:2] - eval_wp[:2]))
        initially_inside_radius = bool(initial_eval_dist <= goal_reach_threshold)

        # Configure condition interceptors
        current_pose_holder: List[np.ndarray] = [init_pos.copy()]
        frozen_world_goal_holder: List[Optional[np.ndarray]] = [None]

        if condition_name == "C5_swapped_task_token":
            swapped_tid = int(spec["swapped_task_token"])
            agent.set_branch_interceptor("task_id", lambda _tid, st=swapped_tid: st)
        else:
            nom_tid = int(spec["task_token"])
            agent.set_branch_interceptor("task_id", lambda _tid, nt=nom_tid: nt)

        if condition_name == "C2_frozen_world_goal":
            def _freeze_world_goal(g_blended_t, _v=None, _m=None):
                if frozen_world_goal_holder[0] is None:
                    g0 = g_blended_t.squeeze(0).detach().cpu().numpy()
                    frozen_world_goal_holder[0] = body_goal_to_world_pose(current_pose_holder[0], g0)
                g_body = compute_body_frame_goal_vec(current_pose_holder[0], frozen_world_goal_holder[0])
                return torch.as_tensor(g_body, dtype=g_blended_t.dtype, device=g_blended_t.device).view(1, 4)

            agent.set_branch_interceptor("blended_goal", _freeze_world_goal)

        elif condition_name == "A1_visual_only_no_memory":
            agent.set_branch_interceptor("blended_goal", lambda _g, vis, _mem: vis)

        # Static t=0 contrast diagnostic on identical unmasked initial frame obs_0 (Task A vs Task B)
        # Only requires one quick grounding forward check
        with torch.no_grad():
            img0_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(agent.device) / 255.0
            s0_t = agent.model.encode(img0_t)
            t0_a = torch.tensor([int(spec["task_token"])], dtype=torch.long, device=agent.device)
            t0_b = torch.tensor([int(spec["swapped_task_token"])], dtype=torch.long, device=agent.device)
            g0_task_a = agent.model.infer_goal_vec(s0_t, t0_a).squeeze(0).cpu().numpy()[:2]
            g0_task_b = agent.model.infer_goal_vec(s0_t, t0_b).squeeze(0).cpu().numpy()[:2]
        gt0_a = compute_body_frame_goal_vec(init_pos, eval_wp)[:2]
        gt0_b = compute_body_frame_goal_vec(init_pos, dist_wp)[:2]
        d_model_0 = g0_task_b - g0_task_a
        d_gt_0 = gt0_b - gt0_a
        n_m0, n_g0 = float(np.linalg.norm(d_model_0)), float(np.linalg.norm(d_gt_0))
        step0_contrast_cos = float(np.dot(d_model_0, d_gt_0) / (n_m0 * n_g0)) if (n_m0 > 1e-8 and n_g0 > 1e-8) else 0.0
        step0_contrast_eta_gt = float(np.dot(d_model_0, d_gt_0) / (n_g0 * n_g0)) if n_g0 > 1e-8 else 0.0

        # Step trace accumulators
        tr_pose: List[List[float]] = []
        tr_latent_s: List[np.ndarray] = []
        tr_l1: List[np.ndarray] = []
        tr_l2: List[np.ndarray] = []
        tr_g_vis: List[np.ndarray] = []
        tr_g_mem: List[np.ndarray] = []
        tr_g_blend: List[np.ndarray] = []
        tr_delta_ego: List[np.ndarray] = []
        tr_gt_target: List[np.ndarray] = []
        tr_gt_dist: List[np.ndarray] = []
        tr_a_react: List[np.ndarray] = []
        tr_a_exec: List[np.ndarray] = []
        tr_best_macro_idx: List[int] = []
        tr_pred_macro_disp: List[np.ndarray] = []
        tr_pred_micro_disp: List[np.ndarray] = []
        tr_col_risk: List[float] = []
        tr_dist_target_wp: List[float] = []
        tr_dist_distractor_wp: List[float] = []
        tr_target_vis_px: List[int] = []
        tr_distractor_vis_px: List[int] = []
        tr_masked_px: List[int] = []
        tr_fallback_mask: List[bool] = []
        tr_action_target_cos: List[float] = []
        tr_goal_ang_err_rad: List[float] = []
        tr_vis_ang_err_rad: List[float] = []

        collision_occurred = False
        goal_reached = False
        distractor_reached = False
        first_reach_step: Optional[int] = None
        first_distractor_reach_step: Optional[int] = None
        prev_xy = init_pos[:2].copy()
        total_path = 0.0

        for step in range(max_steps):
            curr_pose = env.get_robot_pose()
            current_pose_holder[0] = curr_pose
            tr_pose.append(curr_pose.tolist())

            gt_t = compute_body_frame_goal_vec(curr_pose, eval_wp)
            gt_d = compute_body_frame_goal_vec(curr_pose, dist_wp)
            tr_gt_target.append(gt_t)
            tr_gt_dist.append(gt_d)
            tr_dist_target_wp.append(float(np.linalg.norm(curr_pose[:2] - eval_wp[:2])))
            tr_dist_distractor_wp.append(float(np.linalg.norm(curr_pose[:2] - dist_wp[:2])))

            seg = render_segmentation_array(env)
            mask_t, px_t, _ = extract_geom_mask_and_stats(seg, gid_target)
            mask_d, px_d, _ = extract_geom_mask_and_stats(seg, gid_distractor)
            tr_target_vis_px.append(px_t)
            tr_distractor_vis_px.append(px_d)

            obs_for_agent = obs
            masked_px = 0
            fallback_used = False
            if condition_name == "C3_target_masked_rgb":
                obs_for_agent = apply_local_background_mask(obs, mask_t)
                masked_px = int(mask_t.sum())
            elif condition_name == "C4_distractor_masked_rgb":
                matched_mask, fallback_used = build_area_matched_distractor_mask(mask_t, mask_d)
                obs_for_agent = apply_local_background_mask(obs, matched_mask)
                masked_px = int(matched_mask.sum())

            tr_masked_px.append(masked_px)
            tr_fallback_mask.append(bool(fallback_used))

            l1_captured[0] = None
            l2_captured[0] = None

            if condition_name == "C6_oracle_waypoint":
                act, diag = agent.act(obs_for_agent, robot_pose=curr_pose, task_spec=spec, goal_vec_body=gt_t)
            else:
                act, diag = agent.act(obs_for_agent, robot_pose=curr_pose, task_spec=spec)

            tr_latent_s.append(np.asarray(diag["latent"], dtype=np.float32))
            tr_l1.append(l1_captured[0] if l1_captured[0] is not None else np.zeros(64, dtype=np.float32))
            tr_l2.append(l2_captured[0] if l2_captured[0] is not None else np.zeros(64, dtype=np.float32))

            g_vis = np.asarray(diag.get("g_visual_t") if diag.get("g_visual_t") is not None else gt_t, dtype=np.float32)
            g_mem = np.asarray(diag.get("g_memory_t") if diag.get("g_memory_t") is not None else gt_t, dtype=np.float32)
            g_bld = np.asarray(diag.get("g_blended_t") if diag.get("g_blended_t") is not None else gt_t, dtype=np.float32)
            d_ego = np.asarray(diag.get("delta_ego_t") if diag.get("delta_ego_t") is not None else np.zeros(3), dtype=np.float32)

            tr_g_vis.append(g_vis)
            tr_g_mem.append(g_mem)
            tr_g_blend.append(g_bld)
            tr_delta_ego.append(d_ego)
            tr_a_react.append(np.asarray(diag.get("a_react", act), dtype=np.float32))
            tr_a_exec.append(np.asarray(act, dtype=np.float32))
            tr_best_macro_idx.append(int(diag.get("best_macro_idx", -1)))
            tr_pred_macro_disp.append(np.asarray(diag.get("pred_macro_disp", np.zeros(3)), dtype=np.float32))
            tr_pred_micro_disp.append(np.asarray(diag.get("pred_micro_disp", np.zeros(3)), dtype=np.float32))
            tr_col_risk.append(float(diag.get("col_risk", 0.0)))

            # Angular errors and action alignment
            ang_gt = float(np.arctan2(gt_t[1], gt_t[0]))
            ang_bld = float(np.arctan2(g_bld[1], g_bld[0]))
            ang_vis = float(np.arctan2(g_vis[1], g_vis[0]))
            tr_goal_ang_err_rad.append(float(abs(np.arctan2(np.sin(ang_bld - ang_gt), np.cos(ang_bld - ang_gt)))))
            tr_vis_ang_err_rad.append(float(abs(np.arctan2(np.sin(ang_vis - ang_gt), np.cos(ang_vis - ang_gt)))))

            act_xy = np.asarray(act[:2], dtype=np.float64)
            gt_xy = np.asarray(gt_t[:2], dtype=np.float64)
            na, ng = float(np.linalg.norm(act_xy)), float(np.linalg.norm(gt_xy))
            tr_action_target_cos.append(float(np.dot(act_xy, gt_xy) / (na * ng)) if (na > 1e-8 and ng > 1e-8) else 0.0)

            step_res = env.step(act)
            obs = step_res[0]
            info = step_res[-1]
            if info.get("collision", False):
                collision_occurred = True

            post_pose = env.get_robot_pose()
            curr_xy = post_pose[:2]
            total_path += float(np.linalg.norm(curr_xy - prev_xy))
            prev_xy = curr_xy.copy()

            if float(np.linalg.norm(curr_xy - dist_wp[:2])) <= goal_reach_threshold:
                distractor_reached = True
                if first_distractor_reach_step is None:
                    first_distractor_reach_step = step + 1

            if float(np.linalg.norm(curr_xy - eval_wp[:2])) <= goal_reach_threshold:
                goal_reached = True
                if first_reach_step is None:
                    first_reach_step = step + 1
                break

        final_pose = env.get_robot_pose()
        final_eval_err = float(np.linalg.norm(final_pose[:2] - eval_wp[:2]))
        final_dist_err = float(np.linalg.norm(final_pose[:2] - dist_wp[:2]))

        success = bool(
            (not initially_inside_radius)
            and goal_reached
            and (not collision_occurred)
            and (final_eval_err <= goal_reach_threshold)
        )
        redirected_to_distractor = bool(
            (not collision_occurred)
            and distractor_reached
            and (final_dist_err <= goal_reach_threshold)
            and (not success)
        )

        return {
            "pair_id": spec["pair_id"],
            "pair_idx": spec["pair_idx"],
            "context": spec["context"],
            "condition": condition_name,
            "target_key": target_key,
            "distractor_key": distractor_key,
            "obs_unmasked_0_sha256": obs_unmasked_0_hash,
            "obs_0_matches_v5_2_hash": bool(obs_unmasked_0_hash == spec.get("expected_obs_a_hash", "")),
            "initially_inside_radius": initially_inside_radius,
            "initial_eval_dist_m": initial_eval_dist,
            "final_eval_goal_error_m": final_eval_err,
            "final_distractor_goal_error_m": final_dist_err,
            "closer_to_distractor": bool(final_dist_err < final_eval_err),
            "path_length_m": float(total_path),
            "executed_steps": len(tr_a_exec),
            "first_reach_step": first_reach_step,
            "goal_reached": goal_reached,
            "distractor_reached": distractor_reached,
            "redirected_to_distractor": redirected_to_distractor,
            "collision": collision_occurred,
            "success": success,
            "step0_static_contrast_cosine": step0_contrast_cos,
            "step0_static_contrast_eta_gt": step0_contrast_eta_gt,
            "step0_blended_goal_ang_err_deg": float(np.degrees(tr_goal_ang_err_rad[0])),
            "mean_step_blended_goal_ang_err_deg": float(np.degrees(np.mean(tr_goal_ang_err_rad))),
            "mean_step_vis_goal_ang_err_deg": float(np.degrees(np.mean(tr_vis_ang_err_rad))),
            "mean_step_action_target_cos": float(np.mean(tr_action_target_cos)),
            "mean_target_visible_pixels": float(np.mean(tr_target_vis_px)),
            "mean_distractor_visible_pixels": float(np.mean(tr_distractor_vis_px)),
            "mean_masked_pixels": float(np.mean(tr_masked_px)),
            "fallback_mask_steps": int(sum(1 for x in tr_fallback_mask if x)),
            "traces": {
                "pose": np.asarray(tr_pose, dtype=np.float32),
                "latent_s": np.asarray(tr_latent_s, dtype=np.float32),
                "l1_act": np.asarray(tr_l1, dtype=np.float32),
                "l2_act": np.asarray(tr_l2, dtype=np.float32),
                "g_visual": np.asarray(tr_g_vis, dtype=np.float32),
                "g_memory": np.asarray(tr_g_mem, dtype=np.float32),
                "g_blended": np.asarray(tr_g_blend, dtype=np.float32),
                "delta_ego": np.asarray(tr_delta_ego, dtype=np.float32),
                "gt_target_body": np.asarray(tr_gt_target, dtype=np.float32),
                "gt_distractor_body": np.asarray(tr_gt_dist, dtype=np.float32),
                "a_react": np.asarray(tr_a_react, dtype=np.float32),
                "action_exec": np.asarray(tr_a_exec, dtype=np.float32),
                "best_macro_idx": np.asarray(tr_best_macro_idx, dtype=np.int32),
                "pred_macro_disp": np.asarray(tr_pred_macro_disp, dtype=np.float32),
                "pred_micro_disp": np.asarray(tr_pred_micro_disp, dtype=np.float32),
                "col_risk": np.asarray(tr_col_risk, dtype=np.float32),
                "dist_target_wp": np.asarray(tr_dist_target_wp, dtype=np.float32),
                "dist_distractor_wp": np.asarray(tr_dist_distractor_wp, dtype=np.float32),
                "target_vis_px": np.asarray(tr_target_vis_px, dtype=np.int32),
                "distractor_vis_px": np.asarray(tr_distractor_vis_px, dtype=np.int32),
                "masked_px": np.asarray(tr_masked_px, dtype=np.int32),
                "action_target_cos": np.asarray(tr_action_target_cos, dtype=np.float32),
                "goal_ang_err_rad": np.asarray(tr_goal_ang_err_rad, dtype=np.float32),
                "vis_ang_err_rad": np.asarray(tr_vis_ang_err_rad, dtype=np.float32),
            },
        }
    finally:
        agent.track = original_track
        for h in hooks:
            h.remove()


def _summarize_condition_records(records: List[Dict[str, Any]], seed: int = 5500) -> Dict[str, Any]:
    n = len(records)
    succ = [bool(r["success"]) for r in records]
    k_succ = sum(1 for x in succ if x)
    errs = [float(r["final_eval_goal_error_m"]) for r in records]
    paths = [float(r["path_length_m"]) for r in records]
    steps = [float(r["executed_steps"]) for r in records]
    cols = [bool(r["collision"]) for r in records]
    ang_errs = [float(r["mean_step_blended_goal_ang_err_deg"]) for r in records]
    vis_ang_errs = [float(r["mean_step_vis_goal_ang_err_deg"]) for r in records]
    act_cos = [float(r["mean_step_action_target_cos"]) for r in records]
    redir = [bool(r["redirected_to_distractor"]) for r in records]
    closer_dist = [bool(r["closer_to_distractor"]) for r in records]

    def _sub_summary(sub: List[Dict[str, Any]]) -> Dict[str, Any]:
        ns = len(sub)
        ks = sum(1 for x in sub if x["success"])
        return {
            "n_episodes": ns,
            "success_count": ks,
            "success_rate": float(ks / ns) if ns else 0.0,
            "success_wilson_95ci": wilson_ci(ks, ns),
            "collision_rate": float(np.mean([x["collision"] for x in sub])) if ns else 0.0,
            "mean_final_goal_error_m": float(np.mean([x["final_eval_goal_error_m"] for x in sub])) if ns else 0.0,
            "median_final_goal_error_m": float(np.median([x["final_eval_goal_error_m"] for x in sub])) if ns else 0.0,
            "mean_action_target_cos": float(np.mean([x["mean_step_action_target_cos"] for x in sub])) if ns else 0.0,
            "mean_step_blended_goal_ang_err_deg": float(np.mean([x["mean_step_blended_goal_ang_err_deg"] for x in sub])) if ns else 0.0,
        }

    return {
        "n_episodes": n,
        "success_count": k_succ,
        "success_rate": float(k_succ / n) if n else 0.0,
        "success_wilson_95ci": wilson_ci(k_succ, n),
        "collision_count": sum(1 for x in cols if x),
        "collision_rate": float(np.mean(cols)) if n else 0.0,
        "redirected_to_distractor_count": sum(1 for x in redir if x),
        "closer_to_distractor_count": sum(1 for x in closer_dist if x),
        "mean_initial_eval_dist_m": float(np.mean([r["initial_eval_dist_m"] for r in records])) if n else 0.0,
        "min_initial_eval_dist_m": float(np.min([r["initial_eval_dist_m"] for r in records])) if n else 0.0,
        "mean_final_goal_error_m": float(np.mean(errs)) if n else 0.0,
        "median_final_goal_error_m": float(np.median(errs)) if n else 0.0,
        "mean_path_length_m": float(np.mean(paths)) if n else 0.0,
        "median_path_length_m": float(np.median(paths)) if n else 0.0,
        "mean_executed_steps": float(np.mean(steps)) if n else 0.0,
        "mean_step_blended_goal_ang_err_deg": float(np.mean(ang_errs)) if n else 0.0,
        "mean_step_vis_goal_ang_err_deg": float(np.mean(vis_ang_errs)) if n else 0.0,
        "mean_step_action_target_cos": float(np.mean(act_cos)) if n else 0.0,
        "mean_target_visible_pixels": float(np.mean([r["mean_target_visible_pixels"] for r in records])) if n else 0.0,
        "mean_distractor_visible_pixels": float(np.mean([r["mean_distractor_visible_pixels"] for r in records])) if n else 0.0,
        "mean_masked_pixels": float(np.mean([r["mean_masked_pixels"] for r in records])) if n else 0.0,
        "total_fallback_mask_steps": int(sum(r["fallback_mask_steps"] for r in records)),
        "by_context": {
            ctx: _sub_summary([r for r in records if r["context"] == ctx])
            for ctx in ("nominal", "illumination_shift")
        },
    }


def _paired_comparison(recs_a: List[Dict[str, Any]], recs_b: List[Dict[str, Any]], name_a: str, name_b: str, seed: int) -> Dict[str, Any]:
    """Compares condition A minus condition B strictly matched by pair_id."""
    assert [r["pair_id"] for r in recs_a] == [r["pair_id"] for r in recs_b], "Matched pair order mismatch"
    succ_a = [bool(r["success"]) for r in recs_a]
    succ_b = [bool(r["success"]) for r in recs_b]
    n10, n01, p_mc = paired_mcnemar_exact(succ_a, succ_b)
    rd_mean, rd_lo, rd_hi = paired_bootstrap_diff_ci([1.0 if x else 0.0 for x in succ_a], [1.0 if x else 0.0 for x in succ_b], seed=seed)

    err_a = [float(r["final_eval_goal_error_m"]) for r in recs_a]
    err_b = [float(r["final_eval_goal_error_m"]) for r in recs_b]
    ed_mean, ed_lo, ed_hi = paired_bootstrap_diff_ci(err_a, err_b, seed=seed + 1)
    p_err_wilc = float(stats.wilcoxon(np.asarray(err_a) - np.asarray(err_b), alternative="two-sided").pvalue) if not np.allclose(err_a, err_b) else 1.0

    ang_a = [float(r["mean_step_blended_goal_ang_err_deg"]) for r in recs_a]
    ang_b = [float(r["mean_step_blended_goal_ang_err_deg"]) for r in recs_b]
    ad_mean, ad_lo, ad_hi = paired_bootstrap_diff_ci(ang_a, ang_b, seed=seed + 2)
    p_ang_wilc = float(stats.wilcoxon(np.asarray(ang_a) - np.asarray(ang_b), alternative="two-sided").pvalue) if not np.allclose(ang_a, ang_b) else 1.0

    cos_a = [float(r["mean_step_action_target_cos"]) for r in recs_a]
    cos_b = [float(r["mean_step_action_target_cos"]) for r in recs_b]
    cd_mean, cd_lo, cd_hi = paired_bootstrap_diff_ci(cos_a, cos_b, seed=seed + 3)

    return {
        "comparison": f"{name_a} - {name_b}",
        "n_matched_pairs": len(recs_a),
        "success_rate_a": float(np.mean(succ_a)),
        "success_rate_b": float(np.mean(succ_b)),
        "success_rate_diff": rd_mean,
        "success_rate_diff_95ci": [rd_lo, rd_hi],
        "mcnemar_n10_a_only": n10,
        "mcnemar_n01_b_only": n01,
        "mcnemar_two_sided_p": p_mc,
        "final_error_diff_m": ed_mean,
        "final_error_diff_95ci_m": [ed_lo, ed_hi],
        "final_error_wilcoxon_two_sided_p": p_err_wilc,
        "goal_ang_err_diff_deg": ad_mean,
        "goal_ang_err_diff_95ci_deg": [ad_lo, ad_hi],
        "goal_ang_err_wilcoxon_two_sided_p": p_ang_wilc,
        "action_target_cos_diff": cd_mean,
        "action_target_cos_diff_95ci": [cd_lo, cd_hi],
        "holm_adjusted_p_success": None,
    }


def generate_figures_from_saved_artifacts(
    summary_path: Path,
    episode_jsonl_path: Path,
    traces_npz_path: Path,
    figures_dir: Path,
) -> List[str]:
    """Generates reproducible Phase 5.5 figures strictly from committed raw artifacts."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    episodes = [json.loads(line) for line in episode_jsonl_path.open("r", encoding="utf-8") if line.strip()]
    npz = np.load(traces_npz_path)

    written: List[str] = []
    short_names = {
        "C1_full_closed_loop": "C1\nFull CL",
        "C2_frozen_world_goal": "C2\nFrozen t=0",
        "C3_target_masked_rgb": "C3\nTgt Mask",
        "C4_distractor_masked_rgb": "C4\nDist Mask",
        "C5_swapped_task_token": "C5\nSwap Tok",
        "C6_oracle_waypoint": "C6\nOracle WP",
        "A1_visual_only_no_memory": "A1\nVis Only",
    }

    # Figure 1: Condition Success Rates & Final Error by Context
    cond_order = [c for c in ALL_CONDITIONS if c in summary["conditions"]]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=180)
    x = np.arange(len(cond_order))
    rates = [summary["conditions"][c]["success_rate"] * 100.0 for c in cond_order]
    lo_err = [(summary["conditions"][c]["success_rate"] - summary["conditions"][c]["success_wilson_95ci"][0]) * 100.0 for c in cond_order]
    hi_err = [(summary["conditions"][c]["success_wilson_95ci"][1] - summary["conditions"][c]["success_rate"]) * 100.0 for c in cond_order]
    colors = ["#2563eb", "#f59e0b", "#dc2626", "#16a34a", "#7c3aed", "#0d9488", "#64748b"][: len(cond_order)]

    axes[0].bar(x, rates, yerr=[lo_err, hi_err], capsize=4, color=colors, alpha=0.88, edgecolor="#111827")
    for idx, val in enumerate(rates):
        axes[0].text(idx, val + hi_err[idx] + 2.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axes[0].set_ylim(0, 118)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([short_names[c] for c in cond_order], fontsize=8.5)
    axes[0].set_ylabel("Closed-Loop Success Rate (%) [95% Wilson CI]")
    axes[0].set_title("Phase 5.5 Matched Cohort Success (N=60 per condition)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    # Panel B: Context-stratified final goal error (nominal vs illumination_shift)
    width = 0.36
    nom_means = [summary["conditions"][c]["by_context"]["nominal"]["mean_final_goal_error_m"] for c in cond_order]
    ill_means = [summary["conditions"][c]["by_context"]["illumination_shift"]["mean_final_goal_error_m"] for c in cond_order]
    axes[1].bar(x - width / 2, nom_means, width=width, label="Nominal (N=30)", color="#3b82f6", alpha=0.88)
    axes[1].bar(x + width / 2, ill_means, width=width, label="Illumination Shift (N=30)", color="#f97316", alpha=0.88)
    axes[1].axhline(0.35, color="#dc2626", linestyle="--", linewidth=1.2, label="Goal Threshold (0.35 m)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([short_names[c] for c in cond_order], fontsize=8.5)
    axes[1].set_ylabel("Mean Final Waypoint Error (m)")
    axes[1].set_title("Final Goal Error Stratified by Illumination Context")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    p1 = figures_dir / "fig1_condition_success_and_error.png"
    fig.savefig(p1)
    plt.close(fig)
    written.append(p1.name)

    # Figure 2: Step-by-Step Closed-Loop Evolution (Distance to Target, Goal Angle Error, Action Target Cosine)
    max_t = 25
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=180)
    plot_conds = [
        ("C1_full_closed_loop", "C1 Full CL", "#2563eb", "-"),
        ("C2_frozen_world_goal", "C2 Frozen t=0", "#f59e0b", "--"),
        ("C3_target_masked_rgb", "C3 Target Masked", "#dc2626", "-"),
        ("C4_distractor_masked_rgb", "C4 Distractor Masked", "#16a34a", "-."),
        ("C5_swapped_task_token", "C5 Swapped Token", "#7c3aed", ":"),
        ("C6_oracle_waypoint", "C6 Oracle WP", "#0d9488", "--"),
    ]
    for cname, label, col, ls in plot_conds:
        eps_c = [ep for ep in episodes if ep["condition"] == cname]
        if not eps_c:
            continue
        dist_mat = np.full((len(eps_c), max_t), np.nan)
        ang_mat = np.full((len(eps_c), max_t), np.nan)
        cos_mat = np.full((len(eps_c), max_t), np.nan)
        for idx, ep in enumerate(eps_c):
            prefix = f"{cname}__{ep['pair_id']}__"
            d_arr = npz[prefix + "dist_target_wp"]
            a_arr = np.degrees(npz[prefix + "goal_ang_err_rad"])
            c_arr = npz[prefix + "action_target_cos"]
            ln = min(max_t, len(d_arr))
            dist_mat[idx, :ln] = d_arr[:ln]
            ang_mat[idx, :ln] = a_arr[:ln]
            cos_mat[idx, :ln] = c_arr[:ln]
        # Carry forward last distance after early reach so survival curves don't jump artificially
        for idx in range(len(eps_c)):
            valid = np.where(~np.isnan(dist_mat[idx]))[0]
            if len(valid) > 0:
                last_i = valid[-1]
                dist_mat[idx, last_i + 1 :] = dist_mat[idx, last_i]

        ts = np.arange(max_t)
        d_mean = np.nanmean(dist_mat, axis=0)
        d_sem = np.nanstd(dist_mat, axis=0) / np.sqrt(np.sum(~np.isnan(dist_mat), axis=0))
        axes[0].plot(ts, d_mean, label=label, color=col, linestyle=ls, linewidth=2.0)
        axes[0].fill_between(ts, d_mean - d_sem, d_mean + d_sem, color=col, alpha=0.12)

        # For active-step angle and action cosine, plot only while >=10 episodes are still running
        cnt_a = np.sum(~np.isnan(ang_mat), axis=0)
        keep = cnt_a >= min(10, len(eps_c))
        a_mean = np.nanmean(ang_mat[:, keep], axis=0)
        axes[1].plot(ts[keep], a_mean, label=label, color=col, linestyle=ls, linewidth=2.0)

        c_mean = np.nanmean(cos_mat[:, keep], axis=0)
        axes[2].plot(ts[keep], c_mean, label=label, color=col, linestyle=ls, linewidth=2.0)

    axes[0].axhline(0.35, color="#111827", linestyle=":", linewidth=1.0, label="Threshold 0.35m")
    axes[0].set_xlabel("Control Step t")
    axes[0].set_ylabel("Distance to Target Waypoint (m)")
    axes[0].set_title("Closed-Loop Approach Trajectories")
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Control Step t (active episodes >= 10)")
    axes[1].set_ylabel("Blended Goal Angular Error (deg)")
    axes[1].set_title("Goal Vector Angle Error Over Time")
    axes[1].grid(alpha=0.3)

    axes[2].axhline(0.0, color="#111827", linestyle=":", linewidth=1.0)
    axes[2].set_xlabel("Control Step t (active episodes >= 10)")
    axes[2].set_ylabel("cos(Executed Action, Target Dir)")
    axes[2].set_title("Executed Action Alignment Toward Target")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    p2 = figures_dir / "fig2_closed_loop_temporal_evolution.png"
    fig.savefig(p2)
    plt.close(fig)
    written.append(p2.name)

    # Figure 3: Static Step-0 Alignment vs Closed-Loop Error Correction & C1 vs C2 Diagnostic
    c1_eps = {ep["pair_id"]: ep for ep in episodes if ep["condition"] == "C1_full_closed_loop"}
    c2_eps = {ep["pair_id"]: ep for ep in episodes if ep["condition"] == "C2_frozen_world_goal"}
    pids = sorted(c1_eps.keys())
    step0_eta = np.array([c1_eps[p]["step0_static_contrast_eta_gt"] for p in pids])
    step0_ang = np.array([c1_eps[p]["step0_blended_goal_ang_err_deg"] for p in pids])
    c1_err = np.array([c1_eps[p]["final_eval_goal_error_m"] for p in pids])
    c2_err = np.array([c2_eps[p]["final_eval_goal_error_m"] for p in pids])
    c1_succ = np.array([c1_eps[p]["success"] for p in pids], dtype=bool)
    c2_succ = np.array([c2_eps[p]["success"] for p in pids], dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=180)
    axes[0].scatter(step0_eta[c1_succ], c1_err[c1_succ], color="#2563eb", label="C1 Success", alpha=0.8, s=42)
    axes[0].scatter(step0_eta[~c1_succ], c1_err[~c1_succ], color="#dc2626", marker="x", label="C1 Fail", alpha=0.9, s=55)
    axes[0].axvline(0.0, color="#6b7280", linestyle="--", linewidth=1.0)
    axes[0].axhline(0.35, color="#dc2626", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("Step-0 Static Task Contrast Projection eta_gt (dimensionless)")
    axes[0].set_ylabel("C1 Final Waypoint Error (m)")
    axes[0].set_title("Static Step-0 Task Contrast vs C1 Closed-Loop Error")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(alpha=0.3)

    axes[1].scatter(step0_ang[c2_succ], c2_err[c2_succ], color="#0d9488", label="C2 Frozen-Goal Success", alpha=0.75, s=42)
    axes[1].scatter(step0_ang[~c2_succ], c2_err[~c2_succ], color="#f97316", marker="^", label="C2 Frozen-Goal Fail", alpha=0.85, s=48)
    axes[1].axhline(0.35, color="#dc2626", linestyle=":", linewidth=1.0, label="Goal Threshold 0.35m")
    axes[1].set_xlabel("Step-0 Single-Target Goal Vector Angular Error (deg)")
    axes[1].set_ylabel("C2 Frozen World-Goal Final Error (m)")
    axes[1].set_title("Step-0 Goal Angle Error Predicts Open-Loop C2 Failure")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p3 = figures_dir / "fig3_static_contrast_vs_closed_loop.png"
    fig.savefig(p3)
    plt.close(fig)
    written.append(p3.name)

    return written


def run_phase5_5_experiment(
    device: str = "cuda",
    max_steps: int = 45,
    goal_reach_threshold: float = 0.35,
    limit: Optional[int] = None,
    seed: int = 5500,
) -> Dict[str, Any]:
    """Executes the complete Phase 5.5 closed-loop mechanistic grounding study."""
    t0 = time.time()
    specs_path = REPO_ROOT / "world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_2_scene_specs.jsonl"
    ckpt_path = CKPT_DIR / "hjepa_best.pt"
    env_xml_path = REPO_ROOT / "world_models_fpga/env/amr_arena.xml"

    specs = load_phase5_matched_cohort(specs_path, limit=limit)
    env = JetAutoEnv()
    agent = HJEPAAgent(str(ckpt_path), device=device, track="observation")
    oracle = OracleAgent()

    # 1. Physical feasibility check using OracleAgent (Track A upper bound)
    oracle_succ = 0
    for spec in specs:
        env.restore_physical_baseline()
        env.apply_task_spec(spec)
        init_pos = np.asarray(spec["initial_pose"], dtype=np.float32)
        env.reset(options={"initial_pos": (float(init_pos[0]), float(init_pos[1]), float(init_pos[2]))})
        eval_wp = np.asarray(spec["waypoints"][-1], dtype=np.float32)
        reached, col = False, False
        for _ in range(max_steps):
            pose = env.get_robot_pose()
            act, _ = oracle.act(np.zeros((64, 64, 3), dtype=np.uint8), robot_pose=pose, target_pose=eval_wp, task_spec=spec)
            _, _, _, _, info = env.step(act)
            if info.get("collision", False):
                col = True
            if float(np.linalg.norm(env.get_robot_pose()[:2] - eval_wp[:2])) <= goal_reach_threshold:
                reached = True
                break
        if reached and not col:
            oracle_succ += 1

    # 2. Execute all conditions on the exact same matched cohort
    results_by_cond: Dict[str, List[Dict[str, Any]]] = {}
    all_episodes_flat: List[Dict[str, Any]] = []
    npz_payload: Dict[str, np.ndarray] = {}

    for cond_name in ALL_CONDITIONS:
        print(f"Running {cond_name} (N={len(specs)})...")
        cond_recs = []
        for spec in specs:
            rec = evaluate_episode_condition(
                agent=agent,
                env=env,
                spec=spec,
                condition_name=cond_name,
                max_steps=max_steps,
                goal_reach_threshold=goal_reach_threshold,
            )
            traces = rec.pop("traces")
            prefix = f"{cond_name}__{rec['pair_id']}__"
            for tk, tval in traces.items():
                npz_payload[prefix + tk] = tval
            cond_recs.append(rec)
            all_episodes_flat.append(rec)
        results_by_cond[cond_name] = cond_recs
        sr = float(np.mean([r["success"] for r in cond_recs]))
        err = float(np.mean([r["final_eval_goal_error_m"] for r in cond_recs]))
        print(f"  -> {cond_name}: success={sr*100:.1f}% ({sum(r['success'] for r in cond_recs)}/{len(cond_recs)}), mean_final_err={err:.3f}m")

    env.close()

    # 3. Summaries & Holm-adjusted 5 Primary Planned Comparisons
    cond_summaries = {
        cname: _summarize_condition_records(recs, seed=seed + idx * 10)
        for idx, (cname, recs) in enumerate(results_by_cond.items())
    }

    primary_pairs = [
        ("C2_vs_C1_frozen_goal", "C2_frozen_world_goal", "C1_full_closed_loop"),
        ("C3_vs_C1_target_masked", "C3_target_masked_rgb", "C1_full_closed_loop"),
        ("C3_vs_C4_target_vs_distractor_mask", "C3_target_masked_rgb", "C4_distractor_masked_rgb"),
        ("C5_vs_C1_swapped_token", "C5_swapped_task_token", "C1_full_closed_loop"),
        ("C6_vs_C1_oracle_waypoint", "C6_oracle_waypoint", "C1_full_closed_loop"),
    ]
    planned_comparisons: Dict[str, Any] = {}
    raw_ps = []
    comp_names = []
    for i, (comp_key, ca, cb) in enumerate(primary_pairs):
        comp = _paired_comparison(results_by_cond[ca], results_by_cond[cb], ca, cb, seed=seed + 100 + i * 10)
        planned_comparisons[comp_key] = comp
        raw_ps.append(comp["mcnemar_two_sided_p"])
        comp_names.append(comp_key)

    adj_ps = holm_bonferroni_adjust(raw_ps)
    for comp_key, adj_p in zip(comp_names, adj_ps):
        planned_comparisons[comp_key]["holm_adjusted_p_success"] = float(adj_p)

    # Auxiliary secondary comparison (not part of the 5-test primary family)
    auxiliary_comparisons = {
        "A1_vs_C1_visual_only_vs_full": _paired_comparison(
            results_by_cond["A1_visual_only_no_memory"],
            results_by_cond["C1_full_closed_loop"],
            "A1_visual_only_no_memory",
            "C1_full_closed_loop",
            seed=seed + 500,
        )
    }

    # Check static step-0 task contrast vs closed-loop C1/C2 on exact same cohort
    c1_recs = results_by_cond["C1_full_closed_loop"]
    step0_etas = [r["step0_static_contrast_eta_gt"] for r in c1_recs]
    step0_coss = [r["step0_static_contrast_cosine"] for r in c1_recs]
    step0_obs_matches = sum(1 for r in c1_recs if r["obs_0_matches_v5_2_hash"])

    out_dir = REPO_ROOT / "world_models_fpga/research/benchmarks/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "task_conditioned_closed_loop_grounding_v5_5_summary.json"
    ep_jsonl_path = out_dir / "task_conditioned_closed_loop_grounding_v5_5_episodes.jsonl"
    cond_csv_path = out_dir / "task_conditioned_closed_loop_grounding_v5_5_condition_table.csv"
    npz_path = out_dir / "task_conditioned_closed_loop_grounding_v5_5_traces.npz"

    with ep_jsonl_path.open("w", encoding="utf-8") as f:
        for row in all_episodes_flat:
            f.write(json.dumps(row) + "\n")

    np.savez_compressed(npz_path, **npz_payload)

    with cond_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "condition",
            "n_episodes",
            "success_count",
            "success_rate",
            "success_ci_low",
            "success_ci_high",
            "collision_rate",
            "mean_final_goal_error_m",
            "median_final_goal_error_m",
            "mean_path_length_m",
            "mean_executed_steps",
            "mean_step_blended_goal_ang_err_deg",
            "mean_step_action_target_cos",
            "nominal_success_rate",
            "illumination_shift_success_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cname in ALL_CONDITIONS:
            cs = cond_summaries[cname]
            writer.writerow(
                {
                    "condition": cname,
                    "n_episodes": cs["n_episodes"],
                    "success_count": cs["success_count"],
                    "success_rate": cs["success_rate"],
                    "success_ci_low": cs["success_wilson_95ci"][0],
                    "success_ci_high": cs["success_wilson_95ci"][1],
                    "collision_rate": cs["collision_rate"],
                    "mean_final_goal_error_m": cs["mean_final_goal_error_m"],
                    "median_final_goal_error_m": cs["median_final_goal_error_m"],
                    "mean_path_length_m": cs["mean_path_length_m"],
                    "mean_executed_steps": cs["mean_executed_steps"],
                    "mean_step_blended_goal_ang_err_deg": cs["mean_step_blended_goal_ang_err_deg"],
                    "mean_step_action_target_cos": cs["mean_step_action_target_cos"],
                    "nominal_success_rate": cs["by_context"]["nominal"]["success_rate"],
                    "illumination_shift_success_rate": cs["by_context"]["illumination_shift"]["success_rate"],
                }
            )

    git_commit, git_clean = get_git_info(REPO_ROOT)
    elapsed = time.time() - t0

    summary_out = {
        "experiment_name": "Phase 5.5 Closed-Loop Mechanistic Grounding Experiment",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_clean": git_clean,
        "elapsed_seconds": float(elapsed),
        "provenance": {
            "specs_path": str(specs_path.relative_to(REPO_ROOT)),
            "specs_sha256": compute_file_sha256(specs_path),
            "checkpoint_path": str(ckpt_path.relative_to(REPO_ROOT)),
            "checkpoint_sha256": compute_file_sha256(ckpt_path),
            "env_xml_sha256": compute_file_sha256(env_xml_path),
            "device": device,
            "max_steps": max_steps,
            "goal_reach_threshold": goal_reach_threshold,
            "n_scenes": len(specs),
            "obs0_hash_verified_matches": step0_obs_matches,
        },
        "physical_solvability_oracle_expert": {
            "n_episodes": len(specs),
            "success_count": oracle_succ,
            "success_rate": float(oracle_succ / max(1, len(specs))),
        },
        "cohort_static_step0_contrast_baseline": {
            "mean_step0_contrast_eta_gt": float(np.mean(step0_etas)),
            "mean_step0_contrast_cosine": float(np.mean(step0_coss)),
        },
        "conditions": cond_summaries,
        "planned_primary_comparisons_holm5": planned_comparisons,
        "auxiliary_comparisons": auxiliary_comparisons,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    figures_dir = REPO_ROOT / "research/review_package/phase5_5/figures"
    written_figs = generate_figures_from_saved_artifacts(summary_path, ep_jsonl_path, npz_path, figures_dir)
    summary_out["generated_figures"] = [str((figures_dir / fn).relative_to(REPO_ROOT)) for fn in written_figs]
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    return summary_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5.5 Closed-Loop Mechanistic Grounding Experiment")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--goal-threshold", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=None, help="Optional subset size for quick pilot check")
    parser.add_argument("--seed", type=int, default=5500)
    args = parser.parse_args()

    out = run_phase5_5_experiment(
        device=args.device,
        max_steps=args.max_steps,
        goal_reach_threshold=args.goal_threshold,
        limit=args.limit,
        seed=args.seed,
    )
    print("\nPhase 5.5 completed in %.1fs." % out["elapsed_seconds"])
    print("Oracle feasibility:", out["physical_solvability_oracle_expert"])
    for k, c in out["planned_primary_comparisons_holm5"].items():
        print(
            f"  {k}: diff={c['success_rate_diff']*100:+.1f}% "
            f"[{c['success_rate_diff_95ci'][0]*100:+.1f}%, {c['success_rate_diff_95ci'][1]*100:+.1f}%] "
            f"(McNemar p={c['mcnemar_two_sided_p']:.4e}, Holm p_adj={c['holm_adjusted_p_success']:.4e})"
        )


if __name__ == "__main__":
    main()
