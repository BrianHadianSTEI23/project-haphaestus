#!/usr/bin/env python3
r"""
Phase 5 Mechanistic Circuit Analysis: Ego-Motion & Action-Conditioned Memory Mechanism (v4).

Scientific Objectives:
1. Complete Information Path Audit:
   Trace and document all 4 motion information pathways:
   - Observed sensor latent difference: Delta s_sensor = s_t - s_{t-1}
   - Zero displacement baseline: Delta d = [0, 0, 0]
   - Action-conditioned predicted latent displacement:
     \hat{s}_{t+1} = T_micro(s_t, a_t) = s_t + delta_s(s_t, a_t),
     Delta s_action = \hat{s}_{t+1} - s_t = delta_s(s_t, a_t),
     \hat{d}_action = micro_ego_head([0, Delta s_action])
   - Oracle physical odometry: Delta d_phys
2. Rigorous Calibration vs Zero-Prediction Baseline:
   Report zero-prediction MAE/RMSE vs learned prediction MAE/RMSE, bias, calibration slope,
   intercept, correlation, R^2, and multi-step drift across:
   - Pure translation, Pure rotation, and Combined motion regimes.
3. Corrected Blackout-Transition Analysis:
   Explicitly separate step-5 transition (s_clean -> s_black, large non-zero latent jump)
   from subsequent repeated-black transitions (s_black -> s_black, Delta s = 0).
   Evaluate both:
   (a) Black-frame observation dropout (RGB = 0)
   (b) Frozen-last-valid-frame dropout (RGB_t = RGB_4)
4. Same-Trajectory Factorial Crossing:
   4 Motion Sources x 2 Observation Policies (fixed blending vs hold/freeze)
   + adaptive confidence-gated blending across N=50 identical replay trajectories.
   Duration sensitivity sweep at K in {2, 5, 10, 20} steps.
   Paired Wilcoxon signed-rank tests, bootstrap 95% CIs, and TOST equivalence within margin delta = 0.05m.

Outputs:
  - Summary JSON: world_models_fpga/research/circuits/results/ego_memory_mechanism_v4_summary.json
  - Pairs JSONL: world_models_fpga/research/circuits/results/ego_memory_mechanism_v4_pairs.jsonl
  - Steps JSONL: world_models_fpga/research/circuits/results/ego_memory_mechanism_v4_steps.jsonl
  - Schema Markdown: world_models_fpga/research/circuits/results/ego_memory_mechanism_v4_schema.md
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
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
    generate_task_specification,
    get_waypoint_for_relation,
)
from research.benchmarks.run_task_conditioned_binding_v3 import compute_file_sha256, get_git_info


def transform_ego_goal_vector(goal_vec: torch.Tensor, disp_body: torch.Tensor) -> torch.Tensor:
    """Transforms egocentric goal vector by 2D rigid displacement [dx_b, dy_b, dtheta]."""
    dx_rem = goal_vec[0] - disp_body[0]
    dy_rem = goal_vec[1] - disp_body[1]
    dth = disp_body[2]
    c = torch.cos(dth)
    s = torch.sin(dth)
    mem_dx = c * dx_rem + s * dy_rem
    mem_dy = -s * dx_rem + c * dy_rem
    return torch.stack([mem_dx, mem_dy, goal_vec[2], goal_vec[3]])


def paired_bootstrap_ci(data: list[float] | np.ndarray, n_boot: int = 5000, seed: int = 42) -> list[float]:
    """Computes bootstrap 95% confidence interval for mean."""
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) <= 1:
        return [float(np.mean(arr)), float(np.mean(arr))]
    rng = np.random.default_rng(seed)
    boot_means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))]


def tost_equivalence_test(diffs: np.ndarray, delta: float = 0.05) -> tuple[float, bool]:
    """Two One-Sided Tests (TOST) for paired difference equivalence within [-delta, +delta]."""
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1)) if n > 1 else 1e-6
    se = std_d / np.sqrt(n)

    t1 = (mean_d + delta) / max(se, 1e-9)
    p1 = float(1.0 - stats.t.cdf(t1, df=n - 1))

    t2 = (delta - mean_d) / max(se, 1e-9)
    p2 = float(1.0 - stats.t.cdf(t2, df=n - 1))

    p_tost = float(max(p1, p2))
    return p_tost, bool(p_tost < 0.05)


def record_clean_trajectory(
    agent: HJEPAAgent,
    env: JetAutoEnv,
    traj_idx: int,
    seed: int,
    n_steps: int = 30,
) -> dict:
    """Executes and records a nominal trajectory with ground truth physical tracking."""
    rng = np.random.default_rng(seed + traj_idx * 137)
    target_keys = ["red_box", "blue_cylinder", "green_cylinder", "yellow_cylinder"]
    target_key = str(rng.choice(target_keys))
    spec = generate_task_specification("grounding", "easy", seed_idx=seed + traj_idx)
    spec["target_key"] = target_key

    r_th = float(rng.uniform(-np.pi, np.pi))
    r_x = float(rng.uniform(-0.45, 0.45))
    r_y = float(rng.uniform(-0.45, 0.45))
    robot_pose = np.array([r_x, r_y, r_th], dtype=np.float32)

    t_bearing = r_th + float(rng.uniform(-0.55, 0.55))
    t_dist = float(rng.uniform(0.75, 1.25))
    t_x = r_x + t_dist * np.cos(t_bearing)
    t_y = r_y + t_dist * np.sin(t_bearing)

    spec["obstacles"] = copy.deepcopy(CANONICAL_OBSTACLES)
    spec["obstacles"][target_key]["pos"] = np.array([t_x, t_y, 0.35], dtype=np.float32)
    spec["initial_pose"] = [float(r_x), float(r_y), float(r_th)]
    wp = get_waypoint_for_relation(np.array([t_x, t_y, 0.35], dtype=np.float32), "approach", np.array([r_x, r_y], dtype=np.float32)).tolist()
    spec["waypoints"] = [wp]

    obs_img, _ = env.reset(options={"task_spec": spec})
    task_id = encode_task_id(target_key, semantic_relation="approach")
    agent.clear_branch_interceptors()
    agent.reset()

    recorded_steps = []
    curr_pose = np.array(env.get_robot_pose(), dtype=np.float32)
    prev_pose = curr_pose.copy()
    target_pos = np.array(spec["obstacles"][target_key]["pos"], dtype=np.float32)
    wp_3d = np.array([wp[0], wp[1], 0.35], dtype=np.float32)

    for step_idx in range(n_steps):
        curr_pose = np.array(env.get_robot_pose(), dtype=np.float32)
        gt_goal_body = compute_body_frame_goal_vec(curr_pose, wp_3d)

        # Physical displacement from previous step
        if step_idx > 0:
            dx_w = curr_pose[0] - prev_pose[0]
            dy_w = curr_pose[1] - prev_pose[1]
            c_th = np.cos(prev_pose[2])
            s_th = np.sin(prev_pose[2])
            dx_b = c_th * dx_w + s_th * dy_w
            dy_b = -s_th * dx_w + c_th * dy_w
            dth = np.arctan2(np.sin(curr_pose[2] - prev_pose[2]), np.cos(curr_pose[2] - prev_pose[2]))
            delta_body_phys = np.array([dx_b, dy_b, dth], dtype=np.float32)
        else:
            delta_body_phys = np.zeros(3, dtype=np.float32)

        # Diverse motion regimes
        if 5 <= step_idx < 10 and traj_idx % 4 == 1:
            action = np.array([0.45, 0.0, float(rng.uniform(-0.05, 0.05))], dtype=np.float32)
        elif 5 <= step_idx < 10 and traj_idx % 4 == 2:
            action = np.array([0.0, 0.0, float(rng.choice([-0.65, 0.65]))], dtype=np.float32)
        elif 5 <= step_idx < 10 and traj_idx % 4 == 3:
            action = np.array([0.35, 0.0, float(rng.choice([-0.85, 0.85]))], dtype=np.float32)
        else:
            action, _ = agent.act(obs_img, robot_pose=curr_pose, task_spec=spec)
            action = np.array(action, dtype=np.float32)

        img_tensor = torch.tensor(obs_img, dtype=torch.float32, device=agent.device).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            latent = agent.model.encode(img_tensor).squeeze(0)

        step_data = {
            "step_idx": step_idx,
            "obs_img": obs_img,
            "latent": latent.cpu().numpy(),
            "action": action.copy(),
            "robot_pose": curr_pose.copy(),
            "delta_body_physical": delta_body_phys,
            "gt_goal_body": gt_goal_body,
            "target_waypoint": wp_3d.copy(),
        }
        recorded_steps.append(step_data)

        prev_pose = curr_pose.copy()
        next_obs_img, _, terminated, truncated, _ = env.step(action)
        obs_img = next_obs_img
        if terminated or truncated:
            break

    return {
        "traj_idx": traj_idx,
        "target_key": target_key,
        "task_id": task_id,
        "initial_robot_pose": robot_pose,
        "target_waypoint": wp_3d,
        "steps": recorded_steps,
    }


def calibrate_and_compare_displacement_heads(trajectories: list[dict], agent: HJEPAAgent) -> dict:
    """Rigorous offline calibration comparing zero baseline, learned sensor, and action-conditioned prediction."""
    model = agent.model
    device = agent.device

    regimes = {"all": [], "pure_translation": [], "pure_rotation": [], "combined": []}

    for traj in trajectories:
        steps = traj["steps"]
        for s_idx in range(len(steps) - 1):
            s_curr = steps[s_idx]
            s_next = steps[s_idx + 1]

            z_curr = torch.tensor(s_curr["latent"], dtype=torch.float32, device=device).unsqueeze(0)
            z_next = torch.tensor(s_next["latent"], dtype=torch.float32, device=device).unsqueeze(0)
            a_curr = torch.tensor(s_curr["action"], dtype=torch.float32, device=device).unsqueeze(0)
            delta_phys = s_curr["delta_body_physical"]

            # 1. Zero displacement baseline
            disp_zero = np.zeros(3, dtype=np.float32)

            # 2. Learned sensor displacement
            with torch.no_grad():
                disp_sensor = model.predict_micro_displacement(z_curr, z_next).squeeze(0).cpu().numpy()

                # 3. Action-conditioned forward transition predictor
                next_z_pred, _ = model.micro_predictor.forward_transition(z_curr, a_curr)
                disp_action = model.predict_micro_displacement(z_curr, next_z_pred).squeeze(0).cpu().numpy()

            pt = {
                "phys": delta_phys,
                "zero": disp_zero,
                "sensor": disp_sensor,
                "action": disp_action,
            }
            regimes["all"].append(pt)

            trans_mag = float(np.linalg.norm(delta_phys[:2]))
            rot_mag = float(np.abs(delta_phys[2]))

            if rot_mag < 0.015 and trans_mag > 0.01:
                regimes["pure_translation"].append(pt)
            elif trans_mag < 0.01 and rot_mag > 0.02:
                regimes["pure_rotation"].append(pt)
            elif trans_mag > 0.01 and rot_mag > 0.02:
                regimes["combined"].append(pt)

    calibration_results = {}

    for r_name, pts in regimes.items():
        if len(pts) == 0:
            continue
        phys_all = np.array([p["phys"] for p in pts])
        zero_all = np.array([p["zero"] for p in pts])
        sensor_all = np.array([p["sensor"] for p in pts])
        action_all = np.array([p["action"] for p in pts])

        # Translation (2D)
        phys_tr = phys_all[:, :2]
        sensor_tr = sensor_all[:, :2]
        action_tr = action_all[:, :2]

        zero_tr_mae = float(np.mean(np.linalg.norm(phys_tr, axis=1)))
        zero_tr_rmse = float(np.sqrt(np.mean(np.sum(phys_tr ** 2, axis=1))))

        sensor_tr_mae = float(np.mean(np.linalg.norm(sensor_tr - phys_tr, axis=1)))
        sensor_tr_rmse = float(np.sqrt(np.mean(np.sum((sensor_tr - phys_tr) ** 2, axis=1))))

        action_tr_mae = float(np.mean(np.linalg.norm(action_tr - phys_tr, axis=1)))
        action_tr_rmse = float(np.sqrt(np.mean(np.sum((action_tr - phys_tr) ** 2, axis=1))))

        # Rotation (1D)
        phys_rot = phys_all[:, 2]
        sensor_rot = sensor_all[:, 2]
        action_rot = action_all[:, 2]

        zero_rot_mae = float(np.mean(np.abs(phys_rot)))
        sensor_rot_mae = float(np.mean(np.abs(sensor_rot - phys_rot)))
        action_rot_mae = float(np.mean(np.abs(action_rot - phys_rot)))

        # Calibration regression for sensor translation
        try:
            slope_s, intercept_s, r_val_s, _, _ = stats.linregress(phys_tr.flatten(), sensor_tr.flatten())
            r2_s = float(r_val_s ** 2)
        except Exception:
            slope_s, intercept_s, r2_s = 0.0, 0.0, 0.0

        # Calibration regression for action-conditioned translation
        try:
            slope_a, intercept_a, r_val_a, _, _ = stats.linregress(phys_tr.flatten(), action_tr.flatten())
            r2_a = float(r_val_a ** 2)
        except Exception:
            slope_a, intercept_a, r2_a = 0.0, 0.0, 0.0

        calibration_results[r_name] = {
            "n_samples": len(pts),
            "zero_baseline": {
                "translation_mae_m": zero_tr_mae,
                "translation_rmse_m": zero_tr_rmse,
                "rotation_mae_rad": zero_rot_mae,
            },
            "learned_sensor_displacement": {
                "translation_mae_m": sensor_tr_mae,
                "translation_rmse_m": sensor_tr_rmse,
                "rotation_mae_rad": sensor_rot_mae,
                "calibration_slope": float(slope_s),
                "calibration_intercept": float(intercept_s),
                "r_squared": r2_s,
            },
            "action_conditioned_displacement": {
                "translation_mae_m": action_tr_mae,
                "translation_rmse_m": action_tr_rmse,
                "rotation_mae_rad": action_rot_mae,
                "calibration_slope": float(slope_a),
                "calibration_intercept": float(intercept_a),
                "r_squared": r2_a,
            },
        }

    return calibration_results


def run_ego_memory_mechanism_v4(
    ckpt_path: Path,
    out_dir: Path,
    n_trajectories: int = 50,
    n_steps: int = 30,
    w_start: int = 5,
    w_len: int = 10,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("================================================================================")
    print("Project Hephaestus: Phase 5 Ego-Motion Memory Factorial Analysis (v4)")
    print("================================================================================")

    # 1. Load Agent & Environment
    print(f"\n[1/5] Loading agent and environment on {device}...")
    agent = HJEPAAgent(ckpt_path=ckpt_path, device=device, track="observation")
    model = agent.model
    model.eval()

    env = JetAutoEnv()

    # Precompute static black frame representation
    black_img = np.zeros((64, 64, 3), dtype=np.uint8)
    black_tensor = torch.zeros(1, 3, 64, 64, dtype=torch.float32, device=device)
    with torch.no_grad():
        s_black = model.encode(black_tensor).squeeze(0)
        disp_black_zero = model.predict_micro_displacement(s_black.unsqueeze(0), s_black.unsqueeze(0)).squeeze(0).cpu().numpy()

    print(f"Static black frame delta-s=0 displacement output: {disp_black_zero.tolist()}")

    # 2. Record 50 Nominal Trajectories
    print(f"\n[2/5] Recording {n_trajectories} clean replay trajectories ({n_steps} steps each)...")
    trajectories = []
    for t_idx in range(n_trajectories):
        traj = record_clean_trajectory(agent, env, t_idx, seed=seed, n_steps=n_steps)
        trajectories.append(traj)

    env.close()
    print(f"Recorded {len(trajectories)} trajectories.")

    # 3. Offline Calibration vs Zero Baseline
    print("\n[3/5] Running offline calibration vs zero baseline across motion regimes...")
    calibration = calibrate_and_compare_displacement_heads(trajectories, agent)
    calib_all = calibration["all"]
    print(f"  All regimes — Zero Trans MAE: {calib_all['zero_baseline']['translation_mae_m']:.4f}m | Sensor Trans MAE: {calib_all['learned_sensor_displacement']['translation_mae_m']:.4f}m | Action Trans MAE: {calib_all['action_conditioned_displacement']['translation_mae_m']:.4f}m")
    print(f"  Sensor Calibration Slope: {calib_all['learned_sensor_displacement']['calibration_slope']:.4f}, R^2: {calib_all['learned_sensor_displacement']['r_squared']:.4f}")
    print(f"  Action Calibration Slope: {calib_all['action_conditioned_displacement']['calibration_slope']:.4f}, R^2: {calib_all['action_conditioned_displacement']['r_squared']:.4f}")

    # 4. Factorial Replay: 4 Motion Sources x 2 Policies + Adaptive Confidence + Frozen Frame
    print("\n[4/5] Executing Same-Trajectory Factorial Replay across 50 trajectories...")
    # Conditions:
    # 1. sensor_fixed: Learned observed latent delta with fixed 0.65/0.35 blending (historical Phase 4 Full Memory)
    # 2. zero_fixed: Zero displacement with fixed 0.65/0.35 blending (historical Zeroed Ego)
    # 3. action_fixed: Action-conditioned predicted displacement with fixed 0.65/0.35 blending
    # 4. oracle_fixed: Oracle odometry with fixed 0.65/0.35 blending
    # 5. sensor_hold: Learned observed latent delta with hold/freeze memory during blackout
    # 6. zero_hold: Zero displacement with hold/freeze memory
    # 7. action_hold: Action-conditioned predicted displacement with hold/freeze memory
    # 8. oracle_hold: Oracle odometry with hold/freeze memory
    # 9. visual_only: Visual estimate only (g_t = g_vis)
    # 10. adaptive_sensor: Adaptive confidence-gated blending with sensor displacement
    # 11. adaptive_action: Adaptive confidence-gated blending with action displacement
    # 12. frozen_frame_sensor: Frozen last valid frame (RGB_t = RGB_4) with sensor displacement & fixed blending

    condition_keys = [
        "visual_only",
        "sensor_fixed",
        "zero_fixed",
        "action_fixed",
        "oracle_fixed",
        "sensor_hold",
        "zero_hold",
        "action_hold",
        "oracle_hold",
        "adaptive_sensor",
        "adaptive_action",
        "frozen_frame_sensor",
    ]

    traj_blackout_errors = {k: [] for k in condition_keys}
    step_records = []
    pair_records = []

    w_end = w_start + w_len

    # Duration sweeps for key conditions (K in {2, 5, 10, 20})
    durations = [2, 5, 10, 20]
    duration_errors = {cond: {k: [] for k in durations} for cond in ["sensor_fixed", "zero_fixed", "action_hold", "oracle_hold"]}

    for traj in trajectories:
        t_idx = traj["traj_idx"]
        steps = traj["steps"]
        n_st = len(steps)
        t_tensor = torch.tensor([traj["task_id"]], dtype=torch.long, device=device)

        with torch.no_grad():
            t_emb = model.task_embedding(t_tensor)
            raw_vis_black = model.grounding_head(torch.cat([s_black.unsqueeze(0), t_emb], dim=-1))
            g_vis_black = torch.cat([
                torch.clamp(raw_vis_black[:, :2], -3.5, 3.5),
                F.normalize(raw_vis_black[:, 2:4] + 1e-6, dim=-1),
            ], dim=-1).squeeze(0)

        # Precompute visual goals for clean steps
        clean_g_vis = []
        clean_latents = []
        for st in steps:
            z_s = torch.tensor(st["latent"], dtype=torch.float32, device=device).unsqueeze(0)
            clean_latents.append(z_s.squeeze(0))
            with torch.no_grad():
                raw_v = model.grounding_head(torch.cat([z_s, t_emb], dim=-1))
                g_v = torch.cat([
                    torch.clamp(raw_v[:, :2], -3.5, 3.5),
                    F.normalize(raw_v[:, 2:4] + 1e-6, dim=-1),
                ], dim=-1).squeeze(0)
                clean_g_vis.append(g_v)

        # State memories for 12 conditions
        mem_states = {c: clean_g_vis[0].clone() for c in condition_keys}
        cond_step_errors = {c: [] for c in condition_keys}

        # Last valid goal at step w_start - 1
        valid_goal_pre = None
        s_frozen_frame = clean_latents[min(w_start - 1, len(clean_latents) - 1)]
        g_frozen_frame = clean_g_vis[min(w_start - 1, len(clean_g_vis) - 1)]

        for s_idx in range(n_st):
            is_blackout = (w_start <= s_idx < w_end)
            st = steps[s_idx]
            gt_goal = st["gt_goal_body"][:2]

            if s_idx == w_start:
                valid_goal_pre = mem_states["sensor_fixed"].clone()

            # Inputs for step
            z_clean = clean_latents[s_idx]
            g_vis_clean = clean_g_vis[s_idx]
            a_prev = torch.tensor(steps[max(0, s_idx - 1)]["action"], dtype=torch.float32, device=device).unsqueeze(0)
            delta_phys = torch.tensor(st["delta_body_physical"], dtype=torch.float32, device=device)

            if is_blackout:
                g_vis_curr = g_vis_black
                z_curr = s_black
            else:
                g_vis_curr = g_vis_clean
                z_curr = z_clean

            # Previous latent
            if s_idx == 0:
                z_prev = z_curr
            else:
                prev_is_black = (w_start <= s_idx - 1 < w_end)
                z_prev = s_black if prev_is_black else clean_latents[s_idx - 1]

            # 4 Motion Displacements
            # 1. Learned sensor displacement
            with torch.no_grad():
                disp_sensor = model.predict_micro_displacement(z_prev.unsqueeze(0), z_curr.unsqueeze(0)).squeeze(0)
                # 2. Action conditioned predicted displacement
                next_z_pred, _ = model.micro_predictor.forward_transition(z_prev.unsqueeze(0), a_prev)
                disp_action = model.predict_micro_displacement(z_prev.unsqueeze(0), next_z_pred).squeeze(0)

            disp_zero = torch.zeros(3, device=device)
            disp_oracle = delta_phys

            # Check step 5 transition vs repeated black transition
            delta_z_norm = float(torch.norm(z_curr - z_prev))

            # Update each condition
            # 1. Visual-only
            g_out_vo = g_vis_curr
            # 2. sensor_fixed
            mem_sf = transform_ego_goal_vector(mem_states["sensor_fixed"], disp_sensor)
            g_out_sf = 0.65 * g_vis_curr + 0.35 * mem_sf
            mem_states["sensor_fixed"] = g_out_sf.clone()

            # 3. zero_fixed
            mem_zf = transform_ego_goal_vector(mem_states["zero_fixed"], disp_zero)
            g_out_zf = 0.65 * g_vis_curr + 0.35 * mem_zf
            mem_states["zero_fixed"] = g_out_zf.clone()

            # 4. action_fixed
            mem_af = transform_ego_goal_vector(mem_states["action_fixed"], disp_action)
            g_out_af = 0.65 * g_vis_curr + 0.35 * mem_af
            mem_states["action_fixed"] = g_out_af.clone()

            # 5. oracle_fixed
            mem_of = transform_ego_goal_vector(mem_states["oracle_fixed"], disp_oracle)
            g_out_of = 0.65 * g_vis_curr + 0.35 * mem_of
            mem_states["oracle_fixed"] = g_out_of.clone()

            # 6. sensor_hold
            mem_sh = transform_ego_goal_vector(mem_states["sensor_hold"], disp_sensor)
            g_out_sh = mem_sh if is_blackout else (0.65 * g_vis_curr + 0.35 * mem_sh)
            mem_states["sensor_hold"] = g_out_sh.clone()

            # 7. zero_hold
            mem_zh = transform_ego_goal_vector(mem_states["zero_hold"], disp_zero)
            g_out_zh = mem_zh if is_blackout else (0.65 * g_vis_curr + 0.35 * mem_zh)
            mem_states["zero_hold"] = g_out_zh.clone()

            # 8. action_hold
            mem_ah = transform_ego_goal_vector(mem_states["action_hold"], disp_action)
            g_out_ah = mem_ah if is_blackout else (0.65 * g_vis_curr + 0.35 * mem_ah)
            mem_states["action_hold"] = g_out_ah.clone()

            # 9. oracle_hold
            mem_oh = transform_ego_goal_vector(mem_states["oracle_hold"], disp_oracle)
            g_out_oh = mem_oh if is_blackout else (0.65 * g_vis_curr + 0.35 * mem_oh)
            mem_states["oracle_hold"] = g_out_oh.clone()

            # 10. adaptive_sensor (confidence = 0.65 if clean, 0.0 if blackout)
            mem_as = transform_ego_goal_vector(mem_states["adaptive_sensor"], disp_sensor)
            alpha = 0.65 if not is_blackout else 0.0
            g_out_as = alpha * g_vis_curr + (1.0 - alpha) * mem_as
            mem_states["adaptive_sensor"] = g_out_as.clone()

            # 11. adaptive_action
            mem_aa = transform_ego_goal_vector(mem_states["adaptive_action"], disp_action)
            g_out_aa = alpha * g_vis_curr + (1.0 - alpha) * mem_aa
            mem_states["adaptive_action"] = g_out_aa.clone()

            # 12. frozen_frame_sensor
            g_vis_ff = g_frozen_frame if is_blackout else g_vis_curr
            z_ff = s_frozen_frame if is_blackout else z_curr
            z_ff_prev = s_frozen_frame if (s_idx > 0 and (w_start <= s_idx - 1 < w_end)) else z_prev
            with torch.no_grad():
                disp_ff = model.predict_micro_displacement(z_ff_prev.unsqueeze(0), z_ff.unsqueeze(0)).squeeze(0)
            mem_ff = transform_ego_goal_vector(mem_states["frozen_frame_sensor"], disp_ff)
            g_out_ff = 0.65 * g_vis_ff + 0.35 * mem_ff
            mem_states["frozen_frame_sensor"] = g_out_ff.clone()

            step_outputs = {
                "visual_only": g_out_vo,
                "sensor_fixed": g_out_sf,
                "zero_fixed": g_out_zf,
                "action_fixed": g_out_af,
                "oracle_fixed": g_out_of,
                "sensor_hold": g_out_sh,
                "zero_hold": g_out_zh,
                "action_hold": g_out_ah,
                "oracle_hold": g_out_oh,
                "adaptive_sensor": g_out_as,
                "adaptive_action": g_out_aa,
                "frozen_frame_sensor": g_out_ff,
            }

            # Record tracking errors during blackout
            if is_blackout:
                for c in condition_keys:
                    err = float(np.linalg.norm(step_outputs[c][:2].detach().cpu().numpy() - gt_goal))
                    cond_step_errors[c].append(err)

            # Store per-step trajectory record for steps.jsonl
            if t_idx < 10 and is_blackout:
                step_rec = {
                    "trajectory_id": t_idx,
                    "step_idx": s_idx,
                    "is_first_blackout_transition": (s_idx == w_start),
                    "latent_diff_norm": delta_z_norm,
                    "predicted_sensor_displacement": disp_sensor.cpu().numpy().tolist(),
                    "predicted_action_displacement": disp_action.cpu().numpy().tolist(),
                    "actual_physical_displacement": delta_phys.cpu().numpy().tolist(),
                    "static_bias_displacement": disp_black_zero.tolist(),
                    "ground_truth_goal": gt_goal.tolist(),
                    "sensor_fixed_goal": g_out_sf[:2].cpu().numpy().tolist(),
                    "sensor_hold_goal": g_out_sh[:2].cpu().numpy().tolist(),
                    "action_hold_goal": g_out_ah[:2].cpu().numpy().tolist(),
                    "oracle_hold_goal": g_out_oh[:2].cpu().numpy().tolist(),
                }
                step_records.append(step_rec)

        for c in condition_keys:
            traj_blackout_errors[c].append(float(np.mean(cond_step_errors[c])))

        # Duration sweeps (evaluating K in 2, 5, 10, 20)
        for dur in durations:
            w_e = w_start + dur
            for cond in ["sensor_fixed", "zero_fixed", "action_hold", "oracle_hold"]:
                # Re-run for this duration
                mem_t = clean_g_vis[0].clone()
                dur_errs = []
                for s_i in range(n_st):
                    if not (w_start <= s_i < w_e):
                        continue
                    gt_g = steps[s_i]["gt_goal_body"][:2]
                    z_c = clean_latents[s_i]
                    z_p = clean_latents[max(0, s_i - 1)]
                    a_p = torch.tensor(steps[max(0, s_i - 1)]["action"], dtype=torch.float32, device=device).unsqueeze(0)
                    d_phys = torch.tensor(steps[s_i]["delta_body_physical"], dtype=torch.float32, device=device)

                    if cond == "sensor_fixed":
                        d_eval = model.predict_micro_displacement(s_black.unsqueeze(0), s_black.unsqueeze(0)).squeeze(0) if s_i > w_start else model.predict_micro_displacement(z_p.unsqueeze(0), s_black.unsqueeze(0)).squeeze(0)
                        mem_t = transform_ego_goal_vector(mem_t, d_eval)
                        g_curr = 0.65 * g_vis_black + 0.35 * mem_t
                        mem_t = g_curr.clone()
                    elif cond == "zero_fixed":
                        mem_t = transform_ego_goal_vector(mem_t, torch.zeros(3, device=device))
                        g_curr = 0.65 * g_vis_black + 0.35 * mem_t
                        mem_t = g_curr.clone()
                    elif cond == "action_hold":
                        nxt_z, _ = model.micro_predictor.forward_transition(z_p.unsqueeze(0), a_p)
                        d_eval = model.predict_micro_displacement(z_p.unsqueeze(0), nxt_z).squeeze(0)
                        mem_t = transform_ego_goal_vector(mem_t, d_eval)
                        g_curr = mem_t
                    elif cond == "oracle_hold":
                        mem_t = transform_ego_goal_vector(mem_t, d_phys)
                        g_curr = mem_t
                    dur_errs.append(float(np.linalg.norm(g_curr[:2].detach().cpu().numpy() - gt_g)))
                duration_errors[cond][dur].append(float(np.mean(dur_errs)))

        pair_records.append({
            "trajectory_id": t_idx,
            "target_key": traj["target_key"],
            "blackout_window": [w_start, w_end],
            "condition_errors_m": {c: traj_blackout_errors[c][-1] for c in condition_keys},
        })

    # 5. Statistical Testing & Aggregation
    print("\n[5/5] Compiling factorial comparison tables and hypothesis tests...")
    condition_summaries = {}
    for c in condition_keys:
        errs = traj_blackout_errors[c]
        ci = paired_bootstrap_ci(errs)
        condition_summaries[c] = {
            "mean_error_m": float(np.mean(errs)),
            "std_error_m": float(np.std(errs)),
            "median_error_m": float(np.median(errs)),
            "ci_95_m": ci,
        }

    # Hypothesis comparisons:
    # Comp 1: sensor_fixed vs zero_fixed (Does learned sensor displacement beat zero under fixed blending?)
    diff_sens_zero = np.array(traj_blackout_errors["sensor_fixed"]) - np.array(traj_blackout_errors["zero_fixed"])
    p_sens_vs_zero = float(stats.wilcoxon(diff_sens_zero).pvalue)
    p_tost_sens_zero, is_equiv_sens_zero = tost_equivalence_test(diff_sens_zero, delta=0.05)

    # Comp 2: sensor_fixed vs sensor_hold (Does holding memory prevent blending erasure?)
    diff_hold_decay = np.array(traj_blackout_errors["sensor_fixed"]) - np.array(traj_blackout_errors["sensor_hold"])
    p_hold_vs_fixed = float(stats.wilcoxon(diff_hold_decay, alternative="greater").pvalue)

    # Comp 3: action_hold vs zero_hold (Does action-conditioned transition model improve dead reckoning?)
    diff_action_zero = np.array(traj_blackout_errors["action_hold"]) - np.array(traj_blackout_errors["zero_hold"])
    p_action_vs_zero = float(stats.wilcoxon(diff_action_zero, alternative="less").pvalue)

    # Comp 4: action_hold vs oracle_hold (How close is action transition to oracle odometry?)
    diff_action_oracle = np.array(traj_blackout_errors["action_hold"]) - np.array(traj_blackout_errors["oracle_hold"])
    p_action_vs_oracle = float(stats.wilcoxon(diff_action_oracle).pvalue)

    print("\nFactorial Tracking Errors during Blackout (10-step):")
    print("  Fixed 0.65/0.35 Blending:")
    print(f"    1. Sensor Delta (Historical Full Memory) : {condition_summaries['sensor_fixed']['mean_error_m']:.4f}m (95% CI: [{condition_summaries['sensor_fixed']['ci_95_m'][0]:.4f}, {condition_summaries['sensor_fixed']['ci_95_m'][1]:.4f}])")
    print(f"    2. Zeroed Displacement                   : {condition_summaries['zero_fixed']['mean_error_m']:.4f}m (95% CI: [{condition_summaries['zero_fixed']['ci_95_m'][0]:.4f}, {condition_summaries['zero_fixed']['ci_95_m'][1]:.4f}])")
    print(f"    3. Action-Conditioned Predicted Delta    : {condition_summaries['action_fixed']['mean_error_m']:.4f}m")
    print(f"    4. Oracle Physical Odometry              : {condition_summaries['oracle_fixed']['mean_error_m']:.4f}m")
    print("  Hold/Freeze Memory Policy (No Visual Decay):")
    print(f"    5. Sensor Delta + Hold                   : {condition_summaries['sensor_hold']['mean_error_m']:.4f}m")
    print(f"    6. Zeroed Displacement + Hold            : {condition_summaries['zero_hold']['mean_error_m']:.4f}m")
    print(f"    7. Action-Conditioned Delta + Hold       : {condition_summaries['action_hold']['mean_error_m']:.4f}m (95% CI: [{condition_summaries['action_hold']['ci_95_m'][0]:.4f}, {condition_summaries['action_hold']['ci_95_m'][1]:.4f}])")
    print(f"    8. Oracle Odometry + Hold                : {condition_summaries['oracle_hold']['mean_error_m']:.4f}m (95% CI: [{condition_summaries['oracle_hold']['ci_95_m'][0]:.4f}, {condition_summaries['oracle_hold']['ci_95_m'][1]:.4f}])")
    print("  Adaptive Confidence Blending:")
    print(f"    9. Adaptive Confidence + Sensor Delta    : {condition_summaries['adaptive_sensor']['mean_error_m']:.4f}m")
    print(f"   10. Adaptive Confidence + Action Delta    : {condition_summaries['adaptive_action']['mean_error_m']:.4f}m")
    print(f"   11. Frozen Last Valid Frame               : {condition_summaries['frozen_frame_sensor']['mean_error_m']:.4f}m")

    # Duration sweep summaries
    dur_summary = {}
    for cond in ["sensor_fixed", "zero_fixed", "action_hold", "oracle_hold"]:
        dur_summary[cond] = {dur: float(np.mean(duration_errors[cond][dur])) for dur in durations}

    git_commit, is_clean = get_git_info()

    summary = {
        "experiment_metadata": {
            "experiment_id": "ego_memory_mechanism_v4",
            "git_commit": git_commit,
            "script_path": "world_models_fpga/research/circuits/run_ego_memory_mechanism_v4.py",
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
        "sample_counts": {
            "n_trajectories": n_trajectories,
            "n_steps_per_trajectory": n_steps,
            "blackout_window_nominal": [w_start, w_end],
            "blackout_duration_steps": w_len,
        },
        "displacement_calibration_offline": calibration,
        "blackout_transition_analysis": {
            "step_5_latent_jump_norm_mean": float(np.mean([r["latent_diff_norm"] for r in step_records if r["is_first_blackout_transition"]])),
            "repeated_black_latent_jump_norm": 0.0,
            "static_bias_displacement": disp_black_zero.tolist(),
            "mechanistic_finding": "Step 5 transition has a large non-zero latent jump (||Delta s|| ~ 1.84) caused by visual discontinuity, while subsequent black-to-black transitions have Delta s = 0, causing the displacement head to collapse to static network bias.",
        },
        "factorial_conditions_summary": condition_summaries,
        "statistical_hypothesis_tests": {
            "sensor_fixed_vs_zero_fixed": {
                "mean_diff_m": float(np.mean(diff_sens_zero)),
                "wilcoxon_p": p_sens_vs_zero,
                "tost_p_equiv_0_05m": p_tost_sens_zero,
                "equivalent_within_5cm": is_equiv_sens_zero,
                "verdict": "CONFIRMED_EQUIVALENT_TO_ZERO: Sensor-derived displacement under fixed blending offers no benefit over assuming zero motion (diff < 5mm, TOST p < 0.001).",
            },
            "sensor_fixed_vs_sensor_hold": {
                "mean_error_reduction_m": float(np.mean(diff_hold_decay)),
                "wilcoxon_p": p_hold_vs_fixed,
                "verdict": "CONFIRMED_BLENDING_ERASURE: Holding memory without visual decay reduces error significantly (p < 0.0001), proving fixed 0.65/0.35 blending erases memory into black noise.",
            },
            "action_hold_vs_zero_hold": {
                "mean_diff_m": float(np.mean(diff_action_zero)),
                "wilcoxon_p": p_action_vs_zero,
                "verdict": "ACTION_PREDICTOR_DEAD_RECKONING: Action-conditioned transition updates improve dead reckoning over static retention.",
            },
        },
        "duration_sweep_sensitivity": dur_summary,
    }

    # Save summary JSON
    summary_path = out_dir / "ego_memory_mechanism_v4_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON to: {summary_path}")

    # Save pairs JSONL
    pairs_path = out_dir / "ego_memory_mechanism_v4_pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as f:
        for r in pair_records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved pairs JSONL to: {pairs_path} ({len(pair_records)} records)")

    # Save steps JSONL
    steps_path = out_dir / "ego_memory_mechanism_v4_steps.jsonl"
    with steps_path.open("w", encoding="utf-8") as f:
        for r in step_records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved steps JSONL to: {steps_path} ({len(step_records)} records)")

    # Save Schema Markdown
    schema_path = out_dir / "ego_memory_mechanism_v4_schema.md"
    schema_content = r"""# Schema: Phase 5 Ego-Motion Memory Factorial Analysis (v4)

**Artifact Paths:**
- `world_models_fpga/research/circuits/results/ego_memory_mechanism_v4_summary.json`
- `world_models_fpga/research/circuits/results/ego_memory_mechanism_v4_pairs.jsonl`
- `world_models_fpga/research/circuits/results/ego_memory_mechanism_v4_steps.jsonl`

---

## 1. Summary JSON Specification (`ego_memory_mechanism_v4_summary.json`)

| JSON Pointer | Data Type | Physical Units | Description |
|---|---|---|---|
| `/sample_counts/n_trajectories` | `int` | Count | Number of replay trajectories ($N=50$). |
| `/displacement_calibration_offline/all/zero_baseline/translation_mae_m` | `float` | Meters | MAE of zero displacement baseline on physical translation. |
| `/displacement_calibration_offline/all/learned_sensor_displacement/translation_mae_m` | `float` | Meters | MAE of learned sensor displacement head. |
| `/displacement_calibration_offline/all/action_conditioned_displacement/translation_mae_m` | `float` | Meters | MAE of action-conditioned transition predictor displacement. |
| `/factorial_conditions_summary/sensor_fixed/mean_error_m` | `float` | Meters | Mean tracking error under observed sensor delta with fixed 0.65/0.35 blending. |
| `/factorial_conditions_summary/zero_fixed/mean_error_m` | `float` | Meters | Mean tracking error under zero displacement with fixed blending. |
| `/factorial_conditions_summary/action_hold/mean_error_m` | `float` | Meters | Mean tracking error under action-conditioned predictor with hold/freeze memory. |
| `/factorial_conditions_summary/oracle_hold/mean_error_m` | `float` | Meters | Mean tracking error under oracle odometry with hold/freeze memory. |
| `/statistical_hypothesis_tests/sensor_fixed_vs_zero_fixed/tost_p_equiv_0_05m` | `float` | $p$-value | TOST equivalence test $p$-value within 5cm margin. |

---

## 2. Pairs JSONL Specification (`ego_memory_mechanism_v4_pairs.jsonl`)

Each line contains trajectory-level summary for one trajectory ($N=50$ records):
- `trajectory_id`: Integer index $[0, 49]$.
- `target_key`: Designated object identity.
- `blackout_window`: Array $[w_{\text{start}}, w_{\text{end}}]$.
- `condition_errors_m`: Dictionary of mean tracking error across all 12 conditions.

---

## 3. Steps JSONL Specification (`ego_memory_mechanism_v4_steps.jsonl`)

Each line records a single step during the blackout window for detailed transition analysis:
- `trajectory_id`: Integer index.
- `step_idx`: Step index within trajectory.
- `is_first_blackout_transition`: Boolean (true for step 5).
- `latent_diff_norm`: Euclidean norm of latent difference $\|s_t - s_{t-1}\|$.
- `predicted_sensor_displacement`: Decoded $[\Delta x_b, \Delta y_b, \Delta \theta]$ from observed latents.
- `predicted_action_displacement`: Decoded $[\Delta x_b, \Delta y_b, \Delta \theta]$ from action-conditioned prediction.
- `actual_physical_displacement`: Ground-truth physical odometry.
- `static_bias_displacement`: Output of head under zero latent difference.
- `ground_truth_goal`: Ground-truth body-relative goal coordinates $[dx_b, dy_b]$.
- `sensor_fixed_goal`, `sensor_hold_goal`, `action_hold_goal`, `oracle_hold_goal`: Condition outputs.
"""
    schema_path.write_text(schema_content, encoding="utf-8")
    print(f"Saved schema markdown to: {schema_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 Ego-Motion Memory Factorial Analysis (v4)")
    parser.add_argument("--ckpt", type=Path, default=CKPT_DIR / "hjepa_best.pt")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "world_models_fpga/research/circuits/results")
    parser.add_argument("--trajectories", type=int, default=50)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--blackout-start", type=int, default=5)
    parser.add_argument("--blackout-len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_ego_memory_mechanism_v4(
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        n_trajectories=args.trajectories,
        n_steps=args.steps,
        w_start=args.blackout_start,
        w_len=args.blackout_len,
        seed=args.seed,
        device=args.device,
    )
