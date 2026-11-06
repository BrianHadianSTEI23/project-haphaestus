"""
Phase 2: Standardized Model Agents for Behavioral Benchmarking
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Wraps Baseline Policy, LeWorldModel (LeWM), and H-JEPA into a uniform interface:
    agent.select_action(obs_rgb, robot_pose, target_pose, goal_dict=None) -> action_norm
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from models.baseline_policy import BaselineNavigationPolicy
from models.le_world_model import LeWorldModel
from models.h_jepa import HJEPAModel

class BaselineAgent:
    def __init__(self, ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.name = "Baseline Policy"
        self.model_id = "baseline"
        self.device = device
        self.model = BaselineNavigationPolicy().to(self.device)
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

    def select_action(self, obs_rgb, robot_pose, target_pose, goal_dict=None):
        """
        Baseline Goal-Conditioned Reactive Policy.
        """
        rx, ry, rth = robot_pose[0], robot_pose[1], robot_pose[2]
        gx, gy, gth = target_pose[0], target_pose[1], target_pose[2]

        # Arena displacement
        dx_w = gx - rx
        dy_w = gy - ry
        target_dist = np.sqrt(dx_w**2 + dy_w**2)

        # Body-frame projection
        c, s = np.cos(rth), np.sin(rth)
        dx_b = c * dx_w + s * dy_w
        dy_b = -s * dx_w + c * dy_w

        ang_to_goal = np.arctan2(dy_b, dx_b) if target_dist > 0.05 else 0.0
        vx_nom = float(np.clip(0.3 * dx_b, -0.4, 0.5))
        vy_nom = float(np.clip(0.3 * dy_b, -0.4, 0.4))
        wz_nom = float(np.clip(1.5 * ang_to_goal, -1.0, 1.0))

        with torch.no_grad():
            act, col_risk, _ = self.model.act(obs_rgb, robot_pose, target_pose, device=self.device)

        # Blend learned policy output with nominal goal orientation
        act[0] = 0.65 * act[0] + 0.35 * vx_nom
        act[1] = 0.65 * act[1] + 0.35 * vy_nom
        act[2] = 0.50 * act[2] + 0.50 * wz_nom
        return np.clip(act, -1.0, 1.0).astype(np.float32)

class LeWMAgent:
    def __init__(self, ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu", num_candidates=24):
        self.name = "LeWorldModel (LeWM)"
        self.model_id = "lewm"
        self.device = device
        self.num_candidates = num_candidates
        self.model = LeWorldModel().to(self.device)
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

    def select_action(self, obs_rgb, robot_pose, target_pose, goal_dict=None):
        """
        LeWorldModel Latent Foresight MPC:
        Evaluates candidate trajectories using LeWM dynamics predictor to minimize distance
        while penalizing latent collision risk.
        """
        rx, ry, rth = robot_pose[0], robot_pose[1], robot_pose[2]
        gx, gy, gth = target_pose[0], target_pose[1], target_pose[2]

        dx_w = gx - rx
        dy_w = gy - ry
        target_dist = np.sqrt(dx_w**2 + dy_w**2)

        c, s = np.cos(rth), np.sin(rth)
        dx_b = c * dx_w + s * dy_w
        dy_b = -s * dx_w + c * dy_w
        ang_to_goal = np.arctan2(dy_b, dx_b) if target_dist > 0.05 else 0.0

        vx_nom = float(np.clip(0.35 * dx_b, -0.4, 0.6))
        vy_nom = float(np.clip(0.35 * dy_b, -0.4, 0.4))
        wz_nom = float(np.clip(1.6 * ang_to_goal, -1.0, 1.0))

        img_t = torch.from_numpy(obs_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        with torch.no_grad():
            s_curr = self.model.encode(img_t)

        # Generate action candidates around nominal
        cands = [[vx_nom, vy_nom, wz_nom]]
        rng = np.random.default_rng(int((rx + ry) * 100) % 10000)
        for _ in range(self.num_candidates - 1):
            vx_v = np.clip(vx_nom + rng.uniform(-0.25, 0.25), -0.5, 0.9)
            vy_v = np.clip(vy_nom + rng.uniform(-0.35, 0.35), -0.7, 0.7)
            wz_v = np.clip(wz_nom + rng.uniform(-0.35, 0.35), -1.0, 1.0)
            cands.append([vx_v, vy_v, wz_v])

        cands_t = torch.tensor(cands, dtype=torch.float32, device=self.device)
        s_batch = s_curr.repeat(self.num_candidates, 1)

        with torch.no_grad():
            next_s, col_logits = self.model.predict_next(s_batch, cands_t)
            col_probs = torch.sigmoid(col_logits).squeeze(-1).cpu().numpy()

        best_score = -float("inf")
        best_act = np.array([vx_nom, vy_nom, wz_nom], dtype=np.float32)
        dt = 0.1

        for i, cand in enumerate(cands):
            vx_s = cand[0] * 0.6
            vy_s = cand[1] * 0.4
            wz_s = cand[2] * 1.5

            fut_th = rth + wz_s * dt
            fut_dx_w = dx_w - (np.cos(fut_th) * vx_s - np.sin(fut_th) * vy_s) * dt
            fut_dy_w = dy_w - (np.sin(fut_th) * vx_s + np.cos(fut_th) * vy_s) * dt
            fut_dist = np.sqrt(fut_dx_w**2 + fut_dy_w**2)
            dist_gain = (target_dist - fut_dist)

            col_pen = 3.5 * col_probs[i]
            score = dist_gain - col_pen

            if score > best_score:
                best_score = score
                best_act = np.array(cand, dtype=np.float32)

        return np.clip(best_act, -1.0, 1.0).astype(np.float32)

class HJEPAAgent:
    def __init__(self, ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu", num_candidates=24):
        self.name = "H-JEPA Hierarchical"
        self.model_id = "hjepa"
        self.device = device
        self.num_candidates = num_candidates
        self.model = HJEPAModel(stride_k=5).to(self.device)
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

    def select_action(self, obs_rgb, robot_pose, target_pose, goal_dict=None):
        """
        H-JEPA Hierarchical Foresight MPC:
        Level 1 (Macro Stride K=5): Evaluates multi-step path progress and macro obstacle collision over 0.5s horizon.
        """
        rx, ry, rth = robot_pose[0], robot_pose[1], robot_pose[2]
        gx, gy, gth = target_pose[0], target_pose[1], target_pose[2]

        dx_w = gx - rx
        dy_w = gy - ry
        target_dist = np.sqrt(dx_w**2 + dy_w**2)

        c, s = np.cos(rth), np.sin(rth)
        dx_b = c * dx_w + s * dy_w
        dy_b = -s * dx_w + c * dy_w
        ang_to_goal = np.arctan2(dy_b, dx_b) if target_dist > 0.05 else 0.0

        vx_nom = float(np.clip(0.35 * dx_b, -0.4, 0.6))
        vy_nom = float(np.clip(0.35 * dy_b, -0.4, 0.4))
        wz_nom = float(np.clip(1.6 * ang_to_goal, -1.0, 1.0))

        img_t = torch.from_numpy(obs_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        with torch.no_grad():
            s_curr = self.model.encode(img_t)

        cands = [[vx_nom, vy_nom, wz_nom]]
        rng = np.random.default_rng(int((rx + ry) * 100) % 10000)
        for _ in range(self.num_candidates - 1):
            vx_v = np.clip(vx_nom + rng.uniform(-0.25, 0.25), -0.5, 0.9)
            vy_v = np.clip(vy_nom + rng.uniform(-0.35, 0.35), -0.7, 0.7)
            wz_v = np.clip(wz_nom + rng.uniform(-0.35, 0.35), -1.0, 1.0)
            cands.append([vx_v, vy_v, wz_v])

        cands_t = torch.tensor(cands, dtype=torch.float32, device=self.device)
        s_batch = s_curr.repeat(self.num_candidates, 1)

        with torch.no_grad():
            next_S, macro_col_logits = self.model.macro_predictor(s_batch, cands_t)
            col_probs = torch.sigmoid(macro_col_logits).squeeze(-1).cpu().numpy()

        best_score = -float("inf")
        best_act = np.array([vx_nom, vy_nom, wz_nom], dtype=np.float32)
        dt_macro = 0.5 # Stride K=5 at 10 Hz

        for i, cand in enumerate(cands):
            vx_s = cand[0] * 0.6
            vy_s = cand[1] * 0.4
            wz_s = cand[2] * 1.5

            fut_th = rth + wz_s * dt_macro
            fut_dx_w = dx_w - (np.cos(fut_th) * vx_s - np.sin(fut_th) * vy_s) * dt_macro
            fut_dy_w = dy_w - (np.sin(fut_th) * vx_s + np.cos(fut_th) * vy_s) * dt_macro
            fut_dist = np.sqrt(fut_dx_w**2 + fut_dy_w**2)
            dist_gain = (target_dist - fut_dist)

            col_pen = 4.0 * col_probs[i]
            score = dist_gain - col_pen

            if score > best_score:
                best_score = score
                best_act = np.array(cand, dtype=np.float32)

        return np.clip(best_act, -1.0, 1.0).astype(np.float32)
