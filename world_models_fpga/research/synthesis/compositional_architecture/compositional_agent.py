"""
Phase 8: Architecture Synthesis & Compositional Intelligence
Script: compositional_agent.py
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Implements the Candidate Compositional Intelligence Architecture derived from navigation principles:
  1. Perception & Semantic Grounding (Extracts body-frame geometric landmarks)
  2. Structured World Representation (16-dim compact invariant latent state)
  3. Predictive Dynamics & Foresight (Action-conditioned residual latent transition)
  4. Hierarchical Decision Mechanism (Macro subgoal planner with safety barrier)
  5. Action Selection & Feedback Tracking (Agile Mecanum micro steering)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from research.extraction.extracted_model.reconstructed_compact.model_reconstructed_compact import ReconstructedCompactModel
from models.h_jepa import HJEPAModel
from models.le_world_model import LeWorldModel

class CompositionalNavigationAgent:
    """
    Modular, Compositional Architecture derived from Principles P1–P5.
    Interfaces each cognitive module with clean, verifiable contracts:
      - encode(img) -> s_t (16-dim geometric state)
      - evaluate_macro_subgoals(s_t, goal_spec) -> S^*_{t+K}, col_risk
      - track_micro_action(s_t, S^*_{t+K}) -> a_t [vx, vy, wz]
    """
    def __init__(self, compact_ckpt=None, hjepa_ckpt=None, lewm_ckpt=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.compact_model = ReconstructedCompactModel(latent_dim=16, stride_k=5, hidden_dim=32).to(self.device)
        if compact_ckpt and os.path.exists(compact_ckpt):
            self.compact_model.load_state_dict(torch.load(compact_ckpt, map_location=self.device))
        self.compact_model.eval()

        # Modular components
        self.perception_encoder = self.compact_model.encoder
        self.macro_predictor = self.compact_model.macro_predictor
        self.micro_policy = self.compact_model.micro_policy

        # External circuit donors for cross-model experiments
        self.lewm_model = None
        if lewm_ckpt and os.path.exists(lewm_ckpt):
            self.lewm_model = LeWorldModel().to(self.device)
            self.lewm_model.load_state_dict(torch.load(lewm_ckpt, map_location=self.device))
            self.lewm_model.eval()

        self.hjepa_model = None
        if hjepa_ckpt and os.path.exists(hjepa_ckpt):
            self.hjepa_model = HJEPAModel(stride_k=5).to(self.device)
            self.hjepa_model.load_state_dict(torch.load(hjepa_ckpt, map_location=self.device))
            self.hjepa_model.eval()

        # Sequential task plan state
        self.current_subgoal_idx = 0
        self.subgoal_queue = []

    def set_subgoal_sequence(self, waypoints_list):
        """Sets a multi-stage compositional goal sequence."""
        self.subgoal_queue = waypoints_list
        self.current_subgoal_idx = 0

    def select_action(self, obs_rgb, robot_pose, current_target=None, goal_dict=None, num_macro_candidates=24):
        """
        Closed-loop execution of the 5-stage compositional architecture:
          Stage 1: Perception & semantic grounding
          Stage 2: Structured world representation
          Stage 3: Predictive dynamics
          Stage 4: Hierarchical decision & barrier evaluation
          Stage 5: Action selection & feedback
        """
        rx, ry, rth = robot_pose[0], robot_pose[1], robot_pose[2]

        # Multi-stage task sequencer
        if self.subgoal_queue and self.current_subgoal_idx < len(self.subgoal_queue):
            active_target = self.subgoal_queue[self.current_subgoal_idx]
            dist_to_subgoal = np.sqrt((active_target[0] - rx)**2 + (active_target[1] - ry)**2)
            if dist_to_subgoal < 0.28: # Subgoal reached threshold
                if self.current_subgoal_idx < len(self.subgoal_queue) - 1:
                    self.current_subgoal_idx += 1
                    active_target = self.subgoal_queue[self.current_subgoal_idx]
        else:
            active_target = current_target if current_target is not None else np.array([0.0, 0.0, 0.0])

        gx, gy = active_target[0], active_target[1]

        # 1. Perception & Geometric Grounding (P3 & P4)
        img_t = torch.from_numpy(obs_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        with torch.no_grad():
            s_curr = self.perception_encoder(img_t) # 16-dim compact latent

        # Arena displacement in body frame
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

        # 2. Predictive Macro Rollouts (P1, P2, P5)
        candidates = [[vx_nom, vy_nom, wz_nom]]
        rng = np.random.default_rng(int((rx + ry) * 1000) % 50000)
        for _ in range(num_macro_candidates - 1):
            vx_v = np.clip(vx_nom + rng.uniform(-0.25, 0.25), -0.5, 0.9)
            vy_v = np.clip(vy_nom + rng.uniform(-0.35, 0.35), -0.7, 0.7)
            wz_v = np.clip(wz_nom + rng.uniform(-0.35, 0.35), -1.0, 1.0)
            candidates.append([vx_v, vy_v, wz_v])

        cands_t = torch.tensor(candidates, dtype=torch.float32, device=self.device)
        s_batch = s_curr.repeat(num_macro_candidates, 1)

        with torch.no_grad():
            next_S, col_logits = self.macro_predictor(s_batch, cands_t)
            col_probs = torch.sigmoid(col_logits).squeeze(-1).cpu().numpy()

        best_score = -float("inf")
        best_act = np.array([vx_nom, vy_nom, wz_nom], dtype=np.float32)
        best_subgoal_S = next_S[0:1]
        dt_macro = 0.5 # 5 steps at 10 Hz

        for i, cand in enumerate(candidates):
            vx_s = cand[0] * 0.6
            vy_s = cand[1] * 0.4
            wz_s = cand[2] * 1.5

            fut_th = rth + wz_s * dt_macro
            fut_dx_w = dx_w - (np.cos(fut_th) * vx_s - np.sin(fut_th) * vy_s) * dt_macro
            fut_dy_w = dy_w - (np.sin(fut_th) * vx_s + np.cos(fut_th) * vy_s) * dt_macro
            fut_dist = np.sqrt(fut_dx_w**2 + fut_dy_w**2)
            dist_gain = (target_dist - fut_dist)

            # Anticipatory Safety Barrier (P5)
            col_pen = 4.5 * col_probs[i]
            score = dist_gain - col_pen

            if score > best_score:
                best_score = score
                best_act = np.array(cand, dtype=np.float32)
                best_subgoal_S = next_S[i:i+1]

        # 3. Micro Action Execution & Feedback Tracking (P3)
        with torch.no_grad():
            micro_act = self.micro_policy(s_curr, best_subgoal_S).squeeze(0).cpu().numpy()

        # Blend policy output with optimal kinematic steering
        final_act = 0.60 * micro_act + 0.40 * best_act
        return np.clip(final_act, -1.0, 1.0).astype(np.float32)
