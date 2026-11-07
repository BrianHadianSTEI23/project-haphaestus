"""
Phase 6: Compositional Multi-Stage Spatial Agent (Repaired)

Composes the trained ReconstructedCompactModel (or H-JEPA / LeWM) across sequential
relational subgoals in MuJoCo without analytical velocity blending or privileged
obstacle coordinate access.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from research.benchmarks.model_wrappers import compute_body_frame_goal_vec
from research.extraction.extracted_model.reconstructed_compact.model_reconstructed_compact import (
    ReconstructedCompactModel,
)


class CompositionalSpatialAgent:
    """
    Executes multi-stage sequential relational navigation using the distilled
    16D ReconstructedCompactModel (39,804 parameters) via:
      1. CompactEncoder: visual observation -> s_t (16D)
      2. CompactMacroPredictor: evaluates candidate 3-DOF intents around current stage's
         body-frame goal vector to select a collision-free macro subgoal S*_{t+K}
      3. CompactMicroPolicy: maps (s_t, S*_{t+K}) -> holonomic action [vx, vy, wz]
    """
    def __init__(self, compact_ckpt_path: str | None = None, device: str = "cpu"):
        self.device = device
        self.model = ReconstructedCompactModel(latent_dim=16, stride_k=5, hidden_dim=32).to(device)
        if compact_ckpt_path and os.path.exists(compact_ckpt_path):
            self.model.load_state_dict(torch.load(compact_ckpt_path, map_location=device, weights_only=False))
        self.model.eval()

    @torch.no_grad()
    def select_action(self, obs_rgb: np.ndarray, curr_pose: np.ndarray, target_pose: np.ndarray) -> tuple[np.ndarray, dict]:
        goal_vec_body = compute_body_frame_goal_vec(curr_pose, target_pose)
        action, info = self.model.act(obs_rgb, goal_vec_body=goal_vec_body, device=self.device)
        return action, {
            "collision_risk": info["col_risk"],
            "macro_subgoal_norm": float(np.linalg.norm(info["subgoal"])),
        }
