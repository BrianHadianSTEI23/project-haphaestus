"""
Phase 3: Model-Specific Activation Capture & Tensor Mapping
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Maps internal layer representations for:
  - Baseline Navigation Policy
  - LeWorldModel (LeWM)
  - H-JEPA Hierarchical Architecture
Extracts all 10 mandatory mechanistic discovery representations:
  1. Raw image embedding
  2. Goal embedding
  3. Current world-state representation
  4. Action embedding
  5. Predicted future representation
  6. Prediction residual
  7. Macro-level latent representations
  8. Micro-level latent representations
  9. Policy outputs
  10. Intermediate controller outputs (collision risk)
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from research.instrumentation.hooks import HookManager

class ModelLayerRegistry:
    """
    Identifies and registers named hook probes for each model architecture.
    """
    @staticmethod
    def instrument_baseline(model: nn.Module, hook_mgr: HookManager):
        """
        Instruments BaselineNavigationPolicy:
          encoder.conv1..conv4
          encoder.fc: Linear (1024 -> 32) -> visual latent z
          fc1: Linear (36 -> 64)
          fc2: Linear (64 -> 64)
          action_head: Linear (64 -> 3)
          collision_head: Linear (64 -> 1)
        """
        hook_mgr.register_forward_capture(model.encoder.conv1, "baseline.conv1")
        hook_mgr.register_forward_capture(model.encoder.conv2, "baseline.conv2")
        hook_mgr.register_forward_capture(model.encoder.conv3, "baseline.conv3")
        hook_mgr.register_forward_capture(model.encoder.conv4, "baseline.conv4")
        hook_mgr.register_forward_capture(model.encoder.fc, "baseline.visual_latent_z")

        hook_mgr.register_forward_capture(model.fc1, "baseline.policy_fc1")
        hook_mgr.register_forward_capture(model.fc2, "baseline.policy_fc2")
        hook_mgr.register_forward_capture(model.action_head, "baseline.action_head")
        hook_mgr.register_forward_capture(model.collision_head, "baseline.col_head")

    @staticmethod
    def instrument_lewm(model: nn.Module, hook_mgr: HookManager):
        """
        Instruments LeWorldModel:
          encoder[0..8]: Conv layers + Flatten
          encoder[9]: Linear (1024 -> 32)
          encoder[10]: LayerNorm -> Latent s_t
          pred_fc1: Linear (35 -> 64)
          pred_fc2: Linear (64 -> 64)
          pred_delta: Linear (64 -> 32) -> Delta s
          col_fc: Linear (64 -> 1) -> Collision logit
        """
        hook_mgr.register_forward_capture(model.encoder[0], "lewm.conv1")
        hook_mgr.register_forward_capture(model.encoder[2], "lewm.conv2")
        hook_mgr.register_forward_capture(model.encoder[4], "lewm.conv3")
        hook_mgr.register_forward_capture(model.encoder[6], "lewm.conv4")
        hook_mgr.register_forward_capture(model.encoder[8], "lewm.flatten")
        hook_mgr.register_forward_capture(model.encoder[10], "lewm.latent_state_s")

        hook_mgr.register_forward_capture(model.pred_fc1, "lewm.pred_fc1")
        hook_mgr.register_forward_capture(model.pred_fc2, "lewm.pred_fc2")
        hook_mgr.register_forward_capture(model.pred_delta, "lewm.delta_s")
        hook_mgr.register_forward_capture(model.col_fc, "lewm.col_head")

    @staticmethod
    def instrument_hjepa(model: nn.Module, hook_mgr: HookManager):
        """
        Instruments H-JEPA Hierarchical Model:
          Shared Visual Encoder -> s_t
          Macro Predictor: macro_predictor.fc1, fc2, delta_head, col_head
          Micro Predictor Policy: pol_fc1, pol_fc2, action_head
          Micro Predictor Dynamics: dyn_fc1, dyn_fc2, delta_head, col_head
        """
        hook_mgr.register_forward_capture(model.encoder[0], "hjepa.conv1")
        hook_mgr.register_forward_capture(model.encoder[4], "hjepa.conv3")
        hook_mgr.register_forward_capture(model.encoder[8], "hjepa.flatten")
        hook_mgr.register_forward_capture(model.encoder[10], "hjepa.shared_latent_s")

        # Macro Predictor (Tier 1)
        hook_mgr.register_forward_capture(model.macro_predictor.fc1, "hjepa.macro_fc1")
        hook_mgr.register_forward_capture(model.macro_predictor.fc2, "hjepa.macro_fc2")
        hook_mgr.register_forward_capture(model.macro_predictor.delta_head, "hjepa.macro_delta_S")
        hook_mgr.register_forward_capture(model.macro_predictor.col_head, "hjepa.macro_col_head")

        # Micro Predictor (Tier 2)
        hook_mgr.register_forward_capture(model.micro_predictor.pol_fc1, "hjepa.micro_pol_fc1")
        hook_mgr.register_forward_capture(model.micro_predictor.action_head, "hjepa.micro_action_head")
        hook_mgr.register_forward_capture(model.micro_predictor.dyn_fc1, "hjepa.micro_dyn_fc1")
        hook_mgr.register_forward_capture(model.micro_predictor.delta_head, "hjepa.micro_delta_s")
        hook_mgr.register_forward_capture(model.micro_predictor.col_head, "hjepa.micro_col_head")
