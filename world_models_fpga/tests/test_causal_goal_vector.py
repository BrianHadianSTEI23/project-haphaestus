"""Unit test verifying causal visual goal-vector grounding head computation."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from models.h_jepa import HJEPAModel
from models.baseline_policy import encode_task_id


def test_infer_goal_vector_shape_and_bounds():
    """Verify that infer_goal_vec produces bounded [dx, dy, sin_dth, cos_dth]."""
    model = HJEPAModel(latent_dim=32, stride_k=5, hidden_dim=64)
    z = torch.randn(8, 32)
    task_ids = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1], dtype=torch.long)

    g_vec = model.infer_goal_vec(z, task_ids)
    assert g_vec.shape == (8, 4)

    # dx, dy bounded in [-3.5, 3.5]
    dx = g_vec[:, 0]
    dy = g_vec[:, 1]
    assert torch.all(dx >= -3.5) and torch.all(dx <= 3.5)
    assert torch.all(dy >= -3.5) and torch.all(dy <= 3.5)

    # sin^2 + cos^2 ≈ 1
    sin_th = g_vec[:, 2]
    cos_th = g_vec[:, 3]
    norm_trig = sin_th ** 2 + cos_th ** 2
    torch.testing.assert_close(norm_trig, torch.ones_like(norm_trig), atol=1e-4, rtol=1e-4)


def test_reactive_action_bounded():
    """Verify that predict_reactive_action produces actions in [-1, 1]^3."""
    model = HJEPAModel(latent_dim=32, stride_k=5, hidden_dim=64)
    z = torch.randn(8, 32)
    g_vec = torch.randn(8, 4)

    action = model.predict_reactive_action(z, g_vec)
    assert action.shape == (8, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
