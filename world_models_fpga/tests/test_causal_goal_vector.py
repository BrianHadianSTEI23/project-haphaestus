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


def test_hjepa_goal_memory_blending_invariants():
    """Verify that HJEPAAgent accurately tracks ego-motion and blends goal memory: 0.65*vis + 0.35*mem."""
    from research.benchmarks.model_wrappers import HJEPAAgent
    agent = HJEPAAgent(device="cpu", track="observation")
    agent.reset()

    obs_0 = np.zeros((64, 64, 3), dtype=np.uint8)
    task_spec = {"target_key": "red_box", "semantic_relation": "approach", "instruction": "approach red box"}

    # Step 0: Initial call, memory is initialized directly from visual goal
    act_0, info_0 = agent.act(obs_0, task_spec=task_spec)
    assert info_0["g_memory_prev"] is None
    np.testing.assert_allclose(info_0["g_visual_t"], info_0["g_blended_t"], atol=1e-5)
    np.testing.assert_allclose(info_0["g_memory_t"], info_0["g_blended_t"], atol=1e-5)

    # Step 1: Memory blending active: g_blended = 0.65 * g_vis + 0.35 * g_mem_updated
    obs_1 = np.ones((64, 64, 3), dtype=np.uint8) * 128
    act_1, info_1 = agent.act(obs_1, task_spec=task_spec)

    g_vis_1 = info_1["g_visual_t"]
    g_mem_upd_1 = info_1["g_memory_t"]
    g_blend_1 = info_1["g_blended_t"]

    expected_blend = 0.65 * g_vis_1 + 0.35 * g_mem_upd_1
    np.testing.assert_allclose(g_blend_1, expected_blend, atol=1e-5)


def test_hjepa_full_controller_pipeline_diagnostics():
    """Verify that HJEPAAgent returns all required multi-tier hierarchical diagnostics."""
    from research.benchmarks.model_wrappers import HJEPAAgent
    agent = HJEPAAgent(device="cpu", track="observation")
    agent.reset()

    obs = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    task_spec = {"target_key": "blue_cylinder", "semantic_relation": "approach", "instruction": "reach blue cylinder"}

    act, info = agent.act(obs, task_spec=task_spec)

    # Verify diagnostic keys
    expected_keys = [
        "g_visual_t", "g_memory_prev", "delta_ego_t", "g_memory_t", "g_blended_t",
        "a_react", "macro_cand_seqs", "best_macro_idx", "macro_subgoal",
        "a_proposal", "best_micro_idx", "micro_action", "pred_macro_disp", "pred_micro_disp"
    ]
    for k in expected_keys:
        assert k in info, f"Missing diagnostic key: {k}"

    # Verify action bounds
    assert act.shape == (3,)
    assert np.all(act >= -1.0) and np.all(act <= 1.0)
    assert info["macro_cand_seqs"].shape == (25, 15)
    assert info["macro_subgoal"].shape == (32,)

