import os
import numpy as np
import pytest
import torch

os.environ["MUJOCO_GL"] = "egl"

from research.benchmarks.model_wrappers import HJEPAAgent


def test_handwritten_memory_update_invariants():
    """Verify geometric update invariants of the handwritten goal memory in HJEPAAgent."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = HJEPAAgent(device=device, track="observation")
    spec = {"target_key": "purple_box", "semantic_relation": "approach", "instruction": "approach purple box"}
    obs_1 = np.ones((64, 64, 3), dtype=np.uint8) * 100
    obs_2 = np.ones((64, 64, 3), dtype=np.uint8) * 150

    agent.clear_branch_interceptors()
    agent.reset()

    # Step 1: memory initialized from visual goal
    _, diag_1 = agent.act(obs_1, task_spec=spec)
    assert diag_1["g_memory_prev"] is None
    np.testing.assert_allclose(diag_1["g_memory_t"], diag_1["g_visual_t"], atol=1e-5)
    np.testing.assert_allclose(diag_1["g_blended_t"], diag_1["g_visual_t"], atol=1e-5)

    # Step 2: memory updated with delta_ego and blended
    _, diag_2 = agent.act(obs_2, task_spec=spec)
    assert diag_2["g_memory_prev"] is not None
    # Verify fixed 0.65 / 0.35 blending equation
    expected_blended = 0.65 * diag_2["g_visual_t"] + 0.35 * diag_2["g_memory_t"]
    np.testing.assert_allclose(diag_2["g_blended_t"], expected_blended, atol=1e-5)


def test_visual_only_interceptor_bypasses_memory():
    """Verify that visual_only hook forces g_blended == g_visual regardless of memory state."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = HJEPAAgent(device=device, track="observation")
    spec = {"target_key": "purple_box", "semantic_relation": "approach", "instruction": "approach purple box"}
    obs = np.ones((64, 64, 3), dtype=np.uint8) * 120

    agent.clear_branch_interceptors()
    agent.reset()
    # Configure visual-only interceptor
    agent.set_branch_interceptor("blended_goal", lambda g, v, m: v)

    # Step 1
    _, diag_1 = agent.act(obs, task_spec=spec)
    np.testing.assert_allclose(diag_1["g_blended_t"], diag_1["g_visual_t"], atol=1e-5)

    # Step 2
    _, diag_2 = agent.act(obs, task_spec=spec)
    np.testing.assert_allclose(diag_2["g_blended_t"], diag_2["g_visual_t"], atol=1e-5)


def test_corrupted_memory_interceptor():
    """Verify that memory corruption hook perturbs g_memory without altering g_visual."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = HJEPAAgent(device=device, track="observation")
    spec = {"target_key": "purple_box", "semantic_relation": "approach", "instruction": "approach purple box"}
    obs = np.ones((64, 64, 3), dtype=np.uint8) * 80

    agent.clear_branch_interceptors()
    agent.reset()
    agent.act(obs, task_spec=spec)  # step 1

    # Step 2 clean
    agent_clean = HJEPAAgent(device=device, track="observation")
    agent_clean.reset()
    agent_clean.act(obs, task_spec=spec)
    _, diag_clean_2 = agent_clean.act(obs, task_spec=spec)

    # Step 2 corrupted
    agent.set_branch_interceptor("memory_updated", lambda mem: mem + 5.0)
    _, diag_corrupt_2 = agent.act(obs, task_spec=spec)

    # Visual goal unchanged
    np.testing.assert_allclose(diag_corrupt_2["g_visual_t"], diag_clean_2["g_visual_t"], atol=1e-5)
    # Memory updated is shifted by 5.0
    np.testing.assert_allclose(diag_corrupt_2["g_memory_t"], diag_clean_2["g_memory_t"] + 5.0, atol=1e-4)
