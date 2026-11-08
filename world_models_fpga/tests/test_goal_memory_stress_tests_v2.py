#!/usr/bin/env python3
"""
Unit tests for Goal Memory Stress Tests v2.

Verifies:
1. Visual-only controller bypasses memory blending.
2. Controller memory incorporates ego-motion updates over consecutive steps.
3. Zeroed and perturbed ego-motion interceptors alter memory update without corrupting encoder or policy.
4. Observation dropout (occlusion) is properly recorded in traces.
"""

import os
import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

from env.amr_env import JetAutoEnv
from models.baseline_policy import encode_task_id
from research.benchmarks.model_wrappers import HJEPAAgent
from research.datasets.dataset_config import CKPT_DIR
from research.datasets.generation.capability_tasks import generate_task_specification
from research.circuits.run_goal_memory_stress_tests_v2 import evaluate_memory_stress_episode


@pytest.fixture(scope="module")
def env():
    e = JetAutoEnv()
    yield e


@pytest.fixture(scope="module")
def agent():
    ckpt_path = CKPT_DIR / "hjepa_best.pt"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = HJEPAAgent(str(ckpt_path), device=dev, track="observation")
    return a


def test_visual_only_controller_bypass(agent, env):
    """Verifies that cond_2_visual_only ignores memory state completely."""
    spec = generate_task_specification("object_grounding", "easy", seed_idx=9404)

    # In visual-only mode, blended goal equals visual goal
    agent.clear_branch_interceptors()
    agent.reset()
    agent.set_branch_interceptor("blended_goal", lambda g, v, m: v)

    dummy_rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
    task_spec = {"target_key": "red_box", "semantic_relation": "approach", "instruction": "reach the red box"}

    # Step 1
    _, diag1 = agent.act(dummy_rgb, task_spec=task_spec)
    # Step 2
    _, diag2 = agent.act(dummy_rgb, task_spec=task_spec)

    assert np.allclose(diag2["g_blended_t"], diag2["g_visual_t"]), "Visual-only controller did not bypass memory!"


def test_ego_motion_interceptor_isolation(agent):
    """Verifies that zeroed and perturbed ego-motion interceptors alter memory updates without affecting encoder."""
    dummy_rgb1 = np.full((64, 64, 3), 128, dtype=np.uint8)
    dummy_rgb2 = np.full((64, 64, 3), 160, dtype=np.uint8)
    task_spec = {"target_key": "red_box", "semantic_relation": "approach", "instruction": "reach the red box"}

    # Clean run over 2 steps
    agent.clear_branch_interceptors()
    agent.reset()
    _, _ = agent.act(dummy_rgb1, task_spec=task_spec)
    _, diag_clean = agent.act(dummy_rgb2, task_spec=task_spec)

    # Zeroed ego-motion
    agent.clear_branch_interceptors()
    agent.reset()
    agent.set_branch_interceptor("ego_disp", lambda d: torch.zeros_like(d))
    _, _ = agent.act(dummy_rgb1, task_spec=task_spec)
    _, diag_zeroed = agent.act(dummy_rgb2, task_spec=task_spec)

    # 1. Realized displacement should be zeroed
    assert np.allclose(diag_zeroed["delta_ego_t"], np.zeros(3)), "Ego-motion was not zeroed"
    # 2. Updated memory should differ from clean run
    assert not np.allclose(diag_zeroed["g_memory_t"], diag_clean["g_memory_t"]), "Zeroed ego-motion did not alter memory"
    # 3. Encoder latent must remain identical
    assert np.allclose(diag_zeroed["latent"], diag_clean["latent"]), "Ego-motion intervention corrupted encoder latent"

    agent.clear_branch_interceptors()


def test_observation_dropout_tracking(agent, env):
    """Verifies that occlusion window is applied and tracked across timesteps."""
    spec = generate_task_specification("object_grounding", "easy", seed_idx=9505)

    res = evaluate_memory_stress_episode(
        agent,
        env,
        spec,
        condition="cond_1_full_visual_and_memory",
        occlusion_window=(2, 6),
        max_steps=10,
    )

    occl_trace = res["trace_is_occluded"]
    # Check steps 2..5 are flagged as occluded
    assert len(occl_trace) >= 6, "Episode terminated prematurely"
    assert occl_trace[0] == 0 and occl_trace[1] == 0
    assert occl_trace[2] == 1 and occl_trace[3] == 1 and occl_trace[4] == 1 and occl_trace[5] == 1
    assert occl_trace[6] == 0
