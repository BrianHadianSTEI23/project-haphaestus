import os
import numpy as np
import pytest
import torch

os.environ["MUJOCO_GL"] = "egl"

from env.amr_env import JetAutoEnv
from models.baseline_policy import encode_task_id
from research.benchmarks.model_wrappers import HJEPAAgent
from research.benchmarks.run_task_identity_visual_grounding import (
    check_target_visibility,
    build_experimental_suites,
    assert_task_spec_valid,
)
from research.datasets.generation.capability_tasks import generate_task_specification


def test_target_visibility_checker():
    """Verify that check_target_visibility accurately detects target presence via segmentation."""
    env = JetAutoEnv()
    spec = generate_task_specification("object_grounding", "easy", seed_idx=6100)
    env.apply_task_spec(spec)
    init_pos = spec["initial_pose"]
    env.reset(options={"initial_pos": (float(init_pos[0]), float(init_pos[1]), float(init_pos[2]))})

    t_key = spec["target_key"]
    vis_info = check_target_visibility(env, t_key, min_visible_pixels=20)

    assert "is_visible" in vis_info
    assert "visible_pixels" in vis_info
    assert "mask_area" in vis_info
    assert "centroid_uv" in vis_info
    assert vis_info["visible_pixels"] > 0
    assert vis_info["is_visible"] is True
    assert vis_info["centroid_uv"] is not None
    assert 0.0 <= vis_info["centroid_uv"][0] <= 64.0
    assert 0.0 <= vis_info["centroid_uv"][1] <= 64.0


def test_task_token_ground_truth_separation():
    """Verify that in the visual/token conflict condition, ground truth and token remain distinct."""
    suites = build_experimental_suites(n_episodes=5, base_seed=9500)
    conflict_specs = suites["cond_6_visual_token_conflict"]

    for spec in conflict_specs:
        assert "env_ground_truth_target" in spec
        assert "agent_task_token_target" in spec
        assert "agent_task_token" in spec
        # Ground truth object and agent token target must be distinct objects
        assert spec["env_ground_truth_target"] != spec["agent_task_token_target"]
        assert spec["object_a_waypoint"] != spec["object_b_waypoint"]


def test_branch_specific_intervention_independence():
    """Verify that intervening on goal branches leaves model.encode() and reactive action proposal unchanged."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = HJEPAAgent(device=device, track="observation")
    obs_rgb = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    spec = {"target_key": "purple_box", "semantic_relation": "approach", "instruction": "approach purple box"}

    # Run clean (2 steps to populate memory)
    agent.clear_branch_interceptors()
    agent.reset()
    agent.act(obs_rgb, task_spec=spec)
    _, diag_clean = agent.act(obs_rgb, task_spec=spec)
    latent_clean = diag_clean["latent"].copy()
    g_vis_clean = diag_clean["g_visual_t"].copy()

    # Intervene on memory_updated only
    agent.clear_branch_interceptors()
    agent.reset()
    agent.act(obs_rgb, task_spec=spec)
    agent.set_branch_interceptor("memory_updated", lambda mem: torch.ones_like(mem) * 99.0)
    _, diag_mem_mod = agent.act(obs_rgb, task_spec=spec)

    # model.encode() and raw visual goal must be completely identical
    np.testing.assert_allclose(diag_mem_mod["latent"], latent_clean, atol=1e-6)
    np.testing.assert_allclose(diag_mem_mod["g_visual_t"], g_vis_clean, atol=1e-6)
    # But blended goal must differ
    assert not np.allclose(diag_mem_mod["g_blended_t"], diag_clean["g_blended_t"])
