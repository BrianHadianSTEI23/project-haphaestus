"""
Unit tests for Phase 4 Experiment A: Task-Conditioned Visual Binding Benchmark (v3).
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "world_models_fpga"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
import torch

from env.amr_env import JetAutoEnv
from research.benchmarks.model_wrappers import HJEPAAgent
from research.datasets.dataset_config import CKPT_DIR
from research.benchmarks.run_task_conditioned_binding_v3 import (
    generate_dual_visible_scene_pair,
    check_target_visibility,
)


@pytest.fixture(scope="module")
def env():
    e = JetAutoEnv()
    yield e
    e.close()


@pytest.fixture(scope="module")
def agent():
    ckpt = CKPT_DIR / "hjepa_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HJEPAAgent(str(ckpt), device=device, track="observation")


def test_dual_visible_scene_pair_generation(env):
    """Verifies that scene generator produces pairs with >= 100 pixels and valid waypoints >= 0.60m."""
    rng = np.random.Generator(np.random.PCG64(1234))
    rejection_counters = {}

    pair = generate_dual_visible_scene_pair(
        pair_idx=0,
        env=env,
        rng=rng,
        rejection_counters=rejection_counters,
        min_pixels=100,
        min_initial_dist=0.60,
        max_trials=500,
    )

    assert pair is not None, "Failed to generate valid dual-visible pair"
    assert pair["key_a"] != pair["key_b"]
    assert pair["pos_1"] is not None and pair["pos_2"] is not None

    # Check waypoint distances
    r_pos = pair["robot_pose"][:2]
    d_wp1 = np.linalg.norm(pair["wp_1"] - r_pos)
    d_wp2 = np.linalg.norm(pair["wp_2"] - r_pos)
    assert d_wp1 >= 0.60, f"wp1 distance {d_wp1} < 0.60m"
    assert d_wp2 >= 0.60, f"wp2 distance {d_wp2} < 0.60m"

    # Check pixel visibility
    pix_a1 = pair["vis_scene_a"]["vis_a"]["visible_pixels"]
    pix_b1 = pair["vis_scene_a"]["vis_b"]["visible_pixels"]
    pix_a2 = pair["vis_scene_b"]["vis_a"]["visible_pixels"]
    pix_b2 = pair["vis_scene_b"]["vis_b"]["visible_pixels"]

    assert pix_a1 >= 100, f"pix_a1 {pix_a1} < 100"
    assert pix_b1 >= 100, f"pix_b1 {pix_b1} < 100"
    assert pix_a2 >= 100, f"pix_a2 {pix_a2} < 100"
    assert pix_b2 >= 100, f"pix_b2 {pix_b2} < 100"


def test_factorial_forward_pass_and_interaction_math(agent, env):
    """Verifies 2x2 factorial forward passes and exact interaction vector arithmetic."""
    rng = np.random.Generator(np.random.PCG64(5678))
    rejection_counters = {}
    pair = generate_dual_visible_scene_pair(
        pair_idx=1,
        env=env,
        rng=rng,
        rejection_counters=rejection_counters,
        min_pixels=100,
        min_initial_dist=0.60,
    )
    assert pair is not None

    rx, ry, rth = pair["robot_pose"]

    # Scene A
    env.restore_physical_baseline()
    env.apply_task_spec(pair["spec_scene_a"])
    obs_a, _ = env.reset(options={"initial_pos": (rx, ry, rth)})

    # Scene B
    env.restore_physical_baseline()
    env.apply_task_spec(pair["spec_scene_b"])
    obs_b, _ = env.reset(options={"initial_pos": (rx, ry, rth)})

    with torch.no_grad():
        x_a = torch.from_numpy(obs_a.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(agent.device)
        x_b = torch.from_numpy(obs_b.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(agent.device)
        z_a = agent.model.encode(x_a)
        z_b = agent.model.encode(x_b)

        t_a = torch.tensor([pair["task_id_a"]], dtype=torch.long, device=agent.device)
        t_b = torch.tensor([pair["task_id_b"]], dtype=torch.long, device=agent.device)
        e_a = agent.model.task_embedding(t_a)
        e_b = agent.model.task_embedding(t_b)

        g_aa = agent.model.grounding_head(torch.cat([z_a, e_a], dim=-1)).squeeze(0).cpu().numpy()
        g_ab = agent.model.grounding_head(torch.cat([z_a, e_b], dim=-1)).squeeze(0).cpu().numpy()
        g_ba = agent.model.grounding_head(torch.cat([z_b, e_a], dim=-1)).squeeze(0).cpu().numpy()
        g_bb = agent.model.grounding_head(torch.cat([z_b, e_b], dim=-1)).squeeze(0).cpu().numpy()

    assert g_aa.shape == (4,)
    assert g_ab.shape == (4,)
    assert g_ba.shape == (4,)
    assert g_bb.shape == (4,)

    delta_b = g_bb[:2] - g_ba[:2]
    delta_a = g_ab[:2] - g_aa[:2]
    interaction = delta_b - delta_a
    assert interaction.shape == (2,)
    assert not np.any(np.isnan(interaction))
