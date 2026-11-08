"""
Unit tests for Phase 4 Experiment C: Ego-Motion Memory Mechanism Benchmark (v3).
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
from research.circuits.run_ego_memory_mechanism_v3 import (
    compute_physical_body_displacement,
    transform_ego_goal_vector,
    record_clean_trajectory,
    replay_trajectory_under_interventions,
    calibrate_micro_ego_head,
    tost_equivalence_test,
    wrap_to_pi,
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


def test_physical_displacement_math():
    """Tests body-frame physical displacement coordinate math."""
    p0 = np.array([1.0, 2.0, 0.0], dtype=np.float32)
    p1 = np.array([1.5, 2.0, 0.0], dtype=np.float32)  # +0.5m forward along x
    d_trans = compute_physical_body_displacement(p0, p1)
    np.testing.assert_allclose(d_trans, [0.5, 0.0, 0.0], atol=1e-5)

    # When robot is turned 90 deg (pi/2)
    p2 = np.array([1.0, 2.0, np.pi / 2.0], dtype=np.float32)
    p3 = np.array([1.0, 2.5, np.pi / 2.0], dtype=np.float32)  # +0.5m in world y, which is body x
    d_turned = compute_physical_body_displacement(p2, p3)
    np.testing.assert_allclose(d_turned, [0.5, 0.0, 0.0], atol=1e-5)

    # Pure rotation
    p4 = np.array([1.0, 2.0, 0.0], dtype=np.float32)
    p5 = np.array([1.0, 2.0, 0.75], dtype=np.float32)
    d_rot = compute_physical_body_displacement(p4, p5)
    np.testing.assert_allclose(d_rot, [0.0, 0.0, 0.75], atol=1e-5)


def test_transform_ego_goal_vector():
    """Tests handwritten coordinate transformation of body-frame goal vector."""
    goal_0 = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float32)  # 1m directly ahead
    disp_fwd = torch.tensor([0.2, 0.0, 0.0], dtype=torch.float32)      # Robot moved 0.2m forward
    goal_up = transform_ego_goal_vector(goal_0, disp_fwd)
    # Remaining distance should be 0.8m ahead
    np.testing.assert_allclose(goal_up[:2].numpy(), [0.8, 0.0], atol=1e-5)

    # Robot turned 90 deg CCW
    disp_turn = torch.tensor([0.0, 0.0, np.pi / 2.0], dtype=torch.float32)
    goal_turned = transform_ego_goal_vector(goal_0, disp_turn)
    # Target that was ahead should now be to the robot's right (body y = -1.0)
    np.testing.assert_allclose(goal_turned[:2].numpy(), [0.0, -1.0], atol=1e-5)


def test_same_trajectory_replay_invariants(agent, env):
    """Verifies same-trajectory recording, calibration, and replay execution across conditions."""
    trajs = []
    for i in range(2):
        t_data = record_clean_trajectory(agent, env, traj_idx=i, seed=42, n_steps=12)
        trajs.append(t_data)

    assert len(trajs) == 2
    assert len(trajs[0]["steps"]) == 12
    assert trajs[0]["initial_distance"] >= 0.60

    # Calibration check
    calib = calibrate_micro_ego_head(agent, trajs)
    assert "translation_mae_m" in calib
    assert "zero_delta_static_bias" in calib
    assert len(calib["zero_delta_static_bias"]) == 3

    # Replay check
    res = replay_trajectory_under_interventions(
        agent=agent,
        traj=trajs[0],
        donor_trajs=trajs,
        blackout_window=(4, 8),
    )

    # Verify that all 8 conditions are populated
    conds = [
        "cond_A_visual_only",
        "cond_B_full_memory",
        "cond_C_frozen_memory",
        "cond_D_zeroed_ego",
        "cond_E_shuffled_ego",
        "cond_F_matched_donor",
        "cond_G_mismatched_donor",
        "cond_H_oracle_odometry",
    ]
    for c in conds:
        assert c in res["cond_blackout_errors"]
        assert res["cond_blackout_errors"][c] >= 0.0
        assert res["cond_goals"][c].shape == (12, 4)


def test_tost_equivalence_test():
    """Verifies TOST equivalence calculation."""
    # Near identical differences
    diffs_close = np.array([0.001, -0.002, 0.001, 0.0, -0.001] * 10)
    p_tost, _, is_equiv = tost_equivalence_test(diffs_close, delta=0.05)
    assert is_equiv is True
    assert p_tost < 0.05

    # Large differences outside margin
    diffs_large = np.array([0.15, 0.12, 0.18, 0.14, 0.16] * 10)
    p_tost_lg, _, is_equiv_lg = tost_equivalence_test(diffs_large, delta=0.05)
    assert is_equiv_lg is False
