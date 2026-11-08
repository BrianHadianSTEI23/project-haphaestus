import copy
import numpy as np
import pytest
import torch
from world_models_fpga.env.amr_env import JetAutoEnv
from world_models_fpga.research.circuits.run_ego_memory_mechanism_v5 import (
    transform_ego_goal_vector,
    tost_equivalence_test,
    GEOM_NAME_MAP,
)
from world_models_fpga.research.datasets.generation.capability_tasks import (
    CANONICAL_OBSTACLES,
    generate_task_specification,
)


def test_transform_ego_goal_vector_invariants():
    """Tests 2D rigid transform on egocentric goal vector."""
    # Robot at origin, goal at [2.0, 0.0]
    goal = torch.tensor([2.0, 0.0, 0.0, 1.0])

    # 1. Pure forward translation of 0.5m: remaining goal should be [1.5, 0.0]
    disp_fwd = torch.tensor([0.5, 0.0, 0.0])
    trans_fwd = transform_ego_goal_vector(goal, disp_fwd)
    assert np.isclose(trans_fwd[0].item(), 1.5, atol=1e-5)
    assert np.isclose(trans_fwd[1].item(), 0.0, atol=1e-5)

    # 2. Pure left rotation of +90 deg (+pi/2):
    # Goal originally directly ahead [2.0, 0.0] should now be at [0.0, -2.0] (to robot's right)
    disp_rot = torch.tensor([0.0, 0.0, np.pi / 2])
    trans_rot = transform_ego_goal_vector(goal, disp_rot)
    assert np.isclose(trans_rot[0].item(), 0.0, atol=1e-5)
    assert np.isclose(trans_rot[1].item(), -2.0, atol=1e-5)


def test_tost_equivalence_math():
    """Tests TOST equivalence procedure with margin delta = 0.05m."""
    rng = np.random.default_rng(42)
    # Tightly centered differences within [-0.01, 0.01] -> equivalent
    diffs_equiv = rng.normal(0.001, 0.005, size=50)
    p_tost, is_equiv = tost_equivalence_test(diffs_equiv, delta=0.05)
    assert is_equiv is True
    assert p_tost < 0.05

    # Large offset differences -> not equivalent
    diffs_nonequiv = rng.normal(0.10, 0.01, size=50)
    p_tost_ne, is_equiv_ne = tost_equivalence_test(diffs_nonequiv, delta=0.05)
    assert is_equiv_ne is False
    assert p_tost_ne > 0.05


def test_simulator_initialization_assertions():
    """Verifies that apply_task_spec and reset install exact physical pose and obstacles in MuJoCo."""
    env = JetAutoEnv()
    target_key = "yellow_cylinder"
    target_pos = np.array([0.85, -0.65, 0.35], dtype=np.float32)
    init_pose = [0.15, -0.25, 0.785]  # ~45 deg

    spec = {
        "target_key": target_key,
        "obstacles": copy.deepcopy(CANONICAL_OBSTACLES),
        "initial_pose": init_pose,
    }
    spec["obstacles"][target_key]["pos"] = target_pos

    # 1. Restore baseline
    env.restore_physical_baseline()
    # 2. Apply task spec
    env.apply_task_spec(spec)
    # 3. Reset with exact initial pos
    obs, info = env.reset(options={"initial_pos": tuple(init_pose)})

    # Verify actual robot pose in MuJoCo
    actual_pose = env.get_robot_pose()
    assert np.allclose(actual_pose, init_pose, atol=1e-3), f"Pose mismatch: {actual_pose} vs {init_pose}"

    # Verify actual obstacle geometry position in MuJoCo
    geom_name = GEOM_NAME_MAP[target_key]
    geom_id = env.geometry_id(geom_name)
    actual_geom_pos = env.model.geom_pos[geom_id]
    assert np.allclose(actual_geom_pos[:2], target_pos[:2], atol=1e-3), (
        f"Obstacle pos mismatch: {actual_geom_pos[:2]} vs {target_pos[:2]}"
    )
    env.close()
