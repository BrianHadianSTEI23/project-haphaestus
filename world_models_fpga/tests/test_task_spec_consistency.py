"""Unit test verifying task specification consistency, physical feasibility, and footprint clearance."""

from __future__ import annotations

import copy
import numpy as np
import pytest

from research.datasets.generation.capability_tasks import (
    CANONICAL_OBSTACLES,
    generate_task_specification,
    get_waypoint_for_relation,
)
from research.benchmarks.run_track_c_visual_grounding import (
    assert_task_spec_valid,
    compute_surface_distance,
)
from models.baseline_policy import encode_task_id, TARGET_KEYS


def test_nominal_task_specs_pass_validation():
    """All nominal object-grounding specifications must pass geometric validation."""
    for i in range(12):
        spec = generate_task_specification("object_grounding", "easy", seed_idx=6100 + i)
        # Should execute without raising AssertionError
        assert_task_spec_valid(spec)
        assert spec["target_key"] in spec["obstacles"]
        tid = encode_task_id(spec["target_key"], spec["semantic_relation"], spec["instruction"])
        assert 0 <= tid < 64


def test_surface_distance_accounts_for_geometry():
    """Verify surface distance handles both cylinders and boxes properly."""
    cyl_obs = {"pos": np.array([0.0, 0.0, 0.35]), "radius": 0.25}
    box_obs = {"pos": np.array([0.0, 0.0, 0.35]), "size": np.array([0.25, 0.25, 0.35])}

    robot_radius = 0.18
    # Test point along x-axis at distance 0.60m from center
    pt = np.array([0.60, 0.0])

    # For cylinder: dist = 0.60 - (0.25 + 0.18) = 0.17m
    cyl_surf = compute_surface_distance(pt, cyl_obs, robot_radius=robot_radius)
    assert abs(cyl_surf - 0.17) < 1e-4

    # For box: dx = max(0, 0.60 - 0.25) = 0.35m. Surface dist = 0.35 - 0.18 = 0.17m
    box_surf = compute_surface_distance(pt, box_obs, robot_radius=robot_radius)
    assert abs(box_surf - 0.17) < 1e-4


def test_assert_task_spec_catches_missing_target():
    spec = generate_task_specification("object_grounding", "easy", seed_idx=6100)
    del spec["target_key"]
    with pytest.raises(AssertionError):
        assert_task_spec_valid(spec)


def test_assert_task_spec_catches_out_of_bounds_initial_pose():
    spec = generate_task_specification("object_grounding", "easy", seed_idx=6100)
    spec["initial_pose"] = np.array([2.50, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(AssertionError):
        assert_task_spec_valid(spec)


def test_assert_task_spec_catches_obstacle_collision():
    spec = generate_task_specification("object_grounding", "easy", seed_idx=6100)
    # Move initial pose right on top of target obstacle center
    t_pos = spec["obstacles"][spec["target_key"]]["pos"]
    spec["initial_pose"] = np.array([t_pos[0], t_pos[1], 0.0], dtype=np.float32)
    with pytest.raises(AssertionError):
        assert_task_spec_valid(spec)


def test_assert_task_spec_catches_inconsistent_instruction():
    spec = generate_task_specification("object_grounding", "easy", seed_idx=6100)
    # Target is purple_box, set instruction to go to cyan cylinder
    spec["instruction"] = "navigate to the cyan cylinder"
    with pytest.raises(AssertionError):
        assert_task_spec_valid(spec)


def test_waypoint_analytic_calculation_and_clearance():
    """Verify analytic waypoint derivation, orientation toward target, and physical clearance."""
    init_pose = np.array([0.0, -1.0, np.pi / 2], dtype=np.float32)
    target_pos = np.array([0.0, 0.5, 0.35], dtype=np.float32)
    clearance = 0.60

    wp = get_waypoint_for_relation(target_pos, "approach", init_pose, clearance=clearance)

    # 1. Distance from waypoint to target center must match clearance
    dist_to_center = float(np.linalg.norm(wp[:2] - target_pos[:2]))
    assert abs(dist_to_center - clearance) < 1e-4

    # 2. Waypoint yaw must face directly toward the target
    target_bearing = float(np.arctan2(target_pos[1] - wp[1], target_pos[0] - wp[0]))
    ang_diff = abs(target_bearing - wp[2])
    if ang_diff > np.pi:
        ang_diff = 2 * np.pi - ang_diff
    assert ang_diff < 1e-4, f"Robot yaw {wp[2]} does not face target bearing {target_bearing}"


def test_true_pairwise_color_swap_invariants():
    """Verify that bidirectional color mutations actually mutate geom RGBA in MuJoCo simulation."""
    from env.amr_env import JetAutoEnv
    env = JetAutoEnv()
    spec = generate_task_specification("object_grounding", "easy", seed_idx=6100)
    t_key = spec["target_key"]
    obs = spec["obstacles"]
    partner_key = [k for k in obs if k != t_key][0]

    t_rgba = list(obs[t_key]["rgba"])
    p_rgba = list(obs[partner_key]["rgba"])
    t_geom = obs[t_key]["geom_name"]
    p_geom = obs[partner_key]["geom_name"]

    spec["color_mutations"] = {
        t_geom: p_rgba,
        p_geom: t_rgba,
    }

    env.apply_task_spec(spec)

    t_gid = env.geometry_id(t_geom)
    p_gid = env.geometry_id(p_geom)

    np.testing.assert_allclose(env.model.geom_rgba[t_gid], p_rgba, atol=1e-4)
    np.testing.assert_allclose(env.model.geom_rgba[p_gid], t_rgba, atol=1e-4)
    env.close()

