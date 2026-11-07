"""Regression test verifying multi-waypoint active subgoal vs final goal alignment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from env.amr_env import JetAutoEnv
from research.datasets.generation.capability_tasks import generate_task_specification
from research.datasets.generation.generate_benchmark_dataset import BenchmarkDatasetGenerator


def test_multi_waypoint_three_waypoint_action_label_alignment():
    """Verify that demonstrator actions align with the active waypoint, not the final goal."""
    generator = BenchmarkDatasetGenerator(seed=42)

    # Generate a temporal_planning task with 3 waypoints: [wp1, wp2, wp3]
    task_spec = generate_task_specification("temporal_planning", "easy", 100)
    waypoints = task_spec["waypoints"]
    assert len(waypoints) >= 3, f"Expected >= 3 waypoints for temporal_planning, got {len(waypoints)}"

    wp0 = np.asarray(waypoints[0], dtype=np.float32)
    wp1 = np.asarray(waypoints[1], dtype=np.float32)
    wp2 = np.asarray(waypoints[2], dtype=np.float32)

    transitions = generator.run_episode(task_spec, max_steps=40)
    generator.env.close()

    assert len(transitions) > 5, "Episode terminated too early"

    # Step 0 transition: robot at init_pose, moving toward wp0
    tr0 = transitions[0]
    np.testing.assert_allclose(tr0["grounded_goal"], wp0, atol=1e-4, err_msg="Step 0 grounded_goal must be wp0")
    np.testing.assert_allclose(tr0["active_subgoal"], wp0, atol=1e-4, err_msg="Step 0 active_subgoal must be wp0")
    np.testing.assert_allclose(tr0["final_goal"], wp2, atol=1e-4, err_msg="Step 0 final_goal must be wp2")
    assert tr0["active_waypoint_index"] == 0
    assert tr0["remaining_waypoint_count"] == len(waypoints)

    # Action direction check: action should point towards wp0, not wp2
    r_pos0 = tr0["robot_pose"]
    dx_to_wp0 = wp0[0] - r_pos0[0]
    dy_to_wp0 = wp0[1] - r_pos0[1]
    th0 = r_pos0[2]
    # Body-frame heading to wp0
    body_dx0 = np.cos(th0) * dx_to_wp0 + np.sin(th0) * dy_to_wp0
    body_dy0 = -np.sin(th0) * dx_to_wp0 + np.cos(th0) * dy_to_wp0
    dir_to_wp0 = np.array([body_dx0, body_dy0])
    dir_to_wp0 /= (np.linalg.norm(dir_to_wp0) + 1e-8)

    action_xy = tr0["action"][:2]
    action_dir = action_xy / (np.linalg.norm(action_xy) + 1e-8)
    cos_sim0 = np.dot(dir_to_wp0, action_dir)
    assert cos_sim0 > 0.30, f"Demonstrator action at step 0 is not pointing toward wp0! cos_sim = {cos_sim0:.4f}"

    # Verify that grounded_goal is NOT wp2 at step 0
    assert np.linalg.norm(tr0["grounded_goal"][:2] - wp2[:2]) > 0.5, "grounded_goal incorrectly matches final goal at step 0"

    # Check that throughout the episode, final_goal stays wp2
    for step_i, tr in enumerate(transitions):
        np.testing.assert_allclose(tr["final_goal"], wp2, atol=1e-4, err_msg=f"Step {step_i} final_goal mutated")
