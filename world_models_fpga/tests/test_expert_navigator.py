import numpy as np

from research.datasets.generation.expert_navigator import ExpertNavigator


def test_geometric_expert_is_deterministic_and_avoids_blocked_straight_path():
    planner = ExpertNavigator()
    spec = {"obstacles": {"blocker": {"pos": np.array([0.6, 0.0, 0.35]), "shape": "cylinder", "radius": 0.25}}}
    pose, waypoint = np.array([0.0, 0.0, 0.0]), np.array([1.5, 0.0, 0.0])
    first = planner.compute_action(pose, waypoint, spec)
    second = planner.compute_action(pose, waypoint, spec)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, np.array([0.7, 0.0, 0.0], dtype=np.float32))
