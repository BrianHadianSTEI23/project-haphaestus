import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from research.datasets.generation.capability_tasks import generate_task_specification
from research.datasets.generation.expert_navigator import ExpertNavigator


def test_geometric_expert_is_deterministic_and_avoids_blocked_straight_path():
    planner = ExpertNavigator()
    spec = {"obstacles": {"blocker": {"pos": np.array([0.6, 0.0, 0.35]), "shape": "cylinder", "radius": 0.25}}}
    pose, waypoint = np.array([0.0, 0.0, 0.0]), np.array([1.5, 0.0, 0.0])
    first = planner.compute_action(pose, waypoint, spec)
    second = planner.compute_action(pose, waypoint, spec)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, np.array([0.7, 0.0, 0.0], dtype=np.float32))


class TestExpertNavigatorAndCapabilities(unittest.TestCase):
    def test_geometric_expert_determinism_and_repulsion(self):
        test_geometric_expert_is_deterministic_and_avoids_blocked_straight_path()

    def test_capability_tasks_populate_real_obstacles_and_color_mutations(self):
        # Object grounding counterfactual must mutate target RGBA
        og_cf = generate_task_specification("object_grounding", "counterfactual", seed_idx=1002)
        target_key = og_cf["target_key"]
        self.assertEqual(og_cf["obstacles"][target_key]["color"], "mutated_blue")
        np.testing.assert_allclose(og_cf["obstacles"][target_key]["rgba"], [0.15, 0.35, 0.85, 1.0], atol=1e-5)

        # Obstacle avoidance tiers must insert real collidable geometries
        for diff in ("easy", "difficult", "counterfactual", "ood"):
            oa_spec = generate_task_specification("obstacle_avoidance", diff, seed_idx=1200)
            self.assertIn("inserted_obstacle", oa_spec["obstacles"])
            self.assertEqual(oa_spec["obstacles"]["inserted_obstacle"]["geom_name"], "obs_inserted")
            if diff == "ood":
                self.assertIn("slit_barrier", oa_spec["obstacles"])
                self.assertEqual(oa_spec["obstacles"]["slit_barrier"]["geom_name"], "obs_barrier")


if __name__ == "__main__":
    unittest.main()
