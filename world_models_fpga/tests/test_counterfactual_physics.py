import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from env.amr_env import JetAutoEnv
from research.datasets.counterfactual.counterfactual_generator import generate_counterfactual_pair
from research.datasets.counterfactual.disambiguation_tasks import generate_disambiguation_tasks
from research.datasets.counterfactual.validate_counterfactuals import EXPERIMENTS, validate_all


def test_all_counterfactuals_change_real_simulator_state():
    assert all(item["status"] == "VALID" for item in validate_all(pair_id=0))
    assert all(item["status"] == "VALID" for item in validate_all(pair_id=3))


class TestCounterfactualPhysics(unittest.TestCase):
    def test_all_counterfactuals_change_real_simulator_state_across_pairs(self):
        for pair_id in (0, 3, 12, 24):
            results = validate_all(pair_id=pair_id)
            for item in results:
                self.assertEqual(item["status"], "VALID", f"Failed on {item['experiment']} pair {pair_id}")

    def test_counterfactual_pairs_are_distinct_across_pair_ids(self):
        for exp in EXPERIMENTS:
            poses = []
            for p in range(25):
                e0, ei = generate_counterfactual_pair(exp, p)
                poses.append(tuple(np.round(e0["initial_pose"], 5).tolist()))
                # Within each pair, E0 and Ei share pair_id and generation_seed
                self.assertEqual(e0["pair_id"], p)
                self.assertEqual(ei["pair_id"], p)
            self.assertEqual(len(set(poses)), 25, f"Duplicate pair initial poses in {exp}")

    def test_disambiguation_slit_barrier_exists_and_collides(self):
        tasks = generate_disambiguation_tasks()
        self.assertEqual(len(tasks), 4)
        slit_task = [t for t in tasks if "visibility_vs_reachability" in t["probe_id"]][0]
        env = JetAutoEnv()
        try:
            env.apply_task_spec(slit_task)
            barrier_id = env.geometry_id("obs_barrier")
            self.assertAlmostEqual(float(env.model.geom_pos[barrier_id][2]), 0.35, places=3)
            # Placing robot directly inside obs_barrier at (-0.5, 0.6) must trigger physical collision
            env.reset(options={"initial_pos": [-0.5, 0.6, 0.0]})
            self.assertTrue(env._check_collision(), "Robot inside obs_barrier must register collision")
            env.restore_physical_baseline()
            self.assertLess(float(env.model.geom_pos[barrier_id][2]), -5.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
