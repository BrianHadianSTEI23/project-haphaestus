import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from research.benchmarks.evaluator import BenchmarkEvaluator
from research.benchmarks.model_wrappers import BaselineAgent, HJEPAAgent, LeWMAgent, OracleAgent
from research.datasets.generation.capability_tasks import generate_task_specification


def test_information_regime_enforcement():
    # OracleAgent must refuse Track B / Track C calls without privileged pose
    oracle = OracleAgent()
    obs = np.zeros((64, 64, 3), dtype=np.uint8)
    try:
        oracle.act(obs, robot_pose=None, target_pose=None)
        assert False, "OracleAgent must raise ValueError when privileged poses are withheld"
    except ValueError:
        pass

    # Track C agents must operate with robot_pose=None, target_pose=None, goal_vec_body=None
    public_spec = {"target_key": "yellow_cylinder", "semantic_relation": "behind", "instruction": "go behind the yellow cylinder"}
    for agent in (
        BaselineAgent(track="observation"),
        LeWMAgent(track="observation"),
        HJEPAAgent(track="observation"),
    ):
        act, info = agent.act(obs, robot_pose=None, target_pose=None, task_spec=public_spec, goal_vec_body=None)
        assert act.shape == (3,)
        assert np.all(np.abs(act) <= 1.0001)
        assert info["track"] == "observation"


class TestBehavioralValidity(unittest.TestCase):
    def test_track_c_withholds_privileged_state(self):
        test_information_regime_enforcement()

    def test_evaluator_runs_tracks_b_and_c_and_ablations(self):
        evaluator = BenchmarkEvaluator()
        try:
            spec = generate_task_specification("obstacle_avoidance", "difficult", seed_idx=1205)
            for track in ("structured", "observation"):
                for mode in ("prediction_used", "reactive", "prediction_ignored", "no_action_conditioning", "no_collision_prediction"):
                    agent = LeWMAgent(track=track, ablation_mode=mode)
                    res = evaluator.evaluate_episode(agent, spec, max_steps=6)
                    self.assertEqual(res["track"], track)
                    self.assertEqual(res["ablation_mode"], mode)
                    self.assertEqual(res["steps_taken"], 6)
        finally:
            evaluator.close()


if __name__ == "__main__":
    unittest.main()
