"""
Phase 2: Unified Benchmark Evaluator Engine
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Evaluates models under identical initial conditions, instructions, and random seeds:
  - Task Success Rate (%)
  - Collision Rate (%)
  - Path Efficiency (eta = d_geodesic / d_actual)
  - Goal Localization Error (m)
  - Recovery from unexpected disturbances
  - Inference Latency (mean, median, p99 in microseconds)
"""

import os
import sys
import time
import numpy as np
import torch
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from env.amr_env import JetAutoEnv
from research.datasets.generation.capability_tasks import CANONICAL_OBSTACLES

class BenchmarkEvaluator:
    def __init__(self, xml_path=None):
        self.env = JetAutoEnv(xml_path=xml_path)
        self.geom_ids = {}
        for k, v in CANONICAL_OBSTACLES.items():
            gid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, v["geom_name"])
            self.geom_ids[k] = gid

    def _apply_environment(self, obstacles_dict):
        for k, info in obstacles_dict.items():
            if k in self.geom_ids:
                gid = self.geom_ids[k]
                self.env.model.geom_pos[gid] = info["pos"]
                if "rgba" in info:
                    self.env.model.geom_rgba[gid] = info["rgba"]

    def _reset_canonical(self):
        self._apply_environment(CANONICAL_OBSTACLES)

    def evaluate_episode(self, agent, task_spec, max_steps=40, perturbation=None):
        """
        Runs one episode for the agent under task_spec.
        Optional perturbation: dict(step=10, impulse=[0, 0.8, 0])
        """
        self._apply_environment(task_spec["obstacles"])
        init_pos = task_spec["initial_pose"]
        obs, info = self.env.reset(options={"initial_pos": (init_pos[0], init_pos[1], init_pos[2])})

        waypoints = task_spec["waypoints"]
        wp_idx = 0
        target_goal = waypoints[-1] if len(waypoints) > 0 else init_pos

        trajectory = [init_pos[:2].copy()]
        collision_count = 0
        latencies = []
        recovered = False

        for t in range(max_steps):
            curr_pose = self.env.get_robot_pose()
            curr_target = waypoints[wp_idx] if wp_idx < len(waypoints) else target_goal

            # Distance to current waypoint
            dist_curr = np.linalg.norm(curr_target[:2] - curr_pose[:2])
            if dist_curr < 0.22:
                if wp_idx < len(waypoints) - 1:
                    wp_idx += 1
                    curr_target = waypoints[wp_idx]

            # Measure inference latency with high-precision timer
            t_start = time.perf_counter()
            action = agent.select_action(obs, curr_pose, curr_target, goal_dict=task_spec)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1e6)

            # Apply mid-trajectory unexpected perturbation if configured
            if perturbation and t == perturbation.get("step", 10):
                impulse = perturbation.get("impulse", [0.0, 0.8, 0.0])
                # Direct kinetic kick to robot slide/hinge velocities in MuJoCo
                self.env.data.qvel[self.env.jx_id] += impulse[0]
                self.env.data.qvel[self.env.jy_id] += impulse[1]
                self.env.data.qvel[self.env.jz_id] += impulse[2]

            obs, reward, term, trunc, step_info = self.env.step(action)
            if step_info["collision"]:
                collision_count += 1

            new_pose = self.env.get_robot_pose()
            trajectory.append(new_pose[:2].copy())

            # Check if final target reached
            final_dist = np.linalg.norm(target_goal[:2] - new_pose[:2])
            if final_dist < 0.22 and wp_idx >= len(waypoints) - 1:
                if perturbation:
                    recovered = True
                break

        final_pose = self.env.get_robot_pose()
        final_dist = float(np.linalg.norm(target_goal[:2] - final_pose[:2]))
        goal_reached = (final_dist < 0.25)
        task_success = (goal_reached and collision_count == 0)

        # Path efficiency: geodesic distance / actual integrated path length
        geodesic_dist = float(np.linalg.norm(target_goal[:2] - init_pos[:2]))
        actual_path_len = float(sum(np.linalg.norm(trajectory[i+1] - trajectory[i]) for i in range(len(trajectory)-1)))
        path_efficiency = float(geodesic_dist / (actual_path_len + 1e-6)) if actual_path_len > 0.1 else 1.0
        path_efficiency = min(1.0, max(0.0, path_efficiency))

        self._reset_canonical()

        return {
            "episode_id": task_spec.get("episode_id", "ep"),
            "task_success": task_success,
            "goal_reached": goal_reached,
            "collision_count": collision_count,
            "had_collision": (collision_count > 0),
            "final_distance": final_dist,
            "path_efficiency": path_efficiency,
            "trajectory_length": actual_path_len,
            "steps": len(trajectory) - 1,
            "mean_latency_us": float(np.mean(latencies)),
            "p99_latency_us": float(np.percentile(latencies, 99)),
            "recovered": recovered
        }

    def evaluate_suite(self, agent, task_specs, max_steps=40, perturbation=None):
        results = []
        for spec in task_specs:
            res = self.evaluate_episode(agent, spec, max_steps=max_steps, perturbation=perturbation)
            results.append(res)

        success_rate = float(np.mean([r["task_success"] for r in results])) * 100.0
        goal_rate = float(np.mean([r["goal_reached"] for r in results])) * 100.0
        col_rate = float(np.mean([r["had_collision"] for r in results])) * 100.0
        avg_eff = float(np.mean([r["path_efficiency"] for r in results]))
        avg_err = float(np.mean([r["final_distance"] for r in results]))
        mean_lat = float(np.mean([r["mean_latency_us"] for r in results]))
        p99_lat = float(np.mean([r["p99_latency_us"] for r in results]))
        rec_rate = float(np.mean([r["recovered"] for r in results])) * 100.0 if perturbation else 0.0

        return {
            "model_id": agent.model_id,
            "model_name": agent.name,
            "total_episodes": len(results),
            "success_rate": success_rate,
            "goal_reached_rate": goal_rate,
            "collision_rate": col_rate,
            "path_efficiency": avg_eff,
            "goal_localization_error": avg_err,
            "mean_latency_us": mean_lat,
            "p99_latency_us": p99_lat,
            "recovery_rate": rec_rate,
            "episodes": results
        }

    def close(self):
        self.env.close()
