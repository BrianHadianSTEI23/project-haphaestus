"""
Phase 2: Unified Benchmark Evaluator Engine (Repaired for Physical & Information Integrity)

Evaluates agents under identical initial conditions, physical MuJoCo interventions
(via JetAutoEnv.apply_task_spec), and strict information regimes:
  - Track A ('oracle'): agent receives privileged robot_pose, target_pose, and task_spec.
  - Track B ('structured'): agent receives ONLY camera RGB obs + body-frame goal_vec_body.
    World coordinates and obstacle layouts are withheld.
  - Track C ('observation'): agent receives ONLY camera RGB obs + discrete task instruction
    (target_key, semantic_relation). All poses and goal coordinates are withheld.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from env.amr_env import JetAutoEnv
from research.benchmarks.model_wrappers import compute_body_frame_goal_vec


class BenchmarkEvaluator:
    def __init__(self, xml_path: str | None = None):
        self.env = JetAutoEnv(xml_path=xml_path)

    def _invoke_agent(self, agent, obs: np.ndarray, curr_pose: np.ndarray, curr_wp: np.ndarray, task_spec: dict):
        track = getattr(agent, "track", "structured")
        if track == "oracle":
            out = agent.act(obs, robot_pose=curr_pose, target_pose=curr_wp, task_spec=task_spec)
        elif track == "observation":
            # Track C: strictly withhold all poses, waypoints, and obstacle coordinates
            public_spec = {
                "target_key": task_spec.get("target_key", ""),
                "semantic_relation": task_spec.get("semantic_relation", "approach"),
                "instruction": task_spec.get("instruction", ""),
            }
            out = agent.act(obs, robot_pose=None, target_pose=None, task_spec=public_spec, goal_vec_body=None)
        else:
            # Track B ('structured'): provide only egocentric body-frame relative goal vector
            goal_vec_body = compute_body_frame_goal_vec(curr_pose, curr_wp)
            out = agent.act(obs, robot_pose=None, target_pose=None, task_spec=None, goal_vec_body=goal_vec_body)

        if isinstance(out, tuple):
            return out[0], out[1]
        return out, {}

    def evaluate_episode(
        self,
        agent,
        task_spec: dict,
        max_steps: int = 45,
        perturbation: dict | None = None,
        inject_disturbance: bool = False,
    ) -> dict:
        """
        Runs one closed-loop episode in MuJoCo for the agent under task_spec.
        """
        self.env.apply_task_spec(task_spec)
        if hasattr(agent, "reset"):
            agent.reset()

        init_pos = np.asarray(task_spec["initial_pose"], dtype=np.float32)
        obs, _ = self.env.reset(options={"initial_pos": (float(init_pos[0]), float(init_pos[1]), float(init_pos[2]))})

        waypoints = task_spec.get("waypoints", [init_pos])
        wp_idx = 0
        final_wp = np.asarray(waypoints[-1], dtype=np.float32) if len(waypoints) > 0 else init_pos

        # Geodesic reference path length along waypoints
        geodesic_dist = float(np.linalg.norm(waypoints[0][:2] - init_pos[:2]))
        for i in range(len(waypoints) - 1):
            geodesic_dist += float(np.linalg.norm(waypoints[i + 1][:2] - waypoints[i][:2]))

        # Ensure sufficient steps for multi-waypoint / far-corner geodesic distances (max speed ~0.04m/step)
        if "max_steps" in task_spec:
            effective_max_steps = int(task_spec["max_steps"])
        elif max_steps is not None and max_steps != 45:
            # Explicit custom step budget passed by caller (e.g. unit test or short rollout)
            effective_max_steps = int(max_steps)
        else:
            effective_max_steps = max(int(max_steps or 45), int(geodesic_dist / 0.024) + 25)

        trajectory = [init_pos[:2].copy()]
        collision_count = 0
        latencies = []
        recovered = False
        goal_reached = False
        actual_path_dist = 0.0
        prev_xy = init_pos[:2].copy()

        for t in range(effective_max_steps):
            curr_pose = self.env.get_robot_pose()
            curr_wp = np.asarray(waypoints[wp_idx], dtype=np.float32)

            dist_wp = float(np.linalg.norm(curr_wp[:2] - curr_pose[:2]))
            if dist_wp < 0.25:
                if wp_idx < len(waypoints) - 1:
                    wp_idx += 1
                    curr_wp = np.asarray(waypoints[wp_idx], dtype=np.float32)
                else:
                    goal_reached = True

            # Optional mid-episode disturbance
            if inject_disturbance and t == 12:
                self.env.data.qpos[self.env.jx_id] += 0.25
                self.env.data.qpos[self.env.jy_id] -= 0.20
                self.env.data.qpos[self.env.jz_id] += 0.35
                curr_pose = self.env.get_robot_pose()
            elif perturbation and t == perturbation.get("step", 10):
                impulse = perturbation.get("impulse", [0.0, 0.8, 0.0])
                self.env.data.qvel[self.env.jx_id] += impulse[0]
                self.env.data.qvel[self.env.jy_id] += impulse[1]
                self.env.data.qvel[self.env.jz_id] += impulse[2]

            t0 = time.perf_counter()
            action, _ = self._invoke_agent(agent, obs, curr_pose, curr_wp, task_spec)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1e6)

            if task_spec.get("capability") == "action_prediction" and "action_probe_vector" in task_spec:
                action = np.asarray(task_spec["action_probe_vector"], dtype=np.float32)

            obs, _, _, _, step_info = self.env.step(action)
            if step_info["collision"]:
                collision_count += 1

            next_pose = self.env.get_robot_pose()
            actual_path_dist += float(np.linalg.norm(next_pose[:2] - prev_xy))
            prev_xy = next_pose[:2].copy()
            trajectory.append(prev_xy.copy())

            if float(np.linalg.norm(final_wp[:2] - next_pose[:2])) < 0.25 and wp_idx >= len(waypoints) - 1:
                goal_reached = True
                if perturbation or inject_disturbance:
                    recovered = True
                if t > 5:
                    break

        end_pose = self.env.get_robot_pose()
        final_goal_err = float(np.linalg.norm(final_wp[:2] - end_pose[:2]))
        if final_goal_err < 0.28:
            goal_reached = True

        if task_spec.get("is_failure_example", False):
            is_success = bool(collision_count == 0)
        elif task_spec.get("capability") == "action_prediction":
            is_success = True
        else:
            is_success = bool(goal_reached and (collision_count == 0))

        if actual_path_dist > 1e-4:
            path_eff = float(np.clip(geodesic_dist / max(actual_path_dist, geodesic_dist), 0.0, 1.0))
        else:
            path_eff = 0.0

        self.env.restore_physical_baseline()

        return {
            "episode_id": task_spec.get("episode_id", "ep"),
            "track": getattr(agent, "track", "structured"),
            "ablation_mode": getattr(agent, "ablation_mode", "prediction_used"),
            "success": is_success,
            "task_success": is_success,
            "goal_reached": bool(goal_reached),
            "collisions": int(collision_count),
            "collision_count": int(collision_count),
            "collision_free": bool(collision_count == 0),
            "had_collision": bool(collision_count > 0),
            "path_efficiency": path_eff,
            "final_goal_error": final_goal_err,
            "final_distance": final_goal_err,
            "trajectory_length": actual_path_dist,
            "steps_taken": len(latencies),
            "steps": len(latencies),
            "mean_latency_us": float(np.mean(latencies)),
            "median_latency_us": float(np.median(latencies)),
            "p99_latency_us": float(np.percentile(latencies, 99)),
            "recovered": bool(recovered),
        }

    def evaluate_suite(self, agent, task_specs, max_steps=45, perturbation=None):
        results = [self.evaluate_episode(agent, spec, max_steps=max_steps, perturbation=perturbation) for spec in task_specs]
        return {
            "model_name": getattr(agent, "name", "Agent"),
            "track": getattr(agent, "track", "structured"),
            "ablation_mode": getattr(agent, "ablation_mode", "prediction_used"),
            "total_episodes": len(results),
            "success_rate": float(np.mean([r["success"] for r in results])) * 100.0,
            "goal_reached_rate": float(np.mean([r["goal_reached"] for r in results])) * 100.0,
            "collision_rate": float(np.mean([r["had_collision"] for r in results])) * 100.0,
            "path_efficiency": float(np.mean([r["path_efficiency"] for r in results])),
            "goal_localization_error": float(np.mean([r["final_goal_error"] for r in results])),
            "mean_latency_us": float(np.mean([r["mean_latency_us"] for r in results])),
            "p99_latency_us": float(np.mean([r["p99_latency_us"] for r in results])),
            "recovery_rate": float(np.mean([r["recovered"] for r in results])) * 100.0 if perturbation else 0.0,
            "episodes": results,
        }

    def close(self):
        self.env.close()
