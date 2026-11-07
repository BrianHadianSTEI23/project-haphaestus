"""Privileged, deterministic geometric demonstration planner.

The expert has access to task geometry supplied by the simulator. It is used only
to generate supervision labels; it is not evidence that a learned agent inferred
obstacle geometry from pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PlannerConfig:
    vx_limit: float = 0.6
    vy_limit: float = 0.4
    wz_limit: float = 1.5
    robot_radius: float = 0.20
    safety_margin: float = 0.06
    horizon_steps: int = 10
    dt: float = 0.1


class ExpertNavigator:
    """Finite-horizon lattice planner using Mecanum body-frame kinematics.

    Commands are propagated with the same body-to-world transform and limits as
    ``JetAutoEnv``. A candidate is rejected on predicted contact with physical
    obstacle geometry or an arena wall; feasible candidates trade goal progress,
    clearance, heading, and control effort.
    """

    def __init__(self, vx_limit=0.6, vy_limit=0.4, wz_limit=1.5, *, horizon_steps=10, dt=0.1):
        self.config = PlannerConfig(vx_limit, vy_limit, wz_limit, horizon_steps=horizon_steps, dt=dt)
        self._commands = np.asarray(
            [[vx, vy, wz] for vx in (-0.7, 0.0, 0.7)
             for vy in (-0.7, 0.0, 0.7)
             for wz in (-0.7, 0.0, 0.7)], dtype=np.float32
        )

    @staticmethod
    def _obstacle_radius(obstacle: dict) -> float:
        if obstacle.get("shape", "cylinder") == "box":
            size = np.asarray(obstacle.get("size", [0.25, 0.25]), dtype=float)
            return float(np.linalg.norm(size[:2]))
        return float(obstacle.get("radius", 0.25))

    def _minimum_clearance(self, xy: np.ndarray, obstacles: Iterable[dict]) -> float:
        clearances = []
        for obstacle in obstacles:
            center = np.asarray(obstacle["pos"], dtype=float)[:2]
            clearances.append(np.linalg.norm(xy - center) - self._obstacle_radius(obstacle) - self.config.robot_radius)
        clearances.append(2.5 - self.config.robot_radius - np.max(np.abs(xy)))
        return float(min(clearances))

    def _rollout(self, pose: np.ndarray, action: np.ndarray, obstacles: Iterable[dict]) -> tuple[np.ndarray, float]:
        x, y, heading = map(float, pose)
        min_clearance = float("inf")
        for _ in range(self.config.horizon_steps):
            vx = float(action[0]) * self.config.vx_limit
            vy = float(action[1]) * self.config.vy_limit
            wz = float(action[2]) * self.config.wz_limit
            x += (np.cos(heading) * vx - np.sin(heading) * vy) * self.config.dt
            y += (np.sin(heading) * vx + np.cos(heading) * vy) * self.config.dt
            heading = float(np.arctan2(np.sin(heading + wz * self.config.dt), np.cos(heading + wz * self.config.dt)))
            min_clearance = min(min_clearance, self._minimum_clearance(np.array([x, y]), obstacles))
        return np.array([x, y, heading], dtype=np.float32), min_clearance

    def compute_action(self, current_pose, current_waypoint, task_spec, is_last_waypoint=True):
        if task_spec.get("capability") == "action_prediction" and "action_probe_vector" in task_spec:
            return np.clip(np.asarray(task_spec["action_probe_vector"], dtype=np.float32), -1.0, 1.0)
        if task_spec.get("is_failure_example", False):
            return np.array([0.9, 0.0, 0.0], dtype=np.float32)

        pose = np.asarray(current_pose, dtype=np.float32)
        waypoint = np.asarray(current_waypoint, dtype=np.float32)
        obstacles = list(task_spec.get("obstacles", {}).values())
        initial_distance = float(np.linalg.norm(waypoint[:2] - pose[:2]))
        best_action = np.zeros(3, dtype=np.float32)
        best_cost = float("inf")

        for action in self._commands:
            final_pose, min_clearance = self._rollout(pose, action, obstacles)
            if min_clearance < self.config.safety_margin:
                continue
            remaining = float(np.linalg.norm(waypoint[:2] - final_pose[:2]))
            desired_heading = float(waypoint[2]) if is_last_waypoint and remaining < 0.35 else float(np.arctan2(waypoint[1] - final_pose[1], waypoint[0] - final_pose[0]))
            heading_error = abs(float(np.arctan2(np.sin(desired_heading - final_pose[2]), np.cos(desired_heading - final_pose[2]))))
            progress = initial_distance - remaining
            clearance_penalty = 0.0 if min_clearance >= 0.30 else (0.30 - min_clearance) ** 2
            stationary_penalty = 0.25 if np.allclose(action, 0.0) and initial_distance > 0.22 else 0.0
            cost = remaining + 0.15 * heading_error + 3.0 * clearance_penalty + 0.015 * float(np.dot(action, action)) - 0.05 * progress + stationary_penalty
            if cost < best_cost:
                best_cost = cost
                best_action = action.copy()

        # An all-infeasible lattice returns a safe stop instead of an arbitrary path.
        return np.clip(best_action, -1.0, 1.0).astype(np.float32)
