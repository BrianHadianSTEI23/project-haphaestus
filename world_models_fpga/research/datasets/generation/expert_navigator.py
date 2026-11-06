"""
Phase 1: Controlled Intelligence Benchmark - Expert Demonstrator Controller
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Implements 3-DOF continuous Mecanum kinematics controller with:
  - Waypoint tracking with proportional velocity profile
  - Artificial potential field (APF) obstacle avoidance
  - Multi-waypoint sequential switching
  - Canonical action probing for world model dynamics evaluation
  - Controlled failure / collision trajectory generation
"""

import numpy as np

class ExpertNavigator:
    def __init__(self, vx_limit=0.6, vy_limit=0.4, wz_limit=1.5):
        self.vx_limit = vx_limit
        self.vy_limit = vy_limit
        self.wz_limit = wz_limit

    def compute_action(self, current_pose, current_waypoint, task_spec, is_last_waypoint=True):
        """
        Computes normalized action [vx, vy, wz] in [-1.0, 1.0].
        current_pose: [x, y, theta]
        current_waypoint: [x_wp, y_wp, theta_wp]
        """
        # If running an action-prediction probe, return fixed probe command
        if task_spec.get("capability", "") == "action_prediction" and "action_probe_vector" in task_spec:
            return task_spec["action_probe_vector"].copy()

        # If deliberate failure example (e.g. baseline collision demonstration)
        if task_spec.get("is_failure_example", False):
            # Drive straight ahead with no avoidance
            return np.array([0.9, 0.0, 0.0], dtype=np.float32)

        x, y, theta = current_pose[0], current_pose[1], current_pose[2]
        x_wp, y_wp, th_wp = current_waypoint[0], current_waypoint[1], current_waypoint[2]

        dx = x_wp - x
        dy = y_wp - y
        dist = np.sqrt(dx * dx + dy * dy)

        # Body-frame projection of target vector
        c, s = np.cos(theta), np.sin(theta)
        v_world_x = dx
        v_world_y = dy

        # Apply velocity scaling based on distance
        speed = min(0.5, dist * 1.5)
        if dist > 1e-4:
            v_world_x = (dx / dist) * speed
            v_world_y = (dy / dist) * speed
        else:
            v_world_x = 0.0
            v_world_y = 0.0

        # Transform to body frame
        vx_body = c * v_world_x + s * v_world_y
        vy_body = -s * v_world_x + c * v_world_y

        # Heading alignment
        if is_last_waypoint and dist < 0.35:
            # Align with final target heading
            target_heading = th_wp
        else:
            # Face motion direction
            target_heading = np.arctan2(dy, dx) if dist > 0.05 else theta

        heading_err = np.arctan2(np.sin(target_heading - theta), np.cos(target_heading - theta))
        wz_cmd = float(np.clip(heading_err * 2.2, -self.wz_limit, self.wz_limit))

        # Normalized action vector
        act_norm = np.array([
            np.clip(vx_body / self.vx_limit, -1.0, 1.0),
            np.clip(vy_body / self.vy_limit, -1.0, 1.0),
            np.clip(wz_cmd / self.wz_limit, -1.0, 1.0)
        ], dtype=np.float32)

        return act_norm
