"""
Phase 1: Controlled Intelligence Benchmark - Capability Task Definitions
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Defines 6 capability categories with controlled variation across:
  1. Easy examples
  2. Difficult examples
  3. Counterfactual examples
  4. Out-of-distribution (OOD) examples
  5. Failure examples
"""

import copy
import numpy as np

# Canonical arena obstacles definition
CANONICAL_OBSTACLES = {
    "yellow_cylinder": {
        "geom_name": "obs_yellow_cyl",
        "pos": np.array([-1.2, -0.9, 0.35], dtype=np.float32),
        "radius": 0.25,
        "height": 0.35,
        "shape": "cylinder",
        "color": "yellow",
        "rgba": np.array([0.95, 0.85, 0.1, 1.0], dtype=np.float32),
        "label": "yellow cylinder",
        "aliases": ["yellow cylinder", "yellow cyl", "yellow object"]
    },
    "red_box": {
        "geom_name": "obs_red_box",
        "pos": np.array([1.1, 0.9, 0.35], dtype=np.float32),
        "size": np.array([0.25, 0.25, 0.35], dtype=np.float32),
        "shape": "box",
        "color": "red",
        "rgba": np.array([0.85, 0.2, 0.2, 1.0], dtype=np.float32),
        "label": "red box",
        "aliases": ["red box", "red cube", "red obstacle"]
    },
    "blue_cylinder": {
        "geom_name": "obs_blue_cyl",
        "pos": np.array([-1.0, 1.2, 0.35], dtype=np.float32),
        "radius": 0.25,
        "height": 0.35,
        "shape": "cylinder",
        "color": "blue",
        "rgba": np.array([0.2, 0.4, 0.85, 1.0], dtype=np.float32),
        "label": "blue cylinder",
        "aliases": ["blue cylinder", "blue cyl", "blue pillar"]
    },
    "green_cylinder": {
        "geom_name": "obs_green_cyl",
        "pos": np.array([1.2, -1.1, 0.35], dtype=np.float32),
        "radius": 0.22,
        "height": 0.35,
        "shape": "cylinder",
        "color": "green",
        "rgba": np.array([0.2, 0.75, 0.3, 1.0], dtype=np.float32),
        "label": "green cylinder",
        "aliases": ["green cylinder", "green cyl", "green pillar"]
    },
    "purple_box": {
        "geom_name": "obs_purple_box",
        "pos": np.array([0.0, 1.7, 0.35], dtype=np.float32),
        "size": np.array([0.3, 0.2, 0.35], dtype=np.float32),
        "shape": "box",
        "color": "purple",
        "rgba": np.array([0.6, 0.2, 0.75, 1.0], dtype=np.float32),
        "label": "purple box",
        "aliases": ["purple box", "purple cube", "purple container"]
    },
    "cyan_cylinder": {
        "geom_name": "obs_cyan_cyl",
        "pos": np.array([-0.2, -1.7, 0.35], dtype=np.float32),
        "radius": 0.25,
        "height": 0.35,
        "shape": "cylinder",
        "color": "cyan",
        "rgba": np.array([0.1, 0.8, 0.8, 1.0], dtype=np.float32),
        "label": "cyan cylinder",
        "aliases": ["cyan cylinder", "cyan cyl", "cyan pillar"]
    }
}

CAPABILITIES = [
    "object_grounding",
    "spatial_reasoning",
    "obstacle_avoidance",
    "action_prediction",
    "temporal_planning",
    "generalization"
]

DIFFICULTY_TIERS = [
    "easy",
    "difficult",
    "counterfactual",
    "ood",
    "failure"
]

def get_waypoint_for_relation(lmk_pos, relation, robot_pos, clearance=0.60):
    """
    Computes exact geometric waypoint for a relation.
    Relation semantics:
      - 'behind': outward radially from arena center (away from origin)
      - 'in_front': inward radially towards arena center (between object and origin)
      - 'left_of': 90 deg counter-clockwise from radial direction
      - 'right_of': 90 deg clockwise from radial direction
      - 'approach': directly in front along the robot-to-object line of sight
    """
    pos_2d = lmk_pos[:2]
    center_vec = pos_2d - np.array([0.0, 0.0], dtype=np.float32)
    dist_c = np.linalg.norm(center_vec)
    radial_dir = center_vec / (dist_c + 1e-6)
    tangent_dir = np.array([-radial_dir[1], radial_dir[0]], dtype=np.float32)

    if relation == "behind":
        target_xy = pos_2d + radial_dir * clearance
    elif relation == "in_front":
        target_xy = pos_2d - radial_dir * clearance
    elif relation == "left_of":
        target_xy = pos_2d + tangent_dir * clearance
    elif relation == "right_of":
        target_xy = pos_2d - tangent_dir * clearance
    elif relation == "approach":
        r_to_lmk = pos_2d - robot_pos[:2]
        d = np.linalg.norm(r_to_lmk)
        u = r_to_lmk / (d + 1e-6)
        target_xy = pos_2d - u * clearance
    else:
        target_xy = pos_2d + radial_dir * clearance

    # Facing towards landmark
    facing = pos_2d - target_xy
    target_th = float(np.arctan2(facing[1], facing[0]))
    return np.array([target_xy[0], target_xy[1], target_th], dtype=np.float32)


def generate_task_specification(capability, difficulty, seed_idx, canonical_obstacles=None):
    """
    Generates a deterministic task configuration dictionary for a given capability & difficulty.
    """
    if canonical_obstacles is None:
        obstacles = copy.deepcopy(CANONICAL_OBSTACLES)
    else:
        obstacles = copy.deepcopy(canonical_obstacles)

    rng = np.random.default_rng(seed_idx)
    lmk_keys = list(obstacles.keys())
    target_key = lmk_keys[seed_idx % len(lmk_keys)]
    target_obj = obstacles[target_key]

    task_spec = {
        "capability": capability,
        "difficulty": difficulty,
        "seed_idx": seed_idx,
        "target_key": target_key,
        "obstacles": obstacles,
        "camera_config": "front_cam",
        "custom_geoms": {},
        "is_failure_example": False,
        "waypoints": [],
        "initial_pose": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "instruction": "",
        "semantic_relation": "approach",
        "expected_invariants": []
    }

    # -------------------------------------------------------------
    # 1. OBJECT GROUNDING
    # -------------------------------------------------------------
    if capability == "object_grounding":
        task_spec["semantic_relation"] = "approach"
        verbs = ["approach", "navigate to", "go to", "reach"]
        verb = verbs[seed_idx % len(verbs)]
        task_spec["instruction"] = f"{verb} the {target_obj['label']}"

        t_pos = target_obj["pos"][:2]
        pos_jitter = rng.uniform(-0.12, 0.12, size=2).astype(np.float32)
        yaw_jitter = float(rng.uniform(-0.10, 0.10))

        if difficulty == "easy":
            # Start facing target from 1.2m away
            angle = float(np.arctan2(t_pos[1], t_pos[0]))
            start_pos = t_pos - np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * 1.2 + pos_jitter
            start_yaw = float(np.arctan2(t_pos[1] - start_pos[1], t_pos[0] - start_pos[0])) + yaw_jitter
            task_spec["initial_pose"] = np.array([start_pos[0], start_pos[1], start_yaw], dtype=np.float32)

        elif difficulty == "difficult":
            # Start facing away (180 deg) from 2.2m away
            angle = float(np.arctan2(t_pos[1], t_pos[0])) + 0.3
            start_pos = t_pos - np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * 2.2 + pos_jitter
            start_yaw = angle + np.pi + yaw_jitter  # Facing away
            task_spec["initial_pose"] = np.array([start_pos[0], start_pos[1], start_yaw], dtype=np.float32)

        elif difficulty == "counterfactual":
            # Color swap: target label is mutated in RGBA to test appearance invariance
            task_spec["instruction"] = f"reach the {target_obj['label']}"
            obstacles[target_key]["rgba"] = np.array([0.15, 0.35, 0.85, 1.0], dtype=np.float32)
            obstacles[target_key]["color"] = "mutated_blue"
            start_pos = pos_jitter.copy()
            start_yaw = float(np.arctan2(t_pos[1] - start_pos[1], t_pos[0] - start_pos[0])) + yaw_jitter
            task_spec["initial_pose"] = np.array([start_pos[0], start_pos[1], start_yaw], dtype=np.float32)
            task_spec["expected_invariants"].append("spatial_relation_invariant_to_color")

        elif difficulty == "ood":
            # Perturbed position: landmark moved by up to 0.4m
            shift = rng.uniform(-0.4, 0.4, size=2).astype(np.float32)
            obstacles[target_key]["pos"][:2] += shift
            t_pos = obstacles[target_key]["pos"][:2]
            start_pos = pos_jitter.copy()
            start_yaw = float(np.arctan2(t_pos[1] - start_pos[1], t_pos[0] - start_pos[0])) + yaw_jitter
            task_spec["initial_pose"] = np.array([start_pos[0], start_pos[1], start_yaw], dtype=np.float32)

        elif difficulty == "failure":
            # Trapped against wall (dist = 0.15m from arena boundary), facing wall
            task_spec["initial_pose"] = np.array([2.35, float(pos_jitter[1]), yaw_jitter], dtype=np.float32)
            task_spec["is_failure_example"] = True

        goal_pose = get_waypoint_for_relation(obstacles[target_key]["pos"], "approach", task_spec["initial_pose"])
        task_spec["waypoints"] = [goal_pose]

    # -------------------------------------------------------------
    # 2. SPATIAL REASONING
    # -------------------------------------------------------------
    elif capability == "spatial_reasoning":
        relations = ["behind", "in_front", "left_of", "right_of"]
        rel = relations[seed_idx % len(relations)]
        task_spec["semantic_relation"] = rel
        rel_str = rel.replace("_", " ")
        task_spec["instruction"] = f"go {rel_str} the {target_obj['label']}"

        t_pos = target_obj["pos"][:2]
        pos_jitter = rng.uniform(-0.12, 0.12, size=2).astype(np.float32)
        yaw_jitter = float(rng.uniform(-0.10, 0.10))

        if difficulty == "easy":
            task_spec["initial_pose"] = np.array([pos_jitter[0], pos_jitter[1], float(np.arctan2(t_pos[1] - pos_jitter[1], t_pos[0] - pos_jitter[0])) + yaw_jitter], dtype=np.float32)
        elif difficulty == "difficult":
            # Start on opposite side of target, forcing full detour
            radial = t_pos / np.linalg.norm(t_pos)
            opposite_pos = t_pos + radial * 0.8 + pos_jitter * 0.5
            task_spec["initial_pose"] = np.array([opposite_pos[0], opposite_pos[1], float(np.arctan2(-radial[1], -radial[0])) + yaw_jitter], dtype=np.float32)
        elif difficulty == "counterfactual":
            # Rotate initial heading by 90 deg
            task_spec["initial_pose"] = np.array([pos_jitter[0], pos_jitter[1], float(np.arctan2(t_pos[1] - pos_jitter[1], t_pos[0] - pos_jitter[0]) + np.pi/2) + yaw_jitter], dtype=np.float32)
            task_spec["expected_invariants"].append("allocentric_spatial_relation_invariant_to_robot_heading")
        elif difficulty == "ood":
            # Shift landmark position
            shift = rng.uniform(-0.35, 0.35, size=2).astype(np.float32)
            obstacles[target_key]["pos"][:2] += shift
            task_spec["initial_pose"] = np.array([pos_jitter[0], pos_jitter[1], yaw_jitter], dtype=np.float32)
        elif difficulty == "failure":
            # Waypoint falls near corner wall
            task_spec["initial_pose"] = np.array([-2.2, -2.2 + float(pos_jitter[1]), yaw_jitter], dtype=np.float32)
            task_spec["is_failure_example"] = True

        goal_pose = get_waypoint_for_relation(obstacles[target_key]["pos"], rel, task_spec["initial_pose"])
        # If behind, add tangential flank waypoint for smooth clearance
        if rel == "behind":
            c_vec = obstacles[target_key]["pos"][:2]
            rad = c_vec / np.linalg.norm(c_vec)
            tan = np.array([-rad[1], rad[0]], dtype=np.float32)
            flank = obstacles[target_key]["pos"][:2] + tan * 0.70
            flank_pose = np.array([flank[0], flank[1], goal_pose[2]], dtype=np.float32)
            task_spec["waypoints"] = [flank_pose, goal_pose]
        else:
            task_spec["waypoints"] = [goal_pose]

    # -------------------------------------------------------------
    # 3. OBSTACLE AVOIDANCE
    # -------------------------------------------------------------
    elif capability == "obstacle_avoidance":
        task_spec["semantic_relation"] = "approach"
        task_spec["instruction"] = f"navigate around obstacle to {target_obj['label']}"
        t_pos = target_obj["pos"][:2]
        pos_jitter = rng.uniform(-0.10, 0.10, size=2).astype(np.float32)

        # Robot starts opposite to target with arena center obstacle in line of sight
        start_pos = -t_pos * 0.9 + pos_jitter
        los_vec = t_pos - start_pos
        los_dir = los_vec / (np.linalg.norm(los_vec) + 1e-6)
        ortho_dir = np.array([-los_dir[1], los_dir[0]], dtype=np.float32)
        mid_los = (start_pos + t_pos) * 0.5
        task_spec["initial_pose"] = np.array([start_pos[0], start_pos[1], float(np.arctan2(los_vec[1], los_vec[0]))], dtype=np.float32)

        if difficulty == "easy":
            # Obstacle slightly offset from central line of sight
            obs_xy = mid_los + ortho_dir * 0.18
            obstacles["inserted_obstacle"] = {
                "geom_name": "obs_inserted",
                "pos": np.array([float(obs_xy[0]), float(obs_xy[1]), 0.35], dtype=np.float32),
                "radius": 0.22,
                "height": 0.35,
                "shape": "cylinder",
                "color": "magenta",
                "rgba": np.array([0.8, 0.1, 0.8, 1.0], dtype=np.float32),
            }
        elif difficulty in ("difficult", "counterfactual", "failure"):
            # Intermediate obstacle centered directly between robot and goal
            obstacles["inserted_obstacle"] = {
                "geom_name": "obs_inserted",
                "pos": np.array([float(mid_los[0]), float(mid_los[1]), 0.35], dtype=np.float32),
                "radius": 0.22,
                "height": 0.35,
                "shape": "cylinder",
                "color": "magenta",
                "rgba": np.array([0.8, 0.1, 0.8, 1.0], dtype=np.float32),
            }
            if difficulty == "counterfactual":
                task_spec["expected_invariants"].append("goal_reached_despite_intermediate_obstacle")
            elif difficulty == "failure":
                task_spec["is_failure_example"] = True
        elif difficulty == "ood":
            # Two obstacles creating narrow corridor
            obs1_xy = mid_los + ortho_dir * 0.38
            obs2_xy = mid_los - ortho_dir * 0.38
            obstacles["inserted_obstacle"] = {
                "geom_name": "obs_inserted",
                "pos": np.array([float(obs1_xy[0]), float(obs1_xy[1]), 0.35], dtype=np.float32),
                "radius": 0.22,
                "height": 0.35,
                "shape": "cylinder",
                "color": "magenta",
                "rgba": np.array([0.8, 0.1, 0.8, 1.0], dtype=np.float32),
            }
            obstacles["slit_barrier"] = {
                "geom_name": "obs_barrier",
                "pos": np.array([float(obs2_xy[0]), float(obs2_xy[1]), 0.35], dtype=np.float32),
                "size": np.array([0.15, 0.22, 0.35], dtype=np.float32),
                "radius": 0.22,
                "height": 0.35,
                "shape": "box",
                "color": "dark_gray",
                "rgba": np.array([0.3, 0.3, 0.3, 1.0], dtype=np.float32),
            }

        goal_pose = get_waypoint_for_relation(obstacles[target_key]["pos"], "approach", task_spec["initial_pose"])
        # Intermediate detour waypoint around center
        mid_point = mid_los - ortho_dir * 0.65
        mid_pose = np.array([mid_point[0], mid_point[1], float(np.arctan2(t_pos[1] - mid_point[1], t_pos[0] - mid_point[0]))], dtype=np.float32)
        task_spec["waypoints"] = [mid_pose, goal_pose]

    # -------------------------------------------------------------
    # 4. ACTION-CONDITIONED PREDICTION
    # -------------------------------------------------------------
    elif capability == "action_prediction":
        task_spec["semantic_relation"] = "action_probe"
        action_modes = [
            ("pure_forward", [0.8, 0.0, 0.0]),
            ("pure_reverse", [-0.8, 0.0, 0.0]),
            ("lateral_left", [0.0, 0.8, 0.0]),
            ("lateral_right", [0.0, -0.8, 0.0]),
            ("ccw_yaw", [0.0, 0.0, 0.8]),
            ("cw_yaw", [0.0, 0.0, -0.8]),
            ("diagonal_strafe", [0.6, 0.6, 0.0]),
            ("arc_turn", [0.6, 0.0, 0.6])
        ]
        mode_name, act_vec = action_modes[seed_idx % len(action_modes)]
        pos_jitter = rng.uniform(-0.18, 0.18, size=2).astype(np.float32)
        yaw_jitter = float(rng.uniform(-0.25, 0.25))
        task_spec["instruction"] = f"execute {mode_name}"
        task_spec["action_probe_vector"] = np.array(act_vec, dtype=np.float32)
        task_spec["initial_pose"] = np.array([pos_jitter[0], pos_jitter[1], yaw_jitter], dtype=np.float32)
        task_spec["waypoints"] = [np.array([1.0, 0.0, 0.0], dtype=np.float32)]

    # -------------------------------------------------------------
    # 5. TEMPORAL PLANNING
    # -------------------------------------------------------------
    elif capability == "temporal_planning":
        task_spec["semantic_relation"] = "multi_waypoint"
        task_spec["instruction"] = f"circumnavigate arena to {target_obj['label']}"
        pos_jitter = rng.uniform(-0.12, 0.12, size=2).astype(np.float32)
        yaw_jitter = float(rng.uniform(-0.12, 0.12))
        task_spec["initial_pose"] = np.array([-1.5 + pos_jitter[0], 1.5 + pos_jitter[1], -0.78 + yaw_jitter], dtype=np.float32)

        # Multi-stage sequential waypoints across arena quadrants
        wp1 = np.array([0.0, 1.0, -1.57], dtype=np.float32)
        wp2 = np.array([1.0, 0.0, -3.14], dtype=np.float32)
        wp3 = get_waypoint_for_relation(obstacles[target_key]["pos"], "behind", wp2)
        task_spec["waypoints"] = [wp1, wp2, wp3]

        if difficulty == "failure":
            task_spec["is_failure_example"] = True

    # -------------------------------------------------------------
    # 6. GENERALIZATION
    # -------------------------------------------------------------
    elif capability == "generalization":
        task_spec["semantic_relation"] = "behind"
        task_spec["instruction"] = f"go behind the {target_obj['label']} in permuted layout"

        # Permute obstacle positions (swap quadrants)
        quadrants = [
            np.array([-1.2, -1.0, 0.35]),
            np.array([1.2, -1.0, 0.35]),
            np.array([1.2, 1.0, 0.35]),
            np.array([-1.2, 1.0, 0.35]),
            np.array([0.0, -1.6, 0.35]),
            np.array([0.0, 1.6, 0.35])
        ]
        perm = rng.permutation(len(quadrants))
        for i, k in enumerate(obstacles.keys()):
            obstacles[k]["pos"] = quadrants[perm[i]].astype(np.float32)

        # Unseen initial robot headings: 0, 90, 180, 270 deg
        headings = [0.0, np.pi/2, np.pi, -np.pi/2]
        h = headings[seed_idx % len(headings)]
        task_spec["initial_pose"] = np.array([0.0, 0.0, h], dtype=np.float32)

        goal_pose = get_waypoint_for_relation(obstacles[target_key]["pos"], "behind", task_spec["initial_pose"])
        task_spec["waypoints"] = [goal_pose]

    return task_spec
