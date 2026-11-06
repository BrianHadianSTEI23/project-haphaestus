"""
Phase 1: Controlled Intelligence Benchmark - Disambiguation Test Suites
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Implements explicit disambiguation probe pairs:
  1. Robot-relative left vs World-relative left
  2. Robot-relative front vs Object-relative behind
  3. Geometric proximity vs Visual proximity
  4. Target visibility vs Actual reachability
"""

import copy
import numpy as np

def generate_disambiguation_tasks():
    """
    Returns a dictionary of probe task specifications for the 4 disambiguation axes.
    """
    from research.datasets.generation.capability_tasks import CANONICAL_OBSTACLES, get_waypoint_for_relation
    tasks = []

    # -------------------------------------------------------------
    # 1. Robot-relative left vs World-relative left
    # When robot faces SOUTH (theta = -pi/2), robot's left is EAST (+x),
    # whereas world left is WEST (-x).
    # -------------------------------------------------------------
    base_obs = copy.deepcopy(CANONICAL_OBSTACLES)
    tasks.append({
        "probe_id": "DISAMBIG_01_ego_left_vs_world_left",
        "axis": "coordinate_frame_left",
        "capability": "spatial_reasoning",
        "difficulty": "disambiguation",
        "description": "Robot facing south: egocentric left is east (+x), allocentric left is west (-x)",
        "instruction": "go to the left",
        "initial_pose": np.array([0.0, 0.0, -np.pi/2], dtype=np.float32),
        "obstacles": base_obs,
        "egocentric_target": np.array([1.2, 0.0, -np.pi/2], dtype=np.float32),
        "allocentric_target": np.array([-1.2, 0.0, -np.pi/2], dtype=np.float32),
        "waypoints": [np.array([1.2, 0.0, -np.pi/2], dtype=np.float32)],
        "semantic_relation": "left_of"
    })

    # -------------------------------------------------------------
    # 2. Robot-relative front vs Object-relative behind
    # Robot is positioned outside landmark facing landmark and arena center.
    # Moving straight forward reaches 'in front of robot', which is 'in front of landmark',
    # whereas 'behind landmark' requires moving around to the opposite side.
    # -------------------------------------------------------------
    lmk = "yellow_cylinder"
    l_pos = base_obs[lmk]["pos"][:2]
    tasks.append({
        "probe_id": "DISAMBIG_02_ego_front_vs_object_behind",
        "axis": "perspective_front_vs_behind",
        "capability": "spatial_reasoning",
        "difficulty": "disambiguation",
        "description": "Disambiguates moving ahead in camera frame vs reaching object-relative behind",
        "instruction": "go behind the yellow cylinder",
        "initial_pose": np.array([0.0, 0.0, float(np.arctan2(l_pos[1], l_pos[0]))], dtype=np.float32),
        "obstacles": base_obs,
        "ego_front_point": np.array([-0.6, -0.45, float(np.arctan2(l_pos[1], l_pos[0]))], dtype=np.float32),
        "object_behind_point": np.array([-1.6, -1.2, float(np.arctan2(l_pos[1], l_pos[0]))], dtype=np.float32),
        "waypoints": [np.array([-1.6, -1.2, float(np.arctan2(l_pos[1], l_pos[0]))], dtype=np.float32)],
        "semantic_relation": "behind"
    })

    # -------------------------------------------------------------
    # 3. Geometric proximity vs Visual proximity
    # A large object far away subtends the same visual angle as a small object close by.
    # Disambiguates whether neural latent encodes apparent pixel area or Euclidean range.
    # -------------------------------------------------------------
    obs_geo = copy.deepcopy(CANONICAL_OBSTACLES)
    # Double red box size and move it twice as far
    obs_geo["red_box"]["pos"] = np.array([2.0, 1.8, 0.35], dtype=np.float32)
    obs_geo["red_box"]["size"] = np.array([0.5, 0.5, 0.35], dtype=np.float32)
    tasks.append({
        "probe_id": "DISAMBIG_03_geometric_vs_visual_proximity",
        "axis": "scale_distance_invariance",
        "capability": "object_grounding",
        "difficulty": "disambiguation",
        "description": "Double size at double distance: identical visual footprint, 2x physical Euclidean distance",
        "instruction": "approach the red box",
        "initial_pose": np.array([0.0, 0.0, 0.73], dtype=np.float32),
        "obstacles": obs_geo,
        "visual_apparent_size_deg": 18.5,
        "euclidean_distance": 2.69,
        "waypoints": [np.array([1.5, 1.35, 0.73], dtype=np.float32)],
        "semantic_relation": "approach"
    })

    # -------------------------------------------------------------
    # 4. Target visibility vs Actual reachability
    # Target is visually visible in camera, but direct straight path is blocked
    # by a solid wall / obstacle gap narrower than AMR chassis width (0.26m).
    # -------------------------------------------------------------
    obs_reach = copy.deepcopy(CANONICAL_OBSTACLES)
    # Insert blocking wall between (0,0) and blue cylinder (-1.0, 1.2)
    obs_reach["slit_barrier"] = {
        "geom_name": "obs_barrier",
        "pos": np.array([-0.5, 0.6, 0.35], dtype=np.float32),
        "size": np.array([0.05, 0.7, 0.35], dtype=np.float32),
        "shape": "box",
        "color": "dark_gray",
        "rgba": np.array([0.3, 0.3, 0.3, 1.0], dtype=np.float32)
    }
    tasks.append({
        "probe_id": "DISAMBIG_04_visibility_vs_reachability",
        "axis": "kinematic_reachability",
        "capability": "obstacle_avoidance",
        "difficulty": "disambiguation",
        "description": "Blue cylinder is visible over barrier, but direct linear path is impassable (requires wide detour)",
        "instruction": "approach the blue cylinder",
        "initial_pose": np.array([0.0, 0.0, 2.2], dtype=np.float32),
        "obstacles": obs_reach,
        "is_direct_line_passable": False,
        "waypoints": [
            np.array([0.0, 1.4, 2.2], dtype=np.float32), # Detour around barrier
            np.array([-0.8, 1.2, 0.0], dtype=np.float32)
        ],
        "semantic_relation": "approach"
    })

    return tasks
