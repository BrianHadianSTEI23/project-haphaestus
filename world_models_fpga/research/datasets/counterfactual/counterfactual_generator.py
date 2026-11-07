"""
Phase 1: Controlled Intelligence Benchmark - Counterfactual Generator (E1 - E7)
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Generates matched counterfactual pairs (Baseline E0 vs Counterfactual Ei)
where exactly one physical or semantic variable is mutated to test causal invariance:
  - E1: Position Shift (target landmark relocates to another quadrant)
  - E2: Appearance Mutation (target changes color / reflectance)
  - E3: Robot Heading Rotation (robot starts rotated by 90 deg)
  - E4: Robot Starting Position (robot starts in opposite quadrant)
  - E5: Obstacle Insertion (intervening obstacle placed on line of sight)
  - E6: Geometric Shape Mutation (cylinder swapped with box)
  - E7: Camera Elevation / Viewpoint Mutation (camera pitch altered by 10 deg)
"""

import copy
import numpy as np

def generate_counterfactual_pair(experiment_type: str, pair_id: int):
    """
    Returns a tuple: (baseline_task_spec, counterfactual_task_spec)
    where both share the identical instruction, seed, and reference frames,
    differing strictly along the single manipulated axis.
    """
    generation_seed = 1000 + pair_id
    rng = np.random.default_rng(generation_seed)

    # Base canonical obstacles
    from research.datasets.generation.capability_tasks import CANONICAL_OBSTACLES, get_waypoint_for_relation
    base_obs = copy.deepcopy(CANONICAL_OBSTACLES)

    target_key = "yellow_cylinder"
    target_pos = base_obs[target_key]["pos"][:2].copy()
    relation = "behind"
    instruction = "go behind the yellow cylinder"

    # Pair-specific deterministic initial pose perturbation (pair_id == 0 is exact origin reference)
    if pair_id == 0:
        rx, ry, yaw_jitter = 0.0, 0.0, 0.0
    else:
        rx = float(rng.uniform(-0.22, 0.22))
        ry = float(rng.uniform(-0.22, 0.22))
        yaw_jitter = float(rng.uniform(-0.15, 0.15))

    base_heading = float(np.arctan2(target_pos[1] - ry, target_pos[0] - rx) + yaw_jitter)

    # Base specification (E0)
    e0_spec = {
        "counterfactual_id": f"{experiment_type}_pair_{pair_id:03d}_E0",
        "experiment_type": experiment_type,
        "pair_id": int(pair_id),
        "capability": "spatial_reasoning",
        "difficulty": "counterfactual",
        "is_counterfactual": False,
        "pair_role": "baseline",
        "instruction": instruction,
        "semantic_relation": relation,
        "target_key": target_key,
        "obstacles": copy.deepcopy(base_obs),
        "initial_pose": np.array([rx, ry, base_heading], dtype=np.float32),
        "camera_config": "front_cam",
        "camera_pitch_offset": 0.0,
        "generation_seed": generation_seed,
        "intervention": {"name": "none", "parameters": {}},
        "expected_invariant": "reference_baseline"
    }
    # Waypoints for E0
    g_pose = get_waypoint_for_relation(base_obs[target_key]["pos"], relation, e0_spec["initial_pose"])
    c_vec = base_obs[target_key]["pos"][:2] - e0_spec["initial_pose"][:2]
    rad = c_vec / (np.linalg.norm(c_vec) + 1e-8)
    tan = np.array([-rad[1], rad[0]], dtype=np.float32)
    flank = base_obs[target_key]["pos"][:2] + tan * 0.70
    e0_spec["waypoints"] = [np.array([flank[0], flank[1], g_pose[2]], dtype=np.float32), g_pose]

    # Counterfactual specification (Ei)
    ei_spec = copy.deepcopy(e0_spec)
    ei_spec["counterfactual_id"] = f"{experiment_type}_pair_{pair_id:03d}_Ei"
    ei_spec["is_counterfactual"] = True
    ei_spec["pair_role"] = "mutated"

    # -------------------------------------------------------------
    # E1: Target moves to another position
    # -------------------------------------------------------------
    if experiment_type == "E1_position_shift":
        ei_spec["expected_invariant"] = "spatial_relation_meaning_invariant_to_position"
        new_pos = np.array([0.8, -1.2, 0.35], dtype=np.float32)
        ei_spec["obstacles"][target_key]["pos"] = new_pos
        ei_spec["intervention"] = {"name": "target_position", "parameters": {"position": new_pos.tolist()}}
        # Recompute waypoints with mutated position
        g_pose_mut = get_waypoint_for_relation(new_pos, relation, ei_spec["initial_pose"])
        c_vec_mut = new_pos[:2] - ei_spec["initial_pose"][:2]
        rad_mut = c_vec_mut / (np.linalg.norm(c_vec_mut) + 1e-8)
        tan_mut = np.array([-rad_mut[1], rad_mut[0]], dtype=np.float32)
        flank_mut = new_pos[:2] + tan_mut * 0.70
        ei_spec["waypoints"] = [np.array([flank_mut[0], flank_mut[1], g_pose_mut[2]], dtype=np.float32), g_pose_mut]

    # -------------------------------------------------------------
    # E2: Target changes color (yellow cylinder rendered red or cyan)
    # -------------------------------------------------------------
    elif experiment_type == "E2_color_shift":
        ei_spec["expected_invariant"] = "spatial_relation_invariant_to_color"
        # Turn yellow cylinder into bright blue / red in RGBA
        ei_spec["obstacles"][target_key]["rgba"] = np.array([0.15, 0.35, 0.85, 1.0], dtype=np.float32)
        ei_spec["obstacles"][target_key]["color"] = "mutated_blue"
        ei_spec["intervention"] = {"name": "target_rgba", "parameters": {"rgba": [0.15, 0.35, 0.85, 1.0]}}

    # -------------------------------------------------------------
    # E3: Robot rotates 90 deg initial heading
    # -------------------------------------------------------------
    elif experiment_type == "E3_robot_rotation":
        ei_spec["expected_invariant"] = "allocentric_goal_interpretation_invariant_to_heading"
        rotated_heading = float(e0_spec["initial_pose"][2] + np.pi / 2.0)
        ei_spec["initial_pose"][2] = rotated_heading
        ei_spec["intervention"] = {"name": "robot_yaw", "parameters": {"yaw_radians": rotated_heading}}

    # -------------------------------------------------------------
    # E4: Robot changes starting position to opposite quadrant
    # -------------------------------------------------------------
    elif experiment_type == "E4_robot_position":
        ei_spec["expected_invariant"] = "object_relative_goal_invariant_to_start_pose"
        ei_spec["initial_pose"] = np.array([1.0 + 0.5 * rx, 1.0 + 0.5 * ry, -2.35 + yaw_jitter], dtype=np.float32)
        ei_spec["intervention"] = {"name": "robot_initial_pose", "parameters": {"pose": ei_spec["initial_pose"].tolist()}}

    # -------------------------------------------------------------
    # E5: Add intermediate obstacle directly between robot and target
    # -------------------------------------------------------------
    elif experiment_type == "E5_intermediate_obstacle":
        ei_spec["expected_invariant"] = "goal_identity_invariant_under_path_detour"
        # obs_inserted is a reserved collidable MuJoCo geometry in amr_arena.xml.
        # The generator moves it into the arena via JetAutoEnv.apply_task_spec.
        mid_xy = 0.5 * (e0_spec["initial_pose"][:2] + target_pos[:2])
        obs_pos = np.array([float(mid_xy[0]), float(mid_xy[1]), 0.35], dtype=np.float32)
        ei_spec["obstacles"]["inserted_obstacle"] = {
            "geom_name": "obs_inserted",
            "pos": obs_pos,
            "radius": 0.22,
            "shape": "cylinder",
            "color": "magenta",
            "rgba": np.array([0.8, 0.1, 0.8, 1.0], dtype=np.float32)
        }
        ei_spec["intervention"] = {"name": "inserted_collidable_obstacle", "parameters": {"geom_name": "obs_inserted", "position": obs_pos.tolist(), "shape": "cylinder"}}
        # Add extra avoidance detour waypoint
        detour = np.array([float(mid_xy[0] + 0.3), float(mid_xy[1] - 0.45), e0_spec["waypoints"][0][2]], dtype=np.float32)
        ei_spec["waypoints"] = [detour] + e0_spec["waypoints"]

    # -------------------------------------------------------------
    # E6: Change cylinder to another shape (cylinder -> box)
    # -------------------------------------------------------------
    elif experiment_type == "E6_shape_mutation":
        ei_spec["expected_invariant"] = "relational_reasoning_invariant_to_shape"
        ei_spec["obstacles"][target_key]["shape"] = "box"
        ei_spec["obstacles"][target_key]["size"] = np.array([0.25, 0.25, 0.35], dtype=np.float32)
        ei_spec["intervention"] = {"name": "target_geom_type", "parameters": {"shape": "box", "size": [0.25, 0.25, 0.35]}}

    # -------------------------------------------------------------
    # E7: Change camera viewpoint / pitch
    # -------------------------------------------------------------
    elif experiment_type == "E7_camera_viewpoint":
        ei_spec["expected_invariant"] = "spatial_grounding_invariant_to_camera_pitch"
        ei_spec["camera_pitch_offset"] = 10.0 # +10 deg pitch down
        ei_spec["intervention"] = {"name": "front_camera_pitch", "parameters": {"degrees": 10.0}}

    return e0_spec, ei_spec
