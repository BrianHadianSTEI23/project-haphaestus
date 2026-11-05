import re
import numpy as np

class SemanticGoalGrounder:
    """
    Parses natural language commands (e.g. "go behind the yellow cylinder")
    into semantic goals, ground-truth world-frame waypoints, and verifies
    goal completion via real-time camera visual feedback.
    """

    # Arena obstacles coordinates and visual color specifications
    LANDMARKS = {
        "yellow_cylinder": {
            "aliases": ["yellow cylinder", "yellow cyl", "yellow", "yellow box"],
            "pos": np.array([-1.2, -0.9]),
            "radius": 0.25,
            "hsv_range": ((20, 100, 100), (38, 255, 255)),  # OpenCV HSV: H in [0, 179]
            "rgb_preview": (245, 215, 25),
            "label": "Yellow Cylinder"
        },
        "red_box": {
            "aliases": ["red box", "red obstacle", "red", "red cube"],
            "pos": np.array([1.1, 0.9]),
            "radius": 0.35,
            "hsv_range": ((0, 120, 70), (10, 255, 255), (170, 120, 70), (180, 255, 255)),
            "rgb_preview": (220, 50, 50),
            "label": "Red Box"
        },
        "blue_cylinder": {
            "aliases": ["blue cylinder", "blue cyl", "blue"],
            "pos": np.array([-1.0, 1.2]),
            "radius": 0.25,
            "hsv_range": ((100, 120, 70), (130, 255, 255)),
            "rgb_preview": (50, 100, 220),
            "label": "Blue Cylinder"
        },
        "green_cylinder": {
            "aliases": ["green cylinder", "green cyl", "green"],
            "pos": np.array([1.2, -1.1]),
            "radius": 0.22,
            "hsv_range": ((40, 100, 70), (80, 255, 255)),
            "rgb_preview": (50, 200, 75),
            "label": "Green Cylinder"
        },
        "purple_box": {
            "aliases": ["purple box", "purple cube", "purple"],
            "pos": np.array([0.0, 1.7]),
            "radius": 0.35,
            "hsv_range": ((130, 100, 70), (165, 255, 255)),
            "rgb_preview": (155, 50, 190),
            "label": "Purple Box"
        },
        "cyan_cylinder": {
            "aliases": ["cyan cylinder", "cyan cyl", "cyan"],
            "pos": np.array([-0.2, -1.7]),
            "radius": 0.25,
            "hsv_range": ((82, 120, 70), (98, 255, 255)),
            "rgb_preview": (25, 210, 210),
            "label": "Cyan Cylinder"
        }
    }

    RELATIONS = {
        "behind": ["behind", "rear", "back of", "after"],
        "in_front": ["in front of", "front of", "ahead of", "before"],
        "left_of": ["left of", "to the left of", "leftside"],
        "right_of": ["right of", "to the right of", "rightside"],
        "approach": ["go to", "navigate to", "approach", "reach", "near", "towards", "at"]
    }

    def __init__(self):
        self.landmark_keys = list(self.LANDMARKS.keys())

    def parse_command(self, text: str, robot_pos=np.array([0.0, 0.0])):
        """
        Parses text command (e.g. "go behind the yellow cylinder") into:
          - landmark_key: name of target
          - relation: 'behind', 'in_front', 'left_of', 'right_of', 'approach'
          - target_waypoint: (x_goal, y_goal, theta_goal)
        """
        text_lower = text.lower().strip()

        # 1. Detect target landmark
        matched_landmark = None
        for lmk_key, lmk_info in self.LANDMARKS.items():
            for alias in lmk_info["aliases"]:
                if alias in text_lower:
                    matched_landmark = lmk_key
                    break
            if matched_landmark:
                break

        if not matched_landmark:
            # Default to yellow cylinder if ambiguous or not explicitly named
            matched_landmark = "yellow_cylinder"

        # 2. Detect spatial relation
        matched_relation = "approach"
        for rel_key, keywords in self.RELATIONS.items():
            for kw in keywords:
                if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                    matched_relation = rel_key
                    break
            if matched_relation != "approach":
                break

        # 3. Calculate target waypoint based on landmark position and spatial relation
        lmk_pos = self.LANDMARKS[matched_landmark]["pos"]
        lmk_rad = self.LANDMARKS[matched_landmark]["radius"]
        offset_dist = lmk_rad + 0.38  # Safe clearance distance

        # Arena center vector
        center_vec = lmk_pos - np.array([0.0, 0.0])
        dist_c = np.linalg.norm(center_vec)
        radial_dir = center_vec / (dist_c + 1e-6)  # Points outwards from center
        tangent_dir = np.array([-radial_dir[1], radial_dir[0]]) # Perpendicular

        # Determine if clockwise or counter-clockwise flank is closer to robot
        flank_p1 = lmk_pos + tangent_dir * 0.70
        flank_p2 = lmk_pos - tangent_dir * 0.70
        d1 = np.linalg.norm(flank_p1 - robot_pos[:2])
        d2 = np.linalg.norm(flank_p2 - robot_pos[:2])
        best_flank = flank_p1 if d1 < d2 else flank_p2

        if matched_relation == "behind":
            # Target is behind the obstacle facing outward
            target_xy = lmk_pos + radial_dir * offset_dist
        elif matched_relation == "in_front":
            target_xy = lmk_pos - radial_dir * offset_dist
        elif matched_relation == "left_of":
            target_xy = lmk_pos + tangent_dir * offset_dist
        elif matched_relation == "right_of":
            target_xy = lmk_pos - tangent_dir * offset_dist
        else: # "approach" / "near"
            robot_to_lmk = lmk_pos - robot_pos[:2]
            r_dist = np.linalg.norm(robot_to_lmk)
            approach_dir = robot_to_lmk / (r_dist + 1e-6)
            target_xy = lmk_pos - approach_dir * offset_dist

        # Clamp target inside arena safety boundary [-2.1, 2.1]
        target_xy = np.clip(target_xy, -2.1, 2.1)
        best_flank = np.clip(best_flank, -2.1, 2.1)

        # Desired robot heading faces towards the target landmark
        facing_vec = lmk_pos - target_xy
        target_theta = float(np.arctan2(facing_vec[1], facing_vec[0]))

        goal_dict = {
            "raw_text": text,
            "landmark": matched_landmark,
            "landmark_label": self.LANDMARKS[matched_landmark]["label"],
            "landmark_pos": lmk_pos,
            "relation": matched_relation,
            "target_xy": target_xy,
            "flank_xy": best_flank,
            "has_flank": (matched_relation == "behind"),
            "target_theta": target_theta,
            "target_pose": np.array([target_xy[0], target_xy[1], target_theta], dtype=np.float32)
        }
        return goal_dict

    def verify_goal_with_camera(self, obs_rgb, robot_pose, goal_dict, dist_thresh=0.45):
        """
        Checks if the robot has reached the waypoint AND acquired the target landmark in its camera feedback.
        obs_rgb: (H, W, 3) uint8 image from Astra camera.
        robot_pose: [x, y, theta]
        goal_dict: output of parse_command
        """
        # 1. Spatial Euclidean Distance to target waypoint
        xy_dist = np.linalg.norm(robot_pose[:2] - goal_dict["target_xy"])
        heading_err = abs(np.arctan2(
            np.sin(robot_pose[2] - goal_dict["target_theta"]),
            np.cos(robot_pose[2] - goal_dict["target_theta"])
        ))

        spatial_reached = (xy_dist <= dist_thresh)

        # 2. Visual Detection via Color Segmentation
        lmk_key = goal_dict["landmark"]
        lmk_info = self.LANDMARKS[lmk_key]

        # Convert RGB -> HSV (custom fast numpy implementation)
        r = obs_rgb[..., 0].astype(np.float32) / 255.0
        g = obs_rgb[..., 1].astype(np.float32) / 255.0
        b = obs_rgb[..., 2].astype(np.float32) / 255.0

        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin + 1e-6

        # Hue [0, 180)
        h = np.zeros_like(r)
        mask_r = (cmax == r)
        mask_g = (cmax == g) & ~mask_r
        mask_b = (cmax == b) & ~mask_r & ~mask_g

        h[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 360) / 2.0
        h[mask_g] = (60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120.0) / 2.0
        h[mask_b] = (60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240.0) / 2.0

        # Saturation [0, 255]
        s = np.zeros_like(r)
        s[cmax > 0] = (delta[cmax > 0] / cmax[cmax > 0]) * 255.0

        # Value [0, 255]
        v = cmax * 255.0

        hsv = np.stack([h, s, v], axis=-1)

        # Apply HSV thresholding
        hsv_range = lmk_info["hsv_range"]
        if len(hsv_range) == 2:
            low, high = hsv_range
            color_mask = (
                (hsv[..., 0] >= low[0]) & (hsv[..., 0] <= high[0]) &
                (hsv[..., 1] >= low[1]) & (hsv[..., 1] <= high[1]) &
                (hsv[..., 2] >= low[2]) & (hsv[..., 2] <= high[2])
            )
        else: # Double range for red wrap-around
            l1, h1, l2, h2 = hsv_range
            m1 = (
                (hsv[..., 0] >= l1[0]) & (hsv[..., 0] <= h1[0]) &
                (hsv[..., 1] >= l1[1]) & (hsv[..., 1] <= h1[1]) &
                (hsv[..., 2] >= l1[2]) & (hsv[..., 2] <= h1[2])
            )
            m2 = (
                (hsv[..., 0] >= l2[0]) & (hsv[..., 0] <= h2[0]) &
                (hsv[..., 1] >= l2[1]) & (hsv[..., 1] <= h2[1]) &
                (hsv[..., 2] >= l2[2]) & (hsv[..., 2] <= h2[2])
            )
            color_mask = m1 | m2

        pixel_count = int(np.sum(color_mask))
        total_pixels = obs_rgb.shape[0] * obs_rgb.shape[1]
        visual_fraction = pixel_count / total_pixels

        # Centroid of detection in camera FOV
        if pixel_count > 10:
            ys, xs = np.where(color_mask)
            u_center = float(np.mean(xs))
            v_center = float(np.mean(ys))
            # Normalized centering error in [-1, 1]
            u_err = (u_center - (obs_rgb.shape[1] / 2.0)) / (obs_rgb.shape[1] / 2.0)
            visual_acquired = (visual_fraction >= 0.015) and (abs(u_err) < 0.8)
        else:
            u_center = obs_rgb.shape[1] / 2.0
            v_center = obs_rgb.shape[0] / 2.0
            u_err = 1.0
            visual_acquired = False

        # Goal is successfully verified when within target distance envelope and visually confirmed
        goal_verified = spatial_reached and (visual_acquired or xy_dist < 0.25)

        feedback = {
            "spatial_reached": bool(spatial_reached),
            "visual_acquired": bool(visual_acquired),
            "goal_verified": bool(goal_verified),
            "xy_distance": float(xy_dist),
            "heading_error": float(heading_err),
            "visual_fraction": float(visual_fraction),
            "u_center": float(u_center),
            "v_center": float(v_center),
            "u_err": float(u_err),
            "pixel_count": pixel_count
        }
        return feedback
