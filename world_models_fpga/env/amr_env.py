import copy
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

class JetAutoEnv(gym.Env):
    """
    Gymnasium environment for Hiwonder JetAuto Mecanum AMR in MuJoCo.
    Supports both first-person onboard Astra camera and 3D third-person arena rendering.
    """
    metadata = {"render_modes": ["rgb_array", "human"]}

    # Known obstacle ground-truth configurations in the arena
    OBSTACLES = {
        "yellow_cylinder": {"geom": "obs_yellow_cyl", "pos": np.array([-1.2, -0.9, 0.35]), "radius": 0.25, "height": 0.35, "color": "yellow", "rgb": (242, 217, 25)},
        "red_box":         {"geom": "obs_red_box",    "pos": np.array([1.1, 0.9, 0.35]),   "size": (0.25, 0.25), "height": 0.35, "color": "red",    "rgb": (217, 51, 51)},
        "blue_cylinder":   {"geom": "obs_blue_cyl",   "pos": np.array([-1.0, 1.2, 0.35]),  "radius": 0.25, "height": 0.35, "color": "blue",   "rgb": (51, 102, 217)},
        "green_cylinder":  {"geom": "obs_green_cyl",  "pos": np.array([1.2, -1.1, 0.35]),  "radius": 0.22, "height": 0.35, "color": "green",  "rgb": (51, 191, 76)},
        "purple_box":      {"geom": "obs_purple_box", "pos": np.array([0.0, 1.7, 0.35]),   "size": (0.3, 0.2),   "height": 0.35, "color": "purple", "rgb": (153, 51, 191)},
        "cyan_cylinder":   {"geom": "obs_cyan_cyl",   "pos": np.array([-0.2, -1.7, 0.35]), "radius": 0.25, "height": 0.35, "color": "cyan",   "rgb": (25, 204, 204)},
    }

    def __init__(self, xml_path=None, obs_size=(64, 64), view_size=(360, 480), frame_skip=5):
        super().__init__()
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "amr_arena.xml")

        self.xml_path = xml_path
        self.obs_size = obs_size
        self.view_size = view_size
        self.frame_skip = frame_skip  # dt=0.02 * 5 = 0.1s control step (10 Hz)

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Offscreen renderers for first-person Astra camera and 3D arena camera
        self.fp_renderer = mujoco.Renderer(self.model, height=self.obs_size[0], width=self.obs_size[1])
        self.tp_renderer = mujoco.Renderer(self.model, height=self.view_size[0], width=self.view_size[1])

        # Action space: normalized [vx, vy, wz] in [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space: 64x64x3 RGB
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.obs_size[0], self.obs_size[1], 3), dtype=np.uint8
        )

        # Kinematic limits (JetAuto Mecanum chassis)
        self.vx_limit = 0.6  # m/s forward/reverse
        self.vy_limit = 0.4  # m/s lateral strafe
        self.wz_limit = 1.5  # rad/s yaw rate

        # Cache IDs
        self.robot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "amr_chassis")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.jx_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slide_x")
        self.jy_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slide_y")
        self.jz_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge_z")
        self.front_camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_cam"
        )
        # Runtime mutations alter MjModel arrays.  Keep an exact baseline so each
        # episode starts from the same physical simulator, not from the preceding
        # counterfactual.
        self._baseline_geom_pos = self.model.geom_pos.copy()
        self._baseline_geom_size = self.model.geom_size.copy()
        self._baseline_geom_rgba = self.model.geom_rgba.copy()
        self._baseline_geom_type = self.model.geom_type.copy()
        self._baseline_geom_matid = self.model.geom_matid.copy()
        self._baseline_mat_rgba = self.model.mat_rgba.copy() if self.model.nmat > 0 else None
        self._baseline_cam_quat = self.model.cam_quat.copy()
        self._baseline_geom_friction = self.model.geom_friction.copy()
        self._baseline_light_diffuse = self.model.light_diffuse.copy() if self.model.nlight > 0 else None
        self._baseline_light_pos = self.model.light_pos.copy() if self.model.nlight > 0 else None
        self.active_task_spec = None
        mujoco.mj_forward(self.model, self.data)

    def geometry_id(self, geom_name):
        """Return a checked geometry id; fail early on misspelled mutations."""
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise KeyError(f"Unknown MuJoCo geometry: {geom_name}")
        return geom_id

    def restore_physical_baseline(self):
        """Undo every runtime counterfactual mutation in the MuJoCo model."""
        self.model.geom_pos[:] = self._baseline_geom_pos
        self.model.geom_size[:] = self._baseline_geom_size
        self.model.geom_rgba[:] = self._baseline_geom_rgba
        self.model.geom_type[:] = self._baseline_geom_type
        self.model.geom_matid[:] = self._baseline_geom_matid
        if self._baseline_mat_rgba is not None:
            self.model.mat_rgba[:] = self._baseline_mat_rgba
        self.model.geom_friction[:] = self._baseline_geom_friction
        self.model.cam_quat[:] = self._baseline_cam_quat
        if self._baseline_light_diffuse is not None:
            self.model.light_diffuse[:] = self._baseline_light_diffuse
        if self._baseline_light_pos is not None:
            self.model.light_pos[:] = self._baseline_light_pos
        self.active_task_spec = None
        mujoco.mj_forward(self.model, self.data)

    def apply_task_spec(self, task_spec):
        """Apply a task specification to real MuJoCo state and rendering state.

        This is deliberately the sole mutation path used by dataset generation and
        benchmarking.  It supports E1--E7 with physical geometry, collision shape,
        appearance, and camera changes rather than metadata-only changes.
        """
        self.restore_physical_baseline()
        obstacles = task_spec.get("obstacles", {})
        for key, obstacle in obstacles.items():
            geom_name = obstacle.get("geom_name")
            if not geom_name:
                raise ValueError(f"Obstacle {key!r} has no geom_name")
            geom_id = self.geometry_id(geom_name)
            self.model.geom_pos[geom_id] = np.asarray(obstacle["pos"], dtype=np.float64)
            if "rgba" in obstacle:
                self.model.geom_matid[geom_id] = -1
                self.model.geom_rgba[geom_id] = np.asarray(obstacle["rgba"], dtype=np.float32)

            shape = obstacle.get("shape")
            if shape == "box":
                self.model.geom_type[geom_id] = mujoco.mjtGeom.mjGEOM_BOX
                size = np.asarray(obstacle.get("size", [0.25, 0.25, 0.35]), dtype=np.float64)
                self.model.geom_size[geom_id, :3] = size[:3]
            elif shape == "cylinder":
                self.model.geom_type[geom_id] = mujoco.mjtGeom.mjGEOM_CYLINDER
                self.model.geom_size[geom_id, 0] = float(obstacle.get("radius", 0.25))
                self.model.geom_size[geom_id, 1] = float(obstacle.get("height", 0.35))

        # Explicit floor appearance mutation (nuisance intervention)
        if "floor_rgba" in task_spec:
            self.model.geom_matid[self.floor_geom_id] = -1
            self.model.geom_rgba[self.floor_geom_id] = np.asarray(task_spec["floor_rgba"], dtype=np.float32)

        # Explicit color mutations mapping object/geom keys to RGBA
        color_mutations = task_spec.get("color_mutations", {})
        for obj_or_geom_key, rgba in color_mutations.items():
            geom_name = None
            if obj_or_geom_key in obstacles and "geom_name" in obstacles[obj_or_geom_key]:
                geom_name = obstacles[obj_or_geom_key]["geom_name"]
            elif obj_or_geom_key in obstacles:
                geom_name = obstacles[obj_or_geom_key].get("geom_name")
            else:
                geom_name = obj_or_geom_key
            try:
                g_id = self.geometry_id(geom_name)
                self.model.geom_matid[g_id] = -1
                self.model.geom_rgba[g_id] = np.asarray(rgba, dtype=np.float32)
            except KeyError:
                pass

        # Explicit lighting / illumination mutation
        if "light_diffuse_scale" in task_spec and self._baseline_light_diffuse is not None:
            scale = float(task_spec["light_diffuse_scale"])
            self.model.light_diffuse[:] = self._baseline_light_diffuse * scale
        if "light_diffuse" in task_spec and self._baseline_light_diffuse is not None:
            self.model.light_diffuse[:] = np.asarray(task_spec["light_diffuse"], dtype=np.float32)

        pitch_degrees = float(task_spec.get("camera_pitch_offset", 0.0))
        if pitch_degrees:
            pitch_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_euler2Quat(pitch_quat, np.array([0.0, np.deg2rad(pitch_degrees), 0.0]), "XYZ")
            adjusted = np.empty(4, dtype=np.float64)
            # Post-multiplication applies the pitch in the camera's local frame.
            mujoco.mju_mulQuat(adjusted, self._baseline_cam_quat[self.front_camera_id], pitch_quat)
            self.model.cam_quat[self.front_camera_id] = adjusted

        # Optional embodiment intervention used only by explicit tests.
        friction_scale = float(task_spec.get("floor_friction_scale", 1.0))
        if friction_scale <= 0:
            raise ValueError("floor_friction_scale must be positive")
        self.model.geom_friction[self.floor_geom_id] *= friction_scale
        self.active_task_spec = copy.deepcopy(task_spec)
        mujoco.mj_forward(self.model, self.data)

    def _check_collision(self):
        """Returns True if amr_chassis intersects an obstacle or arena wall.

        Velocity actuators can make MuJoCo's contact solver omit a transient
        contact manifold at reset.  We therefore retain contact inspection and
        add an exact signed geometry-distance fallback.  The fallback still uses
        the simulator's collision geometries, not task metadata.
        """
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if g1 == self.robot_geom_id or g2 == self.robot_geom_id:
                other_geom = g2 if g1 == self.robot_geom_id else g1
                if other_geom != self.floor_geom_id:
                    return True
        robot_contype = self.model.geom_contype[self.robot_geom_id]
        robot_affinity = self.model.geom_conaffinity[self.robot_geom_id]
        robot_body_id = self.model.geom_bodyid[self.robot_geom_id]
        for geom_id in range(self.model.ngeom):
            if geom_id in (self.robot_geom_id, self.floor_geom_id):
                continue
            if self.model.geom_bodyid[geom_id] == robot_body_id:
                continue
            if not ((robot_contype & self.model.geom_conaffinity[geom_id]) or (self.model.geom_contype[geom_id] & robot_affinity)):
                continue
            if mujoco.mj_geomDistance(self.model, self.data, self.robot_geom_id, geom_id, 10.0, None) <= 0.0:
                return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Custom initial pose if provided in options
        if options and "initial_pos" in options:
            rx, ry, r_theta = options["initial_pos"]
        else:
            # Randomize initial robot pose within safe central clearance
            for _ in range(100):
                rx = self.np_random.uniform(-1.5, 1.5)
                ry = self.np_random.uniform(-1.5, 1.5)
                r_theta = self.np_random.uniform(-np.pi, np.pi)

                self.data.qpos[self.jx_id] = rx
                self.data.qpos[self.jy_id] = ry
                self.data.qpos[self.jz_id] = r_theta
                self.data.qvel[:] = 0.0

                mujoco.mj_forward(self.model, self.data)
                if not self._check_collision():
                    break

        self.data.qpos[self.jx_id] = rx
        self.data.qpos[self.jy_id] = ry
        self.data.qpos[self.jz_id] = r_theta
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        obs = self.render_observation()
        info = {
            "collision": False,
            "pos": np.array([rx, ry, r_theta], dtype=np.float32),
            "step_count": 0
        }
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Scale normalized actions to body-frame velocity commands
        vx_cmd = float(action[0] * self.vx_limit)
        vy_cmd = float(action[1] * self.vy_limit)
        wz_cmd = float(action[2] * self.wz_limit)

        # Transform body-frame velocity to world-frame using robot heading
        theta = float(self.data.qpos[self.jz_id])
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        vx_world = cos_t * vx_cmd - sin_t * vy_cmd
        vy_world = sin_t * vx_cmd + cos_t * vy_cmd

        self.data.ctrl[0] = vx_world
        self.data.ctrl[1] = vy_world
        self.data.ctrl[2] = wz_cmd

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self.render_observation()
        is_collision = self._check_collision()

        pos = np.array([
            float(self.data.qpos[self.jx_id]),
            float(self.data.qpos[self.jy_id]),
            float(self.data.qpos[self.jz_id])
        ], dtype=np.float32)

        info = {
            "collision": is_collision,
            "action": action.copy(),
            "action_scaled": np.array([vx_cmd, vy_cmd, wz_cmd], dtype=np.float32),
            "pos": pos,
        }

        reward = -1.0 if is_collision else 0.0
        terminated = False
        truncated = False

        return obs, reward, terminated, truncated, info

    def step_physics(self, action):
        """Pure physics step without offscreen camera rendering, for fast MPC candidate evaluation."""
        action = np.clip(action, -1.0, 1.0)
        vx_cmd = float(action[0] * self.vx_limit)
        vy_cmd = float(action[1] * self.vy_limit)
        wz_cmd = float(action[2] * self.wz_limit)
        theta = float(self.data.qpos[self.jz_id])
        vx_world = np.cos(theta) * vx_cmd - np.sin(theta) * vy_cmd
        vy_world = np.sin(theta) * vx_cmd + np.cos(theta) * vy_cmd
        self.data.ctrl[0] = vx_world
        self.data.ctrl[1] = vy_world
        self.data.ctrl[2] = wz_cmd
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        return self._check_collision()

    def render_observation(self):
        """Renders first-person onboard Astra camera view (64x64x3)."""
        self.fp_renderer.update_scene(self.data, camera="front_cam")
        return self.fp_renderer.render()

    def render_3d_view(self, camera="arena_cam"):
        """Renders high-resolution 3D view for external viewer/dashboard."""
        self.tp_renderer.update_scene(self.data, camera=camera)
        return self.tp_renderer.render()

    def get_robot_pose(self):
        return np.array([
            float(self.data.qpos[self.jx_id]),
            float(self.data.qpos[self.jy_id]),
            float(self.data.qpos[self.jz_id])
        ], dtype=np.float32)

    def close(self):
        self.fp_renderer.close()
        self.tp_renderer.close()
