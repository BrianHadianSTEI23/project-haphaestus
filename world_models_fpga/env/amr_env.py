import os
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

    def _check_collision(self):
        """Returns True if amr_chassis contacts any obstacle or arena wall."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if g1 == self.robot_geom_id or g2 == self.robot_geom_id:
                other_geom = g2 if g1 == self.robot_geom_id else g1
                if other_geom != self.floor_geom_id:
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
                rx = np.random.uniform(-1.5, 1.5)
                ry = np.random.uniform(-1.5, 1.5)
                r_theta = np.random.uniform(-np.pi, np.pi)

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
