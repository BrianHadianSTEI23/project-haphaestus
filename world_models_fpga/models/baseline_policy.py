import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

TARGET_KEYS = (
    "red_box",
    "blue_cylinder",
    "green_cylinder",
    "yellow_cylinder",
    "purple_box",
    "cyan_cylinder",
)

SEMANTIC_RELATIONS = (
    "approach",
    "behind",
    "in_front",
    "left_of",
    "right_of",
    "multi_waypoint",
    "action_probe",
    "none",
)


def encode_task_id(target_key: str = "", semantic_relation: str = "approach", instruction: str = "") -> int:
    """Deterministic discrete task index in [0, 63] from target identity and spatial relation."""
    t_idx = len(TARGET_KEYS)
    for i, key in enumerate(TARGET_KEYS):
        label = key.replace("_", " ")
        if key == target_key or (not target_key and label in instruction.lower()):
            t_idx = i
            break
    r_idx = 0
    for j, rel in enumerate(SEMANTIC_RELATIONS):
        if rel == semantic_relation:
            r_idx = j
            break
    return int(min(63, t_idx * 8 + r_idx))


class VisualFeatureEncoder(nn.Module):
    """
    Compact convolutional encoder mapping 64x64x3 RGB images to a 32-dim visual state vector.
    Architecture optimized for high-speed edge inference and signed Q8.8 fixed-point quantization.
    """
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # 64x64x3 -> 32x32x16 -> 16x16x32 -> 8x8x64 -> 4x4x64
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)

        self.fc = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.flatten(start_dim=1)
        out = self.fc(h)
        return out


class BaselineNavigationPolicy(nn.Module):
    """
    Language-Goal Conditioned Reactive Navigation Policy with Obstacle Avoidance.
    Supports:
      - Track B (structured goal cue): Input s_t (32-dim) + Relative Goal Vector g_t (4-dim: dx, dy, sin_dth, cos_dth)
      - Track C (observation + discrete task ID only): Infers g_t from s_t + task_embedding(task_id) via grounding_head.
    Output: 3-DOF Holonomic Action [vx, vy, wz] in [-1, 1]^3 and Collision Risk Probability c_t in [0, 1].
    """
    def __init__(self, latent_dim=32, goal_dim=4, action_dim=3, hidden_dim=64, num_tasks=64, task_emb_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.encoder = VisualFeatureEncoder(latent_dim=latent_dim)

        in_dim = latent_dim + goal_dim

        # Core Policy MLP (Dense layers quantizable to Q8.8 FPGA DSP blocks)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Action head: predicts continuous [vx, vy, wz]
        self.action_head = nn.Linear(hidden_dim, action_dim)

        # Collision risk head: predicts obstacle proximity / contact risk
        self.collision_head = nn.Linear(hidden_dim, 1)

        # Track C visual task grounding head: predicts body-frame goal vector [dx_b, dy_b, sin_dth, cos_dth]
        # from visual latent state z_vis and discrete task ID (no simulator coordinates required).
        self.task_embedding = nn.Embedding(num_tasks, task_emb_dim)
        self.grounding_head = nn.Sequential(
            nn.Linear(latent_dim + task_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),
        )

    def infer_goal_vec(self, z_vis: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Predicts body-frame goal vector [dx_b, dy_b, sin(dth), cos(dth)] from visual latent + task ID."""
        t_emb = self.task_embedding(task_ids)
        raw = self.grounding_head(torch.cat([z_vis, t_emb], dim=-1))
        dx_dy = torch.clamp(raw[:, :2], -3.5, 3.5)
        trig = F.normalize(raw[:, 2:4] + 1e-6, dim=-1)
        return torch.cat([dx_dy, trig], dim=-1)

    def forward(self, img_tensor, goal_vec=None, task_ids=None):
        """
        img_tensor: (B, 3, 64, 64) normalized in [0, 1]
        goal_vec: (B, 4) [dx_rel, dy_rel, sin(dth), cos(dth)] (if None, inferred via task_ids for Track C)
        """
        z_vis = self.encoder(img_tensor)
        if goal_vec is None:
            if task_ids is None:
                task_ids = torch.zeros(img_tensor.shape[0], dtype=torch.long, device=img_tensor.device)
            goal_vec = self.infer_goal_vec(z_vis, task_ids)
        return self.forward_with_latent(z_vis, goal_vec), z_vis

    def forward_with_latent(self, z_vis, goal_vec):
        """Forward pass directly from latent features (matches FPGA accelerator input)."""
        x = torch.cat([z_vis, goal_vec], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        actions = torch.tanh(self.action_head(h))
        col_risk = torch.sigmoid(self.collision_head(h))
        return actions, col_risk

    @torch.no_grad()
    def act(self, obs_rgb, robot_pose=None, target_pose=None, goal_vec_body=None, task_id: int = 0, device="cpu"):
        """
        Computes 3-DOF action in either Track B (goal_vec_body or relative pose) or Track C (visual + task_id only).
        """
        self.eval()
        img_t = torch.from_numpy(obs_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        z_vis = self.encoder(img_t)

        if goal_vec_body is not None:
            goal_vec = torch.as_tensor(goal_vec_body, dtype=torch.float32, device=device).view(1, 4)
        elif robot_pose is not None and target_pose is not None:
            dx_world = float(target_pose[0] - robot_pose[0])
            dy_world = float(target_pose[1] - robot_pose[1])
            theta = float(robot_pose[2])
            dx_body = np.cos(theta) * dx_world + np.sin(theta) * dy_world
            dy_body = -np.sin(theta) * dx_world + np.cos(theta) * dy_world
            d_theta = float(target_pose[2] - robot_pose[2])
            goal_vec = torch.tensor([[dx_body, dy_body, np.sin(d_theta), np.cos(d_theta)]], dtype=torch.float32, device=device)
        else:
            t_id = torch.tensor([int(task_id)], dtype=torch.long, device=device)
            goal_vec = self.infer_goal_vec(z_vis, t_id)

        actions, col_risk = self.forward_with_latent(z_vis, goal_vec)
        act_np = actions.squeeze(0).cpu().numpy()
        col_np = float(col_risk.squeeze(0).cpu().numpy()[0])
        z_np = z_vis.squeeze(0).cpu().numpy()
        return act_np, col_np, z_np
