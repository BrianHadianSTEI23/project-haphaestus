import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    Language-Goal Conditioned Navigation Policy with Obstacle Avoidance.
    Input: Visual Latent State s_t (32-dim) + Relative Goal Vector g_t (4-dim: dx, dy, sin_dth, cos_dth)
    Output: 3-DOF Holonomic Action [vx, vy, wz] in [-1, 1]^3 and Collision Risk Logit c_t in [0, 1].
    """
    def __init__(self, latent_dim=32, goal_dim=4, action_dim=3, hidden_dim=64):
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

    def forward(self, img_tensor, goal_vec):
        """
        img_tensor: (B, 3, 64, 64) normalized in [0, 1]
        goal_vec: (B, 4) [dx_rel, dy_rel, sin(dth), cos(dth)]
        """
        z_vis = self.encoder(img_tensor)
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
    def act(self, obs_rgb, robot_pose, target_pose, device="cpu"):
        """
        Computes 3-DOF action given raw RGB observation and poses.
        """
        self.eval()
        # Preprocess image
        img_t = torch.from_numpy(obs_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        # Compute robot body-frame relative goal vector
        dx_world = float(target_pose[0] - robot_pose[0])
        dy_world = float(target_pose[1] - robot_pose[1])
        theta = float(robot_pose[2])

        # Rotate world delta into robot body frame
        dx_body = np.cos(theta) * dx_world + np.sin(theta) * dy_world
        dy_body = -np.sin(theta) * dx_world + np.cos(theta) * dy_world

        # Heading error to goal
        d_theta = float(target_pose[2] - robot_pose[2])
        sin_th = np.sin(d_theta)
        cos_th = np.cos(d_theta)

        goal_vec = torch.tensor([[dx_body, dy_body, sin_th, cos_th]], dtype=torch.float32, device=device)

        (actions, col_risk), z_vis = self.forward(img_t, goal_vec)

        act_np = actions.squeeze(0).cpu().numpy()
        col_np = col_risk.squeeze(0).cpu().numpy()[0]
        z_np = z_vis.squeeze(0).cpu().numpy()

        return act_np, col_np, z_np
