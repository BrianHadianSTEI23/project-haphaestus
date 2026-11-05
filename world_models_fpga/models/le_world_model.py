import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SIGRegLoss(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization (SIGReg)
    Reference: LeWorldModel (Maes et al., arXiv:2603.19312)
    
    Projects latent representations onto M random unit directions and enforces
    distributional alignment with a standard normal N(0, 1). Prevents latent
    space representation collapse without needing EMA, decoders, or contrastive pairs.
    """
    def __init__(self, latent_dim=32, num_projections=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_projections = num_projections

    def forward(self, z):
        """
        z: (B, latent_dim)
        Returns: SIGReg loss scalar
        """
        batch_size = z.size(0)
        if batch_size < 4:
            return torch.tensor(0.0, device=z.device)

        # Generate random unit projection directions (Sketched projections)
        proj = torch.randn(self.latent_dim, self.num_projections, device=z.device)
        proj = proj / (torch.norm(proj, dim=0, keepdim=True) + 1e-8)

        # 1D Projections: (B, num_projections)
        p = torch.matmul(z, proj)

        # Match moments to standard normal N(0, 1)
        mean_p = torch.mean(p, dim=0)
        var_p = torch.var(p, dim=0, unbiased=True)
        std_p = torch.sqrt(var_p + 1e-8)
        
        # Standardized projections
        p_norm = (p - mean_p) / std_p

        skew_p = torch.mean(p_norm ** 3, dim=0)
        kurt_p = torch.mean(p_norm ** 4, dim=0)

        # Penalties: Mean -> 0, Var -> 1, Skewness -> 0, Kurtosis -> 3
        loss_mean = torch.mean(mean_p ** 2)
        loss_var = torch.mean((var_p - 1.0) ** 2)
        loss_skew = torch.mean(skew_p ** 2)
        loss_kurt = torch.mean((kurt_p - 3.0) ** 2)

        sigreg_loss = loss_mean + loss_var + 0.1 * loss_skew + 0.05 * loss_kurt
        return sigreg_loss

class LeWorldModel(nn.Module):
    r"""
    LeWorldModel (LeWM): Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.
    Reference: Maes et al., arXiv:2603.19312
    
    Consists of:
      1. Pixel Encoder E_phi: (3, 64, 64) -> s_t in R^32
      2. Predictor T_theta: (s_t, a_t) -> \hat{s}_{t+1} in R^32 (residual)
      3. Grounded Collision Head: (s_t, a_t) -> c_t in [0, 1]
    Trained end-to-end with Next-Latent MSE + SIGReg regularization.
    """
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=64, num_projections=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Pixel Encoder (JEPA visual encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),   # -> 32x32x16
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # -> 16x16x32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 8x8x64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # -> 4x4x64
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # Transition Dynamics Predictor (Q8.8 Fixed-Point FPGA Deployable)
        in_dim = latent_dim + action_dim
        self.pred_fc1 = nn.Linear(in_dim, hidden_dim)
        self.pred_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.pred_delta = nn.Linear(hidden_dim, latent_dim)

        # Grounded Collision Head
        self.col_fc = nn.Linear(hidden_dim, 1)

        self.sigreg = SIGRegLoss(latent_dim=latent_dim, num_projections=num_projections)

    def encode(self, img_tensor):
        """Encodes (B, 3, 64, 64) image into latent state s_t."""
        return self.encoder(img_tensor)

    def predict_next(self, s_t, a_t):
        """
        Latent transition core: (s_t, a_t) -> (\hat{s}_{t+1}, col_logit)
        s_t: (B, latent_dim)
        a_t: (B, action_dim)
        """
        x = torch.cat([s_t, a_t], dim=-1)
        h = F.relu(self.pred_fc1(x))
        h = F.relu(self.pred_fc2(h))

        delta_s = self.pred_delta(h)
        next_s = s_t + delta_s  # Residual transition dynamics
        col_logit = self.col_fc(h)

        return next_s, col_logit

    def compute_loss(self, img_t, action_t, img_next, col_target, lambda_reg=0.2):
        """
        Computes LeWorldModel end-to-end loss:
          L = L_pred + lambda_reg * L_sigreg + 0.2 * L_col
        """
        s_t = self.encode(img_t)
        with torch.no_grad():
            s_next_target = self.encode(img_next)

        pred_s_next, col_logit = self.predict_next(s_t, action_t)

        # 1. Latent Dynamics Prediction Loss
        loss_pred = F.mse_loss(pred_s_next, s_next_target)

        # 2. SIGReg Regularization on latent representations to prevent collapse
        loss_sigreg = self.sigreg(s_t)

        # 3. Grounded Collision Loss
        loss_col = F.binary_cross_entropy_with_logits(col_logit, col_target)

        total_loss = loss_pred + lambda_reg * loss_sigreg + 0.2 * loss_col

        metrics = {
            "loss_total": total_loss.item(),
            "loss_pred": loss_pred.item(),
            "loss_sigreg": loss_sigreg.item(),
            "loss_col": loss_col.item()
        }
        return total_loss, metrics

    @torch.no_grad()
    def plan_mpc(self, s_current, goal_latent, horizon=6, num_samples=64, device="cpu"):
        """
        Model-Predictive Path Planning via Latent Foresight (LeWM):
        Samples candidate 3-DOF action trajectories, unrolls future latent states
        through the predictor core, and picks the trajectory minimizing latent goal error
        and collision risk.
        """
        # Expand current latent across batch of candidate trajectories
        s_batch = s_current.repeat(num_samples, 1)

        # Sample candidate action sequences: (num_samples, horizon, 3)
        # 1. Straight forward, 2. Straight + Strafe, 3. Turn + Forward
        actions_cand = torch.randn(num_samples, horizon, self.action_dim, device=device) * 0.6
        # Bias candidates towards forward velocity
        actions_cand[:, :, 0] += 0.4
        actions_cand = torch.clamp(actions_cand, -1.0, 1.0)

        # Ensure first candidate is pure forward
        actions_cand[0, :, 0] = 0.8
        actions_cand[0, :, 1] = 0.0
        actions_cand[0, :, 2] = 0.0

        cum_col_risk = torch.zeros(num_samples, 1, device=device)
        curr_s = s_batch

        for t in range(horizon):
            a_t = actions_cand[:, t]
            next_s, col_logit = self.predict_next(curr_s, a_t)
            col_p = torch.sigmoid(col_logit)
            cum_col_risk += col_p
            curr_s = next_s

        # Evaluate distance to goal latent
        goal_b = goal_latent.repeat(num_samples, 1)
        goal_dist = torch.norm(curr_s - goal_b, dim=-1, keepdim=True)

        # Total trajectory cost = Goal Distance + Collision Penalty
        cost = goal_dist + 3.0 * cum_col_risk
        best_idx = torch.argmin(cost.squeeze(-1)).item()

        best_action = actions_cand[best_idx, 0].cpu().numpy()
        pred_col = cum_col_risk[best_idx].item() / horizon

        return best_action, pred_col, curr_s[best_idx].cpu().numpy()
