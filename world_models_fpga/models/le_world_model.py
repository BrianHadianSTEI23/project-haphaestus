import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SIGRegLoss(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization (SIGReg).
    Reference: LeWorldModel (Maes et al., arXiv:2603.19312).

    Projects latent representations onto M unit directions and enforces
    distributional alignment with a standard normal N(0, 1) via:
      1. Epps-Pulley empirical characteristic function distance against exp(-t^2 / 2), and
      2. Standardized 4-moment matching (mean=0, var=1, skew=0, kurtosis=3).
    Uses a registered deterministic projection buffer in eval mode so validation
    and tests are strictly reproducible.
    """
    def __init__(self, latent_dim=32, num_projections=32, mode="hybrid", seed=2026):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_projections = num_projections
        self.mode = mode

        gen = torch.Generator()
        gen.manual_seed(seed)
        fixed_proj = torch.randn(latent_dim, num_projections, generator=gen)
        fixed_proj = fixed_proj / (torch.norm(fixed_proj, dim=0, keepdim=True) + 1e-8)
        self.register_buffer("fixed_proj", fixed_proj)

        # Quadrature nodes t in [-2.5, 2.5] for Epps-Pulley characteristic function integral
        t_grid = torch.linspace(-2.5, 2.5, steps=11)
        self.register_buffer("t_grid", t_grid)
        self.register_buffer("target_cf", torch.exp(-0.5 * (t_grid ** 2)))
        self.register_buffer("quad_weights", torch.exp(-(t_grid ** 2)))

    def forward(self, z, deterministic: bool = False):
        """
        z: (B, latent_dim)
        Returns: SIGReg loss scalar
        """
        batch_size = z.size(0)
        if batch_size < 4:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)

        if deterministic or (not self.training):
            proj = self.fixed_proj.to(device=z.device, dtype=z.dtype)
        else:
            proj = torch.randn(self.latent_dim, self.num_projections, device=z.device, dtype=z.dtype)
            proj = proj / (torch.norm(proj, dim=0, keepdim=True) + 1e-8)

        # 1D Projections: (B, M)
        p = torch.matmul(z, proj)

        mean_p = torch.mean(p, dim=0)
        var_p = torch.var(p, dim=0, unbiased=True)
        std_p = torch.sqrt(var_p + 1e-6)
        p_norm = (p - mean_p) / std_p

        skew_p = torch.mean(p_norm ** 3, dim=0)
        kurt_p = torch.mean(p_norm ** 4, dim=0)

        loss_mean = torch.mean(mean_p ** 2)
        loss_var = torch.mean((var_p - 1.0) ** 2)
        loss_skew = torch.mean(skew_p ** 2)
        loss_kurt = torch.mean((kurt_p - 3.0) ** 2)
        moment_loss = loss_mean + loss_var + 0.1 * loss_skew + 0.05 * loss_kurt

        # Epps-Pulley characteristic function distance: E[exp(i t p)] vs exp(-t^2 / 2)
        # p_t: (B, M, K)
        p_t = p.unsqueeze(-1) * self.t_grid.view(1, 1, -1)
        ecf_real = torch.mean(torch.cos(p_t), dim=0)  # (M, K)
        ecf_imag = torch.mean(torch.sin(p_t), dim=0)  # (M, K)
        cf_diff_sq = (ecf_real - self.target_cf.view(1, -1)) ** 2 + (ecf_imag ** 2)
        ep_loss = torch.mean(cf_diff_sq * self.quad_weights.view(1, -1))

        if self.mode == "moment_matching":
            return moment_loss
        if self.mode == "epps_pulley":
            return ep_loss + loss_mean + loss_var
        return moment_loss + 0.5 * ep_loss


class LeWorldModel(nn.Module):
    r"""
    LeWorldModel (LeWM): Action-Conditioned Joint-Embedding Predictive Architecture from Pixels.
    Reference: Maes et al., arXiv:2603.19312.

    Components:
      1. Pixel Encoder E_phi: (3, 64, 64) -> s_t in R^32
      2. Transition Predictor T_theta: (s_t, a_t) -> \hat{s}_{t+1} in R^32 (residual)
      3. Grounded Collision Heads:
         - col_fc(h): transition-conditioned collision logit
         - state_col_head(\hat{s}_{t+1}): next-latent collision logit
      4. Latent Ego-Motion Decoder: (s_t, \hat{s}_{t+1} - s_t) -> \hat{\delta p}_{body} in R^3
         (decodes predicted 1-step body-frame displacement strictly from latent states)
      5. Reactive Policy Head: (s_t, g_{body}) -> a_{reactive} in [-1, 1]^3
      6. Visual Task Grounding Head (Track C): (s_t, task_id) -> \hat{g}_{body} in R^4
    """
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=64, num_projections=32, num_tasks=64, task_emb_dim=16):
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

        # Grounded Collision Head (transition hidden state)
        self.col_fc = nn.Linear(hidden_dim, 1)

        # Next-State Collision Head (evaluates predicted latent state next_s directly)
        self.state_col_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Latent Ego-Motion Displacement Decoder: maps (s_t, next_s - s_t) -> [dx_body, dy_body, dtheta]
        # Does NOT take action a_t as input, forcing next_s to carry action-induced spatial motion.
        self.ego_motion_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Reactive Goal-Conditioned Policy Proposal Head: (s_t, goal_vec) -> action in [-1, 1]^3
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Track C Visual Task Grounding Head: (s_t, task_id) -> goal_vec [dx_b, dy_b, sin_dth, cos_dth]
        self.task_embedding = nn.Embedding(num_tasks, task_emb_dim)
        self.grounding_head = nn.Sequential(
            nn.Linear(latent_dim + task_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

        self.sigreg = SIGRegLoss(latent_dim=latent_dim, num_projections=num_projections)

    def encode(self, img_tensor):
        """Encodes (B, 3, 64, 64) image into latent state s_t."""
        return self.encoder(img_tensor)

    def predict_next(self, s_t, a_t):
        r"""
        Latent transition core: (s_t, a_t) -> (\hat{s}_{t+1}, col_logit)
        s_t: (B, latent_dim)
        a_t: (B, action_dim)
        """
        x = torch.cat([s_t, a_t], dim=-1)
        h = F.relu(self.pred_fc1(x))
        h = F.relu(self.pred_fc2(h))

        delta_s = self.pred_delta(h)
        next_s = s_t + delta_s  # Residual transition dynamics
        state_col_logit = self.state_col_head(next_s)
        col_logit = 0.5 * (self.col_fc(h) + state_col_logit)

        return next_s, col_logit

    def predict_ego_displacement(self, s_t: torch.Tensor, next_s: torch.Tensor) -> torch.Tensor:
        """Decodes 1-step body-frame displacement [dx_b, dy_b, dtheta] strictly from latent transition delta (next_s - s_t)."""
        delta_s = next_s - s_t
        feat = torch.cat([torch.zeros_like(s_t), delta_s], dim=-1)
        return self.ego_motion_head(feat)

    def predict_reactive_action(self, s_t: torch.Tensor, goal_vec: torch.Tensor) -> torch.Tensor:
        """Predicts reactive proposal action in [-1, 1]^3 from current latent s_t and body-frame goal_vec."""
        return torch.tanh(self.policy_head(torch.cat([s_t, goal_vec], dim=-1)))

    def infer_goal_vec(self, s_t: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """Track C: predicts body-frame goal vector [dx_b, dy_b, sin_dth, cos_dth] from s_t + task_id."""
        t_emb = self.task_embedding(task_ids)
        raw = self.grounding_head(torch.cat([s_t, t_emb], dim=-1))
        dx_dy = torch.clamp(raw[:, :2], -3.5, 3.5)
        trig = F.normalize(raw[:, 2:4] + 1e-6, dim=-1)
        return torch.cat([dx_dy, trig], dim=-1)

    def compute_loss(
        self,
        img_t,
        action_t,
        img_next,
        col_target,
        lambda_reg=0.2,
        ego_disp_target=None,
        goal_vec=None,
        task_ids=None,
    ):
        """
        Computes LeWorldModel end-to-end loss:
          L = L_pred + lambda_reg * L_sigreg + 0.2 * L_col + L_ego + L_policy + L_ground
        """
        s_t = self.encode(img_t)
        s_next_online = self.encode(img_next)
        s_next_target = s_next_online.detach()

        pred_s_next, col_logit = self.predict_next(s_t, action_t)

        # 1. Latent Dynamics Prediction Loss
        loss_pred = F.mse_loss(pred_s_next, s_next_target)

        # 2. SIGReg Regularization on latent representations to prevent collapse
        loss_sigreg = 0.5 * (self.sigreg(s_t) + self.sigreg(s_next_online))

        # 3. Grounded Collision Loss (both transition and next-state collision heads)
        state_col_logit = self.state_col_head(pred_s_next)
        loss_col = 0.5 * (
            F.binary_cross_entropy_with_logits(col_logit, col_target)
            + F.binary_cross_entropy_with_logits(state_col_logit, col_target)
        )

        total_loss = loss_pred + lambda_reg * loss_sigreg + 0.2 * loss_col

        # 4. Latent Ego-Motion Displacement Loss (forces both encoder and predictor latents to encode action-conditioned spatial motion)
        loss_ego = torch.tensor(0.0, device=img_t.device)
        if ego_disp_target is not None:
            pred_disp_from_gt = self.predict_ego_displacement(s_t, s_next_online)
            pred_disp_from_pred = self.predict_ego_displacement(s_t.detach(), pred_s_next)
            perm = torch.roll(torch.arange(s_t.shape[0], device=s_t.device), shifts=1, dims=0)
            pred_s_perm, _ = self.predict_next(s_t.detach(), action_t[perm])
            pred_disp_perm = self.predict_ego_displacement(s_t.detach(), pred_s_perm)
            loss_ego = (
                0.35 * F.mse_loss(pred_disp_from_gt, ego_disp_target)
                + 0.35 * F.mse_loss(pred_disp_from_pred, ego_disp_target)
                + 0.30 * F.mse_loss(pred_disp_perm, ego_disp_target[perm])
            )
            total_loss = total_loss + 1.5 * loss_ego

        # 5. Goal-Conditioned Policy Proposal Loss & Track C Visual Task Grounding Loss
        loss_policy = torch.tensor(0.0, device=img_t.device)
        loss_ground = torch.tensor(0.0, device=img_t.device)
        if goal_vec is not None:
            pred_act = self.predict_reactive_action(s_t, goal_vec)
            loss_policy = F.mse_loss(pred_act, action_t)
            total_loss = total_loss + 0.5 * loss_policy
            if task_ids is not None:
                pred_goal = self.infer_goal_vec(s_t, task_ids)
                loss_ground = F.mse_loss(pred_goal, goal_vec)
                total_loss = total_loss + 0.5 * loss_ground

        metrics = {
            "loss_total": total_loss.item(),
            "loss_pred": loss_pred.item(),
            "loss_sigreg": loss_sigreg.item(),
            "loss_col": loss_col.item(),
            "loss_ego": loss_ego.item(),
            "loss_policy": loss_policy.item(),
            "loss_ground": loss_ground.item(),
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
        s_batch = s_current.repeat(num_samples, 1)
        actions_cand = torch.randn(num_samples, horizon, self.action_dim, device=device) * 0.6
        actions_cand[:, :, 0] += 0.4
        actions_cand = torch.clamp(actions_cand, -1.0, 1.0)

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

        goal_b = goal_latent.repeat(num_samples, 1)
        goal_dist = torch.norm(curr_s - goal_b, dim=-1, keepdim=True)
        cost = goal_dist + 3.0 * cum_col_risk
        best_idx = torch.argmin(cost.squeeze(-1)).item()

        best_action = actions_cand[best_idx, 0].cpu().numpy()
        pred_col = cum_col_risk[best_idx].item() / horizon

        return best_action, pred_col, curr_s[best_idx].cpu().numpy()
