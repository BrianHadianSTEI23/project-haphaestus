import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.le_world_model import SIGRegLoss


class MacroPredictor(nn.Module):
    r"""
    High-Level Macro Transition Model (Temporal Stride K=5 steps).
    Predicts coarse latent subgoals \hat{S}_{t+K} = T_macro(S_t, A_{t:t+K-1})
    where A_{t:t+K-1} is the ordered K-step action sequence in R^{3*K}
    (or a 3-DOF constant macro action automatically repeated across K steps).
    """
    def __init__(self, macro_dim=32, macro_act_dim=15, hidden_dim=64):
        super().__init__()
        self.macro_dim = macro_dim
        self.macro_act_dim = macro_act_dim

        in_dim = macro_dim + macro_act_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, macro_dim)
        self.col_head = nn.Linear(hidden_dim, 1)

    def _format_macro_action(self, A_k: torch.Tensor) -> torch.Tensor:
        if A_k.dim() == 3:
            A_k = A_k.reshape(A_k.shape[0], -1)
        if A_k.shape[-1] == self.macro_act_dim:
            return A_k
        if A_k.shape[-1] == 3 and self.macro_act_dim % 3 == 0:
            return A_k.repeat(1, self.macro_act_dim // 3)
        raise ValueError(f"Expected macro action dim {self.macro_act_dim} or 3, got {A_k.shape}")

    def forward(self, S_k, A_k):
        """
        S_k: (B, macro_dim)
        A_k: (B, macro_act_dim) or (B, 3) constant macro candidate
        """
        A_formatted = self._format_macro_action(A_k)
        x = torch.cat([S_k, A_formatted], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        delta_S = self.delta_head(h)
        next_S = S_k + delta_S
        col_logit = self.col_head(h)
        return next_S, col_logit


class MicroPredictor(nn.Module):
    r"""
    Low-Level Micro Transition & Action Tracking Model (Temporal Stride 1 step).
    Maps (s_t, S_subgoal) -> primitive holonomic action [vx, vy, wz]
    and predicts micro state transition \hat{s}_{t+1} = T_micro(s_t, a_t).
    """
    def __init__(self, micro_dim=32, macro_dim=32, action_dim=3, hidden_dim=64):
        super().__init__()
        self.micro_dim = micro_dim
        self.macro_dim = macro_dim
        self.action_dim = action_dim

        # 1. Micro Policy: Conditioned on (current micro state s_t, macro subgoal S_subgoal) -> action a_t
        self.pol_fc1 = nn.Linear(micro_dim + macro_dim, hidden_dim)
        self.pol_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

        # 2. Micro Latent Dynamics Predictor: (s_t, a_t) -> \hat{s}_{t+1}
        self.dyn_fc1 = nn.Linear(micro_dim + action_dim, hidden_dim)
        self.dyn_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, micro_dim)
        self.col_head = nn.Linear(hidden_dim, 1)

    def select_action(self, s_t, S_subgoal):
        """Generates primitive action to track macro subgoal."""
        x = torch.cat([s_t, S_subgoal], dim=-1)
        h = F.relu(self.pol_fc1(x))
        h = F.relu(self.pol_fc2(h))
        action = torch.tanh(self.action_head(h))
        return action

    def forward_transition(self, s_t, a_t):
        """Micro single-step transition."""
        x = torch.cat([s_t, a_t], dim=-1)
        h = F.relu(self.dyn_fc1(x))
        h = F.relu(self.dyn_fc2(h))
        delta_s = self.delta_head(h)
        next_s = s_t + delta_s
        col_logit = self.col_head(h)
        return next_s, col_logit


class HJEPAModel(nn.Module):
    """
    H-JEPA: Hierarchical Joint-Embedding Predictive Architecture.
    Reference: Zhang et al., arXiv:2610.06805.

    Operates at two hierarchical tiers:
      - Tier 1: Macro Predictor planning coarse subgoals at stride K=5
      - Tier 2: Micro Predictor executing primitive holonomic actions at stride 1
    Includes latent ego-motion displacement decoders at both macro (K-step) and
    micro (1-step) scales plus Track C visual task grounding.
    """
    def __init__(self, latent_dim=32, stride_k=5, hidden_dim=64, num_tasks=64, task_emb_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.stride_k = stride_k
        self.hidden_dim = hidden_dim

        # Shared Visual Encoder mapping RGB images to latent embeddings
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # Macro transition receives the ordered K-action interval (or 3-dim constant action candidate)
        self.macro_predictor = MacroPredictor(macro_dim=latent_dim, macro_act_dim=3 * stride_k, hidden_dim=hidden_dim)
        self.micro_predictor = MicroPredictor(micro_dim=latent_dim, macro_dim=latent_dim, action_dim=3, hidden_dim=hidden_dim)

        # Next-state collision head on predicted latent states
        self.state_col_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Macro (K-step) and Micro (1-step) Latent Ego-Motion Displacement Decoders
        # Decode body-frame displacement [dx_b, dy_b, dtheta] strictly from (s_t, s_future - s_t).
        self.macro_ego_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.micro_ego_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Reactive Goal-Conditioned Proposal Head: (s_t, goal_vec) -> action in [-1, 1]^3
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
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

        self.sigreg = SIGRegLoss(latent_dim=latent_dim)

    def encode(self, img_tensor):
        return self.encoder(img_tensor)

    def predict_macro_displacement(self, s_t: torch.Tensor, next_S: torch.Tensor) -> torch.Tensor:
        """Decodes K-step body-frame displacement [dx_b, dy_b, dtheta] strictly from macro latent delta (next_S - s_t)."""
        return self.macro_ego_head(torch.cat([torch.zeros_like(s_t), next_S - s_t], dim=-1))

    def predict_micro_displacement(self, s_t: torch.Tensor, next_s: torch.Tensor) -> torch.Tensor:
        """Decodes 1-step body-frame displacement [dx_b, dy_b, dtheta] strictly from micro latent delta (next_s - s_t)."""
        return self.micro_ego_head(torch.cat([torch.zeros_like(s_t), next_s - s_t], dim=-1))

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
        img_seq,
        action_seq,
        col_seq,
        lambda_reg=0.2,
        micro_ego_target=None,
        macro_ego_target=None,
        goal_vec=None,
        task_ids=None,
    ):
        """
        img_seq: (B, T, 3, 64, 64) with T >= K + 1
        action_seq: (B, T, 3)
        col_seq: (B, T, 1)
        """
        batch_size, seq_len = action_seq.shape[:2]
        device = action_seq.device

        # Encode sequence of images into latents
        flat_imgs = img_seq.view(batch_size * seq_len, 3, 64, 64)
        flat_latents = self.encode(flat_imgs)
        latents = flat_latents.view(batch_size, seq_len, self.latent_dim)

        # 1. Micro Step Losses (Single Step)
        s_t = latents[:, 0]
        a_t = action_seq[:, 0]
        s_t1_online = latents[:, 1]
        s_t1_target = s_t1_online.detach()
        pred_s_t1, micro_col = self.micro_predictor.forward_transition(s_t, a_t)

        loss_micro_pred = F.mse_loss(pred_s_t1, s_t1_target)
        state_micro_col = self.state_col_head(pred_s_t1)
        loss_micro_col = 0.5 * (
            F.binary_cross_entropy_with_logits(micro_col, col_seq[:, 0])
            + F.binary_cross_entropy_with_logits(state_micro_col, col_seq[:, 0])
        )

        # 2. Macro Step Losses (Stride K)
        if seq_len > self.stride_k:
            s_macro_start = latents[:, 0]
            macro_action = action_seq[:, :self.stride_k].reshape(batch_size, -1)
            s_macro_online = latents[:, self.stride_k]
            s_macro_target = s_macro_online.detach()

            pred_s_macro, macro_col = self.macro_predictor(s_macro_start, macro_action)
            loss_macro_pred = F.mse_loss(pred_s_macro, s_macro_target)
            macro_col_target = torch.max(col_seq[:, :self.stride_k], dim=1)[0]
            state_macro_col = self.state_col_head(pred_s_macro)
            loss_macro_col = 0.5 * (
                F.binary_cross_entropy_with_logits(macro_col, macro_col_target)
                + F.binary_cross_entropy_with_logits(state_macro_col, macro_col_target)
            )

            # Train micro policy on BOTH ground-truth macro target and predicted macro subgoal
            pred_action_gt = self.micro_predictor.select_action(s_t, s_macro_target)
            pred_action_pred = self.micro_predictor.select_action(s_t, pred_s_macro.detach())
            loss_action = 0.5 * (F.mse_loss(pred_action_gt, a_t) + F.mse_loss(pred_action_pred, a_t))
        else:
            loss_macro_pred = torch.tensor(0.0, device=device)
            loss_macro_col = torch.tensor(0.0, device=device)
            pred_action = self.micro_predictor.select_action(s_t, s_t1_target)
            loss_action = F.mse_loss(pred_action, a_t)
            s_macro_online = s_t1_online
            pred_s_macro = pred_s_t1

        # 3. SIGReg Regularization across hierarchy
        loss_sigreg = 0.5 * (self.sigreg(s_t) + self.sigreg(s_macro_online))

        total_loss = (
            loss_micro_pred
            + 0.8 * loss_macro_pred
            + loss_action
            + lambda_reg * loss_sigreg
            + 0.2 * (loss_micro_col + loss_macro_col)
        )

        # 4. Latent Ego-Motion Displacement Losses (Micro 1-step & Macro K-step with counterfactual action consistency)
        loss_ego = torch.tensor(0.0, device=device)
        perm = torch.roll(torch.arange(batch_size, device=device), shifts=1, dims=0)
        if micro_ego_target is not None:
            pred_s_t1_perm, _ = self.micro_predictor.forward_transition(s_t.detach(), a_t[perm])
            l_micro_ego = (
                0.35 * F.mse_loss(self.predict_micro_displacement(s_t, s_t1_online), micro_ego_target)
                + 0.35 * F.mse_loss(self.predict_micro_displacement(s_t.detach(), pred_s_t1), micro_ego_target)
                + 0.30 * F.mse_loss(self.predict_micro_displacement(s_t.detach(), pred_s_t1_perm), micro_ego_target[perm])
            )
            loss_ego = loss_ego + l_micro_ego
        if macro_ego_target is not None and seq_len > self.stride_k:
            pred_s_macro_perm, _ = self.macro_predictor(s_macro_start.detach(), macro_action[perm])
            l_macro_ego = (
                0.35 * F.mse_loss(self.predict_macro_displacement(s_t, s_macro_online), macro_ego_target)
                + 0.35 * F.mse_loss(self.predict_macro_displacement(s_t.detach(), pred_s_macro), macro_ego_target)
                + 0.30 * F.mse_loss(self.predict_macro_displacement(s_t.detach(), pred_s_macro_perm), macro_ego_target[perm])
            )
            loss_ego = loss_ego + l_macro_ego
        total_loss = total_loss + 1.5 * loss_ego

        # 5. Reactive Policy Proposal & Track C Grounding Losses
        loss_policy = torch.tensor(0.0, device=device)
        loss_ground = torch.tensor(0.0, device=device)
        if goal_vec is not None:
            loss_policy = F.mse_loss(self.predict_reactive_action(s_t, goal_vec), a_t)
            total_loss = total_loss + 0.5 * loss_policy
            if task_ids is not None:
                loss_ground = F.mse_loss(self.infer_goal_vec(s_t, task_ids), goal_vec)
                total_loss = total_loss + 0.5 * loss_ground

        metrics = {
            "loss_total": total_loss.item(),
            "loss_micro_pred": loss_micro_pred.item(),
            "loss_macro_pred": loss_macro_pred.item(),
            "loss_sigreg": loss_sigreg.item(),
            "loss_action": loss_action.item(),
            "loss_ego": loss_ego.item(),
            "loss_policy": loss_policy.item(),
            "loss_ground": loss_ground.item(),
        }
        return total_loss, metrics

    @torch.no_grad()
    def plan_hierarchical(self, s_current, goal_latent, num_macro_candidates=32, device="cpu"):
        """
        2-Level Hierarchical Planning:
          Step 1 (Macro Level): Select optimal macro subgoal S* across stride K
          Step 2 (Micro Level): Execute micro action tracking S*
        """
        primitive_actions = torch.randn(num_macro_candidates, 3, device=device) * 0.7
        primitive_actions[:, 0] += 0.5  # Bias forward
        primitive_actions = torch.clamp(primitive_actions, -1.0, 1.0)
        primitive_actions[0] = torch.tensor([0.8, 0.0, 0.0], device=device)
        macro_actions = primitive_actions.repeat(1, self.stride_k)

        s_macro_batch = s_current.repeat(num_macro_candidates, 1)
        next_S_macro, col_logits = self.macro_predictor(s_macro_batch, macro_actions)

        goal_b = goal_latent.repeat(num_macro_candidates, 1)
        dist_to_goal = torch.norm(next_S_macro - goal_b, dim=-1)
        col_risks = torch.sigmoid(col_logits).squeeze(-1)

        macro_cost = dist_to_goal + 3.0 * col_risks
        best_macro_idx = torch.argmin(macro_cost).item()

        best_subgoal = next_S_macro[best_macro_idx:best_macro_idx+1]

        action = self.micro_predictor.select_action(s_current, best_subgoal)
        _, micro_col_logit = self.micro_predictor.forward_transition(s_current, action)
        pred_col = torch.sigmoid(micro_col_logit).item()

        return action.squeeze(0).cpu().numpy(), pred_col, best_subgoal.squeeze(0).cpu().numpy()
