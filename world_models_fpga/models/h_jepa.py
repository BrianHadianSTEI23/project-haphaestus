import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.le_world_model import SIGRegLoss

class MacroPredictor(nn.Module):
    r"""
    High-Level Macro Transition Model (Temporal Stride K=5 steps).
    Predicts coarse latent subgoals \hat{S}_{k+1} = T_macro(S_k, A_macro)
    where A_macro is a macro displacement / trajectory intent vector.
    """
    def __init__(self, macro_dim=32, macro_act_dim=3, hidden_dim=64):
        super().__init__()
        self.macro_dim = macro_dim
        self.macro_act_dim = macro_act_dim

        in_dim = macro_dim + macro_act_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, macro_dim)
        self.col_head = nn.Linear(hidden_dim, 1)

    def forward(self, S_k, A_k):
        """
        S_k: (B, macro_dim)
        A_k: (B, macro_act_dim)
        """
        x = torch.cat([S_k, A_k], dim=-1)
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
    Trained end-to-end with SIGReg regularization at both hierarchy levels.
    """
    def __init__(self, latent_dim=32, stride_k=5, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.stride_k = stride_k

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

        self.macro_predictor = MacroPredictor(macro_dim=latent_dim, macro_act_dim=3, hidden_dim=hidden_dim)
        self.micro_predictor = MicroPredictor(micro_dim=latent_dim, macro_dim=latent_dim, action_dim=3, hidden_dim=hidden_dim)

        self.sigreg = SIGRegLoss(latent_dim=latent_dim)

    def encode(self, img_tensor):
        return self.encoder(img_tensor)

    def compute_loss(self, img_seq, action_seq, col_seq, lambda_reg=0.2):
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
        s_t1_target = latents[:, 1].detach()
        pred_s_t1, micro_col = self.micro_predictor.forward_transition(s_t, a_t)

        loss_micro_pred = F.mse_loss(pred_s_t1, s_t1_target)
        loss_micro_col = F.binary_cross_entropy_with_logits(micro_col, col_seq[:, 0])

        # 2. Macro Step Losses (Stride K)
        if seq_len > self.stride_k:
            s_macro_start = latents[:, 0]
            # Macro action is integrated displacement over K steps
            macro_action = torch.mean(action_seq[:, :self.stride_k], dim=1)
            s_macro_target = latents[:, self.stride_k].detach()

            pred_s_macro, macro_col = self.macro_predictor(s_macro_start, macro_action)
            loss_macro_pred = F.mse_loss(pred_s_macro, s_macro_target)
            loss_macro_col = F.binary_cross_entropy_with_logits(macro_col, torch.max(col_seq[:, :self.stride_k], dim=1)[0])
        else:
            loss_macro_pred = torch.tensor(0.0, device=device)
            loss_macro_col = torch.tensor(0.0, device=device)

        # 3. SIGReg Regularization
        loss_sigreg = self.sigreg(s_t)

        total_loss = (
            loss_micro_pred + 0.8 * loss_macro_pred +
            lambda_reg * loss_sigreg +
            0.2 * (loss_micro_col + loss_macro_col)
        )

        metrics = {
            "loss_total": total_loss.item(),
            "loss_micro_pred": loss_micro_pred.item(),
            "loss_macro_pred": loss_macro_pred.item(),
            "loss_sigreg": loss_sigreg.item()
        }
        return total_loss, metrics

    @torch.no_grad()
    def plan_hierarchical(self, s_current, goal_latent, num_macro_candidates=32, device="cpu"):
        """
        2-Level Hierarchical Planning:
          Step 1 (Macro Level): Select optimal macro subgoal S* across stride K
          Step 2 (Micro Level): Execute micro action tracking S*
        """
        # 1. Macro Planning: evaluate coarse macro actions
        macro_actions = torch.randn(num_macro_candidates, 3, device=device) * 0.7
        macro_actions[:, 0] += 0.5  # Bias forward
        macro_actions = torch.clamp(macro_actions, -1.0, 1.0)
        macro_actions[0] = torch.tensor([0.8, 0.0, 0.0], device=device)

        s_macro_batch = s_current.repeat(num_macro_candidates, 1)
        next_S_macro, col_logits = self.macro_predictor(s_macro_batch, macro_actions)

        goal_b = goal_latent.repeat(num_macro_candidates, 1)
        dist_to_goal = torch.norm(next_S_macro - goal_b, dim=-1)
        col_risks = torch.sigmoid(col_logits).squeeze(-1)

        macro_cost = dist_to_goal + 3.0 * col_risks
        best_macro_idx = torch.argmin(macro_cost).item()

        best_subgoal = next_S_macro[best_macro_idx:best_macro_idx+1]

        # 2. Micro Tracking: produce primitive action to reach best_subgoal
        micro_action = self.micro_predictor.select_action(s_current, best_subgoal)
        act_np = micro_action.squeeze(0).cpu().numpy()
        pred_col = col_risks[best_macro_idx].item()

        return act_np, pred_col, best_subgoal.squeeze(0).cpu().numpy()
