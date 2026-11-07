"""
Phase 7: Circuit Extraction & Neural Architecture Reconstruction
Script: model_reconstructed_compact.py

Option B: Reconstructed Compact Neural Architecture (39,804 parameters).
Matches the exact layer dimensions of the RTL hardware cores:
  - CompactEncoder: 64x64x3 -> 16-dim latent state s_t (35,336 params)
  - CompactMacroPredictor: (16-dim s_t + 3-dim action/intent) -> 32 -> 32 -> (16-dim next_S + 1-dim col_logit) (2,257 params)
  - CompactMicroPolicy: (16-dim s_t + 16-dim S_subgoal) -> 32 -> 32 -> 3-dim holonomic action (2,211 params)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactEncoder(nn.Module):
    """
    Synthesized compact visual encoder (35,336 parameters vs 140,304 in original).
    Outputs 16-dimensional latent representation encoding spatial geometry and heading.
    """
    def __init__(self, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), # 4x4
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class CompactMacroPredictor(nn.Module):
    """
    Synthesized compact macro transition predictor (2,257 parameters vs 8,609 in original).
    Matches RTL `macro_predictor_core.v` (19 -> 32 -> 32 -> 17).
    """
    def __init__(self, latent_dim=16, act_dim=3, hidden_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, latent_dim)
        self.col_head = nn.Linear(hidden_dim, 1)

    def forward(self, S_k, A_k):
        x = torch.cat([S_k, A_k], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        delta_S = self.delta_head(h)
        next_S = S_k + delta_S
        col_logit = self.col_head(h)
        return next_S, col_logit


class CompactMicroPolicy(nn.Module):
    """
    Synthesized compact micro policy (2,211 parameters vs 8,515 in original).
    Matches RTL `micro_steering_core.v` (32 -> 32 -> 32 -> 3).
    """
    def __init__(self, latent_dim=16, action_dim=3, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, s_t, S_subgoal):
        x = torch.cat([s_t, S_subgoal], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return torch.tanh(self.action_head(h))


class ReconstructedCompactModel(nn.Module):
    """
    Reconstructed Compact Architecture (Option B).
    Total parameter count: 39,804 (76.0% reduction from 166,037).
    """
    def __init__(self, latent_dim=16, stride_k=5, hidden_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.stride_k = stride_k
        self.encoder = CompactEncoder(latent_dim=latent_dim)
        self.macro_predictor = CompactMacroPredictor(latent_dim=latent_dim, act_dim=3, hidden_dim=hidden_dim)
        self.micro_policy = CompactMicroPolicy(latent_dim=latent_dim, action_dim=3, hidden_dim=hidden_dim)

    def encode(self, img_tensor):
        return self.encoder(img_tensor)

    def forward_macro(self, S_k, A_k):
        return self.macro_predictor(S_k, A_k)

    def select_action(self, s_t, S_subgoal):
        return self.micro_policy(s_t, S_subgoal)

    @torch.no_grad()
    def act(self, obs_rgb: np.ndarray, goal_vec_body: np.ndarray, device: str = "cpu"):
        """
        Closed-loop Track B inference executing the exact 3-stage compact pipeline:
          1. CompactEncoder: obs_rgb -> s_t (16D)
          2. CompactMacroPredictor: evaluates candidate 3-DOF intents around body-frame goal
             to select a collision-free macro subgoal next_S* (16D)
          3. CompactMicroPolicy: (s_t, next_S*) -> primitive action [vx, vy, wz]
        """
        self.eval()
        img_t = torch.from_numpy(obs_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        s_t = self.encode(img_t)

        dx_b, dy_b, sin_dth, cos_dth = [float(x) for x in goal_vec_body]
        dth = float(np.arctan2(sin_dth, cos_dth))
        base_intent = torch.tensor(
            [np.clip(1.1 * dx_b, -1.0, 1.0), np.clip(1.1 * dy_b, -1.0, 1.0), np.clip(1.3 * dth, -1.0, 1.0)],
            dtype=torch.float32,
            device=device,
        )
        offsets = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.55, 0.20],
                [0.0, -0.55, -0.20],
                [-0.25, 0.70, 0.30],
                [-0.25, -0.70, -0.30],
                [0.30, 0.0, 0.50],
                [0.30, 0.0, -0.50],
                [0.75, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        cands = torch.clamp(base_intent.unsqueeze(0) + offsets, -1.0, 1.0)
        s_batch = s_t.repeat(cands.shape[0], 1)
        next_S_all, col_logits = self.forward_macro(s_batch, cands)
        col_probs = torch.sigmoid(col_logits).squeeze(-1)

        # Pick macro candidate balancing intent fidelity and low predicted collision probability
        intent_dev = torch.sum((cands - base_intent.unsqueeze(0)) ** 2, dim=-1)
        macro_cost = intent_dev + 2.2 * col_probs
        best_idx = int(torch.argmin(macro_cost).item())

        best_subgoal = next_S_all[best_idx : best_idx + 1]
        action = self.select_action(s_t, best_subgoal).squeeze(0).cpu().numpy()
        return np.clip(action, -1.0, 1.0).astype(np.float32), {
            "col_risk": float(col_probs[best_idx].item()),
            "latent": s_t.squeeze(0).cpu().numpy(),
            "subgoal": best_subgoal.squeeze(0).cpu().numpy(),
        }

    def count_parameters(self):
        enc = sum(p.numel() for p in self.encoder.parameters())
        macro = sum(p.numel() for p in self.macro_predictor.parameters())
        micro = sum(p.numel() for p in self.micro_policy.parameters())
        total = enc + macro + micro
        return {"encoder": enc, "macro": macro, "micro": micro, "total": total}


if __name__ == "__main__":
    model = ReconstructedCompactModel()
    counts = model.count_parameters()
    print("Reconstructed Compact Model Parameter Inventory:")
    for k, v in counts.items():
        print(f"  {k.capitalize():<12}: {v:,} parameters")
