"""
Phase 7: Circuit Extraction & Neural Architecture Reconstruction
Script: model_reconstructed_compact.py
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Option B: Reconstructed Compact Neural Architecture.
Derived directly from the 5 validated principles (P1–P5):
  - P1: Action-Conditioned Residual Foresight (s_{t+1} = s_t + T_macro(s_t, a_t))
  - P2: Hierarchical Macro-Subgoal Abstraction (K=5 macro stride)
  - P3: Egocentric Relational Coordinate Decoupling (16-dim body-frame latent space)
  - P4: Joint-Embedding Appearance Filtering (Color channels discarded, 74.8% encoder compression)
  - P5: Anticipatory Safety Barrier Function (Integrated collision logit head)
Quantization-ready for AMD Kria KR260 FPGA deployment with signed Q8.8 arithmetic.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class CompactEncoder(nn.Module):
    """
    Synthesized compact visual encoder (35,336 parameters vs 140,304 in original).
    Outputs 16-dimensional latent representation strictly encoding spatial geometry and heading.
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
    Implements P1 (Action-Conditioned Foresight) and P5 (Safety Barrier Function).
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
    Implements P3 (Egocentric Relational Steering) to track macro subgoals.
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
