"""
Phase 7: Circuit Extraction & Neural Architecture Reconstruction
Script: model_weight_preserving.py
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Option A: Weight-Preserving Extracted Model.
Retains exact weights from original pre-trained H-JEPA, but removes all unreferenced
modules (micro dynamics predictor) and applies structured masks to non-causal channels.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.h_jepa import MacroPredictor

class WeightPreservingMicroPolicy(nn.Module):
    def __init__(self, micro_dim=32, macro_dim=32, action_dim=3, hidden_dim=64):
        super().__init__()
        self.pol_fc1 = nn.Linear(micro_dim + macro_dim, hidden_dim)
        self.pol_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def select_action(self, s_t, S_subgoal):
        x = torch.cat([s_t, S_subgoal], dim=-1)
        h = F.relu(self.pol_fc1(x))
        h = F.relu(self.pol_fc2(h))
        return torch.tanh(self.action_head(h))

class WeightPreservingExtractedHJEPA(nn.Module):
    """
    Extracted Model Option A: Weight-Preserving Architecture.
    - Preserves trained weights from hjepa_best.pt.
    - Completely prunes unreferenced micro dynamics predictor (8,609 parameters removed).
    - Preserves full visual encoder and macro planner with masking on non-causal appearance channel.
    """
    def __init__(self, latent_dim=32, stride_k=5, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.stride_k = stride_k

        # Visual Encoder
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

        # Macro Predictor (Stride K=5)
        self.macro_predictor = MacroPredictor(macro_dim=latent_dim, macro_act_dim=3, hidden_dim=hidden_dim)

        # Micro Policy (Action tracking only, no micro dynamics!)
        self.micro_policy = WeightPreservingMicroPolicy(micro_dim=latent_dim, macro_dim=latent_dim, action_dim=3, hidden_dim=hidden_dim)

        # Causal mask: zeroes out dimension 14 (pure color)
        self.register_buffer("causal_mask", torch.ones(latent_dim))
        self.causal_mask[14] = 0.0

    def encode(self, img_tensor):
        z = self.encoder(img_tensor)
        return z * self.causal_mask

    def load_from_original_checkpoint(self, ckpt_path, device="cpu"):
        orig_ckpt = torch.load(ckpt_path, map_location=device)
        own_state = self.state_dict()
        loaded = 0
        skipped = 0

        for name, param in orig_ckpt.items():
            if name in own_state:
                if own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    loaded += 1
            elif "micro_predictor.pol_" in name or "micro_predictor.action_head" in name:
                # Map micro_predictor policy weights into micro_policy
                mapped_name = name.replace("micro_predictor.", "micro_policy.")
                if mapped_name in own_state and own_state[mapped_name].shape == param.shape:
                    own_state[mapped_name].copy_(param)
                    loaded += 1
            else:
                skipped += 1

        print(f"[Option A] Loaded {loaded} tensors ({sum(p.numel() for p in self.parameters()):,} active parameters). Skipped {skipped} redundant tensors.")

    def forward_macro(self, S_k, A_k):
        return self.macro_predictor(S_k, A_k)

    def select_action(self, s_t, S_subgoal):
        return self.micro_policy.select_action(s_t, S_subgoal)

if __name__ == "__main__":
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../checkpoints/hjepa_best.pt"))
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "model_weights_preserved.pt"))

    model = WeightPreservingExtractedHJEPA()
    model.load_from_original_checkpoint(ckpt_path)
    torch.save(model.state_dict(), save_path)
    print(f"[Option A] Saved weight-preserving extracted model to {save_path}")
