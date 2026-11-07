"""
Phase 7: Circuit Extraction & Neural Architecture Reconstruction
Script: model_weight_preserving.py

Option A: Weight-Preserving Extracted Model.
Retains exact weights from pre-trained H-JEPA, removes the micro dynamics predictor branch,
and applies structured masks to non-causal channels.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    - Prunes micro dynamics predictor branch.
    - Preserves visual encoder, macro predictor, and micro policy.
    """
    def __init__(self, latent_dim=32, stride_k=5, hidden_dim=64, macro_act_dim=15):
        super().__init__()
        self.latent_dim = latent_dim
        self.stride_k = stride_k
        self.hidden_dim = hidden_dim

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

        self.macro_predictor = MacroPredictor(macro_dim=latent_dim, macro_act_dim=macro_act_dim, hidden_dim=hidden_dim)
        self.micro_policy = WeightPreservingMicroPolicy(micro_dim=latent_dim, macro_dim=latent_dim, action_dim=3, hidden_dim=hidden_dim)
        self.micro_predictor = self.micro_policy

        self.register_buffer("causal_mask", torch.ones(latent_dim))
        self.causal_mask[14] = 0.0

    def encode(self, img_tensor):
        z = self.encoder(img_tensor)
        return z * self.causal_mask

    def load_from_original_checkpoint(self, ckpt_path, device="cpu"):
        orig_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(orig_ckpt, dict) and "state_dict" in orig_ckpt:
            orig_ckpt = orig_ckpt["state_dict"]

        if "macro_predictor.fc1.weight" in orig_ckpt:
            in_w = orig_ckpt["macro_predictor.fc1.weight"].shape[1]
            act_dim = in_w - self.latent_dim
            if act_dim != self.macro_predictor.macro_act_dim:
                self.macro_predictor = MacroPredictor(
                    macro_dim=self.latent_dim, macro_act_dim=act_dim, hidden_dim=self.hidden_dim
                ).to(device)

        own_state = self.state_dict()
        loaded = 0
        skipped = 0

        for name, param in orig_ckpt.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
            elif "micro_predictor.pol_" in name or "micro_predictor.action_head" in name:
                mapped_name = name.replace("micro_predictor.", "micro_policy.")
                if mapped_name in own_state and own_state[mapped_name].shape == param.shape:
                    own_state[mapped_name].copy_(param)
                    loaded += 1
            else:
                skipped += 1

        print(f"[Option A] Loaded {loaded} tensors ({sum(p.numel() for p in self.parameters()):,} structural parameters). Skipped {skipped} tensors.")

    def load_from_full_hjepa(self, ckpt_path, device="cpu"):
        return self.load_from_original_checkpoint(ckpt_path, device=device)

    def forward_macro(self, S_k, A_k):
        return self.macro_predictor(S_k, A_k)

    def select_action(self, s_t, S_subgoal):
        return self.micro_policy.select_action(s_t, S_subgoal)

    def count_parameters(self):
        enc = sum(p.numel() for p in self.encoder.parameters())
        macro = sum(p.numel() for p in self.macro_predictor.parameters())
        micro = sum(p.numel() for p in self.micro_policy.parameters())
        return {"encoder": enc, "macro": macro, "micro": micro, "total": enc + macro + micro}


WeightPreservingExtractedModel = WeightPreservingExtractedHJEPA


if __name__ == "__main__":
    model = WeightPreservingExtractedHJEPA()
    counts = model.count_parameters()
    print("Weight-Preserving Extracted Model Parameter Inventory:")
    for k, v in counts.items():
        print(f"  {k.capitalize():<12}: {v:,} parameters")
