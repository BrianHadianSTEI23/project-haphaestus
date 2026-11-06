"""
Phase 4: Sparse Autoencoder (SAE) Architecture
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Implements dictionary-learning Sparse Autoencoder with tied pre-bias:
  z = ReLU(W_e * (h - b_d) + b_e)
  h_hat = W_d * z + b_d
Where decoder dictionary columns are constrained to unit norm: ||W_d[:, j]||_2 = 1.
Loss: L_SAE = ||h - h_hat||_2^2 + lambda * ||z||_1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, act_dim=32, dict_dim=128, l1_coeff=0.005):
        super().__init__()
        self.act_dim = act_dim
        self.dict_dim = dict_dim
        self.l1_coeff = l1_coeff

        # Decoder bias acts as untied pre-bias for encoder
        self.b_d = nn.Parameter(torch.zeros(act_dim))
        self.W_e = nn.Parameter(torch.empty(dict_dim, act_dim))
        self.b_e = nn.Parameter(torch.zeros(dict_dim))
        self.W_d = nn.Parameter(torch.empty(act_dim, dict_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.W_e, a=0.0, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.W_d, a=0.0, nonlinearity="relu")
        # Normalize decoder columns to unit norm
        with torch.no_grad():
            self.W_d.data = F.normalize(self.W_d.data, dim=0)

    @torch.no_grad()
    def normalize_decoder(self):
        """Projects decoder dictionary columns onto unit hypersphere."""
        self.W_d.data = F.normalize(self.W_d.data, dim=0)

    def encode(self, h):
        """
        h: (B, act_dim)
        Returns: z (B, dict_dim) sparse feature activations
        """
        h_centered = h - self.b_d
        z = F.relu(torch.matmul(h_centered, self.W_e.t()) + self.b_e)
        return z

    def decode(self, z):
        """
        z: (B, dict_dim)
        Returns: h_hat (B, act_dim) reconstructed activation
        """
        h_hat = torch.matmul(z, self.W_d.t()) + self.b_d
        return h_hat

    def forward(self, h):
        z = self.encode(h)
        h_hat = self.decode(z)
        return h_hat, z

    def compute_loss(self, h):
        h_hat, z = self(h)
        recon_loss = F.mse_loss(h_hat, h, reduction="mean")
        l1_loss = torch.mean(torch.sum(torch.abs(z), dim=-1))
        total_loss = recon_loss + self.l1_coeff * l1_loss

        # Calculate metrics
        with torch.no_grad():
            # Variance explained (R^2)
            total_var = torch.var(h, dim=0).sum() + 1e-8
            resid_var = torch.var(h - h_hat, dim=0).sum()
            r2_score = float(max(0.0, 1.0 - (resid_var / total_var).item()))
            # L0 sparsity (average active features per token)
            l0_sparsity = float(torch.mean((z > 0.0).float().sum(dim=-1)).item())
            # Active features in batch
            active_features = (z > 0.0).any(dim=0).sum().item()

        metrics = {
            "loss_total": total_loss.item(),
            "loss_recon": recon_loss.item(),
            "loss_l1": l1_loss.item(),
            "r2_score": r2_score,
            "l0_sparsity": l0_sparsity,
            "active_features": active_features
        }
        return total_loss, metrics
