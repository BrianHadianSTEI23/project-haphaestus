"""Unit test verifying mathematical invariants of activation patching and subspace intervention."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from research.circuits.run_rigorous_behavioral_circuits import (
    sample_random_subspaces,
    SubspaceTransformHook,
)


def test_subspace_projection_orthogonality():
    """Verify that projection matrix P is idempotent, symmetric, and orthogonal to (I - P)."""
    subspaces = sample_random_subspaces(n_draws=5, k=8, dim=32, seed=42)
    for p_np in subspaces:
        P = torch.from_numpy(p_np).float()
        I = torch.eye(32)

        # Symmetry: P == P^T
        torch.testing.assert_close(P, P.T, atol=1e-5, rtol=1e-5)

        # Idempotence: P @ P == P
        torch.testing.assert_close(P @ P, P, atol=1e-5, rtol=1e-5)

        # Orthogonal complement: P @ (I - P) == 0
        P_orth = I - P
        torch.testing.assert_close(P @ P_orth, torch.zeros_like(P), atol=1e-5, rtol=1e-5)


def test_activation_patching_mathematical_invariants():
    """
    Verify that genuine activation patching strictly satisfies:
      z_patched @ P == z_donor @ P
      z_patched @ (I - P) == z_recip @ (I - P)
    """
    rng = np.random.default_rng(123)
    p_np = sample_random_subspaces(n_draws=1, k=8, dim=32, seed=123)[0]
    P = torch.from_numpy(p_np).float()
    I = torch.eye(32)
    P_orth = I - P

    z_recip = torch.from_numpy(rng.standard_normal((10, 32))).float()
    z_donor = torch.from_numpy(rng.standard_normal((10, 32))).float()

    # Activation patching formula:
    # z_patched = z_recip @ (I - P) + z_donor @ P
    z_patched = (z_recip @ P_orth) + (z_donor @ P)

    # Invariant 1: In the target subspace P, z_patched matches z_donor
    patched_in_sub = z_patched @ P
    donor_in_sub = z_donor @ P
    torch.testing.assert_close(patched_in_sub, donor_in_sub, atol=1e-5, rtol=1e-5)

    # Invariant 2: In the orthogonal complement (I - P), z_patched matches z_recip
    patched_in_orth = z_patched @ P_orth
    recip_in_orth = z_recip @ P_orth
    torch.testing.assert_close(patched_in_orth, recip_in_orth, atol=1e-5, rtol=1e-5)


def test_subspace_transform_hook_execution():
    """Verify SubspaceTransformHook properly intercepts encode."""
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(16, 32)
        def encode(self, x):
            return self.fc(x)

    model = DummyModel()
    p_np = np.zeros((32, 32), dtype=np.float32)
    p_np[:8, :8] = np.eye(8, dtype=np.float32)  # top 8 dims

    x = torch.randn(4, 16)
    z_orig = model.encode(x)

    # 1. Ablate mode: dims 0..7 become zero
    with SubspaceTransformHook(model, mode="ablate", proj=p_np, mean_vec=np.zeros(32)):
        z_abl = model.encode(x)
        assert torch.all(z_abl[:, :8] == 0.0)
        torch.testing.assert_close(z_abl[:, 8:], z_orig[:, 8:])

    # 2. Retain mode: dims 8..31 become zero
    with SubspaceTransformHook(model, mode="retain", proj=p_np, mean_vec=np.zeros(32)):
        z_ret = model.encode(x)
        torch.testing.assert_close(z_ret[:, :8], z_orig[:, :8])
        assert torch.all(z_ret[:, 8:] == 0.0)

    # 3. None mode: passes through untouched
    with SubspaceTransformHook(model, mode="none"):
        z_none = model.encode(x)
        torch.testing.assert_close(z_none, z_orig)
