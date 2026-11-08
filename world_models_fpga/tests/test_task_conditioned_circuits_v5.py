import numpy as np
import pytest
from world_models_fpga.research.circuits.run_task_conditioned_circuits_v5 import (
    compute_treves_rolls_lifetime_sparsity,
    compute_population_sparsity,
    compute_spectral_analysis,
    compute_zero_activation_fraction,
    wrap_angle,
)


def test_treves_rolls_sparsity_invariants():
    """Tests lifetime and population sparseness calculations against known mathematical limits."""
    # 1. Constant activations (all 1s) -> sparsity should be 0.0
    A_ones = np.ones((100, 10), dtype=np.float32)
    life_ones = compute_treves_rolls_lifetime_sparsity(A_ones)
    assert np.isclose(life_ones["mean_lifetime_sparsity"], 0.0, atol=1e-5)
    assert np.isclose(life_ones["mean_raw_ratio"], 1.0, atol=1e-5)

    pop_ones = compute_population_sparsity(A_ones)
    assert np.isclose(pop_ones["mean_population_sparsity"], 0.0, atol=1e-5)

    # 2. Perfectly sparse (one-hot across neurons) -> sparsity should approach 1.0
    A_sparse = np.zeros((100, 10), dtype=np.float32)
    A_sparse[0, :] = 1.0  # only 1 active observation out of 100
    life_sparse = compute_treves_rolls_lifetime_sparsity(A_sparse)
    assert life_sparse["mean_lifetime_sparsity"] > 0.95


def test_spectral_analysis_invariants():
    """Tests participation ratio and threshold dims on known covariance structure."""
    # 100 observations of 10 dimensions with known independent variance
    rng = np.random.default_rng(42)
    # 2 dominant dimensions, 8 noise dimensions
    s1 = rng.normal(0, 10.0, size=(1000, 1))
    s2 = rng.normal(0, 10.0, size=(1000, 1))
    noise = rng.normal(0, 0.01, size=(1000, 8))
    A = np.hstack([s1, s2, noise])

    res = compute_spectral_analysis(A)
    # PR should be approximately 2.0
    assert 1.8 <= res["participation_ratio"] <= 2.2
    # Threshold dims: 50% and 80% should be <= 2
    assert res["threshold_dims"]["dim_var_50pct"] <= 2
    assert res["threshold_dims"]["dim_var_80pct"] <= 2


def test_wrap_angle_invariants():
    """Tests angular wrapping within [-pi, pi]."""
    assert np.isclose(wrap_angle(0.0), 0.0)
    assert np.isclose(wrap_angle(np.pi), np.pi) or np.isclose(wrap_angle(np.pi), -np.pi)
    assert np.isclose(wrap_angle(3 * np.pi), np.pi) or np.isclose(wrap_angle(3 * np.pi), -np.pi)
    assert np.isclose(wrap_angle(-2.5 * np.pi), -0.5 * np.pi)
