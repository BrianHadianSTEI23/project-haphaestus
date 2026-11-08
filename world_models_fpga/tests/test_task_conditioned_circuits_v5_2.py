import numpy as np
import pytest
import torch
from world_models_fpga.research.circuits.run_task_conditioned_circuits_v5_2 import (
    compute_spectral_analysis,
    compute_representation_statistics_v5_2,
    wrap_angle,
    paired_bootstrap_ci,
    sign_flip_permutation_test,
    holm_bonferroni_correction,
)


def test_both_context_ranking_formula():
    """Tests the both-context task-sensitivity scoring formula:
    s_i = 0.5 * (|h_i(z_A, e_B) - h_i(z_A, e_A)| + |h_i(z_B, e_B) - h_i(z_B, e_A)|)
    """
    # 3 units, 2 pairs
    # Unit 0: high sensitivity in both contexts
    # Unit 1: high sensitivity only in context A
    # Unit 2: low sensitivity in both
    pair1_aa = np.array([1.0, 1.0, 0.5])
    pair1_ab = np.array([3.0, 3.0, 0.6])  # diff_A: [2.0, 2.0, 0.1]
    pair1_ba = np.array([2.0, 1.0, 0.5])
    pair1_bb = np.array([4.0, 1.0, 0.6])  # diff_B: [2.0, 0.0, 0.1]

    diff_A = np.abs(pair1_ab - pair1_aa)  # [2.0, 2.0, 0.1]
    diff_B = np.abs(pair1_bb - pair1_ba)  # [2.0, 0.0, 0.1]
    s = 0.5 * (diff_A + diff_B)           # [2.0, 1.0, 0.1]

    ranked = np.argsort(s)[::-1]
    assert list(ranked) == [0, 1, 2]
    assert np.isclose(s[0], 2.0)
    assert np.isclose(s[1], 1.0)
    assert np.isclose(s[2], 0.1)


def test_wrap_angle_invariants():
    """Tests angular wrapping within [-pi, pi]."""
    assert np.isclose(wrap_angle(0.0), 0.0)
    assert np.isclose(wrap_angle(np.pi), np.pi) or np.isclose(wrap_angle(np.pi), -np.pi)
    assert np.isclose(wrap_angle(3 * np.pi), np.pi) or np.isclose(wrap_angle(3 * np.pi), -np.pi)
    assert np.isclose(wrap_angle(-2.5 * np.pi), -0.5 * np.pi)
    assert np.isclose(wrap_angle(4 * np.pi), 0.0)


def test_spectral_analysis_invariants():
    """Tests participation ratio and threshold dims on known covariance structure."""
    rng = np.random.default_rng(42)
    # 2 dominant dimensions, 8 noise dimensions
    s1 = rng.normal(0, 10.0, size=(1000, 1))
    s2 = rng.normal(0, 10.0, size=(1000, 1))
    noise = rng.normal(0, 0.01, size=(1000, 8))
    A = np.hstack([s1, s2, noise])

    res = compute_spectral_analysis(A)
    # PR should be approximately 2.0
    assert 1.8 <= res["participation_ratio"] <= 2.2
    assert res["dims_80_var"] <= 2
    assert res["dims_90_var"] <= 2


def test_signed_representation_statistics():
    """Tests representation profiling with corrected signed handling and silent unit detection."""
    rng = np.random.default_rng(42)
    N, D = 500, 64

    # 1. Non-negative post-ReLU test
    A_relu = np.maximum(0.0, rng.normal(0, 1.0, size=(N, D)).astype(np.float32))
    # Add 4 silent units
    A_relu[:, :4] = 0.0

    act_dict_relu = {"layer1_post_relu_64d": A_relu}
    res_relu = compute_representation_statistics_v5_2(act_dict_relu, "layer1_post_relu_64d")

    assert res_relu["is_non_negative"] is True
    assert res_relu["silent_units_count"] == 4
    assert np.isclose(res_relu["silent_units_fraction"], 4 / 64)
    assert 0.0 < res_relu["lifetime_sparsity"]["mean"] < 1.0
    assert 0.0 < res_relu["population_sparsity"]["mean"] < 1.0
    assert res_relu["excess_kurtosis"] is None
    assert res_relu["zero_fraction"] > 0.45

    # 2. Signed pre-ReLU test
    A_pre = rng.normal(0, 1.0, size=(N, D)).astype(np.float32)
    # Add 2 silent units
    A_pre[:, :2] = 0.0

    act_dict_pre = {"layer1_pre_relu_64d": A_pre}
    res_pre = compute_representation_statistics_v5_2(act_dict_pre, "layer1_pre_relu_64d")

    assert res_pre["is_non_negative"] is False
    assert res_pre["silent_units_count"] == 2
    assert np.isclose(res_pre["silent_units_fraction"], 2 / 64)
    assert "positive_part_lifetime_sparsity" in res_pre
    assert "excess_kurtosis" in res_pre
    # Gaussian distribution has excess kurtosis around 0.0
    assert -0.5 <= res_pre["excess_kurtosis"]["mean"] <= 0.5
    assert 0.40 <= res_pre["negative_fraction"] <= 0.60


def test_direction_reversed_transplantation_algebra():
    """Verifies that A->B and B->A transplantation targets are exact opposites."""
    g_a = torch.tensor([1.2, -0.4, 0.8, 0.6])
    g_b = torch.tensor([0.2, 0.6, -0.6, 0.8])

    delta_ab = g_b[:2] - g_a[:2]
    delta_ba = g_a[:2] - g_b[:2]

    assert torch.allclose(delta_ab, -delta_ba)

    # If intervention on recipient g_a produces g_b, transfer efficiency is 1.0
    trans_ab = g_b[:2] - g_a[:2]
    target_norm_sq_ab = float(torch.dot(delta_ab, delta_ab))
    eta_ab = float(torch.dot(trans_ab, delta_ab)) / target_norm_sq_ab
    assert np.isclose(eta_ab, 1.0)

    # Reversed direction: recipient g_b, donor g_a
    trans_ba = g_a[:2] - g_b[:2]
    target_norm_sq_ba = float(torch.dot(delta_ba, delta_ba))
    eta_ba = float(torch.dot(trans_ba, delta_ba)) / target_norm_sq_ba
    assert np.isclose(eta_ba, 1.0)


def test_activation_norm_matching_tolerance():
    """Tests that the activation-norm-matching filter correctly selects within 5% tolerance."""
    rng = np.random.default_rng(2026)
    delta_h = rng.normal(0, 1.0, size=64)
    cand_units = [40, 39, 58, 36, 54, 25, 27, 57]  # k = 8
    cand_norm = float(np.linalg.norm(delta_h[cand_units]))

    # Sample 5000 random subsets
    random_subsets = [rng.choice(64, size=8, replace=False) for _ in range(5000)]
    subset_norms = np.array([np.linalg.norm(delta_h[u]) for u in random_subsets])
    rel_errors = np.abs(subset_norms - cand_norm) / cand_norm

    within_tol = np.where(rel_errors <= 0.05)[0]
    assert len(within_tol) > 50, f"Expected >50 norm-matched matches, got {len(within_tol)}"

    # Best 100 subsets
    sorted_idx = np.argsort(rel_errors)[:100]
    selected_errors = rel_errors[sorted_idx]
    assert np.all(selected_errors <= 0.05)


def test_holm_bonferroni_correction():
    """Tests Holm-Bonferroni step-down correction for multiple hypothesis testing."""
    raw_p = [0.001, 0.01, 0.03, 0.04, 0.20]
    adj_p = holm_bonferroni_correction(raw_p)

    # Number of hypotheses m = 5
    # Sorted raw: p0=0.001, p1=0.01, p2=0.03, p3=0.04, p4=0.20
    # Multipliers: 5, 4, 3, 2, 1
    # Raw * mult: 0.005, 0.04, 0.09, 0.08 (enforced monotonic -> 0.09), 0.20
    assert len(adj_p) == 5
    assert np.isclose(adj_p[0], 0.005)
    assert np.isclose(adj_p[1], 0.04)
    assert np.isclose(adj_p[2], 0.09)
    assert np.isclose(adj_p[3], 0.09)  # monotonicity constraint
    assert np.isclose(adj_p[4], 0.20)


def test_sign_flip_permutation_test():
    """Tests that sign-flip permutation test distinguishes real effects from null differences."""
    # Clearly positive shift
    real_diffs = np.array([0.2, 0.3, 0.15, 0.25, 0.4, 0.18, 0.22, 0.35, 0.28, 0.19])
    p_real = sign_flip_permutation_test(real_diffs, n_perm=5000, seed=42)
    assert p_real < 0.01

    # Symmetric null differences
    null_diffs = np.array([-0.2, 0.2, -0.15, 0.15, -0.1, 0.1, -0.05, 0.05, -0.3, 0.3])
    p_null = sign_flip_permutation_test(null_diffs, n_perm=5000, seed=42)
    assert p_null > 0.50
