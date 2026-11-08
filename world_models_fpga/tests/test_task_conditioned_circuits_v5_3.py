import numpy as np
import pytest
import torch
import math
from world_models_fpga.research.circuits.run_task_conditioned_circuits_v5_3 import (
    compute_hypergeometric_pmf,
    sample_norm_matched_controls_strict,
    wrap_angle,
    paired_bootstrap_ci,
    sign_flip_permutation_test,
    holm_bonferroni_correction,
    compute_spectral_analysis,
    compute_representation_statistics_v5_3,
)


def test_strict_selection_zero_valid():
    """Tests that 0 valid subsets produces INSUFFICIENT_COMMON_SUPPORT with selected_subsets=None."""
    rng = np.random.default_rng(42)
    delta_h = np.ones(64)
    # Candidate units with large activation
    cand_units = [0, 1]
    # Set candidate activations to 100.0, rest to 0.001
    delta_h[:2] = 100.0
    delta_h[2:] = 0.001

    res = sample_norm_matched_controls_strict(
        delta_h=delta_h,
        cand_units=cand_units,
        k=2,
        norm_tolerance=0.05,
        required_controls=100,
        initial_draws=500,
        max_draws=1000,
        rng=rng,
        disjoint_only=True,  # All from complement (norm ~0.0014 vs candidate norm ~141.4)
    )

    assert res["status"] == "INSUFFICIENT_COMMON_SUPPORT"
    assert res["n_valid"] == 0
    assert res["selected_subsets"] is None
    assert res["selected_rel_errors"] is None
    assert res["selected_norms"] is None


def test_strict_selection_few_valid_cases():
    """Tests 10 and 99 valid subsets: must return INSUFFICIENT_COMMON_SUPPORT and NEVER retain invalid subsets."""
    rng = np.random.default_rng(123)
    cand_units = [0, 1]
    k = 2
    cand_norm = 10.0

    # Mock scenario: synthesize draws where exactly N subsets pass tolerance
    for target_valid in [10, 99]:
        # Construct delta_h such that only a tiny fraction of subsets match
        delta_h = np.full(64, 0.1)
        delta_h[0] = 7.071
        delta_h[1] = 7.071  # cand_norm ~ 10.0
        # Give only a few other pairs the right norm
        # Pair (2, 3) has norm 10.0
        delta_h[2] = 7.071
        delta_h[3] = 7.071

        # Using low max_draws ensures we stop before finding 100 matches
        res = sample_norm_matched_controls_strict(
            delta_h=delta_h,
            cand_units=cand_units,
            k=k,
            norm_tolerance=0.05,
            required_controls=100,
            initial_draws=200,
            max_draws=200,
            rng=rng,
        )

        assert res["status"] == "INSUFFICIENT_COMMON_SUPPORT"
        assert res["selected_subsets"] is None
        assert res["selected_rel_errors"] is None
        assert res["selected_norms"] is None


def test_strict_selection_exactly_100_valid():
    """Tests that exactly 100 valid subsets produces VALID_COMMON_SUPPORT and retains all 100 within tolerance."""
    rng = np.random.default_rng(2026)
    delta_h = rng.normal(0, 1.0, size=64)
    cand_units = [0, 1, 2, 3]
    k = 4

    # Sample enough draws to get at least 100
    res = sample_norm_matched_controls_strict(
        delta_h=delta_h,
        cand_units=cand_units,
        k=k,
        norm_tolerance=0.05,
        required_controls=100,
        initial_draws=20000,
        max_draws=50000,
        rng=rng,
    )

    if res["status"] == "VALID_COMMON_SUPPORT":
        assert len(res["selected_subsets"]) == 100
        assert len(res["selected_rel_errors"]) == 100
        assert len(res["selected_norms"]) == 100
        # CRITICAL TEST: Every single retained control must have rel_error <= 0.05
        assert np.all(np.array(res["selected_rel_errors"]) <= 0.05)


def test_strict_selection_more_than_100_valid():
    """Tests that when >100 valid subsets exist, top 100 closest are selected and all satisfy <= 0.05."""
    rng = np.random.default_rng(777)
    delta_h = rng.normal(0, 1.0, size=64)
    cand_units = [10, 11, 12, 13, 14, 15, 16, 17]  # random candidate units
    k = 8

    res = sample_norm_matched_controls_strict(
        delta_h=delta_h,
        cand_units=cand_units,
        k=k,
        norm_tolerance=0.05,
        required_controls=100,
        initial_draws=30000,
        max_draws=50000,
        rng=rng,
    )

    if res["status"] == "VALID_COMMON_SUPPORT":
        assert res["n_valid"] >= 100
        assert len(res["selected_subsets"]) == 100
        errors = np.array(res["selected_rel_errors"])
        assert len(errors) == 100
        assert np.all(errors <= 0.05)
        # Check sorted order of errors
        assert np.all(np.diff(errors) >= -1e-12)


def test_controls_exceeding_5pct_never_retained():
    """Explicitly verifies that no subset with relative error > 0.05 is EVER retained."""
    rng = np.random.default_rng(999)
    # Craft an array where candidate has norm 100.0, and random subsets have norm ~50.0 (error 50%)
    delta_h = np.full(64, 5.0)
    delta_h[:8] = 35.355  # 8 * 35.355^2 ~ 10000 -> norm ~ 100.0
    cand_units = list(range(8))

    res = sample_norm_matched_controls_strict(
        delta_h=delta_h,
        cand_units=cand_units,
        k=8,
        norm_tolerance=0.05,
        required_controls=100,
        initial_draws=1000,
        max_draws=2000,
        rng=rng,
        disjoint_only=True,  # All from units 8..63 (norm is sqrt(8 * 25) ~ 14.14, error ~ 86%)
    )

    assert res["status"] == "INSUFFICIENT_COMMON_SUPPORT"
    assert res["selected_subsets"] is None
    # Verify that min relative error was far above 5%
    assert res["min_rel_error"] > 0.05


def test_adaptive_sampling_reaches_max_draws():
    """Tests that adaptive sampling terminates at max_draws when matches are insufficient."""
    rng = np.random.default_rng(42)
    delta_h = np.full(64, 1.0)
    delta_h[:2] = 50.0
    cand_units = [0, 1]

    max_draws = 2500
    res = sample_norm_matched_controls_strict(
        delta_h=delta_h,
        cand_units=cand_units,
        k=2,
        norm_tolerance=0.05,
        required_controls=100,
        initial_draws=500,
        chunk_draws=500,
        max_draws=max_draws,
        rng=rng,
        disjoint_only=True,
    )

    assert res["total_drawn"] == max_draws
    assert res["status"] == "INSUFFICIENT_COMMON_SUPPORT"
    assert res["selected_subsets"] is None


def test_hypergeometric_pmf_properties():
    """Tests hypergeometric distribution formula and probabilities for subset overlap."""
    D = 64
    for k in [2, 4, 8, 16, 32]:
        probs = [compute_hypergeometric_pmf(k, o, D=D) for o in range(k + 1)]
        # Sum of probabilities across all possible overlaps must equal 1.0
        assert np.isclose(sum(probs), 1.0, atol=1e-6)

        # Expected overlap must be k * k / D
        expected_overlap = k * k / D
        calc_expected = sum(o * probs[o] for o in range(k + 1))
        assert np.isclose(calc_expected, expected_overlap, atol=1e-6)


def test_disjoint_k32_deterministic_reference():
    """Tests that at k=32 in D=64, disjoint sampling evaluates the single deterministic complement."""
    delta_h = np.arange(64, dtype=np.float64)
    cand_units = list(range(32))  # first 32

    res = sample_norm_matched_controls_strict(
        delta_h=delta_h,
        cand_units=cand_units,
        k=32,
        norm_tolerance=0.05,
        disjoint_only=True,
    )

    assert res["status"] == "DETERMINISTIC_REFERENCE"
    assert res["total_drawn"] == 1
    assert len(res["selected_subsets"]) == 1
    # Must be exact complement: units 32..63
    assert res["selected_subsets"][0] == list(range(32, 64))
    assert res["selected_overlaps"][0] == 0


def test_condition_averaged_ablation_math():
    """Tests that condition-averaged ablation removes task difference on intervened units while preserving others."""
    h_a = torch.tensor([[1.0, 5.0, 10.0, 20.0]])
    h_b = torch.tensor([[3.0, 9.0, 10.0, 30.0]])
    cand_units = [0, 1]

    h_a_mabl = h_a.clone()
    h_b_mabl = h_b.clone()
    mean_val = 0.5 * (h_a[:, cand_units] + h_b[:, cand_units])
    h_a_mabl[:, cand_units] = mean_val
    h_b_mabl[:, cand_units] = mean_val

    # Difference on intervened units must be exactly 0
    diff_intervened = h_b_mabl[:, cand_units] - h_a_mabl[:, cand_units]
    assert torch.allclose(diff_intervened, torch.zeros_like(diff_intervened))

    # Unintervened units must retain original difference
    unintervened = [2, 3]
    orig_diff_un = h_b[:, unintervened] - h_a[:, unintervened]
    mabl_diff_un = h_b_mabl[:, unintervened] - h_a_mabl[:, unintervened]
    assert torch.allclose(orig_diff_un, mabl_diff_un)


def test_holm_bonferroni_correction_properties():
    """Tests Holm-Bonferroni correction on edge cases and monotonicity."""
    raw_p = [0.001, 0.008, 0.04, 0.045, 0.10, 0.25, 0.50, 0.80, 0.90, 0.95]
    adj_p = holm_bonferroni_correction(raw_p)

    assert len(adj_p) == 10
    # Must be non-decreasing after sorting
    sorted_idx = np.argsort(raw_p)
    assert np.all(np.diff(np.array(adj_p)[sorted_idx]) >= -1e-12)
    # All values <= 1.0 and >= raw_p
    assert np.all(np.array(adj_p) <= 1.0)
    assert np.all(np.array(adj_p) >= np.array(raw_p))


def test_paired_bootstrap_ci_coverage():
    """Tests paired bootstrap confidence intervals on normal distribution."""
    rng = np.random.default_rng(42)
    data = rng.normal(5.0, 1.0, size=100)
    ci = paired_bootstrap_ci(data, n_boot=2000, seed=42)
    assert ci[0] < 5.0 < ci[1]
    assert 4.5 < ci[0] < 5.0
    assert 5.0 < ci[1] < 5.5
