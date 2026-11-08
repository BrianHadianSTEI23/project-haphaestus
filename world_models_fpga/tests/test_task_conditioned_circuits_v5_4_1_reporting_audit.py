"""Unit tests for Phase 5.4.1 forensic reconciliation, statistical reporting,

and intelligence-principle corrections.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from world_models_fpga.research.circuits.run_task_conditioned_circuits_v5_4_1_reporting_audit import (
    compute_waypoint_from_object_pose,
    compute_vector_cosine,
    compute_scene_swap_reversal_metrics,
    interpret_scene_swap_reversal,
    compute_expected_intervention_keys,
    audit_intervention_reconciliation,
    compute_pair_level_pooled_candidate_metrics,
    compute_cohort_matched_comparisons,
    generate_summary_table_markdown,
)


# -----------------------------------------------------------------------------
# 1. Scene-Swap Reversal Sign & Ideal Cases
# -----------------------------------------------------------------------------

def test_scene_swap_reversal_sign_and_ideal_cases():
    """Verifies that:

    - d_B = -d_A produces reversal cosine = +1.0 (ideal reversal)
    - d_B = d_A produces reversal cosine = -1.0 (no reversal / invariant orientation)
    - d_B orthogonal to d_A produces reversal cosine = 0.0
    """
    d_a = np.array([0.5, 0.2])

    # Ideal reversal: d_b = -d_a
    d_b_rev = -d_a
    cos_rev, ang_rev = compute_scene_swap_reversal_metrics(d_a, d_b_rev)
    assert np.isclose(cos_rev, 1.0, atol=1e-6)
    assert np.isclose(ang_rev, 0.0, atol=1e-5)
    assert "predominantly reverses" in interpret_scene_swap_reversal(cos_rev)

    # Invariant contrast (no reversal): d_b = d_a
    d_b_same = d_a.copy()
    cos_same, ang_same = compute_scene_swap_reversal_metrics(d_a, d_b_same)
    assert np.isclose(cos_same, -1.0, atol=1e-6)
    assert np.isclose(ang_same, 180.0, atol=1e-5)
    assert "fails to reverse" in interpret_scene_swap_reversal(cos_same)

    # Orthogonal contrast: d_b = [-d_a[1], d_a[0]]
    d_b_orth = np.array([-d_a[1], d_a[0]])
    cos_orth, ang_orth = compute_scene_swap_reversal_metrics(d_a, d_b_orth)
    assert np.isclose(cos_orth, 0.0, atol=1e-6)
    assert np.isclose(ang_orth, 90.0, atol=1e-5)
    assert "approximately orthogonal" in interpret_scene_swap_reversal(cos_orth)


# -----------------------------------------------------------------------------
# 2. Expected Row Counts & Reconciliation Math
# -----------------------------------------------------------------------------

def test_expected_row_counts_mathematical_derivation():
    """Verifies:

    - 2,400 candidate rows (60 pairs x 2 scenes x 2 layers x 5 k x 2 directions)
    - 240,000 dimension-matched rows (2,400 settings x 100 draws)
    - Norm-matched rows derived from Phase 5.3 eligibility (2,054 valid settings x 100 = 205,400)
    - Exact sum = 447,800 records
    """
    pairs_file = Path("world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_3_pairs.jsonl")
    assert pairs_file.exists()

    pairs_data = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            pairs_data.append(json.loads(line))

    assert len(pairs_data) == 60

    expected_keys = compute_expected_intervention_keys(pairs_data)
    cand_keys = {k for k in expected_keys if k[5] == "candidate"}
    dim_keys = {k for k in expected_keys if k[5] == "dimension_matched_random"}
    norm_keys = {k for k in expected_keys if k[5] == "norm_matched_random"}

    assert len(cand_keys) == 2400
    assert len(dim_keys) == 240000
    assert len(norm_keys) == 205400
    assert len(expected_keys) == 447800


# -----------------------------------------------------------------------------
# 3. Reconciliation Anomaly Detection (Missing, Duplicate, Extra, Unmatched)
# -----------------------------------------------------------------------------

def test_reconciliation_anomaly_detection():
    """Synthetic tests verifying that missing, duplicate, extra, and unmatched rows

    are strictly detected and fail the reconciliation audit.
    """
    # Create minimal synthetic expected keys (10 keys)
    synthetic_expected = {
        ("PAIR_000", "scene_a", "layer1_post_relu_64d", "A_to_B", 2, "candidate", 0),
        ("PAIR_000", "scene_a", "layer1_post_relu_64d", "A_to_B", 2, "dimension_matched_random", 0),
        ("PAIR_000", "scene_a", "layer1_post_relu_64d", "A_to_B", 2, "dimension_matched_random", 1),
    }

    # Case A: Perfect stream
    perfect_records = [
        {"pair_id": k[0], "scene_id": k[1], "layer": k[2], "direction": k[3], "k": k[4], "condition": k[5], "draw_id": k[6]}
        for k in synthetic_expected
    ]
    res_perfect = audit_intervention_reconciliation(synthetic_expected, perfect_records)
    assert res_perfect["reconciliation_success"] is True
    assert res_perfect["missing_rows_count"] == 0
    assert res_perfect["duplicate_rows_count"] == 0
    assert res_perfect["unexpected_rows_count"] == 0

    # Case B: Missing row
    records_missing = perfect_records[:-1]
    res_missing = audit_intervention_reconciliation(synthetic_expected, records_missing)
    assert res_missing["reconciliation_success"] is False
    assert res_missing["missing_rows_count"] == 1

    # Case C: Duplicate row
    records_dup = perfect_records + [perfect_records[0]]
    res_dup = audit_intervention_reconciliation(synthetic_expected, records_dup)
    assert res_dup["reconciliation_success"] is False
    assert res_dup["duplicate_rows_count"] == 1

    # Case D: Unexpected / Extra row
    records_extra = perfect_records + [
        {"pair_id": "PAIR_000", "scene_id": "scene_a", "layer": "layer1_post_relu_64d", "direction": "A_to_B", "k": 2, "condition": "dimension_matched_random", "draw_id": 999}
    ]
    res_extra = audit_intervention_reconciliation(synthetic_expected, records_extra)
    assert res_extra["reconciliation_success"] is False
    assert res_extra["unexpected_rows_count"] == 1

    # Case E: Unmatched pair ID
    records_unmatched = [
        {"pair_id": "PAIR_999", "scene_id": "scene_a", "layer": "layer1_post_relu_64d", "direction": "A_to_B", "k": 2, "condition": "candidate", "draw_id": 0}
    ]
    res_unmatched = audit_intervention_reconciliation(synthetic_expected, records_unmatched)
    assert res_unmatched["reconciliation_success"] is False
    assert res_unmatched["unmatched_pairs_count"] == 1


# -----------------------------------------------------------------------------
# 4. Cohort-Matched Pair IDs & Paired Difference Invariant
# -----------------------------------------------------------------------------

def test_cohort_matched_pair_ids_and_paired_difference():
    """Verifies that for every layer and k:

    - Candidate and norm-matched eligible means use identical pair IDs
    - paired_difference == candidate_eligible_mean - norm_matched_eligible_mean
    """
    all_pids = [f"PAIR_{i:03d}" for i in range(10)]
    eligible_pids = {"PAIR_000", "PAIR_001", "PAIR_002", "PAIR_003"}  # 4/10 eligible

    pair_metrics_by_lk = {
        ("layer1_post_relu_64d", 8): {
            pid: {
                "cand_eta": float(i * 0.1),
                "cand_imp": float(i * 0.01),
                "cand_eta_model": float(i * 0.2),
                "norm_eta": float(i * 0.05) if pid in eligible_pids else None,
                "norm_imp": float(i * 0.02) if pid in eligible_pids else None,
            }
            for i, pid in enumerate(all_pids)
        },
        ("layer1_post_relu_64d", 2): {
            pid: {
                "cand_eta": float(i * 0.1),
                "cand_imp": float(i * 0.01),
                "cand_eta_model": float(i * 0.2),
                "norm_eta": float(i * 0.05),
                "norm_imp": float(i * 0.02),
            }
            for i, pid in enumerate(all_pids)
        },
        ("layer1_post_relu_64d", 4): {
            pid: {
                "cand_eta": 0.1, "cand_imp": 0.01, "cand_eta_model": 0.2, "norm_eta": 0.05, "norm_imp": 0.02
            } for pid in all_pids
        },
        ("layer1_post_relu_64d", 16): {
            pid: {
                "cand_eta": 0.1, "cand_imp": 0.01, "cand_eta_model": 0.2, "norm_eta": 0.05, "norm_imp": 0.02
            } for pid in all_pids
        },
        ("layer1_post_relu_64d", 32): {
            pid: {
                "cand_eta": 0.1, "cand_imp": 0.01, "cand_eta_model": 0.2, "norm_eta": 0.05, "norm_imp": 0.02
            } for pid in all_pids
        },
        ("layer2_post_relu_64d", 2): {
            pid: {
                "cand_eta": 0.1, "cand_imp": 0.01, "cand_eta_model": 0.2, "norm_eta": 0.05, "norm_imp": 0.02
            } for pid in all_pids
        },
        ("layer2_post_relu_64d", 4): {
            pid: {
                "cand_eta": 0.1, "cand_imp": 0.01, "cand_eta_model": 0.2, "norm_eta": 0.05, "norm_imp": 0.02
            } for pid in all_pids
        },
        ("layer2_post_relu_64d", 8): {
            pid: {
                "cand_eta": 0.1, "cand_imp": 0.01, "cand_eta_model": 0.2, "norm_eta": 0.05, "norm_imp": 0.02
            } for pid in all_pids
        },
        ("layer2_post_relu_64d", 16): {
            pid: {
                "cand_eta": 0.1, "cand_imp": 0.01, "cand_eta_model": 0.2, "norm_eta": 0.05, "norm_imp": 0.02
            } for pid in all_pids
        },
        ("layer2_post_relu_64d", 32): {
            pid: {
                "cand_eta": 0.1, "cand_imp": 0.01, "cand_eta_model": 0.2, "norm_eta": 0.05, "norm_imp": 0.02
            } for pid in all_pids
        },
    }

    eligible_pids_by_lk = {
        ("layer1_post_relu_64d", 8): eligible_pids,
        ("layer1_post_relu_64d", 2): set(all_pids),
        ("layer1_post_relu_64d", 4): set(all_pids),
        ("layer1_post_relu_64d", 16): set(all_pids),
        ("layer1_post_relu_64d", 32): set(all_pids),
        ("layer2_post_relu_64d", 2): set(all_pids),
        ("layer2_post_relu_64d", 4): set(all_pids),
        ("layer2_post_relu_64d", 8): set(all_pids),
        ("layer2_post_relu_64d", 16): set(all_pids),
        ("layer2_post_relu_64d", 32): set(all_pids),
    }

    comparisons = compute_cohort_matched_comparisons(pair_metrics_by_lk, eligible_pids_by_lk, all_pids)
    c8 = comparisons["layer1_post_relu_64d"]["k_8"]

    assert c8["support_status"] == "EXPLORATORY_INCOMPLETE_SUPPORT"
    assert c8["eligible_pairs_count"] == 4
    assert c8["coverage_pct"] == 0.4

    cm = c8["cohort_matched_comparison"]
    assert cm["eligible_pairs"] == sorted(list(eligible_pids))

    # Verify mathematical identity: paired_diff == cand_mean - norm_mean
    expected_diff_eta = cm["candidate_eligible_mean_eta_gt"] - cm["norm_matched_eligible_mean_eta_gt"]
    expected_diff_imp = cm["candidate_eligible_mean_error_improvement_m"] - cm["norm_matched_eligible_mean_error_improvement_m"]
    assert np.isclose(cm["paired_difference_eta_gt"], expected_diff_eta)
    assert np.isclose(cm["paired_difference_error_improvement_m"], expected_diff_imp)

    # Programmatic markdown generation check
    md_table = generate_summary_table_markdown(comparisons)
    assert "| Layer | $k$ |" in md_table
    assert "EXPLORATORY_INCOMPLETE_SUPPORT" in md_table
    assert "EXPLORATORY_COMPLETE_SUPPORT" in md_table


# -----------------------------------------------------------------------------
# 5. Pair-Level Pooled Candidate Metrics & Bootstrap Behavior
# -----------------------------------------------------------------------------

def test_pair_level_pooled_candidate_metrics_and_bootstrap():
    """Verifies that:

    - The unit of statistical inference is the independent pair (N=60)
    - Bootstrap confidence intervals contain the sample mean and have finite positive width
    """
    rng = np.random.default_rng(42)
    candidate_rows_by_pair = {}

    for i in range(60):
        pid = f"PAIR_{i:03d}"
        rows = []
        for _ in range(40):  # 40 settings per pair
            rows.append({
                "eta_gt": float(rng.normal(-0.07, 0.2)),
                "error_improvement_m": float(rng.normal(0.03, 0.1)),
                "eta_model": float(rng.normal(0.40, 0.15))
            })
        candidate_rows_by_pair[pid] = rows

    pooled = compute_pair_level_pooled_candidate_metrics(candidate_rows_by_pair, n_boot=1000, seed=2026)
    assert pooled["n_independent_pairs"] == 60
    assert pooled["n_total_candidate_rows"] == 2400

    pli = pooled["pair_level_inference"]

    # Check eta_gt
    eta_gt_stat = pli["eta_gt"]
    assert eta_gt_stat["ci_95"][0] <= eta_gt_stat["mean"] <= eta_gt_stat["ci_95"][1]
    assert eta_gt_stat["ci_95"][1] > eta_gt_stat["ci_95"][0]

    # Check error improvement
    err_imp_stat = pli["error_improvement_m"]
    assert err_imp_stat["ci_95"][0] <= err_imp_stat["mean"] <= err_imp_stat["ci_95"][1]

    # Check fraction positive
    pos_stat = pli["fraction_positive_improvement"]
    assert 0.0 <= pos_stat["ci_95"][0] <= pos_stat["mean"] <= pos_stat["ci_95"][1] <= 1.0


# -----------------------------------------------------------------------------
# 6. Waypoint Generation Equation & Standoff Offset (0.60 m)
# -----------------------------------------------------------------------------

def test_waypoint_generation_equation_and_standoff_distance():
    """Verifies that compute_waypoint_from_object_pose implements the exact standoff

    equation: wp = pos_obj - (pos_obj - robot_pose) / dist * 0.60 m.
    """
    robot_pose = np.array([0.0, 0.0])
    target_pos = np.array([2.0, 0.0])  # 2.0 m along x

    wp = compute_waypoint_from_object_pose(robot_pose, target_pos, standoff_m=0.60)

    # Robot to waypoint should be 1.40 m, waypoint to target should be 0.60 m
    assert np.isclose(wp[0], 1.40)
    assert np.isclose(wp[1], 0.00)
    assert np.isclose(np.linalg.norm(target_pos - wp), 0.60)
    assert np.isclose(np.linalg.norm(wp - robot_pose), 1.40)

    # Diagonal test
    robot_pose_diag = np.array([1.0, 1.0])
    target_pos_diag = np.array([4.0, 5.0])  # diff = (3, 4), dist = 5.0
    wp_diag = compute_waypoint_from_object_pose(robot_pose_diag, target_pos_diag, standoff_m=0.60)

    # Unit vector is (3/5, 4/5) = (0.6, 0.8)
    # Standoff offset = 0.60 * (0.6, 0.8) = (0.36, 0.48)
    # Expected waypoint = (4.0 - 0.36, 5.0 - 0.48) = (3.64, 4.52)
    assert np.isclose(wp_diag[0], 3.64)
    assert np.isclose(wp_diag[1], 4.52)
    assert np.isclose(np.linalg.norm(target_pos_diag - wp_diag), 0.60)
