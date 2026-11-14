"""Tests for Phase 5.4.2.1 Exact Intervention Key-Set Reconciliation Audit."""

from __future__ import annotations

import copy

from world_models_fpga.research.circuits.run_task_conditioned_circuits_v5_4_2_1_reconciliation_audit import (
    reconcile_intervention_keys,
)


def _make_synthetic_pairs_and_rows():
    pairs = []
    rows = []
    for pid in ("PAIR_000", "PAIR_001"):
        results = {}
        for sc in ("scene_a", "scene_b"):
            for layer in ("layer1_post_relu_64d", "layer2_post_relu_64d"):
                for direction in ("A_to_B", "B_to_A"):
                    for k in (2, 4):
                        is_valid = (layer == "layer1_post_relu_64d" and k == 2)
                        results[f"{sc}__{layer}__{direction}__k_{k}"] = {
                            "norm_matched_status": "VALID_COMMON_SUPPORT" if is_valid else "INSUFFICIENT_SUPPORT"
                        }
                        rows.append({
                            "pair_id": pid,
                            "scene_id": sc,
                            "layer": layer,
                            "direction": direction,
                            "k": k,
                            "condition": "candidate",
                            "draw_id": 0,
                        })
                        for d in range(3):
                            rows.append({
                                "pair_id": pid,
                                "scene_id": sc,
                                "layer": layer,
                                "direction": direction,
                                "k": k,
                                "condition": "dimension_matched_random",
                                "draw_id": d,
                            })
                            if is_valid:
                                rows.append({
                                    "pair_id": pid,
                                    "scene_id": sc,
                                    "layer": layer,
                                    "direction": direction,
                                    "k": k,
                                    "condition": "norm_matched_random",
                                    "draw_id": d,
                                })
        pairs.append({"pair_id": pid, "results": results})
    return pairs, rows


def test_exact_set_equality_passes_on_clean_data() -> None:
    pairs, rows = _make_synthetic_pairs_and_rows()
    summary, _ = reconcile_intervention_keys(pairs, rows, n_random_draws=3, valid_k_values=(2, 4))
    assert summary["exact_set_equality"] is True
    assert summary["expected_rows"] == summary["observed_rows"]
    assert summary["missing_keys_count"] == 0
    assert summary["unexpected_keys_count"] == 0
    assert summary["duplicate_rows_count"] == 0
    assert summary["unmatched_pair_id_rows_count"] == 0
    assert summary["invalid_combinations_rows_count"] == 0


def test_detects_missing_key() -> None:
    pairs, rows = _make_synthetic_pairs_and_rows()
    rows_miss = rows[:-1]
    summary, _ = reconcile_intervention_keys(pairs, rows_miss, n_random_draws=3, valid_k_values=(2, 4))
    assert summary["exact_set_equality"] is False
    assert summary["missing_keys_count"] == 1
    assert summary["unexpected_keys_count"] == 0


def test_detects_duplicate_and_unexpected_and_unmatched() -> None:
    pairs, rows = _make_synthetic_pairs_and_rows()
    extra_dup = copy.deepcopy(rows[0])
    extra_unexp = {
        "pair_id": "PAIR_000",
        "scene_id": "scene_a",
        "layer": "layer2_post_relu_64d",
        "direction": "A_to_B",
        "k": 4,
        "condition": "norm_matched_random",  # not VALID_COMMON_SUPPORT -> unexpected!
        "draw_id": 0,
    }
    extra_unmatched = {
        "pair_id": "PAIR_999",
        "scene_id": "scene_a",
        "layer": "layer1_post_relu_64d",
        "direction": "A_to_B",
        "k": 2,
        "condition": "candidate",
        "draw_id": 0,
    }
    bad_combo = {
        "pair_id": "PAIR_000",
        "scene_id": "scene_a",
        "layer": "layer99_invalid",
        "direction": "C_to_D",
        "k": 99,
        "condition": "bogus_cond",
        "draw_id": 5,
    }
    summary, _ = reconcile_intervention_keys(
        pairs,
        rows + [extra_dup, extra_unexp, extra_unmatched, bad_combo],
        n_random_draws=3,
        valid_k_values=(2, 4),
    )
    assert summary["exact_set_equality"] is False
    assert summary["duplicate_rows_count"] == 1
    assert summary["unexpected_keys_count"] == 3
    assert summary["unmatched_pair_id_rows_count"] == 1
    assert summary["invalid_combinations_rows_count"] == 1
    assert "PAIR_999" in summary["unmatched_pair_ids"]


def test_detects_missing_plus_extra_when_total_row_count_unchanged() -> None:
    """Crucial: total observed == total expected, yet missing=1 and extra=1."""
    pairs, rows = _make_synthetic_pairs_and_rows()
    # Remove one valid norm_matched draw and add an unexpected draw on an invalid setting
    dropped = None
    filtered = []
    for r in rows:
        if dropped is None and r["condition"] == "norm_matched_random":
            dropped = r
            continue
        filtered.append(r)
    assert dropped is not None
    fake_extra = copy.deepcopy(dropped)
    fake_extra["layer"] = "layer2_post_relu_64d"  # invalid common support for norm_matched_random
    fake_extra["k"] = 4
    rows_swapped = filtered + [fake_extra]
    assert len(rows_swapped) == len(rows)

    summary, _ = reconcile_intervention_keys(pairs, rows_swapped, n_random_draws=3, valid_k_values=(2, 4))
    assert summary["expected_rows"] == summary["observed_rows"]
    assert summary["exact_set_equality"] is False
    assert summary["missing_keys_count"] == 1
    assert summary["unexpected_keys_count"] == 1
    assert summary["per_condition_discrepancies"]["norm_matched_random"]["exact_set_equality"] is False
