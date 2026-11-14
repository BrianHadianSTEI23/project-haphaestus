"""Tests for Phase 5.4.2 Single-Reference-Target Estimand Audit."""

from __future__ import annotations

import numpy as np

from world_models_fpga.research.circuits.run_task_conditioned_circuits_v5_4_2_estimand_audit import (
    compute_single_target_estimands,
    holm_bonferroni_adjust,
    summarize_estimands_for_setting_records,
)


def test_cross_target_bug_vs_single_target_estimands_under_noop_and_shift() -> None:
    """Verifies cross-target difference is invalid on no-op and opposite-direction shifts."""
    recip_gt = np.array([1.0, 1.0])
    donor_gt = np.array([1.0, -1.0])
    # Baseline starts biased closer to donor than recipient
    base_out = np.array([1.0, -0.4])

    # 1. No-op intervention: trans_out == base_out
    noop_est = compute_single_target_estimands(base_out, base_out, recip_gt, donor_gt)
    assert np.isclose(noop_est["donor_target_gain_m"], 0.0)
    assert np.isclose(noop_est["recipient_target_change_m"], 0.0)
    assert np.isclose(noop_est["eta_gt"], 0.0)
    assert np.isclose(noop_est["target_parallel_component_m"], 0.0)
    assert np.isclose(noop_est["intervention_norm_m"], 0.0)
    # Yet legacy cross-target diff spuriously gives 1.4 - 0.6 = +0.8m!
    assert np.isclose(noop_est["legacy_cross_target_error_diff_m"], 0.8)

    # 2. Pure radial donor shift (moves 0.5m directly toward donor_gt)
    trans_good = np.array([1.0, -0.9])
    good_est = compute_single_target_estimands(base_out, trans_good, recip_gt, donor_gt)
    assert np.isclose(good_est["donor_target_gain_m"], 0.5)
    assert np.isclose(good_est["recipient_target_change_m"], 0.5)
    # d_true = [0, -2], d_int = [0, -0.5] -> eta_gt = 0.25 (dimensionless), parallel = 0.5m
    assert np.isclose(good_est["eta_gt"], 0.25)
    assert np.isclose(good_est["target_parallel_component_m"], 0.5)
    assert np.isclose(good_est["orthogonal_residual_gt_m"], 0.0)
    assert np.isclose(good_est["intervention_norm_m"], 0.5)
    assert np.isclose(good_est["norm_ratio"], 0.25)

    # 3. Orthogonal perturbation with zero movement toward donor
    trans_orth = np.array([1.5, -0.4])
    orth_est = compute_single_target_estimands(base_out, trans_orth, recip_gt, donor_gt)
    assert orth_est["donor_target_gain_m"] < 0.0  # moves away from donor!
    assert np.isclose(orth_est["eta_gt"], 0.0)
    assert np.isclose(orth_est["target_parallel_component_m"], 0.0)
    assert np.isclose(orth_est["orthogonal_residual_gt_m"], 0.5)


def test_streaming_pair_level_aggregation_and_baseline_identity_audit() -> None:
    """Verifies pair-level averaging (equal 4-setting weight), baseline check, and Holm family."""
    pair_ids = ["PAIR_000", "PAIR_001"]
    ctx_map = {"PAIR_000": "nominal", "PAIR_001": "illumination_shift"}
    act_lookup = {}
    for pid in pair_ids:
        for sc in ("scene_a", "scene_b"):
            act_lookup[(pid, sc)] = {
                "gt_goal_a": [1.0, 0.5, 0.0, 1.0],
                "gt_goal_b": [1.0, -0.5, 0.0, 1.0],
                "output_task_a": [0.8, 0.2, 0.0, 1.0],
                "output_task_b": [0.8, 0.0, 0.0, 1.0],
            }

    # Eligible strictly on layer1 k=2 and k=4 for both pairs
    eligible = {
        ("layer1_post_relu_64d", 2): {"PAIR_000", "PAIR_001"},
        ("layer1_post_relu_64d", 4): {"PAIR_000", "PAIR_001"},
    }

    records = []
    for pid in pair_ids:
        for sc in ("scene_a", "scene_b"):
            for direction in ("A_to_B", "B_to_A"):
                base_out = [0.8, 0.2] if direction == "A_to_B" else [0.8, 0.0]
                for layer in ("layer1_post_relu_64d", "layer2_post_relu_64d"):
                    for k in (2, 4, 8, 16, 32):
                        # candidate draw
                        records.append({
                            "pair_id": pid,
                            "scene_id": sc,
                            "layer": layer,
                            "direction": direction,
                            "k": k,
                            "condition": "candidate",
                            "draw_id": 0,
                            "baseline_output": base_out,
                            "transplanted_output": [0.8, -0.1] if direction == "A_to_B" else [0.8, 0.1],
                        })
                        # 2 dim-matched + 2 norm-matched draws (norm only for eligible)
                        for draw_id in range(2):
                            records.append({
                                "pair_id": pid,
                                "scene_id": sc,
                                "layer": layer,
                                "direction": direction,
                                "k": k,
                                "condition": "dimension_matched_random",
                                "draw_id": draw_id,
                                "baseline_output": base_out,
                                "transplanted_output": base_out,
                            })
                            if pid in eligible.get((layer, k), set()):
                                records.append({
                                    "pair_id": pid,
                                    "scene_id": sc,
                                    "layer": layer,
                                    "direction": direction,
                                    "k": k,
                                    "condition": "norm_matched_random",
                                    "draw_id": draw_id,
                                    "baseline_output": base_out,
                                    "transplanted_output": base_out,
                                })

    res = summarize_estimands_for_setting_records(
        pair_ids=pair_ids,
        pair_context_map=ctx_map,
        eligible_pids_by_lk=eligible,
        records_iter=records,
        act_lookup=act_lookup,
    )
    assert res["baseline_identity_audit"]["baseline_matches_all_draws"] is True
    assert res["baseline_identity_audit"]["activation_ref_mismatches_gt_1e_6"] == 0
    assert res["layer_k_results"]["layer1_post_relu_64d"]["k_2"]["support_status"] == "EXPLORATORY_COMPLETE_SUPPORT"
    assert res["layer_k_results"]["layer1_post_relu_64d"]["k_8"]["support_status"] == "EXPLORATORY_INCOMPLETE_SUPPORT"
    assert len(res["holm_family_definition"]) == 4


def test_holm_bonferroni_adjust_monotonicity() -> None:
    adj = holm_bonferroni_adjust([0.04, 0.005, 0.25, 0.03])
    # Sorted: 0.005 (*4=0.02), 0.03 (*3=0.09), 0.04 (*2=0.09 monotonic), 0.25 (*1=0.25)
    assert np.allclose(adj, [0.09, 0.02, 0.25, 0.09])
