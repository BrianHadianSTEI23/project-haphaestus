"""Unit tests for Phase 5.5.1.1 H1 estimand correction, L/R bias audit, and integrity verification."""

import json
from pathlib import Path

import numpy as np
import pytest

from world_models_fpga.research.benchmarks.run_task_conditioned_closed_loop_grounding_v5_5_1_1_audit import (
    _crossover_category,
    _recompute_unassigned_success_symmetric,
    analyze_H1_and_biases,
    mantel_haenszel_stratified,
    stuart_maxwell_3x3,
)


def test_crossover_4way_partition_and_unassigned_symmetric():
    assert _crossover_category("assigned", "assigned")[0] == "both_first_assigned"
    c_coll1 = _crossover_category("assigned", "unassigned")
    c_coll2 = _crossover_category("unassigned", "assigned")
    assert c_coll1 == ("same_physical_object_collapse", "collapse_onto_obj1")
    assert c_coll2 == ("same_physical_object_collapse", "collapse_onto_obj2")
    assert _crossover_category("unassigned", "unassigned")[0] == "both_first_unassigned"
    for a, b in [("assigned", "neither"), ("neither", "unassigned"), ("neither", "neither")]:
        assert _crossover_category(a, b)[0] == "at_least_one_neither"

    # Symmetric collision-free unassigned success requires all 4 conditions
    base = {
        "initially_inside_radius": 0,
        "unassigned_reached": 1,
        "collision": 0,
        "final_unassigned_goal_error_m": 0.30,
    }
    assert _recompute_unassigned_success_symmetric(base) == 1
    assert _recompute_unassigned_success_symmetric({**base, "collision": 1}) == 0
    assert _recompute_unassigned_success_symmetric({**base, "final_unassigned_goal_error_m": 0.40}) == 0
    assert _recompute_unassigned_success_symmetric({**base, "unassigned_reached": 0}) == 0


def test_stuart_maxwell_and_mantel_haenszel():
    # Perfect symmetry -> chi2 = 0, p = 1.0
    m_sym = np.array([[5, 3, 2], [3, 4, 1], [2, 1, 10]], dtype=float)
    res = stuart_maxwell_3x3(m_sym)
    assert res["p"] == pytest.approx(1.0)

    # Strong asymmetry -> small p
    m_asym = np.array([[2, 1, 1], [10, 2, 1], [15, 2, 5]], dtype=float)
    res_a = stuart_maxwell_3x3(m_asym)
    assert res_a["chi2"] > 5.0 and res_a["p"] < 0.05

    # Stratified MH: right=8/10 vs left=2/10 in two strata
    mh = mantel_haenszel_stratified([(8, 2, 2, 8), (8, 2, 2, 8)])
    assert mh["common_odds_ratio_right_vs_left"] == pytest.approx(16.0)
    assert mh["mh_risk_diff_right_minus_left"] == pytest.approx(0.6)
    assert mh["cmh_p"] < 1e-3


def test_synthetic_end_to_end_h1_and_lr_bias():
    # Build 6 synthetic scenes across 1 condition with mirrored A/B and L/R assignments
    from world_models_fpga.models.baseline_policy import TARGET_KEYS

    scenes_by_id = {}
    episodes = []
    scene_pairs = []
    traces_by_ep = {}

    for i in range(6):
        sid = f"SCENE_{i:03d}"
        k1 = TARGET_KEYS[i % len(TARGET_KEYS)]
        k2 = TARGET_KEYS[(i + 1) % len(TARGET_KEYS)]
        scenes_by_id[sid] = {
            "scene_id": sid,
            "scene_idx": i,
            "context": "nominal" if i < 3 else "illumination_shift",
            "obj_1_key": k1,
            "obj_2_key": k2,
            "obj_1_side": "left",
            "obj_2_side": "right",
            "initial_pose": [0.0, 0.0, 0.0],
            "wp_1": [0.8, 0.5, 0.0],
            "wp_2": [0.8, -0.5, 0.0],
        }
        # Right (B) succeeds 5/6 times, Left (A) succeeds 1/6 times
        sA = 1 if i == 0 else 0
        sB = 1 if i < 5 else 0
        fA = "assigned" if sA == 1 else ("unassigned" if i in (1, 2) else "neither")
        fB = "assigned" if sB == 1 else "neither"

        for asg, tk, dk, side, succ, fr in [
            ("A", k1, k2, "left", sA, fA),
            ("B", k2, k1, "right", sB, fB),
        ]:
            epk = f"F1_dynamic_visual_with_memory__{sid}__{asg}"
            un_r = 1 if fr == "unassigned" else 0
            episodes.append(
                {
                    "condition": "F1_dynamic_visual_with_memory",
                    "scene_id": sid,
                    "assignment": asg,
                    "episode_key": epk,
                    "context": scenes_by_id[sid]["context"],
                    "category_pair": f"{min(k1, k2)}__{max(k1, k2)}",
                    "target_key": tk,
                    "distractor_key": dk,
                    "assigned_side": side,
                    "initially_inside_radius": 0,
                    "assigned_reached": succ,
                    "unassigned_reached": un_r,
                    "first_reached_object": fr,
                    "collision": 0,
                    "assigned_success": succ,
                    "final_assigned_goal_error_m": 0.25 if succ else 0.90,
                    "final_unassigned_goal_error_m": 0.28 if un_r else 0.95,
                    "executed_steps": 1,
                }
            )
            traces_by_ep[epk] = [
                {
                    "t": 0,
                    "pose": [0.0, 0.0, 0.0],
                    "g_visual": [0.5, -0.2, 0.0, 1.0],
                    "gt_assigned_body": [0.8, 0.5 if asg == "A" else -0.5, 0.0, 1.0],
                    "gt_unassigned_body": [0.8, -0.5 if asg == "A" else 0.5, 0.0, 1.0],
                }
            ]
        scene_pairs.append(
            {
                "condition": "F1_dynamic_visual_with_memory",
                "scene_id": sid,
                "task_switch_eta_vis_step0": 0.0,
                "task_switch_cos_vis_step0": 0.0,
            }
        )

    summary, pair_rows, cat_rows, ep_rows = analyze_H1_and_biases(
        scenes_by_id=scenes_by_id,
        episodes=episodes,
        scene_pairs=scene_pairs,
        traces_by_ep=traces_by_ep,
        seed=123,
    )
    h1 = summary["H1_reanalysis_by_condition"]["F1_dynamic_visual_with_memory"]
    assert h1["family_C_complete_crossover_matrix_N60"]["counts"] == {
        "both_first_assigned": 1,
        "same_physical_object_collapse": 2,
        "both_first_unassigned": 0,
        "at_least_one_neither": 3,
    }
    lr = summary["left_right_bias_by_condition"]["F1_dynamic_visual_with_memory"]
    assert lr["left_success_count"] == 1
    assert lr["right_success_count"] == 5
    assert lr["paired_table_both_leftOnly_rightOnly_neither"] == [1, 0, 4, 1]
    assert len(pair_rows) == 6
    assert len(ep_rows) == 12
    assert len(cat_rows) == 12
