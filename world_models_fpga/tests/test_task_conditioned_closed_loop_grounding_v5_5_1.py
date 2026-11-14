"""Tests for Phase 5.5.1 Geometry-Separated Task Selectivity & Feedback/Memory Decomposition."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from env.amr_env import JetAutoEnv
from world_models_fpga.research.benchmarks.model_wrappers import HJEPAAgent, OracleAgent, compute_body_frame_goal_vec
from world_models_fpga.research.benchmarks.run_task_conditioned_closed_loop_grounding_v5_5 import (
    body_goal_to_world_pose,
)
from world_models_fpga.research.benchmarks.run_task_conditioned_closed_loop_grounding_v5_5_1 import (
    MAX_JSONL_SHARD_BYTES,
    MIN_WAYPOINT_SEPARATION_M,
    V5_5_1_CONDITIONS,
    build_geometry_separated_cohort,
    compute_axis_projections,
    compute_scene_pair_crossover_record,
    compute_v5_5_1_statistics,
    evaluate_v5_5_1_episode,
    export_trace_shards_jsonl,
    generate_v5_5_1_figures,
    make_assignment_spec,
)
from world_models_fpga.tests.test_task_conditioned_closed_loop_grounding_v5_5 import _DummyEnv, _make_dummy_agent


def test_geometry_separated_cohort_acceptance_and_non_overlapping_reach_regions() -> None:
    """Verifies >= 0.90 m separation, disjoint 0.35m reach circles, rejected candidate logging, and A/B Oracle reachability."""
    env = JetAutoEnv()
    oracle = OracleAgent()
    try:
        scenes, report = build_geometry_separated_cohort(
            env=env,
            oracle=oracle,
            n_nominal=2,
            n_illum=2,
            max_steps=45,
            goal_reach_threshold=0.35,
            seed=9101,
        )
    finally:
        env.close()

    assert len(scenes) == 4
    assert report["n_nominal_scenes"] == 2
    assert report["n_illum_scenes"] == 2
    assert report["total_candidates_evaluated"] >= 4
    assert isinstance(report["rejection_breakdown"], dict)
    for sc in scenes:
        wp1 = np.asarray(sc["wp_1"][:2], dtype=np.float64)
        wp2 = np.asarray(sc["wp_2"][:2], dtype=np.float64)
        sep = float(np.linalg.norm(wp1 - wp2))
        assert sep >= MIN_WAYPOINT_SEPARATION_M
        assert sc["reach_bubble_gap_m"] >= 0.20 - 1e-6
        assert 0.60 <= sc["init_dist_wp1_m"] <= 1.20
        assert 0.60 <= sc["init_dist_wp2_m"] <= 1.20
        assert sc["init_vis_px_1"] >= 60 and sc["init_vis_px_2"] >= 60


def test_within_scene_crossover_and_axis_projection_scoring() -> None:
    """Verifies dual-task crossover selectivity, collapse detection, and signed projection."""
    gt_a = np.array([1.0, 0.5, 0.0, 1.0], dtype=np.float32)
    gt_u = np.array([1.0, -0.5, 0.0, 1.0], dtype=np.float32)
    # Vector sitting right on assigned target
    p_m_pos, c_pos = compute_axis_projections(gt_a[:2], gt_a[:2], gt_u[:2])
    assert p_m_pos == pytest.approx(0.5, abs=1e-5)
    assert c_pos == pytest.approx(0.5, abs=1e-5)
    # Vector sitting right on unassigned target
    p_m_neg, c_neg = compute_axis_projections(gt_u[:2], gt_a[:2], gt_u[:2])
    assert p_m_neg == pytest.approx(-0.5, abs=1e-5)
    assert c_neg == pytest.approx(-0.5, abs=1e-5)

    base = {
        "condition": "F1_dynamic_visual_with_memory",
        "scene_id": "SCENE_000",
        "scene_idx": 0,
        "context": "nominal",
        "category_pair": "blue_cylinder__red_box",
        "final_assigned_goal_error_m": 0.25,
        "final_unassigned_goal_error_m": 1.15,
        "final_margin_unassigned_minus_assigned_m": 0.90,
        "mean_step_coord_vis_norm": 0.4,
    }
    ep_a_good = {**base, "target_key": "red_box", "assigned_success": True, "unassigned_reached": False, "first_reached_object": "assigned"}
    ep_b_good = {**base, "target_key": "blue_cylinder", "assigned_success": True, "unassigned_reached": False, "first_reached_object": "assigned"}
    pair_ok = compute_scene_pair_crossover_record(ep_a_good, ep_b_good, gt_a, gt_u, gt_a, gt_u, gt_a, gt_u)
    assert pair_ok["dual_assigned_success"] is True
    assert pair_ok["dual_first_correct_selection"] is True
    assert pair_ok["same_object_collapse"] is False
    assert pair_ok["pair_selectivity_margin"] == pytest.approx(1.0)
    assert pair_ok["task_switch_eta_vis_step0"] == pytest.approx(1.0)

    # Collapse case: both assignments physically visit obj_1 (A gets assigned, B gets unassigned!)
    ep_b_collapse = {**base, "target_key": "blue_cylinder", "assigned_success": False, "unassigned_reached": True, "first_reached_object": "unassigned"}
    pair_col = compute_scene_pair_crossover_record(ep_a_good, ep_b_collapse, gt_a, gt_a, gt_a, gt_a, gt_a, gt_u)
    assert pair_col["dual_first_correct_selection"] is False
    assert pair_col["same_object_collapse"] is True
    assert pair_col["pair_selectivity_margin"] == pytest.approx(0.0)
    assert pair_col["task_switch_eta_vis_step0"] == pytest.approx(0.0)


def test_factorial_f3_f4_and_o1_override_and_shard_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies F3 vs F4 freezing/memory isolation, O1 override provenance, and <5MB JSONL shard export."""
    import world_models_fpga.research.benchmarks.run_task_conditioned_closed_loop_grounding_v5_5_1 as v551

    monkeypatch.setattr(v551.mujoco, "mj_name2id", lambda _m, _t, name: 1 if "red" in name else 2)
    monkeypatch.setattr(v551, "render_segmentation_array", lambda _e: np.ones((64, 64, 2), dtype=np.int32))

    env = _DummyEnv()
    agent = _make_dummy_agent()
    spec = {
        "scene_id": "SCENE_SYN_0",
        "scene_idx": 0,
        "assignment": "A",
        "context": "nominal",
        "category_pair": "blue_cylinder__red_box",
        "target_key": "red_box",
        "distractor_key": "blue_cylinder",
        "assigned_side": "left",
        "initial_pose": [0.0, 0.0, 0.0],
        "waypoints": [[1.1, 0.45, 0.0]],
        "distractor_waypoint": [1.1, -0.50, 0.0],
        "task_token": 1,
        "swapped_task_token": 2,
    }

    # F3: frozen world visual goal + memory ENABLED
    rec_f3 = evaluate_v5_5_1_episode(agent, env, spec, "F3_frozen_visual_with_memory", max_steps=4, capture_rgb_keyframes=True)
    tr3 = rec_f3["traces"]
    assert rec_f3["memory_branch_enabled"] is True
    assert rec_f3["memory_branch_executed"] is True
    assert np.all(np.isnan(tr3["delta_ego"][0]))
    assert not np.any(np.isnan(tr3["delta_ego"][1:]))
    assert np.allclose(tr3["g_memory"][0], tr3["g_visual"][0], atol=1e-5)
    assert not np.any(np.isnan(tr3["g_memory"][1:]))
    # Verify frozen world anchor matches step-0 re-projection across subsequent poses
    world_v0 = body_goal_to_world_pose(tr3["pose"][0], tr3["g_visual"][0])
    for t in range(len(tr3["pose"])):
        expected_v_t = compute_body_frame_goal_vec(tr3["pose"][t], world_v0)
        assert np.allclose(tr3["g_visual"][t], expected_v_t, atol=1e-5)
    assert not np.allclose(tr3["g_blended"][1:, :2], tr3["g_visual"][1:, :2], atol=1e-4)

    # F4: frozen world visual goal + memory DISABLED
    rec_f4 = evaluate_v5_5_1_episode(agent, env, spec, "F4_frozen_visual_no_memory", max_steps=4)
    tr4 = rec_f4["traces"]
    assert rec_f4["memory_branch_enabled"] is False
    assert rec_f4["memory_branch_executed"] is False
    assert np.all(np.isnan(tr4["g_memory"]))
    assert np.all(np.isnan(tr4["delta_ego"]))
    assert np.allclose(tr4["g_blended"], tr4["g_visual"], atol=1e-6)

    # O1: planner goal override preserves Track C pipeline & _ego_goal_memory
    rec_o1 = evaluate_v5_5_1_episode(agent, env, spec, "O1_track_c_planner_goal_override", max_steps=4)
    assert rec_o1["task_embedding_executed"] is True
    assert rec_o1["grounding_head_executed"] is True
    assert rec_o1["memory_branch_executed"] is True
    assert rec_o1["planner_goal_source"] == "track_c_planner_goal_override"
    assert np.allclose(agent._ego_goal_memory.cpu().numpy(), rec_o1["traces"]["g_blended"][-1], atol=1e-5)

    # Test <5 MB JSONL shard export using small limit to force multiple shards cleanly
    shard_dir = tmp_path / "shards"
    ep_rows = []
    tr_map = {}
    for idx, r in enumerate((rec_f3, rec_f4, rec_o1)):
        rc = dict(r)
        tr = rc.pop("traces")
        rc.pop("keyframes", None)
        ep_rows.append(rc)
        tr_map[rc["episode_key"]] = tr
    manifest = export_trace_shards_jsonl(ep_rows, tr_map, shard_dir, max_shard_bytes=10_000)
    assert manifest["n_shards"] >= 2
    assert manifest["max_observed_shard_bytes"] < 5 * 1024 * 1024
    for s in manifest["shards"]:
        p = shard_dir / s["filename"]
        assert p.exists()
        assert p.stat().st_size == s["bytes"] < 15_000
    # Verify null serialization in F4 rows (no NaN strings in JSON)
    all_lines = []
    for s in manifest["shards"]:
        all_lines.extend((shard_dir / s["filename"]).read_text(encoding="utf-8").splitlines())
    assert len(all_lines) == manifest["total_rows"]
    parsed_f4 = [json.loads(line) for line in all_lines if '"F4_frozen_visual_no_memory"' in line]
    assert len(parsed_f4) == 4
    assert all(row["g_memory"] == [None, None, None, None] for row in parsed_f4)


def test_v5_5_1_statistics_holm5_and_plot_only(tmp_path: Path) -> None:
    """Verifies 5-family Holm-Bonferroni scene-pair statistics & artifact-only figure generation."""
    episodes_by_cond = {c: [] for c in V5_5_1_CONDITIONS}
    pairs_by_cond = {c: [] for c in V5_5_1_CONDITIONS}
    all_eps = []
    all_pairs = []
    npz_payload = {}

    rng = np.random.default_rng(42)
    for cname in V5_5_1_CONDITIONS:
        base_rate = 0.85 if cname in ("F1_dynamic_visual_with_memory", "O1_track_c_planner_goal_override", "M2_distractor_masked_rgb") else 0.35
        for sid_idx in range(6):
            ctx = "nominal" if sid_idx < 3 else "illumination_shift"
            pair_eps = []
            for asg in ("A", "B"):
                succ = bool(rng.uniform() < base_rate)
                ek = f"{cname}__SCENE_{sid_idx:03d}__{asg}"
                ep = {
                    "scene_id": f"SCENE_{sid_idx:03d}",
                    "scene_idx": sid_idx,
                    "assignment": asg,
                    "episode_key": ek,
                    "context": ctx,
                    "category_pair": "blue_cylinder__red_box",
                    "condition": cname,
                    "target_key": "red_box" if asg == "A" else "blue_cylinder",
                    "distractor_key": "blue_cylinder" if asg == "A" else "red_box",
                    "assigned_side": "left" if asg == "A" else "right",
                    "assigned_success": succ,
                    "unassigned_reached": not succ,
                    "first_reached_object": "assigned" if succ else "unassigned",
                    "collision": False,
                    "closer_to_assigned": succ,
                    "final_assigned_goal_error_m": 0.25 if succ else 0.90,
                    "final_unassigned_goal_error_m": 1.10 if succ else 0.30,
                    "final_margin_unassigned_minus_assigned_m": 0.85 if succ else -0.60,
                    "path_length_m": 1.0,
                    "executed_steps": 5,
                    "mean_step_vis_goal_ang_err_deg": 18.0,
                    "mean_step_blended_goal_ang_err_deg": 18.0,
                    "mean_step_planner_goal_ang_err_deg": 18.0,
                    "mean_step_coord_vis_norm": 0.3 if succ else -0.2,
                    "mean_step_action_assigned_cos": 0.8 if succ else -0.3,
                }
                pair_eps.append(ep)
                episodes_by_cond[cname].append(ep)
                all_eps.append(ep)
                npz_payload[f"{ek}__pose"] = np.zeros((5, 3), dtype=np.float32)
                npz_payload[f"{ek}__dist_assigned_wp"] = np.linspace(1.0, 0.3, 5, dtype=np.float32)
                npz_payload[f"{ek}__coord_vis_norm"] = np.linspace(0.0, 0.4, 5, dtype=np.float32)
                npz_payload[f"{ek}__vis_ang_err_rad"] = np.linspace(0.5, 0.2, 5, dtype=np.float32)

            gt_a = np.array([1.0, 0.5, 0.0, 1.0], dtype=np.float32)
            gt_b = np.array([1.0, -0.5, 0.0, 1.0], dtype=np.float32)
            pr = compute_scene_pair_crossover_record(pair_eps[0], pair_eps[1], gt_a, gt_b, gt_a, gt_b, gt_a, gt_b)
            pairs_by_cond[cname].append(pr)
            all_pairs.append(pr)

    stats_out = compute_v5_5_1_statistics(episodes_by_cond, pairs_by_cond, seed=100)
    assert set(stats_out["confirmatory_hypothesis_families_holm5"].keys()) == {
        "H1_task_selectivity_F1",
        "H2_visual_feedback_main_effect",
        "H3_controller_memory_main_effect",
        "H4_object_masking_specificity",
        "H5_planner_goal_override_O1",
    }

    sum_p = tmp_path / "summary.json"
    ep_p = tmp_path / "episodes.jsonl"
    pr_p = tmp_path / "pairs.jsonl"
    npz_p = tmp_path / "traces.npz"
    figs_p = tmp_path / "figs"
    sum_p.write_text(json.dumps({**stats_out, "representative_cases": {"task_switch_comparison": {"scene_id": "SCENE_000", "waypoint_separation_m": 1.0, "wp_1": [1.0, 0.5, 0.0], "wp_2": [1.0, -0.5, 0.0]}}}), encoding="utf-8")
    ep_p.write_text("\n".join(json.dumps(e) for e in all_eps) + "\n", encoding="utf-8")
    pr_p.write_text("\n".join(json.dumps(p) for p in all_pairs) + "\n", encoding="utf-8")
    np.savez_compressed(npz_p, **npz_payload)

    written = generate_v5_5_1_figures(sum_p, ep_p, pr_p, npz_p, figs_p)
    assert "fig1_v5_5_1_selection_and_context_rates.png" in written
    assert "fig2_v5_5_1_2x2_feedback_memory_factorial.png" in written
    assert "fig3_v5_5_1_temporal_dynamics.png" in written
    assert "frames/task_switch_comparison_SCENE_000.png" in written
