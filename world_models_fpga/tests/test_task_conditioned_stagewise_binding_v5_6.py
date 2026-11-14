#!/usr/bin/env python3
"""Unit tests for Phase 5.6 stage-wise task-to-space mechanistic localization (synthetic fixtures)."""

from __future__ import annotations

import numpy as np
import torch

from world_models_fpga.env.amr_env import JetAutoEnv
from world_models_fpga.models.baseline_policy import TARGET_KEYS, encode_task_id
from world_models_fpga.models.h_jepa import HJEPAModel
from world_models_fpga.research.benchmarks.run_task_conditioned_stagewise_binding_v5_6 import (
    MAX_NORM_REL_DIFF,
    _fit_oof_task_projectors,
    canonical_token_catalog,
    compute_contrast_metrics,
    holm_bonferroni,
    make_norm_matched_perturbation,
    run_causal_interventions,
    run_coordinate_and_training_audit,
    run_fixed_image_token_sweep,
    run_stagewise_probes,
)


def _make_synthetic_scenes(n_scenes: int = 12) -> list[dict]:
    rng = np.random.default_rng(12345)
    pairs = [("red_box", "blue_cylinder"), ("green_cylinder", "yellow_cylinder"), ("purple_box", "cyan_cylinder")]
    scenes = []
    for i in range(n_scenes):
        k1, k2 = pairs[i % len(pairs)]
        if (i // len(pairs)) % 2 == 1:
            k1, k2 = k2, k1
        side_a = "left" if i % 2 == 0 else "right"
        side_b = "right" if side_a == "left" else "left"
        sign_a = 1.0 if side_a == "left" else -1.0
        wp1 = np.array([float(rng.uniform(0.8, 1.3)), float(sign_a * rng.uniform(0.35, 0.55))], dtype=np.float32)
        wp2 = np.array([float(rng.uniform(0.8, 1.3)), float(-sign_a * rng.uniform(0.35, 0.55))], dtype=np.float32)
        rgb = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        scenes.append(
            {
                "scene_idx": i,
                "scene_id": f"SYN_{i:03d}",
                "context": "nominal" if i < n_scenes // 2 else "illumination_shift",
                "category_pair": f"{min(k1, k2)}__{max(k1, k2)}",
                "obj_1_key": k1,
                "obj_2_key": k2,
                "obj_1_side": side_a,
                "obj_2_side": side_b,
                "tok_a": int(encode_task_id(k1, "approach", f"approach {k1}")),
                "tok_b": int(encode_task_id(k2, "approach", f"approach {k2}")),
                "rgb0": rgb,
                "rgb0_sha256": f"hash_{i}",
                "init_pose": np.zeros(3, dtype=np.float32),
                "wp1_b": wp1,
                "wp2_b": wp2,
                "obj1_b": (wp1 + np.array([0.45, sign_a * 0.15], dtype=np.float32)),
                "obj2_b": (wp2 + np.array([0.45, -sign_a * 0.15], dtype=np.float32)),
                "left_obj_b": (wp1 if side_a == "left" else wp2),
                "right_obj_b": (wp2 if side_a == "left" else wp1),
                "left_wp_b": (wp1 if side_a == "left" else wp2),
                "right_wp_b": (wp2 if side_a == "left" else wp1),
            }
        )
    return scenes


def test_b2_synthetic_coordinate_frame_and_action_signs() -> None:
    env = JetAutoEnv(obs_size=(64, 64))
    model = HJEPAModel(latent_dim=32, stride_k=5, hidden_dim=64)
    model.eval()
    try:
        audit = run_coordinate_and_training_audit(
            env,
            model=model,
            train_meta_path=None,
            train_npz_path=None,
            ckpt_path=None,
        )
        assert audit["all_synthetic_coordinate_checks_passed"] is True
        assert audit["simulator_action_signs"]["vy_pos_moves_left_ok"] is True
        assert audit["simulator_action_signs"]["wz_pos_rotates_ccw_ok"] is True
        assert audit["mirrored_robot_single_object"]["dy_b_anti_symmetric_ok"] is True
        assert audit["mirrored_robot_single_object"]["u_mirror_sum_approx_63_ok"] is True
        assert audit["controller_forced_goal_symmetry"]["executed"] is True
    finally:
        env.close()


def test_b3_b4_b5_synthetic_pipeline_invariance_and_norm_matching() -> None:
    torch.manual_seed(42)
    model = HJEPAModel(latent_dim=32, stride_k=5, hidden_dim=64)
    model.eval()
    scenes = _make_synthetic_scenes(n_scenes=12)

    catalog = canonical_token_catalog()
    assert len(catalog) >= 48

    b3 = run_fixed_image_token_sweep(model, scenes, device="cpu", token_catalog=catalog)
    assert b3["summary"]["z_t_invariant_across_tokens_ok"] is True
    assert b3["summary"]["n_sweep_rows"] == len(scenes) * len(catalog)
    assert b3["summary"]["h3_jacobian_dgoal_dz_frob"]["mean"] >= 0.0
    assert b3["summary"]["h3_jacobian_dgoal_de_frob"]["mean"] >= 0.0

    b4 = run_stagewise_probes(b3["scene_stage_cache"])
    assert b4["n_probe_rows"] > 0
    assert any(r["split_scheme"] == "GroupKFold_scene" for r in b4["probe_rows"])

    # Verify OOF projectors are strictly orthonormal and leak-free
    oof_l1 = _fit_oof_task_projectors(b3["scene_stage_cache"], "l1", ranks=(4, 8), n_splits=4)
    for sid, rank_map in oof_l1.items():
        assert sid.startswith("SYN_")
        for k, uk in rank_map.items():
            assert uk.shape == (64, k)
            gram = uk.T @ uk
            assert np.allclose(gram, np.eye(k), atol=1e-6)

    b5 = run_causal_interventions(model, scenes, b3["scene_stage_cache"], device="cpu", seed=5600)
    assert b5["n_intervention_rows"] > 0
    for name, stats in b5["summary_by_intervention"].items():
        if "norm_matched" in name:
            max_diff = stats["max_rel_act_norm_diff"]
            assert max_diff is not None and max_diff <= MAX_NORM_REL_DIFF


def test_contrast_metrics_holm_and_norm_perturbation_exact() -> None:
    rng = np.random.default_rng(7)
    delta = np.array([1.2, -0.5, 0.3, 0.0, -2.1], dtype=np.float32)
    pert = make_norm_matched_perturbation(delta, rng)
    rel_err = abs(float(np.linalg.norm(pert)) / float(np.linalg.norm(delta)) - 1.0)
    assert rel_err <= 1e-5

    # Exact shift along ground-truth contrast vector gives eta_gt == 1.0, orth == 0.0
    gr = np.array([1.0, 0.5])
    gt = np.array([1.0, -0.5])
    m = compute_contrast_metrics(baseline_xy=gr, patched_xy=gt, gt_rec_xy=gr, gt_target_xy=gt)
    assert abs(m["eta_gt"] - 1.0) < 1e-9
    assert abs(m["orth"]) < 1e-9
    assert abs(m["donor_target_gain"] - 1.0) < 1e-9

    adj = holm_bonferroni([0.01, 0.04, 0.03, None])
    assert abs(adj[0] - 0.03) < 1e-12
    assert abs(adj[2] - 0.06) < 1e-12
    assert abs(adj[1] - 0.06) < 1e-12
    assert adj[3] is None
