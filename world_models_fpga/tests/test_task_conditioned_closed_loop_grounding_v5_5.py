"""Tests for Phase 5.5 Closed-Loop Mechanistic Grounding helpers and invariants."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from world_models_fpga.research.benchmarks.model_wrappers import HJEPAAgent, compute_body_frame_goal_vec
from world_models_fpga.research.benchmarks.run_task_conditioned_closed_loop_grounding_v5_5 import (
    ALL_CONDITIONS,
    apply_local_background_mask,
    body_goal_to_world_pose,
    build_area_matched_distractor_mask,
    evaluate_episode_condition,
    generate_figures_from_saved_artifacts,
    holm_bonferroni_adjust,
    paired_mcnemar_exact,
    wilson_ci,
)


def test_body_world_goal_round_trip_across_poses() -> None:
    """Verifies exact world-space anchoring and re-projection for C2 frozen goal."""
    rng = np.random.default_rng(123)
    for _ in range(25):
        pose0 = rng.uniform([-1.5, -1.5, -np.pi], [1.5, 1.5, np.pi])
        target_world = rng.uniform([-1.5, -1.5, -np.pi], [1.5, 1.5, np.pi])
        g0_body = compute_body_frame_goal_vec(pose0, target_world)
        recovered_world = body_goal_to_world_pose(pose0, g0_body)
        assert np.allclose(recovered_world[:2], target_world[:2], atol=1e-5)
        # Re-evaluating at a moved pose matches ground-truth transform to recovered world point
        pose1 = rng.uniform([-1.5, -1.5, -np.pi], [1.5, 1.5, np.pi])
        g1_from_frozen = compute_body_frame_goal_vec(pose1, recovered_world)
        g1_from_true = compute_body_frame_goal_vec(pose1, target_world)
        assert np.allclose(g1_from_frozen, g1_from_true, atol=1e-5)


def test_area_matched_distractor_mask_and_background_fill() -> None:
    """Verifies C3/C4 pixel-count parity, disjointness, fallback patch, and perimeter median fill."""
    rgb = np.full((32, 32, 3), 100, dtype=np.uint8)
    mask_t = np.zeros((32, 32), dtype=bool)
    mask_t[8:14, 6:12] = True  # 36 pixels
    rgb[mask_t] = np.array([220, 20, 20], dtype=np.uint8)

    # Case 1: distractor visible and larger than target -> exact subsample to 36 px
    mask_d_big = np.zeros((32, 32), dtype=bool)
    mask_d_big[8:16, 18:26] = True  # 64 pixels
    matched_big, fb1 = build_area_matched_distractor_mask(mask_t, mask_d_big)
    assert not fb1
    assert int(matched_big.sum()) == int(mask_t.sum()) == 36
    assert not np.any(matched_big & mask_t)

    # Case 2: distractor visible and smaller than target -> dilated locally to 36 px
    mask_d_small = np.zeros((32, 32), dtype=bool)
    mask_d_small[10:12, 22:24] = True  # 4 pixels
    matched_small, fb2 = build_area_matched_distractor_mask(mask_t, mask_d_small)
    assert not fb2
    assert int(matched_small.sum()) == 36
    assert not np.any(matched_small & mask_t)

    # Case 3: distractor out of FOV -> mirrored disjoint patch of exact area 36 px with fallback=True
    mask_d_empty = np.zeros((32, 32), dtype=bool)
    matched_fb, fb3 = build_area_matched_distractor_mask(mask_t, mask_d_empty)
    assert fb3
    assert int(matched_fb.sum()) == 36
    assert not np.any(matched_fb & mask_t)

    # Background median fill removes bright red target cleanly with surrounding 100 gray
    filled = apply_local_background_mask(rgb, mask_t)
    assert np.all(filled[mask_t] == 100)
    assert np.all(filled[~mask_t] == rgb[~mask_t])


def test_paired_mcnemar_and_holm_adjustment() -> None:
    n10, n01, p = paired_mcnemar_exact([True] * 10 + [False] * 2, [False] * 10 + [False] * 2)
    assert n10 == 10 and n01 == 0
    assert p == pytest.approx(2.0 * (0.5 ** 10))
    adj = holm_bonferroni_adjust([0.01, 0.04, 0.03, 0.20, 0.50])
    assert adj[0] == pytest.approx(0.05)
    assert adj[2] == pytest.approx(0.12)
    assert adj[1] == pytest.approx(0.12)


def test_generate_figures_handles_100pct_wilson_ci(tmp_path: Path) -> None:
    """Ensures 100% success rates (where Wilson upper bound can round slightly < 1.0) plot without negative yerr."""
    ci_lo, ci_hi = wilson_ci(60, 60)
    summary = {
        "conditions": {
            "C1_full_closed_loop": {
                "success_rate": 0.85,
                "success_wilson_95ci": list(wilson_ci(51, 60)),
                "by_context": {
                    "nominal": {"mean_final_goal_error_m": 0.44},
                    "illumination_shift": {"mean_final_goal_error_m": 0.44},
                },
            },
            "C2_frozen_world_goal": {
                "success_rate": 0.60,
                "success_wilson_95ci": list(wilson_ci(36, 60)),
                "by_context": {
                    "nominal": {"mean_final_goal_error_m": 0.48},
                    "illumination_shift": {"mean_final_goal_error_m": 0.47},
                },
            },
            "C6_oracle_waypoint": {
                "success_rate": 1.0,
                "success_wilson_95ci": [ci_lo, ci_hi],
                "by_context": {
                    "nominal": {"mean_final_goal_error_m": 0.33},
                    "illumination_shift": {"mean_final_goal_error_m": 0.33},
                },
            },
        }
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    episodes = []
    npz_data = {}
    for cname in ("C1_full_closed_loop", "C2_frozen_world_goal", "C6_oracle_waypoint"):
        episodes.append(
            {
                "condition": cname,
                "pair_id": "PAIR_000",
                "step0_static_contrast_eta_gt": -0.2,
                "step0_blended_goal_ang_err_deg": 40.0,
                "final_eval_goal_error_m": 0.3 if cname == "C6_oracle_waypoint" else 0.45,
                "success": cname != "C2_frozen_world_goal",
            }
        )
        prefix = f"{cname}__PAIR_000__"
        npz_data[prefix + "dist_target_wp"] = np.linspace(0.9, 0.3, 5, dtype=np.float32)
        npz_data[prefix + "goal_ang_err_rad"] = np.linspace(0.8, 0.2, 5, dtype=np.float32)
        npz_data[prefix + "action_target_cos"] = np.linspace(0.5, 0.9, 5, dtype=np.float32)

    ep_path = tmp_path / "episodes.jsonl"
    ep_path.write_text("\n".join(json.dumps(e) for e in episodes) + "\n", encoding="utf-8")
    npz_path = tmp_path / "traces.npz"
    np.savez_compressed(npz_path, **npz_data)
    figs_dir = tmp_path / "figs"

    written = generate_figures_from_saved_artifacts(summary_path, ep_path, npz_path, figs_dir)
    assert len(written) == 3
    for fn in written:
        assert (figs_dir / fn).exists()


class _DummyMacroPredictor(nn.Module):
    def forward(self, s_batch, _macro_in):
        return s_batch, torch.zeros((s_batch.shape[0], 1), dtype=s_batch.dtype, device=s_batch.device)


class _DummyMicroPredictor(nn.Module):
    def forward_transition(self, s_batch, _act_in):
        return s_batch, torch.zeros((s_batch.shape[0], 1), dtype=s_batch.dtype, device=s_batch.device)

    def select_action(self, s_t, _subgoal):
        return torch.zeros((s_t.shape[0], 3), dtype=s_t.dtype, device=s_t.device)


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stride_k = 5
        self.task_embedding = nn.Embedding(16, 16)
        self.grounding_head = nn.Sequential(
            nn.Linear(32 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.macro_predictor = _DummyMacroPredictor()
        self.micro_predictor = _DummyMicroPredictor()

    def encode(self, img_t):
        return torch.ones((img_t.shape[0], 32), dtype=img_t.dtype, device=img_t.device) * 0.25

    def infer_goal_vec(self, s_t, t_tok):
        emb = self.task_embedding(t_tok)
        raw = self.grounding_head(torch.cat([s_t, emb], dim=-1))
        return torch.cat([torch.clamp(raw[:, :2], -3.5, 3.5), torch.nn.functional.normalize(raw[:, 2:4] + 1e-6, dim=-1)], dim=-1)

    def predict_reactive_action(self, s_t, goal_t):
        return torch.clamp(goal_t[:, :3], -0.5, 0.5)

    def predict_micro_displacement(self, s_prev, _s_next):
        return torch.tensor([[0.05, 0.01, 0.02]], dtype=s_prev.dtype, device=s_prev.device).repeat(s_prev.shape[0], 1)

    def predict_macro_displacement(self, s_t, _next_s):
        return torch.zeros((s_t.shape[0], 3), dtype=s_t.dtype, device=s_t.device)

    def state_col_head(self, next_s):
        return torch.zeros((next_s.shape[0], 1), dtype=next_s.dtype, device=next_s.device)


class _DummyEnv:
    def __init__(self):
        self.model = SimpleNamespace()
        self.pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def restore_physical_baseline(self):
        self.pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def apply_task_spec(self, _spec):
        pass

    def reset(self, options=None):
        init = options["initial_pos"]
        self.pose = np.array(init, dtype=np.float32)
        return np.full((64, 64, 3), 128, dtype=np.uint8), {}

    def get_robot_pose(self):
        return self.pose.copy()

    def step(self, act):
        self.pose = self.pose + np.array([0.05 * float(act[0]), 0.05 * float(act[1]), 0.0], dtype=np.float32)
        return np.full((64, 64, 3), 128, dtype=np.uint8), 0.0, False, False, {"collision": False}


def _make_dummy_agent() -> HJEPAAgent:
    agent = HJEPAAgent.__new__(HJEPAAgent)
    super(HJEPAAgent, agent).__init__(name="DummyHJEPA", track="observation", ablation_mode="prediction_used", device="cpu")
    agent.model = _DummyModel()
    agent.model.eval()
    return agent


def test_part_a2_track_b_vs_track_c_override_provenance_and_nan_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies C6 Track B vs C7 Track C override provenance, NaN markers, and uncorrupted _ego_goal_memory."""
    import world_models_fpga.research.benchmarks.run_task_conditioned_closed_loop_grounding_v5_5 as v55

    monkeypatch.setattr(v55.mujoco, "mj_name2id", lambda _m, _t, name: 1 if "red" in name else 2)
    monkeypatch.setattr(v55, "render_segmentation_array", lambda _e: np.ones((64, 64, 2), dtype=np.int32))

    assert "C6_track_b_oracle_waypoint_upper_bound" in ALL_CONDITIONS
    assert "C7_track_c_planner_goal_override" in ALL_CONDITIONS

    env = _DummyEnv()
    agent = _make_dummy_agent()
    spec = {
        "pair_id": "SYNTH_000",
        "pair_idx": 0,
        "context": "nominal",
        "target_key": "red_box",
        "distractor_key": "blue_cylinder",
        "initial_pose": [0.0, 0.0, 0.0],
        "waypoints": [[1.2, 0.2, 0.0]],
        "distractor_waypoint": [1.0, -0.6, 0.0],
        "task_token": 1,
        "swapped_task_token": 2,
    }

    # 1. C6 Track B upper bound must NOT execute grounding head/task embedding and must emit explicit NaN/None
    rec_c6 = evaluate_episode_condition(agent, env, spec, "C6_track_b_oracle_waypoint_upper_bound", max_steps=3)
    assert rec_c6["task_embedding_executed"] is False
    assert rec_c6["grounding_head_executed"] is False
    assert rec_c6["memory_branch_executed"] is False
    assert rec_c6["planner_goal_source"] == "track_b_oracle_waypoint"
    assert rec_c6["step0_blended_goal_ang_err_deg"] is None
    assert rec_c6["mean_step_blended_goal_ang_err_deg"] is None
    assert rec_c6["mean_step_vis_goal_ang_err_deg"] is None
    assert rec_c6["mean_step_planner_goal_ang_err_deg"] == pytest.approx(0.0, abs=1e-4)
    for key in ("l1_act", "l2_act", "g_visual", "g_memory", "g_blended", "delta_ego"):
        assert np.all(np.isnan(rec_c6["traces"][key])), f"Expected NaN in Track B trace {key}"
    assert np.allclose(rec_c6["traces"]["planner_goal"], rec_c6["traces"]["gt_target_body"])

    # 2. C7 Track C planner goal override must keep Track C execution intact while overriding ONLY planner goal
    rec_c7 = evaluate_episode_condition(agent, env, spec, "C7_track_c_planner_goal_override", max_steps=3)
    assert rec_c7["task_embedding_executed"] is True
    assert rec_c7["grounding_head_executed"] is True
    assert rec_c7["memory_branch_executed"] is True
    assert rec_c7["planner_goal_source"] == "track_c_planner_goal_override"
    assert rec_c7["mean_step_planner_goal_ang_err_deg"] == pytest.approx(0.0, abs=1e-4)
    assert not np.any(np.isnan(rec_c7["traces"]["l1_act"]))
    assert not np.any(np.isnan(rec_c7["traces"]["l2_act"]))
    assert not np.any(np.isnan(rec_c7["traces"]["g_visual"]))
    assert not np.any(np.isnan(rec_c7["traces"]["g_blended"]))
    # Memory prediction delta_ego is NaN at t=0 (initialized from vis_goal) and executed for t >= 1
    assert np.all(np.isnan(rec_c7["traces"]["delta_ego"][0]))
    assert not np.any(np.isnan(rec_c7["traces"]["delta_ego"][1:]))
    assert np.allclose(rec_c7["traces"]["g_memory"][0], rec_c7["traces"]["g_visual"][0], atol=1e-5)
    assert not np.any(np.isnan(rec_c7["traces"]["g_memory"][1:]))
    # _ego_goal_memory at final step equals Track C g_blended (NOT overwritten by gt_t!)
    assert np.allclose(agent._ego_goal_memory.cpu().numpy(), rec_c7["traces"]["g_blended"][-1], atol=1e-5)
    assert not np.allclose(agent._ego_goal_memory.cpu().numpy()[:2], rec_c7["traces"]["gt_target_body"][-1, :2], atol=1e-3)

    # 3. A1 visual-only disables memory update explicitly across all steps
    rec_a1 = evaluate_episode_condition(agent, env, spec, "A1_visual_only_no_memory", max_steps=3)
    assert rec_a1["memory_branch_executed"] is False
    assert np.all(np.isnan(rec_a1["traces"]["g_memory"]))
    assert np.all(np.isnan(rec_a1["traces"]["delta_ego"]))
    assert agent._ego_goal_memory is None
