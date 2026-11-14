"""Tests for Phase 5.5 Closed-Loop Mechanistic Grounding helpers and invariants."""

from __future__ import annotations

import numpy as np
import pytest

from world_models_fpga.research.benchmarks.model_wrappers import compute_body_frame_goal_vec
from world_models_fpga.research.benchmarks.run_task_conditioned_closed_loop_grounding_v5_5 import (
    apply_local_background_mask,
    body_goal_to_world_pose,
    build_area_matched_distractor_mask,
    holm_bonferroni_adjust,
    paired_mcnemar_exact,
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
