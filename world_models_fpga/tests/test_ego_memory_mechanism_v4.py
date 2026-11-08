"""
Unit tests for Phase 5 Experiment C: Ego-Motion & Action-Conditioned Memory Factorial Analysis (v4).
Validates 4 motion sources, coordinate update invariants, and statistical equivalence testing.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "world_models_fpga"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
import torch

from research.circuits.run_ego_memory_mechanism_v4 import (
    transform_ego_goal_vector,
    tost_equivalence_test,
    paired_bootstrap_ci,
)


def test_transform_ego_goal_vector_invariants():
    """Verifies that 2D rigid transform correctly maps egocentric coordinates."""
    # Initial goal: 1.0m directly ahead: [1.0, 0.0, 0.0, 1.0]
    g_0 = torch.tensor([1.0, 0.0, 0.0, 1.0])

    # 1. Forward displacement of 0.3m: remaining distance must be 0.7m ahead
    disp_fwd = torch.tensor([0.3, 0.0, 0.0])
    g_fwd = transform_ego_goal_vector(g_0, disp_fwd)
    assert torch.isclose(g_fwd[0], torch.tensor(0.7), atol=1e-5)
    assert torch.isclose(g_fwd[1], torch.tensor(0.0), atol=1e-5)

    # 2. Pure yaw rotation of +90 deg (+pi/2 rad): goal should shift to right: [0.0, -1.0]
    disp_rot = torch.tensor([0.0, 0.0, np.pi / 2])
    g_rot = transform_ego_goal_vector(g_0, disp_rot)
    assert torch.isclose(g_rot[0], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(g_rot[1], torch.tensor(-1.0), atol=1e-5)


def test_tost_equivalence_test_invariants():
    """Verifies that TOST correctly rejects or accepts non-inferiority/equivalence."""
    # 1. Identical zero difference: must be equivalent within 0.05m
    diffs_zero = np.zeros(50)
    p_tost, is_equiv = tost_equivalence_test(diffs_zero, delta=0.05)
    assert is_equiv is True
    assert p_tost < 0.001

    # 2. Large difference (0.25m): must NOT be equivalent within 0.05m
    diffs_large = np.ones(50) * 0.25
    p_tost_large, is_equiv_large = tost_equivalence_test(diffs_large, delta=0.05)
    assert is_equiv_large is False
    assert p_tost_large > 0.99


def test_ego_memory_v4_artifacts_integrity():
    """Verifies that ego-memory v4 artifacts exist, parse cleanly, and satisfy schemas."""
    results_dir = REPO_ROOT / "world_models_fpga/research/circuits/results"
    summary_path = results_dir / "ego_memory_mechanism_v4_summary.json"
    pairs_path = results_dir / "ego_memory_mechanism_v4_pairs.jsonl"
    steps_path = results_dir / "ego_memory_mechanism_v4_steps.jsonl"
    schema_path = results_dir / "ego_memory_mechanism_v4_schema.md"

    assert summary_path.exists(), "Summary JSON missing"
    assert pairs_path.exists(), "Pairs JSONL missing"
    assert steps_path.exists(), "Steps JSONL missing"
    assert schema_path.exists(), "Schema Markdown missing"

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    # Invariants
    assert summary["sample_counts"]["n_trajectories"] == 50
    conds = summary["factorial_conditions_summary"]
    assert "sensor_fixed" in conds
    assert "zero_fixed" in conds
    assert "action_hold" in conds
    assert "oracle_hold" in conds

    # Verification of Action-Conditioned calibration superiority over sensor delta
    calib = summary["displacement_calibration_offline"]["all"]
    action_mae = calib["action_conditioned_displacement"]["translation_mae_m"]
    zero_mae = calib["zero_baseline"]["translation_mae_m"]
    assert action_mae < zero_mae, "Action-conditioned prediction must have lower MAE than assuming zero motion"

    # Count pair records
    with pairs_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 50
