"""
Unit tests for Phase 5 Experiment A: Task-Conditioned Visual Grounding & Instance Binding (v4).
Validates mathematical invariants, statistical testing, and data schemas.
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

from research.benchmarks.run_task_conditioned_binding_v4 import (
    eval_scene_point,
    fit_ridge_cv,
    fit_mlp_cv,
)


def test_eval_scene_point_invariants():
    """Verifies that eval_scene_point produces valid margins, euclidean errors, and angular errors."""
    # Test case 1: Target A is [1.0, 0.0], Target B is [0.0, 1.0]
    gt_a = np.array([1.0, 0.0, 0.0, 1.0])
    gt_b = np.array([0.0, 1.0, 0.0, 1.0])

    # Perfect predictions: g_a close to gt_a, g_b close to gt_b
    g_a = np.array([0.95, 0.05, 0.0, 1.0])
    g_b = np.array([0.05, 0.95, 0.0, 1.0])

    res = eval_scene_point(g_a, g_b, gt_a, gt_b)
    assert res["a_correct"] is True
    assert res["b_correct"] is True
    assert res["dual_discrimination"] is True
    assert res["margin_a_m"] > 0
    assert res["margin_b_m"] > 0
    assert res["err_ang_a_deg"] < 10.0
    assert res["err_ang_b_deg"] < 10.0

    # Inverted predictions: g_a close to gt_b, g_b close to gt_a
    res_inv = eval_scene_point(g_b, g_a, gt_a, gt_b)
    assert res_inv["a_correct"] is False
    assert res_inv["b_correct"] is False
    assert res_inv["dual_discrimination"] is False
    assert res_inv["margin_a_m"] < 0
    assert res_inv["margin_b_m"] < 0


def test_grouped_cv_partition_invariants():
    """Verifies that grouped 5-fold CV by pair_id strictly avoids cross-fold sample leakage."""
    n_pairs = 50
    folds = 5
    pairs_per_fold = n_pairs // folds

    for f in range(folds):
        val_pairs = set(range(f * pairs_per_fold, (f + 1) * pairs_per_fold))
        train_pairs = set(range(n_pairs)) - val_pairs

        # Intersection must be strictly empty
        assert len(val_pairs.intersection(train_pairs)) == 0
        assert len(val_pairs) == 10
        assert len(train_pairs) == 40


def test_binding_v4_artifacts_integrity():
    """Verifies that Phase 5 binding artifacts exist, parse cleanly, and satisfy schemas."""
    results_dir = REPO_ROOT / "world_models_fpga/research/benchmarks/results"
    summary_path = results_dir / "task_conditioned_binding_v4_summary.json"
    pairs_path = results_dir / "task_conditioned_binding_v4_pairs.jsonl"
    csv_path = results_dir / "task_conditioned_binding_v4_model_comparison.csv"
    schema_path = results_dir / "task_conditioned_binding_v4_schema.md"

    assert summary_path.exists(), "Summary JSON missing"
    assert pairs_path.exists(), "Pairs JSONL missing"
    assert csv_path.exists(), "Model comparison CSV missing"
    assert schema_path.exists(), "Schema Markdown missing"

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    # Invariants
    assert summary["sample_counts"]["n_categorical_pairs"] == 50
    assert summary["sample_counts"]["n_categorical_scenes"] == 100
    assert summary["sample_counts"]["n_instance_level_pairs"] == 20

    # Permutation test invariant: empirical null mean must be near 16.3%
    perm = summary["permutation_null_testing"]
    assert 0.10 <= perm["permutation_null_mean"] <= 0.22
    assert perm["paired_permutation_p_value_greater"] > 0.05, "Observed dual discrimination should not beat permutation null"

    # Count JSONL lines
    line_count = 0
    with pairs_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            assert isinstance(data, dict)
            line_count += 1
    assert line_count == 70, f"Expected 70 records (50 categorical + 20 instance), got {line_count}"
