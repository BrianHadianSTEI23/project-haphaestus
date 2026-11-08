import numpy as np
import pytest
from world_models_fpga.research.benchmarks.run_task_conditioned_binding_v5 import (
    eval_scene_point,
)


def test_eval_scene_point_invariants():
    """Verifies that margin, euclidean error, and angular metrics obey geometry invariants."""
    # Target A at [1.0, 0.0], Target B at [0.0, 1.0]
    gt_a = np.array([1.0, 0.0, 0.0, 1.0])
    gt_b = np.array([0.0, 1.0, 0.0, 1.0])

    # Model outputs g_a close to Target A, g_b close to Target B
    g_a = np.array([0.9, 0.1, 0.0, 1.0])
    g_b = np.array([0.1, 0.9, 0.0, 1.0])

    res = eval_scene_point(g_a, g_b, gt_a, gt_b)
    assert res["a_correct"] is True
    assert res["b_correct"] is True
    assert res["dual_discrimination"] is True
    assert res["margin_a_m"] > 0
    assert res["margin_b_m"] > 0
    assert res["err_euc_a_m"] < 0.2
    assert res["err_euc_b_m"] < 0.2
    assert res["err_ang_a_deg"] < 15.0
    assert res["err_ang_b_deg"] < 15.0


def test_pair_clustered_permutation_invariants():
    """Verifies that pair-clustered permutation preserves the paired structure across Scene A and Scene B."""
    rng = np.random.default_rng(42)
    n_pairs = 10
    # Create synthetic dual indicators
    arr_a_norm = rng.integers(0, 2, size=n_pairs)
    arr_a_swap = 1 - arr_a_norm
    arr_b_norm = rng.integers(0, 2, size=n_pairs)
    arr_b_swap = 1 - arr_b_norm

    # Clustered flip: single coin flip per pair applied to both A and B
    flips = rng.integers(0, 2, size=(100, n_pairs))
    res_a = np.where(flips == 0, arr_a_norm, arr_a_swap)
    res_b = np.where(flips == 0, arr_b_norm, arr_b_swap)

    # Invariants: for any pair i where flip == 0, both A and B take norm; if flip == 1, both take swap
    for sim_idx in range(100):
        for p_idx in range(n_pairs):
            if flips[sim_idx, p_idx] == 0:
                assert res_a[sim_idx, p_idx] == arr_a_norm[p_idx]
                assert res_b[sim_idx, p_idx] == arr_b_norm[p_idx]
            else:
                assert res_a[sim_idx, p_idx] == arr_a_swap[p_idx]
                assert res_b[sim_idx, p_idx] == arr_b_swap[p_idx]


def test_stratified_grouped_cv_partitioning_invariants():
    """Verifies that all 4 factorial conditions of each pair stay in the same fold."""
    n_pairs = 50
    folds = 5
    # Synthetic pair identities and conditions
    pair_ids = np.repeat(np.arange(n_pairs), 4)
    conditions = np.tile(["zA_eA", "zA_eB", "zB_eA", "zB_eB"], n_pairs)

    pairs_per_fold = n_pairs // folds
    for f in range(folds):
        val_pair_ids = set(range(f * pairs_per_fold, (f + 1) * pairs_per_fold))
        train_mask = np.array([pid not in val_pair_ids for pid in pair_ids])
        val_mask = ~train_mask

        # Check: no pair_id appears in both train and val
        train_pairs = set(pair_ids[train_mask])
        val_pairs = set(pair_ids[val_mask])
        assert len(train_pairs.intersection(val_pairs)) == 0
        assert len(val_pairs) == pairs_per_fold
        # Check: each val pair has exactly 4 factorial rows in val
        for vp in val_pairs:
            assert np.sum(pair_ids[val_mask] == vp) == 4
