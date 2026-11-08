import numpy as np
import pytest
from world_models_fpga.research.circuits.run_ego_memory_reanalysis_v5_2 import (
    evaluate_displacement_metrics,
    paired_bootstrap_ci,
    sign_flip_permutation_test,
)


def test_scalar_vs_vector_r2_known_geometry():
    """Tests the distinction between scalar magnitude R^2 and 2D vector variance explained."""
    # Case 1: Predicted vectors have identical magnitudes as physical vectors,
    # but completely wrong directional signs (e.g. [-dx, -dy]).
    # Scalar magnitude regression R^2 should be 1.0 (perfect magnitude tracking).
    # Vector R^2 should be negative (worse than mean prediction).
    rng = np.random.default_rng(42)
    dx = rng.normal(1.0, 0.2, size=100)
    dy = rng.normal(1.0, 0.2, size=100)
    dth = rng.normal(0.0, 0.05, size=100)

    phys = np.column_stack([dx, dy, dth])
    # Flipped sign direction
    pred_flipped = np.column_stack([-dx, -dy, dth])

    res = evaluate_displacement_metrics(phys, pred_flipped)

    # Scalar magnitude is identical: ||[-dx, -dy]|| == ||[dx, dy]||
    assert np.isclose(res["scalar_magnitude_regression"]["r_squared"], 1.0)
    # Vector R^2 must be strongly negative because predictions are in opposite direction
    assert res["variance_explained_r_squared"]["vector_2d_r_squared"] < 0.0
    assert res["variance_explained_r_squared"]["component_dx_r_squared"] < 0.0
    assert res["variance_explained_r_squared"]["component_dy_r_squared"] < 0.0
    # Rotation was untouched: R^2 should be 1.0
    assert np.isclose(res["variance_explained_r_squared"]["angular_dtheta_r_squared"], 1.0)


def test_perfect_prediction_r2():
    """Tests that perfect displacement predictions yield R^2 = 1.0 for all metrics."""
    rng = np.random.default_rng(1337)
    phys = rng.normal(0.05, 0.02, size=(200, 3))
    res = evaluate_displacement_metrics(phys, phys)

    assert np.isclose(res["translation_mae_m"], 0.0)
    assert np.isclose(res["rotation_mae_rad"], 0.0)
    assert np.isclose(res["scalar_magnitude_regression"]["r_squared"], 1.0)
    assert np.isclose(res["variance_explained_r_squared"]["vector_2d_r_squared"], 1.0)
    assert np.isclose(res["variance_explained_r_squared"]["component_dx_r_squared"], 1.0)
    assert np.isclose(res["variance_explained_r_squared"]["component_dy_r_squared"], 1.0)
    assert np.isclose(res["variance_explained_r_squared"]["angular_dtheta_r_squared"], 1.0)


def test_within_trajectory_permutation_properties():
    """Tests that within-trajectory temporal permutation preserves action distribution."""
    rng = np.random.default_rng(2026)
    # 10 actions during blackout
    actions = [[0.1 * i, 0.05 * i, -0.02 * i] for i in range(10)]

    p_order = rng.permutation(10)
    perm_actions = [actions[i] for i in p_order]

    # Length must be preserved
    assert len(perm_actions) == 10
    # Multiset of actions must be identical
    assert sorted(perm_actions) == sorted(actions)
    # Marginal means must be bit-for-bit identical
    assert np.allclose(np.mean(actions, axis=0), np.mean(perm_actions, axis=0))
    # Temporal order should be changed (prob of identity permutation is 1/10! < 3e-7)
    assert not np.array_equal(actions, perm_actions)
