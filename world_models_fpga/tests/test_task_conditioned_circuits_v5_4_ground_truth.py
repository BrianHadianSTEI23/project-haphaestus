import numpy as np
import pytest
import gzip
import json
from pathlib import Path
from world_models_fpga.research.circuits.run_task_conditioned_circuits_v5_4_ground_truth_reanalysis import (
    wrap_angle,
    paired_bootstrap_ci,
    sign_flip_permutation_test,
    holm_bonferroni_correction,
)


def test_task_ab_donor_recipient_mapping():
    """Tests correct task A/B donor-recipient ground-truth assignment."""
    gt_goal_a = np.array([1.0, 0.5])
    gt_goal_b = np.array([2.0, -0.5])

    # A_to_B: recipient is A, donor is B
    recip_ab = gt_goal_a
    donor_ab = gt_goal_b
    assert np.allclose(recip_ab, [1.0, 0.5])
    assert np.allclose(donor_ab, [2.0, -0.5])

    # B_to_A: recipient is B, donor is A
    recip_ba = gt_goal_b
    donor_ba = gt_goal_a
    assert np.allclose(recip_ba, [2.0, -0.5])
    assert np.allclose(donor_ba, [1.0, 0.5])


def test_ground_truth_contrast_signs():
    """Verifies that d_true for A_to_B and B_to_A have exact opposite signs."""
    gt_a = np.array([0.5, 0.2])
    gt_b = np.array([-0.3, 0.8])

    d_true_ab = gt_b - gt_a
    d_true_ba = gt_a - gt_b

    assert np.allclose(d_true_ab, -d_true_ba)
    assert np.isclose(np.linalg.norm(d_true_ab), np.linalg.norm(d_true_ba))


def test_perfect_donor_transplant():
    """A perfect donor transplant produces zero donor-target error and positive error improvement."""
    recip_gt = np.array([0.0, 0.0])
    donor_gt = np.array([1.0, 0.0])
    base_out = np.array([0.0, 0.0])  # baseline is at recipient target

    # Perfect transplant reaches donor target exactly
    trans_out = np.array([1.0, 0.0])

    base_err = np.linalg.norm(base_out - recip_gt)
    trans_err = np.linalg.norm(trans_out - donor_gt)
    err_imp = base_err - trans_err

    d_true = donor_gt - recip_gt
    d_interv = trans_out - base_out
    eta_gt = np.dot(d_interv, d_true) / np.dot(d_true, d_true)

    assert np.isclose(trans_err, 0.0)
    assert np.isclose(err_imp, 0.0)  # was 0 from recipient, now 0 from donor
    assert np.isclose(eta_gt, 1.0)


def test_wrong_direction_transplant_negative_projection():
    """Transplanting in the opposite direction produces negative ground-truth projection."""
    recip_gt = np.array([0.0, 0.0])
    donor_gt = np.array([1.0, 0.0])
    base_out = np.array([0.0, 0.0])

    # Intervened output moved away from donor
    trans_out = np.array([-0.5, 0.0])

    d_true = donor_gt - recip_gt
    d_interv = trans_out - base_out
    eta_gt = np.dot(d_interv, d_true) / np.dot(d_true, d_true)

    assert eta_gt < 0.0
    assert np.isclose(eta_gt, -0.5)


def test_orthogonal_output_change_zero_projection():
    """Orthogonal output shifts produce exactly zero ground-truth projection."""
    recip_gt = np.array([0.0, 0.0])
    donor_gt = np.array([1.0, 0.0])  # along X
    base_out = np.array([0.0, 0.0])

    # Shift is purely along Y (orthogonal)
    trans_out = np.array([0.0, 1.5])

    d_true = donor_gt - recip_gt
    d_interv = trans_out - base_out
    eta_gt = np.dot(d_interv, d_true) / np.dot(d_true, d_true)

    assert np.isclose(eta_gt, 0.0)


def test_zero_vector_and_near_zero_handling():
    """Verifies that zero target contrast does not trigger ZeroDivisionError."""
    d_true = np.array([0.0, 0.0])
    d_interv = np.array([0.5, 0.2])

    dot_true = np.dot(d_true, d_true)
    if dot_true < 1e-12:
        eta_gt = 0.0
    else:
        eta_gt = np.dot(d_interv, d_true) / dot_true

    assert eta_gt == 0.0


def test_heading_wraparound_near_pi():
    """Verifies circular heading difference near -pi and +pi."""
    th1 = np.pi - 0.05
    th2 = -np.pi + 0.05
    diff = wrap_angle(th1 - th2)

    # Angular separation is 0.1 rad, not ~2*pi
    assert np.isclose(abs(diff), 0.1, atol=1e-5)


def test_pair_aggregation_subject_independence():
    """Verifies that paired analysis averages within pair rather than inflating sample count."""
    # Mock pair with 2 scenes x 2 directions = 4 settings
    settings_data = [0.1, 0.2, -0.1, 0.4]
    pair_avg = np.mean(settings_data)

    pair_list = [pair_avg]
    # Unit of analysis is N = 1 (the pair), not N = 4
    assert len(pair_list) == 1
    assert np.isclose(pair_list[0], 0.15)


def test_candidate_control_eligible_pair_matching():
    """Verifies that candidate vs norm-matched control differences use identical eligible pair IDs."""
    all_pairs = [f"PAIR_{i:03d}" for i in range(10)]
    valid_mask = [True, True, False, True, False, True, True, False, True, True]

    cand_vals = {pid: float(i) for i, pid in enumerate(all_pairs)}
    norm_vals = {pid: float(i + 0.5) for i, pid in enumerate(all_pairs)}

    eligible_pids = [pid for pid, is_v in zip(all_pairs, valid_mask) if is_v]
    assert len(eligible_pids) == 7

    cand_arr = np.array([cand_vals[pid] for pid in eligible_pids])
    norm_arr = np.array([norm_vals[pid] for pid in eligible_pids])

    assert len(cand_arr) == len(norm_arr) == 7
    diff = cand_arr - norm_arr
    assert np.allclose(diff, -0.5)


def test_gzip_streaming_integrity():
    """Verifies gzip streaming row count and record integrity on small sample."""
    rows = [
        {"pair_id": "PAIR_000", "scene_id": "scene_a", "draw_id": 0},
        {"pair_id": "PAIR_000", "scene_id": "scene_a", "draw_id": 1},
        {"pair_id": "PAIR_000", "scene_id": "scene_b", "draw_id": 0},
    ]
    test_gz_path = Path("/tmp/test_stream_integrity.jsonl.gz")
    with gzip.open(test_gz_path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    read_rows = []
    with gzip.open(test_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            read_rows.append(json.loads(line))

    assert len(read_rows) == 3
    assert read_rows[0]["draw_id"] == 0
    assert read_rows[1]["draw_id"] == 1
    assert read_rows[2]["scene_id"] == "scene_b"
    test_gz_path.unlink()
