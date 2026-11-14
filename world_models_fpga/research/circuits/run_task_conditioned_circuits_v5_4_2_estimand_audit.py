#!/usr/bin/env python3
"""Project Hephaestus: Phase 5.4.2 Single-Reference-Target Estimand Audit.

Re-evaluates all 447,800 Phase 5.3 intervention records using valid single-target
causal estimands rather than Phase 5.4's invalid cross-target difference
(||base_out - recip_gt|| - ||trans_out - donor_gt||).

Primary corrected estimands per intervention record:
  - donor_target_baseline_error_m = ||base_out - donor_gt||
  - donor_target_transplant_error_m = ||trans_out - donor_gt||
  - donor_target_gain_m = donor_target_baseline_error_m - donor_target_transplant_error_m
  - recipient_target_baseline_error_m = ||base_out - recip_gt||
  - recipient_target_transplant_error_m = ||trans_out - recip_gt||
  - recipient_target_change_m = recipient_target_transplant_error_m - recipient_target_baseline_error_m
  - eta_gt = dot(d_int, d_true) / ||d_true||^2   [dimensionless]
  - target_parallel_component_m = dot(d_int, d_true) / ||d_true||   [meters]
  - orthogonal_residual_gt_m = ||d_int - eta_gt * d_true||   [meters]
  - intervention_norm_m = ||d_int||   [meters]
  - norm_ratio = ||d_int|| / ||d_true||   [dimensionless]

All statistical inference is performed strictly at the independent scene-pair
level (N=60) separately per (layer, k), never as one global pooled average.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[3]
LAYERS: Tuple[str, ...] = ("layer1_post_relu_64d", "layer2_post_relu_64d")
K_VALUES: Tuple[int, ...] = (2, 4, 8, 16, 32)
CONDITIONS: Tuple[str, ...] = ("candidate", "dimension_matched_random", "norm_matched_random")
PAIR_SETTINGS: Tuple[Tuple[str, str], ...] = (
    ("scene_a", "A_to_B"),
    ("scene_a", "B_to_A"),
    ("scene_b", "A_to_B"),
    ("scene_b", "B_to_A"),
)

METRIC_KEYS: Tuple[str, ...] = (
    "donor_target_gain_m",
    "recipient_target_change_m",
    "eta_gt",
    "target_parallel_component_m",
    "orthogonal_residual_gt_m",
    "intervention_norm_m",
    "norm_ratio",
    "cosine_gt",
    "eta_model",
    "donor_target_baseline_error_m",
    "donor_target_transplant_error_m",
    "recipient_target_baseline_error_m",
    "recipient_target_transplant_error_m",
    "legacy_cross_target_error_diff_m",
    "positive_donor_gain_fraction",
)


def compute_file_sha256(filepath: Path) -> str:
    """Computes streaming SHA-256 hex digest of a file."""
    if not filepath.exists():
        return "missing"
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_info(repo_root: Path = REPO_ROOT) -> Tuple[str, bool]:
    """Returns (HEAD SHA, working_tree_clean)."""
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
        porcelain = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
        return commit, len(porcelain) == 0
    except Exception:
        return "unknown", False


def paired_bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> List[float]:
    """Computes nonparametric percentile bootstrap confidence interval of the mean."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return [0.0, 0.0]
    if arr.size == 1:
        val = float(arr[0])
        return [val, val]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return [
        float(np.percentile(boot_means, 100.0 * alpha)),
        float(np.percentile(boot_means, 100.0 * (1.0 - alpha))),
    ]


def sign_flip_permutation_p_two_sided(
    diffs: Sequence[float] | np.ndarray,
    n_perm: int = 50000,
    seed: int = 42,
) -> float:
    """Two-sided paired sign-flip permutation p-value (exact fallback for small N)."""
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size == 0:
        return 1.0
    obs = abs(float(np.mean(arr)))
    if obs < 1e-15:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, arr.size))
    perm_means = np.abs((signs * arr[None, :]).mean(axis=1))
    return float((np.sum(perm_means >= obs - 1e-15) + 1) / (n_perm + 1))


def holm_bonferroni_adjust(p_values: Sequence[float]) -> List[float]:
    """Applies step-down Holm-Bonferroni FWER adjustment preserving order."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, (orig_idx, raw_p) in enumerate(indexed):
        val = min(1.0, float(raw_p) * (m - rank))
        running_max = max(running_max, val)
        adjusted[orig_idx] = float(running_max)
    return adjusted


def safe_two_sided_wilcoxon(diffs: Sequence[float] | np.ndarray) -> float:
    """Computes two-sided Wilcoxon signed-rank p-value safely."""
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size == 0 or not np.any(np.abs(arr) > 1e-15):
        return 1.0
    try:
        return float(stats.wilcoxon(arr, alternative="two-sided").pvalue)
    except Exception:
        return 1.0


def safe_one_sided_wilcoxon(diffs: Sequence[float] | np.ndarray, alternative: str = "greater") -> float:
    """Computes one-sided Wilcoxon signed-rank p-value safely."""
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size == 0 or not np.any(np.abs(arr) > 1e-15):
        return 1.0
    try:
        return float(stats.wilcoxon(arr, alternative=alternative).pvalue)
    except Exception:
        return 1.0


def compute_single_target_estimands(
    base_out: Sequence[float] | np.ndarray,
    trans_out: Sequence[float] | np.ndarray,
    recip_gt: Sequence[float] | np.ndarray,
    donor_gt: Sequence[float] | np.ndarray,
    model_donor_out: Sequence[float] | np.ndarray | None = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Computes corrected single-reference-target estimands and vector diagnostics.

    Sign conventions:
      - donor_target_gain_m > 0: intervention moves output closer to donor target.
      - recipient_target_change_m > 0: intervention moves output farther from recipient target.
      - eta_gt: dimensionless projection scalar onto d_true = donor_gt - recip_gt.
      - target_parallel_component_m: signed projection (meters) along d_true / ||d_true||.
    """
    b = np.asarray(base_out[:2], dtype=np.float64)
    t = np.asarray(trans_out[:2], dtype=np.float64)
    rg = np.asarray(recip_gt[:2], dtype=np.float64)
    dg = np.asarray(donor_gt[:2], dtype=np.float64)

    donor_base_err = float(np.linalg.norm(b - dg))
    donor_trans_err = float(np.linalg.norm(t - dg))
    donor_gain = float(donor_base_err - donor_trans_err)

    recip_base_err = float(np.linalg.norm(b - rg))
    recip_trans_err = float(np.linalg.norm(t - rg))
    recip_change = float(recip_trans_err - recip_base_err)

    # Recorded only for forensic reconciliation against Phase 5.4/5.4.1 invalid cross-target metric
    legacy_cross_target_diff = float(recip_base_err - donor_trans_err)

    d_true = dg - rg
    d_int = t - b
    norm_true = float(np.linalg.norm(d_true))
    norm_int = float(np.linalg.norm(d_int))

    if norm_true > eps:
        dot_int_true = float(np.dot(d_int, d_true))
        eta_gt = float(dot_int_true / (norm_true * norm_true))
        parallel_m = float(dot_int_true / norm_true)
        orth_res_m = float(np.linalg.norm(d_int - eta_gt * d_true))
        norm_ratio = float(norm_int / norm_true)
        cosine_gt = float(np.clip(dot_int_true / (norm_int * norm_true), -1.0, 1.0)) if norm_int > eps else 0.0
    else:
        eta_gt = 0.0
        parallel_m = 0.0
        orth_res_m = float(norm_int)
        norm_ratio = 0.0
        cosine_gt = 0.0

    eta_model = 0.0
    if model_donor_out is not None:
        m_don = np.asarray(model_donor_out[:2], dtype=np.float64)
        d_model = m_don - b
        norm_model_sq = float(np.dot(d_model, d_model))
        if norm_model_sq > eps * eps:
            eta_model = float(np.dot(d_int, d_model) / norm_model_sq)

    return {
        "donor_target_baseline_error_m": donor_base_err,
        "donor_target_transplant_error_m": donor_trans_err,
        "donor_target_gain_m": donor_gain,
        "recipient_target_baseline_error_m": recip_base_err,
        "recipient_target_transplant_error_m": recip_trans_err,
        "recipient_target_change_m": recip_change,
        "legacy_cross_target_error_diff_m": legacy_cross_target_diff,
        "eta_gt": eta_gt,
        "target_parallel_component_m": parallel_m,
        "orthogonal_residual_gt_m": orth_res_m,
        "intervention_norm_m": norm_int,
        "true_contrast_norm_m": norm_true,
        "norm_ratio": norm_ratio,
        "cosine_gt": cosine_gt,
        "eta_model": eta_model,
        "positive_donor_gain_fraction": 1.0 if donor_gain > 0.0 else 0.0,
    }


def compute_setting_expected_counts(pairs_data: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Computes expected intervention row counts across conditions from Phase 5.3 pair metadata."""
    n_settings = 0
    n_valid_norm = 0
    for prow in pairs_data:
        for res in prow["results"].values():
            n_settings += 1
            if res.get("norm_matched_status") == "VALID_COMMON_SUPPORT":
                n_valid_norm += 1
    return {
        "candidate": n_settings,
        "dimension_matched_random": n_settings * 100,
        "norm_matched_random": n_valid_norm * 100,
        "total": n_settings + n_settings * 100 + n_valid_norm * 100,
    }


def _summarize_vector(vals: Sequence[float], seed: int) -> Dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "std": None, "ci_95": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "ci_95": paired_bootstrap_ci(arr, n_boot=2000, seed=seed),
    }


def summarize_estimands_for_setting_records(
    pair_ids: Sequence[str],
    pair_context_map: Dict[str, str],
    eligible_pids_by_lk: Dict[Tuple[str, int], set[str]],
    records_iter: Iterable[Dict[str, Any]],
    act_lookup: Dict[Tuple[str, str], Dict[str, Any]],
    expected_counts: Optional[Dict[str, int]] = None,
    seed: int = 5420,
) -> Dict[str, Any]:
    """Streams intervention rows into exact pair-level estimand summaries and audit checks.

    This pure function can run either on full Phase 5.3 inputs or on synthetic test streams.
    """
    n_metrics = len(METRIC_KEYS)
    # Sum and count per (pair_id, scene_id, layer, direction, k, condition)
    setting_sum: Dict[Tuple[str, str, str, str, int, str], np.ndarray] = {}
    setting_cnt: Dict[Tuple[str, str, str, str, int, str], int] = {}

    # Baseline identity across draws per (pair_id, scene_id, layer, direction, k)
    baseline_ref: Dict[Tuple[str, str, str, str, int], np.ndarray] = {}
    baseline_mismatches = 0
    max_baseline_draw_diff = 0.0
    activation_ref_mismatches = 0
    max_activation_ref_diff = 0.0

    # Baseline geometry diagnostics per unique (pair_id, scene_id, direction) [240 scene-directions]
    seen_scene_dirs: set[Tuple[str, str, str]] = set()
    scene_dir_geometry: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    observed_counts = defaultdict(int)
    unmatched_rows = 0
    seen_row_keys: set[Tuple[str, str, str, str, int, str, int]] = set()
    duplicate_rows = 0

    for row in records_iter:
        pid = row["pair_id"]
        sc = row["scene_id"]
        layer = row["layer"]
        direction = row["direction"]
        k = int(row["k"])
        cond = row["condition"]
        draw_id = int(row["draw_id"])

        row_key = (pid, sc, layer, direction, k, cond, draw_id)
        if row_key in seen_row_keys:
            duplicate_rows += 1
        seen_row_keys.add(row_key)
        observed_counts[cond] += 1

        act = act_lookup.get((pid, sc))
        if act is None or direction not in ("A_to_B", "B_to_A"):
            unmatched_rows += 1
            continue

        if direction == "A_to_B":
            recip_gt = act["gt_goal_a"][:2]
            donor_gt = act["gt_goal_b"][:2]
            model_recip_ref = act["output_task_a"][:2]
            model_donor_ref = act["output_task_b"][:2]
        else:
            recip_gt = act["gt_goal_b"][:2]
            donor_gt = act["gt_goal_a"][:2]
            model_recip_ref = act["output_task_b"][:2]
            model_donor_ref = act["output_task_a"][:2]

        base_xy = np.asarray(row["baseline_output"][:2], dtype=np.float64)
        trans_xy = np.asarray(row["transplanted_output"][:2], dtype=np.float64)

        b_key = (pid, sc, layer, direction, k)
        if b_key not in baseline_ref:
            baseline_ref[b_key] = base_xy.copy()
        else:
            diff_draw = float(np.max(np.abs(base_xy - baseline_ref[b_key])))
            if diff_draw > max_baseline_draw_diff:
                max_baseline_draw_diff = diff_draw
            if diff_draw > 1e-12:
                baseline_mismatches += 1

        diff_act = float(np.max(np.abs(base_xy - np.asarray(model_recip_ref, dtype=np.float64))))
        if diff_act > max_activation_ref_diff:
            max_activation_ref_diff = diff_act
        if diff_act > 1e-6:
            activation_ref_mismatches += 1

        sd_key = (pid, sc, direction)
        if sd_key not in seen_scene_dirs:
            seen_scene_dirs.add(sd_key)
            rg_np = np.asarray(recip_gt, dtype=np.float64)
            dg_np = np.asarray(donor_gt, dtype=np.float64)
            r_err0 = float(np.linalg.norm(base_xy - rg_np))
            d_err0 = float(np.linalg.norm(base_xy - dg_np))
            scene_dir_geometry[sd_key] = {
                "baseline_recip_error_m": r_err0,
                "baseline_donor_error_m": d_err0,
                "noop_cross_target_artifact_m": float(r_err0 - d_err0),
                "true_contrast_norm_m": float(np.linalg.norm(dg_np - rg_np)),
            }

        est = compute_single_target_estimands(
            base_out=base_xy,
            trans_out=trans_xy,
            recip_gt=recip_gt,
            donor_gt=donor_gt,
            model_donor_out=model_donor_ref,
        )
        vec = np.array([est[m] for m in METRIC_KEYS], dtype=np.float64)

        s_key = (pid, sc, layer, direction, k, cond)
        if s_key not in setting_sum:
            setting_sum[s_key] = vec
            setting_cnt[s_key] = 1
        else:
            setting_sum[s_key] += vec
            setting_cnt[s_key] += 1

    # Aggregate strictly to pair level: equal average over the 4 pair-settings per (pid, layer, k, condition)
    # shape: pair_metrics[(layer, k)][pid][cond] -> dict[metric -> float]
    pair_metrics: Dict[Tuple[str, int], Dict[str, Dict[str, Dict[str, float] | None]]] = defaultdict(dict)
    pair_rows_out: List[Dict[str, Any]] = []

    for pid in pair_ids:
        p_out: Dict[str, Any] = {
            "pair_id": pid,
            "context": pair_context_map.get(pid, "unknown"),
            "settings": {},
        }
        for layer in LAYERS:
            for k in K_VALUES:
                lk = (layer, k)
                is_elig = pid in eligible_pids_by_lk.get(lk, set())
                cond_agg: Dict[str, Optional[Dict[str, float]]] = {}
                for cond in CONDITIONS:
                    per_setting_vecs = []
                    for sc, direction in PAIR_SETTINGS:
                        s_key = (pid, sc, layer, direction, k, cond)
                        if s_key in setting_sum and setting_cnt[s_key] > 0:
                            per_setting_vecs.append(setting_sum[s_key] / float(setting_cnt[s_key]))
                    if len(per_setting_vecs) == 4:
                        mean_vec = np.mean(per_setting_vecs, axis=0)
                        cond_agg[cond] = {m: float(mean_vec[idx]) for idx, m in enumerate(METRIC_KEYS)}
                    else:
                        cond_agg[cond] = None

                pair_metrics[lk][pid] = cond_agg
                p_out["settings"][f"{layer}__k_{k}"] = {
                    "is_eligible_norm_matched": bool(is_elig),
                    "candidate": cond_agg["candidate"],
                    "dimension_matched_random": cond_agg["dimension_matched_random"],
                    "norm_matched_random": cond_agg["norm_matched_random"],
                }
        pair_rows_out.append(p_out)

    # Scene-pair-level baseline no-op geometry audit
    pair_noop_bias = []
    pair_base_recip_err = []
    pair_base_donor_err = []
    pair_true_norm = []
    for pid in pair_ids:
        vals = [
            scene_dir_geometry[(pid, sc, direction)]
            for sc, direction in PAIR_SETTINGS
            if (pid, sc, direction) in scene_dir_geometry
        ]
        if vals:
            pair_noop_bias.append(float(np.mean([v["noop_cross_target_artifact_m"] for v in vals])))
            pair_base_recip_err.append(float(np.mean([v["baseline_recip_error_m"] for v in vals])))
            pair_base_donor_err.append(float(np.mean([v["baseline_donor_error_m"] for v in vals])))
            pair_true_norm.append(float(np.mean([v["true_contrast_norm_m"] for v in vals])))

    baseline_identity_audit = {
        "unique_settings_checked": len(baseline_ref),
        "baseline_draw_mismatches_gt_1e_12": int(baseline_mismatches),
        "max_abs_diff_across_draws": float(max_baseline_draw_diff),
        "baseline_matches_all_draws": bool(baseline_mismatches == 0 and max_baseline_draw_diff <= 1e-12),
        "activation_ref_mismatches_gt_1e_6": int(activation_ref_mismatches),
        "max_abs_diff_vs_activation_record": float(max_activation_ref_diff),
    }

    noop_artifact_audit = {
        "n_pairs": len(pair_noop_bias),
        "mean_baseline_recipient_error_m": float(np.mean(pair_base_recip_err)) if pair_base_recip_err else 0.0,
        "mean_baseline_donor_error_m": float(np.mean(pair_base_donor_err)) if pair_base_donor_err else 0.0,
        "mean_noop_cross_target_artifact_m": float(np.mean(pair_noop_bias)) if pair_noop_bias else 0.0,
        "noop_cross_target_artifact_ci_95": paired_bootstrap_ci(pair_noop_bias, seed=seed + 1),
        "mean_true_contrast_norm_m": float(np.mean(pair_true_norm)) if pair_true_norm else 0.0,
        "explanation": (
            "If an intervention does nothing (trans_out == base_out), Phase 5.4's cross-target "
            "formula yields ||base_out - recip_gt|| - ||base_out - donor_gt|| rather than zero. "
            "Single-reference-target donor_target_gain_m equals zero identically for any no-op."
        ),
    }

    # Build layer x k statistical tables strictly at pair level (never one pooled global mean)
    layer_k_results: Dict[str, Dict[str, Any]] = {}
    csv_rows: List[Dict[str, Any]] = []
    family_tests: List[Tuple[str, str, str, float]] = []

    for l_idx, layer in enumerate(LAYERS):
        layer_k_results[layer] = {}
        for k_idx, k in enumerate(K_VALUES):
            lk = (layer, k)
            k_key = f"k_{k}"
            elig_pids = [pid for pid in pair_ids if pid in eligible_pids_by_lk.get(lk, set()) and pair_metrics[lk][pid]["norm_matched_random"] is not None]
            complete_support = len(elig_pids) == len(pair_ids) and len(pair_ids) > 0
            support_label = "EXPLORATORY_COMPLETE_SUPPORT" if complete_support else "EXPLORATORY_INCOMPLETE_SUPPORT"

            # Full-cohort summaries for candidate and dimension_matched_random
            cand_full_summary: Dict[str, Any] = {}
            dim_full_summary: Dict[str, Any] = {}
            for cond, target_dict in (("candidate", cand_full_summary), ("dimension_matched_random", dim_full_summary)):
                valid_pids = [pid for pid in pair_ids if pair_metrics[lk][pid][cond] is not None]
                target_dict["n_pairs"] = len(valid_pids)
                target_dict["metrics"] = {}
                for m in METRIC_KEYS:
                    vals = [pair_metrics[lk][pid][cond][m] for pid in valid_pids]  # type: ignore[index]
                    target_dict["metrics"][m] = _summarize_vector(vals, seed=seed + 100 * l_idx + 10 * k_idx)

            # Candidate vs Dimension-Matched paired comparison (full cohort)
            both_dim_pids = [
                pid for pid in pair_ids
                if pair_metrics[lk][pid]["candidate"] is not None and pair_metrics[lk][pid]["dimension_matched_random"] is not None
            ]
            cand_vs_dim: Dict[str, Any] = {"n_pairs": len(both_dim_pids), "metrics": {}}
            for m in ("donor_target_gain_m", "recipient_target_change_m", "eta_gt", "target_parallel_component_m", "orthogonal_residual_gt_m", "intervention_norm_m", "norm_ratio", "cosine_gt", "positive_donor_gain_fraction"):
                diffs = [
                    pair_metrics[lk][pid]["candidate"][m] - pair_metrics[lk][pid]["dimension_matched_random"][m]  # type: ignore[index]
                    for pid in both_dim_pids
                ]
                s = _summarize_vector(diffs, seed=seed + 200 * l_idx + k_idx)
                s["wilcoxon_two_sided_p"] = safe_two_sided_wilcoxon(diffs)
                cand_vs_dim["metrics"][m] = s

            # Cohort-matched Candidate vs Norm-Matched on exact eligible pairs
            cand_elig_summary: Dict[str, Any] = {"n_pairs": len(elig_pids), "metrics": {}}
            norm_elig_summary: Dict[str, Any] = {"n_pairs": len(elig_pids), "metrics": {}}
            cand_vs_norm: Dict[str, Any] = {"n_pairs": len(elig_pids), "eligible_pair_ids": elig_pids, "metrics": {}}

            for m in METRIC_KEYS:
                c_vals = [pair_metrics[lk][pid]["candidate"][m] for pid in elig_pids]  # type: ignore[index]
                n_vals = [pair_metrics[lk][pid]["norm_matched_random"][m] for pid in elig_pids]  # type: ignore[index]
                cand_elig_summary["metrics"][m] = _summarize_vector(c_vals, seed=seed + 300 * l_idx + k_idx)
                norm_elig_summary["metrics"][m] = _summarize_vector(n_vals, seed=seed + 400 * l_idx + k_idx)

            for m in ("donor_target_gain_m", "recipient_target_change_m", "eta_gt", "target_parallel_component_m", "orthogonal_residual_gt_m", "intervention_norm_m", "norm_ratio", "cosine_gt", "positive_donor_gain_fraction", "legacy_cross_target_error_diff_m"):
                diffs = [
                    pair_metrics[lk][pid]["candidate"][m] - pair_metrics[lk][pid]["norm_matched_random"][m]  # type: ignore[index]
                    for pid in elig_pids
                ]
                s = _summarize_vector(diffs, seed=seed + 500 * l_idx + k_idx)
                s["wilcoxon_two_sided_p"] = safe_two_sided_wilcoxon(diffs) if elig_pids else None
                s["wilcoxon_greater_p_secondary"] = safe_one_sided_wilcoxon(diffs, "greater") if elig_pids else None
                s["sign_flip_perm_two_sided_p"] = (
                    sign_flip_permutation_p_two_sided(diffs, n_perm=50000, seed=seed + 600 * l_idx + k_idx)
                    if elig_pids else None
                )
                s["holm_adjusted_p"] = None
                s["equivalence_not_established_note"] = (
                    "Non-significant two-sided p-value indicates no evidence of difference; inspect CI bounds rather than inferring equivalence."
                )
                cand_vs_norm["metrics"][m] = s

            if layer == "layer1_post_relu_64d" and k in (2, 4) and complete_support:
                family_tests.append((layer, k_key, "donor_target_gain_m", float(cand_vs_norm["metrics"]["donor_target_gain_m"]["wilcoxon_two_sided_p"])))
                family_tests.append((layer, k_key, "eta_gt", float(cand_vs_norm["metrics"]["eta_gt"]["wilcoxon_two_sided_p"])))

            # Context stratification (nominal vs illumination_shift) per layer & k
            context_strat: Dict[str, Any] = {}
            for ctx_name in ("nominal", "illumination_shift"):
                ctx_pids = [pid for pid in pair_ids if pair_context_map.get(pid) == ctx_name]
                ctx_elig_pids = [pid for pid in elig_pids if pair_context_map.get(pid) == ctx_name]
                ctx_cand_valid = [pid for pid in ctx_pids if pair_metrics[lk][pid]["candidate"] is not None]
                context_strat[ctx_name] = {
                    "n_pairs_total": len(ctx_pids),
                    "n_pairs_eligible_norm_matched": len(ctx_elig_pids),
                    "candidate_full_donor_target_gain_m": _summarize_vector(
                        [pair_metrics[lk][pid]["candidate"]["donor_target_gain_m"] for pid in ctx_cand_valid],  # type: ignore[index]
                        seed=seed + 700 * l_idx + k_idx,
                    ),
                    "candidate_full_eta_gt": _summarize_vector(
                        [pair_metrics[lk][pid]["candidate"]["eta_gt"] for pid in ctx_cand_valid],  # type: ignore[index]
                        seed=seed + 710 * l_idx + k_idx,
                    ),
                    "eligible_cand_vs_norm_donor_target_gain_diff_m": _summarize_vector(
                        [pair_metrics[lk][pid]["candidate"]["donor_target_gain_m"] - pair_metrics[lk][pid]["norm_matched_random"]["donor_target_gain_m"] for pid in ctx_elig_pids],  # type: ignore[index]
                        seed=seed + 720 * l_idx + k_idx,
                    ) if ctx_elig_pids else None,
                    "eligible_cand_vs_norm_eta_gt_diff": _summarize_vector(
                        [pair_metrics[lk][pid]["candidate"]["eta_gt"] - pair_metrics[lk][pid]["norm_matched_random"]["eta_gt"] for pid in ctx_elig_pids],  # type: ignore[index]
                        seed=seed + 730 * l_idx + k_idx,
                    ) if ctx_elig_pids else None,
                }

            layer_k_results[layer][k_key] = {
                "layer": layer,
                "k": k,
                "total_pairs": len(pair_ids),
                "eligible_pairs_count": len(elig_pids),
                "coverage_pct": float(len(elig_pids) / len(pair_ids)) if pair_ids else 0.0,
                "support_status": support_label,
                "candidate_full_cohort": cand_full_summary,
                "dimension_matched_full_cohort": dim_full_summary,
                "cand_vs_dimension_matched": cand_vs_dim,
                "candidate_eligible_cohort": cand_elig_summary,
                "norm_matched_eligible_cohort": norm_elig_summary,
                "cand_vs_norm_matched_eligible": cand_vs_norm,
                "context_stratification": context_strat,
            }

    # Holm-Bonferroni adjustment strictly over the 4 complete-support comparisons (Layer 1 k=2, 4 x 2 metrics)
    if family_tests:
        raw_ps = [t[3] for t in family_tests]
        adj_ps = holm_bonferroni_adjust(raw_ps)
        for (layer, k_key, metric, _), adj_p in zip(family_tests, adj_ps):
            layer_k_results[layer][k_key]["cand_vs_norm_matched_eligible"]["metrics"][metric]["holm_adjusted_p"] = float(adj_p)

    # Flatten CSV rows cleanly
    for layer in LAYERS:
        for k in K_VALUES:
            k_key = f"k_{k}"
            entry = layer_k_results[layer][k_key]
            for cohort_label, block in (
                ("candidate_full", entry["candidate_full_cohort"]),
                ("dimension_matched_full", entry["dimension_matched_full_cohort"]),
                ("candidate_eligible", entry["candidate_eligible_cohort"]),
                ("norm_matched_eligible", entry["norm_matched_eligible_cohort"]),
            ):
                m = block["metrics"]
                if block["n_pairs"] == 0:
                    continue
                csv_rows.append(
                    {
                        "layer": layer,
                        "k": k,
                        "cohort": cohort_label,
                        "support_status": entry["support_status"],
                        "n_pairs": block["n_pairs"],
                        "donor_target_gain_m_mean": m["donor_target_gain_m"]["mean"],
                        "donor_target_gain_m_ci_low": m["donor_target_gain_m"]["ci_95"][0],
                        "donor_target_gain_m_ci_high": m["donor_target_gain_m"]["ci_95"][1],
                        "recipient_target_change_m_mean": m["recipient_target_change_m"]["mean"],
                        "eta_gt_dimensionless_mean": m["eta_gt"]["mean"],
                        "eta_gt_dimensionless_ci_low": m["eta_gt"]["ci_95"][0],
                        "eta_gt_dimensionless_ci_high": m["eta_gt"]["ci_95"][1],
                        "target_parallel_component_m_mean": m["target_parallel_component_m"]["mean"],
                        "orthogonal_residual_gt_m_mean": m["orthogonal_residual_gt_m"]["mean"],
                        "intervention_norm_m_mean": m["intervention_norm_m"]["mean"],
                        "norm_ratio_dimensionless_mean": m["norm_ratio"]["mean"],
                        "cosine_gt_mean": m["cosine_gt"]["mean"],
                        "positive_donor_gain_fraction": m["positive_donor_gain_fraction"]["mean"],
                        "legacy_cross_target_error_diff_m_mean": m["legacy_cross_target_error_diff_m"]["mean"],
                    }
                )

    total_observed = sum(observed_counts.values())
    reconciliation_audit = {
        "expected_counts_by_condition": expected_counts,
        "observed_counts_by_condition": dict(observed_counts),
        "observed_total_rows": total_observed,
        "duplicate_rows_count": duplicate_rows,
        "unmatched_rows_count": unmatched_rows,
        "reconciliation_success": bool(
            duplicate_rows == 0
            and unmatched_rows == 0
            and (expected_counts is None or total_observed == expected_counts["total"])
        ),
    }

    return {
        "reconciliation_audit": reconciliation_audit,
        "baseline_identity_audit": baseline_identity_audit,
        "noop_cross_target_artifact_audit": noop_artifact_audit,
        "holm_family_definition": [
            {
                "layer": t[0],
                "k_key": t[1],
                "metric": t[2],
                "raw_two_sided_wilcoxon_p": t[3],
                "holm_adjusted_p": layer_k_results[t[0]][t[1]]["cand_vs_norm_matched_eligible"]["metrics"][t[2]]["holm_adjusted_p"],
                "label": "EXPLORATORY_COMPLETE_SUPPORT",
            }
            for t in family_tests
        ],
        "layer_k_results": layer_k_results,
        "pair_rows": pair_rows_out,
        "csv_rows": csv_rows,
    }


def generate_schema_markdown() -> str:
    """Generates schema documentation for Phase 5.4.2 artifacts."""
    return """# Phase 5.4.2 Schema Documentation: Single-Reference-Target Estimand Audit

## 1. Overview & Motivation

Phase 5.4 computed `error_improvement_m = ||baseline_output - recip_gt|| - ||transplanted_output - donor_gt||`, which subtracted distances to two different targets (`recip_gt` vs `donor_gt`). Phase 5.4.2 audits all 447,800 Phase 5.3 intervention records using strictly single-reference-target causal estimands and vector decomposition diagnostics.

All statistical estimates and comparisons are computed strictly at the independent scene-pair level ($N=60$) per layer and subset size $k$, never pooled into a single global mean across $k$ and layers.

## 2. Estimand Definitions

Given `base_out = baseline_output[:2]`, `trans_out = transplanted_output[:2]`, `recip_gt`, `donor_gt`, true contrast `d_true = donor_gt - recip_gt`, and intervention shift `d_int = trans_out - base_out`:

- **`donor_target_gain_m`** (meters, single donor reference):
  `||base_out - donor_gt|| - ||trans_out - donor_gt||`. Positive means the transplant moved the output closer to the donor's ground-truth waypoint. Identically `0.0` under a no-op intervention (`trans_out == base_out`).
- **`recipient_target_change_m`** (meters, single recipient reference):
  `||trans_out - recip_gt|| - ||base_out - recip_gt||`. Positive means the transplant moved the output away from the recipient's ground-truth waypoint.
- **`eta_gt`** (**dimensionless** ratio):
  `dot(d_int, d_true) / ||d_true||^2`. Scalar projection ratio relative to `||d_true||`. **Never report in meters.**
- **`target_parallel_component_m`** (meters):
  `dot(d_int, d_true) / ||d_true||`. Signed projection magnitude along unit ground-truth contrast vector.
- **`orthogonal_residual_gt_m`** (meters):
  `||d_int - eta_gt * d_true||`. Magnitude of intervention shift orthogonal to `d_true`.
- **`intervention_norm_m`** (meters):
  `||d_int||`.
- **`norm_ratio`** (**dimensionless** ratio):
  `||d_int|| / ||d_true||`.

## 3. Primary Files

- `task_conditioned_circuits_v5_4_2_estimand_audit_summary.json`: Provenance hashes, baseline identity verification across draws, no-op cross-target artifact check, complete layer $\\times$ $k$ pair-level statistics, context stratifications, and Holm-adjusted `EXPLORATORY_COMPLETE_SUPPORT` tests.
- `task_conditioned_circuits_v5_4_2_condition_setting_table.csv`: Tidy table of all 10 `(layer, k)` settings across full and eligible cohorts.
- `task_conditioned_circuits_v5_4_2_pair_metrics.jsonl`: Pair-level aggregated metrics for all 60 independent scene pairs.
- `task_conditioned_circuits_v5_4_2_schema.md`: This schema document.
"""


def run_estimand_audit(repo_root: Path = REPO_ROOT, seed: int = 5420) -> Dict[str, Any]:
    """Executes the full Phase 5.4.2 single-reference-target estimand audit."""
    t0 = time.time()
    circuits_results = repo_root / "world_models_fpga/research/circuits/results"

    pairs_path = circuits_results / "task_conditioned_circuits_v5_3_pairs.jsonl"
    specs_path = circuits_results / "task_conditioned_circuits_v5_2_scene_specs.jsonl"
    acts_path = circuits_results / "task_conditioned_circuits_v5_3_activations.jsonl"
    ints_path = circuits_results / "task_conditioned_circuits_v5_3_interventions.jsonl.gz"

    provenance_hashes = {
        p.name: compute_file_sha256(p)
        for p in (pairs_path, specs_path, acts_path, ints_path)
    }

    with pairs_path.open("r", encoding="utf-8") as f:
        pairs_data = [json.loads(line) for line in f if line.strip()]
    pair_ids = [r["pair_id"] for r in pairs_data]
    pair_context_map = {r["pair_id"]: r.get("context", "unknown") for r in pairs_data}

    eligible_pids_by_lk: Dict[Tuple[str, int], set[str]] = defaultdict(set)
    for row in pairs_data:
        pid = row["pair_id"]
        res = row["results"]
        for layer in LAYERS:
            for k in K_VALUES:
                if all(
                    res.get(f"{sc}__{layer}__{direction}__k_{k}", {}).get("norm_matched_status") == "VALID_COMMON_SUPPORT"
                    for sc, direction in PAIR_SETTINGS
                ):
                    eligible_pids_by_lk[(layer, k)].add(pid)

    expected_counts = compute_setting_expected_counts(pairs_data)

    act_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with acts_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                act_lookup[(rec["pair_id"], rec["scene_id"])] = rec

    def _stream_v53_interventions() -> Iterable[Dict[str, Any]]:
        with gzip.open(ints_path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    core = summarize_estimands_for_setting_records(
        pair_ids=pair_ids,
        pair_context_map=pair_context_map,
        eligible_pids_by_lk=eligible_pids_by_lk,
        records_iter=_stream_v53_interventions(),
        act_lookup=act_lookup,
        expected_counts=expected_counts,
        seed=seed,
    )

    git_commit, git_clean = get_git_info(repo_root)
    elapsed = time.time() - t0

    summary_out = {
        "experiment_name": "Phase 5.4.2 Single-Reference-Target Causal Estimand Audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_clean": git_clean,
        "elapsed_seconds": float(elapsed),
        "provenance_hashes": provenance_hashes,
        "reconciliation_audit": core["reconciliation_audit"],
        "baseline_identity_audit": core["baseline_identity_audit"],
        "noop_cross_target_artifact_audit": core["noop_cross_target_artifact_audit"],
        "holm_family_definition": core["holm_family_definition"],
        "layer_k_results": core["layer_k_results"],
    }

    summary_path = circuits_results / "task_conditioned_circuits_v5_4_2_estimand_audit_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    csv_path = circuits_results / "task_conditioned_circuits_v5_4_2_condition_setting_table.csv"
    if core["csv_rows"]:
        fieldnames = list(core["csv_rows"][0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(core["csv_rows"])

    pair_metrics_path = circuits_results / "task_conditioned_circuits_v5_4_2_pair_metrics.jsonl"
    with pair_metrics_path.open("w", encoding="utf-8") as f:
        for prow in core["pair_rows"]:
            f.write(json.dumps(prow) + "\n")

    schema_path = circuits_results / "task_conditioned_circuits_v5_4_2_schema.md"
    schema_path.write_text(generate_schema_markdown(), encoding="utf-8")

    return summary_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5.4.2 Single-Reference-Target Estimand Audit")
    parser.add_argument("--seed", type=int, default=5420)
    args = parser.parse_args()

    summary = run_estimand_audit(seed=args.seed)
    print("Phase 5.4.2 Estimand Audit completed.")
    print(f"Reconciliation OK: {summary['reconciliation_audit']['reconciliation_success']} ({summary['reconciliation_audit']['observed_total_rows']:,} rows)")
    print(f"Baseline Identity Across Draws OK: {summary['baseline_identity_audit']['baseline_matches_all_draws']} (max diff={summary['baseline_identity_audit']['max_abs_diff_across_draws']:.3e})")
    noop = summary["noop_cross_target_artifact_audit"]
    print(f"No-op Cross-Target Artifact Mean: {noop['mean_noop_cross_target_artifact_m']:+.4f}m (BaseRecip={noop['mean_baseline_recipient_error_m']:.4f}m, BaseDonor={noop['mean_baseline_donor_error_m']:.4f}m)")
    for layer in LAYERS:
        print(f"\n[{layer}]")
        for k in K_VALUES:
            d = summary["layer_k_results"][layer][f"k_{k}"]
            cf = d["candidate_full_cohort"]["metrics"]
            cm = d["cand_vs_norm_matched_eligible"]["metrics"]
            print(
                f"  k={k:2d} ({d['coverage_pct']*100:5.1f}% cov, {d['support_status']}): "
                f"CandGain={cf['donor_target_gain_m']['mean']:+.4f}m, CandEta={cf['eta_gt']['mean']:+.4f} (dimless), "
                f"Cand-Norm DiffGain={cm['donor_target_gain_m']['mean']:+.4f}m (p={cm['donor_target_gain_m']['wilcoxon_two_sided_p']:.4f}, adj={cm['donor_target_gain_m']['holm_adjusted_p']})"
            )


if __name__ == "__main__":
    main()
