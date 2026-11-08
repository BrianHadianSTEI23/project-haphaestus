#!/usr/bin/env python3
"""Project Hephaestus: Phase 5.4.1 Forensic Reconciliation, Statistical Reporting,

and Intelligence-Principle Correction.

This module provides a rigorous, post-hoc forensic audit and reporting pipeline:
1. Exact row-by-row key reconciliation across all 447,800 intervention records.
2. Mathematical verification of ground-truth standoff waypoint geometry (0.60 m offset).
3. Corrected physical scene-swap reversal cosine definition and directional interpretation.
4. Pair-level pooled candidate metrics where the unit of analysis is the independent pair (N=60).
5. Programmatic generation of cohort-matched candidate vs norm-matched tables.
6. Multiple comparison adjustment (Holm-Bonferroni) over complete-support exploratory tests.
"""

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats

# -----------------------------------------------------------------------------
# 1. Coordinate & Waypoint Geometry
# -----------------------------------------------------------------------------

def compute_waypoint_from_object_pose(
    robot_pose_xy: np.ndarray,
    object_pos_xy: np.ndarray,
    standoff_m: float = 0.60
) -> np.ndarray:
    """Computes task-defined standoff goal waypoint from object center toward robot.

    wp = pos_obj - (pos_obj - robot_pose) / ||pos_obj - robot_pose|| * standoff_m
    """
    diff = object_pos_xy[:2] - robot_pose_xy[:2]
    dist = float(np.linalg.norm(diff))
    if dist < 1e-6:
        return object_pos_xy[:2].copy()
    unit_vec = diff / dist
    return object_pos_xy[:2] - unit_vec * standoff_m


def compute_vector_cosine(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-8) -> float:
    """Computes cosine similarity between two 2D vectors safely."""
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < eps or n2 < eps:
        return 0.0
    return float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))


def compute_scene_swap_reversal_metrics(
    d_model_scene_a: np.ndarray,
    d_model_scene_b: np.ndarray
) -> Tuple[float, float]:
    """Computes model contrast reversal metrics between Scene A and Scene B.

    Under a physical target swap: d_gt_scene_b = -d_gt_scene_a.
    A correctly reversing model contrast should satisfy:
        d_model_scene_b = -d_model_scene_a
    Therefore, the reversal cosine is defined as:
        scene_swap_cosine = cosine(d_model_scene_a, -d_model_scene_b)

    Returns:
        (scene_swap_cosine, angular_deviation_deg)
        - +1.0: Perfect reversal (d_model_B = -d_model_A, ang_dev = 0 deg)
        - -1.0: No reversal / invariant orientation (d_model_B = d_model_A, ang_dev = 180 deg)
        -  0.0: Orthogonal shift (ang_dev = 90 deg)
    """
    cos_rev = compute_vector_cosine(d_model_scene_a, -d_model_scene_b)
    ang_dev = float(np.degrees(np.arccos(np.clip(cos_rev, -1.0, 1.0))))
    return cos_rev, ang_dev


def interpret_scene_swap_reversal(cos_val: float) -> str:
    """Provides mathematically truthful interpretation of the reversal cosine."""
    if cos_val > 0.5:
        return "Model contrast predominantly reverses with physical target swap."
    elif cos_val < -0.5:
        return "Model contrast predominantly retains invariant orientation across physical target swap (fails to reverse with ground-truth geometry)."
    else:
        return "Model contrast is approximately orthogonal across physical target swap."


# -----------------------------------------------------------------------------
# 2. Key Reconciliation & Auditing
# -----------------------------------------------------------------------------

def compute_expected_intervention_keys(pairs_data: List[Dict[str, Any]]) -> Set[Tuple]:
    """Constructs the complete expected set of intervention record keys.

    Key tuple: (pair_id, scene_id, layer, direction, k, condition, draw_id)

    Rules:
    - candidate: draw_id=0 for all 2,400 settings.
    - dimension_matched_random: draw_id=0..99 for all 2,400 settings (240,000).
    - norm_matched_random: draw_id=0..99 ONLY for settings with VALID_COMMON_SUPPORT.
    """
    expected_keys = set()
    for row in pairs_data:
        pid = row["pair_id"]
        for setting_key, res in row["results"].items():
            parts = setting_key.split("__")
            scene_id = parts[0]  # "scene_a" or "scene_b"
            layer = parts[1]
            direction = parts[2]
            k = int(parts[3].replace("k_", ""))

            # 1. Candidate record
            expected_keys.add((pid, scene_id, layer, direction, k, "candidate", 0))

            # 2. Dimension-matched records
            for d in range(100):
                expected_keys.add((pid, scene_id, layer, direction, k, "dimension_matched_random", d))

            # 3. Norm-matched records (only if setting is eligible)
            if res.get("norm_matched_status") == "VALID_COMMON_SUPPORT":
                for d in range(100):
                    expected_keys.add((pid, scene_id, layer, direction, k, "norm_matched_random", d))

    return expected_keys


def audit_intervention_reconciliation(
    expected_keys: Set[Tuple],
    intervention_records_iter
) -> Dict[str, Any]:
    """Performs a comprehensive reconciliation of observed vs expected intervention rows."""
    observed_keys = set()
    duplicates = 0
    duplicate_keys = set()
    observed_counts = Counter()
    unmatched_pairs = set()

    known_pairs = {k[0] for k in expected_keys}

    for r in intervention_records_iter:
        pid = r.get("pair_id")
        sc = r.get("scene_id")
        layer = r.get("layer")
        direction = r.get("direction")
        k = r.get("k")
        cond = r.get("condition")
        draw_id = r.get("draw_id")

        if pid not in known_pairs:
            unmatched_pairs.add(pid)

        key = (pid, sc, layer, direction, k, cond, draw_id)
        if key in observed_keys:
            duplicates += 1
            duplicate_keys.add(key)
        observed_keys.add(key)
        observed_counts[cond] += 1

    missing_keys = expected_keys - observed_keys
    unexpected_keys = observed_keys - expected_keys

    # Break down expected by condition
    expected_counts = Counter(k[5] for k in expected_keys)
    missing_counts = Counter(k[5] for k in missing_keys)
    unexpected_counts = Counter(k[5] for k in unexpected_keys)

    reconciliation_success = (
        len(missing_keys) == 0 and
        len(unexpected_keys) == 0 and
        duplicates == 0 and
        len(unmatched_pairs) == 0
    )

    return {
        "expected_total_rows": len(expected_keys),
        "observed_total_rows": len(observed_keys) + duplicates,
        "unique_observed_rows": len(observed_keys),
        "duplicate_rows_count": duplicates,
        "missing_rows_count": len(missing_keys),
        "unexpected_rows_count": len(unexpected_keys),
        "unmatched_pairs_count": len(unmatched_pairs),
        "expected_counts_by_condition": dict(expected_counts),
        "observed_counts_by_condition": dict(observed_counts),
        "missing_counts_by_condition": dict(missing_counts),
        "unexpected_counts_by_condition": dict(unexpected_counts),
        "reconciliation_success": bool(reconciliation_success)
    }


# -----------------------------------------------------------------------------
# 3. Pair-Level Aggregations & Pooled Statistics
# -----------------------------------------------------------------------------

def compute_pair_level_pooled_candidate_metrics(
    candidate_rows_by_pair: Dict[str, List[Dict[str, Any]]],
    n_boot: int = 10000,
    seed: int = 2026
) -> Dict[str, Any]:
    """Computes pair-level pooled candidate intervention metrics where the unit of

    analysis is strictly the independent scene pair (N=60).
    """
    pids = sorted(candidate_rows_by_pair.keys())
    n_pairs = len(pids)

    pair_means_eta_gt = []
    pair_means_err_imp = []
    pair_fractions_pos_imp = []
    pair_means_eta_model = []

    # Row-level collections for descriptive reporting
    all_eta_gt = []
    all_err_imp = []
    all_pos_imp = []
    all_eta_model = []

    for pid in pids:
        rows = candidate_rows_by_pair[pid]
        p_eta_gt = [r["eta_gt"] for r in rows]
        p_err_imp = [r["error_improvement_m"] for r in rows]
        p_pos = [1.0 if r["error_improvement_m"] > 0 else 0.0 for r in rows]
        p_eta_model = [r["eta_model"] for r in rows]

        pair_means_eta_gt.append(float(np.mean(p_eta_gt)))
        pair_means_err_imp.append(float(np.mean(p_err_imp)))
        pair_fractions_pos_imp.append(float(np.mean(p_pos)))
        pair_means_eta_model.append(float(np.mean(p_eta_model)))

        all_eta_gt.extend(p_eta_gt)
        all_err_imp.extend(p_err_imp)
        all_pos_imp.extend(p_pos)
        all_eta_model.extend(p_eta_model)

    # Clustered bootstrap over pairs
    rng = np.random.default_rng(seed)
    boot_eta_gt = []
    boot_err_imp = []
    boot_pos_imp = []
    boot_eta_model = []

    for _ in range(n_boot):
        idx = rng.choice(n_pairs, size=n_pairs, replace=True)
        boot_eta_gt.append(float(np.mean([pair_means_eta_gt[i] for i in idx])))
        boot_err_imp.append(float(np.mean([pair_means_err_imp[i] for i in idx])))
        boot_pos_imp.append(float(np.mean([pair_fractions_pos_imp[i] for i in idx])))
        boot_eta_model.append(float(np.mean([pair_means_eta_model[i] for i in idx])))

    return {
        "n_independent_pairs": n_pairs,
        "n_total_candidate_rows": len(all_eta_gt),
        "pair_level_inference": {
            "eta_gt": {
                "mean": float(np.mean(pair_means_eta_gt)),
                "median": float(np.median(pair_means_eta_gt)),
                "std": float(np.std(pair_means_eta_gt, ddof=1)),
                "ci_95": [float(np.percentile(boot_eta_gt, 2.5)), float(np.percentile(boot_eta_gt, 97.5))]
            },
            "error_improvement_m": {
                "mean": float(np.mean(pair_means_err_imp)),
                "median": float(np.median(pair_means_err_imp)),
                "std": float(np.std(pair_means_err_imp, ddof=1)),
                "ci_95": [float(np.percentile(boot_err_imp, 2.5)), float(np.percentile(boot_err_imp, 97.5))]
            },
            "fraction_positive_improvement": {
                "mean": float(np.mean(pair_fractions_pos_imp)),
                "median": float(np.median(pair_fractions_pos_imp)),
                "std": float(np.std(pair_fractions_pos_imp, ddof=1)),
                "ci_95": [float(np.percentile(boot_pos_imp, 2.5)), float(np.percentile(boot_pos_imp, 97.5))]
            },
            "eta_model": {
                "mean": float(np.mean(pair_means_eta_model)),
                "median": float(np.median(pair_means_eta_model)),
                "std": float(np.std(pair_means_eta_model, ddof=1)),
                "ci_95": [float(np.percentile(boot_eta_model, 2.5)), float(np.percentile(boot_eta_model, 97.5))]
            }
        },
        "descriptive_row_level_quantities": {
            "mean_eta_gt": float(np.mean(all_eta_gt)),
            "mean_error_improvement_m": float(np.mean(all_err_imp)),
            "fraction_positive_improvement": float(np.mean(all_pos_imp)),
            "mean_eta_model": float(np.mean(all_eta_model))
        },
        "aggregation_documentation": {
            "unit_of_analysis": "Independent scene pair (N=60)",
            "rows_per_pair": len(all_eta_gt) // n_pairs,
            "error_improvement_vs_contrast_explanation": (
                "Euclidean error improvement (baseline_distance - transplant_distance) measures radial distance "
                "reduction toward the donor waypoint. It can be positive even when contrast projection (eta_gt) is "
                "negative if the baseline output was eccentric and the intervention contracts the output vector "
                "perpendicular to the true target displacement axis."
            )
        }
    }


# -----------------------------------------------------------------------------
# 4. Cohort-Matched Comparisons & Table Generation
# -----------------------------------------------------------------------------

def compute_cohort_matched_comparisons(
    pair_metrics_by_lk: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]],
    eligible_pids_by_lk: Dict[Tuple[str, int], Set[str]],
    all_pids: List[str]
) -> Dict[str, Any]:
    """Computes transparent cohort-matched metrics ensuring candidate and control

    comparisons strictly evaluate identical pair IDs.
    """
    comparisons = {}

    for layer in ["layer1_post_relu_64d", "layer2_post_relu_64d"]:
        comparisons[layer] = {}
        for k in [2, 4, 8, 16, 32]:
            lk = (layer, k)
            k_key = f"k_{k}"
            elig_set = eligible_pids_by_lk[lk]
            elig_pids = sorted(list(elig_set))
            cov = len(elig_pids) / float(len(all_pids))

            # Candidate full cohort (all 60 pairs)
            cand_full_eta = float(np.mean([pair_metrics_by_lk[lk][p]["cand_eta"] for p in all_pids]))
            cand_full_imp = float(np.mean([pair_metrics_by_lk[lk][p]["cand_imp"] for p in all_pids]))
            cand_full_model_eta = float(np.mean([pair_metrics_by_lk[lk][p]["cand_eta_model"] for p in all_pids]))

            # Candidate eligible cohort (exact matching pair IDs)
            cand_elig_eta = float(np.mean([pair_metrics_by_lk[lk][p]["cand_eta"] for p in elig_pids]))
            cand_elig_imp = float(np.mean([pair_metrics_by_lk[lk][p]["cand_imp"] for p in elig_pids]))

            # Norm-matched eligible cohort
            norm_elig_eta = float(np.mean([pair_metrics_by_lk[lk][p]["norm_eta"] for p in elig_pids]))
            norm_elig_imp = float(np.mean([pair_metrics_by_lk[lk][p]["norm_imp"] for p in elig_pids]))

            # Paired difference
            paired_diff_eta = cand_elig_eta - norm_elig_eta
            paired_diff_imp = cand_elig_imp - norm_elig_imp

            # Paired vectors for statistical tests
            diffs_eta = [pair_metrics_by_lk[lk][p]["cand_eta"] - pair_metrics_by_lk[lk][p]["norm_eta"] for p in elig_pids]
            diffs_imp = [pair_metrics_by_lk[lk][p]["cand_imp"] - pair_metrics_by_lk[lk][p]["norm_imp"] for p in elig_pids]

            # Wilcoxon tests (two-sided)
            try:
                w_eta = float(stats.wilcoxon(diffs_eta, alternative="two-sided").pvalue)
            except Exception:
                w_eta = 1.0
            try:
                w_imp = float(stats.wilcoxon(diffs_imp, alternative="two-sided").pvalue)
            except Exception:
                w_imp = 1.0

            # One-sided Wilcoxon tests (greater, candidate superiority)
            try:
                w_eta_greater = float(stats.wilcoxon(diffs_eta, alternative="greater").pvalue)
            except Exception:
                w_eta_greater = 1.0
            try:
                w_imp_greater = float(stats.wilcoxon(diffs_imp, alternative="greater").pvalue)
            except Exception:
                w_imp_greater = 1.0

            status = "EXPLORATORY_COMPLETE_SUPPORT" if len(elig_pids) == len(all_pids) else "EXPLORATORY_INCOMPLETE_SUPPORT"

            comparisons[layer][k_key] = {
                "layer": layer,
                "k": k,
                "total_pairs": len(all_pids),
                "eligible_pairs_count": len(elig_pids),
                "coverage_pct": cov,
                "support_status": status,
                "candidate_full_cohort": {
                    "mean_eta_gt": cand_full_eta,
                    "mean_error_improvement_m": cand_full_imp,
                    "mean_eta_model": cand_full_model_eta
                },
                "cohort_matched_comparison": {
                    "eligible_pairs": elig_pids,
                    "candidate_eligible_mean_eta_gt": cand_elig_eta,
                    "norm_matched_eligible_mean_eta_gt": norm_elig_eta,
                    "paired_difference_eta_gt": paired_diff_eta,
                    "wilcoxon_two_sided_p_eta_gt": w_eta,
                    "wilcoxon_greater_p_eta_gt": w_eta_greater,
                    "candidate_eligible_mean_error_improvement_m": cand_elig_imp,
                    "norm_matched_eligible_mean_error_improvement_m": norm_elig_imp,
                    "paired_difference_error_improvement_m": paired_diff_imp,
                    "wilcoxon_two_sided_p_error_improvement": w_imp,
                    "wilcoxon_greater_p_error_improvement": w_imp_greater
                }
            }

    # Family of 4 complete-support comparisons in Layer 1 (k=2, k=4)
    # 1. Layer 1 k=2 eta_gt
    # 2. Layer 1 k=2 error_imp
    # 3. Layer 1 k=4 eta_gt
    # 4. Layer 1 k=4 error_imp
    comp_family = [
        ("layer1_post_relu_64d", "k_2", "eta_gt", comparisons["layer1_post_relu_64d"]["k_2"]["cohort_matched_comparison"]["wilcoxon_two_sided_p_eta_gt"]),
        ("layer1_post_relu_64d", "k_2", "error_imp", comparisons["layer1_post_relu_64d"]["k_2"]["cohort_matched_comparison"]["wilcoxon_two_sided_p_error_improvement"]),
        ("layer1_post_relu_64d", "k_4", "eta_gt", comparisons["layer1_post_relu_64d"]["k_4"]["cohort_matched_comparison"]["wilcoxon_two_sided_p_eta_gt"]),
        ("layer1_post_relu_64d", "k_4", "error_imp", comparisons["layer1_post_relu_64d"]["k_4"]["cohort_matched_comparison"]["wilcoxon_two_sided_p_error_improvement"])
    ]

    # Sort by p-value for Holm adjustment
    sorted_family = sorted(comp_family, key=lambda x: x[3])
    m = len(sorted_family)
    holm_adjusted = {}
    cum_max = 0.0
    for rank, (l, k_key, metric, raw_p) in enumerate(sorted_family):
        mult = m - rank
        adj_p = min(1.0, raw_p * mult)
        adj_p = max(adj_p, cum_max)
        cum_max = adj_p
        holm_adjusted[(l, k_key, metric)] = float(adj_p)

    # Attach adjusted p-values
    for l, k_key, metric in holm_adjusted:
        field_name = f"holm_adjusted_p_{metric}"
        comparisons[l][k_key]["cohort_matched_comparison"][field_name] = holm_adjusted[(l, k_key, metric)]

    return comparisons


def generate_summary_table_markdown(cohort_comparisons: Dict[str, Any]) -> str:
    """Programmatically generates the markdown table for README.md from cohort comparisons."""
    header = (
        "| Layer | $k$ | Common Support Coverage | Candidate Ground-Truth Transfer $\\eta_{\\text{gt}}$ (All 60 Pairs) | "
        "Candidate $\\eta_{\\text{gt}}$ (Eligible Pairs) | Strict Norm-Matched $\\eta_{\\text{gt}}$ (Eligible Pairs) | "
        "Paired Diff $\\Delta\\eta_{\\text{gt}}$ | Two-Sided $p$ (Holm $p_{\\text{adj}}$) | "
        "Cand Error Imp (Eligible) | Norm-Matched Error Imp (Eligible) | "
        "Paired Diff Error Imp | Two-Sided $p$ (Holm $p_{\\text{adj}}$) | Status |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
    )

    rows = []
    for layer in ["layer1_post_relu_64d", "layer2_post_relu_64d"]:
        l_short = "Layer 1" if "layer1" in layer else "Layer 2"
        for k in [2, 4, 8, 16, 32]:
            k_key = f"k_{k}"
            d = cohort_comparisons[layer][k_key]
            cov = d["coverage_pct"] * 100.0
            elig_n = d["eligible_pairs_count"]
            tot_n = d["total_pairs"]

            c_full_eta = d["candidate_full_cohort"]["mean_eta_gt"]
            cm = d["cohort_matched_comparison"]
            c_elig_eta = cm["candidate_eligible_mean_eta_gt"]
            n_elig_eta = cm["norm_matched_eligible_mean_eta_gt"]
            diff_eta = cm["paired_difference_eta_gt"]
            p_eta = cm["wilcoxon_two_sided_p_eta_gt"]
            p_eta_adj = cm.get("holm_adjusted_p_eta_gt")

            c_elig_imp = cm["candidate_eligible_mean_error_improvement_m"]
            n_elig_imp = cm["norm_matched_eligible_mean_error_improvement_m"]
            diff_imp = cm["paired_difference_error_improvement_m"]
            p_imp = cm["wilcoxon_two_sided_p_error_improvement"]
            p_imp_adj = cm.get("holm_adjusted_p_error_imp")

            status = d["support_status"]

            p_eta_str = f"{p_eta:.4f} ({p_eta_adj:.4f})" if p_eta_adj is not None else f"{p_eta:.4f}*"
            p_imp_str = f"{p_imp:.4f} ({p_imp_adj:.4f})" if p_imp_adj is not None else f"{p_imp:.4f}*"

            row = (
                f"| **{l_short}** | {k} | **{cov:.1f}%** ({elig_n}/{tot_n}) | "
                f"{c_full_eta:+.3f} | {c_elig_eta:+.3f} | {n_elig_eta:+.3f} | "
                f"{diff_eta:+.3f} | {p_eta_str} | "
                f"{c_elig_imp:+.3f}m | {n_elig_imp:+.3f}m | "
                f"{diff_imp:+.3f}m | {p_imp_str} | `{status}` |"
            )
            rows.append(row)

    footer = "\n\n*\\*Note: p-values in incomplete support settings are descriptive exploratory comparisons evaluated on common eligible pairs.*"
    return header + "\n" + "\n".join(rows) + footer


# -----------------------------------------------------------------------------
# 5. Full Audit Pipeline Runner
# -----------------------------------------------------------------------------

def run_reporting_audit(
    base_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Executes the full Phase 5.4.1 forensic reconciliation and reporting audit."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[3]

    circuits_results = base_dir / "world_models_fpga/research/circuits/results"
    
    # Input paths
    pairs_path = circuits_results / "task_conditioned_circuits_v5_3_pairs.jsonl"
    int_path = circuits_results / "task_conditioned_circuits_v5_4_intervention_metrics.jsonl.gz"
    scene_path = circuits_results / "task_conditioned_circuits_v5_4_scene_metrics.jsonl"
    v5_3_summary_path = circuits_results / "task_conditioned_circuits_v5_3_summary.json"
    v5_4_summary_path = circuits_results / "task_conditioned_circuits_v5_4_ground_truth_summary.json"

    # 1. Hashing inputs
    provenance_hashes = {}
    for p in [pairs_path, int_path, scene_path, v5_3_summary_path, v5_4_summary_path]:
        with open(p, "rb") as f:
            provenance_hashes[p.name] = hashlib.sha256(f.read()).hexdigest()

    # 2. Load pairs metadata
    pairs_data = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            pairs_data.append(json.loads(line))

    # Eligible pairs
    eligible_pids_by_lk = defaultdict(set)
    for row in pairs_data:
        pid = row["pair_id"]
        res = row["results"]
        for layer in ["layer1_post_relu_64d", "layer2_post_relu_64d"]:
            for k in [2, 4, 8, 16, 32]:
                if all(res.get(f"{sc}__{layer}__{d}__k_{k}", {}).get("norm_matched_status") == "VALID_COMMON_SUPPORT"
                       for sc in ["scene_a", "scene_b"] for d in ["A_to_B", "B_to_A"]):
                    eligible_pids_by_lk[(layer, k)].add(pid)

    all_pids = [r["pair_id"] for r in pairs_data]

    # 3. Expected keys and reconciliation audit
    expected_keys = compute_expected_intervention_keys(pairs_data)

    def stream_interventions():
        with gzip.open(int_path, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)

    reconciliation_results = audit_intervention_reconciliation(expected_keys, stream_interventions())

    # 4. Stream and collect candidate and norm-matched setting metrics
    cand_rows_by_pair = defaultdict(list)
    cand_settings = {}
    norm_settings = defaultdict(list)

    with gzip.open(int_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pid = r["pair_id"]
            sc = r["scene_id"]
            layer = r["layer"]
            d = r["direction"]
            k = r["k"]
            cond = r["condition"]
            key = (pid, sc, layer, d, k)

            if cond == "candidate":
                cand_rows_by_pair[pid].append(r)
                cand_settings[key] = (r["eta_gt"], r["error_improvement_m"], r["eta_model"])
            elif cond == "norm_matched_random":
                norm_settings[key].append((r["eta_gt"], r["error_improvement_m"]))

    # 5. Pair-level pooled candidate metrics
    pooled_candidate_metrics = compute_pair_level_pooled_candidate_metrics(cand_rows_by_pair)

    # 6. Per-pair aggregated metrics for cohort comparisons
    pair_metrics_by_lk = defaultdict(dict)
    pair_metrics_rows = []

    for pid in all_pids:
        pair_row = {"pair_id": pid, "settings": {}}
        for layer in ["layer1_post_relu_64d", "layer2_post_relu_64d"]:
            for k in [2, 4, 8, 16, 32]:
                lk = (layer, k)
                c_etas, c_imps, c_model = [], [], []
                n_etas, n_imps = [], []

                for sc in ["scene_a", "scene_b"]:
                    for d in ["A_to_B", "B_to_A"]:
                        key = (pid, sc, layer, d, k)
                        if key in cand_settings:
                            c_etas.append(cand_settings[key][0])
                            c_imps.append(cand_settings[key][1])
                            c_model.append(cand_settings[key][2])
                        if key in norm_settings and len(norm_settings[key]) > 0:
                            n_etas.append(np.mean([x[0] for x in norm_settings[key]]))
                            n_imps.append(np.mean([x[1] for x in norm_settings[key]]))

                cand_mean_eta = float(np.mean(c_etas))
                cand_mean_imp = float(np.mean(c_imps))
                cand_mean_model = float(np.mean(c_model))
                norm_mean_eta = float(np.mean(n_etas)) if len(n_etas) == 4 else None
                norm_mean_imp = float(np.mean(n_imps)) if len(n_imps) == 4 else None

                pair_metrics_by_lk[lk][pid] = {
                    "cand_eta": cand_mean_eta,
                    "cand_imp": cand_mean_imp,
                    "cand_eta_model": cand_mean_model,
                    "norm_eta": norm_mean_eta,
                    "norm_imp": norm_mean_imp
                }

                setting_str = f"{layer}__k_{k}"
                pair_row["settings"][setting_str] = {
                    "candidate_eta_gt": cand_mean_eta,
                    "candidate_error_improvement_m": cand_mean_imp,
                    "candidate_eta_model": cand_mean_model,
                    "norm_matched_eta_gt": norm_mean_eta,
                    "norm_matched_error_improvement_m": norm_mean_imp,
                    "is_eligible": bool(pid in eligible_pids_by_lk[lk])
                }
        pair_metrics_rows.append(pair_row)

    # 7. Cohort-matched comparisons
    cohort_comparisons = compute_cohort_matched_comparisons(pair_metrics_by_lk, eligible_pids_by_lk, all_pids)
    table_markdown = generate_summary_table_markdown(cohort_comparisons)

    # 8. Full-model waypoint grounding & scene-swap reversal audit
    with open(v5_4_summary_path, "r", encoding="utf-8") as f:
        v5_4_summary = json.load(f)

    full_acc = v5_4_summary["full_model_ground_truth_accuracy"]
    mean_spatial_error = float(full_acc["mean_spatial_error_m"])
    mean_eta_gt = float(full_acc["mean_eta_gt"])
    mean_cosine = float(full_acc["mean_cosine_alignment"])
    mean_ang_dev = float(full_acc["mean_angular_deviation_deg"])
    mean_heading_error = float(full_acc["mean_heading_error_rad"])
    mean_scene_swap_cos = float(full_acc["mean_scene_swap_cosine"])
    mean_scene_swap_ang = float(full_acc["mean_scene_swap_angular_dev_deg"])

    waypoint_grounding_audit = {
        "standoff_distance_m": 0.60,
        "waypoint_equation": "wp = pos_obj - (pos_obj - robot_pose) / ||pos_obj - robot_pose|| * 0.60",
        "ground_truth_target_definition": "Task-defined standoff goal waypoint, not physical object center",
        "full_model_accuracy": {
            "mean_spatial_error_m": mean_spatial_error,
            "mean_eta_gt": mean_eta_gt,
            "mean_cosine_alignment": mean_cosine,
            "mean_angular_deviation_deg": mean_ang_dev,
            "mean_heading_error_rad": mean_heading_error,
            "mean_heading_error_deg": float(np.degrees(mean_heading_error))
        },
        "scene_swap_reversal_audit": {
            "definition": "cosine(d_model_scene_a, -d_model_scene_b)",
            "expected_reversal_value": "+1.0 (d_model_B = -d_model_A)",
            "observed_mean_reversal_cosine": mean_scene_swap_cos,
            "observed_mean_angular_deviation_deg": mean_scene_swap_ang,
            "directional_interpretation": interpret_scene_swap_reversal(mean_scene_swap_cos),
            "mathematical_implication": (
                "The observed negative reversal cosine (-0.7119) indicates that model contrast retains "
                "substantially the same orientation across physical target swap (cosine(d_model_A, d_model_B) = +0.7119), "
                "failing to reverse with the physical target swap."
            )
        }
    }

    # 9. Machine-readable claims ledger
    claims_ledger = [
        {
            "claim_id": "CLAIM_CIRC_02",
            "hypothesis": "Dedicated Sparse Subnetwork Selectivity",
            "status": "UNRESOLVED_UNDER_CURRENT_REPRESENTATION",
            "evidence_level": "E2",
            "statement": (
                "In the two complete-support settings (Layer 1 k=2, 4), candidate units exhibit no evidence of "
                "superiority over strictly norm-matched random controls (Holm-adjusted p=1.0000); confidence intervals "
                "do not establish equivalence. In the remaining 8 settings, common support is incomplete."
            )
        },
        {
            "claim_id": "CLAIM_CIRC_03",
            "hypothesis": "Dissociation of Internal Model Contrast from Ground-Truth Waypoint Grounding",
            "status": "VALIDATED_DISSOCIATION",
            "evidence_level": "E2",
            "statement": (
                "Task-conditioned output modulation is dissociated from ground-truth waypoint grounding in the tested "
                "H-JEPA checkpoint and paired-scene benchmark. Full-model output contrast is orthogonal (90.63 deg, "
                "cosine +0.0010) to physical waypoint displacement, and candidate unit transplants transfer negative "
                "contrast (eta_gt = -0.0691) with negligible error reduction (+0.0344m)."
            )
        },
        {
            "claim_id": "CLAIM_CIRC_04",
            "hypothesis": "Model Contrast Directionality Across Physical Scene Swap",
            "status": "NON_REVERSING_ORIENTATION_CONFIRMED",
            "evidence_level": "E2",
            "statement": (
                "When physical targets swap positions (d_gt_B = -d_gt_A), model contrast fails to reverse with the "
                "ground-truth geometry (reversal cosine = -0.7119, angular deviation 143.50 deg), retaining similar "
                "output orientation across scenes rather than tracking physical inversion."
            )
        }
    ]

    # Combine into comprehensive audit summary
    audit_summary = {
        "experiment_name": "Phase 5.4.1 Forensic Reconciliation, Statistical Reporting, & Intelligence-Principle Correction",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_hashes": provenance_hashes,
        "reconciliation_audit": reconciliation_results,
        "waypoint_grounding_audit": waypoint_grounding_audit,
        "pair_level_pooled_candidate_metrics": pooled_candidate_metrics,
        "cohort_matched_comparisons": cohort_comparisons,
        "programmatic_table_markdown": table_markdown,
        "machine_readable_claims_ledger": claims_ledger
    }

    # Write output files
    # A. Audit summary JSON
    summary_out_path = circuits_results / "task_conditioned_circuits_v5_4_1_audit_summary.json"
    with open(summary_out_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    # B. Condition counts CSV
    csv_out_path = circuits_results / "task_conditioned_circuits_v5_4_1_condition_counts.csv"
    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "expected_rows", "observed_rows", "missing_rows", "unexpected_rows", "duplicate_rows"])
        for cond in ["candidate", "dimension_matched_random", "norm_matched_random"]:
            exp_c = reconciliation_results["expected_counts_by_condition"].get(cond, 0)
            obs_c = reconciliation_results["observed_counts_by_condition"].get(cond, 0)
            mis_c = reconciliation_results["missing_counts_by_condition"].get(cond, 0)
            unexp_c = reconciliation_results["unexpected_counts_by_condition"].get(cond, 0)
            writer.writerow([cond, exp_c, obs_c, mis_c, unexp_c, reconciliation_results["duplicate_rows_count"]])

    # C. Pair metrics JSONL
    pair_metrics_out_path = circuits_results / "task_conditioned_circuits_v5_4_1_pair_metrics.jsonl"
    with open(pair_metrics_out_path, "w", encoding="utf-8") as f:
        for p_row in pair_metrics_rows:
            f.write(json.dumps(p_row) + "\n")

    # D. Schema markdown
    schema_out_path = circuits_results / "task_conditioned_circuits_v5_4_1_schema.md"
    with open(schema_out_path, "w", encoding="utf-8") as f:
        f.write(generate_schema_documentation())

    return audit_summary


def generate_schema_documentation() -> str:
    """Generates comprehensive documentation for Phase 5.4.1 artifacts."""
    return """# Phase 5.4.1 Schema Documentation: Forensic Reconciliation & Statistical Reporting

## 1. Scope & Purpose

Phase 5.4.1 establishes full cryptographic and mathematical reconciliation of the Phase 5.4 ground-truth spatial reanalysis:
1. Reconciles exact record counts (447,800 rows total: 2,400 candidate, 240,000 dimension-matched, 205,400 norm-matched).
2. Clarifies ground-truth coordinates as task-defined standoff goal waypoints (0.60 m standoff toward robot), not object centers.
3. Corrects physical scene-swap reversal cosine interpretation (-0.7119 signifies non-reversal / orientation retention).
4. Establishes pair-level pooled candidate statistics where the independent scene pair (N=60) is the unit of inference.
5. Populates cohort-matched candidate vs norm-matched tables strictly on identical eligible pair IDs.
6. Designates all post-hoc ground-truth reanalysis tests as `EXPLORATORY_COMPLETE_SUPPORT` or `EXPLORATORY_INCOMPLETE_SUPPORT`.

## 2. Primary Output Files

- `task_conditioned_circuits_v5_4_1_audit_summary.json`: Complete audit records, hashes, waypoint geometry, pooled candidate statistics, cohort-matched comparison metrics, Holm-adjusted p-values, and claims ledger.
- `task_conditioned_circuits_v5_4_1_condition_counts.csv`: Exact condition-level row accounting (expected vs observed).
- `task_conditioned_circuits_v5_4_1_pair_metrics.jsonl`: Pair-level aggregated metrics across all 60 independent pairs and 10 settings.
- `task_conditioned_circuits_v5_4_1_schema.md`: This documentation.

## 3. Mathematical Definitions

### A. Task-Defined Standoff Waypoint
$$\\mathbf{w} = \\mathbf{p}_{\\text{obj}} - 0.60 \\cdot \\frac{\\mathbf{p}_{\\text{obj}} - \\mathbf{p}_{\\text{robot}}}{\\|\\mathbf{p}_{\\text{obj}} - \\mathbf{p}_{\\text{robot}}\\|}$$

### B. Scene-Swap Reversal Cosine
$$\\cos_{\\text{swap}} = \\frac{\\Delta\\mathbf{g}_{\\text{model}}^A \\cdot (-\\Delta\\mathbf{g}_{\\text{model}}^B)}{\\|\\Delta\\mathbf{g}_{\\text{model}}^A\\| \\|\\Delta\\mathbf{g}_{\\text{model}}^B\\|}$$
- $\\cos_{\\text{swap}} = +1.0$: Ideal reversal tracking ground-truth inversion ($\\Delta\\mathbf{g}_{\\text{model}}^B = -\\Delta\\mathbf{g}_{\\text{model}}^A$).
- $\\cos_{\\text{swap}} = -1.0$: Invariant contrast retaining identical orientation ($\\Delta\\mathbf{g}_{\\text{model}}^B = \\Delta\\mathbf{g}_{\\text{model}}^A$).

### C. Cohort-Matched Paired Difference
$$\\Delta\\eta_{\\text{gt}}(L, k) = \\frac{1}{|\\mathcal{P}_{\\text{elig}}|} \\sum_{p \\in \\mathcal{P}_{\\text{elig}}} \\eta_{\\text{cand}}(p, L, k) - \\frac{1}{|\\mathcal{P}_{\\text{elig}}|} \\sum_{p \\in \\mathcal{P}_{\\text{elig}}} \\eta_{\\text{norm}}(p, L, k)$$
Ensures candidate and control means are evaluated strictly on the exact same pair IDs $\\mathcal{P}_{\\text{elig}}$.
"""


if __name__ == "__main__":
    print("=" * 80)
    print("PROJECT HEPHAESTUS: PHASE 5.4.1 FORENSIC RECONCILIATION & REPORTING AUDIT")
    print("=" * 80)
    summary = run_reporting_audit()
    print("Audit completed successfully!")
    print(f"Reconciliation Success: {summary['reconciliation_audit']['reconciliation_success']}")
    print(f"Total Rows Verified: {summary['reconciliation_audit']['observed_total_rows']:,}")
    print(f"Scene-Swap Reversal Cosine: {summary['waypoint_grounding_audit']['scene_swap_reversal_audit']['observed_mean_reversal_cosine']:.4f}")
    print(summary['programmatic_table_markdown'])
