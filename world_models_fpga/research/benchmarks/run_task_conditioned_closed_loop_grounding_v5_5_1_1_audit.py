"""
Project Hephaestus — Phase 5.5.1.1: H1 Estimand Correction, Side/Category Bias Audit,
and Scene-Generator Visual Geometry Audit.

Strictly reuses historical Phase 5.5.1 raw outputs without modifying any historical artifact
and without re-running the 840 closed-loop simulations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, chi2, mannwhitneyu, wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_models_fpga.env.amr_env import JetAutoEnv
from world_models_fpga.models.baseline_policy import TARGET_KEYS, encode_task_id
from world_models_fpga.research.benchmarks.model_wrappers import compute_body_frame_goal_vec
from world_models_fpga.research.benchmarks.run_task_conditioned_closed_loop_grounding_v5_5 import (
    extract_geom_mask_and_stats,
    render_segmentation_array,
)
from world_models_fpga.research.datasets.generation.capability_tasks import CANONICAL_OBSTACLES

RESULTS_DIR = REPO_ROOT / "world_models_fpga" / "research" / "benchmarks" / "results"
DEFAULT_SCENES_PATH = RESULTS_DIR / "task_conditioned_closed_loop_grounding_v5_5_1_scenes.jsonl"
DEFAULT_EPISODES_PATH = RESULTS_DIR / "task_conditioned_closed_loop_grounding_v5_5_1_episodes.jsonl"
DEFAULT_PAIRS_PATH = RESULTS_DIR / "task_conditioned_closed_loop_grounding_v5_5_1_scene_pairs.jsonl"
DEFAULT_SUMMARY_PATH = RESULTS_DIR / "task_conditioned_closed_loop_grounding_v5_5_1_summary.json"
DEFAULT_SHARD_MANIFEST_PATH = (
    RESULTS_DIR / "task_conditioned_closed_loop_v5_5_1_trace_export" / "shard_manifest.json"
)

OUT_H1_SUMMARY = RESULTS_DIR / "task_conditioned_closed_loop_grounding_v5_5_1_1_H1_reanalysis_summary.json"
OUT_SCENE_PAIR_CSV = RESULTS_DIR / "task_conditioned_closed_loop_grounding_v5_5_1_1_scene_pair_matrix.csv"
OUT_CAT_SIDE_CSV = RESULTS_DIR / "task_conditioned_closed_loop_grounding_v5_5_1_1_category_side_table.csv"
OUT_EPISODE_EFFECTS_CSV = RESULTS_DIR / "task_conditioned_closed_loop_grounding_v5_5_1_1_episode_effects.csv"
OUT_GEOMETRY_JSON = RESULTS_DIR / "task_conditioned_closed_loop_grounding_v5_5_1_1_geometry_audit.json"
OUT_SCHEMA_MD = RESULTS_DIR / "task_conditioned_closed_loop_grounding_v5_5_1_1_schema.md"

DEFAULT_BOOTSTRAP_RESAMPLES = 4000
DEFAULT_SEED = 5511


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return (max(0.0, float(center - half)), min(1.0, float(center + half)))


def exact_mcnemar_pvalue(n10: int, n01: int) -> float:
    total = int(n10) + int(n01)
    if total == 0:
        return 1.0
    return float(binomtest(int(n10), total, 0.5, alternative="two-sided").pvalue)


def paired_bootstrap_ci(
    values: Sequence[float],
    n_boot: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def paired_wilcoxon_stats(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    pos = int(np.sum(arr > 1e-12))
    neg = int(np.sum(arr < -1e-12))
    tie = int(len(arr) - pos - neg)
    if pos + neg == 0:
        return {"n": int(len(arr)), "pos": pos, "neg": neg, "tie": tie, "stat": 0.0, "p": 1.0}
    res = wilcoxon(arr, zero_method="wilcox", alternative="two-sided")
    return {
        "n": int(len(arr)),
        "pos": pos,
        "neg": neg,
        "tie": tie,
        "stat": float(res.statistic),
        "p": float(res.pvalue),
    }


def stuart_maxwell_3x3(mat: np.ndarray) -> Dict[str, Any]:
    """Stuart-Maxwell marginal homogeneity test on a 3x3 paired contingency matrix."""
    m = np.asarray(mat, dtype=np.float64)
    assert m.shape == (3, 3)
    row_s = m.sum(axis=1)
    col_s = m.sum(axis=0)
    d = row_s[:-1] - col_s[:-1]
    V = np.zeros((2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            if i == j:
                V[i, j] = row_s[i] + col_s[i] - 2.0 * m[i, i]
            else:
                V[i, j] = -(m[i, j] + m[j, i])
    det = float(np.linalg.det(V))
    if abs(det) < 1e-12:
        return {"chi2": 0.0, "df": 2, "p": 1.0, "degenerate": True}
    stat = float(d.T @ np.linalg.inv(V) @ d)
    p = float(chi2.sf(stat, df=2))
    return {"chi2": stat, "df": 2, "p": p, "degenerate": False}


def mantel_haenszel_stratified(
    strata_tables: Sequence[Tuple[int, int, int, int]],
) -> Dict[str, Any]:
    """
    Computes Cochran-Mantel-Haenszel test and common odds ratio plus Mantel-Haenszel
    precision-weighted risk difference across binary 2x2 strata (n11, n10, n01, n00):
    row1 = group 1 (e.g., right), row2 = group 2 (e.g., left);
    col1 = success, col2 = failure.
    Let a = n11 (right_succ), b = n10 (right_fail), c = n01 (left_succ), d = n00 (left_fail).
    """
    num_or = 0.0
    den_or = 0.0
    cmh_num = 0.0
    cmh_var = 0.0
    rd_num = 0.0
    rd_den = 0.0
    valid_strata = 0
    for a, b, c, d in strata_tables:
        n1 = a + b
        n2 = c + d
        m1 = a + c
        m2 = b + d
        n = n1 + n2
        if n <= 1 or n1 == 0 or n2 == 0:
            continue
        valid_strata += 1
        num_or += (a * d) / n
        den_or += (b * c) / n
        cmh_num += a - (n1 * m1) / n
        cmh_var += (n1 * n2 * m1 * m2) / (n * n * (n - 1.0))
        w = (n1 * n2) / n
        rd_num += w * ((a / n1) - (c / n2))
        rd_den += w
    or_mh = float(num_or / den_or) if den_or > 0 else float("inf")
    cmh_chi2 = float((cmh_num * cmh_num) / cmh_var) if cmh_var > 0 else 0.0
    cmh_p = float(chi2.sf(cmh_chi2, df=1)) if cmh_var > 0 else 1.0
    rd_mh = float(rd_num / rd_den) if rd_den > 0 else 0.0
    return {
        "valid_strata": valid_strata,
        "common_odds_ratio_right_vs_left": or_mh,
        "cmh_chi2": cmh_chi2,
        "cmh_p": cmh_p,
        "mh_risk_diff_right_minus_left": rd_mh,
    }


def _recompute_unassigned_success_symmetric(ep: Dict[str, Any]) -> int:
    """Strict symmetric collision-free success on unassigned target."""
    return int(
        (not bool(ep["initially_inside_radius"]))
        and bool(ep["unassigned_reached"])
        and (not bool(ep["collision"]))
        and (float(ep["final_unassigned_goal_error_m"]) <= 0.35)
    )


def _crossover_category(first_A: str, first_B: str) -> Tuple[str, str]:
    """
    Returns (major_4way_category, fine_subpattern) for a scene pair under Assignment A and B.
    Major 4-way categories:
      - both_first_assigned
      - same_physical_object_collapse
      - both_first_unassigned
      - at_least_one_neither
    """
    if first_A == "assigned" and first_B == "assigned":
        return "both_first_assigned", "A_assigned__B_assigned"
    if (first_A == "assigned" and first_B == "unassigned") or (
        first_A == "unassigned" and first_B == "assigned"
    ):
        sub = "collapse_onto_obj1" if first_A == "assigned" else "collapse_onto_obj2"
        return "same_physical_object_collapse", sub
    if first_A == "unassigned" and first_B == "unassigned":
        return "both_first_unassigned", "A_unassigned__B_unassigned"
    return "at_least_one_neither", f"A_{first_A}__B_{first_B}"


def audit_raw_records(
    scenes_path: Path,
    episodes_path: Path,
    scene_pairs_path: Path,
    summary_path: Path,
    shard_manifest_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Part A1: Verifies every invariant across Phase 5.5.1 raw scenes, episodes, scene_pairs,
    summary, and plain-text trace shards.
    Returns (audit_report, scenes_by_id, traces_by_ep_key).
    """
    scenes = load_jsonl(scenes_path)
    episodes = load_jsonl(episodes_path)
    scene_pairs = load_jsonl(scene_pairs_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    shard_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))

    scenes_by_id = {s["scene_id"]: s for s in scenes}
    ep_keys = [e["episode_key"] for e in episodes]
    pair_keys = [(p["condition"], p["scene_id"]) for p in scene_pairs]

    # Shard verification
    shard_dir = shard_manifest_path.parent
    shard_hash_mismatches = 0
    shard_byte_mismatches = 0
    shard_row_mismatches = 0
    traces_by_ep: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_trace_rows = 0

    for sh in shard_manifest["shards"]:
        sp = shard_dir / sh["filename"]
        actual_bytes = sp.stat().st_size
        actual_sha = sha256_file(sp)
        if actual_bytes != int(sh["bytes"]):
            shard_byte_mismatches += 1
        if actual_sha != sh["sha256"]:
            shard_hash_mismatches += 1
        rows_here = load_jsonl(sp)
        if len(rows_here) != int(sh["n_rows"]):
            shard_row_mismatches += 1
        total_trace_rows += len(rows_here)
        for r in rows_here:
            traces_by_ep[r["episode_key"]].append(r)

    ep_key_set = set(ep_keys)
    trace_ep_key_set = set(traces_by_ep.keys())

    # Assignment A/B symmetry & GT coordinate verification
    ep_by_cs: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for e in episodes:
        ep_by_cs[(e["condition"], e["scene_id"])][e["assignment"]] = e

    missing_ab = 0
    pose_mismatches_ab = 0
    rgb_hash_mismatches_ab = 0
    target_swap_errors = 0
    task_token_errors = 0
    gt_wp_ref_errors = 0
    trace_len_mismatches = 0
    simultaneous_first_reach_count = 0
    unassigned_reached_with_collision_count = 0

    for (cond, sid), ab in ep_by_cs.items():
        if set(ab.keys()) != {"A", "B"}:
            missing_ab += 1
            continue
        epA, epB = ab["A"], ab["B"]
        sc = scenes_by_id[sid]

        if epA["obs_unmasked_0_sha256"] != epB["obs_unmasked_0_sha256"]:
            rgb_hash_mismatches_ab += 1
        if epA["target_key"] != sc["obj_1_key"] or epB["target_key"] != sc["obj_2_key"]:
            target_swap_errors += 1
        if epA["distractor_key"] != sc["obj_2_key"] or epB["distractor_key"] != sc["obj_1_key"]:
            target_swap_errors += 1

        exp_tok_A = int(encode_task_id(sc["obj_1_key"], "approach", f"approach {sc['obj_1_key']}"))
        exp_tok_B = int(encode_task_id(sc["obj_2_key"], "approach", f"approach {sc['obj_2_key']}"))
        if exp_tok_A == exp_tok_B:
            task_token_errors += 1

        init_pose = np.asarray(sc["initial_pose"], dtype=np.float32)
        wp1 = np.asarray(sc["wp_1"], dtype=np.float32)
        wp2 = np.asarray(sc["wp_2"], dtype=np.float32)
        gt_a_exp = compute_body_frame_goal_vec(init_pose, wp1)
        gt_b_exp = compute_body_frame_goal_vec(init_pose, wp2)

        trA = sorted(traces_by_ep[epA["episode_key"]], key=lambda x: x["t"])
        trB = sorted(traces_by_ep[epB["episode_key"]], key=lambda x: x["t"])
        if len(trA) != int(epA["executed_steps"]) or len(trB) != int(epB["executed_steps"]):
            trace_len_mismatches += 1
        if trA and trB:
            p0A = np.asarray(trA[0]["pose"], dtype=np.float32)
            p0B = np.asarray(trB[0]["pose"], dtype=np.float32)
            if np.max(np.abs(p0A - init_pose)) > 1e-4 or np.max(np.abs(p0B - init_pose)) > 1e-4:
                pose_mismatches_ab += 1
            gtA0 = np.asarray(trA[0]["gt_assigned_body"], dtype=np.float32)
            gtAu0 = np.asarray(trA[0]["gt_unassigned_body"], dtype=np.float32)
            gtB0 = np.asarray(trB[0]["gt_assigned_body"], dtype=np.float32)
            gtBu0 = np.asarray(trB[0]["gt_unassigned_body"], dtype=np.float32)
            if (
                np.max(np.abs(gtA0 - gt_a_exp)) > 1e-4
                or np.max(np.abs(gtAu0 - gt_b_exp)) > 1e-4
                or np.max(np.abs(gtB0 - gt_b_exp)) > 1e-4
                or np.max(np.abs(gtBu0 - gt_a_exp)) > 1e-4
            ):
                gt_wp_ref_errors += 1

        for ep in (epA, epB):
            fa = ep.get("first_assigned_reach_step")
            fu = ep.get("first_unassigned_reach_step")
            if fa is not None and fu is not None and int(fa) == int(fu):
                simultaneous_first_reach_count += 1
            if bool(ep["unassigned_reached"]) and bool(ep["collision"]):
                unassigned_reached_with_collision_count += 1

    input_hashes = {
        "scenes_jsonl": {"path": str(scenes_path.relative_to(REPO_ROOT)), "bytes": scenes_path.stat().st_size, "sha256": sha256_file(scenes_path)},
        "episodes_jsonl": {"path": str(episodes_path.relative_to(REPO_ROOT)), "bytes": episodes_path.stat().st_size, "sha256": sha256_file(episodes_path)},
        "scene_pairs_jsonl": {"path": str(scene_pairs_path.relative_to(REPO_ROOT)), "bytes": scene_pairs_path.stat().st_size, "sha256": sha256_file(scene_pairs_path)},
        "summary_json": {"path": str(summary_path.relative_to(REPO_ROOT)), "bytes": summary_path.stat().st_size, "sha256": sha256_file(summary_path)},
        "shard_manifest_json": {"path": str(shard_manifest_path.relative_to(REPO_ROOT)), "bytes": shard_manifest_path.stat().st_size, "sha256": sha256_file(shard_manifest_path)},
    }

    checks = {
        "unique_scene_ids": len(scenes_by_id),
        "expected_scene_ids": 60,
        "scene_count_ok": len(scenes_by_id) == 60 and len(scenes) == 60,
        "scene_condition_pair_count": len(scene_pairs),
        "unique_scene_condition_pairs": len(set(pair_keys)),
        "expected_pairs": 420,
        "pair_count_ok": len(scene_pairs) == 420 and len(set(pair_keys)) == 420,
        "episode_count": len(episodes),
        "unique_episode_keys": len(ep_key_set),
        "expected_episodes": 840,
        "episode_count_ok": len(episodes) == 840 and len(ep_key_set) == 840,
        "missing_or_incomplete_AB_pairs": missing_ab,
        "pose_mismatches_AB": pose_mismatches_ab,
        "rgb_hash_mismatches_AB": rgb_hash_mismatches_ab,
        "target_swap_errors": target_swap_errors,
        "task_token_errors": task_token_errors,
        "gt_wp_ref_errors": gt_wp_ref_errors,
        "shard_count": len(shard_manifest["shards"]),
        "total_trace_rows": total_trace_rows,
        "shard_hash_mismatches": shard_hash_mismatches,
        "shard_byte_mismatches": shard_byte_mismatches,
        "shard_row_mismatches": shard_row_mismatches,
        "trace_episode_keys_missing_from_episodes": len(trace_ep_key_set - ep_key_set),
        "episodes_missing_from_traces": len(ep_key_set - trace_ep_key_set),
        "trace_len_mismatches": trace_len_mismatches,
        "simultaneous_first_reach_events": simultaneous_first_reach_count,
        "unassigned_reached_with_collision_count": unassigned_reached_with_collision_count,
    }
    checks["all_integrity_checks_passed"] = bool(
        checks["scene_count_ok"]
        and checks["pair_count_ok"]
        and checks["episode_count_ok"]
        and missing_ab == 0
        and pose_mismatches_ab == 0
        and rgb_hash_mismatches_ab == 0
        and target_swap_errors == 0
        and task_token_errors == 0
        and gt_wp_ref_errors == 0
        and shard_hash_mismatches == 0
        and shard_byte_mismatches == 0
        and shard_row_mismatches == 0
        and checks["trace_episode_keys_missing_from_episodes"] == 0
        and checks["episodes_missing_from_traces"] == 0
        and trace_len_mismatches == 0
    )

    audit_report = {
        "input_hashes": input_hashes,
        "checks": checks,
        "summary_provenance": summary.get("provenance", {}),
    }
    return audit_report, scenes_by_id, traces_by_ep


def analyze_H1_and_biases(
    scenes_by_id: Dict[str, Dict[str, Any]],
    episodes: List[Dict[str, Any]],
    scene_pairs: List[Dict[str, Any]],
    traces_by_ep: Dict[str, List[Dict[str, Any]]],
    seed: int = DEFAULT_SEED,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Implements Part A2 (symmetric endpoints A, B, C, D for H1) and Part A3 (left/right & category biases).
    Returns (summary_dict, scene_pair_matrix_rows, category_side_rows, episode_effect_rows).
    """
    ep_by_cs: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for ep in episodes:
        ep_by_cs[(ep["condition"], ep["scene_id"])][ep["assignment"]] = ep

    sp_lookup = {(p["condition"], p["scene_id"]): p for p in scene_pairs}
    conditions = sorted({ep["condition"] for ep in episodes}, key=lambda c: ["F1_dynamic_visual_with_memory", "F2_dynamic_visual_no_memory", "F3_frozen_visual_with_memory", "F4_frozen_visual_no_memory", "M1_target_masked_rgb", "M2_distractor_masked_rgb", "O1_track_c_planner_goal_override"].index(c) if c in ["F1_dynamic_visual_with_memory", "F2_dynamic_visual_no_memory", "F3_frozen_visual_with_memory", "F4_frozen_visual_no_memory", "M1_target_masked_rgb", "M2_distractor_masked_rgb", "O1_track_c_planner_goal_override"] else 99)
    scene_ids = sorted(scenes_by_id.keys())

    h1_by_condition: Dict[str, Any] = {}
    side_bias_by_condition: Dict[str, Any] = {}
    scene_pair_matrix_rows: List[Dict[str, Any]] = []
    category_side_rows: List[Dict[str, Any]] = []
    episode_effect_rows: List[Dict[str, Any]] = []

    for ep in episodes:
        un_sym = _recompute_unassigned_success_symmetric(ep)
        terminated_on_assigned = int(bool(ep["assigned_reached"]))
        episode_effect_rows.append(
            {
                "condition": ep["condition"],
                "scene_id": ep["scene_id"],
                "assignment": ep["assignment"],
                "episode_key": ep["episode_key"],
                "context": ep["context"],
                "category_pair": ep["category_pair"],
                "target_key": ep["target_key"],
                "distractor_key": ep["distractor_key"],
                "assigned_side": ep["assigned_side"],
                "assigned_reached": int(ep["assigned_reached"]),
                "unassigned_reached": int(ep["unassigned_reached"]),
                "first_reached_object": ep["first_reached_object"],
                "first_is_assigned": int(ep["first_reached_object"] == "assigned"),
                "first_is_unassigned": int(ep["first_reached_object"] == "unassigned"),
                "assigned_success_symmetric": int(ep["assigned_success"]),
                "unassigned_success_symmetric": int(un_sym),
                "collision": int(ep["collision"]),
                "executed_steps": int(ep["executed_steps"]),
                "early_terminated_on_assigned_reach": terminated_on_assigned,
                "final_assigned_goal_error_m": float(ep["final_assigned_goal_error_m"]),
                "final_unassigned_goal_error_m": float(ep["final_unassigned_goal_error_m"]),
            }
        )

    for cond in conditions:
        # Scene-pair vectors (N=60)
        famA_reach_diffs: List[float] = []
        famA_first_diffs: List[float] = []
        famB_succ_diffs: List[float] = []
        eta0_vals: List[float] = []
        cos0_vals: List[float] = []
        eta_mean_vals: List[float] = []

        # Crossover 4-way and 3x3 physical object matrices
        crossover_counts = Counter()
        subpattern_counts = Counter()
        # Rows = Task A physical first hit in [obj_1, obj_2, neither]
        # Cols = Task B physical first hit in [obj_1, obj_2, neither]
        phys_idx = {"obj_1": 0, "obj_2": 1, "neither": 2}
        mat_3x3 = np.zeros((3, 3), dtype=int)

        # Censoring diagnostics
        n_assigned_reached = 0
        n_unassigned_reached = 0
        n_assigned_first = 0
        n_unassigned_first = 0
        n_assigned_succ_sym = 0
        n_unassigned_succ_sym = 0
        n_early_terminated_assigned = 0
        n_unassigned_reached_no_collision = 0
        n_unassigned_reached_with_collision = 0

        # Left vs Right matched scene pairs (N=60)
        lr_both = 0
        lr_left_only = 0
        lr_right_only = 0
        lr_neither = 0
        lr_left_first_counts = Counter()
        lr_right_first_counts = Counter()
        lr_phys_side_hit_counts = Counter()  # which physical side reached first per episode
        lr_by_context = {
            "nominal": {"both": 0, "left_only": 0, "right_only": 0, "neither": 0, "left_succ": 0, "right_succ": 0, "n": 0},
            "illumination_shift": {"both": 0, "left_only": 0, "right_only": 0, "neither": 0, "left_succ": 0, "right_succ": 0, "n": 0},
        }
        cat_side_stats: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"n": 0, "succ": 0, "first_assigned": 0, "first_unassigned": 0, "coll": 0})
        cat_pair_side_stats: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"n": 0, "succ": 0})

        for sid in scene_ids:
            epA = ep_by_cs[(cond, sid)]["A"]
            epB = ep_by_cs[(cond, sid)]["B"]
            sp = sp_lookup[(cond, sid)]

            # Family A
            aA, uA = int(epA["assigned_reached"]), int(epA["unassigned_reached"])
            aB, uB = int(epB["assigned_reached"]), int(epB["unassigned_reached"])
            famA_reach_diffs.append(0.5 * ((aA - uA) + (aB - uB)))

            fA_a = int(epA["first_reached_object"] == "assigned")
            fA_u = int(epA["first_reached_object"] == "unassigned")
            fB_a = int(epB["first_reached_object"] == "assigned")
            fB_u = int(epB["first_reached_object"] == "unassigned")
            famA_first_diffs.append(0.5 * ((fA_a - fA_u) + (fB_a - fB_u)))

            n_assigned_reached += aA + aB
            n_unassigned_reached += uA + uB
            n_assigned_first += fA_a + fB_a
            n_unassigned_first += fA_u + fB_u

            # Family B
            saA, suA = int(epA["assigned_success"]), _recompute_unassigned_success_symmetric(epA)
            saB, suB = int(epB["assigned_success"]), _recompute_unassigned_success_symmetric(epB)
            famB_succ_diffs.append(0.5 * ((saA - suA) + (saB - suB)))
            n_assigned_succ_sym += saA + saB
            n_unassigned_succ_sym += suA + suB

            for ep in (epA, epB):
                if ep["assigned_reached"]:
                    n_early_terminated_assigned += 1
                if ep["unassigned_reached"]:
                    if ep["collision"]:
                        n_unassigned_reached_with_collision += 1
                    else:
                        n_unassigned_reached_no_collision += 1

            # Family C (Crossover)
            major_c, sub_c = _crossover_category(epA["first_reached_object"], epB["first_reached_object"])
            crossover_counts[major_c] += 1
            subpattern_counts[sub_c] += 1

            # Map to physical objects: A assigned=obj_1, B assigned=obj_2
            phys_A = "obj_1" if epA["first_reached_object"] == "assigned" else ("obj_2" if epA["first_reached_object"] == "unassigned" else "neither")
            phys_B = "obj_2" if epB["first_reached_object"] == "assigned" else ("obj_1" if epB["first_reached_object"] == "unassigned" else "neither")
            mat_3x3[phys_idx[phys_A], phys_idx[phys_B]] += 1

            # Family D (Task-switch direction)
            eta0_vals.append(float(sp["task_switch_eta_vis_step0"]))
            cos0_vals.append(float(sp["task_switch_cos_vis_step0"]))
            # Horizon-mean task-switch projection along step-0 axis or instantaneous axis across shared prefix min(lenA, lenB)
            trA = sorted(traces_by_ep[epA["episode_key"]], key=lambda x: x["t"])
            trB = sorted(traces_by_ep[epB["episode_key"]], key=lambda x: x["t"])
            sc = scenes_by_id[sid]
            wp_A = np.asarray(sc["wp_1"][:2], dtype=np.float64)
            wp_B = np.asarray(sc["wp_2"][:2], dtype=np.float64)
            # Use step-0 world transform comparison across full horizon or step-0 axis as baseline
            # Compute mean world-frame projected visual switch when both share steps vs step 0
            common_t = min(len(trA), len(trB))
            eta_steps = []
            d_ab_step0 = np.asarray(trA[0]["gt_assigned_body"][:2], dtype=np.float64) - np.asarray(trA[0]["gt_unassigned_body"][:2], dtype=np.float64)
            norm_sq_0 = float(np.dot(d_ab_step0, d_ab_step0))
            if norm_sq_0 > 1e-12:
                for t_i in range(common_t):
                    gvA_w = np.asarray(trA[t_i]["g_visual"][:2], dtype=np.float64)
                    gvB_w = np.asarray(trB[t_i]["g_visual"][:2], dtype=np.float64)
                    # Note after t>0 trajectories diverge in pose; convert g_visual to world frame to compare against (wp_A - wp_B)
                    pA = trA[t_i]["pose"]
                    pB = trB[t_i]["pose"]
                    cA, sA = math.cos(pA[2]), math.sin(pA[2])
                    cB, sB = math.cos(pB[2]), math.sin(pB[2])
                    wA = np.array([pA[0] + cA * gvA_w[0] - sA * gvA_w[1], pA[1] + sA * gvA_w[0] + cA * gvA_w[1]])
                    wB = np.array([pB[0] + cB * gvB_w[0] - sB * gvB_w[1], pB[1] + sB * gvB_w[0] + cB * gvB_w[1]])
                    d_ab_w = wp_A - wp_B
                    eta_steps.append(float(np.dot(wA - wB, d_ab_w) / np.dot(d_ab_w, d_ab_w)))
            eta_mean_vals.append(float(np.mean(eta_steps)) if eta_steps else float("nan"))

            # Matched Left vs Right pairing per physical scene
            ep_L = epA if epA["assigned_side"] == "left" else epB
            ep_R = epB if epA["assigned_side"] == "left" else epA
            assert ep_L["assigned_side"] == "left" and ep_R["assigned_side"] == "right"
            sL = int(ep_L["assigned_success"])
            sR = int(ep_R["assigned_success"])
            if sL == 1 and sR == 1:
                lr_both += 1
            elif sL == 1 and sR == 0:
                lr_left_only += 1
            elif sL == 0 and sR == 1:
                lr_right_only += 1
            else:
                lr_neither += 1

            ctx = epA["context"]
            lr_by_context[ctx]["n"] += 1
            lr_by_context[ctx]["left_succ"] += sL
            lr_by_context[ctx]["right_succ"] += sR
            if sL == 1 and sR == 1:
                lr_by_context[ctx]["both"] += 1
            elif sL == 1 and sR == 0:
                lr_by_context[ctx]["left_only"] += 1
            elif sL == 0 and sR == 1:
                lr_by_context[ctx]["right_only"] += 1
            else:
                lr_by_context[ctx]["neither"] += 1

            lr_left_first_counts[ep_L["first_reached_object"]] += 1
            lr_right_first_counts[ep_R["first_reached_object"]] += 1
            for ep_side in (ep_L, ep_R):
                fr = ep_side["first_reached_object"]
                if fr == "assigned":
                    lr_phys_side_hit_counts[ep_side["assigned_side"]] += 1
                elif fr == "unassigned":
                    other_side = "right" if ep_side["assigned_side"] == "left" else "left"
                    lr_phys_side_hit_counts[other_side] += 1
                else:
                    lr_phys_side_hit_counts["neither"] += 1

                cs_row = cat_side_stats[(ep_side["target_key"], ep_side["assigned_side"])]
                cs_row["n"] += 1
                cs_row["succ"] += int(ep_side["assigned_success"])
                cs_row["first_assigned"] += int(fr == "assigned")
                cs_row["first_unassigned"] += int(fr == "unassigned")
                cs_row["coll"] += int(ep_side["collision"])

                cp_row = cat_pair_side_stats[(ep_side["category_pair"], ep_side["assigned_side"])]
                cp_row["n"] += 1
                cp_row["succ"] += int(ep_side["assigned_success"])

            scene_pair_matrix_rows.append(
                {
                    "condition": cond,
                    "scene_id": sid,
                    "context": ctx,
                    "category_pair": epA["category_pair"],
                    "obj_1_key": epA["target_key"],
                    "obj_2_key": epB["target_key"],
                    "obj_1_side": epA["assigned_side"],
                    "obj_2_side": epB["assigned_side"],
                    "first_reached_A": epA["first_reached_object"],
                    "first_reached_B": epB["first_reached_object"],
                    "phys_first_A": phys_A,
                    "phys_first_B": phys_B,
                    "crossover_4way": major_c,
                    "crossover_subpattern": sub_c,
                    "assigned_reached_A": aA,
                    "unassigned_reached_A": uA,
                    "assigned_reached_B": aB,
                    "unassigned_reached_B": uB,
                    "assigned_success_sym_A": saA,
                    "unassigned_success_sym_A": suA,
                    "assigned_success_sym_B": saB,
                    "unassigned_success_sym_B": suB,
                    "left_assigned_success": sL,
                    "right_assigned_success": sR,
                    "task_switch_eta_vis_step0": float(sp["task_switch_eta_vis_step0"]),
                    "task_switch_cos_vis_step0": float(sp["task_switch_cos_vis_step0"]),
                    "task_switch_eta_world_horizon_mean": eta_mean_vals[-1],
                }
            )

        # Assemble H1 summary for cond
        n_pairs = len(scene_ids)
        n_eps = 2 * n_pairs
        c_both_a = int(crossover_counts["both_first_assigned"])
        c_coll = int(crossover_counts["same_physical_object_collapse"])
        c_both_u = int(crossover_counts["both_first_unassigned"])
        c_neither = int(crossover_counts["at_least_one_neither"])
        sm_res = stuart_maxwell_3x3(mat_3x3)

        h1_by_condition[cond] = {
            "n_scene_pairs": n_pairs,
            "n_episodes": n_eps,
            "censoring_note": (
                "Closed-loop episodes terminate immediately upon first reaching assigned_wp <= 0.35m; "
                "later unassigned arrivals are therefore unobserved/censored in assigned-reaching episodes. "
                "Competing first-reached outcomes (Family C and first-hit Family A) observe both waypoints identically up to first hit."
            ),
            "censoring_counts": {
                "early_terminated_on_assigned_reach": n_early_terminated_assigned,
                "unassigned_reached_without_collision": n_unassigned_reached_no_collision,
                "unassigned_reached_with_collision": n_unassigned_reached_with_collision,
            },
            "family_A_symmetric_reaching": {
                "assigned_reached_count": n_assigned_reached,
                "assigned_reached_rate": n_assigned_reached / n_eps,
                "unassigned_reached_count": n_unassigned_reached,
                "unassigned_reached_rate": n_unassigned_reached / n_eps,
                "reached_pair_diff_mean": float(np.mean(famA_reach_diffs)),
                "reached_pair_diff_95ci": list(paired_bootstrap_ci(famA_reach_diffs, seed=seed)),
                "reached_wilcoxon": paired_wilcoxon_stats(famA_reach_diffs),
                "first_reached_assigned_count": n_assigned_first,
                "first_reached_assigned_rate": n_assigned_first / n_eps,
                "first_reached_unassigned_count": n_unassigned_first,
                "first_reached_unassigned_rate": n_unassigned_first / n_eps,
                "first_reached_pair_diff_mean": float(np.mean(famA_first_diffs)),
                "first_reached_pair_diff_95ci": list(paired_bootstrap_ci(famA_first_diffs, seed=seed)),
                "first_reached_wilcoxon": paired_wilcoxon_stats(famA_first_diffs),
            },
            "family_B_symmetric_collision_free_success": {
                "assigned_success_symmetric_count": n_assigned_succ_sym,
                "assigned_success_symmetric_rate": n_assigned_succ_sym / n_eps,
                "unassigned_success_symmetric_count": n_unassigned_succ_sym,
                "unassigned_success_symmetric_rate": n_unassigned_succ_sym / n_eps,
                "pair_diff_mean": float(np.mean(famB_succ_diffs)),
                "pair_diff_95ci": list(paired_bootstrap_ci(famB_succ_diffs, seed=seed)),
                "wilcoxon": paired_wilcoxon_stats(famB_succ_diffs),
            },
            "family_C_complete_crossover_matrix_N60": {
                "denominator_scene_pairs": n_pairs,
                "counts": {
                    "both_first_assigned": c_both_a,
                    "same_physical_object_collapse": c_coll,
                    "both_first_unassigned": c_both_u,
                    "at_least_one_neither": c_neither,
                },
                "rates": {
                    "both_first_assigned": c_both_a / n_pairs,
                    "same_physical_object_collapse": c_coll / n_pairs,
                    "both_first_unassigned": c_both_u / n_pairs,
                    "at_least_one_neither": c_neither / n_pairs,
                },
                "wilson_95ci": {
                    "both_first_assigned": list(wilson_ci(c_both_a, n_pairs)),
                    "same_physical_object_collapse": list(wilson_ci(c_coll, n_pairs)),
                    "both_first_unassigned": list(wilson_ci(c_both_u, n_pairs)),
                    "at_least_one_neither": list(wilson_ci(c_neither, n_pairs)),
                },
                "subpatterns": dict(sorted(subpattern_counts.items())),
                "physical_3x3_matrix_rowsA_colsB_obj1_obj2_neither": mat_3x3.tolist(),
                "population_both_assigned_vs_both_unassigned_mcnemar_p": exact_mcnemar_pvalue(c_both_a, c_both_u),
                "population_stuart_maxwell_3x3": sm_res,
                "conditional_exploratory_only_dual_vs_collapse": {
                    "label": "CONDITIONAL_EXPLORATORY_ONLY_EXCLUDES_BOTH_UNASSIGNED_AND_NEITHER",
                    "dual_first_correct": c_both_a,
                    "same_object_collapse": c_coll,
                    "excluded_pairs": c_both_u + c_neither,
                    "binom_two_sided_p": float(binomtest(c_both_a, c_both_a + c_coll, 0.5).pvalue)
                    if (c_both_a + c_coll) > 0
                    else 1.0,
                },
            },
            "family_D_task_switch_direction": {
                "step0_eta_vis": {
                    "mean": float(np.mean(eta0_vals)),
                    "median": float(np.median(eta0_vals)),
                    "95ci": list(paired_bootstrap_ci(eta0_vals, seed=seed)),
                    "wilcoxon": paired_wilcoxon_stats(eta0_vals),
                    "positive_count": int(np.sum(np.asarray(eta0_vals) > 0)),
                    "negative_count": int(np.sum(np.asarray(eta0_vals) < 0)),
                    "sign_test_p": float(binomtest(int(np.sum(np.asarray(eta0_vals) > 0)), len(eta0_vals), 0.5).pvalue),
                },
                "step0_cos_vis": {
                    "mean": float(np.mean(cos0_vals)),
                    "median": float(np.median(cos0_vals)),
                    "95ci": list(paired_bootstrap_ci(cos0_vals, seed=seed)),
                    "wilcoxon": paired_wilcoxon_stats(cos0_vals),
                    "positive_count": int(np.sum(np.asarray(cos0_vals) > 0)),
                    "negative_count": int(np.sum(np.asarray(cos0_vals) < 0)),
                },
                "horizon_mean_world_eta_vis": {
                    "mean": float(np.nanmean(eta_mean_vals)),
                    "95ci": list(paired_bootstrap_ci(eta_mean_vals, seed=seed)),
                    "wilcoxon": paired_wilcoxon_stats(eta_mean_vals),
                },
            },
        }

        # Side bias stats for cond
        left_succ = lr_both + lr_left_only
        right_succ = lr_both + lr_right_only
        lr_diffs = [float(r["left_assigned_success"] - r["right_assigned_success"]) for r in scene_pair_matrix_rows if r["condition"] == cond]
        rl_diffs = [-x for x in lr_diffs]

        # Category-stratified Mantel-Haenszel + logistic regression
        strata_2x2 = []
        for cat in TARGET_KEYS:
            r_row = cat_side_stats[(cat, "right")]
            l_row = cat_side_stats[(cat, "left")]
            strata_2x2.append((r_row["succ"], r_row["n"] - r_row["succ"], l_row["succ"], l_row["n"] - l_row["succ"]))
        mh_res = mantel_haenszel_stratified(strata_2x2)

        # Save category_side rows for CSV
        for cat in TARGET_KEYS:
            for side in ("left", "right"):
                st = cat_side_stats[(cat, side)]
                n_here = st["n"]
                ci_lo, ci_hi = wilson_ci(st["succ"], n_here)
                category_side_rows.append(
                    {
                        "condition": cond,
                        "target_category": cat,
                        "assigned_side": side,
                        "n_episodes": n_here,
                        "assigned_success_count": st["succ"],
                        "assigned_success_rate": st["succ"] / n_here if n_here else 0.0,
                        "wilson_95ci_lo": ci_lo,
                        "wilson_95ci_hi": ci_hi,
                        "first_assigned_count": st["first_assigned"],
                        "first_unassigned_count": st["first_unassigned"],
                        "collision_count": st["coll"],
                    }
                )

        # Logistic regression: unadjusted vs adjusted (category + context)
        from sklearn.linear_model import LogisticRegression
        cond_eps = [ep for ep in episodes if ep["condition"] == cond]
        y_vec = np.array([int(ep["assigned_success"]) for ep in cond_eps], dtype=int)
        x_right = np.array([[1.0 if ep["assigned_side"] == "right" else 0.0] for ep in cond_eps], dtype=np.float64)
        # Design with category + context
        cats = list(TARGET_KEYS)
        X_adj = np.zeros((len(cond_eps), 1 + (len(cats) - 1) + 1), dtype=np.float64)
        for idx_e, ep in enumerate(cond_eps):
            X_adj[idx_e, 0] = 1.0 if ep["assigned_side"] == "right" else 0.0
            c_idx = cats.index(ep["target_key"])
            if c_idx > 0:
                X_adj[idx_e, c_idx] = 1.0
            X_adj[idx_e, -1] = 1.0 if ep["context"] == "illumination_shift" else 0.0
        log_unadj = LogisticRegression(C=1e4, max_iter=1000).fit(x_right, y_vec)
        log_adj = LogisticRegression(C=1e4, max_iter=1000).fit(X_adj, y_vec)

        side_bias_by_condition[cond] = {
            "n_matched_lr_pairs": n_pairs,
            "left_success_count": left_succ,
            "left_success_rate": left_succ / n_pairs,
            "left_wilson_95ci": list(wilson_ci(left_succ, n_pairs)),
            "right_success_count": right_succ,
            "right_success_rate": right_succ / n_pairs,
            "right_wilson_95ci": list(wilson_ci(right_succ, n_pairs)),
            "paired_table_both_leftOnly_rightOnly_neither": [lr_both, lr_left_only, lr_right_only, lr_neither],
            "exact_mcnemar_p": exact_mcnemar_pvalue(lr_left_only, lr_right_only),
            "paired_risk_diff_left_minus_right": float(np.mean(lr_diffs)),
            "paired_risk_diff_left_minus_right_95ci": list(paired_bootstrap_ci(lr_diffs, seed=seed)),
            "paired_risk_diff_right_minus_left": float(np.mean(rl_diffs)),
            "paired_risk_diff_right_minus_left_95ci": list(paired_bootstrap_ci(rl_diffs, seed=seed)),
            "first_reached_when_assigned_left": dict(lr_left_first_counts),
            "first_reached_when_assigned_right": dict(lr_right_first_counts),
            "physical_side_reached_first_counts": dict(lr_phys_side_hit_counts),
            "by_context": {
                ctx: {
                    **vals,
                    "left_rate": vals["left_succ"] / vals["n"],
                    "right_rate": vals["right_succ"] / vals["n"],
                    "exact_mcnemar_p": exact_mcnemar_pvalue(vals["left_only"], vals["right_only"]),
                }
                for ctx, vals in lr_by_context.items()
            },
            "category_x_side": {
                f"{cat}__{side}": dict(cat_side_stats[(cat, side)])
                for cat in TARGET_KEYS
                for side in ("left", "right")
            },
            "category_pair_x_side_success": {
                f"{cp}__{side}": dict(st)
                for (cp, side), st in sorted(cat_pair_side_stats.items())
            },
            "category_stratified_mantel_haenszel": mh_res,
            "logistic_regression_right_vs_left": {
                "unadjusted_log_odds_right": float(log_unadj.coef_[0][0]),
                "unadjusted_odds_ratio_right": float(math.exp(float(log_unadj.coef_[0][0]))),
                "category_context_adjusted_log_odds_right": float(log_adj.coef_[0][0]),
                "category_context_adjusted_odds_ratio_right": float(math.exp(float(log_adj.coef_[0][0]))),
            },
        }

    # Cohort-wide balance check (using F1 = exact mirror of 60 scenes x 2 assignments)
    f1_eps = [ep for ep in episodes if ep["condition"] == "F1_dynamic_visual_with_memory"]
    balance_matrix = {
        cat: {
            "left_n": sum(1 for e in f1_eps if e["target_key"] == cat and e["assigned_side"] == "left"),
            "right_n": sum(1 for e in f1_eps if e["target_key"] == cat and e["assigned_side"] == "right"),
        }
        for cat in TARGET_KEYS
    }

    summary = {
        "cohort_category_x_side_balance": balance_matrix,
        "H1_reanalysis_by_condition": h1_by_condition,
        "left_right_bias_by_condition": side_bias_by_condition,
    }
    return summary, scene_pair_matrix_rows, category_side_rows, episode_effect_rows


def audit_scene_generator_and_visual_geometry(
    scenes_by_id: Dict[str, Dict[str, Any]],
    episodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Part A4: Rerenders the 60 exact committed initial scene configurations using JetAutoEnv
    without modifying model weights or task labels, checks camera extrinsics & RGB SHA-256
    fidelity, and extracts initial image centroids, pixel counts, distances, bearings, and
    background canonical obstacle proximity by side and category.
    """
    ep_hash_lookup = {(e["condition"], e["scene_id"], e["assignment"]): e["obs_unmasked_0_sha256"] for e in episodes}
    env = JetAutoEnv(obs_size=(64, 64))
    cam_id = env.front_camera_id
    cam_pos = env.model.cam_pos[cam_id].copy().tolist()
    cam_quat = env.model.cam_quat[cam_id].copy().tolist()
    cam_mat0 = env.model.cam_mat0[cam_id].copy().tolist()
    cam_fovy = float(env.model.cam_fovy[cam_id])

    per_scene_geom = []
    rgb_hash_matches = 0
    vis_px_mismatches = 0
    side_records: Dict[str, List[Dict[str, float]]] = {"left": [], "right": []}
    cat_records: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    try:
        for sid in sorted(scenes_by_id.keys()):
            sc = scenes_by_id[sid]
            spec = {
                "obstacles": sc["obstacles"],
                "light_diffuse_scale": sc["light_diffuse_scale"],
            }
            env.restore_physical_baseline()
            env.apply_task_spec(spec)
            init_pose = np.asarray(sc["initial_pose"], dtype=np.float32)
            obs, _ = env.reset(options={"initial_pos": (float(init_pose[0]), float(init_pose[1]), float(init_pose[2]))})
            obs_sha = sha256_bytes(obs.tobytes())
            saved_sha = ep_hash_lookup[("F1_dynamic_visual_with_memory", sid, "A")]
            hash_match = bool(obs_sha == saved_sha)
            if hash_match:
                rgb_hash_matches += 1

            seg = render_segmentation_array(env)
            k1, k2 = sc["obj_1_key"], sc["obj_2_key"]
            gid1 = env.geometry_id(CANONICAL_OBSTACLES[k1]["geom_name"])
            gid2 = env.geometry_id(CANONICAL_OBSTACLES[k2]["geom_name"])
            _, px1, c1 = extract_geom_mask_and_stats(seg, gid1)
            _, px2, c2 = extract_geom_mask_and_stats(seg, gid2)
            if px1 != int(sc["init_vis_px_1"]) or px2 != int(sc["init_vis_px_2"]):
                vis_px_mismatches += 1

            bg_vis_px = {}
            min_bg_dist_1 = float("inf")
            min_bg_dist_2 = float("inf")
            p1 = np.asarray(sc["obj_1_pos"][:2], dtype=np.float64)
            p2 = np.asarray(sc["obj_2_pos"][:2], dtype=np.float64)
            for bg_key, bg_cfg in CANONICAL_OBSTACLES.items():
                if bg_key in (k1, k2):
                    continue
                bg_gid = env.geometry_id(bg_cfg["geom_name"])
                _, bg_px, _ = extract_geom_mask_and_stats(seg, bg_gid)
                bg_vis_px[bg_key] = bg_px
                bg_p = np.asarray(sc["obstacles"][bg_key]["pos"][:2], dtype=np.float64)
                min_bg_dist_1 = min(min_bg_dist_1, float(np.linalg.norm(p1 - bg_p)))
                min_bg_dist_2 = min(min_bg_dist_2, float(np.linalg.norm(p2 - bg_p)))

            gt1 = compute_body_frame_goal_vec(init_pose, np.asarray(sc["wp_1"], dtype=np.float32))
            gt2 = compute_body_frame_goal_vec(init_pose, np.asarray(sc["wp_2"], dtype=np.float32))
            obj1_body = compute_body_frame_goal_vec(init_pose, np.asarray(sc["obj_1_pos"], dtype=np.float32))
            obj2_body = compute_body_frame_goal_vec(init_pose, np.asarray(sc["obj_2_pos"], dtype=np.float32))

            rec1 = {
                "init_dist_wp_m": float(sc["init_dist_wp1_m"]),
                "bearing_rad": float(sc["bearing_1_rad"]),
                "abs_bearing_rad": abs(float(sc["bearing_1_rad"])),
                "init_vis_px": float(px1),
                "centroid_u": float(c1[0]) if c1 is not None else float("nan"),
                "centroid_v": float(c1[1]) if c1 is not None else float("nan"),
                "gt_dx_b": float(gt1[0]),
                "gt_dy_b": float(gt1[1]),
                "obj_dx_b": float(obj1_body[0]),
                "obj_dy_b": float(obj1_body[1]),
                "min_bg_obst_dist_m": min_bg_dist_1,
            }
            rec2 = {
                "init_dist_wp_m": float(sc["init_dist_wp2_m"]),
                "bearing_rad": float(sc["bearing_2_rad"]),
                "abs_bearing_rad": abs(float(sc["bearing_2_rad"])),
                "init_vis_px": float(px2),
                "centroid_u": float(c2[0]) if c2 is not None else float("nan"),
                "centroid_v": float(c2[1]) if c2 is not None else float("nan"),
                "gt_dx_b": float(gt2[0]),
                "gt_dy_b": float(gt2[1]),
                "obj_dx_b": float(obj2_body[0]),
                "obj_dy_b": float(obj2_body[1]),
                "min_bg_obst_dist_m": min_bg_dist_2,
            }
            side_records[sc["obj_1_side"]].append(rec1)
            side_records[sc["obj_2_side"]].append(rec2)
            cat_records[k1].append(rec1)
            cat_records[k2].append(rec2)

            per_scene_geom.append(
                {
                    "scene_id": sid,
                    "context": sc["context"],
                    "hash_match_vs_saved_v5_5_1": hash_match,
                    "obj_1_key": k1,
                    "obj_1_side": sc["obj_1_side"],
                    "obj_1_px": px1,
                    "obj_1_centroid_uv": c1,
                    "obj_2_key": k2,
                    "obj_2_side": sc["obj_2_side"],
                    "obj_2_px": px2,
                    "obj_2_centroid_uv": c2,
                    "background_visible_px": bg_vis_px,
                    "min_bg_obst_dist_1_m": min_bg_dist_1,
                    "min_bg_obst_dist_2_m": min_bg_dist_2,
                }
            )
    finally:
        env.close()

    def _agg_metrics(recs: List[Dict[str, float]]) -> Dict[str, Any]:
        keys = ["init_dist_wp_m", "abs_bearing_rad", "init_vis_px", "centroid_u", "centroid_v", "gt_dx_b", "gt_dy_b", "min_bg_obst_dist_m"]
        out = {"n": len(recs)}
        for k in keys:
            arr = np.asarray([r[k] for r in recs], dtype=np.float64)
            out[k] = {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)), "min": float(np.nanmin(arr)), "max": float(np.nanmax(arr))}
        return out

    # Paired L vs R comparison on scene geometry
    l_px = np.asarray([r["init_vis_px"] for r in side_records["left"]], dtype=np.float64)
    r_px = np.asarray([r["init_vis_px"] for r in side_records["right"]], dtype=np.float64)
    l_u = np.asarray([r["centroid_u"] for r in side_records["left"]], dtype=np.float64)
    r_u = np.asarray([r["centroid_u"] for r in side_records["right"]], dtype=np.float64)
    l_dist = np.asarray([r["init_dist_wp_m"] for r in side_records["left"]], dtype=np.float64)
    r_dist = np.asarray([r["init_dist_wp_m"] for r in side_records["right"]], dtype=np.float64)

    return {
        "front_cam_extrinsics": {
            "cam_id": int(cam_id),
            "pos": cam_pos,
            "quat": cam_quat,
            "mat0": cam_mat0,
            "fovy_deg": cam_fovy,
            "image_size": [64, 64],
            "u_center_px": 31.5,
            "u_sign_convention": "u < 31.5 is left (+bearing_rad, +dy_b); u > 31.5 is right (-bearing_rad, -dy_b)",
        },
        "rerender_fidelity": {
            "n_scenes": len(per_scene_geom),
            "obs_unmasked_0_sha256_exact_matches": rgb_hash_matches,
            "init_vis_px_mismatches": vis_px_mismatches,
        },
        "side_geometry_summary": {
            "left": _agg_metrics(side_records["left"]),
            "right": _agg_metrics(side_records["right"]),
            "left_vs_right_mannwhitney": {
                "init_vis_px_p": float(mannwhitneyu(l_px, r_px, alternative="two-sided").pvalue),
                "init_dist_wp_m_p": float(mannwhitneyu(l_dist, r_dist, alternative="two-sided").pvalue),
                "abs_u_offset_from_31_5_p": float(mannwhitneyu(np.abs(l_u - 31.5), np.abs(r_u - 31.5), alternative="two-sided").pvalue),
                "centroid_u_sum_minus_63_mean": float(np.mean(l_u + r_u - 63.0)),
            },
        },
        "category_geometry_summary": {cat: _agg_metrics(cat_records[cat]) for cat in TARGET_KEYS},
        "per_scene_geometry": per_scene_geom,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_schema_md(path: Path) -> None:
    content = """# Phase 5.5.1.1 — H1 Correction, Bias Audit & Geometry Schema

## Generated Artifacts
1. `task_conditioned_closed_loop_grounding_v5_5_1_1_H1_reanalysis_summary.json`
   - `raw_audit`: Verification counts and input/output SHA-256 hashes over all 60 scenes, 420 pairs, 840 episodes, and 39 JSONL trace shards.
   - `cohort_category_x_side_balance`: Exact counts of `(target_category, assigned_side)` across the 60 scenes.
   - `H1_reanalysis_by_condition`: Symmetric Endpoint Families A (reaching + first-hit), B (collision-free `assigned_success_symmetric` vs `unassigned_success_symmetric`), C (complete 4-category `N=60` crossover matrix + 3x3 Stuart-Maxwell + conditional exploratory `3 vs 13` binomial), and D (task-switch direction).
   - `left_right_bias_by_condition`: Exact L/R McNemar tests, paired risk differences, nominal vs illumination stratification, category x side tables, Mantel-Haenszel stratification, and unadjusted vs category+context-adjusted logistic regression.
2. `task_conditioned_closed_loop_grounding_v5_5_1_1_scene_pair_matrix.csv`
   - 420 rows (`7` conditions x `60` scenes) recording Assignment A/B endpoints, `crossover_4way`, `crossover_subpattern`, physical object hits, matched L/R successes, and task-switch projections.
3. `task_conditioned_closed_loop_grounding_v5_5_1_1_category_side_table.csv`
   - 84 rows (`7` conditions x `6` categories x `2` sides) with exact episode counts, success counts, Wilson 95% CIs, first-hit counts, and collisions.
4. `task_conditioned_closed_loop_grounding_v5_5_1_1_episode_effects.csv`
   - 840 rows (`7` conditions x `60` scenes x `2` assignments) containing symmetric success, symmetric reaching, first-hit indicators, and early-termination censoring flags.
5. `task_conditioned_closed_loop_grounding_v5_5_1_1_geometry_audit.json`
   - MuJoCo `front_cam` extrinsics, initial-frame SHA-256 verification, segmentation pixel areas and `[u, v]` centroids for all 60 scenes, and side/category geometric feature distributions.
"""
    path.write_text(content, encoding="utf-8")


def run_audit(
    scenes_path: Path = DEFAULT_SCENES_PATH,
    episodes_path: Path = DEFAULT_EPISODES_PATH,
    scene_pairs_path: Path = DEFAULT_PAIRS_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    shard_manifest_path: Path = DEFAULT_SHARD_MANIFEST_PATH,
    output_dir: Path = RESULTS_DIR,
    skip_rerender: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_audit, scenes_by_id, traces_by_ep = audit_raw_records(
        scenes_path=scenes_path,
        episodes_path=episodes_path,
        scene_pairs_path=scene_pairs_path,
        summary_path=summary_path,
        shard_manifest_path=shard_manifest_path,
    )
    if not raw_audit["checks"]["all_integrity_checks_passed"]:
        raise RuntimeError(f"Phase 5.5.1 raw integrity audit failed: {raw_audit['checks']}")

    episodes = load_jsonl(episodes_path)
    scene_pairs = load_jsonl(scene_pairs_path)
    h1_and_bias, pair_rows, cat_side_rows, ep_rows = analyze_H1_and_biases(
        scenes_by_id=scenes_by_id,
        episodes=episodes,
        scene_pairs=scene_pairs,
        traces_by_ep=traces_by_ep,
    )

    if skip_rerender:
        geom_audit = {"skipped": True}
    else:
        geom_audit = audit_scene_generator_and_visual_geometry(scenes_by_id=scenes_by_id, episodes=episodes)

    p_pair_csv = output_dir / OUT_SCENE_PAIR_CSV.name
    p_cat_csv = output_dir / OUT_CAT_SIDE_CSV.name
    p_ep_csv = output_dir / OUT_EPISODE_EFFECTS_CSV.name
    p_geom = output_dir / OUT_GEOMETRY_JSON.name
    p_schema = output_dir / OUT_SCHEMA_MD.name
    p_summary = output_dir / OUT_H1_SUMMARY.name

    _write_csv(p_pair_csv, pair_rows)
    _write_csv(p_cat_csv, cat_side_rows)
    _write_csv(p_ep_csv, ep_rows)
    p_geom.write_text(json.dumps(geom_audit, indent=2) + "\n", encoding="utf-8")
    _write_schema_md(p_schema)

    full_summary = {
        "audit_name": "Phase 5.5.1.1 H1 Estimand Correction and Bias Audit",
        "raw_audit": raw_audit,
        **h1_and_bias,
        "geometry_audit_summary": {
            k: v for k, v in geom_audit.items() if k != "per_scene_geometry"
        },
        "output_files": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in (p_pair_csv, p_cat_csv, p_ep_csv, p_geom, p_schema)
        },
    }
    p_summary.write_text(json.dumps(full_summary, indent=2) + "\n", encoding="utf-8")
    return full_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5.5.1.1 H1 reanalysis and bias audit.")
    parser.add_argument("--skip-rerender", action="store_true", help="Skip MuJoCo initial-frame geometry rerendering.")
    args = parser.parse_args()

    res = run_audit(skip_rerender=args.skip_rerender)
    f1_h1 = res["H1_reanalysis_by_condition"]["F1_dynamic_visual_with_memory"]
    f1_lr = res["left_right_bias_by_condition"]["F1_dynamic_visual_with_memory"]
    print(f"Integrity checks passed: {res['raw_audit']['checks']['all_integrity_checks_passed']}")
    print(f"F1 crossover N=60: {f1_h1['family_C_complete_crossover_matrix_N60']['counts']}")
    print(
        f"F1 L vs R: {f1_lr['left_success_count']}/60 ({f1_lr['left_success_rate']*100:.1f}%) "
        f"vs {f1_lr['right_success_count']}/60 ({f1_lr['right_success_rate']*100:.1f}%), "
        f"McNemar p={f1_lr['exact_mcnemar_p']:.4e}"
    )


if __name__ == "__main__":
    main()
