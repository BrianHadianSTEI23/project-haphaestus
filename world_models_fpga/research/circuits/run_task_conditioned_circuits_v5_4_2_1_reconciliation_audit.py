#!/usr/bin/env python3
"""Project Hephaestus: Phase 5.4.2.1 Exact Intervention Key-Set Reconciliation Audit.

Enumerates the full expected intervention key set from Phase 5.3 pair metadata:
  (pair_id, scene_id, layer, direction, k, condition, draw_id)
and compares it against the observed compressed intervention records, verifying:
  - expected_rows vs observed_rows
  - exact missing keys
  - exact unexpected keys
  - exact duplicate keys
  - unmatched pair_ids
  - invalid condition/layer/direction/k combinations
  - per-condition discrepancies

Exact set equality is required before declaring reconciliation successful.
Preserves all Phase 5.4.2 artifacts untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
VALID_SCENES: Tuple[str, ...] = ("scene_a", "scene_b")
VALID_LAYERS: Tuple[str, ...] = ("layer1_post_relu_64d", "layer2_post_relu_64d")
VALID_DIRECTIONS: Tuple[str, ...] = ("A_to_B", "B_to_A")
VALID_K_VALUES: Tuple[int, ...] = (2, 4, 8, 16, 32)
VALID_CONDITIONS: Tuple[str, ...] = (
    "candidate",
    "dimension_matched_random",
    "norm_matched_random",
)

InterventionKey = Tuple[str, str, str, str, int, str, int]


def compute_file_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "missing"
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_info(repo_root: Path = REPO_ROOT) -> Tuple[str, bool]:
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


def build_expected_key_set_from_pairs(
    pairs_data: Sequence[Dict[str, Any]],
    n_random_draws: int = 100,
) -> Tuple[Set[InterventionKey], Set[str], Dict[str, int]]:
    """Constructs the exact expected key set from Phase 5.3 pair metadata."""
    expected_keys: Set[InterventionKey] = set()
    expected_pair_ids: Set[str] = set()
    per_cond_expected: Dict[str, int] = {c: 0 for c in VALID_CONDITIONS}

    for prow in pairs_data:
        pid = str(prow["pair_id"])
        expected_pair_ids.add(pid)
        for rkey, res in prow["results"].items():
            parts = rkey.split("__")
            if len(parts) != 4 or not parts[3].startswith("k_"):
                raise ValueError(f"Malformed result setting key in pairs metadata: {rkey!r}")
            sc, layer, direction, k_str = parts
            k = int(k_str.split("_", 1)[1])

            # Candidate: draw_id 0 for every expected setting
            expected_keys.add((pid, sc, layer, direction, k, "candidate", 0))
            per_cond_expected["candidate"] += 1

            # Dimension-matched random: draw_ids 0..n_random_draws-1 for every expected setting
            for d in range(n_random_draws):
                expected_keys.add((pid, sc, layer, direction, k, "dimension_matched_random", d))
            per_cond_expected["dimension_matched_random"] += n_random_draws

            # Norm-matched random: draw_ids 0..n_random_draws-1 ONLY for VALID_COMMON_SUPPORT
            if res.get("norm_matched_status") == "VALID_COMMON_SUPPORT":
                for d in range(n_random_draws):
                    expected_keys.add((pid, sc, layer, direction, k, "norm_matched_random", d))
                per_cond_expected["norm_matched_random"] += n_random_draws

    return expected_keys, expected_pair_ids, per_cond_expected


def reconcile_intervention_keys(
    pairs_data: Sequence[Dict[str, Any]],
    observed_rows_iter: Iterable[Dict[str, Any]],
    n_random_draws: int = 100,
    valid_scenes: Sequence[str] = VALID_SCENES,
    valid_layers: Sequence[str] = VALID_LAYERS,
    valid_directions: Sequence[str] = VALID_DIRECTIONS,
    valid_k_values: Sequence[int] = VALID_K_VALUES,
    valid_conditions: Sequence[str] = VALID_CONDITIONS,
    sample_limit: int = 25,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Performs exact key-set reconciliation comparing expected keys to observed intervention records."""
    expected_keys, expected_pair_ids, per_cond_expected = build_expected_key_set_from_pairs(
        pairs_data, n_random_draws=n_random_draws
    )
    allowed_scenes = set(valid_scenes)
    allowed_layers = set(valid_layers)
    allowed_directions = set(valid_directions)
    allowed_ks = set(int(v) for v in valid_k_values)
    allowed_conditions = set(valid_conditions)

    observed_keys: Set[InterventionKey] = set()
    observed_row_count = 0
    duplicate_row_count = 0
    duplicate_key_counts: Counter[InterventionKey] = Counter()
    unmatched_pair_id_rows = 0
    unmatched_pair_ids: Set[str] = set()
    invalid_combination_rows = 0
    invalid_combination_samples: List[Dict[str, Any]] = []
    per_cond_observed_rows: Dict[str, int] = defaultdict(int)
    per_cond_observed_unique: Dict[str, Set[InterventionKey]] = defaultdict(set)
    per_cond_duplicate_rows: Dict[str, int] = defaultdict(int)

    for row in observed_rows_iter:
        observed_row_count += 1
        pid = str(row.get("pair_id", ""))
        sc = str(row.get("scene_id", ""))
        layer = str(row.get("layer", ""))
        direction = str(row.get("direction", ""))
        raw_k = row.get("k")
        cond = str(row.get("condition", ""))
        raw_draw = row.get("draw_id")

        try:
            k = int(raw_k)
        except (TypeError, ValueError):
            k = -1
        try:
            draw_id = int(raw_draw)
        except (TypeError, ValueError):
            draw_id = -1

        per_cond_observed_rows[cond] += 1

        if pid not in expected_pair_ids:
            unmatched_pair_id_rows += 1
            unmatched_pair_ids.add(pid)

        is_valid_combo = (
            sc in allowed_scenes
            and layer in allowed_layers
            and direction in allowed_directions
            and k in allowed_ks
            and cond in allowed_conditions
            and 0 <= draw_id < n_random_draws
            and (cond != "candidate" or draw_id == 0)
        )
        if not is_valid_combo:
            invalid_combination_rows += 1
            if len(invalid_combination_samples) < sample_limit:
                invalid_combination_samples.append(
                    {
                        "pair_id": pid,
                        "scene_id": sc,
                        "layer": layer,
                        "direction": direction,
                        "k": raw_k,
                        "condition": cond,
                        "draw_id": raw_draw,
                    }
                )

        key: InterventionKey = (pid, sc, layer, direction, k, cond, draw_id)
        if key in observed_keys:
            duplicate_row_count += 1
            per_cond_duplicate_rows[cond] += 1
        observed_keys.add(key)
        duplicate_key_counts[key] += 1
        per_cond_observed_unique[cond].add(key)

    missing_keys = expected_keys - observed_keys
    unexpected_keys = observed_keys - expected_keys
    duplicated_unique_keys = [k for k, cnt in duplicate_key_counts.items() if cnt > 1]

    all_conds = sorted(set(VALID_CONDITIONS) | set(per_cond_observed_rows.keys()))
    per_condition_rows: List[Dict[str, Any]] = []
    per_condition_summary: Dict[str, Any] = {}
    for cond in all_conds:
        exp_cond_keys = {k for k in expected_keys if k[5] == cond}
        obs_cond_keys = per_cond_observed_unique.get(cond, set())
        miss_c = exp_cond_keys - obs_cond_keys
        unexp_c = obs_cond_keys - exp_cond_keys
        exp_cnt = per_cond_expected.get(cond, 0)
        obs_cnt = per_cond_observed_rows.get(cond, 0)
        dup_cnt = per_cond_duplicate_rows.get(cond, 0)
        exact_cond = (
            len(miss_c) == 0
            and len(unexp_c) == 0
            and dup_cnt == 0
            and exp_cnt == obs_cnt
        )
        row_dict = {
            "condition": cond,
            "expected_rows": int(exp_cnt),
            "observed_rows": int(obs_cnt),
            "observed_unique_keys": int(len(obs_cond_keys)),
            "missing_keys_count": int(len(miss_c)),
            "unexpected_keys_count": int(len(unexp_c)),
            "duplicate_keys_count": int(dup_cnt),
            "count_delta_observed_minus_expected": int(obs_cnt - exp_cnt),
            "exact_set_equality": bool(exact_cond),
        }
        per_condition_rows.append(row_dict)
        per_condition_summary[cond] = row_dict

    exact_set_equality = bool(
        len(missing_keys) == 0
        and len(unexpected_keys) == 0
        and duplicate_row_count == 0
        and unmatched_pair_id_rows == 0
        and invalid_combination_rows == 0
        and len(expected_keys) == observed_row_count
        and expected_keys == observed_keys
    )

    def _fmt_keys(keys: Iterable[InterventionKey]) -> List[Dict[str, Any]]:
        return [
            {
                "pair_id": k[0],
                "scene_id": k[1],
                "layer": k[2],
                "direction": k[3],
                "k": k[4],
                "condition": k[5],
                "draw_id": k[6],
            }
            for k in sorted(keys)[:sample_limit]
        ]

    summary = {
        "expected_rows": int(len(expected_keys)),
        "observed_rows": int(observed_row_count),
        "observed_unique_keys": int(len(observed_keys)),
        "missing_keys_count": int(len(missing_keys)),
        "unexpected_keys_count": int(len(unexpected_keys)),
        "duplicate_rows_count": int(duplicate_row_count),
        "duplicated_unique_keys_count": int(len(duplicated_unique_keys)),
        "unmatched_pair_id_rows_count": int(unmatched_pair_id_rows),
        "unmatched_pair_ids": sorted(unmatched_pair_ids),
        "invalid_combinations_rows_count": int(invalid_combination_rows),
        "invalid_combination_samples": invalid_combination_samples,
        "missing_key_samples": _fmt_keys(missing_keys),
        "unexpected_key_samples": _fmt_keys(unexpected_keys),
        "duplicate_key_samples": _fmt_keys(duplicated_unique_keys),
        "per_condition_discrepancies": per_condition_summary,
        "exact_set_equality": exact_set_equality,
        "reconciliation_verdict": (
            "EXACT_SET_EQUALITY_VERIFIED" if exact_set_equality else "RECONCILIATION_DISCREPANCY_DETECTED"
        ),
    }
    return summary, per_condition_rows


def _iter_jsonl_gz(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def run_reconciliation_audit(
    circuits_results_dir: Optional[Path] = None,
    review_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    res_dir = circuits_results_dir or (REPO_ROOT / "world_models_fpga/research/circuits/results")
    rev_dir = review_dir or (REPO_ROOT / "research/review_package/phase5_4_2_1")
    rev_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = res_dir / "task_conditioned_circuits_v5_3_pairs.jsonl"
    interventions_path = res_dir / "task_conditioned_circuits_v5_3_interventions.jsonl.gz"

    pairs_data = [json.loads(line) for line in pairs_path.open("r", encoding="utf-8") if line.strip()]
    summary_core, csv_rows = reconcile_intervention_keys(pairs_data, _iter_jsonl_gz(interventions_path))

    git_commit, git_clean = get_git_info(REPO_ROOT)
    audit_out = {
        "audit_name": "Phase 5.4.2.1 Exact Intervention Key-Set Reconciliation Audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_clean": git_clean,
        "provenance": {
            "pairs_path": str(pairs_path.relative_to(REPO_ROOT)),
            "pairs_sha256": compute_file_sha256(pairs_path),
            "interventions_path": str(interventions_path.relative_to(REPO_ROOT)),
            "interventions_sha256": compute_file_sha256(interventions_path),
            "key_schema": ["pair_id", "scene_id", "layer", "direction", "k", "condition", "draw_id"],
        },
        **summary_core,
    }

    summary_path = res_dir / "task_conditioned_circuits_v5_4_2_1_audit_summary.json"
    csv_path = res_dir / "task_conditioned_circuits_v5_4_2_1_key_reconciliation.csv"
    schema_path = res_dir / "task_conditioned_circuits_v5_4_2_1_schema.md"

    summary_path.write_text(json.dumps(audit_out, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "condition",
            "expected_rows",
            "observed_rows",
            "observed_unique_keys",
            "missing_keys_count",
            "unexpected_keys_count",
            "duplicate_keys_count",
            "count_delta_observed_minus_expected",
            "exact_set_equality",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    schema_text = """# Phase 5.4.2.1 Exact Key-Set Reconciliation Schema

## Key Definition
Every intervention record is uniquely keyed by:
`(pair_id, scene_id, layer, direction, k, condition, draw_id)`

## Expected Key-Set Construction (`task_conditioned_circuits_v5_3_pairs.jsonl`)
- `candidate`: `draw_id == 0` for every `(pair_id, scene_id, layer, direction, k)` setting.
- `dimension_matched_random`: `draw_id in 0..99` for every setting.
- `norm_matched_random`: `draw_id in 0..99` exclusively when `norm_matched_status == "VALID_COMMON_SUPPORT"`.

## Artifacts
- `task_conditioned_circuits_v5_4_2_1_audit_summary.json`: Exact set comparison counts, samples (if any), per-condition discrepancies, and `exact_set_equality` verdict.
- `task_conditioned_circuits_v5_4_2_1_key_reconciliation.csv`: Per-condition expected vs observed counts and missing/unexpected/duplicate breakdowns.
"""
    schema_path.write_text(schema_text, encoding="utf-8")
    return audit_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5.4.2.1 Exact Key-Set Reconciliation Audit")
    parser.parse_args()
    out = run_reconciliation_audit()
    print(f"Expected rows: {out['expected_rows']}, Observed rows: {out['observed_rows']}")
    print(
        f"Missing: {out['missing_keys_count']}, Unexpected: {out['unexpected_keys_count']}, "
        f"Duplicates: {out['duplicate_rows_count']}, Unmatched pairs: {out['unmatched_pair_id_rows_count']}, "
        f"Invalid combos: {out['invalid_combinations_rows_count']}"
    )
    print(f"Verdict: {out['reconciliation_verdict']} (exact_set_equality={out['exact_set_equality']})")


if __name__ == "__main__":
    main()
