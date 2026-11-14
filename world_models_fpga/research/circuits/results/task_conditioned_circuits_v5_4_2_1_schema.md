# Phase 5.4.2.1 Exact Key-Set Reconciliation Schema

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
