# Phase 5.4.2.1 — Data Dictionary

## 1. Summary JSON (`world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_4_2_1_audit_summary.json`)
- `expected_rows`: Integer count (`447800`) of composite keys constructed from `task_conditioned_circuits_v5_3_pairs.jsonl`.
- `observed_rows`: Total intervention rows (`447800`) read from `task_conditioned_circuits_v5_3_interventions.jsonl.gz`.
- `observed_unique_keys`: Distinct composite keys (`447800`) observed.
- `missing_count`, `unexpected_count`, `duplicate_count`, `unmatched_pair_count`, `invalid_combo_count`: Discrepancy counts (all `0`).
- `per_condition`: Condition-level breakdown for `candidate`, `dimension_matched_random`, and `norm_matched_random` (`expected_rows`, `observed_rows`, `missing_count`, `unexpected_count`, `duplicate_count`, `exact_set_equality`).
- `exact_set_equality`: Boolean verdict (`true`).
- `verdict`: String status (`EXACT_SET_EQUALITY_VERIFIED`).

## 2. Condition CSV (`world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_4_2_1_key_reconciliation.csv`)
4-row tabular reconciliation (`condition`, `expected_rows`, `observed_rows`, `missing_count`, `unexpected_count`, `duplicate_count`, `exact_set_equality`) for `candidate`, `dimension_matched_random`, `norm_matched_random`, and `ALL`.

## 3. Schema File (`world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_4_2_1_schema.md`)
Human-readable key-set specification and artifact index.
