# Phase 5.4.2 Data Dictionary

## Output Artifacts (`world_models_fpga/research/circuits/results/`)

1. `task_conditioned_circuits_v5_4_2_estimand_audit_summary.json` — Master audit summary, provenance SHA-256 digests, baseline identity verification, no-op cross-target artifact audit, 4-test Holm family definition, and per `(layer, k)` pair-level summaries (overall and stratified by `nominal` vs `illumination_shift`).
2. `task_conditioned_circuits_v5_4_2_condition_setting_table.csv` — Flat summary table across `layer`, `k`, and `cohort` (`candidate_full`, `dimension_matched_full`, `candidate_eligible`, `norm_matched_eligible`).
3. `task_conditioned_circuits_v5_4_2_pair_metrics.jsonl` — 60 JSON lines (`PAIR_000`–`PAIR_059`), each containing pair-level averages over the 4 pair-settings (`scene_a/b` $\times$ `A_to_B/B_to_A`) per `(layer, k)` and condition.
4. `task_conditioned_circuits_v5_4_2_schema.md` — Mathematical definitions of every estimand and unit convention.

## Units & Sign Conventions

| Field | Units | Definition & Sign Convention |
|---|---|---|
| `donor_target_gain_m` | meters | $\|\text{base} - \mathbf{g}_{\text{donor}}^*\| - \|\text{trans} - \mathbf{g}_{\text{donor}}^*\|$ (`>0`: closer to donor) |
| `recipient_target_change_m` | meters | $\|\text{trans} - \mathbf{g}_{\text{recip}}^*\| - \|\text{base} - \mathbf{g}_{\text{recip}}^*\|$ (`>0`: farther from recipient) |
| `eta_gt` | **dimensionless** | $\langle \mathbf{d}_{\text{int}}, \mathbf{d}_{\text{true}} \rangle / \|\mathbf{d}_{\text{true}}\|^2$ |
| `target_parallel_component_m` | meters | $\langle \mathbf{d}_{\text{int}}, \mathbf{d}_{\text{true}} \rangle / \|\mathbf{d}_{\text{true}}\|$ |
| `orthogonal_residual_gt_m` | meters | $\|\mathbf{d}_{\text{int}} - \eta_{\text{gt}}\mathbf{d}_{\text{true}}\|$ |
| `intervention_norm_m` | meters | $\|\mathbf{d}_{\text{int}}\|$ |
| `norm_ratio` | **dimensionless** | $\|\mathbf{d}_{\text{int}}\| / \|\mathbf{d}_{\text{true}}\|$ |
| `legacy_cross_target_error_diff_m` | meters | $\|\text{base} - \mathbf{g}_{\text{recip}}^*\| - \|\text{trans} - \mathbf{g}_{\text{donor}}^*\|$ (retained strictly labeled `LEGACY_INVALID_CROSS_TARGET`) |
