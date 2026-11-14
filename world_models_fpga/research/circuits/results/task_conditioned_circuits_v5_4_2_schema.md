# Phase 5.4.2 Schema Documentation: Single-Reference-Target Estimand Audit

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

- `task_conditioned_circuits_v5_4_2_estimand_audit_summary.json`: Provenance hashes, baseline identity verification across draws, no-op cross-target artifact check, complete layer $\times$ $k$ pair-level statistics, context stratifications, and Holm-adjusted `EXPLORATORY_COMPLETE_SUPPORT` tests.
- `task_conditioned_circuits_v5_4_2_condition_setting_table.csv`: Tidy table of all 10 `(layer, k)` settings across full and eligible cohorts.
- `task_conditioned_circuits_v5_4_2_pair_metrics.jsonl`: Pair-level aggregated metrics for all 60 independent scene pairs.
- `task_conditioned_circuits_v5_4_2_schema.md`: This schema document.
