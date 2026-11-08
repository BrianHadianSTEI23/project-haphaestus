# Phase 5.4.1 Data Dictionary & Artifact Specification

## Forensic Reconciliation & Reporting Audit

This document provides the formal data dictionary, schema definitions, coordinate frames, and physical units for all artifacts produced in Project Hephaestus Phase 5.4.1.

---

## 1. File Manifest & Storage Locations

All raw experimental outputs are stored in `world_models_fpga/research/circuits/results/`:

| Filename | Format | Description | Record Count | File Size |
| :--- | :--- | :--- | :--- | :--- |
| `task_conditioned_circuits_v5_4_1_audit_summary.json` | JSON | Complete audit summary, programmatic markdown table, provenance hashes, and claims ledger | 1 document | ~32 KB |
| `task_conditioned_circuits_v5_4_1_condition_counts.csv` | CSV | Reconciled expected vs observed row counts and anomaly metrics across conditions | 3 records | ~200 B |
| `task_conditioned_circuits_v5_4_1_pair_metrics.jsonl` | JSON Lines | Independent scene-pair aggregated metrics for all 60 pairs across candidate and norm-matched controls | 60 records | ~171 KB |
| `task_conditioned_circuits_v5_4_1_schema.md` | Markdown | Formal schema specification and mathematical formulas | 1 document | ~2.7 KB |

---

## 2. Coordinate Systems & Units

- **Coordinate System**: 2D body-relative Cartesian frame ($m$).
  - $+X$: Points directly forward along the robot's heading direction.
  - $+Y$: Points lateral left, perpendicular to forward heading.
- **Physical Translations ($x, y$)**: Metres ($m$) relative to robot base origin.
- **Ground-Truth Target Definition**: The evaluated ground truth is strictly the task-defined standoff goal waypoint $\mathbf{w}$ located 0.60 m in front of the target object along the line of sight from the robot:
  $$\mathbf{w} = \mathbf{p}_{\text{obj}} - 0.60 \frac{\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{robot}}}{\|\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{robot}}\|}$$
- **Body-Frame Waypoint ($\mathbf{g}_{\text{gt}}$)**: Waypoint coordinates rotated into the robot's local frame via $R(-\theta_{\text{robot}})$.
- **Goal Displacements ($\mathbf{g}_{\text{model}}, \mathbf{g}_{\text{gt}}$)**: 2D vectors $[x, y]$ in metres.
- **Heading Angles ($\theta$, bearing)**: Radians, wrapped to $[-\pi, \pi]$ using `atan2(sin, cos)`.
- **Contrast Projection Efficiency ($\eta$)**: Dimensionless scalar projection.

---

## 3. Field Specifications

### A. `task_conditioned_circuits_v5_4_1_condition_counts.csv`
- `condition`: Control condition (`"candidate"`, `"dimension_matched_random"`, or `"norm_matched_random"`).
- `expected_rows`: Mathematically derived record count based on experimental parameters.
- `observed_rows`: Count of records encountered in the database.
- `missing_rows`: Expected keys not present in the database (0 across all conditions).
- `unexpected_rows`: Encountered records not matching expected Cartesian key set (0 across all conditions).
- `duplicate_rows`: Records with duplicate primary key tuples (0 across all conditions).

### B. `task_conditioned_circuits_v5_4_1_pair_metrics.jsonl`
One record per independent scene pair ($N=60$):
- `pair_id`: Unique identifier (`PAIR_000` to `PAIR_059`).
- `context`: Visual context regime (`"nominal"` or `"illumination_shift"`).
- `object_pair`: Semantic target pair (e.g., `"red_box_vs_blue_cylinder"`).
- `pooled_candidate`: Aggregated candidate intervention metrics across all layers, $k$, directions, and scenes:
  - `mean_eta_gt`: Mean ground-truth contrast projection efficiency $\eta_{\text{gt}}$.
  - `mean_error_improvement_m`: Mean Euclidean error reduction toward donor target ($m$).
  - `fraction_positive_error_improvement`: Fraction of candidate interventions with $\Delta E > 0$.
  - `mean_eta_model`: Mean internal model contrast reproduction efficiency $\eta_{\text{model}}$.
- `results_by_layer_k`: Detailed breakdown per `(layer, k)` setting:
  - `candidate_eta_gt`: Mean $\eta_{\text{gt}}$ for candidate units on this pair.
  - `candidate_error_improvement_m`: Mean error improvement for candidate units ($m$).
  - `candidate_eta_model`: Mean $\eta_{\text{model}}$ for candidate units.
  - `norm_matched_eta_gt`: Mean $\eta_{\text{gt}}$ across valid norm-matched draws (`null` if ineligible).
  - `norm_matched_error_improvement_m`: Mean error improvement across valid norm-matched draws (`null` if ineligible).
  - `is_eligible`: Boolean flag indicating whether this pair had `VALID_COMMON_SUPPORT` ($N \ge 100$) in all 4 scene/direction quadrants for this `(layer, k)`.

### C. `task_conditioned_circuits_v5_4_1_audit_summary.json`
- `reconciliation_audit`: Mathematical row count derivations and streaming verification flags.
- `waypoint_grounding_audit`: Standoff distance ($0.60$ m), waypoint equation, full-model spatial accuracy, and scene-swap reversal audit.
- `pair_level_pooled_candidate_metrics`: Population mean, median, and pair-cluster 95% bootstrap CIs across the 60 independent pairs.
- `cohort_matched_comparisons`: Stratified comparisons for all 10 `(layer, k)` settings, reporting Candidate (All 60 Pairs), Candidate (Eligible), Norm-Matched (Eligible), Paired Difference, Two-Sided Wilcoxon $p$, Holm-adjusted $p_{\text{adj}}$, and Status.
- `programmatic_table_markdown`: Formatted GitHub Flavored Markdown summary table.
- `machine_readable_claims_ledger`: Structured ledger with formal claims `CLAIM_CIRC_02`, `CLAIM_CIRC_03`, and `CLAIM_CIRC_04`.
