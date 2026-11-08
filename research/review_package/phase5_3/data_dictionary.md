# Phase 5.3 Data Dictionary & Artifact Specification

This document details the structure, coordinates, physical units, and field definitions of all raw data artifacts produced during Project Hephaestus Phase 5.3.

---

## 1. File Manifest & Storage Locations

| Filename | Format | Description | Record Count |
| :--- | :--- | :--- | :--- |
| `task_conditioned_circuits_v5_3_summary.json` | JSON | Aggregate statistics, 10 primary tests, coverage tables, representation stats | 1 document |
| `task_conditioned_circuits_v5_3_pairs.jsonl` | JSON Lines | Pair-level aggregated sufficiency and necessity outcomes | 60 records |
| `task_conditioned_circuits_v5_3_activations.jsonl` | JSON Lines | Internal 64D activations under Task A and Task B, baseline outputs | 120 records |
| `task_conditioned_circuits_v5_3_interventions.jsonl.gz` | Gzipped JSONL | Raw records for every individual causal intervention draw | 447,800 records |
| `task_conditioned_circuits_v5_3_control_audit.jsonl` | JSON Lines | Sampling audit metrics per pair × scene × layer × k × stratum | 8,400 records |
| `task_conditioned_circuits_v5_3_schema.md` | Markdown | Mathematical and structural documentation of all fields | 1 document |

---

## 2. Coordinate Systems & Units

- **Physical Translations (X, Y)**: Expressed in metres ($m$) in the body-relative frame of the AMR.
  - $+X$: Forward facing direction of the robot.
  - $+Y$: Lateral left direction of the robot.
- **Goal Output Vectors**: Formatted as 4D vectors `[x, y, sin(theta), cos(theta)]`, where `[x, y]` is the translational offset in metres, and `[sin(theta), cos(theta)]` represents the unit orientation vector.
- **Angles ($\theta$, bearing)**: Expressed in radians, wrapped into $[-\pi, \pi]$.
- **Activation Vectors**: Dimensionless floating-point values extracted post-ReLU from Layer 1 and Layer 2 (64 dimensions each).
- **Perturbation Norms**: Euclidean $L_2$ norm of the activation difference vector $\|\Delta \mathbf{h}[U]\| = \sqrt{\sum_{i \in U} (h_i^B - h_i^A)^2}$.
- **Relative Error**: Dimensionless ratio $|\|\Delta \mathbf{h}[U]\| - \|\Delta \mathbf{h}[U_{\text{cand}}]\|| / \|\Delta \mathbf{h}[U_{\text{cand}}]\|$.

---

## 3. Detailed Field Definitions

### A. `task_conditioned_circuits_v5_3_summary.json`
- `experiment_name`: Name of benchmark experiment.
- `timestamp_utc`: Execution completion timestamp in ISO 8601 UTC.
- `git_commit`: Execution commit SHA (`7c30728179b2bd04778219f1f2b9746ba9dcedd7`).
- `git_clean`: Boolean, true if repository tree was clean at execution.
- `provenance_hashes`: Cryptographic SHA-256 hashes of checkpoint, scene specs, candidate rankings, and execution script.
- `dataset_composition`: Counts of independent pairs (60), scenes (120), and nominal vs illumination-shifted conditions.
- `representation_statistics`: Full 9-stage profiling (participation ratio, threshold dimensions, lifetime and population sparseness, silent units count).
- `statistical_results`: Primary and secondary statistical tests across layers and $k$ values:
  - `candidate_efficiency_mean`: Mean projected transfer efficiency $\eta_{\text{cand}}$ across 60 pairs.
  - `norm_matched_efficiency_mean`: Mean efficiency $\eta_{\text{norm}}$ of strict norm-matched controls.
  - `dimension_matched_efficiency_mean`: Mean efficiency $\eta_{\text{dim}}$ of unconstrained random controls.
  - `overlap_standardized_efficiency_mean`: Mean efficiency standardized by theoretical hypergeometric overlap weights.
  - `disjoint_efficiency_mean`: Mean efficiency of disjoint controls ($O=0$).
  - `match_coverage_pct`: Fraction of 60 pairs that achieved $\ge 100$ valid norm-matched controls within 5%.
  - `common_support_status`: `"CONFIRMATORY_EVALUATED"` if coverage = 100%, else `"INSUFFICIENT_COMMON_SUPPORT"`.
  - `cand_vs_norm_matched`: Mean paired difference, pair-clustered bootstrap 95% CI, one-sided Wilcoxon p-value, sign-flip permutation p-value (100,000 resamples), Holm-Bonferroni adjusted p-value.
  - `causal_necessity_cand_vs_norm_matched`: Candidate drop mean, norm-matched drop mean, mean drop difference, paired Wilcoxon p-value.
- `mechanistic_conclusion`: Summary verdict, counts of insufficient common support settings and significant tests.

### B. `task_conditioned_circuits_v5_3_pairs.jsonl`
- `pair_id`: Unique string identifier (`PAIR_000` to `PAIR_059`).
- `pair_idx`: Integer index (0 to 59).
- `context`: `"nominal"` or `"illumination_shift"`.
- `results`: Dictionary keyed by `"{scene_id}__{layer}__{direction}__k_{k}"`:
  - `candidate_eta`: Projected transfer efficiency of candidate subset.
  - `norm_matched_eta_mean`: Mean transfer efficiency of retained norm-matched controls.
  - `dim_matched_eta_mean`: Mean transfer efficiency of unconstrained random controls.
  - `overlap_standardized_eta`: Mean transfer efficiency weighted across overlap strata.
  - `disjoint_eta`: Mean transfer efficiency of disjoint controls.
  - `cand_norm`: Perturbation norm of candidate subset.
  - `norm_matched_norm_mean`: Mean perturbation norm of retained controls.
  - `norm_matched_status`: `"VALID_COMMON_SUPPORT"` or `"INSUFFICIENT_COMMON_SUPPORT"`.
  - `n_within_5pct`: Number of valid subsets found within 5% tolerance.
  - `total_drawn`: Number of random draws attempted (up to 200,000).
  - `necessity_drop_ratio`: Condition-averaged ablation drop ratio for candidate units ($1 - \|\Delta\tilde{\mathbf{g}}\| / \|\Delta\mathbf{g}^*\|$).
  - `norm_matched_necessity_drop_mean`: Mean condition-averaged drop ratio for retained norm-matched controls.
  - `delta_bearing`: Mean bearing shift in radians under condition-averaged ablation.
  - `delta_heading`: Mean heading angle shift in radians under condition-averaged ablation.

### C. `task_conditioned_circuits_v5_3_activations.jsonl`
- `pair_id`: String identifier of scene pair.
- `scene_id`: `"scene_a"` or `"scene_b"`.
- `context`: `"nominal"` or `"illumination_shift"`.
- `checkpoint_sha256`: SHA-256 hash of evaluated checkpoint.
- `ranking_version`: SHA-256 hash of frozen candidate unit rankings.
- `l1_act_task_a`: Layer 1 post-ReLU 64D activation vector under Task A.
- `l1_act_task_b`: Layer 1 post-ReLU 64D activation vector under Task B.
- `l2_act_task_a`: Layer 2 post-ReLU 64D activation vector under Task A.
- `l2_act_task_b`: Layer 2 post-ReLU 64D activation vector under Task B.
- `output_task_a`: Baseline 4D goal output under Task A.
- `output_task_b`: Baseline 4D goal output under Task B.
- `gt_goal_a`: Ground-truth 2D body-relative target vector for Task A.
- `gt_goal_b`: Ground-truth 2D body-relative target vector for Task B.

### D. `task_conditioned_circuits_v5_3_control_audit.jsonl`
- `pair_id`: Pair ID.
- `scene_id`: Scene ID.
- `layer`: Layer name (`layer1_post_relu_64d` or `layer2_post_relu_64d`).
- `k`: Subset size (2, 4, 8, 16, 32).
- `stratum`: Stratum identifier (`"general_norm_matched"`, `"overlap_{o}"`, `"disjoint_o_0"`).
- `random_seed`: Deterministic integer seed used for this setting.
- `attempted_draws`: Total random draws executed (up to 200,000).
- `valid_norm_matches`: Total draws with relative error $\le 0.05$.
- `retained_count`: Controls retained (100 if valid, 0 if insufficient).
- `min_rel_error`: Minimum relative error observed across all attempted draws.
- `median_rel_error`: Median relative error across sampled draws.
- `max_rel_error`: Maximum relative error observed.
- `status`: `"VALID_COMMON_SUPPORT"` or `"INSUFFICIENT_COMMON_SUPPORT"`.
- `eligible_confirmatory`: Boolean, true if setting satisfies strict common support.
