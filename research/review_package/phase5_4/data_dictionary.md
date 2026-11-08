# Phase 5.4 Data Dictionary & Artifact Specification

## Ground-Truth Spatial Grounding & Causal Intervention Reanalysis

This document details the structure, coordinates, physical units, and field definitions of all raw and processed artifacts produced in Project Hephaestus Phase 5.4.

---

## 1. File Manifest & Storage Locations

All raw experimental outputs are stored in `world_models_fpga/research/circuits/results/`:

| Filename | Format | Description | Record Count | File Size |
| :--- | :--- | :--- | :--- | :--- |
| `task_conditioned_circuits_v5_4_ground_truth_summary.json` | JSON | Aggregate statistics, 10 primary tests, coverage tables, context & object-pair stratifications | 1 document | ~54 KB |
| `task_conditioned_circuits_v5_4_scene_metrics.jsonl` | JSON Lines | Full-model body-frame ground-truth targets, predictions, errors, contrast geometry, and scene-swap reversal | 120 records | ~120 KB |
| `task_conditioned_circuits_v5_4_intervention_metrics.jsonl.gz` | Gzipped JSONL | Raw causal intervention records augmented with ground-truth coordinates, donor/recipient errors, and contrast projections | 447,800 records | ~60 MB |
| `task_conditioned_circuits_v5_4_schema.md` | Markdown | Formal schema definition and mathematical formulations | 1 document | ~3 KB |

---

## 2. Coordinate Systems & Units

- **Coordinate System**: Body-relative 2D Cartesian frame ($m$).
  - $+X$: Points directly forward along the robot's heading direction.
  - $+Y$: Points lateral left, perpendicular to the robot's heading.
- **Physical Translations ($x, y$)**: Metres ($m$) from the robot base center.
- **Goal Displacements ($\mathbf{g}_{\text{model}}, \mathbf{g}_{\text{gt}}$)**: 2D vectors $[x, y]$ in metres.
- **Heading Angles ($\theta$, bearing)**: Radians, wrapped to $[-\pi, \pi]$ using `atan2(sin, cos)`.
- **Contrast Projection Efficiency ($\eta$)**: Dimensionless scalar projection.
  - $\eta_{\text{model}}$: Projection onto internal model contrast $\Delta\mathbf{g}_{\text{model}}$.
  - $\eta_{\text{gt}}$: Projection onto true physical target displacement $\Delta\mathbf{g}_{\text{gt}}$ or $\mathbf{d}_{\text{true}}$.
- **Angular Deviation**: Degrees ($^\circ$) between contrast vectors, computed via $\arccos(\text{clip}(\cos, -1, 1))$.

---

## 3. Detailed Field Definitions

### A. `task_conditioned_circuits_v5_4_scene_metrics.jsonl` (120 records)
One record per independent pair ($N=60$) and scene (`Scene_A` and `Scene_B`).
- `pair_id`: Unique identifier of the independent scene pair (`PAIR_000` to `PAIR_059`).
- `scene_id`: Identifier of the scene within the pair (`PAIR_xxx_Scene_A` or `PAIR_xxx_Scene_B`).
- `context`: Environmental visual context (`"nominal"` or `"illumination_shift"`).
- `task_A_id`: Task identifier integer for Task A.
- `task_B_id`: Task identifier integer for Task B.
- `gt_target_A`: $[x, y]$ physical coordinates of Target A in body frame ($m$).
- `gt_target_B`: $[x, y]$ physical coordinates of Target B in body frame ($m$).
- `pred_goal_A`: $[x, y]$ predicted egocentric goal vector under Task A ($m$).
- `pred_goal_B`: $[x, y]$ predicted egocentric goal vector under Task B ($m$).
- `error_A`: Euclidean spatial error $\|\mathbf{g}_{\text{pred}}^A - \mathbf{g}_{\text{gt}}^A\|_2$ ($m$).
- `error_B`: Euclidean spatial error $\|\mathbf{g}_{\text{pred}}^B - \mathbf{g}_{\text{gt}}^B\|_2$ ($m$).
- `mean_error`: Average spatial error $(\text{error}_A + \text{error}_B)/2$ ($m$).
- `delta_gt`: True physical displacement vector $\mathbf{g}_{\text{gt}}^B - \mathbf{g}_{\text{gt}}^A$ ($m$).
- `delta_model`: Internal model contrast displacement vector $\mathbf{g}_{\text{pred}}^B - \mathbf{g}_{\text{pred}}^A$ ($m$).
- `norm_delta_gt`: $L_2$ norm $\|\Delta\mathbf{g}_{\text{gt}}\|_2$ ($m$).
- `norm_delta_model`: $L_2$ norm $\|\Delta\mathbf{g}_{\text{model}}\|_2$ ($m$).
- `eta_gt`: Ground-truth contrast transfer efficiency $\frac{\Delta\mathbf{g}_{\text{model}} \cdot \Delta\mathbf{g}_{\text{gt}}}{\|\Delta\mathbf{g}_{\text{gt}}\|^2}$.
- `cos_align`: Cosine alignment $\frac{\Delta\mathbf{g}_{\text{model}} \cdot \Delta\mathbf{g}_{\text{gt}}}{\|\Delta\mathbf{g}_{\text{model}}\| \|\Delta\mathbf{g}_{\text{gt}}\|}$.
- `ang_dev_deg`: Angular deviation between $\Delta\mathbf{g}_{\text{model}}$ and $\Delta\mathbf{g}_{\text{gt}}$ ($^\circ$).
- `bias_midpoint_model`: Midpoint of predicted outputs $(\mathbf{g}_{\text{pred}}^A + \mathbf{g}_{\text{pred}}^B)/2$ ($m$).
- `bias_midpoint_gt`: Midpoint of true targets $(\mathbf{g}_{\text{gt}}^A + \mathbf{g}_{\text{gt}}^B)/2$ ($m$).
- `heading_error_A_rad`: Absolute circular heading error under Task A (rad).
- `heading_error_B_rad`: Absolute circular heading error under Task B (rad).
- `mean_heading_error_rad`: Average absolute heading error (rad).
- `scene_swap_cos`: Cosine similarity of model contrast reversal $\cos(-\Delta\mathbf{g}_{\text{model}}^A, \Delta\mathbf{g}_{\text{model}}^B)$.
- `scene_swap_ang_deg`: Angular deviation from perfect inversion ($^\circ$).

### B. `task_conditioned_circuits_v5_4_intervention_metrics.jsonl.gz` (447,800 records)
One record per individual causal intervention draw across layers, conditions, and directions.
- `pair_id`: Unique identifier of the independent scene pair (`PAIR_000` to `PAIR_059`).
- `scene_id`: Identifier of the scene within the pair.
- `layer`: Internal network stage (`"layer1_post_relu_64d"` or `"layer2_post_relu_64d"`).
- `k`: Subset size ($k \in \{2, 4, 8, 16, 32\}$).
- `condition`: Control condition (`"candidate"`, `"norm_matched_random"`, or `"dimension_matched_random"`).
- `draw_idx`: Draw index of the control subset ($0$ for candidate; $0..99$ for random controls).
- `direction`: Intervention swap direction (`"A_to_B"` or `"B_to_A"`).
- `donor_task`: Donor task label (`"Task_A"` or `"Task_B"`).
- `recipient_task`: Recipient task label (`"Task_B"` or `"Task_A"`).
- `overlap_count`: Overlap units $|U_{\text{cand}} \cap U_{\text{rand}}|$.
- `overlap_ratio`: Overlap fraction $|U_{\text{cand}} \cap U_{\text{rand}}| / k$.
- `is_within_tolerance`: Boolean indicating whether activation norm error is $\le 0.05$.
- `rel_norm_error`: Relative error $|\|\Delta\mathbf{h}[U]\| - \|\Delta\mathbf{h}[U_{\text{cand}}]\|| / \|\Delta\mathbf{h}[U_{\text{cand}}]\|$.
- `g_base`: Baseline recipient predicted goal vector $[x, y]$ ($m$).
- `g_trans`: Transplanted recipient predicted goal vector $[x, y]$ ($m$).
- `g_donor_pred`: Donor intact predicted goal vector $[x, y]$ ($m$).
- `g_donor_gt`: Physical coordinates of donor true target $[x_{\text{gt}}, y_{\text{gt}}]$ ($m$).
- `g_recip_gt`: Physical coordinates of recipient true target $[x_{\text{gt}}, y_{\text{gt}}]$ ($m$).
- `baseline_error`: Distance from baseline output to recipient physical target $\|\mathbf{g}_{\text{base}} - \mathbf{g}_{\text{recip\_gt}}\|_2$ ($m$).
- `transplant_error`: Distance from transplant output to donor physical target $\|\mathbf{g}_{\text{trans}} - \mathbf{g}_{\text{donor\_gt}}\|_2$ ($m$).
- `error_improvement`: Spatial error reduction $\text{baseline\_error} - \text{transplant\_error}$ ($m$).
- `d_true`: True physical displacement between targets $\mathbf{g}_{\text{donor\_gt}} - \mathbf{g}_{\text{recip\_gt}}$ ($m$).
- `d_intervention`: Output shift induced by transplantation $\mathbf{g}_{\text{trans}} - \mathbf{g}_{\text{base}}$ ($m$).
- `eta_gt`: Ground-truth contrast transfer efficiency $\frac{\mathbf{d}_{\text{intervention}} \cdot \mathbf{d}_{\text{true}}}{\|\mathbf{d}_{\text{true}}\|^2}$.
- `cos_gt`: Cosine alignment $\frac{\mathbf{d}_{\text{intervention}} \cdot \mathbf{d}_{\text{true}}}{\|\mathbf{d}_{\text{intervention}}\| \|\mathbf{d}_{\text{true}}\|}$.
- `orth_res_gt`: Orthogonal residual magnitude $\|\mathbf{d}_{\text{intervention}} - \eta_{\text{gt}} \mathbf{d}_{\text{true}}\|_2$ ($m$).
- `trans_mag`: Intervention shift magnitude $\|\mathbf{d}_{\text{intervention}}\|_2$ ($m$).
- `eta_model`: Internal model contrast efficiency preserved from Phase 5.3.

### C. `task_conditioned_circuits_v5_4_ground_truth_summary.json`
- `reconciliation_audit`: Record counts, duplicate checks (0), unmatched checks (0).
- `full_model_ground_truth_accuracy`: Overall mean/median spatial errors, confidence intervals, contrast alignments, circular heading errors, nominal vs illumination-shifted stratifications, and 24 object-pair stratifications.
- `pair_level_comparisons`: Layer-wise and $k$-wise candidate full cohort, dimension-matched cohort, candidate vs dimension-matched tests, and candidate vs norm-matched eligible cohort tests.
- `claims_ledger_entries`: Updated structured entries for CLAIM_CIRC_02 and CLAIM_CIRC_03.
