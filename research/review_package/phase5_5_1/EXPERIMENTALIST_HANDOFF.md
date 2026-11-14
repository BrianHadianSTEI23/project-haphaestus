# Experimentalist Handoff — Phase 5.4.2.1 & Phase 5.5.1 (For Independent Scientific Review)

**To:** Independent Scientific Researcher (ChatGPT)  
**From:** Experimentalist / Implementation Engineer  
**Branch:** `fix/interpretability`  
**Immutable Pre-Phase Baseline SHA:** `8fa284c206aaa8f5e9dd3524189652da28840462`  
**Commit 1 (Pre-Registered Protocol, Audit Fixes & Tests):** `b905289`  
**Commit 2 (Raw Execution Artifacts, Shards & Manifests):** `945ac53`  
**Commit 3 (Figures, Review Packages, Test Log & Report Updates):** Pending this commit (HEAD at push)  
**Checkpoint Evaluated (Frozen):** `world_models_fpga/checkpoints_v4/hjepa_best.pt` (`sha256: ac69688eae1f4128719dfce912da47c8ae55cbf2412fc8c10feba53d93f1f442`)  
**MuJoCo Environment XML:** `sha256: 0e655e50a8119de6402a5f041369c4728d7d2d9cf4bd8bc4bf8a9a582653bb51`

---

## 1. Part A1 Summary — Phase 5.4.2.1 Exact Key-Set Reconciliation Audit

- **Manifest:** [`phase5_4_2_1/execution_manifest.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_4_2_1/execution_manifest.json)
- **Raw Inputs Audited (Unmodified):**
  - `world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_3_pairs.jsonl` (`sha256: 5d9a0dc8abad2a6c6a78a762004f383eeb4f94a0aa427217f33d4a9e95e5eead`)
  - `world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_3_interventions.jsonl.gz` (`sha256: d57fee1c39717ddb061626ae63fbe657fe2a1e5fc70d6922bf3214b2b77ece0f`)
- **Result (`EXACT_SET_EQUALITY_VERIFIED`):**
  - Composite key: `(pair_id, scene_id, layer, direction, k, condition, draw_id)`
  - `Expected rows = 447,800`, `Observed rows = 447,800`, `Observed unique keys = 447,800`
  - `Missing = 0`, `Unexpected = 0`, `Duplicates = 0`, `Unmatched pairs = 0`, `Invalid combos = 0`
  - Per-condition match: `candidate = 2,400 / 2,400`, `dimension_matched_random = 240,000 / 240,000`, `norm_matched_random = 205,400 / 205,400` (`draw_id in 0..99` exclusively when `norm_matched_status == "VALID_COMMON_SUPPORT"`), all `exact_set_equality = true`.

---

## 2. Part A2 Summary — Shared Harness & Phase 5.5 Code Audit Fixes

- **Wrapper & Provenance (`world_models_fpga/research/benchmarks/model_wrappers.py`):** Added `"planner_goal"` interception applied strictly post-`_resolve_goal_vec` (preserving `_ego_goal_memory`), `disable_memory` explicit flag, and step/episode provenance booleans/strings (`task_embedding_executed`, `grounding_head_executed`, `memory_branch_executed`, `planner_goal_source`, `planner_goal_t`).
- **Phase 5.5 Harness Audit (`run_task_conditioned_closed_loop_grounding_v5_5.py`):** Separated Track B structured oracle waypoints (`C6_track_b_oracle_waypoint_upper_bound`) from Track C planner-goal overrides (`C7_track_c_planner_goal_override`), and replaced zero/oracle fallback substitutions with explicit `np.nan` / `None`. Historical Phase 5.5 raw result files were left untouched.

---

## 3. Part B Summary — Phase 5.5.1 Geometry-Separated Closed-Loop Benchmark

### 3.1 Cohort & Geometry Separation (`cohort_report.json`)
- **Cohort File:** `world_models_fpga/research/benchmarks/results/task_conditioned_closed_loop_grounding_v5_5_1_scenes.jsonl` (`sha256: 2cf3f1c2e32239a9a5b31b516da6f6af514829869684a4623ae2a8017a62ba4e`)
- **Cohort Report:** [`task_conditioned_closed_loop_grounding_v5_5_1_cohort_report.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_closed_loop_grounding_v5_5_1_cohort_report.json)
- **Accepted / Evaluated:** **`60`** accepted (`30` `nominal` + `30` `illumination_shift`) out of `755` evaluated (`695` rejected: `waypoint_sep_below_threshold = 346`, `arena_bounds = 253`, `obstacle_clearance = 84`, `oracle_unreachable_B = 8`, `oracle_unreachable_A = 4`).
- **Separation & Visibility Bounds:** `waypoint_separation_m` in **`[0.900183, 1.036776] m`** (`mean = 0.942498 m`), `init_target_dist` in `[0.824470, 1.197321] m`, `min_init_visible_px = 274`.
- **Oracle Expert Solvability:** `Assignment A = 60/60`, `Assignment B = 60/60`.

### 3.2 Condition-Level Outcomes (`60` Scenes x `2` Assignments = `120` Episodes/Condition; `840` Total)

| Condition | Assigned Success `[Wilson 95% CI]` | Unassigned Reached | 1st Reached (`A/U/N`) | Collisions | Mean Err `A` / `U` (m) | Path (m) / Steps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `F1_dynamic_visual_with_memory` | **32/120 (26.7%)** `[19.6%, 35.2%]` | 24/120 (20.0%) | `32 / 24 / 64` | 41/120 (34.2%) | `1.1355 / 1.1245` | `1.1043 / 39.15` |
| `F2_dynamic_visual_no_memory` | **33/120 (27.5%)** `[20.3%, 36.1%]` | 24/120 (20.0%) | `33 / 24 / 63` | 44/120 (36.7%) | `1.1159 / 1.1092` | `1.0983 / 38.85` |
| `F3_frozen_visual_with_memory` | **9/120 (7.5%)** `[4.0%, 13.6%]` | 6/120 (5.0%) | `9 / 6 / 105` | 16/120 (13.3%) | `1.2083 / 1.2096` | `0.7727 / 43.17` |
| `F4_frozen_visual_no_memory` | **9/120 (7.5%)** `[4.0%, 13.6%]` | 5/120 (4.2%) | `9 / 5 / 106` | 16/120 (13.3%) | `1.2064 / 1.2056` | `0.7667 / 43.35` |
| `M1_target_masked_rgb` | **14/120 (11.7%)** `[7.1%, 18.6%]` | 28/120 (23.3%) | `14 / 28 / 78` | 52/120 (43.3%) | `1.2510 / 1.1907` | `1.1530 / 42.55` |
| `M2_distractor_masked_rgb` | **26/120 (21.7%)** `[15.2%, 29.9%]` | 18/120 (15.0%) | `26 / 18 / 76` | 50/120 (41.7%) | `1.1667 / 1.1951` | `1.1120 / 40.33` |
| `O1_track_c_planner_goal_override` | **111/120 (92.5%)** `[86.4%, 96.0%]` | 1/120 (0.8%) | `111 / 1 / 8` | 1/120 (0.8%) | `0.3447 / 0.7847` | `0.8197 / 23.92` |

### 3.3 Scene-Pair Metrics (`N = 60` Scene Pairs) & Stratifications

| Condition | Dual 1st-Correct | Same-Obj Collapse | Pair Selectivity Margin `[95% CI]` | Step-0 Task-Switch `eta_vis` `[95% CI]` | Nominal / Illum | Left / Right |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `F1_dynamic_visual_with_memory` | 3/60 (5.0%) | 13/60 (21.7%) | `+0.0667 [-0.0333, +0.1667]` | `-0.0105 [-0.1033, +0.0847]` | `26.7% / 26.7%` | `8.3% / 45.0%` |
| `F2_dynamic_visual_no_memory` | 3/60 (5.0%) | 12/60 (20.0%) | `+0.0750 [-0.0417, +0.1833]` | `-0.0105 [-0.1034, +0.0853]` | `28.3% / 26.7%` | `8.3% / 46.7%` |
| `F3_frozen_visual_with_memory` | 0/60 (0.0%) | 1/60 (1.7%) | `+0.0250 [-0.0333, +0.0833]` | `-0.0105 [-0.1092, +0.0822]` | `3.3% / 11.7%` | `0.0% / 15.0%` |
| `F4_frozen_visual_no_memory` | 0/60 (0.0%) | 1/60 (1.7%) | `+0.0333 [-0.0250, +0.0917]` | `-0.0105 [-0.1041, +0.0859]` | `5.0% / 10.0%` | `1.7% / 13.3%` |
| `M1_target_masked_rgb` | 0/60 (0.0%) | 5/60 (8.3%) | `-0.1167 [-0.2083, -0.0250]` | `-0.0837 [-0.2514, +0.0775]` | `13.3% / 10.0%` | `3.3% / 20.0%` |
| `M2_distractor_masked_rgb` | 0/60 (0.0%) | 7/60 (11.7%) | `+0.0667 [-0.0250, +0.1500]` | `+0.1132 [+0.0086, +0.2138]` | `20.0% / 23.3%` | `3.3% / 40.0%` |
| `O1_track_c_planner_goal_override` | 51/60 (85.0%) | 1/60 (1.7%) | `+0.9167 [+0.8583, +0.9667]` | `-0.0105 [-0.1062, +0.0819]` | `90.0% / 95.0%` | `86.7% / 98.3%` |

- **F1 Assigned Success by Category (`n = 20` each):** `blue_cylinder = 12/20 (60.0%)`, `green_cylinder = 6/20 (30.0%)`, `cyan_cylinder = 5/20 (25.0%)`, `purple_box = 5/20 (25.0%)`, `red_box = 2/20 (10.0%)`, `yellow_cylinder = 2/20 (10.0%)`.
- **O1 Assigned Success by Category (`n = 20` each):** `blue_cylinder = 19/20 (95.0%)`, `cyan_cylinder = 19/20 (95.0%)`, `purple_box = 19/20 (95.0%)`, `yellow_cylinder = 19/20 (95.0%)`, `green_cylinder = 18/20 (90.0%)`, `red_box = 17/20 (85.0%)`.

### 3.4 Confirmatory 5-Family Holm–Bonferroni Inference (`N = 60` Scene Pairs)

| Family | Contrast | Diff Mean `[95% Pair-Boot CI]` | Raw Wilcoxon $p$ | **Holm5 $p_{\text{adj}}$** | Secondary / Simple Contrasts |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **H1** | `F1`: `assigned - unassigned` | `+0.0667 [-0.0417, +0.1667]` | `2.1687e-01` | `4.3374e-01` | `dual_1st_correct (3)` vs `collapse (13)` binom `p=2.1271e-02`; `step0_eta_vis=-0.0105` (`p=9.4132e-01`) |
| **H2** | `0.5*(F1+F2) - 0.5*(F3+F4)` | `+0.1958 [+0.1292, +0.2667]` | `3.2228e-06` | **`1.2891e-05`** | `F1-F3 = +0.1917` (`p=9.5840e-06`), `F2-F4 = +0.2000` (`p=2.5165e-06`) |
| **H3** | `0.5*(F1+F3) - 0.5*(F2+F4)` | `-0.0042 [-0.0167, +0.0083]` | `5.6370e-01` | `5.6370e-01` | Interaction `-0.0083` (`p=5.6370e-01`); `F1-F2 = -0.0083` (`p=3.1731e-01`); `F3-F4 = 0.0` (`p=1.0`) |
| **H4** | `M1_target - M2_distractor` | `-0.1000 [-0.1583, -0.0333]` | `2.6998e-03` | **`8.0994e-03`** | Within-family Holm `5.3996e-03`; `M1-F1 = -0.1500` (`within_Holm p=7.1569e-04`); `M2-F1 = -0.0500` (`within_Holm p=1.3361e-01`) |
| **H5** | `O1_planner_override - F1` | `+0.6583 [+0.5667, +0.7500]` | `1.3813e-10` | **`6.9065e-10`** | `dual_1st_correct: 51/60 (85.0%) vs 3/60 (5.0%)` |

*(Non-significant comparisons `H1` and `H3` do **not** establish statistical equivalence.)*

---

## 4. Accessible Raw Traces & Visual Inspection Artifacts
- **Plain-Text Trace Shards (`< 5 MB` each):** [`task_conditioned_closed_loop_v5_5_1_trace_export/`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_closed_loop_v5_5_1_trace_export/shard_manifest.json) (`39` JSONL shards, `32,557` step rows, `max_observed_shard_bytes = 4,199,903`).
- **Full Numerical Trace Archive:** [`task_conditioned_closed_loop_v5_5_1_traces.npz`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_closed_loop_v5_5_1_traces.npz) (`32,438,915` bytes).
- **Summary & Condition CSV:** [`task_conditioned_closed_loop_grounding_v5_5_1_summary.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_closed_loop_grounding_v5_5_1_summary.json), [`task_conditioned_closed_loop_grounding_v5_5_1_condition_table.csv`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_closed_loop_grounding_v5_5_1_condition_table.csv).
- **Figures (`--plot-only`):**
  - [`fig1_v5_5_1_selection_and_context_rates.png`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/figures/fig1_v5_5_1_selection_and_context_rates.png)
  - [`fig2_v5_5_1_2x2_feedback_memory_factorial.png`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/figures/fig2_v5_5_1_2x2_feedback_memory_factorial.png)
  - [`fig3_v5_5_1_temporal_dynamics.png`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/figures/fig3_v5_5_1_temporal_dynamics.png)
- **Representative Keyframes (`figures/frames/`):**
  - [`correct_selection_ep__SCENE_000_B__step_0_unmasked.png`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/figures/frames/correct_selection_ep__SCENE_000_B__step_0_unmasked.png) & [`last_obs.png`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/figures/frames/correct_selection_ep__SCENE_000_B__last_obs.png)
  - [`incorrect_selection_ep__SCENE_006_B__step_0_unmasked.png`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/figures/frames/incorrect_selection_ep__SCENE_006_B__step_0_unmasked.png) & [`last_obs.png`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/figures/frames/incorrect_selection_ep__SCENE_006_B__last_obs.png)
  - [`failure_ep__SCENE_000_A__step_0_unmasked.png`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/figures/frames/failure_ep__SCENE_000_A__step_0_unmasked.png) & [`last_obs.png`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/figures/frames/failure_ep__SCENE_000_A__last_obs.png)
  - [`task_switch_comparison_SCENE_026.png`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/figures/frames/task_switch_comparison_SCENE_026.png)

---

## 5. Reproduction Commands

```bash
# Phase 5.4.2.1 exact key-set reconciliation
.venv/bin/python -m world_models_fpga.research.circuits.run_task_conditioned_circuits_v5_4_2_1_reconciliation_audit

# Phase 5.5.1 geometry-separated closed-loop experiment
.venv/bin/python -m world_models_fpga.research.benchmarks.run_task_conditioned_closed_loop_grounding_v5_5_1 --device cuda

# Deterministic figure regeneration from saved artifacts
.venv/bin/python -m world_models_fpga.research.benchmarks.run_task_conditioned_closed_loop_grounding_v5_5_1 --plot-only

# Full regression suite
.venv/bin/pytest world_models_fpga/tests/ -v
```
