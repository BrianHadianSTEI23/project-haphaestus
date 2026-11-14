# Project Hephaestus — Phase 5.5.1 Pre-Registered Experimental Protocol

**Pre-registration stage:** Authored and committed prior to generating the Phase 5.5.1 scene cohort or running any Phase 5.5.1 closed-loop evaluations.
**Branch:** `fix/interpretability`
**Checkpoint (frozen, no retraining):** `world_models_fpga/checkpoints_v4/hjepa_best.pt`
**Reference code:** `world_models_fpga/research/benchmarks/run_task_conditioned_closed_loop_grounding_v5_5_1.py`

## 1. Scientific Motivation and Audit Fixes Carried Forward
Phase 5.5 observed high overall success on a 60-scene cohort, but every scene had close target–distractor waypoint spacing (`~0.276–0.378 m`) with overlapping `0.35 m` success bubbles, evaluated only a single task assignment per scene, and conflated frozen visual goals with frozen blended goals (`C2`). In parallel, forensic audit of Phase 5.5 (`Part A2`) requires explicit `NaN`/`null` representation for unexecuted branches and unambiguous separation of Track B structured oracle waypoints (`C6_track_b_oracle_waypoint_upper_bound`) from Track C planner-goal overrides (`C7_track_c_planner_goal_override` / `O1_track_c_planner_goal_override`).

Phase 5.5.1 separates physical object selection, repeated visual feedback, and controller ego-memory on an independent, geometry-separated scene cohort with mirrored dual task assignments.

## 2. Cohort Construction & Acceptance Criteria (`N = 60` Scene Pairs)
- **Context stratification:** `30` `nominal` illumination (`light_diffuse_scale = 1.0`) + `30` `illumination_shift` (`light_diffuse_scale` outside `[0.85, 1.15]`, sampled in `[0.45, 0.72]` or `[1.25, 1.55]`).
- **Counterbalanced category pairs & orientations:** All `15` unordered object-category pairs from `TARGET_KEYS` (`red_box`, `blue_cylinder`, `green_cylinder`, `yellow_cylinder`, `purple_box`, `cyan_cylinder`) appear twice per context (`30` scenes per context, `60` total), balanced across `Assignment A` vs `Assignment B` and across Left (`bearing > 0`) vs Right (`bearing < 0`) placement relative to initial robot heading.
- **Strict non-overlapping geometry & reachability:**
  1. `||wp_1 - wp_2|| >= 0.90 m` (so `0.35 m` reach circles around `wp_1` and `wp_2` have `>= 0.20 m` gap and zero overlap).
  2. Initial distance to each candidate waypoint lies strictly in `[0.60, 1.20] m`.
  3. Object standoff from walls (`[-2.05, 2.05] m`) and clearance from other canonical obstacles (`>= 0.50 m` from objects, `>= 0.45 m` from waypoints and corridor).
  4. Both candidate objects clearly visible in initial unmasked RGB (`>= 60` pixels each).
  5. `OracleAgent` (Track A privileged navigator) solves BOTH `Assignment A` (`target = obj_1`, `wp_1`) and `Assignment B` (`target = obj_2`, `wp_2`) within `45` steps collision-free inside `<= 0.35 m`.
- **Rejection logging:** Every rejected candidate and its primary rejection reason are counted and persisted to `task_conditioned_closed_loop_grounding_v5_5_1_cohort_report.json`.

## 3. Experimental Design (`60 scenes x 2 assignments x 7 conditions = 840 episodes`)
Every accepted scene is evaluated under BOTH task assignments (`A`: target=`obj_1`, distractor=`obj_2`; `B`: target=`obj_2`, distractor=`obj_1`) across 7 conditions (`max_steps = 45`, `goal_reach_threshold = 0.35 m`, frozen model weights, `track="observation"` throughout):

| Condition ID | Visual Goal Branch | Controller Ego-Memory Branch | RGB / Task Intervention | Planner Goal Input |
| :--- | :--- | :--- | :--- | :--- |
| `F1_dynamic_visual_with_memory` | Dynamic every step `t` | Enabled (`0.65*vis + 0.35*mem`) | None | `g_blended_t` |
| `F2_dynamic_visual_no_memory` | Dynamic every step `t` | Disabled (`disable_memory=True`) | None | `g_visual_t` |
| `F3_frozen_visual_with_memory` | Anchored to `t=0` world projection of `g_visual_0` | Enabled (`0.65*vis_frozen_t + 0.35*mem`) | None | `g_blended_t` |
| `F4_frozen_visual_no_memory` | Anchored to `t=0` world projection of `g_visual_0` | Disabled (`disable_memory=True`) | None | `g_visual_frozen_t` |
| `M1_target_masked_rgb` | Dynamic every step `t` | Enabled | Assigned target geom locally background-masked | `g_blended_t` |
| `M2_distractor_masked_rgb` | Dynamic every step `t` | Enabled | Area-matched unassigned geom locally masked | `g_blended_t` |
| `O1_track_c_planner_goal_override` | Dynamic every step `t` (executed) | Enabled (updates internal `_ego_goal_memory`) | None on latent/grounding/memory | Overridden by ground-truth `gt_assigned_t` |

## 4. Primary Endpoints & Scene-Pair Unit of Analysis
To respect physical scene pairing and task-assignment dependence, confirmatory inference uses the **physical scene pair (`N = 60` independent units)**, aggregating Assignment A and B per scene before pairing across conditions, with paired Wilcoxon signed-rank tests (or exact McNemar for binary scene-pair indicators) and `4000`-resample scene-pair-clustered bootstrap `95%` confidence intervals (`seed = 5510`).

Per-episode endpoints:
- `assigned_success`: `(not initially_inside_radius) and assigned_reached and (not collision) and final_assigned_goal_error_m <= 0.35`
- `assigned_reached` and `unassigned_reached` (each within `0.35 m` of its respective waypoint)
- `first_reached_object` in `{"assigned", "unassigned", "neither"}`
- `final_assigned_goal_error_m`, `final_unassigned_goal_error_m`, `final_margin_unassigned_minus_assigned_m`
- Signed projection of `g_visual` and `g_blended` onto the `(wp_assigned - wp_unassigned)` axis relative to midpoint, and scene-pair task-switch contrast projection `task_switch_eta` along `(wp_A - wp_B)`.

Scene-pair (`N = 60`) aggregated endpoints:
- `pair_mean_assigned_success` in `{0.0, 0.5, 1.0}`
- `dual_assigned_success` in `{False, True}` (both A and B succeed on assigned target)
- `dual_first_correct_selection` in `{False, True}` (both A and B reach assigned target first)
- `same_object_collapse` in `{False, True}` (one assignment reaches assigned first and the other reaches unassigned first, i.e., both converge on the same physical object)
- `pair_selectivity_margin` in `[-1.0, 1.0]` (`0.5 * ((assigned_A - unassigned_A) + (assigned_B - unassigned_B))`)

## 5. Pre-Registered Confirmatory Hypothesis Families (Holm–Bonferroni Controlled)
Across the **5 confirmatory families**, Holm–Bonferroni correction is applied across the 5 family-primary p-values (`alpha = 0.05` two-sided), and where a family contains multiple planned contrasts, within-family Holm adjustment is also reported:
- **H1 (Task-conditioned physical selection in F1):** Pair-level `pair_mean_assigned_success` vs `pair_mean_unassigned_reached` (`Wilcoxon`) + exact binomial/McNemar comparing `dual_first_correct_selection` vs `same_object_collapse`, and `task_switch_eta_visual` vs `0`.
- **H2 (Visual feedback main effect in 2x2 factorial):** `[0.5*(F1 + F2)] - [0.5*(F3 + F4)]` on `pair_mean_assigned_success` (paired Wilcoxon + pair-clustered bootstrap CI), plus simple effects `F1 - F3` and `F2 - F4`.
- **H3 (Controller ego-memory main effect in 2x2 factorial):** `[0.5*(F1 + F3)] - [0.5*(F2 + F4)]` on `pair_mean_assigned_success` (paired Wilcoxon + pair-clustered bootstrap CI), interaction `(F1 - F2) - (F3 - F4)`, plus simple effects `F1 - F2` and `F3 - F4`.
- **H4 (Target vs distractor visual masking specificity):** `M1_target_masked_rgb - M2_distractor_masked_rgb` (primary contrast for H4) on `pair_mean_assigned_success`, plus Holm-adjusted within-family comparisons `M1 - F1` and `M2 - F1`.
- **H5 (Planner/controller attribution via Track C planner-goal override):** `O1_track_c_planner_goal_override - F1_dynamic_visual_with_memory` on `pair_mean_assigned_success` (and `dual_assigned_success`).

## 6. Execution & Reporting Rules
- Zero synthetic or placeholder data; missing/unexecuted fields (e.g., `g_memory_t` and `delta_ego_t` in `F2`/`F4` or `t=0`) are stored explicitly as `NaN` in `.npz` and `null` in `.jsonl`.
- Full traces saved to `.npz` AND plain-text `.jsonl` shards `< 5 MB` each with SHA-256 `shard_manifest.json`.
- Non-significant comparisons are reported neutrally without claiming equivalence. Causal/mechanistic interpretation is deferred to the independent researcher.
