# Phase 5.5.1 — Forensic Audit (Including Part A2 Code & Telemetry Corrections)

## 1. Part A2 Audit Fixes Applied to Phase 5.5 Codebase (`b905289`)
While leaving all historical Phase 5.5 raw result files untouched, the shared benchmark harness and Phase 5.5 script (`world_models_fpga/research/benchmarks/model_wrappers.py`, `run_task_conditioned_closed_loop_grounding_v5_5.py`, and `test_task_conditioned_closed_loop_grounding_v5_5.py`) were corrected as follows:
1. **Explicit `NaN`/`None` Representation Instead of Silent Fallbacks:** Replaced fallback zero-array or oracle substitutions with explicit `np.nan` in trace arrays and `None` (`null`) in JSON/JSONL records when a branch does not execute (e.g., `g_memory` and `delta_ego` when `disable_memory=True` or `delta_ego` at `t=0`).
2. **Track-Boundary Nomenclature Separation:** Explicitly separated Track B structured oracle waypoints (`C6_track_b_oracle_waypoint_upper_bound`) from Track C planner-goal overrides (`C7_track_c_planner_goal_override`), and implemented `"planner_goal"` interception inside `model_wrappers.py` **post-`_resolve_goal_vec`** so the upstream visual encoder, task embedding, grounding head, and `_ego_goal_memory` update still execute normally while only the planner goal vector is replaced.
3. **Explicit Execution Provenance Flags:** Every step and episode now logs `task_embedding_executed`, `grounding_head_executed`, `memory_branch_executed`, `planner_goal_source`, and `planner_goal_t`.

## 2. Geometry-Separation Resolution of the Phase 5.5 Overlap Confound
- **Phase 5.5 Baseline Cohort:** All `60` scenes in `PAIR_000`–`PAIR_059` placed target and distractor waypoints within `0.276–0.378 m` (`mean = 0.317 m < 0.35 m` reach threshold), allowing an unselective forward drive to enter the target bubble (`C1 = 85.0%` despite `r(lat, gt_y) = -0.340`).
- **Phase 5.5.1 Cohort Filtering:** `755` candidates evaluated; `695` rejected (`waypoint_sep_below_threshold = 346`, `arena_bounds = 253`, `obstacle_clearance = 84`, `oracle_unreachable_B = 8`, `oracle_unreachable_A = 4`); `60` accepted (`min_waypoint_separation_m = 0.900183 m`, `max = 1.036776 m`, `mean = 0.942498 m`).
- Under non-overlapping `>= 0.90 m` waypoint separation (`0.35 m` bubbles separated by `>= 0.20 m`), `O1_track_c_planner_goal_override` (`Track C`, only planner goal replaced by `gt_assigned_t`) reaches **`111/120 (92.5%)`** with **`51/60 (85.0%)`** dual-assignment first-correct selection, whereas `F1_dynamic_visual_with_memory` reaches **`32/120 (26.7%)`** assigned vs **`24/120 (20.0%)`** unassigned with **`3/60 (5.0%)`** dual-assignment first-correct selection vs **`13/60 (21.7%)`** same-object collapse.

## 3. Side and Object-Category Biases Exposed Under `F1`
- **Target Side Asymmetry:** In `F1`, episodes whose assigned target started on the **right** reached `27/60 (45.0%)` vs **left** `5/60 (8.3%)` (`F2`: `right = 28/60 (46.7%)`, `left = 5/60 (8.3%)`), whereas `O1` achieved `left = 52/60 (86.7%)` and `right = 59/60 (98.3%)`.
- **Target Category Asymmetry:** In `F1`, success by assigned category spanned `blue_cylinder = 12/20 (60.0%)`, `green_cylinder = 6/20 (30.0%)`, `cyan_cylinder = 5/20 (25.0%)`, `purple_box = 5/20 (25.0%)`, `red_box = 2/20 (10.0%)`, `yellow_cylinder = 2/20 (10.0%)`, whereas `O1` spanned `85.0%–95.0%` across all six categories.

## 4. Artifact & Execution Verification
- `run_task_conditioned_closed_loop_grounding_v5_5_1.py` ran in `137.95 s` on `cuda` from clean commit `b905289` (`task_conditioned_closed_loop_grounding_v5_5_1_summary.json`: `git_commit = b9052895f6f752d82326fa6f6683ec9b061ab4bd`, `git_clean = true`).
- Exported all `32,557` step trace rows across `39` plain-text JSONL shards (`max_observed_shard_bytes = 4,199,903 < 5 MB`) with SHA-256 verification in `shard_manifest.json`.
- Figures were regenerated deterministically via `--plot-only` without simulation re-execution.
