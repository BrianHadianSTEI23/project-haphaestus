# Phase 5.5 — Data Dictionary

## 1. Summary JSON (`world_models_fpga/research/benchmarks/results/task_conditioned_closed_loop_grounding_v5_5_summary.json`)
- `physical_solvability_oracle_expert`: Exact Oracle Expert feasibility count and rate (`60/60 = 1.0`) across `PAIR_000`–`PAIR_059`.
- `cohort_static_step0_contrast_baseline`: Step-0 unmasked dual-task visual contrast projection (`mean_step0_contrast_eta_gt`, dimensionless; `mean_step0_contrast_cosine`).
- `conditions.<COND>`: Per-condition metrics (`n_episodes`, `success_count`, `success_rate`, `success_wilson_95ci`, `collision_count`, `redirected_to_distractor_count`, `closer_to_distractor_count`, `mean_final_goal_error_m`, `median_final_goal_error_m`, `mean_path_length_m`, `mean_executed_steps`, `mean_step_blended_goal_ang_err_deg`, `mean_step_vis_goal_ang_err_deg`, `mean_step_action_target_cos`, pixel visibility stats, `total_fallback_mask_steps`, and `by_context.{nominal,illumination_shift}`).
- `planned_primary_comparisons_holm5`: Paired difference metrics (`success_rate_diff` with 95% bootstrap CI, McNemar `n10`/`n01`/2-sided $p$, **Holm-Bonferroni adjusted $p$** over the 5 pre-registered comparisons, plus paired final-error, angular-error, and action-cosine Wilcoxon tests).
- `auxiliary_comparisons.A1_vs_C1_visual_only_vs_full`: Unadjusted paired comparison of visual-only vs full closed-loop.

## 2. Per-Episode JSONL (`task_conditioned_closed_loop_grounding_v5_5_episodes.jsonl`, 420 records)
Each line is one `(condition, pair_id)` trajectory containing:
- Identity/Provenance: `condition`, `pair_id`, `pair_idx`, `context` (`nominal` | `illumination_shift`), `target_key`, `distractor_key`, `obs_0_matches_v5_2_hash`, `obs_unmasked_0_sha256`.
- Outcome: `success`, `goal_reached`, `collision`, `distractor_reached`, `redirected_to_distractor` (reached distractor before ever reaching target and failed target), `closer_to_distractor` (`final_distractor_goal_error_m < final_eval_goal_error_m`), `executed_steps`, `first_reach_step`, `path_length_m`.
- Geometric Errors: `initial_eval_dist_m`, `final_eval_goal_error_m`, `final_distractor_goal_error_m`.
- Diagnostics: `step0_static_contrast_eta_gt` (dimensionless), `step0_static_contrast_cosine`, `step0_blended_goal_ang_err_deg`, `mean_step_blended_goal_ang_err_deg`, `mean_step_vis_goal_ang_err_deg`, `mean_step_action_target_cos`, `mean_target_visible_pixels`, `mean_distractor_visible_pixels`, `mean_masked_pixels`, `fallback_mask_steps`.

## 3. Condition CSV Table (`task_conditioned_closed_loop_grounding_v5_5_condition_table.csv`)
7-row aggregate table across `C1`–`C6`, `A1`.

## 4. Per-Step Traces Archive (`task_conditioned_closed_loop_grounding_v5_5_traces.npz`)
Keyed as `<condition>__<pair_id>__<field>` for all 420 episodes, recording step-by-step arrays:
- `pose` `(T+1, 3)`, `latent_s` `(T, 32)`, `l1_act` `(T, 64)`, `l2_act` `(T, 64)`
- `g_visual` `(T, 3)`, `g_memory` `(T, 3)`, `g_blended` `(T, 3)`, `delta_ego` `(T, 3)`
- `gt_target_body` `(T, 3)`, `gt_distractor_body` `(T, 3)`
- `a_react` `(T, 3)`, `action_exec` `(T, 3)`, `best_macro_idx` `(T,)`, `pred_macro_disp` `(T, 3)`, `pred_micro_disp` `(T, 3)`, `col_risk` `(T,)`
- `dist_target_wp` `(T,)`, `dist_distractor_wp` `(T,)`, `target_vis_px` `(T,)`, `distractor_vis_px` `(T,)`, `masked_px` `(T,)`, `action_target_cos` `(T,)`, `goal_ang_err_rad` `(T,)`, `vis_ang_err_rad` `(T,)`.
