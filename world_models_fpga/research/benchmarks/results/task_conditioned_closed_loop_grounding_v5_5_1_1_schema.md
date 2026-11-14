# Phase 5.5.1.1 — H1 Correction, Bias Audit & Geometry Schema

## Generated Artifacts
1. `task_conditioned_closed_loop_grounding_v5_5_1_1_H1_reanalysis_summary.json`
   - `raw_audit`: Verification counts and input/output SHA-256 hashes over all 60 scenes, 420 pairs, 840 episodes, and 39 JSONL trace shards.
   - `cohort_category_x_side_balance`: Exact counts of `(target_category, assigned_side)` across the 60 scenes.
   - `H1_reanalysis_by_condition`: Symmetric Endpoint Families A (reaching + first-hit), B (collision-free `assigned_success_symmetric` vs `unassigned_success_symmetric`), C (complete 4-category `N=60` crossover matrix + 3x3 Stuart-Maxwell + conditional exploratory `3 vs 13` binomial), and D (task-switch direction).
   - `left_right_bias_by_condition`: Exact L/R McNemar tests, paired risk differences, nominal vs illumination stratification, category x side tables, Mantel-Haenszel stratification, and unadjusted vs category+context-adjusted logistic regression.
2. `task_conditioned_closed_loop_grounding_v5_5_1_1_scene_pair_matrix.csv`
   - 420 rows (`7` conditions x `60` scenes) recording Assignment A/B endpoints, `crossover_4way`, `crossover_subpattern`, physical object hits, matched L/R successes, and task-switch projections.
3. `task_conditioned_closed_loop_grounding_v5_5_1_1_category_side_table.csv`
   - 84 rows (`7` conditions x `6` categories x `2` sides) with exact episode counts, success counts, Wilson 95% CIs, first-hit counts, and collisions.
4. `task_conditioned_closed_loop_grounding_v5_5_1_1_episode_effects.csv`
   - 840 rows (`7` conditions x `60` scenes x `2` assignments) containing symmetric success, symmetric reaching, first-hit indicators, and early-termination censoring flags.
5. `task_conditioned_closed_loop_grounding_v5_5_1_1_geometry_audit.json`
   - MuJoCo `front_cam` extrinsics, initial-frame SHA-256 verification, segmentation pixel areas and `[u, v]` centroids for all 60 scenes, and side/category geometric feature distributions.
