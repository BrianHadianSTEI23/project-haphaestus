# Phase 5.6 Data Dictionary

## 1. [task_conditioned_stagewise_binding_v5_6_coordinate_audit.json](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_stagewise_binding_v5_6_coordinate_audit.json)
- `synthetic_placements`: `4` canonical single-object coordinate tests (`pure_body_left`, `pure_body_right`, `forward_left`, `forward_right`).
- `rotated_robot_fixed_body_left`: `4` heading invariance tests (`0, +pi/2, pi, -pi/2`).
- `mirrored_robot_single_object`: anti-symmetric robot displacement check around fixed world object.
- `swapped_category_identical_geometry`: `red_box` vs `blue_cylinder` body-frame invariance.
- `simulator_action_signs`: world displacement / yaw check for `+vy` and `+wz` commands.
- `controller_forced_goal_symmetry`: `predict_reactive_action` and `HJEPAAgent(..., track="waypoint")` responses to explicit left/right body goals.
- `training_dataset_goal_distribution`: read-only summary of `splits_v4/train_benchmark.npz` `(bx, by)` across all relations and `"approach"`-only transitions per target category.

## 2. [task_conditioned_stagewise_binding_v5_6_token_sweep.jsonl](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_stagewise_binding_v5_6_token_sweep.jsonl)
`2,940` JSON lines (`60` scenes x `49` canonical tokens):
- `scene_idx`, `scene_id`, `context`, `category_pair`, `obj_1_key`, `obj_2_key`, `obj_1_side`: Cohort scene identifiers.
- `token_id`, `token_target_key`, `token_relation`, `is_canonical_approach`: Vocabulary entry tested.
- `matches_obj_1`, `matches_obj_2`: Whether `token_id` is `"approach"` of `obj_1` or `obj_2`.
- `pred_dx_b`, `pred_dy_b`, `pred_sin_dth`, `pred_cos_dth`, `pred_norm_m`, `pred_bearing_rad`, `pred_side`: Fixed-image `grounding_head` output vector.
- `dist_to_wp1_m`, `dist_to_wp2_m`, `dist_to_obj1_m`, `dist_to_obj2_m`: Euclidean distance (m) from predicted `[dx_b, dy_b]` to scene waypoints/objects.
- `reactive_vx`, `reactive_vy`, `reactive_wz`: `model.predict_reactive_action(z_t, goal_vec)`.

## 3. [task_conditioned_stagewise_binding_v5_6_probe_metrics.csv](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_stagewise_binding_v5_6_probe_metrics.csv)
`546` CSV rows (`E1` linear decodability only; strictly leak-free splits):
- `split_scheme`: `GroupKFold_scene`, `CategoryPairHoldout_5fold`, `TargetCategoryHoldout_strict_no_scene_overlap`, `Context_Nom_to_Illum`, `Context_Illum_to_Nom`.
- `subset`: `all`, `assigned_left`, `assigned_right`, `context_nominal`, `context_illum`.
- `representation`: `z_t`, `e_t`, `h0_cat`, `L1`, `L2`, `pred_xy`.
- `target`: `obj_1_xy`, `obj_2_xy`, `left_obj_xy`, `right_obj_xy`, `assigned_wp_xy`, `unassigned_wp_xy`, `rel_dir_xy`.
- Metrics: `vector_mae_m`, `baseline_vector_mae_m`, `delta_mae_vs_baseline_m`, `delta_mae_ci_low/high`, `eucl_err_m`, `eucl_err_ci_low/high`, `angular_err_deg`, `r2_x`, `r2_y`, `r2_mean`, `cal_slope_x/y`, `cal_intercept_x/y`.

## 4. [task_conditioned_stagewise_binding_v5_6_intervention_pair_metrics.jsonl](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_stagewise_binding_v5_6_intervention_pair_metrics.jsonl)
`2,040` JSON lines (`34` intervention conditions x `60` scenes):
- `family`, `intervention`, `scene_id`, `context`, `donor_scene_id`: Condition identifiers (`donor_scene_id` includes `:A` or `:B` slot for same-target opposite-side donors, or `"random"`, `"zero"`, `"mean"`, `"tok_<id>"`).
- `act_delta_norm`, `ref_act_delta_norm`, `rel_norm_diff`: Activation perturbation magnitude and relative matching error (all matched random controls satisfy `rel_norm_diff <= 0.05`, observed max `1.16e-7`).
- `baseline`, `patched`, `true_contrast`, `contrast_norm`, `donor_target_gain`, `recipient_target_change`, `eta_gt`, `cos_gt`, `parallel`, `orth`, `norm`: Standard single-target and vector-decomposition intervention metrics.
