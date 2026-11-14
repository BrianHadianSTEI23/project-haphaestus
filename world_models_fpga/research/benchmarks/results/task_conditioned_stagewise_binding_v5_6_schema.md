# Phase 5.6 Output Schema

- `task_conditioned_stagewise_binding_v5_6_coordinate_audit.json`: Stage B2 synthetic coordinate-frame, camera extrinsics, simulator action sign, forced-goal controller symmetry, and read-only `train_benchmark.npz` goal distribution checks.
- `task_conditioned_stagewise_binding_v5_6_token_sweep.jsonl`: One row per `(scene_id, token_id)` across `60` scenes x `49` canonical vocabulary tokens (`2,940` rows) reporting fixed-image `goal_vec` predictions, lateral bias, distances to `wp_1/wp_2/obj_1/obj_2`, and reactive policy velocity outputs.
- `task_conditioned_stagewise_binding_v5_6_probe_metrics.csv`: E1 linear probe performance across `z_t`, `e_t`, `h0_cat`, `L1`, `L2`, `pred_xy` and `GroupKFold_scene`, `CategoryPairHoldout_5fold`, `TargetCategoryHoldout_strict_no_scene_overlap`, `Context_Nom_to_Illum`, `Context_Illum_to_Nom` splits with bootstrap 95% CIs and mean-baseline deltas.
- `task_conditioned_stagewise_binding_v5_6_intervention_pair_metrics.jsonl`: One row per `(intervention, scene_id)` across Intervention Families 1 (`e_t`), 2 (`z_t` donor swaps), and 3 (`L1`/`L2` full & OOF subspace patching) and their `<= 5%` norm-matched controls.
- `task_conditioned_stagewise_binding_v5_6_summary.json`: Consolidated Phase 5.6 provenance, B2–B5 statistical summaries, Holm–Bonferroni paired contrasts, and Stage B6 rescue gate determination.
