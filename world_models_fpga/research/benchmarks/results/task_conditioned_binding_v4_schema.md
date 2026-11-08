# Schema: Phase 5 Task-Conditioned Binding & Model Comparison Benchmark (v4)

**Artifact Paths:**
- `world_models_fpga/research/benchmarks/results/task_conditioned_binding_v4_summary.json`
- `world_models_fpga/research/benchmarks/results/task_conditioned_binding_v4_pairs.jsonl`
- `world_models_fpga/research/benchmarks/results/task_conditioned_binding_v4_model_comparison.csv`

---

## 1. Summary JSON Specification (`task_conditioned_binding_v4_summary.json`)

| JSON Pointer | Data Type | Physical Units | Description |
|---|---|---|---|
| `/experiment_metadata/git_commit` | `string` | Commit SHA | Git commit HEAD during execution. |
| `/experiment_metadata/checkpoint_sha256` | `string` | SHA-256 | Exact hash of model checkpoint (`hjepa_best.pt`). |
| `/sample_counts/n_categorical_pairs` | `int` | Count | Number of independent categorical scene pairs ($N=50$). |
| `/sample_counts/n_categorical_scenes` | `int` | Count | Number of physical scenes evaluated ($N=100$). |
| `/scene_level_discrimination/scene_a/dual_discrimination_rate` | `float` | Fraction $[0, 1]$ | Rate where both Token A and Token B select their targets in Scene A. |
| `/scene_level_discrimination/scene_b/dual_discrimination_rate` | `float` | Fraction $[0, 1]$ | Rate where both Token A and Token B select their targets in Scene B. |
| `/scene_level_discrimination/pooled_pair_level/mean_dual_discrimination_rate` | `float` | Fraction $[0, 1]$ | Average dual discrimination rate across all 100 scenes (16.0%). |
| `/scene_level_discrimination/pooled_pair_level/both_scenes_dual_rate` | `float` | Fraction $[0, 1]$ | Rate of pairs where dual discrimination succeeded in *both* scenes (0.0%). |
| `/permutation_null_testing/paired_permutation_p_value_greater` | `float` | $p$-value | Paired permutation test $p$-value against empirical chance null ($p=0.6200$). |
| `/interaction_vector_analysis/mean_interaction_magnitude_m` | `float` | Meters | Mean magnitude $\|I\| = \|(g_{bb} - g_{ba}) - (g_{ab} - g_{aa})\|$ ($0.352$m). |
| `/interaction_vector_analysis/mean_cosine_alignment_with_swap` | `float` | Cosine $[-1, 1]$ | Cosine alignment between $I$ and physical target swap $\Delta P$ ($-0.314$). |
| `/model_comparison_grouped_5fold_cv/comp1_explain_hjepa_output` | `object` | Various | Held-out 5-fold CV metrics for surrogates predicting H-JEPA outputs. |
| `/model_comparison_grouped_5fold_cv/comp2_ground_truth_target` | `object` | Various | Held-out 5-fold CV metrics for models predicting Ground-Truth targets. |
| `/instance_level_binding_experiment/both_scenes_dual_discrimination_rate` | `float` | Fraction $[0, 1]$ | Dual discrimination rate on same-category instances ($0.0\%$). |

---

## 2. Pairs JSONL Specification (`task_conditioned_binding_v4_pairs.jsonl`)

Each line is a valid JSON object.
- **Categorical Pairs (Lines 1–50):**
  - `pair_id`: Unique index $[0, 49]$.
  - `scene_a_evaluation`: Dictionary with `a_correct`, `b_correct`, `dual_discrimination`, margins, Euclidean errors, and angular errors.
  - `scene_b_evaluation`: Corresponding evaluation for Scene B.
  - `pair_dual_discrimination`: Boolean, true iff both Scene A and Scene B achieved dual discrimination.
  - `interaction_analysis`: Interaction vector, magnitude, target swap displacement, cosine alignment, signed projection, angular difference.
- **Instance Binding Pairs (Lines 51–70):**
  - `instance_binding_record`: Contains category, pair keys, scene evaluations, and dual status.

---

## 3. Model Comparison CSV Specification (`task_conditioned_binding_v4_model_comparison.csv`)

| Column | Data Type | Description |
|---|---|---|
| `comparison` | `string` | Either `comp1_explain_hjepa_output` or `comp2_ground_truth_target`. |
| `model_family` | `string` | Model type (`Task-Only`, `Visual-Only`, `Additive`, `Bilinear`, `Nonlinear MLP`, `Visual Baseline (Mean)`, `Oracle Baseline`). |
| `held_out_mse_mean` | `float` | Mean held-out MSE across 5 grouped folds. |
| `held_out_mse_std` | `float` | Standard deviation of held-out MSE across folds. |
| `held_out_mae_mean` | `float` | Mean held-out MAE (meters). |
| `held_out_mae_std` | `float` | Standard deviation of held-out MAE. |
| `held_out_ang_deg_mean` | `float` | Mean held-out angular error (degrees). |
| `held_out_ang_deg_std` | `float` | Standard deviation of angular error. |
