# Phase 5.2 Data Dictionary & Coordinate Frame Standards

**Repository:** `BrianHadianSTEI23/project-haphaestus`  
**Branch:** `fix/interpretability`  
**Commit:** `bc568052e8be10fba4f4a466e913e3dbd8bb4fd6`  
**Role:** Experimentalist and implementation engineer  
**Auditor:** ChatGPT (Senior Independent Scientific Researcher)  

---

## 1. Physical Units & Coordinate System Standards

### 1.1 Coordinate Reference Frames
1. **World Frame (NWU - North-West-Up):**
   - Origin: Arena geometric center $(0, 0, 0)$.
   - $X$-axis: Positive forward (North), range $[-1.55, +1.55]$ m.
   - $Y$-axis: Positive left (West), range $[-1.55, +1.55]$ m.
   - $Z$-axis: Positive vertical (Up), arena floor at $Z = 0$.
   - Robot orientation: Yaw angle $\theta \in [-\pi, +\pi]$ measured counterclockwise from positive $X$-axis.

2. **Robot Body Frame (FLU - Forward-Left-Up):**
   - Origin: Center of differential-drive wheel axle.
   - $X_b$: Forward along heading direction.
   - $Y_b$: Lateral left perpendicular to heading.
   - $Z_b$: Upward along gravity normal.
   - Transformation from World $(x_w, y_w)$ to Body $(x_b, y_b)$:
     $$\begin{bmatrix} x_b \\ y_b \end{bmatrix} = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x_w - x_{\text{robot}} \\ y_w - y_{\text{robot}} \end{bmatrix}$$

### 1.2 Physical Unit Separation Invariant
- **Translational Variables:** Strictly measured in **meters (m)**.
- **Rotational Variables:** Strictly measured in **radians (rad)** or explicitly labeled **degrees ($^\circ$)**.
- **Prohibition:** Unnormalized Euclidean addition of translational and rotational quantities (e.g. $\sqrt{\Delta x^2 + \Delta y^2 + \Delta \theta^2}$) is strictly prohibited.

---

## 2. Artifact Schemas

### 2.1 `task_conditioned_circuits_v5_2_summary.json`
- `dataset_composition`:
  - `total_pairs`: Integer, 60.
  - `nominal_pairs`: Integer, 30.
  - `illumination_shift_pairs`: Integer, 30.
  - `rejection_counters`: Dictionary recording reasons for scene candidate rejections.
- `frozen_candidate_rankings`:
  - `layer1_post_relu_64d`: List of 64 integers, top units ranked by both-context sensitivity.
  - `layer2_post_relu_64d`: List of 64 integers, top units ranked by both-context sensitivity.
- `representation_statistics`:
  - Dictionary across all 9 intermediate network stages.
  - `participation_ratio`: Float, effective dimensionality $\frac{(\sum \lambda_i)^2}{\sum \lambda_i^2}$.
  - `dims_80_var`, `dims_90_var`, `dims_95_var`, `dims_99_var`: Integers, threshold dimension counts.
  - `silent_units_count`: Integer, units with variance $< 10^{-6}$.
  - `is_non_negative`: Boolean, true for post-ReLU stages.
  - Post-ReLU: `lifetime_sparsity` (mean, std), `population_sparsity` (mean, std), `zero_fraction`.
  - Signed stages: `positive_part_lifetime_sparsity`, `positive_part_population_sparsity`, `excess_kurtosis`, `negative_fraction`.
- `statistical_results`:
  - Nested dictionary: `[layer][direction][k_val]`.
  - `candidate_efficiency_mean`: Float, mean $\eta$ for candidate units.
  - `norm_matched_random_efficiency_mean`: Float, mean $\eta$ across 100 norm-matched random controls.
  - `dimension_matched_random_efficiency_mean`: Float, mean $\eta$ across 100 unconstrained random controls.
  - `cand_vs_norm_matched`:
    - `mean_paired_difference`: Float, $\Delta_{\text{norm}} = \eta_{\text{cand}} - \eta_{\text{norm\_rand}}$.
    - `bootstrap_95_ci`: List of 2 floats, pair-clustered 95% CI.
    - `wilcoxon_one_sided_p_value`: Float.
    - `permutation_sign_flip_p_value`: Float (100,000 resamples).
    - `holm_bonferroni_adjusted_p_value`: Float, family-wise corrected p-value across 20 tests.
  - `cand_vs_dimension_matched`:
    - `mean_paired_difference`: Float, $\Delta_{\text{dim}} = \eta_{\text{cand}} - \eta_{\text{dim\_rand}}$.
    - `wilcoxon_one_sided_p_value`: Float.

### 2.2 `task_conditioned_circuits_v5_2_interventions.jsonl`
Line-delimited JSON with 482,400 records. Each record represents one intervention pass:
- `pair_id`: String, e.g. `"PAIR_000"`.
- `scene_id`: String, `"scene_a"` or `"scene_b"`.
- `context`: String, `"nominal"` or `"illumination_shift"`.
- `layer`: String, `"layer1_post_relu_64d"` or `"layer2_post_relu_64d"`.
- `direction`: String, `"A_to_B"` or `"B_to_A"`.
- `k`: Integer, number of intervened units ($2, 4, 8, 16, 32$).
- `condition`: String, `"candidate"`, `"norm_matched_random"`, or `"dimension_matched_random"`.
- `draw_id`: Integer, $0$ for candidate, $0 \dots 99$ for random draws.
- `units`: List of integers, unit indices intervened.
- `subset_norm`: Float, $\|\Delta h_U\|_2$.
- `norm_diff_abs`: Float, $|N_{\text{rand}} - N_{\text{cand}}|$.
- `norm_diff_rel`: Float, relative error $\frac{|N_{\text{rand}} - N_{\text{cand}}|}{N_{\text{cand}}}$.
- `is_within_tolerance`: Boolean, true if $\le 0.05$.
- `projected_transfer_efficiency`: Float, $\eta$.
- `transfer_magnitude_m`: Float, transferred 2D displacement magnitude in meters.
- `directional_cosine`: Float, cosine alignment with target contrast vector.
- `orthogonal_residual_m`: Float, magnitude of residual orthogonal to target vector.
- `heading_transfer_eff`: Float, angular heading transfer ratio derived from $\operatorname{atan2}$.

### 2.3 `ego_memory_reanalysis_v5_2_summary.json`
- `displacement_analysis`:
  - Broken down across `all_transitions`, `clean_transitions_non_blackout`, `blackout_transitions`, and motion regimes (`pure_translation`, `pure_rotation`, `combined`).
  - `zero_baseline`: `translation_mae_m`, `translation_rmse_m`, `rotation_mae_rad`.
  - `learned_sensor_displacement`: Full metrics including scalar regression $R^2$ and 2D vector coordinate $R^2$.
  - `action_conditioned_displacement`: Full metrics including scalar regression $R^2$ and 2D vector coordinate $R^2$.
- `within_trajectory_time_shuffled_control`:
  - `mean_tracking_errors_m`: Mean tracking error in meters across 50 trajectories for all memory policies.
  - `paired_comparisons`: Paired difference, bootstrap 95% CI, Wilcoxon p-value, and sign-flip permutation p-value for:
    - Action Hold vs Within-Trajectory Shuffled Hold
    - Action Hold vs Cross-Trajectory Shuffled Hold
    - Within-Trajectory Shuffled vs Cross-Trajectory Shuffled
