# Phase 4 Data Dictionary: Variables, Metrics, and Physical Units

**Version:** 4.0.0-phase4-mechanistic-resolution  
**Date:** 2026-11-08  
**Repository:** `BrianHadianSTEI23/project-haphaestus` (`fix/interpretability`)

---

## 1. Experiment A: Task-Conditioned Visual Binding Benchmark (`task_conditioned_binding_v3`)

### 1.1 Summary Metrics (`task_conditioned_binding_v3_summary.json`)

| Variable | Type | Range / Units | Definition & Scientific Interpretation |
|---|:---:|:---:|---|
| `a_correct_count` | `int` | $[0, N]$ | Number of dual-visible paired scenes where under Token A, inferred goal is closer to Target A than Target B. |
| `a_correct_rate` | `float` | $[0.0, 1.0]$ | Marginal probability $P(\text{Token A steers toward Target A})$. Chance level is 0.50. |
| `a_correct_binom_p_vs_chance` | `float` | $[0.0, 1.0]$ | Two-sided exact binomial p-value testing $H_0: p = 0.50$. |
| `b_correct_count` | `int` | $[0, N]$ | Number of dual-visible paired scenes where under Token B, inferred goal is closer to Target B than Target A. |
| `b_correct_rate` | `float` | $[0.0, 1.0]$ | Marginal probability $P(\text{Token B steers toward Target B})$. Chance level is 0.50. |
| `b_correct_binom_p_vs_chance` | `float` | $[0.0, 1.0]$ | Two-sided exact binomial p-value testing $H_0: p = 0.50$. |
| `dual_discrimination_count` | `int` | $[0, N]$ | Joint conjunction count: scenes where Token A selects Target A **AND** Token B selects Target B. |
| `dual_discrimination_rate` | `float` | $[0.0, 1.0]$ | Joint probability $P(\text{A-correct} \cap \text{B-correct})$. For an uncoupled binary decision, random chance is $0.5 \times 0.5 = 0.25$ (25.0%). Values $\le 0.25$ indicate absence of dual visual object binding. |
| `dual_discrimination_binom_p_vs_chance` | `float` | $[0.0, 1.0]$ | Exact binomial test against 25.0% chance level ($H_0: p \ge 0.25$). |
| `mean_angular_shift_rad` | `float` | $[0, \pi]$ rad | Mean 2D angular rotation between $g(z_A, e_A)$ and $g(z_A, e_B)$ under bit-for-bit identical visual latent $z_A$. Measures task sensitivity, **NOT** object binding. |
| `mean_interaction_magnitude` | `float` | Meters | Mean Euclidean norm of the second-order interaction vector $I = [g(z_B, e_B) - g(z_B, e_A)] - [g(z_A, e_B) - g(z_A, e_A)]$. |
| `mean_alignment_with_physical_displacement` | `float` | $[-1.0, 1.0]$ | Cosine similarity between interaction vector $I$ and physical target coordinate displacement $\Delta P = P_A - P_B$. |
| `model_1_additive_task_bias_mse` | `float` | $\text{m}^2$ | Held-out test mean squared error of the Additive Model $g(z, e) = a(z) + b(e)$. |
| `model_2_visual_geometry_selection_mse` | `float` | $\text{m}^2$ | Held-out test MSE of the Visual Geometry Selection Model $g(z, e) = \text{Select}(V(z), e)$. |
| `model_3_general_nonlinear_interaction_mse` | `float` | $\text{m}^2$ | Held-out test MSE of the Bilinear / General Nonlinear Interaction Model $g(z, e) = \text{MLP}(z, e)$. |
| `wilcoxon_p_saliency_bias` | `float` | $[0.0, 1.0]$ | Paired Wilcoxon p-value comparing target error when Target A is visually smaller/dimmer vs larger/brighter. |

---

## 2. Experiment B: Internal Mechanistic Circuits (`task_conditioned_circuits_v3`)

### 2.1 Activation & Transfer Metrics (`task_conditioned_circuits_v3_summary.json`)

| Variable | Type | Range / Units | Definition & Scientific Interpretation |
|---|:---:|:---:|---|
| `mean_layer1_transfer_efficiency` | `float` | $[0.0, 1.0]$ | Cosine projection of recipient output shift along donor target direction when replacing Layer 1 post-ReLU activations with donor activations. |
| `mean_layer2_transfer_efficiency` | `float` | $[0.0, 1.0]$ | Cosine projection of recipient output shift along donor target direction when replacing Layer 2 post-ReLU activations with donor activations. |
| `mean_random_control_transfer_efficiency` | `float` | $[0.0, 1.0]$ | Cosine projection when replacing with matched-norm random orthogonal Gaussian vectors. |
| `wilcoxon_p_layer1_vs_random` | `float` | $[0.0, 1.0]$ | Paired Wilcoxon signed-rank test comparing Layer 1 donor patching vs random matched-norm controls. |
| `wilcoxon_p_layer2_vs_random` | `float` | $[0.0, 1.0]$ | Paired Wilcoxon signed-rank test comparing Layer 2 donor patching vs random matched-norm controls. |
| `mean_top16_dims_displacement_shift_m` | `float` | Meters | Mean Euclidean output goal displacement shift when zeroing the 16 most active units in Layer 1. |
| `mean_random16_dims_displacement_shift_m` | `float` | Meters | Mean Euclidean output goal displacement shift when zeroing 16 randomly chosen units in Layer 1. |
| `wilcoxon_p_top16_vs_random16` | `float` | $[0.0, 1.0]$ | Paired Wilcoxon test establishing whether top-16 ablation significantly exceeds random 16D ablation. |

---

## 3. Experiment C: Ego-Motion Memory Mechanism (`ego_memory_mechanism_v3`)

### 3.1 Calibration & Replay Metrics (`ego_memory_mechanism_v3_summary.json`)

| Variable | Type | Range / Units | Definition & Scientific Interpretation |
|---|:---:|:---:|---|
| `translation_mae_m` | `float` | Meters | Mean absolute error of 1-step body-frame translation prediction $[\widehat{\Delta x}_b, \widehat{\Delta y}_b]$ vs physical ground truth. |
| `rotation_mae_rad` | `float` | Radians | Mean absolute error of 1-step body-frame yaw rotation prediction $\widehat{\Delta \theta}$ vs physical ground truth. |
| `zero_delta_static_bias` | `array[float]` | Vector (3D) | Output of `micro_ego_head` when input latent delta is identically zero ($\Delta s = 0$, as in black-frame dropouts). |
| `k1_error_m`, `k5_error_m`, `k10_error_m`, `k20_error_m` | `float` | Meters | Cumulative multi-step displacement tracking error over $K \in \{1, 5, 10, 20\}$ steps. |
| `cond_A_visual_only` | `float` | Meters | Trajectory-mean tracking error during sensory blackout with memory disabled ($g_t = g_{\text{vis}}(s_{\text{black}})$). |
| `cond_B_full_memory` | `float` | Meters | Trajectory-mean tracking error during sensory blackout under full learned memory and fixed 0.65/0.35 blending. |
| `cond_C_frozen_memory` | `float` | Meters | Trajectory-mean tracking error when memory is held constant at the pre-blackout goal vector. |
| `cond_D_zeroed_ego` | `float` | Meters | Trajectory-mean tracking error when ego displacement is clamped to $\mathbf{0}$ during blackout. |
| `cond_E_shuffled_ego` | `float` | Meters | Trajectory-mean tracking error when ego displacement is randomly drawn from donor trajectory steps. |
| `cond_F_matched_donor` | `float` | Meters | Trajectory-mean tracking error when ego displacement is injected from a bearing-matched donor trajectory. |
| `cond_G_mismatched_donor` | `float` | Meters | Trajectory-mean tracking error when ego displacement is injected from an unrelated mismatched donor trajectory. |
| `cond_H_oracle_odometry` | `float` | Meters | Trajectory-mean tracking error when memory is updated with ground-truth physical odometry without black-frame blending decay. |
| `tost_p_value` | `float` | $[0.0, 1.0]$ | Two One-Sided Tests (TOST) p-value assessing equivalence within margin $\pm \delta = \pm 0.05$m. |
| `holm_adjusted_p` | `float` | $[0.0, 1.0]$ | Holm-Bonferroni step-down adjusted p-value across the 5 preregistered confirmatory comparisons. |
