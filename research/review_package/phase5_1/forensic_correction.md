# Phase 5.1 Forensic Correction & Experimental Audit Report

**Repository:** `BrianHadianSTEI23/project-haphaestus`  
**Branch:** `fix/interpretability`  
**Execution Commit:** `1b104decca63d95600d3265a1417d6377e4fb7b6` (Clean: True)  
**Date:** 2026-11-08  
**Role:** Experimentalist, implementation engineer, and data collector  

---

## 1. Executive Summary of Forensic Repairs

This document details the complete forensic audit and corrections implemented during Phase 5.1 to resolve the methodological flaws identified by the independent scientific audit of Phase 5:

1. **Simulator Initialization Repair:** Corrected `JetAutoEnv.reset()` to process `options["task_spec"]` and enforced the strict 5-step simulator mutation protocol with explicit runtime assertions on robot pose, obstacle positions, and target visibility.
2. **Invalidation of Same-Category Binding Evidence:** Audited `encode_task_id()` and `TARGET_KEYS`. Formally documented that instance distinction between visually identical objects without semantic labels is architecturally unrepresentable in the current vocabulary. Invalidated the historical 5.0% result.
3. **Corrected Pair-Level Permutation Testing:** Implemented pair-clustered permutation testing (100,000 resamples) preserving the paired factorial structure across Scene A and Scene B. Reported one-sided and two-sided p-values and cluster-bootstrapped 95% CIs.
4. **Disentangled Interaction Vector Alignment:** Re-evaluated sign-flip tests for interaction vector $I$ against physical target swap $\Delta P$, reporting positive, negative, and two-sided directions explicitly.
5. **Separation of Physical Units:** Eliminated the invalid 4D Euclidean norm mixing meters and trigonometric components. Separated translation ($d_{xy}$ in meters) and orientation (wrapped angular difference in radians/degrees).
6. **Corrected Treves-Rolls Sparseness & Spectral Profiling:** Fixed unit-pooling bug in Treves-Rolls sparseness by computing lifetime sparseness across observations per unit, reporting population sparseness separately, and calculating threshold dimensions across all 9 intermediate stages.
7. **Fixed Ego-Motion Action/Ground-Truth Time Alignment:** Corrected transition target from $t-1 \to t$ to the actual transition $t \to t+1$ caused by `action_t`.
8. **Reconciled Memory Policy Interpretations:** Rectified the narrative contradiction regarding fixed blending vs hold/freeze policy, reported true installed waypoint tracking errors, proved Action Hold equivalence to Oracle Hold via TOST ($p < 0.0001$), and saved complete per-step records for all 50 trajectories.

---

## 2. Issue 1: Simulator Initialization & Mutation Interface

### 2.1 The Vulnerability
In Phase 5 v4, `run_task_conditioned_binding_v4.py` and `run_ego_memory_mechanism_v4.py` called:
```python
env.reset(options={"task_spec": spec})
```
However, inspection of `world_models_fpga/env/amr_env.py` revealed that `JetAutoEnv.reset()` only processed `options["initial_pos"]`. It did not process `options["task_spec"]`, nor did it invoke `env.apply_task_spec()`. Consequently:
1. Obstacles remained in their default canonical positions rather than moving to the configured positions in `spec`.
2. Initial robot pose fell back to the randomizer loop in `reset()`.
3. Waypoints calculated from hypothetical task configurations were decoupled from the actual MuJoCo physics state.

### 2.2 The Implementation Fix
1. In `world_models_fpga/env/amr_env.py`:
   - Updated `reset()` to explicitly process `options["task_spec"]` via `self.apply_task_spec()` when provided.
   - Added support for extracting initial robot pose from `options["task_spec"]["initial_pose"]` or `self.active_task_spec`.
   - Added fallback in `apply_task_spec` to map obstacle keys to `self.OBSTACLES[key]["geom"]` if `geom_name` is omitted.
2. In all v5 benchmark scripts, the 5-step initialization procedure is strictly enforced:
   ```python
   # 1. Restore baseline
   env.restore_physical_baseline()
   # 2. Apply complete task spec
   env.apply_task_spec(spec)
   # 3. Reset with exact initial pose
   obs, info = env.reset(options={"initial_pos": (r_x, r_y, r_th)})
   # 4. Runtime assertions
   actual_pose = env.get_robot_pose()
   assert np.allclose(actual_pose, [r_x, r_y, r_th], atol=1e-3)
   geom_id = env.geometry_id(geom_name)
   assert np.allclose(env.model.geom_pos[geom_id][:2], target_pos[:2], atol=1e-3)
   assert check_target_visibility(env, target_key)["is_visible"]
   # 5. Compute body-frame goal from actual simulator pose
   gt_goal_body = compute_body_frame_goal_vec(actual_pose, wp_3d)
   ```
3. Verification: Ran 50 clean trajectories in `run_ego_memory_mechanism_v5.py` with zero assertion failures. Added unit test `test_simulator_initialization_assertions` to `test_ego_memory_mechanism_v5.py`.

---

## 3. Issue 2: Same-Category Instance Binding Invalidation

### 3.1 Architectural Audit of `encode_task_id()`
Inspection of `world_models_fpga/models/baseline_policy.py`:
- `TARGET_KEYS = ("red_box", "blue_cylinder", "green_cylinder", "yellow_cylinder", "purple_box", "cyan_cylinder")` (6 items).
- `SEMANTIC_RELATIONS = ("approach", "behind", "in_front", "left_of", "right_of", "multi_waypoint", "action_probe", "none")` (8 items).
- `encode_task_id(target_key, semantic_relation)` produces an index:
  $$\text{task\_id} = \min(63, t_{\text{idx}} \times 8 + r_{\text{idx}})$$

### 3.2 Flaw in Phase 5 v4
In Phase 5 v4, the "same-category instance binding" test paired `yellow_cylinder` with `cyan_cylinder`. These objects have distinct visual RGB appearances and map to distinct semantic task keys (`yellow_cylinder` $\to$ task 24, `cyan_cylinder` $\to$ task 40). This evaluated object category classification across different colors, not instance-level binding of visually identical objects. Furthermore, the custom obstacles were not properly installed due to the initialization bug.

### 3.3 Resolution
1. **Invalidation:** The historical 5.0% dual discrimination result is formally INVALIDATED and removed from the claims ledger.
2. **Boundary Condition:** The current task embedding table has no input channel or representation to specify "the left cylinder" vs "the right cylinder" when two visually identical cylinders are present in the arena. Testing identical cylinders under the existing task token would introduce an unresolvable semantic ambiguity.
3. This limitation is documented in `task_conditioned_binding_v5_summary.json` as `ARCHITECTURALLY_UNSUPPORTED`.

---

## 4. Issue 3: Time Alignment in Ego-Motion Calibration

### 4.1 The Alignment Bug
In Phase 5 v4, `steps[t]["delta_body_physical"]` was recorded at step $t$ representing the displacement from step $t-1$ to $t$.
However, the offline calibration routine predicted displacement using `steps[t]["action"]` and `steps[t]["latent"]`, comparing it against `steps[t]["delta_body_physical"]`. Thus, the model's forward prediction for transition $t \to t+1$ was being compared against the historical transition $t-1 \to t$.

### 4.2 The Correction
In `run_ego_memory_mechanism_v5.py`:
For each step $t \in [0, n_{\text{steps}}-1]$:
1. Record `curr_pose = env.get_robot_pose()` and `latent_t = model.encode(obs_t)`.
2. Execute commanded action `action_t = env.step(action_t)`.
3. Record `next_pose = env.get_robot_pose()` and compute exact physical displacement:
   $$dx_w = x_{t+1} - x_t, \quad dy_w = y_{t+1} - y_t$$
   $$\begin{bmatrix} dx_b \\ dy_b \end{bmatrix} = \begin{bmatrix} \cos\theta_t & \sin\theta_t \\ -\sin\theta_t & \cos\theta_t \end{bmatrix} \begin{bmatrix} dx_w \\ dy_w \end{bmatrix}$$
   $$d\theta = \text{atan2}(\sin(\theta_{t+1} - \theta_t), \cos(\theta_{t+1} - \theta_t))$$
4. Store `delta_body_phys` as the transition target for `action_t`.
5. Calibration compares:
   - Zero Baseline: $[0, 0, 0]$ vs `delta_body_phys`
   - Sensor Head: $\text{disp}(s_t, s_{t+1})$ vs `delta_body_phys`
   - Action Predictor: $\text{disp}(s_t, \hat{s}_{t+1}(s_t, a_t))$ vs `delta_body_phys`.

### 4.3 Corrected Calibration Results ($N=1,450$ Transitions)
- **Zero Baseline:** Translation MAE = **0.0287 m**, RMSE = 0.0381 m.
- **Learned Sensor Displacement:** Translation MAE = **0.0274 m**, RMSE = 0.0366 m ($R^2 = 0.0143$, slope = 0.2112).
  - Minimal explanatory power; only marginal error reduction over assuming zero motion.
- **Action-Conditioned Forward Predictor:** Translation MAE = **0.0142 m**, RMSE = 0.0195 m ($R^2 = 0.2549$, slope = 0.4538).
  - Reduces translation error by **50.5%** compared to the zero baseline and 48.2% compared to the sensor head.

---

## 5. Issue 4: Memory Policy Reconciliations & Tracking Dynamics

### 5.1 Reconciling Fixed Blending vs Hold/Freeze
In Phase 5 v4, the report claimed hold/freeze policy improved error over fixed blending, yet the reported means were 1.2772m (fixed) vs 1.5145m (hold).

Under Phase 5.1 with verified simulator configurations across all 50 trajectories:
| Condition | Policy | Mean Tracking Error (m) | Std Error (m) | 95% Bootstrap CI (m) |
| :--- | :--- | :---: | :---: | :---: |
| `visual_only` | Pure Visual ($g_{\text{vis}}$) | 0.7321 | 0.2175 | [0.6710, 0.7933] |
| `sensor_fixed` | Fixed Blending (0.65 / 0.35) | 0.6879 | 0.2116 | [0.6284, 0.7476] |
| `zero_fixed` | Fixed Blending (0.65 / 0.35) | 0.6919 | 0.2160 | [0.6314, 0.7530] |
| `action_fixed` | Fixed Blending (0.65 / 0.35) | 0.6935 | 0.2129 | [0.6333, 0.7535] |
| `oracle_fixed` | Fixed Blending (0.65 / 0.35) | 0.6914 | 0.2118 | [0.6317, 0.7512] |
| `sensor_hold` | Hold / Freeze Dead-Reckon | 0.6330 | 0.3176 | [0.5440, 0.7205] |
| `zero_hold` | Hold / Freeze Dead-Reckon | 0.6499 | 0.2983 | [0.5673, 0.7326] |
| `action_hold` | Hold / Freeze Dead-Reckon | **0.5490** | 0.2706 | [0.4735, 0.6225] |
| `oracle_hold` | Hold / Freeze Dead-Reckon | **0.5368** | 0.2776 | [0.4589, 0.6141] |
| `action_shuffled_hold`| Action Shuffled Control | 0.5768 | 0.2760 | [0.5005, 0.6523] |
| `action_zero_hold` | Action Zero Control | 0.6165 | 0.2902 | [0.5360, 0.6962] |

### 5.2 Mathematical & Behavioral Mechanics
1. **Under Fixed Blending:** All 4 motion sources collapse to $\sim 0.69$ m. At each step during blackout, the controller blends 65% of the static black frame estimate and 35% of the memory:
   $$\text{mem}_t = 0.65 g_{\text{vis\_black}} + 0.35 \text{Transform}(\text{mem}_{t-1}, \Delta_t)$$
   Over a 10-step window, the contribution of the pre-blackout goal vector decays by $0.35^{10} \approx 2.75 \times 10^{-5}$. The goal vector is dominated by the static network bias of the visual head on black frames.
2. **Under Hold/Freeze Policy:** Visual blending is disabled; memory purely dead-reckons the goal vector using the displacement stream.
   - When displacement is driven by the learned sensor head or zero displacement, dead-reckoning drifts (0.6330m and 0.6499m).
   - When displacement is driven by the action-conditioned predictor, tracking error drops to **0.5490 m**, significantly outperforming zero displacement ($\Delta = -0.1009$ m, Wilcoxon $p = 8.09 \times 10^{-9}$), zero action ($\Delta = -0.0675$ m, Wilcoxon $p = 8.79 \times 10^{-9}$), and shuffled action ($\Delta = -0.0278$ m, Wilcoxon $p = 1.49 \times 10^{-4}$).
   - **Equivalence to Oracle Odometry:** Paired difference between Action Hold and Oracle Hold is $+0.0122$ m (95% CI: $[-0.0010, 0.0251]$ m). Two-sided Wilcoxon $p = 0.0663$. Two One-Sided Tests (TOST) with practical equivalence margin $\delta = 0.05$ m yields **$p = 0.0000$ (Equivalence: True)**.

---

## 6. Issue 5: Pair-Clustered Permutation Null & Interaction Alignment

### 6.1 Pair-Clustered Permutation Testing ($N=50$ Pairs, 100 Scenes)
In Phase 5 v4, permutation testing flipped token labels independently in Scene A and Scene B. Because Scene A and Scene B are matched counterfactuals of the same physical pair, independent flipping violates the paired experimental design.

In Phase 5.1 v5, pair-clustered permutation draws one coin flip $s_i \in \{0, 1\}$ per pair and applies the swap consistently to both Scene A and Scene B:
- **Observed Dual Discrimination:** Scene A = 10.0% (5/50), Scene B = 22.0% (11/50), Pooled = **16.0%** (16/100).
- **Pairs with Dual Success in Both Scenes:** **0/50 (0.0%)**.
- **Pair-Clustered Null Mean:** **19.5% (95% CI: [16.0%, 23.0%])**.
- **Permutation p-values (100,000 resamples):**
  - One-sided ($H_1: \text{obs} > \text{null}$): **$p = 0.9890$** (observed performance is below the null mean).
  - Two-sided ($H_1: \text{obs} \ne \text{null}$): **$p = 0.0571$**.
- **Conclusion:** Dual target selection is statistically indistinguishable from chance under the paired null hypothesis.

### 6.2 Sign-Flip Alignment Tests on Interaction Vector $I$
- Mean interaction magnitude $\|I\| = 0.3521$ m (95% CI: $[0.2921, 0.4161]$ m).
- Cosine alignment with physical target swap $\Delta P$:
  - Mean cosine = **0.3140** (95% CI: $[0.1508, 0.4693]$).
  - Mean angular divergence = **$66.9^\circ$** (95% CI: $[55.3^\circ, 78.9^\circ]$).
  - Sign-flip test for positive alignment ($H_1: \cos > 0$): **$p = 0.0002$**.
  - Sign-flip test for negative alignment ($H_1: \cos < 0$): **$p = 0.9999$**.
  - Two-sided test ($H_1: |\cos| > 0$): **$p = 0.0004$**.
- **Mechanistic Classification:** While there is weak positive correlation with target swap, the interaction vector diverges by $66.9^\circ$ from collinearity ($0^\circ$). This confirms **Mechanism 3 (Mixed Nonlinear Interaction)** and falsifies both clean visual binding (which requires collinearity with target swap) and pure additive task bias (which predicts $I = 0$).

---

## 7. Issue 6: Representation Statistics & Internal Circuits

### 7.1 Effective Dimensionality Across 9 Intermediate Stages
| Stage | Dim | Participation Ratio | Var 50% Dim | Var 80% Dim | Var 90% Dim | Mean Lifetime Sparsity | Mean Population Sparsity | Zero Fraction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `visual_latent_z` | 32 | 3.68 | 2 | 4 | 7 | 0.641 | 0.716 | 0.000 |
| `task_embedding_e` | 16 | 3.96 | 2 | 4 | 6 | 0.678 | 0.787 | 0.000 |
| `input_concat_48d` | 48 | 6.61 | 3 | 7 | 11 | 0.653 | 0.728 | 0.000 |
| `layer1_pre_relu_64d` | 64 | 7.02 | 3 | 7 | 12 | 0.647 | 0.692 | 0.000 |
| `layer1_post_relu_64d`| 64 | 7.63 | 3 | 8 | 13 | 0.647 | 0.692 | 0.510 |
| `layer2_pre_relu_64d` | 64 | 6.27 | 2 | 6 | 11 | 0.661 | 0.703 | 0.000 |
| `layer2_post_relu_64d`| 64 | 5.96 | 2 | 6 | 10 | 0.661 | 0.703 | 0.561 |
| `raw_output_4d` | 4 | 1.75 | 1 | 2 | 2 | 0.445 | 0.680 | 0.000 |
| `postprocessed_goal_4d`| 4 | 2.90 | 1 | 2 | 3 | 0.389 | 0.515 | 0.000 |

### 7.2 Key Findings: Low Effective Dimensionality
- The participation ratios of Layer 1 (7.63 / 64) and Layer 2 (5.96 / 64) demonstrate that intermediate representations are **low-dimensional**, with 80% of total variance captured in just 6 to 8 principal components.
- However, causal sufficiency interventions on the held-out split ($N=25$) reveal that candidate task-sensitive units transfer task-conditioned displacement in a monotonically increasing curve with dimension $k$ ($k=4 \to 23.9\%$, $k=8 \to 38.6\%$, $k=16 \to 70.7\%$, $k=32 \to 90.2\%$, $k=64 \to 100.0\%$).
- The top-8 units achieve only 38.6% transfer efficiency in Layer 2 and 65.9% in Layer 1. After Holm-Bonferroni correction, the candidate units do not demonstrate isolated modular selectivity over random controls, confirming a **distributed low-dimensional population code**.
