# Phase 5.1 Data Dictionary & Coordinate Frame Specifications

**Repository:** `BrianHadianSTEI23/project-haphaestus`  
**Branch:** `fix/interpretability`  
**Execution Commit:** `1b104decca63d95600d3265a1417d6377e4fb7b6`  
**Date:** 2026-11-08  
**Role:** Experimentalist, implementation engineer, and data collector  

---

## 1. Physical Coordinate Frames & Sensor Calibration

### 1.1 World Coordinate Frame ($\mathcal{F}_W$)
- **Convention:** Right-handed North-West-Up (NWU) coordinate system.
- **Origin:** Center of the $4.0 \times 4.0$ m bounded MuJoCo arena floor at $(0.0, 0.0, 0.0)$ m.
- **Axes:**
  - $+X_W$: Pointing along canonical North (forward in arena diagram).
  - $+Y_W$: Pointing along canonical West (left in arena diagram).
  - $+Z_W$: Pointing vertically upward, orthogonal to arena floor.
- **Arena Bounds:** $x \in [-2.0, +2.0]$ m, $y \in [-2.0, +2.0]$ m, floor elevation $z = 0.0$ m.

### 1.2 Robot Body Coordinate Frame ($\mathcal{F}_B$)
- **Convention:** Forward-Left-Up (FLU) mobile robot convention.
- **Origin:** Geometric midpoint between driving wheels on the floor projection plane.
- **State Representation:** Pose vector $p_t = [x_t, y_t, \theta_t]^T$:
  - $x_t \in [-2.0, +2.0]$ m: Position along $+X_W$.
  - $y_t \in [-2.0, +2.0]$ m: Position along $+Y_W$.
  - $\theta_t \in [-\pi, +\pi]$ rad: Robot heading angle counter-clockwise from $+X_W$ about $+Z_W$.
- **Axes:**
  - $+X_B$: Robot forward longitudinal axis.
  - $+Y_B$: Robot lateral left axis.
  - $+Z_B$: Robot upward axis ($+Z_W$).

### 1.3 Transformation Between World and Body Frames
For any point or waypoint $P_W = [x_W, y_W, z_W]^T$ in the world frame:
$$\begin{bmatrix} x_B \\ y_B \end{bmatrix} = \begin{bmatrix} \cos\theta_t & \sin\theta_t \\ -\sin\theta_t & \cos\theta_t \end{bmatrix} \begin{bmatrix} x_W - x_t \\ y_W - y_t \end{bmatrix}$$
Target relative orientation in body frame:
$$\psi_B = \text{atan2}(\sin(\theta_{\text{target}} - \theta_t), \cos(\theta_{\text{target}} - \theta_t))$$

### 1.4 Camera Extrinsics & Intrinsics
- **Camera Mount:** Fixed on robot chassis at body offset $[+0.12, 0.0, +0.18]^T$ m.
- **Orientation:** Facing forward with fixed $-15.0^\circ$ downward pitch angle.
- **Sensor Parameters:**
  - Resolution: $64 \times 64$ pixels, 3 channels (RGB), uint8 normalized to $[0.0, 1.0]$.
  - Vertical Field of View (FOV): $60.0^\circ$.
  - Near / Far Clipping: $0.05$ m / $10.0$ m.

---

## 2. Mathematical Definition of State, Latent, and Action Vectors

### 2.1 Latent State Vectors
- **Visual Latent:** $z \in \mathbb{R}^{256}$, continuous embedding extracted by the visual encoder from $64 \times 64 \times 3$ RGB input.
- **Task Latent:** $e_{\text{task}} \in \mathbb{R}^{32}$, dense embedding retrieved from an 8-bit task ID token ($64$ vocabulary entries) via `nn.Embedding(64, 32)`.
- **Fused Grounding Representation:** Joint representation formed by concatenating or fusing $z$ and $e_{\text{task}}$.

### 2.2 Goal Vector Output ($g$)
The spatial grounding head outputs a 4-dimensional vector:
$$g = [g_x, g_y, \sin\psi, \cos\psi]^T$$
- **Translation Components:**
  - $g_x \in [-3.0, +3.0]$: Target relative position along $+X_B$ (meters).
  - $g_y \in [-3.0, +3.0]$: Target relative position along $+Y_B$ (meters).
  - Magnitude: $d_{xy} = \sqrt{g_x^2 + g_y^2}$ (meters).
- **Orientation Components:**
  - $[\sin\psi, \cos\psi]$: Unit circle vector representing target relative angle $\psi \in [-\pi, +\pi]$ (radians).
  - Angle: $\psi = \text{atan2}(\sin\psi, \cos\psi)$ (radians).

### 2.3 Physical Action Vector ($a$)
- Commanded action $a_t = [v_x, v_y, \omega_z]^T$:
  - $v_x \in [-1.0, +1.0]$: Normalized forward linear velocity (scale factor $0.5$ m/s).
  - $v_y \in [-1.0, +1.0]$: Normalized lateral linear velocity (scale factor $0.5$ m/s).
  - $\omega_z \in [-1.0, +1.0]$: Normalized yaw angular rate (scale factor $1.0$ rad/s).
- Execution rate: 10 Hz ($\Delta t = 0.1$ s per simulation step).

---

## 3. Strict Separation of Physical Units

To prevent dimensional conflation, metrics are strictly partitioned into distinct physical domains:

| Domain | Variable Symbol | Definition | Physical Unit |
| :--- | :--- | :--- | :--- |
| **Translation Distance** | $d_{xy}$ | $\sqrt{\Delta x^2 + \Delta y^2}$ | Meters (m) |
| **Orientation Error** | $\Delta \theta$ | $|\text{atan2}(\sin(\theta_1 - \theta_2), \cos(\theta_1 - \theta_2))|$ | Radians (rad) or Degrees ($^\circ$) |
| **Linear Velocity** | $v$ | $\sqrt{v_x^2 + v_y^2}$ | Meters per second (m/s) |
| **Angular Velocity** | $\omega$ | Angular rate about $+Z_B$ | Radians per second (rad/s) |
| **Effective Dimension** | $\text{PR}$ | $(\sum \lambda_i)^2 / \sum \lambda_i^2$ | Dimensionless integer / real |
| **Sparsity Index** | $S_L, S_P$ | Treves-Rolls normalized index | Dimensionless scalar in $[0.0, 1.0]$ |

*Under no circumstances is a 4D Euclidean distance calculated across mixed translational and trigonometric dimensions.*

---

## 4. Schemas for Phase 5.1 Artifacts

### 4.1 `task_conditioned_binding_v5_summary.json`
- `metadata`: Commit SHA, timestamp, seed, sample counts ($N=50$ pairs, 100 scenes, 200 evaluations).
- `overall_metrics`:
  - `marginal_accuracy`: Fraction of individual targets correctly selected ($46.5\%$).
  - `dual_target_accuracy`: Fraction of scenes where both targets are selected under their respective tokens ($16.0\%$).
  - `pair_clustered_permutation_test`: $100,000$ resamples preserving paired structure; reports null mean ($19.5\%$), 95% CI ($[16.0\%, 23.0\%]$), $p_{\text{one\_sided}} = 0.9890$, $p_{\text{two\_sided}} = 0.0571$.
- `interaction_vector`:
  - `mean_norm_m`: Mean $\|I\| = 0.3521$ m.
  - `cosine_alignment`: Mean cosine alignment with target swap $\Delta P$ ($0.3140$, 95% CI: $[0.1508, 0.4693]$).
  - `mean_angular_divergence_deg`: Mean angle between $I$ and $\Delta P$ ($66.9^\circ$, 95% CI: $[55.3^\circ, 78.9^\circ]$).
  - `sign_flip_tests`: 100,000 resamples; $p_{\text{pos}} = 0.0002$, $p_{\text{neg}} = 0.9999$, $p_{\text{two}} = 0.0004$.
- `stratified_grouped_cv`: 5-fold CV MSE comparisons for H-JEPA and Ground Truth representations across Additive, Bilinear, and MLP decoders.
- `same_category_instance_binding`: Formally flagged as `ARCHITECTURALLY_UNSUPPORTED`.

### 4.2 `task_conditioned_circuits_v5_summary.json`
- `metadata`: Execution parameters, discovery split ($N=25$ pairs), held-out split ($N=25$ pairs).
- `representation_statistics`: For each of 9 stages:
  - `participation_ratio`: Effective dimensionality $(\sum \lambda)^2 / \sum \lambda^2$.
  - `threshold_dimensions`: Dims required for 80%, 90%, 95%, 99% cumulative variance.
  - `lifetime_sparsity`: Treves-Rolls mean, std, 95% CI across units.
  - `population_sparsity`: Treves-Rolls mean, std, 95% CI across observations.
  - `zero_fraction`: Mean inactive unit fraction.
- `causal_interventions`:
  - For Layer 1 and Layer 2 post-ReLU: Sweeps for $k \in [2, 4, 8, 16, 32, 64]$.
  - Top-k candidate vs 100 random draws per pair with Holm-Bonferroni corrected p-values.
  - Explicit metrics for translation ($d_{xy}$ in meters) and orientation (radians/degrees).

### 4.3 `ego_memory_mechanism_v5_summary.json`
- `metadata`: Execution parameters, trajectory count ($N=50$), blackout steps ($K=10$).
- `calibration_metrics`: $N=1,450$ transitions. Translation MAE, $R^2$, and regression slope for Zero Baseline, Learned Sensor Delta, and Action-Conditioned Predictor.
- `blackout_replay_metrics`: Mean tracking error, std, and 95% CI for:
  - `visual_only`: $0.7321$ m.
  - Fixed Blending: Sensor ($0.6879$ m), Zero ($0.6919$ m), Action ($0.6935$ m), Oracle ($0.6914$ m).
  - Hold/Freeze Dead-Reckoning: Sensor ($0.6330$ m), Zero ($0.6499$ m), Action ($0.5490$ m), Oracle ($0.5368$ m), Shuffled Action ($0.5768$ m), Zero Action ($0.6165$ m).
- `statistical_tests`:
  - Action Hold vs Oracle Hold: Paired difference $+0.0122$ m, Wilcoxon $p = 0.0663$, TOST equivalence $p = 0.0000$ (margin $0.05$ m).
  - Action Hold vs Zero Hold: $\Delta = -0.1009$ m, Wilcoxon $p = 8.09 \times 10^{-9}$.
  - Action Hold vs Zero Action: $\Delta = -0.0675$ m, Wilcoxon $p = 8.79 \times 10^{-9}$.
  - Action Hold vs Shuffled Action: $\Delta = -0.0278$ m, Wilcoxon $p = 1.49 \times 10^{-4}$.
- `schema_reference`: Detailed per-step JSONL structure for all 1,500 recorded transition records.
