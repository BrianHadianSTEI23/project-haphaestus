# Phase 5 Data Dictionary: Definitions, Ranges, and Conventions

**Date:** 2026-11-08  
**Repository:** `BrianHadianSTEI23/project-haphaestus` (`fix/interpretability`)  
**Baseline Audit Commit:** `fa0d4f97abbe16de9980f7844db90a6921660324`  

---

## 1. Coordinate Systems & Spatial Conventions

- **World Frame:** Right-handed Cartesian coordinates in meters. Origin $(0, 0, 0)$ is at arena center. Floor span is $[-1.6, 1.6]$ meters along $X$ and $Y$. $Z$ points upward.
- **Robot Body Frame:** Origin is at robot base center.
  - $+X_b$: Forward direction of mobile base.
  - $+Y_b$: Left lateral direction.
  - $\theta$: Robot heading angle in radians, measured counterclockwise from world $+X$.
- **Goal Vector Representation:** Body-relative polar-Cartesian 4D vector:
  $$\mathbf{g} = [dx_b, dy_b, \sin \Delta \theta, \cos \Delta \theta]$$
  where $[dx_b, dy_b]$ is relative position to waypoint in meters, clamped to $[-3.5, 3.5]$ meters, and $[\sin \Delta \theta, \cos \Delta \theta]$ is unit heading offset.

---

## 2. Statistical Metrics & Symbols

| Symbol / Field | Physical Units | Mathematical Definition | Valid Range |
|---|---|---|---|
| `dual_discrimination` | Boolean / Rate | $\mathbb{I}(\|g_A - P_A\| < \|g_A - P_B\|) \land \mathbb{I}(\|g_B - P_B\| < \|g_B - P_A\|)$ | $[0.0, 1.0]$ |
| `margin_a_m` | Meters | $d(g_A, P_B) - d(g_A, P_A)$ (distance margin favoring true target) | $(-\infty, +\infty)$ |
| `err_ang_deg` | Degrees | $|\text{atan2}(\sin(\theta_g - \theta_P), \cos(\theta_g - \theta_P))| \times \frac{180}{\pi}$ | $[0.0, 180.0]^\circ$ |
| $I$ (Interaction Vector) | Meters | $[g(z_B, e_B) - g(z_B, e_A)] - [g(z_A, e_B) - g(z_A, e_A)]$ | $(-\infty, +\infty)^2$ |
| $\|I\|$ | Meters | Euclidean norm of interaction vector | $[0.0, +\infty)$ |
| $\cos \theta_{\text{swap}}$ | Dimensionless | $\frac{I \cdot \Delta P}{\|I\| \|\Delta P\|}$ | $[-1.0, 1.0]$ |
| $\text{TE}$ (Transfer Efficiency) | Fraction | $1.0 - \frac{\|g_{\text{transplant}} - g_{\text{donor}}\|}{\|g_{\text{clean}} - g_{\text{donor}}\|}$ | $[0.0, 1.0]$ |
| Treves-Rolls Sparsity | Dimensionless | $\frac{(\frac{1}{D}\sum_i a_i)^2}{\frac{1}{D}\sum_i a_i^2}$ | $[1/D, 1.0]$ |
| Effective Dimensionality | Dimension | $\frac{(\text{Tr}(\Sigma))^2}{\text{Tr}(\Sigma^2)}$ (Participation ratio of eigenvalues) | $[1.0, D]$ |
| Tracking Error | Meters | $\|g_{\text{curr}}[:2] - y_{\text{GT}}[:2]\|$ | $[0.0, +\infty)$ |

---

## 3. Factorial Conditions in Ego-Memory Experiment

1. `visual_only`: Evaluates instantaneous visual grounding output $g_t = g_{\text{vis}}(s_t)$ without memory.
2. `sensor_fixed`: Observed sensor latent delta $\Delta s_{\text{sensor}} = s_t - s_{t-1}$ with fixed 0.65/0.35 blending.
3. `zero_fixed`: Zero displacement baseline $\Delta d = [0, 0, 0]$ with fixed 0.65/0.35 blending.
4. `action_fixed`: Action-conditioned predicted displacement $\Delta s_{\text{action}} = \delta_s(s_{t-1}, a_{t-1})$ with fixed 0.65/0.35 blending.
5. `oracle_fixed`: Privileged ground-truth physical odometry with fixed 0.65/0.35 blending.
6. `sensor_hold`: Observed sensor delta with hold/freeze memory update during dropout.
7. `zero_hold`: Zero displacement with hold/freeze memory update.
8. `action_hold`: Action-conditioned predicted displacement with hold/freeze memory update.
9. `oracle_hold`: Oracle physical odometry with hold/freeze memory update.
10. `adaptive_sensor`: Adaptive confidence-gated blending with sensor displacement.
11. `adaptive_action`: Adaptive confidence-gated blending with action displacement.
12. `frozen_frame_sensor`: Sensory dropout via repeating the last valid camera frame ($\text{RGB}_t = \text{RGB}_4$).
