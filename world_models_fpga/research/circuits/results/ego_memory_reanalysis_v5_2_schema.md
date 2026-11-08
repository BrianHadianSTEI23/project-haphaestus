# Schema: Phase 5.2 Ego-Motion Memory Reanalysis (`ego_memory_reanalysis_v5_2`)

## 1. Primary Artifacts
- **Summary JSON**: [`ego_memory_reanalysis_v5_2_summary.json`](./ego_memory_reanalysis_v5_2_summary.json)

## 2. Methodology & Scientific Purpose

### 2.1 Displacement Head R^2: Scalar Magnitude vs 2D Vector Variance Explained
- **Scalar Magnitude R^2**:
  Evaluates linear regression between physical scalar displacement norm ||Delta x_phys|| and predicted norm ||Delta x_hat||:
  ||Delta x_hat|| = beta_1 * ||Delta x_phys|| + beta_0
  In Phase 5.1, this yielded R^2 = 0.2549 for action-conditioned displacement and R^2 = 0.0143 for sensor displacement on clean transitions.
- **2D Vector Variance Explained (R^2_vector)**:
  Evaluates coordinate-level predictive fit in 2D physical space:
  R^2_vector = 1 - sum(||Delta x_phys - Delta x_hat||^2) / sum(||Delta x_phys - mean(Delta x_phys)||^2)
  In clean transitions, action-conditioned displacement achieves R^2_vector = 0.4324, with R^2(dx) = 0.4167, R^2(dy) = 0.4579, and angular R^2(dtheta) = 0.6802.

### 2.2 Within-Trajectory Time-Shuffled Action Control
- **Limitation of Phase 5.1 Cross-Trajectory Shuffling**:
  Phase 5.1 sampled actions uniformly across all steps of all trajectories. While statistically significant (p = 0.00015), this conflated trajectory-specific action distributions (e.g. trajectories requiring sharp turning vs straight driving) with temporal order sensitivity.
- **Phase 5.2 Within-Trajectory Temporal Permutation**:
  For each trajectory, the 10 actions associated with the blackout window (transitions 4 -> 5, ..., 13 -> 14) are randomly permuted in time N_perm = 50 times.
  This preserves the exact empirical action distribution for each trajectory while destroying temporal coherence.
- **Statistical Results**:
  - Action Hold Mean Error: 0.5490 m (95% CI: [0.4735, 0.6252])
  - Within-Traj Shuffled Hold Mean Error: 0.5565 m (95% CI: [0.4806, 0.6328])
  - Paired Difference: -0.0075 m (Wilcoxon one-sided p = 4.0780e-04, sign-flip p = 4.3000e-04)
