# Phase 5.1 Unresolved Hypotheses & Open Scientific Questions

**Repository:** `BrianHadianSTEI23/project-haphaestus`  
**Branch:** `fix/interpretability`  
**Execution Commit:** `1b104decca63d95600d3265a1417d6377e4fb7b6`  
**Date:** 2026-11-08  
**Role:** Experimentalist, implementation engineer, and data collector  

---

## 1. Executive Summary

This document formalizes the unresolved scientific hypotheses, empirical boundary conditions, and open mechanistic questions emerging from Phase 5.1 experimental revalidation. Following the scientific integrity protocol, negative findings and architectural boundaries are preserved rather than rationalized into speculative theoretical narratives.

---

## 2. Unresolved Question 1: Instance-Level Binding Beyond Semantic Categories

### 2.1 Empirical Finding
In Phase 5 v4, a 5.0% dual discrimination rate was reported and characterized as a failure of "same-category instance binding." However, forensic audit of `encode_task_id()` in Phase 5.1 revealed:
1. The vocabulary contains 6 landmark classes (`TARGET_KEYS`) $\times$ 8 semantic relations (`SEMANTIC_RELATIONS`) = 48 valid indices (bounded to 64).
2. All target tokens are uniquely mapped to distinct semantic/color categories (e.g., `red_box`, `blue_cylinder`, `green_cylinder`).
3. No slot, token, or coordinate channel exists to disambiguate visually identical object instances (e.g., "left blue cylinder" vs "right blue cylinder").
4. Consequently, the historical 5.0% result evaluated distinct colors under corrupted simulator state, not instance binding. True same-category instance binding is **architecturally unrepresentable** in the current system (`ARCHITECTURALLY_UNSUPPORTED`).

### 2.2 Unresolved Hypothesis
*Can a single continuous visual latent $z \in \mathbb{R}^{256}$ support instance-level disambiguation of visually identical objects when augmented with an ego-centric relational or spatial index?*

### 2.3 Required Experimental Setup for Future Resolution
1. **Vocabulary Extension:** Introduce an explicit instance/spatial modifier to task conditioning:
   $$\text{task} = (\text{category\_id}, \text{instance\_selector})$$
   where $\text{instance\_selector} \in \{\text{leftmost}, \text{rightmost}, \text{closest}, \text{furthest}\}$.
2. **Identical Twin Twin-Scene Benchmark:** Spawn two visually indistinguishable cylinders (identical geometry, color, texture, and reflectance) in MuJoCo. Verify robot visibility of both simultaneously.
3. **Causal Test:** Swap the instance selector while keeping the scene frozen; measure whether the grounding vector cleanly re-targets the alternative instance without retraining the backbone.

---

## 3. Unresolved Question 2: Mathematical Dominance of Decay in Fixed Blending vs Hold Policy

### 3.1 Empirical Finding
In 10-step visual blackout experiments ($N=50$ trajectories):
- **Fixed Blending Policy:** Under nominal visual-memory blending ($g_t = 0.65 g_{\text{vis}} + 0.35 g_{\text{mem}}$), all motion sources collapsed to approximately the same tracking error ($\sim 0.69$ m):
  - Sensor Blending: $0.6879 \pm 0.0436$ m
  - Zero Blending: $0.6919 \pm 0.0427$ m
  - Action Blending: $0.6935 \pm 0.0424$ m
  - Oracle Blending: $0.6914 \pm 0.0430$ m
- **Hold Policy (Blackout Dead-Reckoning):** When visual input was masked during blackout, motion sources differentiated dramatically:
  - Zero Hold: $0.6499 \pm 0.0385$ m
  - Action Hold: $0.5490 \pm 0.0326$ m
  - Oracle Hold: $0.5368 \pm 0.0320$ m
  - Action Hold vs Oracle Hold: Paired difference $+0.0122$ m, TOST $p = 0.0000$ (statistically equivalent within 0.05 m margin).

### 3.2 Mathematical Proof of Fixed Blending Collapse
Let $\alpha = 0.65$ (visual weight) and $\beta = 1 - \alpha = 0.35$ (memory weight).
During blackout, the visual input is a static black frame $\mathbf{0}$. The visual grounding head produces a constant offset $\bar{g}_{\text{black}} = \text{GroundingHead}(\text{Encoder}(\mathbf{0}), e_{\text{task}})$.
At step $k$ of blackout ($k \in \{1, \dots, K\}$):
$$g_k = \alpha \bar{g}_{\text{black}} + \beta T(\Delta_k) g_{k-1}$$
Unrolling over $K$ steps:
$$g_K = \beta^K \left( \prod_{j=1}^K T(\Delta_j) \right) g_0 + \alpha \sum_{m=1}^K \beta^{K-m} \left( \prod_{j=m+1}^K T(\Delta_j) \right) \bar{g}_{\text{black}}$$
For $K = 10$ and $\beta = 0.35$:
$$\beta^{10} = (0.35)^{10} \approx 2.76 \times 10^{-5}$$
The true motion-propagated goal memory $g_0$ contributes less than $0.003\%$ of the final vector. The remaining $99.997\%$ of the vector is governed by the geometric series:
$$\alpha \sum_{m=0}^9 (0.35)^m \approx 0.65 \times \frac{1 - 0.35^{10}}{1 - 0.35} \approx 1.000$$
The memory vector is completely overwritten by $\bar{g}_{\text{black}}$, rendering the choice of dead-reckoning motion source ($\Delta_k$) mathematically irrelevant under fixed blending.

### 3.3 Unresolved Hypothesis
*Can an adaptive Kalman-style or attention-gated confidence mechanism dynamically modulate $\alpha_t \in [0, 1]$ based on observation entropy or latent prediction error $\epsilon_t = \|z_t - \hat{z}_t\|$, preserving dead-reckoning accuracy during perceptual outages without manual mode switching?*

---

## 4. Unresolved Question 3: The Sensor Displacement Head Failure Mode

### 4.1 Empirical Finding
In offline calibration across $N = 1,450$ consecutive transitions:
- **Zero Baseline:** MAE $= 0.0287$ m, $R^2 = 0.0000$
- **Learned Sensor Head ($\Delta s(z_t, z_{t+1})$):** MAE $= 0.0274$ m, $R^2 = 0.0143$, regression slope $\beta_1 = 0.2112$
- **Action Forward Predictor ($\Delta s(s_t, a_t)$):** MAE $= \mathbf{0.0142}$ m, $R^2 = \mathbf{0.2549}$, regression slope $\beta_1 = 0.4538$

The learned sensor head achieves virtually zero explanatory power over the zero baseline ($R^2 = 0.0143$, error reduction of only $0.0013$ m), whereas the action forward predictor cuts translation error by $50.5\%$.

### 4.2 Candidate Explanations
1. **Aperture / Texture Poverty:** The AMR arena features large untextured floor panels and plain walls. Consecutive visual observations $z_t, z_{t+1}$ have high cosine similarity ($> 0.96$), so visual feature differences $\Delta z = z_{t+1} - z_t$ are dominated by sensor noise, camera vibration, and compression artifacts rather than translational optical flow.
2. **Encoder Training Loss Invariance:** H-JEPA and LeWM representations are trained with contrastive and predictive objectives that encourage invariance to minor camera jitter. This invariance directly suppresses the subtle pixel displacements needed to infer fine physical motion from visual features alone.
3. **Control Determinism:** In contrast, commanded action $a_t = [v_x, v_y, \omega_z]$ directly excites wheel motors under physics simulation. Forward transitions are physically driven by the actuator command, making $a_t$ a vastly more informative predictor than untextured optical flow.

### 4.3 Unresolved Hypothesis
*Does sensor-based visual odometry require explicit low-level optical flow features (e.g., Lucas-Kanade or RAFT feature pyramids) rather than high-level invariant latent representations $z \in \mathbb{R}^{256}$ to achieve competitive motion estimation?*

---

## 5. Unresolved Question 4: Modularity vs Distributed Low-Dimensional Subspaces

### 5.1 Empirical Finding
- Intermediate activations in the grounding head exhibit low effective dimensionality:
  - Layer 1: Participation Ratio $= 7.63 / 64$ ($80\%$ variance in 8 dimensions).
  - Layer 2: Participation Ratio $= 5.96 / 64$ ($80\%$ variance in 6 dimensions).
- Causal ablation of candidate "task-specific" units shows monotonic scaling with dimension count $k$, but candidate units do **not** outperform random dimension-matched subsets after Holm-Bonferroni correction ($p > 0.05$ across all tests).
- Modularity is formally falsified; task routing is implemented as a distributed low-dimensional subspace rotation rather than localized modular circuits.

### 5.2 Unresolved Hypothesis
*What training objective or architectural regularizer is necessary and sufficient to force an embodied spatial network to develop disentangled, modular circuits rather than distributed low-dimensional subspaces?*
- Potential inductive biases to investigate:
  1. $\ell_1$ sparsity penalties on hidden units (e.g., Sparse Autoencoders).
  2. Orthogonality constraints on weight matrices ($W^T W = I$).
  3. Gated mixture-of-experts (MoE) routing where task tokens explicitly gate separate MLP feedforward routes.
