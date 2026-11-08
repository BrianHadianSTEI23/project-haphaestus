# Phase 5.2 Unresolved Hypotheses, Boundary Conditions, and Negative Results

**Repository:** `BrianHadianSTEI23/project-haphaestus`  
**Branch:** `fix/interpretability`  
**Commit:** `bc568052e8be10fba4f4a466e913e3dbd8bb4fd6`  
**Role:** Experimentalist and implementation engineer  
**Auditor:** ChatGPT (Senior Independent Scientific Researcher)  

---

## 1. Executive Overview

In accordance with the scientific principles of Project Hephaestus, negative results, disconfirmations, and empirical boundaries are preserved transparently. We do not attempt to retroactively rehabilitate falsified hypotheses or inflate partial findings into universal principles.

This document details the unresolved hypotheses, empirical boundary conditions, and formal negative results established in Phase 5.2.

---

## 2. Key Negative Findings & Disconfirmations

### 2.1 Disconfirmation of Dedicated Sparse Binding Circuits (Hypothesis H2)

- **The Hypothesis:**  
  A compact, sparse subnetwork of hidden units ($k \approx 8$, or $12.5\%$ of a 64-dimensional layer) acts as a dedicated causal circuit mediating task-conditioned spatial grounding.
- **The Phase 5.1 Observation:**  
  Candidate units achieved significantly higher sufficiency transfer than uniformly sampled random subsets (Layer 1 $k=8$: $65.9\%$ vs $13.8\%$; Layer 2 $k=8$: $38.6\%$ vs $12.2\%$, both $p < 10^{-4}$).
- **The Phase 5.2 Controlled Refutation:**  
  When random subsets are strictly matched to the candidate subset's task-induced perturbation norm ($\|\Delta h_U\| \le 5\%$ tolerance):
  1. **Layer 1:** Candidate units are *outperformed* by norm-matched random controls ($k=8$: Candidate $\eta = 0.340$, Norm-Matched $\eta = 0.430$, paired difference $\Delta = -0.090, p_{\text{adj}} = 1.0000$).
  2. **Layer 2:** Candidate units maintain a small positive margin ($\Delta \approx +0.06$ to $+0.08$), but at small $k$ ($k=4, 8, 16$) this margin fails to achieve significance under Holm-Bonferroni correction ($k=8: p_{\text{adj}} = 0.5155$; $k=16: p_{\text{adj}} = 0.3434$). Only at $k=32$ (half the network) does significance survive ($p_{\text{adj}} = 0.0117$).
- **Scientific Implication:**  
  The apparent selectivity observed in Phase 5.1 was largely an **activation-magnitude confounder**. In fully connected MLP layers, task modulation is distributed across the representation manifold rather than localized into modular, autonomous sparse circuits.

### 2.2 Mathematical Mechanism of the Layer 1 Confounder

Why did norm-matched random subsets outperform candidate units in Layer 1?
1. **Low Effective Dimensionality:** Spectral analysis shows that Layer 1 post-ReLU activations have a Participation Ratio of only $8.14/64$ ($80\%$ variance in 7 dimensions).
2. **Subspace Alignment Saturation:** The top-ranked candidate units are strongly collinear with the principal eigenvector of $\Delta h$. As $k$ increases, adding more candidate units increases perturbation along the same saturated direction.
3. **Random Manifold Spread:** Randomly selected units that match the Euclidean norm $\|\Delta h_U\|$ are distributed across orthogonal directions of the activation manifold. When passed through downstream weights $W_2$ and ReLU, these distributed activations avoid saturation, producing higher projected output transfer along the target contrast vector.

---

## 3. Boundary Conditions of Validated Mechanisms

### 3.1 Non-Commutativity in Dead-Reckoning (Hypothesis H1 Sensitivity)

- **The Finding:**  
  Action hold dead-reckoning is statistically sensitive to within-trajectory temporal action order ($p = 0.000408$). Permuting actions across time degrades tracking accuracy.
- **The Boundary Condition (Modest Effect Size):**  
  The degradation is only $-0.0075$ m ($7.5$ mm) over a 10-step ($1.0$-second) blackout.
- **Physical Derivation:**  
  In $\mathrm{SE}(2)$, position evolution obeys:
  $$\mathbf{x}(T) = \mathbf{x}(0) + \int_0^T \mathbf{R}(\theta(t)) \mathbf{v}_b(t) \, dt$$
  $$\theta(t) = \theta(0) + \int_0^t \omega(\tau) \, d\tau$$
  For small angular velocities ($\Delta\theta \approx 0.016$ rad/step $\approx 0.94^\circ/\text{step}$):
  $$\mathbf{R}(\theta(t)) \approx \mathbf{I} + \theta(t) \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$
  The position error between the true sequence and a time-permuted sequence is governed by the second-order commutator:
  $$\Delta \mathbf{x}_{\text{perm}} \sim \iint_{t_1 < t_2} \omega(t_1) \mathbf{v}_b(t_2) \, dt_1 \, dt_2$$
  Because angular displacement per step is small, the commutator magnitude is bounded by $\mathcal{O}(\Delta\theta \cdot \|\mathbf{v}\| \cdot T) \approx 0.01$ m.
  Therefore, temporal sequence scrambling only produces modest trajectory drift over short horizons, whereas total commanded energy ($\sum \mathbf{v}$) dominates dead-reckoning performance.

### 3.2 Above-Null Target Binding Limitation

- **The Finding:**  
  Dual discrimination rate under paired counterfactual scenes remains at $16.0\%$ ($16/100$ scenes), which does not exceed the pair-clustered permutation null ($19.5\%$, one-sided $p = 0.9890$).
- **The Limitation:**  
  The current H-JEPA architecture (monolithic visual encoder + MLP task-grounding head) does not possess explicit slot-based or attention-based binding mechanisms. Target conditioning acts as a continuous feature bias on the visual latent, which fails to reliably resolve target-distractor ambiguity when two visually similar objects are present in the arena.

---

## 4. Open Questions for Future Architectural Iterations

1. **How can true functional selectivity be induced in learned world models?**  
   Standard MLP grounding heads naturally form distributed, magnitude-driven representations. Inducing sparse modular circuits may require explicit sparsity regularizers (e.g. $L_1$ weight penalties or Top-K activation routing like Mixture-of-Experts).
2. **What architectural mechanism is required for robust instance-level binding?**  
   Since `encode_task_id()` cannot represent identical object instances, instance-level spatial intelligence requires relational visual transformers or object-centric representations (e.g. Slot Attention).
3. **Adaptive Sensor-Action Fusion:**  
   Currently, the controller uses fixed blending ($0.65/0.35$) during clean steps and hold during blackout. An adaptive Kalman or learned gating mechanism based on visual uncertainty would eliminate the artificial memory decay observed during extended visual dropout.
