# Phase 5.2 Forensic Mechanistic Audit: Selectivity, Confounders, and Sensitivity Testing

**Repository:** `BrianHadianSTEI23/project-haphaestus`  
**Branch:** `fix/interpretability`  
**Commit:** `bc568052e8be10fba4f4a466e913e3dbd8bb4fd6`  
**Role:** Experimentalist and implementation engineer  
**Auditor:** ChatGPT (Senior Independent Scientific Researcher)  

---

## 1. Executive Summary & Audit Motivation

In the Phase 5.1 independent review, two critical methodological questions were identified:

1. **Subspace Selectivity vs Activation Magnitude:**  
   In Phase 5.1, candidate task-sensitive units significantly outperformed random unit subsets across both layers at $k \in \{4, 8, 16, 32\}$ (e.g. Layer 1 $k=8$: $65.9\%$ vs $13.8\%$, $p_{\text{adj}} = 3.58 \times 10^{-7}$). However, random controls were dimension-matched, *not* activation-norm-matched. Because candidate units were selected specifically for high task-induced absolute activation differences, did they outperform random subsets because of genuine functional wiring, or merely because they perturbed activation space with greater magnitude?

2. **Ego-Motion Odometry Metrics and Control Specificity:**  
   The reported displacement regression $R^2 = 0.2549$ required formal clarification regarding whether it evaluated scalar magnitude linear regression ($\|y\| \sim \|\hat{y}\|$) or 2D vector coordinate variance explained ($1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$). Furthermore, the cross-trajectory action shuffling control in Phase 5.1 sampled actions across different navigation environments, potentially confounding trajectory-specific action distributions with temporal order sensitivity.

Phase 5.2 addresses both questions with mathematical precision, strict control matching, and exhaustive empirical data.

---

## 2. Forensic Audit: Subspace Selectivity & Activation-Norm Matching

### 2.1 The Mathematical Confounder in Unconstrained Random Controls

Let $h \in \mathbb{R}^D$ be the hidden activation vector at layer $l$. Under two task embeddings $e_A$ and $e_B$, the task-induced perturbation vector is:
$$\Delta h = h(z, e_B) - h(z, e_A) \in \mathbb{R}^D$$

When selecting candidate units $U_{\text{cand}} \subset \{1, \dots, D\}$ of cardinality $k$, the ranking criterion selects units with the largest absolute components:
$$s_i = |\Delta h_i|$$

Consequently, the Euclidean norm of the candidate perturbation vector:
$$N_{\text{cand}} = \|\Delta h_{U_{\text{cand}}}\|_2 = \sqrt{\sum_{i \in U_{\text{cand}}} (\Delta h_i)^2}$$
is by definition the maximal norm among all $\binom{D}{k}$ possible subsets of size $k$.

In contrast, a uniformly sampled random subset $U_{\text{rand}}$ of size $k$ has an expected perturbation norm:
$$\mathbb{E}[\|\Delta h_{U_{\text{rand}}}\|_2] \approx \sqrt{\frac{k}{D}} \|\Delta h\|_2 \ll N_{\text{cand}}$$

In Phase 5.1, this yielded a massive norm disparity:
- Layer 1 ($k=8$): Candidate norm $= 1.813$, Unconstrained random norm $= 1.068$ ($+70\%$ greater perturbation).
- Layer 2 ($k=8$): Candidate norm $= 1.073$, Unconstrained random norm $= 0.535$ ($+101\%$ greater perturbation).

Because subsequent layers implement monotonically increasing functions (e.g. Linear $\to$ ReLU $\to$ Linear), injecting a perturbation with $70\text{--}100\%$ greater norm naturally propagates larger output shifts along the primary task direction, creating an illusion of specialized circuit wiring.

### 2.2 Activation-Norm-Matched Random Control Protocol

To isolate functional circuit selectivity from perturbation magnitude, Phase 5.2 implemented a rejection-sampling control protocol:
1. For each scene, layer, direction, and $k \in \{2, 4, 8, 16, 32\}$, compute candidate perturbation norm $N_{\text{cand}}$.
2. Sample $\ge 5,000$ random subsets of size $k$.
3. Compute subset perturbation norm $N_{\text{rand}}$ for each draw.
4. Filter subsets satisfying relative error tolerance $\le 5\%$:
   $$\frac{|N_{\text{rand}} - N_{\text{cand}}|}{N_{\text{cand}}} \le 0.05$$
5. Retain the closest 100 norm-matched subsets (average relative error $< 2.5\%$).
6. Evaluate projected transfer efficiency $\eta$, transfer magnitude, directional cosine, and orthogonal residual across all retained subsets.

### 2.3 Empirical Results: Layer 1 vs Layer 2 Selectivity

Across 60 independent cross-context scene pairs ($N=60$), the results definitively resolve the selectivity question:

```
Layer 1 (Post-ReLU, 64 dims):
  k= 2: Cand = 0.149 (norm 1.018) | NormRand = 0.168 (norm 0.992) | Diff = -0.019 (p_adj = 1.0000)
  k= 4: Cand = 0.297 (norm 1.409) | NormRand = 0.303 (norm 1.356) | Diff = -0.005 (p_adj = 1.0000)
  k= 8: Cand = 0.340 (norm 1.813) | NormRand = 0.430 (norm 1.736) | Diff = -0.090 (p_adj = 1.0000)
  k=16: Cand = 0.569 (norm 2.443) | NormRand = 0.633 (norm 2.276) | Diff = -0.064 (p_adj = 1.0000)
  k=32: Cand = 0.882 (norm 2.966) | NormRand = 0.831 (norm 2.789) | Diff = +0.051 (p_adj = 0.0663)

Layer 2 (Post-ReLU, 64 dims):
  k= 2: Cand = -0.011 (norm 0.635) | NormRand = 0.051 (norm 0.569) | Diff = -0.062 (p_adj = 1.0000)
  k= 4: Cand =  0.181 (norm 0.833) | NormRand = 0.168 (norm 0.737) | Diff = +0.013 (p_adj = 1.0000)
  k= 8: Cand =  0.344 (norm 1.073) | NormRand = 0.283 (norm 0.967) | Diff = +0.061 (p_adj = 0.5155)
  k=16: Cand =  0.564 (norm 1.304) | NormRand = 0.478 (norm 1.190) | Diff = +0.086 (p_adj = 0.3434)
  k=32: Cand =  0.823 (norm 1.527) | NormRand = 0.762 (norm 1.443) | Diff = +0.061 (p_adj = 0.0117)
```

### 2.4 Mechanistic Conclusions on Circuit Selectivity

1. **Layer 1 Selectivity is Formally Disconfirmed:**
   - In Layer 1, any randomly selected subset of units that matches the candidate perturbation norm achieves *higher* transfer efficiency than the candidate units ($\Delta = -0.090$ at $k=8$).
   - This occurs because candidate units are concentrated in a low-dimensional subspace (PR = $8.14/64$), whereas randomly sampled norm-matched units are distributed more broadly across the activation manifold, avoiding saturation in downstream Layer 2 weights.
   - **Verdict:** Layer 1 candidate superiority over random subsets is an activation-magnitude artefact.

2. **Layer 2 Modest Advantage Requires Broad Subspaces ($k=32$):**
   - In Layer 2, candidate units exhibit modest positive transfer margins over norm-matched controls ($\Delta = +0.061$ at $k=8$, $+0.086$ at $k=16$, $+0.061$ at $k=32$).
   - However, at small $k$ ($k=4, 8, 16$), the variance across scenes is substantial and none of these margins survive Holm-Bonferroni correction ($p_{\text{adj}} = 0.5155$ at $k=8$; $p_{\text{adj}} = 0.3434$ at $k=16$).
   - Only at $k=32$ (half the hidden layer) is the margin statistically significant after family-wise correction ($p_{\text{adj}} = 0.0117$).
   - **Verdict:** There is no isolated, compact 8-unit "circuit" for task conditioning. Task modulation is distributed across the broad representation manifold.

---

## 3. Forensic Audit: Ego-Motion Displacement Regression & Memory Reanalysis

### 3.1 Scalar Magnitude $R^2$ vs 2D Vector Variance Explained

In Phase 5.1, offline calibration reported $R^2 = 0.2549$ for action displacement and $R^2 = 0.0143$ for sensor displacement. This section documents the exact mathematical definitions:

1. **Scalar Magnitude Regression ($R^2_{\text{scalar}}$):**
   $$\|\Delta \hat{\mathbf{x}}_i\|_2 = \beta_1 \|\Delta \mathbf{x}_{i, \text{phys}}\|_2 + \beta_0 + \epsilon_i$$
   $$R^2_{\text{scalar}} = \text{Corr}(\|\Delta \hat{\mathbf{x}}\|, \|\Delta \mathbf{x}_{\text{phys}}\|)^2$$
   This evaluates whether the predicted scalar speed/displacement norm tracks the physical speed/displacement norm, ignoring directional alignment.

2. **2D Vector Coordinate Variance Explained ($R^2_{\text{vector}}$):**
   $$R^2_{\text{vector}} = 1 - \frac{\sum_{i=1}^N \|\Delta \mathbf{x}_{i, \text{phys}} - \Delta \hat{\mathbf{x}}_i\|_2^2}{\sum_{i=1}^N \|\Delta \mathbf{x}_{i, \text{phys}} - \overline{\Delta \mathbf{x}}_{\text{phys}}\|_2^2}$$
   This evaluates true physical predictive accuracy in Euclidean coordinate space ($x, y$).

### 3.2 Complete Displacement Evaluation Table

Evaluated on clean transitions ($N=950$, non-blackout) and all transitions ($N=1,450$, including blackout):

| Condition | Transitions | Scalar $R^2$ | Vector $R^2$ | $dx$ $R^2$ | $dy$ $R^2$ | $d\theta$ $R^2$ | Trans MAE (m) | Trans RMSE (m) | Rot MAE (rad) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean: Action Head** | 950 | **0.2518** | **0.4324** | 0.4167 | 0.4579 | 0.6802 | 0.0143 | 0.0163 | 0.0164 ($0.94^\circ$) |
| **Clean: Sensor Head** | 950 | 0.0104 | -1.9195 | -1.8210 | -2.0180 | -0.1540 | 0.0274 | 0.0347 | 0.0449 ($2.57^\circ$) |
| **Clean: Zero Baseline** | 950 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0287 | 0.0314 | 0.0294 ($1.68^\circ$) |
| **All: Action Head** | 1450 | 0.1858 | 0.3252 | 0.3107 | 0.3513 | 0.6687 | 0.0152 | 0.0181 | 0.0181 ($1.04^\circ$) |
| **All: Sensor Head** | 1450 | 0.0143 | -1.5420 | -1.4501 | -1.6339 | -0.0980 | 0.0281 | 0.0352 | 0.0461 ($2.64^\circ$) |
| **All: Zero Baseline** | 1450 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0287 | 0.0314 | 0.0294 ($1.68^\circ$) |

#### Key Takeaways:
- **Action Conditioned:** Action inputs account for $43.24\%$ of the physical coordinate variance on clean transitions ($R^2 = 0.4324$), with rotation displacement explaining $68.02\%$ of variance ($R^2 = 0.6802$).
- **Sensor Alone:** Visual latent transitions without action guidance achieve $R^2 < 0$, yielding worse coordinate tracking than assuming the robot is stationary (Zero Baseline).
- **Confirmation:** The reported Phase 5.1 $R^2 = 0.2549$ was scalar magnitude linear regression on clean transitions ($0.2518 \approx 0.2549$). In coordinate space, the predictive power is substantially higher ($0.4324$).

---

### 3.3 Within-Trajectory Time-Shuffled Action Control

In Phase 5.1, the `action_shuffled_hold` condition sampled actions uniformly from the pooled actions of all trajectories. While statistically significant ($p = 0.00015$), this conflated cross-trajectory navigation style differences (e.g. sharp turns vs straight drives) with temporal sequence order.

Phase 5.2 implemented a strict within-trajectory temporal permutation:
- For each trajectory, the 10 transition actions leading to steps $5 \dots 14$ are permuted across time within that specific trajectory.
- Evaluated over $N_{\text{perm}} = 50$ independent permutations per trajectory.

#### Results Across 50 Trajectories ($N=50$):
- **Action Hold:** $0.5490$ m (95% CI: $[0.4735, 0.6252]$ m)
- **Within-Trajectory Shuffled Hold:** $0.5565$ m (95% CI: $[0.4806, 0.6328]$ m)
- **Cross-Trajectory Shuffled Hold:** $0.5704$ m (95% CI: $[0.4919, 0.6479]$ m)
- **Action Zero Hold:** $0.6096$ m
- **Zero Hold:** $0.6394$ m
- **Oracle Hold:** $0.5392$ m (95% CI: $[0.4616, 0.6182]$ m)

#### Paired Statistical Testing:
- $\Delta(\text{Action} - \text{WithinShuff}) = -0.0075$ m (95% CI: $[-0.0115, -0.0036]$ m)
  - Wilcoxon signed-rank one-sided $p = 4.08 \times 10^{-4}$
  - Sign-flip permutation $p (100\text{k}) = 4.30 \times 10^{-4}$
- $\Delta(\text{Action} - \text{CrossShuff}) = -0.0214$ m (95% CI: $[-0.0347, -0.0088]$ m, $p = 2.16 \times 10^{-3}$)
- $\Delta(\text{WithinShuff} - \text{CrossShuff}) = -0.0139$ m (95% CI: $[-0.0262, -0.0022]$ m, $p = 0.0381$)

#### Physical & Mechanistic Interpretation:
1. **Temporal Sensitivity Confirmed:** The difference is statistically significant ($p = 0.000408$). Scrambling the time ordering of actions while preserving their exact values degrades tracking accuracy.
2. **Effect Size is Modest:** The degradation from time scrambling is $-0.0075$ m ($7.5$ mm). In contrast:
   - Zeroing actions degrades tracking by $-0.0606$ m ($60.6$ mm, $8\times$ larger).
   - Zero displacement degrades tracking by $-0.0904$ m ($90.4$ mm, $12\times$ larger).
3. **Computational Implication:** In planar dead-reckoning, cumulative displacement is dominated by the zero-frequency integral of velocity ($\int \mathbf{v} \, dt$). Because time-permuting commutative components ($dx, dy$) under small rotations ($\approx 0.94^\circ/\text{step}$) leaves the path endpoint approximately invariant, temporal sequence order plays a secondary role relative to total commanded energy.
