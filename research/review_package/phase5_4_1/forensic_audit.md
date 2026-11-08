# Phase 5.4.1 Forensic Audit: Mathematical Reconciliation & Directional Geometry

**Role:** Implementation Engineer & Experimentalist  
**Target:** Independent interpretation and synthesis by ChatGPT (Senior Independent Scientific Researcher)  
**Baseline SHA:** `0cafb12240dcd749a7c3e0c8c9612f4ddeca5ea2`  
**Execution SHA:** `d668ce5c0f552eb90a8e26307409b813efc5fc08`  

---

## 1. Mathematical Row Count Derivation & Streaming Key Audit

The Phase 5.3 intervention database (`task_conditioned_circuits_v5_4_intervention_metrics.jsonl.gz`) contains exactly **447,800 records**. Previous documentation loosely referenced "4,800 candidate interventions". A rigorous audit establishes the exact mathematical derivation of every record:

### Experimental Combinatorics:
- **Independent Scene Pairs**: $N = 60$ (30 nominal context, 30 illumination-shifted context).
- **Scenes per Pair**: 2 (`scene_a` and `scene_b`).
- **Network Layers Evaluated**: 2 (`layer1_post_relu_64d` and `layer2_post_relu_64d`).
- **Subset Sizes ($k$)**: 5 ($k \in \{2, 4, 8, 16, 32\}$).
- **Intervention Directions**: 2 ($A \to B$ and $B \to A$).
- **Total Experimental Settings**:
  $$60 \times 2 \times 2 \times 5 \times 2 = \mathbf{2,400}\text{ settings}$$

### Condition Rows Breakdown:
1. **Candidate Interventions**:
   - Each setting evaluates exactly 1 candidate hidden unit draw ($K_c$).
   - Rows: $2,400 \times 1 = \mathbf{2,400}\text{ records}$.
2. **Dimension-Matched Random Controls**:
   - Each setting evaluates 100 unconstrained random coordinate subsets of dimension $k$.
   - Rows: $2,400 \times 100 = \mathbf{240,000}\text{ records}$.
3. **Norm-Matched Random Controls**:
   - Controls are drawn from the strict common-support pool ($\le 5\%$ norm error tolerance).
   - In `task_conditioned_circuits_v5_3_pairs.jsonl`, exactly **2,054 settings** satisfied `VALID_COMMON_SUPPORT` ($N \ge 100$ valid matching draws within 200,000 samples). The remaining 346 settings had `INSUFFICIENT_COMMON_SUPPORT` ($0$ draws retained under zero-fallback policy).
   - Rows: $2,054 \times 100 = \mathbf{205,400}\text{ records}$.

$$\text{Total Records} = 2,400 + 240,000 + 205,400 = \mathbf{447,800}\text{ records}$$

### Genuine Key-Tuple Reconciliation Audit:
Every record in `task_conditioned_circuits_v5_4_intervention_metrics.jsonl.gz` was streamed and verified against the expected Cartesian product of keys `(pair_id, scene_id, layer, direction, k, condition, draw_id)`.
- **Expected Keys**: 447,800
- **Observed Records**: 447,800
- **Missing Keys**: 0
- **Unexpected / Extra Keys**: 0
- **Duplicate Keys**: 0
- **Reconciliation Success**: `True` (derived programmatically from key-set comparison, with zero hardcoding).

---

## 2. Scene-Swap Reversal Vector Algebra & Directional Proof

A crucial forensic correction concerns the interpretation of the scene-swap reversal cosine.

### Mathematical Formulation:
In each paired trial, Scene B physically swaps the spatial locations of Target A and Target B in the arena. Consequently, the true body-frame target displacement inverts:
$$\Delta\mathbf{g}_{\text{gt}}^B = -\Delta\mathbf{g}_{\text{gt}}^A$$

Let $\Delta\mathbf{g}_{\text{model}}^A = \mathbf{g}_{\text{model}}^{A, B} - \mathbf{g}_{\text{model}}^{A, A}$ be the model contrast under Scene A, and $\Delta\mathbf{g}_{\text{model}}^B = \mathbf{g}_{\text{model}}^{B, B} - \mathbf{g}_{\text{model}}^{B, A}$ be the model contrast under Scene B.
If the model's internal contrast veridically tracks the physical target inversion, it must invert its direction:
$$\Delta\mathbf{g}_{\text{model}}^B = -\Delta\mathbf{g}_{\text{model}}^A$$

The reversal cosine evaluates the cosine similarity between the Scene A contrast and the *negated* Scene B contrast:
$$\text{Reversal Cosine} = \cos(\Delta\mathbf{g}_{\text{model}}^A, -\Delta\mathbf{g}_{\text{model}}^B) = \frac{\Delta\mathbf{g}_{\text{model}}^A \cdot (-\Delta\mathbf{g}_{\text{model}}^B)}{\|\Delta\mathbf{g}_{\text{model}}^A\| \|\Delta\mathbf{g}_{\text{model}}^B\|} = -\cos(\Delta\mathbf{g}_{\text{model}}^A, \Delta\mathbf{g}_{\text{model}}^B)$$

### Interpretation of Values:
1. **Ideal Reversal ($\Delta\mathbf{g}_{\text{model}}^B = -\Delta\mathbf{g}_{\text{model}}^A$)**:
   $$\cos(\Delta\mathbf{g}_{\text{model}}^A, -\Delta\mathbf{g}_{\text{model}}^B) = \mathbf{+1.0}\quad (\theta = 0^\circ)$$
2. **Orthogonal / Unrelated Outputs**:
   $$\cos(\Delta\mathbf{g}_{\text{model}}^A, -\Delta\mathbf{g}_{\text{model}}^B) = \mathbf{0.0}\quad (\theta = 90^\circ)$$
3. **Failure to Reverse / Same Orientation Retained ($\Delta\mathbf{g}_{\text{model}}^B = +\Delta\mathbf{g}_{\text{model}}^A$)**:
   $$\cos(\Delta\mathbf{g}_{\text{model}}^A, -\Delta\mathbf{g}_{\text{model}}^B) = \mathbf{-1.0}\quad (\theta = 180^\circ)$$

### Observed Empirical Result:
Evaluating all 60 independent scene pairs:
- **Mean Reversal Cosine**: **$-0.7119$** (95% CI: $[-0.8045, -0.6055]$, median $-0.8804$).
- **Mean Angular Deviation**: **$143.50^\circ$** (median $151.69^\circ$).
- **Direct Cosine**: $\cos(\Delta\mathbf{g}_{\text{model}}^A, \Delta\mathbf{g}_{\text{model}}^B) = \mathbf{+0.7119} > 0$.

### Definitive Directional Conclusion:
The negative reversal cosine ($-0.7119$) proves that H-JEPA's output contrast retains substantially the **same orientation** across scenes ($\Delta\mathbf{g}_{\text{model}}^A \cdot \Delta\mathbf{g}_{\text{model}}^B = +0.7119$). The model contrast **fails to reverse** with the physical target swap. Previous descriptions claiming $-0.7119$ demonstrated "successful reversal" or "sensitivity to physical scene inversion" were mathematically erroneous and are formally rectified.

Furthermore, stale machine-readable values (`-0.0620` and `93.5 deg`) that were inadvertently retained in `task_conditioned_circuits_v5_4_ground_truth_summary.json` have been removed and replaced with the verified values ($-0.7119$ and $143.50^\circ$).

---

## 3. Ground-Truth Task Waypoint Geometry & Standoff Offset

In Project Hephaestus, the navigation agent is trained to navigate to an interaction waypoint located in front of the target, rather than colliding with the target center. In `run_task_conditioned_circuits_v5_3.py:193`, the ground-truth target waypoint $\mathbf{w}$ is generated as:
$$\mathbf{w} = \mathbf{p}_{\text{obj}} - 0.60 \frac{\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{robot}}}{\|\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{robot}}\|}$$

### Geometric Properties:
- $\mathbf{p}_{\text{obj}}$: 2D arena coordinates of the target object center ($m$).
- $\mathbf{p}_{\text{robot}}$: 2D arena coordinates of the robot base center ($m$).
- **Standoff Distance**: Exactly **$0.60$ m** along the line of sight toward the target object.
- **Body-Frame Representation**: The waypoint $\mathbf{w}$ is subsequently transformed into the robot's local body frame via the robot's yaw orientation $\theta_{\text{robot}}$:
  $$\mathbf{g}_{\text{gt}} = R(-\theta_{\text{robot}}) (\mathbf{w} - \mathbf{p}_{\text{robot}})$$

All Phase 5.4 and 5.4.1 ground-truth metrics evaluate prediction accuracy relative to $\mathbf{g}_{\text{gt}}$. The target is therefore correctly referred to as the **ground-truth task waypoint**, not the physical object center.

---

## 4. Pair-Level Pooled Candidate Intervention Metrics

Following strict statistical principles, all intervention summaries are aggregated at the independent scene-pair level ($N=60$), ensuring each pair constitutes an independent unit of inference.

Across the 60 independent pairs:
- **Ground-Truth Contrast Transfer ($\eta_{\text{gt}}$)**:
  - Mean: **$-0.0691$**
  - Median: **$-0.0806$**
  - Pair-Cluster 95% Bootstrap CI: **$[-0.1072, -0.0308]$**
  - *Interpretation*: Hidden activation transplants transfer negative ground-truth contrast, pointing away from the true target displacement.
- **Euclidean Error Improvement ($\Delta E$)**:
  - Mean: **$+0.0344$ m** ($3.4$ cm out of $48.8$ cm baseline error)
  - Median: **$+0.0409$ m**
  - Pair-Cluster 95% Bootstrap CI: **$[+0.0183, +0.0502]$ m**
- **Proportion of Positive Error Improvement**:
  - Mean: **$54.0\%$**
  - Median: **$52.5\%$**
  - Pair-Cluster 95% Bootstrap CI: **$[51.1\%, 56.9\%]$**
- **Internal Model Contrast Transfer ($\eta_{\text{model}}$)**:
  - Mean: **$+0.4134$**
  - Median: **$+0.3927$**
  - Pair-Cluster 95% Bootstrap CI: **$[+0.3759, +0.4545]$**

### Mechanistic Dissociation: $\Delta E > 0$ vs $\eta_{\text{gt}} < 0$:
A critical insight is why Euclidean error improvement can be slightly positive ($+0.0344$ m) while contrast projection is negative ($\eta_{\text{gt}} = -0.0691$):
- Euclidean error improvement $\Delta E = \|\mathbf{g}_{\text{base}} - \mathbf{g}_{\text{donor}}\|_2 - \|\mathbf{g}_{\text{trans}} - \mathbf{g}_{\text{donor}}\|_2$ measures scalar distance reduction. A general contraction of output vector norm toward the origin or toward the training dataset center reduces Euclidean error to any target within that hemisphere.
- Contrast projection $\eta_{\text{gt}} = \frac{\Delta\mathbf{g}_{\text{trans}} \cdot \mathbf{d}_{\text{true}}}{\|\mathbf{d}_{\text{true}}\|^2}$ strictly evaluates directional movement along the true target axis $\mathbf{d}_{\text{true}} = \mathbf{g}_{\text{donor\_gt}} - \mathbf{g}_{\text{recip\_gt}}$.
- The negative $\eta_{\text{gt}}$ proves that the small metric error reduction is an artifact of norm contraction, not purposeful directional steering toward the donor target.

---

## 5. Complete-Support & Incomplete-Support Stratification

To eliminate sampling bias, all statistical comparisons between candidate units and strictly norm-matched random controls are computed strictly on **identical pair IDs** (`paired_diff = cand_eligible - norm_eligible`).

### Complete-Support Settings (Coverage = 100.0%, 60/60 Pairs):
Because these analyses evaluate ground-truth metrics post-hoc on existing Phase 5.3 data, their formal classification is **`EXPLORATORY_COMPLETE_SUPPORT`** (do not use `CONFIRMATORY_EVALUATED`).

To control family-wise error rate across the family of 4 complete-support comparisons, Holm-Bonferroni adjusted p-values are reported:
1. **Layer 1, $k=2$**:
   - Contrast Transfer $\eta_{\text{gt}}$: Candidate $-0.021$ vs Norm-Matched $-0.039$ ($\Delta = +0.018$, Wilcoxon $p = 0.5461$, Holm $p_{\text{adj}} = 1.0000$).
   - Error Improvement: Candidate $+0.005$ m vs Norm-Matched $+0.016$ m ($\Delta = -0.012$ m, Wilcoxon $p = 0.1169$, Holm $p_{\text{adj}} = 0.4675$).
2. **Layer 1, $k=4$**:
   - Contrast Transfer $\eta_{\text{gt}}$: Candidate $-0.042$ vs Norm-Matched $-0.038$ ($\Delta = -0.004$, Wilcoxon $p = 0.8946$, Holm $p_{\text{adj}} = 1.0000$).
   - Error Improvement: Candidate $+0.027$ m vs Norm-Matched $+0.034$ m ($\Delta = -0.007$ m, Wilcoxon $p = 0.7628$, Holm $p_{\text{adj}} = 1.0000$).

**Verdict**: In neither complete-support setting do candidate units show statistically significant superiority over norm-matched random controls (all Holm $p_{\text{adj}} \ge 0.4675$). Confidence intervals do not establish equivalence, as no formal equivalence margin (TOST) was pre-specified.

### Incomplete-Support Settings (Coverage < 100%, 8/10 Settings):
In the remaining 8 settings (Layer 1 $k \in \{8, 16, 32\}$; Layer 2 all $k \in \{2, 4, 8, 16, 32\}$), common support is incomplete (coverage $51.7\%$ to $93.3\%$). All comparisons in these settings are explicitly designated **`EXPLORATORY_INCOMPLETE_SUPPORT`**.
