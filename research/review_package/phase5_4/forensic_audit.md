# Phase 5.4 Forensic Audit & Mechanistic Dissection

## Separation of Model-Contrast Mediation from Ground-Truth Spatial Mediation

**Audit Scope:** Verification of egocentric spatial accuracy, task-contrast geometry, circular heading errors, and intervention efficacy against ground-truth physical target coordinates in Project Hephaestus Phase 5.4.  
**Auditor Role:** Experimentalist & Implementation Engineer.  
**Scientific Target:** Independent evaluation and interpretation by ChatGPT (Senior Independent Scientific Researcher).

---

## 1. The Core Scientific Puzzle: Model Contrast vs Ground-Truth Target

In Phase 5.3, causal mediation was formalized as the projection of the intervention displacement onto the model's own contrast:
$$\eta_{\text{model}} = \frac{(\mathbf{g}_{\text{trans}} - \mathbf{g}_{\text{base}}) \cdot (\mathbf{g}_{\text{donor}} - \mathbf{g}_{\text{base}})}{\|\mathbf{g}_{\text{donor}} - \mathbf{g}_{\text{base}}\|^2}$$

Under this internal definition:
- Candidate units at $k=32$ in Layer 1 achieved $\eta_{\text{model}} = 0.878$ (transferring $87.8\%$ of the model's internal contrast).
- Even at $k=4$ in Layer 1, candidate units achieved $\eta_{\text{model}} = 0.295$.
- Across all candidate interventions, mean $\eta_{\text{model}} = 0.4134$.

This created a critical interpretability question: **Does transplanting these hidden activations actually move the predicted goal toward the physical target object in body space, or does it merely reproduce an idiosyncratic internal shift within the model's output space?**

To determine this, Phase 5.4 evaluated the identical 447,800 interventions directly against ground-truth body-relative coordinates $\mathbf{g}_{\text{gt}} = [x_{\text{gt}}, y_{\text{gt}}]$ obtained from the MuJoCo simulator.

---

## 2. Full-Model Spatial Geometry: The Orthogonality Discovery

> [!IMPORTANT]
> **Ground-Truth Target Definition**: In all analyses below, the ground truth evaluated is the task-defined standoff goal waypoint $\mathbf{w} = \mathbf{p}_{\text{obj}} - 0.60 \frac{\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{robot}}}{\|\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{robot}}\|}$ located 0.60 m in front of the target object in robot body coordinates, exactly reflecting the agent's navigational objective, not the physical object center.

Evaluating H-JEPA's unperturbed predictions across all 120 evaluation scenes (60 independent pairs) revealed an unexpected geometric phenomenon:

### A. Intact Model Accuracy & Error Metrics:
- **Mean Spatial Error**: **0.4881 m** (95% CI: $[0.4607, 0.5162]$ m, median $0.4729$ m).
- **Task A Spatial Error**: Mean $0.4739$ m (median $0.4606$ m).
- **Task B Spatial Error**: Mean $0.5023$ m (median $0.4871$ m).
- **Heading Circular Error**: Mean **$0.6958$ rad ($39.87^\circ$)** (95% CI: $[0.6120, 0.7811]$ rad).

### B. Contrast Vector Alignment:
Let $\Delta\mathbf{g}_{\text{gt}} = \mathbf{g}_{\text{gt}}^B - \mathbf{g}_{\text{gt}}^A$ represent the true physical offset between Target A and Target B in the robot's body frame, and let $\Delta\mathbf{g}_{\text{model}} = \mathbf{g}_{\text{model}}^B - \mathbf{g}_{\text{model}}^A$ represent the shift in H-JEPA's output when the task token is swapped under identical visual input.
- **Mean Cosine Alignment**: **$+0.0010$** (95% CI: $[-0.0682, +0.0727]$, median $-0.0209$).
- **Mean Angular Deviation**: **$90.63^\circ$** (95% CI: $[86.29^\circ, 94.98^\circ]$, median $92.08^\circ$).
- **Ground-Truth Projection Efficiency ($\eta_{\text{gt}}$)**: **$-0.1984$** (95% CI: $[-0.3209, -0.0760]$, median $-0.2238$).

```
         ^ +X (Forward)
         |
         |         g_model^B
         |          *
         |         /
         |        /  Delta g_model (nearly lateral / orthogonal)
         |       /
         |      * g_model^A
         |
         |                      * g_gt^B (True Target B)
         |                     /
         |                    /   Delta g_gt (Physical offset)
         |                   /
         |                  * g_gt^A (True Target A)
         +---------------------------------------------> +Y (Left)
                      Robot Body Origin
```

**Mechanistic Finding:** The displacement vector induced in H-JEPA's output by swapping the task token is **statistically orthogonal ($90.6^\circ$)** to the true physical displacement vector between the two visible objects. The model modulates its output in response to the task token, but the direction of this modulation does not align with the true physical axis separating the targets.

### C. Scene-Swap Reversal Invariants:
In each paired trial, Scene B physically swaps the positions of Target A and Target B, which mathematically guarantees:
$$\Delta\mathbf{g}_{\text{gt}}^B = -\Delta\mathbf{g}_{\text{gt}}^A$$
If H-JEPA's task contrast vector veridically tracked target swapping, its output displacement would also invert ($\Delta\mathbf{g}_{\text{model}}^B = -\Delta\mathbf{g}_{\text{model}}^A$), yielding an ideal reversal cosine of $\cos(\Delta\mathbf{g}_{\text{model}}^A, -\Delta\mathbf{g}_{\text{model}}^B) = +1.0$.
Evaluating the observed model contrast across scenes:
- **Model Scene-Swap Reversal Cosine**: Mean **$-0.7119$** (95% CI: $[-0.8045, -0.6055]$, median $-0.8804$).
- **Angular Deviation from Reversal**: Mean **$143.50^\circ$** (median $151.69^\circ$).

**Directional Interpretation**: Because $\cos(\Delta\mathbf{g}_{\text{model}}^A, -\Delta\mathbf{g}_{\text{model}}^B) = -\cos(\Delta\mathbf{g}_{\text{model}}^A, \Delta\mathbf{g}_{\text{model}}^B) = -0.7119$, the positive direct cosine $\cos(\Delta\mathbf{g}_{\text{model}}^A, \Delta\mathbf{g}_{\text{model}}^B) = +0.7119$ demonstrates that H-JEPA's output contrast retains substantially the *same* orientation across scenes, **failing to reverse** with the physical target swap. This disconfirms sensitivity to physical target coordinate inversion under scene swap.

---

## 3. Forensic Analysis of Causal Interventions

When hidden-unit activations are transplanted from a donor forward pass into a recipient forward pass:

### A. Contrast Transfer Disconnect:
- In Phase 5.3: Candidate units achieved mean $\eta_{\text{model}} = +0.4134$.
- In Phase 5.4: Evaluating those exact same interventions against the donor's physical target yields mean $\eta_{\text{gt}} = \mathbf{-0.0691}$ (95% CI: $[-0.0905, -0.0487]$).
- Candidate unit transplantation projects in the **negative direction** relative to the true physical target displacement $\mathbf{d}_{\text{true}} = \mathbf{g}_{\text{donor\_gt}} - \mathbf{g}_{\text{recip\_gt}}$.

### B. Metric Error Improvement Toward Donor Target:
For each intervention, we define:
$$\text{baseline\_error} = \|\mathbf{g}_{\text{base}} - \mathbf{g}_{\text{recip\_gt}}\|_2$$
$$\text{transplant\_error} = \|\mathbf{g}_{\text{trans}} - \mathbf{g}_{\text{donor\_gt}}\|_2$$
$$\text{error\_improvement} = \text{baseline\_error} - \text{transplant\_error}$$

Across all 2,400 candidate interventions ($60 \text{ pairs} \times 2 \text{ scenes} \times 2 \text{ layers} \times 5\text{ }k\text{'s} \times 2 \text{ directions} \times 1 \text{ candidate draw}$):
- **Mean Error Improvement**: only **$+0.0344$ m** ($3.4$ cm, 95% CI: $[+0.0183, +0.0502]$ m).
- **Proportion Positive**: only **$54.0\%$** of candidate interventions actually reduce spatial error toward the donor target (95% CI: $[51.1\%, 56.9\%]$).
- **Mean Baseline Error**: $0.4881$ m.
- **Mean Transplant Error**: $0.4537$ m.

Transplanting candidate activations barely nudges the predicted goal closer to the donor object ($+3.4$ cm out of $\approx 49$ cm baseline error).

### C. Complete Common Support Evaluations (Exploratory Post-Hoc Status):
Because these analyses evaluate ground-truth metrics post-hoc on existing Phase 5.3 intervention records, the complete-support comparisons are classified as `EXPLORATORY_COMPLETE_SUPPORT` (do not use `CONFIRMATORY_EVALUATED`). In the two settings with 100% complete common support across all 60 pairs:

1. **Layer 1, $k=2$ ($N=60$ pairs)**:
   - Candidate $\eta_{\text{gt}} = -0.021$ vs Strict Norm-Matched Random $\eta_{\text{gt}} = -0.039$.
   - Paired Difference $\Delta\eta_{\text{gt}} = +0.018$ (Two-sided Wilcoxon $p = 0.5461$, Holm-adjusted $p_{\text{adj}} = 1.0000$).
   - Candidate Error Improvement $= +0.005$ m vs Norm-Matched Error Improvement $= \mathbf{+0.016}$ m.
   - Paired Difference in Error Improvement $= -0.012$ m (Two-sided Wilcoxon $p = 0.1169$, Holm-adjusted $p_{\text{adj}} = 0.4675$).

2. **Layer 1, $k=4$ ($N=60$ pairs)**:
   - Candidate $\eta_{\text{gt}} = -0.042$ vs Strict Norm-Matched Random $\eta_{\text{gt}} = -0.038$.
   - Paired Difference $\Delta\eta_{\text{gt}} = -0.004$ (Two-sided Wilcoxon $p = 0.8946$, Holm-adjusted $p_{\text{adj}} = 1.0000$).
   - Candidate Error Improvement $= +0.027$ m vs Norm-Matched Error Improvement $= \mathbf{+0.034}$ m.
   - Paired Difference in Error Improvement $= -0.007$ m (Two-sided Wilcoxon $p = 0.7628$, Holm-adjusted $p_{\text{adj}} = 1.0000$).

**Definitive Statistical Verdict:** In neither setting do candidate units show statistically significant superiority over norm-matched random controls in steering toward the ground-truth physical target (all Holm $p_{\text{adj}} \ge 0.4675$). Furthermore, norm-matched random controls achieve slightly *larger* mean error reductions ($+1.6$ cm and $+3.4$ cm) than candidate units ($+0.5$ cm and $+2.7$ cm). Confidence intervals do not establish equivalence, as no pre-specified equivalence margin was defined.

---

## 4. Context Stratification: Robustness Across Environmental Shifts

The orthogonality and lack of ground-truth mediation hold identically across both nominal and illumination-shifted visual contexts:

| Metric | Nominal Context ($N=30$ pairs) | Illumination Shift ($N=30$ pairs) | Pooled Total ($N=60$ pairs) |
| :--- | :---: | :---: | :---: |
| **Mean Spatial Error** | 0.4829 m | 0.4934 m | **0.4881 m** |
| **Mean $\eta_{\text{gt}}$ (Full Model)** | -0.1850 | -0.2119 | **-0.1984** |
| **Mean Cosine Alignment** | +0.0197 | -0.0177 | **+0.0010** |
| **Mean Heading Error** | 0.7178 rad ($41.1^\circ$) | 0.6738 rad ($38.6^\circ$) | **0.6958 rad ($39.9^\circ$)** |
| **Scene-Swap Reversal Cosine** | -0.6735 | -0.7503 | **-0.7119** |

Environmental illumination changes do not disrupt or alter the underlying geometry: the model's inductive bias is structural to its visual grounding weights, not a transient lighting artifact.

---

## 5. Summary of Scientific Rectifications

1. **Separation of Metrics**: All future interpretability reporting must strictly differentiate:
   - `model_contrast_efficiency` ($\eta_{\text{model}}$): reproduction of internal model output shift.
   - `ground_truth_contrast_efficiency` ($\eta_{\text{gt}}$): projection onto true physical target displacement.
   - `error_improvement`: metric reduction in body-frame distance to true target ($m$).
2. **Falsification of Ground-Truth Circuit Steering**: The claim that task-sensitive hidden units form an autonomous circuit mediating ground-truth spatial steering is **completely falsified**. Candidate unit transplantation reproduces internal model contrast without steering toward the physical target.
3. **Formal Statistical Status**: For Layer 1 $k=2, 4$, the hypothesis of candidate superiority over norm-matched random controls is rejected; confidence intervals are reported without claiming proven equivalence.
