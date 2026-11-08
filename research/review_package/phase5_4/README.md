# Phase 5.4 Independent Scientific Review Package

## Ground-Truth Spatial Grounding & Causal Intervention Reanalysis

**Repository:** `BrianHadianSTEI23/project-haphaestus`  
**Branch:** `fix/interpretability`  
**Baseline Commit SHA:** `8cb40a31766fed308dcd8e8102516b071b1da940`  
**Execution / Source Commit SHA:** `7d2d7611c970915c029c68eaf38067b860ad9f7e`  
**Timestamp:** `2026-11-08T15:22:43Z`  
**Environment:** Linux x86_64 (`6.8.0-138-generic`), Python 3.13.16, PyTorch 2.1.0, MuJoCo 3.17.0 (`MUJOCO_GL=egl`), NVIDIA RTX 3050 (`cuda`)  
**Test Suite:** 105/105 unit tests passing in 29.53s ([`test_run.log`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_4/test_run.log))  
**Role Assignment:** Experimentalist and implementation engineer. All numerical metrics, geometric measurements, and statistical comparisons are reported transparently for ChatGPT (Senior Independent Scientific Researcher) to audit and interpret.

---

## 1. Executive Summary & Core Scientific Findings

Phase 5.4 was commissioned to resolve the foundational interpretability question arising from Phase 5.3:
> **In Phase 5.3, causal mediation was measured as reproduction of internal model contrast ($\eta_{\text{model}} = \frac{\Delta\mathbf{g}_{\text{trans}} \cdot \Delta\mathbf{g}_{\text{model}}}{\|\Delta\mathbf{g}_{\text{model}}\|^2}$). Does task-conditioned output modulation and hidden-layer transplantation actually steer the predicted egocentric goal vector toward the *true physical target* in body-frame coordinates ($\Delta\mathbf{g}_{\text{gt}}$)?**

To answer this, Phase 5.4 executed a post-hoc reanalysis of the complete Phase 5.3 dataset (60 independent cross-context pairs, 120 scenes, 447,800 intervention records: 2,400 candidate, 240,000 dimension-matched random, 205,400 norm-matched random) without modifying any historical data, retraining H-JEPA, or regenerating simulator scenes.

> [!IMPORTANT]
> **Ground-Truth Target Definition**: The evaluated ground truth is the task-defined standoff goal waypoint $\mathbf{w} = \mathbf{p}_{\text{obj}} - 0.60 \frac{\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{robot}}}{\|\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{robot}}\|}$ located 0.60 m in front of the target object in robot body coordinates, exactly matching the agent's navigation objective, not the physical object center.

### Core Discoveries:

1. **Orthogonality of Model Contrast and Physical Target Contrast ($90.6^\circ$ Divergence)**:
   - While H-JEPA achieves a mean body-frame spatial prediction error of **0.4881 m** (95% CI: $[0.4607, 0.5162]$ m) across the 120 evaluation scenes, its internal task-contrast vector ($\Delta\mathbf{g}_{\text{model}} = \mathbf{g}_{\text{model}}^B - \mathbf{g}_{\text{model}}^A$) is **nearly orthogonal** to the true physical displacement between Target A and Target B standoff waypoints ($\Delta\mathbf{g}_{\text{gt}} = \mathbf{g}_{\text{gt}}^B - \mathbf{g}_{\text{gt}}^A$).
   - Mean cosine alignment between $\Delta\mathbf{g}_{\text{model}}$ and $\Delta\mathbf{g}_{\text{gt}}$ is **$+0.0010$** (95% CI: $[-0.0682, +0.0727]$), corresponding to a mean angular deviation of **$90.63^\circ$** (median $92.08^\circ$).
   - Mean ground-truth contrast projection efficiency of the intact model is negative: **$\eta_{\text{gt}} = -0.1984$** (95% CI: $[-0.3209, -0.0760]$).
   - Mean heading circular error is **$0.6958$ rad ($39.87^\circ$)** (95% CI: $[0.6120, 0.7811]$ rad).
   - Under physical scene swap where physical object coordinates invert ($\Delta\mathbf{g}_{\text{gt}}^B = -\Delta\mathbf{g}_{\text{gt}}^A$), the model output displacement fails to reverse, exhibiting a negative reversal cosine of **$-0.7119$** (mean angular deviation $143.50^\circ$). Because $\cos(\Delta\mathbf{g}_{\text{model}}^A, -\Delta\mathbf{g}_{\text{model}}^B) = -\cos(\Delta\mathbf{g}_{\text{model}}^A, \Delta\mathbf{g}_{\text{model}}^B) = -0.7119$, this negative value demonstrates that the model's internal contrast retains substantially the *same* orientation across scenes ($\Delta\mathbf{g}_{\text{model}}^A \cdot \Delta\mathbf{g}_{\text{model}}^B = +0.7119 > 0$), failing to invert with the physical objects.

2. **Dissociation of Hidden-Unit Transplant Outcomes**:
   - In Phase 5.3, candidate unit transplantation produced substantial transfer of the model's internal contrast ($\eta_{\text{model}} \approx +0.413$).
   - In Phase 5.4, evaluating those identical 447,800 interventions against true body-frame target coordinates reveals that candidate transplantation produces **negative ground-truth contrast transfer**: mean $\eta_{\text{gt}} = -0.0691$ (95% CI: $[-0.1072, -0.0308]$ across 60 independent pairs).
   - The mean error improvement toward the donor target across candidate interventions is only **$+0.0344$ m** (95% CI: $[+0.0183, +0.0502]$ m), and error improves in only **$54.0\%$** of intervention instances (95% CI: $[51.1\%, 56.9\%]$).
   - Hidden-layer transplantation reproduces the model's idiosyncratic internal representation shift, but that shift does not mediate steering toward the physical target.

3. **No Evidence of Candidate Superiority at Complete Support; Equivalence Not Established**:
   - In the only two settings with 100% complete common support across all 60 independent pairs:
     - **Layer 1, $k=2$**: Candidate $\eta_{\text{gt}} = -0.021$ vs Norm-Matched Random $\eta_{\text{gt}} = -0.039$ ($\Delta = +0.018$, Wilcoxon $p = 0.5461$, Holm-adjusted $p_{\text{adj}} = 1.0000$). Error improvement: Candidate $+0.005$ m vs Norm-Matched $+0.016$ m ($\Delta = -0.012$ m, Wilcoxon $p = 0.1169$, Holm-adjusted $p_{\text{adj}} = 0.4675$).
     - **Layer 1, $k=4$**: Candidate $\eta_{\text{gt}} = -0.042$ vs Norm-Matched Random $\eta_{\text{gt}} = -0.038$ ($\Delta = -0.004$, Wilcoxon $p = 0.8946$, Holm-adjusted $p_{\text{adj}} = 1.0000$). Error improvement: Candidate $+0.027$ m vs Norm-Matched $+0.034$ m ($\Delta = -0.007$ m, Wilcoxon $p = 0.7628$, Holm-adjusted $p_{\text{adj}} = 1.0000$).
   - **Methodological Standard**: Because these analyses are post-hoc ground-truth evaluations, complete-support comparisons are classified as `EXPLORATORY_COMPLETE_SUPPORT` (do not use `CONFIRMATORY_EVALUATED`). We do not claim the null hypothesis is supported or that equivalence is proven, because no formal equivalence margin (TOST) was pre-specified. The correct scientific conclusion is: **No evidence of candidate superiority in the two complete-support settings; confidence intervals do not establish equivalence.**

4. **Incomplete Support Settings (8/10)**:
   - For the remaining 8 settings ($k=8, 16, 32$ in Layer 1; all $k \in \{2, 4, 8, 16, 32\}$ in Layer 2), coverage remains incomplete ($51.7\%$ to $93.3\%$). Descriptive metrics on common eligible pairs are reported transparently and explicitly classified as `EXPLORATORY_INCOMPLETE_SUPPORT`.

---

## 2. Experimental Summary Table (N = 60 Independent Pairs, 120 Scenes)

All pair-level comparisons aggregate outcomes across Scene A and Scene B and across both swap directions ($A \to B$ and $B \to A$), maintaining the independent pair ($N=1$) as the proper unit of analysis. Paired differences are computed strictly on identical pair IDs ($\text{Paired Diff} = \text{Cand Eligible} - \text{Norm Eligible}$).

| Layer | $k$ | Common Support Coverage | Candidate Ground-Truth Transfer $\eta_{\text{gt}}$ (All 60 Pairs) | Candidate $\eta_{\text{gt}}$ (Eligible Pairs) | Strict Norm-Matched $\eta_{\text{gt}}$ (Eligible Pairs) | Paired Diff $\Delta\eta_{\text{gt}}$ | Two-Sided $p$ (Holm $p_{\text{adj}}$) | Cand Error Imp (Eligible) | Norm-Matched Error Imp (Eligible) | Paired Diff Error Imp | Two-Sided $p$ (Holm $p_{\text{adj}}$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Layer 1** | 2 | **100.0%** (60/60) | -0.021 | -0.021 | -0.039 | +0.018 | 0.5461 (1.0000) | +0.005m | +0.016m | -0.012m | 0.1169 (0.4675) | `EXPLORATORY_COMPLETE_SUPPORT` |
| **Layer 1** | 4 | **100.0%** (60/60) | -0.042 | -0.042 | -0.038 | -0.004 | 0.8946 (1.0000) | +0.027m | +0.034m | -0.007m | 0.7628 (1.0000) | `EXPLORATORY_COMPLETE_SUPPORT` |
| **Layer 1** | 8 | **93.3%** (56/60) | -0.061 | -0.067 | -0.028 | -0.039 | 0.1334* | +0.043m | +0.029m | +0.014m | 0.1192* | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 1** | 16 | **76.7%** (46/60) | -0.157 | -0.093 | -0.029 | -0.064 | 0.0296* | +0.046m | +0.015m | +0.031m | 0.0000* | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 1** | 32 | **61.7%** (37/60) | -0.149 | -0.237 | -0.232 | -0.005 | 0.6545* | +0.048m | +0.017m | +0.031m | 0.0001* | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 2** | 2 | **81.7%** (49/60) | +0.000 | +0.001 | -0.009 | +0.011 | 0.4834* | +0.001m | +0.015m | -0.014m | 0.0019* | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 2** | 4 | **65.0%** (39/60) | +0.002 | -0.002 | -0.045 | +0.044 | 0.0021* | +0.030m | +0.025m | +0.005m | 0.2645* | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 2** | 8 | **66.7%** (40/60) | -0.058 | -0.070 | -0.085 | +0.014 | 0.9417* | +0.045m | +0.041m | +0.003m | 0.1969* | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 2** | 16 | **51.7%** (31/60) | -0.067 | -0.083 | -0.133 | +0.050 | 0.3570* | +0.040m | +0.050m | -0.010m | 0.8544* | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 2** | 32 | **80.0%** (48/60) | -0.138 | -0.186 | -0.187 | +0.001 | 0.3728* | +0.027m | +0.044m | -0.017m | 0.0005* | `EXPLORATORY_INCOMPLETE_SUPPORT` |

*\*Note: p-values in incomplete support settings are descriptive exploratory comparisons evaluated on common eligible pairs.*

---

## 3. Package Contents

This review package provides complete provenance for independent audit:
- [`README.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_4/README.md): High-level overview, role assignment, and core findings.
- [`forensic_audit.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_4/forensic_audit.md): Deep-dive forensic analysis of model-contrast vs ground-truth spatial dissociation.
- [`data_dictionary.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_4/data_dictionary.md): Comprehensive schema of all variables, coordinate systems, and intervention fields.
- [`protocol_deviations.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_4/protocol_deviations.md): Methodological compliance records and execution constraints.
- [`revised_claims_ledger.csv`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_4/revised_claims_ledger.csv): Updated machine-readable status of hypotheses and claims.
- [`unresolved_hypotheses.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_4/unresolved_hypotheses.md): Explicitly catalogued open scientific questions regarding grounding geometry.
- [`test_run.log`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_4/test_run.log): Complete console output of 105 passing pytest unit tests.
- [`execution_manifest.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_4/execution_manifest.json): Exact SHA-256 hashes of all scripts, input artifacts, output files, and documentation.
