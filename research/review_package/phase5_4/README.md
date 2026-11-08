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

To answer this, Phase 5.4 executed a post-hoc reanalysis of the complete Phase 5.3 dataset (60 independent cross-context pairs, 120 scenes, 447,800 intervention records) without modifying any historical data, retraining H-JEPA, or regenerating simulator scenes.

### Core Discoveries:

1. **Orthogonality of Model Contrast and Physical Target Contrast ($90.6^\circ$ Divergence)**:
   - While H-JEPA achieves a mean body-frame spatial prediction error of **0.4881 m** (95% CI: $[0.4607, 0.5162]$ m) across the 120 evaluation scenes, its internal task-contrast vector ($\Delta\mathbf{g}_{\text{model}} = \mathbf{g}_{\text{model}}^B - \mathbf{g}_{\text{model}}^A$) is **nearly orthogonal** to the true physical displacement between Target A and Target B ($\Delta\mathbf{g}_{\text{gt}} = \mathbf{g}_{\text{gt}}^B - \mathbf{g}_{\text{gt}}^A$).
   - Mean cosine alignment between $\Delta\mathbf{g}_{\text{model}}$ and $\Delta\mathbf{g}_{\text{gt}}$ is **$+0.0010$** (95% CI: $[-0.0682, +0.0727]$), corresponding to a mean angular deviation of **$90.63^\circ$** (median $92.08^\circ$).
   - Mean ground-truth contrast projection efficiency of the intact model is negative: **$\eta_{\text{gt}} = -0.1984$** (95% CI: $[-0.3209, -0.0760]$).
   - Mean heading circular error is **$0.6958$ rad ($39.87^\circ$)** (95% CI: $[0.6120, 0.7811]$ rad).
   - Under physical scene swap where physical object coordinates invert ($\Delta\mathbf{g}_{\text{gt}}^B = -\Delta\mathbf{g}_{\text{gt}}^A$), the model output displacement reverses with mean cosine **$-0.7119$** (mean angular deviation $143.50^\circ$), proving directional sensitivity to scene inversion, yet internally skewed away from the true physical target displacement axis.

2. **Dissociation of Hidden-Unit Transplant Outcomes**:
   - In Phase 5.3, candidate unit transplantation produced substantial transfer of the model's internal contrast ($\eta_{\text{model}} \approx +0.413$).
   - In Phase 5.4, evaluating those identical 447,800 interventions against true body-frame target coordinates reveals that candidate transplantation produces **negative ground-truth contrast transfer**: mean $\eta_{\text{gt}} = -0.0691$ (95% CI: $[-0.0905, -0.0487]$).
   - The mean error improvement toward the donor target across all candidate interventions is only **$+0.0344$ m** (95% CI: $[0.0210, 0.0480]$ m), and error improves in only **$54.0\%$** of intervention instances.
   - Hidden-layer transplantation reproduces the model's idiosyncratic internal representation shift, but that shift does not mediate steering toward the physical target.

3. **No Evidence of Candidate Superiority at Complete Support; Equivalence Not Established**:
   - In the only two settings with 100% complete common support across all 60 independent pairs:
     - **Layer 1, $k=2$**: Candidate $\eta_{\text{gt}} = -0.021$ vs Norm-Matched Random $\eta_{\text{gt}} = -0.039$ ($\Delta = +0.018$, 95% CI: $[-0.007, +0.046]$, Wilcoxon $p = 0.2730$). Error improvement: Candidate $+0.005$ m vs Norm-Matched $+0.016$ m ($\Delta = -0.012$ m, 95% CI: $[-0.028, +0.004]$ m, Wilcoxon $p = 0.9416$).
     - **Layer 1, $k=4$**: Candidate $\eta_{\text{gt}} = -0.042$ vs Norm-Matched Random $\eta_{\text{gt}} = -0.038$ ($\Delta = -0.004$, 95% CI: $[-0.036, +0.026]$, Wilcoxon $p = 0.5527$). Error improvement: Candidate $+0.027$ m vs Norm-Matched $+0.034$ m ($\Delta = -0.007$ m, 95% CI: $[-0.027, +0.014]$ m, Wilcoxon $p = 0.6186$).
   - **Methodological Standard**: We do not claim the null hypothesis is supported or that equivalence is proven, because no formal equivalence margin (TOST) was pre-specified. The correct scientific conclusion is: **No evidence of candidate superiority in the two complete-support settings; confidence intervals do not establish equivalence.**

4. **Incomplete Support Settings (8/10)**:
   - For the remaining 8 settings ($k=8, 16, 32$ in Layer 1; all $k \in \{2, 4, 8, 16, 32\}$ in Layer 2), coverage remains incomplete ($51.7\%$ to $93.3\%$). Descriptive metrics on common eligible pairs are reported transparently and explicitly classified as `EXPLORATORY_INCOMPLETE_SUPPORT`.

---

## 2. Experimental Summary Table (N = 60 Independent Pairs, 120 Scenes)

All pair-level comparisons aggregate outcomes across Scene A and Scene B and across both swap directions ($A \to B$ and $B \to A$), maintaining the independent pair ($N=1$) as the proper unit of analysis.

| Layer | $k$ | Common Support Coverage | Candidate Ground-Truth Transfer $\eta_{\text{gt}}$ | Norm-Matched Ground-Truth Transfer $\eta_{\text{gt}}$ | Paired Diff $\Delta\eta_{\text{gt}}$ [95% CI] | Wilcoxon $p$ ($\eta_{\text{gt}}$) | Candidate Error Improvement (m) | Norm-Matched Error Improvement (m) | Paired Diff Error Imp (m) [95% CI] | Wilcoxon $p$ (Error Imp) | Candidate Model Transfer $\eta_{\text{model}}$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Layer 1** | 2 | **100.0%** (60/60) | -0.021 | -0.039 | +0.018 [-0.007, +0.046] | 0.2730 | +0.005 | **+0.016** | -0.012 [-0.028, +0.004] | 0.9416 | 0.147 | `CONFIRMATORY_EVALUATED` |
| **Layer 1** | 4 | **100.0%** (60/60) | -0.042 | -0.038 | -0.004 [-0.036, +0.026] | 0.5527 | +0.027 | **+0.034** | -0.007 [-0.027, +0.014] | 0.6186 | 0.295 | `CONFIRMATORY_EVALUATED` |
| **Layer 1** | 8 | 93.3% (56/60) | -0.061 | -0.022 | -0.039 [-0.076, -0.003] | 0.9632* | +0.043 | +0.029 | +0.014 [-0.008, +0.035] | 0.1065* | 0.342 | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 1** | 16 | 76.7% (46/60) | -0.157 | -0.093 | -0.064 [-0.141, +0.013] | 0.9324* | +0.062 | +0.031 | +0.031 [+0.005, +0.057] | 0.0101* | 0.570 | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 1** | 32 | 61.7% (37/60) | -0.149 | -0.144 | -0.005 [-0.089, +0.079] | 0.5098* | +0.033 | +0.002 | +0.031 [-0.004, +0.067] | 0.0416* | 0.878 | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 2** | 2 | 81.7% (49/60) | +0.000 | -0.011 | +0.011 [-0.008, +0.030] | 0.1652* | +0.001 | +0.015 | -0.014 [-0.033, +0.005] | 0.9255* | -0.011 | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 2** | 4 | 65.0% (39/60) | +0.002 | -0.042 | +0.044 [+0.013, +0.074] | 0.0026* | +0.051 | +0.046 | +0.005 [-0.023, +0.034] | 0.3701* | 0.181 | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 2** | 8 | 66.7% (40/60) | -0.058 | -0.072 | +0.014 [-0.024, +0.053] | 0.2443* | +0.060 | +0.057 | +0.003 [-0.028, +0.034] | 0.4206* | 0.344 | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 2** | 16 | 51.7% (31/60) | -0.067 | -0.117 | +0.050 [-0.024, +0.124] | 0.0965* | +0.041 | +0.051 | -0.010 [-0.044, +0.024] | 0.7088* | 0.564 | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 2** | 32 | 80.0% (48/60) | -0.138 | -0.139 | +0.001 [-0.082, +0.084] | 0.4907* | +0.022 | +0.039 | -0.017 [-0.048, +0.014] | 0.8653* | 0.823 | `EXPLORATORY_INCOMPLETE_SUPPORT` |

*\*Note: p-values in settings with incomplete coverage are strictly exploratory and evaluated only over the common eligible pairs.*

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
