# Phase 5.3 Independent Scientific Review Package

## Valid Norm-Matched Causal Mediation & Necessity Revalidation

**Repository:** `BrianHadianSTEI23/project-haphaestus`  
**Branch:** `fix/interpretability`  
**Execution Commit SHA:** `7c30728179b2bd04778219f1f2b9746ba9dcedd7`  
**Parent / Baseline Commit SHA:** `6bda2a8ecfd4d2a21e0353e1fb7bc45cba1f9518`  
**Timestamp:** `2026-11-08T14:41:34Z`  
**Environment:** Linux x86_64 (`6.8.0-138-generic`), Python 3.13.16, PyTorch 2.1.0, MuJoCo 3.17.0 (`MUJOCO_GL=egl`), NVIDIA RTX 3050 (`cuda`)  
**Test Suite:** 95/95 unit tests passing in 32.62s (`test_run.log`)  
**Role Assignment:** Experimentalist and implementation engineer. Raw numerical data, controlled interventions, and statistical tests are reported transparently for ChatGPT (Senior Independent Scientific Researcher) to audit and interpret.

---

## 1. Executive Summary & Scientific Findings

Phase 5.3 was commissioned to resolve the critical flaw uncovered in Phase 5.2:
> **In Phase 5.2, when fewer than 100 random controls had activation perturbation norms within 5% tolerance of the candidate subset, the nearest 100 subsets from the general pool were retained regardless of error. Did this compromise the conclusion of "Selective Subspace Confirmed"?**

### Core Findings Under Strict Norm-Matching Without Nearest-Outside Fallback:

1. **Phase 5.2 "Selective Subspace Confirmed" Claim is FORMALLY WITHDRAWN**:
   - In Phase 5.2, across multiple settings (especially $k=32$ in Layer 2 and smaller $k$ in Layer 1), many retained "norm-matched" controls had relative errors substantially exceeding 5% (up to 15-30% error).
   - When strict filtering is enforced ($\text{rel\_error} \le 0.05$) with adaptive sampling up to 200,000 draws:
     - **8 out of 10 primary settings exhibit `INSUFFICIENT_COMMON_SUPPORT`** (< 100% pair coverage).
     - At $k=8, 16, 32$ in Layer 1 and all $k \in \{2, 4, 8, 16, 32\}$ in Layer 2, the candidate subset's task-induced activation perturbation norm lies in an extreme tail that random unit subsets cannot reliably match within 5% tolerance across all 60 independent pairs.

2. **Complete Common Support Settings ($k=2, 4$ in Layer 1) Show ZERO Candidate Advantage**:
   - In the only two settings where 100% of pairs had $\ge 100$ valid norm-matched controls:
     - **Layer 1, $k=2$**: Candidate transfer efficiency = 0.147 vs Norm-Matched Random = 0.179 ($\Delta = -0.031$, $p_{\text{adj}} = 1.0000$). Candidate condition-averaged necessity drop = 0.020 vs Norm-Matched Random = 0.061.
     - **Layer 1, $k=4$**: Candidate transfer efficiency = 0.295 vs Norm-Matched Random = 0.337 ($\Delta = -0.041$, $p_{\text{adj}} = 1.0000$). Candidate condition-averaged necessity drop = 0.105 vs Norm-Matched Random = 0.128.
   - **Conclusion**: When activation perturbation magnitude is strictly equalized, candidate units do not exhibit greater causal sufficiency or causal necessity than arbitrary random unit subsets. The apparent advantage over unconstrained random subsets is an activation-magnitude artefact.

3. **Mechanistic Verdict**:
   - **`Unidentifiable Under Current Representation`**: 8/10 settings exhibit `INSUFFICIENT_COMMON_SUPPORT`.
   - In those settings, the candidate units occupy a perturbation-magnitude regime where uniform random sampling cannot provide valid norm-matched counterfactuals within 5% tolerance. The effect is fundamentally unidentifiable as functional selectivity versus magnitude modulation using this control design.

---

## 2. Experimental Summary Table (N = 60 Independent Pairs, 120 Scenes)

| Layer | $k$ | Common Support Coverage | Candidate Transfer $\eta$ | Unconstrained Random $\eta$ | Strict Norm-Matched $\eta$ | Overlap-Standardized $\eta$ | Paired Diff $\Delta\eta$ | Holm-Bonferroni $p_{\text{adj}}$ | Candidate Drop | Norm-Matched Drop | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Layer 1** | 2 | **100.0%** (60/60) | 0.147 | 0.033 | **0.179** | 0.192 | -0.031 | 1.0000 | 0.020 | **0.061** | `CONFIRMATORY_EVALUATED` (Null Supported) |
| **Layer 1** | 4 | **100.0%** (60/60) | 0.295 | 0.057 | **0.337** | 0.328 | -0.041 | 1.0000 | 0.105 | **0.128** | `CONFIRMATORY_EVALUATED` (Null Supported) |
| **Layer 1** | 8 | 93.3% (56/60) | 0.342 | 0.127 | 0.453 | 0.447 | N/A | N/A | 0.176 | 0.147 | `INSUFFICIENT_COMMON_SUPPORT` |
| **Layer 1** | 16 | 76.7% (46/60) | 0.570 | 0.251 | 0.682 | 0.720 | N/A | N/A | 0.446 | 0.269 | `INSUFFICIENT_COMMON_SUPPORT` |
| **Layer 1** | 32 | 61.7% (37/60) | 0.878 | 0.497 | 0.863 | 0.881 | N/A | N/A | 0.693 | 0.478 | `INSUFFICIENT_COMMON_SUPPORT` |
| **Layer 2** | 2 | 81.7% (49/60) | -0.011 | 0.030 | 0.052 | 0.094 | N/A | N/A | -0.025 | 0.036 | `INSUFFICIENT_COMMON_SUPPORT` |
| **Layer 2** | 4 | 65.0% (39/60) | 0.181 | 0.062 | 0.173 | 0.174 | N/A | N/A | 0.138 | 0.131 | `INSUFFICIENT_COMMON_SUPPORT` |
| **Layer 2** | 8 | 66.7% (40/60) | 0.344 | 0.124 | 0.305 | 0.324 | N/A | N/A | 0.249 | 0.221 | `INSUFFICIENT_COMMON_SUPPORT` |
| **Layer 2** | 16 | 51.7% (31/60) | 0.564 | 0.248 | 0.516 | 0.529 | N/A | N/A | 0.380 | 0.374 | `INSUFFICIENT_COMMON_SUPPORT` |
| **Layer 2** | 32 | 80.0% (48/60) | 0.823 | 0.504 | 0.799 | 0.805 | N/A | N/A | 0.709 | 0.658 | `INSUFFICIENT_COMMON_SUPPORT` |

---

## 3. Package Contents

This review package provides complete cryptographic and data provenance for independent audit:
- [`README.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_3/README.md): High-level overview, role assignment, and core findings.
- [`forensic_audit.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_3/forensic_audit.md): Deep-dive forensic breakdown of Phase 5.2 invalid matching flaw and formal withdrawal of selective subspace claims.
- [`protocol_deviations.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_3/protocol_deviations.md): Complete list of methodological corrections and compliance records.
- [`execution_manifest.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_3/execution_manifest.json): Exact SHA-256 hashes of all scripts, checkpoints, raw artifacts, and tests.
- [`data_dictionary.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_3/data_dictionary.md): Comprehensive schema of all variables, coordinate frames, and intervention records.
- [`revised_claims_ledger.csv`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_3/revised_claims_ledger.csv): Updated status of hypotheses H1-H7 and intelligence principles E1-E3.
- [`unresolved_hypotheses.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_3/unresolved_hypotheses.md): Explicitly catalogued open scientific questions and bounds of common support.
- [`test_run.log`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_3/test_run.log): Complete console output of 95 passing pytest unit tests.
