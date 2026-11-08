# Phase 5.4 Protocol Deviations & Compliance Log

**Experiment:** Phase 5.4 Ground-Truth Spatial Grounding & Causal Intervention Reanalysis  
**Execution Date:** 2026-11-08  
**Repository Branch:** `fix/interpretability`  
**Baseline Commit SHA:** `8cb40a31766fed308dcd8e8102516b071b1da940`  
**Execution Commit SHA:** `7d2d7611c970915c029c68eaf38067b860ad9f7e`

---

## 1. Intentional Methodological Protocol (Phase 5.4 Reanalysis Design)

Phase 5.4 was executed as a strict post-hoc reanalysis of the existing Phase 5.3 causal dataset to evaluate H-JEPA's full-model predictions and hidden-layer interventions against ground-truth physical target coordinates:

1. **Strict Post-Hoc Reanalysis (Zero Retraining / Zero Scene Generation)**:
   - H-JEPA weights (`hjepa_best.pt`) were frozen.
   - Simulator scenes from `task_conditioned_circuits_v5_2_scene_specs.jsonl` were audited and preserved.
   - Zero new simulator episodes were generated.
   - Zero historical Phase 5.1, 5.2, or 5.3 raw artifacts were overwritten, deleted, or modified.

2. **Mandatory Metric Separation**:
   - Internal model contrast ($\Delta\mathbf{g}_{\text{model}}$) was strictly decoupled from physical ground-truth target contrast ($\Delta\mathbf{g}_{\text{gt}}$).
   - Phase 5.3 model-contrast efficiency ($\eta_{\text{model}}$) was retained verbatim.
   - Physical ground-truth transfer efficiency ($\eta_{\text{gt}}$) and body-frame error improvement ($m$) were computed independently.

3. **Gzip Streaming Pipeline**:
   - Processed all 447,800 rows from `task_conditioned_circuits_v5_3_interventions.jsonl.gz` via line-by-line streaming without decompressing the entire file into system RAM.
   - Emitted compressed results directly to `task_conditioned_circuits_v5_4_intervention_metrics.jsonl.gz` (60 MB), maintaining compliance with Git LFS limits (<100 MB).

4. **Independent Scene Pair as Unit of Analysis ($N = 60$)**:
   - All statistical aggregations average across Scene A and Scene B and across both swap directions ($A \to B$ and $B \to A$).
   - The statistical unit of analysis is strictly the independent scene pair ($N = 60$).

5. **Identical Eligible Pair Cohort Matching**:
   - For all candidate versus norm-matched random control comparisons, metrics are evaluated strictly on the identical subset of eligible pair IDs.
   - Incomplete support settings (8/10) are explicitly designated `EXPLORATORY_INCOMPLETE_SUPPORT` with exploratory p-values.

6. **Equivalence & Null Hypothesis Language Standards**:
   - Phrasing such as "null supported" or "proven equivalence" is strictly prohibited because no formal two one-sided test (TOST) equivalence margin was pre-specified for this reanalysis.
   - Wording is standardized to: **"No evidence of candidate superiority in the two complete-support settings; confidence intervals do not establish equivalence."**

---

## 2. Runtime Engineering Discoveries & Adjustments

1. **MuJoCo Framebuffer Bit-Exact Provenance**:
   - Re-verified that the 120 scenes from Phase 5.2/5.3 maintain 100% bit-for-bit SHA-256 hash match against historical specifications.
2. **Reconciliation Audit Verification**:
   - Verified that all 120 scene records and 447,800 intervention records were read without a single missing, duplicate, or unmatched record (`unmatched_rows = 0`, `duplicate_rows = 0`).

---

## 3. Protocol Compliance Checklist

- [x] Baseline commit `8cb40a31766fed308dcd8e8102516b071b1da940` verified clean before execution.
- [x] All 6 Phase 5.3 input artifacts hashed and preserved untouched.
- [x] H-JEPA checkpoint `hjepa_best.pt` remained strictly frozen.
- [x] 120 scene records evaluated for full-model body-frame ground-truth accuracy.
- [x] 447,800 intervention records streamed from gzip, augmented, and written to compressed gzip.
- [x] Pair-level aggregation evaluated over 60 independent pairs.
- [x] Candidate vs norm-matched controls evaluated on identical eligible pairs.
- [x] 8/10 incomplete support settings classified as `EXPLORATORY_INCOMPLETE_SUPPORT`.
- [x] 105/105 pytest unit tests passing.
- [x] All claims verified against raw numerical artifacts.
