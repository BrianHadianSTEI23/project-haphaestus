# Phase 5 Code Audit & Reproducibility Verification

**Audit Date:** 2026-11-08  
**Investigator:** Senior Mechanistic Interpretability, Robotics, and Computational Architecture Research Team  
**Repository:** `BrianHadianSTEI23/project-haphaestus` (`fix/interpretability`)  
**Baseline Commit:** `fa0d4f97abbe16de9980f7844db90a6921660324`  

---

## 1. Audit Overview

This audit forensically inspects the Phase 4 code, metadata, execution logs, and result artifacts to resolve five critical discrepancies identified between preregistration, implementation, and scientific claims.

---

## 2. Resolution of the Five Historical Discrepancies

### Discrepancy 1: Preregistered vs Executed Sample Size in Circuit Analysis
- **Observation:** `research/review_package/phase4/preregistration.json` specified `sample_size: 50` for `experiment_B_internal_mechanistic_circuits`, but `task_conditioned_circuits_v3_pairs.jsonl` contains only 30 records, and `summary.json` reports $N=30$.
- **Root Cause:** In `world_models_fpga/research/circuits/run_task_conditioned_circuits_v3.py`, line 384 defined `parser.add_argument("--pairs", type=int, default=30)`. The script was executed without overriding `--pairs 50`, resulting in a premature termination after 30 scene pairs.
- **Correction in Phase 5:** In Phase 5 (`run_task_conditioned_circuits_v4.py`), the sample size is fixed to $N=50$ independent scene pairs (100 physical scenes). The sample is partitioned into a 25-pair discovery split and a 25-pair held-out evaluation split, ensuring that candidate dimensions identified on discovery are strictly frozen prior to held-out causal evaluation.

---

### Discrepancy 2: Model Comparison Sample Count and Formulation Flaws
- **Observation:** `research/review_package/phase4/README.md` claimed "20 held-out scenes" for the model comparison, but `run_task_conditioned_binding_v3.py` lines 597–608 constructed a 50/50 split of all collected samples (`split_idx = n_total_samples // 2`).
- **Additional Methodological Flaws:**
  1. *Unequal Target Comparison:* In v3, Model 1 (Additive) and Model 3 (Ridge) predicted H-JEPA's output $g_{\text{pred}}$, while Model 2 (Selection) evaluated $g_{\text{pred}}$ against ground-truth coordinates $y_{\text{GT}}$, calculating H-JEPA's test error rather than fitting a visual selection surrogate.
  2. *Missing Bilinear Terms:* Model 3 in v3 was simply a linear ridge regression on $[z, \text{onehot}(e)]$ without cross-product interaction terms.
  3. *Un-grouped Split:* The 50/50 split did not group by `pair_id`, allowing conditions from the same scene pair to leak across train and test splits.
- **Correction in Phase 5:** Phase 5 decouples the analysis into two distinct model comparisons:
  - **Comparison 1 (Surrogates of H-JEPA Output):** Fits task-only, visual-only, additive, bilinear ($z \otimes e$), and 2-layer nonlinear MLP surrogates to explain H-JEPA's predicted outputs.
  - **Comparison 2 (Ground-Truth Spatial Grounding):** Fits the identical models to predict ground-truth target coordinates, including task-independent visual and oracle baselines.
  - Both comparisons employ **Grouped 5-Fold Cross-Validation by `pair_id`**, ensuring all 4 factorial conditions of an independent pair reside in the same validation fold.

---

### Discrepancy 3: Execution Commit Provenance
- **Observation:** `research/review_package/phase4/discrepancy_log.md` identified `bbad9132a268a395c3298fb4a08dc3b8edbf30fd` as the execution commit, whereas `research/review_package/phase4/README.md` and experiment metadata identified `8eab3cdd5e90f40fa3a7f33b068f0d455b0e7237`.
- **Root Cause & History:**
  1. Commit `bbad913`: Initial implementation of Phase 4 benchmark scripts, schemas, and tests.
  2. Commit `1cd9db2`: Integrated dynamic git metadata collection (`git rev-parse HEAD`, `git status`).
  3. Commit `8eab3cd`: Bug fix in `model_wrappers.py` and `run_task_conditioned_binding_v3.py` ensuring `compute_body_frame_goal_vec` handles 2D inputs and passes 3D waypoints.
  4. Execution occurred immediately after commit `8eab3cd`. The discrepancy in `discrepancy_log.md` was an un-updated draft reference to the initial commit `bbad913`.
- **Correction in Phase 5:** All Phase 5 scripts dynamically query and log the exact runtime git HEAD, script SHA-256, and checkpoint SHA-256 into all output JSON summaries.

---

### Discrepancy 4: `working_tree_clean: false` Status
- **Observation:** Metadata in Phase 4 JSON summaries reported `"working_tree_clean": false`.
- **Root Cause:** Benchmark scripts were invoked while result files from earlier runs were already present in `results/` as untracked/modified artifacts. A check of `git status --porcelain` during execution returned these output files, marking the tree dirty. No source code was dirty during execution.
- **Correction in Phase 5:** In Phase 5, the execution harness verifies that all code files match committed hashes, and dynamically tracks untracked files to guarantee clean provenance.

---

### Discrepancy 5: Implementation Correctness vs. Scientific Validity
- **Observation:** The project previously cited "55 passing unit tests" as confirmation of its scientific findings.
- **Scientific Principle:** Unit tests confirm implementation invariants (e.g., tensor shapes, absence of NaNs, range bounds, mathematical helper consistency). They do **not** confirm empirical or theoretical hypotheses (e.g., whether the agent performs visual object binding, or whether memory is updated by ego-motion).
- **Enforced Boundary in Phase 5:**
  - Unit tests are strictly used to verify mathematical operations, tensor transformations, and execution pipelines.
  - Scientific claims must be justified exclusively by controlled causal experiments, paired permutation tests, effect sizes, and out-of-distribution generalization.
