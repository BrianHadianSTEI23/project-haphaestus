# Phase 4 Review Package: Mechanistic Identification of Spatial Decoding and Ego-Motion Memory

**Baseline Audit Commit:** `8229f786309fb6d9ea669e8a783793c92ceca564`  
**Committed Experiment Source:** `8eab3cdd5e90f40fa3a7f33b068f0d455b0e7237`  
**Date:** 2026-11-08  
**Repository:** `BrianHadianSTEI23/project-haphaestus` (`fix/interpretability`)

---

## 1. Executive Summary

Phase 4 executes the preregistered mechanistic benchmarks and surgical internal interventions designed to answer the core scientific questions left open at Phase 3:
1. **Does H-JEPA implement task-conditioned visual object binding, or merely an additive spatial prior?**
   - **Answer:** **Additive spatial prior.** Dual discrimination under identical visual input is only 10.0% (significantly worse than 25% chance level, $p = 0.0133$). Additive task bias model fits held-out scenes with MSE 0.0334 vs 0.1357 for visual geometry selection.
2. **Does the learned `micro_ego_head` perform valid dead-reckoning during visual sensory deprivation?**
   - **Answer:** **No.** Under black-frame dropout, latent delta $\Delta s = 0$, causing the head to output constant static bias $[0.0129, 0.0014, 0.0001]$. Same-trajectory replay across 50 trajectories proves full memory tracking error (0.7047m) is statistically equivalent to zeroing ego motion (0.7082m, diff $-0.0035$m, TOST $p < 0.001$).
3. **Why did frozen memory and mismatched donor memory beat full memory in Phase 3 v2?**
   - **Answer:** Fixed controller blending ($g_t = 0.65 g_{\text{vis}} + 0.35 g_{\text{mem}}$) exponentially erases memory ($0.35^{10} \approx 2.75 \times 10^{-5}$) into arbitrary static black-frame noise. Frozen memory and donor traces bypassed this decay.

---

## 2. Deliverables & Artifact Index

```text
research/review_package/phase4/
├── README.md                 <- This document: Phase 4 overview & index
├── preregistration.json      <- Preregistered hypotheses, sample sizes, and tests
├── phase3_reanalysis.md      <- Mathematical & empirical audit of Phase 3 v2
├── data_dictionary.md        <- Definitions, ranges, and physical units
├── discrepancy_log.md        <- Resolution of contradictions between claims and data
├── claims_evidence.csv       <- Updated claims ledger with Phase 4 evidence
├── interpretation_ready.md   <- Final synthesis and computational principles
└── test_run.log              <- Complete verbatim log of the test suite

world_models_fpga/
├── research/benchmarks/
│   ├── run_task_conditioned_binding_v3.py
│   └── results/
│       ├── task_conditioned_binding_v3_summary.json
│       ├── task_conditioned_binding_v3_pairs.jsonl
│       ├── task_conditioned_binding_v3_traces.npz
│       └── task_conditioned_binding_v3_schema.md
│
├── research/circuits/
│   ├── run_task_conditioned_circuits_v3.py
│   ├── run_ego_memory_mechanism_v3.py
│   └── results/
│       ├── task_conditioned_circuits_v3_summary.json
│       ├── task_conditioned_circuits_v3_pairs.jsonl
│       ├── ego_memory_mechanism_v3_summary.json
│       ├── ego_memory_mechanism_v3_pairs.jsonl
│       ├── ego_memory_mechanism_v3_traces.npz
│       └── ego_memory_mechanism_v3_schema.md
│
└── tests/
    ├── test_task_conditioned_binding_v3.py
    ├── test_task_conditioned_circuits_v3.py
    └── test_ego_memory_mechanism_v3.py
```

---

## 3. Key Empirical Results Summary

| Experiment | Unit / Sample Size | Primary Metric | Observed Value | Control / Comparison | p-value | Scientific Verdict |
|---|:---:|---|:---:|:---:|:---:|---|
| **Exp A: Task Binding** | 50 scene pairs (100 scenes) | Dual-target discrimination rate | **10.0%** (5/50) | Random 50/50 Chance (25.0%) | $p = 0.0133$ | **FALSIFIED AS OBJECT BINDING.** Agent fails to bind designated objects from visual geometry. |
| **Exp A: Model Selection** | 20 held-out scenes | Model Held-out MSE | **0.0334** | Visual Selection (0.1357) | $4\times$ lower MSE | **CONFIRMED ADDITIVE BIAS.** Additive prior model outperforms geometric selection. |
| **Exp B: Circuit Modularity** | 30 scene pairs | Layer 1 Donor Transfer | **100.0%** | Random Control (14.4%) | $p = 9.13 \times 10^{-7}$ | **CONFIRMED DISTRIBUTED MLP.** Computation distributed across layers without modular isolation. |
| **Exp C: Ego-Motion Memory** | 50 replay trajectories | Blackout Tracking Error | **0.7047m** | Zeroed Ego-Motion (0.7082m) | TOST $p < 0.001$ | **CONFIRMED EQUIVALENCE TO ZERO.** Learned head provides no benefit over assuming no motion (diff $-3.5$mm). |
| **Exp C: Memory Erasure** | 50 replay trajectories | Blackout Tracking Error | **0.7047m** | Oracle Odometry (0.5581m) | $p = 0.0012$ | **CONFIRMED BLENDING DECAY.** Fixed blending destroys memory; oracle odometry reduces error by $-0.147$m. |
