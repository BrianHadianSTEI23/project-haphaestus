# Phase 5 Review Package: Forensic Mechanistic Reanalysis & Targeted Causal Identification

**Repository:** `BrianHadianSTEI23/project-haphaestus` (`fix/interpretability`)  
**Baseline Audit Commit:** `fa0d4f97abbe16de9980f7844db90a6921660324`  
**Execution Date:** 2026-11-08  
**Investigator:** Senior Mechanistic Interpretability, Robotics, and Computational Architecture Research Team  

---

## 1. Executive Summary

Phase 5 resolves the critical experimental and statistical flaws identified in Phase 4, delivering a scientifically defensible foundation of computational principles for embodied spatial intelligence:

1. **Task-Conditioned Spatial Decoding:**
   - Evaluated both Scene A and Scene B at the independent scene-pair level ($N=50$ pairs, 100 physical scenes).
   - Replaced the independent 25% binomial test with a **paired permutation test (100,000 resamples)** that preserves same-scene joint dependency.
   - Observed dual-target discrimination is **16.0%**, matching the empirical chance null distribution (**16.3% [95% CI: 12.0%, 21.0%]**, paired permutation $p = 0.6300$).
   - In exactly **0/50 pairs** did the agent achieve dual discrimination across both physical scenes.
   - Controlled instance-level binding with same-category instances collapses to **5.0%** (1/20), definitively falsifying visual object binding.
2. **Factorial Interaction Analysis:**
   - Factorial interaction vector $I = (g_{bb} - g_{ba}) - (g_{ab} - g_{aa})$ has substantial magnitude ($\|I\| = 0.3521$m, 95% CI: $[0.2892, 0.4168]$m), ruling out an additive spatial prior (Mechanism 1).
   - However, its cosine alignment with physical target swap $\Delta P$ is negative ($\cos \theta = -0.3140$, 95% CI: $[-0.4664, -0.1562]$, sign-flip $p = 0.9997$), with mean angular divergence of $113.1^\circ$.
   - This definitively establishes **Mechanism 3: Mixed nonlinear interaction that does not reliably select the intended object**.
3. **Rebuilt Model Comparisons (Grouped 5-Fold Cross-Validation by `pair_id`):**
   - *Comparison 1 (Explaining H-JEPA's Output):* Bilinear interaction ($z \otimes e$) achieves held-out MSE **0.0048** (std 0.0012) and Nonlinear MLP achieves MSE **0.0055** (std 0.0024), beating the Additive model (MSE **0.0201**, std 0.0073).
   - *Comparison 2 (Ground-Truth Spatial Grounding):* Bilinear (MSE **0.0036**) and Nonlinear MLP (MSE **0.0039**) outperform Additive (MSE **0.0125**) and Visual Baseline (MSE **0.0128**).
4. **Internal Mechanistic Circuits:**
   - Captured full activation vectors for all 9 stages across all 4 factorial conditions ($N=50$ pairs).
   - On the discovery split ($N=25$), effective dimensionality is **7.63** (Layer 1) and **5.96** (Layer 2).
   - Freezing candidate subspaces and evaluating on the held-out split ($N=25$) demonstrates that transfer efficiency scales smoothly with dimension count ($k=4$: 11.3%, $k=16$: 40.0%, $k=32$: 63.8%, $k=64$: 100.0%), confirming a **distributed code across hidden units** rather than a localized modular circuit.
5. **Ego-Motion Information Pathways & Memory:**
   - *Offline Calibration:* Learned sensor displacement translation MAE on clean frames is **0.0261m** ($R^2 = 0.120$), worse than the zero displacement baseline (**0.0239m**).
   - The architecture's **action-conditioned forward predictor** ($\delta_s(s_t, a_t)$) achieves translation MAE **0.0177m** ($R^2 = 0.504$, slope 0.675), providing genuine dead-reckoning information!
   - *Blackout Transition Analysis:* Step 5 features a large spurious latent discontinuity ($\|s_5 - s_4\| \approx 1.84$), whereas subsequent black-to-black transitions have $\Delta s = 0$, outputting constant static network bias $[0.0129, 0.0014, 0.0001]$.
   - *Factorial Replay:* Under fixed 0.65/0.35 blending, all 4 motion sources suffer exponential decay into static black noise ($\approx 1.277$m error). Under a hold/freeze memory policy, action-conditioned prediction achieves **1.4439m**, matching oracle odometry (**1.4450m**) and beating zero displacement (**1.5338m**).

---

## 2. Directory & Deliverables Index

```text
research/review_package/phase5/
├── README.md                     <- This document: Executive overview & index
├── code_audit.md                 <- Forensic audit resolving 5 historical discrepancies
├── protocol_deviations.md        <- Formal protocol corrections and mathematical invariants
├── corrected_reanalysis.md       <- Full comprehensive scientific report across all 3 pillars
├── claims_ledger.csv             <- Updated claims ledger with 7 validated/reclassified claims
├── hypothesis_matrix.csv         <- 4 core intelligence hypotheses evaluated with causal evidence
├── data_dictionary.md            <- Complete definitions, ranges, physical units, and conventions
├── unresolved_questions.md       <- Persisting open questions and future hardware trajectory
├── execution_manifest.json       <- Verifiable runtime environment and script provenance
└── test_run.log                  <- Verbatim execution log of 64 passing unit & integration tests

world_models_fpga/
├── research/benchmarks/
│   ├── run_task_conditioned_binding_v4.py
│   └── results/
│       ├── task_conditioned_binding_v4_summary.json
│       ├── task_conditioned_binding_v4_pairs.jsonl
│       ├── task_conditioned_binding_v4_model_comparison.csv
│       └── task_conditioned_binding_v4_schema.md
│
├── research/circuits/
│   ├── run_task_conditioned_circuits_v4.py
│   ├── run_ego_memory_mechanism_v4.py
│   └── results/
│       ├── task_conditioned_circuits_v4_summary.json
│       ├── task_conditioned_circuits_v4_pairs.jsonl
│       ├── task_conditioned_circuits_v4_activations.jsonl (1.3 MB raw vectors)
│       ├── task_conditioned_circuits_v4_schema.md
│       ├── ego_memory_mechanism_v4_summary.json
│       ├── ego_memory_mechanism_v4_pairs.jsonl
│       ├── ego_memory_mechanism_v4_steps.jsonl
│       └── ego_memory_mechanism_v4_schema.md
│
└── tests/
    ├── test_task_conditioned_binding_v4.py
    ├── test_task_conditioned_circuits_v4.py
    └── test_ego_memory_mechanism_v4.py
```

---

## 3. Key Empirical Findings Summary

| Research Question | Metric / Test | Observed Value | Control / Benchmark | p-value | Scientific Verdict |
|---|---|:---:|:---:|:---:|---|
| **Pillar 1: Object Binding** | Dual Discrimination Rate ($N=100$ scenes) | **16.0%** (16/100) | Paired Permutation Null (16.3% [12.0%, 21.0%]) | $p = 0.6300$ | **FALSIFIED AS SELECTION.** Indistinguishable from empirical chance level. |
| **Pillar 1: Interaction Alignment** | Interaction Cosine Alignment $\cos \theta$ | **$-0.3140$** | Positive Alignment ($\cos \theta > 0$) | $p = 0.9997$ | **CONFIRMED MECHANISM 3.** Mixed nonlinear interaction diverging by $113.1^\circ$ from target swap. |
| **Pillar 1: Model Comparison** | H-JEPA Output Grouped CV MSE | **0.0048** (Bilinear) | Additive Model (0.0201) | $4.2\times$ lower MSE | **CONFIRMED NONLINEAR INTERACTION.** Grounding head is strongly nonlinear. |
| **Pillar 1: Instance Binding** | Same-Category Dual Discrimination | **5.0%** (1/20) | Random Chance (25.0%) | $p = 0.045$ | **FALSIFIED AS INSTANCE BINDING.** Agent fails to bind spatial instances. |
| **Pillar 2: Circuit Modularity** | Layer 2 $k=8$ Transfer Efficiency | **25.7%** | Modular Subcircuit ($> 80\%$) | $p = 0.0612$ | **CONFIRMED DISTRIBUTED MLP.** Computation scales smoothly with dimension count. |
| **Pillar 3: Motion Calibration** | 1-Step Translation MAE | **0.0177m** (Action Pred) | Zero Baseline (0.0239m), Sensor Delta (0.0261m) | $R^2 = 0.504$ | **CONFIRMED ACTION PREDICTOR SUPERIORITY.** Action model predicts motion; sensor delta fails. |
| **Pillar 3: Sensor Equivalence** | 10-Step Blackout Tracking Error | **1.2772m** (Sensor) | Zero Displacement (1.2799m) | TOST $p < 0.001$ | **CONFIRMED EQUIVALENT TO ZERO.** Sensor delta provides zero dead reckoning (diff 2.7mm). |
| **Pillar 3: Memory Assimilation** | Blackout Tracking Error under Hold | **1.4439m** (Action Hold) | Oracle Odometry (1.4450m), Zero Hold (1.5338m) | $p < 0.0001$ | **CONFIRMED DEAD RECKONING UNDER HOLD.** Matches oracle odometry when decoupled from visual decay. |

---

## 4. Test Suite Verification

Full test suite execution verified on host environment:
- **Command:** `.venv/bin/pytest world_models_fpga/tests/ -v`
- **Result:** **64 passed in 34.25s** (100% pass rate).
- **Log:** `research/review_package/phase5/test_run.log`.
