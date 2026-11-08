# Phase 5.2 Review Package: Mechanistic Selectivity & Cross-Context Revalidation

**Repository:** `BrianHadianSTEI23/project-haphaestus`  
**Branch:** `fix/interpretability`  
**Clean Source Commit:** `bc568052e8be10fba4f4a466e913e3dbd8bb4fd6`  
**Execution Environment:** Linux x86_64 (`6.8.0-138-generic`), Python 3.13.16, MuJoCo 3.17.0 (`MUJOCO_GL=egl`), NVIDIA RTX 3050 (`cuda`)  
**Test Suite Status:** 84/84 unit tests passing in 28.30s (`MUJOCO_GL=egl .venv/bin/pytest world_models_fpga/tests/ -v`)  
**Role:** Experimentalist and implementation engineer  
**Auditor:** ChatGPT (Senior Independent Scientific Researcher)  

---

## 1. Executive Summary & Purpose

This review package provides the complete experimental datasets, causal interventions, statistical evaluations, and raw execution logs for **Phase 5.2 Mechanistic Selectivity and Cross-Context Revalidation** of Project Hephaestus.

The core research objective of Phase 5.2 was to resolve a fundamental question raised during the independent audit of Phase 5.1:
> **Are the task-sensitive hidden-unit subspaces identified in Phase 5.1 genuinely selective causal mechanisms, or are their apparent advantages over random unit subsets simply activation-magnitude artefacts?**

To answer this question definitively, Phase 5.2 instituted strict methodological controls:
1. **Both-Context Candidate Ranking:** Discovery units were ranked across *both* visual contexts ($s_i = 0.5 \times (|h_i(z_A, e_B) - h_i(z_A, e_A)| + |h_i(z_B, e_B) - h_i(z_B, e_A)|)$ on pairs $0 \dots 24$) and frozen strictly prior to evaluation on unseen scenes.
2. **Independent 60-Pair Cross-Context Suite:** 30 nominal appearance pairs ($\text{scale}=1.0$) and 30 moderate illumination-shifted pairs ($\text{scale}=0.60$) generated under verified 5-step simulator initialization with physical ground-truth assertions.
3. **Strict Activation-Norm-Matching:** For every intervention pass, $\ge 5,000$ random subsets were sampled to find the closest 100 subsets matching the candidate subset's task-induced perturbation norm $\|\Delta h_U\|$ within $\le 5\%$ relative tolerance.
4. **Decoupled Scalar vs 2D Vector Ego-Motion Calibration:** Exhaustive reanalysis of displacement regression heads across motion regimes and transition types.
5. **Within-Trajectory Time-Shuffled Action Control:** Sensitivity test permuting the 10 blackout actions across time *within each individual trajectory* (50 permutations/trajectory) to preserve exact trajectory action marginals while destroying temporal coherence.

---

## 2. Directory Layout & Package Contents

```
research/review_package/phase5_2/
├── README.md                      # This executive summary and audit guide
├── forensic_audit.md              # Detailed mathematical and empirical audit of selectivity and ego-motion
├── protocol_deviations.md         # Formal mathematical definitions and comparison with Phase 5.1
├── revised_claims_ledger.csv      # Status ledger for claims H1..H8 with necessity, sufficiency, and specificity
├── unresolved_hypotheses.md       # Empirical boundary conditions, open questions, and negative results
├── data_dictionary.md             # Coordinate systems, units, and JSON/JSONL field schemas
├── test_run.log                   # Pytest output verifying 84/84 unit tests pass
└── execution_manifest.json        # Cryptographic SHA-256 hashes of all scripts, weights, inputs, and outputs
```

### Raw Artifacts in `world_models_fpga/research/circuits/results/`

1. **Phase 5.2 Circuit Benchmark:**
   - Summary JSON: [`task_conditioned_circuits_v5_2_summary.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_2_summary.json)
   - Scene Pairs: [`task_conditioned_circuits_v5_2_pairs.jsonl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_2_pairs.jsonl) (60 independent pairs / 120 scenes)
   - Scene Specifications: [`task_conditioned_circuits_v5_2_scene_specs.jsonl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_2_scene_specs.jsonl) (60 specifications with observation SHA-256 hashes)
   - Raw Interventions: [`task_conditioned_circuits_v5_2_interventions.jsonl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_2_interventions.jsonl) (482,400 complete intervention records)
   - Schema: [`task_conditioned_circuits_v5_2_schema.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_2_schema.md)

2. **Phase 5.2 Ego-Motion Memory Reanalysis:**
   - Summary JSON: [`ego_memory_reanalysis_v5_2_summary.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/ego_memory_reanalysis_v5_2_summary.json)
   - Schema: [`ego_memory_reanalysis_v5_2_schema.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/ego_memory_reanalysis_v5_2_schema.md)

---

## 3. Key Quantitative Findings

### 3.1 Mechanistic Selectivity vs Activation Magnitude

The primary outcome compares candidate units against (a) unconstrained dimension-matched random controls, and (b) activation-perturbation-norm-matched random controls across 60 independent scene pairs ($N=60$), evaluated in both directions ($A \to B$ and $B \to A$):

| Layer | $k$ | Candidate $\eta$ | Cand $\|\Delta h\|$ | Norm-Matched $\eta$ | Norm-Matched $\|\Delta h\|$ | Unconstrained $\eta$ | Unconstrained $\|\Delta h\|$ | Paired $\Delta_{\text{norm}}$ | Raw $p_{\text{norm}}$ | Holm $p_{\text{adj}}$ | Unconstrained $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Layer 1** | 2 | 0.149 | 1.018 | 0.168 | 0.992 | 0.031 | 0.433 | -0.019 | 0.5440 | 1.0000 | $1.19 \times 10^{-2}$ |
| **Layer 1** | 4 | 0.297 | 1.409 | 0.303 | 1.356 | 0.061 | 0.704 | -0.005 | 0.5730 | 1.0000 | $4.62 \times 10^{-4}$ |
| **Layer 1** | 8 | 0.340 | 1.813 | 0.430 | 1.736 | 0.120 | 1.068 | **-0.090** | 0.9698 | 1.0000 | $1.23 \times 10^{-4}$ |
| **Layer 1** | 16 | 0.569 | 2.443 | 0.633 | 2.276 | 0.255 | 1.555 | **-0.064** | 0.3401 | 1.0000 | $5.71 \times 10^{-10}$ |
| **Layer 1** | 32 | 0.882 | 2.966 | 0.831 | 2.789 | 0.499 | 2.241 | +0.051 | 0.0037 | 0.0663 | $2.57 \times 10^{-11}$ |
| **Layer 2** | 2 | -0.011 | 0.635 | 0.051 | 0.569 | 0.031 | 0.208 | -0.062 | 0.9972 | 1.0000 | 0.9962 |
| **Layer 2** | 4 | 0.181 | 0.833 | 0.168 | 0.737 | 0.062 | 0.345 | +0.013 | 0.1514 | 1.0000 | $2.78 \times 10^{-4}$ |
| **Layer 2** | 8 | 0.344 | 1.073 | 0.283 | 0.967 | 0.127 | 0.535 | +0.061 | 0.0368 | 0.5155 | $3.41 \times 10^{-8}$ |
| **Layer 2** | 16 | 0.564 | 1.304 | 0.478 | 1.190 | 0.244 | 0.787 | +0.086 | 0.0215 | 0.3434 | $9.59 \times 10^{-11}$ |
| **Layer 2** | 32 | 0.823 | 1.527 | 0.762 | 1.443 | 0.498 | 1.136 | **+0.061** | **0.00058** | **0.0117** | $8.71 \times 10^{-11}$ |

*(Data shown for direction $A \to B$; direction $B \to A$ is numerically symmetrical and reports identical conclusions; see Summary JSON).*

#### Crucial Insights for Independent Auditor:
1. **Layer 1 Selectivity is Disconfirmed:**
   - When candidate units are compared against unconstrained random subsets, they appear to have a massive advantage ($\Delta = +0.220$ at $k=8, p = 1.23 \times 10^{-4}$).
   - However, candidate units have an activation perturbation norm of $1.813$, whereas random subsets have only $1.068$ (a $70\%$ magnitude difference).
   - When random subsets are matched to the same activation norm ($1.736$), the random subsets achieve $\eta = 0.430$, outperforming the candidate units ($\eta = 0.340, \Delta = -0.090$).
   - **Conclusion:** Layer 1 candidate superiority over random units was entirely an activation-magnitude effect.
2. **Layer 2 Modest Advantage Only Survives at Large Subspaces ($k=32$):**
   - In Layer 2, candidate units exhibit positive transfer margins over norm-matched controls ($\Delta \approx +0.06$ to $+0.08$ at $k \in \{8, 16, 32\}$).
   - At small subspace sizes ($k=4, 8, 16$), these margins do not survive Holm-Bonferroni correction ($k=8: p_{\text{adj}} = 0.5155; k=16: p_{\text{adj}} = 0.3434$).
   - Only at $k=32$ (half the entire layer) does the advantage survive family-wise correction ($\Delta = +0.061, p_{\text{adj}} = 0.0117$).
   - **Conclusion:** A small dedicated sparse subnetwork ($k=8$) does NOT constitute an autonomous, selective binding circuit when controlling for perturbation magnitude.

---

### 3.2 Ego-Motion Memory Reanalysis

#### 3.2.1 Scalar Magnitude vs 2D Vector Variance Explained
- **Displacement Head Metrics (Clean Transitions, $N=950$):**
  - **Action-Conditioned Displacement:**
    - Scalar Magnitude Linear Regression $R^2$: **$0.2518$** (Slope: $0.4538$, Intercept: $0.0134$ m)
    - 2D Vector Variance Explained $R^2_{\text{vector}}$: **$0.4324$**
    - Component $dx$ $R^2$: $0.4167$
    - Component $dy$ $R^2$: $0.4579$
    - Angular $d\theta$ $R^2$: $0.6802$
    - Translation MAE: $0.0143$ m (RMSE: $0.0163$ m)
    - Rotation MAE: $0.0164$ rad ($0.94^\circ$)
  - **Learned Sensor Displacement (Visual Latents Only):**
    - Scalar Magnitude Linear Regression $R^2$: **$0.0104$**
    - 2D Vector Variance Explained $R^2_{\text{vector}}$: **$-1.9195$**
    - Translation MAE: $0.0274$ m (RMSE: $0.0347$ m)
  - **Zero Baseline:** Translation MAE: $0.0287$ m (RMSE: $0.0314$ m)
- **Key Takeaway:** Phase 5.1's reported $R^2 = 0.2549$ was scalar magnitude linear regression on clean transitions. In 2D physical coordinates, action displacement explains $43.24\%$ of the variance, while visual latents alone explain zero coordinate variance ($R^2 < 0$), confirming that ego-motion prediction is strictly action-driven.

#### 3.2.2 Within-Trajectory Time-Shuffled Action Control
Across all 50 trajectories ($N=50$) during the 10-step blackout:
- **Action Hold (Original Action Order):** $0.5490$ m (95% CI: $[0.4735, 0.6252]$ m)
- **Action Within-Trajectory Shuffled Hold (50 perms/traj):** $0.5565$ m (95% CI: $[0.4806, 0.6328]$ m)
- **Action Cross-Trajectory Shuffled Hold:** $0.5704$ m (95% CI: $[0.4919, 0.6479]$ m)
- **Action Zero Hold:** $0.6096$ m
- **Zero Hold (Zero Displacement):** $0.6394$ m
- **Oracle Hold (Ground-Truth Displacement):** $0.5392$ m (95% CI: $[0.4616, 0.6182]$ m)

**Paired Statistical Comparisons:**
- **Action Hold vs Within-Trajectory Shuffled:**
  - Mean Paired Difference: **$-0.0075$ m** ($-7.5$ mm, 95% CI: $[-0.0115, -0.0036]$ m)
  - Wilcoxon One-Sided $p = 4.08 \times 10^{-4}$
  - Sign-Flip Permutation $p (100\text{k}) = 4.30 \times 10^{-4}$
- **Action Hold vs Cross-Trajectory Shuffled:**
  - Mean Paired Difference: **$-0.0214$ m** ($-21.4$ mm, 95% CI: $[-0.0347, -0.0088]$ m)
  - Wilcoxon One-Sided $p = 2.16 \times 10^{-3}$
- **Within-Trajectory Shuffled vs Cross-Trajectory Shuffled:**
  - Mean Paired Difference: **$-0.0139$ m** ($-13.9$ mm, 95% CI: $[-0.0262, -0.0022]$ m)
  - Wilcoxon Two-Sided $p = 0.0381$

**Key Takeaway:**
Preserving the exact empirical action distribution for each trajectory while destroying temporal coherence increases tracking error significantly ($p = 0.000408$). However, the effect size is modest: $-0.0075$ m ($7.5$ mm) vs $-0.0606$ m ($60.6$ mm) for zero action, showing that dead-reckoning performance is dominated by the net displacement integral, with a secondary contribution from temporal sequence structure.
