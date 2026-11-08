# Empirical Behavioral and Ablation Evaluation Report

**Author:** Brian A. Hadian & Senior Research Team  
**Branch:** `fix/interpretability`  
**Dataset Version:** `4.0.0-multiwaypoint-aligned` (`splits_v4`)  
**Date:** 2026-11-08  
**Evaluation Platform:** Closed-loop MuJoCo AMR Simulation (`JetAutoEnv`) + RTX 3050 GPU  
**Artifact Dependencies:** `behavioral_eval_results.json`, `matched_temporal_controls.json`, `track_c_visual_grounding.json`, `generalization_eval_results.json`  

---

## 1. Executive Summary

This report documents the comprehensive empirical behavioral evaluation of Project Hephaestus across three distinct information regimes (**Track A: Oracle**, **Track B: Structured**, and **Track C: Pure Observation**), six capability domains, multi-seed variance analysis across three independent training seeds (`42`, `1337`, `2026`), controlled causal ablations, matched temporal-prediction controls, physical counterfactual experiments (**E1–E7**), visual-spatial disambiguation suites, mid-trajectory kinetic disturbance recovery, and held-out latent prediction audits.

All evaluations were executed in closed-loop physics simulation using real model checkpoints. Discovered experimental confounds (such as candidate search mismatches in temporal hierarchy ablations and coordinate leakage in Track B) have been systematically repaired with rigorous matched protocols.

---

## 2. Multi-Track Behavioral Benchmark Results

### 2.1 Information Regimes Definition
- **Track A (Privileged Oracle):** Upper bound using exact simulator coordinates and geometric lattice planner (`ExpertNavigator`).
- **Track B (Structured Egocentric):** Camera RGB ($64\times 64\times 3$) + body-frame egocentric goal vector $(\Delta x_b, \Delta y_b, \sin\Delta\theta, \cos\Delta\theta)$.
- **Track C (Pure Observation):** Camera RGB ($64\times 64\times 3$) + discrete task instruction ID only. Zero privileged simulator poses or coordinates provided at inference.

### 2.2 Capability Performance (Track A & Track B, Primary Seed 42)

Evaluated across 10 deterministic held-out episodes per capability ($N=60$ episodes per agent):

| Capability | Oracle (Track A) | Random (Track C) | Baseline (Track B) | LeWM (Track B) | H-JEPA (Track B) |
|---|---|---|---|---|---|
| **Object Grounding** | **80.0%** ($\pm 26.1\%$) | 0.0% | 40.0% ($\pm 32.0\%$) | 90.0% ($\pm 19.6\%$) | **100.0%** ($\pm 0.0\%$) |
| **Spatial Reasoning** | **60.0%** ($\pm 32.0\%$) | 0.0% | 30.0% ($\pm 29.9\%$) | 20.0% ($\pm 26.1\%$) | **60.0%** ($\pm 32.0\%$) |
| **Obstacle Avoidance** | **60.0%** ($\pm 32.0\%$) | 0.0% | 30.0% ($\pm 29.9\%$) | 10.0% ($\pm 19.6\%$) | **20.0%** ($\pm 26.1\%$) |
| **Action Prediction** | **100.0%** ($\pm 0.0\%$) | 0.0% | 10.0% ($\pm 19.6\%$) | 20.0% ($\pm 26.1\%$) | **20.0%** ($\pm 26.1\%$) |
| **Temporal Planning** | 0.0% (Err: 2.64m) | 0.0% | 0.0% (Col: 100%) | 0.0% (Col: 100%) | 0.0% (Err: **2.16m**) |
| **Generalization** | **80.0%** ($\pm 26.1\%$) | 0.0% | 0.0% ($\pm 0.0\%$) | **20.0%** ($\pm 26.1\%$) | **20.0%** ($\pm 26.1\%$) |
| **Mean Overall Success** | **63.3%** | 0.0% | 18.3% | 26.7% | **36.7%** |
| **Mean Collision Rate** | **8.3%** | 8.3% | 53.3% | 56.7% | **48.3%** |
| **Mean Goal Error** | **0.981 m** | 1.834 m | 0.836 m | 0.743 m | **0.677 m** |

**Key Findings:**
1. H-JEPA achieves **100.0%** success on Object Grounding and **60.0%** on Spatial Reasoning (matching the Privileged Oracle), significantly outperforming Baseline.
2. H-JEPA achieves the lowest overall mean goal error (**0.677 m** vs 0.836 m Baseline and 0.743 m LeWM).

---

### 2.3 Multi-Seed Stability in Track B (Seeds: 42, 1337, 2026)

Evaluated across 30 representative benchmark episodes for each model checkpoint across all three independently trained seeds:

| Architecture | Seed 42 Success | Seed 1337 Success | Seed 2026 Success | Mean Success | Multi-Seed Std | Mean Collision |
|---|---|---|---|---|---|---|
| **Baseline Navigation Policy** | 40.0% | 13.3% | 20.0% | **24.4%** | $\pm 13.9\%$ | 42.2% |
| **LeWorldModel (LeWM)** | 26.7% | 26.7% | 26.7% | **26.7%** | $\pm 0.0\%$ | 55.6% |
| **H-JEPA Hierarchical** | 40.0% | 36.7% | 26.7% | **34.4%** | $\pm 6.9\%$ | **52.2%** |

---

### 2.4 Track C: Pure Visual Spatial Grounding Benchmark (`track_c_visual_grounding.json`)

To eliminate the confound of privileged coordinate leakage in Track B, we implemented `run_track_c_visual_grounding.py` evaluating agents strictly under first-person $64\times 64\times 3$ RGB vision with a discrete task token ($N=20$ episodes per suite, zero simulator coordinates, asserted waypoint clearance $\ge 0.45$m):

| Test Regime | Description | H-JEPA Success | LeWM Success | Baseline Success | $\Delta$ vs Nominal | Exact McNemar $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Nominal Grounding** | Clean arena, target specified by discrete ID | **20/20 (100.0%)** | 19/20 (95.0%) | 8/20 (40.0%) | Baseline (0.260m err) | — |
| **2A. Floor Nuisance** | Floor texture dark slate/blue | **1/20 (5.0%)** | — | — | **-95.0%** [-100%, -85%] | **$p < 0.0001$** |
| **2B. Illumination Nuisance**| 50% ambient lighting dimming | **17/20 (85.0%)** | — | — | -15.0% | $p = 0.2500$ |
| **2C1. Target Appearance** | Target object texture mutation | **14/20 (70.0%)** | — | — | -30.0% | **$p = 0.0312$** |
| **2C2. Distractor Appearance**| Distractor objects texture mutation | **20/20 (100.0%)** | — | — | 0.0% | $p = 1.0000$ |
| **3. Semantic Color Swap** | Target and distractor swap visual colors | **20/20 (100.0%)** | — | — | 0.0% | $p = 1.0000$ |
| **4. Landmark Rearrangement**| Physically swapping target & obstacle positions | **1/20 (5.0%)** | — | — | **-95.0%** (1.700m err) | **$p < 0.0001$** |
| **5. Layout Generalization** | Rejection-sampled configurations (surface clearance) | **19/20 (95.0%)** | — | — | -5.0% | $p = 1.0000$ |
| **6. Rigid SE(2) Rotation (+90 deg)** | Global rigid arena rotation by $+90^\circ$ | **20/20 (100.0%)** | — | — | +0.0% | $p = 1.0000$ |
| **6. Rigid SE(2) Rotation (+180 deg)**| Global rigid arena rotation by $+180^\circ$ | **20/20 (100.0%)** | — | — | +0.0% | $p = 1.0000$ |
| **6. Rigid SE(2) Rotation (-90 deg)** | Global rigid arena rotation by $-90^\circ$ | **20/20 (100.0%)** | — | — | +0.0% | $p = 1.0000$ |

**Key Insights:**
1. Under pure visual observation, predictive world models (H-JEPA 100.0%, LeWM 95.0%) dramatically outperform the reactive baseline (40.0%), demonstrating that predictive latent representations provide the spatial memory required to navigate to visually grounded goals.
2. Under global rigid SE(2) rotations ($\pm 90^\circ, 180^\circ$), where relative arena geometry is preserved, H-JEPA maintains **100.0% success** ($p = 1.0000$), with global latent cosine similarity of 0.9745. However, egocentric camera yaw rotation drops latent cosine to 0.3450; the visual encoder lacks steerable group-equivariant filters.
3. Shifting floor texture collapses success to **5.0% ($p < 0.0001$)**, proving extreme sensitivity to surface textures, while 50% lighting dimming preserves 85.0% success.
4. Physically swapping landmark positions collapses success to **5.0% (1.700m goal error, $p < 0.0001$)**. The robot steers to the nominal physical coordinate, proving navigation via an **allocentric spatial map prior** rather than dynamic visual tracking.
5. Rejection-sampled layout generalization preserves **95.0% success** ($p = 1.0000$) when valid surface clearance ($\ge 0.05$m) is enforced.
6. Mutating target appearance reduces success to **70.0% ($p = 0.0312$)**, whereas mutating distractor appearance preserves **100.0%**, and color swapping preserves **100.0%** by routing according to categorical task tokens rather than dynamic color parsing.

---

## 3. Matched Temporal-Prediction Controls & Ablations

### 3.1 Resolving Historical Confounds (Finding A & B)
In legacy reports, `no_temporal_hierarchy` was claimed to cause a collapse to 3.3%. However, raw evaluation logs showed:
- `prediction_used`: 36.7% success, 63.3% collisions.
- `no_temporal_hierarchy`: 36.7% success, 60.0% collisions.
The ablation produced zero performance drop. Furthermore, the evaluation compared 25 hand-designed two-phase sequences against single-step action searches.

### 3.2 Repaired Matched Protocol (`run_matched_temporal_controls.py`)
To isolate learned temporal prediction from search confounds, all conditions evaluated the **exact same 24 candidate action sequences of length $K=5$** across $N = 50$ matched episodes:

| Condition | Description | Success Rate | Collision Rate | Goal Error (m) | $\Delta \text{Success}$ vs Reactive | McNemar Exact $p$-value | Holm-Adjusted $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cond A** | Reactive Policy ($K=0$, no lookahead) | 36.0% | 40.0% | 0.537m | Baseline | — | — |
| **Cond B** | Direct Macro Prediction ($K=5$, 1 pass) | 46.0% | 48.0% | 0.537m | +10.0% | $p = 0.0625$ | $p = 0.2500$ |
| **Cond C** | Recursive Micro Unrolling ($K=5$, 5 steps) | **50.0%** | **36.0%** | **0.458m** | +14.0% | $p = 0.7266$ (vs Macro) | $p = 1.0000$ |
| **Cond D** | Oracle Physics MPC (MuJoCo simulator clone) | 46.0% | **8.0%** | **0.390m** | +10.0% | — | — |
| **Cond E** | Shuffled Action-Outcome Binding (10 perms) | 34.0% | 46.0% | 0.551m | -2.0% | $p = 0.03125$ (vs Macro) | $p = 0.1562$ |
| **Cond E2**| Random Candidate Selection | 26.0% | 20.0% | 0.794m | -10.0% | $p = 0.00195$ (vs Macro)| **$p = 0.0117$** |
| **Cond F** | Candidate Search without Prediction (Prior-Only)| 36.0% | 40.0% | 0.537m | 0.0% | $p = 1.0000$ (Identical) | — |
| **Cond G** | Straight-Line Kinematic Heuristic (Non-Predictive)| 46.0% | 48.0% | 0.540m | +10.0% | $p = 1.0000$ (vs Macro) | $p = 1.0000$ |

### 3.3 Open-Loop Kinematic Fidelity across Horizons ($K \in \{1, 3, 5, 10\}$)

| Horizon $K$ | Micro Recursive Latent MSE | Micro Recursive Disp Err (m) | LeWM Recursive Disp Err (m) | Macro Direct Latent MSE | Macro Direct Disp Err (m) | $\Delta \text{Disp Err}$ (Macro vs Micro) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K=1$** | 0.0262 | 0.0180m | 0.0200m | — | — | — |
| **$K=3$** | 0.1217 | 0.0491m | 0.0588m | — | — | — |
| **$K=5$** | 0.2272 | 0.0814m | 0.0975m | **0.1932** | **0.0460m** | **-0.0354m (-43.5%)** |
| **$K=10$** | 0.4804 | 0.1612m | 0.2025m | — | — | — |

### 3.4 Scientific Findings:
1. **Direct Macro vs. Recursive Micro:** In closed loop, direct macro (46.0%) achieves parity with recursive micro (50.0%, McNemar exact $p = 0.7266$, Wilcoxon $p = 0.4750$). Direct macro does not statistically dominate recursive micro in task success.
2. **Kinematic Heuristic Equivalence:** Candidate selection via non-predictive open-loop unicycle kinematic projection (`cond_G_straight_line_heuristic`) achieves identical success (**46.0%**, $n_{10}=0, n_{01}=0, p = 1.0000$). Forward candidate search alone drives a major portion of navigation behavior.
3. **Multiplicity-Controlled Lookahead Evaluation:** Macro lookahead significantly outperforms random candidate selection (46.0% vs 26.0%, Holm-adjusted $p = 0.0117$). However, under Holm–Bonferroni multiplicity control across 6 pairwise comparisons, macro vs multi-permutation shuffled binding (34.0%, unadjusted $p = 0.03125$) yields adjusted $p = 0.1562 > 0.05$.
4. **Computational Latency Amortization:** The genuine scientific principle of hierarchical macro temporal abstraction is **$4.30\times$ measured GPU latency reduction** (median 0.371 ms vs 1.594 ms, 50 warm-up runs, 200 repeats; theoretical RTL cycle model 4,327 vs 21,635 cycles, or 45k vs 190k cycles, $4.22\times$) paired with **open-loop kinematic drift suppression** (0.0460m vs 0.0814m physical error at $K=5$, a 43.5% error reduction).
5. **Condition A and Condition F Equivalence:** Mathematical and empirical analysis confirms that scoring candidates solely by alignment to the reactive action $a_{\text{react}}$ identically selects candidate 0, making Condition F mathematically and behaviorally equivalent to Condition A ($\Delta = 0.0\%$, identical trajectory).
6. **Falsification of Prospective Safety Barriers (`run_predictive_safety_benchmark.py`):** On 40 collision-critical episodes, intact neural collision prediction yields a **70.0% collision rate**, identical to shuffled collision binding (70.0%, $p = 1.0000$), disabled collision head (70.0%, $p = 1.0000$), and random predictor (70.0%, $p = 1.0000$). In contrast, oracle physics MPC cuts collisions to 7.5% ($p = 5.96 \times 10^{-8}$). Offline discrimination on the macro $K=5$ horizon (AUROC = 1.0000, Brier = 0.0003, ECE = 0.0041, FNR = 0.0%) fails to provide closed-loop safety under temporal constraints, refuting formal Control Barrier Function (CBF) claims.

---

## 4. Generalization & Counterfactual Invariance (E1–E7)

Evaluated across 8 physical counterfactual pairs per experiment with verified unique hashes:

| Experiment | Oracle Ret. | Baseline Ret. | LeWM Ret. | H-JEPA Ret. | H-JEPA Ei Success |
|---|---|---|---|---|---|
| **E1: Landmark Position Shift** | 12.5% | 0.0% | 0.0% | 0.0% | 0.0% |
| **E2: Appearance / Color Shift** | **100.0%** | 71.4% | 0.0% | **50.0%** | 12.5% |
| **E3: Robot Yaw Rotation ($\pm 90^\circ$)** | **100.0%** | 0.0% | 0.0% | **100.0%** | **100.0%** |
| **E4: Robot Position Offset** | **100.0%** | **100.0%** | 0.0% | **50.0%** | 50.0% |
| **E5: Intermediate Obstacle Insertion** | **100.0%** | 57.1% | 0.0% | 0.0% | 0.0% |
| **E6: Shape Mutation (Cylinder $\leftrightarrow$ Box)** | 25.0% | 28.6% | 0.0% | 0.0% | 0.0% |
| **E7: Camera Viewpoint Shift ($\pm 15^\circ$)** | **100.0%** | 57.1% | 0.0% | **25.0%** | 25.0% |

### Disambiguation Probes (DISAMBIG_01 .. DISAMBIG_04)

| Probe Condition | Oracle Success | Baseline Success | LeWM Success | H-JEPA Success | H-JEPA Goal Error |
|---|---|---|---|---|---|
| `DISAMBIG_01`: Ego Left vs World Left | **100.0%** | 66.7% | 0.0% | 0.0% | 0.477 m |
| `DISAMBIG_02`: Ego Front vs Object Behind | **66.7%** | 0.0% | 0.0% | 0.0% | 0.896 m |
| `DISAMBIG_03`: Geometric vs Visual Proximity | 0.0% | 0.0% | 0.0% | **100.0%** | **0.237 m** |
| `DISAMBIG_04`: Visibility vs Reachability | 0.0% | 0.0% | 0.0% | 0.0% | 0.245 m |

---

## 5. Robustness & Dynamic Disturbance Recovery

Evaluated under an unexpected lateral kinetic push ($\Delta v_y = 0.8\text{ m/s}$, $\Delta \omega_z = 0.5\text{ rad/s}$) injected at step $t=10$:

| Model | Recovery Rate | Collision Rate | Final Goal Error | Trajectory Divergence |
|---|---|---|---|---|
| **CNN Reactive Baseline** | 80.0% | 30.0% | 0.294 m | Diverges immediately |
| **LeWorldModel (Single-Scale)** | 55.0% | 25.0% | 0.349 m | Delayed recovery |
| **H-JEPA (Hierarchical)** | **100.0%** | **20.0%** | **0.234 m** | **Rapid macro replanning** |

---

## 6. Action-Conditioned Latent Prediction Audit

Evaluated on 2,642 held-out test transitions from `held_out_test_benchmark.npz` across $K=5$ strides:

| Evaluation Metric | LeWorldModel (LeWM) | H-JEPA Micro ($K=1$) | H-JEPA Macro ($K=5$) |
|---|---|---|---|
| **Latent Transition MSE** | 0.04988 | **0.02844** | 0.19096 |
| **Latent Cosine Similarity** | 0.9713 | **0.9810** | 0.8647 |
| **Collision Accuracy** | 99.3% | — | **99.9%** |
| **5-Step Recursive Rollout MSE** | 0.35643 | — | **0.19096 (Direct Macro)** |

---

## 7. Status of Preregistered Hypotheses

| Hypothesis | Preregistered Claim | Empirical Evidence | Verdict |
|---|---|---|---|
| **H1 (World-Model Utility)** | Action-conditioned prediction guides navigation | Outperforms random actions (46.0% vs 26.0%, Holm-adjusted $p = 0.0117$). Matched by straight-line kinematic heuristic (46.0%, $p = 1.0000$). Shuffled binding Holm-adjusted $p = 0.1562$. | **CALIBRATED (Level E3)** |
| **H3 (Temporal Hierarchy)** | Macro lookahead ($K=5$) avoids compounding drift and reduces compute | Cuts 5-step drift from 0.081m to 0.046m (-43.5%); amortizes GPU latency by $4.30\times$ (median 0.371 ms vs 1.594 ms; RTL cycle model 4,327 vs 21,635 cycles); achieves closed-loop parity with micro (46.0% vs 50.0%, $p = 0.7266$). | **REFINED (Amortization & Stability)** |
| **H4 (Invariance Structure)** | Spatial representations exhibit visual invariance and spatial equivariance | SE(2) global equivariance (100.0%, latent cosine = 0.9745); allocentric map prior (5.0% landmark relocation, 1.700m error, $p < 0.0001$); categorical token routing (100.0% color swap); extreme floor texture sensitivity (5.0%, $p < 0.0001$). | **CALIBRATED (Level E3)** |
| **H-Safety (Predictive Barrier)** | Latent collision lookahead provides a prospective safety barrier | 70.0% closed-loop collisions across intact, shuffled, and disabled heads ($p = 1.0000$), while oracle cuts collisions to 7.5% ($p < 10^{-7}$). Offline AUROC=1.0 does not yield closed-loop safety. | **FALSIFIED AS PROSPECTIVE BARRIER** |
