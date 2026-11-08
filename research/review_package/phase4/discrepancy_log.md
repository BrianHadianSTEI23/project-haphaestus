# Phase 4 Discrepancy Log: Reconciling Prior Hypotheses with Mechanistic Evidence

**Baseline Audit Commit:** `8229f786309fb6d9ea669e8a783793c92ceca564`  
**Execution Commit:** `bbad9132a268a395c3298fb4a08dc3b8edbf30fd`  
**Date:** 2026-11-08  
**Investigator:** Senior Mechanistic Interpretability, Robotics, and Computer Architecture Research Team  

---

## 1. Overview & Protocol Discrepancies

This document logs all empirical, mathematical, and interpretational discrepancies identified between historical Phase 3 claims and the controlled Phase 4 mechanistic experiments (`task_conditioned_binding_v3`, `task_conditioned_circuits_v3`, and `ego_memory_mechanism_v3`).

In accordance with strict scientific integrity principles:
- No negative result has been discarded or suppressed.
- No hypothesis has been promoted to E3 without satisfying necessity, sufficiency, and cross-condition causal validation.
- All discrepancies are traced to architectural and mathematical root causes in the codebase.

---

## 2. Discrepancy Table: Claims vs Raw Mechanistic Evidence

| Discrepancy ID | Historical Phase 3 Claim | Phase 4 Ground Truth Evidence | Mathematical / Architectural Root Cause | Reconciled Status |
|---|---|---|---|---|
| **DISC_01** | *"Angular shift of 0.869 rad ($49.8^\circ$) under identical RGB proves task-conditioned visual object binding"* (CLAIM_02 in Phase 3) | In 50 dual-visible pairs (100 scenes, all $\ge 100$ px), Token A steers to Target A in **42.0%** of scenes, Token B steers to Target B in **38.0%**, and joint dual discrimination is only **10.0%** ($p = 0.0133$ vs 25% chance). Held-out Additive Model MSE is **0.0334** vs **0.1357** for Visual Geometry Selection. | The grounding head $g(z, e) = \text{MLP}([z, e])$ learns a dominant additive task bias subspace ($a(z) + b(e)$). Changing $e$ rotates the output vector by $\approx 48^\circ$, but the rotation vector is uncorrelated with the physical position of the alternate target. | **FALSIFIED AS OBJECT BINDING.** Reclassified as Task-Conditioned Spatial Prior / Mixed Scene Decoding (E2). |
| **DISC_02** | *"Learned `micro_ego_head` maintains egocentric goal tracking during temporary visual blackout"* (CLAIM_04) | In 50 replay trajectories under 10-step blackout, Full Memory error is **0.7047m**, while Zeroed Ego-Motion error is **0.7082m** (difference is $-0.0035$m / 3.5mm; TOST equivalence $p < 0.001$). Frozen memory achieves **0.6727m**. | Under black-frame dropout, $\text{RGB}_t = 0 \implies s_t = s_{\text{black}} \implies \Delta s = 0$. The `micro_ego_head` receives zero input delta and outputs only static bias $[0.0129, 0.0014, 0.0001]$. It is physically incapable of perceiving robot motion during blackout. | **FALSIFIED AS EGO-MOTION TRACKING.** Reclassified as Black-Frame Blending Decay. |
| **DISC_03** | *"Mismatched donor memory (0.438m) beating full memory (0.648m) in Phase 3 v2 was an anomalous outlier"* | Replay confirms that any vector pointing generally forward/target-ward outperforms the black-frame corrupted full memory. | Fixed controller blending $g_t = 0.65 \cdot g_{\text{vis}} + 0.35 \cdot g_{\text{mem}}$ exponentially erases memory by $0.35^{10} \approx 2.7 \times 10^{-5}$ into the arbitrary static black-frame vector. Donor memory traces and frozen memory bypass this decay. | **EXPLAINED & RESOLVED.** The anomaly was driven by exponential black-frame erasure, not donor informativeness. |
| **DISC_04** | *"Visual-only control achieved higher closed-loop navigation (83.3%) than full memory (80.0%) in Phase 3 v2 due to stochastic noise"* | Same-trajectory replay eliminates closed-loop path divergence and proves visual-only tracking error is **0.7460m** vs **0.7047m** for full memory. | In Phase 3 v2, different conditions took actions generated from different goal estimates, generating divergent physical trajectories. Evaluating tracking error across divergent paths confounded controller accuracy with locomotion history. Same-trajectory replay resolves this. | **RESOLVED VIA SAME-TRAJECTORY REPLAY.** Full memory slightly outperforms visual-only ($p = 8.88 \times 10^{-15}$), but purely due to pre-blackout vector retention, not ego-motion updates. |
| **DISC_05** | *"Grounding head implements a localized, modular circuit in Layer 1 or Layer 2"* | Surgical hidden-layer donor patching achieves 100.0% transfer at Layer 1 and 100.0% at Layer 2. Top-16 ablation shifts output by 0.526m vs 0.321m for random 16D ablation ($p = 0.0012$). | The 3-layer MLP grounding head ($48 \to 64 \to 64 \to 4$) computes a distributed transformation without modular bottlenecking. Selective ablation does not isolate an independent object-binding subcircuit. | **CONFIRMED DISTRIBUTED MLP.** Falsified for modular circuit isolation. |

---

## 3. Methodological Invariants Enforced in Phase 4

1. **Rejection Sampling & Geometric Validity:**
   - Both target and distractor objects guaranteed visible with $\ge 100$ segmentation pixels rendered by MuJoCo classic OpenGL/EGL renderer.
   - Initial robot pose to waypoint distance strictly $\ge 0.60$m.
   - Initial object separation $\ge 0.52$m.
2. **Elimination of Trajectory Divergence Confound:**
   - Evaluated 50 identical recorded trajectories through all 8 memory conditions.
   - Ensured bit-for-bit identical robot poses, observations, and ground-truth body-relative vectors.
3. **Statistical Multiplicity & Equivalence Testing:**
   - Pre-registered 5 confirmatory family comparisons.
   - Controlled family-wise error rate via Holm-Bonferroni correction.
   - Tested non-inferiority/equivalence within margin $\delta = 0.05$m using Two One-Sided Tests (TOST).
