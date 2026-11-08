# Phase 5 Unresolved Questions & Future Scientific Trajectory

**Date:** 2026-11-08  
**Repository:** `BrianHadianSTEI23/project-haphaestus` (`fix/interpretability`)  
**Status:** Post-Phase 5 Reanalysis  

---

## 1. Overview

In accordance with strict scientific integrity principles, this document articulates the key unresolved computational and mechanistic questions that persist following the Phase 5 forensic reanalysis and targeted causal interventions.

---

## 2. Unresolved Scientific Questions

### Question 1: Architectural Inductive Biases for True Instance-Level Object Binding
- **Current Finding:** Concatenating visual latents $z \in \mathbb{R}^{32}$ with discrete task embeddings $e \in \mathbb{R}^{16}$ into a multi-layer perceptron grounding head produces a mixed nonlinear interaction ($\|I\| = 0.352$m) that does **not** perform geometric selection ($\cos \theta = -0.314$, dual discrimination $16.0\%$, identical to the $16.3\%$ permutation null). On same-category instances, dual discrimination collapses to $5.0\%$.
- **Unresolved Issue:** What minimal architectural mechanism is necessary and sufficient to implement true object-centric variable binding in predictive world models?
  - *Candidate 1:* Slot-based object representations (e.g., Slot Attention, DINOSAUR) where task tokens bind to discrete object tokens via spatial attention.
  - *Candidate 2:* Cross-attention coordinate queries where spatial coordinates are decoded via query-key dot products rather than feedforward MLP concatenation.
  - *Candidate 3:* Relative spatial coordinate heads that explicitly predict bounding box centroids and semantic segmentation masks before body-relative polar transformation.

### Question 2: Real-World Sensorimotor Drift in Action-Conditioned Dynamics
- **Current Finding:** In clean simulation, the action-conditioned forward transition model $T_{\text{micro}}(s_t, a_t)$ produces single-step displacement predictions with translation MAE **0.0177m** ($R^2 = 0.504$), substantially outperforming observed sensor latent differences (MAE 0.0261m, $R^2 = 0.120$) and matching privileged oracle odometry during 10-step sensory deprivation ($1.4439$m vs $1.4450$m).
- **Unresolved Issue:** In physical deployment on the Kria KR260 SOM and JetAuto hardware, open-loop motor commands $a_t$ are subject to unmodeled wheel slip, surface friction variations, motor deadbands, and battery voltage drop.
  - Does the action-conditioned predictor maintain its dead-reckoning accuracy on real AMR robots without hardware wheel encoder and IMU proprioceptive sensor fusion?
  - Can wheel odometry and IMU measurements be incorporated as explicit proprioceptive latents in H-JEPA's hierarchical transition predictor?

### Question 3: Dynamic Confidence Calibration in Closed-Loop Autonomous Navigation
- **Current Finding:** Offline same-trajectory replay established that fixed 0.65/0.35 blending exponentially destroys memory into static black-frame noise ($0.35^{10} \approx 2.75 \times 10^{-5}$), collapsing tracking error across all motion sources to $\approx 1.277$m. Adaptive confidence gating ($\alpha_t = 0$ upon dropout) preserves valid spatial tracking ($1.4439$m).
- **Unresolved Issue:** In real-world environments, sensory degradation is rarely binary (clean vs completely black). Partial occlusions, glare, motion blur, and visual clutter introduce continuous variations in observation reliability.
  - How can H-JEPA's encoder dynamically infer observation confidence $\alpha_t \in [0, 1]$ directly from visual latent uncertainty or prediction residual $\|s_t - \hat{s}_t\|$?
  - What are the stability guarantees of confidence-gated memory updates under closed-loop locomotion where navigation actions are driven by the estimated goal?

### Question 4: Cross-Embodiment Generalization of the Intelligence Principles
- **Current Finding:** The four computational principles derived in Phase 5:
  1. *Relational Binding Principle:* Concatenative MLP conditioning fails to implement variable binding.
  2. *Distributed Representation Principle:* MLP grounding heads form polysemantic distributed codes without modular subcircuits.
  3. *Motor-Predictive Observability Principle:* Sensory latent differences cannot observe motion during sensory deprivation; action-conditioned transition models are necessary.
  4. *Adaptive Assimilation Principle:* Fixed convex blending rules suffer exponential erasure under non-stationary observation quality.
- **Unresolved Issue:** Are these principles universal across non-holonomic differential-drive mobile bases, quadrupedal robots, and robotic manipulators, or are they specific to holonomic planar AMR locomotion?
