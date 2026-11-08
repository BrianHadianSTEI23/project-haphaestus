# Phase 4 Interpretation-Ready Synthesis: Computational Principles of Embodied Spatial Intelligence

**Baseline Audit Commit:** `8229f786309fb6d9ea669e8a783793c92ceca564`  
**Execution Commit:** `8eab3cdd5e90f40fa3a7f33b068f0d455b0e7237`  
**Date:** 2026-11-08  
**Repository:** `BrianHadianSTEI23/project-haphaestus` (`fix/interpretability`)

---

## 1. Executive Summary: What Was Proven vs What Was Disproven

Phase 4 concludes the controlled mechanistic investigation of task-conditioned spatial decoding and ego-motion memory in Project Hephaestus. Through 2x2 factorial evaluations on 100 dual-visible scenes, intermediate activation patching across all 9 grounding head stages, and same-trajectory replay across 50 recorded paths, the exact computational reality of the model has been isolated.

### 1.1 What Was Proven
1. **Additive Task-Conditioned Spatial Prior, Not Visual Object Binding:**
   - In 50 bit-for-bit identical paired scenes with both objects visible ($\ge 100$ pixels each), changing the task token rotates the inferred goal vector by **$47.87^\circ$** ($0.8355$ rad).
   - However, marginal accuracy pointing toward Target A is **42.0%** ($p = 0.322$ vs 50% chance), Target B is **38.0%** ($p = 0.119$ vs 50% chance), and joint dual discrimination is only **10.0%** (5/50 pairs).
   - Testing against 25% chance level yields $p = 0.0133$, proving the model performs **significantly worse than random guessing** for binary object selection because both tokens steer toward a common canonical arena sector.
   - Model comparison on held-out test splits decisively favors the **Additive Task Bias Model ($MSE = 0.0334$)** over the **Visual Geometry Selection Model ($MSE = 0.1357$)** by more than 4-to-1.
2. **Latent Delta Collapse Under Sensory Deprivation:**
   - During black-frame dropouts, $\text{RGB}_t = 0 \implies s_t = s_{\text{black}} \implies \Delta s = 0$.
   - The learned `micro_ego_head` receives an identically zero latent delta and outputs its static network bias: $[0.0129, 0.0014, 0.0001]$.
   - The model has no internal recurrent state or motor-efference copy; it is physically impossible for it to estimate self-motion when the visual latent does not change.
3. **Equivalence of Full Memory and Zeroed Ego-Motion:**
   - Same-trajectory replay of 50 common trajectories under 10-step sensory blackout proves that **Full Learned Memory (tracking error 0.7047m) is statistically equivalent to Zeroed Ego-Motion (tracking error 0.7082m)**:
     - Mean paired difference: **$-0.0035$m (3.5 millimeters)**.
     - TOST equivalence test within margin $\pm 0.05$m: **$p < 0.0001$ (confirmed equivalent)**.
   - The learned `micro_ego_head` provides no meaningful egocentric coordinate update over assuming the robot remained stationary.
4. **Exponential Memory Erasure via Fixed Blending:**
   - Fixed controller blending $g_t = 0.65 \cdot g_{\text{vis}} + 0.35 \cdot g_{\text{mem}}$ causes memory to decay as $0.35^{10} \approx 2.75 \times 10^{-5}$. Within 5–10 steps, the memory vector is replaced by the ungrounded static vector of a black image.
   - Frozen pre-blackout memory achieves **0.6727m** tracking error, and Oracle Odometry achieves **0.5581m** ($p = 0.0012$), proving that preventing decay into black-frame noise substantially restores spatial persistence.

### 1.2 What Was Disproven & Falsified
- **FALSIFIED:** Task-conditioned visual object binding (CLAIM_02). The grounding head does not select among visual objects from scene geometry; it perturbs an additive prior.
- **FALSIFIED:** Ego-motion spatial persistence under visual sensory deprivation (CLAIM_04). The learned displacement head does not track self-motion during visual dropout.
- **FALSIFIED:** Modular hidden-layer subcircuits in the grounding head (CLAIM_05). Representation is distributed across MLP weights.

---

## 2. Reusable Computational Principles of Embodied Spatial Intelligence

### Principle 1: Additive Prior Steering vs Geometric Object Binding
- **Computational Statement:** In feedforward perception-to-goal networks without explicit token-to-feature attention mechanisms or spatial object slot bottlenecks, categorical task embeddings act as additive bias vectors. They translate and rotate the output manifold within a low-dimensional subspace rather than indexing or routing spatial coordinates of specific visual geometry.
- **Assumptions & Boundary Conditions:** Evaluated in H-JEPA's 3-layer MLP grounding head ($48 \to 64 \to 64 \to 4$). Holds when task embeddings are concatenated with global visual latents without localized cross-attention.
- **Architectural Implication:** To achieve true task-conditioned object binding, world models must replace global latent concatenation with cross-attention over spatial feature maps or explicit object slot representations (e.g., Slot Attention or spatial queries).

### Principle 2: Latent Delta Degeneracy Under Sensory Deprivation
- **Computational Statement:** Any feedforward self-motion estimator whose inputs are strictly deltas between consecutive sensory latents ($\Delta s = s_t - s_{t-1}$) collapses to static network bias when sensory input ceases ($\Delta s \to \mathbf{0}$). It cannot perform dead-reckoning.
- **Assumptions & Boundary Conditions:** Valid across all sensory modalities (RGB, depth, lidar) when sensory dropout produces static latent representations.
- **Architectural Implication:** Embodied world models requiring dead-reckoning under sensory deprivation must integrate motor-action efference copies ($a_{t-1}$) or proprioceptive IMU/wheel-odometry streams into their transition models, or maintain an autoregressive recurrent state ($h_t = f(h_{t-1}, a_{t-1})$).

### Principle 3: Blending Instability in Static-Weight Controller Memory
- **Computational Statement:** Fixed linear blending between an observation-derived estimate and an accumulated memory vector ($g_t = \alpha g_{\text{obs}} + (1-\alpha) g_{\text{mem}}$) is unstable under sensory degradation. The memory decays exponentially as $(1-\alpha)^K$, substituting ungrounded sensor corruption for valid spatial history.
- **Architectural Implication:** Controllers must employ adaptive, confidence-gated, or Kalman-filter-like uncertainty weighting where $\alpha \to 0$ when observation variance/confidence drops below an empirical threshold.
