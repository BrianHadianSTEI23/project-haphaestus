# Phase 5 Protocol Deviations & Methodological Corrections

**Date:** 2026-11-08  
**Repository:** `BrianHadianSTEI23/project-haphaestus` (`fix/interpretability`)  
**Baseline Audit Commit:** `fa0d4f97abbe16de9980f7844db90a6921660324`  

---

## 1. Executive Summary of Deviations

This document logs all methodological, statistical, and architectural adjustments implemented in Phase 5 to repair experimental flaws identified in Phase 4.

| Component | Phase 4 Protocol | Phase 5 Protocol | Scientific Rationale |
|---|---|---|---|
| **Binding Discrimination** | Evaluated Scene A only ($N=50$ scenes). | Evaluated both Scene A and Scene B at the independent scene-pair level ($N=50$ pairs, 100 scenes). | Eliminates asymmetric evaluation bias and enforces proper statistical pairing. |
| **Null Distribution** | Independent 25% binomial test ($p=0.0133$). | Paired permutation test (100,000 resamples) preserving same-scene joint dependency ($p=0.6200$). | Falsifies the independence assumption; empirical null mean is 16.2%, matching observed 16.0%. |
| **Factorial Interaction $I$** | Scalar interaction magnitude $\|I\|$ without alignment to physical target swap. | Vector interaction $I$, bootstrap 95% CIs, and sign-flip permutation test of cosine alignment with physical target swap. | Directly differentiates between additive prior ($I=0$), geometric correction ($\cos \theta > 0$), and uncoordinated nonlinear interaction. |
| **Model Comparison** | Confounded H-JEPA surrogate fitting with ground-truth error; 50/50 un-grouped split; missing bilinear terms. | Two separate comparisons (Surrogate vs Ground Truth); Grouped 5-Fold Cross-Validation by `pair_id`; explicit bilinear Kronecker terms. | Guarantees identical dependent variables and prevents data leakage across factorial conditions within a pair. |
| **Instance-Level Binding** | Evaluated only different semantic categories (`red_box` vs `blue_cylinder`). | Tests same-category instances (e.g., two cylinders or boxes) distinguished by positional/relational condition. | Distinguishes semantic category recognition from spatial instance-level binding. |
| **Intermediate Circuits** | Stored only activation norms; replaced 100% of layer activations (trivial identity check). | Records full 9-stage activation vectors in JSONL; uses discovery split to freeze candidate dimensions; tests necessity, sufficiency, rescue, and matched random controls. | Disentangles true circuit modularity from whole-layer identity reconstruction. |
| **Ego-Motion Information Path** | Evaluated only sensor latent difference $\Delta s = s_t - s_{t-1}$ under black frames ($\Delta s = 0$). | Audits complete path; tests action-conditioned transition model $\delta_s(s_t, a_t)$ via `micro_predictor.forward_transition`; distinguishes step-5 transition from repeated black frames. | Isolates whether architecture contains action-conditioned motion estimation independent of camera feed. |
| **Memory Policy** | Confounded odometry quality with fixed 0.65/0.35 blending decay into black-frame static bias. | Full $4 \times 2$ factorial (4 motion sources $\times$ 2 memory policies: fixed blending vs hold/freeze) + adaptive confidence gating. | Decouples displacement prediction accuracy from memory retention decay. |

---

## 2. Mathematical Formulations of Phase 5 Invariants

### 2.1 Independent Scene-Pair Dual Discrimination
For each pair $i \in \{1, \dots, N\}$, we observe four factorial outputs:
$$g_{aa} = F(z_A, e_A), \quad g_{ab} = F(z_A, e_B), \quad g_{ba} = F(z_B, e_A), \quad g_{bb} = F(z_B, e_B)$$
alongside ground-truth body-relative coordinates:
$$\text{Scene A}: P_{1,A} = \text{Target A}, \, P_{2,A} = \text{Target B}$$
$$\text{Scene B}: P_{2,B} = \text{Target A}, \, P_{1,B} = \text{Target B}$$

A scene configuration achieves dual discrimination iff both token queries select their respective target over the distractor:
$$\text{Dual}_A = \mathbb{I}(\|g_{aa} - P_{1,A}\| < \|g_{aa} - P_{2,A}\|) \land \mathbb{I}(\|g_{ab} - P_{2,A}\| < \|g_{ab} - P_{1,A}\|)$$
$$\text{Dual}_B = \mathbb{I}(\|g_{ba} - P_{2,B}\| < \|g_{ba} - P_{1,B}\|) \land \mathbb{I}(\|g_{bb} - P_{1,B}\| < \|g_{bb} - P_{2,B}\|)$$

The pair-level discrimination score is $\bar{D}_i = \frac{1}{2}(\text{Dual}_{A,i} + \text{Dual}_{B,i})$.

### 2.2 Factorial Interaction Vector
$$I_i = (g_{bb,i} - g_{ba,i}) - (g_{ab,i} - g_{aa,i})$$
Physical target swap vector for Target B:
$$\Delta P_i = P_{1,B,i} - P_{2,A,i}$$
Cosine alignment:
$$\cos \theta_i = \frac{I_i \cdot \Delta P_i}{\|I_i\| \|\Delta P_i\|}$$

### 2.3 Paired Permutation Test
Under the null hypothesis $H_0$ that the task token $e$ does not bind to the visual target $P$, the assignment of token outputs $(g_1, g_2)$ to targets $(P_A, P_B)$ is exchangeable within each scene. For each permutation $k$, a Rademacher random variable $S_k \in \{-1, +1\}$ decides whether the token roles are preserved or swapped. The exact paired permutation $p$-value is:
$$p = \frac{1}{K} \sum_{k=1}^K \mathbb{I}(\bar{D}^{(k)} \ge \bar{D}_{\text{obs}})$$

### 2.4 Action-Conditioned Dynamics Transition
While sensor-derived displacement relies on observed consecutive latents:
$$\Delta s_{\text{sensor}} = s_t - s_{t-1}$$
the action-conditioned transition model predicts:
$$\hat{s}_{t+1} = T_{\text{micro}}(s_t, a_t) = s_t + \delta_s(s_t, a_t)$$
$$\Delta s_{\text{action}} = \hat{s}_{t+1} - s_t = \delta_s(s_t, a_t)$$
Displacement is decoded via:
$$\hat{d}_{\text{action}} = \text{micro\_ego\_head}([0, \Delta s_{\text{action}}])$$
This formulation provides motion estimation under sensory deprivation without camera observations.
