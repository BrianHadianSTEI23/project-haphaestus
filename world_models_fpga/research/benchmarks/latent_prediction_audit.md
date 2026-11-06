# Phase 2: Action-Conditioned Latent Prediction Audit

**Author:** Brian A. Hadian  
**Project:** Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA  
**Target Platform:** AMD Kria KR260 Robotics Starter Kit (Zynq UltraScale+ XCK26 SOM)  
**Host Architecture:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.6, EGL Headless Renderer)  
**Reference Directive:** Section 3 (Phase 2, Future Latent Prediction Evaluation) in `research_direction.md`

---

## 1. Objective & Mandate

As specified in `research_direction.md`:
> *"For action-conditioned prediction, evaluate the predicted future latent representation against the target encoder representation. For H-JEPA, evaluate each temporal level independently."*

This audit measures the accuracy of the world models' internal forward simulation engines against actual visual observations encoded by the frozen visual target encoder:
1. **LeWorldModel (LeWM)**: Single-step transition model $\hat{s}_{t+1} = s_t + T_{\text{predictor}}(s_t, a_t)$.
2. **H-JEPA Micro-Level ($K=1$)**: Single-step primitive dynamics $\hat{s}_{t+1} = s_t + T_{\text{micro}}(s_t, a_t)$.
3. **H-JEPA Macro-Level ($K=5$)**: Multi-step trajectory model $\hat{S}_{k+1} = T_{\text{macro}}(S_k, \bar{A}_{\text{macro}})$ across $0.5\,\text{s}$ planning horizons.

All models were evaluated across **2,885 held-out test transitions** and **2,405 contiguous macro sequences** ($K=5$).

---

## 2. Latent Prediction Audit Results

| Model & Hierarchy Tier | Horizon ($T$) | Latent MSE $||\hat{s} - s^*||_2^2$ | Cosine Similarity ($\cos\theta$) | Collision Prediction Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: |
| **LeWorldModel (LeWM)** | 1 step ($0.1\,\text{s}$) | 0.08479 | **0.9552** | **83.3%** |
| **H-JEPA Micro-Level** | 1 step ($0.1\,\text{s}$) | **0.04461** | **0.9751** | — |
| **H-JEPA Macro-Level** | 5 steps ($0.5\,\text{s}$) | 0.41986 | **0.7726** | **81.3%** |

---

## 3. Analysis of Hierarchical Dynamics

### 3.1 Micro-Level Precision ($K=1$)
- Both LeWM and H-JEPA achieve outstanding directional alignment with the ground-truth visual latent space ($0.9552$ and $0.9751$ cosine similarity, respectively).
- H-JEPA Micro-Level achieves an ultra-low MSE of **0.04461**, demonstrating that conditioning the low-level policy on macro subgoals stabilizes the single-step forward dynamics.

### 3.2 Macro-Level Subgoal Prediction ($K=5$)
- Over a multi-step horizon of $K=5$ control steps ($0.5\,\text{s}$), H-JEPA maintains a strong cosine similarity of **0.7726**.
- The macro collision prediction head achieves **81.3% accuracy** in forecasting whether a 5-step action trajectory will trigger a collision with an obstacle or arena boundary.
- *Significance for Hardware Acceleration*: Macro-level planning requires evaluating the predictor only once every 5 timesteps (2 Hz instead of 10 Hz), reducing FPGA activation memory bandwidth by $5\times$ while maintaining high collision avoidance fidelity.
