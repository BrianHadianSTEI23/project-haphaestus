# Implementation Deviations & Literature Gap Analysis
**Project:** Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA  
**Author:** Brian A. Hadian  
**Phase:** Phase 0 — Repository Audit & Experimental Reproducibility  
**Date:** November 6, 2026  

---

## 1. Executive Summary of Audit Findings

In accordance with Section 2 ("Important corrections before starting") and Phase 0 of the Master Research Directive, this document identifies all architectural deviations between the current practical implementation and the theoretical formulations in the reference literature ([arXiv:2603.19312](https://arxiv.org/abs/2603.19312) and [arXiv:2610.06805](https://arxiv.org/abs/2610.06805)).

Documenting these deviations transparently is mandatory so that subsequent mechanistic discoveries are not distorted by invalid assumptions.

---

## 2. LeWorldModel (LeWM) Architectural Analysis (arXiv:2603.19312)

### Reference Paper Specification:
* **Authors**: Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero (Meta FAIR / NYU).
* **Concept**: End-to-end JEPA directly from raw pixels to latent space using next-embedding prediction and Sketched Isotropic Gaussian Regularization (**SIGReg**) to eliminate representation collapse without requiring stop-gradients, EMA target encoders, pixel decoders, or multi-term loss weighting.
* **Original Network Scale**: $\approx 15\,\text{Million}$ parameters, ViT / ResNet backbone, tested on 2D/3D DMControl tasks.

### Current Implementation in Repository:
1. **Backbone Scale**: Compressed to a compact 4-layer Convolutional Feature Pyramid (140,304 parameters) with 32-dim latent state $s_t \in \mathbb{R}^{32}$.
   - *Justification*: The 15M-parameter ViT backbone would exceed the on-chip BRAM (1.2 MB) and DSP slices (1,248 DSP48E2) of the AMD Kria XCK26 FPGA. Downscaling to 148k parameters enables full on-chip execution.
2. **SIGReg Loss Formulation**:
   - The paper projects latents onto $M$ 1D vectors $u_m$ and computes empirical characteristic functions or moment discrepancies.
   - Our implementation projects onto $M=32$ random unit vectors and computes penalties on $\mu=0, \sigma^2=1, \text{skew}=0, \text{kurt}=3$. This matches the paper's moment-matching formulation of SIGReg.
3. **Planning Horizon**:
   - The paper evaluates trajectory rollouts using latent MPC (CEM). Our repository implements a 6-step horizon ($0.6\,\text{s}$) with 24-32 candidate actions.

---

## 3. H-JEPA Architectural Analysis (arXiv:2610.06805)

### Reference Paper Specification:
* **Authors**: Wancong Zhang, Basile Terver, Michael Rabbat, Yann LeCun, Randall Balestriero.
* **Concept**: Hierarchical multi-timescale action-conditioned JEPA. Stacks multiple temporal abstraction levels (e.g. strides of 5, 10, 20 steps).
* **Planning**: Top-down planning where the highest level optimizes coarse macro trajectories in its own abstract latent space, passing predicted subgoals to intermediate levels down to primitive motor actions.

### Current Implementation in Repository:
1. **Hierarchy Depth**:
   - The paper demonstrates up to 3 hierarchical levels on Visual AntMaze.
   - Our implementation establishes **2 hierarchical tiers**:
     - **Macro Tier**: Stride $K=5$ ($0.5\,\text{s}$ at 10 Hz), predicting coarse subgoals $\hat{S}_{k+1}$.
     - **Micro Tier**: Stride $1$ ($0.1\,\text{s}$ primitive control loop), executing continuous 3-DOF velocities tracking the macro subgoal.
   - *Justification*: Stride $K=5$ is well-matched to the physical dynamics of the JetAuto AMR (top speed $0.6\,\text{m/s}$ moves $\approx 0.30\,\text{m}$ per macro step, matching obstacle clearance envelopes).
2. **Latent Space Alignment**:
   - The paper uses distinct latent dimensionalities across levels. Our implementation uses identical 32-dimensional latent vectors for both Macro and Micro levels, sharing the CNN visual encoder.

---

## 4. Hardware Accelerator Scope & Acceleration Boundary

> [!WARNING]
> **Critical Architectural Distinction (Directive Section 2.C)**:
> The reported $0.48\,\mu\text{s}$ (48 clock cycles @ 100 MHz) measurement is the **accelerator kernel compute latency** for the 3-layer MLP predictor / policy core in programmable logic (PL). It is NOT the complete end-to-end perception-to-actuation pipeline.

### Full Pipeline Latency Accounting:
1. **Camera Image Capture**: $2.1\,\text{ms}$ (Offscreen rendering / Astra camera driver)
2. **Visual Feature Encoder (CNN)**: $4.8\,\text{ms}$ on ARM Cortex-A53 CPU / $0.06\,\text{ms}$ on NVIDIA RTX 3050 GPU
3. **PS-to-PL Bus Transfer (AXI-Lite)**: $0.72\,\mu\text{s}$
4. **FPGA Core Accelerator Compute**: **$0.48\,\mu\text{s}$ (48 cycles)**
5. **PL-to-PS Bus Transfer (AXI-Lite)**: $0.64\,\mu\text{s}$
6. **Actuation & Physics Step**: $100\,\text{ms}$ (10 Hz nominal control rate)

### Implication for Research:
When evaluating FPGA speedup during the mechanistic circuit extraction phases, we must independently benchmark:
- **Kernel-level compute speedup**: $1,237\times$ over ARM CPU.
- **End-to-end inference speedup**: When the CNN encoder is executed on CPU/GPU vs. when the extracted circuit is fully integrated.
- The circuit extraction phase will target discovering whether the 140k-parameter CNN encoder can also be pruned or replaced with an extracted compact circuit to accelerate the entire end-to-end pipeline.
