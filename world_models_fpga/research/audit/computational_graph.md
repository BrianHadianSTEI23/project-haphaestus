# Computational Graph & Operational Boundaries
**Project:** Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA  
**Author:** Brian A. Hadian  
**Phase:** Phase 0 — Repository Audit & Experimental Reproducibility  
**Date:** November 6, 2026  

---

## 1. Operational Classification Taxonomy

To prevent confounding neural network capabilities with deterministic heuristics or offline training artifacts, all operations across the pipeline are strictly categorized into five mutually exclusive classes:

1. **[L] Learned Operations**: Operations parameterized by weights optimized via gradient descent.
2. **[NL] Non-Learned / Deterministic Operations**: Heuristic geometry, physics stepping, parsing, and thresholding.
3. **[TR] Training-Only Operations**: Backpropagation, AdamW optimizer, cosine annealing, and SIGReg projection penalties.
4. **[INF] Inference-Time Operations**: Forward passes, feature normalization, and sampling rollouts.
5. **[FPGA] Operations Accelerated in PL Hardware**: RTL modules mapped to AMD Kria KR260 FPGA fabric.

---

## 2. Global Computational Graph

```mermaid
flowchart TD
    classDef learned fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#ffffff;
    classDef nonlearned fill:#334155,stroke:#64748b,stroke-width:1px,color:#ffffff;
    classDef fpga fill:#065f46,stroke:#10b981,stroke-width:2px,color:#ffffff;
    classDef train fill:#831843,stroke:#f43f5e,stroke-width:1px,color:#ffffff;

    subgraph Perception ["1. Perception Pipeline"]
        RGB["Raw RGB Frame (64x64x3)<br/>[NL / INF]"]
        CONV1["Conv2d 3->16 (stride 2)<br/>[L / INF]"]
        CONV2["Conv2d 16->32 (stride 2)<br/>[L / INF]"]
        CONV3["Conv2d 32->64 (stride 2)<br/>[L / INF]"]
        CONV4["Conv2d 64->64 (stride 2)<br/>[L / INF]"]
        FC_ENC["Linear 1024->32 + LN<br/>[L / INF]"]
        LATENT["Latent State s_t in R^32<br/>[INF]"]
    end

    subgraph Semantics ["2. Semantic Grounding Pipeline"]
        TEXT["User Command String<br/>[NL]"]
        REGEX["Regex / Landmark Alias Match<br/>[NL]"]
        PREP_SOLV["Spatial Preposition Solver<br/>[NL]"]
        GOAL_VEC["Relative Goal Vector g_t in R^4<br/>[NL / INF]"]
    end

    subgraph TrainingLoss ["3. Training-Only Supervision"]
        SIGREG["SIGReg 1D Projections & Moments<br/>[TR]"]
        MSE_DYN["Next-Latent MSE Loss ||s_{t+1} - s*||^2<br/>[TR]"]
        BCE_COL["BCE Collision Risk Loss<br/>[TR]"]
        ADAMW["AdamW Optimizer Updates<br/>[TR]"]
    end

    subgraph Accelerators ["4. Hardware Acceleration Boundary (AMD Kria KR260)"]
        AXI_IN["AXI4-Lite Write Registers 0x10-0x9F<br/>[NL / FPGA]"]
        FPGA_MAC1["Layer 1 MAC Engine (36/35 -> 64, ReLU)<br/>[L / FPGA]"]
        FPGA_MAC2["Layer 2 MAC Engine (64 -> 64, ReLU)<br/>[L / FPGA]"]
        FPGA_OUT["Layer 3 Output Heads + Saturated Q8.8<br/>[L / FPGA]"]
        AXI_OUT["AXI4-Lite Read Registers 0xA0-0x13F<br/>[NL / FPGA]"]
    end

    subgraph Physics ["5. Physical Execution & Feedback"]
        ACT_NORM["Action Clipping [-1, 1]<br/>[NL]"]
        MUJOCO["MuJoCo AMR Physics (frame_skip=5)<br/>[NL]"]
        CAM_VERIF["HSV Segmentation & Verification<br/>[NL]"]
    end

    RGB --> CONV1 --> CONV2 --> CONV3 --> CONV4 --> FC_ENC --> LATENT
    TEXT --> REGEX --> PREP_SOLV --> GOAL_VEC

    LATENT --> AXI_IN
    GOAL_VEC --> AXI_IN
    AXI_IN --> FPGA_MAC1 --> FPGA_MAC2 --> FPGA_OUT --> AXI_OUT

    LATENT -.-> SIGREG -.-> ADAMW
    LATENT -.-> MSE_DYN -.-> ADAMW
    FPGA_OUT -.-> BCE_COL -.-> ADAMW

    AXI_OUT --> ACT_NORM --> MUJOCO --> RGB
    MUJOCO --> CAM_VERIF

    class CONV1,CONV2,CONV3,CONV4,FC_ENC learned;
    class RGB,TEXT,REGEX,PREP_SOLV,GOAL_VEC,ACT_NORM,MUJOCO,CAM_VERIF nonlearned;
    class FPGA_MAC1,FPGA_MAC2,FPGA_OUT,AXI_IN,AXI_OUT fpga;
    class SIGREG,MSE_DYN,BCE_COL,ADAMW train;
```

---

## 3. Acceleration Boundary Breakdown

The physical execution split between PS (ARM Cortex-A53) and PL (FPGA Fabric) benchmarked directly on the AMD Kria KR260 Starter Kit:

![Kria KR260 Baseline Audit Telemetry](kr260_baseline_audit_screenshot.png)

| Pipeline Stage | Executing Processor | Data Format | Compute Cost | Latency Profile (Empirical KR260) |
| :--- | :--- | :---: | :---: | :---: |
| **Observation Capture** | PS (Host Linux / MuJoCo) | `uint8` $[64, 64, 3]$ | Negligible (Offscreen buffer) | $\approx 2.1\,\text{ms}$ |
| **Visual Encoder (CNN)** | PS ARM Cortex-A53 | `float32` | 140,304 MACs | $4.62\,\text{ms}$ |
| **PS-to-PL Feature Transfer** | AXI4-Lite Bus (`0xA000_0010`) | Signed `Q8.8` (16-bit) | 36 Register Writes | $\approx 0.72\,\mu\text{s}$ |
| **Predictor / Policy Core** | **KR260 FPGA PL (UltraScale+)** | **Signed `Q8.8` (Fixed)** | **6,788 - 8,609 MACs** | **$0.470\,\mu\text{s}$ (47 clock cycles)** |
| **PL-to-PS Result Transfer** | AXI4-Lite Bus (`0xA000_00A0`) | Signed `Q8.8` (16-bit) | 4 to 32 Register Reads | $\approx 0.64\,\mu\text{s}$ |
| **Baseline Full Cortex-A53**| PS ARM Cortex-A53 (1.333 GHz) | `float32` | 147,028 params | $5.381\,\text{ms}$ ($185.8\,\text{FPS}$) |
| **Baseline PL Speedup** | KR260 PL vs Cortex-A53 Head | Signed `Q8.8` vs Float32 | MLP Forward | $\mathbf{1,238.6\times}$ |
| **Actuator Command & Sim** | PS (Linux / MuJoCo) | Continuous Body Velocities | 3-DOF Kinematic Integration | $0.10\,\text{s}$ control timestep (10 Hz) |
