# Architecture Inventory: Mechanistic Model Audit
**Project:** Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA  
**Author:** Brian A. Hadian  
**Phase:** Phase 0 — Repository Audit & Experimental Reproducibility  
**Date:** November 6, 2026  

---

## 1. Executive Summary of Evaluated Models

Three model architectures operate within the JetAuto Mecanum AMR environment. Each architecture contains a distinct combination of learned neural representations, learned dynamics predictors, and deterministic control wrappers.

| Model Identifier | Primary Computational Role | Total Trainable Parameters | Latent State Space | Planning Horizon | Key Literature Reference |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline Navigation Policy** | Goal-conditioned reactive visual control with collision scoring | **147,028** | $z \in \mathbb{R}^{32}$ | 1 step ($0.1\,\text{s}$) | Standard Vision-Language Actor-Critic |
| **LeWorldModel (LeWM)** | Non-generative action-conditioned visual foresight | **148,913** | $s \in \mathbb{R}^{32}$ | 6 steps ($0.6\,\text{s}$) | Maes et al., [arXiv:2603.19312](https://arxiv.org/abs/2603.19312) |
| **H-JEPA** | 2-tier hierarchical joint-embedding world model | **166,037** | $S, s \in \mathbb{R}^{32}$ | 2 levels (Macro $K=5$, Micro $1$) | Zhang et al., [arXiv:2610.06805](https://arxiv.org/abs/2610.06805) |

---

## 2. Module-by-Module Technical Breakdown

### 2.1 Model 1: Baseline Navigation Policy (`BaselineNavigationPolicy`)
* **File Location**: `world_models_fpga/models/baseline_policy.py`
* **Checkpoint File**: `world_models_fpga/checkpoints/baseline_best.pt` (Verified, SHA256 validated)
* **Total Parameters**: 147,028 (100% trainable during Phase 0 training)

#### Internal Submodules:
1. **`VisualFeatureEncoder` (140,240 parameters)**:
   - `conv1`: `Conv2d(3, 16, kernel_size=4, stride=2, padding=1)` $\to 32 \times 32 \times 16$ (784 params)
   - `conv2`: `Conv2d(16, 32, kernel_size=4, stride=2, padding=1)` $\to 16 \times 16 \times 32$ (8,224 params)
   - `conv3`: `Conv2d(32, 64, kernel_size=4, stride=2, padding=1)` $\to 8 \times 8 \times 64$ (32,832 params)
   - `conv4`: `Conv2d(64, 64, kernel_size=4, stride=2, padding=1)` $\to 4 \times 4 \times 64$ (65,600 params)
   - `fc`: `Linear(1024, 32)` $\to$ latent vector $z_t \in \mathbb{R}^{32}$ (32,800 params)
   - Activation: Elementwise $\operatorname{ReLU}$
2. **Core Policy MLP (6,788 parameters)**:
   - Input Concatenation: $[z_t, g_{\text{rel}}]$, where $g_{\text{rel}} = [\Delta x_{\text{body}}, \Delta y_{\text{body}}, \sin\Delta\theta, \cos\Delta\theta] \in \mathbb{R}^4$ (Total input dimension = 36)
   - `fc1`: `Linear(36, 64)` + $\operatorname{ReLU}$ (2,368 params)
   - `fc2`: `Linear(64, 64)` + $\operatorname{ReLU}$ (4,160 params)
   - `action_head`: `Linear(64, 3)` + $\tanh$ activation $\to [v_x, v_y, \omega_z] \in [-1, 1]^3$ (195 params)
   - `collision_head`: `Linear(64, 1)` + $\sigma$ activation $\to c_t \in [0, 1]$ (65 params)

---

### 2.2 Model 2: LeWorldModel (`LeWorldModel`)
* **File Location**: `world_models_fpga/models/le_world_model.py`
* **Checkpoint File**: `world_models_fpga/checkpoints/lewm_best.pt`
* **Total Parameters**: 148,913

#### Internal Submodules:
1. **JEPA Visual Pixel Encoder (140,304 parameters)**:
   - 4-layer convolutional feature pyramid identical to Baseline + `LayerNorm(32)` normalization layer (64 params).
   - Maps raw image $x_t \in \mathbb{R}^{3 \times 64 \times 64} \to s_t \in \mathbb{R}^{32}$.
2. **Latent Transition Predictor Core (8,544 parameters)**:
   - Concatenates $[s_t, a_t]$ where $s_t \in \mathbb{R}^{32}$ and $a_t \in \mathbb{R}^3$ (Total input dimension = 35).
   - `pred_fc1`: `Linear(35, 64)` + $\operatorname{ReLU}$ (2,304 params)
   - `pred_fc2`: `Linear(64, 64)` + $\operatorname{ReLU}$ (4,160 params)
   - `pred_delta`: `Linear(64, 32)` (2,080 params) $\to$ residual transition $\Delta s_t$.
   - Transition formulation: $\hat{s}_{t+1} = s_t + \Delta s_t$.
3. **Grounded Collision Risk Head (65 parameters)**:
   - `col_fc`: `Linear(64, 1)` $\to$ predicted collision logit $\hat{c}_{t+1}$.
4. **Regularization Component (Non-parametric / 0 weights)**:
   - `SIGRegLoss`: Sketched Isotropic Gaussian Regularizer with $M=32$ random 1D projections matching first 4 standardized moments ($\mu=0, \sigma^2=1, \text{skew}=0, \text{kurt}=3$).

---

### 2.3 Model 3: Hierarchical JEPA (`HJEPAModel`)
* **File Location**: `world_models_fpga/models/h_jepa.py`
* **Checkpoint File**: `world_models_fpga/checkpoints/hjepa_best.pt`
* **Total Parameters**: 166,037

#### Internal Submodules:
1. **Shared Visual Encoder**: Identical 140,304 parameter Conv4-LN architecture.
2. **Macro Predictor (8,609 parameters)**:
   - Operates at temporal stride $K=5$ ($0.5\,\text{s}$).
   - Maps $[S_k, A^{\text{macro}}_k] \in \mathbb{R}^{32 + 3}$ through 2-layer MLP (`fc1: 35->64`, `fc2: 64->64`) to predicted macro subgoal $\hat{S}_{k+1} = S_k + \Delta S_k$ and macro collision risk $\hat{C}_{k+1}$.
3. **Micro Tracking Predictor & Policy (17,124 parameters)**:
   - Micro Policy (`pol_fc1: 64->64`, `pol_fc2: 64->64`, `action_head: 64->3`): Conditioned on $[s_t, S_{\text{subgoal}}] \in \mathbb{R}^{32 + 32}$ to output primitive 3-DOF action $a_t = [v_x, v_y, \omega_z]^\top$.
   - Micro Transition Predictor (`dyn_fc1: 35->64`, `dyn_fc2: 64->64`, `delta_head: 64->32`, `col_head: 64->1`): Predicts local single-step dynamics $\hat{s}_{t+1} = s_t + \Delta s_t$.

---

## 3. Learned vs. Engineered Component Boundaries

The system must NOT be evaluated as a single black-box. The audit establishes the exact dividing lines between neural components and deterministic software:

```
[ Natural Language Command ] 
             │
             ▼
┌────────────────────────────────────────┐
│  ENGINEERED / DETERMINISTIC NLP LAYER  │  (nlp/grounding.py)
│  - String parsing & regex alias match  │  - Zero neural weights
│  - Spatial offset vector calculation   │  - Exact geometric formulas
│  - Clearance flank waypoint routing    │
└───────────────────┬────────────────────┘
                    │  Outputs: Target pose (x*, y*, theta*) & flank (xf, yf)
                    ▼
┌────────────────────────────────────────┐
│      LEARNED NEURAL NETWORK LAYER      │  (models/*.py)
│  - 4-layer visual convolution          │  - 147k - 166k learned weights
│  - Latent dynamics transition          │  - Quantized to Q8.8 fixed-point
│  - Collision risk logit prediction     │
└───────────────────┬────────────────────┘
                    │  Outputs: Action [vx, vy, wz] / Latent delta
                    ▼
┌────────────────────────────────────────┐
│   DETERMINISTIC CONTROLLER & HARNESS   │  (sim_app_3d/sim_app_3d.py)
│  - Holonomic velocity clipping         │  - MuJoCo physics integration
│  - Target arrival threshold checks     │  - Real-time video streaming
│  - Camera HSV color segmentation check │
└────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Audit Finding**: In the current interactive application, obstacle circumnavigation uses an engineered clearance flank waypoint within `SemanticGoalGrounder`, while the learned world model predicts forward collision probabilities. When conducting mechanistic discovery, we must test both end-to-end policy actions and isolated latent dynamics predictions without the engineered flank to expose raw model capabilities.
