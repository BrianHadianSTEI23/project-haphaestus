# Phase 3: Activation Trace & Representation Catalog

**Author:** Brian A. Hadian  
**Project:** Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA  
**Target Platform:** AMD Kria KR260 Robotics Starter Kit (Zynq UltraScale+ XCK26 SOM)  
**Host Architecture:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.6, PyTorch 2.14.1+cu126)  
**Reference Directive:** Section 3 (Phase 3, "What to Capture") in `research_direction.md`

---

## 1. Mapping of the Ten Required Intermediate Representations

In accordance with Phase 3 instructions, the table below maps each of the ten mandatory mechanistic discovery representations to its exact PyTorch submodule path, tensor dimension, and functional role across all three architectures:

```
+--------------------------------------------------------------------------------------------------------------------+
|                                    INTERMEDIATE REPRESENTATION REGISTRY                                            |
+----+-----------------------------+---------------------------+---------------------------------+-------------------+
| #  | Representation Concept      | Target Model Architecture | PyTorch Submodule Path          | Tensor Shape      |
+----+-----------------------------+---------------------------+---------------------------------+-------------------+
| 1  | Raw Image Embedding         | Baseline Policy           | encoder.fc (baseline.visual_z)  | (B, 32) float32   |
|    |                             | LeWorldModel              | encoder[10] (lewm.latent_s)     | (B, 32) float32   |
|    |                             | H-JEPA Hierarchical       | encoder[10] (hjepa.shared_s)    | (B, 32) float32   |
+----+-----------------------------+---------------------------+---------------------------------+-------------------+
| 2  | Goal Embedding              | Baseline Policy           | goal_vec input                  | (B, 4) float32    |
|    |                             | LeWM / H-JEPA             | goal_dict["target_pose"]        | (B, 3) float32    |
+----+-----------------------------+---------------------------+---------------------------------+-------------------+
| 3  | Current World-State Rep     | LeWorldModel              | encoder[10] (lewm.latent_s)     | (B, 32) float32   |
|    |                             | H-JEPA Hierarchical       | encoder[10] (hjepa.shared_s)    | (B, 32) float32   |
+----+-----------------------------+---------------------------+---------------------------------+-------------------+
| 4  | Action Embedding            | LeWorldModel              | actions input to pred_fc1       | (B, 3) float32    |
|    |                             | H-JEPA Macro              | macro_actions to macro_fc1      | (B, 3) float32    |
|    |                             | H-JEPA Micro              | primitive actions to dyn_fc1    | (B, 3) float32    |
+----+-----------------------------+---------------------------+---------------------------------+-------------------+
| 5  | Predicted Future Rep        | LeWorldModel              | s_t + delta_s                   | (B, 32) float32   |
|    |                             | H-JEPA Macro (K=5)        | S_k + delta_S                   | (B, 32) float32   |
|    |                             | H-JEPA Micro (K=1)        | s_t + delta_s                   | (B, 32) float32   |
+----+-----------------------------+---------------------------+---------------------------------+-------------------+
| 6  | Prediction Residual         | LeWorldModel              | pred_delta (lewm.delta_s)       | (B, 32) float32   |
|    |                             | H-JEPA Macro              | macro_predictor.delta_head      | (B, 32) float32   |
|    |                             | H-JEPA Micro              | micro_predictor.delta_head      | (B, 32) float32   |
+----+-----------------------------+---------------------------+---------------------------------+-------------------+
| 7  | Macro-Level Latent Rep      | H-JEPA Hierarchical       | macro_predictor output S_{k+1}  | (B, 32) float32   |
+----+-----------------------------+---------------------------+---------------------------------+-------------------+
| 8  | Micro-Level Latent Rep      | H-JEPA Hierarchical       | micro_predictor output s_{t+1}  | (B, 32) float32   |
+----+-----------------------------+---------------------------+---------------------------------+-------------------+
| 9  | Policy Actuator Outputs     | Baseline Policy           | action_head (tanh)              | (B, 3) float32    |
|    |                             | LeWM / H-JEPA             | Latent MPC Argmin Selector      | (B, 3) float32    |
|    |                             | H-JEPA Micro Policy       | micro_predictor.action_head     | (B, 3) float32    |
+----+-----------------------------+---------------------------+---------------------------------+-------------------+
| 10 | Controller Outputs          | Baseline Policy           | collision_head (sigmoid)        | (B, 1) float32    |
|    | (Collision Risk Logits)     | LeWorldModel              | col_fc (lewm.col_head)          | (B, 1) float32    |
|    |                             | H-JEPA Macro              | macro_predictor.col_head        | (B, 1) float32    |
|    |                             | H-JEPA Micro              | micro_predictor.col_head        | (B, 1) float32    |
+----+-----------------------------+---------------------------+---------------------------------+-------------------+
```

---

## 2. Pathway Specificity by Architecture

### 2.1 Baseline Goal Fusion Pathway
In `BaselineNavigationPolicy`, goal information enters directly after the convolutional encoder:
$$\mathbf{x}_{\text{fused}} = [\mathbf{z}_{\text{vis}} \,\|\, \mathbf{g}_{\text{rel}}] \in \mathbb{R}^{32 + 4 = 36}$$
- Captured at `baseline.visual_latent_z` ($\mathbf{z}_{\text{vis}}$) and `baseline.policy_fc1` ($\mathbf{x}_{\text{fused}}$).
- Identifies how spatial prepositions modulate motor commands through dense weights $W_{\text{fc1}} \in \mathbb{R}^{64 \times 36}$.

### 2.2 LeWorldModel Encoder vs Predictor Separation
- **Encoder Pathway**: `lewm.conv1` $\to$ `lewm.conv4` $\to$ `lewm.flatten` $\to$ `lewm.latent_state_s` ($s_t \in \mathbb{R}^{32}$).
- **Predictor Pathway**: `[s_t, a_t]` $\to$ `lewm.pred_fc1` $\to$ `lewm.pred_fc2` $\to$ `lewm.delta_s` ($\Delta s$).
- Enables independent patching of current visual features vs future dynamics.

### 2.3 H-JEPA Hierarchical Level Decoupling
- **Shared Visual Encoder**: Captures universal semantic visual states $s_t$.
- **Macro Planning Level ($K=5$)**: Maps $(S_k, \bar{A}_{\text{macro}}) \to \hat{S}_{k+1}$.
- **Micro Execution Level ($K=1$)**: Maps $(s_t, S_{\text{subgoal}}^*) \to a_t$ and tracks fine dynamics $(s_t, a_t) \to \hat{s}_{t+1}$.
- Both levels instrumented independently for hierarchical causal discovery in Phase 5.
