# Model Interfaces & Calling Specifications
**Project:** Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA  
**Author:** Brian A. Hadian  
**Phase:** Phase 0 — Repository Audit & Experimental Reproducibility  
**Date:** November 6, 2026  

---

## 1. Baseline Navigation Policy Interface

### Forward Pass Specification
```python
# Function Signature:
def forward(self, img_tensor: torch.Tensor, goal_vec: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
```
* **Inputs**:
  - `img_tensor`: `torch.FloatTensor` of shape `(B, 3, 64, 64)`, values normalized in $[0.0, 1.0]$.
  - `goal_vec`: `torch.FloatTensor` of shape `(B, 4)`, representing $[\Delta x_{\text{body}}, \Delta y_{\text{body}}, \sin\Delta\theta, \cos\Delta\theta]$ in meters/radians.
* **Outputs**:
  - `(actions, col_risk)`:
    - `actions`: `torch.FloatTensor` of shape `(B, 3)`, continuous normalized velocities $[v_x, v_y, \omega_z] \in [-1.0, 1.0]^3$ ($\tanh$ bounded).
    - `col_risk`: `torch.FloatTensor` of shape `(B, 1)`, predicted collision probability $c \in [0.0, 1.0]$ ($\sigma$ bounded).
  - `z_vis`: `torch.FloatTensor` of shape `(B, 32)`, intermediate visual latent feature vector.

### Hardware Core Sub-Interface (FPGA Accelerator Input):
```python
def forward_with_latent(self, z_vis: torch.Tensor, goal_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
```
* Concatenates $[z_{\text{vis}}, g_{\text{vec}}] \in \mathbb{R}^{32 + 4 = 36}$, matching AXI-Lite register map `0x010` - `0x09C`.

---

## 2. LeWorldModel (LeWM) Interface

### Encode Specification
```python
def encode(self, img_tensor: torch.Tensor) -> torch.Tensor:
```
* **Input**: `img_tensor` `(B, 3, 64, 64)`, normalized $[0.0, 1.0]$.
* **Output**: `s_t` `(B, 32)`, unit-normalized latent state ($\text{LayerNorm}(32)$).

### Latent Transition Core Specification (FPGA Accelerator Input)
```python
def predict_next(self, s_t: torch.Tensor, a_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
```
* **Inputs**:
  - `s_t`: `torch.FloatTensor` of shape `(B, 32)`, current latent state.
  - `a_t`: `torch.FloatTensor` of shape `(B, 3)`, 3-DOF action vector in $[-1.0, 1.0]^3$.
  - Total input vector: 35 features in signed Q8.8 fixed-point format.
* **Outputs**:
  - `next_s`: `torch.FloatTensor` of shape `(B, 32)`, predicted future latent state $\hat{s}_{t+1} = s_t + \Delta s_t$.
  - `col_logit`: `torch.FloatTensor` of shape `(B, 1)`, unnormalized collision logit.

### Training Loss Specification
```python
def compute_loss(self, img_t: torch.Tensor, action_t: torch.Tensor, img_next: torch.Tensor, col_target: torch.Tensor, lambda_reg: float = 0.2) -> Tuple[torch.Tensor, Dict[str, float]]:
```
* Total Loss: $\mathcal{L} = \mathcal{L}_{\text{MSE}}(\hat{s}_{t+1}, \operatorname{sg}(s_{t+1})) + \lambda_{\text{reg}} \mathcal{L}_{\text{SIGReg}}(s_t) + 0.2 \mathcal{L}_{\text{BCE}}(\text{col\_logit}, c)$.

---

## 3. H-JEPA Interface

### Macro-Predictor Specification (Temporal Stride $K=5$)
```python
def forward(self, S_k: torch.Tensor, A_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
```
* **Inputs**:
  - `S_k`: `torch.FloatTensor` of shape `(B, 32)`, current macro latent state at step $t_k$.
  - `A_k`: `torch.FloatTensor` of shape `(B, 3)`, macro action / average velocity vector over $K=5$ steps.
* **Outputs**:
  - `next_S`: `torch.FloatTensor` of shape `(B, 32)`, predicted macro subgoal $\hat{S}_{k+1}$.
  - `col_logit`: `torch.FloatTensor` of shape `(B, 1)`, predicted macro collision logit.

### Micro-Predictor Specification (Temporal Stride 1)
```python
def select_action(self, s_t: torch.Tensor, S_subgoal: torch.Tensor) -> torch.Tensor:
```
* Conditioned on $[s_t, S_{\text{subgoal}}] \in \mathbb{R}^{32 + 32 = 64}$ to yield primitive action $a_t \in [-1.0, 1.0]^3$.

```python
def forward_transition(self, s_t: torch.Tensor, a_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
```
* Conditioned on $[s_t, a_t] \in \mathbb{R}^{32 + 3 = 35}$ to yield local transition $\hat{s}_{t+1} = s_t + \Delta s_t$.

---

## 4. Hardware AXI4-Lite Memory-Mapped Register Map

All models map to the unified memory-mapped register space on the AMD Kria KR260 (`Base Address: 0xA000_0000`):

| Address Offset | Register Name | Direction | Data Representation | Function |
| :--- | :--- | :---: | :---: | :--- |
| `0x000` | `REG_CTRL` | R/W | 32-bit uint | Bit 0: Start, Bit 1: Done, Bits 3:2: Model Mode (`00`: Baseline, `01`: LeWM, `10`: H-JEPA) |
| `0x004` | `REG_PERF` | R | 32-bit uint | Measured PL execution latency in clock cycles (10 ns / cycle) |
| `0x008` | `REG_IRQ` | R/W | 32-bit uint | Bit 0: Interrupt enable |
| `0x010` - `0x09C` | `IN_FEAT[0..35]` | W | Signed Q8.8 | 36 words: Latent state $s_t$ ($0..31$), Goal vector or action ($32..35$) |
| `0x0A0` - `0x11C` | `OUT_FEAT[0..31]` | R | Signed Q8.8 | 32 words: Predicted future latent $\hat{s}_{t+1}$ or actions ($v_x, v_y, \omega_z$) |
| `0x120` | `OUT_COL` | R | Signed Q8.8 | Predicted collision risk score |
