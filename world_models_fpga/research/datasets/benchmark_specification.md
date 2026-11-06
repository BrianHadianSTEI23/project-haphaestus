# Phase 1: Controlled Intelligence Benchmark Specification

**Author:** Brian A. Hadian  
**Project:** Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA  
**Target Platform:** AMD Kria KR260 Robotics Starter Kit (Zynq UltraScale+ XCK26 SOM)  
**Host Architecture:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.6, EGL Headless Renderer)  
**Reference Directive:** Section 3 (Phase 1) & Master Research Directive in `research_direction.md`

---

## 1. Executive Summary & Objective

Standard end-to-end reinforcement learning or imitation learning policies evaluated on random navigation episodes suffer from **confounded attribution**: one cannot distinguish whether a robot reached a target because it recognized the target object, reasoned about spatial prepositions, or simply followed memorized spatial shortcuts.

In accordance with Section 3 (Phase 1) of the Master Research Directive, this benchmark establishes a **controlled experimental laboratory** in MuJoCo. By holding specific physical, visual, and semantic variables constant while systematically intervening on others, we isolate the internal neural representations responsible for:
1. Identifying objects grounded in natural language (`object_grounding`).
2. Relative spatial relation mapping (`spatial_reasoning`).
3. Reactive and predictive collision avoidance (`obstacle_avoidance`).
4. Action-conditioned forward world model simulation (`action_prediction`).
5. Multi-waypoint long-horizon navigation (`temporal_planning`).
6. Zero-shot layout and orientation invariance (`generalization`).

---

## 2. Six Controlled Capability Categories

Every capability is evaluated across five systematic difficulty tiers:
1. **Easy**: Direct line-of-sight, close proximity ($1.0 - 1.2\,\text{m}$), aligned robot heading.
2. **Difficult**: Distant start ($2.2 - 2.5\,\text{m}$), heading reversed ($180^\circ$ away), or opposite quadrant start.
3. **Counterfactual**: Exactly one semantic or physical property mutated while preserving others.
4. **Out-of-Distribution (OOD)**: Spatial coordinates or obstacle arrangements outside canonical clusters.
5. **Failure / Baseline Collision**: Trap configurations or un-steered paths establishing baseline failure modes.

```
+----------------------------------------------------------------------------------------------------+
|                                 6 CAPABILITY CATEGORIES                                            |
+--------------------------+----------------------------------------+--------------------------------+
| Capability               | Experimental Task                      | Measurement & Metrics          |
+--------------------------+----------------------------------------+--------------------------------+
| Object Grounding         | "Approach <color> <shape>"             | Goal Euclidean error, IoU      |
| Spatial Reasoning        | "Go behind / left / right of <object>" | Relative waypoint alignment    |
| Obstacle Avoidance       | Reach goal with intervening obstacle   | Collision rate, min clearance  |
| Action Prediction        | Canonical 3-DOF action impulse probes  | Latent transition MSE ||s'-s|| |
| Temporal Planning        | Multi-waypoint arena circumnavigation  | Episode success, horizon (T)   |
| Generalization           | Quadrant permutation, rotated heading  | Generalization transfer gap    |
+--------------------------+----------------------------------------+--------------------------------+
```

### 2.1 Object Grounding
- **Semantic Commands**: `"approach the {target}"`, `"navigate to {target}"`, `"reach the {target}"`.
- **Target Vocabulary**: 6 colored landmarks (`yellow_cylinder`, `red_box`, `blue_cylinder`, `green_cylinder`, `purple_box`, `cyan_cylinder`).
- **Core Mechanism Under Test**: Visual feature extraction binding linguistic token embeddings to spatial image coordinates in latent space $z \in \mathbb{R}^{32}$.

### 2.2 Relative Spatial Reasoning
- **Semantic Prepositions**:
  - `"behind"`: Radial vector outward from arena center: $\mathbf{p}_{\text{behind}} = \mathbf{p}_{\text{lmk}} + \hat{\mathbf{r}} \cdot d_{\text{clearance}}$.
  - `"in front of"`: Radial vector inward towards arena center: $\mathbf{p}_{\text{front}} = \mathbf{p}_{\text{lmk}} - \hat{\mathbf{r}} \cdot d_{\text{clearance}}$.
  - `"left of"`: Tangential counter-clockwise vector: $\mathbf{p}_{\text{left}} = \mathbf{p}_{\text{lmk}} + \hat{\boldsymbol{\tau}} \cdot d_{\text{clearance}}$.
  - `"right of"`: Tangential clockwise vector: $\mathbf{p}_{\text{right}} = \mathbf{p}_{\text{lmk}} - \hat{\boldsymbol{\tau}} \cdot d_{\text{clearance}}$.
- **Core Mechanism Under Test**: Allocentric vs egocentric coordinate transformations and relational vector arithmetic.

### 2.3 Obstacle Avoidance
- **Setup**: Direct straight path between robot start and goal is obstructed by an obstacle with variable lateral offset.
- **Tiers**:
  - Offset avoidance ($\Delta y = 0.35\,\text{m}$): minor course correction.
  - Centered obstruction ($\Delta y = 0.0\,\text{m}$): requires full circumnavigation with flanking maneuver.
  - Slalom / Chicane (OOD): dual sequential obstacles forming a narrow corridor.
- **Metrics**: Minimum obstacle clearance $d_{\min}$, collision flag $C \in \{0, 1\}$, path tortuosity ratio $\mathcal{T} = \frac{L_{\text{actual}}}{L_{\text{geodesic}}}$.

### 2.4 Action-Conditioned Prediction
- **Setup**: Open-loop execution of pure canonical 3-DOF Mecanum impulses:
  1. Pure forward translation: $[+v_x, 0, 0]$
  2. Pure reverse translation: $[-v_x, 0, 0]$
  3. Pure lateral left strafe: $[0, +v_y, 0]$
  4. Pure lateral right strafe: $[0, -v_y, 0]$
  5. Pure counter-clockwise yaw: $[0, 0, +\omega_z]$
  6. Pure clockwise yaw: $[0, 0, -\omega_z]$
  7. Diagonal holonomic strafe: $[+v_x, +v_y, 0]$
  8. Coordinated arc turn: $[+v_x, 0, +\omega_z]$
- **Core Mechanism Under Test**: Evaluates whether LeWM predictor and H-JEPA micro-dynamics accurately predict $s_{t+1}$ conditioned on 3-DOF action $a_t$ without visual leakage.

### 2.5 Temporal Planning
- **Setup**: Sequential multi-waypoint navigation ($W_1 \to W_2 \to W_3$) across opposite quadrants of the 5m $\times$ 5m arena.
- **Horizon**: $T \in [30, 50]$ timesteps ($3.0 - 5.0\,\text{s}$ at 10 Hz).
- **Core Mechanism Under Test**: Evaluates whether H-JEPA macro-latent $S_k$ ($K=5$) preserves goal persistence over extended temporal horizons.

### 2.6 Generalization
- **Permuted Arena Layout**: Obstacle positions permuted across quadrants I, II, III, and IV.
- **Orientation Invariance**: Initial robot headings uniformly sampled from $\{0, \frac{\pi}{2}, \pi, -\frac{\pi}{2}\}$.
- **Metrics**: Transfer drop $\Delta = \text{Success}_{\text{canonical}} - \text{Success}_{\text{permuted}}$.

---

## 3. Transition Data Schema

Every timestep is captured in a rigid 15-field transition record:

```python
transition = {
    "episode_id": str,              # e.g., "spatial_reasoning_difficult_0105"
    "timestep": int,                # 0 to T-1
    "rgb": np.ndarray,             # (64, 64, 3) uint8 Astra camera observation
    "next_rgb": np.ndarray,        # (64, 64, 3) uint8 observation at t+1
    "robot_pose": np.ndarray,      # (3,) float32 [x, y, theta] in Arena frame
    "robot_velocity": np.ndarray,  # (3,) float32 [vx, vy, omega_z] instantaneous
    "object_positions": np.ndarray,# (6, 3) float32 ground-truth obstacle coordinates
    "object_classes": np.ndarray,  # (6,) int32 [0=cylinder, 1=box]
    "object_colors": np.ndarray,   # (6, 4) float32 RGBA color values
    "language_instruction": str,   # e.g., "go behind the yellow cylinder"
    "grounded_goal": np.ndarray,   # (3,) float32 target waypoint [x_g, y_g, theta_g]
    "action": np.ndarray,          # (3,) float32 executed normalized command
    "next_action": np.ndarray,     # (3,) float32 lookahead command at t+1
    "collision": bool,             # Contact between robot chassis and obstacle/wall
    "goal_reached": bool,          # Distance to target waypoint < 0.22 m
    "episode_success": bool        # True if terminal state reached collision-free
}
```

> [!IMPORTANT]
> **Data Separation Guarantee**: Ground-truth physical variables (`robot_pose`, `object_positions`, `robot_velocity`) are strictly stored for post-hoc probing, SAE validation, and attribution graphs. Neural models operate strictly from visual observations (`rgb`) and natural language embeddings.

---

## 4. Dataset Partitioning (Zero Frame-Leakage Splits)

To prevent observation leakage across splits, partitioning is strictly performed by **environment configuration seeds**, ensuring that entire episodes with identical initial conditions or spatial trajectories never span across splits:

| Split Name | Episodes | Transitions | Purpose |
| :--- | :---: | :---: | :--- |
| **Train Benchmark** | 400 | ~7,500 | Training SAEs, diagnostic probes, and behavioral baselines |
| **Validation Benchmark** | 100 | ~1,900 | Hyperparameter tuning and validation loss monitoring |
| **Held-out Test Benchmark**| 100 | ~1,900 | Final zero-shot capability benchmarking |
| **Counterfactual Suite (E1-E7)**| 350 | ~6,500 | Direct causal intervention and invariance discovery |
| **Disambiguation Suite** | 100 | ~2,000 | Resolving ego- vs allocentric and perceptual ambiguities |
| **Total Benchmark** | **1,050** | **~19,800** | Full Phase 1 Benchmark Corpus |

---

## 5. Storage Format & Reproducibility

- Binary Shards: High-compression NumPy `.npz` archive files storing uint8 images and float32 kinematic tensors.
- Index Metadata: Companion `.json` files index every episode with seed, instruction, capability, difficulty tier, and success flags.
- Deterministic Reproduction: All episodes generated with explicit random seeds starting from `seed=42`, logged in the metadata.
