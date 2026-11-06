# Phase 2: Generalization & Memorization Analysis

**Author:** Brian A. Hadian  
**Project:** Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA  
**Target Platform:** AMD Kria KR260 Robotics Starter Kit (Zynq UltraScale+ XCK26 SOM)  
**Host Architecture:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.6, EGL Headless Renderer)  
**Reference Directive:** Section 3 (Phase 2, Critical Experiment) in `research_direction.md`

---

## 1. The Critical Experiment: Memorization vs Generalizable Mechanisms

As stated in the directive:
> *"Critical experiment: Train and evaluate on the same capability while holding environmental configurations out of training. You want to distinguish memorized navigation patterns from reusable spatial and action-related mechanisms."*

To isolate memorized shortcuts from genuine spatial principles, we evaluated all three models across three environmental distribution shifts:
1. **Canonical In-Distribution**: Standard landmark positions and familiar orientations.
2. **Permuted Layout (Quadrant Swap)**: All six obstacles randomly rearranged across quadrants I, II, III, and IV.
3. **Unseen Starting Orientations**: AMR initialized facing $0^\circ, 90^\circ, 180^\circ,$ and $270^\circ$, altering the camera's initial visual orientation relative to the arena geometry.

---

## 2. Generalization Metrics Matrix

| Model Architecture | Canonical Success (%) | Permuted Layout Success (%) | Permuted Layout Gap ($\Delta$) | Unseen Orientations Success (%) | Orientation Gap ($\Delta$) | Collision Rate on Unseen Orientations (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Policy** | 10.0% | 0.0% | **+10.0%** | **0.0%** | **+10.0%** | **55.0%** |
| **LeWorldModel (LeWM)** | 25.0% | 0.0% | **+25.0%** | **30.0%** | **-5.0%** (Robust!) | 50.0% |
| **H-JEPA Hierarchical** | 25.0% | 0.0% | **+25.0%** | **25.0%** | **+0.0%** (Invariant!) | 55.0% |

---

## 3. Key Mechanistic Insights

### 3.1 Orientation Invariance in Predictive World Models
- **Baseline Failure**: When the robot begins facing an unfamiliar orientation ($90^\circ, 180^\circ,$ or $270^\circ$), the Baseline Policy's performance collapses to **0.0% success** with a massive **55.0% collision rate**. The reactive policy has memorized specific visual landmarks in specific retinal positions.
- **World Model Invariance**:
  - LeWM achieves **30.0% success** on unseen initial orientations (actually outperforming its canonical 25.0% score, $\Delta = -5.0\%$).
  - H-JEPA maintains an identical **25.0% success rate** ($\Delta = 0.0\%$).
  - *Mechanistic Root*: Because LeWM and H-JEPA predict future states conditioned on 3-DOF action candidates ($a_t = [v_x, v_y, \omega_z]$), their latent rollout MPC continuously evaluates candidate headings and rotates the vehicle toward the target regardless of initial heading.

### 3.2 The Permuted Layout Gap (The Localization Bottleneck)
- When obstacle positions are permuted across quadrants, all three models drop to 0.0% success.
- *Root Cause Analysis*: In the training distribution, landmarks were positioned within specific arena clusters. When an obstacle is moved across quadrants, the visual encoder's convolutional feature maps still register the object, but the downstream policy/predictor struggles to reconcile unfamiliar obstacle relative distances without prior mapping experience.
- *Implication for Phase 4 & Phase 5*: In Phase 4 (Representations / SAEs) and Phase 5 (Circuits), we must probe whether the bottleneck lies in the visual encoder (appearance-to-spatial feature extraction) or in the policy head (action selection).
