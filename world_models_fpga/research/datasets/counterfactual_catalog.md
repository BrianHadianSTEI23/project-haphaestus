# Phase 1: Counterfactual & Disambiguation Experiment Catalog

**Author:** Brian A. Hadian  
**Project:** Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA  
**Target Platform:** AMD Kria KR260 Robotics Starter Kit (Zynq UltraScale+ XCK26 SOM)  
**Host Architecture:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.6, EGL Headless Renderer)  
**Reference Directive:** Section 2.3 & Section 3 (Phase 1) in `research_direction.md`

---

## 1. Overview & Methodological Role

To uncover causal mechanisms in neural networks, observational correlations are insufficient: one must perform **controlled interventions**. 

If a neural representation changes when an object changes color, but the navigation goal does not, the representation is **entangled** with appearance rather than encoding pure spatial geometry. Conversely, if a latent circuit fires invariantly across position, orientation, and color variations, that circuit constitutes an authentic, reusable computational mechanism.

This catalog establishes:
1. **The Seven Controlled Counterfactual Intervention Suites (E1 – E7)**: Matched pairs $(E_0, E_i)$ differing strictly along one manipulated physical or semantic axis.
2. **The Four Disambiguation Probes**: Tasks designed to resolve fundamental frame-of-reference and perceptual ambiguities.

---

## 2. Seven Counterfactual Intervention Suites (E1 – E7)

Every test suite consists of 25 matched pairs (50 episodes, ~1,500 transitions), sharing the identical natural language instruction `"go behind the yellow cylinder"` and initial conditions, with exactly one factor intervened upon:

```
+------------------------------------------------------------------------------------------------------------------+
|                                    COUNTERFACTUAL INTERVENTION MATRIX (E1 - E7)                                  |
+----+-----------------------+------------------------------------------+------------------------------------------+
| ID | Intervention Type     | Physical Mutation Applied                | Expected Invariant Hypothesis            |
+----+-----------------------+------------------------------------------+------------------------------------------+
| E1 | Position Shift        | Landmark moved: [-1.2, -0.9] -> [0.8,-1.2] | Semantic meaning of "behind" preserved   |
| E2 | Color Shift           | Yellow cylinder colored bright blue/red  | Spatial relation invariant to appearance |
| E3 | Robot Heading         | Initial yaw rotated by +90 deg (+pi/2)   | Allocentric goal interpretation constant |
| E4 | Robot Start Position  | Robot starts in opposite quadrant (IV)   | Object-relative goal pose invariant      |
| E5 | Obstacle Insertion    | Intermediate cylinder placed on LOS      | Goal identity invariant under detour     |
| E6 | Shape Mutation        | Yellow cylinder replaced with yellow box | Relational reasoning invariant to shape  |
| E7 | Viewpoint Mutation    | Camera pitch inclined by +10 deg         | Grounding invariant to camera attitude   |
+----+-----------------------+------------------------------------------+------------------------------------------+
```

### Detailed Suite Profiles

#### E1 — Position Invariance
- **Baseline ($E_0$)**: Landmark at canonical coordinates $\mathbf{p}_0 = [-1.2, -0.9, 0.35]$.
- **Mutation ($E_1$)**: Landmark relocated to quadrant IV at $\mathbf{p}_1 = [0.8, -1.2, 0.35]$.
- **Causal Hypothesis**: The latent subspace encoding the spatial preposition *"behind"* must perform relative vector addition ($\mathbf{p}_{\text{target}} = \mathbf{p}_{\text{landmark}} + \Delta \mathbf{r}$) rather than memorizing fixed arena coordinates.

#### E2 — Appearance & Color Decoupling
- **Baseline ($E_0$)**: Yellow cylinder with RGBA $[0.95, 0.85, 0.1, 1.0]$.
- **Mutation ($E_2$)**: Target geometry rendered with deep blue RGBA $[0.15, 0.35, 0.85, 1.0]$.
- **Causal Hypothesis**: The object grounding circuit should fail or redirect if bound to color tokens, while the downstream spatial reasoning and motor policy heads should remain functionally decoupled from color encodings.

#### E3 — Heading & Frame of Reference
- **Baseline ($E_0$)**: Robot initial heading $\theta_0 = -2.49\,\text{rad}$ (facing landmark).
- **Mutation ($E_3$)**: Robot initial heading rotated by $+90^\circ$ ($\theta_3 = -0.92\,\text{rad}$, facing tangent).
- **Causal Hypothesis**: Allocentric spatial reasoning must rotate the perceived body-frame vector by $-\theta$ to maintain world-relative target stability.

#### E4 — Starting Quadrant Shift
- **Baseline ($E_0$)**: Robot starts at arena origin $(0, 0)$.
- **Mutation ($E_4$)**: Robot starts in quadrant II at $(1.0, 1.0)$ facing $-135^\circ$.
- **Causal Hypothesis**: The trajectory path changes completely, but the final convergence point ($\mathbf{p}_{\text{goal}}$) remains invariant.

#### E5 — Intervening Obstacle Insertion
- **Baseline ($E_0$)**: Clear direct path between robot and yellow cylinder flank.
- **Mutation ($E_5$)**: Intermediate obstacle inserted at $(-0.6, -0.45)$, directly on the line of sight.
- **Causal Hypothesis**: The predictive collision-avoidance circuit must activate, injecting a lateral avoidance command while preserving the final goal attractor.

#### E6 — Geometric Shape Mutation
- **Baseline ($E_0$)**: Target geometry is a cylinder ($r = 0.25\,\text{m}, h = 0.35\,\text{m}$).
- **Mutation ($E_6$)**: Target geometry replaced with a rectangular prism ($0.25 \times 0.25 \times 0.35\,\text{m}$).
- **Causal Hypothesis**: Spatial prepositions like *"behind"* are defined with respect to the landmark centroid; shape contours should not alter relational waypoint computation.

#### E7 — Viewpoint & Sensor Pitch Perturbation
- **Baseline ($E_0$)**: Astra camera mounted at standard tilt angle ($5^\circ$ downward).
- **Mutation ($E_7$)**: Camera pitch tilted downward by $+10^\circ$ ($15^\circ$ total tilt), compressing the horizon in the upper field of view.
- **Causal Hypothesis**: Robust spatial representations should tolerate pitch variations without degrading linear velocity commands.

---

## 3. Four Disambiguation Probes

Ambiguous instructions and optical illusions pose major failure modes for spatial AI. We define four explicit disambiguation tasks:

```
+------------------------------------------------------------------------------------------------------------------+
|                                        DISAMBIGUATION PROBE MATRIX                                               |
+----+----------------------------+------------------------------------------+-------------------------------------+
| ID | Probe Name                 | Physical & Semantic Setup                | Target Disambiguation Focus         |
+----+----------------------------+------------------------------------------+-------------------------------------+
| D1 | Ego vs Allocentric Left    | Robot facing South (theta = -pi/2)       | Ego Left (+x / East) vs Allocentric |
|    |                            | Command: "go to the left"                | Left (-x / West)                    |
+----+----------------------------+------------------------------------------+-------------------------------------+
| D2 | Ego Front vs Object Behind | Robot facing landmark from outside       | Camera Forward vs Radial Detour     |
|    |                            | Command: "go behind the yellow cylinder" | Behind Landmark                     |
+----+----------------------------+------------------------------------------+-------------------------------------+
| D3 | Geometric vs Visual Scale  | 2x large box placed at 2x distance       | Apparent Retinal Area vs True       |
|    |                            | Subtends identical 18.5 deg FOV          | Physical Metric Euclidean Range     |
+----+----------------------------+------------------------------------------+-------------------------------------+
| D4 | Visibility vs Reachability | Target visible through 0.15m gap         | Optical Line of Sight vs            |
|    |                            | (Chassis width = 0.26m: impassable)      | Kinematic Passability / Reachability|
+----+----------------------------+------------------------------------------+-------------------------------------+
```

### D1 — Egocentric Left vs Allocentric West
- When the AMR faces South ($\theta = -\frac{\pi}{2}$), its egocentric left ($+y_{\text{body}}$) points East ($+x_{\text{world}}$). An allocentric interpretation of *"left"* maps to West ($-x_{\text{world}}$).
- **Benchmark Resolution**: We annotate both interpretations to measure whether the model's policy head naturally aligns with ego- or allocentric frames.

### D2 — Retinal Front vs Object Behind
- A common failure in spatial grounding is driving straight toward an object when commanded to go *"behind"* it, because the object appears directly in front of the camera.
- **Benchmark Resolution**: Measures whether the policy plans a flanking maneuver around the landmark or collides head-on into the frontal face.

### D3 — Apparent Retinal Area vs Metric Euclidean Range
- A red box of size $0.5 \times 0.5\,\text{m}$ at distance $2.69\,\text{m}$ produces the exact same pixel footprint as a $0.25 \times 0.25\,\text{m}$ box at $1.35\,\text{m}$.
- **Benchmark Resolution**: Probes whether the latent features encode metric depth or merely 2D visual area.

### D4 — Optical Visibility vs Kinematic Reachability
- A blue cylinder is clearly visible through a narrow gap ($0.15\,\text{m}$), but the AMR chassis width is $0.26\,\text{m}$.
- **Benchmark Resolution**: Tests whether the model attempts an impossible straight path or recognizes the barrier and takes the necessary detour.
