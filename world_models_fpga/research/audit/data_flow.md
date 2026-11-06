# Data Flow & Coordinate Reference Frames
**Project:** Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA  
**Author:** Brian A. Hadian  
**Phase:** Phase 0 — Repository Audit & Experimental Reproducibility  
**Date:** November 6, 2026  

---

## 1. Coordinate Reference Systems & Conventions

Accurate causal interpretability requires rigorous tracking of coordinate frames. The project uses four distinct coordinate systems:

```
+-----------------------------------------------------------------------------------------+
|                                    ARENA WORLD FRAME                                    |
| Origin: Center of floor (0.0, 0.0, 0.0). Bounds: X in [-2.5, 2.5] m, Y in [-2.5, 2.5] m. |
| Heading: 0 rad aligns with +X axis. Counter-clockwise rotation is positive.            |
+-----------------------------------------------------------------------------------------+
                                             │
                       Rotation Matrix R(-theta) & Translation
                                             ▼
+-----------------------------------------------------------------------------------------+
|                                  ROBOT AMR BODY FRAME                                   |
| Origin: Center of AMR wheelbase footprint.                                              |
| +X_body: Forward direction (Chassis nose). Velocity v_x in [-0.6, +0.6] m/s.           |
| +Y_body: Lateral strafe left (Mecanum 45 deg rollers). Velocity v_y in [-0.4, +0.4] m/s.|
| +Z_body: Vertical upward axis. Yaw rate omega_z in [-1.5, +1.5] rad/s.                  |
+-----------------------------------------------------------------------------------------+
                                             │
                        Rigid Body Offset [x=+0.105m, z=+0.165m]
                                             ▼
+-----------------------------------------------------------------------------------------+
|                               ASTRA PRO SENSOR CAMERA FRAME                             |
| Pose: Mounted at x=+0.105m, z=+0.165m relative to body (total z=0.213m above ground).   |
| Optical Axis: Looking forward (+X_body), 5 deg downward tilt toward floor.              |
| FOV: 60 degrees horizontal field of view. Offscreen Buffer: (64, 64, 3) RGB uint8.     |
+-----------------------------------------------------------------------------------------+
                                             │
                              OpenCV HSV Transform & Projection
                                             ▼
+-----------------------------------------------------------------------------------------+
|                                  IMAGE PLANE COORDINATES                                |
| Resolution: u in [0, 63] (horizontal column), v in [0, 63] (vertical row).              |
| Normalized Centering Error: u_err = (u - 32) / 32 in [-1.0, 1.0].                       |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Velocity Transformation & Kinematics

The Hiwonder JetAuto Mecanum AMR operates with continuous 3-DOF planar holonomic kinematics:

$$\begin{bmatrix} v_{x,\text{world}} \\ v_{y,\text{world}} \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} v_{x,\text{body}} \\ v_{y,\text{body}} \end{bmatrix}$$

$$\omega_{z,\text{world}} = \omega_{z,\text{body}}$$

The raw continuous control input $a = [a_0, a_1, a_2] \in [-1.0, 1.0]^3$ maps to physical velocities via:

$$v_{x,\text{body}} = a_0 \times 0.6\,\text{m/s}$$
$$v_{y,\text{body}} = a_1 \times 0.4\,\text{m/s} \quad (\text{Mecanum strafe slip calibration})$$
$$\omega_{z,\text{body}} = a_2 \times 1.5\,\text{rad/s}$$

*Simulation Physics Timestep*: MuJoCo base timestep is $\Delta t_{\text{sim}} = 0.02\,\text{s}$. With `frame_skip = 5`, the high-level policy and world model execute at:

$$\Delta t_{\text{ctrl}} = 0.02 \times 5 = 0.10\,\text{s} \quad (\text{Control Rate } f_s = 10.0\,\text{Hz})$$

---

## 3. End-to-End Data Flow Pipeline

### Step 1: Semantic Parsing & Goal Formulation
1. Text command parsed by `SemanticGoalGrounder.parse_command(text, robot_pose)`
2. Target landmark extracted $\to \mathbf{p}_{\text{lmk}} = (x_{\text{lmk}}, y_{\text{lmk}})$
3. Spatial preposition evaluated $\to$ target waypoint $\mathbf{p}^* = (x^*, y^*)$ and desired orientation $\theta^* = \operatorname{atan2}(y_{\text{lmk}} - y^*, x_{\text{lmk}} - x^*)$.
4. Flank waypoint $\mathbf{p}_{\text{flank}} = \mathbf{p}_{\text{lmk}} \pm \mathbf{n}_{\text{tangent}} \times 0.70\,\text{m}$ calculated for circumnavigating behind obstacles.

### Step 2: Observation & Latent Feature Extraction
1. Onboard Astra camera renders first-person frame: $I_t \in \mathbb{R}^{64 \times 64 \times 3}$, normalized to $[0.0, 1.0]$.
2. Visual encoder projects $I_t \to s_t \in \mathbb{R}^{32}$.
3. Relative goal vector computed in robot body frame:
   $$\begin{bmatrix} \Delta x_{\text{body}} \\ \Delta y_{\text{body}} \end{bmatrix} = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x^* - x_{\text{robot}} \\ y^* - y_{\text{robot}} \end{bmatrix}$$
   $$\mathbf{g}_{\text{rel}} = [\Delta x_{\text{body}}, \Delta y_{\text{body}}, \sin(\theta^* - \theta), \cos(\theta^* - \theta)] \in \mathbb{R}^4$$

### Step 3: Model-Specific Planning & Inference
* **Baseline Policy**:
  - Concatenates $[s_t, \mathbf{g}_{\text{rel}}] \to \text{MLP} \to [v_x, v_y, \omega_z], \hat{c}$.
* **LeWorldModel (LeWM)**:
  - Samples $N=24$ candidate velocity actions $a^{(i)} = [v_x, v_y, \omega_z]^{(i)}$.
  - Predicts future latents $\hat{s}_{t+1}^{(i)} = s_t + T_\theta(s_t, a^{(i)})$ and collision probabilities $\hat{c}^{(i)} = \sigma(\text{col\_head}(h^{(i)}))$.
  - Minimizes cost $\mathcal{J} = \Delta d_{\text{goal}} + \lambda_{\text{col}} \hat{c}^{(i)}$ to pick optimal action.
* **H-JEPA**:
  - Macro predictor evaluates coarse displacement candidates over stride $K=5$ ($0.5\,\text{s}$).
  - Micro tracking policy outputs primitive action $a_t$ conditioned on $[s_t, \hat{S}_{\text{macro}}]$.

### Step 4: Verification via Closed-Loop Camera Servoing
1. HSV thresholding extracts candidate target pixels.
2. Centroid $(u, v)$ computed; if $N_{\text{pixels}} \ge 0.015 \times 64 \times 64$, visual target is acquired (`visual_acquired = True`).
3. Goal verified when $\|\mathbf{p}_{\text{robot}} - \mathbf{p}^*\| \le 0.35\,\text{m}$ and visual lock confirmed.
