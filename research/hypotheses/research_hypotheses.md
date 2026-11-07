# Falsifiable Research Hypotheses (`research_hypotheses.md`)

**Central Research Question:**  
> What computational mechanisms allow an embodied agent to construct, maintain, and use spatial knowledge to predict environmental consequences, navigate toward goals, generalize across changes in its environment and embodiment, and act safely under temporal constraints?

All hypotheses below are preregistered prior to executing repaired experiments. Every candidate principle must be evaluated against the 9-point scientific standard: (1) Definition, (2) Mechanism, (3) Behavioral relevance, (4) Causal evidence, (5) Necessity, (6) Sufficiency, (7) Transfer, (8) Prediction, and (9) Limitations.

Evidence levels follow the strict ladder:
- **`E0 — Speculative`**: Plausible hypothesis, no direct empirical evidence.
- **`E1 — Observational`**: Measured correlation, probe decodability, or descriptive similarity without causal intervention.
- **`E2 — Interventional`**: Controlled causal intervention (ablation, activation patching, or closed-loop mechanism removal/restoration) demonstrating a statistically significant effect on task behavior or target output relative to random controls.
- **`E3 — Generalized`**: Replicated causal evidence across independent seeds, held-out layouts, counterfactual transformations, or embodiments, with verified predictive power on an experiment not used to formulate the hypothesis.

---

## Hypothesis H1 — Action-Conditioned Latent Predictive Planning

1. **Definition:** An embodied world model encodes visual observations $o_t \mapsto s_t = E_\phi(o_t) \in \mathbb{R}^d$ and learns an action-conditioned residual transition operator $\hat{s}_{t+1} = s_t + T_\theta(s_t, a_t)$ coupled with a learned goal-progress estimator $U_\psi(\hat{s}_{t+1}, g)$ and collision-risk estimator $r_\omega(\hat{s}_{t+1}, a_t)$ such that selecting $a_t^* = \arg\max_{a \in \mathcal{A}} [U_\psi(\hat{s}_{t+1}, g) - \lambda r_\omega(\hat{s}_{t+1}, a)]$ improves closed-loop goal reaching and collision avoidance over a reactive policy without access to privileged simulator pose.
2. **Variables & Measurements:**
   - Independent variables: candidate action $a_t$, action-conditioning toggle ($a_t$ vs. zero/shuffled $a_t$), prediction-use toggle (scoring via $\hat{s}_{t+1}$ vs. ignoring $\hat{s}_{t+1}$).
   - Dependent variables: 1-step and multi-step latent prediction MSE $\|\hat{s}_{t+1} - s_{t+1}\|_2^2$, cosine similarity, closed-loop goal-reaching rate, collision rate, and goal distance on held-out episodes across Track B (structured egocentric goal cue) and Track C (observation + discrete task ID only).
3. **Alternative Explanations:**
   - Apparent planning success could arise from privileged analytical pose propagation in the wrapper (as in the legacy `model_wrappers.py`) or from a purely reactive mapping $s_t \mapsto a_t$ where $\Delta s = T_\theta(s_t, a_t)$ has negligible influence.
4. **Falsification Criteria:**
   - H1 is **falsified** at the behavioral level if: (a) removing privileged pose propagation causes the world-model planner to fail to beat a reactive baseline, or (b) zeroing/shuffling the action input $a_t$ to $T_\theta(s_t, a_t)$ or setting $\Delta s = 0$ produces no statistically significant degradation in closed-loop navigation or multi-step prediction.

---

## Hypothesis H2 — Egocentric Relational Spatial Representation

1. **Definition:** The visual encoder and downstream policy represent target and obstacle locations primarily in robot body-frame (egocentric polar/Cartesian) coordinates $\mathbf{v}_{\text{body}} = R(-\theta_{\text{robot}})(\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{robot}})$ rather than world-frame allocentric coordinates $\mathbf{p}_{\text{obj}}$, inducing group equivariance of body-frame vectors under planar robot translations and yaw rotations $g \in SE(2)$.
2. **Variables & Measurements:**
   - Linear ridge probe $R^2$ and RMSE (tuned on validation split, evaluated on held-out test split across 3 training seeds and compared against an untrained random CNN baseline) for:
     - Egocentric relative object vector $(\Delta x_{\text{body}}, \Delta y_{\text{body}})$ and bearing $\phi_{\text{body}}$
     - Obstacle clearance $d_{\min}$
     - Allocentric robot pose $(x, y, \theta)$ and world object position $(x_{\text{obj}}, y_{\text{obj}})$
   - Disambiguation suite (`DISAMBIG_01` ego-left vs. world-left, `DISAMBIG_02` ego-front vs. object-behind) and Counterfactual `E3` (90° robot yaw rotation) & `E4` (robot start position shift).
3. **Alternative Explanations:**
   - In a fixed canonical arena with visible background walls and multiple colored landmarks, a single first-person camera frame contains global landmark constellations that allow decoding *both* allocentric position and egocentric bearing, or random convolutional features may already support high probe $R^2$ without learned specialization.
4. **Falsification Criteria:**
   - The strong claim of "pure egocentric decoupling ($R^2_{\text{world}} = 0$)" is **falsified** if world position/orientation is also decodable above random baseline, or if trained encoder decodability does not exceed an untrained random CNN projection, or if rotating the robot (`E3`) causes severe closed-loop failure in observation-only mode when the target leaves the camera field of view.

---

## Hypothesis H3 — Hierarchical Multi-Timescale Temporal Abstraction ($K=5$)

1. **Definition:** A two-tier hierarchical world model (H-JEPA) that predicts a $K$-step macro subgoal $\hat{S}_{t+K} = S_t + T_{\text{macro}}(S_t, a_{t:t+K-1})$ from an ordered $K$-step action sequence and conditions a trained micro policy $a_t = \pi_{\text{micro}}(s_t, \hat{S}_{t+K})$ improves long-horizon multi-waypoint navigation and reduces planning compute relative to single-step ($K=1$) prediction.
2. **Variables & Measurements:**
   - Macro latent prediction MSE and macro collision AUROC at exact temporal alignment $t \to t+K$ across horizons $K \in \{1, 3, 5, 8\}$.
   - Closed-loop behavioral ablation comparing: (a) full H-JEPA ($\hat{S}_{t+K}$ drives $\pi_{\text{micro}}$), (b) macro-bypassed H-JEPA ($\hat{S}_{t+K} = \mathbf{0}$ or $\hat{S}_{t+K} = s_t$), and (c) random subgoal injection.
3. **Alternative Explanations:**
   - If $\pi_{\text{micro}}$ simply memorizes a reactive mapping from $s_t$ while ignoring $\hat{S}_{t+K}$, or if $K=5$ prediction is less accurate than $K=1$ rollout due to compounding uncertainty, macro hierarchy adds parameters without functional benefit.
4. **Falsification Criteria:**
   - H3 is **falsified** if ablating $\hat{S}_{t+K}$ (replacing it with $s_t$ or zeros) does not degrade $\pi_{\text{micro}}$ action accuracy or closed-loop task performance, or if $K=5$ planning fails to outperform $K=1$ planning on multi-waypoint tasks.

---

## Hypothesis H4 — Anticipatory Collision-Risk Estimation (Predictive Safety)

1. **Definition:** Training a collision-prediction head $r_\omega(s_t, a_t) = \sigma(W_{\text{col}} h(s_t, a_t))$ on action-conditioned transitions yields a calibrated collision-risk estimator whose risk score increases monotonically as obstacle clearance decreases for forward-moving actions, and penalizing candidate actions by $r_\omega(s_t, a_t)$ causally reduces physical collision rates in closed-loop navigation.
2. **Variables & Measurements:**
   - Held-out collision classification AUROC, Brier calibration score, and Spearman rank correlation between predicted risk $r_\omega(s_t, a_t)$ and negative obstacle clearance $-d_{\min}$.
   - Closed-loop collision rate on `obstacle_avoidance` and `E5_intermediate_obstacle` suites with safety penalty enabled ($\lambda_{\text{col}} > 0$) vs. ablated ($\lambda_{\text{col}} = 0$).
3. **Alternative Explanations:**
   - Because collisions are rare in expert demonstrations (class imbalance), a naive classifier can achieve $>80\%$ accuracy by predicting "no collision" everywhere while having poor AUROC/recall and zero causal effect on obstacle avoidance.
4. **Falsification Criteria:**
   - H4 is **falsified** if held-out collision AUROC is near chance ($\le 0.55$), or if ablating the collision head ($\lambda_{\text{col}} = 0$) does not increase collision rate in closed-loop obstacle avoidance.

---

## Hypothesis H5 — Task-Relevant Appearance Invariance vs. Grounding Sensitivity

1. **Definition:** Self-supervised joint-embedding prediction (LeWM / H-JEPA) exhibits lower latent sensitivity to object color mutations (`E2_color_shift`) than supervised policies, while preserving geometric sensitivity (`E1_position_shift`, `E5_intermediate_obstacle`, `E6_shape_mutation`).
2. **Variables & Measurements:**
   - Normalized counterfactual latent sensitivity $S_k(E_i) = \mathbb{E}[\|z(E_0) - z(E_i)\| / \sigma_z]$ across all 7 physically verified counterfactual suites `E1`–`E7`.
   - Closed-loop transfer retention on `E2_color_shift` (mutated target color) vs. `E6_shape_mutation` (cylinder $\to$ box) and `E7_camera_viewpoint` (+10° pitch).
3. **Alternative Explanations:**
   - An encoder that is completely insensitive to color (`E2`) cannot distinguish a "yellow cylinder" from a "blue cylinder" when the task instruction is color-grounded ("go behind the yellow cylinder") unless target identity is supplied externally! Thus, total color invariance is actually a *liability* for color-referenced object grounding, whereas partial invariance or late task-conditioned filtering is required.
4. **Falsification Criteria:**
   - H5 is **falsified** if: (a) JEPA encoders show equal or higher `E2` sensitivity than the baseline encoder on physically verified counterfactuals, or (b) ablating color-sensitive dimensions impairs color-grounded navigation in Track C.

---

## Hypothesis H6 — Principle-Guided Architecture Compression & Hardware Equivalence

1. **Definition:** Removing modules that have zero inference-time dataflow contribution (e.g., H-JEPA's 1-step micro dynamics head when macro planning drives micro policy) and distilling into a lower-width student (`ReconstructedCompactModel`, 39,804 parameters) via genuine teacher representation + prediction distillation preserves closed-loop navigation performance while enabling bit-exact Q8.8 fixed-point execution on an FPGA feature-conditioned accelerator.
2. **Variables & Measurements:**
   - Structural parameter count, nonzero parameter count, and exact MACs.
   - Comparison of: (a) Supervised compact training from scratch, (b) Behavior cloning only, (c) Genuine teacher representation + prediction distillation, and (d) Weight-preserving structural module removal.
   - Closed-loop behavioral success and collision rates across Track A/B/C.
   - Bit-exact equivalence $\|y_{\text{RTL}} - y_{\text{Q8.8}}\|_2 = 0$, quantization error $\|y_{\text{Q8.8}} - y_{\text{FP32}}\|_2 / \|y_{\text{FP32}}\|_2$, exact RTL cycle count (4,327 cycles per single candidate pass; $24 \times 2,179 + 2,145$ cycles for 24-candidate MPC), Vivado 2023.2 post-synthesis resource utilization and timing on `xck26-sfvc784-2LV-c`, and live ARM Cortex-A53 + INA260 power telemetry on the physical Kria KR260 board.
3. **Falsification Criteria:**
   - H6 is **falsified** if teacher distillation fails to improve student representation/prediction fidelity or closed-loop behavior over training from scratch, or if Q8.8 quantization degrades closed-loop navigation decisions relative to FP32.
