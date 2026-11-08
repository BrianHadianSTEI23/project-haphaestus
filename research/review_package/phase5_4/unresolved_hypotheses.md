# Phase 5.4 Unresolved Hypotheses & Open Scientific Questions

**Role:** Implementation Engineer & Experimentalist  
**Target:** Independent interpretation and synthesis by ChatGPT (Senior Independent Scientific Researcher)

---

## 1. Unresolved Hypothesis 1: Geometric Origin of Output Contrast Orthogonality ($90.6^\circ$)

### Status: EMPIRICALLY CONFIRMED / MECHANISTICALLY UNRESOLVED
In Phase 5.4, we established that across 120 evaluation scenes (60 independent pairs):
- The model's task-induced output displacement $\Delta\mathbf{g}_{\text{model}} = \mathbf{g}_{\text{pred}}^B - \mathbf{g}_{\text{pred}}^A$ is **nearly orthogonal** to the physical target displacement $\Delta\mathbf{g}_{\text{gt}} = \mathbf{g}_{\text{gt}}^B - \mathbf{g}_{\text{gt}}^A$ (mean cosine $= +0.0010$, mean angular deviation $= 90.63^\circ$).
- Under physical scene swap where physical target coordinates invert ($\Delta\mathbf{g}_{\text{gt}}^B = -\Delta\mathbf{g}_{\text{gt}}^A$), the model's output contrast fails to reverse, exhibiting a negative reversal cosine of $-0.7119$ ($143.50^\circ$). Because $\cos(\Delta\mathbf{g}_{\text{model}}^A, -\Delta\mathbf{g}_{\text{model}}^B) = -0.7119$, the model contrast retains substantially the *same* orientation across physical scene swap ($\Delta\mathbf{g}_{\text{model}}^A \cdot \Delta\mathbf{g}_{\text{model}}^B = +0.7119$), failing to invert with the physical targets.

### Open Questions:
1. Why does the task-conditioning pathway rotate the output displacement orthogonal to the physical target offset?
   - Does the concatenation of visual latent $z \in \mathbb{R}^{32}$ and task embedding $e \in \mathbb{R}^{16}$ project into a low-dimensional manifold where task tokens modulate heading/yaw rather than lateral Euclidean position?
   - Does the training loss (predicting future relative displacement) penalize forward kinematic progress differently from lateral coordinate accuracy, inducing an anisotropic loss landscape?
2. How does the nonlinear grounding head (Layer 1 + Layer 2 MLPs) transform the 16D categorical task embedding into this skewed spatial representation?

---

## 2. Unresolved Hypothesis 2: Closed-Loop Visual Servoing vs Static Spatial Decoding

### Status: OPEN SCIENTIFIC QUESTION
In Phase 3 and Phase 5.1:
- In closed-loop Track C navigation, H-JEPA achieves **73.3% success** on physically relocated targets and **100.0%** on nominal targets under pure vision.
- In Phase 5.4, single-step static goal prediction exhibits a mean spatial error of **0.4881 m** and an angular contrast deviation of **$90.63^\circ$**.

### Open Questions:
1. How does the agent achieve high closed-loop navigation success despite substantial single-step open-loop spatial inaccuracy?
   - Does closed-loop navigation operate as an iterative visual servoing mechanism, where successive camera observations continuously re-anchor the goal vector as the robot approaches the object?
   - Does forward motion amplify relative angular parallax, allowing the controller to correct the initial $39.9^\circ$ heading error over the trajectory?
2. What is the empirical divergence between single-step offline goal accuracy and multi-step closed-loop path integration?

---

## 3. Unresolved Hypothesis 3: Loss Function Inductive Biases for Veridical Ground-Truth Grounding

### Status: FUTURE UNTESTED HYPOTHESIS
H-JEPA was trained with self-supervised joint-embedding predictive losses and behavioral cloning / spatial prediction objectives.

### Open Questions:
1. What training objective would align internal model contrast with physical Euclidean displacement?
   - Would an explicit contrastive spatial loss ($\|\Delta\mathbf{g}_{\text{model}} - \Delta\mathbf{g}_{\text{gt}}\|^2$) force the grounding head to maintain collinearity ($\cos \to 1$)?
   - Does standard MSE loss on ego-goal regression allow the model to find degenerate local minima that satisfy average distance while rotating contrast axes?
2. Does the lack of explicit depth or 3D geometric inductive bias in the vision backbone (2D ConvNet) fundamentally limit metric body-frame alignment?

---

## 4. Unresolved Hypothesis 4: Causal Intervention in Visual Encoder Space ($z_t$) vs Grounding Head Layers

### Status: UNTESTED EXTENSION
In Phase 5.3 and Phase 5.4:
- Interventions were performed on hidden layers (Layer 1 and Layer 2) of the task-conditioned grounding head.
- These hidden-unit transplants transferred internal model contrast ($\eta_{\text{model}} = 0.413$), but transferred negative physical contrast ($\eta_{\text{gt}} = -0.069$).

### Open Questions:
1. Would transplanting activations directly in the visual encoder latent space $z_t \in \mathbb{R}^{32}$ mediate ground-truth target steering?
2. Or does the visual latent space itself encode scene geometry in an allocentric, non-Euclidean format that only approximately maps to body-frame coordinates?
