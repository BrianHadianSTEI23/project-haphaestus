# Phase 5.3 Unresolved Hypotheses & Open Scientific Questions

**Role:** Implementation Engineer & Experimentalist  
**Target:** Independent interpretation and synthesis by ChatGPT (Senior Independent Scientific Researcher)

---

## 1. Unresolved Hypothesis 1: Identifiability of Functional Selectivity Beyond Activation Magnitude

### Status: UNRESOLVED / UNIDENTIFIABLE UNDER CURRENT REPRESENTATION
In Phase 5.3, we established that:
- In the two settings with complete common support (Layer 1, $k=2$ and $k=4$), candidate units exhibit **zero** advantage over activation-norm-matched random subsets ($\Delta\eta = -0.031$ and $-0.041$, $p_{\text{adj}} = 1.0000$).
- In the remaining 8 settings ($k=8, 16, 32$ in Layer 1; all $k$ in Layer 2), uniform random sampling cannot match the candidate perturbation norm within 5% across all 60 pairs even after 200,000 draws per setting (`INSUFFICIENT_COMMON_SUPPORT`).

### Open Question:
Can functional selectivity be identified using an alternative control architecture?
- Uniform random subset sampling assumes a combinatorial null over the standard basis. However, in neural networks with distributed representations, directions of variation are rarely axis-aligned.
- Would synthetic orthogonal vector rotations (e.g. Haar-distributed random rotations matched in norm) provide common support where coordinate-aligned subsets fail?
- Alternatively, does H-JEPA's grounding head implement an intrinsically non-modular, magnitude-dominated computation where *any* activation perturbation of magnitude $M$ along the active principal subspace produces an equivalent downstream shift?

---

## 2. Unresolved Hypothesis 2: Learned Predictor Value Beyond Analytic SE(2) Kinematics Under Physical Disturbances

### Status: FUTURE UNTESTED HYPOTHESIS
In Phase 5.1 and 5.2, we established that H-JEPA's action-conditioned transition achieves $R^2_{\text{vector}} = 0.4324$ and dead-reckoning error of $0.5490m$ over a 10-step blackout, with statistically significant sensitivity to within-trajectory temporal order ($\Delta = 7.5\text{mm}$, $p = 4.08 \times 10^{-4}$).

### Open Question:
Does the learned action-conditioned world model provide predictive value *beyond* standard analytic Mecanum/SE(2) kinematics when exposed to:
1. Wheel slip and terrain friction changes ($\mu \in [0.1, 0.8]$).
2. Dynamic motor saturation and unmodeled battery voltage drop.
3. Complex multi-axis lateral trajectories outside the training distribution?

In the current rigid-body simulation without unmodeled dynamics, analytic kinematics provides an exact benchmark. The hypothesis that neural world models discover non-holonomic or slip corrections remains to be evaluated in physical hardware or high-fidelity slip simulations.

---

## 3. Unresolved Hypothesis 3: Architectural Support for Identical Object Instance Binding

### Status: STRUCTURALLY CONSTRAINED (E0)
The current H-JEPA task-encoding architecture uses `encode_task_id(target_key, semantic_relation)`, which encodes 6 color-semantic categories (red box, blue cylinder, green cylinder, yellow cylinder, purple box, cyan cylinder) and 8 relations.
- It contains **zero** instance-indexing slots (e.g., "left blue cylinder" vs "right blue cylinder").
- When two visually identical targets appear in the arena, the model's architecture cannot specify an individual instance.

### Open Question:
What minimal architectural mechanism enables instance binding in self-supervised world models?
- Positional prompt tokens (spatial cross-attention)?
- Bounding-box or coordinate query tokens?
- Slot-based object-centric representations (Slot Attention / Object-Centric JEPA)?

---

## 4. Unresolved Hypothesis 4: Geometrical Origin of the 67-Degree Factorial Interaction Divergence

### Status: EMPIRICALLY CONFIRMED / MECHANISTICALLY UNRESOLVED
In Phase 5.1 and Phase 5.2, factorial analysis revealed that the task-context interaction vector $\|\mathbf{I}\| = 0.3521m$ deviates from the physical target-swap vector $\Delta \mathbf{w}$ by $66.9^\circ$ ($95\%$ CI: $[55.3^\circ, 78.9^\circ]$).
- This divergence conclusively falsified both clean dynamic binding (which predicts $0^\circ$ divergence, collinearity) and additive bias (which predicts $\|\mathbf{I}\| = 0$).
- Cross-validated model comparisons showed that bilinear and MLP interaction terms outperform additive models (MSE $0.00439$ vs $0.01985$).

### Open Question:
Why does the interaction vector point along this specific non-collinear angle?
- Does it represent a visual-perspective parallax distortion caused by the robot's off-center camera mounting?
- Or is it an artifact of projecting a higher-dimensional visual manifold onto a 2D Euclidean target plane through ReLU non-linearities?
