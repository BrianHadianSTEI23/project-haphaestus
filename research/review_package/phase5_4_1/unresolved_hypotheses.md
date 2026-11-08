# Phase 5.4.1 Unresolved Hypotheses & Open Scientific Questions

**Role:** Implementation Engineer & Experimentalist  
**Target:** Independent interpretation and synthesis by ChatGPT (Senior Independent Scientific Researcher)  
**Baseline SHA:** `0cafb12240dcd749a7c3e0c8c9612f4ddeca5ea2`  
**Execution SHA:** `d668ce5c0f552eb90a8e26307409b813efc5fc08`  

---

## 1. Unresolved Hypothesis 1: Representation Coding Status of Non-Selective Latent Subspaces

### Status: UNRESOLVED / FORMALLY OPEN FOR INTERPRETATION
- **Prior Premature Promotion**: Phase 5.1/5.2 promoted "The Principle of Diffuse Population Coding (Modularity Falsified)" based on the observation that candidate unit ablations did not produce greater performance drops than random controls.
- **Forensic Correction**: Phase 5.4.1 formally withdrew this principle. In strict norm-matched controls (Phase 5.3), candidate units show no evidence of superiority over norm-matched random controls at complete support ($p_{\text{adj}} = 1.0000$).
- **Open Scientific Questions for ChatGPT**:
  1. Does the empirical equivalence between candidate subsets and norm-matched random controls justify inferring a *diffuse population code*? Or does it merely demonstrate that the candidate identification criterion failed to isolate a causally privileged subspace under standard basis coordinates?
  2. Could the causal mechanism reside in an overcomplete, non-orthogonal basis (e.g., polysemantic superposition), rendering coordinate-aligned axis sampling uninformative?
  3. Under what formal conditions is a neural representation entitled to be classified as a "population code" versus "representationally unidentifiable under standard basis interventions"?

---

## 2. Unresolved Hypothesis 2: Geometric Origin of Output Contrast Retention Across Physical Scene Swap

### Status: EMPIRICALLY CONFIRMED / MECHANISTICALLY UNRESOLVED
- **Empirical Finding**: In Phase 5.4.1, evaluating all 60 independent paired scenes reveals that when physical target objects swap locations ($\Delta\mathbf{g}_{\text{gt}}^B = -\Delta\mathbf{g}_{\text{gt}}^A$), H-JEPA's output contrast vector **fails to reverse**:
  $$\cos(\Delta\mathbf{g}_{\text{model}}^A, -\Delta\mathbf{g}_{\text{model}}^B) = -0.7119 \implies \Delta\mathbf{g}_{\text{model}}^A \cdot \Delta\mathbf{g}_{\text{model}}^B = +0.7119 > 0$$
  The model output contrast retains substantially the *same orientation* across scenes regardless of the physical object swap.
- **Open Scientific Questions for ChatGPT**:
  1. Why does the model retain the same contrast orientation across scenes despite physical object inversion? Does the visual encoder anchor to arena-frame visual landmarks (walls, floor textures) rather than tracking relative object positions?
  2. Does the task-token modulation vector $[\widehat{dx}, \widehat{dy}]$ operate as an allocentric directional prior rather than an object-relative displacement?
  3. What training dynamics lead an architecture trained on relative displacement prediction to develop an invariant output contrast that ignores physical scene inversion?

---

## 3. Unresolved Hypothesis 3: Contrast Projection Dissociation vs Closed-Loop Navigation Parity

### Status: OPEN PARADOX
- **The Empirical Paradox**:
  - In single-step static prediction (Phase 5.4), H-JEPA's output modulation is orthogonal to true physical displacement ($90.63^\circ$, $\cos = +0.0010$), and candidate unit transplantation transfers negative physical contrast ($\eta_{\text{gt}} = -0.0691$) with minimal metric error reduction ($+0.0344$ m).
  - In closed-loop navigation (Track C, Phase 3 v2 & Phase 5.1), H-JEPA achieves **73.3% success** on physically relocated targets (median path length 0.488 m) and **100.0% nominal success** under pure RGB vision without metric coordinates.
- **Open Scientific Questions for ChatGPT**:
  1. How does an agent whose instantaneous task contrast is orthogonal to the true target displacement successfully navigate to relocated targets in closed loop?
  2. Does closed-loop visual servoing operate via continuous Jacobian descent, where recurrent visual feedback corrects for static open-loop miscalibration at every 10 Hz decision step?
  3. Does the agent rely on a coarse heading alignment followed by proximity-triggered visual capture, rendering precise metric goal triangulation unnecessary for navigation success?
