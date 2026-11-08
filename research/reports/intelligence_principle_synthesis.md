# Project Hephaestus: Computational Principles of Embodied Spatial Intelligence

**Author:** Mechanistic Interpretability & Robotics Research Team  
**Repository:** `BrianHadianSTEI23/project-haphaestus`  
**Branch:** `fix/interpretability`  
**Primary Artifacts:**  
- `world_models_fpga/research/benchmarks/results/task_identity_visual_grounding.json`  
- `world_models_fpga/research/benchmarks/results/task_identity_visual_grounding_traces.npz`  
- `world_models_fpga/research/circuits/results/goal_path_interventions.json`  
- `world_models_fpga/research/benchmarks/results/predictive_safety_benchmark.json`  
- `world_models_fpga/research/benchmarks/results/matched_temporal_controls.json`  
- `world_models_fpga/research/circuits/results/rigorous_behavioral_circuits.json`  
- `world_models_fpga/research/benchmarks/results/track_c_visual_grounding.json`  

---

## 1. Executive Scientific Summary

Project Hephaestus investigates the computational mechanisms through which an autonomous embodied agent transforms first-person visual observations and task context into spatial representations, evaluates prospective actions, and navigates under real-time constraints.

Through fine-grained causal interventions, branch isolation, and counterfactual testing, we have derived an empirically verified, falsification-tested account of the computation performed by the Hierarchical Joint-Embedding Predictive Architecture (H-JEPA).

### Central Empirical Findings:
1. **Dynamic Visual Grounding is Active and Saliency-Constrained:**
   When an intended physical target is relocated to a novel arena position while remaining within the robot camera field of view, H-JEPA dynamically tracks and reaches the relocated target with **93.3% success (28/30)** and **0.257m** final goal error ($p = 0.5000$ vs nominal baseline). When the visual scene is held fixed and the task token is swapped, success collapses to **3.3% (1/30)** with **1.524m** error ($p = 3.73 \times 10^{-9}$, Holm-adjusted $p = 1.86 \times 10^{-8}$). H-JEPA does not execute a blind feedforward semantic map from token alone; dynamic visual target extraction requires visual evidence matching the categorical token.
2. **Goal Memory is a Controller Heuristic, Not Learned Neural Memory:**
   Egocentric goal memory (`_ego_goal_memory`), previous latent tracking, and fixed 0.65/0.35 blending are handwritten controller logic in `model_wrappers.py`, containing zero learned recurrent parameters in `HJEPAModel`. Across 7 controlled memory interventions, a pure visual controller (`visual_only`, 0% memory) achieves **96.7% success (29/30)**, matching the default blended controller (96.7%, $p = 1.0000$). Navigation does not depend on controller memory integration.
3. **C1 Subspace Falsification (Unresolved Representation Hypothesis):**
   Targeted ablation of candidate 8D subspace C1 produces a minor 3.3% drop (93.3% $\rightarrow$ 90.0%, $p = 0.9032$ vs random 8D drop of 20.8% $\pm$ 14.8%, $z = -1.177$). Random 8D subspace patching achieves higher offline donor cosine (0.9854) than candidate C1 (0.9779), and random patch rescue (90.9%) is indistinguishable from matched patch rescue (93.3%, $p = 0.6129$). C1 is falsified as a modular circuit. Because random projections achieve parity, this does not confirm a selective diffuse population code; C1 remains an unresolved representation hypothesis.
4. **Prospective Collision Filter Falsification:**
   In closed-loop navigation, intact, shuffled, and disabled collision heads all yield identical **70.0% collision rates** ($p = 1.0000$). Offline collision discrimination exhibits AUROC = 1.000 with clustered bootstrap 95% CI = **[0.5, 1.0]** across 20 episode clusters. Prospective candidate scoring operates as an empirical soft penalty, not a formal Control Barrier Function.
5. **Kinematic Equivalence of Macro Lookahead:**
   Learned $K=5$ macro rollouts beat random candidate selection (46.0% vs 26.0%, Holm-adjusted $p = 0.0117$), but are statistically indistinguishable from 1-step micro lookahead (46.0% vs 50.0%, $p = 0.7266$, adjusted $p = 1.0000$) and match open-loop kinematic straight-line projection identically (46.0% vs 46.0%, $p = 1.0000$).

---

## 2. Systematic Edge-by-Edge Evidence Matrix

| Causal Graph Edge | Proposed Computation | Observational Evidence | Causal Intervention Evidence | Necessity Status | Sufficiency / Rescue Status | Generalization Status | Known Failure Conditions & Counterexamples |
|---|---|---|---|:---:|:---:|:---:|---|
| **$x_t \rightarrow z_t$** | Visual feature compression to 32D latent vector | Latent correlation with body displacement ($R^2 = 0.81$) | Gaussian noise injection drops navigation success | **Confirmed** (visual input required) | **Sufficient** for local steering | Robust to 50% lighting reduction (85.0% success); fails under floor appearance shift (5.0% success, $p = 3.81 \times 10^{-6}$) | Extreme floor appearance shifts collapse latent embedding distribution |
| **$\text{TaskID} \rightarrow e_t$** | Categorical indexing into 16D embedding | Distinct cluster centroids per landmark key | Swapping token under fixed scene collapses navigation toward newly designated object to 3.3% ($p < 10^{-8}$) | **Confirmed** (required for task routing) | **Conditioning Prior** (requires matching visual features) | Generalizes across canonical arena layouts (95.0% success) | Fails when categorical token conflicts with prominent visual landmark |
| **$(z_t, e_t) \rightarrow \hat{g}_t$** | Task-conditioned visual target extraction | Grounding head achieves 0.103m mean step displacement error | Relocated visible target achieves 93.3% success (0.257m error). C1 ablation yields 3.3% drop ($p=0.9032$); random 8D patch achieves $\cos = 0.9854$ vs C1 $\cos = 0.9779$ | **Confirmed** ($z_t$ and $e_t$ jointly mediate goal inference) | **Unresolved Representation** (C1 falsified as modular circuit; random parity precludes selective rescue claim) | Generalizes across relocated target positions when visible in camera frustum | Unconstrained relocation without search policy (targets behind robot) drops success to 30.0% |
| **$(z_t, \hat{g}_t) \rightarrow a_{\text{react}}$** | Reactive holonomic proportional velocity steering | Actions align with goal vector ($\cos = 0.87$) | Clamping $\hat{g}_t$ to zero reduces action magnitude to near-zero | **Confirmed** (goal vector drives steering) | **Sufficient** in open space (100% nominal) | Robust to 50% lighting reduction (85.0% success) | High obstacle clutter without lookahead incurs 40.0% collisions |
| **$(z_t, A_K) \rightarrow \hat{S}_{t+K}$** | Learned action-conditioned forward macro prediction | Open-loop 5-step displacement error = 0.046m ($K=5$) | Macro beats random candidate selection ($p=0.0117$), but matches kinematic straight-line heuristic identically ($p=1.0000$) | **Non-Essential** ($p=1.0000$ vs kinematic heuristic) | **Redundant** with straight-line kinematic projection | Closed-loop macro matches 1-step micro lookahead ($p=0.7266$, adj $p=1.0000$) | Low-slip nominal domains provide no uplift over geometric unicycle extrapolation |
| **$(z_t, A_K) \rightarrow p_{\text{col}}$** | Learned prospective collision risk estimation | Offline AUROC = 1.0000 (cluster CI [0.5, 1.0]), Brier = 0.0003 | Intact head (70% col) matches disabled (70%) and shuffled (70%, $p=1.0000$) | **Falsified Online** (does not reduce collisions) | **Non-Sufficient** (fails prospective barrier function) | Offline metric holds across held-out split | Control delay and soft penalty cost weighting fail to prevent closed-loop collisions |

---

## 3. Detailed Mechanistic Hypotheses Resolution

### Hypothesis H1 — Visual Goal Vector Representation & Circuit C1
> *Hypothesis:* A localized 8D linear subspace (C1) in the 32D latent visual space selectively and causally mediates task-conditioned goal-vector extraction.

- **Empirical Results:**
  - Clean baseline navigation: 93.3% success (28/30).
  - C1 targeted ablation: 90.0% success (27/30, drop = 3.3%).
  - Random 8D subspace ablation drop: 20.8% $\pm$ 14.8% ($p = 0.9032$, $z = -1.177$).
  - Retain-only C1 sufficiency: 16.7% success.
  - Matched donor patch: 93.3% success vs Mismatched donor patch: 83.3% success.
  - Random 8D patch rescue: **90.9% success** ($p = 0.6129$ vs matched patch).
  - Offline donor cosine: C1 = 0.9779 $\pm$ 0.0356 (angular error 0.1430 rad) vs Random 8D = **0.9854 $\pm$ 0.0078** (Random is higher!).
  - Perturbation diagnostics: C1 variance fraction removed = 23.51% (Mahalanobis dist 94.991, norm ratio 0.875) matching random 8D ($23.99\% \pm 3.38\%$, Mahalanobis dist $94.322 \pm 25.492$, norm ratio 0.871).
- **Scientific Resolution:** **FALSIFIED AS A MODULAR CIRCUIT; UNRESOLVED REPRESENTATION HYPOTHESIS.** C1 is non-selective for visual goal grounding. Because random 8D subspace patching achieves higher offline donor cosine and equivalent behavioral rescue, donor activation transfer cannot be claimed as selective circuit rescue. C1 remains an unresolved representation hypothesis beyond the definitive falsification of this localized 8D linear subspace.

---

### Hypothesis H2 — Dynamic Visual Grounding vs Task-Conditioned Canonical Expectation
> *Hypothesis:* Does H-JEPA dynamically ground the current physical location of a target from RGB observations, or does it predominantly retrieve a task-conditioned canonical spatial expectation?

- **Empirical Results (Phase 3 Factorial Experiment, $N=30$ per condition):**
  1. **Canonical Baseline (Cond 1):** 30/30 (100.0%) success, 0.289m final error, visual goal angular error 0.244 rad.
  2. **Relocated Target with Verified Visibility (Cond 2):** 28/30 (**93.3% success**), **0.257m final error**, only 2/30 closer to canonical ($p = 0.5000$, Holm-adjusted $p = 1.0000$).
  3. **Distractor-Only Relocation (Cond 3):** 30/30 (100.0%) success, 0.289m final error ($p = 1.0000$).
  4. **Same RGB, Changed Task Token (Cond 4):** 1/30 (**3.3% success**), **1.524m final error**, 27/30 closer to canonical ($p = 3.73 \times 10^{-9}$, Holm-adjusted $p = 1.86 \times 10^{-8}$). Visual goal angular error jumps to 1.844 rad.
  5. **Target Appearance Reassigned (Cond 5):** 26/30 (86.7%) success, 0.336m error ($p = 0.1250$, Holm-adjusted $p = 0.3750$).
  6. **Visual / Token Conflict (Cond 6):** 22/30 (73.3%) success, 0.384m error ($p = 0.0078$, Holm-adjusted $p = 0.03125$).
- **Scientific Resolution:** **CONFIRMED DYNAMIC VISUAL GROUNDING (BOUNDED BY VISUAL SALIENCY).**
  - When the target is physically relocated within the camera field of view, the agent dynamically grounds the object and navigates to its actual location (93.3% success), refuting the claim that the agent is bound to canonical training coordinates.
  - The agent does not execute a blind feedforward semantic map from token alone: when the token is swapped under a fixed RGB scene, navigation fails (3.3% success), proving that goal generation requires visual saliency evidence matching the categorical token.

---

### Hypothesis H3 — Controller Goal Memory vs Learned Recurrent State
> *Hypothesis:* Does the agent's spatial navigation depend causally on temporal goal memory integration, and does memory represent a learned recurrent neural mechanism?

- **Empirical Results (Phase 3 Memory Interventions, $N=30$ per condition):**
  1. `cond_1_full_blended` (default controller, 0.65 visual + 0.35 memory): 29/30 (96.7%) success, 0.301m error.
  2. `cond_2_visual_only` (pure visual grounding, 0% memory): 29/30 (**96.7% success**), **0.304m error** ($p = 1.0000$, $\Delta = 0.0\%$).
  3. `cond_3_memory_reset_step` (memory zeroed every step): 28/30 (93.3%) success, 0.320m error ($p = 1.0000$).
  4. `cond_4_frozen_memory` (memory frozen at $t=0$ state): 29/30 (96.7%) success, 0.293m error ($p = 1.0000$).
  5. `cond_5_corrupted_memory` (Gaussian noise injected into memory): 30/30 (100.0%) success, 0.298m error ($p = 1.0000$).
  6. `cond_6_matched_donor_memory`: 30/30 (100.0%) success, 0.292m error ($p = 1.0000$).
  7. `cond_7_mismatched_donor_memory`: 30/30 (100.0%) success, 0.290m error ($p = 1.0000$).
- **Scientific Resolution:** **CONFIRMED CONTROLLER HEURISTIC (NON-ESSENTIAL).**
  - Goal memory is an ephemeral Python controller buffer in `model_wrappers.py`, not a learned recurrent neural parameter in `HJEPAModel`.
  - Closed-loop navigation does not depend on memory integration; pure visual grounding achieves exact parity with default blending.

---

### Hypothesis H4 — Predictive Dynamics versus Geometric Kinematics
> *Hypothesis:* Learned action-conditioned prediction may provide additional value only when real physical dynamics make simple geometric projection inadequate.

- **Empirical Results ($N=50$ matched episodes):**
  - Reactive Baseline ($K=0$): 36.0% success, 40.0% collisions.
  - Direct Macro $K=5$ (1-pass): 46.0% success, 48.0% collisions.
  - Recursive Micro $K=5$ (5-pass): 50.0% success, 36.0% collisions ($p = 0.7266$ vs Macro).
  - Straight-Line Kinematic Heuristic (non-predictive): **46.0% success, 48.0% collisions** ($p = 1.0000$ vs Macro).
  - Shuffled Action–Outcome Binding: 34.0% success (unadjusted $p = 0.03125$, Holm-adjusted $p = 0.1562$).
  - Random Candidate Selection: 26.0% success (Holm-adjusted $p = 0.0117$ vs Macro).
  - Latency speedup: Direct Macro is **$4.30\times$ faster** than Recursive Micro (0.371 ms vs 1.594 ms).
- **Scientific Resolution:** **CONFIRMED KINEMATIC EQUIVALENCE IN NOMINAL DOMAINS.** Learned macro world modeling matches open-loop kinematic unicycle projection identically ($p = 1.0000$). Uplift over random selection (+20.0%, $p = 0.0117$) reflects candidate pruning along the goal heading, not deep prospective dynamic simulation.

---

### Hypothesis H5 — Prospective Collision Safety & The Closed-Loop Reality
> *Hypothesis:* A prospective collision prediction head trained offline provides effective closed-loop obstacle avoidance.

- **Empirical Results ($N=40$ collision-critical episodes):**
  - Offline Metrics ($K=5$ horizon): AUROC = 1.0000, Brier = 0.0003, ECE = 0.0041, FNR (@0.5) = 0.0%.
  - Clustered Bootstrap AUROC 95% CI: **[0.5, 1.0]** across 20 episode clusters.
  - Closed-Loop Intact Collision Head: **70.0% collisions** (27.5% success, 0.666m error).
  - Closed-Loop Shuffled Collision Binding: **70.0% collisions** ($p = 1.0000$).
  - Closed-Loop Disabled Collision Head: **70.0% collisions** ($p = 1.0000$).
  - Closed-Loop Random Predictor: 57.5% collisions ($p = 0.0625$).
  - Closed-Loop Oracle Physics Simulation: **7.5% collisions** ($p = 5.96 \times 10^{-8}$).
- **Scientific Resolution:** **FALSIFIED AS A CLOSED-LOOP SAFETY MECHANISM.** Near-perfect offline discriminative metrics do not translate to online obstacle avoidance. Discrete candidate penalty scoring degenerates to the unconstrained baseline (70.0% collisions) due to control delay and lack of reflexive distance margins.

---

### Hypothesis H6 — Coordinate Symmetry versus Representational Equivariance
> *Hypothesis:* The system may be invariant to global coordinate-frame changes without encoding a nontrivial equivariant spatial transformation.

- **Empirical Results ($N=20$ episodes per angle):**
  - Rigid Global Arena Rotation (+90°, 180°, -90°): 20/20 (100.0%) success retention across all angles.
  - Global Rigid Latent Cosine Similarity: **0.9745**.
  - Egocentric Camera Yaw Shift (+90°): Latent Cosine Similarity collapses to **0.3450**.
  - Linear Operator Fit $T_g$: $R^2 < 0.20$ on held-out camera rotations.
- **Scientific Resolution:** **CONFIRMED (GLOBAL INVARIANCE $\neq$ REPRESENTATIONAL EQUIVARIANCE).** Rigid global arena rotation rotates the robot and environment together, preserving the egocentric camera perspective ($x_{\text{rot}} \approx x_{\text{nom}}$). High latent similarity reflects viewpoint preservation, not learned steerable group-equivariant representations ($f(g \cdot x) \neq T_g f(x)$).

---

## 4. Reconciled Computational Principles of Embodied Spatial Intelligence

### Principle 1: Saliency-Gated Visual Grounding
Task-conditioned visual goal grounding requires matching physical visual evidence. When the target object is visible in the camera field of view, the visual encoder and grounding head dynamically compute egocentric displacement (93.3% success). Categorical task tokens cannot redirect navigation in the absence of corresponding visual saliency features (3.3% success under token swap, $p < 10^{-8}$).

### Principle 2: Controller Memory Decoupling
Temporal goal integration implemented as a controller-level vector differencing heuristic contributes zero marginal spatial information over instantaneous reactive visual grounding in observable environments (pure visual controller achieves identical 96.7% success, $p = 1.0000$).

### Principle 3: Kinematic Equivalence of Macro Lookahead
In rigid, low-slip holonomic domains, action-conditioned latent forward prediction is functionally equivalent to deterministic kinematic unicycle projection ($p = 1.0000$). Learned predictive lookahead provides no marginal utility over geometric extrapolation unless non-linear contact dynamics or wheel slip are present.

### Principle 4: The Closed-Loop Prospective Safety Gap
Offline collision prediction fidelity (AUROC = 1.000, cluster CI [0.5, 1.0]) does not prevent online collisions when integrated as a soft candidate penalty (intact, shuffled, and disabled heads all yield 70.0% collisions). Reliable collision avoidance requires instantaneous closed-loop physical boundaries or control barrier functions rather than feedforward candidate scoring.

### Principle 5: Viewpoint Invariance versus Latent Equivariance
Behavioral retention under global coordinate transformations ($SE(2)$ rigid rotation) reflects egocentric viewpoint preservation ($\cos = 0.975$) rather than mathematical group equivariance. Egocentric camera rotations destroy latent alignment ($\cos = 0.345$), indicating that standard CNN backbones lack steerable equivariant filter banks.

---

## 5. Architectural Blueprint for Embodied Spatial Intelligence

Based on these empirical discoveries and falsifications, future embodied world-model architectures should implement:

1. **Active Visual Search Policy Decoupled from Tracking:** Reactive visual grounding succeeds reliably when objects are visible (93.3%), but fails when objects fall outside the field of view. Target acquisition requires an explicit exploration/search state machine.
2. **Deterministic Kinematic Proposer with Slip-Specialized Neural Residuals:** Implement trajectory candidate generation using deterministic kinematic unicycle models on FPGA/edge silicon (0.18 ms latency), reserving learned neural world models strictly for predicting non-linear wheel slip, terrain compliance, and dynamic obstacles.
3. **Hard Reactive Control Barrier Layer:** Replace soft prospective candidate penalties with a dedicated, high-frequency (100 Hz) Control Barrier Function (CBF) operating directly on range sensor boundaries.
4. **Group-Equivariant Visual Backbones:** Integrate $SE(2)$-steerable convolutional kernels ($G$-CNNs) to ensure mathematical equivariance under camera yaw and perspective changes.
