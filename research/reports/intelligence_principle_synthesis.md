# Project Hephaestus: Computational Principles of Embodied Spatial Intelligence

**Author:** Mechanistic Interpretability & Robotics Research Team  
**Repository:** `BrianHadianSTEI23/project-haphaestus`  
**Branch:** `fix/interpretability`  
**Primary Artifacts:**  
- `world_models_fpga/research/benchmarks/results/task_identity_visual_grounding_v2.json`  
- `world_models_fpga/research/benchmarks/results/task_identity_visual_grounding_v2_traces.npz`  
- `world_models_fpga/research/circuits/results/goal_memory_stress_tests_v2.json`  
- `world_models_fpga/research/circuits/results/goal_memory_stress_tests_v2_traces.npz`  
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
1. **Task-Conditioned Spatial Decoding is a Mixed Nonlinear Interaction, Not Visual Object Binding (Phase 5.1 v5):**
   Evaluating both Scene A and Scene B at the independent scene-pair level ($N=50$ pairs, 100 scenes, verified 5-step simulator mutation protocol) with a pair-clustered permutation test (100,000 resamples) reveals that observed dual-target discrimination is **16.0%**, demonstrating that above-null target binding was not shown relative to the permutation null distribution (**19.5% [95% CI: 16.0%, 23.0%]**, $p_{\text{one\_sided}} = 0.9890$, $p_{\text{two\_sided}} = 0.0571$). Exactly **0/50 pairs** achieved dual discrimination across both scenes. The historical 5.0% same-category instance claim is invalidated and classified as `ARCHITECTURALLY_UNSUPPORTED` because `encode_task_id()` lacks vocabulary slots for identical instance indexing. The factorial interaction vector $I = (g_{bb} - g_{ba}) - (g_{ab} - g_{aa})$ has substantial magnitude ($\|I\| = 0.3521$m, 95% CI: $[0.2921, 0.4161]$m), ruling out an additive spatial prior (Mechanism 1). Its cosine alignment with target swap $\Delta P$ is positive ($\overline{\cos} = +0.3140$, 95% CI: $[0.1508, 0.4693]$, sign-flip $p_{\text{pos}} = 0.00016, p_{\text{two}} = 0.00044$), diverging by $66.9^\circ$ (95% CI: $[55.3^\circ, 78.9^\circ]$) from physical target swap. Stratified grouped 5-fold CV by `pair_id` confirms H-JEPA outputs are best explained by a **Bilinear Interaction Model ($MSE = 0.00544$)** and **Nonlinear MLP ($MSE = 0.00439$)**, outperforming Additive ($MSE = 0.01985$) and Task-Only ($MSE = 0.03723$). These surrogate fits indicate context-dependent input-output relationships rather than a literal bilinear circuit.
2. **Ego-Motion Latent Differences Collapse to Static Bias; Action Predictor Implements Dead Reckoning (Phase 5.1 & Phase 5.2):**
   Offline calibration with corrected $t \to t+1$ time alignment ($N=1,450$ transitions) proves learned sensor displacement translation MAE on clean frames is **0.0274m** ($R^2 = 0.0143$ on magnitude regression, slope 0.2112; 2D vector coordinate $R^2 = -1.9195$), providing minimal predictive value over the zero displacement baseline (**0.0287m**). Under blackout, sensor latent delta $\Delta s = 0$, outputting constant static network bias. In contrast, the architecture's **action-conditioned forward predictor** ($\delta_s(s_t, a_t)$) achieves translation MAE **0.0142m** on clean transitions ($R^2 = 0.2518$ on scalar magnitude regression, slope 0.4538; 2D vector coordinate $R^2 = 0.4324$ with $dx$ $R^2 = 0.4167$, $dy$ $R^2 = 0.4579$, and $d\theta$ $R^2 = 0.6802$), cutting translation error by 50.5%. Under same-trajectory factorial replay ($N=50$), fixed 0.65/0.35 blending erases all motion sources into static black noise ($\approx 0.69$m error) due to $(0.35)^{10} \approx 2.76 \times 10^{-5}$ memory decay; fixed-blending vs hold tracking differences (0.0549m, 95% CI: [-0.0416, 0.1528]m, Wilcoxon $p = 0.1719$) are not statistically significant. When evaluated under hold/freeze memory policy, action-conditioned prediction achieves **0.5490m**, statistically equivalent to oracle odometry (**0.5392m**, diff +0.0098m, TOST $p = 6.85 \times 10^{-7}$ at 0.05m margin) under tested simulation conditions, significantly beating zero hold (**0.6499m**, $p = 8.09 \times 10^{-9}$), zero action (**0.6165m**, $p = 8.79 \times 10^{-9}$), and cross-shuffled action (**0.5704m**, $p = 2.16 \times 10^{-3}$). In Phase 5.2, within-trajectory time-shuffled action control (50 perms/trajectory) achieves 0.5565m, yielding paired difference $\Delta = -0.0075$ m ($-7.5$ mm, 95% CI: [-0.0115, -0.0036] m, Wilcoxon $p = 4.08 \times 10^{-4}$, sign-flip $p = 4.30 \times 10^{-4}$). This demonstrates statistically significant sensitivity to temporal action order while revealing that the net displacement integral dominates dead-reckoning under small commutative rotations in $\mathrm{SE}(2)$. The memory accumulator is handwritten controller logic, not learned recurrent memory.
3. **Internal Mechanistic Identification, Activation-Norm Matched Controls, & Ground-Truth Dissociation (Phase 5.1–5.5):**
   Capturing full activation vectors for all 9 stages on evaluation scene pairs reveals low effective dimensionality across stages (visual latent $z$ PR: 4.13/32; task embedding $e$ PR: 3.92/16; input concat PR: 7.03/48; Layer 1 post-ReLU PR: 8.14/64; Layer 2 post-ReLU PR: 7.14/64). On held-out scenes, candidate top-k units significantly outperform unconstrained dimension-matched random subsets (Layer 1 $k=8$: candidate 34.0% vs random 12.0%, $\Delta = +0.220, p = 1.23 \times 10^{-4}$; Layer 2 $k=8$: candidate 34.4% vs random 12.7%, $\Delta = +0.216, p = 3.41 \times 10^{-8}$). However, an independent audit of Phase 5.2 revealed that when $< 100$ valid matches existed within 5% tolerance, controls outside tolerance were silently retained, invalidating the historical "Selective Subspace Confirmed" claim. Phase 5.3 strictly enforced tolerance gating ($\le 5\%$ error mandatory; no fallback to invalid controls), adaptive sampling up to 200,000 draws, and matched necessity controls across 60 independent pairs (120 scenes, 447,800 interventions: 2,400 candidate, 240,000 dimension-matched, 205,400 norm-matched). Results show that in the two settings with 100% complete common support (Layer 1 $k=2, 4$), candidate units exhibit no evidence of superiority over strictly norm-matched random controls in sufficiency (k=2: 0.147 vs 0.179, $\Delta = -0.031, p_{\text{adj}} = 1.0000$; k=4: 0.295 vs 0.337, $\Delta = -0.041, p_{\text{adj}} = 1.0000$) and lower necessity drops (k=2: 0.020 vs 0.061; k=4: 0.105 vs 0.128); confidence intervals do not establish equivalence. In the remaining 8 out of 10 settings, common support is incomplete (coverage 51.7% to 93.3%) because candidate perturbation norms occupy an extreme combinatorial tail that random subsets cannot match within 5% tolerance even after 200,000 draws. Under strict common support, candidate advantage is unidentifiable from an activation-magnitude effect. Furthermore, Phase 5.4 ground-truth reanalysis established that static internal model contrast ($\Delta\mathbf{g}_{\text{model}}$) is dissociated from physical target steering in `hjepa_best.pt` on `PAIR_000`–`PAIR_059`: task-induced output displacement is nearly orthogonal ($90.63^\circ$, cosine $+0.0010$) to $\Delta\mathbf{g}_{\text{gt}}$, and reverses only partially under scene swap ($\cos = -0.7119$). In **Phase 5.4.2**, auditing single-target estimands across all 447,800 rows (`2,400` deterministic settings, `max_abs_diff_across_draws = 0.0`) invalidated Phase 5.4's confounded cross-target error difference (`+0.0344 m`, which carries a `+0.0037 m` no-op artifact), yielding null candidate selectivity at complete support (Layer 1 `k=2`: `donor_target_gain_m` `+0.0008 m` vs `+0.0125 m`, Holm $p = 0.4675$, dimensionless `eta_gt` `-0.0206` vs `-0.0386`, Holm $p = 1.0000$; `k=4`: `+0.0234 m` vs `+0.0302 m`, Holm $p = 1.0000$, `eta_gt` `-0.0425` vs `-0.0384`, Holm $p = 1.0000$; equivalence not established). In **Phase 5.5** closed-loop testing ($N=60$, Oracle `60/60`), `C1` succeeds `51/60 (85.0%)` despite step-0 `eta_gt = -0.551556`; continuous visual updates are causally necessary (`C2` drops to `37/60 = 61.7%`, Holm $p = 2.5940 \times 10^{-3}$) to sustain forward visual servoing across `0.317 m < 0.35 m` closely spaced targets, while oracle waypoints (`C6`) reach `60/60 (100.0%, +15.0%, Holm p = 1.5625e-02)`.
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

- **Empirical Results (Phase 3 v2 Controlled Grounding, $N=30$ per condition):**
  1. **Canonical Baseline (Cond 1):** 30/30 (100.0%) success, 0.328m final error, initial distance 0.587m, path length 0.270m.
  2. **Relocated Target with Guaranteed Minimum Distance (Cond 2):** 22/30 (**73.3% success**), 0.652m final error (median 0.343m), initial distance 0.770m (min 0.666m), median path length **0.488m** (mean 0.706m). Eliminates initial distance confound. Degrades significantly vs canonical ($p = 0.0078$, Holm-adjusted $p = 0.03125$).
  3. **Distractor-Only Relocation (Cond 3):** 30/30 (100.0%) success, 0.328m final error ($p = 1.0000$).
  4. **Dual-Visible Identical RGB, Token A (Cond 4a):** 20/30 (66.7%) success, 0.497m error, median path 0.555m.
  5. **Dual-Visible Identical RGB, Token B (Cond 4b):** 21/30 (70.0%) success, 0.483m error, median path 0.504m. Changing only the task token rotates the inferred visual goal vector by **$0.869$ rad ($49.8^\circ$)** on the identical visual latent. Strict dual discrimination is achieved in **20.0% (6/30)** of episodes.
  6. **Visual / Token Conflict Factorial (Cond 5):** 26/30 (86.7%) success, 0.371m error ($p = 0.1250$, Holm-adjusted $p = 0.3750$).
  7. **Target Appearance Reassigned (Cond 6):** 29/30 (96.7%) success, 0.341m error ($p = 1.0000$).
- **Scientific Resolution:** **CONFIRMED DYNAMIC VISUAL GROUNDING (LOCOMOTION VERIFIED, SALIENCY-CONSTRAINED).**
  - When the target is physically relocated within the camera field of view with guaranteed initial distance ($\ge 0.60$m), the agent performs genuine locomotion (median path 0.488m) to reach the relocated target (73.3% success). The prior 93.3% figure in v1 was inflated by 21/30 episodes starting within the goal threshold.
  - In paired scenes where candidate objects A and B are simultaneously visible in identical RGB frames, changing only the task token rotates the inferred goal vector by $49.8^\circ$, proving task tokens steer visual goal inference, but compete with spatial priors (20% dual discrimination rate).

---

### Hypothesis H3 — Controller Goal Memory vs Learned Recurrent State Under Occlusion
> *Hypothesis:* Does the agent's spatial navigation depend causally on temporal goal memory integration, and does memory represent a learned recurrent neural mechanism?

- **Empirical Results (Phase 3 v2 Memory Stress Tests Under 10-Step Sensory Blackout, $N=30$ matched episodes):**
  1. `cond_1_full_visual_and_memory` (default controller, 0.65 visual + 0.35 memory): 24/30 (80.0%) success, occlusion window error 0.648m.
  2. `cond_2_visual_only` (pure visual grounding, 0% memory): 25/30 (**83.3% success**), occlusion window error **0.745m**. During the 10-step sensory blackout (steps 5..15), **visual-only control incurs $+0.097$m higher tracking error** (95% CI $[0.062, 0.134]$m, Wilcoxon $p = 3.79 \times 10^{-6}$) vs full memory. After vision is restored, final navigation success recovers to 83.3% (non-inferior within $\delta=0.10$ margin to full memory 80.0%).
  3. `cond_3_memory_reset_step`: 25/30 (83.3%) success, occlusion error 0.567m.
  4. `cond_4_frozen_memory`: 30/30 (100.0%) success, occlusion error 0.490m.
  5. `cond_5_corrupted_memory` (+0.50 Gaussian noise): 28/30 (93.3%) success, occlusion error 0.685m.
  6. `cond_6_zeroed_ego_motion`: 24/30 (80.0%) success, occlusion error 0.629m.
  7. `cond_7_perturbed_ego_motion`: 25/30 (83.3%) success, occlusion error 0.643m.
  8. `cond_8_matched_donor_memory`: 29/30 (96.7%) success, occlusion error 0.501m.
  9. `cond_9_mismatched_donor_memory`: 27/30 (90.0%) success, occlusion error 0.438m.
- **Scientific Resolution:** **CONTROLLER HEURISTIC CAUSALLY MAINTAINS OCCLUSION TRACKING.**
  - Goal memory is an ephemeral Python controller buffer in `model_wrappers.py` utilizing learned 2D `micro_ego_head` displacements; zero learned recurrent state in `HJEPAModel`.
  - While visual-only achieves parity under continuous vision, controller memory provides a statistically significant tracking benefit during temporary sensory blackout ($+0.097$m error increase when omitted, $p = 3.79 \times 10^{-6}$). Final navigation recovers once vision is restored.

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
- **Scientific Resolution:** **SAMPLE PARITY WITH KINEMATIC PROJECTION IN NOMINAL DOMAINS (EQUIVALENCE NOT ESTABLISHED SOLELY FROM p=1.0).** Learned macro world modeling matches open-loop kinematic unicycle projection identically in sample outcomes ($n_{10}=0, n_{01}=0$, $46.0\%$ vs $46.0\%$; $p = 1.0000$ does not establish statistical equivalence). Uplift over random selection (+20.0%, $p = 0.0117$) reflects candidate pruning along the goal heading, not deep prospective dynamic simulation.

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

### Principle 1: Saliency-Gated Visual Grounding & Locomotion Calibration
Task-conditioned visual goal grounding requires matching physical visual evidence. When the target object is visible in the camera field of view and initial distance is properly controlled ($\ge 0.60$m), the visual encoder and grounding head dynamically compute egocentric displacement driving genuine locomotion (73.3% success, median path 0.488m), showing calibrated degradation vs canonical baseline ($p = 0.03125$). Categorical task tokens interact causally with visual saliency to rotate goal vectors ($49.8^\circ$ shift under identical RGB frames), but cannot steer navigation when visual evidence is completely absent (3.3% success under token swap, $p < 10^{-8}$).

### Principle 2: Controller Memory Decoupling & Occlusion Buffer
Temporal goal integration implemented as a controller-level vector accumulator paired with learned `micro_ego_head` displacements contributes zero marginal spatial information over instantaneous reactive visual grounding in continuously observable environments (pure visual controller achieves identical 96.7% success, $p = 1.0000$). However, during temporary visual blackout, controller memory causally preserves egocentric tracking (+0.097m error penalty if omitted, $p = 3.79 \times 10^{-6}$), acting as an ephemeral dead-reckoning bridge until visual observations resume.

### Principle 3: Kinematic Equivalence of Macro Lookahead
In rigid, low-slip holonomic domains, action-conditioned latent forward prediction achieves identical sample navigation success to deterministic kinematic unicycle projection ($46.0\%$ vs $46.0\%$, $n_{10}=0, n_{01}=0$; statistical equivalence is not claimed from $p = 1.0000$). Learned predictive lookahead provides no observed marginal uplift over geometric extrapolation unless non-linear contact dynamics or wheel slip are present.

### Principle 4: The Closed-Loop Prospective Safety Gap
Offline collision prediction fidelity (AUROC = 1.000, cluster CI [0.5, 1.0]) does not prevent online collisions when integrated as a soft candidate penalty (intact, shuffled, and disabled heads all yield 70.0% collisions). Reliable collision avoidance requires instantaneous closed-loop physical boundaries or control barrier functions rather than feedforward candidate scoring.

### Principle 5: Viewpoint Invariance versus Latent Equivariance
Behavioral retention under global coordinate transformations ($SE(2)$ rigid rotation) reflects egocentric viewpoint preservation ($\cos = 0.975$) rather than mathematical group equivariance. Egocentric camera rotations destroy latent alignment ($\cos = 0.345$), indicating that standard CNN backbones lack steerable equivariant filter banks.

### Principle 6: Dissociation of Static Internal Contrast from Ground-Truth Waypoint Grounding & Geometry-Separated Closed-Loop Decomposition (Phases 5.4.2, 5.4.2.1, 5.5, 5.5.1)
In the tested H-JEPA checkpoint (`hjepa_best.pt`) and paired-scene benchmark (`PAIR_000`–`PAIR_059`), static task-conditioned output modulation is dissociated from physical target displacement ($\Delta\mathbf{g}_{\text{gt}}$, orthogonal at $90.63^\circ$, cosine $+0.0010$). Under **Phase 5.4.2** corrected single-target estimands and **Phase 5.4.2.1** exact composite key-set reconciliation (`Expected = 447,800`, `Observed = 447,800`, `observed_unique_keys = 447,800`, `Missing = 0, Unexpected = 0, Duplicates = 0, EXACT_SET_EQUALITY_VERIFIED`; `2,400` deterministic settings, `max_abs_diff_across_draws = 0.0`), removing Phase 5.4's confounded cross-target error difference (`+0.0344 m`, which carries a `+0.0037 m` no-op artifact) shows no evidence of candidate superiority over strictly norm-matched controls at complete support (Layer 1 `k=2` `donor_target_gain_m` `+0.0008 m` vs `+0.0125 m`, Holm $p = 0.4675$, dimensionless `eta_gt` `-0.0206` vs `-0.0386`, Holm $p = 1.0000$; `k=4` `+0.0234 m` vs `+0.0302 m`, Holm $p = 1.0000$, dimensionless `eta_gt` `-0.0425` vs `-0.0384`, Holm $p = 1.0000$; equivalence not established). **Phase 5.5** closed-loop evaluation ($N=60$, Oracle `60/60`) showed `C1 = 51/60 (85.0%)` because target–distractor waypoints averaged `0.317 m < 0.35 m` reach radius, enabling non-selective forward visual servoing (`r(lat, gt_y) = -0.340`), with Part A2 subsequently separating `C6_track_b_oracle_waypoint_upper_bound` from `C7_track_c_planner_goal_override` and recording unexecuted fields as `NaN`/`None`. Under **Phase 5.5.1** on an independent geometry-separated cohort (`N = 60` scenes, `755` evaluated / `695` rejected, `min_waypoint_separation_m = 0.900183 m`, `mean = 0.942498 m`, `Oracle A = 60/60`, `Oracle B = 60/60`, `840` episodes):
- **`H1` (`F1` task selectivity):** Assigned success `32/120 (26.7%)` vs unassigned reached `24/120 (20.0%)`, pair diff `+0.0667 [-0.0417, +0.1667]`, `Holm5 p = 0.4337`; `dual_first_correct = 3/60 (5.0%)` vs `same_object_collapse = 13/60 (21.7%)` (`binom p = 0.0213`), `step0_eta_vis = -0.0105` (`p = 0.9413`), accompanied by right-side (`45.0%` vs left `8.3%`) and category (`blue_cylinder = 60.0%` vs `red_box / yellow_cylinder = 10.0%`) skew.
- **`H2` (Repeated visual feedback):** `0.5*(F1+F2) - 0.5*(F3+F4) = +0.1958 [+0.1292, +0.2667]` (`27.1%` vs `7.5%`), **`Holm5 p = 1.2891e-05`** (`F1-F3 = +0.1917`, `F2-F4 = +0.2000`).
- **`H3` (Controller ego-memory):** `0.5*(F1+F3) - 0.5*(F2+F4) = -0.0042 [-0.0167, +0.0083]`, `Holm5 p = 0.5637` (interaction `-0.0083, p = 0.5637`; `F1-F2 = -0.0083, p = 0.3173`; `F3-F4 = 0.0, p = 1.0`; equivalence not established and visual blackout excluded).
- **`H4` (Object-specific masking):** `M1_target_masked (11.7%) - M2_distractor_masked (21.7%) = -0.1000 [-0.1583, -0.0333]`, **`Holm5 p = 0.0081`** (`M1 - F1 = -0.1500, within_Holm p = 7.1569e-04`; `M2 - F1 = -0.0500, within_Holm p = 0.1336`).
- **`H5` (Track C planner-goal override):** `O1 = 111/120 (92.5%, dual_first_correct = 51/60 = 85.0%, left = 86.7%, right = 98.3%, categories 85.0%–95.0%)` vs `F1 (26.7%)`, diff `+0.6583 [+0.5667, +0.7500]`, **`Holm5 p = 6.9065e-10`**. Raw findings remain open for independent interpretation rather than elevated into a new speculative principle.
- **Phase 5.5.1.1 (Symmetric H1 Estimand Audit & Side/Category Confound Localization, `phase5_5_1_1/`):** Complete `60`-scene, `420`-pair, `840`-episode replay (`32,557` step rows, `39` shards, `0` mismatches, `60/60` bit-for-bit initial-frame SHA-256 matches, `all_integrity_checks_passed = true`) reconciles censoring and symmetry:
  - **Symmetric `F1` H1 Families:** In `F1`, `32` assigned approaches terminated early while among unassigned approaches `14` completed without collision and `10` collided during full `45`-step continuation. Family A (`first_waypoint_reached`: `32/120 = 26.7%` vs `24/120 = 20.0%`) yields `diff = +0.0667 [-0.0333, +0.1667], Wilcoxon p = 0.2169`; Family B (`first_waypoint_reached_no_collision`: `32/120 = 26.7%` vs `10/120 = 8.3%`) yields `diff = +0.1833 [+0.0917, +0.2750], Wilcoxon p = 3.6702e-04`; complete-case `N = 60` Family C records `both_first_assigned = 3/60 (5.0%)`, `same_physical_object_collapse = 13/60 (21.7%)` (`3` onto `obj_1`, `10` onto `obj_2`), `both_first_unassigned = 2/60 (3.3%)`, `at_least_one_neither = 42/60 (70.0%)` (`3x3` Stuart–Maxwell `chi2 = 2.2513, p = 0.3244`; conditional `3 vs 13` binomial `p = 0.02127`, exploratory); Family D reports `step0_eta_vis = -0.0105, p = 0.9413`.
  - **Left/Right Asymmetry & Confound Decomposition:** `F1` exhibits a right-side skew (`Left 5/60 = 8.3%` vs `Right 27/60 = 45.0%`, paired `[3, 2, 24, 31]`, RD `+0.3667 [+0.2167, +0.5000]`, exact McNemar `p = 1.0490e-05`), which persists under `M2` (`Left 2/60 = 3.3%` vs `Right 24/60 = 40.0%`, `p = 1.0490e-05`) and after category stratification (`Mantel–Haenszel OR = 8.17`, `CMH chi2 = 21.30, p = 3.9270e-06`; logistic adjusted `OR = 12.90`), while initial image areas (`448.1 vs 450.1 px, p = 0.7231`) and horizontal centroids (`u_L + u_R - 63 = +0.244 px`) are symmetric.
- **Phase 5.6 (Stage-Wise Task-to-Space Mechanistic Localization, `phase5_6/`):** Executed strictly pre-registered Stage B2–B6 checks (`task_conditioned_stagewise_binding_v5_6_*`) on `hjepa_best.pt`:
  - **Stage B2 (Coordinate Frame & Split Symmetry):** Synthetic MuJoCo `front_cam`, `compute_body_frame_goal_vec` (`roundtrip_err_m <= 2.4e-8`), `JetAutoEnv.step` (`+vy -> +0.0402 m` world `y`, `+wz -> +0.2365 rad`), and forced-goal controllers (`predict_reactive_action`, `HJEPAAgent(..., track="waypoint")`) all pass right-handed sign checks (`all_synthetic_coordinate_checks_passed = true`). However, `splits_v4/train_benchmark.npz` (`408` episodes, `13,188` transitions, `96` unique object layouts) shows `"approach"` transitions (`136` episodes, `4,154` transitions) are `75.9%` Right (`by_mean = -0.1890 m`, e.g., `blue_cylinder 97.6% R`, `green_cylinder 94.7% R`, `yellow_cylinder 73.3% R`, `red_box 71.2% R`, `cyan_cylinder 63.2% R`, `purple_box 61.3% R`), vs `46.5% L / 53.5% R` (`by_mean = +0.0310 m`) across all relations.
  - **Stage B3 & H1/H3/H4 (Task Embedding & Fixed-Image Goal Sweep):** `H1` (`task_embedding` collapse) is falsified (`6` `"approach"` embeddings have L2 norms `3.410..4.760`, mean pairwise L2 `5.391`, mean cosine `+0.093`, centered effective rank `4.623/5`; all `49` tokens `eff_rank = 14.767/16`). Over `2,940` sweep rows (`z_t_reencode_max_abs_diff = 0.0`), `||d g_xy / d e_t||_F = 0.7636 [0.7283, 0.8008]` vs `||d g_xy / d z_t||_F = 0.8524 [0.8177, 0.8888]` (ratio `0.8968`), and counterfactual `A->B` token swaps move `pred_xy` by `0.5855 m [0.5169, 0.6545]`, yet `eta_gt = -0.0105 [-0.1053, +0.0814]` (`Wilcoxon p = 0.9413`). Across `360` `"approach"` evaluations (`60x6`), `85.3%` predict Right (`dy_mean = -0.3510 m`), mirroring training right-side prevalence (`blue_cylinder 98.3% R, cyan_cylinder 98.3% R, green_cylinder 85.0% R, purple_box 85.0% R, yellow_cylinder 73.3% R, red_box 71.7% R`), with `(dx_b, dy_b)` ANOVA variance decomposed into `[0.495, 0.502]` scene (`z_t`), `[0.284, 0.181]` token (`e_t`), and `[0.221, 0.317]` interaction.
  - **Stage B4 (E1 Stage-Wise Probes, `546` rows, zero scene leakage):** Under `GroupKFold_scene`, `z_t` predicts `left_obj_xy` (`eucl = 0.084 m, dMAE = +0.0034 m, R2 = +0.068`) and `right_obj_xy` (`eucl = 0.099 m, dMAE = +0.0035 m, R2 = -0.084`) only negligibly above fold-mean baseline; all layers (`z_t, e_t, h0_cat, L1, L2, pred_xy`) fail to linearly decode category-bound `obj_1_xy` (`R2 <= +0.005`), `obj_2_xy` (`R2 <= +0.010`), `assigned_wp_xy` (`R2 <= +0.008`), and `rel_dir_xy` (`R2 <= 0.000`; `TargetCategoryHoldout` `L1 R2 = -2.191`, `L2 R2 = -1.383`).
  - **Stage B5/B6 (E2 Causal Interventions, `2,040` rows):** Opposite-side same-target-key donors exist for `60/60` scenes and all `<= 5%` norm-matched controls satisfy tolerance (`max_rel_act_norm_diff <= 1.16e-7`), yet `0/12` Holm–Bonferroni paired contrasts vs norm-matched random controls or orthogonal complements reach significance (`12/12` `holm_p_delta_eta = 1.0`, `holm_p_delta_gain = 1.0`; donor flip `eta_gt = -0.0793`; OOF subspace `k=8,16` `eta_gt = -0.0729..-0.0058`). Stage B6 closed-loop rescue did not trigger (`b6_triggered = false`, `NOT_TRIGGERED_NO_CAUSAL_RESCUE_CANDIDATE`). Raw findings remain recorded for independent scientific interpretation rather than converted into new principles.

---

## 5. Architectural Blueprint for Embodied Spatial Intelligence

Based on these empirical discoveries and falsifications, future embodied world-model architectures should implement:

1. **Active Visual Search Policy Decoupled from Tracking:** Reactive visual grounding succeeds reliably when objects are visible (93.3%), but fails when objects fall outside the field of view. Target acquisition requires an explicit exploration/search state machine.
2. **Deterministic Kinematic Proposer with Slip-Specialized Neural Residuals:** Implement trajectory candidate generation using deterministic kinematic unicycle models on FPGA/edge silicon (0.18 ms latency), reserving learned neural world models strictly for predicting non-linear wheel slip, terrain compliance, and dynamic obstacles.
3. **Hard Reactive Control Barrier Layer:** Replace soft prospective candidate penalties with a dedicated, high-frequency (100 Hz) Control Barrier Function (CBF) operating directly on range sensor boundaries.
4. **Group-Equivariant Visual Backbones:** Integrate $SE(2)$-steerable convolutional kernels ($G$-CNNs) to ensure mathematical equivariance under camera yaw and perspective changes.
