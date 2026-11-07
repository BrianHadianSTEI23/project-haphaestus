# Scientific Claims Reassessment and Provenance Audit

**Branch:** `fix/interpretability`  
**Audit Date:** 2026-11-07  
**Scope:** Rigorous reassessment of all empirical claims, behavioral benchmarks, causal circuit attributions, invariance tests, model distillations, and hardware benchmarks across Project Hephaestus.

---

## 1. Executive Summary & Standard of Scientific Integrity

The purpose of this reassessment is to eliminate any discrepancy between what the source code actually establishes and what the written reports claim. A foundational principle of experimental science is that:
1. **Correlation or linear decodability does not establish causality.**
2. **Sensitivity of a model's internal activations does not establish behavioral necessity or sufficiency.**
3. **An experiment whose evaluation metric overrides the agent's behavior cannot establish task competency.**
4. **An intervention evaluated on the exact data used to discover its candidate subspace is methodologically circular.**
5. **A multidimensional target reduced to its Euclidean norm loses all directional information.**
6. **A hardware simulation does not constitute physical silicon execution.**
7. **Attaching contemporary commit hashes and dataset checksums to historical JSON outputs without verifying their execution provenance invalidates scientific attribution.**

In accordance with Priority 0, all nine evaluation categories in [`experiment_manifest.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/validation/experiment_manifest.json) have been categorized as **`LEGACY_UNVERIFIED`** until each experiment is systematically repaired, executed under controlled protocols with embedded execution metadata, and independently validated.

---

## 2. Systematic Audit & Reassessment Matrix

| Claim ID | Previous Stated Claim | Previous Status & Evidence | Methodological Flaw / Confound Discovered | Reassessed Status | Required Remediation |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **`H1`** | **Hierarchical Temporal Abstraction (K=5) suppresses compounding drift by 46.4%** | `VERIFIED` (E3) | The closed-loop H-JEPA macro planner was evaluated using a hand-designed 24-candidate detour library with a 2-phase motion primitive, while ablations (`no_temporal_hierarchy`) used a single-step primitive or different candidate sets, confounding planning search with temporal abstraction. | **`REQUIRES_REEXECUTION`** | Implement matched candidate library, matched scoring function, matched encoder, and equal evaluation budgets across Reactive, Prediction-Ignored, Recursive 1-step, Direct K-step ($K \in \{1, 3, 5, 10\}$), and Macro-to-Micro planners. |
| **`H2`** | **Modular Egocentric Coordinate Subspace is Falsified / Non-Selective** | `REJECTED` (Modular) / `WEAKENED` (Distributed) | **Critical Code Bug:** In [`test_necessity_sufficiency.py`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/test_necessity_sufficiency.py), `_discover_top_k_subspace()` reduced multidimensional targets to `np.linalg.norm()`. For C2, the target was $[\sin\theta, \cos\theta]$, whose norm is identically $1.0$. The correlation was computed against a constant vector, rendering candidate feature selection arbitrary noise. | **`WITHDRAWN (Invalid Method)` $\to$ `REQUIRES_REEXECUTION`** | Replace scalar norm with multi-output regression / canonical correlation analysis (CCA) preserving 2D heading and displacement. Recompute C2 from training data only and re-evaluate on held-out episodes. |
| **`H3`** | **Factorized Affordance-Appearance Equivariance** | `VERIFIED` (E3) | Color invariance was evaluated in Track B, where an explicit simulator-derived body-frame goal vector $[d_x, d_y]$ was externally provided to the agent. Because the goal coordinates were given, the agent did not need to visually identify the target from color cues. | **`DOWNGRADED` (E2 Output Sensitivity) $\to$ `REQUIRES_REEXECUTION`** | Create distinct tests: (1) external goal cues, (2) image-derived target identity, (3) color-grounded target selection, and (4) geometry-derived spatial relationships. Ensure agents are not rewarded for ignoring color when the task instruction requires color grounding. |
| **`H4`** | **Latent Lookahead Safety Barrier is Causally Necessary (+23.4% collisions when ablated)** | `VERIFIED` (E3) | While the collision surge under ablation is observable, behavioral evaluation aggregated `action_prediction` probe tasks where the evaluator overrode agent actions and unconditionally marked them 100% successful. | **`DOWNGRADED` (Contaminated Aggregate) $\to$ `REQUIRES_REEXECUTION`** | Remove `action_prediction` override from navigation success. Evaluate action prediction separately via MAE and cosine similarity. Re-run closed-loop ablations with clean navigation metrics. |
| **`H5`** | **Cross-Architecture Latent Geometric Transfer via Procrustes Alignment** | `VERIFIED` (E2) | Procrustes transfer evaluated open-loop cosine similarity between predicted latents of transplanted predictors, without verifying whether transplanted predictions yield valid physical states or successful closed-loop navigation. | **`DOWNGRADED` (Open-Loop Latent Alignment Only) $\to$ `REQUIRES_REEXECUTION`** | Evaluate transplanted predictors against actual held-out future observations / physical states and test closed-loop navigation transfer. |
| **`H6`** | **Bit-Exact Hardware Predictability & 70.49x Acceleration on KR260 FPGA** | `VERIFIED` (E3) | 1. FPGA PL latency was measured via cycle-accurate Verilog RTL simulation (`iverilog`), not physical bitstream execution on the KR260 FPGA PL. 2. Vivado 2023.2 synthesis at 200 MHz exhibited negative slack (WNS `-2.836 ns`, $F_{\max} \approx 127.62$ MHz). 3. Board power was sampled from whole-board telemetry, not PL rails. 4. Top module executes single-pass macro/micro inference, not the full 24-MPC loop in hardware logic. | **`DOWNGRADED` (RTL Simulation & PS Execution Only)** | Strictly separate RTL simulation, post-synthesis timing estimates at timing-closed frequency (125 MHz), physical PS execution, and physical PL status. Clarify that top module accelerates single macro/micro passes. |
| **`BEHAV-AGG`** | **Track B Navigation Success Rate (55.6% for H-JEPA)** | `VERIFIED` | In [`evaluator.py`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/evaluator.py), when `task_spec["capability"] == "action_prediction"`, agent actions were overridden with `action_probe_vector` and episodes marked unconditionally `is_success = True`. | **`WITHDRAWN (Contaminated Aggregate)`** | Separate navigation tasks from action-prediction probe tasks. Report goal reaching, collision avoidance, and temporal planning as distinct distributions with paired confidence intervals. |
| **`LABEL-ALIGN`** | **Dataset Action Supervision Alignment** | `VERIFIED` | In [`generate_benchmark_dataset.py`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/datasets/generation/generate_benchmark_dataset.py), multi-waypoint tasks stored the final waypoint as `grounded_goal`, while expert actions were computed toward intermediate waypoints. | **`FLAWED SUPERVISION` $\to$ `REQUIRES_REEXECUTION`** | Store active waypoint/subgoal, remaining waypoint sequence, and action horizon per transition. Retrain models that depend on goal-conditioned action supervision. |
| **`INFO-REG`** | **Characterization of Track B and Track C Information Regimes** | `VERIFIED` | Track B was described as "vision-only", ignoring the externally provided body-frame goal vector $[d_x, d_y]$. Track C was described as "natural language understanding", whereas `encode_task_id()` maps discrete categorical IDs. | **`DOWNGRADED / RE-CHARACTERIZED`** | Explicitly define Track B as "Structured-Goal Navigation (Simulator-Derived Egocentric Vector)" and Track C as "Image + Categorical Task-ID Conditioned Navigation". |
| **`CIRC-CAUSAL`**| **Circuit Registry C1–C5 Behavioral Necessity & Sufficiency** | `VERIFIED` | [`test_necessity_sufficiency.py`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/test_necessity_sufficiency.py) measured changes in model output tensors (representation sensitivity), NOT closed-loop behavioral performance in the simulator. | **`DOWNGRADED` (Representation Sensitivity Only)** | Implement closed-loop behavioral interventions over independent episodes with matched random and unrelated feature controls. |
| **`PATCH-PROV`** | **Activation Patching Subspace Recovery** | `VERIFIED` | [`run_activation_patching.py`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/run_activation_patching.py) discovered top-k dimensions on the donor/recipient evaluation data themselves, causing circular in-sample discovery. | **`FLAWED METHOD` $\to$ `REQUIRES_REEXECUTION`** | Discover candidate features strictly on training split; evaluate patching on independent evaluation split. |

---

## 3. Detailed Audit of Critical Flaws

### 3.1 The C2 Yaw-Target Euclidean Norm Bug
In `world_models_fpga/research/circuits/test_necessity_sufficiency.py`:
```python
def _discover_top_k_subspace(latents: np.ndarray, target: np.ndarray, k: int = 8) -> np.ndarray:
    if target.ndim > 1:
        target = np.linalg.norm(target, axis=-1)
    t_c = (target - np.mean(target)) / (np.std(target) + 1e-8)
    corrs = np.zeros(latents.shape[1], dtype=np.float32)
    for d in range(latents.shape[1]):
        l_d = latents[:, d]
        corrs[d] = abs(float(np.mean(((l_d - np.mean(l_d)) / (np.std(l_d) + 1e-8)) * t_c)))
    return np.argsort(-corrs)[:k]
```
For C2 (`egocentric_transformation`):
```python
c2_dims_h = _discover_top_k_subspace(s_tr_h, np.stack([np.sin(tr_poses[:, 2]), np.cos(tr_poses[:, 2])], axis=-1), k=8)
```
Because $\sin^2\theta + \cos^2\theta \equiv 1$, `np.linalg.norm(...)` evaluates to an array of constant $1.0$s. Consequently, `np.std(target)` is zero, correlation values are essentially zero noise, and the selected top-8 dimensions are arbitrary. Concluding that C2 was "falsified" because these arbitrarily selected dimensions had low selectivity was a direct consequence of this code bug.

Furthermore, taking the Euclidean norm of other multidimensional targets:
- `tr_gvecs[:, :2]` ($[d_x, d_y]$) collapses 2D direction into scalar distance $\sqrt{d_x^2 + d_y^2}$.
- `tr_acts` ($[v_x, v_y, \omega_z]$) collapses linear and angular velocity into speed.
- `tr_goals[:, :2] - tr_poses[:, :2]` collapses directional displacement into distance.

All multidimensional target correlations must be computed using multi-output linear regression $R^2$ or component-wise canonical correlation analysis.

### 3.2 Action Prediction Episode Contamination
In `world_models_fpga/research/benchmarks/evaluator.py`:
```python
if task_spec.get("capability") == "action_prediction" and "action_probe_vector" in task_spec:
    action = np.asarray(task_spec["action_probe_vector"], dtype=np.float32)
...
elif task_spec.get("capability") == "action_prediction":
    is_success = True
```
When `action_prediction` tasks were included in evaluation suites:
1. The robot's action was overridden with the probe vector, completely bypassing the policy under test.
2. The episode was unconditionally declared successful regardless of robot motion or collisions.
3. This artificially inflated the aggregate success rate and masked navigation failures.

### 3.3 Multi-Waypoint Action Supervision Misalignment
In `world_models_fpga/research/datasets/generation/generate_benchmark_dataset.py`:
- `goal_target = waypoints[-1]`
- `action = self.navigator.compute_action(curr_pose, curr_wp, ...)`
- `trans["grounded_goal"] = goal_target`
- `trans["action"] = action`

For a trajectory with 3 waypoints $[W_0, W_1, W_2]$, while the robot is traversing toward $W_0$, the demonstrator action $\mathbf{a}_t$ is steering toward $W_0$. However, `grounded_goal` records $W_2$. When a neural model is trained to predict $\mathbf{a}_t$ conditioned on `grounded_goal`, it is supervised to produce actions steering toward $W_0$ when told the goal is $W_2$, injecting contradictory supervisory signals.

---

## 4. Remediation Roadmap Across Phases

1. **Phase 1 (Complete):** Strict provenance manifest implemented; all 9 categories labeled `LEGACY_UNVERIFIED`; this reassessment document established.
2. **Phase 2 (Priority 1):** Remove action-prediction override from navigation success; evaluate action prediction separately; align multi-waypoint active subgoals vs. final goals; add 3-waypoint regression test; regenerate clean benchmark data and retrain affected models.
3. **Phase 3 (Priority 2):** Fix C2 yaw-target bug using multi-output regression / CCA; separate representation sensitivity from closed-loop behavioral necessity/sufficiency; split discovery and evaluation in activation patching.
4. **Phase 4 (Priority 3):** Implement matched experimental comparison for temporal prediction (Reactive, Prediction-ignored, Recursive 1-step, Direct K-step, Macro-to-micro) with matched candidate sets, scoring functions, and parameter budgets.
5. **Phase 5 (Priority 4):** Rebuild appearance and geometry invariance tests distinguishing external cues from image-derived target identity and color grounding.
6. **Phase 6 (Priority 5):** Re-evaluate cross-model Procrustes transfer on physical states and closed-loop navigation.
7. **Phase 7 (Priority 6):** Correct hardware evidence: separate RTL simulation from physical PS execution, report timing-closed frequency (125 MHz), and clarify single-pass acceleration.
8. **Phase 8 (Priority 7):** Update all reports, downgrade/withdraw unsupported claims, and synthesize only scientifically defensible principles.
