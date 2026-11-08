# Phase 3 Re-Analysis & Empirical Audit: Deconstructing Task-Conditioned Grounding and Ego-Motion Memory

**Audit Baseline Commit:** `8229f786309fb6d9ea669e8a783793c92ceca564`  
**Date:** 2026-11-08  
**Investigator:** Senior Mechanistic Interpretability, Robotics, and Computer Architecture Research Team  
**Repository:** `BrianHadianSTEI23/project-haphaestus` (`fix/interpretability`)

---

## 1. Executive Summary

This document performs a rigorous, skeptical re-analysis of the experimental data generated during Phase 3 v2 (`task_identity_visual_grounding_v2.json` and `goal_memory_stress_tests_v2.json`). While the v2 experiments successfully eliminated the initial-distance artifact of v1 (where 21/30 episodes started inside the goal threshold), careful inspection reveals critical experimental and interpretational confounds that preclude claiming:
1. That H-JEPA implements a task-conditioned *visual object binding* mechanism.
2. That the learned `micro_ego_head` performs a causally beneficial egocentric coordinate transformation during visual sensory deprivation.

This re-analysis provides the mathematical, empirical, and architectural foundation for the controlled interventions executed in Phase 4.

---

## 2. Audit of Task-Conditioned Visual Grounding (v2)

### 2.1 Raw Empirical Findings
From `world_models_fpga/research/benchmarks/results/task_identity_visual_grounding_v2.json` ($N=30$ per condition):
- **Canonical Baseline (`cond_1`):** Success **30/30 (100.0%)**, mean initial distance 0.587m, mean path length 0.270m, mean final error 0.328m, step visual displacement error **0.304m**, step visual angular error **0.241 rad**.
- **Relocated Target Visible (`cond_2`):** Success **22/30 (73.3%)**, mean initial distance 0.770m (min 0.666m), median path length **0.488m** (mean 0.706m), mean final error 0.652m (median 0.343m), step visual displacement error **0.754m**, step visual angular error **0.943 rad**.
  - McNemar exact vs Canonical: $n_{10} = 8, n_{01} = 0, p = 0.0078125$, Holm-adjusted $p = 0.03125$.
  - Closer to canonical waypoint: 8/30 (26.7%).
- **Same-RGB Paired Token Contrast (`cond_4a` vs `cond_4b`):**
  - Evaluated on 30 bit-for-bit identical initial RGB images (SHA-256 match verified across all 30 pairs, both objects visible with $>1000$ pixels each).
  - Closed-loop success under Token A: **20/30 (66.7%)**, median path length 0.555m, final error 0.497m.
  - Closed-loop success under Token B: **21/30 (70.0%)**, median path length 0.504m, final error 0.483m.
  - Mean angular shift of inferred goal vector $[\widehat{dx}_b, \widehat{dy}_b]$: **$0.8686$ rad ($49.77^\circ$)**.

### 2.2 Deep Trace Re-Analysis: Object Binding Breakdown
Re-analyzing the step-0 traces from `task_identity_visual_grounding_v2_traces.npz` across all 30 paired scenes yields the exact target-selection rates:
- **A-Correct Count:** **16 / 30 (53.3%)** — Under Token A, the inferred goal vector is closer to physical Target A than to Target B in 16 of 30 scenes.
- **B-Correct Count:** **16 / 30 (53.3%)** — Under Token B, the inferred goal vector is closer to physical Target B than to Target A in 16 of 30 scenes.
- **Dual-Discrimination Count:** **6 / 30 (20.0%)** — In only 6 of 30 paired scenes does Token A steer closer to Target A *and* Token B steer closer to Target B.
- **A-Only Correct:** 10 / 30 (33.3%).
- **B-Only Correct:** 10 / 30 (33.3%).
- **Neither Correct:** 4 / 30 (13.3%).

### 2.3 What the Metrics Measure vs What They Do NOT Measure

| Metric | Stated Interpretation in Phase 3 | What It Actually Measures | What It Does NOT Measure |
|---|---|---|---|
| **Mean Angular Shift ($0.869$ rad)** | "Task token redirects goal vector toward designated target" | The sensitivity of the grounding head function $f(z, e)$ to task embedding $e$. The output vector changes when $e$ changes. | It does **NOT** measure whether the rotation points toward the designated physical object. A fixed random orthogonal vector addition $f(z, e) = a(z) + R(e)$ produces large angular shifts without looking at target locations. |
| **Dual Discrimination (6/30, 20%)** | Reported as a secondary qualification of steering | The joint conjunction: $P(\text{A-correct} \cap \text{B-correct})$. Under pure random 50/50 guessing, $0.5 \times 0.5 = 25\%$. | 20.0% is **at or below chance level** for a binary object selector. |
| **A-Correct & B-Correct (53.3%)** | Conflated into closed-loop navigation success (~70%) | Marginal accuracy of pointing toward the designated object in a 2-target scene. | 53.3% is indistinguishable from coin-flipping ($p = 0.855$ binomial test against 50%). |
| **Closed-Loop Success (66.7% / 70.0%)** | Interpreted as evidence of target discrimination | The robot reaches *some* target within the $0.35$m radius when guided by a reactive policy in a bounded arena. | It does not prove the robot reached the *designated* target vs the unintended distractor. |

### 2.4 Unsupported Scientific Claims
1. **Unsupported Claim:** *"Changing task token under bit-for-bit identical dual-visible RGB observation redirects goal vector toward designated object"* (CLAIM_02 in Phase 3).
   - **Correction:** The task token rotates the output vector ($49.8^\circ$), but the direction of rotation does not align with the relative geometry of the designated target (accuracy is 53.3% marginally, 20.0% jointly).
2. **Unsupported Claim:** *"H-JEPA dynamically grounds relocated physical targets from RGB"* (CLAIM_01).
   - **Correction:** While H-JEPA reaches relocated targets in 73.3% of episodes when visible, its mean step displacement error increases by $+148\%$ (from 0.304m to 0.754m), and 26.7% of trajectories steer toward canonical coordinates rather than the visible object. The mechanism is a mixture of spatial prior and visual perturbation, not a clean visual coordinate decoder.

---

## 3. Audit of Controller Goal Memory Under Blackout (v2)

### 3.1 Raw Empirical Findings
From `world_models_fpga/research/circuits/results/goal_memory_stress_tests_v2.json` ($N=30$ matched episodes per condition, 10-step visual sensory blackout at steps 5..15):

| Condition | Success Count | Success Rate | Blackout Goal Error (m) | Blackout Angular Error (rad) | Final Error (m) |
|---|:---:|:---:|:---:|:---:|:---:|
| **`cond_1_full_visual_and_memory`** | 24 | 80.0% | **0.648** | 1.063 | 0.415 |
| **`cond_2_visual_only`** | 25 | 83.3% | **0.745** | 1.176 | 0.405 |
| **`cond_3_memory_reset_step`** | 25 | 83.3% | 0.567 | 1.266 | 0.434 |
| **`cond_4_frozen_memory`** | 30 | 100.0% | **0.490** | **0.831** | **0.336** |
| **`cond_5_corrupted_memory`** | 28 | 93.3% | 0.685 | 1.169 | 0.379 |
| **`cond_6_zeroed_ego_motion`** | 24 | 80.0% | **0.629** | 1.021 | 0.417 |
| **`cond_7_perturbed_ego_motion`** | 25 | 83.3% | 0.643 | 1.060 | 0.411 |
| **`cond_8_matched_donor_memory`** | 29 | 96.7% | 0.501 | 0.889 | 0.349 |
| **`cond_9_mismatched_donor_memory`** | 27 | 90.0% | **0.438** | **0.821** | 0.400 |

### 3.2 Paradoxical Anomalies in the Raw Evidence
Four glaring contradictions exist in the raw evidence:
1. **Frozen memory (0.490m) substantially outperforms the full memory model (0.648m)** during blackout.
2. **Zeroing the learned ego-motion displacement (0.629m) yields *lower* tracking error than using the learned ego-motion head (0.648m)**.
3. **Mismatched donor memory from an unrelated episode (0.438m) achieves the lowest tracking error of any condition**, beating full memory by $-0.210$m ($p = 7.15 \times 10^{-6}$).
4. **Visual-only control achieves higher closed-loop navigation success (83.3%) than full memory (80.0%)**, despite having higher tracking error during the blackout window.

### 3.3 Root-Cause Architectural Diagnostics

#### Confound A: Zero Latent Delta During Blackout
In `h_jepa.py`, displacement prediction is defined as:
```python
def predict_micro_displacement(self, s_t: torch.Tensor, next_s: torch.Tensor) -> torch.Tensor:
    return self.micro_ego_head(torch.cat([torch.zeros_like(s_t), next_s - s_t], dim=-1))
```
During a black-frame dropout, the input frame is identically zero: $\text{RGB}_t = 0$.
The visual encoder outputs a constant static latent: $s_t = \text{Encoder}(0) = s_{\text{black}}$.
Consequently:
$$\Delta s = s_t - s_{t-1} = s_{\text{black}} - s_{\text{black}} = \mathbf{0}$$
The learned `micro_ego_head` receives an identically zero latent delta at every step of the blackout window! It is physically impossible for the network to perceive robot motion from visual latents when the visual input does not change. Any predicted displacement is purely the static network bias: $b_{\text{ego}} = \text{micro\_ego\_head}(\mathbf{0})$.

#### Confound B: Exponential Blending Decay Toward Black-Frame Garbage
In `model_wrappers.py`, the controller blends visual and memory goals at every step:
$$g_t = 0.65 \cdot g_{\text{vis}}(s_t) + 0.35 \cdot g_{\text{mem}}$$
During blackout, $g_{\text{vis}}(s_{\text{black}})$ is an ungrounded static vector computed from the black frame. At each timestep:
- Step 1 of blackout: $g_t = 0.65 \cdot g_{\text{black}} + 0.35 \cdot g_{\text{mem}}$
- Step 2 of blackout: $g_t = 0.65 \cdot g_{\text{black}} + 0.35 \cdot (0.65 \cdot g_{\text{black}} + 0.35 \cdot g_{\text{mem}}) = 0.8775 \cdot g_{\text{black}} + 0.1225 \cdot g_{\text{mem}}$
- Step 10 of blackout: $g_t \approx 1.0 \cdot g_{\text{black}} + 2.75 \times 10^{-5} \cdot g_{\text{mem}}$

The memory vector is effectively erased and replaced by the arbitrary static black-frame vector!
In contrast, in `cond_4_frozen_memory`, the hook constantly overrode `mem_updated` with the initial step-0 visual goal, repeatedly re-injecting 35% of the true initial goal vector. That is why frozen memory performed better.

#### Confound C: Donor Memory Injection from Unoccluded Clean Trajectories
In `cond_8` and `cond_9`, the donor trace was recorded from a *clean baseline episode* where the robot had full visual observation and successfully navigated to the target. Thus, donor memory traces were valid, moving goal vectors pointing toward targets. In canonical arena setups, targets are situated at similar radii (~1.2m); hence, even a mismatched donor trace pointed in a realistic forward direction, outperforming the black-frame corrupted full memory model.

#### Confound D: Closed-Loop Physical Trajectory Confound
Because each condition took actions generated from its own goal estimate, different conditions produced *different physical trajectories*. Evaluating goal-vector tracking error across divergent physical paths confounds the quality of the memory update with the robot's physical locomotion.

---

## 4. Reconciled Evaluation Matrix: What Was Proven vs What Was Disproven

| Topic | Phase 3 Claim | Reconciled Status | Root Cause & Evidence |
|---|---|---|---|
| **Visual Grounding** | Confirmed Dynamic Grounding (93.3% v1, 73.3% v2) | **Partially Grounded, Heavily Biased** | Active locomotion verified (median path 0.488m), but step goal error is high (0.754m vs 0.304m) and 26.7% steer to canonical. |
| **Task Token Steering** | Task token redirects goal vector toward designated object ($49.8^\circ$ shift) | **Falsified as Object Binding; Confirmed as Arbitrary Vector Rotation** | Angular shift is real ($49.8^\circ$), but A-correct is 53.3% and dual-discrimination is 20.0% (chance level). |
| **Goal Memory** | Controller memory maintains tracking under blackout ($+0.097$m visual-only error) | **Confounded by Static Black-Frame Leakage and Blending Decay** | Latent delta is zero during blackout ($\Delta s = 0$). Memory decays into black-frame static vector. Frozen memory and mismatched donors beat full memory. |
| **Ego-Motion Head** | `micro_ego_head` provides egocentric coordinate tracking | **Unverified / Possibly Counter-Productive** | Zeroing ego-motion (0.629m) beats the learned head (0.648m). Direct translation/rotation accuracy was never tested. |

---

## 5. Phase 4 Experimental Requirements & Competing Hypotheses

To resolve these open questions, Phase 4 implements three new controlled experiments:

### 5.1 Experiment A: Task-Conditioned Visual Binding (`run_task_conditioned_binding_v3.py`)
- **Hypothesis 1 (Visual Object Binding):** Task token $e$ selects visible object $k$, and visual geometry $V(z)$ determines egocentric vector: $g(z, e) = \text{Select}(V_A(z), V_B(z), e)$.
- **Hypothesis 2 (Spatial Association Prior):** Task token $e$ retrieves a canonical learned spatial coordinate, largely independent of visual geometry: $g(z, e) = g_{\text{prior}}(e) + \epsilon(z)$.
- **Hypothesis 3 (Mixed Scene-to-Goal Non-Linear Decoding):** Task token and visual latent interact through distributed non-linear MLP layers without explicit object segmentation: $g(z, e) = \text{MLP}(z, e)$.
- **Experimental Test:** 100+ dual-visible scenes ($>100$ px segmentation each, $>0.60$m distance), 2x2 factorial evaluation ($g_{AA}, g_{AB}, g_{BA}, g_{BB}$), interaction metric $I$, and an independent visual saliency manipulation (area/color mutations).

### 5.2 Experiment B: Internal Mechanistic Interventions (`run_task_conditioned_circuits_v3.py`)
- Instrument intermediate activations of `grounding_head` (input 48D, layer 1 pre/post ReLU, layer 2 pre/post ReLU, raw 4D output).
- Surgical activation patching between matched and mismatched donor scenes to identify where object identity binds to spatial coordinates.

### 5.3 Experiment C: Ego-Memory Mechanism & Same-Trajectory Replay (`run_ego_memory_mechanism_v3.py`)
- **Direct Ego-Motion Head Calibration:** Offline test of `micro_ego_head` against ground-truth translation $[dx, dy]$ and rotation $d\theta$ across varying linear and angular velocities.
- **Same-Trajectory Replay (Mandatory):** Replay identical recorded robot trajectories through all 8 memory conditions (visual-only, full memory, frozen, zeroed, shuffled, matched donor, mismatched donor, oracle odometry), eliminating closed-loop path divergence.
- **Dynamic Black-Frame Dropouts:** Blackouts of duration $K \in \{2, 5, 10, 20\}$ across straight, turning, and combined motions.
- **Fixed vs Adaptive Blending:** Evaluate fixed 0.65/0.35 against uncertainty-weighted and memory-only rules.
