# Project Hephaestus — Scientific Review & Evidence Reconciliation Package

**Branch:** `fix/interpretability`  
**Target Architecture:** Hierarchical Joint-Embedding Predictive Architecture (H-JEPA) for Autonomous Mobile Robots (AMR)  
**Execution Environment:** NVIDIA RTX 3050 Laptop GPU / AMD Kria KR260 SOM | PyTorch 2.14.1+cu126 | MuJoCo 3.1.6 (EGL headless)

---

## 1. Executive Summary & Purpose

This review package provides the scientific audit, empirical reconciliation, and mechanistic verification for Project Hephaestus. The objective is to discover and validate **reusable computational principles of embodied spatial intelligence through mechanistic interpretability**.

The core investigation answers:
> **Does H-JEPA dynamically ground the current physical location of a target from RGB, or does it predominantly retrieve a task-conditioned canonical spatial expectation? How do task identity, visual geometry, and controller memory interact?**

### Key Scientific Findings:
1. **Dynamic Visual Grounding is Causally Confirmed (Under Camera Visibility):**
   When the target object is physically relocated to a novel arena position and verified to lie within the robot's front camera field of view, H-JEPA dynamically tracks and navigates to the relocated target with **93.3% success (28/30)** and **0.257m** mean goal error (eval waypoint distance drops from 0.57m to 0.26m; only 2/30 finish closer to canonical). The previous 30% success rate in unconstrained relocation arose because targets were placed behind or out of view of the front camera, which requires an active visual search policy rather than reactive visual tracking.
2. **Visual Saliency Bounds Semantic Token Steering:**
   When the physical RGB scene is held fixed and the task token is swapped to designate a different arena object, success collapses to **3.3% (1/30)** with **1.524m** error ($p = 3.73 \times 10^{-9}$, Holm-adjusted $p = 1.86 \times 10^{-8}$). Angular error jumps to 1.844 rad, with 27/30 trajectories remaining closer to canonical. This proves that H-JEPA does **not** execute a blind feedforward semantic map from token alone; goal generation requires visual saliency evidence matching the categorical token.
3. **Goal Memory is a Controller Heuristic, Not Learned Recurrent Memory:**
   Egocentric goal memory (`_ego_goal_memory`), previous latent tracking, and fixed 0.65/0.35 blending are handwritten controller logic in `model_wrappers.py`, containing **zero learned recurrent parameters** in `HJEPAModel`. Across 7 controlled memory path interventions, a pure visual controller (`visual_only`, 0% memory) achieves **96.7% success (29/30)**, matching the default blended controller (96.7%, $p = 1.0000$). Memory corruption and mismatched donor injection also achieve $\ge 96.7\%$ success. The agent's spatial behavior is driven by reactive visual grounding and world-model candidate scoring, not recurrent memory integration.
4. **C1 Subspace is Falsified as a Modular Circuit (Unresolved Representation Hypothesis):**
   Ablating C1 yields a minimal 3.3% drop ($p = 0.9032$ vs random 8D drop of 20.8% $\pm$ 14.8%, $z = -1.177$). Random 8D subspace patching achieves a *higher* offline donor cosine (0.9854) than candidate C1 (0.9779), and random patch rescue (90.9%) is indistinguishable from matched patch rescue (93.3%, $p = 0.6129$). C1 is falsified as a selective circuit; donor activation transfer does not constitute selective rescue. C1 remains an unresolved representation hypothesis beyond the falsification of this localized linear subspace.
5. **Prospective Collision Safety Head Provides No Closed-Loop Collision Reduction:**
   In closed-loop navigation, intact, shuffled, and disabled collision heads all produce identical **70.0% collision rates** ($p = 1.0000$). Offline collision discrimination exhibits AUROC = 1.000 with a clustered bootstrap 95% CI of **[0.5, 1.0]** across 20 episode clusters. The safety filter operates as an empirical soft penalty, not a formal Control Barrier Function.
6. **Temporal Hierarchy Beats Random Selection but Equals 1-Step Lookahead:**
   Multi-step macro rollouts ($K=5$) significantly outperform random candidate selection (46.0% vs 26.0%, Holm-adjusted $p = 0.0117$), but do not provide a statistically significant closed-loop advantage over recursive 1-step micro lookahead (46.0% vs 50.0%, $p = 0.7266$, adjusted $p = 1.0000$).

---

## 2. Review Package Contents

| File | Purpose | Description |
|---|---|---|
| [`claims_evidence.csv`](claims_evidence.csv) | Exhaustive Claims & Evidence Ledger | All claims linked to script symbols, lines, raw JSON pointers, sample sizes, observed values, p-values, CIs, and caveats. |
| [`agent_computation_graph.json`](agent_computation_graph.json) | Complete Computation Graph | Graph formally distinguishing learned neural modules (encoder, task embed, grounding head, ego-motion head, predictors) from handwritten controller logic (memory buffer, geometric update, blending, MPC scorer). |
| [`discrepancy_log.md`](discrepancy_log.md) | Discrepancy & Reconciliation Catalog | Enumerates all 8 numerical and conceptual discrepancies between prior draft reports and raw JSON artifacts with exact corrections. |
| [`experiment_index.json`](experiment_index.json) | Provenance & Hash Manifest | Cryptographic index containing script paths, SHA256 hashes, commands, checkpoint hashes, random seeds, and artifact paths. |

---

## 3. Primary Experiment Artifacts

All raw experiment artifacts are version-controlled, independently reproducible, and stored in the repository:

1. **Task Identity vs Dynamic Visual Grounding Benchmark:**
   - Script: [`world_models_fpga/research/benchmarks/run_task_identity_visual_grounding.py`](../../world_models_fpga/research/benchmarks/run_task_identity_visual_grounding.py)
   - Summary JSON: [`world_models_fpga/research/benchmarks/results/task_identity_visual_grounding.json`](../../world_models_fpga/research/benchmarks/results/task_identity_visual_grounding.json)
   - Numerical Traces NPZ: [`world_models_fpga/research/benchmarks/results/task_identity_visual_grounding_traces.npz`](../../world_models_fpga/research/benchmarks/results/task_identity_visual_grounding_traces.npz)
2. **Goal Memory Path Interventions Benchmark:**
   - Script: [`world_models_fpga/research/circuits/run_goal_path_interventions.py`](../../world_models_fpga/research/circuits/run_goal_path_interventions.py)
   - Summary JSON: [`world_models_fpga/research/circuits/results/goal_path_interventions.json`](../../world_models_fpga/research/circuits/results/goal_path_interventions.json)
3. **Predictive Safety Benchmark:**
   - Script: [`world_models_fpga/research/benchmarks/run_predictive_safety_benchmark.py`](../../world_models_fpga/research/benchmarks/run_predictive_safety_benchmark.py)
   - Summary JSON: [`world_models_fpga/research/benchmarks/results/predictive_safety_benchmark.json`](../../world_models_fpga/research/benchmarks/results/predictive_safety_benchmark.json)
4. **Matched Temporal Controls Benchmark:**
   - Script: [`world_models_fpga/research/benchmarks/run_matched_temporal_controls.py`](../../world_models_fpga/research/benchmarks/run_matched_temporal_controls.py)
   - Summary JSON: [`world_models_fpga/research/benchmarks/results/matched_temporal_controls.json`](../../world_models_fpga/research/benchmarks/results/matched_temporal_controls.json)
5. **Track C Visual Grounding Suite:**
   - Script: [`world_models_fpga/research/benchmarks/run_track_c_visual_grounding.py`](../../world_models_fpga/research/benchmarks/run_track_c_visual_grounding.py)
   - Summary JSON: [`world_models_fpga/research/benchmarks/results/track_c_visual_grounding.json`](../../world_models_fpga/research/benchmarks/results/track_c_visual_grounding.json)
6. **Rigorous Behavioral Circuits (C1):**
   - Script: [`world_models_fpga/research/circuits/run_rigorous_behavioral_circuits.py`](../../world_models_fpga/research/circuits/run_rigorous_behavioral_circuits.py)
   - Summary JSON: [`world_models_fpga/research/circuits/results/rigorous_behavioral_circuits.json`](../../world_models_fpga/research/circuits/results/rigorous_behavioral_circuits.json)

---

## 4. Verification & Testing

All unit tests under `world_models_fpga/tests/` verify:
- Object visibility calculation via camera frustum and rendered segmentation.
- Strict separation of environment ground truth, intended target identity, and agent task token.
- Independence of branch-specific interventions (ensuring goal branch interventions leave `model.encode()` and policy heads unchanged).
- Geometric memory update invariants.
- Exact bit-level reproducibility of random safety and baseline generators.

Run the test suite with:
```bash
.venv/bin/pytest world_models_fpga/tests/
```
All 41 tests pass cleanly.
