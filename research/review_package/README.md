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
1. **Dynamic Visual Grounding is Causally Confirmed Under Guaranteed Distance (v2):**
   When target relocation is strictly constrained so initial waypoint distance is $\ge 0.60$m (margin $\ge 0.25$m, mean initial distance $0.770$m) and physical relocation is $\ge 0.75$m with post-action evaluation, H-JEPA achieves **73.3% success (22/30)** and **median path length 0.488m** (mean $0.706$m). This eliminates the v1 initial distance confound where 21/30 episodes started inside the $0.35$m goal threshold. Active locomotion is confirmed, but performance exhibits a statistically significant degradation compared to canonical baseline ($100.0\%$, McNemar $p = 0.0078$, Holm-adjusted $p = 0.03125$).
2. **Task-Token Conditioning Under Dual-Visible Identical RGB:**
   In 30 bit-for-bit identical paired scenes where candidate objects A and B are simultaneously visible in the identical camera frame (both $\ge 25$ px, neither occluded), changing only the task token rotates the inferred visual goal vector by **$0.869$ rad ($49.8^\circ$)** on the identical visual latent. Closed-loop navigation achieves **66.7% success (20/30)** under Token A and **70.0% success (21/30)** under Token B. Dual-target discrimination (steering closer to A under Token A and closer to B under Token B) occurs in **20.0% (6/30)** of episodes, demonstrating that categorical token conditioning interacts causally with visual saliency but competes with canonical spatial expectations.
3. **Controller Memory Value Under Sensory Deprivation (v2):**
   Under continuous visual observation, pure visual control achieves parity with blended control (difference = 0.0%). However, under temporary 10-step camera blackout (steps 5..15), **visual-only control suffers a statistically significant increase in tracking error ($+0.097$m higher error, 95% CI $[0.062, 0.134]$, Wilcoxon $p = 3.79 \times 10^{-6}$)** compared to the blended controller. Final navigation success upon visual restoration recovers to 83.3% (non-inferior within $\delta=0.10$ margin of full memory 80.0%). Controller memory (handwritten accumulator paired with learned `micro_ego_head` displacements) causally maintains egocentric goal tracking during visual sensory deprivation.
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
| [`agent_computation_graph.json`](agent_computation_graph.json) | Complete Computation Graph | Graph formally distinguishing learned neural modules (encoder, task embed, grounding head, ego-motion head, predictors) from handwritten controller logic (memory buffer, geometric update, blending, MPC scorer) with cautious evidence statuses. |
| [`discrepancy_log.md`](discrepancy_log.md) | Discrepancy & Reconciliation Catalog | Enumerates all 10 numerical and conceptual discrepancies between prior draft reports and raw JSON artifacts with exact corrections. |
| [`experiment_index.json`](experiment_index.json) | Provenance & Hash Manifest | Cryptographic index containing script paths, SHA256 hashes, commands, checkpoint hashes, random seeds, and artifact paths (including v2 suites). |
| [`test_run.log`](test_run.log) | Clean Pytest Execution Log | Real verbatim pytest execution trace verifying all 48 test suite unit and integration tests passing. |

---

## 3. Primary Experiment Artifacts

All raw experiment artifacts are version-controlled, independently reproducible, and stored in the repository:

1. **Controlled Task Identity & Visual Grounding v2 Benchmark:**
   - Script: [`world_models_fpga/research/benchmarks/run_task_identity_visual_grounding_v2.py`](../../world_models_fpga/research/benchmarks/run_task_identity_visual_grounding_v2.py)
   - Summary JSON: [`world_models_fpga/research/benchmarks/results/task_identity_visual_grounding_v2.json`](../../world_models_fpga/research/benchmarks/results/task_identity_visual_grounding_v2.json)
   - Numerical Traces NPZ: [`world_models_fpga/research/benchmarks/results/task_identity_visual_grounding_v2_traces.npz`](../../world_models_fpga/research/benchmarks/results/task_identity_visual_grounding_v2_traces.npz)
2. **Goal Memory Stress Tests Under Sensory Blackout v2:**
   - Script: [`world_models_fpga/research/circuits/run_goal_memory_stress_tests_v2.py`](../../world_models_fpga/research/circuits/run_goal_memory_stress_tests_v2.py)
   - Summary JSON: [`world_models_fpga/research/circuits/results/goal_memory_stress_tests_v2.json`](../../world_models_fpga/research/circuits/results/goal_memory_stress_tests_v2.json)
   - Numerical Traces NPZ: [`world_models_fpga/research/circuits/results/goal_memory_stress_tests_v2_traces.npz`](../../world_models_fpga/research/circuits/results/goal_memory_stress_tests_v2_traces.npz)
3. **Task Identity vs Dynamic Visual Grounding Benchmark (v1 Historical):**
   - Script: [`world_models_fpga/research/benchmarks/run_task_identity_visual_grounding.py`](../../world_models_fpga/research/benchmarks/run_task_identity_visual_grounding.py)
   - Summary JSON: [`world_models_fpga/research/benchmarks/results/task_identity_visual_grounding.json`](../../world_models_fpga/research/benchmarks/results/task_identity_visual_grounding.json)
4. **Goal Memory Path Interventions Benchmark (v1 Historical):**
   - Script: [`world_models_fpga/research/circuits/run_goal_path_interventions.py`](../../world_models_fpga/research/circuits/run_goal_path_interventions.py)
   - Summary JSON: [`world_models_fpga/research/circuits/results/goal_path_interventions.json`](../../world_models_fpga/research/circuits/results/goal_path_interventions.json)
5. **Predictive Safety Benchmark:**
   - Script: [`world_models_fpga/research/benchmarks/run_predictive_safety_benchmark.py`](../../world_models_fpga/research/benchmarks/run_predictive_safety_benchmark.py)
   - Summary JSON: [`world_models_fpga/research/benchmarks/results/predictive_safety_benchmark.json`](../../world_models_fpga/research/benchmarks/results/predictive_safety_benchmark.json)
6. **Matched Temporal Controls Benchmark:**
   - Script: [`world_models_fpga/research/benchmarks/run_matched_temporal_controls.py`](../../world_models_fpga/research/benchmarks/run_matched_temporal_controls.py)
   - Summary JSON: [`world_models_fpga/research/benchmarks/results/matched_temporal_controls.json`](../../world_models_fpga/research/benchmarks/results/matched_temporal_controls.json)
7. **Track C Visual Grounding Suite:**
   - Script: [`world_models_fpga/research/benchmarks/run_track_c_visual_grounding.py`](../../world_models_fpga/research/benchmarks/run_track_c_visual_grounding.py)
   - Summary JSON: [`world_models_fpga/research/benchmarks/results/track_c_visual_grounding.json`](../../world_models_fpga/research/benchmarks/results/track_c_visual_grounding.json)
8. **Rigorous Behavioral Circuits (C1):**
   - Script: [`world_models_fpga/research/circuits/run_rigorous_behavioral_circuits.py`](../../world_models_fpga/research/circuits/run_rigorous_behavioral_circuits.py)
   - Summary JSON: [`world_models_fpga/research/circuits/results/rigorous_behavioral_circuits.json`](../../world_models_fpga/research/circuits/results/rigorous_behavioral_circuits.json)

---

## 4. Verification & Testing

All unit tests under `world_models_fpga/tests/` verify:
- Object visibility calculation via camera frustum and rendered segmentation.
- Strict separation of environment ground truth, intended target identity, and agent task token.
- Independence of branch-specific interventions (`task_id`, `task_embedding`, `grounding_latent`, `visual_goal`, `memory_prev`, `memory_updated`, `blended_goal`).
- Geometric memory update invariants and occlusion window tracking behavior.
- Exact bit-level reproducibility of random safety and baseline generators.

Run the test suite with:
```bash
.venv/bin/pytest world_models_fpga/tests/
```
All 48 tests pass cleanly (see [`test_run.log`](test_run.log)).
