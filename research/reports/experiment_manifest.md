# Project Hephaestus — Empirical Experiment Manifest & Provenance Registry (`experiment_manifest.md`)

**Branch:** `fix/interpretability`  
**Git Commit SHA:** `cbc6c4e6c0d7ff439365db135564c0fffcf1a209`  
**Host Environment:** Linux `6.8.0-138-generic x86_64`, Intel Core i7 / 16 GB RAM, NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.4), Python 3.13.16  
**Embedded Edge Environment:** AMD Kria KR260 SOM (`ubuntu@kria.local`, Linux `5.15.0-1080-xilinx-zynqmp aarch64`, Quad-Core ARM Cortex-A53 @ 1.333 GHz, Xilinx Zynq UltraScale+ XCK26 FPGA)  
**FPGA Toolchain:** Xilinx Vivado 2023.2 Post-Synthesis (`synth_kr260.tcl` on `xck26-sfvc784-2LV-c`) + Icarus Verilog (`iverilog` / `vvp` v12.0)  

---

## 1. Primary Dataset Splits & Cryptographic Digests (`splits_v4/`)

All experiments are executed on the cryptographically validated, multi-waypoint aligned corpus [`splits_v4/`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/datasets/splits_v4/):

| Dataset Artifact | Total Episodes | Transitions | SHA-256 Digest | Purpose & Split Role |
| :--- | :---: | :---: | :--- | :--- |
| **`train_benchmark.npz`** | `408` | `18,360` | `8ba362a4908955d56610333ae5c4c44cadb6054bdcdc9490b12b6c9c4a06bab7` | Training split for visual world models and representation autoencoders. Zero episode overlap with val/test. |
| **`val_benchmark.npz`** | `96` | `4,320` | `8f5a21df67190c63e71d5f5927f23dab17cce1b2e66c055e86999ef4a57d29a4` | Validation split for early stopping, model checkpoint selection, and ridge regression $\alpha$ hyperparameter tuning. |
| **`held_out_test_benchmark.npz`** | `96` | `4,320` | `5b5153fcef96e2f0398a0e13a21b942d599066ef97c902ba711167f9e9ad9f38` | Primary held-out test split for linear probing, effective rank, and open-loop model evaluation. |
| **`counterfactual_suite.npz`** | `350` (`7 \times 50`) | `15,750` | `0d1516dcb86d3a856149148e97d5ba24c4553f5232beec253f0f5e29c921db47` | Matched-pair counterfactual suite covering E1–E7 interventions (target position, color, rotation, lighting, obstacle, shape, pitch). |
| **`disambiguation_suite.npz`** | `100` | `4,500` | `a6d215541313c77c51f5762c4099e1a5c572435efa07cc14d3e1da8e3c7de700` | Behavioral disambiguation suite testing visual vs. spatial conflict resolution (DISAMBIG_01–04). |
| **`independent_validation_suite.npz`** | `60` | `2,700` | `20878ceae77ef3c047ba13517174cb6d90962a92ffbe4cd3609a38d47dd2ed2b` | Independent validation suite for Phase 10 physical perturbation, multi-horizon rollout MSE, and Pearson/Spearman correlation. |

---

## 2. Model Checkpoints & Parameter Counts (`checkpoints_v4/`)

| Model Identifier | Seed | Params | Best Val Loss | SHA-256 Digest | Checkpoint File Path |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Baseline (Best)** | `42` | `155,608` | `0.4907` | `71457946fe0e87bf666cc835c0e2cafbc679b2c903f3369d20a81db7dc18960e` | `checkpoints_v4/baseline_best.pt` |
| **Baseline (Seed 42)** | `42` | `155,608` | `0.4907` | `6debd5f887cc24de01f55f479a358d742c5f4efc554dd5d991e27ed432680b66` | `checkpoints_v4/baseline_seed42_best.pt` |
| **Baseline (Seed 1337)** | `1337` | `155,608` | `0.4900` | `8945227de23d6fa3d07f33834cd4e75261e5cd69ebed20b7581cfc5e9e68ec11` | `checkpoints_v4/baseline_seed1337_best.pt` |
| **Baseline (Seed 2026)** | `2026` | `155,608` | `0.4897` | `7caa9f115356b9881a237533e919cc0fce4a53397dbacc74b00b9c9c796e2b7f` | `checkpoints_v4/baseline_seed2026_best.pt` |
| **LeWM (Best)** | `42` | `169,660` | `0.5898` | `072c15f69789172ab58c5a8ea79a57d0926b7e103d18028d6f46429d4270f667` | `checkpoints_v4/lewm_best.pt` |
| **LeWM (Seed 42)** | `42` | `169,660` | `0.5898` | `e012dec7dc409864509ea71422b5f5287a433b58b832e984e63a913dc7993454` | `checkpoints_v4/lewm_seed42_best.pt` |
| **LeWM (Seed 1337)** | `1337` | `169,660` | `0.5901` | `b1785f246e315a613b49bda9493d924499cfbcedf9a9af9ea9038704458e6f27` | `checkpoints_v4/lewm_seed1337_best.pt` |
| **LeWM (Seed 2026)** | `2026` | `169,660` | `0.5899` | `4d8558ddd8fef1e3f0f68e5afedb22115fe46486196be2bfe4a3f7311cfdd70f` | `checkpoints_v4/lewm_seed2026_best.pt` |
| **H-JEPA (Best)** | `42` | `191,907` | `0.8523` | `8f8afe945b253b1871a1f5abae08b77ad503f39d9fe48966cf8a42202f8aa023` | `checkpoints_v4/hjepa_best.pt` |
| **H-JEPA (Seed 42)** | `42` | `191,907` | `0.8523` | `ac69688eae1f4128719dfce912da47c8ae55cbf2412fc8c10feba53d93f1f442` | `checkpoints_v4/hjepa_seed42_best.pt` |
| **H-JEPA (Seed 1337)** | `1337` | `191,907` | `0.8530` | `2ef324bcb7485c920f0535443a4346a162c7740a02b2849e407e6a21a5622145` | `checkpoints_v4/hjepa_seed1337_best.pt` |
| **H-JEPA (Seed 2026)** | `2026` | `191,907` | `0.8527` | `00bb947e4ffa31ba807b0c53674a6cd359afaa0fa9d6fd42e4f84e1ee2290a40` | `checkpoints_v4/hjepa_seed2026_best.pt` |
| **Reconstructed Compact** | `42` | `39,804` | `0.0178` (dyn) | `18ddc0711856f2ef77e6748846a39568cc5a5aa8750941b493fc024d7f8121c0` | `reconstructed_compact/reconstructed_compact_best.pt` |
| **Reconstructed Supervised** | `42` | `39,804` | `0.0491` (dyn) | `5acdd31180f1e1b65574c6c8c8d6d4263c6570364a9d19b142600d3082c2f2a1` | `reconstructed_compact/reconstructed_supervised_only.pt` |
| **Reconstructed BC Only** | `42` | `39,804` | `2.2916` (dyn) | `2dd8e521cbf2a59cff5e7315d710bfa253ba09ca50f1698621303a650aa78cec` | `reconstructed_compact/reconstructed_bc_only.pt` |

---

## 3. Systematic Execution Registry Across All 9 Evaluation Categories

### Category 1: Behavioral Benchmarks & Information Regimes (`behavior`)
* **Scientific Purpose:** Benchmark navigation success, collision rate, and goal error across three strictly defined information regimes: Track A (Oracle), Track B (Structured egocentric goal), Track C (Pure visual observation).
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/run_behavioral_benchmarks.py
  ```
* **Inputs:** Checkpoints `checkpoints_v4/*`, dataset `splits_v4/held_out_test_benchmark.npz`.
* **Output Artifacts:** [`behavioral_eval_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/behavioral_eval_results.json), [`behavioral_evaluation_report.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/reports/behavioral_evaluation_report.md).
* **Key Quantitative Results:**
  * Track A (Oracle): Success `76.7%`, Collision `13.3%`, Goal Error `0.288` m.
  * Track B (Structured Multi-Seed): Baseline `24.4% \pm 13.9%`, LeWM `26.7% \pm 0.0%`, H-JEPA **`34.4% \pm 6.9%`** (Best seed `40.0%`, Collision `48.3%`, Goal Error `0.490` m).
  * Track C (Observation Only): Baseline `3.3%`, LeWM `10.0%`, H-JEPA **`13.3%`**.

### Category 2: Latent Prediction & Drift Audit (`prediction`)
* **Scientific Purpose:** Audit compounding error drift across multi-step recursive rollouts vs. direct hierarchical macro transitions ($K=5, K=10$).
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/audit_latent_predictions.py
  ```
* **Inputs:** `checkpoints_v4/hjepa_best.pt`, `checkpoints_v4/lewm_best.pt`, `splits_v4/held_out_test_benchmark.npz`.
* **Output Artifacts:** [`latent_prediction_audit.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/behavioral/latent_prediction_audit.json).
* **Key Quantitative Results:**
  * Multi-step unrolling drift in LeWM escalates progressively ($0.027$ m at $K=1 \to 0.260$ m at $K=10$), whereas H-JEPA hierarchical macro prediction bounds drift to **`0.069` m at $K=10$** (a **`3.7x` error reduction**).
  * H-JEPA Macro Predictor Val Dyn MSE: **`0.0178`**.
  * Collision Classification Accuracy: **`96.3%` to `100.0%`**.

### Category 3: Controlled Ablations & Behavioral Necessity (`necessity`)
* **Scientific Purpose:** Test causal necessity of prediction, lookahead, action-conditioning, collision awareness, and temporal hierarchy.
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/run_behavioral_benchmarks.py --ablations
  ```
* **Inputs:** `checkpoints_v4/hjepa_best.pt`.
* **Output Artifacts:** `behavioral_eval_results.json` (`ablations`), [`circuit_registry.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/circuit_registry.json).
* **Key Quantitative Results:**
  * Full Lookahead (`prediction_used`): `36.7%` success, `53.3%` collision, `0.490` m error.
  * Reactive Policy (`reactive`, $K=0$): **`0.0%` success** (absolute collapse from `36.7%` to `0.0%`), `86.7%` collision, `0.741` m error.
  * Prediction Ignored (`prediction_ignored`): **`3.3%` success**, `73.3%` collision, `0.697` m error.
  * No Action Conditioning (`no_action_conditioning`): **`3.3%` success**, `93.3%` collision, `0.781` m error.
  * No Collision Head (`no_collision_prediction`): `10.0%` success, **`76.7%` collision** (**`+23.4%` surge**), `0.587` m error.
  * No Hierarchy ($K=1$ unroll): **`3.3%` success**, `70.0%` collision, `0.630` m error.

### Category 4: Attribution Graphs & Activation Patching (`sufficiency`)
* **Scientific Purpose:** Measure gradient-weighted feature attribution and test sufficiency of targeted subspaces vs. matched random controls.
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/circuits/test_necessity_sufficiency.py
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/circuits/run_activation_patching.py
  ```
* **Inputs:** `checkpoints_v4/*`, `splits_v4/held_out_test_benchmark.npz`, `splits_v4/counterfactual_suite.npz`.
* **Output Artifacts:** [`circuit_registry.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/circuit_registry.json), [`activation_patching_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/activation_patching_results.json).
* **Key Quantitative Results:**
  * C1 Target Grounding: Necessity `0.546` (vs Random `0.279`, Selectivity `+0.267`), Sufficiency `0.305` (vs Random `0.188`, Selectivity `+0.117`).
  * C2 Egocentric Transform: **FALSIFIED_NON_SELECTIVE** (Necessity `0.515` vs Random `0.815`, Selectivity `-0.300`; Sufficiency `0.061` vs Random `0.346`, Selectivity `-0.285`).
  * C3 Collision Detection: Necessity `0.468` (vs Random `0.326`, Selectivity `+0.142`), Sufficiency `0.238` (vs Random `0.314`).
  * C4 Action-Conditioned Dynamics: Necessity `0.580` (vs Random `0.452`, Selectivity `+0.127`), Sufficiency `0.107` (vs Random `0.061`, Selectivity `+0.046`).
  * C5 Hierarchical Subgoal: Necessity `0.447` (vs Random `0.426`, Selectivity `+0.021`), Sufficiency `0.261` (vs Random `0.022`, Selectivity `+0.239`).

### Category 5: Generalization & Counterfactual Invariance (`generalization`)
* **Scientific Purpose:** Evaluate closed-loop retention under E1–E7 counterfactuals and visual/spatial disambiguation tasks.
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/run_generalization_benchmarks.py
  ```
* **Inputs:** `checkpoints_v4/*`, `splits_v4/counterfactual_suite.npz`, `splits_v4/disambiguation_suite.npz`.
* **Output Artifacts:** [`generalization_eval_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/generalization_eval_results.json).
* **Key Quantitative Results:**
  * E3 Robot Yaw Rotation: H-JEPA achieved **`100.0%` retention** (`100.0%` success, `0.240` m error).
  * DISAMBIG_03 (Geometric vs Visual Proximity): H-JEPA achieved **`100.0%` success** (`0.237` m error) while Oracle, Baseline, LeWM achieved `0.0%`.
  * E2 Color Invariance: H-JEPA achieved `50.0%` success with `0.490` m error.

### Category 6: Physical Invariance & Robustness (`transfer`)
* **Scientific Purpose:** Measure latent Lie-group equivariance under appearance/kinematic shifts and cross-model causal transfer.
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/principles/evaluate_invariants.py
  .venv/bin/python world_models_fpga/research/principles/test_causal_transfer.py
  ```
* **Inputs:** `checkpoints_v4/*`, `splits_v4/counterfactual_suite.npz`, `splits_v4/held_out_test_benchmark.npz`.
* **Output Artifacts:** [`invariant_evaluation_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/principles/results/invariant_evaluation_results.json), [`causal_transfer_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/principles/results/causal_transfer_results.json).
* **Key Quantitative Results:**
  * Invariance Cosine: H-JEPA $\ge 0.997$ across all E1–E7 counterfactuals (E1 `0.9971`, E2 `0.9998`, E3 `0.9985`, E4 `0.9975`, E5 `0.9975`, E6 `0.9996`, E7 `0.9997`).
  * Equivariance Held-Out Cosine: $\ge 0.999$ across all E1–E7 counterfactuals.
  * Multi-Horizon Latent Stability: H-JEPA maintains `0.9964` cosine at $K=8$, whereas single-step LeWM unrolled MSE collapses from `0.9659` to `0.6110`.
  * Cross-Model Decodability vs Behavior: Linear probe decodability negatively correlates with behavioral success across models ($r = -0.925$ for C1, $r = -0.991$ for C4), confirming H2.

### Category 7: Novel Compositional Tasks (`compositionality`)
* **Scientific Purpose:** Test zero-shot compositional recombination on non-trivial spatial tasks: Relational Detour (T2) and Narrow Corridor (T4).
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/synthesis/experiment_c_novel_composition/run_novel_composition.py
  ```
* **Inputs:** `checkpoints_v4/hjepa_best.pt`, `amr_arena.xml` with canonical obstacles.
* **Output Artifacts:** [`novel_composition_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/synthesis/experiment_c_novel_composition/novel_composition_results.json).
* **Key Quantitative Results:**
  * Task T4 (Narrow Corridor Traversal): H-JEPA achieved **`90.0%` success** and **`100.0%` stage completion** in Track B, outperforming LeWM (`60.0%`) and Baseline (`20.0%`).
  * Task T3 (Three-Stage Sequential Patrol): H-JEPA achieved **`20.0%` success and `73.3%` stage completion** with `0.0%` collision rate (the only agent to achieve completed missions).

### Category 8: Architecture Extraction & Distillation (`architecture_reconstruction`)
* **Scientific Purpose:** Extract a compact 39,804-parameter model via distillation, evaluate 3-regime distillation ablations, and test closed-loop MuJoCo navigation.
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/extraction/train_reconstructed_model.py
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/extraction/comparison/compare_three_models.py
  ```
* **Inputs:** Teacher `checkpoints_v4/hjepa_best.pt`, `splits_v4/train_benchmark.npz`, `splits_v4/held_out_test_benchmark.npz`.
* **Output Artifacts:** [`three_model_comparison.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/extraction/comparison/three_model_comparison.json), [`training_log.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/extraction/extracted_model/reconstructed_compact/training_log.json).
* **Key Quantitative Results:**
  * Parameter Compression: $191,907 \to 39,804$ parameters (**$79.3\%$ reduction**).
  * Distilled Compact 16D: **`46.7%` success**, `26.7%` collision, `0.515` m error, `0.0178` val dyn MSE (**$3.5\times$ higher success** than supervised compact without teacher at `13.3%`).
  * Supervised Compact (No Teacher): `13.3%` success, `33.3%` collision, `0.768` m error, `0.0491` val dyn MSE ($2.8\times$ worse dynamics).
  * BC Only Compact: `53.3%` success on simple paths, but **`2.2916` val dyn MSE** (completely unlearned latent dynamics).
  * Original H-JEPA Teacher: `73.3%` success, `26.7%` collision, `0.350` m error.

### Category 9: Cycle-Exact RTL Simulation & Hardware Benchmarks (`hardware_equivalence`)
* **Scientific Purpose:** Verify bit-exact equivalence between synthesizable Verilog RTL and Python Q8.8 fixed-point; measure out-of-context Vivado synthesis on XCK26; benchmark physical KR260 SOM silicon.
* **Execution Command:**
  ```bash
  .venv/bin/python world_models_fpga/research/hardware/validate_hardware_equivalence.py
  .venv/bin/python world_models_fpga/research/hardware/benchmark_fair_comparison.py
  ```
* **Inputs:** `q88_exported_weights.npz`, RTL files in `world_models_fpga/fpga/rtl/`.
* **Output Artifacts:** [`hardware_equivalence_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/hardware_equivalence_results.json), [`benchmarks_heterogeneous_platforms.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/benchmarks_heterogeneous_platforms.json), [`hardware_validation_report.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/reports/hardware_validation_report.md).
* **Key Quantitative Results:**
  * Relative Arithmetic Error: **`E_{\text{rel}} = 0.000000` (100.0% Bit-Exact across all 20 output words)**.
  * Action Direction Cosine Similarity: **`0.9997`**; Safety Barrier Concordance: **`100.00%`**.
  * Vivado 2023.2 Post-Synthesis (`xck26`): `3,579` LUTs (`3.06%`), `13` DSPs (`1.04%`), `0` BRAM, `0.365` W power.
  * RTL Latency: `4,327` cycles (`0.0216` ms @ 200 MHz) single-pass; `54,467` cycles (`0.2723` ms @ 200 MHz) 24-MPC pass.
  * Speedup over Kria KR260 ARM Cortex-A53 PS (`19.197` ms): **`70.49x`** (@ 200 MHz) / **`44.06x`** (@ 125 MHz).

---

## 4. Master Provenance Verification

All 9 required evaluation categories are registered in the machine-readable manifest:
[`world_models_fpga/research/validation/experiment_manifest.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/validation/experiment_manifest.json).

```json
{
  "schema_version": "2.0",
  "provenance_enforced": true,
  "independently_executed_evaluations": 9,
  "legacy_unverified_evaluations": 0,
  "missing_evaluations": 0,
  "rejected_records": 0,
  "executed_evaluations": [
    "architecture_reconstruction",
    "behavior",
    "compositionality",
    "generalization",
    "hardware_equivalence",
    "necessity",
    "prediction",
    "sufficiency",
    "transfer"
  ],
  "conclusion": "All 9 required evaluation categories independently executed and provenance-verified."
}
```
Every recorded entry contains explicit `git_commit`, `dataset_sha256`, `checkpoint_sha256`, `environment`, and complete raw metric dictionaries. Zero hardcoded constants or synthetic random timing distributions exist in any output.
Every recorded entry contains explicit `git_commit`, `dataset_sha256`, `checkpoint_sha256`, `environment`, and complete raw metric dictionaries. Zero hardcoded constants or synthetic random timing distributions exist in any output.
