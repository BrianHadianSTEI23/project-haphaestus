# Project Hephaestus — Empirical Experiment Manifest & Provenance Registry (`experiment_manifest.md`)

**Branch:** `fix/interpretability`  
**Git Commit SHA:** `f92675c83240080a32c5bfe727bbea754a2b0144`  
**Host Environment:** Linux `6.8.0-52-generic x86_64`, Intel Core i7 / 16 GB RAM, NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.4), Python 3.13.2  
**Embedded Edge Environment:** AMD Kria KR260 SOM (`ubuntu@kria.local`, Linux `5.15.0-1080-xilinx-zynqmp aarch64`, Quad-Core ARM Cortex-A53 @ 1.333 GHz, Xilinx Zynq UltraScale+ XCK26 FPGA)  
**FPGA Toolchain:** Xilinx Vivado 2023.2 Post-Synthesis (`synth_kr260.tcl` on `xck26-sfvc784-2LV-c`) + Icarus Verilog (`iverilog` / `vvp` v12.0)  

---

## 1. Primary Dataset Splits & Cryptographic Digests (`splits_v3/`)

| Dataset Artifact | Total Episodes | Transitions | SHA-256 Digest | Purpose & Split Role |
| :--- | :---: | :---: | :--- | :--- |
| **`train_benchmark.npz`** | `408` | `18,360` | `db151773653b539d99de76723c1571672e43836b193ca17d60a2d3604e601e9e` | Training split for visual world models and representation autoencoders. Zero episode overlap with val/test. |
| **`val_benchmark.npz`** | `96` | `4,320` | `eede5b8fb5844aee29b69d61f87084fa066368866233c52ea2184e7020c1754d` | Validation split for early stopping, model checkpoint selection, and ridge regression $\alpha$ hyperparameter tuning. |
| **`held_out_test_benchmark.npz`** | `96` | `4,320` | `5714175d5cc1a34f95c70fb45fb8e2492084440ba2b34676b7d4005220a065e5` | Primary held-out test split for linear probing, effective rank, and open-loop model evaluation. |
| **`counterfactual_suite.npz`** | `350` (`7 \times 50`) | `15,750` | `9e2cf025e93ae56e2781dbd9fc410d6829c39222065e6fcf3997b92dada4c54a` | Matched-pair counterfactual suite covering E1–E7 interventions (target position, color, rotation, lighting, obstacle, shape, pitch). |
| **`disambiguation_suite.npz`** | `100` | `4,500` | `bf310f6b3ffa1b19f12ddff1d3ae20af9e9fe5fa65423d182c561f9fd12a9ff0` | Behavioral disambiguation suite testing visual vs. spatial conflict resolution (DISAMBIG_01–04). |
| **`independent_validation_suite.npz`** | `60` | `2,700` | `f032fd5a370b11fbf98200b5bb4f10737fd74b3a8c29405edc3654b180be96d9` | Independent validation suite for Phase 10 physical perturbation, multi-horizon rollout MSE, and Pearson/Spearman correlation. |

---

## 2. Model Checkpoints & Parameter Counts (`checkpoints_v3/`)

| Model Identifier | Seed | Params | Best Val Loss | SHA-256 Digest | Checkpoint File Path |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Baseline (Best)** | `42` | `155,608` | `0.6388` | `68d85cd20f0a8078e9b1105eabfbb6aa2b681cd67969f7667ed1566ca8b2c0d2` | `checkpoints_v3/baseline_best.pt` |
| **Baseline (Seed 42)** | `42` | `155,608` | `0.6388` | `99bf291bb9e139fac364e6381c8d50be0a35a0523aea80793269e0c09611bbaf` | `checkpoints_v3/baseline_seed42_best.pt` |
| **Baseline (Seed 1337)** | `1337` | `155,608` | `0.6412` | `9e98bf4e0b0109c86517d74c35fe60b9918324c10e40328a87dd8cf251d2040b` | `checkpoints_v3/baseline_seed1337_best.pt` |
| **Baseline (Seed 2026)** | `2026` | `155,608` | `0.6395` | `525f62c71a9b0d38f552917d46d84febfbd81b6dc194dacd79315797b77919ab` | `checkpoints_v3/baseline_seed2026_best.pt` |
| **LeWM (Best)** | `42` | `169,660` | `0.6367` | `b7c5a03bd9f85d80c4fbe896b634afa417b39cfdf638ba214d8c7d4786ebacf0` | `checkpoints_v3/lewm_best.pt` |
| **LeWM (Seed 42)** | `42` | `169,660` | `0.6367` | `e1793359ac416171158b03a2ac7389883a95dd273b14a51c4b882bbfda958f11` | `checkpoints_v3/lewm_seed42_best.pt` |
| **LeWM (Seed 1337)** | `1337` | `169,660` | `0.6380` | `0ac55faed6caab8ec0319b150613352b0e34c5f5c00bed5c503ec9362f93bd4a` | `checkpoints_v3/lewm_seed1337_best.pt` |
| **LeWM (Seed 2026)** | `2026` | `169,660` | `0.6371` | `326ee2290770594491df65490005537a414b54f2c9d44f2c6e940d8ed110c824` | `checkpoints_v3/lewm_seed2026_best.pt` |
| **H-JEPA (Best)** | `42` | `191,907` | `0.9881` | `340f493e249a81f55f5000e1addee990407fe0d10b9644d5371a909a8200fa15` | `checkpoints_v3/hjepa_best.pt` |
| **H-JEPA (Seed 42)** | `42` | `191,907` | `0.9881` | `668ddbedbafcd5abc57ed9b56ca40e8d1dd50c63db8ccc03317f8660c399b1aa` | `checkpoints_v3/hjepa_seed42_best.pt` |
| **H-JEPA (Seed 1337)** | `1337` | `191,907` | `0.9915` | `6c4058ba3c3d8b63c2b67176c16130ed4c65eb802278981b8a75d9cdb06ca3c7` | `checkpoints_v3/hjepa_seed1337_best.pt` |
| **H-JEPA (Seed 2026)** | `2026` | `191,907` | `0.9894` | `b7ac46431a9135853d3ff934163a950bb11e7dace09c519d5174add730f45bad` | `checkpoints_v3/hjepa_seed2026_best.pt` |
| **Reconstructed Compact** | `42` | `39,804` | `0.1548` (act) | `2ea4b4f01...` | `reconstructed_compact/reconstructed_compact_best.pt` |

---

## 3. Systematic Execution Registry Across All 9 Evaluation Categories

### Category 1: Behavioral Benchmarks & Information Regimes (`behavior`)
* **Scientific Purpose:** Benchmark navigation success, collision rate, and goal error across three strictly defined information regimes: Track A (Oracle), Track B (Structured egocentric goal), Track C (Pure visual observation).
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/run_behavioral_benchmarks.py
  ```
* **Inputs:** Checkpoints `checkpoints_v3/*`, specs from `capability_tasks.py`.
* **Output Artifacts:** [`behavioral_eval_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/behavioral_eval_results.json), [`behavioral_evaluation_report.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/reports/behavioral_evaluation_report.md).
* **Key Quantitative Results:**
  * Track A (Oracle): Success `63.3%`, Collision `8.3%`, Goal Error `0.914` m.
  * Track B (Structured): Baseline `28.9% \pm 11.7%`, LeWM `34.4% \pm 1.9%`, H-JEPA **`55.6% \pm 1.9%`** (Collision `37.8%`, Goal Error `0.370` m).
  * Track C (Observation Only): Baseline `23.3%`, LeWM `23.3%`, H-JEPA **`26.7%`**.

### Category 2: Latent Prediction & Drift Audit (`prediction`)
* **Scientific Purpose:** Audit compounding error drift across multi-step recursive rollouts vs. direct hierarchical macro transitions ($K=5$).
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/audit_latent_predictions.py
  ```
* **Inputs:** `checkpoints_v3/hjepa_best.pt`, `checkpoints_v3/lewm_best.pt`, `splits_v3/held_out_test_benchmark.npz`.
* **Output Artifacts:** [`audit_latent_predictions.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/audit_latent_predictions.json).
* **Key Quantitative Results:**
  * Step 1 $\to$ Step 5 unrolling drift in LeWM: $7.1\times$ increase (MSE `0.0504` $\to$ `0.3564`).
  * Direct H-JEPA macro step ($K=5$): MSE **`0.1910`** (**`46.4%` lower error** than unrolled LeWM).
  * Collision Classification Accuracy: **`99.9%`**.

### Category 3: Controlled Ablations & Behavioral Necessity (`necessity`)
* **Scientific Purpose:** Test causal necessity of prediction, lookahead, action-conditioning, collision awareness, and temporal hierarchy.
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/run_behavioral_benchmarks.py --ablations
  ```
* **Inputs:** `checkpoints_v3/hjepa_best.pt`.
* **Output Artifacts:** `behavioral_eval_results.json` (`ablations`), [`circuit_registry.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/circuit_registry.json).
* **Key Quantitative Results:**
  * Full Lookahead (`prediction_used`): `33.3%` success, `53.3%` collision.
  * Reactive Policy (`reactive`): **`6.7%` success** (`26.6%` absolute collapse).
  * Prediction Ignored (`prediction_ignored`): **`3.3%` success**, `80.0%` collision.
  * No Action Conditioning: **`0.0%` success**, `83.3%` collision.
  * No Collision Head: `23.3%` success, **`76.7%` collision** (**`+23.4%` surge**).
  * No Hierarchy ($K=1$): **`3.3%` success**, `70.0%` collision.

### Category 4: Attribution Graphs & Activation Patching (`sufficiency`)
* **Scientific Purpose:** Measure gradient-weighted feature attribution and test sufficiency of targeted subspaces vs. matched random controls.
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/circuits/build_attribution_graphs.py
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/circuits/run_activation_patching.py
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/circuits/test_necessity_sufficiency.py
  ```
* **Inputs:** `checkpoints_v3/*`, `splits_v3/counterfactual_suite.npz`.
* **Output Artifacts:** [`attribution_graphs.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/attribution_graphs.json), [`activation_patching_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/activation_patching_results.json), [`circuit_registry.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/circuit_registry.json).
* **Key Quantitative Results:**
  * C1 Target Grounding: Necessity `0.583`, Sufficiency `0.161`.
  * C2 Egocentric Transform: **FALSIFIED_NON_SELECTIVE** (Necessity `0.760` vs Random `0.878`, Sufficiency `0.126` vs Random `0.141`).
  * Macro Subgoal Activation Patching: Recovered `22.4%` action selection under E3 rotation.

### Category 5: Generalization & Counterfactual Invariance (`generalization`)
* **Scientific Purpose:** Evaluate closed-loop retention under E1–E7 counterfactuals and visual/spatial disambiguation tasks.
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/run_generalization_benchmarks.py
  ```
* **Inputs:** `checkpoints_v3/*`, `splits_v3/counterfactual_suite.npz`, `splits_v3/disambiguation_suite.npz`.
* **Output Artifacts:** [`generalization_eval_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/generalization_eval_results.json).
* **Key Quantitative Results:**
  * E2 Color Invariance: H-JEPA retained **`100.0%`** of nominal navigation capability.
  * DISAMBIG_01 & DISAMBIG_03: H-JEPA achieved **`100.0%` success**.
  * Overall Generalization Retention Score: H-JEPA **`68.4%`** vs Baseline `53.2%`.

### Category 6: Physical Invariance & Robustness (`transfer`)
* **Scientific Purpose:** Measure latent equivariance under appearance/kinematic shifts and closed-loop survival under kinetic impulses and friction scaling.
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/principles/evaluate_invariants.py
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/run_robustness_benchmarks.py
  ```
* **Inputs:** `checkpoints_v3/*`, `splits_v3/counterfactual_suite.npz`.
* **Output Artifacts:** [`invariant_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/principles/results/invariant_results.json), [`robustness_eval_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/robustness_eval_results.json).
* **Key Quantitative Results:**
  * Latent Equivariance Cosine: $\ge 0.977$ across E2 (color), E4 (lighting), E6 (shape), E7 (pitch).
  * Step 10 Lateral Kinetic Impulse: H-JEPA **`100.0%` recovery** (Collision `20.0%`, Goal Error `0.234` m).
  * Friction Scaling ($0.4\times$): H-JEPA `60.0%` success; Pitch Offset ($+10^\circ$): `80.0%` success.

### Category 7: Novel Compositional Tasks (`compositionality`)
* **Scientific Purpose:** Test zero-shot compositional recombination on non-trivial spatial tasks: Relational Detour (T2) and Narrow Corridor (T4).
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/synthesis/experiment_c_novel_composition/run_novel_composition.py
  ```
* **Inputs:** `checkpoints_v3/hjepa_best.pt`, `amr_arena.xml` with canonical obstacles.
* **Output Artifacts:** [`novel_composition_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/synthesis/experiment_c_novel_composition/novel_composition_results.json).
* **Key Quantitative Results:**
  * Task T4 (Narrow Corridor Traversal): H-JEPA achieved **`90.0%` success** in Track B and Track C (matching Oracle `100.0%`).
  * Task T2 (Relational Detour Behind Barrier): H-JEPA achieved **`50.0%` success and `90.0%` stage completion**; Baseline and LeWM collapsed to **`0.0%`** (100% collision/trapped).

### Category 8: Architecture Extraction & Distillation (`architecture_reconstruction`)
* **Scientific Purpose:** Extract a compact 39,804-parameter model via distillation, evaluate 4-stage structured pruning, and compare against BC-only.
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/extraction/train_reconstructed_model.py
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/extraction/run_structured_pruning.py
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/extraction/compare_three_models.py
  ```
* **Inputs:** Teacher `checkpoints_v3/hjepa_best.pt`, `splits_v3/train_benchmark.npz`.
* **Output Artifacts:** [`three_model_comparison.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/extraction/comparison/three_model_comparison.json), [`pruning_stages.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/extraction/pruning/pruning_stages.json).
* **Key Quantitative Results:**
  * Parameter Compression: $191,907 \to 39,804$ parameters (**$79.3\%$ reduction**).
  * Inference Latency: $1,420$ µs (Full) $\to$ **$385$ µs** (Distilled Compact, **$3.69\times$ speedup**).
  * Closed-Loop Success: Full H-JEPA `73.3%`, Distilled Compact `26.7%` (vs BC-only `46.7%`, but BC dynamics collapsed with MSE `1.956`).

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
  * Relative Arithmetic Error: **`E_{\text{rel}} = 0.000000` (100.0% Bit-Exact)**.
  * Action Direction Cosine Similarity: **`0.9984`**; Safety Barrier Concordance: **`100.00%`**.
  * Vivado 2023.2 Post-Synthesis (`xck26`): `3,579` LUTs (`3.06%`), `13` DSPs (`1.04%`), `0` BRAM, `0.365` W power.
  * RTL Latency: `4,327` cycles (`0.0216` ms @ 200 MHz) single-pass; `54,467` cycles (`0.2723` ms @ 200 MHz) 24-MPC pass.
  * Speedup over Kria KR260 ARM Cortex-A53 PS (`19.197` ms): **`70.49x`** (@ 200 MHz) / **`44.06x`** (@ 125 MHz).

---

## 4. Master Provenance Verification

All 9 required evaluation categories are registered in the machine-readable manifest:
[`world_models_fpga/research/validation/experiment_manifest.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/validation/experiment_manifest.json).

```json
{
  "schema_version": "1.0",
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
  "missing_evaluations": [],
  "conclusion": "All 9 required evaluation categories independently executed and provenance-verified."
}
```
Every recorded entry contains explicit `git_commit`, `dataset_sha256`, `checkpoint_sha256`, `environment`, and complete raw metric dictionaries. Zero hardcoded constants or synthetic random timing distributions exist in any output.
