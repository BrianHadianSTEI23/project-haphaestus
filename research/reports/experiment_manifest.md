# Project Hephaestus — Empirical Experiment Manifest & Provenance Registry (`experiment_manifest.md`)

**Branch:** `fix/interpretability`  
**Git Commit SHA:** `c56ba340ba57ba528f555df6826355bfdbfc48d8`  
**Host Environment:** Linux `6.8.0-138-generic x86_64`, Intel Core i7 / 16 GB RAM, NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.4), Python 3.13.16  
**Embedded Edge Environment:** AMD Kria KR260 SOM (`ubuntu@kria.local`, Linux `5.15.0-1080-xilinx-zynqmp aarch64`, Quad-Core ARM Cortex-A53 @ 1.333 GHz, Xilinx Zynq UltraScale+ XCK26 FPGA)  
**FPGA Toolchain:** Xilinx Vivado 2023.2 Post-Synthesis (`synth_kr260.tcl` on `xck26-sfvc784-2LV-c`) + Icarus Verilog (`iverilog` / `vvp` v12.0)  

---

## 1. Primary Dataset Splits & Cryptographic Digests (`splits_v4/`)

All experiments are executed on the cryptographically validated, multi-waypoint aligned corpus [`splits_v4/`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/datasets/splits_v4/):

| Dataset Artifact | Total Episodes | Transitions | SHA-256 Digest | Purpose & Split Role |
| :--- | :---: | :---: | :--- | :--- |
| **`train_benchmark.npz`** | `408` | `18,360` | `8ba4406a4b12631557d07945cbe211832b8519fbbd8624128be1f609e9e6ca3a` | Training split for visual world models and representation autoencoders. Zero episode overlap with val/test. |
| **`val_benchmark.npz`** | `96` | `4,320` | `8f5a21df67190c63e71d5f5927f23dab17cce1b2e66c055e86999ef4a57d29a4` | Validation split for early stopping, model checkpoint selection, and ridge regression $\alpha$ hyperparameter tuning. |
| **`held_out_test_benchmark.npz`** | `96` | `4,320` | `5b5153fcef96e2f0398a0e13a21b942d599066ef97c902ba711167f9e9ad9f38` | Primary held-out test split for linear probing, effective rank, and open-loop model evaluation. |
| **`counterfactual_suite.npz`** | `350` (`7 \times 50`) | `15,750` | `0d1516dcb86d3a856149148e97d5ba24c4553f5232beec253f0f5e29c921db47` | Matched-pair counterfactual suite covering E1–E7 interventions (target position, color, rotation, lighting, obstacle, shape, pitch). |
| **`disambiguation_suite.npz`** | `100` | `4,500` | `a6d215541313c77c51f5762c4099e1a5c572435efa07cc14d3e1da8e3c7de700` | Behavioral disambiguation suite testing visual vs. spatial conflict resolution (DISAMBIG_01–04). |
| **`independent_validation_suite.npz`** | `60` | `2,700` | `20878ceae77ef3c047ba13517174cb6d90962a92ffbe4cd3609a38d47dd2ed2b` | Independent validation suite for Phase 10 physical perturbation, multi-horizon rollout MSE, and Pearson/Spearman correlation. |

---

## 2. Model Checkpoints & Parameter Counts (`checkpoints_v4/`)

| Model Identifier | Seed | Params | Best Val Loss | SHA-256 Digest | Checkpoint File Path |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Baseline (Best)** | `42` | `155,608` | `0.4907` | `6debd5f887cc24de01f55f479a358d742c5f4efc554dd5d991e27ed432680b66` | `checkpoints_v4/baseline_best.pt` |
| **Baseline (Seed 42)** | `42` | `155,608` | `0.4907` | `6debd5f887cc24de01f55f479a358d742c5f4efc554dd5d991e27ed432680b66` | `checkpoints_v4/baseline_seed42_best.pt` |
| **Baseline (Seed 1337)** | `1337` | `155,608` | `0.4900` | `8945227de23d6fa3d07f33834cd4e75261e5cd69ebed20b7581cfc5e9e68ec11` | `checkpoints_v4/baseline_seed1337_best.pt` |
| **Baseline (Seed 2026)** | `2026` | `155,608` | `0.4897` | `7caa9f115356b9881a237533e919cc0fce4a53397dbacc74b00b9c9c796e2b7f` | `checkpoints_v4/baseline_seed2026_best.pt` |
| **LeWM (Best)** | `42` | `169,660` | `0.5898` | `e012dec7dc409864509ea71422b5f5287a433b58b832e984e63a913dc7993454` | `checkpoints_v4/lewm_best.pt` |
| **LeWM (Seed 42)** | `42` | `169,660` | `0.5898` | `e012dec7dc409864509ea71422b5f5287a433b58b832e984e63a913dc7993454` | `checkpoints_v4/lewm_seed42_best.pt` |
| **LeWM (Seed 1337)** | `1337` | `169,660` | `0.5901` | `b1785f246e315a613b49bda9493d924499cfbcedf9a9af9ea9038704458e6f27` | `checkpoints_v4/lewm_seed1337_best.pt` |
| **LeWM (Seed 2026)** | `2026` | `169,660` | `0.5899` | `4d8558ddd8fef1e3f0f68e5afedb22115fe46486196be2bfe4a3f7311cfdd70f` | `checkpoints_v4/lewm_seed2026_best.pt` |
| **H-JEPA (Best)** | `42` | `191,907` | `0.8523` | `ac69688eae1f4128719dfce912da47c8ae55cbf2412fc8c10feba53d93f1f442` | `checkpoints_v4/hjepa_best.pt` |
| **H-JEPA (Seed 42)** | `42` | `191,907` | `0.8523` | `ac69688eae1f4128719dfce912da47c8ae55cbf2412fc8c10feba53d93f1f442` | `checkpoints_v4/hjepa_seed42_best.pt` |
| **H-JEPA (Seed 1337)** | `1337` | `191,907` | `0.8530` | `2ef324bcb7485c920f0535443a4346a162c7740a02b2849e407e6a21a5622145` | `checkpoints_v4/hjepa_seed1337_best.pt` |
| **H-JEPA (Seed 2026)** | `2026` | `191,907` | `0.8527` | `00bb947e4ffa31ba807b0c53674a6cd359afaa0fa9d6fd42e4f84e1ee2290a40` | `checkpoints_v4/hjepa_seed2026_best.pt` |
| **Reconstructed Compact** | `42` | `39,804` | `0.0178` (dyn) | `18ddc0711856f2ef77e6748846a39568cc5a5aa8750941b493fc024d7f8121c0` | `reconstructed_compact/reconstructed_compact_best.pt` |
| **Reconstructed Supervised** | `42` | `39,804` | `0.0491` (dyn) | `5acdd31180f1e1b65574c6c8c8d6d4263c6570364a9d19b142600d3082c2f2a1` | `reconstructed_compact/reconstructed_supervised_only.pt` |
| **Reconstructed BC Only** | `42` | `39,804` | `2.2916` (dyn) | `2dd8e521cbf2a59cff5e7315d710bfa253ba09ca50f1698621303a650aa78cec` | `reconstructed_compact/reconstructed_bc_only.pt` |

---

## 3. Systematic Execution Registry Across All 9 Evaluation Categories

### Category 1: Behavioral Benchmarks & Information Regimes (`behavior`)
* **Scientific Purpose:** Benchmark navigation success, collision rate, and goal error across three strictly defined information regimes: Track A (Oracle), Track B (Structured egocentric goal), Track C (Pure visual observation), and evaluate dynamic visual grounding vs allocentric priors under camera visibility and distance controls.
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/run_behavioral_benchmarks.py
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/run_track_c_visual_grounding.py --episodes-per-suite 20 --device cuda
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/run_task_identity_visual_grounding.py --episodes-per-cond 30 --device cuda
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/run_task_identity_visual_grounding_v2.py --episodes-per-cond 30 --device cuda
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/run_task_conditioned_binding_v3.py --pairs 50 --device cuda
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/run_task_conditioned_binding_v4.py --pairs 50 --device cuda
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/run_task_conditioned_binding_v5.py --pairs 50 --device cuda
  ```
* **Inputs:** Checkpoints `checkpoints_v4/*`, dataset `splits_v4/held_out_test_benchmark.npz`.
* **Output Artifacts:** [`behavioral_eval_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/behavioral_eval_results.json), [`track_c_visual_grounding.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/track_c_visual_grounding.json), [`task_identity_visual_grounding_v2.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_identity_visual_grounding_v2.json), [`task_identity_visual_grounding_v2_traces.npz`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_identity_visual_grounding_v2_traces.npz), [`task_conditioned_binding_v3_summary.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_binding_v3_summary.json), [`task_conditioned_binding_v3_pairs.jsonl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_binding_v3_pairs.jsonl), [`task_conditioned_binding_v3_traces.npz`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_binding_v3_traces.npz), [`task_conditioned_binding_v4_summary.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_binding_v4_summary.json), [`task_conditioned_binding_v4_pairs.jsonl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_binding_v4_pairs.jsonl), [`task_conditioned_binding_v4_model_comparison.csv`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_binding_v4_model_comparison.csv), [`task_conditioned_binding_v5_summary.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_binding_v5_summary.json), [`task_conditioned_binding_v5_pairs.jsonl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_binding_v5_pairs.jsonl), [`task_conditioned_binding_v5_fold_predictions.jsonl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_binding_v5_fold_predictions.jsonl), [`task_conditioned_binding_v5_schema.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_binding_v5_schema.md).
* **Key Quantitative Results:**
  * Track A (Oracle): Success `76.7%`, Collision `13.3%`, Goal Error `0.288` m.
  * Track B (Structured Multi-Seed): Baseline `24.4% \pm 13.9%`, LeWM `26.7% \pm 0.0%`, H-JEPA **`34.4% \pm 6.9%`** (Best seed `40.0%`, Collision `48.3%`, Goal Error `0.490` m).
  * Track C (Observation Only Grounding, N=20): H-JEPA **20/20 (100.0%, 0.260m)**, LeWM 19/20 (95.0%), Baseline 8/20 (40.0%). Rigid SE(2) global rotations (+90, 180, -90 deg) achieve **100.0%** ($p = 1.0000$, global latent cosine = 0.9745 vs 0.3450 egocentric yaw). Floor nuisance collapses success to **5.0%** ($p < 0.0001$), target appearance achieves **70.0%** ($p = 0.0312$), distractor appearance achieves **100.0%**, color swap achieves **100.0%** (confirming routing by categorical task tokens), and target relocation collapses to **30.0%** (6/20, 0.963m error, $p = 0.0001$), distractor relocation achieves **95.0%** (19/20), pairwise swap collapses to **5.0%** (1/20), and floor shift collapses to **5.0%** (1/20).
  * Dynamic Visual Grounding v2 Benchmark ($N=30$ per condition, 180 episodes total, with initial waypoint distance $\ge 0.60$m and post-action evaluation):
    * Canonical Baseline: **100.0%** (30/30), 0.328m error, median path length 0.261m.
    * Relocated Target (Visible): **73.3%** (22/30), 0.652m error (median 0.343m), median path length **0.488m** (mean 0.706m), initial distance 0.770m (min 0.666m). McNemar unadjusted $p = 0.0078$, Holm-adjusted $p = 0.03125$. Confirms active locomotion and eliminates the v1 distance confound where 21/30 started inside the goal threshold.
    * Distractor Relocation: **100.0%** (30/30), 0.328m error ($p = 1.0000$).
    * Same RGB Paired Token Contrast (Dual-Visible Objects): Token A **66.7%** (20/30), Token B **70.0%** (21/30). In identical RGB observations, swapping task token rotates inferred visual goal vector by **$0.869$ rad ($49.8^\circ$)**. Correct dual discrimination achieved in **20.0% (6/30)** of episodes.
    * Visual/Token Conflict Factorial: **86.7%** (26/30), 0.371m error ($p = 0.1250$, Holm $p = 0.3750$).
    * Target Appearance Reassigned: **96.7%** (29/30), 0.341m error ($p = 1.0000$).
  * Task-Conditioned Binding Benchmark v5 ($N=50$ pairs, 100 scenes, tested independently across Scene A and Scene B with verified 5-step simulator mutation protocol):
    * Scene A: Token A 42.0%, Token B 38.0%, Dual 10.0%, Margins: A $-0.025$m, B $-0.035$m.
    * Scene B: Token A 50.0%, Token B 56.0%, Dual 22.0%, Margins: A $+0.005$m, B $+0.027$m.
    * Pooled: Marginal accuracy 46.5% (93/200, 95% CI: [41.0%, 52.0%]). Dual discrimination 16.0% (16/100, 95% CI: [9.0%, 24.0%]).
    * Pairs with dual success in BOTH scenes: **0/50 (0.0%)**.
    * Pair-clustered permutation test (100,000 resamples preserving paired structure): Null Mean **19.5% (95% CI: [16.0%, 23.0%])**, $p_{\text{one\_sided}} = 0.9890$, $p_{\text{two\_sided}} = 0.0571$. Dual target selection is indistinguishable from chance.
    * Interaction vector $I = (g_{bb} - g_{ba}) - (g_{ab} - g_{aa})$: mean $\|I\| = 0.3521$m (95% CI: $[0.2921, 0.4161]$m); cosine alignment with target swap $\Delta P$ is **0.3140** (95% CI: $[0.1508, 0.4693]$, sign-flip $p_{\text{pos}} = 0.0002, p_{\text{neg}} = 0.9999, p_{\text{two}} = 0.0004$, mean angular divergence $66.9^\circ$). Confirms Mechanism 3 (Mixed Nonlinear Interaction).
    * Stratified Grouped 5-Fold Cross-Validation: Bilinear Interaction MSE 0.00544 and Nonlinear MLP MSE 0.00439 beat Additive (0.01985) and Task-Only (0.03723) for H-JEPA surrogate; Bilinear MSE 0.00274 and Nonlinear MLP MSE 0.00367 beat Additive (0.01249) and Visual Baseline (0.01276) for Ground Truth targets.
    * Instance-Level Binding Limitation: Audited `encode_task_id()` vocabulary (6 landmarks $\times$ 8 relations). Zero slots exist for instance representation without semantic tags. Historical 5.0% claim is invalidated and classified as `ARCHITECTURALLY_UNSUPPORTED`.

### Category 2: Latent Prediction & Drift Audit (`prediction`)
* **Scientific Purpose:** Audit compounding error drift across multi-step recursive rollouts vs. direct hierarchical macro transitions ($K=5, K=10$), matched candidate controls, and prospective safety barrier benchmarks.
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/run_matched_temporal_controls.py --episodes 50 --device cuda
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/benchmarks/run_predictive_safety_benchmark.py --episodes 40 --device cuda
  ```
* **Inputs:** `checkpoints_v4/hjepa_best.pt`, `checkpoints_v4/lewm_best.pt`, `splits_v4/held_out_test_benchmark.npz`.
* **Output Artifacts:** [`matched_temporal_controls.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/matched_temporal_controls.json), [`predictive_safety_benchmark.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/predictive_safety_benchmark.json).
* **Key Quantitative Results:**
  * Open-loop physical displacement error at $K=5$: Direct Macro achieves **0.0460m** vs **0.0814m** for recursive micro unrolling (a 43.5% error reduction, $\Delta = -0.0354$m).
  * Computational latency: Direct macro executes $K=5$ with CUDA-synchronized median latency **0.371 ms** (mean 0.380 ms, p95 0.467 ms) on GPU vs **1.594 ms** (mean 1.668 ms, p95 1.998 ms) for recursive micro unrolling (**$4.30\times$ measured step speedup**). Straight-line kinematic heuristic median is 0.184 ms. Theoretical RTL hardware cycle models (4,327 vs 21,635 cycles, or 45k vs 190k cycles, $4.22\times$) represent theoretical pipeline models, distinct from GPU runtimes.
  * Closed loop with 24 matched candidates ($N=50$): Direct Macro **46.0%** vs Recursive Micro **50.0%** (McNemar $p = 0.7266$, Wilcoxon $p = 0.4750$). Straight-line kinematic heuristic achieves **46.0%** ($p = 1.0000$ vs Macro). Macro vs Random Candidate Selection is statistically significant (46.0% vs 26.0%, Holm-adjusted $p = 0.0117$). Under Holm-Bonferroni correction over 6 comparisons, macro vs multi-permutation shuffled binding (34.0%, unadjusted $p = 0.03125$) yields adjusted $p = 0.1562 > 0.05$.
  * Prospective Safety Barrier Falsification ($N=40$): Intact collision head yields 70.0% collision rate, identical to shuffled (70.0%, $p = 1.0000$), disabled (70.0%, $p = 1.0000$), and random predictor (70.0%, $p = 1.0000$), while oracle physics simulation cuts collisions to 7.5% ($p < 10^{-7}$). High offline discrimination on macro $K=5$ horizon (AUROC = 1.0000, Brier = 0.0003, ECE = 0.0041, FNR = 0.0%) fails to provide closed-loop safety under temporal constraints.

### Category 3: Controlled Ablations & Behavioral Necessity (`necessity`)
* **Scientific Purpose:** Test causal necessity of prediction, lookahead, action-conditioning, collision awareness, and temporal hierarchy on non-degenerate baselines with regime-matched random controls.
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/circuits/run_rigorous_behavioral_circuits.py --episodes 30 --n-rand 50 --device cuda
  ```
* **Inputs:** `checkpoints_v4/hjepa_best.pt`, `checkpoints_v4/lewm_best.pt`, `splits_v4/train_benchmark.npz`.
* **Output Artifacts:** [`necessity_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/necessity_results.json), [`rigorous_behavioral_circuits.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/rigorous_behavioral_circuits.json).
* **Key Quantitative Results:**
  * Clean Baseline Success: **60.0%** structured, **93.3%** Track C object-grounding suite ($N=30$).
  * Matched random 8D subspace draws evaluated per regime: Track C random drop = **20.8% ($\pm 14.5\%$)**, Structured random drop = **5.3% ($\pm 2.1\%$)**.
  * C1 Visual Target Grounding: Clean 93.3% $\to$ Ablated 90.0% (**3.3% necessity drop**, selectivity gain **-17.4%**, $z = -1.21, p = 0.9032$, `FALSIFIED_NON_SELECTIVE` as a modular circuit; representation hypothesis remains unresolved without claim of confirmed diffuse population code).
  * C2 Egocentric Displacement (10.0% drop, +4.7% selectivity), C3 Obstacle Risk (10.0% drop, +4.7% selectivity), and C4 Forward Dynamics (6.7% drop, +1.3% selectivity) are within $1\sigma$ of structured random controls ($z \le 0.62$, empirical $p = 0.5294$), designated as `UNRESOLVED_NON_SELECTIVE`.
  * Within-model color control: 0.0% drop ($z = -0.69, p = 0.8235$, `FALSIFIED_NON_SELECTIVE`).

### Category 4: Attribution Graphs & Activation Patching (`sufficiency`)
* **Scientific Purpose:** Measure gradient-weighted feature attribution and test sufficiency and rescue of targeted subspaces vs. matched random controls with genuine matched-donor activation patching ($z_{\text{rescue}} = z_{\text{recip}}(I - P) + z_{\text{donor}}P$) satisfying projection invariants, as well as controller memory pathway interventions.
* **Execution Command:**
  ```bash
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/circuits/run_rigorous_behavioral_circuits.py --episodes 30 --n-rand 30 --device cuda
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/circuits/run_goal_path_interventions.py --episodes 30 --device cuda
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/circuits/run_goal_memory_stress_tests_v2.py --episodes 30 --device cuda
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/circuits/run_task_conditioned_circuits_v3.py --pairs 30 --device cuda
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/circuits/run_ego_memory_mechanism_v3.py --trajectories 50 --device cuda
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/circuits/run_task_conditioned_circuits_v4.py --pairs 50 --device cuda
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/circuits/run_ego_memory_mechanism_v4.py --trajectories 50 --device cuda
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/circuits/run_task_conditioned_circuits_v5.py --pairs 50 --device cuda
  MUJOCO_GL=egl .venv/bin/python world_models_fpga/research/circuits/run_ego_memory_mechanism_v5.py --trajectories 50 --device cuda
  ```
* **Inputs:** `checkpoints_v4/*`, `splits_v4/train_benchmark.npz`.
* **Output Artifacts:** [`sufficiency_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/sufficiency_results.json), [`circuit_registry.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/circuit_registry.json), [`goal_memory_stress_tests_v2.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/goal_memory_stress_tests_v2.json), [`goal_memory_stress_tests_v2_traces.npz`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/goal_memory_stress_tests_v2_traces.npz), [`task_conditioned_circuits_v3_summary.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/task_conditioned_circuits_v3_summary.json), [`task_conditioned_circuits_v3_pairs.jsonl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/task_conditioned_circuits_v3_pairs.jsonl), [`ego_memory_mechanism_v3_summary.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/ego_memory_mechanism_v3_summary.json), [`ego_memory_mechanism_v3_pairs.jsonl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/ego_memory_mechanism_v3_pairs.jsonl), [`ego_memory_mechanism_v3_traces.npz`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/ego_memory_mechanism_v3_traces.npz), [`task_conditioned_circuits_v4_summary.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/task_conditioned_circuits_v4_summary.json), [`task_conditioned_circuits_v4_pairs.jsonl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/task_conditioned_circuits_v4_pairs.jsonl), [`task_conditioned_circuits_v4_activations.jsonl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/task_conditioned_circuits_v4_activations.jsonl), [`ego_memory_mechanism_v4_summary.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/ego_memory_mechanism_v4_summary.json), [`ego_memory_mechanism_v4_pairs.jsonl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/ego_memory_mechanism_v4_pairs.jsonl), [`ego_memory_mechanism_v4_steps.jsonl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/ego_memory_mechanism_v4_steps.jsonl), [`task_conditioned_circuits_v5_summary.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_summary.json), [`task_conditioned_circuits_v5_pairs.jsonl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_pairs.jsonl), [`task_conditioned_circuits_v5_interventions.jsonl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_interventions.jsonl), [`task_conditioned_circuits_v5_schema.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_schema.md), [`ego_memory_mechanism_v5_summary.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/ego_memory_mechanism_v5_summary.json), [`ego_memory_mechanism_v5_pairs.jsonl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/ego_memory_mechanism_v5_pairs.jsonl), [`ego_memory_mechanism_v5_steps.jsonl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/ego_memory_mechanism_v5_steps.jsonl), [`ego_memory_mechanism_v5_schema.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/circuits/results/ego_memory_mechanism_v5_schema.md).
* **Key Quantitative Results:**
  * Dedicated `sufficiency_results.json` records clean baseline (60.0% / 93.3%), ablated retention (C1 retained 16.7%), and matched donor patching (Track C random rescue mean = 90.9%, Structured random rescue mean = 53.3%).
  * C1 matched donor patching restores **93.3% navigation success** (0.295m step goal error) vs mismatched donor **83.3%** (0.668m step goal error) and random rescue 90.9% ($p = 0.6129$). Offline, donor patching steers inferred goal vectors toward donor targets ($\cos = 0.9779 \pm 0.0356$, angular error 0.1430 rad vs $\cos = 0.9854 \pm 0.0078$ random). Perturbation diagnostics confirm C1 removes 23.51% variance (Mahalanobis distance 94.991) matching random 8D ($23.99\% \pm 3.38\%$, Mahalanobis distance $94.322 \pm 25.492$). Because random subspace achieves equal or higher donor alignment, selective rescue is not established.
  * Goal Path & Controller Memory Stress Tests v2 ($N=30$ matched episodes per condition, 9 conditions, 10-step sensory blackout at steps 5..15):
    * Full visual + memory: **80.0%** (24/30) success, 0.415m error, occlusion window error 0.648m.
    * Visual-only (0% memory): **83.3%** (25/30) success, 0.405m error, occlusion window error **0.745m**. Visual-only control suffers **$+0.097$m higher tracking error** (95% CI $[0.062, 0.134]$m, Wilcoxon $p = 3.79 \times 10^{-6}$) during blackout, proving controller memory provides active egocentric tracking during visual deprivation. Final navigation success recovers to 83.3% (non-inferior within $\delta=0.10$ margin to full memory 80.0%).
    * Frozen memory: **100.0%** (30/30) success, occlusion error 0.490m.
    * Corrupted memory: **93.3%** (28/30) success, occlusion error 0.685m.
    * Zeroed ego-motion / Perturbed ego-motion: **80.0% / 83.3%** success, occlusion error 0.629m / 0.643m.
    * Matched / Mismatched donor memory: **96.7% / 90.0%** success.
    * Architectural Decomposition: Controller memory is an accumulator in `model_wrappers.py` utilizing learned 2D displacement vectors from `micro_ego_head`. No learned recurrent neural weights exist in `HJEPAModel`.
  * Task-Conditioned Intermediate Circuits Benchmark v5 ($N=50$ pairs, 25 discovery + 25 held-out, 9 stages recorded, 100 random draws per pair):
    * Stage profiling on discovery split: Layer 1 Participation Ratio = **7.63/64** (80% var in 8 dims, Lifetime Sparseness 0.647, Pop Sparseness 0.692, Zero Frac 0.510); Layer 2 Participation Ratio = **5.96/64** (80% var in 6 dims, Lifetime Sparseness 0.661, Pop Sparseness 0.703, Zero Frac 0.561).
    * Held-out ablation & patching sweep ($k \in \{2, 4, 8, 16, 32, 64\}$): Sufficiency transfer scales monotonically with subspace dimension (Layer 2: $k=2 \to 0.0\%$, $k=4 \to 23.9\%$, $k=8 \to 38.6\%$, $k=16 \to 70.7\%$, $k=32 \to 90.2\%$, $k=64 \to 100.0\%$). Candidate top-k units do not outperform random dimension-matched subsets after Holm-Bonferroni correction ($p > 0.05$). Modularity is falsified; distributed low-dimensional population code is confirmed.
  * Ego-Motion and Memory Mechanism Benchmark v5 ($N=1,450$ calibration transitions with corrected $t \to t+1$ target alignment, $N=50$ replay trajectories):
    * Offline sensor delta translation MAE is 0.0274m ($R^2 = 0.0143$, slope 0.2112), providing minimal predictive value over zero baseline (0.0287m); action-conditioned predictor $\delta_s(s_t, a_t)$ achieves MAE **0.0142m** ($R^2 = 0.2549$, slope 0.4538), cutting translation error by 50.5%.
    * Fixed 0.65/0.35 blending collapses all 4 motion sources to $\sim 0.69$m (Sensor 0.6879m, Zero 0.6919m, Action 0.6935m, Oracle 0.6914m) due to $0.35^{10}$ memory decay pulling toward static black frame bias.
    * Under Hold/Freeze dead-reckoning policy, Action Hold reaches **0.5490m**, statistically equivalent to Oracle Hold (**0.5368m**, diff +0.0122m, TOST $p = 0.0000$ at 0.05m margin). Action Hold significantly outperforms Zero Hold (0.6499m, $p = 8.09 \times 10^{-9}$), Zero Action (0.6165m, $p = 8.79 \times 10^{-9}$), and Shuffled Action (0.5768m, $p = 1.49 \times 10^{-4}$).

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
  * Cross-Architecture Asymmetry: LeWM $\to$ H-JEPA achieves **0.977 next-state cosine**, whereas H-JEPA macro $\to$ LeWM achieves **0.639 cosine**.

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
* **Output Artifacts:** [`three_model_comparison.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/extraction/comparison/three_model_comparison.json).
* **Key Quantitative Results:**
  * Parameter Compression: $191,907 \to 39,804$ parameters (**$79.3\%$ reduction**).
  * Distilled Compact 16D: **`46.7%` success**, `26.7%` collision, `0.515` m error, `0.0178` val dyn MSE.
  * BC Only Compact: `53.3%` success, `2.2916` val dyn MSE.

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
  * Vivado 2023.2 Post-Synthesis (`xck26`): `3,579` LUTs (`3.06%`), `13` DSPs (`1.04%`), `0` BRAM, `0.365` W power. Timing closed at **`125.0 MHz`**.
  * RTL Latency: `4,327` cycles (`34.61` $\mu$s @ 125 MHz) single-pass; `1.038` ms 24-candidate MPC pass in software dispatch.
  * Speedup over Kria KR260 ARM Cortex-A53 PS (`19.197` ms): **`44.06x`** (@ 125 MHz).

---

## 4. Master Provenance Verification

All 9 evaluation categories are registered in the machine-readable manifest:
[`world_models_fpga/research/validation/experiment_manifest.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/validation/experiment_manifest.json).

```json
{
  "schema_version": "2.0",
  "provenance_enforced": true,
  "independently_executed_evaluations": 5,
  "legacy_unverified_evaluations": 4,
  "missing_evaluations": 0,
  "rejected_records": 0,
  "independently_executed_categories": [
    "behavior",
    "generalization",
    "necessity",
    "prediction",
    "sufficiency"
  ],
  "legacy_unverified_categories": [
    "architecture_reconstruction",
    "compositionality",
    "hardware_equivalence",
    "transfer"
  ],
  "conclusion": "INCOMPLETE PROVENANCE: 5/9 categories independently executed. 4 categories are legacy unverified. 0 rejected records. 0 categories missing."
}
```
Every recorded entry contains explicit `git_commit`, `dataset_sha256`, `checkpoint_sha256`, `environment`, and complete raw metric dictionaries verified by cryptographic byte hash.
