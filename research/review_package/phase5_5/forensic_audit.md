# Phase 5.5 — Forensic Audit

## 1. Execution & Provenance Verification

- **Scene cohort:** Exact $N = 60$ paired episodes (`PAIR_000`–`PAIR_059`: 30 `nominal` with `light_diffuse_scale=1.0`, 30 `illumination_shift` with `light_diffuse_scale=0.60`) loaded from `world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_2_scene_specs.jsonl` (`sha256: 5edfcf5d8e1638b9e624ea7fb796ab87a951ce88ef81b5244af339c9d7f34337`).
- **Checkpoint & Simulator:** `world_models_fpga/checkpoints_v4/hjepa_best.pt` (`sha256: ac69688eae1f4128719dfce912da47c8ae55cbf2412fc8c10feba53d93f1f442`) running under CUDA (`torch` GPU) and MuJoCo (`JetAutoEnv`, XML `sha256: 0e655e50a8119de6402a5f041369c4728d7d2d9cf4bd8bc4bf8a9a582653bb51`).
- **Physical solvability control:** `OracleAgent` (`ExpertNavigator` with true 3D `target_pose` and obstacle map) succeeded on **60/60 (100.0%)** scenes under identical physics, horizon (`max_steps=45`), and reach threshold (`goal_reach_threshold=0.35 m`).
- **Initial observation pixel-hash audit (`obs_0_matches_v5_2_hash`):** Exact SHA-256 bit-identity against `obs_a_hash` in `v5_2_scene_specs.jsonl` held for **34/60** scenes (`19/30` nominal, `15/30` illumination_shift). On the remaining 26 scenes, MuJoCo rendering produced minor GPU/driver floating-point pixel differences while initializing the exact pose, targets, lighting, and waypoints recorded in `v5_2_scene_specs.jsonl`. Crucially, cohort step-0 static contrast (`mean_step0_contrast_eta_gt = -0.551556`, `mean_step0_contrast_cosine = -0.141440`) replicates Phase 5.4's negative static task-contrast alignment.

## 2. Diagnostic Discovery: Why High Closed-Loop Success Co-occurs with Poor Static Selectivity

Step-level trajectory decomposition from `task_conditioned_closed_loop_grounding_v5_5_traces.npz` resolves the Phase 5.4 vs closed-loop puzzle without invoking hidden target-selective circuitry:

1. **Spatial Geometry of `PAIR_000`–`PAIR_059` Creates a Forward-Drive Reach Overlap:**
   - At $t=0$ in robot body frame: target waypoint $x \in [0.585, 0.734]\text{ m}$, $|y|$ mean $= 0.155\text{ m}$ (distance $0.680\text{ m}$); distractor waypoint $x \in [0.592, 0.738]\text{ m}$, $|y|$ mean $= 0.156\text{ m}$ (distance $0.690\text{ m}$).
   - Mean Euclidean distance between target and distractor waypoints is **$0.317\text{ m} < 0.35\text{ m}$ reach radius**.
   - As a result, driving forward by $\sim 0.40\text{ m}$ enters the target's $0.35\text{ m}$ reach radius frequently even when lateral steering is unselective (`C1` reaches `both = 25/60`, `onlyT = 26/60`; `C3` reaches `both = 31/60`, `onlyT = 14/60`; `C4` reaches `both = 30/60`, `onlyT = 16/60`).
2. **Lateral Steering Remains Unselective in `C1`:**
   - Initial body-frame lateral correlation of `g_visual[0].y` with ground-truth target $y$ is **$r = -0.288$** (and $+0.311$ with distractor $y$); actual executed body-frame lateral displacement correlates negatively with target $y$ (**$r = -0.340$** in `C1`, vs **$+0.550$** in `C6_oracle_waypoint`).
   - Swapping the task token (`C5_swapped_task_token`) flips the initial lateral sign (`r(gv0_y, gt_y) = +0.261`), yielding `52/60 (86.7%)` with trajectory correlation `r = +0.100` — confirming that task conditioning in this checkpoint does not robustly align `g_visual` toward the requested target.
3. **Why `C2_frozen_world_goal` Collapses Without Collisions (`-23.3%`, Holm $p = 0.00259$):**
   - Static `g_visual[0]` underestimates forward distance (`+0.444 m` vs true `+0.662 m`) and sometimes mispoints almost orthogonally (e.g., `yellow_cylinder`: mean `g_visual[0] = (-0.0006, +0.3261) m`, yielding `1/10` success in `C2` over `42.8` steps vs `8/10` in `C1`).
   - Freezing that stale point in world space causes `g_blended.x` to decay as the robot moves (`+0.444 -> +0.141 m`), angular error to climb (`42.3° -> 69.9°`), and forward velocity to drop (`ax: 0.740 -> 0.378`), leaving episodes stalled just outside the `0.35 m` threshold (`mean_final_goal_error_m = 0.4761 m`, `0` collisions).

## 3. Verification Integrity

- Pre-registered protocol (`research/review_package/phase5_5/preregistered_protocol.md`) was finalized **before** any full simulation execution.
- All 4 unit tests in `world_models_fpga/tests/test_task_conditioned_closed_loop_grounding_v5_5.py` pass after the final `--plot-only` Wilson CI clamp edit (`4 passed`).
- Historical Phase 5.1–5.4.1 result files were left completely untouched.
