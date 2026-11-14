# Experimentalist Handoff — Phase 5.4.2 & Phase 5.5 (For Independent Scientific Review)

**To:** Independent Scientific Researcher (ChatGPT)  
**From:** Experimentalist / Implementation Engineer  
**Branch:** `fix/interpretability`  
**Immutable Pre-Phase Baseline SHA:** `3362ccd9ea1b006012d2f04636f25019eb28e1da`  
**Code / Fix Commits (Commit 1 series):** `3e2154c` (Phase 5.4.2 + 5.5 implementation & tests), `72dbcfc` (3D target pose fix for `OracleAgent`), `b604db8` (100% Wilson CI `yerr` clamp & `--plot-only` + test)  
**Execution Data / Review Package Commit (Commit 2):** `fc2925c`  
**Report Reconciliation + Handoff Commit (Commit 3):** Pending this commit (HEAD at push)  
**Checkpoint Audited:** `world_models_fpga/checkpoints_v4/hjepa_best.pt` (`sha256: ac69688eae1f4128719dfce912da47c8ae55cbf2412fc8c10feba53d93f1f442`)  
**Paired Scene Cohort:** `world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_2_scene_specs.jsonl` ($N = 60$ pairs `PAIR_000`–`PAIR_059`: 30 `nominal` `light_diffuse_scale=1.0`, 30 `illumination_shift` `light_diffuse_scale=0.60`; $0$ excluded pairs).

---

## 1. Part A Summary — Phase 5.4.2 Estimand Audit & Baseline Reconciliation

1. **Data Integrity & Exact Determinism:**
   - All **447,800** intervention rows (`2,400` candidate, `240,000` dimension-matched random, `205,400` norm-matched random) in `task_conditioned_circuits_v5_3_interventions.jsonl.gz` reconcile with `0` duplicate rows and `0` unmatched rows (`reconciliation_success = true`).
   - Across all `2,400` unique `(pair_id, layer, subset_size_k, direction)` settings, baseline model outputs are bit-for-bit deterministic across all draws (`max_abs_diff_across_draws = 0.0`, `baseline_draw_mismatches_gt_1e_12 = 0`, `max_abs_diff_vs_activation_record = 0.0`).
2. **Invalidation of Phase 5.4 Cross-Target Error Difference (`+0.0344 m`):**
   - Computing cross-target error differences under a zero-perturbation no-op yields a nonzero artifact of **`+0.003741 m`** (95% CI `[-0.010619, +0.017515] m`, where `mean_baseline_recipient_error_m = 0.488122 m`, `mean_baseline_donor_error_m = 0.484382 m`, `mean_true_contrast_norm_m = 0.317304 m`), because negative recipient deterioration inflates the score even without donor improvement.
3. **Single-Target Estimands on Complete-Support Holm Family ($m=4$ tests, `EXPLORATORY_COMPLETE_SUPPORT`):**
   - **Layer 1 `k=2`:**
     - `donor_target_gain_m`: Candidate `+0.0008 m` vs Norm-Matched `+0.0125 m`, diff `-0.0117 m` `[-0.0280, +0.0049] m`, Wilcoxon $p = 0.1169$, **Holm $p = 0.4675$**.
     - `eta_gt` (**dimensionless**): Candidate `-0.0206` vs Norm-Matched `-0.0386`, diff `+0.0181` `[-0.0077, +0.0441]`, Wilcoxon $p = 0.5461$, **Holm $p = 1.0000$**.
   - **Layer 1 `k=4`:**
     - `donor_target_gain_m`: Candidate `+0.0234 m` vs Norm-Matched `+0.0302 m`, diff `-0.0067 m` `[-0.0282, +0.0135] m`, Wilcoxon $p = 0.7628$, **Holm $p = 1.0000$**.
     - `eta_gt` (**dimensionless**): Candidate `-0.0425` vs Norm-Matched `-0.0384`, diff `-0.0041` `[-0.0333, +0.0269]`, Wilcoxon $p = 0.8946$, **Holm $p = 1.0000$**.
4. **Vector Decomposition Diagnostic at Larger $k$ (Incomplete Common Support 51.7%–93.3%):**
   - At Layer 1 `k=16`, Candidate `donor_target_gain_m = +0.0579 m` co-occurs with `recipient_target_change_m = -0.0783 m`, negative `target_parallel_component_m = -0.0493 m`, negative `eta_gt = -0.1566` (dimensionless), `orthogonal_residual_gt_m = 0.2577 m`, and `intervention_norm_m = 0.4253 m` — a non-selective inward/radial displacement rather than norm contraction or selective donor steering.

---

## 2. Part B Summary — Phase 5.5 Closed-Loop Mechanistic Grounding ($N = 60$ Scenes, 420 Episodes)

### 2.1 Condition-Level Table (Plus Oracle Feasibility = `60/60 (100.0%)`)
Step-0 static contrast baseline on the exact cohort: `mean_step0_contrast_eta_gt = -0.551556` (dimensionless), `mean_step0_contrast_cosine = -0.141440`. (`obs_0` matched `v5_2` `obs_a_hash` bit-for-bit on `34/60` scenes; exact poses, lighting, and waypoints restored for all `60/60`).

| Condition | Success | Wilson 95% CI | Nominal / Illum | Collisions | Mean Final Err (m) | Steps | Blend Angle Err | Action Target Cos | Closer to Distractor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `C1_full_closed_loop` | **51/60 (85.0%)** | `[73.9%, 91.9%]` | `83.3% / 86.7%` | 3 (5.0%) | 0.4417 | 17.63 | 46.43° | +0.6501 | 29/60 |
| `C2_frozen_world_goal` | **37/60 (61.7%)** | `[49.0%, 72.9%]` | `60.0% / 63.3%` | 0 (0.0%) | 0.4761 | 24.23 | 63.61° | +0.4984 | 30/60 |
| `C3_target_masked_rgb` | **45/60 (75.0%)** | `[62.8%, 84.2%]` | `66.7% / 83.3%` | 6 (10.0%) | 0.5096 | 23.02 | 61.91° | +0.4553 | 39/60 |
| `C4_distractor_masked_rgb` | **46/60 (76.7%)** | `[64.6%, 85.6%]` | `76.7% / 76.7%` | 3 (5.0%) | 0.4591 | 22.28 | 63.83° | +0.5161 | 36/60 |
| `C5_swapped_task_token` | **52/60 (86.7%)** | `[75.8%, 93.1%]` | `86.7% / 86.7%` | 2 (3.3%) | 0.3907 | 15.72 | 41.23° | +0.7303 | 25/60 |
| `C6_oracle_waypoint` | **60/60 (100.0%)** | `[94.0%, 100.0%]` | `100.0% / 100.0%` | 0 (0.0%) | 0.3294 | 9.20 | 0.00° | +0.9597 | 17/60 |
| `A1_visual_only_no_memory` | **50/60 (83.3%)** | `[72.0%, 90.7%]` | `80.0% / 86.7%` | 5 (8.3%) | 0.4282 | 17.55 | 46.52° | +0.6572 | 30/60 |

### 2.2 Pre-Registered Matched-Pair Comparisons (`holm5` + `A1` Auxiliary)

| Comparison | Success Diff `[95% CI]` | McNemar `(n10, n01)` | Raw $p$ | **Holm5 $p_{\text{adj}}$** | Err Diff (m) | Wilcoxon $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `C2_frozen_world_goal - C1` | **-23.3%** `[-36.7%, -11.7%]` | `(1, 15)` | `5.1880e-04` | **`2.5940e-03`** | +0.0344 | `2.7211e-02` |
| `C3_target_masked_rgb - C1` | -10.0% `[-18.3%, -1.7%]` | `(1, 7)` | `7.0312e-02` | `2.1094e-01` | +0.0678 | `6.0489e-02` |
| `C3_target_masked_rgb - C4` | -1.7% `[-11.7%, +8.3%]` | `(4, 5)` | `1.0000e+00` | `1.0000e+00` | +0.0505 | `1.0692e-01` |
| `C5_swapped_task_token - C1` | +1.7% `[-11.7%, +15.0%]` | `(8, 7)` | `1.0000e+00` | `1.0000e+00` | -0.0510 | `5.6111e-01` |
| `C6_oracle_waypoint - C1` | **+15.0%** `[+6.7%, +25.0%]` | `(9, 0)` | `3.9062e-03` | **`1.5625e-02`** | -0.1124 | `1.1456e-04` |
| `A1_visual_only - C1` (Aux) | -1.7% `[-5.0%, 0.0%]` | `(0, 1)` | `1.0000e+00` | *unadjusted* | -0.0136 | `7.9285e-01` |

*(Non-significant comparisons `C3-C1`, `C3-C4`, `C5-C1`, `A1-C1` do **NOT** establish statistical equivalence.)*

---

## 3. Decision Matrix for Hypotheses H1–H3 & Mechanistic Diagnosis

- **H1 — Closed-Loop Visual Feedback Necessity (`C2` vs `C1`):** **SUPPORTED (`Holm p = 0.00259`)**. Trace inspection reveals `g_visual[0]` underestimates forward range (`mean x = +0.444 m` vs true `+0.662 m`, with `yellow_cylinder` pointing orthogonally `(-0.0006, +0.3261) m`). Freezing world coordinates decays `g_blended.x` (`+0.444 -> +0.141 m`), inflates goal angular error (`42.3° -> 69.9°`), cuts forward drive (`ax: 0.740 -> 0.378`, forward disp `0.311 m`), and stalls short (`0` collisions, `yellow_cylinder = 1/10`). Closed-loop visual re-estimation (`C1`) sustains forward visual servoing (`ax: 0.740 -> 0.597`, forward disp `0.401 m`).
- **H2 — Ground-Truth Waypoint Accuracy Bottleneck (`C6` vs `C1`):** **SUPPORTED (`Holm p = 0.01563`)**. Feeding ground-truth body waypoints lifts success from `85.0% -> 100.0%`, halves steps (`17.63 -> 9.20`), reduces collisions (`3 -> 0`), and flips lateral trajectory correlation with target $y$ from **`r = -0.340` (`C1`) to `r = +0.550` (`C6`)**.
- **H3 — Target-Selective Visual Grounding & Task-Token Mediation (`C3`, `C4`, `C5`):** **NOT SUPPORTED / EXPOSED CONFOUND**. Neither masking target vs distractor (`C3-C4`: `75.0%` vs `76.7%`, `p=1.0`) nor swapping tokens (`C5`: `86.7%`, `p=1.0`) collapses success. Crucially, `traces.npz` explains **both** high `C1` success and negative static `eta_gt`:
  1. In `PAIR_000`–`PAIR_059`, target and distractor waypoints sit side-by-side ahead at $x \approx 0.66\text{ m}$ (`|y| = 0.155 m`) with **mean pairwise distance $0.317\text{ m} < 0.35\text{ m}$ reach radius**.
  2. Any sustained $\sim 0.40\text{ m}$ forward servo enters the overlapping $0.35\text{ m}$ target bubble (`C1` reaches `both = 25/60`, `onlyT = 26/60`; `C3` reaches `both = 31/60`, `onlyT = 14/60`, ending closer to distractor `39/60` with `r(lat, gd_y) = +0.477`).
  3. In `C1`, initial `g_visual.y` and final lateral displacement correlate *negatively* with target $y$ (`r = -0.288` and `r = -0.340`), whereas swapping tokens (`C5`) flips correlation slightly positive (`+0.261` and `+0.100`). Thus `C1` succeeds mainly via a **non-selective forward visual servo exploiting the `0.317 m < 0.35 m` scene geometry**, NOT selective target-token spatial decoding.

---

## 4. Exact Reproduction Commands

```bash
# Part A: Phase 5.4.2 single-target estimand audit
.venv/bin/python -m world_models_fpga.research.circuits.run_task_conditioned_circuits_v5_4_2_estimand_audit

# Part B: Phase 5.5 closed-loop mechanistic grounding (CUDA / MuJoCo)
.venv/bin/python -m world_models_fpga.research.benchmarks.run_task_conditioned_closed_loop_grounding_v5_5

# Part B: Regenerate Phase 5.5 figures strictly from saved artifacts
.venv/bin/python -m world_models_fpga.research.benchmarks.run_task_conditioned_closed_loop_grounding_v5_5 --plot-only

# Full regression suite
.venv/bin/pytest world_models_fpga/tests/
```

---

## 5. Open Questions for Independent Researcher (ChatGPT)

1. Should an adversarial **wide-separation dual-target cohort** (e.g., pairwise target–distractor separation $\ge 0.80\text{ m} \gg 0.35\text{ m}$, or opposite-hemisphere objects) now be designed to eliminate the `0.317 m < 0.35 m` forward-overlap confound and test selective task-token navigation directly?
2. Should `C5` vs `C1`'s flipped initial lateral correlation (`-0.288 -> +0.261`) prompt an audit of training label alignment / `task_embedding` indexing in `checkpoints_v4/hjepa_best.pt` before any further interpretability work on `grounding_head`?
3. Given that `A1` (`83.3%`) showed negligible incremental effect in continuously visible dual-target scenes while earlier v2 blackout tests (`run_goal_memory_stress_tests_v2.py`) showed a tracking-error advantage under 10-step blackout, is goal memory best framed strictly as a **blackout-conditional stabilizer** rather than a core grounding mechanism?
