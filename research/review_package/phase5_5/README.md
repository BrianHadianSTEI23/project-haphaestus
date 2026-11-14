# Phase 5.5 — Closed-Loop Mechanistic Grounding Review Package

## Overview

Phase 5.5 resolves the apparent paradox between Phase 5.4's negative/misaligned static step-0 task contrast and H-JEPA's high closed-loop navigation success on the exact $N=60$ paired scene cohort (`PAIR_000`–`PAIR_059`: 30 `nominal` and 30 `illumination_shift`, drawn from `task_conditioned_circuits_v5_2_scene_specs.jsonl`).

All 7 closed-loop evaluation conditions (`C1`–`C6`, `A1`, $7 \times 60 = 420$ MuJoCo simulation episodes) plus the 60 physical-feasibility Oracle Expert checks were executed under CUDA acceleration using checkpoint `world_models_fpga/checkpoints_v4/hjepa_best.pt` (`sha256: ac69688e...1442`) against fixed physical benchmark threshold `goal_reach_threshold = 0.35 m`, `max_steps = 45`.

---

## Key Empirical Outcomes (N = 60 Matched Scenes per Condition)

| Condition | Success | Wilson 95% CI | Nominal / Illum | Collisions | Mean Final Error (m) | Mean Steps | Blended Angle Error (deg) | Mean Action Target Cos | Closer to Distractor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Oracle ExpertNavigator` | **60/60 (100.0%)** | — | — | — | — | — | — | — | — |
| `C1_full_closed_loop` | **51/60 (85.0%)** | `[73.9%, 91.9%]` | `83.3% / 86.7%` | 3 (5.0%) | 0.4417 | 17.63 | 46.43° | +0.6501 | 29/60 |
| `C2_frozen_world_goal` | **37/60 (61.7%)** | `[49.0%, 72.9%]` | `60.0% / 63.3%` | 0 (0.0%) | 0.4761 | 24.23 | 63.61° | +0.4984 | 30/60 |
| `C3_target_masked_rgb` | **45/60 (75.0%)** | `[62.8%, 84.2%]` | `66.7% / 83.3%` | 6 (10.0%) | 0.5096 | 23.02 | 61.91° | +0.4553 | 39/60 |
| `C4_distractor_masked_rgb` | **46/60 (76.7%)** | `[64.6%, 85.6%]` | `76.7% / 76.7%` | 3 (5.0%) | 0.4591 | 22.28 | 63.83° | +0.5161 | 36/60 |
| `C5_swapped_task_token` | **52/60 (86.7%)** | `[75.8%, 93.1%]` | `86.7% / 86.7%` | 2 (3.3%) | 0.3907 | 15.72 | 41.23° | +0.7303 | 25/60 |
| `C6_oracle_waypoint` | **60/60 (100.0%)** | `[94.0%, 100.0%]` | `100.0% / 100.0%` | 0 (0.0%) | 0.3294 | 9.20 | 0.00° | +0.9597 | 17/60 |
| `A1_visual_only_no_memory` | **50/60 (83.3%)** | `[72.0%, 90.7%]` | `80.0% / 86.7%` | 5 (8.3%) | 0.4282 | 17.55 | 46.52° | +0.6572 | 30/60 |

Cohort step-0 static contrast baseline across the same $N=60$ scenes:
- `mean_step0_contrast_eta_gt = -0.551556` (dimensionless)
- `mean_step0_contrast_cosine = -0.141440`

---

## Pre-Registered Primary Comparisons (5-Test Holm–Bonferroni Family)

| Comparison | Success Diff [95% Boot CI] | McNemar `(n10, n01)` | Exact 2-sided $p$ | **Holm Adjusted $p$** | Final Error Diff (m) | Wilcoxon $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `C2_frozen_world_goal - C1_full_closed_loop` | **-23.3%** `[-36.7%, -11.7%]` | `(1, 15)` | `5.1880e-04` | **`2.5940e-03`** | +0.0344 | `2.7211e-02` |
| `C3_target_masked_rgb - C1_full_closed_loop` | -10.0% `[-18.3%, -1.7%]` | `(1, 7)` | `7.0312e-02` | `2.1094e-01` | +0.0678 | `6.0489e-02` |
| `C3_target_masked_rgb - C4_distractor_masked_rgb` | -1.7% `[-11.7%, +8.3%]` | `(4, 5)` | `1.0000e+00` | `1.0000e+00` | +0.0505 | `1.0692e-01` |
| `C5_swapped_task_token - C1_full_closed_loop` | +1.7% `[-11.7%, +15.0%]` | `(8, 7)` | `1.0000e+00` | `1.0000e+00` | -0.0510 | `5.6111e-01` |
| `C6_oracle_waypoint - C1_full_closed_loop` | **+15.0%** `[+6.7%, +25.0%]` | `(9, 0)` | `3.9062e-03` | **`1.5625e-02`** | -0.1124 | `1.1456e-04` |

Auxiliary comparison (outside Holm5 primary family):
- `A1_visual_only_no_memory - C1_full_closed_loop`: `-1.7%` (`50/60` vs `51/60`, 95% CI `[-5.0%, 0.0%]`, `(0, 1)`, McNemar $p=1.0000$). Note: non-significant difference does NOT establish behavioral equivalence.

---

## Mechanistic Findings Explained Directly by the Traces

1. **Continuous closed-loop goal re-estimation is causally necessary, but primarily as a sustained forward visual servo (`C2` vs `C1`):**
   - At $t=0$, `C1`'s visual goal prediction is foreshortened and misaligned (`mean g_visual[0].x = +0.444 m` vs true target `+0.662 m`; blended angular error `42.3°`; `yellow_cylinder` `g_visual[0] = (-0.001, +0.326) m`).
   - When frozen at $t=0$ in world coordinates (`C2`), moving forward shrinks/rotates the stale goal vector (`g_blended.x: +0.444 -> +0.141 m`, angular error `42.3° -> 69.9°`, forward action component `ax: +0.740 -> +0.378`, `frac(ax > 0.2) = 0.722` vs `0.911` in `C1`), causing stalls (`0` collisions, `24.2` mean steps, `-23.3%` success, Holm $p = 0.00259$). Re-computing `g_visual(obs_t, token)` closed-loop sustains forward velocity (`ax: 0.740 -> 0.597`, forward displacement `0.401 m` vs `0.311 m`).
2. **Waypoint accuracy bottleneck (`C6` vs `C1`):**
   - Replacing `g_blended` with the ground-truth waypoint raises success from `85.0%` to `100.0%` (`+15.0%`, Holm $p = 0.01563$), halves steps (`17.63 -> 9.20`), reduces collisions to `0`, and flips lateral displacement correlation with target $y$ from `r = -0.340` (`C1`) to `r = +0.550` (`C6`).
3. **Why high `C1` (`85.0%`), `C3` (`75.0%`), and `C5` (`86.7%`) success co-exists with poor task selectivity:**
   - **Scene-geometry confound in `PAIR_000`–`PAIR_059`:** Both objects sit ahead of the robot ($x_{\text{tgt}} \in [0.585, 0.734]\text{ m}$, $|y_{\text{tgt}}| = 0.155\text{ m}$; $x_{\text{dist}} \in [0.592, 0.738]\text{ m}$, $|y_{\text{dist}}| = 0.156\text{ m}$) with **mean pairwise waypoint separation $0.317\text{ m} < 0.35\text{ m}$ reach radius**. Consequently, sustained forward driving of $\sim 0.40\text{ m}$ reaches the $0.35\text{ m}$ target bubble in most episodes even without selective target steering (`C1` reaches `both` target and distractor in `25/60` episodes, ends closer to distractor in `29/60`, and has lateral correlation `r(lat, gt_y) = -0.340`).
   - **Visual target masking (`C3`) & task-token swap (`C5`) are statistically inconclusive on binary success rate, yet trace geometry exposes non-selectivity:** Masking the target (`C3`) still reaches `75.0%` because driving forward (`0.440 m`) toward the remaining distractor (`r(lat, gd_y) = +0.477`, `both = 31/60`, `closerD = 39/60`) frequently enters the overlapping $0.35\text{ m}$ target bubble. Conversely, swapping the task token (`C5`) flips initial lateral correlation from negative (`r(gv_y, gt_y) = -0.288` in `C1`) to slightly positive (`+0.261` in `C5`; trajectory `r(lat, gt_y) = +0.100`). Neither `C3 - C4` (`p=1.0`) nor `C5 - C1` (`p=1.0`) permits claiming equivalence.

---

## Figures & Artifacts

- Figures (`research/review_package/phase5_5/figures/`):
  - `fig1_condition_success_and_error.png` — Matched cohort success rates with Wilson 95% CIs and context-stratified errors.
  - `fig2_closed_loop_temporal_evolution.png` — Control-step evolution of target distance, goal angular error, and executed action cosine alignment.
  - `fig3_static_contrast_vs_closed_loop.png` — Relationship between step-0 static contrast (`eta_gt`, dimless) / initial angular error and closed-loop vs frozen-goal final error.
- Raw Outputs (`world_models_fpga/research/benchmarks/results/`):
  - `task_conditioned_closed_loop_grounding_v5_5_summary.json`
  - `task_conditioned_closed_loop_grounding_v5_5_episodes.jsonl` (420 records)
  - `task_conditioned_closed_loop_grounding_v5_5_condition_table.csv`
  - `task_conditioned_closed_loop_grounding_v5_5_traces.npz`
- Source & Tests:
  - `world_models_fpga/research/benchmarks/run_task_conditioned_closed_loop_grounding_v5_5.py`
  - `world_models_fpga/tests/test_task_conditioned_closed_loop_grounding_v5_5.py`
