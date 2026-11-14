# Project Hephaestus — Phase 5.5.1: Geometry-Separated Task Selectivity, Visual Feedback, and Controller-Memory Decomposition

**Branch:** `fix/interpretability`  
**Pre-Registered Code & Protocol Commit:** `b905289`  
**Execution Data & Manifest Commit:** `945ac53`  
**Independent Scientific Reviewer:** ChatGPT  
**Role of Package:** Experimentalist implementation, raw execution artifacts, confirmatory statistics, and forensic audit (no speculative theory synthesis).

## 1. Scientific Objective
Phase 5.5 showed high closed-loop success (`C1 = 51/60 = 85.0%`) on a cohort (`PAIR_000`–`PAIR_059`) whose target and distractor waypoints were separated by `0.276–0.378 m` (`mean = 0.317 m < 0.35 m` reach radius), alongside carried-over telemetry ambiguities (`Part A2`). Phase 5.5.1 tests physical object selectivity, repeated visual feedback, and controller ego-memory on an **independent, geometry-separated `N = 60` paired-scene cohort** (`waypoint_separation >= 0.90 m`, dual mirrored task assignments `A` and `B`, `840` total episodes).

## 2. Cohort Construction (`N = 60` Accepted Scenes)
From `755` generated candidates (`695` rejected: `waypoint_sep_below_threshold = 346`, `arena_bounds = 253`, `obstacle_clearance = 84`, `oracle_unreachable_B = 8`, `oracle_unreachable_A = 4`), `60` scenes passed all geometric, visibility, and Oracle reachability criteria (`30` `nominal`, `30` `illumination_shift`):
- `waypoint_separation_m`: `min = 0.900183 m`, `mean = 0.942498 m`, `max = 1.036776 m` (`0.35 m` reach circles separated by `>= 0.20 m` gap).
- `initial_target_dist`: `[0.824470, 1.197321] m`.
- `min_init_visible_px`: `274` pixels (`>= 60` required).
- `OracleAgent` feasibility: `60/60` on `Assignment A` and `60/60` on `Assignment B`.

## 3. Summary of Raw Condition Results (`120` Episodes = `60` Pairs x `2` Assignments Each)

| Condition | Assigned Success `[95% CI]` | Unassigned Reached | 1st Reached (`A/U/N`) | Dual 1st-Correct | Same-Obj Collapse | Pair Selectivity Margin | Collisions | Nominal / Illum | Left / Right |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `F1_dynamic_visual_with_memory` | **32/120 (26.7%)** `[19.6%, 35.2%]` | 24/120 (20.0%) | `32 / 24 / 64` | 3/60 (5.0%) | 13/60 (21.7%) | `+0.067` `[-0.033, +0.167]` | 41/120 (34.2%) | `26.7% / 26.7%` | `8.3% / 45.0%` |
| `F2_dynamic_visual_no_memory` | **33/120 (27.5%)** `[20.3%, 36.1%]` | 24/120 (20.0%) | `33 / 24 / 63` | 3/60 (5.0%) | 12/60 (20.0%) | `+0.075` `[-0.042, +0.183]` | 44/120 (36.7%) | `28.3% / 26.7%` | `8.3% / 46.7%` |
| `F3_frozen_visual_with_memory` | **9/120 (7.5%)** `[4.0%, 13.6%]` | 6/120 (5.0%) | `9 / 6 / 105` | 0/60 (0.0%) | 1/60 (1.7%) | `+0.025` `[-0.033, +0.083]` | 16/120 (13.3%) | `3.3% / 11.7%` | `0.0% / 15.0%` |
| `F4_frozen_visual_no_memory` | **9/120 (7.5%)** `[4.0%, 13.6%]` | 5/120 (4.2%) | `9 / 5 / 106` | 0/60 (0.0%) | 1/60 (1.7%) | `+0.033` `[-0.025, +0.092]` | 16/120 (13.3%) | `5.0% / 10.0%` | `1.7% / 13.3%` |
| `M1_target_masked_rgb` | **14/120 (11.7%)** `[7.1%, 18.6%]` | 28/120 (23.3%) | `14 / 28 / 78` | 0/60 (0.0%) | 5/60 (8.3%) | `-0.117` `[-0.208, -0.025]` | 52/120 (43.3%) | `13.3% / 10.0%` | `3.3% / 20.0%` |
| `M2_distractor_masked_rgb` | **26/120 (21.7%)** `[15.2%, 29.9%]` | 18/120 (15.0%) | `26 / 18 / 76` | 0/60 (0.0%) | 7/60 (11.7%) | `+0.067` `[-0.025, +0.150]` | 50/120 (41.7%) | `20.0% / 23.3%` | `3.3% / 40.0%` |
| `O1_track_c_planner_goal_override` | **111/120 (92.5%)** `[86.4%, 96.0%]` | 1/120 (0.8%) | `111 / 1 / 8` | 51/60 (85.0%) | 1/60 (1.7%) | `+0.917` `[+0.858, +0.967]` | 1/120 (0.8%) | `90.0% / 95.0%` | `86.7% / 98.3%` |

## 4. Confirmatory 5-Family Holm–Bonferroni Results (`N = 60` Physical Scene Pairs)
- **H1 (`F1` task selectivity):** `pair_mean_assigned_success (0.2667) - pair_mean_unassigned_reached (0.2000) = +0.0667` `[-0.0417, +0.1667]`, Wilcoxon `p = 0.216872`, **`Holm5_p = 0.433744`**; `dual_first_correct (3/60)` vs `same_object_collapse (13/60)` binomial `p = 0.021271`; `step0_eta_vis = -0.0105` `[-0.1078, +0.0812]`, `p = 0.941316`.
- **H2 (Visual feedback main effect):** `0.5*(F1+F2) - 0.5*(F3+F4) = +0.1958` `[+0.1292, +0.2667]`, Wilcoxon `p = 3.2228e-06`, **`Holm5_p = 1.2891e-05`** (`F1 - F3 = +0.1917, p = 9.5840e-06`; `F2 - F4 = +0.2000, p = 2.5165e-06`).
- **H3 (Controller ego-memory main effect):** `0.5*(F1+F3) - 0.5*(F2+F4) = -0.0042` `[-0.0167, +0.0083]`, Wilcoxon `p = 0.563703`, **`Holm5_p = 0.563703`**; interaction `(F1-F2)-(F3-F4) = -0.0083`, `p = 0.563703`; simple `F1-F2 = -0.0083, p = 0.317311`; simple `F3-F4 = 0.0000, p = 1.0000`. (Equivalence is not claimed from non-significant contrasts.)
- **H4 (Object masking specificity):** `M1 - M2 = -0.1000` `[-0.1583, -0.0333]`, Wilcoxon `p = 0.002700`, **`Holm5_p = 0.008099`** (within-family Holm `0.005400`); `M1 - F1 = -0.1500` `[-0.2167, -0.0833]`, within-family Holm `7.1569e-04`; `M2 - F1 = -0.0500` `[-0.1083, +0.0083]`, within-family Holm `0.133614`.
- **H5 (Planner goal override):** `O1 - F1 = +0.6583` `[+0.5667, +0.7500]`, Wilcoxon `p = 1.3813e-10`, **`Holm5_p = 6.9065e-10`**.

## 5. Contents of Review Package
- [`preregistered_protocol.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/preregistered_protocol.md)
- [`EXPERIMENTALIST_HANDOFF.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/EXPERIMENTALIST_HANDOFF.md)
- [`forensic_audit.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/forensic_audit.md)
- [`data_dictionary.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/data_dictionary.md)
- [`protocol_deviations.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/protocol_deviations.md)
- [`revised_claims_ledger.csv`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/revised_claims_ledger.csv)
- [`unresolved_hypotheses.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/unresolved_hypotheses.md)
- [`execution_manifest.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/execution_manifest.json)
- [`execution_run.log`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/execution_run.log)
- [`test_run.log`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/test_run.log)
- `figures/` (`fig1_v5_5_1_selection_and_context_rates.png`, `fig2_v5_5_1_2x2_feedback_memory_factorial.png`, `fig3_v5_5_1_temporal_dynamics.png`, and 7 inspection PNGs in `figures/frames/`)
