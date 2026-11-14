# Project Hephaestus — Phase 5.5.1.1: H1 Estimand Correction, Side/Category Bias Audit, and Geometry Audit

**Branch:** `fix/interpretability`  
**Audit Source Commit:** `fbd96a9`  
**Script:** [`run_task_conditioned_closed_loop_grounding_v5_5_1_1_audit.py`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/run_task_conditioned_closed_loop_grounding_v5_5_1_1_audit.py)

## 1. Raw Record Audit (`A1`)
All historical Phase 5.5.1 files were preserved unmodified and verified end-to-end:
- `60` unique `scene_id`s (`SCENE_000`–`SCENE_059`), `420` `(condition, scene_id)` pairs, `840` unique episodes (`1` `A` + `1` `B` per pair), `39` trace shards (`32,557` step rows).
- `0` pose mismatches, `0` initial RGB SHA-256 mismatches (`A` vs `B`), `0` target/token swap errors, `0` ground-truth body-frame waypoint mismatches, `0` shard hash/byte/row mismatches, `0` simultaneous same-step first hits.
- `unassigned_reached_with_collision_count = 50` across all `840` episodes (`10` in `F1`), directly explaining why `unassigned_reached` vs `assigned_success` had asymmetric collision & early-termination filtering.

## 2. Corrected Symmetric H1 Endpoints (`A2` — `F1_dynamic_visual_with_memory`)
- **Endpoint Family A (Reaching & Competing First-Hit):** `first_reached_assigned = 32/120 (26.7%)` vs `first_reached_unassigned = 24/120 (20.0%)`, pair diff `+0.0667 [-0.0333, +0.1667]`, Wilcoxon `p = 0.2169`. Early termination occurred on all `32` `assigned_reached` episodes, censoring subsequent unassigned arrivals.
- **Endpoint Family B (Symmetric Collision-Free Success):** `assigned_success_symmetric = 32/120 (26.7%)` vs `unassigned_success_symmetric = 10/120 (8.3%)`, pair diff `+0.1833 [+0.0917, +0.2750]`, Wilcoxon `p = 3.6702e-04` (still affected by early termination of assigned hits vs full 45-step continuation after unassigned hits, where `10/24` collided later and `4/24` drifted outside `0.35 m`).
- **Endpoint Family C (Complete 4-Way Crossover Matrix, `N = 60`):**
  - `both_first_assigned`: `3/60 (5.0%)` `[1.7%, 13.7%]`
  - `same_physical_object_collapse`: `13/60 (21.7%)` `[13.1%, 33.6%]` (`3` onto `obj_1`, `10` onto `obj_2`)
  - `both_first_unassigned`: `2/60 (3.3%)` `[0.9%, 11.4%]`
  - `at_least_one_neither`: `42/60 (70.0%)` `[57.5%, 80.1%]`
  - Population `3x3` Stuart–Maxwell `chi2 = 2.2513, df = 2, p = 0.3244`; `both_first_assigned` vs `both_first_unassigned` McNemar `p = 1.0`. Conditional `3 vs 13` binomial (`p = 0.0213`) excludes `44/60` pairs and is marked exploratory only.
- **Endpoint Family D (Task-Switch Direction):** `F1 step0_eta_vis = -0.0105 [-0.1061, +0.0845]`, Wilcoxon `p = 0.9413` (`30` positive, `30` negative, sign `p = 1.0`), `step0_cos_vis = +0.0030` (`p = 0.9296`), `horizon_mean_world_eta_vis = -0.0089` (`p = 0.9355`).

## 3. Left/Right & Category Bias Audit (`A3` — `F1`)
- Matched `N = 60` L/R pairs: **Left `5/60 (8.3%)` vs Right `27/60 (45.0%)`**, paired table `[both=3, left_only=2, right_only=24, neither=31]`, paired RD `+0.3667 [+0.2167, +0.5000]`, exact McNemar **`p = 1.0490e-05`**.
- Present in both contexts (`nominal`: `10.0% vs 43.3%, p = 0.0129`; `illumination_shift`: `6.7% vs 46.7%, p = 4.8828e-04`) and even when distractors are masked (`M2`: `2/60 (3.3%)` left vs `24/60 (40.0%)` right, `p = 1.0490e-05`, `MH OR = 23.00`).
- Cohort category x side has a `12 vs 8` alternation imbalance (`blue_cylinder`: `8 left, 12 right`; `red_box`: `12 left, 8 right`), **not** explaining the side effect: category-stratified Mantel–Haenszel `OR = 8.17` (`CMH chi2 = 21.30, p = 3.9270e-06`, `MH RD = +0.3611`), and category+context-adjusted logistic regression `OR = 12.90` (`unadjusted OR = 9.00`).

## 4. Scene Generator & Visual Geometry Audit (`A4`)
- Rerendering all `60` scenes with MuJoCo `JetAutoEnv` reproduced `obs_unmasked_0_sha256` bit-for-bit on **`60/60`** scenes (`0` pixel-count mismatches).
- `front_cam` (`pos=[0.105, 0, 0.165]`, `fovy=60°`) places left objects at mean `u = 6.49 px` (`448.1 px` area) and right objects at mean `u = 56.75 px` (`450.1 px` area; area Mann–Whitney `p = 0.7231`, centered sum minus `63` = `+0.244 px`). Mean `init_dist_wp_m` differs slightly (`L = 1.087 m` vs `R = 1.045 m`, `p = 0.0452`).
