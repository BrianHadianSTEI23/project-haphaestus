# Experimentalist Handoff — Phase 5.5.1.1 (H1 Correction, Bias Audit & Geometry Audit)

**To:** Independent Scientific Researcher (ChatGPT)  
**From:** Experimentalist / Implementation Engineer  
**Branch:** `fix/interpretability`  
**Audit Source Commit:** `fbd96a9`  
**Artifact Manifest:** [`execution_manifest.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1_1/execution_manifest.json)

## 1. Raw Record & Shard Integrity (`A1`)
All `60` scenes, `420` scene-condition pairs, `840` episodes, and `39` JSONL shards (`32,557` trace rows) passed every integrity check (`all_integrity_checks_passed = true`, `0` missing/duplicate/swap/pose/waypoint/shard-hash mismatches). Rerendering all `60` initial states matched `obs_unmasked_0_sha256` bit-for-bit on **`60/60`** scenes.

## 2. Corrected Symmetric H1 Outcomes (`A2`, `F1_dynamic_visual_with_memory`)
- **Family A (Symmetric Waypoint-Reaching / Competing First-Hit):** `first_reached_assigned = 32/120 (26.7%)` vs `first_reached_unassigned = 24/120 (20.0%)`, pair diff `+0.0667 [-0.0333, +0.1667]`, Wilcoxon `p = 0.2169`. Note early termination occurs immediately on all `32` `assigned_reached` episodes, censoring any later unassigned arrivals.
- **Family B (Symmetric Collision-Free Success):** `assigned_success_symmetric = 32/120 (26.7%)` vs `unassigned_success_symmetric = 10/120 (8.3%)`, pair diff `+0.1833 [+0.0917, +0.2750]`, Wilcoxon `p = 3.6702e-04`. Notice `10` of the `24` `unassigned_reached` episodes collided later in the 45-step rollout and `4` drifted outside `0.35 m`, whereas assigned-hitting episodes stop before further collision risk.
- **Family C (Complete 4-Way Crossover Matrix over All `N = 60` Scene Pairs):**
  - `both_first_assigned`: **`3/60 (5.0%)`** `[1.7%, 13.7%]`
  - `same_physical_object_collapse`: **`13/60 (21.7%)`** `[13.1%, 33.6%]` (`3` onto `obj_1`, `10` onto `obj_2`)
  - `both_first_unassigned`: **`2/60 (3.3%)`** `[0.9%, 11.4%]`
  - `at_least_one_neither`: **`42/60 (70.0%)`** `[57.5%, 80.1%]`
  - Population `3x3` Stuart–Maxwell `chi2 = 2.2513, p = 0.3244`; `both_first_assigned` vs `both_first_unassigned` McNemar `p = 1.0`. The conditional `3 vs 13` binomial (`p = 0.0213`) excludes `44/60` pairs and is retained solely as `CONDITIONAL_EXPLORATORY_ONLY`.
- **Family D (Task-Switch Direction):** `step0_eta_vis = -0.0105 [-0.1061, +0.0845]`, Wilcoxon `p = 0.9413` (`30` > 0, `30` < 0); `step0_cos_vis = +0.0030` (`p = 0.9296`); `horizon_mean_world_eta_vis = -0.0089` (`p = 0.9355`).

## 3. Left/Right & Category Bias (`A3`)
- **Matched L/R (`N = 60`, `F1`):** Left **`5/60 (8.3%)`** vs Right **`27/60 (45.0%)`**, paired `[both=3, left_only=2, right_only=24, neither=31]`, RD `+0.3667 [+0.2167, +0.5000]`, exact McNemar **`p = 1.0490e-05`**.
- Stratified by context (`F1`): `nominal` `10.0% vs 43.3% (p = 0.0129)`; `illumination_shift` `6.7% vs 46.7% (p = 4.8828e-04)`. Also persists under `M2_distractor_masked_rgb`: `2/60 (3.3%) vs 24/60 (40.0%) (p = 1.0490e-05)`.
- Category x side balance shows `12/8` alternation (`blue_cylinder` `8L/12R`, `red_box` `12L/8R`), which does **not** explain the side bias: Mantel–Haenszel common `OR = 8.17` (`p = 3.9270e-06`, `MH RD = +0.3611`), category+context-adjusted logistic `OR = 12.90`. Within `F1`, `blue_cylinder` is `1/8 (12.5%)` left vs `11/12 (91.7%)` right; `green_cylinder` is `0/12 (0.0%)` left vs `6/8 (75.0%)` right.

## 4. Scene Geometry Audit (`A4`)
`front_cam` (`pos=[0.105, 0, 0.165]`, `fovy=60°`) renders symmetric object areas (`L=448.1 px` vs `R=450.1 px`, `p = 0.7231`) and horizontal centroids (`L mean u=6.49 px`, `R mean u=56.75 px`, `u_L + u_R - 63 = +0.244 px`), with a `4.1 cm` initial waypoint distance difference (`L=1.087 m` vs `R=1.045 m`, `p = 0.0452`).

## 5. Reproduction
```bash
.venv/bin/python -m world_models_fpga.research.benchmarks.run_task_conditioned_closed_loop_grounding_v5_5_1_1_audit
.venv/bin/pytest world_models_fpga/tests/test_task_conditioned_closed_loop_grounding_v5_5_1_1_audit.py -v
```
