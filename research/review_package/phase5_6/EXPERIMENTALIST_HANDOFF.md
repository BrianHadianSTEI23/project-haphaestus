# Phase 5.6 Experimentalist Handoff (For Independent Scientific Interpretation)

**Scope Completed:** Parts **A** (Phase 5.5.1.1 H1/Side/Geometry Audit, committed at `fbd96a9` / `ddf07db`) and **B2–B6** (Phase 5.6 Stage-Wise Localization on frozen `hjepa_best.pt`, pre-registered at `8479eb2`, raw results at `0b315e7`).

---

## Core Bounded Measurements Delivered (Strictly Factual)

1. **Part A — Phase 5.5.1.1 (`F1`, `N = 60` scenes / `120` paired episodes):**
   - Reached Assigned First vs Unassigned First (Family A): `32/120 (26.7%)` vs `24/120 (20.0%)`, diff `+0.0667 [-0.0333, +0.1667]`, Wilcoxon `p = 0.2169`.
   - Symmetric Collision-Free (Family B): `32/120 (26.7%)` vs `10/120 (8.3%)`, `p = 3.67e-4` (`10` unassigned-hit episodes collided during remainder of 45-step horizon; assigned episodes terminated immediately upon hit).
   - Complete `N = 60` Crossover (Family C): `both_assigned = 3/60 (5.0%)`, `same_object_collapse = 13/60 (21.7%)`, `both_unassigned = 2/60 (3.3%)`, `at_least_one_neither = 42/60 (70.0%)`, Stuart–Maxwell `p = 0.3244`.
   - Side Skew (A3): Right `27/60 (45.0%)` vs Left `5/60 (8.3%)`, exact McNemar `p = 1.049e-5`; persists in `M2_distractor_masked_rgb` (`2/60 vs 24/60, p = 1.049e-5`); category-stratified Mantel–Haenszel `OR = 8.17` (`p = 3.93e-6`); category+context logistic `OR = 12.90`.
   - Visual Geometry (A4): `60/60` SHA-256 initial frames match bit-for-bit; L/R pixel areas (`448.1 vs 450.1 px, p = 0.7231`) and horizontal centroids (`u_L + u_R - 63 = +0.244 px`) are symmetric.

2. **Part B2–B6 — Phase 5.6 Localization (`hjepa_best.pt`):**
   - **Coordinates & Controller (`B2`):** `all_synthetic_coordinate_checks_passed = true` (`roundtrip <= 2.4e-8 m`, `u_L + u_R = 63.00 px`, `+vy` moves left, `+wz` rotates CCW). Forced Left vs Right goals `[+1.0, +-0.45]` drive both reactive policy and hierarchical planner in the commanded lateral directions.
   - **Training Labels (`B2`):** `train_benchmark.npz` (`408` episodes, `13,188` transitions, `96` unique object layouts) has `75.9%` body-right waypoints across all `4,154` `"approach"` transitions (`blue_cylinder` `97.6%` R, `green_cylinder` `94.7%` R).
   - **Task Embeddings & Fixed-Image Sweep (`B3`):** `task_embedding` is well-separated (`6` `"approach"` L2 `4.44..6.79`, eff-rank `4.62/5`), and `grounding_head` has strong `e_t` Jacobian (`||J_e||_F / ||J_z||_F = 0.897`) plus `0.586 m` counterfactual `A->B` prediction shifts, **yet zero waypoint contrast alignment** (`eta_gt = -0.0105, p = 0.9413`). Across `360` `"approach"` predictions, **`85.3%` point body-right (`dy_mean = -0.351 m`)** — including `96.7%` right (`dy = -0.427 m`) when the assigned object is physically on the Left.
   - **E1 Stage Probes (`B4`):** Under strictly leak-free `GroupKFold_scene`, `z_t` shows only `~3.4 mm` MAE improvement (`R2 = +0.068` Left, `-0.084` Right) over narrow-band fold mean, and all layers (`z_t, h0, L1, L2`) fail (`R2 <= +0.01`) to linearly decode category-bound `obj_1_xy`, `assigned_wp_xy`, or `rel_dir_xy` (`TargetCategoryHoldout` `R2 = -2.19` in `L1`, `-1.38` in `L2`).
   - **E2 Causal Interventions (`B5`–`B6`):** Across `34` intervention rows x `60` scenes (`2,040` rows) covering `e_t`, `z_t` same-target opposite-side donor swaps, full `L1/L2` swaps, and out-of-fold subspaces (`k = 8, 16`), **none** outperform `<= 5%` norm-matched random controls (`12/12` Holm-adjusted `p = 1.0`). Stage `B6` closed-loop rescue was therefore **not triggered** (`NOT_TRIGGERED_NO_CAUSAL_RESCUE_CANDIDATE`).

*Independent interpretation of mechanisms and candidate future architectural/data remedies is left to ChatGPT.*
