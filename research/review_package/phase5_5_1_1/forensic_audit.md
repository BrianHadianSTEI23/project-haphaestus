# Phase 5.5.1.1 — Forensic Audit

## 1. H1 Asymmetry Diagnosed and Corrected
In Phase 5.5.1, `H1` contrasted `pair_mean_assigned_success` against `pair_mean_unassigned_reached`. Three structural asymmetries were confirmed directly in raw traces:
1. **Collision asymmetry:** `unassigned_reached_with_collision = 10/24` in `F1` (`50` across all `840` episodes), whereas `assigned_success` forbids collisions. Requiring strict symmetric collision-free success drops `F1 unassigned_success_symmetric` from `24/120 (20.0%)` to `10/120 (8.3%)` (`Family B p = 3.6702e-04`), still subject to stopping-rule censoring.
2. **Episode-termination censoring asymmetry:** Episodes break out of the closed loop on step `first_assigned_reach_step` (`32` times in `F1`), whereas episodes hitting `unassigned_wp` first continue up to `45` steps, exposing those `24` episodes to later collisions (`10`) or drift outside `0.35 m` (`4`).
3. **Conditional crossover truncation:** The `3 vs 13` binomial (`p = 0.0213`) drops `44/60` pairs (`2` `both_first_unassigned`, `42` `at_least_one_neither`). Under the full `N = 60` denominator, `3x3` Stuart–Maxwell symmetry yields `chi2 = 2.2513, p = 0.3244`, and `both_first_assigned` (`3/60`) vs `both_first_unassigned` (`2/60`) yields McNemar `p = 1.0`.

## 2. Left/Right and Category Imbalance Forensic Finding
- Alternating `side_sign` over `15` unordered pairs per context produced `12/8` cell counts (`red_box/green_cylinder/purple_box = 12 left, 8 right`; `blue_cylinder/yellow_cylinder/cyan_cylinder = 8 left, 12 right`).
- Adjusting for target category (`MH OR = 8.17, p = 3.9270e-06`; logistic `OR = 12.90`) rules out cohort imbalance as the sole cause of the right-side skew (`blue_cylinder` itself is `1/8 (12.5%)` on left vs `11/12 (91.7%)` on right; `M2` with distractor masked is `2/60 (3.3%)` left vs `24/60 (40.0%)` right, `p = 1.0490e-05`).
- Camera extrinsics (`front_cam` centered at `y=0`) and pixel areas (`L=448.1 px` vs `R=450.1 px`, `p=0.7231`) show no severe geometric generator asymmetry, aside from a `4.1 cm` initial waypoint distance difference (`p = 0.0452`) whose causal contribution is directly bounded by `O1` (`left = 86.7%, right = 98.3%`).
