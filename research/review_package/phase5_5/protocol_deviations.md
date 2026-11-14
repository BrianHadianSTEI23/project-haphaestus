# Phase 5.5 — Protocol Deviations Log

All design elements followed `research/review_package/phase5_5/preregistered_protocol.md` prior to execution, with the following explicitly documented post-simulation / implementation deviations:

1. **Wilson Upper-Bound Floating-Point Clamp in Figure 1 Plotting (`--plot-only` patch, commit `b604db8`):**
   - During the initial Phase 5.5 run, all 60 Oracle Expert feasibility checks, all 420 simulation episodes, and all 4 primary data files (`summary.json`, `episodes.jsonl`, `condition_table.csv`, `traces.npz`) saved cleanly to disk.
   - Immediately afterward, `generate_figures_from_saved_artifacts` raised `ValueError: 'yerr' must not contain negative values` in `matplotlib` because `wilson_ci(60, 60)` returned `0.9999999999999998` for `C6_oracle_waypoint` (`100.0%`), giving `hi_err = -2.2e-14`.
   - **Resolution:** Clamped `lo_err` and `hi_err` with `max(0.0, ...)` in `generate_figures_from_saved_artifacts`, added CLI flag `--plot-only` to regenerate all PNGs strictly from saved artifacts without re-running MuJoCo physics or overwriting any numerical metric, and added `test_generate_figures_handles_100pct_wilson_ci` to verify zero regression. Only `summary["generated_figures"]` was populated on `--plot-only`.

2. **Initial Observation SHA-256 Match Rate (`34/60` vs `60/60`):**
   - `obs_0_matches_v5_2_hash` matched `v5_2_scene_specs.jsonl` `obs_a_hash` bit-for-bit on `34/60` scenes and differed slightly at render pixel level on `26/60` scenes under current OpenGL/MuJoCo rendering. Exact scene pose, waypoints, lighting scale, and task assignments were restored from `v5_2_scene_specs.jsonl` in all 60 scenes, and step-0 static contrast baseline replicated Phase 5.4 (`mean_step0_contrast_eta_gt = -0.551556`).

3. **Benchmark Scope Observation (Dual-Visible Spatial Proximity):**
   - Although the pre-registered protocol correctly reused `PAIR_000`–`PAIR_059` for strict comparability with Phase 5.3/5.4, trace diagnostics revealed that the two objects' waypoints in `v5_2_scene_specs.jsonl` are separated on average by `0.317 m < 0.35 m` (`goal_reach_threshold`), partially confounding pure target-vs-distractor discrimination with generic forward progress (`C3_target_masked_rgb` = `75.0%`). Reported transparently in `forensic_audit.md` and `unresolved_hypotheses.md`.
