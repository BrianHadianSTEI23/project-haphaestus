# Phase 5.5.1 — Protocol Deviations Log

**Pre-registered protocol:** [`preregistered_protocol.md`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_5_1/preregistered_protocol.md) (committed at `b905289` prior to cohort generation and simulation execution).

## Summary
- **Deviations from pre-registered design:** **None (`0`).**
- **Verification of pre-registered parameters:**
  - Cohort sample size: `60` scenes (`30` `nominal`, `30` `illumination_shift`), minimum waypoint separation `0.900183 m >= 0.90 m`, minimum initial visibility `274 px >= 60 px`, Oracle A & B feasibility `60/60` & `60/60`.
  - Experimental matrix: All `7` conditions (`F1`–`F4`, `M1`, `M2`, `O1`) x `60` scenes x `2` assignments = `840` episodes executed cleanly from commit `b905289` (`git_clean = true`).
  - Statistical family & multiplicity control: Exact `N = 60` scene-pair unit of analysis, `4000`-resample scene-pair-clustered bootstrap (`seed = 5510`), and 5-family Holm–Bonferroni correction executed identically to specification.
  - Telemetry & export format: Missing/unexecuted branches recorded as `NaN` in `.npz` and `null` in JSONL; all `39` plain-text JSONL trace shards stayed under the `5 MB` cap (`max_observed_shard_bytes = 4,199,903`).
