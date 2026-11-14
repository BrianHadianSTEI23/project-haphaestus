# Phase 5.6 Protocol Deviations Log

1. **Pre-Execution `env.reset()` Unpacking & `sys.path` Bug Fix:**
   - Immediately after committing initial pre-registration (`a2f434b`), launching `run_task_conditioned_stagewise_binding_v5_6.py` failed during Stage B2 (`TypeError: expected np.ndarray (got tuple)` because `JetAutoEnv.reset()` returns `(obs_rgb, info)` and `HJEPAModel` imports `models.*`). Zero sweep, probe, or intervention data were generated on `a2f434b`.
   - Fixed the unpacking (`neutral_rgb, _` and `rgb0, _`) plus `FPGA_ROOT` path addition, extended unit tests to cover `run_coordinate_and_training_audit(..., model=model)`, passed all `3` unit tests (`10.30s`), and committed clean revision `8479eb2` **before** executing Phase 5.6 cleanly with `git_working_tree_clean: true`.
2. **Same-Target Opposite-Side Donor Slot Usage:**
   - Pre-registration originally drafted Stage B5 donor swaps using opposite-side donor scenes (`wp1_b`). Prior to committing source (`a2f434b`), inspection showed all `60/60` cohort scenes have a same-context donor containing the recipient's **exact `obj_1_key`** on the opposite lateral side (in either slot `A` or `B`). Accordingly, `F2_donor_flipped_lr` and `F3_L*_donor_full`/`oof_subspace` paired recipient `stage_A` with that exact target's donor slot (`stage_A` or `stage_B`), guaranteeing zero task-token mismatch (`e_donor == e_a`).
3. **Zero/Mean Ablation Origin Reference (`[0, 0]`):**
   - For necessity ablations (`F2_zero`, `F2_mean`, `F3_L*_zero`, `F3_L*_mean`) where no physical second target exists, `[0.0, 0.0]` was passed as `gt_target_xy`. Consequently, positive `donor_target_gain` on `F3_L1_zero` (`+0.5491 m`) measures prediction shrinkage toward `[0, 0]` rather than selective switching toward a valid object; `eta_gt` remains `-0.1023` (`p = 0.042`, opposite sign / unaligned) and is documented explicitly in `forensic_audit.md`.
4. **Stage B6 Non-Execution by Pre-Registered Rule:**
   - `b6_triggered` evaluated `false` across all evaluated candidates (`max eta_gt = -0.0105 < +0.30`, `min p_vs_matched_random = 0.0991 > 0.01`). Per pre-registration, closed-loop intervention rollouts were not executed.
