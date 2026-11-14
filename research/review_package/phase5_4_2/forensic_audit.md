# Phase 5.4.2 Forensic Audit: Cross-Target Estimand Error & Vector Decomposition

## 1. Root Cause Identified in Phase 5.4

In `world_models_fpga/research/circuits/run_task_conditioned_circuits_v5_4_ground_truth_reanalysis.py` (lines 404–412):
- `base_err = np.linalg.norm(base_out - recip_gt)`
- `trans_err = np.linalg.norm(trans_out - donor_gt)`
- `err_imp = base_err - trans_err`

This subtracted distance to two distinct waypoints (`recip_gt` vs `donor_gt`).

## 2. Empirical Verification of Confound & Geometry

1. **No-Op Cross-Target Artifact:** Evaluated across all 60 pairs (240 scene-directions), `mean_baseline_recipient_error_m = 0.4881 m` vs `mean_baseline_donor_error_m = 0.4844 m`, yielding a no-op artifact of `+0.0037 m` (95% CI `[-0.0106, +0.0175] m`). Under single-reference-target `donor_target_gain_m`, any no-op equals `0.0000 m` identically.
2. **Baseline Identity Across Draws:** Across all 447,800 intervention records (2,400 settings), `baseline_output` has `max_abs_diff_across_draws = 0.0` (0 mismatches) and `max_abs_diff_vs_activation_record = 0.0` against `task_conditioned_circuits_v5_3_activations.jsonl`.
3. **Why Donor Distance Can Shrink While `eta_gt < 0`:** In every `(layer, k)` setting where `candidate` exhibits positive `donor_target_gain_m` (e.g., Layer 1 $k=4, 8, 16, 32$ and Layer 2 $k=4, 8, 16, 32$), `recipient_target_change_m` is simultaneously negative (`-0.0160 m` to `-0.0783 m`), meaning transplanted outputs move closer to **both** `donor_gt` and `recip_gt` while projecting negatively onto $\mathbf{d}_{\text{true}} = \mathbf{g}_{\text{donor}}^* - \mathbf{g}_{\text{recip}}^*$ (`target_parallel_component_m = -0.0129 m` to `-0.0493 m` in Layer 1) with large orthogonal shifts (`0.1480 m` to `0.3866 m`).
4. **Difference from Legacy Reporting:** Phase 5.4.1 reported `k=2` `error_improvement_m` Holm-adjusted $p = 1.0000$; under corrected single-target `donor_target_gain_m`, Layer 1 `k=2` yields two-sided Wilcoxon $p = 0.1169$ and Holm-adjusted $p_{\text{adj}} = 0.4675$ (`cand = +0.0008 m` vs `norm = +0.0125 m`, diff `-0.0117 m`). Neither complete-support setting favors candidates, and confidence intervals do not establish equivalence.
