# Phase 5.4.2 Review Package — Single-Reference-Target Estimand Audit

**Experiment:** Phase 5.4.2 Single-Reference-Target Causal Estimand Audit  
**Script:** `world_models_fpga/research/circuits/run_task_conditioned_circuits_v5_4_2_estimand_audit.py`  
**Unit Tests:** `world_models_fpga/tests/test_task_conditioned_circuits_v5_4_2_estimand_audit.py`  
**Source Commit:** `3e2154cd1b821e765bb26df3d92053d19c6da25e`  

---

## 1. Scientific Correction Summary

Phase 5.4 (`run_task_conditioned_circuits_v5_4_ground_truth_reanalysis.py`) evaluated causal hidden-unit transplantation using a **cross-target error difference**:
$$\text{error\_improvement\_m}_{\text{legacy}} = \|\text{base\_out} - \mathbf{g}_{\text{recip}}^*\| - \|\text{trans\_out} - \mathbf{g}_{\text{donor}}^*\|$$
Because the two terms measured distance to two different ground-truth waypoints ($\mathbf{g}_{\text{recip}}^*$ vs $\mathbf{g}_{\text{donor}}^*$), a pure no-op intervention ($\text{trans\_out} = \text{base\_out}$) yielded a non-zero artifact ($+0.0037\text{ m}$ mean across 60 pairs, 95% CI $[-0.0106, +0.0175]\text{ m}$; baseline recipient error $0.4881\text{ m}$ vs baseline donor error $0.4844\text{ m}$).

Phase 5.4.2 streams all **447,800** Phase 5.3 raw intervention records (`task_conditioned_circuits_v5_3_interventions.jsonl.gz`) paired with `task_conditioned_circuits_v5_3_activations.jsonl` and `task_conditioned_circuits_v5_3_pairs.jsonl` (without modifying or overwriting any Phase 5.1–5.4.1 file) and computes valid **single-reference-target estimands** plus full vector decomposition diagnostics, aggregated strictly at the **independent scene-pair level ($N=60$) per layer and subset size $k$**:

- **`donor_target_gain_m`** (m): $\|\text{base\_out} - \mathbf{g}_{\text{donor}}^*\| - \|\text{trans\_out} - \mathbf{g}_{\text{donor}}^*\|$ ($>0$: closer to donor; $=0$ identically for no-op)
- **`recipient_target_change_m`** (m): $\|\text{trans\_out} - \mathbf{g}_{\text{recip}}^*\| - \|\text{base\_out} - \mathbf{g}_{\text{recip}}^*\|$ ($>0$: farther from recipient)
- **`eta_gt`** (**dimensionless**): $\frac{\langle \mathbf{d}_{\text{int}}, \mathbf{d}_{\text{true}} \rangle}{\|\mathbf{d}_{\text{true}}\|^2}$
- **`target_parallel_component_m`** (m): $\frac{\langle \mathbf{d}_{\text{int}}, \mathbf{d}_{\text{true}} \rangle}{\|\mathbf{d}_{\text{true}}\|}$
- **`orthogonal_residual_gt_m`** (m): $\|\mathbf{d}_{\text{int}} - \eta_{\text{gt}}\mathbf{d}_{\text{true}}\|$
- **`intervention_norm_m`** (m): $\|\mathbf{d}_{\text{int}}\|$
- **`norm_ratio`** (**dimensionless**): $\|\mathbf{d}_{\text{int}}\| / \|\mathbf{d}_{\text{true}}\|$

---

## 2. Baseline Identity & Record Reconciliation Verification

- **Records audited:** 447,800 total (2,400 `candidate`, 240,000 `dimension_matched_random`, 205,400 `norm_matched_random`), 0 duplicates, 0 unmatched rows (`reconciliation_success = true`).
- **Baseline identity across draws:** Across all 2,400 `(pair_id, scene_id, layer, direction, k)` settings, `baseline_output` is **bit-for-bit identical** across `candidate`, `dimension_matched_random`, and `norm_matched_random` draws (`max_abs_diff_across_draws = 0.0`, `baseline_draw_mismatches_gt_1e_12 = 0`), and matches `output_task_a` / `output_task_b` in `task_conditioned_circuits_v5_3_activations.jsonl` (`max_abs_diff_vs_activation_record = 0.0`).

---

## 3. Layer-by-$k$ Pair-Level Results ($N=60$ Independent Pairs)

All inferential tests on complete-support post-hoc settings are explicitly labeled `EXPLORATORY_COMPLETE_SUPPORT`. Non-significant two-sided $p$-values indicate no evidence of difference and **never** establish equivalence.

| Layer | $k$ | Support ($n/60$) | Cand $\text{Gain}_{\text{donor}}$ (m) | Cand $\text{Chg}_{\text{recip}}$ (m) | Cand $\eta_{\text{gt}}$ (dimless) | Cand Parallel (m) | Cand Orth (m) | Cand $\Delta\text{Gain}$ vs NormCtrl (m) [95% CI] | Two-Sided $p$ (Holm $p_{\text{adj}}$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Layer 1** | 2 | 60/60 (100.0%) | `+0.0008` | `+0.0045` | `-0.0206` | `-0.0064` | `0.1103` | `-0.0117` `[-0.0280, +0.0049]` | `0.1169` (`0.4675`) | `EXPLORATORY_COMPLETE_SUPPORT` |
| **Layer 1** | 4 | 60/60 (100.0%) | `+0.0234` | `-0.0160` | `-0.0425` | `-0.0129` | `0.1480` | `-0.0067` `[-0.0282, +0.0135]` | `0.7628` (`1.0000`) | `EXPLORATORY_COMPLETE_SUPPORT` |
| **Layer 1** | 8 | 56/60 (93.3%) | `+0.0390` | `-0.0361` | `-0.0613` | `-0.0188` | `0.1534` | `+0.0141` `[-0.0012, +0.0300]` | `0.1192` (`—`) | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 1** | 16 | 46/60 (76.7%) | `+0.0579` | `-0.0783` | `-0.1566` | `-0.0493` | `0.2577` | `+0.0314` `[+0.0224, +0.0410]` | `0.0000` (`—`) | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 1** | 32 | 37/60 (61.7%) | `+0.0297` | `-0.0335` | `-0.1487` | `-0.0466` | `0.3866` | `+0.0312` `[+0.0170, +0.0456]` | `0.0001` (`—`) | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 2** | 2 | 49/60 (81.7%) | `-0.0027` | `+0.0016` | `+0.0002` | `+0.0001` | `0.0579` | `-0.0138` `[-0.0218, -0.0063]` | `0.0019` (`—`) | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 2** | 4 | 39/60 (65.0%) | `+0.0469` | `-0.0464` | `+0.0017` | `+0.0007` | `0.1295` | `+0.0051` `[-0.0043, +0.0136]` | `0.2645` (`—`) | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 2** | 8 | 40/60 (66.7%) | `+0.0561` | `-0.0573` | `-0.0576` | `-0.0178` | `0.1646` | `+0.0032` `[-0.0071, +0.0120]` | `0.1969` (`—`) | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 2** | 16 | 31/60 (51.7%) | `+0.0368` | `-0.0382` | `-0.0674` | `-0.0206` | `0.2409` | `-0.0104` `[-0.0295, +0.0061]` | `0.8544` (`—`) | `EXPLORATORY_INCOMPLETE_SUPPORT` |
| **Layer 2** | 32 | 48/60 (80.0%) | `+0.0186` | `-0.0285` | `-0.1383` | `-0.0437` | `0.3308` | `-0.0174` `[-0.0258, -0.0095]` | `0.0005` (`—`) | `EXPLORATORY_INCOMPLETE_SUPPORT` |

## 4. Key Mechanistic Observation

Where candidate donor gains appear positive at larger $k$ (e.g., Layer 1 $k=16$: `donor_target_gain_m = +0.0579 m`), **`recipient_target_change_m` is simultaneously negative (`-0.0783 m`)**—meaning the intervention moved outputs closer to **both** targets at once—while the signed component along $\mathbf{d}_{\text{true}}$ is negative (`target_parallel_component_m = -0.0493 m`, dimensionless $\eta_{\text{gt}} = -0.1566$) and dominated by off-axis displacement (`orthogonal_residual_gt_m = 0.2577 m`, `intervention_norm_m = 0.4253 m`). Across both complete-support tests (Layer 1 $k=2, 4$), candidate units show no evidence of superiority over strictly norm-matched random controls on either `donor_target_gain_m` (Holm $p_{\text{adj}} = 0.4675, 1.0000$) or $\eta_{\text{gt}}$ (Holm $p_{\text{adj}} = 1.0000, 1.0000$).
