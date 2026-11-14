# Phase 5.6 Forensic Audit — Stage-Wise Localization (B3–B6)

**Summary JSON:** [task_conditioned_stagewise_binding_v5_6_summary.json](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_stagewise_binding_v5_6_summary.json)

---

## 1. Stage B3 — Task-Embedding Geometry & Fixed-Image Sweep (`N = 60 x 49 = 2,940` rows)

1. **`z_t` Invariance Verified:** `z_t_reencode_max_abs_diff = 0.0` (`z_t_invariant_across_tokens_ok = true`).
2. **No Task-Embedding Collapse (H1 Falsified):**
   - `6` `"approach"` tokens (`0, 8, 16, 24, 32, 40`): L2 norms `3.410..4.760` (`mean 4.005`); pairwise L2 distances `4.438..6.786` (`mean 5.391`); mean pairwise cosine `+0.093` (`-0.204..+0.394`); centered effective rank `4.623` (out of max `5`).
   - All `49` canonical tokens: pairwise L2 mean `5.653`, mean cosine `+0.0029`, centered effective rank `14.767 / 16`.
3. **`grounding_head` Responds Strongly to `e_t`—Along Wrong / Unbound Directions (H3):**
   - Autograd Jacobians on `N = 120` `(scene, A/B)` endpoints: `||d g_xy / d z_t||_F = 0.8524 [0.8177, 0.8888]`, `||d g_xy / d e_t||_F = 0.7636 [0.7283, 0.8008]`, ratio **`0.8968 [0.8767, 0.9181]`**.
   - Swapping `A -> B` task token on identical `I_0` changes `g_vis[:2]` by mean norm **`0.5855 m [0.5169, 0.6545]`** (median `0.5259 m`), and within-scene SD across the 6 `"approach"` tokens averages `0.2762 m`.
   - **Yet directional alignment with ground-truth waypoint contrast is null:** `eta_gt = -0.0105 [-0.1053, +0.0814]` (`Wilcoxon p = 0.9413`).
4. **Severe Fixed Rightward & Token-Category Bias in `g_vis` (H4):**
   - Across all `360` `"approach"` evaluations (`60` scenes x `6` tokens), mean predicted lateral offset is **`dy_b = -0.3510 m`**, with **`85.3%` (`307/360`) negative (`dy_b < 0`, Right)**.
   - Even stratified by physical ground truth under `Assignment A`, scenes where the assigned target sits on the **Left** (`n = 30`) receive `dy_b = -0.427 ± 0.227 m` (**`96.7%` predicted Right**), virtually indistinguishable from Right-assigned scenes (`dy_b = -0.404 ± 0.321 m`, **`93.3%` predicted Right**).
   - Individual `"approach"` tokens match their training-set lateral skews in `train_benchmark.npz`: `blue_cylinder` predicts `98.3%` Right (`dy = -0.563 m`, `dx = +0.349 m`), `cyan_cylinder` `98.3%` Right (`dy = -0.507 m`), `green_cylinder` `85.0%` Right (`dy = -0.278 m`), `purple_box` `85.0%` Right (`dy = -0.390 m`), `yellow_cylinder` `73.3%` Right (`dy = -0.223 m`), `red_box` `71.7%` Right (`dy = -0.145 m`).

---

## 2. Stage B4 — Stage-Wise E1 Linear Probes (`546` summary rows; strictly leak-free)

*Important bound: linear decodability is E1 evidence only.* Under `GroupKFold_scene` (zero same-scene leakage):

| Target (Body Frame) | `z_t` (32D) | `h0_cat` (48D) | `L1` (64D) | `L2` (64D) | `pred_xy` (2D) |
|---|---|---|---|---|---|
| `left_obj_xy` | `eucl=0.084 m, dMAE=+0.0034 m, R2=+0.068` | `eucl=0.083 m, dMAE=+0.0039 m, R2=+0.067` | `eucl=0.089 m, dMAE=+0.0002 m, R2=-0.019` | `eucl=0.096 m, dMAE=-0.0038 m, R2=-0.257` | `eucl=0.091 m, R2=-0.077` |
| `right_obj_xy` | `eucl=0.099 m, dMAE=+0.0035 m, R2=-0.084` | `eucl=0.101 m, dMAE=+0.0022 m, R2=-0.075` | `eucl=0.102 m, dMAE=+0.0023 m, R2=-0.060` | `eucl=0.134 m, dMAE=-0.0160 m, R2=-2.513` | `eucl=0.105 m, R2=-0.037` |
| `obj_1_xy` | `eucl=0.750 m, dMAE=-0.0078 m, R2=-0.029` | `eucl=0.750 m, R2=-0.032` | `eucl=0.742 m, R2=-0.025` | `eucl=0.789 m, R2=-0.435` | `eucl=0.736 m, R2=+0.005` |
| `obj_2_xy` | `eucl=0.744 m, dMAE=-0.0065 m, R2=-0.006` | `eucl=0.744 m, R2=-0.010` | `eucl=0.735 m, R2=+0.010` | `eucl=0.782 m, R2=-0.415` | `eucl=0.732 m, R2=-0.021` |
| `assigned_wp_xy` | `eucl=0.474 m, dMAE=-0.0003 m, R2=+0.008` | `eucl=0.494 m, R2=-0.040` | `eucl=0.499 m, R2=-0.056` | `eucl=0.491 m, R2=-0.160` | `eucl=0.475 m, R2=-0.001` |
| `rel_dir_xy` | `eucl=0.942 m, dMAE=0.0000 m, R2=0.000` | `eucl=0.983 m, R2=-0.047` | `eucl=0.992 m, R2=-0.064` | `eucl=0.975 m, R2=-0.210` | `eucl=0.945 m, R2=-0.004` |

- Under `TargetCategoryHoldout_strict_no_scene_overlap` and `CategoryPairHoldout_5fold`, linear decodability of `assigned_wp_xy` and `rel_dir_xy` deteriorates further (`L1 -> rel_dir_xy`: `R2 = -2.191`, `L2 -> rel_dir_xy`: `R2 = -1.383`).
- Across the entire grounding hierarchy (`z_t -> h0 -> L1 -> L2 -> pred_xy`), **no representation supports linear out-of-sample decoding of category-selected object positions or `assigned_wp - unassigned_wp` contrast**, and `z_t` captures at best `~3.4 mm` (`dMAE`) of slot-wise Left/Right variance above the narrow-band fold mean.

---

## 3. Stage B5 — Causal Interventions & Holm–Bonferroni Matched Controls (`E2`, `2,040` rows)

Key observation on feed-forward deterministic equivalence: in `grounding_head(cat(z_t, e_t))`, when donor and recipient share the exact same task embedding `e_a` (as ensured by `_select_matched_donors` matching `obj_1_key` on the opposite side), full-vector overrides at `z_t`, `L1`, and `L2` yield the exact same deterministic output (`F2_donor_flipped_lr == F3_L1_donor_full == F3_L2_donor_full`, mean `eta_gt = -0.0793`). Similarly, `F1_counterfactual_swap == F3_L1_cf_swap == F3_L2_cf_swap` (`eta_gt = -0.0105`). Distinct layer-specific tests are supplied by **independent layer-wise `<= 5%` norm-matched random perturbations** (all achieving `max_rel_act_norm_diff <= 1.2e-7`) and **out-of-fold subspaces `U_k`** (`k = 8, 16`).

### Pre-Registered Causal Contrasts (All `12/12` Null After Holm–Bonferroni)

| Paired Contrast | Treatment `eta_gt` | Control `eta_gt` | `delta_eta` [95% CI] | Raw `p` (`eta` / `gain`) | Holm `p` (`eta` / `gain`) |
|---|---|---|---|---|---|
| `F1_cf_swap` vs `F1_norm_matched_random` | `-0.0105` | `+0.1204` | `-0.1310 [-0.3207, +0.0718]` | `0.0991` / `0.3810` | `1.0` / `1.0` |
| `F2_donor_flipped_lr` vs `F2_norm_matched_random` | `-0.0793` | `+0.0927` | `-0.1720 [-0.3398, -0.0107]` | `0.0962` / `0.5315` | `1.0` / `1.0` |
| `F3_L1_cf_swap` vs `F3_L1_cf_norm_matched_random` | `-0.0105` | `+0.0916` | `-0.1021 [-0.2444, +0.0358]` | `0.2273` / `0.9883` | `1.0` / `1.0` |
| `F3_L2_cf_swap` vs `F3_L2_cf_norm_matched_random` | `-0.0105` | `-0.1077` | `+0.0972 [-0.0346, +0.2265]` | `0.1851` / `0.0918` | `1.0` / `1.0` |
| `F3_L1_donor_full` vs `F3_L1_donor_norm_matched_random` | `-0.0793` | `-0.0500` | `-0.0293 [-0.2073, +0.1491]` | `0.5412` / `0.3097` | `1.0` / `1.0` |
| `F3_L2_donor_full` vs `F3_L2_donor_norm_matched_random` | `-0.0793` | `+0.0438` | `-0.1231 [-0.2609, +0.0138]` | `0.2002` / `0.3062` | `1.0` / `1.0` |
| `F3_L1_oof_subspace_k8` vs `rand_k8` | `-0.0389` | `-0.0489` | `+0.0100 [-0.0485, +0.0681]` | `0.5910` / `0.5219` | `1.0` / `1.0` |
| `F3_L1_oof_subspace_k8` vs `orth_k8` | `-0.0389` | `-0.0212` | `-0.0177 [-0.1206, +0.0831]` | `0.5510` / `0.3972` | `1.0` / `1.0` |
| `F3_L1_oof_subspace_k16` vs `rand_k16` | `-0.0058` | `-0.0266` | `+0.0208 [-0.0983, +0.1323]` | `0.6481` / `0.3423` | `1.0` / `1.0` |
| `F3_L2_oof_subspace_k8` vs `rand_k8` | `-0.0235` | `-0.0104` | `-0.0131 [-0.0686, +0.0407]` | `0.6323` / `0.6964` | `1.0` / `1.0` |
| `F3_L2_oof_subspace_k8` vs `orth_k8` | `-0.0235` | `-0.0558` | `+0.0323 [-0.0775, +0.1512]` | `0.7910` / `0.7740` | `1.0` / `1.0` |
| `F3_L2_oof_subspace_k16` vs `rand_k16` | `-0.0729` | `-0.0168` | `-0.0561 [-0.1421, +0.0251]` | `0.2389` / `0.5363` | `1.0` / `1.0` |

### Note on Zero/Mean Ablation Target Reference
For zero/mean necessity ablations (`F2_zero`, `F2_mean`, `F3_L*_zero`, `F3_L*_mean`), the nominal reference coordinate for `gt_target_xy` was `[0, 0]` (`eta_gt` spans `-0.1023` to `+0.1357`; `F3_L1_zero` reduces Euclidean distance to origin `[0, 0]` by `+0.5491 m` and incidentally moves closer to `wp_a` by `-0.2559 m` solely because unablated `g_vis` predicts strong negative lateral bias `dy = -0.43 m`, whereas ablating `e_t <- 0` (`F1_zero`) reduces distance toward `wp_b` by `+0.1223 m` while remaining unaligned at `eta_gt = -0.0113, p = 0.8309`). None represent task-selective spatial grounding.

---

## 4. Stage B6 Pre-Registered Gate Decision

- Evaluated candidates: `F1_counterfactual_swap`, `F3_L1_cf_swap`, `F3_L2_cf_swap`.
- Observed metrics for all three: `eta_gt_mean = -0.0105` (gate requires `>= +0.30`), `donor_target_gain_mean_m = +0.0363 m` (gate requires `>= +0.10 m`), `p_vs_zero = 0.9413` (`< 0.01` required), `p_vs_matched_random = 0.0991 / 0.2273 / 0.1851` (`< 0.01` required).
- Result: **`b6_triggered = false` (`b6_status = NOT_TRIGGERED_NO_CAUSAL_RESCUE_CANDIDATE`)**.
