# Phase 5.6 Pre-Registered Protocol — Stage-Wise Task-to-Space Mechanistic Localization

**Pre-registration stage:** Authored and committed prior to running any Phase 5.6 fixed-image token sweep, E1 linear probe, or E2 neural intervention on `hjepa_best.pt`.
**Checkpoint under study:** `world_models_fpga/checkpoints_v4/hjepa_best.pt` (SHA-256: `ac69688eae1f4128719dfce912da47c8ae55cbf2412fc8c10feba53d93f1f442`, unmodified).
**Scene Cohort Under Study:** `N = 60` independent geometry-separated scenes (`30` `nominal`, `30` `illumination_shift`; `120` paired `A/B` episodes) frozen from `world_models_fpga/research/benchmarks/results/task_conditioned_closed_loop_grounding_v5_5_1_scenes.jsonl`.
**Entry Script:** `world_models_fpga/research/benchmarks/run_task_conditioned_stagewise_binding_v5_6.py`
**Unit Tests:** `world_models_fpga/tests/test_task_conditioned_stagewise_binding_v5_6.py`

---

## 1. Epistemic Ground Rules

1. **E1 vs. E2 Separation:** Linear probe decodability (`E1`) measures correlation / linear accessibility only and will NEVER be reported as causal circuit localization (`E2`).
2. **Intervention Evidence Standards (`E2`):** Causal claims require explicit separation of necessity (ablation), sufficiency (donor swap shifts output toward donor target), rescue (restoring task contrast), and specificity (exceeding `<= 5%` norm-matched random controls and orthogonal subspace controls).
3. **Training Data / Checkpoint Constraints:** No model retraining (`hjepa_best.pt` is frozen). No modification or deletion of Phase 5.1–5.5.1.1 raw artifacts.
4. **Statistical Reporting:** Non-significant contrasts do NOT establish equivalence. Null/negative findings will be retained verbatim. All scene-level paired comparisons use scene-cluster bootstrap CIs (`4000` resamples, `seed=5600`) and two-sided Wilcoxon signed-rank tests with Holm–Bonferroni multiplicity adjustment within each hypothesis family.

---

## 2. Pre-Registered Hypotheses (`H1`–`H5`)

- **H1 (Task-Embedding Invariance / Collapse):** Test whether `e_t = model.task_embedding(task_id)` exhibits non-trivial pairwise L2 and cosine separation across the 6 `"approach"` target tokens (`0, 8, 16, 24, 32, 40`) and all `48` `TARGET_KEYS x SEMANTIC_RELATIONS` tokens (+ fallback `48`), OR collapses/clusters trivially.
- **H2 (Encoder Visual Preservation vs. Erasure):** Test under leak-free scene-grouped (`GroupKFold`) Ridge CV whether `z_t = model.encode(I_t)` (`32D`) linearly preserves body-frame coordinates of BOTH candidate objects (`obj_1_xy`, `obj_2_xy`, `left_obj_xy`, `right_obj_xy`, `assigned_wp_xy`, `unassigned_wp_xy`) substantially better than a trivial Mean Baseline, OR whether `z_t` already erases one object / introduces strong left-right representation asymmetry.
- **H3 (Early Fusion & Grounding-Head Attenuation of `e_t`):** In `h0 = [z_t || e_t]` (`48D`) -> `L1` (`64D` post-ReLU) -> `L2` (`64D` post-ReLU) -> `raw` (`4D`), test whether `grounding_head` encodes task-selected quantities (`assigned_wp`, `rel_dir = assigned_wp - unassigned_wp`) in `L1`/`L2` AND whether output `infer_goal_vec` exhibits meaningful target sensitivity (`||delta_pred||`, `eta_vis`, Jacobian `||d goal / d e_t||` vs `||d goal / d z_t||`) or attenuates/ignores `e_t` in favor of a task-invariant visual/default goal prior.
- **H4 (Localization of Rightward and Category-Side Bias):** Partition the `45.0%` Right vs `8.3%` Left behavioral asymmetry found in Phase 5.5.1.1 across four potential stages: (a) MuJoCo camera/simulator/training coordinate convention, (b) `z_t` left vs right decoding precision/bias, (c) `grounding_head` default lateral bias (`dy_b < 0`) and `e_t`-coupled category offsets vs training-split `(target_key, semantic_relation)` empirical `(bx, by)` distribution, and (d) downstream planner/reactive controller symmetry when fed explicit `+dy_b` vs `-dy_b` targets.
- **H5 (Causal Sufficiency, Necessity, Rescue, and Specificity across `e_t`, `z_t`, `L1`, `L2`):** Identify which component(s) satisfy controlled causal sufficiency, necessity, rescue, and specificity on fixed counterfactual images and matched cross-scene donor swaps relative to `<= 5%` norm-matched random controls.

---

## 3. Experimental Pipeline

### Stage B2: Coordinate Frame, Extrinsics, Action Sign, & Training Alignment Verification
Run deterministic synthetic checks in `JetAutoEnv(obs_size=(64, 64))`:
1. Pure body-left (`+dy_b > 0`), pure body-right (`-dy_b < 0`), forward-left, forward-right object placements -> verify MuJoCo `front_cam` segmentation centroid `(u < 31.5)` for left and `(u > 31.5)` for right.
2. Rotated robot heading (`theta in {0, +pi/2, pi, -pi/2}`) with fixed world objects -> verify body-frame formula `[c*dx + s*dy, -s*dx + c*dy]` matches expected quadrant.
3. Mirrored robot poses around a single world object -> verify sign flip of `dy_b` and camera `u` reflection.
4. Identical geometry with swapped categories (`red_box <-> blue_cylinder`) -> verify zero geometry shift.
5. `JetAutoEnv.step` sign check (`act = [0, +0.3, 0]` produces `+vy_body`; `act = [0, 0, +0.5]` produces `+wz` CCW heading increase).
6. Downstream controller (`HJEPAAgent`, `track="waypoint"`) response to symmetric forced goals `[+1.0, +0.45, 0, 1]` vs `[+1.0, -0.45, 0, 1]`.
7. Read-only audit of `train_pipeline.py` training-goal convention and `world_models_fpga/research/datasets/splits_v4/train_benchmark.npz` + `train_benchmark_metadata.json` (`bx`, `by` distributions by `target_key` and `semantic_relation`).

Output: `world_models_fpga/research/benchmarks/results/task_conditioned_stagewise_binding_v5_6_coordinate_audit.json`.

### Stage B3: Fixed-Image Task-Token & Representation Sweep
- Render exact initial RGB frames `I_0` for all `N = 60` cohort scenes and verify bit-for-bit SHA-256 against `v5_5_1_traces.npz` / Phase 5.5.1.1 geometry audit.
- Compute `z_t = model.encode(I_0)` once per scene (verifying deterministic invariance across token evaluations `< 1e-7`).
- Sweep all `49` canonical token IDs (`48` `TARGET_KEYS x SEMANTIC_RELATIONS` + `48` `"none"` fallback), recording `[dx_b, dy_b, sin_dth, cos_dth]`, `||xy||`, bearing `atan2(dy_b, dx_b)`, signed lateral sign `dy_b < 0` vs `dy_b > 0`, projections onto `wp_1`, `wp_2`, `obj_1`, `obj_2`, counterfactual contrast metrics (`eta_vis`, `cos_vis`, `delta_norm`) for `approach(obj_1)` vs `approach(obj_2)`, output sensitivity to `e_t` vs `z_t` via exact autograd Jacobians `||d goal_xy / d e_t||_F` vs `||d goal_xy / d z_t||_F`, and controller velocity `[vx, vy, wz]` on `I_0`.
- Output: `world_models_fpga/research/benchmarks/results/task_conditioned_stagewise_binding_v5_6_token_sweep.jsonl`.

### Stage B4: Stage-Wise Linear Probes (`E1` Only)
- Representations (`X`): `z_t (32D)`, `e_t (16D)`, `h0 = [z_t || e_t] (48D)`, `L1_post_relu (64D)`, `L2_post_relu (64D)`, and `pred_xy (2D)` for reference; compared against fold-mean **Mean Baseline**.
- Body-frame targets (`Y`, shape `(120, 2)` across `60 scenes x {A, B}`): `obj_1_xy`, `obj_2_xy`, `left_obj_xy`, `right_obj_xy`, `assigned_wp_xy`, `unassigned_wp_xy`, and `rel_dir_xy = assigned_wp_xy - unassigned_wp_xy`.
- Strictly leak-free splits:
  1. `GroupKFold(n_splits=5)` by `scene_id` (both `A` and `B` of a scene always stay in the same fold; `StandardScaler` + `Ridge` alpha selected strictly inside train folds from `[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]`).
  2. `CategoryPairHoldout`: 5-fold grouped by `category_pair` (3 unique unordered pairs held out per fold).
  3. `TargetCategoryHoldout`: Leave-One-Target-Category-Out (6 folds holding out all episodes where `target_key == k`).
  4. `ContextTransfer`: Train on `nominal` (`n=30` scenes) -> Test on `illumination_shift` (`n=30` scenes), and reverse.
  5. `SideStratified`: Report test metrics overall and split by `assigned_side == "left"` vs `"right"`, plus left/right error parity on `left_obj_xy` vs `right_obj_xy` inside `GroupKFold`.
- Metrics: Vector MAE (m), Euclidean error (m), Angular error (deg), `R2_x`, `R2_y`, `R2_mean`, Calibration slope/intercept, `delta_MAE_vs_baseline`, with 1000-resample scene-cluster bootstrap CIs (`seed=5600`).
- Output: `world_models_fpga/research/benchmarks/results/task_conditioned_stagewise_binding_v5_6_probe_metrics.csv`.

### Stage B5: Controlled Fixed-Image Causal Interventions Across Stages 1–3 (`E2`)
Three intervention families evaluated on identical pre-registered metrics (`baseline`, `patched`, `true_contrast`, `donor_target_gain`, `recipient_target_change`, `eta_gt`, `parallel`, `orth`, `norm`, `rel_norm_diff` for controls):

1. **Family 1 (`Task Embedding e_t`):** On fixed `I_0`:
   - `F1_counterfactual_swap` (`A <-> B` token swap on identical image, `N=60` scene pairs).
   - `F1_zero` (`e_t <- 0`), `F1_mean` (`e_t <- mean_48_tokens`), `F1_mismatched_third` (`e_t` replaced by another `approach` token not present in the scene's pair), `F1_norm_matched_random` (`||delta e||` matched within `<= 5%` to `||e(B) - e(A)||_2`).
2. **Family 2 (`Visual Latent z_t` Matched Donor Swaps & Controls):**
   - For each recipient scene `i` (`N=60`), deterministically select:
     - `F2_donor_same_lr_diff_cat`: donor `j != i` with identical `(context, obj_1_side)` minimizing `||wp_1_j - wp_1_i|| + ||wp_2_j - wp_2_i||` under distinct object categories (`cat_pair_j != cat_pair_i`).
     - `F2_donor_flipped_lr`: donor `j != i` in the same `context` containing the recipient's **exact `obj_1_key` target** placed on the opposite lateral side (`sign(dy_donor) == -sign(dy_rec)`), keeping recipient `e_t` fixed so we cleanly test visual-location vs task-embedding dominance without task-token mismatch!
   - Paired controls: `F2_norm_matched_random` (`||delta z||` matched within `<= 5%` to each donor delta `z_donor - z_rec`), `F2_zero`, `F2_mean`.
3. **Family 3 (`Grounding Head L1 & L2` Full-Vector & Out-of-Fold Subspace Patching):**
   - Full-vector first (zero fitting bias): `F3_L1_cf_swap`, `F3_L2_cf_swap` (`A -> B` on same scene), `F3_L1_donor_full`, `F3_L2_donor_full` (donor from `F2_donor_flipped_lr`, taking slot `A` or `B` where donor `e_t == recipient e_a`), plus `F3_L1_zero`, `F3_L2_zero`, `F3_L1_mean`, `F3_L2_mean`, and `<= 5%` norm-matched random controls (`F3_L1_cf_norm_matched_random`, `F3_L2_cf_norm_matched_random`, `F3_L1_donor_norm_matched_random`, `F3_L2_donor_norm_matched_random`).
   - Out-of-fold subspaces (`k = 8` and `k = 16` rank): strictly on each `4/5` training fold of `GroupKFold`, fit orthogonal basis `U_task` (`SVD` of Ridge weight matrix mapping `L -> [rel_dir_xy, assigned_wp_xy, unassigned_wp_xy]` augmented with centered within-scene `A vs B` contrast directions; `P_task = U_task U_task^T`) and test on the held-out `1/5` scenes:
     - `F3_L1_oof_subspace_k8`, `F3_L1_oof_subspace_k16`, `F3_L2_oof_subspace_k8`, `F3_L2_oof_subspace_k16` (`L_patch = L_rec + P_task(L_donor - L_rec)`).
     - Compare against orthogonal complement `F3_*_oof_orth_k{8,16}` (`I - P_task`) AND norm-matched rank-`k` random orthogonal projector control `F3_*_oof_rand_subspace_k{8,16}_norm_matched` (`<= 5%` norm match to `P_task(L_donor - L_rec)`).

Output: `world_models_fpga/research/benchmarks/results/task_conditioned_stagewise_binding_v5_6_intervention_pair_metrics.jsonl` (sharded < 5 MB if needed, otherwise single JSONL verified < 5 MB).

### Stage B6: Pre-Registered Optional Closed-Loop Rescue Gate
Closed-loop verification on `N = 12` balanced nominal scenes (`6` Left-assigned, `6` Right-assigned) is executed **ONLY IF** at least one Stage-B5 counterfactual/subspace intervention clears ALL FOUR quantitative rescue criteria:
1. Mean `eta_gt >= +0.30` on held-out scenes;
2. Scene-paired two-sided Wilcoxon `p < 0.01` vs zero-shift AND vs matched `<= 5%` random-perturbation control;
3. Positive `donor_target_gain` with mean reduction `>= +0.10 m` in Euclidean error toward intended goal;
4. Explicit deployment boundary disclosure (e.g., whether intervention is non-privileged vs privileged diagnostic swap).

If no Stage-B5 intervention satisfies all four criteria, Stage B6 is **NOT triggered** (`b6_status: "NOT_TRIGGERED_NO_CAUSAL_RESCUE_CANDIDATE"`), preventing post-hoc fishing expeditions.
