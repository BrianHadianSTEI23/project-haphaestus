# Phase 5.6 Review Package — Stage-Wise Task-to-Space Mechanistic Localization

**Experiment:** Phase 5.6 (`run_task_conditioned_stagewise_binding_v5_6.py`)  
**Frozen Checkpoint:** `world_models_fpga/checkpoints_v4/hjepa_best.pt` (`sha256: ac69688eae1f4128719dfce912da47c8ae55cbf2412fc8c10feba53d93f1f442`)  
**Cohort:** `N = 60` independent geometry-separated scenes (`30` nominal, `30` illumination_shift; `0` RGB-hash mismatches against saved cohort frames).  
**Pre-Registered Source Revision:** `8479eb25c562dc72760ca7c229d86550a4e620db` (`git_working_tree_clean: true` at execution).  
**Raw Results Revision:** `0b315e7`.

---

## Executive Empirical Summary (H1–H5 Status)

| Pre-Registered Hypothesis | Question | Empirical Result in Frozen Checkpoint | Status |
|---|---|---|---|
| **H1 (Task-Embedding Collapse)** | Do `"approach"` embeddings collapse or stay separated? | `6` `"approach"` embeddings span L2 `4.438..6.786` (`mean 5.391`, mean cos `+0.093`, centered eff-rank `4.623/5`); all `49` tokens eff-rank `14.767/16`. | **FALSIFIED** (no embedding collapse) |
| **H2 (Visual Preservation vs. Erasure)** | Does `z_t` (`32D`) preserve both object locations vs lose side/object structure (`E1` probes)? | Under `GroupKFold_scene`, `z_t` predicts `left_obj_xy` (`eucl=0.084 m, dMAE=+0.0034 m, R2=+0.068`) and `right_obj_xy` (`eucl=0.099 m, dMAE=+0.0035 m, R2=-0.084`) only negligibly above fold-mean baseline, and fails on `obj_1_xy` (`R2=-0.029`), `obj_2_xy` (`R2=-0.006`), `assigned_wp_xy` (`R2=+0.008`), and `rel_dir_xy` (`R2=0.000`). | **PARTIALLY CLARIFIED** (`E1`: weak slot-level variance capture; no linear category-bound spatial map at any layer) |
| **H3 (Early Fusion & `e_t` Sensitivity)** | Does `grounding_head` ignore `e_t` completely, or shift along ungrounded offsets? | Autograd Jacobian `||d g_xy / d e_t||_F = 0.764` vs `||d g_xy / d z_t||_F = 0.852` (ratio `0.897`); token swaps shift `goal_vec[:2]` by `0.586 m [0.517, 0.655]`, **but** alignment with physical waypoint contrast is null (`eta_gt = -0.0105 [-0.105, +0.081], p = 0.9413`). Stage-wise `L1`/`L2` E1 probes yield `R2 <= -0.046` on `rel_dir_xy` (`TargetCategoryHoldout` `R2 = -2.191 / -1.383`). | **CHARACTERIZED** (`e_t` strongly perturbs output without spatial object-to-token binding) |
| **H4 (Rightward & Category-Side Bias Origin)** | Where does Phase 5.5.1.1's `45.0%` Right vs `8.3%` Left reach skew originate? | **Not** simulator/camera (`all_synthetic_coordinate_checks_passed=true`) and **not** controller sign inversion (`left_steers_left_ok=true`, `right_steers_right_ok=true`). All `6` `"approach"` tokens predict body-right (`85.3%` `dy_b < 0`, mean `-0.351 m`; `96.7%` right even when assigned object is on Left). In `train_benchmark.npz`, `75.9%` of `4,154` `"approach"` transitions lie on the Right (`by_mean = -0.189 m`; `blue_cylinder` `97.6%` Right -> B3 `98.3%` Right; `green_cylinder` `94.7%` Right -> B3 `85.0%` Right). | **LOCALIZED TO GROUNDING HEAD + TRAINING PRIOR** (`g_vis` rightward category/default skew) |
| **H5 (Causal Sufficiency, Necessity, Rescue & Specificity — E2)** | Does any layer (`e_t`, `z_t`, `L1`, `L2`, OOF subspaces `k=8,16`) pass controlled causal switching/rescue above `<=5%` norm-matched random controls? | Across all `2,040` pair rows and `12` Holm–Bonferroni paired contrasts, **zero** counterfactual, donor-swap, or OOF-subspace interventions clear `<=5%` random controls or orthogonal complements (`all 12 holm_p_delta_eta = 1.0`, `all 12 holm_p_delta_gain = 1.0`). | **NO SELECTIVE BINDING CIRCUIT FOUND** (`B6_status = NOT_TRIGGERED_NO_CAUSAL_RESCUE_CANDIDATE`) |

---

## Package Contents

- [preregistered_protocol.md](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_6/preregistered_protocol.md) — Pre-registered hypotheses, splits, interventions, multiplicity rules, and B6 gate (`committed at a2f434b`).
- [coordinate_frame_audit.md](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_6/coordinate_frame_audit.md) — Stage B2 deterministic MuJoCo, controller, and `train_benchmark.npz` audit.
- [forensic_audit.md](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_6/forensic_audit.md) — Full B3/B4/B5 empirical analysis and Stage B6 non-trigger verification.
- [data_dictionary.md](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_6/data_dictionary.md) — Schema and column definitions for all Phase 5.6 artifacts.
- [unresolved_hypotheses.md](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_6/unresolved_hypotheses.md) — Open questions and what retraining / architectural interventions are required next.
- [revised_claims_ledger.csv](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_6/revised_claims_ledger.csv) — Machine-readable summary of Phase 5.6 claims (`E1`/`E2` separated).
- [execution_manifest.json](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_6/execution_manifest.json) — Exact SHA-256 hashes of inputs, frozen checkpoint, script, and outputs.
- [execution_run.log](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_6/execution_run.log) — Complete stdout/stderr execution trace.
- [test_run.log](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_6/test_run.log) — Full `pytest` verification log.
- [protocol_deviations.md](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_6/protocol_deviations.md) — Disclosure of minor pre-execution bug fix and reporting conventions.
- [EXPERIMENTALIST_HANDOFF.md](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/research/review_package/phase5_6/EXPERIMENTALIST_HANDOFF.md) — Bounded experimentalist summary for independent scientific interpretation.
