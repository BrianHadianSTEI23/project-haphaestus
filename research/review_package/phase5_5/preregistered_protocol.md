# Phase 5.5 Pre-Registered Experimental Protocol: Closed-Loop Mechanistic Grounding

**Status:** PRE-REGISTERED PRIOR TO EXECUTION OF CONDITIONS C1–C6  
**Repository:** `BrianHadianSTEI23/project-haphaestus` (`fix/interpretability`)  
**Script:** `world_models_fpga/research/benchmarks/run_task_conditioned_closed_loop_grounding_v5_5.py`  

---

## 1. Scientific Motivation

Phase 5.4 demonstrated on 60 dual-visible paired scenes (`PAIR_000`–`PAIR_059`) that H-JEPA's static $t=0$ task-conditioned output contrast $\Delta\mathbf{g}_{\text{model}}(t=0)$ is nearly orthogonal ($90.63^\circ$, $\cos = +0.0010$, $\eta_{\text{gt}} = -0.1984$) to physical target waypoint contrast $\Delta\mathbf{g}_{\text{gt}}$. Yet earlier closed-loop Track C evaluations (e.g., `track_c_visual_grounding.json`, `task_identity_visual_grounding_v2.json`) reported 73.3%–100.0% closed-loop success on visible targets.

Existing traces (`task_identity_visual_grounding_v2_traces.npz`) indicate that on dual-visible `cond_4a`/`cond_4b`, initial blended goal vectors already exhibit large error ($0.55$–$0.56$m, $37^\circ$–$43^\circ$ angular error) and still end at $\sim 0.70$m ($62^\circ$–$64^\circ$) while reaching the $0.35$m goal threshold in $66.7\%$–$70.0\%$ of episodes. However, existing datasets did not test whether closed-loop success survives freezing the step-0 world goal, masking ongoing target pixels vs distractor pixels, or replacing inferred goals with an oracle waypoint inside the exact same paired Phase 5.4 scenes and planner.

## 2.Matched Scene Cohort & Invariants

- **Cohort:** All $N=60$ verified dual-visible scenes (`scene_a`) from `world_models_fpga/research/circuits/results/task_conditioned_circuits_v5_2_scene_specs.jsonl` (`PAIR_000`–`PAIR_059`), comprising:
  - `nominal` context: $N=30$ scenes (`light_diffuse_scale = 1.0`)
  - `illumination_shift` context: $N=30$ scenes (`light_diffuse_scale = 0.60`)
- **Geometry & Visibility Invariants Verified Before Evaluation:**
  - Zero episodes start inside `goal_reach_threshold = 0.35`m (initial waypoint distance $\ge 0.602$m).
  - Both target (`key_a` at `pos_1`) and visible distractor (`key_b` at `pos_2`) pass $\ge 100$ visible pixels in `front_cam` at $t=0$.
  - Initial `obs_0` SHA-256 digest matches stored `obs_a_hash`.
  - Physical solvability reference check executed with `OracleAgent` (`ExpertNavigator`).
- **Horizon & Success Criteria (unchanged from `v2`):**
  - `max_steps = 45`, `goal_reach_threshold = 0.35`m.
  - `success = (not initially_inside_radius) and goal_reached and (not collision) and (final_eval_goal_error_m <= 0.35)`.

## 3. Experimental Conditions

1. **`C1_full_closed_loop`**: Default Track C `HJEPAAgent(track="observation")` (`0.65 * g_visual + 0.35 * g_memory`).
2. **`C2_frozen_world_goal`**: At $t=0$, the initial blended goal $\mathbf{g}_0$ is transformed from initial pose $\mathbf{p}_0$ to fixed world coordinates $\mathbf{w}_0^{\text{frozen}}$. For all $t \ge 0$, `goal_t` is computed purely by transforming $\mathbf{w}_0^{\text{frozen}}$ into current robot body frame $\mathbf{p}_t$, disabling ongoing visual goal recomputation while keeping encoder/planner RGB inputs live.
3. **`C3_target_masked_rgb`**: At every step $t$, pixels belonging to `target_key`'s MuJoCo segmentation mask in `front_cam` are replaced with the local 2-px perimeter background median RGB before encoding.
4. **`C4_distractor_masked_rgb`**: At every step $t$, an exact pixel-area-matched mask (`|mask_C4| == |mask_target|`) centered on `distractor_key`'s segmentation mask (expanding locally or using an opposite-bearing background patch strictly disjoint from target if distractor exits FOV, with fallback logged per step) is replaced via the identical local perimeter median fill rule.
5. **`C5_swapped_task_token`**: Identical RGB scene and target evaluation (`wp_1` for `key_a`), with `task_id` set to `encode_task_id(key_b, "approach")`. Secondary endpoint also logs arrival at `wp_2` (`key_b`).
6. **`C6_oracle_waypoint`**: Identical `HJEPAAgent` planner/policy (`track="structured"`) given ground-truth body-frame goal vector toward `wp_1` at every step.

### Auxiliary Secondary Condition (Separate from C1–C6 Family)
- **`A1_visual_only_no_memory`**: `HJEPAAgent(track="observation")` with `blended_goal` set strictly to `g_visual_t` (0% controller memory weight), isolating controller memory vs visual branch contribution during continuous observation.

## 4. Hypotheses & Pre-Registered Statistical Analysis Plan

- **Unit of Inference:** Matched scene ($N=60$ paired episodes across conditions; additionally stratified by `nominal` $N=30$ and `illumination_shift` $N=30$).
- **Primary Family of 5 Paired Hypothesis Comparisons (Holm-Bonferroni adjusted across the 5 comparisons, $\alpha = 0.05$):**
  1. **H1 (Closed-Loop Recomputation Necessity):** `C2_frozen_world_goal` vs `C1_full_closed_loop` — tests whether step-0 world goal suffices or continuous visual feedback improves closed-loop success and reduces error.
  2. **H2a (Target Visual Appearance Necessity):** `C3_target_masked_rgb` vs `C1_full_closed_loop` — tests whether ongoing target pixels are causally required for closed-loop steering.
  3. **H2b (Target vs Distractor Masking Specificity):** `C3_target_masked_rgb` vs `C4_distractor_masked_rgb` — controls for area-matched visual occlusion perturbation.
  4. **H2c (Task-Token Routing Necessity):** `C5_swapped_task_token` vs `C1_full_closed_loop` — tests whether categorical task token shifts trajectory choice away from original target on identical visuals.
  5. **H3 (Grounding vs Downstream Planner Bottleneck):** `C6_oracle_waypoint` vs `C1_full_closed_loop` — isolates the performance loss attributable to inferred goal inaccuracies within the exact same H-JEPA planner.
- **Statistical Tests:**
  - Binary `success`: Two-sided exact McNemar test on matched discordances `(n10, n01)`, paired risk difference with 2,000-resample paired bootstrap 95% CI, Holm-Bonferroni adjusted $p$-values across the 5 primary comparisons.
  - Continuous endpoints (`final_eval_goal_error_m`, `path_length_m`, `mean_goal_angle_err_deg`, `mean_action_target_cos`): Two-sided Wilcoxon signed-rank test + paired bootstrap 95% CI of mean paired difference.
  - Equivalence is NEVER concluded from non-significant $p$-values alone; full 95% CIs are reported for all differences.
