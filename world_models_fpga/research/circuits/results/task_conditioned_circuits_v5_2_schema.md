# Schema: Phase 5.2 Task-Conditioned Circuits Revalidation Artifacts

**Execution Commit:** `bc568052e8be10fba4f4a466e913e3dbd8bb4fd6` (Clean: `False`)  
**Timestamp:** `2026-11-08T13:33:18.013394+00:00`  
**Dataset Size:** 60 independent scene pairs (30 nominal, 30 illumination shifted)

---

## 1. Primary Artifacts

1. `task_conditioned_circuits_v5_2_summary.json`:
   - `dataset_composition`: Pair counts, illumination parameters, rejection counters.
   - `frozen_candidate_rankings`: Top units scored via both-context sensitivity on discovery split (pairs 0..24).
   - `representation_statistics`: Effective dimensionality (participation ratio), eigenvalue spectra, cumulative explained variance thresholds, positive-part and signed sparseness, excess kurtosis, and silent units across all 9 stages.
   - `statistical_results`: Primary comparison (Candidate vs Norm-Matched Random) and secondary comparison (Candidate vs Dimension-Matched Random) with paired effect sizes, pair-clustered bootstrap 95% CIs, Wilcoxon signed-rank tests, sign-flip permutation tests, and Holm-Bonferroni adjusted p-values across 20 planned tests.
   - `mechanistic_conclusion`: Formal verdict on whether candidate selectivity is a genuine causal mechanism or an activation-magnitude artefact.

2. `task_conditioned_circuits_v5_2_pairs.jsonl`:
   - One line per independent pair ($N=60$).
   - Contains paired candidate transfer efficiencies, norm-matched random means, dimension-matched random means, and necessity retention/drop ratios.

3. `task_conditioned_circuits_v5_2_interventions.jsonl.gz`:
   - Complete record of EVERY intervention pass (gzipped to comply with GitHub 100MB repository limit) (candidate + all random draws).
   - Fields:
     - `pair_id`: Pair identifier (e.g., `PAIR_000`).
     - `scene_id`: `scene_a` or `scene_b`.
     - `context`: `nominal` or `illumination_shift`.
     - `layer`: `layer1_post_relu_64d` or `layer2_post_relu_64d`.
     - `direction`: `A_to_B` or `B_to_A`.
     - `k`: Number of intervened dimensions (k in [2, 4, 8, 16, 32]).
     - `condition`: `candidate`, `norm_matched_random`, or `dimension_matched_random`.
     - `draw_id`: 0 for candidate; 0..99 for random draws.
     - `units`: List of intervened unit indices.
     - `subset_norm`: Task-induced activation perturbation norm sqrt(sum((delta h_i)^2)).
     - `norm_diff_abs`: |N_rand - N_cand|.
     - `norm_diff_rel`: |N_rand - N_cand| / N_cand.
     - `is_within_tolerance`: True if relative norm error <= 0.05 (5%).
     - `projected_transfer_efficiency`: Projected transfer fraction along task-contrast direction (eta).
     - `transfer_magnitude_m`: Euclidean norm of transferred XY displacement in meters.
     - `directional_cosine`: Cosine alignment with target task-contrast vector.
     - `orthogonal_residual_m`: Transferred vector magnitude orthogonal to intended contrast vector.
     - `heading_transfer_eff`: Heading transfer ratio derived from atan2(sin, cos).

4. `task_conditioned_circuits_v5_2_scene_specs.jsonl`:
   - Exact physical specifications of all 60 scene pairs.
   - Robot pose, target positions, approach waypoints, obstacle dictionary, visibility pixel counts, and rendered camera observation SHA-256 hashes.
