# Schema: Phase 5 Internal Mechanistic Circuit Analysis (v4)

**Artifact Paths:**
- `world_models_fpga/research/circuits/results/task_conditioned_circuits_v4_summary.json`
- `world_models_fpga/research/circuits/results/task_conditioned_circuits_v4_pairs.jsonl`
- `world_models_fpga/research/circuits/results/task_conditioned_circuits_v4_activations.jsonl`

---

## 1. Activations JSONL Specification (`task_conditioned_circuits_v4_activations.jsonl`)

Each line corresponds to one independent scene pair ($N=50$ lines, valid JSON):
- `pair_id`: Unique integer $[0, 49]$.
- `task_ids`: Array of two task IDs $[e_A, e_B]$.
- `conditions`: Dictionary with 4 factorial keys (`cond_aa`, `cond_ab`, `cond_ba`, `cond_bb`).
  Each condition records full intermediate activation vectors for all 9 stages:
  - `visual_latent_z`: `array[float]`, shape $(32,)$.
  - `task_embedding_e`: `array[float]`, shape $(16,)$.
  - `input_concat_48d`: `array[float]`, shape $(48,)$.
  - `layer1_pre_relu_64d`: `array[float]`, shape $(64,)$.
  - `layer1_post_relu_64d`: `array[float]`, shape $(64,)$.
  - `layer2_pre_relu_64d`: `array[float]`, shape $(64,)$.
  - `layer2_post_relu_64d`: `array[float]`, shape $(64,)$.
  - `raw_output_4d`: `array[float]`, shape $(4,)$.
  - `postprocessed_goal_4d`: `array[float]`, shape $(4,)$.

---

## 2. Summary JSON Specification (`task_conditioned_circuits_v4_summary.json`)

| JSON Pointer | Data Type | Physical Units | Description |
|---|---|---|---|
| `/sample_counts/total_pairs_instrumented` | `int` | Count | Total pairs evaluated ($N=50$). |
| `/sample_counts/discovery_split_pairs` | `int` | Count | Discovery split sample count ($25$). |
| `/sample_counts/held_out_evaluation_pairs` | `int` | Count | Held-out split sample count ($25$). |
| `/discovery_split_analysis/layer_1/effective_dimensionality` | `float` | Dimension | Participation ratio of covariance eigenvalues for Layer 1. |
| `/discovery_split_analysis/layer_2/effective_dimensionality` | `float` | Dimension | Participation ratio of covariance eigenvalues for Layer 2. |
| `/held_out_causal_interventions/layer_2/k_8/sufficiency_te_cand_mean` | `float` | Fraction $[0, 1]$ | Transfer efficiency transplanting candidate top-8 units. |
| `/held_out_causal_interventions/layer_2/k_8/sufficiency_te_rand_mean` | `float` | Fraction $[0, 1]$ | Transfer efficiency transplanting matched random 8 units. |
| `/held_out_causal_interventions/layer_2/k_8/sufficiency_wilcoxon_p_cand_vs_rand` | `float` | $p$-value | Paired Wilcoxon signed-rank test comparing candidate vs random subset. |
| `/modularity_determination/organization_status` | `string` | Categorical | `CONFIRMED_DISTRIBUTED_ORGANIZATION`. |

---

## 3. Pairs JSONL Specification (`task_conditioned_circuits_v4_pairs.jsonl`)

Each line contains held-out causal intervention results for one pair ($N=25$ lines):
- `pair_id`: Unique integer $[25, 49]$.
- `clean_donor_diff_m`: Baseline Euclidean distance between recipient and donor goal vectors.
- `layer_interventions`: Nested results across `layer_1` and `layer_2` for $k \in [2, 4, 8, 16, 32, 64]$, reporting necessity shift, sufficiency transfer efficiency, rescue percentage, and random control outcomes.
