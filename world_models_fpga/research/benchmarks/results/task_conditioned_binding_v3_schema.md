# Schema Documentation: Task-Conditioned Visual Binding Benchmark (v3)

**Artifacts Covered:**
- `task_conditioned_binding_v3_summary.json`
- `task_conditioned_binding_v3_pairs.jsonl`
- `task_conditioned_binding_v3_traces.npz`

---

## 1. Summary JSON Schema (`task_conditioned_binding_v3_summary.json`)

| JSON Pointer | Data Type | Units / Range | Description |
|---|:---:|:---:|---|
| `/experiment_metadata/source_git_commit` | `string` | Git SHA (40 chars) | Exact commit baseline from which evaluation was launched. |
| `/experiment_metadata/script_sha256` | `string` | Hex digest (64 chars) | SHA-256 hash of `run_task_conditioned_binding_v3.py`. |
| `/experiment_metadata/checkpoint_sha256` | `string` | Hex digest (64 chars) | SHA-256 of `checkpoints_v4/hjepa_best.pt`. |
| `/sample_counts/n_pairs_evaluated` | `integer` | Count $\ge 50$ | Number of independent scene pairs evaluated. |
| `/sample_counts/n_total_scenes` | `integer` | Count $\ge 100$ | Number of distinct dual-visible scene configurations. |
| `/same_scene_token_steering_results/a_correct_rate` | `float` | $[0.0, 1.0]$ | Fraction of pairs where Token A steers closer to physical Target A than Target B. |
| `/same_scene_token_steering_results/a_correct_binom_p_vs_chance` | `float` | $[0.0, 1.0]$ | Two-sided exact binomial p-value testing $H_0: p = 0.50$. |
| `/same_scene_token_steering_results/b_correct_rate` | `float` | $[0.0, 1.0]$ | Fraction of pairs where Token B steers closer to physical Target B than Target A. |
| `/same_scene_token_steering_results/b_correct_binom_p_vs_chance` | `float` | $[0.0, 1.0]$ | Two-sided exact binomial p-value testing $H_0: p = 0.50$. |
| `/same_scene_token_steering_results/dual_discrimination_rate` | `float` | $[0.0, 1.0]$ | Conjunction: fraction where Token A $\to$ A AND Token B $\to$ B. |
| `/same_scene_token_steering_results/dual_discrimination_binom_p_vs_chance` | `float` | $[0.0, 1.0]$ | Exact binomial test against 25% chance level ($0.5 \times 0.5$). |
| `/same_scene_token_steering_results/mean_angular_shift_rad` | `float` | $[0, \pi]$ rad | Mean angular rotation between $g(z_A, e_A)$ and $g(z_A, e_B)$ under identical RGB. |
| `/interaction_analysis/mean_interaction_magnitude` | `float` | Meters | Mean Euclidean norm of interaction vector $\|I\| = \|(g_{BB}-g_{BA})-(g_{AB}-g_{AA})\|$. |
| `/interaction_analysis/mean_alignment_with_physical_displacement` | `float` | $[-1.0, 1.0]$ | Mean cosine alignment between interaction vector and physical object swap $\Delta P$. |
| `/model_comparison_held_out/model_1_additive_task_bias_mse` | `float` | $\text{m}^2$ | Held-out MSE of additive task model $g(z, e) = a(z) + b(e)$. |
| `/model_comparison_held_out/model_2_visual_geometry_selection_mse` | `float` | $\text{m}^2$ | Held-out MSE of pure visual object selection model. |
| `/model_comparison_held_out/model_3_general_nonlinear_interaction_mse` | `float` | $\text{m}^2$ | Held-out MSE of bilinear / general interaction model. |
| `/saliency_factorial_results/wilcoxon_p_saliency_bias` | `float` | $[0.0, 1.0]$ | Wilcoxon signed-rank test comparing error to Target A when A is less vs more salient. |

---

## 2. Pairs JSONL Schema (`task_conditioned_binding_v3_pairs.jsonl`)

Each line is a JSON object with schema:
```json
{
  "pair_id": "string (PAIR_000)",
  "scene_seed": "integer",
  "object_identities": ["string (key_A)", "string (key_B)"],
  "physical_positions": [[x1, y1, z1], [x2, y2, z2]],
  "approach_waypoints": [[wx1, wy1], [wx2, wy2]],
  "initial_robot_pose": [rx, ry, rth],
  "image_sha256": ["sha256_scene_a", "sha256_scene_b"],
  "object_segmentation_pixels": [[pix_a1, pix_b1], [pix_a2, pix_b2]],
  "image_centroids": [[[u_a1, v_a1], [u_b1, v_b1]], [[u_a2, v_a2], [u_b2, v_b2]]],
  "task_ids": [id_a, id_b],
  "factorial_outputs": {
    "g_aa": [dx, dy, sin_th, cos_th],
    "g_ab": [dx, dy, sin_th, cos_th],
    "g_ba": [dx, dy, sin_th, cos_th],
    "g_bb": [dx, dy, sin_th, cos_th]
  },
  "ground_truth_body_goals": {
    "gt_a_scene_a": [dx, dy, sin_th, cos_th],
    "gt_b_scene_a": [dx, dy, sin_th, cos_th],
    "gt_a_scene_b": [dx, dy, sin_th, cos_th],
    "gt_b_scene_b": [dx, dy, sin_th, cos_th]
  },
  "same_scene_discrimination": {
    "d_aa_to_target_a": "float",
    "d_aa_to_target_b": "float",
    "d_ab_to_target_b": "float",
    "d_ab_to_target_a": "float",
    "a_correct": "boolean",
    "b_correct": "boolean",
    "dual_discrimination": "boolean",
    "angular_shift_rad": "float"
  },
  "interaction_analysis": {
    "delta_a": [dx, dy],
    "delta_b": [dx, dy],
    "interaction_vec": [dx, dy],
    "interaction_magnitude": "float",
    "alignment_with_physical_displacement": "float"
  },
  "saliency_factorial": {
    "saliency_a_less_b": {"vis_a_pixels": "int", "vis_b_pixels": "int", "g_sal_token_a": [...], "d_to_target_a": "float"},
    "saliency_a_more_b": {"vis_a_pixels": "int", "vis_b_pixels": "int", "g_sal_token_a": [...], "d_to_target_a": "float"}
  }
}
```

---

## 3. Traces NPZ Schema (`task_conditioned_binding_v3_traces.npz`)

Compressed dictionary storing raw floating-point arrays:
- `pair_{idx:03d}_z_a`: 32D visual latent vector from Scene A.
- `pair_{idx:03d}_z_b`: 32D visual latent vector from Scene B.
- `pair_{idx:03d}_g_aa`: 4D goal vector $[dx, dy, \sin\theta, \cos\theta]$ for Scene A with Token A.
- `pair_{idx:03d}_g_ab`: 4D goal vector for Scene A with Token B.
- `pair_{idx:03d}_g_ba`: 4D goal vector for Scene B with Token A.
- `pair_{idx:03d}_g_bb`: 4D goal vector for Scene B with Token B.
