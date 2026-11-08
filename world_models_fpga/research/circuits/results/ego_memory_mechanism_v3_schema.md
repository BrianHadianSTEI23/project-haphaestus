# Schema Documentation: Ego-Motion Memory Mechanism Benchmark (v3)

**Artifacts Covered:**
- `ego_memory_mechanism_v3_summary.json`
- `ego_memory_mechanism_v3_pairs.jsonl`
- `ego_memory_mechanism_v3_traces.npz`

---

## 1. Summary JSON Schema (`ego_memory_mechanism_v3_summary.json`)

| JSON Pointer | Data Type | Units / Range | Description |
|---|:---:|:---:|---|
| `/experiment_metadata/source_git_commit` | `string` | Git SHA (40 chars) | Exact commit baseline from which evaluation was launched. |
| `/experiment_metadata/script_sha256` | `string` | Hex digest (64 chars) | SHA-256 hash of `run_ego_memory_mechanism_v3.py`. |
| `/experiment_metadata/checkpoint_sha256` | `string` | Hex digest (64 chars) | SHA-256 of `checkpoints_v4/hjepa_best.pt`. |
| `/sample_counts/n_trajectories_evaluated` | `integer` | Count $\ge 50$ | Number of independent replay trajectories evaluated. |
| `/sample_counts/n_total_steps` | `integer` | Count $\ge 1500$ | Total timesteps recorded across all trajectories. |
| `/micro_ego_head_calibration/translation_mae_m` | `float` | Meters | Mean absolute error of 1-step translation prediction $[\Delta x_b, \Delta y_b]$. |
| `/micro_ego_head_calibration/translation_rmse_m` | `float` | Meters | Root-mean-square error of 1-step translation prediction. |
| `/micro_ego_head_calibration/rotation_mae_rad` | `float` | Radians | Mean absolute error of 1-step yaw rotation prediction $\Delta \theta$. |
| `/micro_ego_head_calibration/rotation_rmse_rad` | `float` | Radians | Root-mean-square error of 1-step yaw rotation prediction. |
| `/micro_ego_head_calibration/translation_bias_m` | `array[float]` | Meters | Mean signed error $[\mathbb{E}(\widehat{\Delta x}_b - \Delta x_b), \mathbb{E}(\widehat{\Delta y}_b - \Delta y_b)]$. |
| `/micro_ego_head_calibration/rotation_bias_rad` | `float` | Radians | Mean signed error $\mathbb{E}(\widehat{\Delta \theta} - \Delta \theta)$. |
| `/micro_ego_head_calibration/zero_delta_static_bias` | `array[float]` | Vector (3D) | Output of `micro_ego_head(0)` when $\Delta s = 0$ (blackout input). |
| `/multi_step_error_accumulation/k1_error_m` | `float` | Meters | K=1 step displacement tracking error. |
| `/multi_step_error_accumulation/k5_error_m` | `float` | Meters | K=5 step cumulative displacement tracking error. |
| `/multi_step_error_accumulation/k10_error_m` | `float` | Meters | K=10 step cumulative displacement tracking error. |
| `/multi_step_error_accumulation/k20_error_m` | `float` | Meters | K=20 step cumulative displacement tracking error. |
| `/same_trajectory_replay_results/cond_A_visual_only/mean_blackout_error_m` | `float` | Meters | Mean tracking error during blackout under visual-only control. |
| `/same_trajectory_replay_results/cond_B_full_memory/mean_blackout_error_m` | `float` | Meters | Mean tracking error during blackout under full learned memory. |
| `/same_trajectory_replay_results/cond_C_frozen_memory/mean_blackout_error_m` | `float` | Meters | Mean tracking error during blackout under frozen pre-blackout memory. |
| `/same_trajectory_replay_results/cond_D_zeroed_ego/mean_blackout_error_m` | `float` | Meters | Mean tracking error during blackout with ego displacement zeroed. |
| `/same_trajectory_replay_results/cond_E_shuffled_ego/mean_blackout_error_m` | `float` | Meters | Mean tracking error during blackout with shuffled ego displacements. |
| `/same_trajectory_replay_results/cond_F_matched_donor/mean_blackout_error_m` | `float` | Meters | Mean tracking error during blackout with matched donor ego displacements. |
| `/same_trajectory_replay_results/cond_G_mismatched_donor/mean_blackout_error_m` | `float` | Meters | Mean tracking error during blackout with mismatched donor ego displacements. |
| `/same_trajectory_replay_results/cond_H_oracle_odometry/mean_blackout_error_m` | `float` | Meters | Mean tracking error during blackout with ground-truth robot odometry. |
| `/preregistered_paired_comparisons` | `array[object]` | Records | Wilcoxon signed-rank tests, Holm-adjusted p-values, paired bootstrap 95% CIs, and TOST equivalence for 5 comparisons. |
| `/blending_ablation_results/mean_error_fixed_nominal` | `float` | Meters | Blackout error under fixed 0.65/0.35 blending. |
| `/blending_ablation_results/mean_error_pure_memory` | `float` | Meters | Blackout error under 0.0/1.0 memory-only blending. |
| `/blending_ablation_results/mean_error_adaptive_blending` | `float` | Meters | Blackout error under confidence-gated adaptive blending. |

---

## 2. Trajectories JSONL Schema (`ego_memory_mechanism_v3_pairs.jsonl`)

Each line represents one complete trajectory replay ($N \ge 50$ records):
```json
{
  "trajectory_id": "TRAJ_000",
  "trajectory_seed": 1000,
  "target_key": "red_box",
  "initial_robot_pose": [x, y, theta],
  "target_waypoint": [x, y],
  "initial_distance_m": 0.75,
  "trajectory_length_steps": 30,
  "blackout_window": [5, 15],
  "calibration_metrics": {
    "mean_trans_mae_m": 0.035,
    "mean_rot_mae_rad": 0.082
  },
  "condition_blackout_tracking_errors_m": {
    "cond_A_visual_only": 0.724,
    "cond_B_full_memory": 0.648,
    "cond_C_frozen_memory": 0.490,
    "cond_D_zeroed_ego": 0.629,
    "cond_E_shuffled_ego": 0.652,
    "cond_F_matched_donor": 0.501,
    "cond_G_mismatched_donor": 0.438,
    "cond_H_oracle_odometry": 0.125
  },
  "blending_ablation_errors_m": {
    "fixed_065_035": 0.648,
    "visual_only_10_00": 0.724,
    "memory_only_00_10": 0.412,
    "adaptive_gated": 0.385
  },
  "duration_sweep_errors_m": {
    "k2_steps": 0.285,
    "k5_steps": 0.432,
    "k10_steps": 0.648,
    "k20_steps": 0.895
  }
}
```

---

## 3. Traces NPZ Schema (`ego_memory_mechanism_v3_traces.npz`)

The NPZ file contains dense step-level arrays for every trajectory and condition:
- `traj_ids`: Array of string identifiers shape `(N,)`.
- `robot_poses`: Array of physical robot poses `(N, T, 3)`.
- `gt_goals_body`: Array of ground-truth body-relative target vectors `(N, T, 4)`.
- `pred_ego_displacements`: Decoded micro-ego displacements `(N, T, 3)`.
- `actual_body_displacements`: Ground-truth physical body displacements `(N, T, 3)`.
- `condition_goals`: Dict/structured array of predicted goal vectors `(N, T, 8, 4)` across the 8 conditions.
- `is_blackout`: Boolean mask `(N, T)` indicating blackout steps.
