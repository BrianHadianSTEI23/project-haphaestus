# Schema: Phase 5 Ego-Motion Memory Factorial Analysis (v4)

**Artifact Paths:**
- `world_models_fpga/research/circuits/results/ego_memory_mechanism_v4_summary.json`
- `world_models_fpga/research/circuits/results/ego_memory_mechanism_v4_pairs.jsonl`
- `world_models_fpga/research/circuits/results/ego_memory_mechanism_v4_steps.jsonl`

---

## 1. Summary JSON Specification (`ego_memory_mechanism_v4_summary.json`)

| JSON Pointer | Data Type | Physical Units | Description |
|---|---|---|---|
| `/sample_counts/n_trajectories` | `int` | Count | Number of replay trajectories ($N=50$). |
| `/displacement_calibration_offline/all/zero_baseline/translation_mae_m` | `float` | Meters | MAE of zero displacement baseline on physical translation. |
| `/displacement_calibration_offline/all/learned_sensor_displacement/translation_mae_m` | `float` | Meters | MAE of learned sensor displacement head. |
| `/displacement_calibration_offline/all/action_conditioned_displacement/translation_mae_m` | `float` | Meters | MAE of action-conditioned transition predictor displacement. |
| `/factorial_conditions_summary/sensor_fixed/mean_error_m` | `float` | Meters | Mean tracking error under observed sensor delta with fixed 0.65/0.35 blending. |
| `/factorial_conditions_summary/zero_fixed/mean_error_m` | `float` | Meters | Mean tracking error under zero displacement with fixed blending. |
| `/factorial_conditions_summary/action_hold/mean_error_m` | `float` | Meters | Mean tracking error under action-conditioned predictor with hold/freeze memory. |
| `/factorial_conditions_summary/oracle_hold/mean_error_m` | `float` | Meters | Mean tracking error under oracle odometry with hold/freeze memory. |
| `/statistical_hypothesis_tests/sensor_fixed_vs_zero_fixed/tost_p_equiv_0_05m` | `float` | $p$-value | TOST equivalence test $p$-value within 5cm margin. |

---

## 2. Pairs JSONL Specification (`ego_memory_mechanism_v4_pairs.jsonl`)

Each line contains trajectory-level summary for one trajectory ($N=50$ records):
- `trajectory_id`: Integer index $[0, 49]$.
- `target_key`: Designated object identity.
- `blackout_window`: Array $[w_{\text{start}}, w_{\text{end}}]$.
- `condition_errors_m`: Dictionary of mean tracking error across all 12 conditions.

---

## 3. Steps JSONL Specification (`ego_memory_mechanism_v4_steps.jsonl`)

Each line records a single step during the blackout window for detailed transition analysis:
- `trajectory_id`: Integer index.
- `step_idx`: Step index within trajectory.
- `is_first_blackout_transition`: Boolean (true for step 5).
- `latent_diff_norm`: Euclidean norm of latent difference $\|s_t - s_{t-1}\|$.
- `predicted_sensor_displacement`: Decoded $[\Delta x_b, \Delta y_b, \Delta \theta]$ from observed latents.
- `predicted_action_displacement`: Decoded $[\Delta x_b, \Delta y_b, \Delta \theta]$ from action-conditioned prediction.
- `actual_physical_displacement`: Ground-truth physical odometry.
- `static_bias_displacement`: Output of head under zero latent difference.
- `ground_truth_goal`: Ground-truth body-relative goal coordinates $[dx_b, dy_b]$.
- `sensor_fixed_goal`, `sensor_hold_goal`, `action_hold_goal`, `oracle_hold_goal`: Condition outputs.
