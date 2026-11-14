# Phase 5.6 Stage B2 — Coordinate Frame, Camera Extrinsics, Controller, & Training Distribution Audit

**Raw Artifact:** [task_conditioned_stagewise_binding_v5_6_coordinate_audit.json](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/benchmarks/results/task_conditioned_stagewise_binding_v5_6_coordinate_audit.json)

---

## 1. Simulator & Camera Coordinate Verification (`all_synthetic_coordinate_checks_passed = true`)

All synthetic MuJoCo geometric tests passed bit-for-bit consistency without sign inversions:

| Synthetic Check | Setup | Body-Frame Goal `(dx_b, dy_b)` | Roundtrip Err | Camera `front_cam` Pixel `(u, v)` | Side Consistency |
|---|---|---|---|---|---|
| `pure_body_left` | Robot `(0,0,0)`, target `(1.35, +0.55)` | `(+0.7943, +0.3236)` | `1.97e-8 m` | `u = 8.69 px`, `696 px` | `left == left` |
| `pure_body_right` | Robot `(0,0,0)`, target `(1.35, -0.55)` | `(+0.7943, -0.3236)` | `1.97e-8 m` | `u = 54.31 px`, `696 px` | `right == right` |
| `forward_left` | Robot `(0,0,0)`, target `(1.50, +0.35)` | `(+0.9157, +0.2137)` | `2.40e-8 m` | `u = 16.78 px`, `704 px` | `left == left` |
| `forward_right` | Robot `(0,0,0)`, target `(1.50, -0.35)` | `(+0.9157, -0.2137)` | `2.40e-8 m` | `u = 46.23 px`, `703 px` | `right == right` |
| `rotated_robot_fixed_body_left` (`0, +pi/2, pi, -pi/2`) | Fixed `+0.45 m` body-left offset across 4 world headings | `(+0.8288, +0.2664)` across all 4 headings | `6.79e-9 m` | `u = 10.98..11.11 px` (`840..843 px`) | `4/4 left` |
| `mirrored_robot_single_object` | Single world object `(1.5, 0)` observed from `(0, -0.45, 0)` vs `(0, +0.45, 0)` | `+0.2776 m` vs `-0.2776 m` | `6.94e-9 m` | `u_L = 11.79 px`, `u_R = 51.21 px` (`sum = 63.00 px`) | Exact anti-symmetry |
| `swapped_category_identical_geometry` | `red_box` vs `blue_cylinder` at `(1.35, +0.55)` | Identical `(+0.7943, +0.3236)` | `1.97e-8 m` | `u = 9.57 px` vs `8.69 px` | Identical body target |

### Simulator Action Signs (`JetAutoEnv.step`)
- Commanding `[0.0, +0.5, 0.0]` (`+vy`) for 3 steps from `(0, 0, 0)` shifts world `y` by **`+0.0402 m`** (`vy_pos_moves_left_ok = true`).
- Commanding `[0.0, 0.0, +0.6]` (`+wz`) for 3 steps rotates heading by **`+0.2365 rad`** (`wz_pos_rotates_ccw_ok = true`).

---

## 2. Downstream Controller Forced-Goal Symmetry Check

With fixed neutral initial image and `track="waypoint"` explicit body-frame goals `[+1.0, +0.45, 0, 1]` (left) vs `[+1.0, -0.45, 0, 1]` (right):
- **`model.predict_reactive_action`:**
  - Forced Left: `[+0.0359, +0.3224, +0.0656]` (`+vy`, `+wz` — moves/rotates left).
  - Forced Right: `[+0.1020, -0.0118, -0.0743]` (`-vy`, `-wz` — moves/rotates right).
- **Full Hierarchical Planner (`HJEPAAgent`, `track="waypoint"`, `ablation_mode="full"`):**
  - Forced Left: `[+0.6500, +0.5500, 0.0000]` (`hierarchical_left_steers_left_ok = true`).
  - Forced Right: `[+0.9085, -0.1774, -0.0299]` (`hierarchical_right_steers_right_ok = true`).

Thus, the downstream controller obeys body-frame lateral signs cleanly and does **not** explain the 5.4x rightward reach skew (`45.0% vs 8.3%`). Notice that when forced left, the controller in fact emits a larger lateral strafe magnitude (`+0.550`) than right (`-0.177`), reinforcing that the observed behavioral rightward dominance arises **upstream** in `grounding_head` goal prediction (`g_vis`), not from controller sign inversion.

---

## 3. Read-Only Audit of Training Data (`train_benchmark.npz`)

Inspecting `world_models_fpga/research/datasets/splits_v4/train_benchmark.npz` (`sha256: 8ba362a4...`, `408` episodes, `13,188` transitions, **`96` unique object layouts**):
- **All relations combined (`n = 13,188`):** `bx_mean = +0.6557 m`, `by_mean = +0.0310 m` (`46.5%` Left / `53.5%` Right).
- **Relation `approach` ONLY (`136` episodes, `4,154` transitions — the exact relation tested in Phase 5.5/5.5.1/5.6):**
  - Overall `approach`: `by_mean = -0.1890 m`, **`75.9%` Right (`by < 0`) vs `24.1%` Left (`by > 0`)**.
  - `blue_cylinder` (`n = 538`): **`97.6%` Right** (`by_mean = -0.3456 m`).
  - `green_cylinder` (`n = 524`): **`94.7%` Right** (`by_mean = -0.3548 m`).
  - `yellow_cylinder` (`n = 1,034`): **`73.3%` Right** (`by_mean = -0.1621 m`).
  - `red_box` (`n = 1,027`): **`71.2%` Right** (`by_mean = -0.1567 m`).
  - `cyan_cylinder` (`n = 511`): **`63.2%` Right** (`by_mean = -0.1559 m`).
  - `purple_box` (`n = 520`): **`61.3%` Right** (`by_mean = -0.0095 m`).

### Consequence
The simulator, MuJoCo `front_cam`, `compute_body_frame_goal_vec`, training pipeline transform, and downstream controller share an identical, sign-consistent right-handed body frame. Conversely, the frozen training split's `"approach"` episodes are heavily skewed toward body-right goals (`75.9%` overall; `>94%` for `blue_cylinder` and `green_cylinder`) across only `96` repeated object layouts.
