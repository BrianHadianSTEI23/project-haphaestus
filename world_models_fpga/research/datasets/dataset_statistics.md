# Phase 1: Benchmark Dataset Integrity & Quantitative Statistics

**Author:** Brian A. Hadian  
**Project:** Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA  
**Target Platform:** AMD Kria KR260 Robotics Starter Kit (Zynq UltraScale+ XCK26 SOM)  
**Host Architecture:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.6, EGL Headless Renderer)  
**Reference Directive:** Section 3 (Phase 1) & Master Research Directive in `research_direction.md`

---

## 1. Quantitative Corpus Summary

The Phase 1 Controlled Intelligence Benchmark corpus comprises **1,050 high-fidelity episodes** and **33,895 structured transitions**, captured deterministically in MuJoCo at 10 Hz control frequency ($dt = 0.1\,\text{s}$).

| Split Identifier | Episodes | Transitions | Compressed Size | Avg. Episode Length | Collision Rate | Primary Objective |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `train_benchmark` | 408 | 12,595 | 59.72 MB | 30.9 steps | 23.45% | Training SAEs, diagnostic probes, and baselines |
| `val_benchmark` | 96 | 2,911 | 13.51 MB | 30.3 steps | 17.66% | Validation loss tracking and tuning |
| `held_out_test_benchmark` | 96 | 2,885 | 14.37 MB | 30.1 steps | 16.71% | Zero-shot held-out environment benchmark |
| `counterfactual_suite` | 350 | 12,250 | 61.78 MB | 35.0 steps | 7.35% | Causal intervention pairs (E1 – E7) |
| `disambiguation_suite` | 100 | 3,254 | 16.40 MB | 32.5 steps | 7.65% | Resolving frame-of-reference & scale ambiguity |
| **Total Benchmark Corpus** | **1,050** | **33,895** | **165.78 MB** | **32.3 steps** | **14.52%** | **Full Mechanistic Discovery Suite** |

---

## 2. Capability Distribution Breakdown

Each capability is tested across all 5 difficulty tiers (Easy, Difficult, Counterfactual, OOD, Failure/Baseline Collision):

| Capability Category | Base Episodes | Transitions | Target Variables & Focus |
| :--- | :---: | :---: | :--- |
| **Object Grounding** | 100 | 1,975 | 6 Landmark objects, color-shape token binding |
| **Spatial Reasoning** | 100 | 3,159 | 4 Prepositions (*behind*, *in front*, *left of*, *right of*) |
| **Obstacle Avoidance**| 100 | 3,425 | Intervening obstacles, corridor clearance, reactive steering |
| **Action Prediction** | 100 | 3,332 | 8 Canonical 3-DOF Mecanum impulse probes |
| **Temporal Planning** | 100 | 3,000 | Multi-waypoint sequential routes ($W_1 \to W_2 \to W_3$) |
| **Generalization** | 100 | 3,500 | Permuted arena layouts, 4 heading quadrants |
| **Counterfactual (E1-E7)** | 350 | 12,250 | Matched pair physical & perceptual mutations |
| **Disambiguation** | 100 | 3,254 | Frame of reference (Ego vs Allocentric) |
| **Total** | **1,050** | **33,895** | **Comprehensive Benchmark Coverage** |

---

## 3. Data Integrity & Verification Checklist

Automated verification was conducted using [verify_dataset.py](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/datasets/verify_dataset.py). All checks passed:

- [x] **Tensor Dimension Integrity**: All 33,895 transitions strictly adhere to:
  - `rgb`: $(64, 64, 3)$ uint8
  - `next_rgb`: $(64, 64, 3)$ uint8
  - `robot_pose`: $(3,)$ float32 $[x, y, \theta]$
  - `robot_velocity`: $(3,)$ float32 $[v_x, v_y, \omega_z]$
  - `object_positions`: $(6, 3)$ float32
  - `object_classes`: $(6,)$ int32
  - `object_colors`: $(6, 4)$ float32 RGBA
  - `actions`: $(3,)$ float32 $[v_x, v_y, \omega_z]$
  - `next_actions`: $(3,)$ float32
  - `collisions`: bool
  - `goals_reached`: bool
  - `episode_successes`: bool
  - `episode_starts`: bool
- [x] **Numerical Stability**: 0 NaN, 0 Inf values across all kinematic and action channels.
- [x] **Zero Observation Leakage**: Split assignment is executed by environment configuration seeds; frames from the same episode never span across multiple splits.
- [x] **Sensor-State Decoupling**: Ground-truth physical poses and object coordinates are isolated for mechanistic analysis and diagnostic probing; models only receive camera pixels and text tokens.
- [x] **Deterministic Reproducibility**: All episodes generated from seed base `seed=42` with explicit per-episode RNG offsets recorded in companion `.json` metadata.

---

## 4. Phase 1 Acceptance Criteria Evaluation

In accordance with Phase 1 acceptance criteria defined in `research_direction.md`:

| Requirement | Target | Achieved Status | Evidence |
| :--- | :---: | :---: | :--- |
| **Reproducible Episodes** | $\ge 100$ per capability | **100 - 350 per capability** | 1,050 episodes total across 6 capabilities + E1-E7 |
| **Controlled Variations** | Position, heading, appearance, obstacles | **Verified across 5 tiers** | Easy, Difficult, Counterfactual, OOD, Failure tiers |
| **Split Separation** | Independent Train, Val, Held-out Test | **Completed & Sharded** | `train_benchmark`, `val_benchmark`, `held_out_test_benchmark` |
| **Ground-Truth Physical State** | Recorded for post-hoc analysis | **100% Captured** | Poses, velocities, 6-object 3D coordinates, and colors |

**Status:** Phase 1 implementation and data generation fully satisfied. Ready for Beamer presentation compilation.
