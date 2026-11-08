"""
Track C Pure Visual Spatial Grounding Benchmark (Scientifically Repaired Protocol)

Strict Information Regime:
- Track C ('observation'): The agent receives ONLY camera RGB (64x64x3) and discrete task ID.
- Zero simulator coordinates ([dx, dy, sin_th, cos_th] strictly withheld).
- Evaluates five paired counterfactual test suites against matched nominal episodes:
  1. Nominal Visual Grounding: Baseline canonical tasks.
  2. Appearance Nuisance: Floor texture and illumination altered; geometry and semantics held constant.
  3. Semantically Relevant Appearance: Target color swapped with distractor; tests visual grounding vs location memorization.
  4. Visual Geometry Grounding: Physical landmark positions swapped; tests visual steering vs allocentric prior.
  5. Layout Generalization: Non-canonical randomized obstacle layouts.
  6. Rigid SE(2) Transformations: Global rigid rotation of robot pose, all obstacles, and target waypoints.
- Reports exact contingency tables, paired differences, McNemar exact tests, 95% bootstrap CIs,
  and saves full episode-level raw records.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Add project root and world_models_fpga
repo_root = Path(__file__).resolve().parents[3]
pkg_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(pkg_root))

import numpy as np
import torch
from scipy.stats import binom

from env.amr_env import JetAutoEnv
from models.h_jepa import HJEPAModel
from models.le_world_model import LeWorldModel
from models.baseline_policy import BaselineNavigationPolicy
from research.benchmarks.model_wrappers import BaselineAgent, HJEPAAgent, LeWMAgent
from research.datasets.generation.capability_tasks import generate_task_specification, CANONICAL_OBSTACLES


def _sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _git_clean() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return len(out) == 0
    except Exception:
        return False


def _paired_contingency(nom_succ: list[bool], mut_succ: list[bool]) -> dict:
    """Computes exact 2x2 contingency table and McNemar exact binomial p-value."""
    n11 = sum(1 for n, m in zip(nom_succ, mut_succ) if n and m)
    n00 = sum(1 for n, m in zip(nom_succ, mut_succ) if not n and not m)
    n10 = sum(1 for n, m in zip(nom_succ, mut_succ) if n and not m)  # nom succ, mut fail
    n01 = sum(1 for n, m in zip(nom_succ, mut_succ) if not n and m)  # nom fail, mut succ

    discordant = n10 + n01
    if discordant == 0:
        p_val = 1.0
    else:
        k = min(n10, n01)
        p_val = float(min(1.0, 2.0 * binom.cdf(k, discordant, 0.5)))

    diffs = np.array([float(m) - float(n) for n, m in zip(nom_succ, mut_succ)], dtype=np.float32)
    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(42)
    if len(diffs) > 1:
        boot_diffs = [np.mean(rng.choice(diffs, size=len(diffs), replace=True)) for _ in range(2000)]
        ci_low = float(np.percentile(boot_diffs, 2.5))
        ci_high = float(np.percentile(boot_diffs, 97.5))
    else:
        ci_low, ci_high = mean_diff, mean_diff

    return {
        "n11_both_success": n11,
        "n00_both_fail": n00,
        "n10_nom_succ_mut_fail": n10,
        "n01_nom_fail_mut_succ": n01,
        "discordant_count": discordant,
        "mcnemar_exact_p_value": p_val,
        "mean_diff": mean_diff,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def evaluate_suite_track_c(agent, specs: list[dict], max_steps: int = 50) -> dict:
    """Evaluates an agent strictly under Track C (zero privileged coordinates)."""
    env = JetAutoEnv()
    records = []
    for spec in specs:
        env.restore_physical_baseline()
        env.apply_task_spec(spec)
        init_pos = np.asarray(spec["initial_pose"], dtype=np.float32)
        obs, _ = env.reset(options={"initial_pos": (float(init_pos[0]), float(init_pos[1]), float(init_pos[2]))})
        final_wp = np.asarray(spec.get("waypoints", [init_pos])[-1], dtype=np.float32)

        if hasattr(agent, "reset"):
            agent.reset()

        collision_hit = False
        goal_reached = False
        prev_xy = init_pos[:2].copy()
        total_path = 0.0

        for _ in range(max_steps):
            curr_pose = env.get_robot_pose()
            dist = float(np.linalg.norm(curr_pose[:2] - final_wp[:2]))
            if dist <= 0.28:
                goal_reached = True
                break

            # In Track C, pass ONLY obs and task_spec (with discrete instructions).
            public_spec = {
                "target_key": spec.get("target_key", ""),
                "semantic_relation": spec.get("semantic_relation", "approach"),
                "instruction": spec.get("instruction", ""),
            }
            act_np, _ = agent.act(obs, robot_pose=None, target_pose=None, task_spec=public_spec, goal_vec_body=None)

            step_res = env.step(act_np)
            obs = step_res[0]
            info = step_res[-1]
            if info.get("collision", False):
                collision_hit = True
            curr_xy = env.get_robot_pose()[:2]
            total_path += float(np.linalg.norm(curr_xy - prev_xy))
            prev_xy = curr_xy.copy()

        final_p = env.get_robot_pose()
        final_err = float(np.linalg.norm(final_p[:2] - final_wp[:2]))
        succ = bool(goal_reached and not collision_hit)
        records.append({
            "episode_id": spec.get("episode_id", "ep"),
            "target_key": spec.get("target_key", ""),
            "success": succ,
            "goal_reached": goal_reached,
            "collision": collision_hit,
            "final_goal_error_m": final_err,
            "path_length_m": total_path,
        })
    env.close()

    n = len(records)
    n_succ = sum(1 for r in records if r["success"])
    n_goal = sum(1 for r in records if r["goal_reached"])
    n_col = sum(1 for r in records if r["collision"])
    mean_err = float(np.mean([r["final_goal_error_m"] for r in records]))

    return {
        "episodes_evaluated": n,
        "success_count": n_succ,
        "success_rate": float(n_succ / max(1, n)),
        "goal_reached_count": n_goal,
        "goal_reached_rate": float(n_goal / max(1, n)),
        "collision_count": n_col,
        "collision_rate": float(n_col / max(1, n)),
        "mean_final_goal_error_m": mean_err,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Track C Pure Visual Spatial Grounding Benchmark")
    parser.add_argument("--episodes-per-suite", type=int, default=20, help="Number of episodes per test suite (N >= 20)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=str(pkg_root / "research/benchmarks/results/track_c_visual_grounding.json"))
    args = parser.parse_args()

    ckpt_dir = pkg_root / "checkpoints_v4"
    hjepa_path = ckpt_dir / "hjepa_best.pt"
    lewm_path = ckpt_dir / "lewm_best.pt"
    base_path = ckpt_dir / "baseline_best.pt"
    test_npz = pkg_root / "research/datasets/splits_v4/held_out_test_benchmark.npz"

    print("=================================================================")
    print("  TRACK C PURE VISUAL SPATIAL GROUNDING BENCHMARK")
    print(f"  Device: {args.device} | Checkpoint: {hjepa_path.name}")
    print(f"  Episodes per suite: {args.episodes_per_suite} | Privileged coordinates: ZERO")
    print("=================================================================")

    # Initialize Track C agents
    agent_h = HJEPAAgent(str(hjepa_path), device=args.device, track="observation")
    agent_l = LeWMAgent(str(lewm_path), device=args.device, track="observation")
    agent_b = BaselineAgent(str(base_path), device=args.device, track="observation")

    n_eps = args.episodes_per_suite

    # --- 1. Nominal Track C Grounding Suite ---
    print("\n--- 1. Evaluating Nominal Track C Visual Grounding ---")
    nominal_specs = []
    for i in range(n_eps):
        spec = generate_task_specification("object_grounding", "easy", seed_idx=6100 + i)
        spec["episode_id"] = f"nom_{i:02d}"
        nominal_specs.append(spec)

    res_nom_h = evaluate_suite_track_c(agent_h, nominal_specs)
    res_nom_l = evaluate_suite_track_c(agent_l, nominal_specs)
    res_nom_b = evaluate_suite_track_c(agent_b, nominal_specs)
    print(f"  [H-JEPA Nominal]   Succ: {res_nom_h['success_count']}/{n_eps} ({res_nom_h['success_rate']*100:5.1f}%) | Col: {res_nom_h['collision_rate']*100:5.1f}% | GoalErr: {res_nom_h['mean_final_goal_error_m']:.3f}m")
    print(f"  [LeWM Nominal]     Succ: {res_nom_l['success_count']}/{n_eps} ({res_nom_l['success_rate']*100:5.1f}%) | Col: {res_nom_l['collision_rate']*100:5.1f}% | GoalErr: {res_nom_l['mean_final_goal_error_m']:.3f}m")
    print(f"  [Baseline Nominal] Succ: {res_nom_b['success_count']}/{n_eps} ({res_nom_b['success_rate']*100:5.1f}%) | Col: {res_nom_b['collision_rate']*100:5.1f}% | GoalErr: {res_nom_b['mean_final_goal_error_m']:.3f}m")

    nom_h_succ = [r["success"] for r in res_nom_h["records"]]

    # --- 2. Appearance Nuisance: Floor Texture & Illumination ---
    print("\n--- 2. Evaluating Appearance Nuisance (Floor & Lighting) ---")
    nuisance_specs = []
    for i in range(n_eps):
        spec = copy.deepcopy(nominal_specs[i])
        spec["episode_id"] = f"nuisance_{i:02d}"
        spec["floor_rgba"] = [0.15, 0.20, 0.35, 1.0]  # dark slate/blue floor
        spec["light_diffuse_scale"] = 0.7  # 30% ambient lighting drop
        nuisance_specs.append(spec)

    res_nuis_h = evaluate_suite_track_c(agent_h, nuisance_specs)
    nuis_h_succ = [r["success"] for r in res_nuis_h["records"]]
    pair_nuis = _paired_contingency(nom_h_succ, nuis_h_succ)
    print(f"  [H-JEPA Nuisance]  Succ: {res_nuis_h['success_count']}/{n_eps} ({res_nuis_h['success_rate']*100:5.1f}%) | Delta: {pair_nuis['mean_diff']*100:+5.1f}% [CI95: {pair_nuis['ci95_low']*100:+5.1f}%, {pair_nuis['ci95_high']*100:+5.1f}%] | p={pair_nuis['mcnemar_exact_p_value']:.4f}")

    # --- 3. Semantically Relevant Appearance: Color Swap with Distractor ---
    print("\n--- 3. Evaluating Semantically Relevant Appearance (Color Swap) ---")
    sem_specs = []
    for i in range(n_eps):
        spec = copy.deepcopy(nominal_specs[i])
        spec["episode_id"] = f"sem_color_{i:02d}"
        target = spec.get("target_key", "yellow_cylinder")
        # Swap colors between target cylinder (yellow) and distractor cylinder (blue)
        # Target cylinder painted blue, distractor cylinder painted yellow
        spec["color_mutations"] = {
            "obs_yellow_cyl": [0.2, 0.4, 0.85, 1.0],  # blue
            "obs_blue_cyl": [0.95, 0.85, 0.1, 1.0],   # yellow
        }
        # Update waypoint to the cylinder that is now yellow (obs_blue_cyl pos)
        if "obs_blue_cyl" in [o.get("geom_name") for o in spec.get("obstacles", {}).values()]:
            for o in spec["obstacles"].values():
                if o.get("geom_name") == "obs_blue_cyl":
                    spec["waypoints"] = [o["pos"].copy()]
        sem_specs.append(spec)

    res_sem_h = evaluate_suite_track_c(agent_h, sem_specs)
    sem_h_succ = [r["success"] for r in res_sem_h["records"]]
    pair_sem = _paired_contingency(nom_h_succ, sem_h_succ)
    print(f"  [H-JEPA Sem Color] Succ: {res_sem_h['success_count']}/{n_eps} ({res_sem_h['success_rate']*100:5.1f}%) | Delta: {pair_sem['mean_diff']*100:+5.1f}% [CI95: {pair_sem['ci95_low']*100:+5.1f}%, {pair_sem['ci95_high']*100:+5.1f}%] | p={pair_sem['mcnemar_exact_p_value']:.4f}")

    # --- 4. Visual Geometry Grounding: Landmark Physical Rearrangement ---
    print("\n--- 4. Evaluating Visual Geometry Grounding (Physical Landmark Swap) ---")
    rearrange_specs = []
    for i in range(n_eps):
        spec = copy.deepcopy(nominal_specs[i])
        spec["episode_id"] = f"rearr_{i:02d}"
        # Physically swap target yellow_cylinder and distractor red_box positions
        obs = spec.get("obstacles", {})
        if "yellow_cylinder" in obs and "red_box" in obs:
            pos_y = obs["yellow_cylinder"]["pos"].copy()
            pos_r = obs["red_box"]["pos"].copy()
            obs["yellow_cylinder"]["pos"] = pos_r
            obs["red_box"]["pos"] = pos_y
            spec["waypoints"] = [pos_r.copy()]  # target is physically at pos_r now
        rearrange_specs.append(spec)

    res_rearr_h = evaluate_suite_track_c(agent_h, rearrange_specs)
    rearr_h_succ = [r["success"] for r in res_rearr_h["records"]]
    pair_rearr = _paired_contingency(nom_h_succ, rearr_h_succ)
    print(f"  [H-JEPA Rearrange] Succ: {res_rearr_h['success_count']}/{n_eps} ({res_rearr_h['success_rate']*100:5.1f}%) | Delta: {pair_rearr['mean_diff']*100:+5.1f}% [CI95: {pair_rearr['ci95_low']*100:+5.1f}%, {pair_rearr['ci95_high']*100:+5.1f}%] | p={pair_rearr['mcnemar_exact_p_value']:.4f} | GoalErr: {res_rearr_h['mean_final_goal_error_m']:.3f}m")

    # --- 5. Layout Generalization: Randomized Obstacle Layouts ---
    print("\n--- 5. Evaluating Layout Generalization (Randomized Configurations) ---")
    rng_layout = np.random.default_rng(777)
    rand_specs = []
    for i in range(n_eps):
        spec = copy.deepcopy(nominal_specs[i])
        spec["episode_id"] = f"rand_layout_{i:02d}"
        for k, o_dict in spec.get("obstacles", {}).items():
            delta_xy = rng_layout.uniform(-0.35, 0.35, size=2)
            o_dict["pos"][:2] = np.clip(o_dict["pos"][:2] + delta_xy, -1.8, 1.8)
        # Update target waypoint to new target position
        t_key = spec.get("target_key", "yellow_cylinder")
        if t_key in spec.get("obstacles", {}):
            spec["waypoints"] = [spec["obstacles"][t_key]["pos"].copy()]
        rand_specs.append(spec)

    res_rand_h = evaluate_suite_track_c(agent_h, rand_specs)
    rand_h_succ = [r["success"] for r in res_rand_h["records"]]
    pair_rand = _paired_contingency(nom_h_succ, rand_h_succ)
    print(f"  [H-JEPA Layout]    Succ: {res_rand_h['success_count']}/{n_eps} ({res_rand_h['success_rate']*100:5.1f}%) | Delta: {pair_rand['mean_diff']*100:+5.1f}% [CI95: {pair_rand['ci95_low']*100:+5.1f}%, {pair_rand['ci95_high']*100:+5.1f}%] | p={pair_rand['mcnemar_exact_p_value']:.4f}")

    # --- 6. Rigid SE(2) Global Transformation (90, 180, 270 deg) ---
    print("\n--- 6. Evaluating Rigid SE(2) Global Transformations ---")
    se2_angles = [np.pi / 2, np.pi, -np.pi / 2]
    se2_results = {}
    for ang in se2_angles:
        deg = int(round(np.degrees(ang)))
        c_a, s_a = np.cos(ang), np.sin(ang)
        R = np.array([[c_a, -s_a], [s_a, c_a]], dtype=np.float32)

        se2_specs = []
        for i in range(n_eps):
            spec = copy.deepcopy(nominal_specs[i])
            spec["episode_id"] = f"se2_{deg:+d}_{i:02d}"
            # Transform initial pose
            p = spec["initial_pose"]
            new_xy = R @ p[:2]
            spec["initial_pose"] = np.array([new_xy[0], new_xy[1], p[2] + ang], dtype=np.float32)

            # Transform all obstacles
            for k, o_dict in spec.get("obstacles", {}).items():
                o_xy = R @ o_dict["pos"][:2]
                o_dict["pos"] = np.array([o_xy[0], o_xy[1], o_dict["pos"][2]], dtype=np.float32)

            # Transform waypoints
            wps = []
            for wp in spec.get("waypoints", []):
                w_xy = R @ wp[:2]
                wps.append(np.array([w_xy[0], w_xy[1], wp[2]], dtype=np.float32))
            spec["waypoints"] = wps
            se2_specs.append(spec)

        res_se2_h = evaluate_suite_track_c(agent_h, se2_specs)
        se2_succ = [r["success"] for r in res_se2_h["records"]]
        pair_se2 = _paired_contingency(nom_h_succ, se2_succ)
        se2_results[f"rot_{deg:+}deg"] = {
            "success_count": res_se2_h["success_count"],
            "success_rate": res_se2_h["success_rate"],
            "mean_goal_error_m": res_se2_h["mean_final_goal_error_m"],
            "paired_stats": pair_se2,
            "records": res_se2_h["records"],
        }
        print(f"  [SE(2) {deg:+4d} deg] Succ: {res_se2_h['success_count']}/{n_eps} ({res_se2_h['success_rate']*100:5.1f}%) | Delta: {pair_se2['mean_diff']*100:+5.1f}% | p={pair_se2['mcnemar_exact_p_value']:.4f}")

    # Build execution metadata
    metadata = {
        "command_executed": f"python world_models_fpga/research/benchmarks/run_track_c_visual_grounding.py --episodes-per-suite {n_eps} --device {args.device}",
        "execution_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_sha(),
        "working_tree_clean": _git_clean(),
        "dataset_path": str(test_npz.relative_to(repo_root)),
        "dataset_sha256": _sha256(test_npz),
        "checkpoint_path": str(hjepa_path.relative_to(repo_root)),
        "checkpoint_sha256": _sha256(hjepa_path),
        "checkpoints": [
            {"name": "hjepa_best.pt", "path": str(hjepa_path.relative_to(repo_root)), "sha256": _sha256(hjepa_path)},
            {"name": "lewm_best.pt", "path": str(lewm_path.relative_to(repo_root)), "sha256": _sha256(lewm_path)},
            {"name": "baseline_best.pt", "path": str(base_path.relative_to(repo_root)), "sha256": _sha256(base_path)},
        ],
        "random_seed": 42,
        "configuration": {
            "track": "Track C (pure observation, zero simulator coordinates)",
            "episodes_per_suite": n_eps,
            "max_steps": 50,
        },
        "environment": {
            "platform": sys.platform,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }

    full_output = {
        "execution_metadata": metadata,
        "summary": {
            "hjepa_nominal": res_nom_h,
            "lewm_nominal": res_nom_l,
            "baseline_nominal": res_nom_b,
            "appearance_nuisance": {
                "results": res_nuis_h,
                "paired_contingency": pair_nuis,
            },
            "semantically_relevant_appearance": {
                "results": res_sem_h,
                "paired_contingency": pair_sem,
            },
            "visual_geometry_grounding": {
                "results": res_rearr_h,
                "paired_contingency": pair_rearr,
            },
            "layout_generalization": {
                "results": res_rand_h,
                "paired_contingency": pair_rand,
            },
            "se2_rigid_transformations": se2_results,
        },
    }

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(full_output, indent=2) + "\n")
    print(f"\n[Saved Artifact] -> {out_p}")


if __name__ == "__main__":
    main()
