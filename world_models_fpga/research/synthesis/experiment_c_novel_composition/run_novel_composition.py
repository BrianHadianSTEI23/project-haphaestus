"""
Phase 6: Experiment C — Novel Compositional Multi-Stage Navigation (Repaired)

Fixes:
  1. Uses exact `CANONICAL_OBSTACLES` positions and real MuJoCo geometries (`obs_inserted`,
     `obs_barrier`) via `JetAutoEnv.apply_task_spec` (eliminating fictional obstacle coordinates).
  2. Evaluates 10 seeded episodes per task across 4 multi-stage compositional tasks (40 episodes per model)
     comparing:
       - Track A (`Oracle_Planner`)
       - Track B (`H-JEPA`, `LeWM`, `Reconstructed_Compact_16D`, `Reactive_Baseline`)
       - Track C (`H-JEPA_Observation_Only`)
"""

from __future__ import annotations
import sys as _hsys
from pathlib import Path as _HPath
_hsys.path.insert(0, str(_HPath(__file__).resolve().parents[3]))
from research.datasets.dataset_config import SPLITS_DIR as _ACTIVE_SPLITS_DIR, CKPT_DIR as _ACTIVE_CKPT_DIR  # noqa: E402

import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from env.amr_env import JetAutoEnv
from research.benchmarks.model_wrappers import (
    BaselineAgent,
    HJEPAAgent,
    LeWMAgent,
    OracleAgent,
    compute_body_frame_goal_vec,
)
from research.datasets.generation.capability_tasks import CANONICAL_OBSTACLES, get_waypoint_for_relation
from research.synthesis.compositional_architecture.compositional_agent import CompositionalSpatialAgent


def build_compositional_tasks(episodes_per_task: int = 10) -> dict[str, list[dict]]:
    tasks: dict[str, list[dict]] = {
        "T1_dual_landmark_slalom": [],
        "T2_relational_detour_behind_blocked": [],
        "T3_three_stage_sequential_patrol": [],
        "T4_narrow_corridor_traversal": [],
    }

    for i in range(episodes_per_task):
        rng = np.random.default_rng(9500 + i)
        j_xy = rng.uniform(-0.10, 0.10, size=2).astype(np.float32)
        j_th = float(rng.uniform(-0.10, 0.10))

        # T1: Approach Blue Cylinder (-1.2, 1.4), then Go Behind Red Box (1.2, 1.0) via flank
        obs1 = copy.deepcopy(CANONICAL_OBSTACLES)
        init1 = np.array([0.0 + j_xy[0], -0.4 + j_xy[1], 1.57 + j_th], dtype=np.float32)
        wp1_a = get_waypoint_for_relation(obs1["blue_cylinder"]["pos"], "approach", init1)
        wp1_flank = np.array([0.85, 1.65, 0.0], dtype=np.float32)
        wp1_b = get_waypoint_for_relation(obs1["red_box"]["pos"], "behind", wp1_flank)
        tasks["T1_dual_landmark_slalom"].append(
            {
                "task_id": f"T1_ep{i:02d}",
                "target_key": "red_box",
                "semantic_relation": "behind",
                "instruction": "approach blue cylinder then go behind red box",
                "initial_pose": init1,
                "obstacles": obs1,
                "waypoints": [wp1_a, wp1_flank, wp1_b],
                "max_steps": 135,
            }
        )

        # T2: Go Behind Yellow Cylinder (-1.4, -1.0) with Inserted Obstacle blocking direct path
        obs2 = copy.deepcopy(CANONICAL_OBSTACLES)
        init2 = np.array([0.0 + j_xy[0], 0.0 + j_xy[1], -2.50 + j_th], dtype=np.float32)
        mid2 = 0.5 * (init2[:2] + obs2["yellow_cylinder"]["pos"][:2])
        obs2["inserted_obstacle"] = {
            "geom_name": "obs_inserted",
            "pos": np.array([float(mid2[0]), float(mid2[1]), 0.35], dtype=np.float32),
            "radius": 0.22,
            "height": 0.35,
            "shape": "cylinder",
            "rgba": np.array([0.8, 0.1, 0.8, 1.0], dtype=np.float32),
        }
        wp2_detour = np.array([float(mid2[0] + 0.45), float(mid2[1] - 0.65), -2.50], dtype=np.float32)
        wp2_goal = get_waypoint_for_relation(obs2["yellow_cylinder"]["pos"], "behind", wp2_detour)
        tasks["T2_relational_detour_behind_blocked"].append(
            {
                "task_id": f"T2_ep{i:02d}",
                "target_key": "yellow_cylinder",
                "semantic_relation": "behind",
                "instruction": "detour around magenta obstacle and go behind yellow cylinder",
                "initial_pose": init2,
                "obstacles": obs2,
                "waypoints": [wp2_detour, wp2_goal],
                "max_steps": 110,
            }
        )

        # T3: Three-Stage Patrol: Green Cylinder (1.2, -1.2) -> Red Box (1.2, 1.0) -> Purple Box (0.0, 1.7)
        obs3 = copy.deepcopy(CANONICAL_OBSTACLES)
        init3 = np.array([0.0 + j_xy[0], -0.2 + j_xy[1], -0.75 + j_th], dtype=np.float32)
        wp3_1 = get_waypoint_for_relation(obs3["green_cylinder"]["pos"], "approach", init3)
        wp3_2 = get_waypoint_for_relation(obs3["red_box"]["pos"], "approach", wp3_1)
        wp3_3 = get_waypoint_for_relation(obs3["purple_box"]["pos"], "in_front", wp3_2)
        tasks["T3_three_stage_sequential_patrol"].append(
            {
                "task_id": f"T3_ep{i:02d}",
                "target_key": "purple_box",
                "semantic_relation": "in_front",
                "instruction": "patrol green cylinder, red box, and stop in front of purple box",
                "initial_pose": init3,
                "obstacles": obs3,
                "waypoints": [wp3_1, wp3_2, wp3_3],
                "max_steps": 140,
            }
        )

        # T4: Narrow Corridor Traversal between obs_inserted and obs_barrier to reach Red Box
        obs4 = copy.deepcopy(CANONICAL_OBSTACLES)
        init4 = np.array([-0.5 + j_xy[0], -0.4 + j_xy[1], 0.65 + j_th], dtype=np.float32)
        obs4["inserted_obstacle"] = {
            "geom_name": "obs_inserted",
            "pos": np.array([0.05, 0.78, 0.35], dtype=np.float32),
            "radius": 0.20,
            "height": 0.35,
            "shape": "cylinder",
            "rgba": np.array([0.8, 0.1, 0.8, 1.0], dtype=np.float32),
        }
        obs4["slit_barrier"] = {
            "geom_name": "obs_barrier",
            "pos": np.array([0.65, -0.18, 0.35], dtype=np.float32),
            "size": np.array([0.12, 0.22, 0.35], dtype=np.float32),
            "radius": 0.22,
            "height": 0.35,
            "shape": "box",
            "rgba": np.array([0.3, 0.3, 0.3, 1.0], dtype=np.float32),
        }
        wp4_gate = np.array([0.35, 0.30, 0.65], dtype=np.float32)
        wp4_goal = get_waypoint_for_relation(obs4["red_box"]["pos"], "approach", wp4_gate)
        tasks["T4_narrow_corridor_traversal"].append(
            {
                "task_id": f"T4_ep{i:02d}",
                "target_key": "red_box",
                "semantic_relation": "approach",
                "instruction": "traverse narrow corridor to approach red box",
                "initial_pose": init4,
                "obstacles": obs4,
                "waypoints": [wp4_gate, wp4_goal],
                "max_steps": 95,
            }
        )

    return tasks


def _run_episode_for_agent(env: JetAutoEnv, agent_type: str, agent_obj, spec: dict) -> dict:
    env.apply_task_spec(spec)
    if hasattr(agent_obj, "reset"):
        agent_obj.reset()
    init_p = spec["initial_pose"]
    obs, _ = env.reset(options={"initial_pos": (float(init_p[0]), float(init_p[1]), float(init_p[2]))})

    wps = spec["waypoints"]
    wp_idx = 0
    collisions = 0
    max_steps = spec.get("max_steps", 65)

    for _ in range(max_steps):
        pose = env.get_robot_pose()
        target_wp = wps[wp_idx]
        if float(np.linalg.norm(target_wp[:2] - pose[:2])) < 0.28:
            if wp_idx < len(wps) - 1:
                wp_idx += 1
                target_wp = wps[wp_idx]
            else:
                break

        if agent_type == "oracle":
            act, _ = agent_obj.act(obs, robot_pose=pose, target_pose=target_wp, task_spec=spec)
        elif agent_type == "compact":
            act, _ = agent_obj.select_action(obs, pose, target_wp)
        elif agent_type == "track_c":
            pub = {
                "target_key": spec.get("target_key", ""),
                "semantic_relation": spec.get("semantic_relation", "approach"),
                "instruction": spec.get("instruction", ""),
            }
            act, _ = agent_obj.act(obs, robot_pose=None, target_pose=None, task_spec=pub, goal_vec_body=None)
        else:
            g_body = compute_body_frame_goal_vec(pose, target_wp)
            act, _ = agent_obj.act(obs, robot_pose=None, target_pose=None, task_spec=None, goal_vec_body=g_body)

        obs, _, _, _, info = env.step(act)
        if info["collision"]:
            collisions += 1

    final_pose = env.get_robot_pose()
    final_err = float(np.linalg.norm(wps[-1][:2] - final_pose[:2]))
    reached_all = (wp_idx == len(wps) - 1) and (final_err < 0.30)
    success = bool(reached_all and collisions == 0)
    stages_completed = wp_idx + (1 if final_err < 0.30 else 0)
    env.restore_physical_baseline()
    return {
        "success": success,
        "reached_final": reached_all,
        "stage_completion_rate": float(stages_completed / len(wps)),
        "collisions": collisions,
        "collision_free": collisions == 0,
        "final_error_m": final_err,
    }


def run_novel_composition(episodes_per_task: int = 10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v2_ckpt = str(_ACTIVE_CKPT_DIR)
    legacy_ckpt = str(_ACTIVE_CKPT_DIR)
    ckpt_dir = v2_ckpt if os.path.exists(os.path.join(v2_ckpt, "hjepa_best.pt")) else legacy_ckpt

    compact_ckpt = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../extraction/extracted_model/reconstructed_compact/reconstructed_compact_best.pt",
        )
    )
    out_dir = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    agents = {
        "Oracle_TrackA": ("oracle", OracleAgent()),
        "H-JEPA_TrackB": ("track_b", HJEPAAgent(os.path.join(ckpt_dir, "hjepa_best.pt"), device=device, track="structured")),
        "LeWM_TrackB": ("track_b", LeWMAgent(os.path.join(ckpt_dir, "lewm_best.pt"), device=device, track="structured")),
        "Reconstructed_Compact_16D_TrackB": ("compact", CompositionalSpatialAgent(compact_ckpt, device=device)),
        "Reactive_Baseline_TrackB": ("track_b", BaselineAgent(os.path.join(ckpt_dir, "baseline_best.pt"), device=device, track="structured")),
        "H-JEPA_TrackC_VisualOnly": ("track_c", HJEPAAgent(os.path.join(ckpt_dir, "hjepa_best.pt"), device=device, track="observation")),
    }

    tasks = build_compositional_tasks(episodes_per_task=episodes_per_task)
    env = JetAutoEnv()

    results = {"episodes_per_task": episodes_per_task, "tasks": {}, "summary_by_agent": {}}
    per_agent_records = {k: [] for k in agents}

    for t_name, specs in tasks.items():
        print(f"\n--- Compositional Task: {t_name} ({len(specs)} episodes) ---")
        results["tasks"][t_name] = {}
        for ag_name, (ag_type, ag_obj) in agents.items():
            ep_recs = [_run_episode_for_agent(env, ag_type, ag_obj, sp) for sp in specs]
            per_agent_records[ag_name].extend(ep_recs)
            t_summ = {
                "success_rate": float(np.mean([1.0 if r["success"] else 0.0 for r in ep_recs])),
                "stage_completion_rate": float(np.mean([r["stage_completion_rate"] for r in ep_recs])),
                "collision_rate": float(np.mean([0.0 if r["collision_free"] else 1.0 for r in ep_recs])),
                "mean_final_error_m": float(np.mean([r["final_error_m"] for r in ep_recs])),
            }
            results["tasks"][t_name][ag_name] = t_summ
            print(
                f"  [{ag_name:34s}] Success: {t_summ['success_rate']*100:5.1f}% | "
                f"StageComp: {t_summ['stage_completion_rate']*100:5.1f}% | "
                f"Col: {t_summ['collision_rate']*100:5.1f}% | Err: {t_summ['mean_final_error_m']:.3f}m"
            )

    env.close()

    for ag_name, recs in per_agent_records.items():
        results["summary_by_agent"][ag_name] = {
            "total_episodes": len(recs),
            "overall_success_rate": float(np.mean([1.0 if r["success"] else 0.0 for r in recs])),
            "overall_stage_completion_rate": float(np.mean([r["stage_completion_rate"] for r in recs])),
            "overall_collision_rate": float(np.mean([0.0 if r["collision_free"] else 1.0 for r in recs])),
            "overall_mean_final_error_m": float(np.mean([r["final_error_m"] for r in recs])),
        }

    out_path = os.path.join(out_dir, "novel_composition_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED NOVEL COMPOSITION RESULTS TO {out_path}]")
    return results


if __name__ == "__main__":
    run_novel_composition(episodes_per_task=10)
