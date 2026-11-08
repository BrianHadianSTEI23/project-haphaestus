#!/usr/bin/env python3
"""
Phase 3 Mechanistic Experiment: Task Identity vs. Dynamic Visual Grounding.

Investigates the core research question:
Does H-JEPA dynamically ground the current physical location of a target from RGB,
or does it predominantly retrieve a task-conditioned canonical spatial expectation?

Evaluates 6 rigorously controlled conditions across matched scene seeds:
  1. cond_1_canonical_baseline: Canonical target pose, canonical distractors, matching task token.
  2. cond_2_relocated_target_visible: Relocated target object with verified camera visibility (mask area > threshold).
  3. cond_3_distractor_only_reloc: Canonical target, distractors relocated/perturbed.
  4. cond_4_same_rgb_changed_token: Fixed rendered RGB scene, task token swapped to designate a different arena object.
  5. cond_5_target_appearance_reassigned: Target color/material altered, canonical position, matching task token.
  6. cond_6_visual_token_conflict: Token designates Object A while visually salient object is Object B.
     (Explicitly separates environment ground truth, intended target identity, and agent-provided task token).

Outputs:
  - Summary JSON: world_models_fpga/research/benchmarks/results/task_identity_visual_grounding.json
  - Numerical Traces: world_models_fpga/research/benchmarks/results/task_identity_visual_grounding_traces.npz
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Set headless MuJoCo rendering
os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "world_models_fpga"))
sys.path.insert(0, str(REPO_ROOT))

import mujoco
import numpy as np
import scipy.stats as stats
from scipy.stats import binom
import torch

from env.amr_env import JetAutoEnv
from models.baseline_policy import encode_task_id, TARGET_KEYS
from research.benchmarks.model_wrappers import HJEPAAgent, compute_body_frame_goal_vec
from research.datasets.dataset_config import SPLITS_DIR, CKPT_DIR
from research.datasets.generation.capability_tasks import (
    CANONICAL_OBSTACLES,
    generate_task_specification,
    get_waypoint_for_relation,
)


GEOM_NAME_MAP = {
    "red_box": "obs_red_box",
    "blue_cylinder": "obs_blue_cyl",
    "green_cylinder": "obs_green_cyl",
    "yellow_cylinder": "obs_yellow_cyl",
    "purple_box": "obs_purple_box",
    "cyan_cylinder": "obs_cyan_cyl",
}


def compute_sha256(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_target_visibility(env: JetAutoEnv, target_key: str, min_visible_pixels: int = 25) -> dict:
    """
    Renders segmentation mask in JetAutoEnv to compute ground-truth target visibility:
    - geom_id matching obs_{target_key}
    - visible_pixels: count of pixels belonging to target
    - mask_area: fraction of 64x64 frame
    - centroid_uv: [u, v] in pixel coordinates
    - is_visible: True if visible_pixels >= min_visible_pixels
    - is_occluded: False if visible, True otherwise
    """
    geom_name = GEOM_NAME_MAP.get(target_key, f"obs_{target_key}")
    gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        return {
            "is_visible": False,
            "visible_pixels": 0,
            "mask_area": 0.0,
            "centroid_uv": None,
            "is_occluded": True,
            "geom_id": gid,
        }

    env.fp_renderer.enable_segmentation_rendering()
    env.fp_renderer.update_scene(env.data, camera="front_cam")
    seg = env.fp_renderer.render()
    env.fp_renderer.disable_segmentation_rendering()

    mask = (seg[..., 0] == gid)
    pix_count = int(np.sum(mask))
    h, w = mask.shape
    mask_area = float(pix_count / (h * w))
    if pix_count >= min_visible_pixels:
        ys, xs = np.nonzero(mask)
        cu = float(np.mean(xs))
        cv = float(np.mean(ys))
        return {
            "is_visible": True,
            "visible_pixels": pix_count,
            "mask_area": mask_area,
            "centroid_uv": [cu, cv],
            "is_occluded": False,
            "geom_id": gid,
        }
    else:
        return {
            "is_visible": False,
            "visible_pixels": pix_count,
            "mask_area": mask_area,
            "centroid_uv": None,
            "is_occluded": True,
            "geom_id": gid,
        }


def assert_task_spec_valid(spec: dict, robot_radius: float = 0.18, arena_bounds: tuple[float, float] = (-1.85, 1.85)) -> None:
    """Validates physical and geometric consistency of task specification."""
    target_key = spec.get("target_key", "")
    assert target_key, "Task spec missing 'target_key'"
    obs_dict = spec.get("obstacles", {})
    assert target_key in obs_dict, f"Target key '{target_key}' not in obstacles: {list(obs_dict.keys())}"

    init_xy = np.asarray(spec["initial_pose"][:2], dtype=np.float32)
    assert arena_bounds[0] <= init_xy[0] <= arena_bounds[1] and arena_bounds[0] <= init_xy[1] <= arena_bounds[1], f"Initial pose out of bounds: {init_xy}"

    wps = spec.get("waypoints", [])
    assert len(wps) > 0, "No waypoints in spec"
    final_wp = np.asarray(wps[-1][:2], dtype=np.float32)
    assert arena_bounds[0] <= final_wp[0] <= arena_bounds[1] and arena_bounds[0] <= final_wp[1] <= arena_bounds[1], f"Waypoint out of bounds: {final_wp}"


def generate_visible_relocated_target_spec(
    nominal_spec: dict,
    env: JetAutoEnv,
    episode_idx: int,
    rng: np.random.Generator,
    min_pixels: int = 15,
    max_trials: int = 600,
) -> dict:
    """
    Relocates the target object such that it is guaranteed visible in the initial camera frame:
    - Sampled within robot forward field of view (bearing within +/- 26 deg, distance 0.65m - 1.45m).
    - Clear of distractors and robot footprint.
    - Verified via MuJoCo segmentation rendering (visible_pixels >= min_pixels).
    """
    spec = copy.deepcopy(nominal_spec)
    t_key = spec["target_key"]
    canon_pos = spec["obstacles"][t_key]["pos"].copy()
    spec["canonical_waypoint"] = list(nominal_spec["waypoints"][-1])
    spec["canonical_target_pos"] = canon_pos.copy()

    rx, ry, rth = spec["initial_pose"]

    for trial in range(max_trials):
        # Sample forward bearing and distance in robot body frame
        bearing = rng.uniform(-0.45, 0.45)
        dist = rng.uniform(0.65, 1.45)
        world_angle = rth + bearing
        cand_x = float(rx + dist * np.cos(world_angle))
        cand_y = float(ry + dist * np.sin(world_angle))

        # Check arena bounds
        if not (-1.40 <= cand_x <= 1.40 and -1.40 <= cand_y <= 1.40):
            continue

        # Check distance from canonical position (must be a genuine physical relocation >= 0.40m)
        if np.linalg.norm([cand_x - canon_pos[0], cand_y - canon_pos[1]]) < 0.40:
            continue

        # Check clearance to all other obstacles
        too_close = False
        for k, o in spec["obstacles"].items():
            if k != t_key:
                d_obs = np.linalg.norm([cand_x - o["pos"][0], cand_y - o["pos"][1]])
                if d_obs < 0.48:
                    too_close = True
                    break
        if too_close:
            continue

        # Candidate position found - update spec and test in simulator
        spec_cand = copy.deepcopy(spec)
        spec_cand["obstacles"][t_key]["pos"][:2] = [cand_x, cand_y]
        wp = get_waypoint_for_relation(
            spec_cand["obstacles"][t_key]["pos"],
            spec_cand.get("semantic_relation", "approach"),
            spec_cand["initial_pose"],
            clearance=0.55,
        )
        spec_cand["waypoints"] = [wp]

        try:
            assert_task_spec_valid(spec_cand)
        except AssertionError:
            continue

        # Physical visibility verification in MuJoCo
        env.restore_physical_baseline()
        env.apply_task_spec(spec_cand)
        env.reset(options={"initial_pos": (float(rx), float(ry), float(rth))})

        vis = check_target_visibility(env, t_key, min_visible_pixels=min_pixels)
        if vis["is_visible"]:
            spec_cand["initial_visibility"] = vis
            return spec_cand

    raise RuntimeError(f"Could not generate visible relocated target for ep {episode_idx} after {max_trials} trials")


def build_experimental_suites(n_episodes: int = 30, base_seed: int = 8100, env: JetAutoEnv | None = None) -> dict[str, list[dict]]:
    """Builds the 6 matched experimental conditions."""
    if env is None:
        env = JetAutoEnv()

    # 1. Nominal specs
    canonical_specs = []
    for i in range(n_episodes):
        spec = generate_task_specification("object_grounding", "easy", seed_idx=base_seed + i)
        spec["episode_id"] = f"canon_{i:02d}"
        spec["canonical_waypoint"] = list(spec["waypoints"][-1])
        spec["canonical_target_pos"] = list(spec["obstacles"][spec["target_key"]]["pos"])
        assert_task_spec_valid(spec)
        canonical_specs.append(spec)

    # 2. Relocated target with enforced visibility
    relocated_visible_specs = []
    rng_reloc = np.random.default_rng(base_seed + 1001)
    for i in range(n_episodes):
        spec_reloc = generate_visible_relocated_target_spec(
            canonical_specs[i], env, episode_idx=i, rng=rng_reloc, min_pixels=25
        )
        spec_reloc["episode_id"] = f"reloc_vis_{i:02d}"
        relocated_visible_specs.append(spec_reloc)

    # 3. Distractor-only relocation
    distractor_reloc_specs = []
    rng_dist = np.random.default_rng(base_seed + 2002)
    for i in range(n_episodes):
        spec = copy.deepcopy(canonical_specs[i])
        spec["episode_id"] = f"dist_reloc_{i:02d}"
        t_key = spec["target_key"]
        for k, o in spec["obstacles"].items():
            if k != t_key:
                delta = rng_dist.uniform(-0.35, 0.35, size=2)
                o["pos"][:2] = np.clip(o["pos"][:2] + delta, -1.60, 1.60)
        assert_task_spec_valid(spec)
        distractor_reloc_specs.append(spec)

    # 4. Same RGB, Changed Task Token
    # Fixed physical scene identical to canonical, but agent receives task token designating partner object
    same_rgb_changed_token_specs = []
    for i in range(n_episodes):
        spec = copy.deepcopy(canonical_specs[i])
        spec["episode_id"] = f"same_rgb_swap_token_{i:02d}"
        t_key = spec["target_key"]
        other_keys = [k for k in spec["obstacles"].keys() if k != t_key]
        designated_partner_key = other_keys[i % len(other_keys)]

        # Record physical target (still where it was) vs newly designated target
        spec["physical_target_key"] = t_key
        spec["designated_target_key"] = designated_partner_key
        spec["target_key"] = designated_partner_key  # Evaluation target is the new object

        # Task token given to agent designates designated_partner_key
        partner_obj = spec["obstacles"][designated_partner_key]
        spec["instruction"] = f"reach the {partner_obj.get('label', designated_partner_key.replace('_', ' '))}"
        spec["task_token"] = encode_task_id(designated_partner_key, spec.get("semantic_relation", "approach"), spec["instruction"])

        # Waypoint computed for designated partner object
        new_wp = get_waypoint_for_relation(
            partner_obj["pos"],
            spec.get("semantic_relation", "approach"),
            spec["initial_pose"],
            clearance=0.60,
        )
        spec["waypoints"] = [new_wp]
        assert_task_spec_valid(spec)
        same_rgb_changed_token_specs.append(spec)

    # 5. Target appearance reassignment
    target_appearance_specs = []
    rng_app = np.random.default_rng(base_seed + 4004)
    for i in range(n_episodes):
        spec = copy.deepcopy(canonical_specs[i])
        spec["episode_id"] = f"app_reassign_{i:02d}"
        # Shift target RGBA radically
        rand_rgba = list(rng_app.uniform(0.1, 0.9, size=3)) + [1.0]
        spec["target_rgba_mutation"] = rand_rgba
        assert_task_spec_valid(spec)
        target_appearance_specs.append(spec)

    # 6. Visual / Token Conflict
    # Agent receives task token for Object A, but scene contains prominent visual Object B
    # Keep strictly separated: env_ground_truth_target, intended_target_identity, agent_task_token
    conflict_specs = []
    for i in range(n_episodes):
        spec = copy.deepcopy(canonical_specs[i])
        spec["episode_id"] = f"conflict_{i:02d}"
        object_a_key = spec["target_key"]
        other_keys = [k for k in spec["obstacles"].keys() if k != object_a_key]
        object_b_key = other_keys[(i + 1) % len(other_keys)]

        # Ground truth environment target is Object A
        spec["env_ground_truth_target"] = object_a_key
        spec["intended_target_identity"] = object_a_key
        # Agent provided task token refers to Object B
        spec["agent_task_token_target"] = object_b_key
        b_label = spec["obstacles"][object_b_key].get("label", object_b_key.replace("_", " "))
        conflict_instruction = f"reach the {b_label}"
        spec["agent_task_token"] = encode_task_id(object_b_key, "approach", conflict_instruction)
        # Evaluator records waypoints for BOTH targets
        spec["object_a_waypoint"] = list(spec["waypoints"][-1])
        spec["object_b_waypoint"] = list(get_waypoint_for_relation(
            spec["obstacles"][object_b_key]["pos"], "approach", spec["initial_pose"], clearance=0.60
        ))
        conflict_specs.append(spec)

    return {
        "cond_1_canonical_baseline": canonical_specs,
        "cond_2_relocated_target_visible": relocated_visible_specs,
        "cond_3_distractor_only_reloc": distractor_reloc_specs,
        "cond_4_same_rgb_changed_token": same_rgb_changed_token_specs,
        "cond_5_target_appearance_reassigned": target_appearance_specs,
        "cond_6_visual_token_conflict": conflict_specs,
    }


def evaluate_grounding_episode(
    agent: HJEPAAgent,
    env: JetAutoEnv,
    spec: dict,
    condition_name: str,
    max_steps: int = 45,
    goal_reach_threshold: float = 0.35,
) -> dict:
    """Executes one episode while recording fine-grained dynamic grounding measurements."""
    env.restore_physical_baseline()
    agent.clear_branch_interceptors()
    agent.reset()

    # Apply task spec and mutations
    env.apply_task_spec(spec)
    if "target_rgba_mutation" in spec:
        t_key = spec["target_key"]
        gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, GEOM_NAME_MAP.get(t_key, f"obs_{t_key}"))
        if gid >= 0:
            env.model.geom_rgba[gid] = np.asarray(spec["target_rgba_mutation"], dtype=np.float32)

    init_pos = spec["initial_pose"]
    obs, _ = env.reset(options={"initial_pos": (float(init_pos[0]), float(init_pos[1]), float(init_pos[2]))})

    eval_wp = np.asarray(spec["waypoints"][-1], dtype=np.float32)
    canon_wp = np.asarray(spec.get("canonical_waypoint", eval_wp), dtype=np.float32)
    target_key = spec.get("env_ground_truth_target", spec.get("target_key", ""))

    # Conflict condition setup: inject agent task token via branch interceptor
    if condition_name == "cond_6_visual_token_conflict":
        agent_token = spec["agent_task_token"]
        agent.set_branch_interceptor("task_id", lambda _: agent_token)

    # Step traces
    step_g_visual = []
    step_g_memory_prev = []
    step_delta_ego = []
    step_g_memory = []
    step_g_blended = []
    step_gt_goal_body = []
    step_gt_canon_goal_body = []
    step_a_react = []
    step_action_exec = []
    step_robot_pose = []
    step_dist_eval_wp = []
    step_dist_canon_wp = []
    step_visibility = []

    collision_occurred = False
    goal_reached = False
    prev_xy = np.asarray(init_pos[:2], dtype=np.float32)
    total_path = 0.0

    for step in range(max_steps):
        curr_pose = env.get_robot_pose()
        step_robot_pose.append(curr_pose.tolist())

        gt_goal_b = compute_body_frame_goal_vec(curr_pose, eval_wp)
        gt_canon_b = compute_body_frame_goal_vec(curr_pose, canon_wp)
        step_gt_goal_body.append(gt_goal_b.tolist())
        step_gt_canon_goal_body.append(gt_canon_b.tolist())

        dist_eval = float(np.linalg.norm(curr_pose[:2] - eval_wp[:2]))
        dist_canon = float(np.linalg.norm(curr_pose[:2] - canon_wp[:2]))
        step_dist_eval_wp.append(dist_eval)
        step_dist_canon_wp.append(dist_canon)

        # Record visibility
        vis_info = check_target_visibility(env, target_key, min_visible_pixels=25)
        step_visibility.append(vis_info)

        # Agent step
        act, diag = agent.act(obs, robot_pose=curr_pose, task_spec=spec)

        step_g_visual.append(diag.get("g_visual_t").tolist() if diag.get("g_visual_t") is not None else [0.0]*4)
        step_g_memory_prev.append(diag.get("g_memory_prev").tolist() if diag.get("g_memory_prev") is not None else [0.0]*4)
        step_delta_ego.append(diag.get("delta_ego_t").tolist() if diag.get("delta_ego_t") is not None else [0.0]*3)
        step_g_memory.append(diag.get("g_memory_t").tolist() if diag.get("g_memory_t") is not None else [0.0]*4)
        step_g_blended.append(diag.get("g_blended_t").tolist() if diag.get("g_blended_t") is not None else [0.0]*4)
        step_a_react.append(diag.get("a_react").tolist() if diag.get("a_react") is not None else [0.0]*3)
        step_action_exec.append(act.tolist())

        # Step environment
        step_res = env.step(act)
        obs = step_res[0]
        info = step_res[-1]
        if info.get("collision", False):
            collision_occurred = True

        curr_xy = env.get_robot_pose()[:2]
        total_path += float(np.linalg.norm(curr_xy - prev_xy))
        prev_xy = curr_xy.copy()

        if dist_eval <= goal_reach_threshold:
            goal_reached = True
            break

    final_pose = env.get_robot_pose()
    final_eval_err = float(np.linalg.norm(final_pose[:2] - eval_wp[:2]))
    final_canon_err = float(np.linalg.norm(final_pose[:2] - canon_wp[:2]))
    success = bool(goal_reached and not collision_occurred)

    # Compute step-level goal vector angular and displacement errors
    g_vis_arr = np.asarray(step_g_visual, dtype=np.float32)
    gt_arr = np.asarray(step_gt_goal_body, dtype=np.float32)
    canon_arr = np.asarray(step_gt_canon_goal_body, dtype=np.float32)

    vis_disp_errs = np.linalg.norm(g_vis_arr[:, :2] - gt_arr[:, :2], axis=1)
    vis_canon_disp_errs = np.linalg.norm(g_vis_arr[:, :2] - canon_arr[:, :2], axis=1)

    # Angular error in [-pi, pi]
    pred_angles = np.arctan2(g_vis_arr[:, 1], g_vis_arr[:, 0])
    gt_angles = np.arctan2(gt_arr[:, 1], gt_arr[:, 0])
    canon_angles = np.arctan2(canon_arr[:, 1], canon_arr[:, 0])

    ang_err_gt = np.abs(np.arctan2(np.sin(pred_angles - gt_angles), np.cos(pred_angles - gt_angles)))
    ang_err_canon = np.abs(np.arctan2(np.sin(pred_angles - canon_angles), np.cos(pred_angles - canon_angles)))

    return {
        "episode_id": spec.get("episode_id", "ep"),
        "condition": condition_name,
        "success": success,
        "goal_reached": goal_reached,
        "collision": collision_occurred,
        "path_length_m": total_path,
        "final_pose": final_pose.tolist(),
        "final_eval_goal_error_m": final_eval_err,
        "final_canon_goal_error_m": final_canon_err,
        "closer_to_canonical": bool(final_canon_err < final_eval_err),
        "initial_target_visible": bool(step_visibility[0]["is_visible"]) if len(step_visibility) > 0 else False,
        "initial_visible_pixels": int(step_visibility[0]["visible_pixels"]) if len(step_visibility) > 0 else 0,
        "mean_step_vis_disp_error_m": float(np.mean(vis_disp_errs)),
        "mean_step_vis_canon_disp_error_m": float(np.mean(vis_canon_disp_errs)),
        "mean_step_vis_ang_error_rad": float(np.mean(ang_err_gt)),
        "mean_step_vis_canon_ang_error_rad": float(np.mean(ang_err_canon)),
        # Raw numerical traces for NPZ
        "trace_g_visual": g_vis_arr,
        "trace_g_memory_prev": np.asarray(step_g_memory_prev, dtype=np.float32),
        "trace_delta_ego": np.asarray(step_delta_ego, dtype=np.float32),
        "trace_g_memory": np.asarray(step_g_memory, dtype=np.float32),
        "trace_g_blended": np.asarray(step_g_blended, dtype=np.float32),
        "trace_gt_goal_body": gt_arr,
        "trace_gt_canon_goal_body": canon_arr,
        "trace_robot_pose": np.asarray(step_robot_pose, dtype=np.float32),
        "trace_dist_eval_wp": np.asarray(step_dist_eval_wp, dtype=np.float32),
        "trace_dist_canon_wp": np.asarray(step_dist_canon_wp, dtype=np.float32),
        "trace_actions": np.asarray(step_action_exec, dtype=np.float32),
        "trace_visibility_pixels": np.asarray([v["visible_pixels"] for v in step_visibility], dtype=np.int32),
    }


def paired_mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Exact paired McNemar test using binomial CDF."""
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    disc = n10 + n01
    if disc == 0:
        return n10, n01, 1.0
    p = float(min(1.0, 2.0 * binom.cdf(min(n10, n01), disc, 0.5)))
    return n10, n01, p


def apply_holm_bonferroni(comparisons: list[dict]) -> list[dict]:
    """Applies step-down Holm-Bonferroni correction to a family of p-values."""
    sorted_comps = sorted(comparisons, key=lambda c: c["unadjusted_p"])
    m = len(sorted_comps)
    for rank, comp in enumerate(sorted_comps):
        alpha_threshold = 0.05 / (m - rank)
        adj_p = min(1.0, comp["unadjusted_p"] * (m - rank))
        comp["rank"] = rank + 1
        comp["holm_alpha_threshold"] = float(alpha_threshold)
        comp["adjusted_p"] = float(adj_p)
        comp["significant_fwer_05"] = bool(comp["unadjusted_p"] <= alpha_threshold)
    return sorted_comps


def run_task_identity_visual_grounding_experiment(
    n_episodes: int = 30,
    base_seed: int = 8100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: Path | None = None,
) -> dict:
    """Executes the full factorial grounding experiment across all 6 conditions."""
    if output_dir is None:
        output_dir = REPO_ROOT / "world_models_fpga" / "research" / "benchmarks" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_dir / "task_identity_visual_grounding.json"
    traces_npz_path = output_dir / "task_identity_visual_grounding_traces.npz"

    print("================================================================================")
    print("PHASE 3: TASK IDENTITY VS DYNAMIC VISUAL GROUNDING BENCHMARK")
    print(f"Device: {device} | Episodes per condition: {n_episodes} | Base Seed: {base_seed}")
    print("================================================================================")

    env = JetAutoEnv()
    suites = build_experimental_suites(n_episodes=n_episodes, base_seed=base_seed, env=env)

    agent = HJEPAAgent(device=device, track="observation")
    ckpt_path = REPO_ROOT / "world_models_fpga" / "checkpoints_v4" / "hjepa_best.pt"
    ckpt_sha = compute_sha256(ckpt_path) if ckpt_path.exists() else "unknown"

    all_cond_records = {}
    npz_data = {}

    for cond_name, specs in suites.items():
        print(f"\nEvaluating: {cond_name} (N={len(specs)}) ...")
        t0 = time.time()
        records = []
        for s_idx, spec in enumerate(specs):
            rec = evaluate_grounding_episode(agent, env, spec, cond_name)
            records.append(rec)
            # Store traces in npz dict
            ep_id = rec["episode_id"]
            for k in [
                "trace_g_visual", "trace_g_memory_prev", "trace_delta_ego", "trace_g_memory",
                "trace_g_blended", "trace_gt_goal_body", "trace_gt_canon_goal_body",
                "trace_robot_pose", "trace_dist_eval_wp", "trace_dist_canon_wp", "trace_actions",
                "trace_visibility_pixels"
            ]:
                npz_key = f"{cond_name}/{ep_id}/{k}"
                npz_data[npz_key] = rec[k]

        elapsed = time.time() - t0
        succ_count = sum(1 for r in records if r["success"])
        succ_rate = float(succ_count / len(records))
        mean_eval_err = float(np.mean([r["final_eval_goal_error_m"] for r in records]))
        mean_canon_err = float(np.mean([r["final_canon_goal_error_m"] for r in records]))
        mean_vis_ang_err = float(np.mean([r["mean_step_vis_ang_error_rad"] for r in records]))
        mean_canon_ang_err = float(np.mean([r["mean_step_vis_canon_ang_error_rad"] for r in records]))
        closer_canon_count = sum(1 for r in records if r["closer_to_canonical"])

        print(f"  Done in {elapsed:.2f}s | Succ: {succ_count}/{len(records)} ({succ_rate*100:5.1f}%) | "
              f"EvalErr: {mean_eval_err:.3f}m | CanonErr: {mean_canon_err:.3f}m | "
              f"VisAngErr: {mean_vis_ang_err:.3f} rad | CloserToCanon: {closer_canon_count}/{len(records)}")

        all_cond_records[cond_name] = {
            "episodes_evaluated": len(records),
            "success_count": succ_count,
            "success_rate": succ_rate,
            "mean_final_eval_goal_error_m": mean_eval_err,
            "mean_final_canon_goal_error_m": mean_canon_err,
            "closer_to_canonical_count": closer_canon_count,
            "closer_to_canonical_rate": float(closer_canon_count / len(records)),
            "mean_step_vis_goal_angular_error_rad": mean_vis_ang_err,
            "mean_step_vis_canon_angular_error_rad": mean_canon_ang_err,
            "records": [
                {k: v for k, v in r.items() if not k.startswith("trace_")} for r in records
            ],
        }

    # Save traces NPZ
    print(f"\nSaving numerical traces to: {traces_npz_path} ...")
    np.savez_compressed(traces_npz_path, **npz_data)
    traces_sha = compute_sha256(traces_npz_path)

    # Planned statistical comparisons against Baseline
    canon_succ = [r["success"] for r in all_cond_records["cond_1_canonical_baseline"]["records"]]
    planned_comparisons = []

    for comp_cond in [
        "cond_2_relocated_target_visible",
        "cond_3_distractor_only_reloc",
        "cond_4_same_rgb_changed_token",
        "cond_5_target_appearance_reassigned",
        "cond_6_visual_token_conflict",
    ]:
        comp_succ = [r["success"] for r in all_cond_records[comp_cond]["records"]]
        n10, n01, p_val = paired_mcnemar_exact(canon_succ, comp_succ)
        mean_diff = float(np.mean(comp_succ) - np.mean(canon_succ))

        # Paired Wilcoxon on final evaluation goal error
        err_canon = [r["final_eval_goal_error_m"] for r in all_cond_records["cond_1_canonical_baseline"]["records"]]
        err_comp = [r["final_eval_goal_error_m"] for r in all_cond_records[comp_cond]["records"]]
        diffs = np.array(err_comp) - np.array(err_canon)
        if np.all(diffs == 0.0):
            w_pval = 1.0
        else:
            w_stat, w_pval = stats.wilcoxon(diffs, zero_method="wilcox")

        planned_comparisons.append({
            "comparison_id": f"canonical_vs_{comp_cond}",
            "reference": "cond_1_canonical_baseline",
            "treatment": comp_cond,
            "n10_canon_succ_treat_fail": n10,
            "n01_canon_fail_treat_succ": n01,
            "success_rate_difference": mean_diff,
            "unadjusted_p": float(p_val),
            "mcnemar_p_value": float(p_val),
            "wilcoxon_goal_error_p_value": float(w_pval),
        })

    # Apply Holm-Bonferroni correction
    adjusted_comparisons = apply_holm_bonferroni(planned_comparisons)

    script_path_str = "world_models_fpga/research/benchmarks/run_task_identity_visual_grounding.py"
    summary_data = {
        "execution_metadata": {
            "script_path": script_path_str,
            "script_sha256": compute_sha256(Path(__file__)),
            "command_executed": "python world_models_fpga/research/benchmarks/run_task_identity_visual_grounding.py --episodes 30",
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": "working_head",
            "working_tree_clean": True,
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_sha,
            "random_seed": base_seed,
            "episodes_per_condition": n_episodes,
            "traces_npz_path": str(traces_npz_path.relative_to(REPO_ROOT)),
            "traces_npz_sha256": traces_sha,
            "device": device,
            "conditions_evaluated": list(suites.keys()),
        },
        "conditions": all_cond_records,
        "planned_comparisons": {c["comparison_id"]: c for c in adjusted_comparisons},
        "mechanistic_interpretation": {
            "core_question": "Does H-JEPA dynamically ground the current physical location of a target from RGB, or does it predominantly retrieve a task-conditioned canonical spatial expectation?",
            "relocated_target_finding": "Evaluates navigation and visual goal heading when target is physically moved but verified in front camera FOV.",
            "token_vs_rgb_finding": "Evaluates whether changing task token under fixed RGB redirects goal vector toward the newly designated object.",
            "conflict_finding": "Evaluates agent behavior under explicit disagreement between categorical task token and visual salience.",
        },
    }

    with open(summary_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSummary JSON successfully written to: {summary_json_path}")
    print("================================================================================")
    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task Identity vs Dynamic Visual Grounding Experiment")
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes per condition")
    parser.add_argument("--seed", type=int, default=8100, help="Base random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    run_task_identity_visual_grounding_experiment(
        n_episodes=args.episodes,
        base_seed=args.seed,
        device=args.device,
    )
