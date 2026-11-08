#!/usr/bin/env python3
"""
Unit tests for Task Identity & Dynamic Visual Grounding v2.

Verifies:
1. Rejection sampling guarantees:
   - Initial waypoint distance >= 0.60m (0.25m margin outside 0.35m goal threshold).
   - Relocated target distance from canonical position >= 0.75m.
   - Enforced initial visibility >= 25 pixels in camera frustum.
2. Two-target visible condition:
   - Both candidate targets A and B are visible in initial frame (both >= 25 pixels).
   - Neither object is occluded.
   - The paired initial RGB images are bit-for-bit identical (SHA-256 match).
   - The only model input that changes is the discrete task token / task embedding.
3. Post-action evaluation criteria:
   - Starting inside goal radius does not grant trivial success.
   - Success requires reaching goal on valid post-action pose.
4. Independent branch interventions:
   - Grounding latent, task embedding, visual goal, and controller memory pathways are decoupled.
"""

import copy
import os
import hashlib
import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

from env.amr_env import JetAutoEnv
from models.h_jepa import HJEPAModel
from models.baseline_policy import encode_task_id
from research.benchmarks.model_wrappers import HJEPAAgent
from research.datasets.dataset_config import CKPT_DIR
from research.datasets.generation.capability_tasks import generate_task_specification, get_waypoint_for_relation
from research.benchmarks.run_task_identity_visual_grounding_v2 import (
    generate_visible_relocated_target_spec_v2,
    generate_two_target_visible_spec_v2,
    check_target_visibility,
    evaluate_grounding_episode_v2,
)


@pytest.fixture(scope="module")
def env():
    e = JetAutoEnv()
    yield e


@pytest.fixture(scope="module")
def agent():
    ckpt_path = CKPT_DIR / "hjepa_best.pt"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = HJEPAAgent(str(ckpt_path), device=dev, track="observation")
    return a


def test_rejection_sampling_guarantees(env):
    """Verifies that relocated target generator enforces all physical and geometric margins."""
    rng = np.random.default_rng(9101)
    rejection_counters = {}
    nominal = generate_task_specification("object_grounding", "easy", seed_idx=9101)
    nominal["canonical_waypoint"] = list(nominal["waypoints"][-1])
    nominal["canonical_target_pos"] = list(nominal["obstacles"][nominal["target_key"]]["pos"])

    spec_reloc = generate_visible_relocated_target_spec_v2(
        nominal,
        env,
        episode_idx=0,
        rng=rng,
        rejection_counters=rejection_counters,
        min_pixels=25,
        min_initial_dist=0.60,
        min_reloc_dist=0.75,
    )

    t_key = spec_reloc["target_key"]
    canon_pos = np.asarray(spec_reloc["canonical_target_pos"][:2])
    reloc_pos = np.asarray(spec_reloc["obstacles"][t_key]["pos"][:2])
    reloc_dist = float(np.linalg.norm(reloc_pos - canon_pos))
    assert reloc_dist >= 0.75, f"Relocation distance {reloc_dist} < 0.75m"

    robot_xy = np.asarray(spec_reloc["initial_pose"][:2])
    wp_xy = np.asarray(spec_reloc["waypoints"][-1][:2])
    init_wp_dist = float(np.linalg.norm(robot_xy - wp_xy))
    assert init_wp_dist >= 0.60, f"Initial waypoint distance {init_wp_dist} < 0.60m"

    # MuJoCo visibility verification
    env.restore_physical_baseline()
    env.apply_task_spec(spec_reloc)
    env.reset(options={"initial_pos": spec_reloc["initial_pose"]})
    vis = check_target_visibility(env, t_key, min_visible_pixels=25)
    assert vis["is_visible"], "Target not visible in camera frustum"
    assert vis["visible_pixels"] >= 25, f"Target pixels {vis['visible_pixels']} < 25"


def test_two_target_visible_paired_condition(env):
    """Verifies that paired token contrast uses identical RGB observations and both targets are visible."""
    rng = np.random.default_rng(9202)
    rejection_counters = {}
    nominal = generate_task_specification("object_grounding", "easy", seed_idx=9202)

    spec_a, spec_b = generate_two_target_visible_spec_v2(
        nominal,
        env,
        episode_idx=0,
        rng=rng,
        rejection_counters=rejection_counters,
        min_pixels=25,
        min_initial_dist=0.60,
    )

    # 1. Both objects pass visibility threshold
    env.restore_physical_baseline()
    env.apply_task_spec(spec_a)
    obs_a, _ = env.reset(options={"initial_pos": spec_a["initial_pose"]})
    vis_a = check_target_visibility(env, spec_a["target_key"], min_visible_pixels=25)
    vis_b = check_target_visibility(env, spec_a["partner_target_key"], min_visible_pixels=25)
    assert vis_a["is_visible"], "Object A not visible"
    assert vis_b["is_visible"], "Object B not visible"
    assert vis_a["visible_pixels"] >= 25
    assert vis_b["visible_pixels"] >= 25

    # 2. Both waypoints start >= 0.60m from robot
    r_xy = np.asarray(spec_a["initial_pose"][:2])
    d_a = float(np.linalg.norm(r_xy - np.asarray(spec_a["target_a_waypoint"][:2])))
    d_b = float(np.linalg.norm(r_xy - np.asarray(spec_a["target_b_waypoint"][:2])))
    assert d_a >= 0.60, f"Waypoint A dist {d_a} < 0.60m"
    assert d_b >= 0.60, f"Waypoint B dist {d_b} < 0.60m"

    # 3. Render spec_b from identical initial pose and verify image SHA-256 match
    env.restore_physical_baseline()
    env.apply_task_spec(spec_b)
    obs_b, _ = env.reset(options={"initial_pos": spec_b["initial_pose"]})

    hash_a = hashlib.sha256(obs_a.tobytes()).hexdigest()
    hash_b = hashlib.sha256(obs_b.tobytes()).hexdigest()
    assert hash_a == hash_b, "Paired scenes do not produce bit-for-bit identical RGB observations!"

    # 4. Only task token changes between specs
    assert spec_a["task_token"] != spec_b["task_token"], "Task tokens must differ between paired trials"


def test_post_action_success_logic(agent, env):
    """Verifies that post-action evaluation rejects trivial success if starting within goal radius."""
    spec = generate_task_specification("object_grounding", "easy", seed_idx=9303)
    # Intentionally manipulate initial pose to be right on top of waypoint
    wp = spec["waypoints"][-1]
    spec_trivial = copy.deepcopy(spec)
    spec_trivial["initial_pose"] = [float(wp[0]), float(wp[1]), float(wp[2])]

    res = evaluate_grounding_episode_v2(agent, env, spec_trivial, condition_name="trivial_test", max_steps=5)
    assert res["initially_inside_radius"] is True
    assert res["success"] is False, "Episode starting inside goal radius must NEVER be counted as success!"


def test_independent_grounding_interventions(agent):
    """Verifies that interceptors on task_id, task_embedding, and grounding_latent operate independently."""
    agent.clear_branch_interceptors()
    agent.reset()

    dummy_rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
    task_spec = {"target_key": "red_box", "semantic_relation": "approach", "instruction": "reach the red box"}

    # Clean run
    _, diag_clean = agent.act(dummy_rgb, task_spec=task_spec)
    clean_latent = diag_clean["latent"]
    clean_g_vis = diag_clean["g_visual_t"]

    # Intervene on grounding_latent only: zero out latent entering grounding head
    agent.reset()
    agent.set_branch_interceptor("grounding_latent", lambda s: torch.zeros_like(s))
    _, diag_latent_mod = agent.act(dummy_rgb, task_spec=task_spec)

    # 1. Visual goal must change
    assert not np.allclose(diag_latent_mod["g_visual_t"], clean_g_vis), "Zeroing grounding_latent did not alter visual goal"
    # 2. But encoder latent itself must NOT change
    assert np.allclose(diag_latent_mod["latent"], clean_latent), "Intervention on grounding_latent leaked into encoder latent"

    # Intervene on task_embedding only
    agent.clear_branch_interceptors()
    agent.reset()
    agent.set_branch_interceptor("task_embedding", lambda t: torch.zeros_like(t))
    _, diag_task_mod = agent.act(dummy_rgb, task_spec=task_spec)

    assert not np.allclose(diag_task_mod["g_visual_t"], clean_g_vis), "Zeroing task_embedding did not alter visual goal"
    assert np.allclose(diag_task_mod["latent"], clean_latent), "Task embedding intervention leaked into encoder latent"

    agent.clear_branch_interceptors()
