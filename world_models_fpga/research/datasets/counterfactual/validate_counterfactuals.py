"""Executable validity checks for all seven physical counterfactuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from env.amr_env import JetAutoEnv
from research.datasets.counterfactual.counterfactual_generator import generate_counterfactual_pair

EXPERIMENTS = (
    "E1_position_shift", "E2_color_shift", "E3_robot_rotation", "E4_robot_position",
    "E5_intermediate_obstacle", "E6_shape_mutation", "E7_camera_viewpoint",
)


def _snapshot(env: JetAutoEnv, spec: dict) -> dict:
    env.apply_task_spec(spec)
    target_id = env.geometry_id(spec["obstacles"][spec["target_key"]]["geom_name"])
    inserted_id = env.geometry_id("obs_inserted")
    image, _ = env.reset(seed=spec["generation_seed"], options={"initial_pos": tuple(spec["initial_pose"].tolist())})
    return {
        "target_position": env.model.geom_pos[target_id].copy(),
        "target_rgba": env.model.geom_rgba[target_id].copy(),
        "target_type": int(env.model.geom_type[target_id]),
        "inserted_position": env.model.geom_pos[inserted_id].copy(),
        "inserted_type": int(env.model.geom_type[inserted_id]),
        "camera_quat": env.model.cam_quat[env.front_camera_id].copy(),
        "image": image.copy(),
    }


def validate_pair(experiment: str, pair_id: int = 0) -> dict:
    baseline, intervention = generate_counterfactual_pair(experiment, pair_id)
    env = JetAutoEnv()
    try:
        before, after = _snapshot(env, baseline), _snapshot(env, intervention)
        if experiment == "E1_position_shift":
            assert not np.allclose(before["target_position"], after["target_position"])
        elif experiment == "E2_color_shift":
            assert not np.allclose(before["target_rgba"], after["target_rgba"])
        elif experiment == "E3_robot_rotation":
            assert not np.isclose(baseline["initial_pose"][2], intervention["initial_pose"][2])
            assert not np.array_equal(before["image"], after["image"])
        elif experiment == "E4_robot_position":
            assert not np.allclose(baseline["initial_pose"], intervention["initial_pose"])
            assert not np.array_equal(before["image"], after["image"])
        elif experiment == "E5_intermediate_obstacle":
            assert after["inserted_position"][2] > 0 and after["inserted_type"] == mujoco.mjtGeom.mjGEOM_CYLINDER
            env.reset(seed=intervention["generation_seed"], options={"initial_pos": tuple(after["inserted_position"][:2]) + (0.0,)})
            assert env._check_collision(), "inserted obstacle is not collidable"
        elif experiment == "E6_shape_mutation":
            assert before["target_type"] == mujoco.mjtGeom.mjGEOM_CYLINDER
            assert after["target_type"] == mujoco.mjtGeom.mjGEOM_BOX
            assert not np.array_equal(before["image"], after["image"])
        elif experiment == "E7_camera_viewpoint":
            assert not np.allclose(before["camera_quat"], after["camera_quat"]), "camera orientation unchanged"
            assert not np.array_equal(before["image"], after["image"]), "rendered camera observation unchanged"
        return {"experiment": experiment, "pair_id": pair_id, "status": "VALID"}
    finally:
        env.close()


def validate_all(pair_id: int = 0) -> list[dict]:
    return [validate_pair(experiment, pair_id) for experiment in EXPERIMENTS]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-id", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {"pair_id": args.pair_id, "results": validate_all(args.pair_id)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
