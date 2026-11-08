import os
import numpy as np
import pytest
import mujoco

os.environ["MUJOCO_GL"] = "egl"
from env.amr_env import JetAutoEnv


def test_floor_appearance_mutation():
    """Verify that applying floor_rgba actually alters rendered pixels and is restored cleanly."""
    env = JetAutoEnv()
    baseline_obs = env.render_observation()

    # Apply task spec with floor_rgba
    spec = {"floor_rgba": [0.15, 0.20, 0.35, 1.0]}
    env.apply_task_spec(spec)
    mutated_obs = env.render_observation()

    pixel_diff = np.max(np.abs(baseline_obs.astype(float) - mutated_obs.astype(float)))
    assert pixel_diff > 30.0, f"Expected floor mutation to alter observation, got max diff {pixel_diff}"

    # Verify restoration
    env.restore_physical_baseline()
    restored_obs = env.render_observation()
    np.testing.assert_array_equal(baseline_obs, restored_obs)
    env.close()


def test_obstacle_color_mutation():
    """Verify that color_mutations actually changes geom RGBA and rendered pixels."""
    env = JetAutoEnv()
    # Position robot facing yellow cylinder
    env.reset(options={"initial_pos": (-0.5, -0.9, np.pi)})
    baseline_obs = env.render_observation()

    # Mutate yellow cylinder to bright green
    spec = {
        "obstacles": {
            "yellow_cylinder": {
                "geom_name": "obs_yellow_cyl",
                "pos": [-1.2, -0.9, 0.35],
                "shape": "cylinder",
            }
        },
        "color_mutations": {"obs_yellow_cyl": [0.1, 0.9, 0.1, 1.0]},
    }
    env.apply_task_spec(spec)
    env.reset(options={"initial_pos": (-0.5, -0.9, np.pi)})
    mutated_obs = env.render_observation()

    pixel_diff = np.max(np.abs(baseline_obs.astype(float) - mutated_obs.astype(float)))
    assert pixel_diff > 30.0, f"Expected color mutation to alter observation, got max diff {pixel_diff}"

    # Verify restoration
    env.restore_physical_baseline()
    env.reset(options={"initial_pos": (-0.5, -0.9, np.pi)})
    restored_obs = env.render_observation()
    np.testing.assert_array_equal(baseline_obs, restored_obs)
    env.close()


def test_physical_obstacle_repositioning():
    """Verify that changing obstacle pos physically relocates the MuJoCo geom."""
    env = JetAutoEnv()
    cyl_id = env.geometry_id("obs_yellow_cyl")
    orig_pos = env.model.geom_pos[cyl_id].copy()

    # Physically move the cylinder
    new_pos = np.array([1.5, 1.5, 0.35], dtype=np.float64)
    spec = {
        "obstacles": {
            "yellow_cylinder": {
                "geom_name": "obs_yellow_cyl",
                "pos": new_pos,
                "shape": "cylinder",
            }
        }
    }
    env.apply_task_spec(spec)
    assert np.allclose(env.model.geom_pos[cyl_id], new_pos), "Physical geom_pos did not update"

    # Verify restoration
    env.restore_physical_baseline()
    assert np.allclose(env.model.geom_pos[cyl_id], orig_pos), "Physical geom_pos did not restore"
    env.close()


def test_illumination_mutation():
    """Verify that light_diffuse_scale alters rendered pixels."""
    env = JetAutoEnv()
    baseline_obs = env.render_observation()

    spec = {"light_diffuse_scale": 0.2}
    env.apply_task_spec(spec)
    dim_obs = env.render_observation()

    pixel_diff = np.max(np.abs(baseline_obs.astype(float) - dim_obs.astype(float)))
    assert pixel_diff > 30.0, f"Expected lighting mutation to alter observation, got max diff {pixel_diff}"

    env.restore_physical_baseline()
    restored_obs = env.render_observation()
    np.testing.assert_array_equal(baseline_obs, restored_obs)
    env.close()
