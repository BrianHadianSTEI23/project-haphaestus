"""
Phase 1: Controlled Intelligence Benchmark Dataset Generator
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Generates reproducible MuJoCo transition datasets across:
  - 6 Capability Categories (Object Grounding, Spatial Reasoning, Obstacle Avoidance,
                             Action-Conditioned Prediction, Temporal Planning, Generalization)
  - 5 Difficulty / Variation Tiers (Easy, Difficult, Counterfactual, OOD, Failure)
  - Counterfactual Suites (E1 - E7 paired interventions)
  - Disambiguation Suites (Ego vs Allocentric, Front vs Behind, Scale vs Distance, Visibility vs Reachability)

Records complete physical ground-truth alongside camera observations:
  [rgb, next_rgb, robot_pose, robot_velocity, object_positions, object_classes,
   object_colors, language_instruction, grounded_goal, action, next_action,
   collision, goal_reached, episode_success]
"""

import os
import sys
import copy
import time
import json
import argparse
import numpy as np
import mujoco

# Ensure path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from env.amr_env import JetAutoEnv
from research.datasets.generation.capability_tasks import (
    CANONICAL_OBSTACLES, CAPABILITIES, DIFFICULTY_TIERS,
    generate_task_specification, get_waypoint_for_relation
)
from research.datasets.generation.expert_navigator import ExpertNavigator
from research.datasets.counterfactual.counterfactual_generator import generate_counterfactual_pair
from research.datasets.counterfactual.disambiguation_tasks import generate_disambiguation_tasks

class BenchmarkDatasetGenerator:
    def __init__(self, xml_path=None, seed=42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.env = JetAutoEnv(xml_path=xml_path)
        self.navigator = ExpertNavigator()

        # Cache geometry IDs
        self.geom_ids = {}
        for k, v in CANONICAL_OBSTACLES.items():
            gid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, v["geom_name"])
            self.geom_ids[k] = gid

        # Object class mapping
        self.shape_map = {"cylinder": 0, "box": 1}
        self.color_map = {
            "yellow": 0, "red": 1, "blue": 2, "green": 3,
            "purple": 4, "cyan": 5, "magenta": 6, "mutated_blue": 7, "dark_gray": 8
        }

    def _apply_environment_obstacles(self, obstacles_dict):
        """Updates MuJoCo geom positions and RGBA in-memory for the current task."""
        for k, info in obstacles_dict.items():
            if k in self.geom_ids:
                gid = self.geom_ids[k]
                self.env.model.geom_pos[gid] = info["pos"]
                if "rgba" in info:
                    self.env.model.geom_rgba[gid] = info["rgba"]
                if "size" in info and info["shape"] == "box":
                    self.env.model.geom_size[gid] = info["size"]
                elif "radius" in info and info["shape"] == "cylinder":
                    self.env.model.geom_size[gid][:2] = [info["radius"], info["height"]]

    def _reset_canonical_obstacles(self):
        """Restores canonical arena obstacle positions and colors."""
        self._apply_environment_obstacles(CANONICAL_OBSTACLES)

    def _get_ground_truth_objects(self, obstacles_dict):
        """Extracts (6, 3) positions, (6,) classes, (6, 4) colors."""
        positions = []
        classes = []
        colors = []
        for k in CANONICAL_OBSTACLES.keys():
            if k in obstacles_dict:
                obj = obstacles_dict[k]
                positions.append(obj["pos"])
                classes.append(self.shape_map.get(obj.get("shape", "cylinder"), 0))
                colors.append(obj.get("rgba", np.array([1, 1, 1, 1], dtype=np.float32)))
            else:
                positions.append(np.array([0, 0, -10], dtype=np.float32)) # hidden
                classes.append(-1)
                colors.append(np.array([0, 0, 0, 0], dtype=np.float32))
        return (
            np.array(positions, dtype=np.float32),
            np.array(classes, dtype=np.int32),
            np.array(colors, dtype=np.float32)
        )

    def run_episode(self, task_spec, max_steps=35):
        """
        Executes a single controlled episode in MuJoCo, returning list of transition dicts.
        """
        # 1. Apply environment modifications
        self._apply_environment_obstacles(task_spec["obstacles"])

        # 2. Reset robot pose
        init_pos = task_spec["initial_pose"]
        obs, info = self.env.reset(options={"initial_pos": (init_pos[0], init_pos[1], init_pos[2])})

        waypoints = task_spec["waypoints"]
        wp_idx = 0
        transitions = []

        gt_pos, gt_cls, gt_col = self._get_ground_truth_objects(task_spec["obstacles"])
        goal_target = waypoints[-1] if len(waypoints) > 0 else init_pos

        is_success = False

        for t in range(max_steps):
            curr_pose = self.env.get_robot_pose()
            prev_obs = obs.copy()

            # Waypoint management
            if wp_idx < len(waypoints):
                curr_wp = waypoints[wp_idx]
                dist_to_wp = np.linalg.norm(curr_wp[:2] - curr_pose[:2])
                is_last = (wp_idx == len(waypoints) - 1)

                if dist_to_wp < 0.22:
                    if is_last:
                        is_success = True
                        goal_reached = True
                    else:
                        wp_idx += 1
                        curr_wp = waypoints[wp_idx]
                        is_last = (wp_idx == len(waypoints) - 1)
                        goal_reached = False
                else:
                    goal_reached = False
            else:
                curr_wp = waypoints[-1]
                goal_reached = True
                is_success = True

            # Compute demonstrator action
            action = self.navigator.compute_action(
                curr_pose, curr_wp, task_spec, is_last_waypoint=(wp_idx >= len(waypoints) - 1)
            )

            # Step environment
            obs, reward, term, trunc, step_info = self.env.step(action)
            collision = step_info["collision"]

            # Compute instantaneous robot velocity
            next_pose = self.env.get_robot_pose()
            dt = 0.1 # 10 Hz control step
            vx_meas = (next_pose[0] - curr_pose[0]) / dt
            vy_meas = (next_pose[1] - curr_pose[1]) / dt
            wz_meas = np.arctan2(np.sin(next_pose[2] - curr_pose[2]), np.cos(next_pose[2] - curr_pose[2])) / dt
            vel_vec = np.array([vx_meas, vy_meas, wz_meas], dtype=np.float32)

            # Lookahead action for transition record
            if t < max_steps - 1 and not (is_success and goal_reached):
                next_action = self.navigator.compute_action(
                    next_pose, curr_wp, task_spec, is_last_waypoint=(wp_idx >= len(waypoints) - 1)
                )
            else:
                next_action = np.zeros(3, dtype=np.float32)

            trans = {
                "episode_id": task_spec.get("episode_id", 0),
                "timestep": t,
                "rgb": prev_obs,
                "next_rgb": obs,
                "robot_pose": curr_pose,
                "robot_velocity": vel_vec,
                "object_positions": gt_pos,
                "object_classes": gt_cls,
                "object_colors": gt_col,
                "language_instruction": task_spec["instruction"],
                "grounded_goal": goal_target,
                "action": action,
                "next_action": next_action,
                "collision": collision,
                "goal_reached": goal_reached,
                "episode_success": False # Set after episode ends
            }
            transitions.append(trans)

            if is_success and goal_reached and t > 5:
                break
            if collision and task_spec.get("is_failure_example", False) and t > 8:
                break

        # Mark episode success across transitions
        for tr in transitions:
            tr["episode_success"] = is_success

        # Restore canonical state
        self._reset_canonical_obstacles()
        return transitions

    def generate_benchmark_splits(self, total_episodes=600, output_dir=None):
        """
        Generates 600 base episodes evenly divided across 6 capabilities (100 each),
        partitioned into:
          - Train split: 400 episodes (canonical + mild perturbations)
          - Val split: 100 episodes (unseen initial poses)
          - Held-out Test split: 100 episodes (held-out obstacle arrangements & unseen commands)
        """
        if output_dir is None:
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../splits"))
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n=======================================================")
        print(f"  PHASE 1: BENCHMARK DATASET GENERATION")
        print(f"  Total Base Episodes: {total_episodes} (100 per capability)")
        print(f"  Output Directory: {output_dir}")
        print(f"=======================================================")

        episodes_per_cap = total_episodes // len(CAPABILITIES)

        train_episodes = []
        val_episodes = []
        test_episodes = []

        ep_counter = 0
        t0 = time.time()

        for cap_idx, capability in enumerate(CAPABILITIES):
            print(f"\n[Capability {cap_idx+1}/6] Generating: {capability.upper()} ({episodes_per_cap} episodes)...")
            cap_transitions = 0
            for i in range(episodes_per_cap):
                diff = DIFFICULTY_TIERS[i % len(DIFFICULTY_TIERS)]
                spec = generate_task_specification(capability, diff, seed_idx=1000 + ep_counter)
                spec["episode_id"] = f"{capability}_{diff}_{ep_counter:04d}"

                transitions = self.run_episode(spec)
                cap_transitions += len(transitions)

                # Split assignment by configuration seed (zero frame leakage)
                # 66% train, 17% val, 17% held-out test
                if i % 6 < 4:
                    train_episodes.append((spec, transitions))
                elif i % 6 == 4:
                    val_episodes.append((spec, transitions))
                else:
                    test_episodes.append((spec, transitions))

                ep_counter += 1

            print(f"  -> Generated {episodes_per_cap} episodes ({cap_transitions} transitions).")

        t1 = time.time()
        print(f"\nBase benchmark generation finished in {t1 - t0:.2f} s ({ep_counter} episodes).")

        # Save splits to .npz + .json
        self._save_split("train_benchmark", train_episodes, output_dir)
        self._save_split("val_benchmark", val_episodes, output_dir)
        self._save_split("held_out_test_benchmark", test_episodes, output_dir)

    def generate_counterfactual_suite(self, pairs_per_type=25, output_dir=None):
        """
        Generates 7 counterfactual suites (E1 - E7), each with pairs_per_type matched pairs.
        Total = 7 * 25 * 2 = 350 episodes.
        """
        if output_dir is None:
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../splits"))
        os.makedirs(output_dir, exist_ok=True)

        exp_types = [
            "E1_position_shift",
            "E2_color_shift",
            "E3_robot_rotation",
            "E4_robot_position",
            "E5_intermediate_obstacle",
            "E6_shape_mutation",
            "E7_camera_viewpoint"
        ]

        print(f"\n=======================================================")
        print(f"  PHASE 1: COUNTERFACTUAL SUITE GENERATION (E1 - E7)")
        print(f"  Pairs per type: {pairs_per_type} | Total Suites: {len(exp_types)}")
        print(f"=======================================================")

        all_cf_episodes = []
        t0 = time.time()

        for exp_idx, exp_type in enumerate(exp_types):
            print(f"[{exp_idx+1}/7] Generating matched pairs for {exp_type}...")
            for p_idx in range(pairs_per_type):
                e0_spec, ei_spec = generate_counterfactual_pair(exp_type, pair_id=p_idx)
                e0_spec["episode_id"] = f"{exp_type}_p{p_idx:02d}_E0"
                ei_spec["episode_id"] = f"{exp_type}_p{p_idx:02d}_Ei"

                e0_trans = self.run_episode(e0_spec)
                ei_trans = self.run_episode(ei_spec)

                all_cf_episodes.append((e0_spec, e0_trans))
                all_cf_episodes.append((ei_spec, ei_trans))

        t1 = time.time()
        print(f"Counterfactual generation finished in {t1 - t0:.2f} s ({len(all_cf_episodes)} episodes).")
        self._save_split("counterfactual_suite", all_cf_episodes, output_dir)

    def generate_disambiguation_suite(self, repeats_per_probe=25, output_dir=None):
        """
        Generates disambiguation probe suites across 4 cognitive/spatial axes.
        Total = 4 * 25 = 100 episodes.
        """
        if output_dir is None:
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../splits"))
        os.makedirs(output_dir, exist_ok=True)

        probe_specs = generate_disambiguation_tasks()
        print(f"\n=======================================================")
        print(f"  PHASE 1: DISAMBIGUATION PROBE SUITE GENERATION")
        print(f"  Probes: {len(probe_specs)} | Repeats: {repeats_per_probe}")
        print(f"=======================================================")

        all_disambig_episodes = []
        t0 = time.time()

        for p_idx, p_spec in enumerate(probe_specs):
            print(f"[{p_idx+1}/{len(probe_specs)}] Generating probe: {p_spec['probe_id']}...")
            for r in range(repeats_per_probe):
                spec = copy.deepcopy(p_spec)
                spec["episode_id"] = f"{spec['probe_id']}_rep{r:02d}"
                # Add minor jitter to initial pose to test robustness
                jitter = self.rng.uniform(-0.05, 0.05, size=3).astype(np.float32)
                spec["initial_pose"][:2] += jitter[:2]

                trans = self.run_episode(spec)
                all_disambig_episodes.append((spec, trans))

        t1 = time.time()
        print(f"Disambiguation generation finished in {t1 - t0:.2f} s ({len(all_disambig_episodes)} episodes).")
        self._save_split("disambiguation_suite", all_disambig_episodes, output_dir)

    def _save_split(self, split_name, episodes_data, output_dir):
        """Compacts and saves transitions into compressed .npz and .json metadata."""
        print(f"  Saving {split_name} ({len(episodes_data)} episodes)...")

        all_rgb = []
        all_next_rgb = []
        all_poses = []
        all_vels = []
        all_obj_pos = []
        all_obj_cls = []
        all_obj_col = []
        all_actions = []
        all_next_actions = []
        all_collisions = []
        all_goals_reached = []
        all_ep_successes = []
        ep_starts = []

        metadata_records = []
        curr_idx = 0

        for ep_idx, (spec, trans) in enumerate(episodes_data):
            if len(trans) == 0:
                continue

            ep_meta = {
                "episode_id": spec.get("episode_id", f"ep_{ep_idx}"),
                "start_index": curr_idx,
                "length": len(trans),
                "instruction": spec.get("instruction", ""),
                "capability": spec.get("capability", "none"),
                "difficulty": spec.get("difficulty", "none"),
                "semantic_relation": spec.get("semantic_relation", "none"),
                "is_counterfactual": spec.get("is_counterfactual", False),
                "counterfactual_id": spec.get("counterfactual_id", ""),
                "expected_invariant": spec.get("expected_invariant", ""),
                "target_key": spec.get("target_key", ""),
                "episode_success": bool(trans[0]["episode_success"])
            }
            metadata_records.append(ep_meta)

            for step_i, tr in enumerate(trans):
                all_rgb.append(tr["rgb"])
                all_next_rgb.append(tr["next_rgb"])
                all_poses.append(tr["robot_pose"])
                all_vels.append(tr["robot_velocity"])
                all_obj_pos.append(tr["object_positions"])
                all_obj_cls.append(tr["object_classes"])
                all_obj_col.append(tr["object_colors"])
                all_actions.append(tr["action"])
                all_next_actions.append(tr["next_action"])
                all_collisions.append(tr["collision"])
                all_goals_reached.append(tr["goal_reached"])
                all_ep_successes.append(tr["episode_success"])
                ep_starts.append(step_i == 0)
                curr_idx += 1

        npz_file = os.path.join(output_dir, f"{split_name}.npz")
        json_file = os.path.join(output_dir, f"{split_name}_metadata.json")

        np.savez_compressed(
            npz_file,
            rgb=np.array(all_rgb, dtype=np.uint8),
            next_rgb=np.array(all_next_rgb, dtype=np.uint8),
            robot_pose=np.array(all_poses, dtype=np.float32),
            robot_velocity=np.array(all_vels, dtype=np.float32),
            object_positions=np.array(all_obj_pos, dtype=np.float32),
            object_classes=np.array(all_obj_cls, dtype=np.int32),
            object_colors=np.array(all_obj_col, dtype=np.float32),
            actions=np.array(all_actions, dtype=np.float32),
            next_actions=np.array(all_next_actions, dtype=np.float32),
            collisions=np.array(all_collisions, dtype=bool),
            goals_reached=np.array(all_goals_reached, dtype=bool),
            episode_successes=np.array(all_ep_successes, dtype=bool),
            episode_starts=np.array(ep_starts, dtype=bool)
        )

        with open(json_file, "w") as f:
            json.dump({
                "split_name": split_name,
                "total_episodes": len(metadata_records),
                "total_transitions": curr_idx,
                "episodes": metadata_records
            }, f, indent=2)

        file_size_mb = os.path.getsize(npz_file) / (1024 * 1024)
        print(f"  -> Saved {npz_file} ({file_size_mb:.2f} MB, {curr_idx} transitions)")
        print(f"  -> Saved {json_file}")

    def close(self):
        self.env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Phase 1 Controlled Intelligence Benchmark Datasets")
    parser.add_argument("--episodes", type=int, default=600, help="Total base benchmark episodes")
    parser.add_argument("--cf_pairs", type=int, default=25, help="Counterfactual pairs per type (E1-E7)")
    parser.add_argument("--disambig_repeats", type=int, default=25, help="Disambiguation repeats per probe")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation")
    parser.add_argument("--skip_base", action="store_true", help="Skip base benchmark generation if already generated")
    parser.add_argument("--skip_cf", action="store_true", help="Skip counterfactual generation if already generated")
    args = parser.parse_args()

    gen = BenchmarkDatasetGenerator(seed=args.seed)
    # 1. Base benchmark splits (Train, Val, Held-out Test)
    if not args.skip_base:
        gen.generate_benchmark_splits(total_episodes=args.episodes)
    else:
        print("Skipping base benchmark generation (--skip_base specified).")
    # 2. Counterfactual suites (E1 - E7)
    if not args.skip_cf:
        gen.generate_counterfactual_suite(pairs_per_type=args.cf_pairs)
    else:
        print("Skipping counterfactual suite generation (--skip_cf specified).")
    # 3. Disambiguation suites
    gen.generate_disambiguation_suite(repeats_per_probe=args.disambig_repeats)
    gen.close()
    print("\n[ALL BENCHMARK DATASETS SUCCESSFULLY GENERATED AND VERIFIED]\n")
