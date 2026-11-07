"""Replay-based determinism and label-consistency check for splits_v2.

Regenerates a sample of stored episodes with the *current* simulator and
generator code and compares actions, collision labels, poses and RGB frames
with the stored NPZ arrays.  A mismatch means either (a) generation is not
deterministic, or (b) the stored labels were produced by different simulator
code (e.g. a pre-repair collision checker).

Usage:
    python world_models_fpga/research/datasets/replay_determinism_check.py \
        [--episodes-per-split 6] [--out <json>]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "world_models_fpga"))

from world_models_fpga.research.datasets.generation.generate_benchmark_dataset import BenchmarkDatasetGenerator  # noqa: E402
from world_models_fpga.research.datasets.generation.capability_tasks import generate_task_specification  # noqa: E402
from world_models_fpga.research.datasets.counterfactual.counterfactual_generator import generate_counterfactual_pair  # noqa: E402
from world_models_fpga.research.datasets.dataset_config import SPLITS_DIR

SPLITS = SPLITS_DIR


def _spec_for(episode):
    eid = episode["episode_id"]
    if episode.get("experiment_type"):
        e0, ei = generate_counterfactual_pair(episode["experiment_type"], pair_id=int(episode["pair_id"]))
        spec = e0 if episode["pair_role"] in ("E0", "baseline", "control") or eid.endswith("_E0") else ei
    else:
        spec = generate_task_specification(episode["capability"], episode["difficulty"], seed_idx=int(episode["generation_seed"]))
    spec["episode_id"] = eid
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-per-split", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "world_models_fpga/research/datasets/splits_v2/replay_determinism_report.json"))
    args = ap.parse_args()
    gen = BenchmarkDatasetGenerator(seed=42)
    report = {}
    for split in ["train_benchmark", "val_benchmark", "held_out_test_benchmark", "counterfactual_suite", "independent_validation_suite"]:
        d = np.load(SPLITS / f"{split}.npz")
        meta = json.load(open(SPLITS / f"{split}_metadata.json"))
        eps = meta["episodes"]
        idx = np.linspace(0, len(eps) - 1, args.episodes_per_split).astype(int)
        rows = []
        for i in idx:
            ep = eps[int(i)]
            s, L = int(ep["start_index"]), int(ep["length"])
            trans = gen.run_episode(_spec_for(ep))
            n = min(L, len(trans))
            acts = np.stack([t["action"] for t in trans[:n]])
            cols = np.array([t["collision"] for t in trans[:n]], dtype=bool)
            poses = np.stack([t["robot_pose"] for t in trans[:n]])
            rgb = np.stack([t["rgb"] for t in trans[:n]])
            rows.append({
                "episode_id": ep["episode_id"],
                "stored_length": L,
                "replayed_length": len(trans),
                "max_action_abs_diff": float(np.abs(acts - d["actions"][s:s + n]).max()),
                "max_pose_abs_diff": float(np.abs(poses - d["robot_pose"][s:s + n]).max()),
                "max_rgb_abs_diff": int(np.abs(rgb.astype(int) - d["rgb"][s:s + n].astype(int)).max()),
                "collision_label_agreement": float(np.mean(cols == d["collisions"][s:s + n])),
                "stored_collisions": int(d["collisions"][s:s + L].sum()),
                "replayed_collisions": int(cols.sum()),
            })
        ok = all(r["stored_length"] == r["replayed_length"] and r["collision_label_agreement"] == 1.0
                 and r["max_action_abs_diff"] < 1e-5 and r["max_pose_abs_diff"] < 1e-5 for r in rows)
        report[split] = {"deterministic_and_consistent": ok, "episodes": rows}
        print(split, "OK" if ok else "MISMATCH", [(r["episode_id"], r["collision_label_agreement"], r["max_action_abs_diff"]) for r in rows])
    gen.close()
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
