"""Dataset-integrity validation for versioned, episode-level navigation data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = {
    "rgb", "next_rgb", "robot_pose", "robot_velocity", "object_positions",
    "object_classes", "object_colors", "actions", "next_actions", "collisions",
    "goals_reached", "episode_successes", "episode_starts", "grounded_goals",
    "transition_episode_indices",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_split(npz_path: str | Path, metadata_path: str | Path) -> dict:
    """Validate one generated split and return machine-readable measurements.

    Raises ``AssertionError`` for invalid data, preventing a training run from
    quietly consuming a leaky or malformed dataset.
    """
    npz_path, metadata_path = Path(npz_path), Path(metadata_path)
    with metadata_path.open() as file:
        metadata = json.load(file)
    assert metadata.get("dataset_version") == "2.0.0-physical-counterfactuals", "legacy/unversioned split is not valid for repaired experiments"
    assert metadata.get("npz_sha256") == _sha256(npz_path), "NPZ checksum does not match metadata"
    data = np.load(npz_path, allow_pickle=False)
    missing = REQUIRED_ARRAYS - set(data.files)
    assert not missing, f"missing required arrays: {sorted(missing)}"
    n = len(data["rgb"])
    assert n > 0
    assert data["rgb"].shape == (n, 64, 64, 3) and data["rgb"].dtype == np.uint8
    assert data["next_rgb"].shape == (n, 64, 64, 3) and data["next_rgb"].dtype == np.uint8
    for name, shape in {"robot_pose": (n, 3), "robot_velocity": (n, 3), "actions": (n, 3), "next_actions": (n, 3), "grounded_goals": (n, 3)}.items():
        assert data[name].shape == shape, f"invalid {name} shape: {data[name].shape}"
        assert np.isfinite(data[name]).all(), f"non-finite values in {name}"
    assert data["object_positions"].shape == (n, 6, 3)
    assert np.isfinite(data["object_positions"]).all()
    assert np.all(np.abs(data["actions"]) <= 1.00001), "action labels exceed normalized limits"
    starts = np.flatnonzero(data["episode_starts"])
    assert len(starts) == metadata["total_episodes"] == len(metadata["episodes"])
    assert len(starts) and starts[0] == 0
    assert np.array_equal(np.unique(data["transition_episode_indices"]), np.arange(len(starts)))
    episode_ids = set()
    for episode_index, (start, episode) in enumerate(zip(starts, metadata["episodes"])):
        length = int(episode["length"])
        assert int(episode["start_index"]) == int(start)
        assert length > 0 and start + length <= n
        assert not data["episode_starts"][start + 1:start + length].any(), "episode boundary inside episode"
        assert np.all(data["transition_episode_indices"][start:start + length] == episode_index)
        assert episode["episode_id"] not in episode_ids, "duplicate episode id"
        episode_ids.add(episode["episode_id"])
        assert episode.get("label_provenance"), "label provenance absent"
        assert episode.get("config_sha256"), "config_sha256 absent"
        assert episode.get("environment_parameters", {}).get("physical_geometries") is not None

    if metadata.get("split_name") == "counterfactual_suite":
        # Verify that pairs within each experiment_type are genuinely distinct (not duplicate copies)
        by_exp_e0_hashes: dict[str, set[str]] = {}
        for episode in metadata["episodes"]:
            if not episode.get("is_counterfactual", False):
                exp = episode.get("experiment_type", "")
                by_exp_e0_hashes.setdefault(exp, set()).add(episode["config_sha256"])
        for exp, hashes in by_exp_e0_hashes.items():
            assert len(hashes) >= 20, f"Duplicate counterfactual pairs detected in {exp}: only {len(hashes)} unique E0 configs"

    return {"npz": str(npz_path), "episodes": len(starts), "transitions": n, "sha256": metadata["npz_sha256"], "status": "VALID"}


def verify_all_splits(splits_dir: str | Path) -> list[dict]:
    root = Path(splits_dir)
    results = []
    for split in (
        "train_benchmark",
        "val_benchmark",
        "held_out_test_benchmark",
        "counterfactual_suite",
        "disambiguation_suite",
        "independent_validation_suite",
    ):
        npz_path, metadata_path = root / f"{split}.npz", root / f"{split}_metadata.json"
        if npz_path.exists() or metadata_path.exists():
            assert npz_path.exists() and metadata_path.exists(), f"incomplete split {split}"
            results.append(validate_split(npz_path, metadata_path))
    ids_by_split = []
    seeds_by_base_split = []
    for result in results:
        stem = Path(result["npz"]).stem
        with Path(result["npz"]).with_name(stem + "_metadata.json").open() as file:
            episodes = json.load(file)["episodes"]
            ids_by_split.append((stem, {e["episode_id"] for e in episodes}))
            if stem in ("train_benchmark", "val_benchmark", "held_out_test_benchmark", "independent_validation_suite"):
                seeds_by_base_split.append((stem, {int(e["generation_seed"]) for e in episodes}))
    for i, (left_name, left_ids) in enumerate(ids_by_split):
        for right_name, right_ids in ids_by_split[i + 1:]:
            assert not left_ids & right_ids, f"episode leakage between {left_name} and {right_name}"
    for i, (left_name, left_seeds) in enumerate(seeds_by_base_split):
        for right_name, right_seeds in seeds_by_base_split[i + 1:]:
            assert not left_seeds & right_seeds, f"seed leakage between {left_name} and {right_name}"
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("splits_dir", nargs="?", default=Path(__file__).with_name("splits"))
    args = parser.parse_args()
    for result in verify_all_splits(args.splits_dir):
        print(json.dumps(result, sort_keys=True))
