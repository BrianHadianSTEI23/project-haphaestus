"""Reproducible training for the repaired, episode-disjoint dataset format.

This entry point refuses legacy splits: their transition-level random split and
unknown provenance make them unsuitable for a scientific training claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.baseline_policy import BaselineNavigationPolicy
from models.h_jepa import HJEPAModel
from models.le_world_model import LeWorldModel
from research.datasets.verify_dataset import validate_split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


class EpisodeSequenceDataset(Dataset):
    """Contiguous samples that never cross an episode boundary."""

    def __init__(self, npz_path: str | Path, metadata_path: str | Path, stride_k: int):
        self.validation = validate_split(npz_path, metadata_path)
        self.data = np.load(npz_path, allow_pickle=False)
        self.sequence_length = stride_k + 1
        self.indices: list[int] = []
        starts = np.flatnonzero(self.data["episode_starts"])
        for start, end in zip(starts, list(starts[1:]) + [len(self.data["rgb"])]):
            self.indices.extend(range(int(start), int(end) - self.sequence_length + 1))
        if not self.indices:
            raise ValueError(f"No complete {self.sequence_length}-observation sequences in {npz_path}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        t = self.indices[index]
        end = t + self.sequence_length
        images = torch.from_numpy(self.data["rgb"][t:end]).permute(0, 3, 1, 2).float().div(255.0)
        actions = torch.from_numpy(self.data["actions"][t:end]).float()
        collisions = torch.from_numpy(self.data["collisions"][t:end].astype(np.float32)).unsqueeze(-1)
        pose = self.data["robot_pose"][t]
        goal = self.data["grounded_goals"][t]
        dx, dy = goal[:2] - pose[:2]
        c, s = np.cos(pose[2]), np.sin(pose[2])
        goal_vector = torch.tensor([c * dx + s * dy, -s * dx + c * dy, np.sin(goal[2] - pose[2]), np.cos(goal[2] - pose[2])], dtype=torch.float32)
        return images, actions, collisions, goal_vector


def _mean_metrics(metrics: list[dict]) -> dict:
    return {key: float(np.mean([entry[key] for entry in metrics])) for key in metrics[0]}


def _run_epoch(model_name, model, loader, optimizer, device) -> dict:
    is_training = optimizer is not None
    model.train(is_training)
    rows = []
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    for images, actions, collisions, goal in loader:
        images, actions, collisions, goal = images.to(device), actions.to(device), collisions.to(device), goal.to(device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        if model_name == "baseline":
            (pred_action, pred_risk), _ = model(images[:, 0], goal)
            action_loss = mse(pred_action, actions[:, 0])
            # Baseline emits probabilities, not logits.
            risk_loss = nn.functional.binary_cross_entropy(pred_risk, collisions[:, 0])
            loss = action_loss + 0.2 * risk_loss
            metrics = {"loss_total": float(loss.detach()), "loss_action": float(action_loss.detach()), "loss_collision": float(risk_loss.detach())}
        elif model_name == "lewm":
            loss, measured = model.compute_loss(images[:, 0], actions[:, 0], images[:, 1], collisions[:, 0])
            metrics = measured
        else:
            loss, measured = model.compute_loss(images, actions, collisions)
            metrics = measured
        if is_training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        rows.append(metrics)
    return _mean_metrics(rows)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable"


def train_models(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_ds = EpisodeSequenceDataset(args.train_npz, args.train_metadata, args.stride_k)
    val_ds = EpisodeSequenceDataset(args.val_npz, args.val_metadata, args.stride_k)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args) | {
        "device": str(device), "git_commit": _git_sha(), "torch": torch.__version__,
        "python": platform.python_version(), "train_dataset": train_ds.validation, "val_dataset": val_ds.validation,
    }
    (out_dir / "training_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    models = {
        "baseline": BaselineNavigationPolicy(),
        "lewm": LeWorldModel(),
        "hjepa": HJEPAModel(stride_k=args.stride_k),
    }
    results = {}
    for name, model in models.items():
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        history, best_value, best_epoch = [], float("inf"), None
        checkpoint_path = out_dir / f"{name}_best.pt"
        for epoch in range(1, args.epochs + 1):
            train_metrics = _run_epoch(name, model, train_loader, optimizer, device)
            with torch.no_grad():
                val_metrics = _run_epoch(name, model, val_loader, None, device)
            record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
            history.append(record)
            value = val_metrics["loss_total"]
            if value < best_value:
                best_value, best_epoch = value, epoch
                torch.save({"state_dict": model.state_dict(), "epoch": epoch, "validation_loss": value, "training_config": run_config}, checkpoint_path)
        checksum = _sha256(checkpoint_path)
        (out_dir / f"{name}_history.json").write_text(json.dumps(history, indent=2) + "\n")
        (out_dir / f"{name}_best.metadata.json").write_text(json.dumps({"selection": "minimum validation loss", "best_epoch": best_epoch, "validation_loss": best_value, "sha256": checksum}, indent=2) + "\n")
        results[name] = {"best_epoch": best_epoch, "best_validation_loss": best_value, "checkpoint": str(checkpoint_path), "sha256": checksum}
    (out_dir / "training_results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--val-npz", required=True)
    parser.add_argument("--val-metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stride-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    print(json.dumps(train_models(parser.parse_args()), indent=2))
