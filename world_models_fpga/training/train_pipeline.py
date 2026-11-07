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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.baseline_policy import BaselineNavigationPolicy, encode_task_id
from models.h_jepa import HJEPAModel
from models.le_world_model import LeWorldModel
from research.datasets.verify_dataset import validate_split
from research.datasets.dataset_config import SPLITS_DIR, CKPT_DIR


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
        raw_npz = np.load(npz_path, allow_pickle=False)
        rgb = raw_npz["rgb"]
        self.rgb_t = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div_(255.0).contiguous()
        self.actions_t = torch.from_numpy(raw_npz["actions"].astype(np.float32)).contiguous()
        self.collisions_t = torch.from_numpy(raw_npz["collisions"].astype(np.float32)).unsqueeze(-1).contiguous()
        robot_pose = raw_npz["robot_pose"].astype(np.float32)
        grounded_goals = raw_npz["grounded_goals"].astype(np.float32)
        episode_starts = raw_npz["episode_starts"]
        with Path(metadata_path).open() as f:
            self.metadata = json.load(f)
        self.stride_k = stride_k
        self.sequence_length = stride_k + 1
        self.indices: list[int] = []
        N = len(rgb)
        task_ids = np.zeros(N, dtype=np.int64)

        starts = np.flatnonzero(episode_starts)
        episodes_meta = self.metadata.get("episodes", [])
        for ep_i, (start, end) in enumerate(zip(starts, list(starts[1:]) + [N])):
            if ep_i < len(episodes_meta):
                ep_m = episodes_meta[ep_i]
                tid = encode_task_id(
                    ep_m.get("target_key", ""),
                    ep_m.get("semantic_relation", "approach"),
                    ep_m.get("instruction", ""),
                )
            else:
                tid = 0
            task_ids[int(start):int(end)] = tid
            self.indices.extend(range(int(start), int(end) - self.sequence_length + 1))

        if not self.indices:
            raise ValueError(f"No complete {self.sequence_length}-observation sequences in {npz_path}")

        self.task_ids_t = torch.from_numpy(task_ids)

        # Vectorized body-frame targets for all indices
        c = np.cos(robot_pose[:, 2])
        s = np.sin(robot_pose[:, 2])
        dx_g = grounded_goals[:, 0] - robot_pose[:, 0]
        dy_g = grounded_goals[:, 1] - robot_pose[:, 1]
        dth_g = grounded_goals[:, 2] - robot_pose[:, 2]
        goal_vecs = np.stack([c * dx_g + s * dy_g, -s * dx_g + c * dy_g, np.sin(dth_g), np.cos(dth_g)], axis=-1).astype(np.float32)
        self.goal_vecs_t = torch.from_numpy(goal_vecs).contiguous()

        pose_t1 = np.roll(robot_pose, -1, axis=0)
        dx1 = pose_t1[:, 0] - robot_pose[:, 0]
        dy1 = pose_t1[:, 1] - robot_pose[:, 1]
        dth1 = np.arctan2(np.sin(pose_t1[:, 2] - robot_pose[:, 2]), np.cos(pose_t1[:, 2] - robot_pose[:, 2]))
        micro_egos = np.stack([c * dx1 + s * dy1, -s * dx1 + c * dy1, dth1], axis=-1).astype(np.float32)
        self.micro_egos_t = torch.from_numpy(micro_egos).contiguous()

        pose_tk = np.roll(robot_pose, -stride_k, axis=0)
        dxk = pose_tk[:, 0] - robot_pose[:, 0]
        dyk = pose_tk[:, 1] - robot_pose[:, 1]
        dthk = np.arctan2(np.sin(pose_tk[:, 2] - robot_pose[:, 2]), np.cos(pose_tk[:, 2] - robot_pose[:, 2]))
        macro_egos = np.stack([c * dxk + s * dyk, -s * dxk + c * dyk, dthk], axis=-1).astype(np.float32)
        self.macro_egos_t = torch.from_numpy(macro_egos).contiguous()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        t = self.indices[index]
        end = t + self.sequence_length
        return (
            self.rgb_t[t:end],
            self.actions_t[t:end],
            self.collisions_t[t:end],
            self.goal_vecs_t[t],
            self.micro_egos_t[t],
            self.macro_egos_t[t],
            self.task_ids_t[t],
        )


def _mean_metrics(metrics: list[dict]) -> dict:
    return {key: float(np.mean([entry[key] for entry in metrics])) for key in metrics[0]}


def _grad_norm(model: nn.Module) -> float:
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += float(p.grad.detach().pow(2).sum().item())
    return float(np.sqrt(total_sq))


def _run_epoch(model_name, model, loader, optimizer, device) -> dict:
    is_training = optimizer is not None
    model.train(is_training)
    rows = []
    mse = nn.MSELoss()
    for images, actions, collisions, goal, micro_ego, macro_ego, task_id in loader:
        images = images.to(device)
        actions = actions.to(device)
        collisions = collisions.to(device)
        goal = goal.to(device)
        micro_ego = micro_ego.to(device)
        macro_ego = macro_ego.to(device)
        task_id = task_id.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        if model_name == "baseline":
            (pred_action, pred_risk), z_vis = model(images[:, 0], goal)
            action_loss = mse(pred_action, actions[:, 0])
            risk_loss = F.binary_cross_entropy(pred_risk, collisions[:, 0])
            pred_goal_c = model.infer_goal_vec(z_vis, task_id)
            ground_loss = mse(pred_goal_c, goal)
            loss = action_loss + 0.2 * risk_loss + 0.5 * ground_loss
            metrics = {
                "loss_total": float(loss.detach()),
                "loss_action": float(action_loss.detach()),
                "loss_collision": float(risk_loss.detach()),
                "loss_ground": float(ground_loss.detach()),
            }
        elif model_name == "lewm":
            loss, measured = model.compute_loss(
                images[:, 0],
                actions[:, 0],
                images[:, 1],
                collisions[:, 0],
                ego_disp_target=micro_ego,
                goal_vec=goal,
                task_ids=task_id,
            )
            metrics = measured
        else:
            loss, measured = model.compute_loss(
                images,
                actions,
                collisions,
                micro_ego_target=micro_ego,
                macro_ego_target=macro_ego,
                goal_vec=goal,
                task_ids=task_id,
            )
            metrics = measured

        if is_training:
            loss.backward()
            grad_norm = _grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            metrics["grad_norm"] = grad_norm
        rows.append(metrics)
    return _mean_metrics(rows)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unavailable"


def train_models(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_ds = EpisodeSequenceDataset(args.train_npz, args.train_metadata, args.stride_k)
    val_ds = EpisodeSequenceDataset(args.val_npz, args.val_metadata, args.stride_k)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in str(args.seeds).split(",")] if getattr(args, "seeds", None) else [int(args.seed)]
    all_seed_results = {}

    for seed in seeds:
        set_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
        run_config = vars(args) | {
            "seed": seed,
            "device": str(device),
            "git_commit": _git_sha(),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "train_dataset": train_ds.validation,
            "val_dataset": val_ds.validation,
        }
        (out_dir / f"training_config_seed{seed}.json").write_text(json.dumps(run_config, indent=2) + "\n")
        if seed == seeds[0]:
            (out_dir / "training_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

        models = {
            "baseline": BaselineNavigationPolicy(),
            "lewm": LeWorldModel(),
            "hjepa": HJEPAModel(stride_k=args.stride_k),
        }
        seed_results = {}
        for name, model in models.items():
            set_seed(seed)
            model.to(device)
            param_count = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            history, best_value, best_epoch = [], float("inf"), None
            seed_ckpt_path = out_dir / f"{name}_seed{seed}_best.pt"
            default_ckpt_path = out_dir / f"{name}_best.pt"

            for epoch in range(1, args.epochs + 1):
                train_metrics = _run_epoch(name, model, train_loader, optimizer, device)
                with torch.no_grad():
                    val_metrics = _run_epoch(name, model, val_loader, None, device)
                record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
                history.append(record)
                value = val_metrics["loss_total"]
                if value < best_value:
                    best_value, best_epoch = value, epoch
                    # Save raw state_dict for seamless downstream loading
                    torch.save(model.state_dict(), seed_ckpt_path)
                    if seed == seeds[0]:
                        torch.save(model.state_dict(), default_ckpt_path)
            print(f"[Seed {seed}] {name:<8} | Best Epoch: {best_epoch:2d} | Val Loss: {best_value:.4f} | Final Train Loss: {history[-1]['train']['loss_total']:.4f}", flush=True)

            checksum = _sha256(seed_ckpt_path)
            meta_payload = {
                "selection": "minimum validation loss",
                "seed": seed,
                "parameter_count": param_count,
                "best_epoch": best_epoch,
                "validation_loss": best_value,
                "sha256": checksum,
                "training_config": run_config,
            }
            (out_dir / f"{name}_seed{seed}_history.json").write_text(json.dumps(history, indent=2) + "\n")
            (out_dir / f"{name}_seed{seed}_best.metadata.json").write_text(json.dumps(meta_payload, indent=2) + "\n")
            if seed == seeds[0]:
                (out_dir / f"{name}_history.json").write_text(json.dumps(history, indent=2) + "\n")
                (out_dir / f"{name}_best.metadata.json").write_text(json.dumps(meta_payload, indent=2) + "\n")

            seed_results[name] = {
                "seed": seed,
                "parameter_count": param_count,
                "best_epoch": best_epoch,
                "best_validation_loss": best_value,
                "final_train_loss": history[-1]["train"]["loss_total"],
                "final_grad_norm": history[-1]["train"].get("grad_norm", 0.0),
                "checkpoint": str(seed_ckpt_path),
                "sha256": checksum,
            }
        all_seed_results[f"seed_{seed}"] = seed_results

    summary = {
        "primary_seed": seeds[0],
        "seeds": seeds,
        "results_by_seed": all_seed_results,
        "primary_results": all_seed_results[f"seed_{seeds[0]}"],
    }
    (out_dir / "training_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-npz", default=str(SPLITS_DIR / "train_benchmark.npz"))
    parser.add_argument("--train-metadata", default=str(SPLITS_DIR / "train_benchmark_metadata.json"))
    parser.add_argument("--val-npz", default=str(SPLITS_DIR / "val_benchmark.npz"))
    parser.add_argument("--val-metadata", default=str(SPLITS_DIR / "val_benchmark_metadata.json"))
    parser.add_argument("--output-dir", default=str(CKPT_DIR))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stride-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="42,1337,2026")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    print(json.dumps(train_models(parser.parse_args()), indent=2))
