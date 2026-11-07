"""
Phase 6: Genuine Teacher-Student Distillation & Compact Model Training (Repaired)

Fixes:
  1. Uses the loaded `HJEPAModel` teacher during training (distilling projected 32D->16D teacher
     latent states, macro subgoal states, collision logits, and teacher action predictions).
  2. Fixes the validation bug that computed `F.mse_loss(v_pred_next, v_curr)` (now evaluates
     `v_pred_macro` against `v_macro_gt`).
  3. Trains and compares three explicit compact regimes on the exact 39,804-parameter hardware
     architecture (`ReconstructedCompactModel`):
       - `distilled_compact`: Teacher distillation + ground-truth hierarchical dynamics (`reconstructed_compact_best.pt`)
       - `supervised_compact_no_teacher`: Ground-truth dynamics + action supervision without teacher
       - `bc_only_compact`: Behavior cloning only without latent transition losses
"""

from __future__ import annotations
import sys as _hsys
from pathlib import Path as _HPath
_hsys.path.insert(0, str(_HPath(__file__).resolve().parents[4]))
from research.datasets.dataset_config import SPLITS_DIR as _ACTIVE_SPLITS_DIR, CKPT_DIR as _ACTIVE_CKPT_DIR  # noqa: E402

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.h_jepa import HJEPAModel
from research.extraction.extracted_model.reconstructed_compact.model_reconstructed_compact import (
    ReconstructedCompactModel,
)


class CompactDistillationDataset(Dataset):
    def __init__(self, npz_path: str, stride_k: int = 5):
        data = np.load(npz_path)
        self.rgb_t = torch.from_numpy(data["rgb"]).permute(0, 3, 1, 2).float().div_(255.0).contiguous()
        self.next_rgb_t = torch.from_numpy(data["next_rgb"]).permute(0, 3, 1, 2).float().div_(255.0).contiguous()
        poses = data["robot_pose"].astype(np.float32)
        goals = data["grounded_goals"].astype(np.float32)
        self.actions_t = torch.from_numpy(data["actions"].astype(np.float32)).contiguous()
        self.collisions_t = torch.from_numpy(data["collisions"].astype(np.float32)).unsqueeze(-1).contiguous()
        self.ep_indices = data["transition_episode_indices"]
        self.stride_k = stride_k
        self.N = len(self.rgb_t)

        self.macro_targets = np.zeros(self.N, dtype=np.int64)
        for i in range(self.N):
            target_idx = min(i + stride_k, self.N - 1)
            while target_idx > i and self.ep_indices[target_idx] != self.ep_indices[i]:
                target_idx -= 1
            self.macro_targets[i] = target_idx

        dx = goals[:, 0] - poses[:, 0]
        dy = goals[:, 1] - poses[:, 1]
        th = poses[:, 2]
        c, s = np.cos(th), np.sin(th)
        dth = np.arctan2(np.sin(goals[:, 2] - th), np.cos(goals[:, 2] - th))
        intents = np.stack(
            [
                np.clip(1.1 * (c * dx + s * dy), -1.0, 1.0),
                np.clip(1.1 * (-s * dx + c * dy), -1.0, 1.0),
                np.clip(1.3 * dth, -1.0, 1.0),
            ],
            axis=-1,
        ).astype(np.float32)
        self.intents_t = torch.from_numpy(intents).contiguous()

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        m_idx = self.macro_targets[idx]
        return (
            self.rgb_t[idx],
            self.next_rgb_t[idx],
            self.rgb_t[m_idx],
            self.intents_t[idx],
            self.actions_t[idx],
            self.collisions_t[idx],
        )


def _train_regime(
    regime_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    teacher: HJEPAModel,
    teacher_proj: torch.Tensor,
    device: str,
    epochs: int = 12,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ReconstructedCompactModel(latent_dim=16, stride_k=5, hidden_dim=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        steps = 0
        for img_t, img_next, img_macro, intent_3d, act, col in train_loader:
            img_t = img_t.to(device)
            img_next = img_next.to(device)
            img_macro = img_macro.to(device)
            intent_3d = intent_3d.to(device)
            act = act.to(device)
            col = col.to(device)

            optimizer.zero_grad()
            v_curr = model.encode(img_t)
            v_macro_online = model.encode(img_macro)
            v_macro_gt = v_macro_online.detach()

            with torch.no_grad():
                t_curr_32 = teacher.encode(img_t)
                t_macro_32, t_col_mac = teacher.macro_predictor(t_curr_32, act)
                t_act = teacher.micro_predictor.select_action(t_curr_32, t_macro_32)
                t_curr_16 = F.layer_norm(t_curr_32 @ teacher_proj, (16,))
                t_macro_16 = F.layer_norm(t_macro_32 @ teacher_proj, (16,))

            # Macro transition on both realized action and body-frame goal intent
            v_sub_act, col_logit_act = model.forward_macro(v_curr, act)
            v_sub_intent, col_logit_int = model.forward_macro(v_curr, intent_3d)

            pred_act_from_sub = model.select_action(v_curr, v_sub_intent)
            pred_act_from_gt = model.select_action(v_curr, v_macro_gt)

            loss_bc = 0.5 * (F.mse_loss(pred_act_from_sub, act) + F.mse_loss(pred_act_from_gt, act))
            loss_col = 0.5 * (
                F.binary_cross_entropy_with_logits(col_logit_act, col)
                + F.binary_cross_entropy_with_logits(col_logit_int, col)
            )

            if regime_name == "bc_only_compact":
                loss = loss_bc + 0.3 * loss_col
            elif regime_name == "supervised_compact_no_teacher":
                loss_sub = 0.5 * (F.mse_loss(v_sub_act, v_macro_gt) + F.mse_loss(v_sub_intent, v_macro_gt))
                loss = loss_bc + 0.6 * loss_sub + 0.3 * loss_col
            else:
                # Full Teacher-Student Distillation + Ground-Truth Supervision
                loss_sub_gt = 0.5 * (F.mse_loss(v_sub_act, v_macro_gt) + F.mse_loss(v_sub_intent, v_macro_gt))
                loss_sub_distill = 0.5 * (F.mse_loss(v_sub_act, t_macro_16) + F.mse_loss(v_sub_intent, t_macro_16))
                loss_repr_distill = 0.5 * (F.mse_loss(v_curr, t_curr_16) + F.mse_loss(v_macro_online, t_macro_16))
                loss_act_distill = F.mse_loss(pred_act_from_sub, t_act)
                loss = (
                    0.7 * loss_bc
                    + 0.3 * loss_act_distill
                    + 0.4 * loss_sub_gt
                    + 0.4 * loss_sub_distill
                    + 0.4 * loss_repr_distill
                    + 0.3 * loss_col
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            ep_loss += float(loss.item())
            steps += 1

        # Validation (comparing v_sub_act against v_macro_gt, NOT v_curr!)
        model.eval()
        val_act_mse, val_sub_mse, v_steps = 0.0, 0.0, 0
        with torch.no_grad():
            for img_t, _, img_macro, intent_3d, act, _ in val_loader:
                img_t = img_t.to(device)
                img_macro = img_macro.to(device)
                intent_3d = intent_3d.to(device)
                act = act.to(device)

                v_curr = model.encode(img_t)
                v_macro_gt = model.encode(img_macro)
                v_sub_act, _ = model.forward_macro(v_curr, act)
                v_sub_intent, _ = model.forward_macro(v_curr, intent_3d)
                p_act = model.select_action(v_curr, v_sub_intent)

                val_act_mse += float(F.mse_loss(p_act, act).item())
                val_sub_mse += float(F.mse_loss(v_sub_act, v_macro_gt).item())
                v_steps += 1

        rec = {
            "epoch": epoch,
            "train_loss": ep_loss / max(steps, 1),
            "val_action_mse": val_act_mse / max(v_steps, 1),
            "val_dynamics_mse": val_sub_mse / max(v_steps, 1),
        }
        history.append(rec)

    return model, history


def train_reconstructed_model(epochs: int = 12):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v2_splits = str(_ACTIVE_SPLITS_DIR)
    legacy_splits = str(_ACTIVE_SPLITS_DIR)
    splits_dir = v2_splits if os.path.exists(os.path.join(v2_splits, "train_benchmark.npz")) else legacy_splits

    v2_ckpt = str(_ACTIVE_CKPT_DIR)
    legacy_ckpt = str(_ACTIVE_CKPT_DIR)
    ckpt_dir = v2_ckpt if os.path.exists(os.path.join(v2_ckpt, "hjepa_best.pt")) else legacy_ckpt

    out_dir = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    print("=================================================================")
    print("  PHASE 6: TEACHER-STUDENT DISTILLATION OF COMPACT MODEL (16D)")
    print(f"  Device: {device} | Splits: {splits_dir}")
    print("=================================================================")

    tr_ds = CompactDistillationDataset(os.path.join(splits_dir, "train_benchmark.npz"))
    va_ds = CompactDistillationDataset(os.path.join(splits_dir, "val_benchmark.npz"))
    tr_loader = DataLoader(tr_ds, batch_size=128, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=128, shuffle=False)

    teacher = HJEPAModel(stride_k=5).to(device)
    teacher.load_state_dict(torch.load(os.path.join(ckpt_dir, "hjepa_best.pt"), map_location=device, weights_only=False), strict=False)
    teacher.eval()

    with torch.no_grad():
        sample_imgs = tr_ds.rgb_t[:1024].to(device)
        z_t = teacher.encode(sample_imgs)
        z_c = z_t - z_t.mean(dim=0, keepdim=True)
        _, _, Vh = torch.linalg.svd(z_c, full_matrices=False)
        teacher_proj = Vh[:16, :].T.contiguous()

    regimes = {}
    for regime_name, ckpt_filename in (
        ("distilled_compact", "reconstructed_compact_best.pt"),
        ("supervised_compact_no_teacher", "reconstructed_supervised_only.pt"),
        ("bc_only_compact", "reconstructed_bc_only.pt"),
    ):
        print(f"\n--- Training Compact Regime: {regime_name} ---")
        mdl, hist = _train_regime(regime_name, tr_loader, va_loader, teacher, teacher_proj, device, epochs=epochs)
        save_path = os.path.join(out_dir, ckpt_filename)
        torch.save(mdl.state_dict(), save_path)
        if regime_name == "distilled_compact":
            torch.save(mdl.state_dict(), os.path.join(out_dir, "model_reconstructed_compact.pt"))
        regimes[regime_name] = {
            "checkpoint": save_path,
            "final_metrics": hist[-1],
            "history": hist,
        }
        print(
            f"  [{regime_name:30s}] Final Val Action MSE: {hist[-1]['val_action_mse']:.4f} | "
            f"Val Dyn MSE: {hist[-1]['val_dynamics_mse']:.4f}"
        )

    summary = {
        "architecture": "ReconstructedCompactModel",
        "total_parameters": ReconstructedCompactModel().count_parameters(),
        "latent_dim": 16,
        "hidden_dim": 32,
        "epochs": epochs,
        "regimes": regimes,
        "history": regimes["distilled_compact"]["history"],
    }
    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    train_reconstructed_model(epochs=12)
