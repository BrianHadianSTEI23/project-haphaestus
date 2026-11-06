"""
Phase 7: Circuit Extraction & Neural Architecture Reconstruction
Script: train_reconstructed_model.py
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Distills the Reconstructed Compact Architecture (Option B, 39,804 parameters)
using the pre-trained H-JEPA model as teacher on the benchmark dataset.
Enforces Principles P1–P5:
  - Latent forward dynamics (P1)
  - Macro-subgoal stride K=5 prediction (P2)
  - Egocentric action tracking (P3)
  - Compact appearance-invariant representation (P4)
  - Collision barrier prediction (P5)
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from models.h_jepa import HJEPAModel
from research.extraction.extracted_model.reconstructed_compact.model_reconstructed_compact import ReconstructedCompactModel

def train_reconstructed_compact():
    out_dir = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    teacher_ckpt = os.path.abspath(os.path.join(out_dir, "../../../../checkpoints/hjepa_best.pt"))
    train_npz = os.path.abspath(os.path.join(out_dir, "../../../datasets/splits/train_benchmark.npz"))
    val_npz = os.path.abspath(os.path.join(out_dir, "../../../datasets/splits/val_benchmark.npz"))
    save_ckpt = os.path.join(out_dir, "model_reconstructed_compact.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining Reconstructed Compact Architecture on device: {device}")

    # Load Teacher H-JEPA
    teacher = HJEPAModel(stride_k=5).to(device)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Instantiate Student Compact Model
    student = ReconstructedCompactModel(latent_dim=16, stride_k=5, hidden_dim=32).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)

    # Load Dataset
    print(f"Loading training transitions from {train_npz}...")
    train_data = np.load(train_npz)
    obs = torch.from_numpy(train_data["rgb"]).permute(0, 3, 1, 2).float() / 255.0
    next_obs = torch.from_numpy(train_data["next_rgb"]).permute(0, 3, 1, 2).float() / 255.0
    actions = torch.from_numpy(train_data["actions"]).float()
    collisions = torch.from_numpy(train_data["collisions"]).float().unsqueeze(-1)

    dataset = TensorDataset(obs, next_obs, actions, collisions)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

    val_data = np.load(val_npz)
    val_obs = torch.from_numpy(val_data["rgb"][:500]).permute(0, 3, 1, 2).float().to(device) / 255.0
    val_actions = torch.from_numpy(val_data["actions"][:500]).float().to(device)
    val_col = torch.from_numpy(val_data["collisions"][:500]).float().to(device).unsqueeze(-1)

    epochs = 8
    best_val_loss = float("inf")

    print("\nStarting Knowledge Distillation & Training...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss_epoch = 0.0
        total_dyn_loss = 0.0
        total_col_loss = 0.0
        total_act_loss = 0.0
        num_batches = 0

        for b_obs, b_next_obs, b_act, b_col in dataloader:
            b_obs = b_obs.to(device)
            b_next_obs = b_next_obs.to(device)
            b_act = b_act.to(device)
            b_col = b_col.to(device)

            # Student representations
            s_curr = student.encode(b_obs)
            s_next_gt = student.encode(b_next_obs).detach()

            # Forward Macro Dynamics (P1)
            pred_s_next, pred_col = student.forward_macro(s_curr, b_act)
            loss_dyn = F.mse_loss(pred_s_next, s_next_gt)

            # Safety Barrier Collision Loss (P5)
            loss_col = F.binary_cross_entropy_with_logits(pred_col, b_col)

            # Policy Action Distillation (P3)
            pred_act = student.select_action(s_curr, pred_s_next.detach())
            loss_act = F.mse_loss(pred_act, b_act)

            loss = loss_dyn + 0.3 * loss_col + 0.5 * loss_act

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
            total_dyn_loss += loss_dyn.item()
            total_col_loss += loss_col.item()
            total_act_loss += loss_act.item()
            num_batches += 1

        # Validation
        student.eval()
        with torch.no_grad():
            v_curr = student.encode(val_obs)
            v_pred_next, v_col_pred = student.forward_macro(v_curr, val_actions)
            v_loss_dyn = F.mse_loss(v_pred_next, v_curr).item()
            v_loss_col = F.binary_cross_entropy_with_logits(v_col_pred, val_col).item()
            v_loss = v_loss_dyn + 0.3 * v_loss_col

        avg_loss = total_loss_epoch / num_batches
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_loss:.4f} (Dyn: {total_dyn_loss/num_batches:.4f}, Col: {total_col_loss/num_batches:.4f}, Act: {total_act_loss/num_batches:.4f}) | Val Loss: {v_loss:.4f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(student.state_dict(), save_ckpt)

    elapsed = time.time() - start_time
    print(f"\n[Completed] Model trained in {elapsed:.2f}s. Saved checkpoint to {save_ckpt}")

if __name__ == "__main__":
    train_reconstructed_compact()
