import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.baseline_policy import BaselineNavigationPolicy
from models.le_world_model import LeWorldModel
from models.h_jepa import HJEPAModel
from env.amr_env import JetAutoEnv
from nlp.grounding import SemanticGoalGrounder

class NavigationTransitionDataset(Dataset):
    """
    Dataset wrapping (obs_t, action_t, obs_t+1, col_t) for LeWM, H-JEPA, and Baseline.
    """
    def __init__(self, npz_path, seq_len=6):
        data = np.load(npz_path)
        self.obs = data["obs"]               # (N, 64, 64, 3) uint8
        self.actions = data["actions"]       # (N, 3) float32
        self.collisions = data["collisions"] # (N,) bool
        self.ep_starts = data["episode_starts"]
        self.seq_len = seq_len
        self.total_len = len(self.obs)

        # Build contiguous sequence indices
        self.indices = []
        ep_indices = np.where(self.ep_starts)[0]
        ep_indices = np.append(ep_indices, self.total_len)

        for ep_i in range(len(ep_indices) - 1):
            s_idx = ep_indices[ep_i]
            e_idx = ep_indices[ep_i + 1]
            for t in range(s_idx, e_idx - seq_len):
                self.indices.append(t)

        print(f"Loaded {len(self.indices)} sequences (T={seq_len}) from {self.total_len} total transitions.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        T = self.seq_len

        # Images: (T, 3, 64, 64) normalized in [0, 1]
        imgs = torch.from_numpy(self.obs[t : t + T]).permute(0, 3, 1, 2).float() / 255.0
        acts = torch.from_numpy(self.actions[t : t + T]).float()
        cols = torch.from_numpy(self.collisions[t : t + T]).float().unsqueeze(-1)

        return imgs, acts, cols

def train_models(epochs=12, batch_size=64, seq_len=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f"  TRAINING PIPELINE: Baseline Policy, LeWM, H-JEPA")
    print(f"  Target Hardware: RTX 3050 Laptop GPU ({device})")
    print(f"============================================================")

    dataset_path = "/home/hadynata/vs-code-data/Python/world_models/dataset/amr_nav_15k.npz"
    dataset = NavigationTransitionDataset(dataset_path, seq_len=seq_len)
    
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    ckpt_dir = "/home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. TRAIN BASELINE NAVIGATION POLICY
    # ---------------------------------------------------------
    print("\n[1/3] Training Baseline Navigation Policy (Vision-Language / Goal-Conditioned)...")
    baseline = BaselineNavigationPolicy().to(device)
    base_opt = torch.optim.AdamW(baseline.parameters(), lr=1e-3, weight_decay=1e-4)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=device))

    baseline_losses = []
    for epoch in range(1, epochs + 1):
        baseline.train()
        ep_loss = 0.0
        for imgs, acts, cols in train_loader:
            imgs, acts, cols = imgs.to(device), acts.to(device), cols.to(device)
            # Create synthetic goal vector towards forward direction with jitter
            dx_rel = acts[:, 0, 0:1] * 1.5 + torch.randn_like(acts[:, 0, 0:1]) * 0.1
            dy_rel = acts[:, 0, 1:2] * 1.5 + torch.randn_like(acts[:, 0, 1:2]) * 0.1
            sin_th = torch.sin(acts[:, 0, 2:3])
            cos_th = torch.cos(acts[:, 0, 2:3])
            goal_vec = torch.cat([dx_rel, dy_rel, sin_th, cos_th], dim=-1)

            base_opt.zero_grad()
            (pred_act, pred_col), _ = baseline(imgs[:, 0], goal_vec)
            loss_act = mse_fn(pred_act, acts[:, 0])
            loss_col = bce_fn(pred_col, cols[:, 0])
            loss = loss_act + 0.3 * loss_col

            loss.backward()
            base_opt.step()
            ep_loss += loss.item() * len(imgs)

        ep_loss /= len(train_ds)
        baseline_losses.append(ep_loss)
        if epoch % 3 == 0 or epoch == epochs:
            print(f"  [Baseline] Epoch {epoch:02d}/{epochs:02d} - Loss: {ep_loss:.4f}")

    torch.save(baseline.state_dict(), os.path.join(ckpt_dir, "baseline_best.pt"))
    print("  -> Baseline checkpoint saved to checkpoints/baseline_best.pt")

    # ---------------------------------------------------------
    # 2. TRAIN LeWorldModel (LeWM) with SIGReg
    # ---------------------------------------------------------
    print("\n[2/3] Training LeWorldModel (arXiv:2603.19312) with SIGReg Loss...")
    lewm = LeWorldModel().to(device)
    lewm_opt = torch.optim.AdamW(lewm.parameters(), lr=1e-3, weight_decay=1e-4)

    lewm_losses = []
    for epoch in range(1, epochs + 1):
        lewm.train()
        ep_loss = 0.0
        ep_sigreg = 0.0
        for imgs, acts, cols in train_loader:
            imgs, acts, cols = imgs.to(device), acts.to(device), cols.to(device)
            lewm_opt.zero_grad()
            loss, metrics = lewm.compute_loss(imgs[:, 0], acts[:, 0], imgs[:, 1], cols[:, 0], lambda_reg=0.25)
            loss.backward()
            lewm_opt.step()
            ep_loss += loss.item() * len(imgs)
            ep_sigreg += metrics["loss_sigreg"] * len(imgs)

        ep_loss /= len(train_ds)
        ep_sigreg /= len(train_ds)
        lewm_losses.append(ep_loss)
        if epoch % 3 == 0 or epoch == epochs:
            print(f"  [LeWM] Epoch {epoch:02d}/{epochs:02d} - Total Loss: {ep_loss:.4f} | SIGReg: {ep_sigreg:.4f}")

    torch.save(lewm.state_dict(), os.path.join(ckpt_dir, "lewm_best.pt"))
    print("  -> LeWM checkpoint saved to checkpoints/lewm_best.pt")

    # ---------------------------------------------------------
    # 3. TRAIN H-JEPA (Hierarchical JEPA)
    # ---------------------------------------------------------
    print("\n[3/3] Training H-JEPA (arXiv:2610.06805) with Macro-Stride K=5...")
    hjepa = HJEPAModel(stride_k=5).to(device)
    hjepa_opt = torch.optim.AdamW(hjepa.parameters(), lr=1e-3, weight_decay=1e-4)

    hjepa_losses = []
    for epoch in range(1, epochs + 1):
        hjepa.train()
        ep_loss = 0.0
        ep_macro = 0.0
        for imgs, acts, cols in train_loader:
            imgs, acts, cols = imgs.to(device), acts.to(device), cols.to(device)
            hjepa_opt.zero_grad()
            loss, metrics = hjepa.compute_loss(imgs, acts, cols, lambda_reg=0.20)
            loss.backward()
            hjepa_opt.step()
            ep_loss += loss.item() * len(imgs)
            ep_macro += metrics["loss_macro_pred"] * len(imgs)

        ep_loss /= len(train_ds)
        ep_macro /= len(train_ds)
        hjepa_losses.append(ep_loss)
        if epoch % 3 == 0 or epoch == epochs:
            print(f"  [H-JEPA] Epoch {epoch:02d}/{epochs:02d} - Total Loss: {ep_loss:.4f} | Macro MSE: {ep_macro:.4f}")

    torch.save(hjepa.state_dict(), os.path.join(ckpt_dir, "hjepa_best.pt"))
    print("  -> H-JEPA checkpoint saved to checkpoints/hjepa_best.pt")

    # Generate training curves plot
    plt.figure(figsize=(10, 4), dpi=150)
    plt.plot(baseline_losses, label="Baseline Policy Loss", color="royalblue", lw=2)
    plt.plot(lewm_losses, label="LeWorldModel (LeWM + SIGReg) Loss", color="darkorange", lw=2)
    plt.plot(hjepa_losses, label="H-JEPA (Hierarchical World Model) Loss", color="forestgreen", lw=2)
    plt.title("Training Convergence: Baseline vs. LeWM vs. H-JEPA on AMR Dynamics", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(ckpt_dir, "training_curves_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\n[OK] Training curves saved to: {plot_path}")
    print(f"All 3 model weights trained and ready!")

if __name__ == "__main__":
    train_models(epochs=12, batch_size=64)
