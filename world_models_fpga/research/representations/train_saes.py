"""
Phase 4: Sparse Autoencoder (SAE) Training Pipeline
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Trains Sparse Autoencoders across:
  - Models: LeWorldModel (s_t), H-JEPA (s_t), Baseline Policy (z)
  - Dictionary Expansion Sizes: 64 (2x), 128 (4x), 256 (8x)
  - Random Initialization Seeds: 42, 1337 (for stability analysis)
Evaluates:
  - Reconstruction Error (MSE, R^2)
  - L0 Sparsity
  - Dead Feature Fraction
  - Activation Frequency Distribution
"""

import sys as _hsys
from pathlib import Path as _HPath
_hsys.path.insert(0, str(_HPath(__file__).resolve().parents[2]))
from research.datasets.dataset_config import SPLITS_DIR as _ACTIVE_SPLITS_DIR, CKPT_DIR as _ACTIVE_CKPT_DIR  # noqa: E402
import os
import sys
import json
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from models.baseline_policy import BaselineNavigationPolicy
from models.le_world_model import LeWorldModel
from models.h_jepa import HJEPAModel
from research.representations.sae_model import SparseAutoencoder

def extract_activations(data_npz_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Extracts frozen visual latent representations for Baseline (z), LeWM (s), and H-JEPA (s).
    """
    print(f"Loading dataset from: {data_npz_path}")
    data = np.load(data_npz_path)
    obs = data["rgb"] # (N, 64, 64, 3)
    N = len(obs)
    print(f"Extracting latent activations across {N} frames on {device}...")

    v2_ckpt = str(_ACTIVE_CKPT_DIR)
    legacy_ckpt = str(_ACTIVE_CKPT_DIR)
    ckpt_dir = v2_ckpt if os.path.exists(os.path.join(v2_ckpt, "lewm_best.pt")) else legacy_ckpt

    base_model = BaselineNavigationPolicy().to(device)
    base_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "baseline_best.pt"), map_location=device, weights_only=False), strict=False)
    base_model.eval()

    lewm_model = LeWorldModel().to(device)
    lewm_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "lewm_best.pt"), map_location=device, weights_only=False), strict=False)
    lewm_model.eval()

    hjepa_model = HJEPAModel(stride_k=5).to(device)
    hjepa_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "hjepa_best.pt"), map_location=device, weights_only=False), strict=False)
    hjepa_model.eval()

    batch_size = 256
    num_batches = (N + batch_size - 1) // batch_size

    base_z_list = []
    lewm_s_list = []
    hjepa_s_list = []

    with torch.no_grad():
        for b in range(num_batches):
            s_idx = b * batch_size
            e_idx = min(N, (b + 1) * batch_size)
            imgs = torch.from_numpy(obs[s_idx:e_idx]).permute(0, 3, 1, 2).float().to(device) / 255.0

            z_b = base_model.encoder(imgs)
            s_l = lewm_model.encode(imgs)
            s_h = hjepa_model.encode(imgs)

            base_z_list.append(z_b.cpu())
            lewm_s_list.append(s_l.cpu())
            hjepa_s_list.append(s_h.cpu())

    return {
        "baseline_z": torch.cat(base_z_list, dim=0),
        "lewm_s": torch.cat(lewm_s_list, dim=0),
        "hjepa_s": torch.cat(hjepa_s_list, dim=0)
    }

def train_single_sae(h_tensor, dict_dim, seed=42, epochs=25, batch_size=128, lr=1e-3, l1_coeff=0.005, device="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    act_dim = h_tensor.shape[-1]
    sae = SparseAutoencoder(act_dim=act_dim, dict_dim=dict_dim, l1_coeff=l1_coeff).to(device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr, weight_decay=1e-5)

    dataset = TensorDataset(h_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    feature_activation_counts = torch.zeros(dict_dim, device=device)
    total_samples = 0

    for epoch in range(1, epochs + 1):
        sae.train()
        epoch_recon = 0.0
        epoch_l1 = 0.0
        for (batch_h,) in loader:
            batch_h = batch_h.to(device)
            optimizer.zero_grad()
            loss, metrics = sae.compute_loss(batch_h)
            loss.backward()
            optimizer.step()
            sae.normalize_decoder()

            epoch_recon += metrics["loss_recon"] * len(batch_h)
            epoch_l1 += metrics["loss_l1"] * len(batch_h)

    # Final evaluation pass over full dataset
    sae.eval()
    all_z = []
    all_recon = []
    with torch.no_grad():
        for (batch_h,) in DataLoader(dataset, batch_size=256, shuffle=False):
            batch_h = batch_h.to(device)
            h_hat, z = sae(batch_h)
            all_z.append(z)
            all_recon.append(h_hat)
            feature_activation_counts += (z > 0.0).sum(dim=0)
            total_samples += len(batch_h)

    full_z = torch.cat(all_z, dim=0)
    full_h_hat = torch.cat(all_recon, dim=0)
    full_h = h_tensor.to(device)

    # Compute final metrics
    mse_recon = float(F.mse_loss(full_h_hat, full_h).item())
    total_var = float(torch.var(full_h, dim=0).sum().item())
    resid_var = float(torch.var(full_h - full_h_hat, dim=0).sum().item())
    r2_score = float(max(0.0, 1.0 - (resid_var / (total_var + 1e-8))))

    l0_per_sample = float(torch.mean((full_z > 0.0).float().sum(dim=-1)).item())
    dead_features = int((feature_activation_counts == 0).sum().item())
    dead_fraction = float(dead_features / dict_dim)

    results = {
        "dict_dim": dict_dim,
        "seed": seed,
        "epochs": epochs,
        "mse_recon": mse_recon,
        "r2_score": r2_score,
        "l0_sparsity": l0_per_sample,
        "dead_features": dead_features,
        "dead_fraction": dead_fraction,
        "activation_frequency": (feature_activation_counts / total_samples).cpu().tolist()
    }
    return sae, results

def train_all_saes(train_npz, output_dir=None):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "sae_features"))
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activations = extract_activations(train_npz, device=device)

    dict_dims = [64, 128, 256]
    seeds = [42, 1337]

    all_sae_metrics = {}
    csv_rows = []

    print("\n=======================================================")
    print("  PHASE 4: SPARSE AUTOENCODER (SAE) TRAINING")
    print(f"  Dictionary Dims: {dict_dims} | Seeds: {seeds}")
    print(f"  Target Device: {device}")
    print("=======================================================\n")

    for model_key, h_tensor in activations.items():
        print(f"\n--- Training SAEs on [{model_key.upper()}] (h in R^{h_tensor.shape[-1]}) ---")
        all_sae_metrics[model_key] = {}

        for dict_dim in dict_dims:
            seed_decoders = {}
            for s in seeds:
                t0 = time.time()
                sae, metrics = train_single_sae(
                    h_tensor, dict_dim=dict_dim, seed=s, epochs=20, batch_size=128, device=device
                )
                t1 = time.time()

                # Store normalized decoder atoms (act_dim, dict_dim) -> transpose to (dict_dim, act_dim)
                w_dec = sae.W_d.detach().cpu().T
                w_dec = w_dec / (torch.norm(w_dec, dim=-1, keepdim=True) + 1e-8)
                seed_decoders[s] = w_dec

                run_id = f"{model_key}_d{dict_dim}_s{s}"
                all_sae_metrics[model_key][run_id] = metrics

                # Save trained model checkpoint
                ckpt_path = os.path.join(output_dir, f"{run_id}.pt")
                torch.save({
                    "model_key": model_key,
                    "dict_dim": dict_dim,
                    "seed": s,
                    "state_dict": sae.state_dict(),
                    "metrics": metrics
                }, ckpt_path)
                if dict_dim == 128 and s == 42:
                    short_name = {"baseline_z": "Baseline", "lewm_s": "LeWM", "hjepa_s": "H-JEPA"}[model_key]
                    res_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
                    os.makedirs(res_dir, exist_ok=True)
                    torch.save(sae.state_dict(), os.path.join(res_dir, f"sae_{short_name}.pt"))

                csv_rows.append([
                    model_key, dict_dim, s,
                    f"{metrics['r2_score']:.4f}",
                    f"{metrics['mse_recon']:.5f}",
                    f"{metrics['l0_sparsity']:.1f}",
                    metrics['dead_features'],
                    f"{metrics['dead_fraction']*100:.1f}%",
                    f"{t1 - t0:.1f}s"
                ])

                print(f"  [{run_id:20s}] R^2: {metrics['r2_score']:.4f} | "
                      f"MSE: {metrics['mse_recon']:.5f} | "
                      f"L0: {metrics['l0_sparsity']:4.1f}/{dict_dim} | "
                      f"Dead: {metrics['dead_features']:2d} ({metrics['dead_fraction']*100:4.1f}%) | "
                      f"Time: {t1-t0:4.1f}s")

            # Compute cross-seed dictionary atom stability between seed 42 and seed 1337
            if 42 in seed_decoders and 1337 in seed_decoders:
                cos_mat = torch.abs(seed_decoders[42] @ seed_decoders[1337].T)
                max_match_42 = float(torch.max(cos_mat, dim=1)[0].mean().item())
                max_match_1337 = float(torch.max(cos_mat, dim=0)[0].mean().item())
                stability = 0.5 * (max_match_42 + max_match_1337)
                all_sae_metrics[model_key][f"d{dict_dim}_cross_seed_stability"] = stability
                print(f"  >> [Dict={dict_dim}] Cross-Seed Atom Stability (Seed 42 vs 1337): {stability:.4f}")

    # Save summary JSON and CSV
    json_path = os.path.join(output_dir, "sae_training_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_sae_metrics, f, indent=2)
    res_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "sae_analysis.json"), "w") as f:
        json.dump(all_sae_metrics, f, indent=2)

    csv_path = os.path.join(output_dir, "sae_training_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model Activation", "Dict Dim", "Seed", "R^2 Score", "Recon MSE", "L0 Sparsity", "Dead Features", "Dead %", "Train Time"])
        writer.writerows(csv_rows)

    print(f"\nSaved SAE training results to:")
    print(f"  -> {json_path}")
    print(f"  -> {csv_path}")
    return all_sae_metrics

if __name__ == "__main__":
    v2_npz = str(_ACTIVE_SPLITS_DIR / "train_benchmark.npz")
    legacy_npz = str(_ACTIVE_SPLITS_DIR / "train_benchmark.npz")
    train_all_saes(v2_npz if os.path.exists(v2_npz) else legacy_npz)
