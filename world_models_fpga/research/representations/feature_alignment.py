"""
Phase 4: Feature Stability & Representation Alignment Analysis
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Computes:
1. Centered Kernel Alignment (Linear CKA) across:
   - Baseline (z) vs LeWM (s) vs H-JEPA (s) vs H-JEPA Macro (S)
2. SAE Dictionary Cosine Alignment across random initialization seeds:
   - Seed 42 vs Seed 1337 for each dictionary dimension (d=64, 128, 256)
   - Measures maximum cosine similarity and reproducible feature fraction (|cos| >= 0.85)
3. Cross-layer / Cross-model feature correlation
"""

import sys as _hsys
from pathlib import Path as _HPath
_hsys.path.insert(0, str(_HPath(__file__).resolve().parents[2]))
from research.datasets.dataset_config import SPLITS_DIR as _ACTIVE_SPLITS_DIR, CKPT_DIR as _ACTIVE_CKPT_DIR  # noqa: E402
import os
import sys
import json
import csv
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from models.baseline_policy import BaselineNavigationPolicy
from models.le_world_model import LeWorldModel
from models.h_jepa import HJEPAModel
from research.representations.sae_model import SparseAutoencoder

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Computes Linear Centered Kernel Alignment (CKA) between representations X and Y.
    Numerically efficient via Frobenius norm of cross-covariance:
      CKA(X, Y) = ||Y_c^T X_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)
    """
    X_c = X - np.mean(X, axis=0)
    Y_c = Y - np.mean(Y, axis=0)

    # Cross-covariance
    cross_cov = Y_c.T @ X_c
    cross_norm_sq = np.sum(cross_cov ** 2)

    # Auto-covariances
    x_cov = X_c.T @ X_c
    y_cov = Y_c.T @ Y_c
    x_norm = np.linalg.norm(x_cov, 'fro')
    y_norm = np.linalg.norm(y_cov, 'fro')

    if x_norm < 1e-12 or y_norm < 1e-12:
        return 0.0

    cka = cross_norm_sq / (x_norm * y_norm)
    return float(np.clip(cka, 0.0, 1.0))

def compute_sae_dictionary_alignment(sae_ckpt_1: str, sae_ckpt_2: str, threshold: float = 0.85):
    """
    Computes bipartite cosine alignment between decoder dictionary columns W_d of two SAEs.
    Only considers non-dead features (active in training).
    """
    data1 = torch.load(sae_ckpt_1, map_location="cpu")
    data2 = torch.load(sae_ckpt_2, map_location="cpu")

    W_d1 = data1["state_dict"]["W_d"].numpy() # (act_dim, dict_dim)
    W_d2 = data2["state_dict"]["W_d"].numpy()

    # Filter out dead features
    freq1 = np.array(data1["metrics"]["activation_frequency"])
    freq2 = np.array(data2["metrics"]["activation_frequency"])

    active_idx1 = np.where(freq1 > 0.0001)[0]
    active_idx2 = np.where(freq2 > 0.0001)[0]

    if len(active_idx1) == 0 or len(active_idx2) == 0:
        return {
            "mean_max_cosine": 0.0,
            "reproducible_fraction": 0.0,
            "reproducible_count": 0,
            "active_features_1": len(active_idx1),
            "active_features_2": len(active_idx2)
        }

    # Normalize columns
    col1 = W_d1[:, active_idx1] / (np.linalg.norm(W_d1[:, active_idx1], axis=0, keepdims=True) + 1e-8)
    col2 = W_d2[:, active_idx2] / (np.linalg.norm(W_d2[:, active_idx2], axis=0, keepdims=True) + 1e-8)

    # Pairwise cosine matrix: (num_act1, num_act2)
    cosine_matrix = np.abs(col1.T @ col2)

    # For each feature in dictionary 1, find its best matching feature in dictionary 2
    max_match_1to2 = np.max(cosine_matrix, axis=1)
    reproducible = np.sum(max_match_1to2 >= threshold)

    return {
        "mean_max_cosine": float(np.mean(max_match_1to2)),
        "max_cosine": float(np.max(max_match_1to2)),
        "min_cosine": float(np.min(max_match_1to2)),
        "reproducible_fraction": float(reproducible / len(active_idx1)),
        "reproducible_count": int(reproducible),
        "active_features_1": int(len(active_idx1)),
        "active_features_2": int(len(active_idx2))
    }

def run_feature_alignment_analysis(val_npz_path, sae_dir=None, output_dir=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "feature_alignment"))
    if sae_dir is None:
        sae_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "sae_features"))
    os.makedirs(output_dir, exist_ok=True)

    print("\n=======================================================")
    print("  PHASE 4: REPRESENTATION ALIGNMENT & STABILITY (CKA)")
    print(f"  Validation Suite: {val_npz_path}")
    print(f"  Device: {device}")
    print("=======================================================\n")

    val_data = np.load(val_npz_path)
    obs = val_data["rgb"] # (N, 64, 64, 3)
    N = len(obs)

    ckpt_dir = str(_ACTIVE_CKPT_DIR)

    base_model = BaselineNavigationPolicy().to(device)
    base_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "baseline_best.pt"), map_location=device))
    base_model.eval()

    lewm_model = LeWorldModel().to(device)
    lewm_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "lewm_best.pt"), map_location=device))
    lewm_model.eval()

    hjepa_model = HJEPAModel(stride_k=5).to(device)
    hjepa_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "hjepa_best.pt"), map_location=device))
    hjepa_model.eval()

    # Extract representations across all validation frames
    z_base_list, s_lewm_list, s_hjepa_list, S_macro_list = [], [], [], []
    batch_size = 256
    with torch.no_grad():
        for b in range(0, N, batch_size):
            imgs = torch.from_numpy(obs[b : b + batch_size]).permute(0, 3, 1, 2).float().to(device) / 255.0
            zb = base_model.encoder(imgs)
            sl = lewm_model.encode(imgs)
            sh = hjepa_model.encode(imgs)
            # Compute macro-predicted state across horizon K=5
            a_fwd = torch.tensor([[0.8, 0.0, 0.0]], device=device).repeat(len(imgs), 1)
            sm, _ = hjepa_model.macro_predictor(sh, a_fwd)

            z_base_list.append(zb.cpu().numpy())
            s_lewm_list.append(sl.cpu().numpy())
            s_hjepa_list.append(sh.cpu().numpy())
            S_macro_list.append(sm.cpu().numpy())

    rep_dict = {
        "Baseline (z)": np.concatenate(z_base_list, axis=0),
        "LeWM (s_t)": np.concatenate(s_lewm_list, axis=0),
        "H-JEPA Micro (s_t)": np.concatenate(s_hjepa_list, axis=0),
        "H-JEPA Macro (S_t)": np.concatenate(S_macro_list, axis=0)
    }

    # 1. Compute Pairwise CKA Matrix
    rep_names = list(rep_dict.keys())
    num_reps = len(rep_names)
    cka_matrix = np.zeros((num_reps, num_reps))

    print("--- Computing Pairwise Linear CKA Similarity ---")
    for i in range(num_reps):
        for j in range(num_reps):
            if i == j:
                cka_matrix[i, j] = 1.0
            elif i < j:
                val = linear_cka(rep_dict[rep_names[i]], rep_dict[rep_names[j]])
                cka_matrix[i, j] = val
                cka_matrix[j, i] = val
            print(f"  CKA({rep_names[i]:18s}, {rep_names[j]:18s}) = {cka_matrix[i, j]:.4f}")

    cka_results = {
        "representations": rep_names,
        "cka_matrix": cka_matrix.tolist()
    }

    # 2. Compute SAE Stability Across Random Seeds (Seed 42 vs 1337)
    sae_stability_results = {}
    stability_csv_rows = []

    dict_dims = [64, 128, 256]
    models = ["baseline_z", "lewm_s", "hjepa_s"]

    print("\n--- Computing SAE Dictionary Seed Stability (Seed 42 vs 1337) ---")
    for m in models:
        sae_stability_results[m] = {}
        for d in dict_dims:
            ckpt1 = os.path.join(sae_dir, f"{m}_d{d}_s42.pt")
            ckpt2 = os.path.join(sae_dir, f"{m}_d{d}_s1337.pt")

            if os.path.exists(ckpt1) and os.path.exists(ckpt2):
                stab = compute_sae_dictionary_alignment(ckpt1, ckpt2, threshold=0.85)
                sae_stability_results[m][f"d{d}"] = stab
                print(f"  [{m.upper():12s} d={d:3d}] Mean Max Cosine: {stab['mean_max_cosine']:.4f} | "
                      f"Reproducible (>0.85): {stab['reproducible_fraction']*100:4.1f}% "
                      f"({stab['reproducible_count']}/{stab['active_features_1']})")
                stability_csv_rows.append([
                    m, d, f"{stab['mean_max_cosine']:.4f}", f"{stab['max_cosine']:.4f}",
                    f"{stab['reproducible_fraction']*100:.1f}%", stab['reproducible_count'], stab['active_features_1']
                ])
            else:
                print(f"  Skipping {m} d={d}: checkpoints not found yet.")

    # Save outputs
    cka_json = os.path.join(output_dir, "cka_similarity.json")
    with open(cka_json, "w") as f:
        json.dump(cka_results, f, indent=2)

    stab_json = os.path.join(output_dir, "sae_seed_stability.json")
    with open(stab_json, "w") as f:
        json.dump(sae_stability_results, f, indent=2)

    stab_csv = os.path.join(output_dir, "sae_seed_stability.csv")
    with open(stab_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Dict Dim", "Mean Max Cosine", "Peak Cosine", "Reproducible Fraction", "Reproducible Count", "Active Count"])
        writer.writerows(stability_csv_rows)

    print(f"\nSaved Alignment & Stability results to:")
    print(f"  -> {cka_json}")
    print(f"  -> {stab_json}")
    print(f"  -> {stab_csv}")

    return cka_results, sae_stability_results

if __name__ == "__main__":
    val_npz = str(_ACTIVE_SPLITS_DIR / "val_benchmark.npz")
    run_feature_alignment_analysis(val_npz)
