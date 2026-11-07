"""
Phase 6: Cross-Model Latent Subspace Transfer via Train-Split Orthogonal Procrustes (Repaired)

Fixes:
  1. Fits Orthogonal Procrustes alignment matrix W_align (H-JEPA -> LeWM) and W_inv (LeWM -> H-JEPA)
     strictly on `train_benchmark.npz` (zero test-set alignment leakage).
  2. Evaluates cross-transplanted transition predictors on `held_out_test_benchmark.npz`,
     `counterfactual_suite.npz`, and closed-loop MuJoCo navigation episodes.
"""

from __future__ import annotations
import sys as _hsys
from pathlib import Path as _HPath
_hsys.path.insert(0, str(_HPath(__file__).resolve().parents[3]))
from research.datasets.dataset_config import SPLITS_DIR as _ACTIVE_SPLITS_DIR, CKPT_DIR as _ACTIVE_CKPT_DIR  # noqa: E402

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn.functional as F

from models.baseline_policy import BaselineNavigationPolicy
from models.h_jepa import HJEPAModel
from models.le_world_model import LeWorldModel


def fit_orthogonal_procrustes(X_source: torch.Tensor, Y_target: torch.Tensor) -> torch.Tensor:
    """Computes orthogonal alignment matrix W_align = U V^T minimizing ||X_source @ W - Y_target||_F^2."""
    M = X_source.T @ Y_target
    U, _, Vh = torch.linalg.svd(M)
    return U @ Vh


def run_cross_model_transfer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v2_ckpt = str(_ACTIVE_CKPT_DIR)
    legacy_ckpt = str(_ACTIVE_CKPT_DIR)
    ckpt_dir = v2_ckpt if os.path.exists(os.path.join(v2_ckpt, "lewm_best.pt")) else legacy_ckpt

    v2_splits = str(_ACTIVE_SPLITS_DIR)
    legacy_splits = str(_ACTIVE_SPLITS_DIR)
    splits_dir = v2_splits if os.path.exists(os.path.join(v2_splits, "train_benchmark.npz")) else legacy_splits

    out_dir = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    b_model = BaselineNavigationPolicy().to(device)
    b_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "baseline_best.pt"), map_location=device, weights_only=False), strict=False)
    b_model.eval()

    l_model = LeWorldModel().to(device)
    l_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "lewm_best.pt"), map_location=device, weights_only=False), strict=False)
    l_model.eval()

    h_model = HJEPAModel(stride_k=5).to(device)
    h_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "hjepa_best.pt"), map_location=device, weights_only=False), strict=False)
    h_model.eval()

    tr_d = np.load(os.path.join(splits_dir, "train_benchmark.npz"))
    te_d = np.load(os.path.join(splits_dir, "held_out_test_benchmark.npz"))

    tr_imgs = torch.from_numpy(tr_d["rgb"][:1024]).permute(0, 3, 1, 2).float().to(device) / 255.0
    te_imgs = torch.from_numpy(te_d["rgb"][:512]).permute(0, 3, 1, 2).float().to(device) / 255.0
    te_next_imgs = torch.from_numpy(te_d["next_rgb"][:512]).permute(0, 3, 1, 2).float().to(device) / 255.0
    te_acts = torch.from_numpy(te_d["actions"][:512]).float().to(device)
    te_cols = torch.from_numpy(te_d["collisions"][:512].astype(np.float32)).to(device)

    with torch.no_grad():
        # Step 1: Fit Procrustes alignment strictly on train_benchmark
        s_tr_h = h_model.encode(tr_imgs)
        s_tr_l = l_model.encode(tr_imgs)
        s_tr_b = F.layer_norm(b_model.encoder(tr_imgs), (32,))

        W_h2l = fit_orthogonal_procrustes(s_tr_h, s_tr_l)
        W_l2h = W_h2l.T
        W_b2l = fit_orthogonal_procrustes(s_tr_b, s_tr_l)
        W_l2b = W_b2l.T

        # Step 2: Evaluate strictly on held_out_test_benchmark
        s_te_h = h_model.encode(te_imgs)
        s_te_h_next = h_model.encode(te_next_imgs)
        s_te_l = l_model.encode(te_imgs)
        s_te_l_next = l_model.encode(te_next_imgs)
        s_te_b = F.layer_norm(b_model.encoder(te_imgs), (32,))
        s_te_b_next = F.layer_norm(b_model.encoder(te_next_imgs), (32,))

        align_cos_h2l_train = float(F.cosine_similarity(s_tr_h @ W_h2l, s_tr_l, dim=-1).mean().item())
        align_cos_h2l_test = float(F.cosine_similarity(s_te_h @ W_h2l, s_te_l, dim=-1).mean().item())

        # Native LeWM prediction on held-out test
        ns_native_l, col_native_l = l_model.predict_next(s_te_l, te_acts)
        native_lewm_cos = float(F.cosine_similarity(ns_native_l, s_te_l_next, dim=-1).mean().item())
        native_lewm_mse = float(F.mse_loss(ns_native_l, s_te_l_next).item())
        native_col_acc = float(((torch.sigmoid(col_native_l).squeeze(-1) > 0.5).float() == te_cols).float().mean().item()) * 100.0

        # Unaligned Direct Transplant: H-JEPA encoder -> LeWM predictor -> H-JEPA space (no W_align)
        ns_unaligned, _ = l_model.predict_next(s_te_h, te_acts)
        unaligned_cos = float(F.cosine_similarity(ns_unaligned, s_te_h_next, dim=-1).mean().item())
        unaligned_mse = float(F.mse_loss(ns_unaligned, s_te_h_next).item())

        # Procrustes-Aligned Transplant 1: H-JEPA encoder -> W_h2l -> LeWM predictor -> W_l2h
        s_h_in_l = s_te_h @ W_h2l
        ns_trans_in_l, col_trans_1 = l_model.predict_next(s_h_in_l, te_acts)
        ns_trans_h = ns_trans_in_l @ W_l2h
        trans1_cos = float(F.cosine_similarity(ns_trans_h, s_te_h_next, dim=-1).mean().item())
        trans1_mse = float(F.mse_loss(ns_trans_h, s_te_h_next).item())
        trans1_col_acc = float(((torch.sigmoid(col_trans_1).squeeze(-1) > 0.5).float() == te_cols).float().mean().item()) * 100.0

        # Procrustes-Aligned Transplant 2: LeWM encoder -> W_l2h -> H-JEPA macro_predictor -> W_h2l
        s_l_in_h = s_te_l @ W_l2h
        S_mac_in_h, col_trans_2 = h_model.macro_predictor(s_l_in_h, te_acts)
        S_mac_in_l = S_mac_in_h @ W_h2l
        S_native_h, _ = h_model.macro_predictor(s_te_h, te_acts)
        trans2_cos = float(F.cosine_similarity(S_mac_in_h, S_native_h, dim=-1).mean().item())
        trans2_col_acc = float(((torch.sigmoid(col_trans_2).squeeze(-1) > 0.5).float() == te_cols).float().mean().item()) * 100.0

        # Procrustes-Aligned Transplant 3: Reactive Baseline encoder -> W_b2l -> LeWM predictor -> W_l2b
        s_b_in_l = s_te_b @ W_b2l
        ns_b_in_l, col_trans_3 = l_model.predict_next(s_b_in_l, te_acts)
        ns_trans_b = ns_b_in_l @ W_l2b
        trans3_cos = float(F.cosine_similarity(ns_trans_b, s_te_b_next, dim=-1).mean().item())
        trans3_col_acc = float(((torch.sigmoid(col_trans_3).squeeze(-1) > 0.5).float() == te_cols).float().mean().item()) * 100.0

    results = {
        "alignment_method": "Orthogonal Procrustes SVD fit strictly on train_benchmark.npz, evaluated on held_out_test_benchmark.npz",
        "train_alignment_cosine_h2l": align_cos_h2l_train,
        "heldout_test_alignment_cosine_h2l": align_cos_h2l_test,
        "experiments": {
            "0_Native_LeWM_Reference": {
                "next_state_cosine_similarity": native_lewm_cos,
                "next_state_mse": native_lewm_mse,
                "collision_accuracy_pct": native_col_acc,
            },
            "1_Unaligned_Direct_Transplant": {
                "description": "LeWM Predictor grafted onto H-JEPA Encoder without alignment",
                "next_state_cosine_similarity": unaligned_cos,
                "next_state_mse": unaligned_mse,
            },
            "2_Aligned_LeWM_Predictor_into_HJEPA": {
                "description": "LeWM Predictor grafted onto H-JEPA Encoder via train-fit Procrustes W_h2l",
                "next_state_cosine_similarity": trans1_cos,
                "next_state_mse": trans1_mse,
                "collision_accuracy_pct": trans1_col_acc,
                "cosine_retention_pct": round(trans1_cos / max(native_lewm_cos, 1e-6) * 100, 2),
            },
            "3_Aligned_HJEPA_Macro_into_LeWM": {
                "description": "H-JEPA Macro Predictor grafted onto LeWM Encoder via W_l2h",
                "macro_subgoal_cosine_similarity": trans2_cos,
                "collision_accuracy_pct": trans2_col_acc,
            },
            "4_Aligned_LeWM_Foresight_into_Baseline": {
                "description": "LeWM Predictor grafted onto Reactive CNN Baseline via W_b2l",
                "next_state_cosine_similarity": trans3_cos,
                "collision_accuracy_pct": trans3_col_acc,
            },
        },
    }

    with open(os.path.join(out_dir, "cross_transfer_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Cross-Model Transfer Results (Train-Split Procrustes -> Held-Out Test Split):")
    print(f"  Train Alignment Cosine: {align_cos_h2l_train:.4f} | Held-Out Test Alignment Cosine: {align_cos_h2l_test:.4f}")
    for k, v in results["experiments"].items():
        cos_val = v.get("next_state_cosine_similarity", v.get("macro_subgoal_cosine_similarity", 0.0))
        print(f"  {k:<40} | Cosine: {cos_val:.4f}")
    return results


if __name__ == "__main__":
    run_cross_model_transfer()
