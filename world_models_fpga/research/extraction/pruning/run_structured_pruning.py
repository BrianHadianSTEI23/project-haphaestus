"""
Phase 6: Structured Channel & Subspace Pruning Pipeline (Repaired)

Fixes:
  1. Eliminates the synthetic `int(p4 - pruned_fine * 0.3)` parameter estimate.
  2. Explicitly separates:
       - `structural_parameters`: dense tensor allocation in PyTorch modules
       - `active_nonzero_parameters`: exact count of unmasked weights (|w| > 1e-8)
       - `structural_compression_ratio` vs `active_nonzero_compression_ratio`
  3. Uses `<CKPT_DIR>/hjepa_best.pt` and `<SPLITS_DIR>/val_benchmark.npz` (see research/datasets/dataset_config.py).
"""

from __future__ import annotations
import sys as _hsys
from pathlib import Path as _HPath
_hsys.path.insert(0, str(_HPath(__file__).resolve().parents[3]))
from research.datasets.dataset_config import SPLITS_DIR as _ACTIVE_SPLITS_DIR, CKPT_DIR as _ACTIVE_CKPT_DIR  # noqa: E402

import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.h_jepa import HJEPAModel
from research.extraction.extracted_model.weight_preserving.model_weight_preserving import (
    WeightPreservingExtractedModel,
)


def evaluate_retention(model, orig_model, val_imgs, val_acts, device="cpu"):
    model.eval()
    orig_model.eval()
    with torch.no_grad():
        s_orig = orig_model.encode(val_imgs)
        S_macro_orig, _ = orig_model.macro_predictor(s_orig, val_acts)
        act_orig = orig_model.micro_predictor.select_action(s_orig, S_macro_orig)

        if hasattr(model, "micro_policy"):
            s_curr = model.encode(val_imgs)
            S_macro, _ = model.forward_macro(s_curr, val_acts)
            act_curr = model.select_action(s_curr, S_macro)
        else:
            s_curr = model.encode(val_imgs)
            S_macro, _ = model.macro_predictor(s_curr, val_acts)
            act_curr = model.micro_predictor.select_action(s_curr, S_macro)

        cos_sim = float(F.cosine_similarity(act_curr, act_orig, dim=-1).mean().item())
        mse_diff = float(F.mse_loss(act_curr, act_orig).item())
        retention = float(np.clip(1.0 - mse_diff, 0.0, 1.0))
    return {"retention": retention, "cosine_similarity": cos_sim, "action_mse": mse_diff}


def count_nonzero_params(model: nn.Module) -> tuple[int, int]:
    structural = sum(p.numel() for p in model.parameters())
    nonzero = sum(int((p.detach().abs() > 1e-8).sum().item()) for p in model.parameters())
    return structural, nonzero


def run_pruning_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v2_ckpt = str(_ACTIVE_CKPT_DIR)
    legacy_ckpt = str(_ACTIVE_CKPT_DIR)
    ckpt_dir = v2_ckpt if os.path.exists(os.path.join(v2_ckpt, "hjepa_best.pt")) else legacy_ckpt

    v2_splits = str(_ACTIVE_SPLITS_DIR)
    legacy_splits = str(_ACTIVE_SPLITS_DIR)
    splits_dir = v2_splits if os.path.exists(os.path.join(v2_splits, "val_benchmark.npz")) else legacy_splits

    out_dir = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    orig_model = HJEPAModel(stride_k=5).to(device)
    orig_model.load_state_dict(
        torch.load(os.path.join(ckpt_dir, "hjepa_best.pt"), map_location=device, weights_only=False),
        strict=False,
    )
    orig_model.eval()

    val_data = np.load(os.path.join(splits_dir, "val_benchmark.npz"))
    val_imgs = torch.from_numpy(val_data["rgb"][:512]).permute(0, 3, 1, 2).float().to(device) / 255.0
    val_acts = torch.from_numpy(val_data["actions"][:512]).float().to(device)

    orig_struct, orig_nz = count_nonzero_params(orig_model)
    stages = []

    stages.append(
        {
            "stage": 0,
            "name": "Original_Full_HJEPAModel",
            "structural_parameters": orig_struct,
            "active_nonzero_parameters": orig_nz,
            "structural_reduction_pct": 0.0,
            "active_nonzero_reduction_pct": 0.0,
            "retention": 1.0,
            "cosine_similarity": 1.0,
            "action_mse": 0.0,
        }
    )

    # Stage 1: Remove Micro Dynamics Prediction Branch (WeightPreservingExtractedModel)
    wp_model = WeightPreservingExtractedModel(stride_k=5).to(device)
    wp_model.load_from_full_hjepa(os.path.join(ckpt_dir, "hjepa_best.pt"), device=device)
    s1_struct, s1_nz = count_nonzero_params(wp_model)
    m1 = evaluate_retention(wp_model, orig_model, val_imgs, val_acts, device)
    stages.append(
        {
            "stage": 1,
            "name": "Stage1_Dead_Branch_Eliminated",
            "structural_parameters": s1_struct,
            "active_nonzero_parameters": s1_nz,
            "structural_reduction_pct": round((1.0 - s1_struct / orig_struct) * 100, 2),
            "active_nonzero_reduction_pct": round((1.0 - s1_nz / orig_nz) * 100, 2),
            **m1,
        }
    )

    # Stage 2: Low-Saliency Conv4 Filter Masking (25% lowest L1-norm filters zeroed)
    wp_s2 = copy.deepcopy(wp_model)
    with torch.no_grad():
        conv4 = wp_s2.encoder[6]
        l1_norms = conv4.weight.abs().mean(dim=(1, 2, 3))
        k_prune = int(0.25 * conv4.out_channels)
        prune_idx = torch.argsort(l1_norms)[:k_prune]
        conv4.weight[prune_idx] = 0.0
        conv4.bias[prune_idx] = 0.0
    s2_struct, s2_nz = count_nonzero_params(wp_s2)
    m2 = evaluate_retention(wp_s2, orig_model, val_imgs, val_acts, device)
    stages.append(
        {
            "stage": 2,
            "name": "Stage2_Conv4_Channel_Masking_25pct",
            "structural_parameters": s2_struct,
            "active_nonzero_parameters": s2_nz,
            "structural_reduction_pct": round((1.0 - s2_struct / orig_struct) * 100, 2),
            "active_nonzero_reduction_pct": round((1.0 - s2_nz / orig_nz) * 100, 2),
            **m2,
        }
    )

    # Stage 3: Non-Selective Latent Bottleneck Masking (8 lowest-variance latent dims zeroed)
    wp_s3 = copy.deepcopy(wp_s2)
    with torch.no_grad():
        fc_enc = wp_s3.encoder[9]
        row_norms = fc_enc.weight.abs().mean(dim=1)
        low_lat_idx = torch.argsort(row_norms)[:8]
        fc_enc.weight[low_lat_idx] = 0.0
        fc_enc.bias[low_lat_idx] = 0.0
    s3_struct, s3_nz = count_nonzero_params(wp_s3)
    m3 = evaluate_retention(wp_s3, orig_model, val_imgs, val_acts, device)
    stages.append(
        {
            "stage": 3,
            "name": "Stage3_Latent_Subspace_Masking_24dim",
            "structural_parameters": s3_struct,
            "active_nonzero_parameters": s3_nz,
            "structural_reduction_pct": round((1.0 - s3_struct / orig_struct) * 100, 2),
            "active_nonzero_reduction_pct": round((1.0 - s3_nz / orig_nz) * 100, 2),
            **m3,
        }
    )

    # Stage 4: Magnitude Threshold Sparsification (|w| < 0.015 zeroed)
    wp_s4 = copy.deepcopy(wp_s3)
    with torch.no_grad():
        for p in wp_s4.parameters():
            if p.dim() >= 2:
                p.mul_((p.abs() >= 0.015).float())
    s4_struct, s4_nz = count_nonzero_params(wp_s4)
    m4 = evaluate_retention(wp_s4, orig_model, val_imgs, val_acts, device)
    stages.append(
        {
            "stage": 4,
            "name": "Stage4_Magnitude_Sparsified_Model",
            "structural_parameters": s4_struct,
            "active_nonzero_parameters": s4_nz,
            "structural_reduction_pct": round((1.0 - s4_struct / orig_struct) * 100, 2),
            "active_nonzero_reduction_pct": round((1.0 - s4_nz / orig_nz) * 100, 2),
            **m4,
        }
    )

    save_wp_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../extracted_model/weight_preserving/weight_preserving_best.pt")
    )
    os.makedirs(os.path.dirname(save_wp_path), exist_ok=True)
    torch.save(wp_s4.state_dict(), save_wp_path)

    with open(os.path.join(out_dir, "pruning_stages.json"), "w") as f:
        json.dump({"stages": stages}, f, indent=2)

    print("Structured Pruning Results (Structural vs Active Non-Zero Parameters):")
    for st in stages:
        print(
            f"  Stage {st['stage']}: {st['name']:<38} | Struct: {st['structural_parameters']:7,d} | "
            f"NonZero: {st['active_nonzero_parameters']:7,d} (-{st['active_nonzero_reduction_pct']:4.1f}%) | "
            f"CosSim: {st['cosine_similarity']:.4f}"
        )
    return stages


if __name__ == "__main__":
    run_pruning_pipeline()
