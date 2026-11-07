"""
Phase 4: Activation Statistics & Quantization Profiling (FPGA Q8.8 Readiness)
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Profiles distribution, dynamic range, sparsity, and fixed-point headroom across:
  - Baseline Policy (conv1..conv4, visual_latent, mlp1, policy_actions)
  - LeWorldModel (conv1..conv4, s_t, trans_fc, policy_fc)
  - H-JEPA (conv1..conv4, s_t, micro_pred, macro_pred)

Computes:
  - Mean, Std Dev, Min, Max
  - Q8.8 Dynamic Range Compatibility ([-128.0, +127.996])
  - Saturation rate (% out of range)
  - Zero Sparsity (% exact zero after ReLU)
  - Effective Bit Width requirement: ceil(log2(max(|min|, |max|) + 1)) + frac_bits
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
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from models.baseline_policy import BaselineNavigationPolicy
from models.le_world_model import LeWorldModel
from models.h_jepa import HJEPAModel
from research.instrumentation.model_inspector import ModelInspector

def compute_tensor_stats(arr: np.ndarray, name: str):
    """
    Computes rigorous statistical metrics on activation array.
    """
    flat = arr.flatten()
    n = len(flat)
    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat))
    min_val = float(np.min(flat))
    max_val = float(np.max(flat))

    # Q8.8 range: [-128.0, 127.996]
    sat_under = np.sum(flat < -128.0)
    sat_over = np.sum(flat > 127.996)
    sat_rate = float((sat_under + sat_over) / n)

    # Zero sparsity (|x| < 1e-5)
    zero_sparsity = float(np.sum(np.abs(flat) < 1e-5) / n)

    # Peak absolute value
    peak_abs = max(abs(min_val), abs(max_val))
    # Required integer bits (including sign)
    int_bits_req = int(np.ceil(np.log2(peak_abs + 1e-6))) + 1 if peak_abs > 0 else 1

    return {
        "layer_name": name,
        "shape": list(arr.shape),
        "total_elements": n,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "peak_abs": float(peak_abs),
        "int_bits_req": int_bits_req,
        "q8_8_overflow_rate": sat_rate,
        "zero_sparsity": zero_sparsity,
        "q8_8_safe": bool(sat_rate == 0.0 and peak_abs <= 127.0)
    }

def run_activation_profiling(dataset_npz_path, output_dir=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "activation_statistics"))
    os.makedirs(output_dir, exist_ok=True)

    print("\n=======================================================")
    print("  PHASE 4: ACTIVATION STATISTICS & Q8.8 PROFILING")
    print(f"  Dataset: {dataset_npz_path}")
    print(f"  Device: {device}")
    print("=======================================================\n")

    data = np.load(dataset_npz_path)
    obs = data["rgb"][:1000] # Profile 1000 validation frames
    N = len(obs)

    ckpt_dir = str(_ACTIVE_CKPT_DIR)

    models_config = [
        ("baseline", BaselineNavigationPolicy, "baseline_best.pt"),
        ("lewm", LeWorldModel, "lewm_best.pt"),
        ("hjepa", lambda: HJEPAModel(stride_k=5), "hjepa_best.pt")
    ]

    all_stats = {}
    csv_rows = []

    for m_type, m_cls, ckpt_name in models_config:
        print(f"\n--- Profiling Activations: [{m_type.upper()}] ---")
        model = m_cls().to(device)
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, ckpt_name), map_location=device))
        model.eval()

        inspector = ModelInspector(model, m_type)

        batch_size = 100
        collected_acts = {}

        for b in range(0, N, batch_size):
            imgs = torch.from_numpy(obs[b : b + batch_size]).permute(0, 3, 1, 2).float().to(device) / 255.0

            if m_type == "baseline":
                goal_vecs = torch.tensor([[1.0, 0.0, 0.0, 1.0]], device=device).repeat(len(imgs), 1)
                acts = inspector.capture_activations(imgs, goal_vecs)
            else:
                acts = inspector.capture_activations(imgs)

            for k, tensor in acts.items():
                if k == "model_output":
                    continue
                if isinstance(tensor, torch.Tensor):
                    if k not in collected_acts:
                        collected_acts[k] = []
                    collected_acts[k].append(tensor.cpu().numpy())

        # Compute stats per layer
        all_stats[m_type] = {}
        for k, list_arr in collected_acts.items():
            full_arr = np.concatenate(list_arr, axis=0)
            stats = compute_tensor_stats(full_arr, k)
            all_stats[m_type][k] = stats

            csv_rows.append([
                m_type, k, str(stats["shape"][1:]),
                f"{stats['mean']:.4f}", f"{stats['std']:.4f}",
                f"{stats['min']:.4f}", f"{stats['max']:.4f}",
                f"{stats['peak_abs']:.4f}", stats["int_bits_req"],
                f"{stats['zero_sparsity']*100:.1f}%",
                f"{stats['q8_8_overflow_rate']*100:.4f}%",
                "PASS" if stats["q8_8_safe"] else "FAIL"
            ])

            print(f"  [{k:22s}] Range: [{stats['min']:7.2f}, {stats['max']:7.2f}] | "
                  f"Peak: {stats['peak_abs']:6.2f} | "
                  f"IntBits: {stats['int_bits_req']:2d} | "
                  f"ZeroSparsity: {stats['zero_sparsity']*100:4.1f}% | "
                  f"Q8.8 Safe: {'YES' if stats['q8_8_safe'] else 'OVERFLOW'}")

        inspector.close()

    # Save summary JSON and CSV
    json_path = os.path.join(output_dir, "activation_stats_summary.json")
    with open(json_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    csv_path = os.path.join(output_dir, "activation_stats_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Layer", "Shape", "Mean", "Std", "Min", "Max",
            "Peak Abs", "Int Bits Req", "Zero Sparsity", "Q8.8 Overflow Rate", "Q8.8 Safe"
        ])
        writer.writerows(csv_rows)

    print(f"\nSaved Activation Statistics to:")
    print(f"  -> {json_path}")
    print(f"  -> {csv_path}")

    return all_stats

if __name__ == "__main__":
    val_npz = str(_ACTIVE_SPLITS_DIR / "val_benchmark.npz")
    run_activation_profiling(val_npz)
