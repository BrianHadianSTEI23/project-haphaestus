"""
Phase 5: Computational Circuit Reconstruction - Activation Patching
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Primary causal experiment:
- Clean Scenario: Robot navigates to target behind yellow cylinder.
- Corrupted Scenario: Mutated scene (E1 Position Shift, E5 Obstacle Insertion).
Selectively patches activations at layer l:
  h_l^patched = h_l^clean
Measures Causal Recovery:
  Recovery = (S_patched - S_corrupted) / (S_clean - S_corrupted)
Also measures Reverse Degradation (injecting corrupted patch into clean run).
Evaluates causal bottleneck across early conv, late conv, latent clusters, and predictor heads.
"""

import os
import sys
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from models.baseline_policy import BaselineNavigationPolicy
from models.le_world_model import LeWorldModel
from models.h_jepa import HJEPAModel
from research.instrumentation.model_inspector import ModelInspector

def action_similarity_score(a_test: torch.Tensor, a_clean: torch.Tensor) -> float:
    """
    Computes fidelity score S relative to clean ground-truth action:
    S = cos_sim(a_test, a_clean) - 0.5 * ||a_test - a_clean||_2
    """
    cos_sim = F.cosine_similarity(a_test, a_clean, dim=-1).mean().item()
    l2_dist = torch.norm(a_test - a_clean, dim=-1).mean().item()
    return float(cos_sim - 0.5 * l2_dist)

def run_patching_for_model(model_type, model_cls, ckpt_name, cf_npz_path, output_dir, device):
    print(f"\n=======================================================")
    print(f"  RUNNING ACTIVATION PATCHING: [{model_type.upper()}]")
    print(f"=======================================================")

    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../checkpoints/{ckpt_name}"))
    model = model_cls().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    inspector = ModelInspector(model, model_type)

    # Load matched counterfactual pairs
    cf_data = np.load(cf_npz_path)
    obs = cf_data["rgb"]
    ep_starts = np.where(cf_data["episode_starts"])[0]

    # Use first 20 pairs of E1 (Position shift) as Corruption A
    # and first 20 pairs of E5 (Obstacle insertion) as Corruption B
    corruptions = [
        {"name": "E1_position_shift", "start_pair": 0, "num_pairs": 20},
        {"name": "E5_obstacle_insertion", "start_pair": 100, "num_pairs": 20}
    ]

    # Define candidate patching layers per model
    if model_type == "baseline":
        layers_to_patch = [
            "baseline.conv1", "baseline.conv2", "baseline.conv3", "baseline.conv4",
            "baseline.visual_latent_z", "baseline.policy_fc1", "baseline.action_head"
        ]
        # Semantic sub-clusters in latent_z (dim 32)
        cluster_indices = {
            "cluster_appearance": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 19, 21, 22, 27, 28, 29, 30],
            "cluster_heading": [10, 16, 25],
            "cluster_position": [13, 20, 23, 26, 31],
            "cluster_geometry": [24]
        }
    elif model_type == "lewm":
        layers_to_patch = [
            "lewm.conv1", "lewm.conv2", "lewm.conv3", "lewm.conv4",
            "lewm.latent_state_s", "lewm.pred_fc1", "lewm.delta_s"
        ]
        cluster_indices = {
            "cluster_appearance": [0, 2, 4, 13, 17, 19, 22, 24, 27],
            "cluster_heading": [1, 3, 7, 8, 10, 11, 15, 20, 21, 29],
            "cluster_geometry": [5, 6, 9, 14, 23, 25, 26, 28, 30, 31],
            "cluster_position": [12, 16, 18]
        }
    elif model_type == "hjepa":
        layers_to_patch = [
            "hjepa.conv1", "hjepa.shared_latent_s", "hjepa.macro_fc1",
            "hjepa.macro_delta_S", "hjepa.micro_pol_fc1", "hjepa.micro_action_head"
        ]
        cluster_indices = {
            "cluster_appearance": [8],
            "cluster_heading": [0, 9, 16, 19, 20, 22, 23, 24],
            "cluster_geometry": [1, 2, 3, 5, 12, 13, 14, 18, 25, 26, 28, 31],
            "cluster_position": [4, 6, 7, 10, 11, 15, 17, 21, 27, 29, 30]
        }

    results = {}

    for corr in corruptions:
        corr_name = corr["name"]
        results[corr_name] = {}
        print(f"\n--- Testing Corruption: {corr_name} ---")

        # Collect Clean and Corrupted frames (t=0 of each pair)
        clean_frames = []
        corrupt_frames = []

        for p in range(corr["num_pairs"]):
            e0_ep = corr["start_pair"] + p * 2
            ei_ep = e0_ep + 1
            clean_frames.append(obs[ep_starts[e0_ep]])
            corrupt_frames.append(obs[ep_starts[ei_ep]])

        clean_t = torch.from_numpy(np.array(clean_frames)).permute(0, 3, 1, 2).float().to(device) / 255.0
        corrupt_t = torch.from_numpy(np.array(corrupt_frames)).permute(0, 3, 1, 2).float().to(device) / 255.0

        # Baseline goals
        goal_vecs = torch.tensor([[1.0, 0.0, 0.0, 1.0]], device=device).repeat(len(clean_t), 1)

        # 1. Clean forward pass
        if model_type == "baseline":
            clean_out, _ = model(clean_t, goal_vecs)
            a_clean = clean_out[0]
            clean_acts = inspector.capture_activations(clean_t, goal_vecs)
        elif model_type == "lewm":
            s_clean = model.encode(clean_t)
            cand_a = torch.tensor([[0.8, 0.0, 0.0]], device=device).repeat(len(clean_t), 1)
            pred_s, col_l = model.predict_next(s_clean, cand_a)
            col_p = torch.sigmoid(col_l)
            a_clean = cand_a * (1.0 - col_p) + torch.tensor([[0.1, 0.6, 0.4]], device=device) * col_p
            clean_acts = inspector.capture_activations(clean_t)
        elif model_type == "hjepa":
            s_clean = model.encode(clean_t)
            cand_a = torch.tensor([[0.8, 0.0, 0.0]], device=device).repeat(len(clean_t), 1)
            S_macro, col_l = model.macro_predictor(s_clean, cand_a)
            a_clean = model.micro_predictor.select_action(s_clean, S_macro)
            clean_acts = inspector.capture_activations(clean_t)

        # 2. Corrupted forward pass
        if model_type == "baseline":
            corrupt_out, _ = model(corrupt_t, goal_vecs)
            a_corrupt = corrupt_out[0]
            corrupt_acts = inspector.capture_activations(corrupt_t, goal_vecs)
        elif model_type == "lewm":
            s_corrupt = model.encode(corrupt_t)
            pred_s_corrupt, col_l_c = model.predict_next(s_corrupt, cand_a)
            col_p_c = torch.sigmoid(col_l_c)
            a_corrupt = cand_a * (1.0 - col_p_c) + torch.tensor([[0.1, 0.6, 0.4]], device=device) * col_p_c
            corrupt_acts = inspector.capture_activations(corrupt_t)
        elif model_type == "hjepa":
            s_corrupt = model.encode(corrupt_t)
            S_macro_corrupt, col_l_c = model.macro_predictor(s_corrupt, cand_a)
            a_corrupt = model.micro_predictor.select_action(s_corrupt, S_macro_corrupt)
            corrupt_acts = inspector.capture_activations(corrupt_t)

        S_clean = action_similarity_score(a_clean, a_clean)
        S_corrupt = action_similarity_score(a_corrupt, a_clean)
        delta_base = max(1e-5, S_clean - S_corrupt)

        # 3. Patch each candidate layer
        for l_name in layers_to_patch:
            if l_name not in clean_acts:
                continue

            h_clean = clean_acts[l_name].to(device)
            # Patch corrupted run with clean activation
            with inspector.patch_activation(l_name, h_clean):
                if model_type == "baseline":
                    patched_out, _ = model(corrupt_t, goal_vecs)
                    a_patched = patched_out[0]
                elif model_type == "lewm":
                    s_patched = model.encode(corrupt_t)
                    pred_s_p, col_l_p = model.predict_next(s_patched, cand_a)
                    col_p_p = torch.sigmoid(col_l_p)
                    a_patched = cand_a * (1.0 - col_p_p) + torch.tensor([[0.1, 0.6, 0.4]], device=device) * col_p_p
                elif model_type == "hjepa":
                    s_patched = model.encode(corrupt_t)
                    S_macro_p, col_l_p = model.macro_predictor(s_patched, cand_a)
                    a_patched = model.micro_predictor.select_action(s_patched, S_macro_p)

            S_patched = action_similarity_score(a_patched, a_clean)
            recovery = float(np.clip((S_patched - S_corrupt) / delta_base, 0.0, 1.0))

            # Reverse Patching: patch clean run with corrupted activation
            h_corrupt = corrupt_acts[l_name].to(device)
            with inspector.patch_activation(l_name, h_corrupt):
                if model_type == "baseline":
                    rev_out, _ = model(clean_t, goal_vecs)
                    a_rev = rev_out[0]
                elif model_type == "lewm":
                    s_rev = model.encode(clean_t)
                    pred_s_r, col_l_r = model.predict_next(s_rev, cand_a)
                    col_p_r = torch.sigmoid(col_l_r)
                    a_rev = cand_a * (1.0 - col_p_r) + torch.tensor([[0.1, 0.6, 0.4]], device=device) * col_p_r
                elif model_type == "hjepa":
                    s_rev = model.encode(clean_t)
                    S_macro_r, col_l_r = model.macro_predictor(s_rev, cand_a)
                    a_rev = model.micro_predictor.select_action(s_rev, S_macro_r)

            S_rev = action_similarity_score(a_rev, a_clean)
            corruption_transfer = float(np.clip((S_clean - S_rev) / delta_base, 0.0, 1.0))

            results[corr_name][l_name] = {
                "forward_recovery": recovery,
                "reverse_corruption_transfer": corruption_transfer,
                "is_causal_bottleneck": bool(recovery >= 0.70 and corruption_transfer >= 0.60)
            }

            print(f"  Layer [{l_name:24s}] Recovery: {recovery*100:5.1f}% | Transfer: {corruption_transfer*100:5.1f}% | Causal: {results[corr_name][l_name]['is_causal_bottleneck']}")

        # 4. Patch Discovered Sub-Clusters in Latent Representation
        latent_key = [k for k in layers_to_patch if "latent" in k or "shared" in k][0]
        if latent_key in clean_acts:
            h_clean_full = clean_acts[latent_key].to(device)
            h_corrupt_full = corrupt_acts[latent_key].to(device)

            for c_name, dims in cluster_indices.items():
                # Partially patch only the specified channel dimensions
                h_patched = h_corrupt_full.clone()
                h_patched[:, dims] = h_clean_full[:, dims]

                with inspector.patch_activation(latent_key, h_patched):
                    if model_type == "baseline":
                        patched_out, _ = model(corrupt_t, goal_vecs)
                        a_patched = patched_out[0]
                    elif model_type == "lewm":
                        s_patched = model.encode(corrupt_t)
                        pred_s_p, col_l_p = model.predict_next(s_patched, cand_a)
                        col_p_p = torch.sigmoid(col_l_p)
                        a_patched = cand_a * (1.0 - col_p_p) + torch.tensor([[0.1, 0.6, 0.4]], device=device) * col_p_p
                    elif model_type == "hjepa":
                        s_patched = model.encode(corrupt_t)
                        S_macro_p, col_l_p = model.macro_predictor(s_patched, cand_a)
                        a_patched = model.micro_predictor.select_action(s_patched, S_macro_p)

                S_patched = action_similarity_score(a_patched, a_clean)
                recovery = float(np.clip((S_patched - S_corrupt) / delta_base, 0.0, 1.0))

                results[corr_name][f"{latent_key}.{c_name}"] = {
                    "forward_recovery": recovery,
                    "num_dims": len(dims),
                    "is_causal_bottleneck": bool(recovery >= 0.65)
                }
                print(f"  Cluster [{c_name:22s} ({len(dims):2d} dims)] Recovery: {recovery*100:5.1f}% | Causal: {results[corr_name][f'{latent_key}.{c_name}']['is_causal_bottleneck']}")

    inspector.close()
    return results

def run_all_patching():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "activation_patching"))
    os.makedirs(out_dir, exist_ok=True)
    cf_npz = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/splits/counterfactual_suite.npz"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_patch_data = {}
    csv_rows = []

    configs = [
        ("baseline", BaselineNavigationPolicy, "baseline_best.pt"),
        ("lewm", LeWorldModel, "lewm_best.pt"),
        ("hjepa", lambda: HJEPAModel(stride_k=5), "hjepa_best.pt")
    ]

    for m_type, m_cls, ckpt in configs:
        res = run_patching_for_model(m_type, m_cls, ckpt, cf_npz, out_dir, device)
        all_patch_data[m_type] = res
        for corr_name, layers in res.items():
            for l_name, m_dict in layers.items():
                csv_rows.append([
                    m_type, corr_name, l_name,
                    f"{m_dict['forward_recovery']*100:.1f}%",
                    f"{m_dict.get('reverse_corruption_transfer', 0.0)*100:.1f}%",
                    "YES" if m_dict.get("is_causal_bottleneck", False) else "NO"
                ])

    json_path = os.path.join(out_dir, "patching_results.json")
    with open(json_path, "w") as f:
        json.dump(all_patch_data, f, indent=2)

    csv_path = os.path.join(out_dir, "patching_recovery.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Corruption", "Layer / Cluster", "Forward Recovery %", "Reverse Transfer %", "Causal Bottleneck"])
        writer.writerows(csv_rows)

    print(f"\nSaved Activation Patching Results to:")
    print(f"  -> {json_path}")
    print(f"  -> {csv_path}")

if __name__ == "__main__":
    run_all_patching()
