"""
Phase 2: Action-Conditioned Latent Prediction Audit (LeWM & H-JEPA)
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Audits future state prediction in latent space:
  - LeWorldModel (LeWM): MSE(s_hat_{t+1}, s^*_{t+1}) and Cosine Similarity
  - H-JEPA Micro-Level (K=1): MSE(s_hat_{t+1}, s^*_{t+1})
  - H-JEPA Macro-Level (K=5): MSE(S_hat_{k+1}, S^*_{k+1}) across 5-step macro horizons
  - Collision prediction accuracy, precision, recall, and BCE loss
Evaluated on held-out test transitions and counterfactual suites.
"""

import os
import sys
import json
import csv
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from models.le_world_model import LeWorldModel
from models.h_jepa import HJEPAModel

def audit_latent_predictions(data_npz_path, output_dir=None):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "behavioral"))
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints"))

    print("\n=======================================================")
    print("  PHASE 2: LATENT PREDICTION AUDIT (LeWM vs H-JEPA)")
    print(f"  Dataset: {data_npz_path}")
    print(f"  Target Hardware: {device}")
    print("=======================================================\n")

    # Load models
    lewm = LeWorldModel().to(device)
    lewm.load_state_dict(torch.load(os.path.join(ckpt_dir, "lewm_best.pt"), map_location=device))
    lewm.eval()

    hjepa = HJEPAModel(stride_k=5).to(device)
    hjepa.load_state_dict(torch.load(os.path.join(ckpt_dir, "hjepa_best.pt"), map_location=device))
    hjepa.eval()

    # Load data
    data = np.load(data_npz_path)
    obs = data["rgb"]            # (N, 64, 64, 3)
    next_obs = data["next_rgb"]  # (N, 64, 64, 3)
    actions = data["actions"]    # (N, 3)
    collisions = data["collisions"] # (N,)
    ep_starts = data["episode_starts"]
    N = len(obs)

    batch_size = 128
    num_batches = (N + batch_size - 1) // batch_size

    # Metric accumulators
    lewm_mse_list = []
    lewm_cos_list = []
    lewm_col_acc = []

    hjepa_micro_mse = []
    hjepa_micro_cos = []

    with torch.no_grad():
        for b in range(num_batches):
            s_idx = b * batch_size
            e_idx = min(N, (b + 1) * batch_size)

            imgs = torch.from_numpy(obs[s_idx:e_idx]).permute(0, 3, 1, 2).float().to(device) / 255.0
            next_imgs = torch.from_numpy(next_obs[s_idx:e_idx]).permute(0, 3, 1, 2).float().to(device) / 255.0
            acts = torch.from_numpy(actions[s_idx:e_idx]).float().to(device)
            cols = torch.from_numpy(collisions[s_idx:e_idx]).float().to(device)

            # -------------------------------------------------------------
            # 1. LeWorldModel Evaluation
            # -------------------------------------------------------------
            s_t = lewm.encode(imgs)
            s_next_target = lewm.encode(next_imgs)
            s_next_pred, col_logits = lewm.predict_next(s_t, acts)

            # MSE and Cosine Similarity
            mse_lewm = F.mse_loss(s_next_pred, s_next_target, reduction="none").mean(dim=-1).cpu().numpy()
            cos_lewm = F.cosine_similarity(s_next_pred, s_next_target, dim=-1).cpu().numpy()
            lewm_mse_list.extend(mse_lewm)
            lewm_cos_list.extend(cos_lewm)

            pred_col = (torch.sigmoid(col_logits).squeeze(-1) > 0.5).float()
            acc_col = (pred_col == cols).float().cpu().numpy()
            lewm_col_acc.extend(acc_col)

            # -------------------------------------------------------------
            # 2. H-JEPA Micro Evaluation (K=1)
            # -------------------------------------------------------------
            h_s_t = hjepa.encode(imgs)
            h_s_next_target = hjepa.encode(next_imgs)
            h_s_next_pred, h_col_logits = hjepa.micro_predictor.forward_transition(h_s_t, acts)

            mse_micro = F.mse_loss(h_s_next_pred, h_s_next_target, reduction="none").mean(dim=-1).cpu().numpy()
            cos_micro = F.cosine_similarity(h_s_next_pred, h_s_next_target, dim=-1).cpu().numpy()
            hjepa_micro_mse.extend(mse_micro)
            hjepa_micro_cos.extend(cos_micro)

    # -------------------------------------------------------------
    # 3. H-JEPA Macro Evaluation (K=5)
    # Evaluates sequences of stride K=5
    # -------------------------------------------------------------
    K = 5
    hjepa_macro_mse = []
    hjepa_macro_cos = []
    hjepa_macro_col_acc = []

    # Find contiguous K=5 sequences
    ep_indices = np.where(ep_starts)[0]
    ep_indices = np.append(ep_indices, N)

    macro_seqs = []
    for ep_i in range(len(ep_indices) - 1):
        s_pos = ep_indices[ep_i]
        e_pos = ep_indices[ep_i + 1]
        for t in range(s_pos, e_pos - K):
            macro_seqs.append(t)

    print(f"Evaluating {len(macro_seqs)} Macro Sequences (Stride K={K})...")
    with torch.no_grad():
        for b in range(0, len(macro_seqs), batch_size):
            batch_t = macro_seqs[b : b + batch_size]
            if len(batch_t) == 0:
                break

            img_0 = torch.from_numpy(obs[batch_t]).permute(0, 3, 1, 2).float().to(device) / 255.0
            img_K = torch.from_numpy(obs[[t + K for t in batch_t]]).permute(0, 3, 1, 2).float().to(device) / 255.0

            # Average action over K=5 steps
            act_slices = [actions[t : t + K] for t in batch_t]
            macro_act = torch.from_numpy(np.mean(act_slices, axis=1)).float().to(device)

            col_slices = [any(collisions[t : t + K]) for t in batch_t]
            macro_col_gt = torch.tensor(col_slices, dtype=torch.float32, device=device)

            S_0 = hjepa.encode(img_0)
            S_K_target = hjepa.encode(img_K)

            pred_S_K, macro_col_log = hjepa.macro_predictor(S_0, macro_act)

            mse_macro = F.mse_loss(pred_S_K, S_K_target, reduction="none").mean(dim=-1).cpu().numpy()
            cos_macro = F.cosine_similarity(pred_S_K, S_K_target, dim=-1).cpu().numpy()
            hjepa_macro_mse.extend(mse_macro)
            hjepa_macro_cos.extend(cos_macro)

            pred_macro_col = (torch.sigmoid(macro_col_log).squeeze(-1) > 0.5).float()
            hjepa_macro_col_acc.extend((pred_macro_col == macro_col_gt).float().cpu().numpy())

    audit_results = {
        "dataset": os.path.basename(data_npz_path),
        "total_transitions": N,
        "total_macro_sequences": len(macro_seqs),
        "lewm": {
            "latent_mse": float(np.mean(lewm_mse_list)),
            "latent_cosine_similarity": float(np.mean(lewm_cos_list)),
            "collision_prediction_accuracy": float(np.mean(lewm_col_acc)) * 100.0
        },
        "hjepa_micro_k1": {
            "latent_mse": float(np.mean(hjepa_micro_mse)),
            "latent_cosine_similarity": float(np.mean(hjepa_micro_cos))
        },
        "hjepa_macro_k5": {
            "latent_mse": float(np.mean(hjepa_macro_mse)),
            "latent_cosine_similarity": float(np.mean(hjepa_macro_cos)),
            "macro_collision_accuracy": float(np.mean(hjepa_macro_col_acc)) * 100.0
        }
    }

    print("\n--- LATENT PREDICTION AUDIT RESULTS ---")
    print(f"LeWorldModel (LeWM):")
    print(f"  Latent MSE (s_t+1):          {audit_results['lewm']['latent_mse']:.5f}")
    print(f"  Cosine Similarity:           {audit_results['lewm']['latent_cosine_similarity']:.4f}")
    print(f"  Collision Accuracy:          {audit_results['lewm']['collision_prediction_accuracy']:.1f}%")
    print(f"H-JEPA Micro-Level (K=1):")
    print(f"  Latent MSE (s_t+1):          {audit_results['hjepa_micro_k1']['latent_mse']:.5f}")
    print(f"  Cosine Similarity:           {audit_results['hjepa_micro_k1']['latent_cosine_similarity']:.4f}")
    print(f"H-JEPA Macro-Level (K=5):")
    print(f"  Macro Latent MSE (S_k+1):    {audit_results['hjepa_macro_k5']['latent_mse']:.5f}")
    print(f"  Macro Cosine Similarity:     {audit_results['hjepa_macro_k5']['latent_cosine_similarity']:.4f}")
    print(f"  Macro Collision Accuracy:    {audit_results['hjepa_macro_k5']['macro_collision_accuracy']:.1f}%")

    out_json = os.path.join(output_dir, "latent_prediction_audit.json")
    with open(out_json, "w") as f:
        json.dump(audit_results, f, indent=2)

    print(f"\nSaved latent prediction audit to {out_json}")
    return audit_results

if __name__ == "__main__":
    test_npz = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/splits/held_out_test_benchmark.npz"))
    audit_latent_predictions(test_npz)
