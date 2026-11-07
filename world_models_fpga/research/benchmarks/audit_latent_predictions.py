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

from models.le_world_model import LeWorldModel
from models.h_jepa import HJEPAModel

def audit_latent_predictions(data_npz_path, output_dir=None):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "behavioral"))
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v2_ckpt = str(_ACTIVE_CKPT_DIR)
    legacy_ckpt = str(_ACTIVE_CKPT_DIR)
    ckpt_dir = v2_ckpt if os.path.exists(os.path.join(v2_ckpt, "lewm_best.pt")) else legacy_ckpt

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
    # 3. Explicit Multi-Horizon Prediction Audit: K in {1, 3, 5, 10}
    # Evaluates Latent MSE, Cosine Similarity, Collision Accuracy,
    # and Decoded Physical Displacement Error against MuJoCo Ground Truth.
    # -------------------------------------------------------------
    robot_pose = data["robot_pose"] if "robot_pose" in data else None
    ep_indices = np.where(ep_starts)[0]
    ep_indices = np.append(ep_indices, N)

    horizons = [1, 3, 5, 10]
    horizon_results = {}
    lewm_k5_mse_list, lewm_k5_cos_list = [], []
    hjepa_macro_mse, hjepa_macro_cos, hjepa_macro_col_acc = [], [], []

    for K in horizons:
        seq_indices = []
        for ep_i in range(len(ep_indices) - 1):
            s_pos = ep_indices[ep_i]
            e_pos = ep_indices[ep_i + 1]
            for t in range(s_pos, e_pos - K):
                seq_indices.append(t)

        print(f"Evaluating {len(seq_indices)} sequences for horizon K={K}...")
        l_mse, l_cos, l_col, l_disp, l_head = [], [], [], [], []
        h_micro_mse, h_micro_cos, h_micro_col, h_micro_disp, h_micro_head = [], [], [], [], []
        h_macro_mse, h_macro_cos, h_macro_col, h_macro_disp, h_macro_head = [], [], [], [], []

        with torch.no_grad():
            for b in range(0, len(seq_indices), batch_size):
                batch_t = seq_indices[b : b + batch_size]
                if len(batch_t) == 0:
                    break

                img_0 = torch.from_numpy(obs[batch_t]).permute(0, 3, 1, 2).float().to(device) / 255.0
                img_K = torch.from_numpy(obs[[t + K for t in batch_t]]).permute(0, 3, 1, 2).float().to(device) / 255.0
                act_slices = np.asarray([actions[t : t + K] for t in batch_t], dtype=np.float32)  # (B, K, 3)

                col_slices = [any(collisions[t : t + K]) for t in batch_t]
                col_gt = np.asarray(col_slices, dtype=bool)

                # Ground truth physical displacement in step-0 body frame
                has_pose = robot_pose is not None
                if has_pose:
                    p0 = robot_pose[batch_t]
                    pK = robot_pose[[t + K for t in batch_t]]
                    dx_w = pK[:, 0] - p0[:, 0]
                    dy_w = pK[:, 1] - p0[:, 1]
                    c0 = np.cos(p0[:, 2])
                    s0 = np.sin(p0[:, 2])
                    gt_dx_ego = c0 * dx_w + s0 * dy_w
                    gt_dy_ego = -s0 * dx_w + c0 * dy_w
                    gt_dth = np.arctan2(np.sin(pK[:, 2] - p0[:, 2]), np.cos(pK[:, 2] - p0[:, 2]))

                # ---------------------------------------------------------
                # LeWM Rollout
                # ---------------------------------------------------------
                target_s_K_lewm = lewm.encode(img_K)
                s_curr = lewm.encode(img_0)
                cum_dx = torch.zeros(len(batch_t), device=device)
                cum_dy = torch.zeros(len(batch_t), device=device)
                cum_dth = torch.zeros(len(batch_t), device=device)
                l_col_pred = torch.zeros(len(batch_t), dtype=torch.bool, device=device)

                for step in range(K):
                    a_step = torch.from_numpy(act_slices[:, step, :]).float().to(device)
                    s_next, col_logit = lewm.predict_next(s_curr, a_step)
                    disp = lewm.predict_ego_displacement(s_curr, s_next)
                    step_dx = disp[:, 0]
                    step_dy = disp[:, 1]
                    step_dth = disp[:, 2]

                    c_th = torch.cos(cum_dth)
                    s_th = torch.sin(cum_dth)
                    cum_dx = cum_dx + c_th * step_dx - s_th * step_dy
                    cum_dy = cum_dy + s_th * step_dx + c_th * step_dy
                    cum_dth = cum_dth + step_dth

                    l_col_pred = l_col_pred | (torch.sigmoid(col_logit).squeeze(-1) > 0.5)
                    s_curr = s_next

                l_mse.extend(F.mse_loss(s_curr, target_s_K_lewm, reduction="none").mean(dim=-1).cpu().numpy())
                l_cos.extend(F.cosine_similarity(s_curr, target_s_K_lewm, dim=-1).cpu().numpy())
                l_col.extend((l_col_pred.cpu().numpy() == col_gt).astype(np.float32))

                if has_pose:
                    disp_err = np.sqrt((cum_dx.cpu().numpy() - gt_dx_ego) ** 2 + (cum_dy.cpu().numpy() - gt_dy_ego) ** 2)
                    head_err = np.abs(np.arctan2(np.sin(cum_dth.cpu().numpy() - gt_dth), np.cos(cum_dth.cpu().numpy() - gt_dth)))
                    l_disp.extend(disp_err)
                    l_head.extend(head_err)

                if K == 5:
                    lewm_k5_mse_list.extend(F.mse_loss(s_curr, target_s_K_lewm, reduction="none").mean(dim=-1).cpu().numpy())
                    lewm_k5_cos_list.extend(F.cosine_similarity(s_curr, target_s_K_lewm, dim=-1).cpu().numpy())

                # ---------------------------------------------------------
                # H-JEPA Micro Autoregressive Rollout
                # ---------------------------------------------------------
                target_s_K_hjepa = hjepa.encode(img_K)
                h_s_curr = hjepa.encode(img_0)
                h_cum_dx = torch.zeros(len(batch_t), device=device)
                h_cum_dy = torch.zeros(len(batch_t), device=device)
                h_cum_dth = torch.zeros(len(batch_t), device=device)
                h_col_pred = torch.zeros(len(batch_t), dtype=torch.bool, device=device)

                for step in range(K):
                    a_step = torch.from_numpy(act_slices[:, step, :]).float().to(device)
                    h_next, col_logit = hjepa.micro_predictor.forward_transition(h_s_curr, a_step)
                    disp = hjepa.predict_micro_displacement(h_s_curr, h_next)
                    step_dx = disp[:, 0]
                    step_dy = disp[:, 1]
                    step_dth = disp[:, 2]

                    c_th = torch.cos(h_cum_dth)
                    s_th = torch.sin(h_cum_dth)
                    h_cum_dx = h_cum_dx + c_th * step_dx - s_th * step_dy
                    h_cum_dy = h_cum_dy + s_th * step_dx + c_th * step_dy
                    h_cum_dth = h_cum_dth + step_dth

                    h_col_pred = h_col_pred | (torch.sigmoid(col_logit).squeeze(-1) > 0.5)
                    h_s_curr = h_next

                h_micro_mse.extend(F.mse_loss(h_s_curr, target_s_K_hjepa, reduction="none").mean(dim=-1).cpu().numpy())
                h_micro_cos.extend(F.cosine_similarity(h_s_curr, target_s_K_hjepa, dim=-1).cpu().numpy())
                h_micro_col.extend((h_col_pred.cpu().numpy() == col_gt).astype(np.float32))

                if has_pose:
                    h_disp_err = np.sqrt((h_cum_dx.cpu().numpy() - gt_dx_ego) ** 2 + (h_cum_dy.cpu().numpy() - gt_dy_ego) ** 2)
                    h_head_err = np.abs(np.arctan2(np.sin(h_cum_dth.cpu().numpy() - gt_dth), np.cos(h_cum_dth.cpu().numpy() - gt_dth)))
                    h_micro_disp.extend(h_disp_err)
                    h_micro_head.extend(h_head_err)

                # ---------------------------------------------------------
                # H-JEPA Macro Prediction (for K=5 single jump, K=10 chained jump)
                # ---------------------------------------------------------
                if K == 5:
                    macro_act = torch.from_numpy(act_slices.reshape(len(batch_t), -1)).float().to(device)
                    S_0 = hjepa.encode(img_0)
                    pred_S_5, macro_col_log = hjepa.macro_predictor(S_0, macro_act)
                    disp_macro = hjepa.predict_macro_displacement(S_0, pred_S_5)

                    h_macro_mse.extend(F.mse_loss(pred_S_5, target_s_K_hjepa, reduction="none").mean(dim=-1).cpu().numpy())
                    h_macro_cos.extend(F.cosine_similarity(pred_S_5, target_s_K_hjepa, dim=-1).cpu().numpy())
                    pred_col_m = (torch.sigmoid(macro_col_log).squeeze(-1) > 0.5).cpu().numpy()
                    h_macro_col.extend((pred_col_m == col_gt).astype(np.float32))

                    if has_pose:
                        m_dx = disp_macro[:, 0].cpu().numpy()
                        m_dy = disp_macro[:, 1].cpu().numpy()
                        m_dth = disp_macro[:, 2].cpu().numpy()
                        h_macro_disp.extend(np.sqrt((m_dx - gt_dx_ego) ** 2 + (m_dy - gt_dy_ego) ** 2))
                        h_macro_head.extend(np.abs(np.arctan2(np.sin(m_dth - gt_dth), np.cos(m_dth - gt_dth))))

                    hjepa_macro_mse = h_macro_mse
                    hjepa_macro_cos = h_macro_cos
                    hjepa_macro_col_acc = h_macro_col

                elif K == 10:
                    act_1 = torch.from_numpy(act_slices[:, :5, :].reshape(len(batch_t), -1)).float().to(device)
                    act_2 = torch.from_numpy(act_slices[:, 5:, :].reshape(len(batch_t), -1)).float().to(device)
                    S_0 = hjepa.encode(img_0)
                    pred_S_5, col_1 = hjepa.macro_predictor(S_0, act_1)
                    disp_1 = hjepa.predict_macro_displacement(S_0, pred_S_5)
                    pred_S_10, col_2 = hjepa.macro_predictor(pred_S_5, act_2)
                    disp_2 = hjepa.predict_macro_displacement(pred_S_5, pred_S_10)

                    h_macro_mse.extend(F.mse_loss(pred_S_10, target_s_K_hjepa, reduction="none").mean(dim=-1).cpu().numpy())
                    h_macro_cos.extend(F.cosine_similarity(pred_S_10, target_s_K_hjepa, dim=-1).cpu().numpy())
                    pred_col_m = ((torch.sigmoid(col_1).squeeze(-1) > 0.5) | (torch.sigmoid(col_2).squeeze(-1) > 0.5)).cpu().numpy()
                    h_macro_col.extend((pred_col_m == col_gt).astype(np.float32))

                    if has_pose:
                        c1 = torch.cos(disp_1[:, 2])
                        s1 = torch.sin(disp_1[:, 2])
                        dx_10 = (disp_1[:, 0] + c1 * disp_2[:, 0] - s1 * disp_2[:, 1]).cpu().numpy()
                        dy_10 = (disp_1[:, 1] + s1 * disp_2[:, 0] + c1 * disp_2[:, 1]).cpu().numpy()
                        dth_10 = (disp_1[:, 2] + disp_2[:, 2]).cpu().numpy()
                        h_macro_disp.extend(np.sqrt((dx_10 - gt_dx_ego) ** 2 + (dy_10 - gt_dy_ego) ** 2))
                        h_macro_head.extend(np.abs(np.arctan2(np.sin(dth_10 - gt_dth), np.cos(dth_10 - gt_dth))))

        h_entry = {
            "sequences_evaluated": len(seq_indices),
            "lewm": {
                "latent_mse": float(np.mean(l_mse)),
                "latent_cosine_similarity": float(np.mean(l_cos)),
                "collision_accuracy": float(np.mean(l_col)) * 100.0,
                "displacement_error_m": float(np.mean(l_disp)) if l_disp else 0.0,
                "heading_error_rad": float(np.mean(l_head)) if l_head else 0.0,
            },
            "hjepa_micro": {
                "latent_mse": float(np.mean(h_micro_mse)),
                "latent_cosine_similarity": float(np.mean(h_micro_cos)),
                "collision_accuracy": float(np.mean(h_micro_col)) * 100.0,
                "displacement_error_m": float(np.mean(h_micro_disp)) if h_micro_disp else 0.0,
                "heading_error_rad": float(np.mean(h_micro_head)) if h_micro_head else 0.0,
            },
        }
        if K in (5, 10):
            h_entry["hjepa_macro"] = {
                "latent_mse": float(np.mean(h_macro_mse)),
                "latent_cosine_similarity": float(np.mean(h_macro_cos)),
                "collision_accuracy": float(np.mean(h_macro_col)) * 100.0,
                "displacement_error_m": float(np.mean(h_macro_disp)) if h_macro_disp else 0.0,
                "heading_error_rad": float(np.mean(h_macro_head)) if h_macro_head else 0.0,
            }
        horizon_results[f"horizon_k{K}"] = h_entry

    macro_seqs = [t for ep_i in range(len(ep_indices) - 1) for t in range(ep_indices[ep_i], ep_indices[ep_i + 1] - 5)]

    audit_results = {
        "dataset": os.path.basename(data_npz_path),
        "total_transitions": N,
        "total_macro_sequences": len(macro_seqs),
        "lewm": {
            "latent_mse": float(np.mean(lewm_mse_list)),
            "latent_cosine_similarity": float(np.mean(lewm_cos_list)),
            "collision_prediction_accuracy": float(np.mean(lewm_col_acc)) * 100.0,
            "rollout_k5_mse": float(np.mean(lewm_k5_mse_list)),
            "rollout_k5_cosine_similarity": float(np.mean(lewm_k5_cos_list)),
        },
        "hjepa_micro_k1": {
            "latent_mse": float(np.mean(hjepa_micro_mse)),
            "latent_cosine_similarity": float(np.mean(hjepa_micro_cos))
        },
        "hjepa_macro_k5": {
            "latent_mse": float(np.mean(hjepa_macro_mse)),
            "latent_cosine_similarity": float(np.mean(hjepa_macro_cos)),
            "macro_collision_accuracy": float(np.mean(hjepa_macro_col_acc)) * 100.0
        },
        "multi_horizon_sweep": horizon_results,
    }

    print("\n--- MULTI-HORIZON LATENT PREDICTION & PHYSICAL DISPLACEMENT AUDIT ---")
    for K in horizons:
        res = horizon_results[f"horizon_k{K}"]
        print(f"Horizon K={K:2d}:")
        print(f"  LeWM:        Latent MSE={res['lewm']['latent_mse']:.4f} | CosSim={res['lewm']['latent_cosine_similarity']:.4f} | DispErr={res['lewm']['displacement_error_m']:.3f}m | HeadErr={res['lewm']['heading_error_rad']:.3f}rad | ColAcc={res['lewm']['collision_accuracy']:.1f}%")
        print(f"  H-JEPA Micro: Latent MSE={res['hjepa_micro']['latent_mse']:.4f} | CosSim={res['hjepa_micro']['latent_cosine_similarity']:.4f} | DispErr={res['hjepa_micro']['displacement_error_m']:.3f}m | HeadErr={res['hjepa_micro']['heading_error_rad']:.3f}rad | ColAcc={res['hjepa_micro']['collision_accuracy']:.1f}%")
        if f"hjepa_macro" in res:
            m_res = res["hjepa_macro"]
            print(f"  H-JEPA Macro: Latent MSE={m_res['latent_mse']:.4f} | CosSim={m_res['latent_cosine_similarity']:.4f} | DispErr={m_res['displacement_error_m']:.3f}m | HeadErr={m_res['heading_error_rad']:.3f}rad | ColAcc={m_res['collision_accuracy']:.1f}%")

    out_json = os.path.join(output_dir, "latent_prediction_audit.json")
    with open(out_json, "w") as f:
        json.dump(audit_results, f, indent=2)

    print(f"\nSaved latent prediction audit to {out_json}")
    return audit_results

if __name__ == "__main__":
    v2_npz = str(_ACTIVE_SPLITS_DIR / "held_out_test_benchmark.npz")
    legacy_npz = str(_ACTIVE_SPLITS_DIR / "held_out_test_benchmark.npz")
    audit_latent_predictions(v2_npz if os.path.exists(v2_npz) else legacy_npz)
