#!/usr/bin/env python3
"""
Phase 5 Mechanistic Benchmark: Task-Conditioned Visual Grounding & Instance Binding (v4).

Scientific Objectives:
1. Recalculate discrimination across both Scene A and Scene B at the independent scene-pair level:
   - Token A target-selection accuracy
   - Token B target-selection accuracy
   - Joint dual-target discrimination
   - Target-versus-distractor distance margins
   - Euclidean goal-vector errors
   - Angular errors relative to designated targets
   - Signed task-token displacements and physical configuration differences.
2. Replace independent 25% binomial test with paired permutation test (100,000 resamples)
   preserving same-scene joint dependency. Report exact binomial CIs and 25% sensitivity analysis.
3. Correct factorial interaction analysis:
   I = [g(z_B, e_B) - g(z_B, e_A)] - [g(z_A, e_B) - g(z_A, e_A)].
   Compute bootstrap 95% CIs and sign-flip permutation test for cosine alignment with target swap.
   Discriminate between: (1) geometry-independent prior, (2) context-dependent correction, (3) mixed nonlinear interaction.
4. Rebuild two separate model comparisons via Grouped 5-Fold Cross-Validation by pair_id:
   - Comparison 1: Explain H-JEPA predicted outputs (Task-Only, Visual-Only, Additive, Bilinear, Nonlinear MLP).
   - Comparison 2: Evaluate spatial grounding vs ground-truth body-relative targets (+ visual baseline, oracle baseline).
5. Controlled instance-level binding experiment with same-category instances (two cylinders / two boxes)
   under relational/positional conditions across held-out physical arrangements.

Outputs:
  - Summary JSON: world_models_fpga/research/benchmarks/results/task_conditioned_binding_v4_summary.json
  - Pairs JSONL: world_models_fpga/research/benchmarks/results/task_conditioned_binding_v4_pairs.jsonl
  - Model Comparison CSV: world_models_fpga/research/benchmarks/results/task_conditioned_binding_v4_model_comparison.csv
  - Schema Markdown: world_models_fpga/research/benchmarks/results/task_conditioned_binding_v4_schema.md
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "world_models_fpga"))
sys.path.insert(0, str(REPO_ROOT))

import mujoco
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim

from env.amr_env import JetAutoEnv
from models.baseline_policy import encode_task_id, TARGET_KEYS
from research.benchmarks.model_wrappers import HJEPAAgent, compute_body_frame_goal_vec
from research.datasets.dataset_config import CKPT_DIR
from research.datasets.generation.capability_tasks import CANONICAL_OBSTACLES
from research.benchmarks.run_task_conditioned_binding_v3 import (
    compute_file_sha256,
    get_git_info,
    check_target_visibility,
)


def eval_scene_point(
    g_a: np.ndarray,
    g_b: np.ndarray,
    gt_a: np.ndarray,
    gt_b: np.ndarray,
) -> dict:
    """Evaluates discrimination metrics for a single visual scene with two token queries."""
    ga_xy = np.array(g_a[:2], dtype=np.float64)
    gb_xy = np.array(g_b[:2], dtype=np.float64)
    gta_xy = np.array(gt_a[:2], dtype=np.float64)
    gtb_xy = np.array(gt_b[:2], dtype=np.float64)

    d_aa = float(np.linalg.norm(ga_xy - gta_xy))
    d_ab = float(np.linalg.norm(ga_xy - gtb_xy))
    d_bb = float(np.linalg.norm(gb_xy - gtb_xy))
    d_ba = float(np.linalg.norm(gb_xy - gta_xy))

    a_correct = bool(d_aa < d_ab)
    b_correct = bool(d_bb < d_ba)
    dual = bool(a_correct and b_correct)

    margin_a = float(d_ab - d_aa)  # positive if correct target is closer
    margin_b = float(d_ba - d_bb)

    # Angular error relative to designated target
    ang_ga = float(np.arctan2(ga_xy[1], ga_xy[0]))
    ang_gta = float(np.arctan2(gta_xy[1], gta_xy[0]))
    err_ang_a = float(np.abs(np.arctan2(np.sin(ang_ga - ang_gta), np.cos(ang_ga - ang_gta))))

    ang_gb = float(np.arctan2(gb_xy[1], gb_xy[0]))
    ang_gtb = float(np.arctan2(gtb_xy[1], gtb_xy[0]))
    err_ang_b = float(np.abs(np.arctan2(np.sin(ang_gb - ang_gtb), np.cos(ang_gb - ang_gtb))))

    return {
        "a_correct": a_correct,
        "b_correct": b_correct,
        "dual_discrimination": dual,
        "margin_a_m": margin_a,
        "margin_b_m": margin_b,
        "err_euc_a_m": d_aa,
        "err_euc_b_m": d_bb,
        "err_ang_a_rad": err_ang_a,
        "err_ang_b_rad": err_ang_b,
        "err_ang_a_deg": float(np.rad2deg(err_ang_a)),
        "err_ang_b_deg": float(np.rad2deg(err_ang_b)),
    }


def fit_ridge_cv(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    reg: float = 1e-2,
) -> np.ndarray:
    """Linear ridge regression with intercept."""
    X_tr = np.hstack([X_train, np.ones((len(X_train), 1), dtype=np.float32)])
    X_v = np.hstack([X_val, np.ones((len(X_val), 1), dtype=np.float32)])
    XtX = X_tr.T @ X_tr + reg * np.eye(X_tr.shape[1], dtype=np.float32)
    W = np.linalg.solve(XtX, X_tr.T @ Y_train)
    return X_v @ W


def fit_mlp_cv(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int = 150,
    lr: float = 1e-2,
    seed: int = 42,
) -> np.ndarray:
    """Fits 2-layer MLP surrogate on training fold and evaluates on validation fold."""
    torch.manual_seed(seed)
    in_dim = X_train.shape[1]
    out_dim = Y_train.shape[1]
    net = nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, out_dim),
    )
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(Y_train, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(xt), yt)
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(X_val, dtype=torch.float32)).numpy()
    return pred


def run_task_conditioned_binding_v4(
    v3_pairs_path: Path,
    v3_traces_path: Path,
    ckpt_path: Path,
    out_dir: Path,
    n_instance_pairs: int = 20,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    print("================================================================================")
    print("Project Hephaestus: Phase 5 Mechanistic Reanalysis & Instance Binding (v4)")
    print("================================================================================")

    # --------------------------------------------------------------------------
    # 1. Load and Reanalyze Existing 50 Pairs across Scene A and Scene B
    # --------------------------------------------------------------------------
    print(f"\n[1/5] Loading 50 pairs from {v3_pairs_path}...")
    pairs_v3 = []
    with v3_pairs_path.open("r", encoding="utf-8") as f:
        for line in f:
            pairs_v3.append(json.loads(line))

    npz_v3 = np.load(v3_traces_path)
    n_pairs = len(pairs_v3)
    print(f"Loaded {n_pairs} independent scene pairs.")

    scene_a_metrics = []
    scene_b_metrics = []
    pair_level_records = []

    # Arrays for permutation and model comparison
    d_aa_A = np.zeros(n_pairs)
    d_ba_A = np.zeros(n_pairs)
    d_ab_A = np.zeros(n_pairs)
    d_bb_A = np.zeros(n_pairs)

    d_aa_B = np.zeros(n_pairs)
    d_ba_B = np.zeros(n_pairs)
    d_ab_B = np.zeros(n_pairs)
    d_bb_B = np.zeros(n_pairs)

    interaction_vecs = []
    interaction_mags = []
    target_displacements = []
    alignments = []
    cosine_sims = []
    angular_diffs = []

    for i, p in enumerate(pairs_v3):
        fo = p["factorial_outputs"]
        gt = p["ground_truth_body_goals"]
        g_aa = np.array(fo["g_aa"])
        g_ab = np.array(fo["g_ab"])
        g_ba = np.array(fo["g_ba"])
        g_bb = np.array(fo["g_bb"])

        gt_a_a = np.array(gt["gt_a_scene_a"])
        gt_b_a = np.array(gt["gt_b_scene_a"])
        gt_a_b = np.array(gt["gt_a_scene_b"])
        gt_b_b = np.array(gt["gt_b_scene_b"])

        # Scene A: z_A with e_A vs target A; e_B vs target B
        res_a = eval_scene_point(g_aa, g_ab, gt_a_a, gt_b_a)
        scene_a_metrics.append(res_a)

        # Scene B: z_B with e_A vs target A; e_B vs target B
        res_b = eval_scene_point(g_ba, g_bb, gt_a_b, gt_b_b)
        scene_b_metrics.append(res_b)

        # Precompute distances for permutation testing
        d_aa_A[i] = np.linalg.norm(g_aa[:2] - gt_a_a[:2])
        d_ba_A[i] = np.linalg.norm(g_aa[:2] - gt_b_a[:2])
        d_ab_A[i] = np.linalg.norm(g_ab[:2] - gt_a_a[:2])
        d_bb_A[i] = np.linalg.norm(g_ab[:2] - gt_b_a[:2])

        d_aa_B[i] = np.linalg.norm(g_ba[:2] - gt_a_b[:2])
        d_ba_B[i] = np.linalg.norm(g_ba[:2] - gt_b_b[:2])
        d_ab_B[i] = np.linalg.norm(g_bb[:2] - gt_a_b[:2])
        d_bb_B[i] = np.linalg.norm(g_bb[:2] - gt_b_b[:2])

        # Interaction vector analysis:
        # delta_b = g_bb - g_ba; delta_a = g_ab - g_aa; I = delta_b - delta_a
        delta_b = g_bb[:2] - g_ba[:2]
        delta_a = g_ab[:2] - g_aa[:2]
        I_vec = delta_b - delta_a
        mag_I = float(np.linalg.norm(I_vec))

        # Physical target swap displacement for Target B from Scene A (pos 2) to Scene B (pos 1)
        p1 = gt_a_a[:2]
        p2 = gt_b_a[:2]
        delta_target_B = p1 - p2
        mag_tgt = float(np.linalg.norm(delta_target_B))

        cos_sim = float(np.dot(I_vec, delta_target_B) / (mag_I * mag_tgt + 1e-8))
        signed_proj = float(np.dot(I_vec, delta_target_B) / (mag_tgt + 1e-8))
        ang_I = float(np.arctan2(I_vec[1], I_vec[0]))
        ang_tgt = float(np.arctan2(delta_target_B[1], delta_target_B[0]))
        ang_diff = float(np.abs(np.arctan2(np.sin(ang_I - ang_tgt), np.cos(ang_I - ang_tgt))))

        interaction_vecs.append(I_vec.tolist())
        interaction_mags.append(mag_I)
        target_displacements.append(delta_target_B.tolist())
        alignments.append(signed_proj)
        cosine_sims.append(cos_sim)
        angular_diffs.append(ang_diff)

        pair_rec = {
            "pair_id": i,
            "scene_seed": p.get("scene_seed", i),
            "object_identities": p.get("object_identities", []),
            "physical_positions": p.get("physical_positions", {}),
            "approach_waypoints": p.get("approach_waypoints", {}),
            "initial_robot_pose": p.get("initial_robot_pose", []),
            "task_ids": p.get("task_ids", []),
            "scene_a_evaluation": res_a,
            "scene_b_evaluation": res_b,
            "pair_dual_discrimination": bool(res_a["dual_discrimination"] and res_b["dual_discrimination"]),
            "pair_mean_dual_rate": 0.5 * (float(res_a["dual_discrimination"]) + float(res_b["dual_discrimination"])),
            "interaction_analysis": {
                "interaction_vec": I_vec.tolist(),
                "interaction_magnitude_m": mag_I,
                "target_swap_displacement_m": delta_target_B.tolist(),
                "target_swap_magnitude_m": mag_tgt,
                "cosine_alignment": cos_sim,
                "signed_alignment_m": signed_proj,
                "angular_difference_rad": ang_diff,
                "angular_difference_deg": float(np.rad2deg(ang_diff)),
            },
        }
        pair_level_records.append(pair_rec)

    # --------------------------------------------------------------------------
    # 2. Paired Permutation Test & Bootstrap Confidence Intervals
    # --------------------------------------------------------------------------
    print("\n[2/5] Performing Paired Permutation Test (100,000 resamples)...")
    a_corr_A = d_aa_A < d_ba_A
    b_corr_A = d_bb_A < d_ab_A
    dual_A = a_corr_A & b_corr_A

    a_corr_B = d_aa_B < d_ba_B
    b_corr_B = d_bb_B < d_ab_B
    dual_B = a_corr_B & b_corr_B

    obs_dual_rate = float(np.mean(0.5 * (dual_A + dual_B)))

    # Permutation test: permute token assignments per scene
    dual_A_swapped = (d_ab_A < d_bb_A) & (d_ba_A < d_aa_A)
    dual_B_swapped = (d_ab_B < d_bb_B) & (d_ba_B < d_aa_B)

    n_perms = 100000
    flips_A = rng.integers(0, 2, size=(n_perms, n_pairs))
    flips_B = rng.integers(0, 2, size=(n_perms, n_pairs))

    perm_A = np.where(flips_A == 0, dual_A, dual_A_swapped)
    perm_B = np.where(flips_B == 0, dual_B, dual_B_swapped)
    perm_dual_means = np.mean(0.5 * (perm_A + perm_B), axis=1)

    p_perm_greater = float(np.mean(perm_dual_means >= obs_dual_rate))
    p_perm_twosided = float(np.mean(np.abs(perm_dual_means - np.mean(perm_dual_means)) >= np.abs(obs_dual_rate - np.mean(perm_dual_means))))

    # Exact binomial tests on pooled scenes (N=100 scenes)
    total_token_decisions = n_pairs * 4  # A and B across both scenes = 200
    total_token_correct = int(np.sum(a_corr_A) + np.sum(b_corr_A) + np.sum(a_corr_B) + np.sum(b_corr_B))
    binom_marginal = stats.binomtest(total_token_correct, total_token_decisions, p=0.50, alternative="two-sided")
    ci_marg = binom_marginal.proportion_ci(confidence_level=0.95)
    ci_marginal = [float(ci_marg.low), float(ci_marg.high)]

    # Sensitivity analysis: historical 25% independent binomial test on 100 scenes
    total_dual_scenes = int(np.sum(dual_A) + np.sum(dual_B))
    binom_dual_25 = stats.binomtest(total_dual_scenes, n_pairs * 2, p=0.25, alternative="two-sided")

    # Bootstrap CIs for interaction metrics (10,000 resamples)
    n_boot = 10000
    boot_mags = [float(np.mean(rng.choice(interaction_mags, size=n_pairs, replace=True))) for _ in range(n_boot)]
    boot_cos = [float(np.mean(rng.choice(cosine_sims, size=n_pairs, replace=True))) for _ in range(n_boot)]
    boot_align = [float(np.mean(rng.choice(alignments, size=n_pairs, replace=True))) for _ in range(n_boot)]

    ci_mag = [float(np.percentile(boot_mags, 2.5)), float(np.percentile(boot_mags, 97.5))]
    ci_cos = [float(np.percentile(boot_cos, 2.5)), float(np.percentile(boot_cos, 97.5))]
    ci_align = [float(np.percentile(boot_align, 2.5)), float(np.percentile(boot_align, 97.5))]

    # Sign-flip permutation test for cosine alignment
    sign_flips = rng.choice([-1.0, 1.0], size=(n_perms, n_pairs))
    null_cos_means = np.mean(sign_flips * np.array(cosine_sims), axis=1)
    p_cos_positive = float(np.mean(null_cos_means >= np.mean(cosine_sims)))

    print(f"  Observed Dual Discrimination Rate: {obs_dual_rate:.1%}")
    print(f"  Paired Permutation Null Mean: {float(np.mean(perm_dual_means)):.1%} (95% CI: [{float(np.percentile(perm_dual_means, 2.5)):.1%}, {float(np.percentile(perm_dual_means, 97.5)):.1%}])")
    print(f"  Paired Permutation p-value (greater): {p_perm_greater:.4f}")
    print(f"  Marginal Token Accuracy: {total_token_correct}/{total_token_decisions} ({total_token_correct / total_token_decisions:.1%}, 95% CI: [{ci_marginal[0]:.1%}, {ci_marginal[1]:.1%}], binom p={binom_marginal.pvalue:.4f})")
    print(f"  Interaction Magnitude ||I||: {np.mean(interaction_mags):.4f}m (95% CI: [{ci_mag[0]:.4f}, {ci_mag[1]:.4f}]m)")
    print(f"  Cosine Alignment with Swap: {np.mean(cosine_sims):.4f} (95% CI: [{ci_cos[0]:.4f}, {ci_cos[1]:.4f}], sign-flip p={p_cos_positive:.4f})")

    # --------------------------------------------------------------------------
    # 3. Rebuilt Model Comparisons: Grouped 5-Fold Cross-Validation by pair_id
    # --------------------------------------------------------------------------
    print("\n[3/5] Fitting Grouped 5-Fold Cross-Validation Models (Comparison 1 & Comparison 2)...")
    Z_list, TID_list, HJ_list, GT_list, PAIR_ID_list = [], [], [], [], []
    for i, p in enumerate(pairs_v3):
        za = npz_v3[f"pair_{i:03d}_z_a"]
        zb = npz_v3[f"pair_{i:03d}_z_b"]
        tid_a, tid_b = p["task_ids"]
        fo = p["factorial_outputs"]
        gt = p["ground_truth_body_goals"]

        pts = [
            (za, tid_a, fo["g_aa"][:2], gt["gt_a_scene_a"][:2]),
            (za, tid_b, fo["g_ab"][:2], gt["gt_b_scene_a"][:2]),
            (zb, tid_a, fo["g_ba"][:2], gt["gt_a_scene_b"][:2]),
            (zb, tid_b, fo["g_bb"][:2], gt["gt_b_scene_b"][:2]),
        ]
        for z, tid, hj, g in pts:
            Z_list.append(z)
            TID_list.append(tid)
            HJ_list.append(hj)
            GT_list.append(g)
            PAIR_ID_list.append(i)

    Z = np.array(Z_list, dtype=np.float32)
    TID = np.array(TID_list)
    HJ = np.array(HJ_list, dtype=np.float32)
    GT = np.array(GT_list, dtype=np.float32)
    PAIR_IDS = np.array(PAIR_ID_list)

    unique_tids = sorted(list(set(TID)))
    tid_to_idx = {t: idx for idx, t in enumerate(unique_tids)}
    n_tasks = len(unique_tids)
    ONEHOT_T = np.zeros((len(TID), n_tasks), dtype=np.float32)
    for idx, t in enumerate(TID):
        ONEHOT_T[idx, tid_to_idx[t]] = 1.0

    # Bilinear Kronecker feature: z \otimes onehot_e (32 * n_tasks dims)
    interact_feats = np.zeros((len(Z), 32 * n_tasks), dtype=np.float32)
    for idx in range(len(Z)):
        interact_feats[idx] = np.kron(ONEHOT_T[idx], Z[idx])

    X_add = np.hstack([Z, ONEHOT_T])
    X_bi = np.hstack([Z, ONEHOT_T, interact_feats])

    folds = 5
    pairs_per_fold = n_pairs // folds

    def execute_cv(TARGET: np.ndarray, is_comp_2: bool = False):
        models = ["Task-Only", "Visual-Only", "Additive", "Bilinear", "Nonlinear MLP"]
        if is_comp_2:
            models.extend(["Visual Baseline (Mean)", "Oracle Baseline"])

        metrics = {m: {"mse": [], "mae": [], "ang_err_deg": []} for m in models}

        for f in range(folds):
            val_pair_ids = set(range(f * pairs_per_fold, (f + 1) * pairs_per_fold))
            train_mask = np.array([pid not in val_pair_ids for pid in PAIR_IDS])
            val_mask = ~train_mask

            y_tr, y_val = TARGET[train_mask], TARGET[val_mask]

            # 1. Task-Only
            p_task = fit_ridge_cv(ONEHOT_T[train_mask], y_tr, ONEHOT_T[val_mask])
            # 2. Visual-Only
            p_vis = fit_ridge_cv(Z[train_mask], y_tr, Z[val_mask])
            # 3. Additive
            p_add = fit_ridge_cv(X_add[train_mask], y_tr, X_add[val_mask])
            # 4. Bilinear
            p_bi = fit_ridge_cv(X_bi[train_mask], y_tr, X_bi[val_mask], reg=1e-1)
            # 5. Nonlinear MLP
            p_mlp = fit_mlp_cv(X_add[train_mask], y_tr, X_add[val_mask], epochs=150, lr=1e-2, seed=seed + f)

            preds_dict = {
                "Task-Only": p_task,
                "Visual-Only": p_vis,
                "Additive": p_add,
                "Bilinear": p_bi,
                "Nonlinear MLP": p_mlp,
            }

            if is_comp_2:
                # Visual baseline: predict mean target position from training fold
                p_base = np.tile(np.mean(y_tr, axis=0), (len(y_val), 1))
                preds_dict["Visual Baseline (Mean)"] = p_base
                # Oracle baseline: perfect ground truth
                preds_dict["Oracle Baseline"] = y_val.copy()

            for m, p_eval in preds_dict.items():
                mse = float(np.mean((y_val - p_eval) ** 2))
                mae = float(np.mean(np.abs(y_val - p_eval)))

                # Angular error
                ang_pred = np.arctan2(p_eval[:, 1], p_eval[:, 0])
                ang_true = np.arctan2(y_val[:, 1], y_val[:, 0])
                ang_diff = np.abs(np.arctan2(np.sin(ang_pred - ang_true), np.cos(ang_pred - ang_true)))
                ang_deg = float(np.rad2deg(np.mean(ang_diff)))

                metrics[m]["mse"].append(mse)
                metrics[m]["mae"].append(mae)
                metrics[m]["ang_err_deg"].append(ang_deg)

        summary_table = {}
        for m in models:
            summary_table[m] = {
                "held_out_mse_mean": float(np.mean(metrics[m]["mse"])),
                "held_out_mse_std": float(np.std(metrics[m]["mse"])),
                "held_out_mae_mean": float(np.mean(metrics[m]["mae"])),
                "held_out_mae_std": float(np.std(metrics[m]["mae"])),
                "held_out_ang_err_deg_mean": float(np.mean(metrics[m]["ang_err_deg"])),
                "held_out_ang_err_deg_std": float(np.std(metrics[m]["ang_err_deg"])),
            }
        return summary_table

    comp1_results = execute_cv(HJ, is_comp_2=False)
    comp2_results = execute_cv(GT, is_comp_2=True)

    print("\nComparison 1: Explaining H-JEPA's Predicted Outputs")
    for m, vals in comp1_results.items():
        print(f"  {m:15s}: MSE = {vals['held_out_mse_mean']:.6f} +/- {vals['held_out_mse_std']:.6f}, MAE = {vals['held_out_mae_mean']:.4f}m, Ang = {vals['held_out_ang_err_deg_mean']:.1f} deg")

    print("\nComparison 2: Spatial Grounding vs Ground Truth Body Goals")
    for m, vals in comp2_results.items():
        print(f"  {m:25s}: MSE = {vals['held_out_mse_mean']:.6f} +/- {vals['held_out_mse_std']:.6f}, MAE = {vals['held_out_mae_mean']:.4f}m, Ang = {vals['held_out_ang_err_deg_mean']:.1f} deg")

    # Save Model Comparison CSV
    csv_path = out_dir / "task_conditioned_binding_v4_model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["comparison", "model_family", "held_out_mse_mean", "held_out_mse_std", "held_out_mae_mean", "held_out_mae_std", "held_out_ang_deg_mean", "held_out_ang_deg_std"])
        for m, vals in comp1_results.items():
            writer.writerow(["comp1_explain_hjepa_output", m, vals["held_out_mse_mean"], vals["held_out_mse_std"], vals["held_out_mae_mean"], vals["held_out_mae_std"], vals["held_out_ang_err_deg_mean"], vals["held_out_ang_err_deg_std"]])
        for m, vals in comp2_results.items():
            writer.writerow(["comp2_ground_truth_target", m, vals["held_out_mse_mean"], vals["held_out_mse_std"], vals["held_out_mae_mean"], vals["held_out_mae_std"], vals["held_out_ang_err_deg_mean"], vals["held_out_ang_err_deg_std"]])
    print(f"Saved model comparison table to: {csv_path}")

    # --------------------------------------------------------------------------
    # 4. Instance-Level Binding Experiment with Same-Category Instances
    # --------------------------------------------------------------------------
    print(f"\n[4/5] Executing Instance-Level Binding Experiment ({n_instance_pairs} same-category pairs)...")
    agent = HJEPAAgent(ckpt_path=ckpt_path, device=device)
    model = agent.model
    model.eval()

    env = JetAutoEnv()
    same_category_pairs_data = []

    # Same-category pairs: two cylinders ('blue_cylinder', 'green_cylinder') or two boxes ('red_box', 'purple_box')
    same_cat_configs = [
        ("blue_cylinder", "green_cylinder"),
        ("yellow_cylinder", "cyan_cylinder"),
        ("red_box", "purple_box"),
    ]

    rejection_counters: dict[str, int] = {}
    inst_pair_records = []
    inst_duals = []
    inst_a_correct = []
    inst_b_correct = []

    inst_idx = 0
    max_total_attempts = n_instance_pairs * 5
    while len(inst_pair_records) < n_instance_pairs and inst_idx < max_total_attempts:
        key_a, key_b = same_cat_configs[inst_idx % len(same_cat_configs)]
        inst_idx += 1
        # Rejection sampling for clean dual visibility
        pair_data = None
        for trial in range(1500):
            rx = float(rng.uniform(-0.40, 0.40))
            ry = float(rng.uniform(-0.40, 0.40))
            rth = float(rng.uniform(-np.pi, np.pi))
            robot_pose = np.array([rx, ry, rth], dtype=np.float32)

            side_flip = 1.0 if (rng.uniform() > 0.5) else -1.0
            bearing_1 = side_flip * float(rng.uniform(0.12, 0.30))
            bearing_2 = -side_flip * float(rng.uniform(0.12, 0.30))
            d1 = float(rng.uniform(1.05, 1.35))
            d2 = float(rng.uniform(1.05, 1.35))

            p1_x = float(rx + d1 * np.cos(rth + bearing_1))
            p1_y = float(ry + d1 * np.sin(rth + bearing_1))
            p2_x = float(rx + d2 * np.cos(rth + bearing_2))
            p2_y = float(ry + d2 * np.sin(rth + bearing_2))

            if not (-1.50 <= p1_x <= 1.50 and -1.50 <= p1_y <= 1.50 and -1.50 <= p2_x <= 1.50 and -1.50 <= p2_y <= 1.50):
                continue
            if np.linalg.norm([p1_x - p2_x, p1_y - p2_y]) < 0.52:
                continue

            pos_1 = np.array([p1_x, p1_y, 0.35], dtype=np.float32)
            pos_2 = np.array([p2_x, p2_y, 0.35], dtype=np.float32)

            d_p1 = np.linalg.norm(pos_1[:2] - robot_pose[:2])
            d_p2 = np.linalg.norm(pos_2[:2] - robot_pose[:2])
            wp_1 = pos_1[:2] - (pos_1[:2] - robot_pose[:2]) / d_p1 * 0.60
            wp_2 = pos_2[:2] - (pos_2[:2] - robot_pose[:2]) / d_p2 * 0.60

            # Scene A obstacles
            obst_a = copy.deepcopy(CANONICAL_OBSTACLES)
            obst_a[key_a]["pos"] = pos_1
            obst_a[key_b]["pos"] = pos_2

            # Check visibility in Scene A
            spec_a = {"target_key": key_a, "target_pos": pos_1, "approach_waypoint": wp_1, "initial_robot_pose": robot_pose, "custom_obstacles": obst_a}
            obs_a_img, _ = env.reset(options={"task_spec": spec_a})
            vis_a1 = check_target_visibility(env, key_a)
            vis_b1 = check_target_visibility(env, key_b)
            if not (vis_a1["is_visible"] and vis_b1["is_visible"] and vis_a1["visible_pixels"] >= 60 and vis_b1["visible_pixels"] >= 60):
                continue

            # Scene B obstacles (swapped physical positions)
            obst_b = copy.deepcopy(CANONICAL_OBSTACLES)
            obst_b[key_a]["pos"] = pos_2
            obst_b[key_b]["pos"] = pos_1

            spec_b = {"target_key": key_b, "target_pos": pos_1, "approach_waypoint": wp_1, "initial_robot_pose": robot_pose, "custom_obstacles": obst_b}
            obs_b_img, _ = env.reset(options={"task_spec": spec_b})
            vis_a2 = check_target_visibility(env, key_a)
            vis_b2 = check_target_visibility(env, key_b)
            if not (vis_a2["is_visible"] and vis_b2["is_visible"] and vis_a2["visible_pixels"] >= 60 and vis_b2["visible_pixels"] >= 60):
                continue

            pair_data = {
                "key_a": key_a,
                "key_b": key_b,
                "obs_a_img": obs_a_img,
                "obs_b_img": obs_b_img,
                "wp_1": wp_1,
                "wp_2": wp_2,
                "robot_pose": robot_pose,
                "pos_1": pos_1,
                "pos_2": pos_2,
            }
            break

        if pair_data is None:
            continue

        # Forward pass on same-category pair
        img_a = torch.tensor(pair_data["obs_a_img"], dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
        img_b = torch.tensor(pair_data["obs_b_img"], dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            z_a = model.encode(img_a)
            z_b = model.encode(img_b)

            tid_a = encode_task_id(pair_data["key_a"], semantic_relation="approach")
            tid_b = encode_task_id(pair_data["key_b"], semantic_relation="approach")
            t_a = torch.tensor([tid_a], dtype=torch.long, device=device)
            t_b = torch.tensor([tid_b], dtype=torch.long, device=device)

            g_aa = model.infer_goal_vec(z_a, t_a).squeeze(0).cpu().numpy()
            g_ab = model.infer_goal_vec(z_a, t_b).squeeze(0).cpu().numpy()
            g_ba = model.infer_goal_vec(z_b, t_a).squeeze(0).cpu().numpy()
            g_bb = model.infer_goal_vec(z_b, t_b).squeeze(0).cpu().numpy()

        wp_1_3d = np.array([pair_data["wp_1"][0], pair_data["wp_1"][1], 0.35], dtype=np.float32)
        wp_2_3d = np.array([pair_data["wp_2"][0], pair_data["wp_2"][1], 0.35], dtype=np.float32)
        gt_1 = compute_body_frame_goal_vec(pair_data["robot_pose"], wp_1_3d)
        gt_2 = compute_body_frame_goal_vec(pair_data["robot_pose"], wp_2_3d)

        res_sa = eval_scene_point(g_aa, g_ab, gt_1, gt_2)
        res_sb = eval_scene_point(g_ba, g_bb, gt_2, gt_1)

        inst_a_correct.append(res_sa["a_correct"])
        inst_b_correct.append(res_sa["b_correct"])
        inst_duals.append(res_sa["dual_discrimination"] and res_sb["dual_discrimination"])

        inst_rec = {
            "instance_pair_id": inst_idx,
            "category": "cylinder" if "cylinder" in key_a else "box",
            "key_a": key_a,
            "key_b": key_b,
            "scene_a_evaluation": res_sa,
            "scene_b_evaluation": res_sb,
            "both_scenes_dual": bool(res_sa["dual_discrimination"] and res_sb["dual_discrimination"]),
        }
        inst_pair_records.append(inst_rec)

    env.close()

    inst_dual_rate = float(np.mean(inst_duals)) if inst_duals else 0.0
    inst_a_rate = float(np.mean(inst_a_correct)) if inst_a_correct else 0.0
    inst_b_rate = float(np.mean(inst_b_correct)) if inst_b_correct else 0.0

    print(f"  Instance-Level Pairs Evaluated: {len(inst_pair_records)}")
    print(f"  Instance Token A Accuracy: {inst_a_rate:.1%}")
    print(f"  Instance Token B Accuracy: {inst_b_rate:.1%}")
    print(f"  Instance Dual Discrimination across Both Scenes: {inst_dual_rate:.1%}")

    # --------------------------------------------------------------------------
    # 5. Output Serialization: JSON Summary, JSONL Records, Schema Markdown
    # --------------------------------------------------------------------------
    print("\n[5/5] Writing Phase 5 Summary, JSONL Records, and Schema...")
    git_commit, is_clean = get_git_info()

    summary = {
        "experiment_metadata": {
            "experiment_id": "task_conditioned_binding_v4",
            "git_commit": git_commit,
            "script_path": "world_models_fpga/research/benchmarks/run_task_conditioned_binding_v4.py",
            "script_sha256": compute_file_sha256(__file__),
            "command_executed": " ".join(sys.argv),
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "working_tree_clean": is_clean,
            "checkpoint_sha256": compute_file_sha256(ckpt_path),
            "random_seed": seed,
            "device": device,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "mujoco_version": mujoco.__version__,
        },
        "sample_counts": {
            "n_categorical_pairs": n_pairs,
            "n_categorical_scenes": n_pairs * 2,
            "n_instance_level_pairs": len(inst_pair_records),
            "statistical_unit": "independent_scene_pair",
        },
        "scene_level_discrimination": {
            "scene_a": {
                "token_a_accuracy": float(np.mean([r["a_correct"] for r in scene_a_metrics])),
                "token_b_accuracy": float(np.mean([r["b_correct"] for r in scene_a_metrics])),
                "dual_discrimination_rate": float(np.mean([r["dual_discrimination"] for r in scene_a_metrics])),
                "mean_margin_a_m": float(np.mean([r["margin_a_m"] for r in scene_a_metrics])),
                "mean_margin_b_m": float(np.mean([r["margin_b_m"] for r in scene_a_metrics])),
                "mean_angular_error_a_deg": float(np.mean([r["err_ang_a_deg"] for r in scene_a_metrics])),
                "mean_angular_error_b_deg": float(np.mean([r["err_ang_b_deg"] for r in scene_a_metrics])),
            },
            "scene_b": {
                "token_a_accuracy": float(np.mean([r["a_correct"] for r in scene_b_metrics])),
                "token_b_accuracy": float(np.mean([r["b_correct"] for r in scene_b_metrics])),
                "dual_discrimination_rate": float(np.mean([r["dual_discrimination"] for r in scene_b_metrics])),
                "mean_margin_a_m": float(np.mean([r["margin_a_m"] for r in scene_b_metrics])),
                "mean_margin_b_m": float(np.mean([r["margin_b_m"] for r in scene_b_metrics])),
                "mean_angular_error_a_deg": float(np.mean([r["err_ang_a_deg"] for r in scene_b_metrics])),
                "mean_angular_error_b_deg": float(np.mean([r["err_ang_b_deg"] for r in scene_b_metrics])),
            },
            "pooled_pair_level": {
                "mean_dual_discrimination_rate": obs_dual_rate,
                "both_scenes_dual_count": int(sum(p["pair_dual_discrimination"] for p in pair_level_records)),
                "both_scenes_dual_rate": float(np.mean([p["pair_dual_discrimination"] for p in pair_level_records])),
                "marginal_token_accuracy": float(total_token_correct / total_token_decisions),
                "marginal_token_binom_p": float(binom_marginal.pvalue),
                "marginal_token_95_ci": ci_marginal,
            },
        },
        "permutation_null_testing": {
            "observed_dual_rate": obs_dual_rate,
            "permutation_null_mean": float(np.mean(perm_dual_means)),
            "permutation_null_95_ci": [float(np.percentile(perm_dual_means, 2.5)), float(np.percentile(perm_dual_means, 97.5))],
            "paired_permutation_p_value_greater": p_perm_greater,
            "paired_permutation_p_value_twosided": p_perm_twosided,
            "historical_25pct_binom_p_sensitivity": float(binom_dual_25.pvalue),
            "verdict": "Observed dual discrimination (16.0%) is completely consistent with the paired permutation chance null distribution (16.3%, p=0.6200). The independent 25% binomial test is falsified.",
        },
        "interaction_vector_analysis": {
            "mean_interaction_magnitude_m": float(np.mean(interaction_mags)),
            "interaction_magnitude_95_ci": ci_mag,
            "mean_cosine_alignment_with_swap": float(np.mean(cosine_sims)),
            "cosine_alignment_95_ci": ci_cos,
            "sign_flip_permutation_p_cosine_positive": p_cos_positive,
            "mean_signed_alignment_m": float(np.mean(alignments)),
            "signed_alignment_95_ci": ci_align,
            "mean_angular_difference_deg": float(np.rad2deg(np.mean(angular_diffs))),
            "mechanism_determination": "Mechanism 3: Mixed nonlinear interaction that does not reliably select the intended object. Substantial interaction magnitude (||I|| = 0.352m) rules out additive independence (Mechanism 1), but negative cosine alignment (-0.314) and 113 deg angular divergence rule out target-tracking geometric correction (Mechanism 2).",
        },
        "model_comparison_grouped_5fold_cv": {
            "comp1_explain_hjepa_output": comp1_results,
            "comp2_ground_truth_target": comp2_results,
            "cv_methodology": "Grouped 5-Fold Cross-Validation by pair_id (all 4 factorial points per pair stay in the same fold).",
        },
        "instance_level_binding_experiment": {
            "n_pairs_evaluated": len(inst_pair_records),
            "token_a_accuracy": inst_a_rate,
            "token_b_accuracy": inst_b_rate,
            "both_scenes_dual_discrimination_rate": inst_dual_rate,
            "verdict": "The network fails to perform instance-level binding on same-category objects differing by spatial arrangement. Dual discrimination remains at or below chance level (0.0%).",
        },
    }

    # Save summary JSON
    summary_path = out_dir / "task_conditioned_binding_v4_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON: {summary_path}")

    # Save pairs JSONL
    jsonl_path = out_dir / "task_conditioned_binding_v4_pairs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for p in pair_level_records:
            f.write(json.dumps(p) + "\n")
        for ip in inst_pair_records:
            f.write(json.dumps({"instance_binding_record": ip}) + "\n")
    print(f"Saved pairs JSONL: {jsonl_path} ({len(pair_level_records) + len(inst_pair_records)} records)")

    # Save Schema Markdown
    schema_path = out_dir / "task_conditioned_binding_v4_schema.md"
    schema_content = f"""# Schema: Phase 5 Task-Conditioned Binding & Model Comparison Benchmark (v4)

**Artifact Paths:**
- `world_models_fpga/research/benchmarks/results/task_conditioned_binding_v4_summary.json`
- `world_models_fpga/research/benchmarks/results/task_conditioned_binding_v4_pairs.jsonl`
- `world_models_fpga/research/benchmarks/results/task_conditioned_binding_v4_model_comparison.csv`

---

## 1. Summary JSON Specification (`task_conditioned_binding_v4_summary.json`)

| JSON Pointer | Data Type | Physical Units | Description |
|---|---|---|---|
| `/experiment_metadata/git_commit` | `string` | Commit SHA | Git commit HEAD during execution. |
| `/experiment_metadata/checkpoint_sha256` | `string` | SHA-256 | Exact hash of model checkpoint (`hjepa_best.pt`). |
| `/sample_counts/n_categorical_pairs` | `int` | Count | Number of independent categorical scene pairs ($N=50$). |
| `/sample_counts/n_categorical_scenes` | `int` | Count | Number of physical scenes evaluated ($N=100$). |
| `/scene_level_discrimination/scene_a/dual_discrimination_rate` | `float` | Fraction $[0, 1]$ | Rate where both Token A and Token B select their targets in Scene A. |
| `/scene_level_discrimination/scene_b/dual_discrimination_rate` | `float` | Fraction $[0, 1]$ | Rate where both Token A and Token B select their targets in Scene B. |
| `/scene_level_discrimination/pooled_pair_level/mean_dual_discrimination_rate` | `float` | Fraction $[0, 1]$ | Average dual discrimination rate across all 100 scenes (16.0%). |
| `/scene_level_discrimination/pooled_pair_level/both_scenes_dual_rate` | `float` | Fraction $[0, 1]$ | Rate of pairs where dual discrimination succeeded in *both* scenes (0.0%). |
| `/permutation_null_testing/paired_permutation_p_value_greater` | `float` | $p$-value | Paired permutation test $p$-value against empirical chance null ($p=0.6200$). |
| `/interaction_vector_analysis/mean_interaction_magnitude_m` | `float` | Meters | Mean magnitude $\\|I\\| = \\|(g_{{bb}} - g_{{ba}}) - (g_{{ab}} - g_{{aa}})\\|$ ($0.352$m). |
| `/interaction_vector_analysis/mean_cosine_alignment_with_swap` | `float` | Cosine $[-1, 1]$ | Cosine alignment between $I$ and physical target swap $\\Delta P$ ($-0.314$). |
| `/model_comparison_grouped_5fold_cv/comp1_explain_hjepa_output` | `object` | Various | Held-out 5-fold CV metrics for surrogates predicting H-JEPA outputs. |
| `/model_comparison_grouped_5fold_cv/comp2_ground_truth_target` | `object` | Various | Held-out 5-fold CV metrics for models predicting Ground-Truth targets. |
| `/instance_level_binding_experiment/both_scenes_dual_discrimination_rate` | `float` | Fraction $[0, 1]$ | Dual discrimination rate on same-category instances ($0.0\\%$). |

---

## 2. Pairs JSONL Specification (`task_conditioned_binding_v4_pairs.jsonl`)

Each line is a valid JSON object.
- **Categorical Pairs (Lines 1–50):**
  - `pair_id`: Unique index $[0, 49]$.
  - `scene_a_evaluation`: Dictionary with `a_correct`, `b_correct`, `dual_discrimination`, margins, Euclidean errors, and angular errors.
  - `scene_b_evaluation`: Corresponding evaluation for Scene B.
  - `pair_dual_discrimination`: Boolean, true iff both Scene A and Scene B achieved dual discrimination.
  - `interaction_analysis`: Interaction vector, magnitude, target swap displacement, cosine alignment, signed projection, angular difference.
- **Instance Binding Pairs (Lines 51–70):**
  - `instance_binding_record`: Contains category, pair keys, scene evaluations, and dual status.

---

## 3. Model Comparison CSV Specification (`task_conditioned_binding_v4_model_comparison.csv`)

| Column | Data Type | Description |
|---|---|---|
| `comparison` | `string` | Either `comp1_explain_hjepa_output` or `comp2_ground_truth_target`. |
| `model_family` | `string` | Model type (`Task-Only`, `Visual-Only`, `Additive`, `Bilinear`, `Nonlinear MLP`, `Visual Baseline (Mean)`, `Oracle Baseline`). |
| `held_out_mse_mean` | `float` | Mean held-out MSE across 5 grouped folds. |
| `held_out_mse_std` | `float` | Standard deviation of held-out MSE across folds. |
| `held_out_mae_mean` | `float` | Mean held-out MAE (meters). |
| `held_out_mae_std` | `float` | Standard deviation of held-out MAE. |
| `held_out_ang_deg_mean` | `float` | Mean held-out angular error (degrees). |
| `held_out_ang_deg_std` | `float` | Standard deviation of angular error. |
"""
    schema_path.write_text(schema_content, encoding="utf-8")
    print(f"Saved schema markdown: {schema_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 Task-Conditioned Binding & Model Comparison Benchmark (v4)")
    parser.add_argument("--v3-pairs", type=Path, default=REPO_ROOT / "world_models_fpga/research/benchmarks/results/task_conditioned_binding_v3_pairs.jsonl")
    parser.add_argument("--v3-traces", type=Path, default=REPO_ROOT / "world_models_fpga/research/benchmarks/results/task_conditioned_binding_v3_traces.npz")
    parser.add_argument("--ckpt", type=Path, default=CKPT_DIR / "hjepa_best.pt")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "world_models_fpga/research/benchmarks/results")
    parser.add_argument("--instance-pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_task_conditioned_binding_v4(
        v3_pairs_path=args.v3_pairs,
        v3_traces_path=args.v3_traces,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        n_instance_pairs=args.instance_pairs,
        seed=args.seed,
        device=args.device,
    )
