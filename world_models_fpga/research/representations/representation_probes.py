"""
Phase 4: Supervised Representation Probes
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Trains diagnostic linear & ridge probes for 8 physical & semantic targets:
  1. Robot Position (x, y)
  2. Robot Orientation (sin theta, cos theta)
  3. Landmark Object Position (x_obj, y_obj)
  4. Relative Object Vector (dx_rel, dy_rel)
  5. Minimum Obstacle Clearance (d_min)
  6. Target Goal Direction (delta theta)
  7. Future Robot Position (x_t+1, y_t+1)
  8. Executed Action (vx, vy, wz)
Evaluates decodability across Baseline (z), LeWM (s), and H-JEPA (s) on held-out test data.
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

class RidgeProbe:
    """
    Closed-form ridge regression probe: W = (X^T X + alpha I)^(-1) X^T Y
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.W = None
        self.b = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        # Center data
        self.x_mean = np.mean(X, axis=0)
        self.y_mean = np.mean(Y, axis=0)
        Xc = X - self.x_mean
        Yc = Y - self.y_mean

        d = Xc.shape[1]
        cov = Xc.T @ Xc + self.alpha * np.eye(d)
        self.W = np.linalg.solve(cov, Xc.T @ Yc)
        self.b = self.y_mean - self.x_mean @ self.W
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b

    def score(self, X: np.ndarray, Y: np.ndarray):
        Y_pred = self.predict(X)
        resid = np.sum((Y - Y_pred) ** 2, axis=0)
        total = np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0) + 1e-8
        r2_per_dim = np.maximum(0.0, 1.0 - (resid / total))
        rmse_per_dim = np.sqrt(np.mean((Y - Y_pred) ** 2, axis=0))
        return float(np.mean(r2_per_dim)), float(np.mean(rmse_per_dim)), Y_pred

def extract_features_and_targets(npz_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Extracts latent features h from models and physical probe targets from MuJoCo ground truth.
    """
    data = np.load(npz_path)
    obs = data["rgb"]
    poses = data["robot_pose"]          # [x, y, theta]
    obj_pos = data["object_positions"]  # (N, 6, 3)
    actions = data["actions"]          # [vx, vy, wz]
    N = len(obs)

    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints"))
    base_model = BaselineNavigationPolicy().to(device)
    base_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "baseline_best.pt"), map_location=device))
    base_model.eval()

    lewm_model = LeWorldModel().to(device)
    lewm_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "lewm_best.pt"), map_location=device))
    lewm_model.eval()

    hjepa_model = HJEPAModel(stride_k=5).to(device)
    hjepa_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "hjepa_best.pt"), map_location=device))
    hjepa_model.eval()

    # Extract model latents
    batch_size = 256
    z_base, s_lewm, s_hjepa = [], [], []
    with torch.no_grad():
        for b in range(0, N, batch_size):
            imgs = torch.from_numpy(obs[b : b + batch_size]).permute(0, 3, 1, 2).float().to(device) / 255.0
            z_base.append(base_model.encoder(imgs).cpu().numpy())
            s_lewm.append(lewm_model.encode(imgs).cpu().numpy())
            s_hjepa.append(hjepa_model.encode(imgs).cpu().numpy())

    features = {
        "baseline_z": np.concatenate(z_base, axis=0),
        "lewm_s": np.concatenate(s_lewm, axis=0),
        "hjepa_s": np.concatenate(s_hjepa, axis=0)
    }

    # Construct Ground-Truth Physical Targets
    # 1. Robot Position (x, y)
    t_pos = poses[:, :2]

    # 2. Robot Orientation (sin, cos)
    th = poses[:, 2]
    t_ori = np.stack([np.sin(th), np.cos(th)], axis=-1)

    # 3. Primary Object Position (Yellow cylinder: obj 0)
    t_obj = obj_pos[:, 0, :2]

    # 4. Relative Object Vector (body frame)
    dx_w = t_obj[:, 0] - t_pos[:, 0]
    dy_w = t_obj[:, 1] - t_pos[:, 1]
    c_th, s_th = np.cos(th), np.sin(th)
    dx_b = c_th * dx_w + s_th * dy_w
    dy_b = -s_th * dx_w + c_th * dy_w
    t_rel_obj = np.stack([dx_b, dy_b], axis=-1)

    # 5. Obstacle Clearance (distance to nearest obstacle)
    dists_all = np.linalg.norm(obj_pos[:, :, :2] - t_pos[:, np.newaxis, :], axis=-1)
    t_clearance = np.min(dists_all, axis=-1, keepdims=True)

    # 6. Target Goal Direction
    t_goal_dir = np.arctan2(dy_b, dx_b)[:, np.newaxis]

    # 7. Future Position (lookahead 1 step)
    t_fut_pos = np.roll(t_pos, shift=-1, axis=0)

    # 8. Executed Action (vx, vy, wz)
    t_action = actions[:, :3]

    targets = {
        "robot_position": t_pos,
        "robot_orientation": t_ori,
        "object_position": t_obj,
        "relative_object_vector": t_rel_obj,
        "obstacle_clearance": t_clearance,
        "goal_direction": t_goal_dir,
        "future_position": t_fut_pos,
        "action_identity": t_action
    }
    return features, targets

def train_and_evaluate_probes(train_npz, test_npz, output_dir=None):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "probe_results"))
    os.makedirs(output_dir, exist_ok=True)

    print("\n=======================================================")
    print("  PHASE 4: SUPERVISED REPRESENTATION PROBING")
    print("=======================================================\n")

    train_feats, train_targets = extract_features_and_targets(train_npz)
    test_feats, test_targets = extract_features_and_targets(test_npz)

    probe_results = {}
    csv_rows = []

    for model_key in ["baseline_z", "lewm_s", "hjepa_s"]:
        print(f"\nEvaluating Probes on [{model_key.upper()}]...")
        probe_results[model_key] = {}
        X_train = train_feats[model_key]
        X_test = test_feats[model_key]

        for target_name, Y_train in train_targets.items():
            Y_test = test_targets[target_name]
            probe = RidgeProbe(alpha=10.0)
            probe.fit(X_train, Y_train)
            r2, rmse, _ = probe.score(X_test, Y_test)

            probe_results[model_key][target_name] = {
                "r2_score": r2,
                "rmse": rmse,
                "weights_norm": float(np.linalg.norm(probe.W))
            }

            csv_rows.append([
                model_key, target_name, f"{r2:.4f}", f"{rmse:.4f}"
            ])
            print(f"  [{target_name:24s}] R^2: {r2:.4f} | RMSE: {rmse:.4f}")

    # Save summary JSON and CSV
    json_path = os.path.join(output_dir, "probe_performance.json")
    with open(json_path, "w") as f:
        json.dump(probe_results, f, indent=2)

    csv_path = os.path.join(output_dir, "probe_performance.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model Latent", "Probe Target", "Test R^2 Score", "Test RMSE"])
        writer.writerows(csv_rows)

    print(f"\nSaved representation probe results to:")
    print(f"  -> {json_path}")
    print(f"  -> {csv_path}")
    return probe_results

if __name__ == "__main__":
    train_npz = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/splits/train_benchmark.npz"))
    test_npz = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/splits/held_out_test_benchmark.npz"))
    train_and_evaluate_probes(train_npz, test_npz)
