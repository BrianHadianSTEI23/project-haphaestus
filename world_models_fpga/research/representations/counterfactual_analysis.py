"""
Phase 4: Counterfactual Feature Analysis (E1 - E7)
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Measures the sensitivity of intermediate latent representations and SAE features to isolated
physical and semantic counterfactual mutations:
  - E1: Position Shift (target object relocated)
  - E2: Appearance Mutation (target color / texture altered)
  - E3: Robot Heading Rotation (initial heading rotated by 90 deg)
  - E4: Robot Starting Position (robot starts in opposite quadrant)
  - E5: Obstacle Insertion (intervening obstacle placed along line of sight)
  - E6: Geometric Shape Mutation (cylinder swapped with rectangular box)
  - E7: Camera Elevation / Viewpoint Mutation (camera pitch altered by 10 deg)

Calculates feature sensitivity:
  S_k(E) = mean_{pairs} [ |h_k(E0) - h_k(Ei)| / (std(h_k) + eps) ]
Classifies each latent dimension into functional semantic categories:
  - Appearance
  - Geometry / Object Location
  - Egocentric Heading
  - Global Starting Pose
  - Obstacle Clearance
  - Viewpoint / Perspective
"""

import os
import sys
import json
import csv
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from models.baseline_policy import BaselineNavigationPolicy
from models.le_world_model import LeWorldModel
from models.h_jepa import HJEPAModel

EXPERIMENT_TYPES = [
    "E1_position_shift",
    "E2_color_shift",
    "E3_robot_rotation",
    "E4_robot_start_pos",
    "E5_obstacle_insertion",
    "E6_shape_mutation",
    "E7_camera_elevation"
]

CATEGORY_MAP = {
    "E1_position_shift": "Geometry / Target Location",
    "E2_color_shift": "Visual Appearance / Color",
    "E3_robot_rotation": "Egocentric Heading",
    "E4_robot_start_pos": "Global Position",
    "E5_obstacle_insertion": "Obstacle / Clearance",
    "E6_shape_mutation": "Object Class / Geometry",
    "E7_camera_elevation": "Viewpoint / Perspective"
}

def analyze_counterfactuals(cf_npz_path, output_dir=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "counterfactual_analysis"))
    os.makedirs(output_dir, exist_ok=True)

    print("\n=======================================================")
    print("  PHASE 4: COUNTERFACTUAL FEATURE SENSITIVITY ANALYSIS")
    print(f"  Suite File: {cf_npz_path}")
    print(f"  Device: {device}")
    print("=======================================================\n")

    cf_data = np.load(cf_npz_path)
    obs = cf_data["rgb"] # (12250, 64, 64, 3)
    ep_starts = np.where(cf_data["episode_starts"])[0]
    total_episodes = len(ep_starts)
    steps_per_ep = 35
    assert total_episodes == 350, f"Expected 350 episodes, got {total_episodes}"

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

    # Extract initial frame (t=0) for all 350 episodes
    t0_indices = ep_starts
    t0_rgb = obs[t0_indices] # (350, 64, 64, 3)

    with torch.no_grad():
        imgs = torch.from_numpy(t0_rgb).permute(0, 3, 1, 2).float().to(device) / 255.0
        z_base = base_model.encoder(imgs).cpu().numpy()
        s_lewm = lewm_model.encode(imgs).cpu().numpy()
        s_hjepa = hjepa_model.encode(imgs).cpu().numpy()

    models_latents = {
        "baseline_z": z_base,
        "lewm_s": s_lewm,
        "hjepa_s": s_hjepa
    }

    # Structure: 7 experiments * 25 pairs * 2 episodes = 350 episodes
    # Each pair consists of:
    #   idx_0 = exp_idx * 50 + pair_idx * 2
    #   idx_1 = exp_idx * 50 + pair_idx * 2 + 1
    pairs_per_type = 25
    sensitivity_results = {}
    classification_results = {}
    csv_rows = []

    for m_name, latents in models_latents.items():
        std_devs = np.std(latents, axis=0) + 1e-6
        dim = latents.shape[-1]
        sensitivity_results[m_name] = {}
        classification_results[m_name] = {}

        exp_sens_matrix = np.zeros((len(EXPERIMENT_TYPES), dim))

        for e_idx, e_type in enumerate(EXPERIMENT_TYPES):
            deltas = []
            for p in range(pairs_per_type):
                ep0_idx = e_idx * 50 + p * 2
                epi_idx = ep0_idx + 1
                diff = np.abs(latents[ep0_idx] - latents[epi_idx]) / std_devs
                deltas.append(diff)

            mean_delta = np.mean(deltas, axis=0) # (dim,)
            exp_sens_matrix[e_idx] = mean_delta
            sensitivity_results[m_name][e_type] = {
                "category": CATEGORY_MAP[e_type],
                "mean_sensitivity": float(np.mean(mean_delta)),
                "max_sensitivity": float(np.max(mean_delta)),
                "per_dim_sensitivity": mean_delta.tolist()
            }

        # Classify each dimension according to maximum response
        dom_exp_indices = np.argmax(exp_sens_matrix, axis=0)
        dom_exp_values = np.max(exp_sens_matrix, axis=0)

        feature_profiles = []
        for d in range(dim):
            top_exp = EXPERIMENT_TYPES[dom_exp_indices[d]]
            profile = {
                "dim": d,
                "dominant_factor": top_exp,
                "category": CATEGORY_MAP[top_exp],
                "sensitivity": float(dom_exp_values[d]),
                "is_selective": bool(dom_exp_values[d] > 1.2 * np.mean(exp_sens_matrix[:, d]))
            }
            feature_profiles.append(profile)

            csv_rows.append([
                m_name, d, profile["category"], top_exp, f"{dom_exp_values[d]:.4f}", profile["is_selective"]
            ])

        classification_results[m_name] = feature_profiles

    # Write sensitivity and classification summaries
    sens_json = os.path.join(output_dir, "counterfactual_sensitivity.json")
    with open(sens_json, "w") as f:
        json.dump(sensitivity_results, f, indent=2)

    class_json = os.path.join(output_dir, "feature_classification.json")
    with open(class_json, "w") as f:
        json.dump(classification_results, f, indent=2)

    csv_path = os.path.join(output_dir, "feature_classification.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Dimension", "Dominant Category", "Experiment Type", "Sensitivity", "Is Selective"])
        writer.writerows(csv_rows)

    print(f"Saved Counterfactual Analysis results to:")
    print(f"  -> {sens_json}")
    print(f"  -> {class_json}")
    print(f"  -> {csv_path}")

    # Print summary table
    print("\n--- Model Representation Selectivity Breakdown ---")
    for m_name in models_latents.keys():
        print(f"\n[{m_name.upper()}]:")
        counts = {}
        for prof in classification_results[m_name]:
            cat = prof["category"]
            counts[cat] = counts.get(cat, 0) + 1
        for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {cat:32s}: {cnt:2d} dimensions ({cnt/32*100:4.1f}%)")

    return sensitivity_results, classification_results

if __name__ == "__main__":
    cf_npz = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/splits/counterfactual_suite.npz"))
    analyze_counterfactuals(cf_npz)
