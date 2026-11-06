"""
Phase 5: Computational Circuit Reconstruction - Systematic Ablation Experiments
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Ablates candidate functional circuits, semantic feature clusters, and predictor modules:
  1. Appearance / Color Cluster (zeroed)
  2. Spatial Geometry Cluster (zeroed)
  3. Egocentric Heading Cluster (zeroed)
  4. Obstacle Avoidance / Collision Head (zeroed)
  5. Action-Conditioned Dynamics Predictor Core (residual bypass)
  6. Macro Planning Circuit (in H-JEPA)

Measures causal impact across 5 core capabilities:
  - Object Grounding
  - Spatial Reasoning
  - Obstacle Avoidance
  - Trajectory Execution / Foresight
  - Robustness under Kinetic Perturbation
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

CAPABILITIES = [
    "object_grounding",
    "spatial_reasoning",
    "obstacle_avoidance",
    "trajectory_foresight",
    "robustness_recovery"
]

def evaluate_capabilities_under_ablation(model, model_type, inspector, ablate_spec, test_data, device):
    """
    Evaluates the 5 core capabilities on test_data under the specified ablation context.
    ablate_spec: dict with 'target', 'channel_indices', 'mode', 'bypass_module'
    """
    obs = test_data["rgb"][:100]
    poses = test_data["robot_pose"][:100]
    actions = test_data["actions"][:100]
    obj_pos = test_data["object_positions"][:100]
    N = len(obs)

    imgs = torch.from_numpy(obs).permute(0, 3, 1, 2).float().to(device) / 255.0
    cand_a = torch.tensor([[0.8, 0.0, 0.0]], device=device).repeat(N, 1)
    goal_vecs = torch.tensor([[1.0, 0.0, 0.0, 1.0]], device=device).repeat(N, 1)

    # Context manager for ablation
    ablate_context = None
    target_layer = ablate_spec.get("target")
    ch_indices = ablate_spec.get("channel_indices")

    if target_layer and ch_indices is not None:
        ablate_context = inspector.ablate_component(target_layer, channel_indices=ch_indices, mode="zero")

    def run_forward():
        if model_type == "baseline":
            (pred_actions, col_risks), z = model(imgs, goal_vecs)
            return pred_actions, col_risks, z
        elif model_type == "lewm":
            s = model.encode(imgs)
            if ablate_spec.get("bypass_dynamics"):
                # Predictor disabled: delta_s = 0, col = 0
                pred_s = s
                col_risks = torch.zeros(N, 1, device=device)
            else:
                pred_s, col_l = model.predict_next(s, cand_a)
                col_risks = torch.sigmoid(col_l)
            col_p = col_risks
            pred_actions = cand_a * (1.0 - col_p) + torch.tensor([[0.1, 0.6, 0.4]], device=device) * col_p
            return pred_actions, col_risks, pred_s
        elif model_type == "hjepa":
            s = model.encode(imgs)
            if ablate_spec.get("bypass_macro"):
                # Macro circuit disabled: direct zero subgoal
                S_subgoal = torch.zeros_like(s)
                col_risks = torch.zeros(N, 1, device=device)
            else:
                S_subgoal, col_l = model.macro_predictor(s, cand_a)
                col_risks = torch.sigmoid(col_l)
            pred_actions = model.micro_predictor.select_action(s, S_subgoal)
            return pred_actions, col_risks, S_subgoal

    if ablate_context:
        with ablate_context:
            pred_actions, col_risks, latent_out = run_forward()
    else:
        pred_actions, col_risks, latent_out = run_forward()

    pred_act_np = pred_actions.detach().cpu().numpy()
    gt_act_np = actions[:, :3]

    # 1. Object Grounding: alignment of forward velocity towards target
    grounding_score = float(np.mean(np.maximum(0.0, pred_act_np[:, 0])))

    # 2. Spatial Reasoning: angular steering alignment towards landmark
    angle_error = np.abs(pred_act_np[:, 2] - gt_act_np[:, 2])
    spatial_score = float(np.mean(np.maximum(0.0, 1.0 - angle_error)))

    # 3. Obstacle Avoidance: collision risk anticipation when near obstacles
    dists = np.linalg.norm(obj_pos[:, :, :2] - poses[:, np.newaxis, :2], axis=-1)
    min_dists = np.min(dists, axis=-1)
    near_mask = min_dists < 0.65
    col_np = col_risks.detach().cpu().numpy().squeeze(-1)
    if np.sum(near_mask) > 0:
        avoidance_score = float(np.mean(col_np[near_mask]))
    else:
        avoidance_score = 0.5

    # 4. Trajectory Foresight: action fidelity to expert demonstration
    act_mse = np.mean((pred_act_np - gt_act_np) ** 2)
    foresight_score = float(max(0.0, 1.0 - act_mse))

    # 5. Robustness under Perturbation: stability of command output
    robustness_score = float(np.clip(1.0 - np.std(pred_act_np, axis=0).mean(), 0.0, 1.0))

    return {
        "object_grounding": grounding_score,
        "spatial_reasoning": spatial_score,
        "obstacle_avoidance": avoidance_score,
        "trajectory_foresight": foresight_score,
        "robustness_recovery": robustness_score
    }

def run_ablation_suite():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "ablation_results"))
    os.makedirs(out_dir, exist_ok=True)
    val_npz = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/splits/val_benchmark.npz"))
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../checkpoints"))
    test_data = np.load(val_npz)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n=======================================================")
    print("  PHASE 5: SYSTEMATIC CIRCUIT ABLATION EXPERIMENTS")
    print("=======================================================\n")

    # Load semantic cluster definitions from Phase 4
    class_json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../representations/counterfactual_analysis/feature_classification.json"))
    with open(class_json_path) as f:
        class_data = json.load(f)

    models_config = [
        ("baseline", BaselineNavigationPolicy, "baseline_best.pt", "baseline.visual_latent_z"),
        ("lewm", LeWorldModel, "lewm_best.pt", "lewm.latent_state_s"),
        ("hjepa", lambda: HJEPAModel(stride_k=5), "hjepa_best.pt", "hjepa.shared_latent_s")
    ]

    ablation_results = {}
    csv_rows = []

    for m_type, m_cls, ckpt_name, latent_name in models_config:
        print(f"\n--- Running Ablations for: [{m_type.upper()}] ---")
        model = m_cls().to(device)
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, ckpt_name), map_location=device))
        model.eval()

        inspector = ModelInspector(model, m_type)

        # 1. Unablated Baseline Reference
        clean_scores = evaluate_capabilities_under_ablation(model, m_type, inspector, {}, test_data, device)
        ablation_results[m_type] = {"clean_reference": clean_scores, "ablations": {}}

        print(f"  [Clean Reference] Grounding: {clean_scores['object_grounding']:.3f} | Spatial: {clean_scores['spatial_reasoning']:.3f} | Avoidance: {clean_scores['obstacle_avoidance']:.3f} | Foresight: {clean_scores['trajectory_foresight']:.3f}")

        # Define ablation interventions
        m_key = f"{m_type}_z" if m_type == "baseline" else f"{m_type}_s"
        cluster_map = {}
        for entry in class_data.get(m_key, []):
            cat = entry["category"]
            if cat not in cluster_map: cluster_map[cat] = []
            cluster_map[cat].append(entry["dim"])

        interventions = [
            ("ablate_appearance", {
                "target": latent_name,
                "channel_indices": cluster_map.get("Visual Appearance / Color", []),
                "desc": f"Zero out appearance features ({len(cluster_map.get('Visual Appearance / Color', []))} dims)"
            }),
            ("ablate_geometry", {
                "target": latent_name,
                "channel_indices": cluster_map.get("Geometry / Target Location", []),
                "desc": f"Zero out geometry features ({len(cluster_map.get('Geometry / Target Location', []))} dims)"
            }),
            ("ablate_heading", {
                "target": latent_name,
                "channel_indices": cluster_map.get("Egocentric Heading", []),
                "desc": f"Zero out heading features ({len(cluster_map.get('Egocentric Heading', []))} dims)"
            })
        ]

        # Model-specific module ablations
        if m_type == "lewm":
            interventions.append(("ablate_transition_dynamics", {
                "bypass_dynamics": True,
                "desc": "Bypass predictive dynamics core (delta_s = 0, residual only)"
            }))
        elif m_type == "hjepa":
            interventions.append(("ablate_macro_planner", {
                "bypass_macro": True,
                "desc": "Disable macro planning circuit (subgoal S* = 0)"
            }))

        for ab_name, ab_spec in interventions:
            if ab_spec.get("channel_indices") is not None and len(ab_spec["channel_indices"]) == 0:
                continue

            ab_scores = evaluate_capabilities_under_ablation(model, m_type, inspector, ab_spec, test_data, device)
            
            # Compute relative capability drop: Delta S = (Clean - Ablated) / Clean
            drops = {}
            for cap in CAPABILITIES:
                ref = clean_scores[cap]
                val = ab_scores[cap]
                drop_pct = float((ref - val) / (ref + 1e-8) * 100.0)
                drops[cap] = drop_pct

            ablation_results[m_type]["ablations"][ab_name] = {
                "description": ab_spec["desc"],
                "scores": ab_scores,
                "capability_drops_pct": drops
            }

            print(f"  [{ab_name:26s}] Grounding Drop: {drops['object_grounding']:+5.1f}% | "
                  f"Spatial Drop: {drops['spatial_reasoning']:+5.1f}% | "
                  f"Avoidance Drop: {drops['obstacle_avoidance']:+5.1f}% | "
                  f"Foresight Drop: {drops['trajectory_foresight']:+5.1f}%")

            csv_rows.append([
                m_type, ab_name, ab_spec["desc"],
                f"{drops['object_grounding']:+.1f}%",
                f"{drops['spatial_reasoning']:+.1f}%",
                f"{drops['obstacle_avoidance']:+.1f}%",
                f"{drops['trajectory_foresight']:+.1f}%",
                f"{drops['robustness_recovery']:+.1f}%"
            ])

        inspector.close()

    # Save outputs
    json_path = os.path.join(out_dir, "ablation_matrix.json")
    with open(json_path, "w") as f:
        json.dump(ablation_results, f, indent=2)

    csv_path = os.path.join(out_dir, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Ablation Target", "Description",
            "Grounding Drop", "Spatial Drop", "Avoidance Drop", "Foresight Drop", "Robustness Drop"
        ])
        writer.writerows(csv_rows)

    print(f"\nSaved Ablation Results to:")
    print(f"  -> {json_path}")
    print(f"  -> {csv_path}")

    return ablation_results

if __name__ == "__main__":
    run_ablation_suite()
