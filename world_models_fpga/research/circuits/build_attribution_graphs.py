"""
Phase 4: Mechanistic Attribution Graph Builder (Repaired)

Fixes:
  1. Eliminates all hardcoded edge weights (0.84, 0.72, 0.91, etc.) and hardcoded registry tables.
  2. Actually invokes `compute_gradient_attribution` (gradient x activation E[|dY/dh * h|])
     across layers of Baseline, LeWM, and H-JEPA on held-out test transitions.
  3. Delegates `circuit_registry.json` generation to `test_necessity_sufficiency()` so there
     is a single authoritative, empirically measured circuit registry.
"""

from __future__ import annotations
import sys as _hsys
from pathlib import Path as _HPath
_hsys.path.insert(0, str(_HPath(__file__).resolve().parents[2]))
from research.datasets.dataset_config import SPLITS_DIR as _ACTIVE_SPLITS_DIR, CKPT_DIR as _ACTIVE_CKPT_DIR  # noqa: E402

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from models.baseline_policy import BaselineNavigationPolicy
from models.h_jepa import HJEPAModel
from models.le_world_model import LeWorldModel
from research.circuits.test_necessity_sufficiency import test_necessity_sufficiency


def compute_gradient_attribution(model, loss_scalar: torch.Tensor, layer_activations: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """
    Computes empirical gradient-times-activation attribution:
      Attr(h_i) = E_batch[ | (dY / dh_i) * h_i | ]
    """
    attributions = {}
    for layer_name, act_tensor in layer_activations.items():
        if not act_tensor.requires_grad:
            continue
        grad = torch.autograd.grad(
            outputs=loss_scalar,
            inputs=act_tensor,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        if grad is not None:
            attr = torch.abs(grad * act_tensor)
            if attr.dim() == 4:
                attr_score = attr.mean(dim=(0, 2, 3)).detach().cpu().numpy()
            else:
                attr_score = attr.mean(dim=0).detach().cpu().numpy()
            attributions[layer_name] = attr_score
        else:
            attributions[layer_name] = np.zeros(act_tensor.shape[1] if act_tensor.dim() > 1 else 1, dtype=np.float32)
    return attributions


def _make_hook(d, k):
    def hook(m, i, o):
        o.retain_grad()
        d[k] = o
        return None
    return hook


def _norm_weight(val: float, scale: float) -> float:
    return float(np.round(np.clip(val / (scale + 1e-8), 0.01, 1.0), 4))


def build_attribution_graphs(output_dir: str | None = None):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "attribution_graphs"))
    os.makedirs(output_dir, exist_ok=True)
    res_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
    os.makedirs(res_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    v2_splits = str(_ACTIVE_SPLITS_DIR)
    legacy_splits = str(_ACTIVE_SPLITS_DIR)
    splits_dir = v2_splits if os.path.exists(os.path.join(v2_splits, "held_out_test_benchmark.npz")) else legacy_splits

    v2_ckpt = str(_ACTIVE_CKPT_DIR)
    legacy_ckpt = str(_ACTIVE_CKPT_DIR)
    ckpt_dir = v2_ckpt if os.path.exists(os.path.join(v2_ckpt, "lewm_best.pt")) else legacy_ckpt

    print("\n=======================================================")
    print("  PHASE 4: MECHANISTIC ATTRIBUTION GRAPH CONSTRUCTION")
    print(f"  Output Directory: {output_dir}")
    print("=======================================================\n")

    te_d = np.load(os.path.join(splits_dir, "held_out_test_benchmark.npz"))
    n = min(256, len(te_d["rgb"]))
    imgs = torch.from_numpy(te_d["rgb"][:n]).permute(0, 3, 1, 2).float().to(device) / 255.0
    acts = torch.from_numpy(te_d["actions"][:n]).float().to(device)
    poses = te_d["robot_pose"][:n]
    goals = te_d["grounded_goals"][:n]
    dx, dy = goals[:, 0] - poses[:, 0], goals[:, 1] - poses[:, 1]
    th = poses[:, 2]
    c, s = np.cos(th), np.sin(th)
    dth = goals[:, 2] - poses[:, 2]
    g_vecs = torch.from_numpy(
        np.stack([c * dx + s * dy, -s * dx + c * dy, np.sin(dth), np.cos(dth)], axis=-1).astype(np.float32)
    ).to(device)

    # 1. Baseline Policy Gradient x Activation Attribution
    b_model = BaselineNavigationPolicy().to(device)
    b_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "baseline_best.pt"), map_location=device, weights_only=False), strict=False)
    b_model.eval()
    b_acts = {}
    b_hooks = [
        b_model.encoder.conv2.register_forward_hook(_make_hook(b_acts, "conv2")),
        b_model.encoder.conv4.register_forward_hook(_make_hook(b_acts, "conv4")),
        b_model.fc1.register_forward_hook(_make_hook(b_acts, "fc1")),
        b_model.fc2.register_forward_hook(_make_hook(b_acts, "fc2")),
    ]
    imgs_b = imgs.clone().requires_grad_(True)
    g_b = g_vecs.clone().requires_grad_(True)
    z_b = b_model.encoder(imgs_b)
    z_b.retain_grad()
    b_acts["z_vis"] = z_b
    (pred_act_b, pred_col_b) = b_model.forward_with_latent(z_b, g_b)
    b_acts["goal_vec"] = g_b
    target_b = torch.norm(pred_act_b, dim=-1).mean() + pred_col_b.mean()
    attr_b = compute_gradient_attribution(b_model, target_b, b_acts)
    for h in b_hooks:
        h.remove()

    # 2. LeWorldModel Gradient x Activation Attribution
    l_model = LeWorldModel().to(device)
    l_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "lewm_best.pt"), map_location=device, weights_only=False), strict=False)
    l_model.eval()
    l_acts = {}
    l_hooks = [
        l_model.pred_fc1.register_forward_hook(_make_hook(l_acts, "pred_fc1")),
        l_model.pred_fc2.register_forward_hook(_make_hook(l_acts, "pred_fc2")),
    ]
    imgs_l = imgs.clone().requires_grad_(True)
    a_l = acts.clone().requires_grad_(True)
    s_l = l_model.encode(imgs_l)
    s_l.retain_grad()
    l_acts["s_t"] = s_l
    l_acts["a_t"] = a_l
    ns_l, col_l = l_model.predict_next(s_l, a_l)
    ns_l.retain_grad()
    l_acts["next_s"] = ns_l
    disp_l = l_model.predict_ego_displacement(s_l, ns_l)
    target_l = torch.norm(ns_l - s_l, dim=-1).mean() + torch.norm(disp_l, dim=-1).mean() + torch.sigmoid(col_l).mean()
    attr_l = compute_gradient_attribution(l_model, target_l, l_acts)
    for h in l_hooks:
        h.remove()

    # 3. H-JEPA Gradient x Activation Attribution
    h_model = HJEPAModel().to(device)
    h_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "hjepa_best.pt"), map_location=device, weights_only=False), strict=False)
    h_model.eval()
    h_acts = {}
    h_hooks = [
        h_model.macro_predictor.fc1.register_forward_hook(_make_hook(h_acts, "macro_fc1")),
        h_model.micro_predictor.pol_fc1.register_forward_hook(_make_hook(h_acts, "micro_pol_fc1")),
        h_model.micro_predictor.dyn_fc1.register_forward_hook(_make_hook(h_acts, "micro_dyn_fc1")),
    ]
    imgs_h = imgs.clone().requires_grad_(True)
    a_h = acts.clone().requires_grad_(True)
    s_h = h_model.encode(imgs_h)
    s_h.retain_grad()
    h_acts["s_t"] = s_h
    h_acts["a_t"] = a_h
    S_mac_h, col_mac_h = h_model.macro_predictor(s_h, a_h)
    S_mac_h.retain_grad()
    h_acts["S_macro"] = S_mac_h
    act_h = h_model.micro_predictor.select_action(s_h, S_mac_h)
    ns_h, col_mic_h = h_model.micro_predictor.forward_transition(s_h, act_h)
    target_h = (
        torch.norm(act_h, dim=-1).mean()
        + torch.norm(S_mac_h - s_h, dim=-1).mean()
        + torch.sigmoid(col_mac_h).mean()
        + torch.norm(ns_h - s_h, dim=-1).mean()
        + torch.sigmoid(col_mic_h).mean()
    )
    attr_h = compute_gradient_attribution(h_model, target_h, h_acts)
    for h in h_hooks:
        h.remove()

    b_scale = float(max(np.mean(v) for v in attr_b.values()))
    l_scale = float(max(np.mean(v) for v in attr_l.values()))
    h_scale = float(max(np.mean(v) for v in attr_h.values()))

    graphs = {
        "Baseline": {
            "nodes": [
                {"id": "encoder.conv2", "type": "conv_feature", "mean_attribution": float(np.mean(attr_b["conv2"]))},
                {"id": "encoder.conv4", "type": "conv_feature", "mean_attribution": float(np.mean(attr_b["conv4"]))},
                {"id": "z_vis", "type": "latent_state", "mean_attribution": float(np.mean(attr_b["z_vis"]))},
                {"id": "goal_vec", "type": "structured_goal_input", "mean_attribution": float(np.mean(attr_b["goal_vec"]))},
                {"id": "fc1", "type": "mlp_hidden", "mean_attribution": float(np.mean(attr_b["fc1"]))},
                {"id": "fc2", "type": "mlp_hidden", "mean_attribution": float(np.mean(attr_b["fc2"]))},
                {"id": "action_out", "type": "output", "top_latent_dims": np.argsort(-attr_b["z_vis"])[:8].tolist()},
            ],
            "edges": [
                {"source": "encoder.conv2", "target": "encoder.conv4", "weight": _norm_weight(float(np.mean(attr_b["conv2"])), b_scale)},
                {"source": "encoder.conv4", "target": "z_vis", "weight": _norm_weight(float(np.mean(attr_b["conv4"])), b_scale)},
                {"source": "z_vis", "target": "fc1", "weight": _norm_weight(float(np.mean(attr_b["z_vis"])), b_scale)},
                {"source": "goal_vec", "target": "fc1", "weight": _norm_weight(float(np.mean(attr_b["goal_vec"])), b_scale)},
                {"source": "fc1", "target": "fc2", "weight": _norm_weight(float(np.mean(attr_b["fc1"])), b_scale)},
                {"source": "fc2", "target": "action_out", "weight": _norm_weight(float(np.mean(attr_b["fc2"])), b_scale)},
            ],
        },
        "LeWM": {
            "nodes": [
                {"id": "s_t", "type": "latent_state", "mean_attribution": float(np.mean(attr_l["s_t"])), "top_dims": np.argsort(-attr_l["s_t"])[:8].tolist()},
                {"id": "a_t", "type": "action_input", "mean_attribution": float(np.mean(attr_l["a_t"]))},
                {"id": "pred_fc1", "type": "transition_hidden", "mean_attribution": float(np.mean(attr_l["pred_fc1"]))},
                {"id": "pred_fc2", "type": "transition_hidden", "mean_attribution": float(np.mean(attr_l["pred_fc2"]))},
                {"id": "next_s", "type": "predicted_latent", "mean_attribution": float(np.mean(attr_l["next_s"]))},
            ],
            "edges": [
                {"source": "s_t", "target": "pred_fc1", "weight": _norm_weight(float(np.mean(attr_l["s_t"])), l_scale)},
                {"source": "a_t", "target": "pred_fc1", "weight": _norm_weight(float(np.mean(attr_l["a_t"])), l_scale)},
                {"source": "pred_fc1", "target": "pred_fc2", "weight": _norm_weight(float(np.mean(attr_l["pred_fc1"])), l_scale)},
                {"source": "pred_fc2", "target": "next_s", "weight": _norm_weight(float(np.mean(attr_l["pred_fc2"])), l_scale)},
            ],
        },
        "H-JEPA": {
            "nodes": [
                {"id": "s_t", "type": "latent_state", "mean_attribution": float(np.mean(attr_h["s_t"])), "top_dims": np.argsort(-attr_h["s_t"])[:8].tolist()},
                {"id": "a_t", "type": "macro_action_input", "mean_attribution": float(np.mean(attr_h["a_t"]))},
                {"id": "macro_fc1", "type": "macro_hidden", "mean_attribution": float(np.mean(attr_h["macro_fc1"]))},
                {"id": "S_macro", "type": "macro_subgoal", "mean_attribution": float(np.mean(attr_h["S_macro"]))},
                {"id": "micro_pol_fc1", "type": "micro_policy_hidden", "mean_attribution": float(np.mean(attr_h["micro_pol_fc1"]))},
                {"id": "micro_dyn_fc1", "type": "micro_dynamics_hidden", "mean_attribution": float(np.mean(attr_h["micro_dyn_fc1"]))},
            ],
            "edges": [
                {"source": "s_t", "target": "macro_fc1", "weight": _norm_weight(float(np.mean(attr_h["s_t"])), h_scale)},
                {"source": "a_t", "target": "macro_fc1", "weight": _norm_weight(float(np.mean(attr_h["a_t"])), h_scale)},
                {"source": "macro_fc1", "target": "S_macro", "weight": _norm_weight(float(np.mean(attr_h["macro_fc1"])), h_scale)},
                {"source": "S_macro", "target": "micro_pol_fc1", "weight": _norm_weight(float(np.mean(attr_h["S_macro"])), h_scale)},
                {"source": "s_t", "target": "micro_pol_fc1", "weight": _norm_weight(float(np.mean(attr_h["s_t"])), h_scale)},
                {"source": "micro_pol_fc1", "target": "micro_dyn_fc1", "weight": _norm_weight(float(np.mean(attr_h["micro_pol_fc1"])), h_scale)},
            ],
        },
    }

    graph_path_1 = os.path.join(output_dir, "attribution_graphs.json")
    graph_path_2 = os.path.join(res_dir, "attribution_graphs.json")
    with open(graph_path_1, "w") as f:
        json.dump(graphs, f, indent=2)
    with open(graph_path_2, "w") as f:
        json.dump(graphs, f, indent=2)

    registry = test_necessity_sufficiency()
    print(f"Saved empirical attribution graphs to {graph_path_1} and {graph_path_2}")
    return graphs, registry


if __name__ == "__main__":
    build_attribution_graphs()
