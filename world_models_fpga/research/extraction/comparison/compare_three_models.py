"""
Phase 6: Three-Model & Distillation Ablation Comparison (Repaired)

Evaluates:
  1. Original Full H-JEPA
  2. Weight-Preserving Pruned H-JEPA (Option A)
  3. Reconstructed Compact Model — Teacher-Distilled (Option B, 39,804 params)
  4. Reconstructed Compact Model — Supervised Only (No Teacher Ablation)
  5. Reconstructed Compact Model — Behavior Cloning Only (No Latent Dynamics Ablation)
Reports structural parameters, active non-zero parameters, open-loop held-out test metrics,
and closed-loop MuJoCo behavioral navigation performance.
"""

from __future__ import annotations
import sys as _hsys
from pathlib import Path as _HPath
_hsys.path.insert(0, str(_HPath(__file__).resolve().parents[3]))
from research.datasets.dataset_config import SPLITS_DIR as _ACTIVE_SPLITS_DIR, CKPT_DIR as _ACTIVE_CKPT_DIR  # noqa: E402

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn.functional as F

from models.h_jepa import HJEPAModel
from research.benchmarks.evaluator import BenchmarkEvaluator
from research.benchmarks.model_wrappers import BaseAgentWrapper, HJEPAAgent
from research.datasets.generation.capability_tasks import generate_task_specification
from research.extraction.extracted_model.reconstructed_compact.model_reconstructed_compact import (
    ReconstructedCompactModel,
)
from research.extraction.extracted_model.weight_preserving.model_weight_preserving import (
    WeightPreservingExtractedHJEPA,
)


class CompactAgentWrapper(BaseAgentWrapper):
    def __init__(self, name: str, ckpt_path: str, device: str = "cpu"):
        super().__init__(name=name, track="structured", ablation_mode="compact_distilled", device=device)
        self.model = ReconstructedCompactModel(latent_dim=16, stride_k=5, hidden_dim=32).to(device)
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        self.model.eval()

    def act(self, obs_rgb, robot_pose=None, target_pose=None, task_spec=None, goal_vec_body=None):
        return self.model.act(obs_rgb, goal_vec_body=goal_vec_body, device=self.device)


def compare_three_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v2_ckpt = str(_ACTIVE_CKPT_DIR)
    legacy_ckpt = str(_ACTIVE_CKPT_DIR)
    ckpt_dir = v2_ckpt if os.path.exists(os.path.join(v2_ckpt, "hjepa_best.pt")) else legacy_ckpt

    v2_splits = str(_ACTIVE_SPLITS_DIR)
    legacy_splits = str(_ACTIVE_SPLITS_DIR)
    splits_dir = v2_splits if os.path.exists(os.path.join(v2_splits, "held_out_test_benchmark.npz")) else legacy_splits

    out_dir = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    hjepa_ckpt = os.path.join(ckpt_dir, "hjepa_best.pt")
    wp_ckpt = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../extracted_model/weight_preserving/weight_preserving_best.pt")
    )
    rc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../extracted_model/reconstructed_compact"))
    rc_distill_ckpt = os.path.join(rc_dir, "reconstructed_compact_best.pt")
    rc_sup_ckpt = os.path.join(rc_dir, "reconstructed_supervised_only.pt")
    rc_bc_ckpt = os.path.join(rc_dir, "reconstructed_bc_only.pt")

    orig_model = HJEPAModel(stride_k=5).to(device)
    orig_model.load_state_dict(torch.load(hjepa_ckpt, map_location=device, weights_only=False), strict=False)
    orig_model.eval()

    wp_model = WeightPreservingExtractedHJEPA(stride_k=5).to(device)
    if os.path.exists(wp_ckpt):
        wp_model.load_from_original_checkpoint(hjepa_ckpt, device=device)
        wp_model.load_state_dict(torch.load(wp_ckpt, map_location=device, weights_only=False), strict=False)
    else:
        wp_model.load_from_original_checkpoint(hjepa_ckpt, device=device)
    wp_model.eval()

    rc_model = ReconstructedCompactModel(latent_dim=16, stride_k=5, hidden_dim=32).to(device)
    if os.path.exists(rc_distill_ckpt):
        rc_model.load_state_dict(torch.load(rc_distill_ckpt, map_location=device, weights_only=False))
    rc_model.eval()

    te_d = np.load(os.path.join(splits_dir, "held_out_test_benchmark.npz"))
    imgs = torch.from_numpy(te_d["rgb"][:512]).permute(0, 3, 1, 2).float().to(device) / 255.0
    acts = torch.from_numpy(te_d["actions"][:512]).float().to(device)
    cols = torch.from_numpy(te_d["collisions"][:512].astype(np.float32)).to(device)

    def _eval_open_loop(mdl, is_compact=False):
        t0 = time.perf_counter()
        with torch.no_grad():
            if is_compact:
                s = mdl.encode(imgs)
                S_mac, c_log = mdl.forward_macro(s, acts)
                p_act = mdl.select_action(s, S_mac)
            else:
                s = mdl.encode(imgs)
                S_mac, c_log = mdl.macro_predictor(s, acts)
                p_act = mdl.micro_predictor.select_action(s, S_mac)
        dt_ms = (time.perf_counter() - t0) * 1000.0 / len(imgs)
        act_mse = float(F.mse_loss(p_act, acts).item())
        act_cos = float(F.cosine_similarity(p_act, acts, dim=-1).mean().item())
        col_acc = float(((torch.sigmoid(c_log).squeeze(-1) > 0.5).float() == cols).float().mean().item()) * 100.0
        struct_p = sum(p.numel() for p in mdl.parameters())
        nz_p = sum(int((p.detach().abs() > 1e-8).sum().item()) for p in mdl.parameters())
        return {
            "structural_parameters": struct_p,
            "active_nonzero_parameters": nz_p,
            "action_mse": act_mse,
            "action_cosine_similarity": act_cos,
            "collision_accuracy_pct": col_acc,
            "latency_ms_per_frame": dt_ms,
        }

    open_orig = _eval_open_loop(orig_model, is_compact=False)
    open_wp = _eval_open_loop(wp_model, is_compact=False)
    open_rc = _eval_open_loop(rc_model, is_compact=True)

    # Closed-loop MuJoCo comparison across Original, Distilled Compact, Supervised-Only Compact, and BC-Only Compact
    evaluator = BenchmarkEvaluator()
    eval_specs = (
        [generate_task_specification("object_grounding", "easy", seed_idx=5400 + i) for i in range(5)]
        + [generate_task_specification("spatial_reasoning", "easy", seed_idx=5500 + i) for i in range(5)]
        + [generate_task_specification("obstacle_avoidance", "difficult", seed_idx=5600 + i) for i in range(5)]
    )

    closed_loop_agents = {
        "Original_HJEPA": HJEPAAgent(hjepa_ckpt, device=device, track="structured"),
        "Reconstructed_Distilled_16D": CompactAgentWrapper("Reconstructed_Distilled_16D", rc_distill_ckpt, device=device),
        "Reconstructed_Supervised_NoTeacher_16D": CompactAgentWrapper("Reconstructed_Supervised_NoTeacher_16D", rc_sup_ckpt, device=device),
        "Reconstructed_BC_Only_16D": CompactAgentWrapper("Reconstructed_BC_Only_16D", rc_bc_ckpt, device=device),
    }

    closed_loop_results = {}
    for ag_name, ag in closed_loop_agents.items():
        recs = [evaluator.evaluate_episode(ag, sp) for sp in eval_specs]
        closed_loop_results[ag_name] = {
            "success_rate": float(np.mean([1.0 if r["success"] else 0.0 for r in recs])),
            "goal_reached_rate": float(np.mean([1.0 if r["goal_reached"] else 0.0 for r in recs])),
            "collision_rate": float(np.mean([0.0 if r["collision_free"] else 1.0 for r in recs])),
            "path_efficiency": float(np.mean([r["path_efficiency"] for r in recs])),
            "mean_goal_error_m": float(np.mean([r["final_goal_error"] for r in recs])),
        }
        print(
            f"  [Closed-Loop | {ag_name:38s}] Success: {closed_loop_results[ag_name]['success_rate']*100:5.1f}% | "
            f"Col: {closed_loop_results[ag_name]['collision_rate']*100:5.1f}% | "
            f"GoalErr: {closed_loop_results[ag_name]['mean_goal_error_m']:.3f}m"
        )

    evaluator.close()

    comparison = {
        "open_loop_comparison": {
            "Original_HJEPA": open_orig,
            "Weight_Preserving_Pruned": open_wp,
            "Reconstructed_Compact_16D": open_rc,
        },
        "closed_loop_and_distillation_ablations": closed_loop_results,
    }

    out_path = os.path.join(out_dir, "three_model_comparison.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[SAVED THREE-MODEL COMPARISON TO {out_path}]")
    return comparison


if __name__ == "__main__":
    compare_three_models()
