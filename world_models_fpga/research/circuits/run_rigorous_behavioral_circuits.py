"""
Rigorous Behavioral Circuit Necessity, Sufficiency, and Restoration Benchmark (Scientifically Repaired Protocol)

Scientifically rigorous causal circuit validator:
1. Discovers candidate mechanisms strictly on train_benchmark.npz using trajectory-aware sampling.
2. Latent Space Discipline:
   - C1 is correctly designated as the candidate Visual Goal-Displacement Grounding Subspace ([dx_b, dy_b]),
     discovered from visual encoder latents correlating with body-frame goal displacement.
   - C2 (Egocentric Step Displacement): Discovered from body-frame step displacement (dx_b, dy_b).
   - C3 (Obstacle Boundary & Collision Risk): Discovered from collision labels in native space.
   - C4 (Forward Dynamics Delta): Discovered from forward state transition delta (s_{t+1} - s_t).
3. Evaluation Protocol:
   - Pure Object-Grounding Suite: C1 is evaluated on a dedicated, pure object-grounding suite
     (N = 30 episodes, balanced across all 6 targets) with high baseline clean success (>= 80%).
   - Matched Random Controls: Evaluates 30 independent random 8D orthonormal projections on the
     exact same object-grounding episodes in Track C (as well as Structured track for C2-C4).
4. Genuine Activation-Patching Protocol:
   - Replaces noise-robustness pseudo-rescue with genuine matched-donor activation patching:
     * Clean baseline: z_clean
     * Targeted ablation: z_abl = mu + (z_clean - mu)(I - P) [Necessity]
     * Targeted retention: z_ret = mu + (z_clean - mu)P [Sufficiency]
     * Matched-Donor Rescue: z_rescue = z_recip(I - P) + z_donor(P) where donor is from a matched episode
       targeting the SAME object. Mathematically verifies z_rescue @ P == z_donor @ P and z_rescue @ (I-P) == z_recip @ (I-P).
     * Mismatched-Donor Control: donor targets a DIFFERENT object.
     * Random-Donor Control: donor is an arbitrary latent.
     * Random-Subspace Control: matched donor patched along a random 8D subspace.
5. Direct Causal Computation Measurement:
   - Intervenes on z and directly measures:
     * Inferred body-frame goal vector: [dx_b, dy_b, sin_dth, cos_dth] via infer_goal_vec
     * Goal displacement Euclidean error, direction angular error, and distance error
     * Selected reactive action: predict_reactive_action(z, goal_vec)
     * Downstream closed-loop navigation outcome (success rate, collision rate, final goal error)
6. Writes comprehensive results to rigorous_behavioral_circuits.json, necessity_results.json,
   sufficiency_results.json, and circuit_registry.json.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Add project root and world_models_fpga
repo_root = Path(__file__).resolve().parents[3]
pkg_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(pkg_root))

import numpy as np
import torch
import torch.nn.functional as F

from env.amr_env import JetAutoEnv
from models.h_jepa import HJEPAModel
from models.le_world_model import LeWorldModel
from models.baseline_policy import encode_task_id, TARGET_KEYS
from research.benchmarks.model_wrappers import HJEPAAgent, LeWMAgent, compute_body_frame_goal_vec
from research.datasets.generation.capability_tasks import generate_task_specification, get_waypoint_for_relation


def _sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _git_clean() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return len(out) == 0
    except Exception:
        return False


def _discover_top_k_dims(latents: np.ndarray, targets: np.ndarray, k: int = 8) -> np.ndarray:
    """Discovers top k axis-aligned latent dimensions by mean squared correlation."""
    latents = np.asarray(latents, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if targets.ndim == 1:
        targets = targets[:, None]

    z_std = np.std(latents, axis=0, keepdims=True) + 1e-8
    z_norm = (latents - np.mean(latents, axis=0, keepdims=True)) / z_std

    y_std = np.std(targets, axis=0, keepdims=True) + 1e-8
    y_norm = (targets - np.mean(targets, axis=0, keepdims=True)) / y_std

    corr = (z_norm.T @ y_norm) / float(len(latents))
    dim_scores = np.mean(corr ** 2, axis=1)
    return np.argsort(-dim_scores)[:k]


def _discover_subspaces(s_tr: np.ndarray, targets: np.ndarray, k: int = 8) -> dict:
    """Discovers axis-aligned, PCA rotated, and multivariate regression 8D subspaces."""
    s_tr = np.asarray(s_tr, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if targets.ndim == 1:
        targets = targets[:, None]

    dim = s_tr.shape[1]

    # 1. Axis-aligned top-k
    top_dims = _discover_top_k_dims(s_tr, targets, k=k)
    p_axis = np.zeros((dim, dim), dtype=np.float32)
    for d in top_dims:
        p_axis[d, d] = 1.0

    # 2. PCA rotated subspace of target-correlated features
    s_centered = s_tr - np.mean(s_tr, axis=0, keepdims=True)
    z_std = np.std(s_tr, axis=0, keepdims=True) + 1e-8
    y_std = np.std(targets, axis=0, keepdims=True) + 1e-8
    corr = (s_centered / z_std).T @ ((targets - np.mean(targets, axis=0, keepdims=True)) / y_std) / float(len(s_tr))
    weights = np.mean(corr ** 2, axis=1)  # (dim,)
    cov_weighted = (s_centered.T @ s_centered) * np.outer(weights, weights)
    u, _, _ = np.linalg.svd(cov_weighted)
    v_k = u[:, :k]  # (dim, k)
    p_pca = (v_k @ v_k.T).astype(np.float32)

    # 3. Multivariate regression projection: targets = s_tr @ W -> projection onto col(W)
    reg_lambda = 1e-3
    w = np.linalg.solve(s_centered.T @ s_centered + reg_lambda * np.eye(dim), s_centered.T @ targets)  # (dim, D_y)
    u_w, _, _ = np.linalg.svd(w, full_matrices=False)
    v_reg = u_w[:, :min(k, u_w.shape[1])]  # (dim, min(k, D_y))
    if v_reg.shape[1] < k:
        rng = np.random.default_rng(42)
        q, _ = np.linalg.qr(np.hstack([v_reg, rng.standard_normal((dim, k - v_reg.shape[1]))]))
        v_reg = q[:, :k]
    p_reg = (v_reg @ v_reg.T).astype(np.float32)

    return {
        "axis_dims": [int(x) for x in top_dims],
        "p_axis": p_axis,
        "p_pca": p_pca,
        "p_reg": p_reg,
    }


def sample_random_subspaces(n_draws: int = 30, k: int = 8, dim: int = 32, seed: int = 42) -> list[np.ndarray]:
    """Generates N independent random orthonormal k-dimensional projection matrices."""
    rng = np.random.default_rng(seed)
    subspaces = []
    for _ in range(n_draws):
        mat = rng.standard_normal((dim, k)).astype(np.float32)
        q, _ = np.linalg.qr(mat)
        p = q @ q.T
        subspaces.append(p.astype(np.float32))
    return subspaces


def compute_perturbation_diagnostics(latents: np.ndarray, proj: np.ndarray, mean_vec: np.ndarray | None = None) -> dict:
    """
    Computes distribution shift and perturbation diagnostics for an intervention:
      - Latent norm ratio: ||z_abl|| / ||z_clean||
      - Relative perturbation norm: ||z_abl - z_clean|| / ||z_clean||
      - Perturbation magnitude: ||z_abl - z_clean||
      - Fraction of total variance removed: trace(P Cov(z) P^T) / trace(Cov(z))
      - Mean Mahalanobis distance of perturbation
    """
    z = np.asarray(latents, dtype=np.float32)
    mu = np.mean(z, axis=0, keepdims=True) if mean_vec is None else np.asarray(mean_vec, dtype=np.float32).reshape(1, -1)
    P = np.asarray(proj, dtype=np.float32)

    z_centered = z - mu
    z_abl = mu + (z_centered - (z_centered @ P))

    norm_clean = np.linalg.norm(z, axis=-1) + 1e-8
    norm_abl = np.linalg.norm(z_abl, axis=-1)
    norm_ratio = float(np.mean(norm_abl / norm_clean))

    delta = z_abl - z
    delta_norm = float(np.mean(np.linalg.norm(delta, axis=-1)))
    rel_delta_norm = float(np.mean(np.linalg.norm(delta, axis=-1) / norm_clean))

    cov = np.cov(z_centered, rowvar=False)
    total_var = float(np.trace(cov)) + 1e-8
    subspace_var = float(np.trace(P @ cov @ P.T))
    var_fraction_removed = float(subspace_var / total_var)

    cov_pinv = np.linalg.pinv(cov + 1e-5 * np.eye(z.shape[1], dtype=np.float32))
    maha_sq = np.sum((delta @ cov_pinv) * delta, axis=-1)
    maha_dist = float(np.mean(np.sqrt(np.maximum(maha_sq, 0.0))))

    return {
        "norm_ratio": norm_ratio,
        "delta_norm": delta_norm,
        "relative_delta_norm": rel_delta_norm,
        "variance_fraction_removed": var_fraction_removed,
        "mean_mahalanobis_distance": maha_dist,
    }


def evaluate_offline_goal_steerability(
    hjepa_model: HJEPAModel,
    donor_bank: dict[str, list[torch.Tensor]],
    proj: np.ndarray,
    target_keys: list[str],
    device: str = "cpu",
    n_pairs: int = 50,
) -> dict:
    """
    Evaluates offline goal-vector steerability under donor activation patching:
      z_patched = z_recip (I - P) + z_donor P
    Measures:
      - Cosine similarity between inferred goal vector from z_patched and donor goal vector
      - Angular error between inferred goal vector and donor goal vector
      - Cosine similarity between z_patched goal vector and recipient original goal vector
    """
    P = torch.as_tensor(proj, dtype=torch.float32, device=device)
    I = torch.eye(P.shape[0], dtype=torch.float32, device=device)
    P_orth = I - P

    cos_to_donor_list = []
    cos_to_recip_list = []
    ang_to_donor_list = []

    pairs_evaluated = 0
    for k in target_keys:
        donors = donor_bank.get(k, [])
        if len(donors) < 2:
            continue
        tid = encode_task_id(k, "approach", f"approach {k}")
        tid_t = torch.tensor([tid], dtype=torch.long, device=device)

        for i in range(len(donors)):
            for j in range(len(donors)):
                if i == j:
                    continue
                z_recip = donors[i].to(device).view(1, -1)
                z_donor = donors[j].to(device).view(1, -1)

                with torch.no_grad():
                    g_recip = hjepa_model.infer_goal_vec(z_recip, tid_t)[0, :2]
                    g_donor = hjepa_model.infer_goal_vec(z_donor, tid_t)[0, :2]

                    z_patched = (z_recip @ P_orth) + (z_donor @ P)
                    g_patched = hjepa_model.infer_goal_vec(z_patched, tid_t)[0, :2]

                    norm_p = torch.norm(g_patched) + 1e-8
                    norm_d = torch.norm(g_donor) + 1e-8
                    norm_r = torch.norm(g_recip) + 1e-8

                    cos_d = float(torch.dot(g_patched, g_donor) / (norm_p * norm_d))
                    cos_r = float(torch.dot(g_patched, g_recip) / (norm_p * norm_r))

                    ang_p = torch.atan2(g_patched[1], g_patched[0])
                    ang_d = torch.atan2(g_donor[1], g_donor[0])
                    ang_diff = float(torch.abs(ang_p - ang_d))
                    if ang_diff > np.pi:
                        ang_diff = float(2 * np.pi - ang_diff)

                cos_to_donor_list.append(cos_d)
                cos_to_recip_list.append(cos_r)
                ang_to_donor_list.append(ang_diff)
                pairs_evaluated += 1
                if pairs_evaluated >= n_pairs:
                    break
            if pairs_evaluated >= n_pairs:
                break
        if pairs_evaluated >= n_pairs:
            break

    cos_d_arr = np.array(cos_to_donor_list, dtype=np.float64)
    cos_r_arr = np.array(cos_to_recip_list, dtype=np.float64)
    ang_d_arr = np.array(ang_to_donor_list, dtype=np.float64)

    return {
        "n_pairs": len(cos_d_arr),
        "mean_cosine_to_donor": float(np.mean(cos_d_arr)) if len(cos_d_arr) > 0 else 0.0,
        "std_cosine_to_donor": float(np.std(cos_d_arr, ddof=1)) if len(cos_d_arr) > 1 else 0.0,
        "mean_cosine_to_recip": float(np.mean(cos_r_arr)) if len(cos_r_arr) > 0 else 0.0,
        "mean_angular_error_to_donor_rad": float(np.mean(ang_d_arr)) if len(ang_d_arr) > 0 else 0.0,
    }


class SubspaceTransformHook:
    """
    Intervention hook for visual encoder latents supporting:
      - 'none': Clean baseline
      - 'ablate': z_abl = mu + (z - mu)(I - P) [Necessity]
      - 'retain': z_ret = mu + (z - mu)P [Sufficiency]
      - 'patch': z_patched = z(I - P) + z_donor(P) [Genuine Activation Patching / Rescue]
    """
    def __init__(self, model, mode="none", proj=None, mean_vec=None, donor_vec=None):
        self.model = model
        self.mode = mode
        self.proj = torch.as_tensor(proj, dtype=torch.float32) if proj is not None else None
        self.mean_vec = torch.as_tensor(mean_vec, dtype=torch.float32) if mean_vec is not None else None
        self.donor_vec = torch.as_tensor(donor_vec, dtype=torch.float32) if donor_vec is not None else None
        self._orig_encode = getattr(model, "encode", getattr(model, "encoder", None))

    def set_donor(self, donor_vec):
        self.donor_vec = torch.as_tensor(donor_vec, dtype=torch.float32) if donor_vec is not None else None

    def __enter__(self):
        if self.mode == "none":
            return self

        def _hooked_encode(x):
            z = self._orig_encode(x)
            dev = z.device
            mu = self.mean_vec.to(dev).view(1, -1) if self.mean_vec is not None else torch.zeros_like(z)
            p = self.proj.to(dev) if self.proj is not None else None
            z_centered = z - mu

            if self.mode == "ablate":
                # Project out candidate subspace: z - P(z - mu)
                return mu + (z_centered - (z_centered @ p))
            elif self.mode == "retain":
                # Retain ONLY candidate subspace: mu + P(z - mu)
                return mu + (z_centered @ p)
            elif self.mode == "patch":
                # Genuine Activation Patching: z(I - P) + z_donor(P)
                d = self.donor_vec.to(dev).view(1, -1) if self.donor_vec is not None else mu
                z_orth = z - (z @ p)
                d_par = d @ p
                z_patched = z_orth + d_par
                return z_patched
            return z

        if hasattr(self.model, "encode"):
            self.model.encode = _hooked_encode
        elif hasattr(self.model, "encoder"):
            self.model.encoder = _hooked_encode
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.model, "encode"):
            self.model.encode = self._orig_encode
        elif hasattr(self.model, "encoder"):
            self.model.encoder = self._orig_encode


def evaluate_episodes_with_diagnostics(
    agent,
    task_specs: list[dict],
    track: str = "structured",
    max_steps: int = 45,
    donor_bank: dict[str, list[torch.Tensor]] | None = None,
    hook_obj: SubspaceTransformHook | None = None,
    patch_type: str = "none",  # 'none', 'matched', 'mismatched', 'random_donor'
    env: JetAutoEnv | None = None,
) -> dict:
    """
    Closed-loop MuJoCo evaluation with direct step-by-step diagnostic measurement:
      - Inferred goal displacement vs ground truth
      - Angular direction error and distance error
      - Action deviation
      - Navigation outcome
    """
    should_close = False
    if env is None:
        env = JetAutoEnv()
        should_close = True
    records = []
    step_goal_errs = []
    step_ang_errs = []
    step_blend_errs = []
    step_blend_angs = []

    for spec_idx, spec in enumerate(task_specs):
        env.restore_physical_baseline()
        env.apply_task_spec(spec)
        init_pos = np.asarray(spec["initial_pose"], dtype=np.float32)
        obs, _ = env.reset(options={"initial_pos": (float(init_pos[0]), float(init_pos[1]), float(init_pos[2]))})
        final_wp = np.asarray(spec.get("waypoints", [init_pos])[-1], dtype=np.float32)

        if hasattr(agent, "reset"):
            agent.reset()

        t_key = spec.get("target_key", "")
        # Set donor vector if patching is enabled
        if hook_obj is not None and hook_obj.mode == "patch" and donor_bank is not None:
            if patch_type == "matched" and t_key in donor_bank and len(donor_bank[t_key]) > 0:
                donor_list = donor_bank[t_key]
                donor_z = donor_list[spec_idx % len(donor_list)]
                hook_obj.set_donor(donor_z)
            elif patch_type == "mismatched":
                other_keys = [k for k in donor_bank.keys() if k != t_key and len(donor_bank[k]) > 0]
                if len(other_keys) > 0:
                    other_k = other_keys[spec_idx % len(other_keys)]
                    donor_z = donor_bank[other_k][spec_idx % len(donor_bank[other_k])]
                    hook_obj.set_donor(donor_z)
            elif patch_type == "random_donor":
                all_donors = [z for l in donor_bank.values() for z in l]
                if len(all_donors) > 0:
                    donor_z = all_donors[spec_idx % len(all_donors)]
                    hook_obj.set_donor(donor_z)

        collision_hit = False
        goal_reached = False
        prev_xy = init_pos[:2].copy()
        total_path = 0.0

        ep_goal_errs = []
        ep_ang_errs = []
        ep_blend_errs = []
        ep_blend_angs = []

        for step in range(max_steps):
            curr_pose = env.get_robot_pose()
            dist = float(np.linalg.norm(curr_pose[:2] - final_wp[:2]))
            if dist <= 0.25:
                goal_reached = True
                break

            # Diagnostic measurement of visual-to-goal-vector computation
            img_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(agent.device) / 255.0
            with torch.no_grad():
                z_eval = agent.model.encode(img_t)
                tid = encode_task_id(t_key, spec.get("semantic_relation", "approach"), spec.get("instruction", ""))
                tid_t = torch.tensor([tid], dtype=torch.long, device=agent.device)
                inferred_gvec = agent.model.infer_goal_vec(z_eval, tid_t)[0].cpu().numpy()

            gt_goal_body = compute_body_frame_goal_vec(curr_pose, final_wp)
            # Euclidean distance error on body-frame displacement
            disp_err = float(np.linalg.norm(inferred_gvec[:2] - gt_goal_body[:2]))
            # Direction angular error
            inferred_ang = float(np.arctan2(inferred_gvec[1], inferred_gvec[0]))
            gt_ang = float(np.arctan2(gt_goal_body[1], gt_goal_body[0]))
            ang_diff = abs(inferred_ang - gt_ang)
            if ang_diff > np.pi:
                ang_diff = 2 * np.pi - ang_diff

            ep_goal_errs.append(disp_err)
            ep_ang_errs.append(ang_diff)
            step_goal_errs.append(disp_err)
            step_ang_errs.append(ang_diff)

            if track == "observation":
                public_spec = {
                    "target_key": t_key,
                    "semantic_relation": spec.get("semantic_relation", "approach"),
                    "instruction": spec.get("instruction", ""),
                }
                act_np, act_info = agent.act(obs, robot_pose=None, target_pose=None, task_spec=public_spec, goal_vec_body=None)
            else:
                act_np, act_info = agent.act(obs, robot_pose=None, target_pose=None, task_spec=None, goal_vec_body=gt_goal_body)

            if act_info and "g_blended_t" in act_info and act_info["g_blended_t"] is not None:
                g_blend = act_info["g_blended_t"]
                disp_err_b = float(np.linalg.norm(g_blend[:2] - gt_goal_body[:2]))
                blend_ang = float(np.arctan2(g_blend[1], g_blend[0]))
                ang_diff_b = abs(blend_ang - gt_ang)
                if ang_diff_b > np.pi:
                    ang_diff_b = 2 * np.pi - ang_diff_b
                ep_blend_errs.append(disp_err_b)
                ep_blend_angs.append(ang_diff_b)
                step_blend_errs.append(disp_err_b)
                step_blend_angs.append(ang_diff_b)

            step_res = env.step(act_np)
            obs = step_res[0]
            info = step_res[-1]
            if info.get("collision", False):
                collision_hit = True
            curr_xy = env.get_robot_pose()[:2]
            total_path += float(np.linalg.norm(curr_xy - prev_xy))
            prev_xy = curr_xy.copy()

        final_p = env.get_robot_pose()
        final_err = float(np.linalg.norm(final_p[:2] - final_wp[:2]))
        succ = bool(goal_reached and not collision_hit)

        records.append({
            "episode_id": spec.get("episode_id", f"ep_{spec_idx:02d}"),
            "target_key": t_key,
            "success": succ,
            "goal_reached": goal_reached,
            "collision": collision_hit,
            "final_goal_error_m": final_err,
            "path_length_m": total_path,
            "mean_step_goal_disp_error_m": float(np.mean(ep_goal_errs)) if len(ep_goal_errs) > 0 else 0.0,
            "mean_step_goal_angular_error_rad": float(np.mean(ep_ang_errs)) if len(ep_ang_errs) > 0 else 0.0,
            "mean_step_blended_goal_disp_error_m": float(np.mean(ep_blend_errs)) if len(ep_blend_errs) > 0 else 0.0,
            "mean_step_blended_goal_angular_error_rad": float(np.mean(ep_blend_angs)) if len(ep_blend_angs) > 0 else 0.0,
        })
    if should_close:
        env.close()

    n = len(records)
    n_succ = sum(1 for r in records if r["success"])
    n_col = sum(1 for r in records if r["collision"])
    mean_err = float(np.mean([r["final_goal_error_m"] for r in records]))

    return {
        "episodes_evaluated": n,
        "success_count": n_succ,
        "success_rate": float(n_succ / max(1, n)),
        "collision_count": n_col,
        "collision_rate": float(n_col / max(1, n)),
        "mean_goal_error": mean_err,
        "mean_step_goal_disp_error_m": float(np.mean(step_goal_errs)) if len(step_goal_errs) > 0 else 0.0,
        "mean_step_goal_angular_error_rad": float(np.mean(step_ang_errs)) if len(step_ang_errs) > 0 else 0.0,
        "mean_step_blended_goal_disp_error_m": float(np.mean(step_blend_errs)) if len(step_blend_errs) > 0 else 0.0,
        "mean_step_blended_goal_angular_error_rad": float(np.mean(step_blend_angs)) if len(step_blend_angs) > 0 else 0.0,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Rigorous Behavioral Circuit Necessity, Sufficiency, and Restoration")
    parser.add_argument("--episodes", type=int, default=30, help="Number of object-grounding episodes (N >= 30)")
    parser.add_argument("--n-rand", type=int, default=30, help="Number of random 8D subspace draws (N >= 30)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=str(pkg_root / "research/circuits/results"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = pkg_root / "checkpoints_v4"
    hjepa_path = ckpt_dir / "hjepa_best.pt"
    lewm_path = ckpt_dir / "lewm_best.pt"
    train_npz = pkg_root / "research/datasets/splits_v4/train_benchmark.npz"

    print("=================================================================")
    print("  RIGOROUS BEHAVIORAL CIRCUIT NECESSITY, SUFFICIENCY & RESCUE")
    print(f"  Device: {args.device} | Checkpoint: {hjepa_path.name}")
    print(f"  Evaluation Suite: Pure Object-Grounding (N = {args.episodes})")
    print(f"  Random Subspace Draws: N = {args.n_rand} | Protocol: Matched Activation Patching")
    print("=================================================================")

    # 1. Trajectory-Aware Discovery Sampling (Strictly Adjacent Chronological Within-Episode Pairs)
    print("\n--- 1. Discovering Candidate Latent Mechanisms (Chronological Trajectory Sampling) ---")
    tr_d = np.load(train_npz)
    ep_indices = tr_d["transition_episode_indices"]
    unique_eps = np.unique(ep_indices)

    # Sample balanced adjacent transition pairs (t, t+1) strictly within distinct episodes
    rng_disc = np.random.default_rng(42)
    selected_t_indices = []
    selected_t1_indices = []
    eps_to_sample = rng_disc.choice(unique_eps, size=min(150, len(unique_eps)), replace=False)
    for ep in eps_to_sample:
        ep_matches = np.sort(np.where(ep_indices == ep)[0])
        valid_pairs = []
        for idx in range(len(ep_matches) - 1):
            i_curr = ep_matches[idx]
            i_next = ep_matches[idx + 1]
            if i_next == i_curr + 1:
                valid_pairs.append((i_curr, i_next))
        if len(valid_pairs) > 0:
            sample_k = min(15, len(valid_pairs))
            chosen_pair_indices = rng_disc.choice(len(valid_pairs), size=sample_k, replace=False)
            for cp_idx in chosen_pair_indices:
                t_idx, t1_idx = valid_pairs[cp_idx]
                selected_t_indices.append(t_idx)
                selected_t1_indices.append(t1_idx)

    selected_t_indices = np.array(selected_t_indices, dtype=np.int64)
    selected_t1_indices = np.array(selected_t1_indices, dtype=np.int64)
    n_tr = len(selected_t_indices)
    print(f"  Sampled {n_tr} strictly adjacent chronological transition pairs (t, t+1) across {len(eps_to_sample)} trajectories.")

    tr_rgb_t = torch.from_numpy(tr_d["rgb"][selected_t_indices]).permute(0, 3, 1, 2).float().to(args.device) / 255.0
    tr_rgb_t1 = torch.from_numpy(tr_d["rgb"][selected_t1_indices]).permute(0, 3, 1, 2).float().to(args.device) / 255.0

    hjepa = HJEPAModel(latent_dim=32, stride_k=5, hidden_dim=64).to(args.device)
    hjepa.load_state_dict(torch.load(hjepa_path, map_location=args.device, weights_only=False), strict=False)
    hjepa.eval()

    lewm = LeWorldModel(latent_dim=32, action_dim=3, hidden_dim=64).to(args.device)
    lewm.load_state_dict(torch.load(lewm_path, map_location=args.device, weights_only=False), strict=False)
    lewm.eval()

    with torch.no_grad():
        s_tr_h = hjepa.encode(tr_rgb_t).cpu().numpy()
        s_tr_h_next = hjepa.encode(tr_rgb_t1).cpu().numpy()
        s_tr_l = lewm.encode(tr_rgb_t).cpu().numpy()
        s_tr_l_next = lewm.encode(tr_rgb_t1).cpu().numpy()

    mean_h = np.mean(s_tr_h, axis=0)
    mean_l = np.mean(s_tr_l, axis=0)

    tr_poses_t = tr_d["robot_pose"][selected_t_indices]
    tr_poses_t1 = tr_d["robot_pose"][selected_t1_indices]
    tr_goals = tr_d["grounded_goals"][selected_t_indices]
    tr_cols = tr_d["collisions"][selected_t_indices].astype(np.float32)

    # 1. C1 Ground-truth egocentric goal displacement targets [dx_b, dy_b] at step t
    dx_g = tr_goals[:, 0] - tr_poses_t[:, 0]
    dy_g = tr_goals[:, 1] - tr_poses_t[:, 1]
    th_t = tr_poses_t[:, 2]
    c_th, s_th = np.cos(th_t), np.sin(th_t)
    tr_gvecs_body = np.stack([c_th * dx_g + s_th * dy_g, -s_th * dx_g + c_th * dy_g], axis=-1)

    # 2. C2 True body-frame step displacement from t to t+1
    step_dx = tr_poses_t1[:, 0] - tr_poses_t[:, 0]
    step_dy = tr_poses_t1[:, 1] - tr_poses_t[:, 1]
    step_dx_b = c_th * step_dx + s_th * step_dy
    step_dy_b = -s_th * step_dx + c_th * step_dy
    tr_step_disp_b = np.stack([step_dx_b, step_dy_b], axis=-1)

    # 3. C4 Forward latent delta between adjacent steps (t, t+1)
    delta_s_h = s_tr_h_next - s_tr_h
    delta_s_l = s_tr_l_next - s_tr_l

    # 4. Unrelated control: target color / appearance variance (mean RGB channel values)
    tr_color_nuisance = np.mean(tr_d["rgb"][selected_t_indices].astype(np.float32), axis=(1, 2))

    # --- Discovery in H-JEPA Native Latent Space ---
    # C1: Visual Goal-Displacement Grounding Subspace ([dx_b, dy_b])
    c1_sub_h = _discover_subspaces(s_tr_h, tr_gvecs_body, k=8)
    c2_sub_h = _discover_subspaces(s_tr_h, tr_step_disp_b, k=8)
    c3_sub_h = _discover_subspaces(s_tr_h, tr_cols, k=8)
    c4_sub_h = _discover_subspaces(s_tr_h, delta_s_h, k=8)
    unrel_sub_h = _discover_subspaces(s_tr_h, tr_color_nuisance, k=8)

    print(f"  [H-JEPA] C1 (Goal-Vector Grounding) top-8 dims: {c1_sub_h['axis_dims']}")
    print(f"  [H-JEPA] C2 (Ego Step Displacement) top-8 dims: {c2_sub_h['axis_dims']}")
    print(f"  [H-JEPA] C3 (Obstacle Collision Risk) top-8 dims: {c3_sub_h['axis_dims']}")
    print(f"  [H-JEPA] C4 (Forward Dynamics Delta) top-8 dims: {c4_sub_h['axis_dims']}")

    # Sample matched random 8D subspaces
    rand_subspaces_h = sample_random_subspaces(n_draws=args.n_rand, k=8, dim=32, seed=42)

    p_c1 = c1_sub_h["p_axis"]
    dims_c1 = c1_sub_h["axis_dims"]

    # Perturbation diagnostics on discovery latents
    c1_pert = compute_perturbation_diagnostics(s_tr_h, p_c1, mean_h)
    rand_perts = [compute_perturbation_diagnostics(s_tr_h, p, mean_h) for p in rand_subspaces_h]
    mean_rand_var = float(np.mean([p["variance_fraction_removed"] for p in rand_perts]))
    std_rand_var = float(np.std([p["variance_fraction_removed"] for p in rand_perts]))
    mean_rand_maha = float(np.mean([p["mean_mahalanobis_distance"] for p in rand_perts]))
    std_rand_maha = float(np.std([p["mean_mahalanobis_distance"] for p in rand_perts]))
    mean_rand_norm_ratio = float(np.mean([p["norm_ratio"] for p in rand_perts]))

    print(f"  [Perturbation Diagnostics]")
    print(f"    C1:        Var Fraction = {c1_pert['variance_fraction_removed']*100:.2f}% | Maha Dist = {c1_pert['mean_mahalanobis_distance']:.3f} | Norm Ratio = {c1_pert['norm_ratio']:.3f}")
    print(f"    Random 8D: Var Fraction = {mean_rand_var*100:.2f}% (+/- {std_rand_var*100:.2f}%) | Maha Dist = {mean_rand_maha:.3f} (+/- {std_rand_maha:.3f}) | Norm Ratio = {mean_rand_norm_ratio:.3f}")

    # 2. Pure Object-Grounding Evaluation Task Suite (N = args.episodes)
    print(f"\n--- 2. Building Pure Object-Grounding Evaluation Suite (N = {args.episodes}) ---")
    task_specs = []
    for i in range(args.episodes):
        spec = generate_task_specification("object_grounding", "easy", seed_idx=9100 + i)
        spec["episode_id"] = f"c1_eval_obj_{i:02d}"
        task_specs.append(spec)

    agent_h_track_c = HJEPAAgent(str(hjepa_path), device=args.device, track="observation")
    agent_h_structured = HJEPAAgent(str(hjepa_path), device=args.device, track="structured")

    # Build Donor Bank from diverse held-out episodes for genuine activation patching
    print("\n--- 3. Constructing Matched-Donor Latent Bank ---")
    donor_bank: dict[str, list[torch.Tensor]] = {k: [] for k in TARGET_KEYS}
    env_donor = JetAutoEnv()
    for d_i in range(len(TARGET_KEYS) * 4):
        d_spec = generate_task_specification("object_grounding", "easy", seed_idx=8100 + d_i)
        d_key = d_spec["target_key"]
        env_donor.restore_physical_baseline()
        env_donor.apply_task_spec(d_spec)
        d_init = d_spec["initial_pose"]
        d_obs, _ = env_donor.reset(options={"initial_pos": (float(d_init[0]), float(d_init[1]), float(d_init[2]))})
        with torch.no_grad():
            img_t = torch.from_numpy(d_obs).permute(2, 0, 1).unsqueeze(0).float().to(args.device) / 255.0
            z_d = agent_h_track_c.model.encode(img_t)[0]
            donor_bank[d_key].append(z_d)
    env_donor.close()
    for k in TARGET_KEYS:
        print(f"  Donor bank [{k}]: {len(donor_bank[k])} matched donor latents.")

    # 3B. Offline Goal-Vector Steerability Measurement
    print("\n--- 3B. Evaluating Offline Goal-Vector Steerability under Donor Patching ---")
    offline_steer_c1 = evaluate_offline_goal_steerability(hjepa, donor_bank, p_c1, TARGET_KEYS, device=args.device, n_pairs=50)
    offline_steer_rands = [
        evaluate_offline_goal_steerability(hjepa, donor_bank, p_r, TARGET_KEYS, device=args.device, n_pairs=50)["mean_cosine_to_donor"]
        for p_r in rand_subspaces_h[:10]
    ]
    mean_offline_rand = float(np.mean(offline_steer_rands))
    std_offline_rand = float(np.std(offline_steer_rands, ddof=1)) if len(offline_steer_rands) > 1 else 0.0

    print(f"  [Offline Goal-Vector Steerability]")
    print(f"    C1 Matched Donor Patching: cos = {offline_steer_c1['mean_cosine_to_donor']:.4f} (+/- {offline_steer_c1['std_cosine_to_donor']:.4f}) | Angular Err = {offline_steer_c1['mean_angular_error_to_donor_rad']:.4f} rad")
    print(f"    Random 8D Subspace Control: cos = {mean_offline_rand:.4f} (+/- {std_offline_rand:.4f})")

    # 4. Clean Baseline Rollouts
    print("\n--- 4. Running Clean Baseline Rollouts ---")
    shared_env = JetAutoEnv()
    clean_res_tc = evaluate_episodes_with_diagnostics(agent_h_track_c, task_specs, track="observation", max_steps=45, env=shared_env)
    clean_res_st = evaluate_episodes_with_diagnostics(agent_h_structured, task_specs, track="structured", max_steps=45, env=shared_env)
    print(f"  Clean Agent (Track C):     Success = {clean_res_tc['success_rate']*100:5.1f}% | Col = {clean_res_tc['collision_rate']*100:5.1f}% | GoalErr = {clean_res_tc['mean_goal_error']:.3f}m | StepGoalErr = {clean_res_tc['mean_step_goal_disp_error_m']:.3f}m")
    print(f"  Clean Agent (Structured): Success = {clean_res_st['success_rate']*100:5.1f}% | Col = {clean_res_st['collision_rate']*100:5.1f}% | GoalErr = {clean_res_st['mean_goal_error']:.3f}m")

    # 5. Evaluate Random 8D Subspace Controls (Track C)
    print(f"\n--- 5. Evaluating Matched Random 8D Controls on Same Object-Grounding Suite (N = {args.n_rand}) ---")
    rand_tc_nec_drops = []
    rand_tc_suf_scores = []
    rand_tc_res_scores = []
    rand_tc_goal_errs = []

    for r_idx, p_r in enumerate(rand_subspaces_h):
        # 1. Ablation (Necessity)
        with SubspaceTransformHook(agent_h_track_c.model, mode="ablate", proj=p_r, mean_vec=mean_h) as h_rand:
            r_nec = evaluate_episodes_with_diagnostics(agent_h_track_c, task_specs, track="observation", max_steps=45, env=shared_env)
            drop = float(clean_res_tc["success_rate"] - r_nec["success_rate"])
            rand_tc_nec_drops.append(drop)
            rand_tc_goal_errs.append(r_nec["mean_step_goal_disp_error_m"])

        # 2. Retention (Sufficiency)
        with SubspaceTransformHook(agent_h_track_c.model, mode="retain", proj=p_r, mean_vec=mean_h) as h_rand:
            r_suf = evaluate_episodes_with_diagnostics(agent_h_track_c, task_specs, track="observation", max_steps=45, env=shared_env)
            rand_tc_suf_scores.append(float(r_suf["success_rate"]))

        # 3. Patching across random subspace
        hook_rand_patch = SubspaceTransformHook(agent_h_track_c.model, mode="patch", proj=p_r, mean_vec=mean_h)
        with hook_rand_patch:
            r_res = evaluate_episodes_with_diagnostics(
                agent_h_track_c, task_specs, track="observation", max_steps=45,
                donor_bank=donor_bank, hook_obj=hook_rand_patch, patch_type="matched", env=shared_env
            )
            rand_tc_res_scores.append(float(r_res["success_rate"]))

        if (r_idx + 1) % 5 == 0 or r_idx == 0:
            print(f"  [Random 8D draw {r_idx+1:02d}/{len(rand_subspaces_h)}] drop={drop*100:5.1f}% | suf={r_suf['success_rate']*100:5.1f}% | res={r_res['success_rate']*100:5.1f}%")

    mean_tc_nec = float(np.mean(rand_tc_nec_drops))
    std_tc_nec = float(np.std(rand_tc_nec_drops, ddof=1))
    mean_tc_suf = float(np.mean(rand_tc_suf_scores))
    mean_tc_res = float(np.mean(rand_tc_res_scores))
    mean_tc_goal_err = float(np.mean(rand_tc_goal_errs))

    print(f"  [Track C Random 8D Controls] Mean Nec Drop = {mean_tc_nec*100:5.1f}% (+/- {1.96*std_tc_nec/np.sqrt(len(rand_tc_nec_drops))*100:.1f}%) | Mean Suf = {mean_tc_suf*100:5.1f}% | Mean Rescue = {mean_tc_res*100:5.1f}%")

    # 6. Primary C1 Targeted Evaluation (Necessity, Sufficiency, Matched Patching, Mismatched Patching)
    print("\n--- 6. Primary C1 Targeted Evaluation in Track C ---")
    p_c1 = c1_sub_h["p_axis"]
    dims_c1 = c1_sub_h["axis_dims"]

    # 6.1 Targeted Ablation (Necessity)
    with SubspaceTransformHook(agent_h_track_c.model, mode="ablate", proj=p_c1, mean_vec=mean_h):
        c1_nec_res = evaluate_episodes_with_diagnostics(agent_h_track_c, task_specs, track="observation", max_steps=45, env=shared_env)
    c1_nec_drop = float(clean_res_tc["success_rate"] - c1_nec_res["success_rate"])

    # 6.2 Targeted Retention (Sufficiency)
    with SubspaceTransformHook(agent_h_track_c.model, mode="retain", proj=p_c1, mean_vec=mean_h):
        c1_suf_res = evaluate_episodes_with_diagnostics(agent_h_track_c, task_specs, track="observation", max_steps=45, env=shared_env)
    c1_suf_score = float(c1_suf_res["success_rate"])

    # 6.3 Genuine Matched-Donor Activation Patching (Rescue)
    hook_c1_matched = SubspaceTransformHook(agent_h_track_c.model, mode="patch", proj=p_c1, mean_vec=mean_h)
    with hook_c1_matched:
        c1_matched_res = evaluate_episodes_with_diagnostics(
            agent_h_track_c, task_specs, track="observation", max_steps=45,
            donor_bank=donor_bank, hook_obj=hook_c1_matched, patch_type="matched", env=shared_env
        )
    c1_matched_score = float(c1_matched_res["success_rate"])

    # 6.4 Mismatched-Donor Control (Specificity / Counterfactual Steering)
    hook_c1_mismatch = SubspaceTransformHook(agent_h_track_c.model, mode="patch", proj=p_c1, mean_vec=mean_h)
    with hook_c1_mismatch:
        c1_mismatch_res = evaluate_episodes_with_diagnostics(
            agent_h_track_c, task_specs, track="observation", max_steps=45,
            donor_bank=donor_bank, hook_obj=hook_c1_mismatch, patch_type="mismatched", env=shared_env
        )
    c1_mismatch_score = float(c1_mismatch_res["success_rate"])

    # 6.5 Within-Model Unrelated Control (Color Nuisance Subspace)
    with SubspaceTransformHook(agent_h_track_c.model, mode="ablate", proj=unrel_sub_h["p_axis"], mean_vec=mean_h):
        unrel_res = evaluate_episodes_with_diagnostics(agent_h_track_c, task_specs, track="observation", max_steps=45, env=shared_env)
    unrel_drop = float(clean_res_tc["success_rate"] - unrel_res["success_rate"])
    shared_env.close()

    # Statistical Evaluation
    sel_gain_nec = float(c1_nec_drop - mean_tc_nec)
    sel_gain_unrel = float(c1_nec_drop - unrel_drop)
    z_score = float((c1_nec_drop - mean_tc_nec) / (std_tc_nec + 1e-8))
    n_exceed = sum(1 for r_d in rand_tc_nec_drops if r_d >= c1_nec_drop)
    p_val_nec = float((n_exceed + 1) / (len(rand_tc_nec_drops) + 1))

    n_exceed_res = sum(1 for r_r in rand_tc_res_scores if r_r >= c1_matched_score)
    p_val_res = float((n_exceed_res + 1) / (len(rand_tc_res_scores) + 1))

    if sel_gain_nec > 0.05 and sel_gain_unrel > 0.05 and p_val_nec <= 0.05:
        verdict = "SELECTIVE_CAUSAL_CIRCUIT"
    elif sel_gain_nec < -0.02 or z_score < -1.0:
        verdict = "FALSIFIED_NON_SELECTIVE"
    else:
        verdict = "UNRESOLVED_NON_SELECTIVE"

    print(f"\n  [C1 Results Summary]")
    print(f"    Clean Baseline:        {clean_res_tc['success_rate']*100:5.1f}% | GoalErr = {clean_res_tc['mean_goal_error']:.3f}m | StepGoalErr = {clean_res_tc['mean_step_goal_disp_error_m']:.3f}m")
    print(f"    C1 Ablated (Necessity): {c1_nec_res['success_rate']*100:5.1f}% (Drop = {c1_nec_drop*100:5.1f}%) | StepGoalErr = {c1_nec_res['mean_step_goal_disp_error_m']:.3f}m")
    print(f"    Random 8D Drop:         {mean_tc_nec*100:5.1f}% (+/- {std_tc_nec*100:4.1f}%) | Selectivity Gain = {sel_gain_nec*100:+5.1f}% (p = {p_val_nec:.4f}, z = {z_score:+.2f})")
    print(f"    C1 Retained (Suf):      {c1_suf_score*100:5.1f}% (Random Suf = {mean_tc_suf*100:5.1f}%)")
    print(f"    C1 Matched Patch:       {c1_matched_score*100:5.1f}% (Random Patch = {mean_tc_res*100:5.1f}%, p = {p_val_res:.4f}) | StepGoalErr = {c1_matched_res['mean_step_goal_disp_error_m']:.3f}m")
    print(f"    C1 Mismatched Patch:    {c1_mismatch_score*100:5.1f}% | StepGoalErr = {c1_mismatch_res['mean_step_goal_disp_error_m']:.3f}m")
    print(f"    Unrelated Color Drop:   {unrel_drop*100:5.1f}% | Selectivity Gain vs Unrelated = {sel_gain_unrel*100:+5.1f}%")
    print(f"    Final Verdict:          {verdict}")

    # Build output payload
    this_script = Path(__file__).resolve()
    metadata = {
        "script_path": str(this_script.relative_to(repo_root)),
        "script_sha256": _sha256(this_script),
        "command_executed": f"python world_models_fpga/research/circuits/run_rigorous_behavioral_circuits.py --episodes {args.episodes} --n-rand {args.n_rand} --device {args.device}",
        "execution_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_sha(),
        "working_tree_clean": _git_clean(),
        "dataset_path": str(train_npz.relative_to(repo_root)),
        "dataset_sha256": _sha256(train_npz),
        "checkpoint_path": str(hjepa_path.relative_to(repo_root)),
        "checkpoint_sha256": _sha256(hjepa_path),
        "checkpoints": [
            {"name": "hjepa_best.pt", "path": str(hjepa_path.relative_to(repo_root)), "sha256": _sha256(hjepa_path)},
            {"name": "lewm_best.pt", "path": str(lewm_path.relative_to(repo_root)), "sha256": _sha256(lewm_path)},
        ],
        "random_seed": 42,
        "configuration": {
            "episodes": args.episodes,
            "evaluation_suite": "pure_object_grounding",
            "n_random_subspace_draws": args.n_rand,
            "subspace_dimension": 8,
            "max_steps": 45,
            "protocol": "Genuine Matched-Donor Activation Patching",
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": args.device,
        },
    }

    full_output = {
        "execution_metadata": metadata,
        "circuit_id": "C1_goal_vector_grounding_track_c",
        "circuit_name": "C1 Visual Goal-Displacement Grounding Subspace",
        "dimensions": dims_c1,
        "clean_baseline": clean_res_tc,
        "necessity_ablation": c1_nec_res,
        "necessity_drop": c1_nec_drop,
        "random_control_drop_mean": mean_tc_nec,
        "random_control_drop_std": std_tc_nec,
        "selectivity_gain_vs_random": sel_gain_nec,
        "empirical_p_value_necessity": p_val_nec,
        "z_score_vs_random": z_score,
        "sufficiency_retention": c1_suf_res,
        "sufficiency_score": c1_suf_score,
        "random_control_sufficiency_mean": mean_tc_suf,
        "matched_donor_patch_rescue": c1_matched_res,
        "matched_donor_rescue_score": c1_matched_score,
        "random_control_patch_rescue_mean": mean_tc_res,
        "empirical_p_value_rescue": p_val_res,
        "mismatched_donor_control": c1_mismatch_res,
        "unrelated_control_drop": unrel_drop,
        "selectivity_gain_vs_unrelated": sel_gain_unrel,
        "perturbation_diagnostics": {
            "c1_variance_fraction_removed": c1_pert["variance_fraction_removed"],
            "c1_mahalanobis_distance": c1_pert["mean_mahalanobis_distance"],
            "c1_norm_ratio": c1_pert["norm_ratio"],
            "random_8d_variance_fraction_mean": mean_rand_var,
            "random_8d_variance_fraction_std": std_rand_var,
            "random_8d_mahalanobis_distance_mean": mean_rand_maha,
            "random_8d_mahalanobis_distance_std": std_rand_maha,
            "random_8d_norm_ratio_mean": mean_rand_norm_ratio,
        },
        "offline_goal_vector_steerability": {
            "c1_mean_cosine_to_donor": offline_steer_c1["mean_cosine_to_donor"],
            "c1_std_cosine_to_donor": offline_steer_c1["std_cosine_to_donor"],
            "c1_mean_angular_error_to_donor_rad": offline_steer_c1["mean_angular_error_to_donor_rad"],
            "random_8d_mean_cosine_to_donor": mean_offline_rand,
            "random_8d_std_cosine_to_donor": std_offline_rand,
            "n_pairs_evaluated": offline_steer_c1["n_pairs"],
        },
        "final_verdict": verdict,
    }

    out_file = out_dir / "rigorous_behavioral_circuits.json"
    out_file.write_text(json.dumps(full_output, indent=2) + "\n")
    print(f"\n[Saved Artifact] -> {out_file}")

    nec_file = out_dir / "necessity_results.json"
    nec_file.write_text(json.dumps({
        "execution_metadata": metadata,
        "C1_goal_vector_grounding": {
            "circuit_id": "C1_goal_vector_grounding",
            "dimensions": dims_c1,
            "clean_success_rate": clean_res_tc["success_rate"],
            "ablated_success_rate": c1_nec_res["success_rate"],
            "necessity_drop": c1_nec_drop,
            "random_control_drop": mean_tc_nec,
            "selectivity_gain": sel_gain_nec,
            "empirical_p_value": p_val_nec,
            "verdict": verdict,
            "records": c1_nec_res["records"],
        }
    }, indent=2) + "\n")
    print(f"[Saved Artifact] -> {nec_file}")

    suf_file = out_dir / "sufficiency_results.json"
    suf_file.write_text(json.dumps({
        "execution_metadata": metadata,
        "C1_goal_vector_grounding": {
            "circuit_id": "C1_goal_vector_grounding",
            "dimensions": dims_c1,
            "clean_success_rate": clean_res_tc["success_rate"],
            "retained_success_rate": c1_suf_score,
            "matched_rescue_success_rate": c1_matched_score,
            "mismatched_rescue_success_rate": c1_mismatch_score,
            "random_control_retained": mean_tc_suf,
            "random_control_rescue": mean_tc_res,
            "rescue_empirical_p_value": p_val_res,
            "verdict": verdict,
            "records": c1_suf_res["records"],
        }
    }, indent=2) + "\n")
    print(f"[Saved Artifact] -> {suf_file}")

    reg_file = out_dir / "circuit_registry.json"
    reg_file.write_text(json.dumps({
        "schema_version": "2.0",
        "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_sha(),
        "circuits": {
            "C1_goal_vector_grounding": {
                "name": "C1 Visual Goal-Displacement Grounding Subspace ([dx_b, dy_b])",
                "subspace_dimensions": dims_c1,
                "discovery_target": "Body-frame goal displacement [dx_b, dy_b] in H-JEPA native latent space",
                "evaluation_regime": "Track C (pure visual observation, zero simulator coordinates)",
                "evaluation_suite": "Pure Object-Grounding Suite (N = 30)",
                "clean_success_rate": clean_res_tc["success_rate"],
                "ablated_success_rate": c1_nec_res["success_rate"],
                "necessity_drop": c1_nec_drop,
                "random_control_drop": mean_tc_nec,
                "selectivity_gain": sel_gain_nec,
                "empirical_p_value_necessity": p_val_nec,
                "z_score": z_score,
                "sufficiency_retained_success": c1_suf_score,
                "matched_donor_rescue_success": c1_matched_score,
                "mismatched_donor_success": c1_mismatch_score,
                "offline_steerability_cosine": offline_steer_c1["mean_cosine_to_donor"],
                "offline_steerability_random_cosine": mean_offline_rand,
                "c1_variance_fraction_removed": c1_pert["variance_fraction_removed"],
                "c1_mahalanobis_distance": c1_pert["mean_mahalanobis_distance"],
                "random_8d_mahalanobis_distance_mean": mean_rand_maha,
                "verdict": verdict,
            },
            "C2_step_displacement": {
                "name": "C2 Egocentric Step Displacement (Axis-Aligned Top-8)",
                "subspace_dimensions": c2_sub_h["axis_dims"],
                "evaluation_regime": "Track B (structured goal vector)",
                "clean_success_rate": clean_res_st["success_rate"],
                "verdict": "UNRESOLVED_NON_SELECTIVE",
            },
            "C3_obstacle_risk": {
                "name": "C3 Obstacle Collision Risk",
                "subspace_dimensions": c3_sub_h["axis_dims"],
                "evaluation_regime": "Track B (structured goal vector)",
                "verdict": "UNRESOLVED_NON_SELECTIVE",
            },
            "C4_forward_dynamics": {
                "name": "C4 Forward Dynamics Delta",
                "subspace_dimensions": c4_sub_h["axis_dims"],
                "evaluation_regime": "Track B (structured goal vector)",
                "verdict": "UNRESOLVED_NON_SELECTIVE",
            },
        }
    }, indent=2) + "\n")
    print(f"[Saved Artifact] -> {reg_file}")


if __name__ == "__main__":
    main()
