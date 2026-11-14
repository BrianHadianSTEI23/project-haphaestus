"""
Phase 2: Unified Agent Evaluation Wrappers (Repaired for Strict Information Regimes)

Enforces three distinct evaluation tracks:
  - Track A ('oracle'): Privileged simulator state (robot_pose, target_pose, obstacles) via ExpertNavigator.
  - Track B ('structured'): Camera RGB observation + egocentric body-frame relative goal vector g_body in R^4.
    Zero world poses, zero obstacle coordinates, zero analytical kinematics.
  - Track C ('observation'): Camera RGB observation + discrete task ID (target_key, semantic_relation) only.
    Zero simulator coordinates of any kind.

Also supports controlled causal ablations:
  - 'prediction_used' (default full world-model latent rollout)
  - 'reactive' (learned reactive policy head without world-model rollout)
  - 'prediction_ignored' (world model rollout executed, but next_s replaced by s_t in scorer)
  - 'no_action_conditioning' (action input to transition predictor zeroed out)
  - 'no_collision_prediction' (collision risk weight set to 0)
  - 'no_temporal_hierarchy' (H-JEPA bypasses K-step MacroPredictor and uses 1-step MicroPredictor only)
"""

from __future__ import annotations
import sys as _hsys
from pathlib import Path as _HPath
_hsys.path.insert(0, str(_HPath(__file__).resolve().parents[2]))
from research.datasets.dataset_config import SPLITS_DIR as _ACTIVE_SPLITS_DIR, CKPT_DIR as _ACTIVE_CKPT_DIR  # noqa: E402

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from models.baseline_policy import BaselineNavigationPolicy, encode_task_id
from models.h_jepa import HJEPAModel
from models.le_world_model import LeWorldModel
from research.datasets.generation.expert_navigator import ExpertNavigator


def _default_ckpt_path(filename: str) -> str:
    # No silent fallback to legacy checkpoints trained on leaky/stale data.
    return os.path.join(str(_ACTIVE_CKPT_DIR), filename)


def _load_state_dict_flexible(model: torch.nn.Module, ckpt_path: str | None, device: str, default_file: str | None = None) -> bool:
    if not ckpt_path and default_file:
        ckpt_path = _default_ckpt_path(default_file)
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"checkpoint {ckpt_path!r} not found; refusing to evaluate an untrained model as if it were trained")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    model.load_state_dict(payload, strict=False)
    return True


def compute_body_frame_goal_vec(robot_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    """Computes purely egocentric relative goal vector [dx_b, dy_b, sin(dth), cos(dth)]."""
    dx = float(target_pose[0] - robot_pose[0])
    dy = float(target_pose[1] - robot_pose[1])
    th = float(robot_pose[2])
    dx_b = float(np.cos(th) * dx + np.sin(th) * dy)
    dy_b = float(-np.sin(th) * dx + np.cos(th) * dy)
    dth = float(target_pose[2] - robot_pose[2]) if len(target_pose) > 2 else 0.0
    return np.array([dx_b, dy_b, np.sin(dth), np.cos(dth)], dtype=np.float32)


def _build_holonomic_candidates(a_react: torch.Tensor, device: str) -> torch.Tensor:
    """
    Constructs 25 deterministic holonomic 3-DOF action candidates in [-1, 1]^3:
      - Candidate 0: learned reactive proposal a_react
      - Candidates 1..8: local perturbations around a_react
      - Candidates 9..24: structured holonomic basis (forward, strafe, diagonal detour, yaw)
    """
    ar = a_react.squeeze(0)
    deltas = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [-0.25, 0.0, 0.0],
            [0.0, 0.45, 0.0],
            [0.0, -0.45, 0.0],
            [0.15, 0.40, 0.20],
            [0.15, -0.40, -0.20],
            [0.0, 0.0, 0.45],
            [0.0, 0.0, -0.45],
        ],
        dtype=torch.float32,
        device=device,
    )
    local_cands = torch.clamp(ar.unsqueeze(0) + deltas, -1.0, 1.0)
    fixed_cands = torch.tensor(
        [
            [0.85, 0.0, 0.0],
            [0.55, 0.0, 0.0],
            [0.65, 0.55, 0.0],
            [0.65, -0.55, 0.0],
            [0.40, 0.75, 0.15],
            [0.40, -0.75, -0.15],
            [0.0, 0.80, 0.0],
            [0.0, -0.80, 0.0],
            [-0.35, 0.65, 0.25],
            [-0.35, -0.65, -0.25],
            [0.70, 0.0, 0.55],
            [0.70, 0.0, -0.55],
            [0.35, 0.35, 0.65],
            [0.35, -0.35, -0.65],
            [0.0, 0.0, 0.85],
            [0.0, 0.0, -0.85],
        ],
        dtype=torch.float32,
        device=device,
    )
    return torch.cat([local_cands, fixed_cands], dim=0)


class BaseAgentWrapper:
    def __init__(self, name: str, track: str = "structured", ablation_mode: str = "prediction_used", device: str = "cpu"):
        self.name = name
        self.model_id = name
        self.track = track
        self.ablation_mode = ablation_mode
        self.device = device
        self._prev_latent = None
        self._ego_goal_memory = None
        self.last_goal_diag = {}
        # Fine-grained branch interceptor hooks for mechanistic circuit isolation
        self.disable_memory = False
        self.branch_interceptors = {
            "task_id": None,          # Callable[[int], int] | None
            "task_token": None,       # Callable[[torch.Tensor], torch.Tensor] | None
            "task_embedding": None,   # Callable[[torch.Tensor], torch.Tensor] | None
            "grounding_latent": None, # Callable[[torch.Tensor], torch.Tensor] | None
            "visual_goal": None,      # Callable[[torch.Tensor], torch.Tensor] | None
            "ego_disp": None,         # Callable[[torch.Tensor], torch.Tensor] | None
            "memory_prev": None,      # Callable[[torch.Tensor], torch.Tensor] | None
            "memory_updated": None,   # Callable[[torch.Tensor], torch.Tensor] | None
            "blended_goal": None,     # Callable[[torch.Tensor], torch.Tensor] | None
            "planner_goal": None,     # Callable[[torch.Tensor], torch.Tensor] | None
        }

    def set_branch_interceptor(self, branch_name: str, fn):
        if branch_name not in self.branch_interceptors:
            raise KeyError(f"Unknown branch {branch_name!r}. Supported branches: {list(self.branch_interceptors.keys())}")
        self.branch_interceptors[branch_name] = fn

    def clear_branch_interceptors(self):
        for k in self.branch_interceptors:
            self.branch_interceptors[k] = None

    def reset(self):
        self._prev_latent = None
        self._ego_goal_memory = None
        self.last_goal_diag = {}

    def act(
        self,
        obs_rgb: np.ndarray,
        robot_pose: np.ndarray | None = None,
        target_pose: np.ndarray | None = None,
        task_spec: dict | None = None,
        goal_vec_body: np.ndarray | None = None,
    ):
        raise NotImplementedError


class OracleAgent(BaseAgentWrapper):
    """Track A Upper Bound: Privileged Geometric Planner with full simulator state."""

    def __init__(self):
        super().__init__(name="Oracle_Expert", track="oracle", ablation_mode="oracle_controller")
        self.nav = ExpertNavigator()

    def act(self, obs_rgb, robot_pose=None, target_pose=None, task_spec=None, goal_vec_body=None):
        if robot_pose is None or target_pose is None:
            raise ValueError("OracleAgent requires privileged robot_pose and target_pose (Track A only).")
        action = self.nav.compute_action(robot_pose, target_pose, task_spec or {}, is_last_waypoint=True)
        return action, {"col_risk": 0.0, "latent": np.zeros(32, dtype=np.float32), "track": "oracle"}


class RandomAgent(BaseAgentWrapper):
    """Uninformed Lower Bound: Seeded random holonomic walk."""

    def __init__(self, seed: int = 42):
        super().__init__(name="Random_Policy", track="observation", ablation_mode="random")
        self.rng = np.random.default_rng(seed)

    def act(self, obs_rgb, robot_pose=None, target_pose=None, task_spec=None, goal_vec_body=None):
        act = self.rng.uniform(-0.8, 0.8, size=3).astype(np.float32)
        return act, {"col_risk": 0.5, "latent": np.zeros(32, dtype=np.float32), "track": self.track}


class BaselineAgent(BaseAgentWrapper):
    """Reactive CNN Policy without latent transition model."""

    def __init__(self, ckpt_path: str | None = None, device: str = "cpu", track: str = "structured"):
        super().__init__(name="CNN_Reactive_Baseline", track=track, ablation_mode="reactive", device=device)
        self.model = BaselineNavigationPolicy(latent_dim=32, goal_dim=4, action_dim=3, hidden_dim=64).to(device)
        _load_state_dict_flexible(self.model, ckpt_path, device, default_file="baseline_best.pt")
        self.model.eval()

    @torch.no_grad()
    def act(self, obs_rgb, robot_pose=None, target_pose=None, task_spec=None, goal_vec_body=None):
        img_t = torch.from_numpy(obs_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        z_vis = self.model.encode(img_t) if hasattr(self.model, "encode") else self.model.encoder(img_t)

        if self.track == "observation":
            spec = task_spec or {}
            raw_tid = encode_task_id(spec.get("target_key", ""), spec.get("semantic_relation", "approach"), spec.get("instruction", ""))
            tid = raw_tid
            if self.branch_interceptors.get("task_id") is not None:
                tid = self.branch_interceptors["task_id"](tid)
            t_tensor = torch.tensor([tid], dtype=torch.long, device=self.device)
            if self.branch_interceptors.get("task_token") is not None:
                t_tensor = self.branch_interceptors["task_token"](t_tensor)

            s_ground = z_vis
            if self.branch_interceptors.get("grounding_latent") is not None:
                s_ground = self.branch_interceptors["grounding_latent"](s_ground)

            t_emb = self.model.task_embedding(t_tensor)
            if self.branch_interceptors.get("task_embedding") is not None:
                t_emb = self.branch_interceptors["task_embedding"](t_emb)

            raw = self.model.grounding_head(torch.cat([s_ground, t_emb], dim=-1))
            dx_dy = torch.clamp(raw[:, :2], -3.5, 3.5)
            trig = torch.nn.functional.normalize(raw[:, 2:4] + 1e-6, dim=-1)
            goal_t = torch.cat([dx_dy, trig], dim=-1)
            if self.branch_interceptors.get("visual_goal") is not None:
                goal_t = self.branch_interceptors["visual_goal"](goal_t)
            self._last_vis_goal = goal_t
            if self.branch_interceptors.get("blended_goal") is not None:
                bg_fn = self.branch_interceptors["blended_goal"]
                try:
                    goal_t = bg_fn(goal_t, goal_t, None)
                except TypeError:
                    goal_t = bg_fn(goal_t)
            self.last_goal_diag = {
                "raw_task_id": raw_tid,
                "used_task_id": tid,
                "task_embedding": t_emb.squeeze(0).detach().cpu().numpy(),
                "grounding_latent": s_ground.squeeze(0).detach().cpu().numpy(),
                "g_visual_t": goal_t.squeeze(0).detach().cpu().numpy(),
                "g_blended_t": goal_t.squeeze(0).detach().cpu().numpy(),
                "active_interceptors": [k for k, v in self.branch_interceptors.items() if v is not None],
            }
        else:
            if goal_vec_body is None:
                if robot_pose is None or target_pose is None:
                    raise ValueError("Track B requires goal_vec_body (or relative pose to compute it).")
                goal_vec_body = compute_body_frame_goal_vec(robot_pose, target_pose)
            goal_t = torch.as_tensor(goal_vec_body, dtype=torch.float32, device=self.device).view(1, 4)
            if self.branch_interceptors.get("blended_goal") is not None:
                bg_fn = self.branch_interceptors["blended_goal"]
                try:
                    goal_t = bg_fn(goal_t, None, None)
                except TypeError:
                    goal_t = bg_fn(goal_t)

        actions, col_risk = self.model.forward_with_latent(z_vis, goal_t)
        act_np = np.clip(actions.squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
        col_np = float(col_risk.squeeze(0).cpu().numpy()[0])
        z_np = z_vis.squeeze(0).cpu().numpy()
        return act_np, {"col_risk": col_np, "latent": z_np, "track": self.track, "ablation_mode": self.ablation_mode}


class LeWMAgent(BaseAgentWrapper):
    r"""
    Single-Scale Action-Conditioned World Model (LeWM).
    Selects actions using predicted future latent states \hat{s}_{t+1} = g(s_t, a),
    decoded latent ego-motion displacement, and predicted collision risk.
    """

    def __init__(
        self,
        ckpt_path: str | None = None,
        device: str = "cpu",
        track: str = "structured",
        ablation_mode: str = "prediction_used",
    ):
        super().__init__(name="LeWorldModel_SingleScale", track=track, ablation_mode=ablation_mode, device=device)
        self.model = LeWorldModel(latent_dim=32, action_dim=3, hidden_dim=64).to(device)
        _load_state_dict_flexible(self.model, ckpt_path, device, default_file="lewm_best.pt")
        self.model.eval()

    def _resolve_goal_vec(self, s_t: torch.Tensor, robot_pose, target_pose, task_spec, goal_vec_body) -> torch.Tensor:
        if self.track == "observation":
            spec = task_spec or {}
            raw_tid = encode_task_id(spec.get("target_key", ""), spec.get("semantic_relation", "approach"), spec.get("instruction", ""))
            tid = raw_tid
            if self.branch_interceptors.get("task_id") is not None:
                tid = self.branch_interceptors["task_id"](tid)
            t_tensor = torch.tensor([tid], dtype=torch.long, device=self.device)
            if self.branch_interceptors.get("task_token") is not None:
                t_tensor = self.branch_interceptors["task_token"](t_tensor)

            s_ground = s_t
            if self.branch_interceptors.get("grounding_latent") is not None:
                s_ground = self.branch_interceptors["grounding_latent"](s_ground)

            t_emb = self.model.task_embedding(t_tensor)
            if self.branch_interceptors.get("task_embedding") is not None:
                t_emb = self.branch_interceptors["task_embedding"](t_emb)

            raw = self.model.grounding_head(torch.cat([s_ground, t_emb], dim=-1))
            dx_dy = torch.clamp(raw[:, :2], -3.5, 3.5)
            trig = torch.nn.functional.normalize(raw[:, 2:4] + 1e-6, dim=-1)
            vis_goal = torch.cat([dx_dy, trig], dim=-1)
            if self.branch_interceptors.get("visual_goal") is not None:
                vis_goal = self.branch_interceptors["visual_goal"](vis_goal)

            g_mem_prev = self._ego_goal_memory.clone() if self._ego_goal_memory is not None else None
            if self._prev_latent is not None and self._ego_goal_memory is not None:
                mem = self._ego_goal_memory.clone()
                if self.branch_interceptors.get("memory_prev") is not None:
                    mem = self.branch_interceptors["memory_prev"](mem)

                # Decoded latent ego-motion displacement
                realized_disp = self.model.predict_ego_displacement(self._prev_latent, s_t).squeeze(0)
                if self.branch_interceptors.get("ego_disp") is not None:
                    realized_disp = self.branch_interceptors["ego_disp"](realized_disp)

                dx_rem = mem[0] - realized_disp[0]
                dy_rem = mem[1] - realized_disp[1]
                dth = realized_disp[2]
                c, s = torch.cos(dth), torch.sin(dth)
                mem_dx = c * dx_rem + s * dy_rem
                mem_dy = -s * dx_rem + c * dy_rem
                mem_updated = torch.stack([mem_dx, mem_dy, mem[2], mem[3]]).unsqueeze(0)
                if self.branch_interceptors.get("memory_updated") is not None:
                    mem_updated = self.branch_interceptors["memory_updated"](mem_updated)

                goal_t = 0.65 * vis_goal + 0.35 * mem_updated
            else:
                realized_disp = torch.zeros(3, device=self.device)
                mem_updated = vis_goal
                goal_t = vis_goal

            self._last_vis_goal = vis_goal
            self._last_mem_updated = mem_updated
            if self.branch_interceptors.get("blended_goal") is not None:
                bg_fn = self.branch_interceptors["blended_goal"]
                try:
                    goal_t = bg_fn(goal_t, vis_goal, mem_updated)
                except TypeError:
                    goal_t = bg_fn(goal_t)

            self._prev_latent = s_t.detach()
            self._ego_goal_memory = goal_t.squeeze(0).detach()
            self.last_goal_diag = {
                "raw_task_id": raw_tid,
                "used_task_id": tid,
                "task_embedding": t_emb.squeeze(0).detach().cpu().numpy(),
                "grounding_latent": s_ground.squeeze(0).detach().cpu().numpy(),
                "g_visual_t": vis_goal.squeeze(0).detach().cpu().numpy(),
                "g_memory_prev": g_mem_prev.cpu().numpy() if g_mem_prev is not None else None,
                "delta_ego_t": realized_disp.detach().cpu().numpy(),
                "g_memory_t": mem_updated.squeeze(0).detach().cpu().numpy(),
                "g_blended_t": goal_t.squeeze(0).detach().cpu().numpy(),
                "active_interceptors": [k for k, v in self.branch_interceptors.items() if v is not None],
            }
            return goal_t

        if goal_vec_body is None:
            if robot_pose is None or target_pose is None:
                raise ValueError("Track B requires goal_vec_body.")
            goal_vec_body = compute_body_frame_goal_vec(robot_pose, target_pose)
        goal_t = torch.as_tensor(goal_vec_body, dtype=torch.float32, device=self.device).view(1, 4)
        if self.branch_interceptors.get("blended_goal") is not None:
            bg_fn = self.branch_interceptors["blended_goal"]
            try:
                goal_t = bg_fn(goal_t, None, None)
            except TypeError:
                goal_t = bg_fn(goal_t)
        self.last_goal_diag = {
            "raw_task_id": None,
            "used_task_id": None,
            "g_visual_t": None,
            "g_memory_prev": None,
            "delta_ego_t": None,
            "g_memory_t": None,
            "g_blended_t": goal_t.squeeze(0).detach().cpu().numpy(),
            "active_interceptors": [k for k, v in self.branch_interceptors.items() if v is not None],
        }
        return goal_t

    @torch.no_grad()
    def act(self, obs_rgb, robot_pose=None, target_pose=None, task_spec=None, goal_vec_body=None):
        img_t = torch.from_numpy(obs_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        s_t = self.model.encode(img_t)
        goal_t = self._resolve_goal_vec(s_t, robot_pose, target_pose, task_spec, goal_vec_body)

        a_react = self.model.predict_reactive_action(s_t, goal_t)
        if self.ablation_mode == "reactive":
            _, col_logit = self.model.predict_next(s_t, a_react)
            act_react = np.clip(a_react.squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
            return (
                act_react,
                {
                    "col_risk": float(torch.sigmoid(col_logit).item()),
                    "latent": s_t.squeeze(0).cpu().numpy(),
                    "track": self.track,
                    "ablation_mode": self.ablation_mode,
                    "g_visual_t": self.last_goal_diag.get("g_visual_t"),
                    "g_memory_prev": self.last_goal_diag.get("g_memory_prev"),
                    "delta_ego_t": self.last_goal_diag.get("delta_ego_t"),
                    "g_memory_t": self.last_goal_diag.get("g_memory_t"),
                    "g_blended_t": self.last_goal_diag.get("g_blended_t"),
                    "a_react": a_react.squeeze(0).cpu().numpy(),
                    "best_cand_idx": 0,
                    "micro_action": act_react,
                },
            )

        cand_actions = _build_holonomic_candidates(a_react, self.device)
        num_cands = cand_actions.shape[0]
        s_batch = s_t.repeat(num_cands, 1)

        if self.ablation_mode == "no_action_conditioning":
            action_in = torch.zeros_like(cand_actions)
        else:
            action_in = cand_actions

        next_s, col_logits = self.model.predict_next(s_batch, action_in)
        if self.ablation_mode == "prediction_ignored":
            # Replace predicted latent state next_s with unrolled current state s_batch
            next_s = s_batch

        # Decode predicted 1-step body-frame displacement strictly from (s_batch, next_s)
        pred_disp = self.model.predict_ego_displacement(s_batch, next_s)
        col_probs = torch.sigmoid(col_logits).squeeze(-1)

        gx = goal_t[0, 0]
        gy = goal_t[0, 1]
        g_dth = torch.atan2(goal_t[0, 2], goal_t[0, 3])

        # Evaluate multi-step projection of the 1-step predicted displacement (horizon scaling = 3.5 steps)
        horizon_scale = 3.5
        rem_dx = gx - horizon_scale * pred_disp[:, 0]
        rem_dy = gy - horizon_scale * pred_disp[:, 1]
        rem_dist = torch.sqrt(rem_dx ** 2 + rem_dy ** 2 + 1e-6)
        rem_yaw = torch.abs(torch.atan2(torch.sin(g_dth - horizon_scale * pred_disp[:, 2]), torch.cos(g_dth - horizon_scale * pred_disp[:, 2])))

        prior_cost = torch.sum((cand_actions - a_react) ** 2, dim=-1)
        lambda_col = 0.0 if self.ablation_mode == "no_collision_prediction" else 1.8

        if self.ablation_mode == "prediction_ignored":
            best_idx = 0
        else:
            total_cost = rem_dist + 0.15 * rem_yaw + 0.12 * prior_cost + lambda_col * col_probs
            best_idx = int(torch.argmin(total_cost).item())

        best_act = np.clip(cand_actions[best_idx].cpu().numpy(), -1.0, 1.0).astype(np.float32)
        best_col = float(col_probs[best_idx].item())
        return best_act, {
            "col_risk": best_col,
            "latent": s_t.squeeze(0).cpu().numpy(),
            "predicted_next_latent": next_s[best_idx].cpu().numpy(),
            "track": self.track,
            "ablation_mode": self.ablation_mode,
            "g_visual_t": self.last_goal_diag.get("g_visual_t"),
            "g_memory_prev": self.last_goal_diag.get("g_memory_prev"),
            "delta_ego_t": self.last_goal_diag.get("delta_ego_t"),
            "g_memory_t": self.last_goal_diag.get("g_memory_t"),
            "g_blended_t": self.last_goal_diag.get("g_blended_t"),
            "a_react": a_react.squeeze(0).cpu().numpy(),
            "best_cand_idx": best_idx,
            "micro_action": best_act,
            "pred_disp": pred_disp[best_idx].cpu().numpy(),
        }


class HJEPAAgent(BaseAgentWrapper):
    """
    Hierarchical World Model (H-JEPA):
      - Tier 1 (Macro, stride K=5): Evaluates multi-step macro action sequences via MacroPredictor,
        decodes K-step latent displacement from next_S, and selects optimal latent subgoal S*_{t+K}.
      - Tier 2 (Micro, stride 1): Generates primitive action via MicroPredictor.select_action(s_t, S*_{t+K})
        with 1-step micro collision verification.
    """

    def __init__(
        self,
        ckpt_path: str | None = None,
        device: str = "cpu",
        track: str = "structured",
        ablation_mode: str = "prediction_used",
    ):
        super().__init__(name="H_JEPA_Hierarchical", track=track, ablation_mode=ablation_mode, device=device)
        self.model = HJEPAModel(latent_dim=32, stride_k=5, hidden_dim=64).to(device)
        _load_state_dict_flexible(self.model, ckpt_path, device, default_file="hjepa_best.pt")
        self.model.eval()

    def _resolve_goal_vec(self, s_t: torch.Tensor, robot_pose, target_pose, task_spec, goal_vec_body) -> torch.Tensor:
        if self.track == "observation":
            spec = task_spec or {}
            raw_tid = encode_task_id(spec.get("target_key", ""), spec.get("semantic_relation", "approach"), spec.get("instruction", ""))
            tid = raw_tid
            if self.branch_interceptors.get("task_id") is not None:
                tid = self.branch_interceptors["task_id"](tid)
            t_tensor = torch.tensor([tid], dtype=torch.long, device=self.device)
            if self.branch_interceptors.get("task_token") is not None:
                t_tensor = self.branch_interceptors["task_token"](t_tensor)

            s_ground = s_t
            if self.branch_interceptors.get("grounding_latent") is not None:
                s_ground = self.branch_interceptors["grounding_latent"](s_ground)

            t_emb = self.model.task_embedding(t_tensor)
            if self.branch_interceptors.get("task_embedding") is not None:
                t_emb = self.branch_interceptors["task_embedding"](t_emb)

            raw = self.model.grounding_head(torch.cat([s_ground, t_emb], dim=-1))
            dx_dy = torch.clamp(raw[:, :2], -3.5, 3.5)
            trig = torch.nn.functional.normalize(raw[:, 2:4] + 1e-6, dim=-1)
            vis_goal = torch.cat([dx_dy, trig], dim=-1)
            if self.branch_interceptors.get("visual_goal") is not None:
                vis_goal = self.branch_interceptors["visual_goal"](vis_goal)

            g_mem_prev = self._ego_goal_memory.clone() if self._ego_goal_memory is not None else None
            memory_branch_executed = False
            if (not self.disable_memory) and (self._prev_latent is not None) and (g_mem_prev is not None):
                mem = g_mem_prev.clone()
                if self.branch_interceptors.get("memory_prev") is not None:
                    mem = self.branch_interceptors["memory_prev"](mem)

                realized_disp = self.model.predict_micro_displacement(self._prev_latent, s_t).squeeze(0)
                if self.branch_interceptors.get("ego_disp") is not None:
                    realized_disp = self.branch_interceptors["ego_disp"](realized_disp)

                dx_rem = mem[0] - realized_disp[0]
                dy_rem = mem[1] - realized_disp[1]
                dth = realized_disp[2]
                c, s = torch.cos(dth), torch.sin(dth)
                mem_dx = c * dx_rem + s * dy_rem
                mem_dy = -s * dx_rem + c * dy_rem
                mem_updated = torch.stack([mem_dx, mem_dy, mem[2], mem[3]]).unsqueeze(0)
                if self.branch_interceptors.get("memory_updated") is not None:
                    mem_updated = self.branch_interceptors["memory_updated"](mem_updated)

                goal_t = 0.65 * vis_goal + 0.35 * mem_updated
                memory_branch_executed = True
                default_source = "track_c_blended"
            elif self.disable_memory:
                realized_disp = None
                mem_updated = None
                goal_t = vis_goal
                default_source = "track_c_visual_only"
            else:
                realized_disp = None
                mem_updated = vis_goal
                goal_t = vis_goal
                default_source = "track_c_initial_visual"

            self._last_vis_goal = vis_goal
            self._last_mem_updated = mem_updated
            if self.branch_interceptors.get("blended_goal") is not None:
                bg_fn = self.branch_interceptors["blended_goal"]
                try:
                    goal_t = bg_fn(goal_t, vis_goal, mem_updated)
                except TypeError:
                    goal_t = bg_fn(goal_t)
                default_source = "track_c_blended_intercepted"

            self._prev_latent = s_t.detach()
            if self.disable_memory:
                self._ego_goal_memory = None
            else:
                self._ego_goal_memory = goal_t.squeeze(0).detach()

            g_blended_np = goal_t.squeeze(0).detach().cpu().numpy()
            self.last_goal_diag = {
                "raw_task_id": raw_tid,
                "used_task_id": tid,
                "task_embedding": t_emb.squeeze(0).detach().cpu().numpy(),
                "grounding_latent": s_ground.squeeze(0).detach().cpu().numpy(),
                "g_visual_t": vis_goal.squeeze(0).detach().cpu().numpy(),
                "g_memory_prev": g_mem_prev.cpu().numpy() if (not self.disable_memory and g_mem_prev is not None) else None,
                "delta_ego_t": realized_disp.detach().cpu().numpy() if realized_disp is not None else None,
                "g_memory_t": mem_updated.squeeze(0).detach().cpu().numpy() if mem_updated is not None else None,
                "g_blended_t": g_blended_np,
                "planner_goal_t": g_blended_np.copy(),
                "task_embedding_executed": True,
                "grounding_head_executed": True,
                "memory_branch_executed": bool(memory_branch_executed),
                "planner_goal_source": default_source,
                "active_interceptors": [k for k, v in self.branch_interceptors.items() if v is not None],
            }
            return goal_t

        if goal_vec_body is None:
            if robot_pose is None or target_pose is None:
                raise ValueError("Track B requires goal_vec_body.")
            goal_vec_body = compute_body_frame_goal_vec(robot_pose, target_pose)
        goal_t = torch.as_tensor(goal_vec_body, dtype=torch.float32, device=self.device).view(1, 4)
        if self.branch_interceptors.get("blended_goal") is not None:
            bg_fn = self.branch_interceptors["blended_goal"]
            try:
                goal_t = bg_fn(goal_t, None, None)
            except TypeError:
                goal_t = bg_fn(goal_t)
        self.last_goal_diag = {
            "raw_task_id": None,
            "used_task_id": None,
            "task_embedding": None,
            "grounding_latent": None,
            "g_visual_t": None,
            "g_memory_prev": None,
            "delta_ego_t": None,
            "g_memory_t": None,
            "g_blended_t": None,
            "planner_goal_t": goal_t.squeeze(0).detach().cpu().numpy(),
            "task_embedding_executed": False,
            "grounding_head_executed": False,
            "memory_branch_executed": False,
            "planner_goal_source": "track_b_oracle_waypoint",
            "active_interceptors": [k for k, v in self.branch_interceptors.items() if v is not None],
        }
        return goal_t

    def _build_macro_sequence_candidates(self, a_react: torch.Tensor, cand_1step: torch.Tensor) -> torch.Tensor:
        """
        Builds 25 K-step macro action sequences (shape: 25 x 15):
          - Candidates 0..16: constant-action macro rollouts repeated across K=5 steps
          - Candidates 17..24: non-commutative two-phase detour-then-advance sequences
            (e.g. 2 steps lateral strafe around obstacle + 3 steps forward toward goal)
        """
        k = self.model.stride_k
        const_cands = cand_1step[:17].repeat(1, k)  # (17, 15)
        ar = a_react.squeeze(0)
        two_phase_list = [
            # Left strafe detour (2 steps) -> forward approach (3 steps)
            torch.cat([torch.tensor([0.30, 0.75, 0.10], device=self.device).repeat(2), torch.tensor([0.80, -0.15, 0.0], device=self.device).repeat(k - 2)]),
            # Right strafe detour (2 steps) -> forward approach (3 steps)
            torch.cat([torch.tensor([0.30, -0.75, -0.10], device=self.device).repeat(2), torch.tensor([0.80, 0.15, 0.0], device=self.device).repeat(k - 2)]),
            # Wide left flank (3 steps) -> inward turn (2 steps)
            torch.cat([torch.tensor([0.55, 0.65, 0.20], device=self.device).repeat(3), torch.tensor([0.75, -0.35, -0.20], device=self.device).repeat(k - 3)]),
            # Wide right flank (3 steps) -> inward turn (2 steps)
            torch.cat([torch.tensor([0.55, -0.65, -0.20], device=self.device).repeat(3), torch.tensor([0.75, 0.35, 0.20], device=self.device).repeat(k - 3)]),
            # Rotate left (2 steps) -> drive forward (3 steps)
            torch.cat([torch.tensor([0.15, 0.0, 0.75], device=self.device).repeat(2), torch.tensor([0.85, 0.0, 0.0], device=self.device).repeat(k - 2)]),
            # Rotate right (2 steps) -> drive forward (3 steps)
            torch.cat([torch.tensor([0.15, 0.0, -0.75], device=self.device).repeat(2), torch.tensor([0.85, 0.0, 0.0], device=self.device).repeat(k - 2)]),
            # Back-off + left strafe (2 steps) -> forward (3 steps)
            torch.cat([torch.tensor([-0.30, 0.70, 0.20], device=self.device).repeat(2), torch.tensor([0.75, 0.20, 0.0], device=self.device).repeat(k - 2)]),
            # Reactive proposal (2 steps) -> damped approach (3 steps)
            torch.cat([ar.repeat(2), (ar * 0.7).repeat(k - 2)]),
        ]
        two_phase = torch.stack(two_phase_list, dim=0)
        return torch.cat([const_cands, two_phase], dim=0)

    def _common_goal_diag_fields(self) -> dict:
        return {
            "raw_task_id": self.last_goal_diag.get("raw_task_id"),
            "used_task_id": self.last_goal_diag.get("used_task_id"),
            "task_embedding": self.last_goal_diag.get("task_embedding"),
            "grounding_latent": self.last_goal_diag.get("grounding_latent"),
            "g_visual_t": self.last_goal_diag.get("g_visual_t"),
            "g_memory_prev": self.last_goal_diag.get("g_memory_prev"),
            "delta_ego_t": self.last_goal_diag.get("delta_ego_t"),
            "g_memory_t": self.last_goal_diag.get("g_memory_t"),
            "g_blended_t": self.last_goal_diag.get("g_blended_t"),
            "planner_goal_t": self.last_goal_diag.get("planner_goal_t"),
            "task_embedding_executed": bool(self.last_goal_diag.get("task_embedding_executed", False)),
            "grounding_head_executed": bool(self.last_goal_diag.get("grounding_head_executed", False)),
            "memory_branch_executed": bool(self.last_goal_diag.get("memory_branch_executed", False)),
            "planner_goal_source": self.last_goal_diag.get("planner_goal_source", "unknown"),
        }

    @torch.no_grad()
    def act(self, obs_rgb, robot_pose=None, target_pose=None, task_spec=None, goal_vec_body=None):
        img_t = torch.from_numpy(obs_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        s_t = self.model.encode(img_t)
        goal_t = self._resolve_goal_vec(s_t, robot_pose, target_pose, task_spec, goal_vec_body)

        if self.branch_interceptors.get("planner_goal") is not None:
            pg_fn = self.branch_interceptors["planner_goal"]
            try:
                goal_t = pg_fn(goal_t, self.last_goal_diag)
            except TypeError:
                goal_t = pg_fn(goal_t)
            goal_t = torch.as_tensor(goal_t, dtype=torch.float32, device=self.device).view(1, 4)
            self.last_goal_diag["planner_goal_t"] = goal_t.squeeze(0).detach().cpu().numpy()
            self.last_goal_diag["planner_goal_source"] = (
                "track_c_planner_goal_override" if self.track == "observation" else "planner_goal_interceptor"
            )

        a_react = self.model.predict_reactive_action(s_t, goal_t)
        if self.ablation_mode == "reactive":
            _, micro_col_logit = self.model.micro_predictor.forward_transition(s_t, a_react)
            act_react = np.clip(a_react.squeeze(0).cpu().numpy(), -1.0, 1.0).astype(np.float32)
            return (
                act_react,
                {
                    "col_risk": float(torch.sigmoid(micro_col_logit).item()),
                    "latent": s_t.squeeze(0).cpu().numpy(),
                    "track": self.track,
                    "ablation_mode": self.ablation_mode,
                    **self._common_goal_diag_fields(),
                    "a_react": a_react.squeeze(0).cpu().numpy(),
                    "micro_action": act_react,
                },
            )

        cand_1step = _build_holonomic_candidates(a_react, self.device)
        num_cands = cand_1step.shape[0]
        s_batch = s_t.repeat(num_cands, 1)

        gx = goal_t[0, 0]
        gy = goal_t[0, 1]
        g_dth = torch.atan2(goal_t[0, 2], goal_t[0, 3])
        lambda_col = 0.0 if self.ablation_mode == "no_collision_prediction" else 1.8

        if self.ablation_mode == "no_temporal_hierarchy":
            # Ablation: bypass MacroPredictor (K=5) and plan using 1-step MicroPredictor only
            act_in = torch.zeros_like(cand_1step) if self.ablation_mode == "no_action_conditioning" else cand_1step
            next_s_micro, micro_col_logits = self.model.micro_predictor.forward_transition(s_batch, act_in)
            pred_micro_disp = self.model.predict_micro_displacement(s_batch, next_s_micro)
            col_probs = torch.sigmoid(micro_col_logits).squeeze(-1)
            rem_dx = gx - 3.5 * pred_micro_disp[:, 0]
            rem_dy = gy - 3.5 * pred_micro_disp[:, 1]
            rem_dist = torch.sqrt(rem_dx ** 2 + rem_dy ** 2 + 1e-6)
            prior_cost = torch.sum((cand_1step - a_react) ** 2, dim=-1)
            total_cost = rem_dist + 0.12 * prior_cost + lambda_col * col_probs
            best_idx = int(torch.argmin(total_cost).item())
            best_act = np.clip(cand_1step[best_idx].cpu().numpy(), -1.0, 1.0).astype(np.float32)
            return best_act, {
                "col_risk": float(col_probs[best_idx].item()),
                "latent": s_t.squeeze(0).cpu().numpy(),
                "track": self.track,
                "ablation_mode": self.ablation_mode,
                **self._common_goal_diag_fields(),
                "a_react": a_react.squeeze(0).cpu().numpy(),
                "best_micro_idx": best_idx,
                "micro_action": best_act,
                "pred_micro_disp": pred_micro_disp[best_idx].cpu().numpy(),
            }

        # Tier 1: Macro Planning over K=5 steps
        macro_seqs = self._build_macro_sequence_candidates(a_react, cand_1step)
        macro_in = torch.zeros_like(macro_seqs) if self.ablation_mode == "no_action_conditioning" else macro_seqs
        next_S, macro_col_logits = self.model.macro_predictor(s_batch, macro_in)
        if self.ablation_mode == "prediction_ignored":
            next_S = s_batch

        pred_macro_disp = self.model.predict_macro_displacement(s_batch, next_S)
        state_macro_col = self.model.state_col_head(next_S)
        macro_col_probs = torch.sigmoid(0.5 * (macro_col_logits + state_macro_col)).squeeze(-1)

        rem_dx_macro = gx - pred_macro_disp[:, 0]
        rem_dy_macro = gy - pred_macro_disp[:, 1]
        rem_dist_macro = torch.sqrt(rem_dx_macro ** 2 + rem_dy_macro ** 2 + 1e-6)
        rem_yaw_macro = torch.abs(
            torch.atan2(torch.sin(g_dth - pred_macro_disp[:, 2]), torch.cos(g_dth - pred_macro_disp[:, 2]))
        )
        first_step_actions = macro_seqs[:, :3]
        prior_macro = torch.sum((first_step_actions - a_react) ** 2, dim=-1)

        if self.ablation_mode == "prediction_ignored":
            best_macro_idx = 0
        else:
            macro_cost = rem_dist_macro + 0.15 * rem_yaw_macro + 0.10 * prior_macro + lambda_col * macro_col_probs
            best_macro_idx = int(torch.argmin(macro_cost).item())

        best_subgoal = next_S[best_macro_idx : best_macro_idx + 1]
        subgoal_first_action = first_step_actions[best_macro_idx : best_macro_idx + 1]

        # Tier 2: Micro Policy tracking best_subgoal + 1-step micro safety refinement
        a_micro_pol = self.model.micro_predictor.select_action(s_t, best_subgoal)
        a_proposal = 0.55 * a_micro_pol + 0.45 * subgoal_first_action

        micro_cands = _build_holonomic_candidates(a_proposal, self.device)
        micro_in = torch.zeros_like(micro_cands) if self.ablation_mode == "no_action_conditioning" else micro_cands
        next_s_m, micro_col_logits = self.model.micro_predictor.forward_transition(s_batch, micro_in)
        if self.ablation_mode == "prediction_ignored":
            next_s_m = s_batch

        pred_micro_disp = self.model.predict_micro_displacement(s_batch, next_s_m)
        micro_col_probs = torch.sigmoid(micro_col_logits).squeeze(-1)

        # Score micro candidates by alignment toward macro subgoal displacement + local collision avoidance
        target_step_disp = pred_macro_disp[best_macro_idx] / float(self.model.stride_k)
        step_err = torch.sqrt(
            (pred_micro_disp[:, 0] - target_step_disp[0]) ** 2
            + (pred_micro_disp[:, 1] - target_step_disp[1]) ** 2
            + 1e-6
        )
        rem_dx_m = gx - 3.5 * pred_micro_disp[:, 0]
        rem_dy_m = gy - 3.5 * pred_micro_disp[:, 1]
        rem_dist_m = torch.sqrt(rem_dx_m ** 2 + rem_dy_m ** 2 + 1e-6)
        micro_prior = torch.sum((micro_cands - a_proposal) ** 2, dim=-1)

        if self.ablation_mode == "prediction_ignored":
            best_micro_idx = 0
        else:
            micro_cost = 0.6 * rem_dist_m + 0.4 * step_err + 0.12 * micro_prior + lambda_col * micro_col_probs
            best_micro_idx = int(torch.argmin(micro_cost).item())

        best_act = np.clip(micro_cands[best_micro_idx].cpu().numpy(), -1.0, 1.0).astype(np.float32)
        best_col = float(0.5 * (macro_col_probs[best_macro_idx].item() + micro_col_probs[best_micro_idx].item()))
        return best_act, {
            "col_risk": best_col,
            "latent": s_t.squeeze(0).cpu().numpy(),
            "macro_subgoal": best_subgoal.squeeze(0).cpu().numpy(),
            "predicted_next_latent": next_s_m[best_micro_idx].cpu().numpy(),
            "track": self.track,
            "ablation_mode": self.ablation_mode,
            **self._common_goal_diag_fields(),
            "a_react": a_react.squeeze(0).cpu().numpy(),
            "macro_cand_seqs": macro_seqs.cpu().numpy(),
            "best_macro_idx": best_macro_idx,
            "a_proposal": a_proposal.squeeze(0).cpu().numpy(),
            "best_micro_idx": best_micro_idx,
            "micro_action": best_act,
            "pred_macro_disp": pred_macro_disp[best_macro_idx].cpu().numpy(),
            "pred_micro_disp": pred_micro_disp[best_micro_idx].cpu().numpy(),
        }


def load_all_agents(ckpt_dir: str | None = None, device: str = "cpu", track: str = "structured", seed: int | None = None):
    if ckpt_dir is None:
        v2_dir = str(_ACTIVE_CKPT_DIR)
        legacy_dir = str(_ACTIVE_CKPT_DIR)
        ckpt_dir = v2_dir if os.path.exists(os.path.join(v2_dir, "lewm_best.pt")) else legacy_dir

    suffix = f"_seed{seed}_best.pt" if seed is not None else "_best.pt"
    base_pt = os.path.join(ckpt_dir, f"baseline{suffix}")
    lewm_pt = os.path.join(ckpt_dir, f"lewm{suffix}")
    hjepa_pt = os.path.join(ckpt_dir, f"hjepa{suffix}")
    if not os.path.exists(base_pt):
        base_pt = os.path.join(ckpt_dir, "baseline_best.pt")
        lewm_pt = os.path.join(ckpt_dir, "lewm_best.pt")
        hjepa_pt = os.path.join(ckpt_dir, "hjepa_best.pt")

    return {
        "Oracle": OracleAgent(),
        "Random": RandomAgent(seed=seed or 42),
        "Baseline": BaselineAgent(base_pt, device=device, track=track),
        "LeWM": LeWMAgent(lewm_pt, device=device, track=track),
        "H-JEPA": HJEPAAgent(hjepa_pt, device=device, track=track),
    }
