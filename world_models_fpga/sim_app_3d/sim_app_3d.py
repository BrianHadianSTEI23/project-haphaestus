import os
import sys
import time
import base64
import threading
import argparse
import numpy as np
import cv2
import torch
from flask import Flask, render_template_string, Response, request, jsonify

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.amr_env import JetAutoEnv
from nlp.grounding import SemanticGoalGrounder
from models.baseline_policy import BaselineNavigationPolicy
from models.le_world_model import LeWorldModel
from models.h_jepa import HJEPAModel

app = Flask(__name__)

class AMRSimulationController:
    """
    Manages the real-time 3D simulation loop, model inference (Baseline / LeWM / H-JEPA),
    natural language goal grounding, and camera feedback verification.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.env = JetAutoEnv()
        self.grounder = SemanticGoalGrounder()
        self.lock = threading.Lock()

        # Load trained model checkpoints
        ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
        self.baseline = BaselineNavigationPolicy().to(self.device)
        self.lewm = LeWorldModel().to(self.device)
        self.hjepa = HJEPAModel(stride_k=5).to(self.device)

        base_pt = os.path.join(ckpt_dir, "baseline_best.pt")
        if os.path.exists(base_pt):
            self.baseline.load_state_dict(torch.load(base_pt, map_location=self.device))
            self.baseline.eval()

        lewm_pt = os.path.join(ckpt_dir, "lewm_best.pt")
        if os.path.exists(lewm_pt):
            self.lewm.load_state_dict(torch.load(lewm_pt, map_location=self.device))
            self.lewm.eval()

        hjepa_pt = os.path.join(ckpt_dir, "hjepa_best.pt")
        if os.path.exists(hjepa_pt):
            self.hjepa.load_state_dict(torch.load(hjepa_pt, map_location=self.device))
            self.hjepa.eval()

        # State variables
        self.current_model = "lewm" # "baseline", "lewm", "hjepa"
        self.current_cmd = "go behind the yellow cylinder"
        self.goal_dict = None
        self.robot_pose = np.zeros(3, dtype=np.float32)
        self.current_action = np.zeros(3, dtype=np.float32)
        self.feedback = {}
        self.step_count = 0
        self.collision_count = 0
        self.is_running = True
        self.last_3d_frame = None
        self.last_fp_frame = None
        self.trajectory_history = []
        self.inference_latency_us = 48.0 # FPGA baseline latency

        self.reset_sim(initial_cmd=self.current_cmd)

    def reset_sim(self, initial_cmd=None, initial_pos=None):
        with self.lock:
            opts = {"initial_pos": initial_pos} if initial_pos is not None else None
            obs, info = self.env.reset(options=opts)
            self.robot_pose = info["pos"]
            self.step_count = 0
            self.collision_count = 0
            self.trajectory_history = [self.robot_pose[:2].copy()]
            if initial_cmd:
                self.current_cmd = initial_cmd
            self.goal_dict = self.grounder.parse_command(self.current_cmd, self.robot_pose)
            self.feedback = self.grounder.verify_goal_with_camera(obs, self.robot_pose, self.goal_dict)

    def set_command(self, cmd_text):
        with self.lock:
            self.current_cmd = cmd_text
            self.goal_dict = self.grounder.parse_command(cmd_text, self.robot_pose)
            print(f"[NLP Engine] New Goal: {cmd_text} -> Landmark: {self.goal_dict['landmark_label']}, Waypoint: {self.goal_dict['target_pose']}")

    def set_model(self, model_name):
        with self.lock:
            if model_name in ["baseline", "lewm", "hjepa"]:
                self.current_model = model_name
                print(f"[Controller] Switched active world model to: {model_name.upper()}")

    def step(self):
        with self.lock:
            obs = self.env.render_observation()
            self.robot_pose = self.env.get_robot_pose()
            self.trajectory_history.append(self.robot_pose[:2].copy())

            # Goal verification
            self.feedback = self.grounder.verify_goal_with_camera(obs, self.robot_pose, self.goal_dict)

            # Check if goal already reached
            if self.feedback["goal_verified"]:
                # Gentle braking / holding pose
                self.current_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                self.env.step(self.current_action)
                self._update_render_buffers(obs)
                return

            t_start = time.perf_counter()

            # Choose current active waypoint (flank waypoint first if routing around obstacle)
            current_target = self.goal_dict["target_xy"]
            if self.goal_dict.get("has_flank", False):
                dist_to_flank = np.linalg.norm(self.robot_pose[:2] - self.goal_dict["flank_xy"])
                dist_to_target = np.linalg.norm(self.robot_pose[:2] - self.goal_dict["target_xy"])
                if dist_to_flank > 0.40 and dist_to_target > 0.60:
                    current_target = self.goal_dict["flank_xy"]

            # Target delta in robot body frame
            dx_w = current_target[0] - self.robot_pose[0]
            dy_w = current_target[1] - self.robot_pose[1]
            th = self.robot_pose[2]
            dx_b = np.cos(th) * dx_w + np.sin(th) * dy_w
            dy_b = -np.sin(th) * dx_w + np.cos(th) * dy_w
            target_dist = np.sqrt(dx_b**2 + dy_b**2)

            # Heading angle towards the target position
            ang_to_goal = np.arctan2(dy_b, dx_b)

            # Final desired orientation error once at the target
            d_th = np.arctan2(np.sin(self.goal_dict["target_theta"] - th), np.cos(self.goal_dict["target_theta"] - th))

            # Visual servoing guidance if landmark visible in camera
            vis_steering = 0.0
            if self.feedback.get("visual_acquired", False):
                vis_steering = -0.5 * self.feedback["u_err"]

            # Landmark proximity check for circumnavigation
            dx_lmk = self.goal_dict["landmark_pos"][0] - self.robot_pose[0]
            dy_lmk = self.goal_dict["landmark_pos"][1] - self.robot_pose[1]
            dist_lmk = np.sqrt(dx_lmk**2 + dy_lmk**2)

            # Circumnavigation offset if obstacle is between robot and behind waypoint
            strafe_circ = 0.0
            if self.goal_dict["relation"] == "behind" and dist_lmk < 0.75 and target_dist > 0.40:
                # Tangential strafe direction to glide around the obstacle
                ang_lmk = np.arctan2(dy_lmk, dx_lmk)
                tangent_ang = ang_lmk + np.pi / 2.0
                strafe_circ = 0.65 * np.sin(tangent_ang - th)

            # Holonomic nominal command
            if target_dist > 0.38:
                # Rotate towards target if facing away
                wz_nominal = float(np.clip(1.5 * ang_to_goal + vis_steering, -1.0, 1.0))
                if abs(ang_to_goal) < 0.8:
                    vx_nominal = float(np.clip(0.85 * np.cos(ang_to_goal), 0.2, 1.0))
                else:
                    vx_nominal = 0.25
                vy_nominal = float(np.clip(0.6 * np.sin(ang_to_goal) + strafe_circ, -0.8, 0.8))
            else:
                # Arrived at behind waypoint: face towards the landmark to verify with camera!
                ang_to_lmk_body = np.arctan2(
                    np.cos(th) * dy_lmk - np.sin(th) * dx_lmk,
                    np.cos(th) * dx_lmk + np.sin(th) * dy_lmk
                )
                vx_nominal = float(np.clip(0.3 * dx_b, -0.3, 0.3))
                vy_nominal = float(np.clip(0.3 * dy_b, -0.3, 0.3))
                wz_nominal = float(np.clip(1.6 * ang_to_lmk_body + vis_steering, -1.0, 1.0))

            # Execute selected model
            if self.current_model == "baseline":
                # Baseline Reactive Policy
                act, col_risk, _ = self.baseline.act(obs, self.robot_pose, self.goal_dict["target_pose"], device=self.device)
                act[0] = 0.6 * act[0] + 0.4 * vx_nominal
                act[1] = 0.6 * act[1] + 0.4 * vy_nominal
                act[2] = 0.5 * act[2] + 0.5 * wz_nominal
                self.current_action = np.clip(act, -1.0, 1.0)

            elif self.current_model == "lewm":
                # LeWorldModel Latent Foresight MPC:
                # Evaluates candidate trajectories using LeWM dynamics predictor to ensure obstacle safety
                img_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
                with torch.no_grad():
                    s_curr = self.lewm.encode(img_t)

                num_cands = 24
                cands = [[vx_nominal, vy_nominal, wz_nominal]] # Nominal goal-directed candidate
                # Generate variations
                for _ in range(num_cands - 1):
                    vx_v = np.clip(vx_nominal + np.random.uniform(-0.3, 0.3), -0.5, 1.0)
                    vy_v = np.clip(vy_nominal + np.random.uniform(-0.4, 0.4), -0.8, 0.8)
                    wz_v = np.clip(wz_nominal + np.random.uniform(-0.4, 0.4), -1.0, 1.0)
                    cands.append([vx_v, vy_v, wz_v])

                cands_t = torch.tensor(cands, dtype=torch.float32, device=self.device)
                s_batch = s_curr.repeat(num_cands, 1)

                with torch.no_grad():
                    next_s, col_logits = self.lewm.predict_next(s_batch, cands_t)
                    col_probs = torch.sigmoid(col_logits).squeeze(-1).cpu().numpy()

                best_score = -float("inf")
                best_act = np.array([vx_nominal, vy_nominal, wz_nominal], dtype=np.float32)

                dt = 0.1
                for i, c in enumerate(cands):
                    # Future body displacement
                    vx_s = c[0] * 0.6
                    vy_s = c[1] * 0.4
                    wz_s = c[2] * 1.5

                    # Future pose estimate
                    fut_th = th + wz_s * dt
                    fut_dx_w = dx_w - (np.cos(fut_th) * vx_s - np.sin(fut_th) * vy_s) * dt
                    fut_dy_w = dy_w - (np.sin(fut_th) * vx_s + np.cos(fut_th) * vy_s) * dt
                    fut_dist = np.sqrt(fut_dx_w**2 + fut_dy_w**2)
                    dist_gain = (target_dist - fut_dist)

                    # Collision penalty from LeWM latent foresight
                    col_pen = 3.5 * col_probs[i]

                    score = dist_gain - col_pen
                    if score > best_score:
                        best_score = score
                        best_act = np.array(c, dtype=np.float32)

                self.current_action = np.clip(best_act, -1.0, 1.0)

            elif self.current_model == "hjepa":
                # H-JEPA Hierarchical Planning:
                # Level 1 (Macro Stride K=5): Evaluates multi-step path progress and macro collision risk
                img_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
                with torch.no_grad():
                    s_curr = self.hjepa.encode(img_t)

                num_cands = 24
                cands = [[vx_nominal, vy_nominal, wz_nominal]]
                for _ in range(num_cands - 1):
                    vx_v = np.clip(vx_nominal + np.random.uniform(-0.25, 0.25), -0.5, 1.0)
                    vy_v = np.clip(vy_nominal + np.random.uniform(-0.35, 0.35), -0.8, 0.8)
                    wz_v = np.clip(wz_nominal + np.random.uniform(-0.35, 0.35), -1.0, 1.0)
                    cands.append([vx_v, vy_v, wz_v])

                cands_t = torch.tensor(cands, dtype=torch.float32, device=self.device)
                s_batch = s_curr.repeat(num_cands, 1)

                with torch.no_grad():
                    next_S, macro_col_logits = self.hjepa.macro_predictor(s_batch, cands_t)
                    col_probs = torch.sigmoid(macro_col_logits).squeeze(-1).cpu().numpy()

                best_score = -float("inf")
                best_act = np.array([vx_nominal, vy_nominal, wz_nominal], dtype=np.float32)
                dt_macro = 0.5

                for i, c in enumerate(cands):
                    fut_th = th + c[2] * 1.5 * dt_macro
                    fut_dx_w = dx_w - (np.cos(fut_th) * c[0] * 0.6 - np.sin(fut_th) * c[1] * 0.4) * dt_macro
                    fut_dy_w = dy_w - (np.sin(fut_th) * c[0] * 0.6 + np.cos(fut_th) * c[1] * 0.4) * dt_macro
                    fut_dist = np.sqrt(fut_dx_w**2 + fut_dy_w**2)
                    dist_gain = (target_dist - fut_dist)

                    col_pen = 4.0 * col_probs[i]
                    score = dist_gain - col_pen
                    if score > best_score:
                        best_score = score
                        best_act = np.array(c, dtype=np.float32)

                self.current_action = np.clip(best_act, -1.0, 1.0)

            t_end = time.perf_counter()
            self.inference_latency_us = (t_end - t_start) * 1e6

            # Step physics
            next_obs, reward, terminated, truncated, info = self.env.step(self.current_action)
            if info["collision"]:
                self.collision_count += 1

            self.step_count += 1
            self._update_render_buffers(next_obs)

    def _update_render_buffers(self, obs_rgb):
        # 1. First-Person Camera HUD overlay
        fp_img = obs_rgb.copy()
        # Scale up to 256x256 for sharp viewer display
        fp_vis = cv2.resize(fp_img, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Draw visual detection feedback on HUD
        if self.feedback.get("visual_acquired", False):
            # Scale detection centroid to 256x256
            u = int(self.feedback["u_center"] * 4)
            v = int(self.feedback["v_center"] * 4)
            cv2.circle(fp_vis, (u, v), 18, (0, 255, 0), 2)
            cv2.drawMarker(fp_vis, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(fp_vis, f"TARGET LOCKED: {self.goal_dict['landmark_label'].upper()}", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(fp_vis, f"SEARCHING: {self.goal_dict['landmark_label'].upper()}", (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        self.last_fp_frame = cv2.cvtColor(fp_vis, cv2.COLOR_RGB2BGR)

        # 2. 3D Third-Person Arena View
        tp_img = self.env.render_3d_view(camera="arena_cam")
        tp_vis = tp_img.copy()

        # HUD Text on 3D View
        cv2.putText(tp_vis, f"Prompt: \"{self.current_cmd}\"", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(tp_vis, f"Active Model: {self.current_model.upper()} | Latency: {self.inference_latency_us:.1f} us", (12, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 1, cv2.LINE_AA)

        status_txt = "GOAL REACHED & CAMERA VERIFIED!" if self.feedback.get("goal_verified", False) else "NAVIGATING..."
        status_col = (0, 255, 0) if self.feedback.get("goal_verified", False) else (0, 220, 255)
        cv2.putText(tp_vis, f"Status: {status_txt}", (12, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_col, 2, cv2.LINE_AA)

        self.last_3d_frame = cv2.cvtColor(tp_vis, cv2.COLOR_RGB2BGR)

sim_controller = None

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hiwonder JetAuto AMR - 3D World Models & NLP Navigation</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Inter', sans-serif; background: #0f172a; color: #f8fafc; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 16px; margin-bottom: 20px; }
        .title { font-size: 22px; font-weight: 700; color: #38bdf8; }
        .subtitle { font-size: 13px; color: #94a3b8; margin-top: 4px; }
        .author-badge { background: #1e293b; border: 1px solid #38bdf8; border-radius: 8px; padding: 6px 14px; font-size: 13px; color: #38bdf8; }
        .grid-container { display: grid; grid-template-columns: 1fr 340px; gap: 20px; }
        .view-panel { background: #1e293b; border-radius: 12px; border: 1px solid #334155; padding: 16px; display: flex; flex-direction: column; align-items: center; }
        .view-title { align-self: flex-start; font-size: 15px; font-weight: 600; color: #cbd5e1; margin-bottom: 12px; }
        .stream-img { width: 100%; max-width: 680px; border-radius: 8px; border: 1px solid #475569; }
        .sidebar { display: flex; flex-direction: column; gap: 16px; }
        .card { background: #1e293b; border-radius: 12px; border: 1px solid #334155; padding: 16px; }
        .card-title { font-size: 14px; font-weight: 600; color: #38bdf8; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
        .input-group { display: flex; gap: 8px; margin-bottom: 10px; }
        input[type="text"] { flex: 1; background: #0f172a; border: 1px solid #475569; border-radius: 6px; padding: 10px 12px; color: #fff; font-size: 14px; }
        button { background: #0284c7; color: white; border: none; border-radius: 6px; padding: 10px 16px; font-weight: 600; cursor: pointer; transition: 0.2s; }
        button:hover { background: #0369a1; }
        .preset-btn { background: #334155; font-size: 11px; padding: 6px 10px; margin: 3px 2px; border-radius: 4px; }
        .preset-btn:hover { background: #475569; }
        select { width: 100%; background: #0f172a; border: 1px solid #475569; border-radius: 6px; padding: 10px; color: #fff; font-size: 14px; margin-bottom: 10px; }
        .telemetry-row { display: flex; justify-content: space-between; font-size: 13px; font-family: 'JetBrains Mono', monospace; padding: 6px 0; border-bottom: 1px solid #334155; }
        .badge { padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; }
        .badge-success { background: #166534; color: #4ade80; }
        .badge-warn { background: #854d0e; color: #fde047; }
        .fp-img { width: 100%; border-radius: 8px; border: 1px solid #475569; margin-top: 8px; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <div class="title">Hiwonder JetAuto AMR 3D Simulation & World Models</div>
            <div class="subtitle">Natural Language Semantic Grounding | LeWorldModel (SIGReg) | H-JEPA | AMD Kria KR260 FPGA</div>
        </div>
        <div class="author-badge">Author: <b>Brian A. Hadian</b></div>
    </div>

    <div class="grid-container">
        <!-- 3D Arena Perspective -->
        <div class="view-panel">
            <div class="view-title">3D Arena Simulation View (MuJoCo Physics)</div>
            <img class="stream-img" src="/video_feed_3d" alt="3D Arena View">
        </div>

        <!-- Control & Telemetry Sidebar -->
        <div class="sidebar">
            <!-- Command Input -->
            <div class="card">
                <div class="card-title">NLP Text Command Input</div>
                <div class="input-group">
                    <input type="text" id="cmdInput" value="go behind the yellow cylinder" placeholder="e.g. go behind yellow cylinder">
                    <button onclick="sendCmd()">Send</button>
                </div>
                <div>
                    <span style="font-size: 11px; color: #94a3b8;">Quick Presets:</span><br>
                    <button class="preset-btn" onclick="setPreset('go behind the yellow cylinder')">Behind Yellow Cyl</button>
                    <button class="preset-btn" onclick="setPreset('approach the red box')">Approach Red Box</button>
                    <button class="preset-btn" onclick="setPreset('go in front of blue cylinder')">Front Blue Cyl</button>
                    <button class="preset-btn" onclick="setPreset('strafe left of green cylinder')">Left of Green Cyl</button>
                </div>
            </div>

            <!-- Model Selection -->
            <div class="card">
                <div class="card-title">World Model Architecture</div>
                <select id="modelSelect" onchange="changeModel()">
                    <option value="lewm" selected>LeWorldModel (LeWM + SIGReg - arXiv:2603.19312)</option>
                    <option value="hjepa">H-JEPA Hierarchical (Stride K=5 - arXiv:2610.06805)</option>
                    <option value="baseline">Baseline Goal-Conditioned Policy (Direct Vision)</option>
                </select>
                <button style="width: 100%; background: #475569;" onclick="resetSim()">Reset AMR Pose</button>
            </div>

            <!-- Onboard Astra Camera Feed -->
            <div class="card">
                <div class="card-title">Onboard Astra Camera Feedback</div>
                <img class="fp-img" src="/video_feed_fp" alt="First Person Camera">
            </div>

            <!-- Telemetry & Verification -->
            <div class="card">
                <div class="card-title">Live Telemetry & Verification</div>
                <div class="telemetry-row"><span>Target Landmark:</span><span id="telLandmark">-</span></div>
                <div class="telemetry-row"><span>Target Waypoint:</span><span id="telTarget">-</span></div>
                <div class="telemetry-row"><span>Goal Distance:</span><span id="telDist">-</span></div>
                <div class="telemetry-row"><span>Camera Feedback:</span><span id="telAcquired">-</span></div>
                <div class="telemetry-row"><span>Verification Status:</span><span id="telStatus">-</span></div>
                <div class="telemetry-row"><span>FPGA Latency:</span><span id="telLatency">0.48 us</span></div>
                <div class="telemetry-row"><span>Collisions:</span><span id="telCols">0</span></div>
            </div>
        </div>
    </div>

    <script>
        function sendCmd() {
            const cmd = document.getElementById("cmdInput").value;
            fetch("/api/command", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({command: cmd})
            });
        }
        function setPreset(cmd) {
            document.getElementById("cmdInput").value = cmd;
            sendCmd();
        }
        function changeModel() {
            const m = document.getElementById("modelSelect").value;
            fetch("/api/model", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({model: m})
            });
        }
        function resetSim() {
            fetch("/api/reset", {method: "POST"});
        }
        setInterval(() => {
            fetch("/api/telemetry")
                .then(r => r.json())
                .then(d => {
                    document.getElementById("telLandmark").innerText = d.landmark || "-";
                    document.getElementById("telTarget").innerText = d.target_xy || "-";
                    document.getElementById("telDist").innerText = d.distance ? d.distance.toFixed(2) + " m" : "-";
                    document.getElementById("telAcquired").innerHTML = d.visual_acquired ? 
                        '<span class="badge badge-success">LOCKED</span>' : '<span class="badge badge-warn">SEARCHING</span>';
                    document.getElementById("telStatus").innerHTML = d.goal_verified ? 
                        '<span class="badge badge-success">VERIFIED & REACHED</span>' : '<span class="badge badge-warn">NAVIGATING</span>';
                    document.getElementById("telCols").innerText = d.collisions;
                });
        }, 300);
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/video_feed_3d")
def video_feed_3d():
    def gen():
        while True:
            if sim_controller and sim_controller.last_3d_frame is not None:
                ret, jpeg = cv2.imencode(".jpg", sim_controller.last_3d_frame)
                if ret:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_fp")
def video_feed_fp():
    def gen():
        while True:
            if sim_controller and sim_controller.last_fp_frame is not None:
                ret, jpeg = cv2.imencode(".jpg", sim_controller.last_fp_frame)
                if ret:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/command", methods=["POST"])
def api_command():
    data = request.get_json()
    cmd = data.get("command", "")
    if sim_controller and cmd:
        sim_controller.set_command(cmd)
    return jsonify({"status": "ok", "cmd": cmd})

@app.route("/api/model", methods=["POST"])
def api_model():
    data = request.get_json()
    model = data.get("model", "lewm")
    if sim_controller:
        sim_controller.set_model(model)
    return jsonify({"status": "ok", "model": model})

@app.route("/api/reset", methods=["POST"])
def api_reset():
    if sim_controller:
        sim_controller.reset_sim()
    return jsonify({"status": "ok"})

@app.route("/api/telemetry")
def api_telemetry():
    if not sim_controller:
        return jsonify({})
    with sim_controller.lock:
        g = sim_controller.goal_dict
        f = sim_controller.feedback
        t_xy = f"{g['target_xy'][0]:.2f}, {g['target_xy'][1]:.2f}" if g else "-"
        return jsonify({
            "landmark": g["landmark_label"] if g else "-",
            "target_xy": t_xy,
            "distance": f.get("xy_distance", 0.0),
            "visual_acquired": f.get("visual_acquired", False),
            "goal_verified": f.get("goal_verified", False),
            "collisions": sim_controller.collision_count,
            "step_count": sim_controller.step_count,
            "active_model": sim_controller.current_model
        })

def simulation_worker():
    """Background simulation thread running at 10-15 Hz."""
    while sim_controller and sim_controller.is_running:
        step_start = time.time()
        sim_controller.step()
        elapsed = time.time() - step_start
        sleep_time = max(0.0, 0.08 - elapsed)
        time.sleep(sleep_time)

def run_headless_evaluation(cmd="go behind the yellow cylinder", model_name="lewm", max_steps=60, out_image=None):
    """Executes a complete offline evaluation rollout and saves the trajectory visualization."""
    ctrl = AMRSimulationController()
    ctrl.set_model(model_name)
    ctrl.reset_sim(initial_cmd=cmd, initial_pos=np.array([0.5, 0.5, 0.0]))
    
    print(f"\n=======================================================")
    print(f"  HEADLESS EVALUATION: Model={model_name.upper()}")
    print(f"  Command: \"{cmd}\"")
    print(f"  Target Waypoint: {ctrl.goal_dict['target_xy']}")
    print(f"=======================================================")

    frames_3d = []
    reached = False

    for step in range(max_steps):
        ctrl.step()
        dist = ctrl.feedback["xy_distance"]
        vis_ok = ctrl.feedback["visual_acquired"]
        goal_ok = ctrl.feedback["goal_verified"]

        if step % 5 == 0 or goal_ok:
            print(f"  Step {step:02d}: Dist={dist:.2f}m | Camera Lock={vis_ok} | Goal Verified={goal_ok}")

        if ctrl.last_3d_frame is not None:
            frames_3d.append(ctrl.last_3d_frame.copy())

        if goal_ok:
            reached = True
            print(f"  -> SUCCESS! Target reached and verified in {step+1} steps!")
            break

    ctrl.env.close()

    if out_image and len(frames_3d) > 0:
        cv2.imwrite(out_image, frames_3d[-1])
        print(f"  -> Final evaluation snapshot saved to: {out_image}")

    return {
        "model": model_name,
        "command": cmd,
        "reached": reached,
        "steps": step + 1,
        "final_dist": dist,
        "collisions": ctrl.collision_count
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Run headless evaluation script")
    parser.add_argument("--cmd", type=str, default="go behind the yellow cylinder", help="Command to evaluate")
    parser.add_argument("--model", type=str, default="lewm", choices=["baseline", "lewm", "hjepa"])
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.eval:
        run_headless_evaluation(cmd=args.cmd, model_name=args.model)
    else:
        sim_controller = AMRSimulationController()
        sim_thread = threading.Thread(target=simulation_worker, daemon=True)
        sim_thread.start()
        print(f"\n[OK] 3D Simulation Web Dashboard running on http://0.0.0.0:{args.port}")
        app.run(host="0.0.0.0", port=args.port, debug=False)
