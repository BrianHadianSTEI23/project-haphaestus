#!/usr/bin/env bash
# ============================================================================
# Script: run_demo.sh
# Project: World Models & Language Navigation on AMD Kria KR260 FPGA
# Author: Brian A. Hadian
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="/home/hadynata/vs-code-data/hdl/project-haphaestus/.venv"

echo "======================================================================"
echo "  Hiwonder JetAuto AMR - 3D World Models & Language Navigation"
echo "  Author: Brian A. Hadian"
echo "======================================================================"

source "${VENV_PATH}/bin/activate"

# Ensure headless EGL rendering if display server is absent
export MUJOCO_GL="${MUJOCO_GL:-egl}"

PORT="${PORT:-8080}"
echo "[1/2] Starting Interactive Mechanistic Studio & 3D Simulation Web Dashboard on http://localhost:${PORT}"
echo "      - Interactive JavaScript Neural Architecture Engine: Live neuron firing & synapse pulses"
echo "      - Interactive 3D Linear Space Design: 3D Orbit, Pan, Zoom, Safety Barrier Hyperplane B(s,a)=0.5"
echo "      - Full Cross-Model Suite: Baseline vs. LeWM vs. H-JEPA vs. Option A vs. Option B (KR260 FPGA) vs. Novel Composition"
echo "      - Deep Weight & Data Transformation Inspection: Hover on neurons to view weights and linear output vectors"
echo "      - Live MuJoCo 3D Arena & Onboard Astra RGB-D Camera HUD"
echo "      - Natural Language Command Dispatching: Live steering and waypoint routing"
echo "----------------------------------------------------------------------"
echo "Author: Brian A. Hadian"
echo "Press Ctrl+C to terminate the dashboard."

python3 "${SCRIPT_DIR}/sim_app_3d/sim_app_3d.py" --port ${PORT}
