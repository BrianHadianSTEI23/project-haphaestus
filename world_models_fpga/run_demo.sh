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

PORT=8080
echo "[1/3] Starting Interactive 3D Simulation Web Dashboard on http://localhost:${PORT}"
echo "      - Real-time 3D MuJoCo Arena Perspective (30 FPS Stream)"
echo "      - Onboard Astra Camera Feed with Target Centroid & Bounding Box HUD"
echo "      - Text Command Input: e.g. 'go behind the yellow cylinder'"
echo "      - Model Selector: Baseline vs. LeWorldModel (LeWM) vs. H-JEPA"
echo "----------------------------------------------------------------------"
echo "Press Ctrl+C to terminate the dashboard."

python3 "${SCRIPT_DIR}/sim_app_3d/sim_app_3d.py" --port ${PORT}
