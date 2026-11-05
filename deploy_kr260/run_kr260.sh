#!/usr/bin/env bash
# ==============================================================================
# AMD Kria KR260 Interactive Test & Program Runner
# Author: Brian A. Hadian
# Project: Hephaestus
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "========================================================================"
echo "  PROJECT HEPHAESTUS: AMD KRIA KR260 UNIFIED RUNNER"
echo "  Author: Brian A. Hadian"
echo "========================================================================"

# Check if running as root (required for /dev/mem physical access on Ubuntu 22.04)
if [[ $EUID -ne 0 ]]; then
    if [[ -c /dev/mem ]]; then
        echo "[INFO] Running as regular user. Physical /dev/mem access requires root/sudo."
        echo "[INFO] You can re-run with: sudo ./run_kr260.sh"
        echo "[INFO] Continuing... (will automatically use MOCK mode if /dev/mem access is denied)"
    fi
fi

# Make scripts executable
chmod +x kr260_menu.py kr260_pl_driver.py kr260_uart_driver.py kr260_fpga_manager.py 2>/dev/null || true

# Launch the interactive menu
python3 kr260_menu.py "$@"
