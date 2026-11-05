#!/usr/bin/env bash
# ==============================================================================
# Project Hephaestus: Unified Laptop Verification Suite for AMD Kria KR260
# Author: Brian A. Hadian
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "========================================================================"
echo "  PROJECT HEPHAESTUS: AMD KRIA KR260 SIMULATION SUITE"
echo "  Author: Brian A. Hadian"
echo "========================================================================"
echo "Workspace: ${ROOT_DIR}"
echo ""

mkdir -p build/sim

echo ">>> [1/4] Running Neural Policy Core Verification (16 Golden Vectors)..."
iverilog -g2012 -o build/sim/sim_policy_core \
    sim/tb_kr260_policy_core.v rtl/igor_kr260_policy_core.v
vvp build/sim/sim_policy_core
echo ">>> Neural Policy Core: PASSED"
echo ""

echo ">>> [2/4] Running AXI4-Lite PS-PL High-Speed Bridge Verification..."
iverilog -g2012 -o build/sim/sim_axi_lite \
    sim/tb_kr260_axi_lite.v rtl/igor_kr260_top.v rtl/igor_kr260_policy_core.v \
    rtl/igor_axi_lite_bridge.v rtl/igor_safety_security_gate.v rtl/igor_status_logger.v \
    rtl/igor_kr260_policy_uart.v rtl/igor_uart_rx.v rtl/igor_uart_tx.v
vvp build/sim/sim_axi_lite
echo ">>> AXI4-Lite Bridge: PASSED"
echo ""

echo ">>> [3/4] Running UART Framing & CRC-8 Protocol Engine Verification..."
iverilog -g2012 -o build/sim/sim_uart \
    sim/tb_kr260_uart.v rtl/igor_kr260_policy_uart.v rtl/igor_kr260_policy_core.v \
    rtl/igor_uart_rx.v rtl/igor_uart_tx.v
vvp build/sim/sim_uart
echo ">>> UART Framing Engine: PASSED"
echo ""

echo ">>> [4/4] Running Hardware Safety & Security Gate Verification..."
iverilog -g2012 -o build/sim/sim_safety \
    sim/tb_kr260_safety_gate.v rtl/igor_safety_security_gate.v
vvp build/sim/sim_safety
echo ">>> Hardware Safety Gate: PASSED"
echo ""

echo "========================================================================"
echo ">> ALL LAPTOP HDL SIMULATION TESTS PASSED SEAMLESSLY (100% VERIFIED)! <<"
echo "========================================================================"
