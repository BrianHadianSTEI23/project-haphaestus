# ==============================================================================
# Vivado Synthesis & Implementation Script for AMD Kria KR260
# Author: Brian A. Hadian
# ==============================================================================

set SCRIPT_DIR [file dirname [info script]]
set ROOT_DIR   [file normalize "${SCRIPT_DIR}/.."]

set PART_NAME  "xck26-sfvc784-2LV-c"
set TOP_MODULE "igor_kr260_top"

puts "========================================================================"
puts "  AMD KRIA KR260 SYNTHESIS - PROJECT HEPHAESTUS"
puts "  Target Part: ${PART_NAME}"
puts "  Author: Brian A. Hadian"
puts "========================================================================"

# Create in-memory project
create_project -in_memory -part ${PART_NAME}

# Add Verilog Sources
read_verilog -sv "${ROOT_DIR}/rtl/igor_kr260_top.v"
read_verilog -sv "${ROOT_DIR}/rtl/igor_kr260_policy_core.v"
read_verilog -sv "${ROOT_DIR}/rtl/igor_axi_lite_bridge.v"
read_verilog -sv "${ROOT_DIR}/rtl/igor_safety_security_gate.v"
read_verilog -sv "${ROOT_DIR}/rtl/igor_status_logger.v"
read_verilog -sv "${ROOT_DIR}/rtl/igor_kr260_policy_uart.v"
read_verilog -sv "${ROOT_DIR}/rtl/igor_uart_rx.v"
read_verilog -sv "${ROOT_DIR}/rtl/igor_uart_tx.v"

# Add Weight Memory Files
foreach bank {0 1 2 3 4 5 6 7} {
    add_files "${ROOT_DIR}/rtl/mem/bank${bank}.mem"
    set_property file_type {Memory File} [get_files "*bank${bank}.mem*"]
}

# Add Constraints
read_xdc "${ROOT_DIR}/constraints/kr260_constraints.xdc"

# Run Out-Of-Context Synthesis for the KR260 Compute Core
puts ">>> Running Out-Of-Context Synthesis for ${TOP_MODULE}..."
synth_design -top ${TOP_MODULE} -part ${PART_NAME} -mode out_of_context

# Generate Utilization & Timing Reports
file mkdir "${ROOT_DIR}/build/reports"
report_utilization -file "${ROOT_DIR}/build/reports/utilization_synth.rpt"
report_timing_summary -file "${ROOT_DIR}/build/reports/timing_synth.rpt"

puts ">>> Synthesis complete! Reports written to build/reports/"
exit
