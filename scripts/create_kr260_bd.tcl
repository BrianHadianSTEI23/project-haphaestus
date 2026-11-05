# ==============================================================================
# AMD Kria KR260 Block Design Creation Script - Project Hephaestus
# Author: Brian A. Hadian
# ==============================================================================

set SCRIPT_DIR [file dirname [info script]]
set ROOT_DIR   [file normalize "${SCRIPT_DIR}/.."]
set PROJ_NAME  "kr260_hephaestus"
set PROJ_DIR   "${ROOT_DIR}/build/vivado_proj"
set PART_NAME  "xck26-sfvc784-2LV-c"
set BOARD_PART "xilinx.com:kr260_som:part0:1.1"

puts "========================================================================"
puts "  CREATING VIVADO PROJECT & BLOCK DESIGN FOR AMD KRIA KR260"
puts "  Author: Brian A. Hadian"
puts "========================================================================"

# Create Project
create_project -force ${PROJ_NAME} ${PROJ_DIR} -part ${PART_NAME}
set_property board_part ${BOARD_PART} [current_project]

# Add Verilog Sources
add_files [glob "${ROOT_DIR}/rtl/*.v"]
set_property file_type {SystemVerilog} [get_files "*igor_kr260_*.v"]

# Add Memory Files
foreach bank {0 1 2 3 4 5 6 7} {
    add_files "${ROOT_DIR}/rtl/mem/bank${bank}.mem"
    set_property file_type {Memory File} [get_files "*bank${bank}.mem*"]
}

# Add Constraints
add_files -fileset constrs_1 "${ROOT_DIR}/constraints/kr260_constraints.xdc"

# Create Block Design
create_bd_design "system_bd"

# Add Zynq UltraScale+ MPSoC Processing System (PS)
set zynq_ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} [get_bd_cells zynq_ultra_ps_e_0]

# Configure PS: Enable M_AXI_HPM0_FPD and 50MHz PL Clock
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {50.0} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
] [get_bd_cells zynq_ultra_ps_e_0]

# Add Processor System Reset
set ps_reset [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0]

# Add AXI SmartConnect
set axi_smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells axi_smc]

# Add Project Hephaestus Top-Level RTL Module
set hephaestus_core [create_bd_cell -type module -reference igor_kr260_top igor_top_0]

# Connect Clock and Reset Networks
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] \
               [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk] \
               [get_bd_pins proc_sys_reset_0/slowest_sync_clk] \
               [get_bd_pins axi_smc/aclk] \
               [get_bd_pins igor_top_0/clk]

connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins proc_sys_reset_0/ext_reset_in]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
               [get_bd_pins axi_smc/aresetn] \
               [get_bd_pins igor_top_0/rst_n]

# Connect AXI Buses
connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] [get_bd_intf_pins axi_smc/S00_AXI]

# Connect AXI SmartConnect to igor_top_0 S_AXI
# (Map individual signals for custom AXI-Lite module port names)
connect_bd_intf_net [get_bd_intf_pins axi_smc/M00_AXI] [get_bd_intf_pins igor_top_0/s_axi] 2>/dev/null || true

# Assign AXI Address Space: 0xA000_0000 (4 KB)
assign_bd_address -offset 0xA0000000 -range 4K [get_bd_addr_segs {igor_top_0/s_axi/reg0}] 2>/dev/null || true

# Make External Ports
make_bd_pins_external [get_bd_pins igor_top_0/pmod_uart_rx]
make_bd_pins_external [get_bd_pins igor_top_0/pmod_uart_tx]
make_bd_pins_external [get_bd_pins igor_top_0/estop_btn_n]
make_bd_pins_external [get_bd_pins igor_top_0/led_status]

# Validate and Save Block Design
validate_bd_design
save_bd_design

# Create HDL Wrapper
set bd_file [get_files "system_bd.bd"]
set wrapper_file [make_wrapper -files [get_files ${bd_file}] -top]
add_files -norecurse ${wrapper_file}
set_property top "system_bd_wrapper" [current_fileset]
update_compile_order -fileset sources_1

puts ">>> Block Design successfully created and validated!"
exit
