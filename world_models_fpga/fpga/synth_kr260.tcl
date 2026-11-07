# ============================================================================
# Script: synth_kr260.tcl
# Target: AMD Kria KR260 SOM (Zynq UltraScale+ xck26-sfvc784-2LV-c)
# Synthesizes top_world_model_accelerator at 200 MHz (5.0 ns clock period)
# and exports post-synthesis utilization, timing, and power reports.
# ============================================================================

set part_name "xck26-sfvc784-2LV-c"
set out_dir "world_models_fpga/research/hardware"
file mkdir $out_dir

read_verilog -sv world_models_fpga/fpga/rtl/macro_predictor_core.v
read_verilog -sv world_models_fpga/fpga/rtl/micro_steering_core.v
read_verilog -sv world_models_fpga/fpga/rtl/top_world_model_accelerator.v

synth_design -top top_world_model_accelerator -part $part_name -mode out_of_context

create_clock -period 5.000 -name clk [get_ports clk]

report_utilization -file "$out_dir/vivado_utilization.rpt"
report_timing_summary -file "$out_dir/vivado_timing.rpt"
report_power -file "$out_dir/vivado_power.rpt"

puts "=== VIVADO SYNTHESIS COMPLETED SUCCESSFULLY ==="
exit
