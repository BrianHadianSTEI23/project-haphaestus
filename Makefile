# ==============================================================================
# Makefile for Project Hephaestus - AMD Kria KR260 Robotics Accelerator
# Author: Brian A. Hadian
# ==============================================================================

.PHONY: all sim synth bd package test test-eth run clean help

all: sim

help:
	@echo "Project Hephaestus - Build & Verification Targets:"
	@echo "  make sim      : Run all Verilog HDL simulations via Icarus Verilog"
	@echo "  make synth    : Run AMD Vivado synthesis for Kria KR260 (xck26-sfvc784-2LV-c)"
	@echo "  make bd       : Create Vivado Zynq MPSoC Block Design with AXI connection"
	@echo "  make package  : Package bitstream, DTBO overlay and shell.json for KR260"
	@echo "  make test     : Run automated software & hardware test suite"
	@echo "  make test-eth : Run automated Ethernet link & UART failover regression test"
	@echo "  make run      : Launch interactive menu-driven KR260 test runner"
	@echo "  make clean    : Clean build artifacts and simulation dump files"

sim:
	@bash scripts/run_simulation.sh

synth:
	@vivado -mode batch -notrace -source scripts/synth_kr260.tcl

bd:
	@vivado -mode batch -notrace -source scripts/create_kr260_bd.tcl

package:
	@python3 scripts/package_kria_firmware.py

test:
	@./run_kr260.sh --auto --mock

test-eth:
	@python3 deploy_kr260/test_ethernet_failover.py

run:
	@./run_kr260.sh

clean:
	@rm -rf build/sim/*.vcd *.vcd build/sim/sim_* build/reports/*.rpt
	@rm -rf .Xil/ vivado*.log vivado*.jou
	@echo "Clean completed."
