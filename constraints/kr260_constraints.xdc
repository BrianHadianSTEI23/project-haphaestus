# ==============================================================================
# AMD Kria KR260 Constraints File - Project Hephaestus
# Author: Brian A. Hadian
# Target Silicon: AMD Zynq UltraScale+ MPSoC (xck26-sfvc784-2LV-c)
# Carrier Board: AMD Kria KR260 Robotics Starter Kit
# ==============================================================================

# ------------------------------------------------------------------------------
# Clock & Timing Constraints
# ------------------------------------------------------------------------------
# 50 MHz PL Base Clock (20 ns period)
create_clock -period 20.000 -name clk [get_ports clk]

# ------------------------------------------------------------------------------
# Pmod Header Serial Communication & Emergency Stop (Bank 45 - LVCMOS33)
# Pin 1: UART RX (from external Jetson / STM32 / Host)
# Pin 2: UART TX (to external host)
# Pin 3: E-Stop Physical Button (Active-low with pull-up)
# ------------------------------------------------------------------------------
set_property PACKAGE_PIN H12 [get_ports pmod_uart_rx]
set_property IOSTANDARD LVCMOS33 [get_ports pmod_uart_rx]
set_property PULLUP true [get_ports pmod_uart_rx]

set_property PACKAGE_PIN G11 [get_ports pmod_uart_tx]
set_property IOSTANDARD LVCMOS33 [get_ports pmod_uart_tx]
set_property DRIVE 8 [get_ports pmod_uart_tx]
set_property SLEW SLOW [get_ports pmod_uart_tx]

set_property PACKAGE_PIN F11 [get_ports estop_btn_n]
set_property IOSTANDARD LVCMOS33 [get_ports estop_btn_n]
set_property PULLUP true [get_ports estop_btn_n]

# ------------------------------------------------------------------------------
# Status Diagnostic LEDs (Bank 45 - LVCMOS33)
# LED[0]: Heartbeat (~1.5 Hz blink)
# LED[1]: Safety Permit (1 = Safe Actuation Permitted)
# LED[2]: Core Done
# LED[3]: Core Busy
# LED[7:4]: Active Fault Code lower nibble
# ------------------------------------------------------------------------------
set_property PACKAGE_PIN J10 [get_ports {led_status[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led_status[0]}]

set_property PACKAGE_PIN J11 [get_ports {led_status[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led_status[1]}]

set_property PACKAGE_PIN K12 [get_ports {led_status[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led_status[2]}]

set_property PACKAGE_PIN K13 [get_ports {led_status[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led_status[3]}]

set_property PACKAGE_PIN G10 [get_ports {led_status[4]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led_status[4]}]

set_property PACKAGE_PIN H11 [get_ports {led_status[5]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led_status[5]}]

set_property PACKAGE_PIN J12 [get_ports {led_status[6]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led_status[6]}]

set_property PACKAGE_PIN F10 [get_ports {led_status[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led_status[7]}]
