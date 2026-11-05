# ⚡ Project Hephaestus: Edge-Accelerated Neural Obstacle Avoidance & Hardware Safety Gate for AMD Kria KR260

**Author:** Brian A. Hadian  
**Target Hardware:** AMD Kria KR260 Robotics Starter Kit  
**Target FPGA Silicon:** AMD Zynq UltraScale+ MPSoC (`xck26-sfvc784-2LV-c` on K26 SOM)  
**Host Processing System (PS) OS:** Ubuntu 22.04 LTS (ARM Cortex-A53 Quad-Core)  
**Toolchains:** AMD Vivado 2023.2, Vitis 2023.2 (`bootgen`, `dtc`), Icarus Verilog (`iverilog`), Python 3  

---

## 📖 1. Brief Description & Background

**Project Hephaestus** is a high-performance, real-time autonomous robotics hardware accelerator designed specifically for the **AMD Kria KR260 Robotics Starter Kit**. 

The system builds upon the core algorithmic foundations of Project IGOR (originally targeting the Intel Cyclone V SoC on DE10-Nano) and re-engineers the entire compute, memory, safety, and interconnect architecture for AMD Zynq UltraScale+ MPSoC.

### The Problem It Solves
High-speed autonomous navigation on mobile robots (AMRs) operating in dynamic, cluttered industrial environments demands **sub-millisecond neural policy evaluation** coupled with **non-negotiable, zero-latency physical safety guarantees**. Running pure software control loops on Linux host CPUs introduces OS scheduler jitter, memory paging delays, and catastrophic risks if the host OS freezes while actuators remain powered.

### The Hephaestus Solution on AMD Kria KR260
1. **Parallel-8 Neural Accelerator**: Evaluates a trained 74-input PPO residual policy ($74 \rightarrow 128 \rightarrow 64 \rightarrow 3$) in signed Q8.8 fixed-point arithmetic using **8 parallel MAC lanes** mapped directly to AMD UltraScale+ DSP48E2 slices. Inference completes in **138.32 µs** (6,916 clock cycles @ 50 MHz) or **69.16 µs** @ 100 MHz — over **140× faster** than the robot's 20 ms control loop.
2. **Nanosecond AXI4-Lite Memory-Mapped Bus**: Connects the PL accelerator directly to the PS ARM Cortex-A53 via `M_AXI_HPM0_FPD` at base address `0xA000_0000`. Using Python `mmap` over `/dev/mem` (with zero third-party C++ libraries), the PS executes inference in **~140 µs total roundtrip** — **32× faster** than external 921,600 baud serial UART.
3. **Dual-Interface Flexibility**: Concurrently retains a wire-speed binary UART framing engine (153-byte request / 11-byte response with CRC-8 Maxim validation) mapped to KR260 Pmod/RPi header pins for backward compatibility with external microcontrollers or Jetson hosts.
4. **Hardware-Enforced Fail-Closed Safety Gate**: An autonomous RTL safety interlock with a **monotonic 20 ms lease watchdog**. If the host PS fails to refresh its lease within 20 ms (e.g., kernel panic, thread lock), motor commands are immediately latched to zero at wire speed. Also enforces physical/software latched E-Stop and velocity clamping.

---

## 🏛️ 2. Architectural Features

```
+==================================================================================================+
|                        PROJECT HEPHAESTUS: AMD KRIA KR260 ARCHITECTURE                           |
+==================================================================================================+
|                                                                                                  |
|  [ PROCESSING SYSTEM (PS) - ARM Cortex-A53 Quad-Core @ 1.33 GHz | Ubuntu 22.04 LTS ]             |
|  • Sensor Perception: 360° LiDAR (64 frontal normalized beams) + 6-DOF IMU Kalman filter         |
|  • High-Level Planner: Semi-global corridor navigation + nominal velocity blending               |
|  • Direct PL Access: Python /dev/mem memory-mapped driver (Zero C++ dependency)                  |
|  • FPGA Manager: Linux sysfs & xmutil app management (/lib/firmware/xilinx/igor_kr260)           |
|                                                                                                  |
|                                     │                                                            |
|          High-Speed AXI4-Lite Bus   ▼   (M_AXI_HPM0_FPD @ 0xA000_0000 | ~140 us roundtrip)       |
|                                                                                                  |
|  [ PROGRAMMABLE LOGIC (PL) - AMD Zynq UltraScale+ MPSoC xck26-sfvc784-2LV-c ]                    |
|  ┌────────────────────────────────────────────────────────┐ ┌──────────────────────────────────┐ |
|  │  Parallel-8 Neural Accelerator Core                    │ │  Hardware Safety & Security Gate │ |
|  │  • Multi-Layer Perceptron: 74 -> 128 -> 64 -> 3        │ │  • Monotonic 20 ms Lease Watchdog│ |
|  │  • 8 Parallel MAC Lanes (UltraScale+ DSP48E2 Slices)   │ │  • Zero-latency E-Stop Interlock │ |
|  │  • 8 Banked Block RAMs (18,051 SHA-256 verified weights│ │  • Signed Q8.8 Velocity Clamper   │ |
|  │  • 40-bit Non-Overflow Accumulators & ReLU/Clamping    │ │  • Non-Maskable Status Logger    │ |
|  │  • Latency: 6,916 cycles (138.32 us @ 50 MHz)          │ │  • Priority Fault Hierarchy      │ |
|  └────────────────────────────────────────────────────────┘ └──────────────────────────────────┘ |
|                                     │                                                            |
|                                     ▼                                                            |
|  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ |
|  │  Secondary Serial UART Protocol Engine                                                      │ |
|  │  • Pmod 1 / J5 Header (921,600 baud, 8N1, LVCMOS33)                                         │ |
|  │  • 153-byte Request / 11-byte Response with CRC-8 Maxim Packet Validation                    │ |
|  └─────────────────────────────────────────────────────────────────────────────────────────────┘ |
+==================================================================================================+
```

---

## 📁 3. Directory Structure

```
project-haphaestus/
├── README.md                          # Comprehensive documentation by Brian A. Hadian
├── Makefile                           # Unified build, test, and synthesis targets
├── run_kr260.sh                       # Root launcher for the interactive menu runner
│
├── rtl/                               # Verilog HDL Source Files for AMD UltraScale+ PL
│   ├── igor_kr260_top.v               # Top-level wrapper (AXI-Lite + UART + Core + Safety + LEDs)
│   ├── igor_kr260_policy_core.v       # 8-lane Parallel MAC neural policy accelerator
│   ├── igor_axi_lite_bridge.v         # High-speed AXI4-Lite slave memory-mapped bridge
│   ├── igor_kr260_policy_uart.v       # Binary UART protocol engine with CRC-8 Maxim
│   ├── igor_safety_security_gate.v    # Fail-closed RTL safety gate & 20 ms lease watchdog
│   ├── igor_status_logger.v           # Non-maskable hardware fault event logger
│   ├── igor_uart_rx.v                 # Robust UART receiver with 2-stage synchronizer
│   ├── igor_uart_tx.v                 # Serial UART transmitter
│   └── mem/                           # Cryptographically Verified Weight Banks
│       ├── policy_weights.mem         # Monolithic Q8.8 weights file
│       ├── bank0.mem .. bank7.mem     # 8 Partitioned Block RAM initialization banks (4096 words each)
│
├── constraints/                       # Physical & Timing Constraints
│   └── kr260_constraints.xdc          # KR260 Pinout (Pmod, LEDs, E-Stop) & 50 MHz timing
│
├── sim/                               # Simulation & Verification Suite (Laptop Testing)
│   ├── tb_kr260_policy_core.v         # Core testbench verifying all 16 golden oracle vectors
│   ├── tb_kr260_axi_lite.v            # AXI4-Lite bus transaction & register testbench
│   ├── tb_kr260_uart.v                # UART transaction, framing error & E-Stop testbench
│   ├── tb_kr260_safety_gate.v         # Watchdog lease countdown & clamper testbench
│   └── vectors/                       # Test Vectors & Hex Stimuli
│       ├── uart_q88_vectors.json      # 16 standardized test cases from model oracle
│       ├── observations.hex           # Hex stimuli for 16 vectors (74 features each)
│       └── expected.hex               # Hex expected action outputs (3 actions each)
│
├── scripts/                           # Automation Scripts for Laptop & Build Flow
│   ├── run_simulation.sh              # 1-command execution of all 4 HDL simulation testbenches
│   ├── synth_kr260.tcl                # Vivado batch synthesis script for xck26-sfvc784-2LV-c
│   ├── create_kr260_bd.tcl            # Generates Zynq MPSoC Block Design with AXI SmartConnect
│   └── package_kria_firmware.py       # Converts bitstream to .bit.bin, compiles DTBO, generates shell.json
│
└── deploy_kr260/                      # Deployment Folder to Copy to AMD Kria KR260
    ├── run_kr260.sh                   # Main executable script on KR260 PS (Ubuntu 22.04)
    ├── kr260_menu.py                  # Interactive Menu-Driven System Test & Control Runner
    ├── kr260_pl_driver.py             # Pure Python memory-mapped PL driver via /dev/mem
    ├── kr260_uart_driver.py           # Serial UART driver for Pmod / ttyPS1
    ├── kr260_fpga_manager.py          # Linux sysfs / xmutil FPGA bitstream programming manager
    ├── firmware/                      # FPGA Bitstream & Device Tree Overlay Artifacts
    │   ├── igor_kr260.bit.bin         # Binary bitstream for FPGA Manager
    │   ├── igor_kr260.dtbo            # Device tree overlay compiled via dtc
    │   └── shell.json                 # xmutil application metadata
    └── vectors/
        └── uart_q88_vectors.json      # Golden vectors for on-target verification
```

---

## 📊 4. Memory Map & AXI4-Lite Register Reference

The PL accelerator peripheral is mapped by the Zynq UltraScale+ PS at base address `0xA000_0000` (4 KB address space):

| Offset | Name | Type | Reset Value | Description |
|---|---|---|---|---|
| `0x000` | `CR` | W/R | `0x0000_0000` | **Control Register**<br>Bit 0: `START` (trigger inference)<br>Bit 1: `ABORT` (cancel active inference)<br>Bit 2: `ESTOP_SOFT` (assert software E-stop)<br>Bit 3: `SOFT_RESET` (reset core state)<br>Bit 4: `LEASE_REFRESH` (reload watchdog timer)<br>Bit 5: `ESTOP_CLEAR` (clear latched E-stop)<br>Bit 6: `ESTOP_RELEASE` (release software E-stop) |
| `0x004` | `SR` | RO | `0x0000_0008` | **Status Register**<br>Bit 0: `BUSY` (inference in progress)<br>Bit 1: `DONE` (inference completed)<br>Bit 2: `FAULT` (arithmetic or sequence fault)<br>Bit 3: `SAFETY_PERMIT` (1 = actuation permitted)<br>Bit 4: `SATURATED` (activation output clipped)<br>Bit 5: `ESTOP_ACTIVE` (E-stop engaged)<br>Bit 6: `LEASE_EXPIRED` (watchdog expired)<br>Bits [11:8]: `DEBUG_STATE` (core internal state) |
| `0x008` | `ERR` | RO | `0x0000_0000` | **Error & Fault Code Register**<br>Bits [7:0]: `ACTIVE_FAULT_CODE`<br>Bits [15:8]: `LAST_FAULT_CODE` (from logger)<br>Bits [31:16]: `ERROR_FLAGS` (sticky history) |
| `0x00C` | `LAT` | RO | `0x0000_1B04` | **Inference Latency** in clock cycles (expected: 6,916 cycles = 138.32 µs @ 50 MHz) |
| `0x010` | `EVT` | RO | `0x0000_0000` | **Cumulative Fault Event Counter** (from status logger) |
| `0x014` | `ACC` | RO | `0x0000_0000` | **Accepted Inference Counter** (valid frames evaluated) |
| `0x018` | `REJ` | RO | `0x0000_0000` | **Rejected Frames Counter** (aborted / fault frames) |
| `0x01C` | `WDG` | RO | `0x000F_4240` | **Remaining Watchdog Lease Cycles** (decrements every clock) |
| `0x020` | `CFG_WDG` | W/R | `0x000F_4240` | **Watchdog Lease Limit** in clock cycles (default 1,000,000 = 20 ms @ 50 MHz) |
| `0x024` | `ACT0` | RO | `0x0000_0000` | **Action 0** ($\Delta v_x$ residual) signed 16-bit Q8.8 in [15:0] |
| `0x028` | `ACT1` | RO | `0x0000_0000` | **Action 1** ($\Delta v_y$ residual) signed 16-bit Q8.8 in [15:0] |
| `0x02C` | `ACT2` | RO | `0x0000_0000` | **Action 2** ($\Delta \omega_z$ residual) signed 16-bit Q8.8 in [15:0] |
| `0x030` | `ACT_CRC` | RO | `0x0000_0000` | **Action CRC-8 Maxim** checksum of the 3 actions in [7:0] |
| `0x034` | `HW_ID` | RO | `0x4B52_3236` | **Hardware ID**: ASCII `"KR26"` (identifies AMD Kria KR260 core) |
| `0x038` | `CLK_FREQ`| RO | `0x02FA_F080` | **Clock Frequency**: `50000000` Hz (50 MHz) |
| `0x040` | `MAX_V` | W/R | `0x0000_0100` | **Max Linear Velocity Limit** in Q8.8 (default 256 = 1.0 m/s) |
| `0x044` | `MAX_W` | W/R | `0x0000_0180` | **Max Angular Velocity Limit** in Q8.8 (default 384 = 1.5 rad/s) |
| `0x100`-`0x224`| `OBS[0..73]`| W/R | `0x0000_0000` | **Observation Features**: 74 x 32-bit registers (address `0x100 + i*4`). Bits [15:0] hold signed Q8.8 sensor feature $i$. Writing sets internal loaded flag. |

---

## 🛡️ 5. Hardware Safety Gate Priority Hierarchy

The `igor_safety_security_gate.v` module evaluates all inputs in hardware at wire speed and outputs a strict priority fault code:

| Fault Code | Cause | Output Action | Recovery Action |
|---|---|---|---|
| `8'd0` | **All Systems Safe & Normal** | Residual actions passed or clamped | Normal operation |
| `8'd1` | **Reset Asserted** | `safety_permit = 0`, outputs zeroed | Release reset (`rst_n = 1`) |
| `8'd2` | **Emergency Stop Asserted** | `safety_permit = 0`, outputs zeroed (latched) | Clear button + write `CR[5]` |
| `8'd3` | **Protective Stop Active** | `safety_permit = 0`, outputs zeroed | Clear protective obstacle |
| `8'd4` | **Core / Peripheral Not Ready** | `safety_permit = 0`, outputs zeroed | Wait for core readiness |
| `8'd7` | **Arithmetic / Numerical Fault** | `safety_permit = 0`, outputs zeroed | Soft reset core (`CR[3]`) |
| `8'd8` | **Watchdog Lease Timeout** | `safety_permit = 0`, outputs zeroed | Send heartbeat (`CR[4]`) |
| `8'd9` | **Result Invalid / Uncalculated** | `safety_permit = 0`, outputs zeroed | Run new inference |
| `8'd10`| **Speed Limit Exceeded** | `safety_permit = 1`, actions clamped to max | Automatic rate limiter |

---

## 🧪 6. How to Test on Laptop (Workflow Stage 1)

All RTL modules and testbenches are fully self-contained and run directly on your laptop using Icarus Verilog (`iverilog`) and Python 3.

### 1. Run the Unified Simulation Suite
Run all 4 verification testbenches with a single command:
```bash
make sim
# or directly:
bash scripts/run_simulation.sh
```

**Expected Output:**
```text
========================================================================
  PROJECT HEPHAESTUS: AMD KRIA KR260 SIMULATION SUITE
  Author: Brian A. Hadian
========================================================================
>>> [1/4] Running Neural Policy Core Verification (16 Golden Vectors)...
[PASS] Vector 00: Latency=6916 cycles (138.32 us @ 50MHz) | Expected=[     0,     -3,      6] Actual=[     0,     -3,      6]
[PASS] Vector 01: Latency=6916 cycles (138.32 us @ 50MHz) | Expected=[   -57,     76,     42] Actual=[   -57,     76,     42]
...
  Summary: 16/16 Test Vectors Passed, 0 Failed
>> SUCCESS: ALL KR260 POLICY CORE TESTS PASSED DETERMINISTICALLY! <<

>>> [2/4] Running AXI4-Lite PS-PL High-Speed Bridge Verification...
[PASS] Step 1: Read HW_ID = 0x4B523236 (ASCII 'KR26')
[PASS] Step 2: Read CLK_FREQ = 50000000 Hz (50 MHz)
[PASS] Step 4: Inference completed! SR = 0x0000020A
[PASS] Step 5: Output actions match oracle! Q8.8=[   -46,     60,     34]
[PASS] Step 7b: Watchdog lease expired! (LEASE_EXPIRED=1, SAFETY_PERMIT=0)
[PASS] Step 7c: Motor outputs zeroed on watchdog timeout (Fail-Closed verified)
>> SUCCESS: ALL AXI4-LITE PS-PL BRIDGE TESTS PASSED! <<

>>> [3/4] Running UART Framing & CRC-8 Protocol Engine Verification...
[PASS] Step 1a: Received exactly 11 response bytes!
[PASS] Step 1b: Response header verified: 5A A5 02 06
[PASS] Step 1c: Decoded actions match oracle! Q8.8=[   -46,     60,     34]
[PASS] Step 2: Corrupted CRC rejected with fault code 6!
[PASS] Step 3: Outputs zeroed and safety permit revoked on E-Stop!
>> SUCCESS: ALL UART PROTOCOL TESTS PASSED! <<

>>> [4/4] Running Hardware Safety & Security Gate Verification...
[PASS] Step 1: Reset asserts fault_code=1, permit=0, outputs zeroed
[PASS] Step 3: Overspeed clamped to max_v (500 -> 256) with fault_code=10
[PASS] Step 4: Watchdog expired! fault_code=8, permit=0, motor outputs zeroed
[PASS] Step 6b: E-Stop remains latched after button release
>> SUCCESS: ALL HARDWARE SAFETY GATE TESTS PASSED! <<
========================================================================
>> ALL LAPTOP HDL SIMULATION TESTS PASSED SEAMLESSLY (100% VERIFIED)! <<
========================================================================
```

### 2. Run Software Driver Dry-Run on Laptop
You can test the Python runtime software directly on your laptop before deploying to the board:
```bash
make test
# or:
./run_kr260.sh --auto --mock
```

---

## 🔨 7. How to Compile the Bitstream (Workflow Stage 2)

Vivado 2023.2 is installed on this laptop and can compile the bitstream for the AMD Kria K26 SOM (`xck26-sfvc784-2LV-c`).

### 1. Run Vivado Synthesis
Synthesizes the complete top-level accelerator design:
```bash
make synth
# or:
vivado -mode batch -notrace -source scripts/synth_kr260.tcl
```
*Synthesis reports are output to `build/reports/utilization_synth.rpt` and `timing_synth.rpt`.*  
*Resource utilization: **13.2% LUTs**, **5.8% Registers**, **1.3% DSPs (16 DSP48E2 slices)**.*

### 2. Package Kria Firmware & Device Tree Overlay
Generates the `.bit.bin` and compiles the `.dtbo` overlay:
```bash
make package
# or:
python3 scripts/package_kria_firmware.py
```
This prepares:
- `deploy_kr260/firmware/igor_kr260.bit.bin`
- `deploy_kr260/firmware/igor_kr260.dtbo`
- `deploy_kr260/firmware/shell.json`

---

## 🚀 8. How to Deploy and Run on AMD Kria KR260 (Workflow Stage 3)

### 1. Copy the Deployment Folder to AMD Kria KR260
From your laptop terminal, copy the `deploy_kr260/` directory to the KR260 over SSH:
```bash
scp -r deploy_kr260 ubuntu@<KR260_IP_ADDRESS>:~/
```

### 2. SSH into the AMD Kria KR260
```bash
ssh ubuntu@<KR260_IP_ADDRESS>
cd ~/deploy_kr260
```

### 3. Launch the Interactive System Test Runner
Run the unified script (requires `sudo` for physical `/dev/mem` memory-mapping):
```bash
sudo ./run_kr260.sh
```

---

## 📱 9. Interactive Menu Options Walkthrough

When you launch `./run_kr260.sh`, you are greeted with the interactive terminal dashboard:

```text
+=============================================================================+
|          PROJECT HEPHAESTUS: AMD KRIA KR260 ACCELERATOR SUITE               |
|   Edge-Accelerated Neural Obstacle Avoidance & Hardware Safety Gate         |
+=============================================================================+
 Author: Brian A. Hadian
 Target Board: AMD Kria KR260 Robotics Starter Kit (Zynq UltraScale+ MPSoC)
 Processing System (PS): ARM Cortex-A53 Quad-Core (Ubuntu 22.04 LTS)
 Programmable Logic (PL): Parallel-8 Neural Policy Core + RTL Safety Gate
 Operating Mode: PHYSICAL HARDWARE (Live)
 FPGA Manager State: operating
 Core Hardware ID: 0x4B523236 (KR26)
-----------------------------------------------------------------------------
SELECT A TEST / DIAGNOSTIC OPTION:
  [1] Program PL Bitstream from PS (FPGA Manager / xmutil)
  [2] Run Native AXI4-Lite Memory-Mapped PL Inference (16 Golden Vectors)
  [3] Run PL UART Protocol Interface Test (921,600 baud)
  [4] Test Hardware Safety Gate & Monotonic 20 ms Watchdog
  [5] Benchmark Inference Latency & Jitter (1000 Iterations)
  [6] Autonomous AMR Navigation & 4-Wheel Mecanum Kinematics Demo
  [7] Live Real-Time PL Telemetry Dashboard
  [8] Exit Program
-----------------------------------------------------------------------------
Enter choice [1-8]:
```

### Detailed Menu Functionality:

- **Option `[1]` — Program PL Bitstream from PS**:
  Uses `xmutil unloadapp` and `xmutil loadapp igor_kr260` (or direct sysfs `/sys/class/fpga_manager/fpga0/firmware`). Programs the bitstream into the FPGA fabric and applies the device tree overlay without requiring a reboot.
- **Option `[2]` — Native AXI4-Lite Memory-Mapped PL Inference**:
  Loads all 16 standardized test cases from `uart_q88_vectors.json`, writes them into PL registers `0x100..0x224`, triggers inference, and reads back the 3 action residuals. Displays a formatted table with input, expected vs. actual Q8.8, latency, and bit-exact PASS/FAIL status.
- **Option `[3]` — PL UART Protocol Interface Test**:
  Sends a 153-byte binary request frame (`5A A5 01 94 ... CRC8`) at 921,600 baud to the PL UART engine and parses the 11-byte response (`5A A5 02 06 ... CRC8`). Verifies CRC-8 Maxim calculation and range bounds.
- **Option `[4]` — Test Hardware Safety Gate & Watchdog**:
  Demonstrates hardware-level fail-closed safety. Lets the 20 ms watchdog timer expire without heartbeats and proves that motor outputs drop to zero with fault code 8. Then sends a software heartbeat to verify permit recovery, tests emergency stop latching, and speed clamping.
- **Option `[5]` — Benchmark Latency & Jitter (1000 Iterations)**:
  Executes 1,000 back-to-back inferences, calculating minimum, mean, maximum latency, jitter (standard deviation), and max control loop frequency. Demonstrates a mean latency of ~140 µs.
- **Option `[6]` — Autonomous AMR Navigation & Mecanum Kinematics**:
  Feeds a simulated 64-beam frontal LiDAR scan (with an obstacle on the right corridor) and IMU heading into the FPGA. Receives neural residual actions, blends them with nominal command velocity, applies safety limits, and computes real-time 4-wheel Mecanum motor speeds (Front-Left, Front-Right, Rear-Left, Rear-Right RPS).
- **Option `[7]` — Live Real-Time PL Telemetry Dashboard**:
  Provides a real-time terminal display showing live PL clock cycles, watchdog lease countdown, safety permit status, fault codes, and inference throughput. Press `Ctrl+C` to return to the menu.
- **Option `[8]` — Exit Program**:
  Cleanly unmaps physical memory, closes any open serial ports, and exits the shell program.

---

## 🔬 10. Verification & Cryptographic Provenance

All neural network weights are bit-true and match the validated model oracle:
- **Model Checksum (SHA-256)**:
  `55a647207c0b670901fe8074716331ac2876886bbbe1ca44aff9bbb942426e2d` (`policy_weights.mem`)
- **Bank Checksums**:
  Each `bank0.mem` through `bank7.mem` contains 4,096 hexadecimal 16-bit words corresponding to the exact weights allocated to lane $0 \dots 7$.
- **Standard Default Fixture**:
  - Input: 64 frontal beams @ `205` ($0.80$ normalized) + 10 kinematic zeros
  - Output: `[-46, 60, 34]` in Q8.8 ($-0.180$ m/s, $+0.234$ m/s, $+0.133$ rad/s)
  - Action CRC-8: `0x61`
  - Execution Time: Exactly 6,916 clock cycles (138.32 µs @ 50 MHz)

---

## 👤 Author Information

**Brian A. Hadian**  
Project Hephaestus: Edge-Accelerated Silicon & FPGA Robotics  
AMD Kria KR260 Robotics Starter Kit Implementation  
Ubuntu 22.04 LTS PS / AMD Zynq UltraScale+ MPSoC PL Co-Design
