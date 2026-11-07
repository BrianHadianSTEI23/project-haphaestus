# Hardware Validation Report — RTL Simulation, Xilinx Vivado 2023.2 Synthesis, and Physical AMD Kria KR260 Silicon Measurements

## 1. Executive Summary & Scientific Separation of Evidence

In accordance with Milestone 7 (`HW-REPORT`, `HW-EQUIV`, `HW-AXI-STREAM`, `HW-CYCLE`), this report strictly separates four distinct categories of hardware evidence so that no analytical estimate, simulation cycle count, or post-synthesis timing figure is conflated with physical board execution:

1. **Cycle-Accurate RTL Simulation (`iverilog` / `vvp`):** Bit-exact verification of the synthesizable Verilog-2001 accelerator [`top_world_model_accelerator.v`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/fpga/rtl/top_world_model_accelerator.v) via [`tb_top_world_model_accelerator.v`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/fpga/rtl/tb_top_world_model_accelerator.v) across both **AXI4-Lite** register-mapped I/O and **AXI4-Stream** (`35`-word DMA packet ingress with `TLAST` auto-start).
2. **Xilinx Vivado 2023.2 Post-Synthesis Analysis (`xck26-sfvc784-2LV-c`):** Real out-of-context synthesis executed via [`synth_kr260.tcl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/fpga/synth_kr260.tcl) on the AMD Kria KR260 Zynq UltraScale+ XCK26 FPGA part, producing [`vivado_utilization.rpt`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/vivado_utilization.rpt), [`vivado_timing.rpt`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/vivado_timing.rpt), and [`vivado_power.rpt`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/vivado_power.rpt).
3. **Physical AMD Kria KR260 Board Execution (`aarch64` Linux on Cortex-A53 PS + Live INA260 / AMS Telemetry):** Direct execution of [`kr260_hardware_benchmark.py`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/deploy_kr260/kr260_hardware_benchmark.py) on the physical Kria KR260 board (`ubuntu@kria.local`, Linux `5.15.0-1080-xilinx-zynqmp aarch64`) using the exported trained Q8.8 weights [`q88_exported_weights.npz`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/q88_exported_weights.npz), recorded in [`kr260_hardware_empirical_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/deploy_kr260/kr260_hardware_empirical_results.json).
4. **Host CPU & GPU Benchmarks (`benchmarks_heterogeneous_platforms.json`):** Measured PyTorch FP32 inference latency on the host CPU and NVIDIA RTX 3050 Laptop GPU via [`benchmark_fair_comparison.py`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/benchmark_fair_comparison.py).

---

## 2. Repairs Executed (`HW-REPORT`, `HW-EQUIV`, `HW-AXI-STREAM`, `HW-CYCLE`)

| Issue ID | Previous Flaw | Scientific Repair | Verification Artifact |
| :--- | :--- | :--- | :--- |
| **`HW-REPORT`** | Resource utilization (`18,450` LUTs, `24,800` FFs, `38` BRAMs, `128` DSP48E2) and power (`0.65 W`) were hardcoded strings in [`benchmark_fair_comparison.py`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/benchmark_fair_comparison.py). | Created [`synth_kr260.tcl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/fpga/synth_kr260.tcl) and ran Xilinx Vivado 2023.2 synthesis on `xck26-sfvc784-2LV-c`. Updated `benchmark_fair_comparison.py` to parse the real Vivado reports directly. | [`vivado_utilization.rpt`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/vivado_utilization.rpt), [`vivado_timing.rpt`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/vivado_timing.rpt), [`vivado_power.rpt`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/vivado_power.rpt) |
| **`HW-EQUIV`** | [`validate_hardware_equivalence.py`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/validate_hardware_equivalence.py) added synthetic `np.random.normal(0, 1e-4)` noise to Python fixed-point outputs instead of invoking RTL simulation, and [`kr260_hardware_benchmark.py`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/deploy_kr260/kr260_hardware_benchmark.py) generated random `np.random.randint` weights. | Exported trained Q8.8 weights to [`q88_exported_weights.npz`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/q88_exported_weights.npz) and `.hex` ROM tables; updated `validate_hardware_equivalence.py` to compile and run `iverilog`/`vvp` on [`tb_top_world_model_accelerator.v`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/fpga/rtl/tb_top_world_model_accelerator.v); updated `kr260_hardware_benchmark.py` to load `q88_exported_weights.npz`. | [`hardware_equivalence_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/hardware_equivalence_results.json), [`kr260_hardware_empirical_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/deploy_kr260/kr260_hardware_empirical_results.json) |
| **`HW-AXI-STREAM`** | `s_axis_tvalid` and `s_axis_tdata` were declared on [`top_world_model_accelerator.v`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/fpga/rtl/top_world_model_accelerator.v) with `s_axis_tready = 1'b1` tied high, ignoring incoming stream words. | Implemented functional 35-word AXI4-Stream packet ingress (`16` words $s_t$ + `3` words $A_k$ + `16` words $S_{\text{subgoal}}$) with proper `s_axis_tready` backpressure (`fsm_state == ST_IDLE && !start_reg`) and `s_axis_tlast` auto-triggering. | [`top_world_model_accelerator.v`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/fpga/rtl/top_world_model_accelerator.v#L140-L185), [`tb_top_world_model_accelerator.v`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/fpga/rtl/tb_top_world_model_accelerator.v#L210-L275) |
| **`HW-CYCLE`** | Single-pass RTL latency (`4,327` cycles = `0.0216 ms` at `200 MHz`) was conflated with 24-candidate MPC latency (`54,467` cycles = `0.2723 ms` at `200 MHz`). | Explicitly separated 1-candidate pass (`4,327` cycles) from 24-candidate MPC + 1 micro policy pass (`54,467` cycles) across all scripts, JSON outputs, and tables. | [`hardware_equivalence_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/hardware_equivalence_results.json), [`benchmarks_heterogeneous_platforms.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/benchmarks_heterogeneous_platforms.json) |

---

## 3. Three-Way Numerical Equivalence (`FP32` vs. `Q8.8` Reference vs. Verilog RTL)

Evaluated across $N = 300$ held-out transitions in [`validate_hardware_equivalence.py`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/validate_hardware_equivalence.py) and cycle-accurate RTL simulation ([`hardware_equivalence_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/hardware_equivalence_results.json)):

| Comparison Metric | Measured Value | Interpretation |
| :--- | :---: | :--- |
| **Verilog RTL vs. Python Q8.8 Reference ($E_{\text{rel}}$)** | **`0.000000` (100.0% Bit-Exact)** | `20 / 20` output words match bit-for-bit across both AXI4-Lite and AXI4-Stream modes |
| **Predicted Next Macro State $\hat{S}_{t+K}$ ($E_{\text{rel}}$ vs. FP32)** | `0.009430` (`0.94%`) | Sub-1% quantization error across 16-D residual macro state transition |
| **Motor Action Vector $[v_x, v_y, \omega_z]$ ($E_{\text{rel}}$ vs. FP32)** | `0.072017` (`7.20%`) | Dominated by piecewise-linear $\tanh$ saturation slope approximation |
| **Collision Risk $B(S_t, A_k)$ ($E_{\text{rel}}$ vs. FP32)** | `0.874024` | Piecewise-linear $\sigma(x) \approx \text{clip}((x + 2)/4, 0, 1)$ clips near-zero tail probabilities to exact `0` |
| **Action Direction Cosine Similarity (Q8.8 vs. FP32)** | **`0.9984`** | Near-identical holonomic velocity heading |
| **Obstacle Safety Barrier Decision Concordance** | **`100.00%`** | Identical binary safety gating ($B(S_t, A_k) > 0.50$) across all 300 transitions |

---

## 4. Real Xilinx Vivado 2023.2 Post-Synthesis Results (`xck26-sfvc784-2LV-c`)

Synthesized out-of-context with a target clock period of `5.000 ns` (`200.0 MHz`) via [`synth_kr260.tcl`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/fpga/synth_kr260.tcl) (`0` errors, `0` critical warnings):

### 4.1 Post-Synthesis Resource Utilization ([`vivado_utilization.rpt`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/vivado_utilization.rpt))

| Resource Type | Used | Available (`xck26`) | Utilization (%) | Previous Hardcoded Claim (Refuted) |
| :--- | :---: | :---: | :---: | :---: |
| **CLB LUTs** | **`3,579`** | `117,120` | **`3.06%`** | `18,450 (15.8%)` |
| **CLB Registers (Flip-Flops)** | **`3,749`** | `234,240` | **`1.60%`** | `24,800 (10.6%)` |
| **Block RAM Tiles (`RAMB36`/`RAMB18`)** | **`0`** | `144` | **`0.00%`** | `38 (26.4%)` |
| **DSP48E2 Slices** | **`13`** | `1,248` | **`1.04%`** | `128 (10.3%)` |

*Architectural Note on BRAM & DSP Usage:* Because `top_world_model_accelerator` implements sequential single-MAC engines (`macro_transition_core` + `micro_policy_core`) whose compact weight tables (`608`–`1,024` words per layer) are written as combinational `case` ROMs in Verilog, Vivado synthesizes those constant weight tables directly into distributed LUT logic/muxes (`3,579` LUTs, `0` Block RAM tiles) and uses `13` DSP48E2 slices for the signed Q8.8 multiply-accumulate units.

### 4.2 Post-Synthesis Timing Analysis ([`vivado_timing.rpt`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/vivado_timing.rpt))

| Timing Parameter | Value | Scientific Assessment |
| :--- | :---: | :--- |
| **Target Clock Constraint** | `5.000 ns` (`200.0 MHz`) | Nominal design target |
| **Worst Negative Slack (WNS at 200 MHz)** | **`-2.836 ns`** | Critical path = `7.836 ns` (14 logic levels through unpipelined `DSP48E2` multiply + bias addition + Q8.8 saturation into `S_next_out_reg`) |
| **Worst Hold Slack (WHS)** | `+0.015 ns` | All hold constraints met |
| **Post-Synthesis Maximum Frequency ($F_{\max}$)** | **`127.62 MHz`** | Closes timing cleanly at **`125.0 MHz`** (`8.000 ns` clock period, `+0.164 ns` positive slack) |

### 4.3 Post-Synthesis Power Analysis ([`vivado_power.rpt`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/vivado_power.rpt))

* **Total On-Chip Power:** **`0.365 W`**
  * **Dynamic Power:** **`0.076 W`** (`0.025 W` clocks, `0.026 W` signals, `0.021 W` CLB logic, `0.004 W` DSPs)
  * **Device Static Power:** **`0.289 W`**

---

## 5. Exact Cycle Counts & Latency at Both Nominal (`200 MHz`) and Timing-Closed (`125 MHz`) Clocks

| Execution Mode | Exact Clock Cycles | Latency @ `200.0 MHz` (Target) | Latency @ `127.62 MHz` ($F_{\max}$) | Latency @ `125.0 MHz` (Timing-Closed) |
| :--- | :---: | :---: | :---: | :---: |
| **Macro Transition Core (`19 -> 32 -> 32 -> 16 + 1`)** | `2,179` cycles | `10.895 us` | `17.074 us` | `17.432 us` |
| **Micro Policy Core (`32 -> 32 -> 32 -> 3`)** | `2,145` cycles | `10.725 us` | `16.808 us` | `17.160 us` |
| **Top-Level FSM Handshake Overhead** | `3` cycles | `0.015 us` | `0.024 us` | `0.024 us` |
| **1 Macro Candidate + 1 Micro Policy Pass (Single-Pass)** | **`4,327` cycles** | **`21.635 us` (`0.0216 ms`)** | **`33.905 us` (`0.0339 ms`)** | **`34.616 us` (`0.0346 ms`)** |
| **24 Macro MPC Candidates + 1 Micro Policy Pass (`24 * 2,180 + 2,147`)** | **`54,467` cycles** | **`272.335 us` (`0.2723 ms`)** | **`426.790 us` (`0.4268 ms`)** | **`435.736 us` (`0.4357 ms`)** |

---

## 6. Physical AMD Kria KR260 Silicon Measurements & Heterogeneous Platform Comparison

Measured directly on the physical AMD Kria KR260 board (`ubuntu@kria.local`, quad-core ARM Cortex-A53 @ `1.333 GHz`, [`kr260_hardware_empirical_results.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/deploy_kr260/kr260_hardware_empirical_results.json)) and compared against host CPU/GPU in [`benchmarks_heterogeneous_platforms.json`](file:///home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/research/hardware/benchmarks_heterogeneous_platforms.json):

### 6.1 Live Physical Kria KR260 INA260 & AMS Sensor Telemetry

* **Board Rail Voltage (`in_voltage0_raw`):** `5.031 V`
* **Board Total Current (`in_current0_raw`):** `676.0 mA`
* **Board Active Power (`in_power0_raw`):** **`3.400 W`**
* **Temperatures:** PS LPD `37.16 °C`, PS FPD `38.08 °C`, PL Fabric `35.78 °C`

### 6.2 End-to-End Heterogeneous Platform Comparison

| Configuration | Platform | Evidence Type | Median Latency (ms) | P95 (ms) | Throughput (FPS) | Power (W) | Energy / Decision (mJ) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Original Baseline (FP32)** | Host CPU | Host CPU Measured | `5.4605` | `5.5344` | `183.1` | `28.000` | `152.89` |
| **Original H-JEPA 24-MPC (FP32)** | Host CPU | Host CPU Measured | `7.0926` | `7.1755` | `141.0` | `28.000` | `198.59` |
| **Original H-JEPA 24-MPC (FP32)** | NVIDIA RTX 3050 GPU | Host GPU Measured | `0.5912` | `0.7984` | `1,691.5` | `35.000` | `20.69` |
| **Extracted Compact 24-MPC (FP32)** | Host CPU | Host CPU Measured | `3.8686` | `5.8878` | `258.5` | `18.500` | `71.57` |
| **Extracted Compact 24-MPC (Q8.8)** | **Kria KR260 ARM Cortex-A53 PS** | **Physical KR260 PS Measured** | **`61.7750`** (`42.58 ms` CNN + `19.20 ms` 24-MPC) | `61.7750` | **`16.2`** | **`3.400`** | **`210.04`** |
| **FPGA PL Kernel (1 Macro + 1 Micro)** | **XCK26 PL (`4,327` cycles)** | **RTL Cycle-Exact + Vivado Synth** | **`0.0216`** (`@200MHz`) / **`0.0346`** (`@125MHz`) | `0.0216` | **`46,221.4`** | **`0.365`** | **`0.0079`** |
| **FPGA PL Kernel (24-MPC + 1 Micro)** | **XCK26 PL (`54,467` cycles)** | **RTL Cycle-Exact + Vivado Synth** | **`0.2723`** (`@200MHz`) / **`0.4357`** (`@125MHz`) | `0.2723` | **`3,671.9`** | **`0.365`** | **`0.0994`** |
| **KR260 Hybrid PS(CNN) + PL(24-MPC)** | **AMD Kria KR260 SOM** | **Measured PS CNN + RTL PL Kernel** | **`42.8655`** (`@200MHz`) / **`43.0289`** (`@125MHz`) | `42.8655` | **`23.3`** | **`3.400`** | **`145.74`** |

### 6.3 Key Hardware Findings

1. **Planning & Control Kernel Acceleration (`70.49x` @ `200 MHz` / `44.06x` @ `125 MHz`):** On the physical Kria KR260 ARM Cortex-A53 PS, executing the 24-candidate macro MPC rollout plus micro policy takes **`19.197 ms`**, whereas the synthesizable RTL PL core executes the exact same Q8.8 computation in **`54,467` cycles** (`0.2723 ms` at `200 MHz`, or `0.4357 ms` at timing-closed `125 MHz`), yielding a **`70.49x`** speedup (`44.06x` at `125 MHz`) and reducing the planning kernel latency to `< 0.5%` of the `100 ms` (`10 Hz`) real-time robot control budget.
2. **Amdahl's Law Bottleneck on Hybrid PS+PL Deployment:** Because the 4-layer Tier-1 visual CNN encoder remains on the ARM Cortex-A53 PS (`42.578 ms`), the hybrid PS(CNN) + PL(24-MPC) end-to-end latency is **`42.866 ms` (`23.3 Hz`)** vs. **`61.775 ms` (`16.2 Hz`)** for PS-only execution (`1.44x` end-to-end SOM speedup). Achieving sub-millisecond end-to-end SOM latency requires migrating the 4-layer Conv2D visual encoder into the PL fabric alongside the macro/micro MLP cores.
