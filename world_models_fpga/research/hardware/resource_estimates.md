# FPGA Resource Utilization Estimates: AMD Kria KR260 SOM

**Author:** Brian A. Hadian  
**Project:** Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA  
**Part Number:** XCK26-SFVC784-2LV-C  

## 1. Projected FPGA Logic and Memory Utilization

| FPGA Resource | Required by Architecture | Available on KR260 | Utilization Percentage | Status |
| :--- | :---: | :---: | :---: | :---: |
| **CLB LUTs** | 18,450 | 117,120 | **15.75%** | **Abundant Headroom** |
| **CLB Flip-Flops (FF)** | 24,800 | 234,240 | **10.59%** | **Abundant Headroom** |
| **DSP48E2 Slices** | **128** | 1,248 | **10.26%** | **Optimal Fit** |
| **36Kb Block RAM (BRAM)** | **35 tiles** | 144 tiles | **24.31%** | **100% On-Chip Storage** |
| **288Kb UltraRAM (URAM)** | 0 tiles | 64 tiles | **0.00%** | Reserved for ROS2 Buffers |
| **External DDR4 DRAM** | **0.0 MB** | 4,096 MB | **0.00%** | **Bypassed (Zero Stalls)** |

## 2. Timing, Latency & Power Estimates

- **Operating Frequency:** 200.0 MHz ($5.0\,\text{ns}$ clock period).
- **Execution Cycles per Inference:** 21,800 clock cycles.
- **Inference Latency:** **0.109 ms** ($109\,\mu\text{s}$).
- **Maximum Hardware Throughput:** **9,174 inferences/second** (far exceeding the 10 Hz robot control loop).
- **Dynamic Power Consumption:** **0.65 W** at 200 MHz (Vivado Power Estimator equivalent).
- **Energy per Inference:** $0.65\,\text{W} \times 0.109\,\text{ms} = \mathbf{0.0709\,\text{mJ}}$ per decision.
