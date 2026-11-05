# Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA
**Author:** Brian A. Hadian  
**Target Board:** AMD Kria KR260 Robotics Starter Kit (Zynq UltraScale+ XCK26 SOM)  
**Host Platform:** Intel Core Ultra 7 + NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.6)

---

## 1. Project Overview

This repository implements an end-to-end robotic system integrating **Natural Language Semantic Grounding**, **Action-Conditioned World Models**, and **Signed Q8.8 Fixed-Point FPGA Hardware Acceleration** for the **Hiwonder JetAuto Mecanum AMR**:

1. **3D Interactive Simulation**:
   - High-fidelity MuJoCo physics arena with the authentic Hiwonder JetAuto Mecanum AMR chassis (3-DOF continuous holonomic kinematics: $v_x \in [-0.6, 0.6]\,\text{m/s}$, $v_y \in [-0.4, 0.4]\,\text{m/s}$, $\omega_z \in [-1.5, 1.5]\,\text{rad/s}$).
   - Onboard Orbbec Astra camera rendering $64 \times 64 \times 3$ RGB observation frames.
   - Dual-view interactive Web Dashboard streaming 30 FPS 3D arena views alongside onboard camera feedback with HUD overlays.
2. **Language-Grounded Semantic Goal Engine**:
   - Parses natural language prompts (e.g., *"go behind the yellow cylinder"*, *"approach the red box"*, *"move to the left of green cylinder"*).
   - Resolves spatial prepositions (*behind*, *front*, *left*, *right*, *near*) into 3D arena waypoints with obstacle clearance flanks.
   - Closed-loop visual verification via real-time HSV color segmentation and camera projective centroid tracking.
3. **World Model Architectures**:
   - **Baseline Policy**: Flat reactive vision-language actor-critic policy.
   - **LeWorldModel (LeWM)**: Action-conditioned Joint-Embedding Predictive Architecture (JEPA) from raw pixels with Sketched Isotropic Gaussian Regularization (**SIGReg**) based on [arXiv:2603.19312](https://arxiv.org/abs/2603.19312).
   - **H-JEPA**: Hierarchical 2-tier JEPA decoupling macro subgoals (temporal stride $K=5$) from micro primitive action tracking based on [arXiv:2610.06805](https://arxiv.org/abs/2610.06805).
4. **FPGA Hardware Accelerator (AMD Kria KR260)**:
   - Synthesizable Verilog RTL (`world_models_accelerator_top.v`) with pipelined DSP vector MAC units and AXI4-Lite slave interface.
   - Signed Q8.8 fixed-point quantization (16-bit word: 1 sign, 7 integer, 8 fractional bits).
   - Execution latency: **48 clock cycles = 0.48 $\mu$s @ 100 MHz** ($1,237\times$ speedup over KR260 ARM Cortex-A53 CPU).

---

## 2. Directory Structure

```
world_models_fpga/
├── env/
│   ├── amr_arena.xml              # MuJoCo 3D arena with 6 colored obstacles & JetAuto AMR
│   └── amr_env.py                 # Gymnasium wrapper with dual first-person & third-person rendering
├── nlp/
│   └── grounding.py               # NLP Semantic Goal Grounder & Astra Camera Verification
├── models/
│   ├── baseline_policy.py         # Baseline goal-conditioned navigation policy
│   ├── le_world_model.py          # LeWorldModel (LeWM + SIGReg) JEPA architecture
│   └── h_jepa.py                  # H-JEPA Hierarchical (Macro Stride K=5 + Micro)
├── training/
│   └── train_pipeline.py          # Master GPU training script on RTX 3050
├── checkpoints/                   # Trained PyTorch weights (.pt) and loss curves
├── fpga/
│   ├── quantization.py            # Signed Q8.8 quantization and hex exporter
│   ├── vectors/                   # Quantized weight tables (.hex)
│   └── rtl/
│       ├── q88_mac_core.v         # Vector DSP MAC unit with saturation arithmetic
│       ├── world_models_accelerator_top.v # AXI4-Lite FPGA top-level accelerator
│       └── tb_world_models_accelerator.v  # Icarus Verilog simulation testbench
├── deploy_kr260/
│   ├── kr260_runner.py            # On-board benchmark runner for AMD Kria KR260
│   └── benchmark_laptop.py        # Laptop GPU and CPU benchmark suite
├── sim_app_3d/
│   └── sim_app_3d.py              # Interactive 3D Web Dashboard & evaluation harness
├── presentation/
│   ├── presentation.tex           # 8-slide LaTeX Beamer presentation
│   ├── presentation.pdf           # Compiled presentation PDF
│   ├── slides/                    # High-res PNG slide renders (slide-1 to slide-8)
│   └── generate_plots.py          # Benchmark comparison bar charts generator
└── run_demo.sh                    # One-command demo launcher script
```

---

## 3. How to Run

### A. Launch Interactive 3D Web Dashboard
```bash
./run_demo.sh
```
Open `http://localhost:8080` in your web browser. You can type any command (e.g. *"go behind the yellow cylinder"*), switch world models in real time, and observe the AMR navigating with onboard camera feedback.

### B. Run Headless Evaluation Across Models
```bash
# Evaluate LeWorldModel on user's prompt
python3 sim_app_3d/sim_app_3d.py --eval --cmd "go behind the yellow cylinder" --model lewm

# Evaluate H-JEPA
python3 sim_app_3d/sim_app_3d.py --eval --cmd "go behind the yellow cylinder" --model hjepa

# Evaluate Baseline Policy
python3 sim_app_3d/sim_app_3d.py --eval --cmd "go behind the yellow cylinder" --model baseline
```

### C. Run FPGA Verilog RTL Simulation
```bash
iverilog -o sim_accel fpga/rtl/world_models_accelerator_top.v fpga/rtl/tb_world_models_accelerator.v
vvp sim_accel
```

### D. Run Benchmarks on AMD Kria KR260
```bash
ssh ubuntu@kria.local "cd ~/deploy_world_models && python3 kr260_runner.py"
```

---

## 4. Benchmark Summary Across Platforms

| Platform | Compute Device | Precision | Latency ($\mu\text{s}$) | Throughput (FPS) | Power (Watts) | Energy / Inf ($\mu\text{J}$) | Speedup vs. ARM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Laptop GPU** | NVIDIA RTX 3050 Laptop GPU | Float32 | $83.71\,\mu\text{s}$ | 11,946 inf/s | $38.5\,\text{W}$ | $3,281.0\,\mu\text{J}$ | $7.1\times$ |
| **Laptop CPU** | Intel Core Ultra 7 (x86_64) | Float32 | $44.64\,\mu\text{s}$ | 22,401 inf/s | $28.0\,\text{W}$ | $1,250.0\,\mu\text{J}$ | $13.3\times$ |
| **KR260 PS** | ARM Cortex-A53 (4 Cores @ 1.2 GHz) | Q8.8 | $593.86\,\mu\text{s}$ | 1,684 inf/s | $3.10\,\text{W}$ | $1,841.0\,\mu\text{J}$ | $1.0\times$ (Ref) |
| **KR260 PL** | **AMD XCK26 FPGA PL (@ 100 MHz)** | **Signed Q8.8** | **$0.480\,\mu\text{s}$** | **2,083,333 inf/s** | **$3.25\,\text{W}$** | **$1.56\,\mu\text{J}$** | **$1,237.2\times$** |
