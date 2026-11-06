# Memory Requirements & Hierarchy Analysis: Option B on AMD Kria KR260

**Author:** Brian A. Hadian  
**Project:** Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA  
**Target Hardware:** AMD Kria KR260 Starter Kit (XCK26-SFVC784-2LV-C)  

## 1. Executive Summary

- **Total Model Parameters:** 39,804 parameters
- **Total Weight Memory:** **77.74 KB** (in signed Q8.8 16-bit precision)
- **Available On-Chip BRAM:** 640 KB (144 blocks of 36 Kb)
- **BRAM Utilization:** **24.31%** (35 of 144 BRAM tiles)
- **External DDR4 DRAM Bandwidth:** **0.0 MB/s** (100% on-chip BRAM residency)

## 2. Layer-by-Layer Memory Footprint Table

| Component | Description | Parameters | Weight Memory | Activation Buffer | BRAM Tiles (36Kb) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **C1_CONV1** | Conv1 (3->8, k4, s2, p1) | 392 | 784 B | 16,384 B | 4 |
| **C2_CONV2** | Conv2 (8->16, k4, s2, p1) | 2,064 | 4,128 B | 8,192 B | 3 |
| **C3_CONV3** | Conv3 (16->32, k4, s2, p1) | 8,224 | 16,448 B | 4,096 B | 5 |
| **C4_CONV4** | Conv4 (32->32, k4, s2, p1) | 16,416 | 32,832 B | 1,024 B | 8 |
| **C5_FC_ENC** | Linear Projection (512->16) | 8,208 | 16,416 B | 32 B | 4 |
| **C6_LN_ENC** | LayerNorm (16) | 32 | 64 B | 32 B | 1 |
| **C7_LAT_REG** | Egocentric Polar Latent Register | 0 | 0 B | 32 B | 1 |
| **C8_PRED_FC1** | Predictor FC1 (19->32) | 640 | 1,280 B | 64 B | 1 |
| **C9_PRED_FC2** | Predictor FC2 (32->32) | 1,056 | 2,112 B | 64 B | 1 |
| **C10_DELTA_HEAD** | Delta Latent Head (32->16) | 528 | 1,056 B | 32 B | 1 |
| **C11_COL_HEAD** | Collision Barrier Head (32->1) | 33 | 66 B | 2 B | 1 |
| **C12_POLICY_FC1** | Policy FC1 (32->32) | 1,056 | 2,112 B | 64 B | 1 |
| **C13_POLICY_FC2** | Policy FC2 (32->32) | 1,056 | 2,112 B | 64 B | 1 |
| **C14_ACTION_HEAD** | Mecanum Action Head (32->3) | 99 | 198 B | 6 B | 1 |
| **TOTAL** | **Entire Synthesized Graph** | **39,804** | **79,608 B (77.74 KB)** | **49,152 B** | **35 of 144 Tiles (24.3%)** |

## 3. Streaming Activation Double-Buffering

To support real-time 64x64 pixel video streaming without stalling the pipeline:
- **Line Buffer Architecture:** Conv1 and Conv2 utilize dual 64-word circular BRAM line buffers (128 words = 256 bytes each).
- **Zero DDR Access:** Pixel streams are processed raster-order as they arrive from the camera interface via AXI4-Stream, requiring zero DRAM frame buffering for feature extraction.
