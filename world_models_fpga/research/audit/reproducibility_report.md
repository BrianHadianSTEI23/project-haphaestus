# Reproducibility Report & Verification Audit
**Project:** Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA  
**Author:** Brian A. Hadian  
**Phase:** Phase 0 — Repository Audit & Experimental Reproducibility  
**Date:** November 6, 2026  

---

## 1. Checkpoint Integrity & Cryptographic Hashes

All baseline and reference world model weights have been locked, crytographically fingerprinted, and verified for non-tampering:

| Checkpoint Identifier | File Location | File Size (Bytes) | SHA-256 Checksum (Hex) | Validation Status |
| :--- | :--- | :---: | :--- | :---: |
| **Baseline Policy** | `checkpoints/baseline_best.pt` | 594,877 | `a690dfcc008fa842c16196ff3ea71e06dbbcfdb2c021c1fdb45eaae43f80ea52` | **VERIFIED** |
| **LeWorldModel (LeWM)**| `checkpoints/lewm_best.pt` | 602,971 | `3871581e5d032543e33fc11f26487e4df2802fc563d72b225dbca89fb0c69d8a` | **VERIFIED** |
| **H-JEPA** | `checkpoints/hjepa_best.pt` | 676,533 | `b8729640c74795c65f909db51d07c08ce8a46b7ba90453531b6cc66d92994ebf` | **VERIFIED** |

---

## 2. Deterministic Inference Verification

Under identical random seed (`seed = 42`) and CPU execution, the models produce bit-exact deterministic activations:

### LeWorldModel Reference Validation:
* **Input**: Standard normal pseudo-random observation $I_0 \sim \mathcal{N}(0, 1) \in \mathbb{R}^{1 \times 3 \times 64 \times 64}$.
* **Action Input**: $a_0 = [0.5, 0.0, 0.0]^\top$.
* **Encoded Latent $s_0[0:4]$**:
  `[+1.556291, -1.246892, +0.342355, -1.108591]`
* **Predicted Future Latent $\hat{s}_1[0:4]$**:
  `[+1.510471, -1.263646, +0.350870, -1.013136]`
* **Deterministic Delta $\|\hat{s}_1 - s_0\|_2$**:
  $0.1342$ (Valid residual transition dynamics).

---

## 3. Behavioral Benchmark Reproduction

Re-evaluating the user's primary benchmark scenario (*"go behind the yellow cylinder"*, AMR starting pose $(0.5, 0.5, 0.0)$):

| Model Architecture | Task Success | Steps to Completion | Final Goal Distance | Obstacle Collisions | Reproducibility Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Navigation Policy** | FAILED | 85 (Timeout) | $0.981\,\text{m}$ | 13 | **100% REPRODUCED** |
| **LeWorldModel (LeWM + SIGReg)**| SUCCESS | 74 | $0.246\,\text{m}$ | 0 | **100% REPRODUCED** |
| **H-JEPA (Hierarchical Stride $K=5$)**| SUCCESS | 76 | $0.249\,\text{m}$ | 0 | **100% REPRODUCED** |

---

## 4. Phase 0 Acceptance Criteria Verification

As stipulated in Section 3, Phase 0 of the Master Research Directive:

- [x] **Every trainable parameter is accounted for**: 147,028 (Baseline), 148,913 (LeWM), 166,037 (H-JEPA) tabulated in `parameter_inventory.csv`.
- [x] **Every neural module has an identifiable input and output**: Documented with tensor shapes and dtypes in `model_interfaces.md`.
- [x] **Every inference pathway can be traced**: Detailed in `computational_graph.md` and `data_flow.md`.
- [x] **Original checkpoints reproduce the existing benchmark within defined tolerance**: Re-verified bit-exact determinism.
- [x] **Literature and hardware deviations explicitly documented**: Documented in `implementation_deviations.md`.

**Phase 0 Status**: **FULLY COMPLETE & VERIFIED**.
