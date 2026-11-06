#!/usr/bin/env python3
"""
Phase 9 — Step 9.3: Bit-Exact Q8.8 Fixed-Point Software Reference
Script: q88_fixed_point_reference.py
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Implements the bit-exact mathematical reference matching the synthesizable Verilog RTL.
Exports:
  1. Hex weight tables for BRAM initialization ($readmemh)
  2. Input test vectors (in_features_q88.hex)
  3. Golden output vectors (golden_outputs_q88.hex)
"""

import os
import sys
import numpy as np
import torch

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(base_dir)
from world_models_fpga.research.extraction.extracted_model.reconstructed_compact.model_reconstructed_compact import ReconstructedCompactModel

def float_to_q88(val):
    """Converts float32 to signed 16-bit Q8.8 fixed-point integer (-32768 to 32767)."""
    scaled = np.round(np.array(val, dtype=np.float32) * 256.0)
    return np.clip(scaled, -32768, 32767).astype(np.int16)

def q88_to_float(val):
    """Converts signed 16-bit Q8.8 fixed-point integer to float32."""
    return np.array(val, dtype=np.float32) / 256.0

def export_to_hex_file(arr, filepath):
    """Exports numpy array as 4-character hex strings (one per line) for Verilog $readmemh."""
    flat = np.array(arr).flatten()
    with open(filepath, "w") as f:
        for val in flat:
            u16 = int(val) & 0xFFFF
            f.write(f"{u16:04X}\n")

class BitExactQ88Model:
    """
    Bit-Exact Signed Q8.8 Fixed-Point Reference Engine.
    Directly mirrors hardware arithmetic:
      - 16-bit signed operands
      - 32-bit accumulation
      - Right shift by 8 bits with rounding
      - 16-bit saturation clipping [-32768, +32767]
    """
    @staticmethod
    def mac_1d(in_vec, weights, bias=None):
        """
        in_vec: int16 (N,)
        weights: int16 (M, N)
        bias: int16 (M,)
        """
        in_32 = in_vec.astype(np.int64)
        w_32 = weights.astype(np.int64)
        # Dot product accumulates in 32-bit/64-bit
        acc = np.dot(w_32, in_32) # Q16.16
        # Hardware rounding: add 128 (0.5 LSB) before right shift 8
        acc_rounded = (acc + 128) >> 8 # Q8.8
        if bias is not None:
            acc_rounded += bias.astype(np.int64)
        return np.clip(acc_rounded, -32768, 32767).astype(np.int16)

    @staticmethod
    def relu(x):
        return np.maximum(x, 0).astype(np.int16)

    @staticmethod
    def pwl_tanh(x):
        """Piecewise linear tanh approximation in Q8.8 (-256 to +256 = -1.0 to +1.0)."""
        # x is Q8.8
        out = np.zeros_like(x)
        # Region 1: |x| < 0.5 (128 in Q8.8) -> y = x
        r1 = np.abs(x) < 128
        out[r1] = x[r1]
        # Region 2: 128 <= |x| < 256 -> y = sign(x) * (64 + |x|/2)
        r2 = (np.abs(x) >= 128) & (np.abs(x) < 256)
        out[r2] = np.sign(x[r2]) * (64 + (np.abs(x[r2]) >> 1))
        # Region 3: |x| >= 256 -> y = +/- 256 (+/- 1.0)
        r3 = np.abs(x) >= 256
        out[r3] = np.sign(x[r3]) * 256
        return out.astype(np.int16)

    @staticmethod
    def pwl_sigmoid(x):
        """Piecewise linear sigmoid approximation in Q8.8 (0 to 256 = 0.0 to 1.0)."""
        # y = 0.5 + 0.25*x for |x| <= 512 (+/- 2.0)
        # 0.5 in Q8.8 is 128, 0.25 is shift right 2
        out = 128 + (x >> 2)
        out = np.clip(out, 0, 256)
        return out.astype(np.int16)

def export_model_q88_weights():
    out_dir = os.path.abspath(os.path.dirname(__file__))
    hex_dir = os.path.join(out_dir, "hex_weights")
    os.makedirs(hex_dir, exist_ok=True)

    ckpt_path = os.path.abspath(os.path.join(out_dir, "../extraction/extracted_model/reconstructed_compact/model_reconstructed_compact.pt"))
    model = ReconstructedCompactModel(latent_dim=16, stride_k=5, hidden_dim=32)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    sd = model.state_dict()
    q88_dict = {}

    print("\n--- EXPORTING Q8.8 WEIGHT TABLES FOR FPGA RTL ---")
    for k, v in sd.items():
        arr = v.cpu().numpy()
        q88_arr = float_to_q88(arr)
        q88_dict[k] = q88_arr
        safe_name = k.replace(".", "_") + ".hex"
        export_to_hex_file(q88_arr, os.path.join(hex_dir, safe_name))
        print(f"Exported: {safe_name:<35} | Shape: {str(q88_arr.shape):<15} | Words: {q88_arr.size:,}")

    # Generate Golden Verification Test Vectors for RTL Testbench
    # Test vector: 16-word latent state S_k + 3-word candidate action A_k = 19 words
    np.random.seed(1234)
    test_S_k = float_to_q88(np.array([0.5, -0.3, 1.2, 0.0, -0.8, 0.4, 0.2, -0.1,
                                      0.9, -0.4, 0.3, 0.1, -0.5, 0.6, -0.2, 0.0], dtype=np.float32))
    test_A_k = float_to_q88(np.array([0.2, 0.0, 0.1], dtype=np.float32)) # Forward + slight steer
    test_in_19 = np.concatenate([test_S_k, test_A_k])

    # Compute Bit-Exact Macro Predictor Golden Output
    # Layer 1: FC1 (19 -> 32)
    w_fc1 = q88_dict["macro_predictor.fc1.weight"]
    b_fc1 = q88_dict["macro_predictor.fc1.bias"]
    h1 = BitExactQ88Model.relu(BitExactQ88Model.mac_1d(test_in_19, w_fc1, b_fc1))

    # Layer 2: FC2 (32 -> 32)
    w_fc2 = q88_dict["macro_predictor.fc2.weight"]
    b_fc2 = q88_dict["macro_predictor.fc2.bias"]
    h2 = BitExactQ88Model.relu(BitExactQ88Model.mac_1d(h1, w_fc2, b_fc2))

    # Head 1: Delta Head (32 -> 16)
    w_delta = q88_dict["macro_predictor.delta_head.weight"]
    b_delta = q88_dict["macro_predictor.delta_head.bias"]
    delta_S = BitExactQ88Model.mac_1d(h2, w_delta, b_delta)
    # Residual Addition: Next_S = S_k + delta_S
    next_S = np.clip(test_S_k.astype(np.int64) + delta_S.astype(np.int64), -32768, 32767).astype(np.int16)

    # Head 2: Collision Head (32 -> 1)
    w_col = q88_dict["macro_predictor.col_head.weight"]
    b_col = q88_dict["macro_predictor.col_head.bias"]
    col_logit = BitExactQ88Model.mac_1d(h2, w_col, b_col)
    col_prob = BitExactQ88Model.pwl_sigmoid(col_logit)

    # Compute Bit-Exact Micro Policy Golden Output
    # Input: Current state (16) + Subgoal (16) = 32 words
    test_subgoal = next_S.copy()
    test_policy_in = np.concatenate([test_S_k, test_subgoal])
    w_p_fc1 = q88_dict["micro_policy.fc1.weight"]
    b_p_fc1 = q88_dict["micro_policy.fc1.bias"]
    p_h1 = BitExactQ88Model.relu(BitExactQ88Model.mac_1d(test_policy_in, w_p_fc1, b_p_fc1))

    w_p_fc2 = q88_dict["micro_policy.fc2.weight"]
    b_p_fc2 = q88_dict["micro_policy.fc2.bias"]
    p_h2 = BitExactQ88Model.relu(BitExactQ88Model.mac_1d(p_h1, w_p_fc2, b_p_fc2))

    w_act = q88_dict["micro_policy.action_head.weight"]
    b_act = q88_dict["micro_policy.action_head.bias"]
    act_raw = BitExactQ88Model.mac_1d(p_h2, w_act, b_act)
    act_q88 = BitExactQ88Model.pwl_tanh(act_raw)

    # Export Testbench hex files
    vec_dir = os.path.abspath(os.path.join(out_dir, "../../fpga/vectors"))
    os.makedirs(vec_dir, exist_ok=True)

    export_to_hex_file(test_in_19, os.path.join(vec_dir, "tb_input_macro_19.hex"))
    export_to_hex_file(next_S, os.path.join(vec_dir, "tb_golden_next_S_16.hex"))
    export_to_hex_file(col_prob, os.path.join(vec_dir, "tb_golden_col_prob_1.hex"))
    export_to_hex_file(test_policy_in, os.path.join(vec_dir, "tb_input_policy_32.hex"))
    export_to_hex_file(act_q88, os.path.join(vec_dir, "tb_golden_action_3.hex"))

    print("\n[Generated Test Vectors in fpga/vectors/]")
    print(f"  Input Macro Vector (19 words):   {os.path.join(vec_dir, 'tb_input_macro_19.hex')}")
    print(f"  Golden Next_S Vector (16 words): {os.path.join(vec_dir, 'tb_golden_next_S_16.hex')}")
    print(f"  Golden Col Prob (1 word):        {os.path.join(vec_dir, 'tb_golden_col_prob_1.hex')}")
    print(f"  Input Policy Vector (32 words):  {os.path.join(vec_dir, 'tb_input_policy_32.hex')}")
    print(f"  Golden Action Vector (3 words):  {os.path.join(vec_dir, 'tb_golden_action_3.hex')}")
    print(f"\nSample Golden Output Values:")
    print(f"  Next_S[0..3] (Q8.8): {next_S[:4]} -> Float: {q88_to_float(next_S[:4])}")
    print(f"  Col Prob (Q8.8):    {col_prob} -> Float: {q88_to_float(col_prob)}")
    print(f"  Action (Q8.8):      {act_q88} -> Float: {q88_to_float(act_q88)}")

if __name__ == "__main__":
    export_model_q88_weights()
