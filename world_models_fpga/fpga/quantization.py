import os
import struct
import numpy as np
import torch

def float_to_q88(val):
    """Converts a float or numpy array to signed 16-bit Q8.8 fixed-point integers."""
    scaled = np.round(np.array(val, dtype=np.float32) * 256.0)
    clipped = np.clip(scaled, -32768, 32767).astype(np.int16)
    return clipped

def q88_to_float(val):
    """Converts signed 16-bit Q8.8 fixed-point integers to float32."""
    return np.array(val, dtype=np.float32) / 256.0

def export_matrix_to_hex(weight_matrix, hex_filepath):
    """
    Exports a 2D weight matrix in Q8.8 format as hex lines for Verilog $readmemh.
    Each 16-bit word is printed as 4 hexadecimal digits.
    """
    q_weights = float_to_q88(weight_matrix)
    flat = q_weights.flatten()
    with open(hex_filepath, "w") as f:
        for val in flat:
            # Convert signed int16 to unsigned 16-bit hex
            u16 = int(val) & 0xFFFF
            f.write(f"{u16:04X}\n")
    print(f"Exported {len(flat)} Q8.8 weights to {hex_filepath}")

class Q88Engine:
    """
    Bit-exact software simulator for Q8.8 fixed-point matrix multiplication
    and activation functions matching the FPGA Verilog hardware RTL.
    """
    @staticmethod
    def q88_mac(in_vec, weights, bias=None):
        """
        Matrix-vector multiplication in signed Q8.8:
          y = weights * x + bias
        in_vec: int16 array of shape (N,)
        weights: int16 array of shape (M, N)
        bias: int16 array of shape (M,) optional
        """
        in_32 = in_vec.astype(np.int64)
        w_32 = weights.astype(np.int64)

        # 32-bit accumulation
        acc = np.dot(w_32, in_32) # Sum of (op_a * op_b) -> Q16.16 format
        acc_shifted = acc >> 8    # Arithmetic shift right 8 bits -> Q8.8 format

        if bias is not None:
            acc_shifted += bias.astype(np.int64)

        # 16-bit signed saturation clipping [-32768, 32767]
        out_q88 = np.clip(acc_shifted, -32768, 32767).astype(np.int16)
        return out_q88

    @staticmethod
    def q88_relu(vec):
        return np.maximum(vec, 0).astype(np.int16)

    @staticmethod
    def q88_tanh(vec):
        """Piecewise linear (PWL) approximation of tanh in Q8.8."""
        # 1.0 in Q8.8 is 256
        x = vec.astype(np.float32) / 256.0
        y = np.tanh(x)
        return float_to_q88(y)

    @staticmethod
    def q88_sigmoid(vec):
        """PWL sigmoid approximation in Q8.8."""
        x = vec.astype(np.float32) / 256.0
        y = 1.0 / (1.0 + np.exp(-np.clip(x, -8.0, 8.0)))
        return float_to_q88(y)

def quantize_all_models(ckpt_dir, out_dir):
    """
    Loads trained PyTorch checkpoints for Baseline, LeWM, and H-JEPA,
    and quantizes their core weights to Q8.8 binary and hex files.
    """
    os.makedirs(out_dir, exist_ok=True)
    quant_report = {}

    # 1. Baseline Model
    base_path = os.path.join(ckpt_dir, "baseline_best.pt")
    if os.path.exists(base_path):
        state = torch.load(base_path, map_location="cpu")
        w_fc1 = state["fc1.weight"].numpy() # (64, 36)
        b_fc1 = state["fc1.bias"].numpy()
        w_fc2 = state["fc2.weight"].numpy() # (64, 64)
        b_fc2 = state["fc2.bias"].numpy()
        w_act = state["action_head.weight"].numpy() # (3, 64)
        b_act = state["action_head.bias"].numpy()
        w_col = state["collision_head.weight"].numpy() # (1, 64)
        b_col = state["collision_head.bias"].numpy()

        export_matrix_to_hex(w_fc1, os.path.join(out_dir, "baseline_w_fc1.hex"))
        export_matrix_to_hex(b_fc1, os.path.join(out_dir, "baseline_b_fc1.hex"))
        export_matrix_to_hex(w_fc2, os.path.join(out_dir, "baseline_w_fc2.hex"))
        export_matrix_to_hex(b_fc2, os.path.join(out_dir, "baseline_b_fc2.hex"))
        export_matrix_to_hex(w_act, os.path.join(out_dir, "baseline_w_act.hex"))
        export_matrix_to_hex(b_act, os.path.join(out_dir, "baseline_b_act.hex"))

        quant_report["baseline"] = {"w_fc1": w_fc1.shape, "w_fc2": w_fc2.shape, "w_act": w_act.shape}

    # 2. LeWorldModel
    lewm_path = os.path.join(ckpt_dir, "lewm_best.pt")
    if os.path.exists(lewm_path):
        state = torch.load(lewm_path, map_location="cpu")
        w_p1 = state["pred_fc1.weight"].numpy() # (64, 35)
        b_p1 = state["pred_fc1.bias"].numpy()
        w_p2 = state["pred_fc2.weight"].numpy() # (64, 64)
        b_p2 = state["pred_fc2.bias"].numpy()
        w_del = state["pred_delta.weight"].numpy() # (32, 64)
        b_del = state["pred_delta.bias"].numpy()

        export_matrix_to_hex(w_p1, os.path.join(out_dir, "lewm_w_fc1.hex"))
        export_matrix_to_hex(b_p1, os.path.join(out_dir, "lewm_b_fc1.hex"))
        export_matrix_to_hex(w_p2, os.path.join(out_dir, "lewm_w_fc2.hex"))
        export_matrix_to_hex(b_p2, os.path.join(out_dir, "lewm_b_fc2.hex"))
        export_matrix_to_hex(w_del, os.path.join(out_dir, "lewm_w_delta.hex"))
        export_matrix_to_hex(b_del, os.path.join(out_dir, "lewm_b_delta.hex"))

        quant_report["lewm"] = {"w_pred1": w_p1.shape, "w_pred2": w_p2.shape, "w_delta": w_del.shape}

    # 3. H-JEPA Model
    hjepa_path = os.path.join(ckpt_dir, "hjepa_best.pt")
    if os.path.exists(hjepa_path):
        state = torch.load(hjepa_path, map_location="cpu")
        w_m1 = state["macro_predictor.fc1.weight"].numpy()
        b_m1 = state["macro_predictor.fc1.bias"].numpy()
        w_m2 = state["macro_predictor.fc2.weight"].numpy()
        b_m2 = state["macro_predictor.fc2.bias"].numpy()
        w_md = state["macro_predictor.delta_head.weight"].numpy()
        b_md = state["macro_predictor.delta_head.bias"].numpy()

        export_matrix_to_hex(w_m1, os.path.join(out_dir, "hjepa_w_macro1.hex"))
        export_matrix_to_hex(b_m1, os.path.join(out_dir, "hjepa_b_macro1.hex"))
        export_matrix_to_hex(w_m2, os.path.join(out_dir, "hjepa_w_macro2.hex"))
        export_matrix_to_hex(b_m2, os.path.join(out_dir, "hjepa_b_macro2.hex"))
        export_matrix_to_hex(w_md, os.path.join(out_dir, "hjepa_w_macro_delta.hex"))
        export_matrix_to_hex(b_md, os.path.join(out_dir, "hjepa_b_macro_delta.hex"))

        quant_report["hjepa"] = {"w_macro1": w_m1.shape, "w_macro2": w_m2.shape, "w_macro_delta": w_md.shape}

    print("Quantization report:", quant_report)
    return quant_report

if __name__ == "__main__":
    ckpt = "/home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/checkpoints"
    out = "/home/hadynata/vs-code-data/hdl/project-haphaestus/world_models_fpga/fpga/vectors"
    quantize_all_models(ckpt, out)
