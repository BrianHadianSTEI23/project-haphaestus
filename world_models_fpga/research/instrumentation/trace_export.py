"""
Phase 3: Trace Export and Large-Scale Activation Logging Utilities
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Provides high-throughput activation recording and serialization:
  - Diagnostic Mode: Full layer tensor dumps for single-episode deep audits
  - Selected Mode: Compressed latent-only archives for SAE training (Phase 4)
  - Statistical Mode: Continuous running statistics (mean, std, sparsity) for large datasets
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List

from research.instrumentation.model_inspector import ModelInspector

class TraceCollector:
    """
    Orchestrates activation extraction over batches or episodes without unbounded RAM growth.
    """
    def __init__(self, inspector: ModelInspector, mode: str = "selected", max_records: int = 10000):
        self.inspector = inspector
        self.mode = mode
        self.max_records = max_records
        self.buffer = {}
        self.running_stats = {}
        self.record_count = 0

    def record_step(self, *inputs):
        """
        Executes hooked forward pass and ingests activations into buffer.
        """
        trace = self.inspector.capture_activations(*inputs)
        self.record_count += 1

        if self.mode == "statistical":
            # Update running moments in-place
            for k, v in trace.items():
                if isinstance(v, torch.Tensor):
                    arr = v.float().numpy()
                    if k not in self.running_stats:
                        self.running_stats[k] = {
                            "count": 0,
                            "sum": np.zeros(arr.shape[1:], dtype=np.float64),
                            "sq_sum": np.zeros(arr.shape[1:], dtype=np.float64),
                            "zeros": 0
                        }
                    st = self.running_stats[k]
                    st["count"] += arr.shape[0]
                    st["sum"] += np.sum(arr, axis=0, dtype=np.float64)
                    st["sq_sum"] += np.sum(arr**2, axis=0, dtype=np.float64)
                    st["zeros"] += np.sum(np.abs(arr) < 1e-6)

        else: # 'selected' or 'diagnostic'
            for k, v in trace.items():
                if self.mode == "selected" and not any(sub in k for sub in ["latent", "delta", "action", "col"]):
                    continue

                if isinstance(v, torch.Tensor):
                    arr = v.numpy()
                    if k not in self.buffer:
                        self.buffer[k] = []
                    self.buffer[k].append(arr)

    def export_and_reset(self, output_path: str):
        """
        Flushes buffered traces to disk and frees RAM.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if self.mode == "statistical":
            final_stats = {}
            for k, st in self.running_stats.items():
                n = max(1, st["count"])
                mean = st["sum"] / n
                var = np.maximum(0.0, (st["sq_sum"] / n) - (mean**2))
                std = np.sqrt(var)
                sparsity = st["zeros"] / (n * mean.size)
                final_stats[k] = {
                    "shape": list(mean.shape),
                    "mean_magnitude": float(np.mean(np.abs(mean))),
                    "mean_std": float(np.mean(std)),
                    "sparsity": float(sparsity),
                    "total_samples": n
                }
            with open(output_path, "w") as f:
                json.dump(final_stats, f, indent=2)
            self.running_stats.clear()

        else:
            npz_dict = {}
            for k, arr_list in self.buffer.items():
                npz_dict[k] = np.concatenate(arr_list, axis=0)
            np.savez_compressed(output_path, **npz_dict)
            self.buffer.clear()

        print(f"Exported {self.record_count} trace steps to {output_path}")
        self.record_count = 0
