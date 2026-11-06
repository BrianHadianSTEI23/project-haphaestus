"""
Phase 3: ModelInspector - Unified Mechanistic Instrumentation Framework
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Implements the unified ModelInspector class required by Section 3 (Phase 3):
  - register_hooks(model, architecture_type)
  - capture_activations(inputs)
  - capture_gradients(loss)
  - extract_weights()
  - patch_activation(layer_name, patch_tensor)
  - ablate_component(layer_name, channel_indices, mode)
  - export_trace(output_path, mode)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional

from research.instrumentation.hooks import HookManager
from research.instrumentation.activation_capture import ModelLayerRegistry

class ModelInspector:
    def __init__(self, model: nn.Module, model_type: str = "baseline"):
        """
        model_type: 'baseline', 'lewm', or 'hjepa'
        """
        self.model = model
        self.model_type = model_type.lower()
        self.device = next(model.parameters()).device
        self.hook_mgr = HookManager()
        self.model.eval()

        # Cache of named modules
        self.named_modules = dict(model.named_modules())
        self.register_hooks(self.model)

    def register_hooks(self, model: nn.Module):
        """
        Registers non-interfering forward capture hooks across all key layers.
        """
        self.hook_mgr.clear()
        if "baseline" in self.model_type:
            ModelLayerRegistry.instrument_baseline(model, self.hook_mgr)
        elif "lewm" in self.model_type:
            ModelLayerRegistry.instrument_lewm(model, self.hook_mgr)
        elif "hjepa" in self.model_type:
            ModelLayerRegistry.instrument_hjepa(model, self.hook_mgr)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def capture_activations(self, *inputs) -> Dict[str, Any]:
        """
        Executes a forward pass and returns captured activation dictionary.
        Guarantees that model is in eval mode and outputs are strictly non-interfered.
        """
        self.hook_mgr.activations.clear()
        with torch.no_grad():
            if hasattr(self.model, "predict_next") and len(inputs) == 2 and inputs[0].dim() == 2 and inputs[1].dim() == 2:
                outputs = self.model.predict_next(*inputs)
            elif hasattr(self.model, "macro_predictor") and len(inputs) == 2 and inputs[0].dim() == 2:
                outputs = self.model.macro_predictor(*inputs)
            elif hasattr(self.model, "encode") and len(inputs) == 1 and inputs[0].dim() == 4:
                outputs = self.model.encode(*inputs)
            else:
                outputs = self.model(*inputs)

        # Build clean detached dictionary
        trace = {}
        for k, v in self.hook_mgr.activations.items():
            if isinstance(v, torch.Tensor):
                trace[k] = v.cpu()
            elif isinstance(v, list):
                trace[k] = [item.cpu() if isinstance(item, torch.Tensor) else item for item in v]
            else:
                trace[k] = v

        def to_cpu(item):
            if isinstance(item, torch.Tensor):
                return item.cpu()
            elif isinstance(item, (tuple, list)):
                return [to_cpu(x) for x in item]
            return item

        trace["model_output"] = to_cpu(outputs)
        return trace

    def capture_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Backpropagates loss and returns captured layer gradients.
        """
        self.hook_mgr.gradients.clear()
        loss.backward(retain_graph=True)

        grad_dict = {}
        for k, v in self.hook_mgr.gradients.items():
            if isinstance(v, torch.Tensor):
                grad_dict[k] = v.cpu()
            elif isinstance(v, list):
                grad_dict[k] = [g.cpu() if isinstance(g, torch.Tensor) else g for g in v]
        return grad_dict

    def extract_weights(self) -> Dict[str, Dict[str, Any]]:
        """
        Extracts weight matrices, bias vectors, shapes, and sparsity stats.
        """
        weight_catalog = {}
        for name, param in self.model.named_parameters():
            p_data = param.detach().cpu().numpy()
            sparsity = float(np.mean(np.abs(p_data) < 1e-6))
            weight_catalog[name] = {
                "shape": list(param.shape),
                "numel": param.numel(),
                "mean": float(np.mean(p_data)),
                "std": float(np.std(p_data)),
                "l1_norm": float(np.sum(np.abs(p_data))),
                "l2_norm": float(np.linalg.norm(p_data)),
                "sparsity": sparsity,
                "requires_grad": param.requires_grad
            }
        return weight_catalog

    def patch_activation(self, layer_name: str, patch_tensor: torch.Tensor):
        """
        Returns a context manager that temporarily replaces layer_name's activation.
        """
        module = self._resolve_module(layer_name)
        return self.hook_mgr.patch_context(module, layer_name, patch_tensor)

    def ablate_component(self, layer_name: str, channel_indices=None, mode: str = "zero", mean_tensor=None):
        """
        Returns a context manager that temporarily ablates channels in layer_name.
        """
        module = self._resolve_module(layer_name)
        return self.hook_mgr.ablate_context(module, layer_name, channel_indices=channel_indices, mode=mode, mean_tensor=mean_tensor)

    def _resolve_module(self, layer_name: str) -> nn.Module:
        """Helper to resolve dot-notated layer name to PyTorch nn.Module."""
        # Check direct named_modules
        if layer_name in self.named_modules:
            return self.named_modules[layer_name]

        # Model-specific short aliases
        if "baseline" in layer_name:
            if layer_name == "baseline.conv1": return self.model.encoder.conv1
            if layer_name == "baseline.conv2": return self.model.encoder.conv2
            if layer_name == "baseline.conv3": return self.model.encoder.conv3
            if layer_name == "baseline.conv4": return self.model.encoder.conv4
            if layer_name == "baseline.visual_latent_z": return self.model.encoder.fc
            if layer_name == "baseline.policy_fc1": return self.model.fc1
            if layer_name == "baseline.policy_fc2": return self.model.fc2
            if layer_name == "baseline.action_head": return self.model.action_head
            if layer_name == "baseline.col_head": return self.model.collision_head
        elif "lewm" in layer_name:
            if layer_name == "lewm.conv1": return self.model.encoder[0]
            if layer_name == "lewm.conv2": return self.model.encoder[2]
            if layer_name == "lewm.conv3": return self.model.encoder[4]
            if layer_name == "lewm.conv4": return self.model.encoder[6]
            if layer_name == "lewm.flatten": return self.model.encoder[8]
            if layer_name == "lewm.latent_state_s": return self.model.encoder[10]
            if layer_name == "lewm.pred_fc1": return self.model.pred_fc1
            if layer_name == "lewm.pred_fc2": return self.model.pred_fc2
            if layer_name == "lewm.delta_s": return self.model.pred_delta
            if layer_name == "lewm.col_head": return self.model.col_fc
        elif "hjepa" in layer_name:
            if layer_name == "hjepa.conv1": return self.model.encoder[0]
            if layer_name == "hjepa.shared_latent_s": return self.model.encoder[10]
            if layer_name == "hjepa.macro_fc1": return self.model.macro_predictor.fc1
            if layer_name == "hjepa.macro_fc2": return self.model.macro_predictor.fc2
            if layer_name == "hjepa.macro_delta_S": return self.model.macro_predictor.delta_head
            if layer_name == "hjepa.macro_col_head": return self.model.macro_predictor.col_head
            if layer_name == "hjepa.micro_pol_fc1": return self.model.micro_predictor.pol_fc1
            if layer_name == "hjepa.micro_action_head": return self.model.micro_predictor.action_head
            if layer_name == "hjepa.micro_dyn_fc1": return self.model.micro_predictor.dyn_fc1
            if layer_name == "hjepa.micro_delta_s": return self.model.micro_predictor.delta_head
            if layer_name == "hjepa.micro_col_head": return self.model.micro_predictor.col_head

        # Recursive traversal
        parts = layer_name.split(".")
        curr = self.model
        for p in parts:
            if hasattr(curr, p):
                curr = getattr(curr, p)
            elif p.isdigit() and isinstance(curr, (nn.Sequential, nn.ModuleList)):
                curr = curr[int(p)]
            else:
                raise AttributeError(f"Module {layer_name} not found in {self.model_type}")
        return curr

    def export_trace(self, trace_dict: Dict[str, Any], output_path: str, mode: str = "diagnostic"):
        """
        Exports collected activations:
          - 'diagnostic': Full uncompressed tensor dump (.npz)
          - 'selected': Only target latent bottleneck representations
          - 'statistical': JSON summary of mean, std, L1, L2, sparsity
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if mode == "statistical":
            stats = {}
            for k, v in trace_dict.items():
                if isinstance(v, torch.Tensor):
                    arr = v.float().numpy()
                    stats[k] = {
                        "shape": list(arr.shape),
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                        "l1_norm": float(np.sum(np.abs(arr))),
                        "l2_norm": float(np.linalg.norm(arr)),
                        "sparsity": float(np.mean(np.abs(arr) < 1e-5))
                    }
            with open(output_path, "w") as f:
                json.dump(stats, f, indent=2)

        elif mode == "selected":
            # Save only latent bottleneck states (z, s, S, delta)
            selected_keys = [k for k in trace_dict.keys() if any(sub in k for sub in ["latent", "delta", "action", "col", "output"])]
            npz_data = {}
            for k in selected_keys:
                val = trace_dict[k]
                if isinstance(val, torch.Tensor):
                    npz_data[k] = val.numpy()
                elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], torch.Tensor):
                    npz_data[k] = val[0].numpy()
            np.savez_compressed(output_path, **npz_data)

        elif mode == "diagnostic":
            # Full tensor dump
            npz_data = {}
            for k, v in trace_dict.items():
                if isinstance(v, torch.Tensor):
                    npz_data[k] = v.numpy()
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, torch.Tensor):
                            npz_data[f"{k}_{i}"] = item.numpy()
            np.savez_compressed(output_path, **npz_data)
        else:
            raise ValueError(f"Unknown trace export mode: {mode}")

    def close(self):
        self.hook_mgr.clear()
