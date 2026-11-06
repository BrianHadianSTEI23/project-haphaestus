"""
Phase 3: Neural Network Instrumentation - Low-Level Hook Framework
Author: Brian A. Hadian
Project: Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA

Provides modular PyTorch hooks for:
  - Activation Capture (Forward Hooks)
  - Gradient Capture (Backward Hooks)
  - Activation Patching (Intervention Hooks)
  - Feature Channel Ablation (Zeroing / Mean Ablation)
  - Context Managers for Safe In-Flight Interventions
"""

import torch
import torch.nn as nn
from contextlib import contextmanager

class HookManager:
    """
    Tracks and cleanly manages lifecycle of PyTorch forward/backward hooks.
    Ensures zero dangling hooks and strictly non-destructive model evaluation.
    """
    def __init__(self):
        self.handles = []
        self.activations = {}
        self.gradients = {}
        self.patch_values = {}
        self.ablation_masks = {}

    def register_forward_capture(self, module: nn.Module, name: str, retain_grad: bool = False):
        """
        Registers a forward hook capturing output activation.
        """
        def hook_fn(mod, inp, out):
            if isinstance(out, tuple):
                # If tuple output (e.g., (next_s, col_logit)), store individual tensors
                self.activations[name] = [o.detach().clone() if isinstance(o, torch.Tensor) else o for o in out]
            elif isinstance(out, torch.Tensor):
                self.activations[name] = out.detach().clone()
            else:
                self.activations[name] = out

            if retain_grad and isinstance(out, torch.Tensor) and out.requires_grad:
                out.retain_grad()

        handle = module.register_forward_hook(hook_fn)
        self.handles.append(handle)
        return handle

    def register_backward_capture(self, module: nn.Module, name: str):
        """
        Registers a backward hook capturing input/output gradients.
        """
        def hook_fn(mod, grad_inp, grad_out):
            if isinstance(grad_out, tuple):
                self.gradients[name] = [g.detach().clone() if isinstance(g, torch.Tensor) else g for g in grad_out]
            elif isinstance(grad_out, torch.Tensor):
                self.gradients[name] = grad_out.detach().clone()

        handle = module.register_full_backward_hook(hook_fn)
        self.handles.append(handle)
        return handle

    def register_patching_hook(self, module: nn.Module, name: str, patch_tensor: torch.Tensor):
        """
        Replaces the module's output with patch_tensor during forward execution.
        """
        def hook_fn(mod, inp, out):
            if isinstance(out, tuple):
                # Replace primary tensor, retain remaining tuple items
                new_out = list(out)
                new_out[0] = patch_tensor.to(out[0].device)
                return tuple(new_out)
            return patch_tensor.to(out.device)

        handle = module.register_forward_hook(hook_fn)
        self.handles.append(handle)
        return handle

    def register_ablation_hook(self, module: nn.Module, name: str, channel_indices=None, mode: str = "zero", mean_tensor=None):
        """
        Ablates specified feature channels (mode='zero' or mode='mean').
        """
        def hook_fn(mod, inp, out):
            is_tuple = isinstance(out, tuple)
            target = out[0] if is_tuple else out

            if not isinstance(target, torch.Tensor):
                return out

            modified = target.clone()
            if channel_indices is None:
                # Ablate entire layer
                if mode == "zero":
                    modified.zero_()
                elif mode == "mean" and mean_tensor is not None:
                    modified = mean_tensor.to(target.device).expand_as(modified)
            else:
                # Ablate specific channel indices
                if mode == "zero":
                    modified[..., channel_indices] = 0.0
                elif mode == "mean" and mean_tensor is not None:
                    modified[..., channel_indices] = mean_tensor[..., channel_indices].to(target.device)

            if is_tuple:
                res = list(out)
                res[0] = modified
                return tuple(res)
            return modified

        handle = module.register_forward_hook(hook_fn)
        self.handles.append(handle)
        return handle

    def clear(self):
        """Removes all registered hooks from models."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.activations.clear()
        self.gradients.clear()
        self.patch_values.clear()
        self.ablation_masks.clear()

    @contextmanager
    def patch_context(self, module: nn.Module, name: str, patch_tensor: torch.Tensor):
        """Context manager for temporary activation replacement."""
        handle = self.register_patching_hook(module, name, patch_tensor)
        try:
            yield
        finally:
            handle.remove()
            if handle in self.handles:
                self.handles.remove(handle)

    @contextmanager
    def ablate_context(self, module: nn.Module, name: str, channel_indices=None, mode: str = "zero", mean_tensor=None):
        """Context manager for temporary ablation."""
        handle = self.register_ablation_hook(module, name, channel_indices=channel_indices, mode=mode, mean_tensor=mean_tensor)
        try:
            yield
        finally:
            handle.remove()
            if handle in self.handles:
                self.handles.remove(handle)
