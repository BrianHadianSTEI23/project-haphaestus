# Phase 3: Neural Network Instrumentation Architecture

**Author:** Brian A. Hadian  
**Project:** Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA  
**Target Platform:** AMD Kria KR260 Robotics Starter Kit (Zynq UltraScale+ XCK26 SOM)  
**Host Architecture:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.6, PyTorch 2.14.1+cu126)  
**Reference Directive:** Section 3 (Phase 3) & Master Research Directive in `research_direction.md`

---

## 1. Executive Summary & Objective

To reverse-engineer the computational mechanisms responsible for language-grounded navigation and world modeling, the neural networks' internal operations must be rendered fully observable without altering their behavior.

In accordance with Phase 3 of `research_direction.md`, this framework implements the unified `ModelInspector` system:
- **PyTorch Forward Hooks**: Non-destructive activation capture at all intermediate layer boundaries.
- **Full Backward Hooks**: Gradient capture for feature attribution and saliency maps.
- **Activation Patching Contexts**: In-flight replacement of target activations (essential for causal circuit discovery in Phase 5).
- **Feature Channel Ablation**: Clean zeroing or dataset-mean imputation across individual latent channels.
- **Configurable Trace Serialization**: Diagnostic (full dump), Selected (bottleneck latents), and Statistical (running moments) storage.

---

## 2. Hook Architecture & Lifecycle Management

The framework centers on `HookManager` and `ModelInspector`:

```
+----------------------------------------------------------------------------------------------------+
|                                    MODELINSPECTOR FRAMEWORK                                        |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|    +-----------------------------+               +--------------------------------------------+    |
|    |      Input Tensors          |               |             ModelInspector                 |    |
|    |  (obs_rgb, goal, action)    |               |  - register_hooks()                        |    |
|    +--------------+--------------+               |  - capture_activations()                   |    |
|                   |                              |  - capture_gradients()                     |    |
|                   v                              |  - extract_weights()                       |    |
|    +-----------------------------+               |  - patch_activation()                      |    |
|    |  Target Neural Network      |               |  - ablate_component()                      |    |
|    |  (Baseline / LeWM / H-JEPA) |               |  - export_trace()                          |    |
|    +--------------+--------------+               +---------------------+----------------------+    |
|                   |                                                    |                           |
|                   | Forward Execution                                  | Hook Attachment           |
|                   v                                                    v                           |
|         [Module Layer Bound] <================================= [HookManager]                     |
|                   |                                              - forward_hook                    |
|                   |---> Detached Clone (CPU/CUDA)                - backward_hook                   |
|                   |---> Patching Replacement (if active)         - patch_hook                      |
|                   |---> Channel Ablation Mask (if active)        - ablation_hook                   |
|                   v                                                    |                           |
|    +-----------------------------+                                     v                           |
|    |        Model Output         |                       +-----------------------------+           |
|    | (Unchanged: ||y'-y||=0.0)   |                       |    Serialized Traces        |           |
|    +-----------------------------+                       | (.npz / .json / stats)      |           |
|                                                          +-----------------------------+           |
+----------------------------------------------------------------------------------------------------+
```

### 2.1 Strict Non-Interference Guarantee
A common pitfall in mechanistic interpretability is hook overhead that mutates in-place buffers or alters graph execution semantics. The `ModelInspector` guarantees:
1. All captured activations are cloned using `.detach().clone()`.
2. Output tensors returned to the caller are mathematically identical to unhooked executions: $||\mathbf{y}_{\text{hooked}} - \mathbf{y}_{\text{clean}}||_\infty = 0.00\times 10^0$.
3. When context managers (`patch_activation`, `ablate_component`) exit, all hook handles are automatically removed from PyTorch's internal `_forward_hooks` dictionary.

---

## 3. Configurable Trace Collection Modes

To avoid unbounded RAM and disk consumption during large-scale evaluation, the framework provides three distinct serialization tiers:

| Mode | Format | Storage Target | Typical Size | Intended Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **Diagnostic Mode** | Compressed `.npz` | All intermediate conv maps, flattenings, and dense layers | ~435 KB / 10 steps | Single-episode in-depth causal inspection & layer audits |
| **Selected Mode** | Compressed `.npz` | Latent representations only ($z, s_t, S_k, \Delta s$, actions) | ~1.2 KB / 10 steps | Training Sparse Autoencoders (SAEs) and diagnostic probes |
| **Statistical Mode**| JSON | Welford running moments (mean, std, L1, L2 norm, sparsity) | ~2.5 KB total | Evaluation over tens of thousands of frames |
