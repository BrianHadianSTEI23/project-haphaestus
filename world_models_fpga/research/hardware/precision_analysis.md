# Numerical Precision Analysis: Signed Q8.8 Fixed-Point Synthesis

**Author:** Brian A. Hadian  
**Project:** Action-Conditioned World Models & Language Navigation on AMD Kria KR260 FPGA  

## 1. Why Signed Q8.8 Fixed-Point Arithmetic?

The AMD Kria KR260 FPGA contains 1,248 **DSP48E2 slices**. Each slice natively supports a signed **27-bit by 18-bit multiplier** with a 48-bit accumulator.

1. **Single-DSP Operation:** A 16-bit $\times$ 16-bit signed multiplication maps into **exactly one DSP48E2 slice** without logic tiling or cascading.
2. **Dynamic Range Suitability:** SAE feature analysis in Phase 4 established that maximum latent activations across all layers remain bounded in $[-10.75, +10.75]$. Signed Q8.8 provides a dynamic range of $[-128.0, +127.99609375]$ with a step resolution of $\epsilon = 2^{-8} = 0.00390625$, giving **$11.9\times$ headroom** before saturation.
3. **Accumulator Headroom:** Using the 48-bit internal accumulator allows summing up to $2^{32}$ products without intermediate overflow before down-scaling and saturation clipping.

## 2. Activation Function Quantization & Approximation

- **ReLU:** $\max(0, x)$ requires 0 DSPs; implemented via a 1-bit sign multiplexer.
- **LayerNorm:** Evaluated using a 5-region Piecewise Linear (PWL) LUT for inverse square root $1/\sqrt{x}$.
- **Sigmoid (Safety Barrier Head):** 4-segment PWL approximation: $\sigma(x) \approx 0.5 + 0.25x$ for $|x| \le 2$, saturated to 0 or 1 elsewhere.
- **Tanh (Motor Policy):** 4-segment PWL approximation: $\tanh(x) \approx x$ for $|x| < 0.5$, $\text{sign}(x) \cdot (0.5 + 0.5|x|)$ up to $|x| < 1.0$, saturated to $\pm 1.0$ elsewhere.
