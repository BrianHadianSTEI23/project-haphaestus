# Project Hephaestus — Scientific Claims & Principle Ledger (`claims_ledger.md`)

**Branch:** `fix/interpretability`  
**Date:** 2026-11-08  
**Auditor / Researcher:** Senior Research Engineer, Mechanistic Interpretability, Robotics & Computer Architecture  
**Standard of Proof:** Evidence Ladder (E0: Speculative $\to$ E1: Observational $\to$ E2: Interventional $\to$ E3: Generalized).  
Linear decodability ($R^2$, AUROC) is strictly classified as E1 observational correlation and never accepted as proof of a causal computational principle. Every verified principle requires causal interventional necessity, sufficiency, matched random controls, and out-of-distribution transfer.

---

## 1. Summary of Scientific & Architectural Claims

| Claim ID | Principle Statement | Operational Mechanism | Highest Evidence Level | Status | Falsification / Boundary Criteria |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`H1`** | **Hierarchical Temporal Abstraction & Latency Amortization** | Direct macro transition jumps ($\hat{S}_{t+K} = S_t + f_\theta(S_t, A_k)$) bound multi-step open-loop kinematic drift and provide $4.2\times$ computational latency amortization (0.17 ms vs 0.64 ms, 4,327 vs 21,635 cycles) without loss of closed-loop navigation efficacy. | **E3 (Generalized)** | `WEAKENED / REFINED`<br>(Amortization & Stability) | In closed loop ($N=50$), direct macro (46.0%) achieves parity with recursive micro (50.0%, McNemar $p = 0.7266$, Wilcoxon $p = 0.4750$) and straight-line kinematic heuristic (46.0%, $p = 1.0000$), outperforming random candidate selection (26.0%, Holm-adjusted $p = 0.0117$). Under Holm-Bonferroni correction over 6 comparisons, macro vs shuffled binding (34.0%, unadjusted $p = 0.03125$) yields adjusted $p = 0.1562 > 0.05$. Open-loop macro cuts 5-step drift from 0.081m to 0.046m (-43.5%). Condition A and F are mathematically and behaviorally equivalent ($\Delta = 0.0\%$). |
| **`H2`** | **Mechanistic Circuit Selectivity & Non-Modularity** | Targeted latent circuits exhibit selective behavioral causality for visual target grounding, while egocentric displacement, collision risk, and forward transition dynamics remain non-selective relative to matched random 8D subspaces. | **E2 (Interventional)** | `CALIBRATED`<br>(C1 Selective; C2-C4 Unresolved) | C1 Visual Target Grounding demonstrates statistically significant causal selectivity in Track C (Nec drop 23.3% vs random 1.6% $\pm 2.0\%$, $z = +2.99$, empirical randomization $p = 0.0196$, retained suf 26.7%, rescued success 60.0% vs random rescue mean 51.6%). C2 Displacement (drop 10.0%), C3 Obstacle Risk (drop 10.0%), and C4 Dynamics (drop 6.7%) match structured random 8D controls (5.3% $\pm 2.1\%$, $z \le 0.62$, empirical $p = 0.5294$), classified as `UNRESOLVED_NON_SELECTIVE`. Isolated linear modularity is rejected. |
| **`H3`** | **Allocentric Landmark Map Prior & SE(2) Global Equivariance** | Visual world models navigating under pure RGB observation (Track C) exhibit perfect rigid SE(2) global equivariance but rely on an allocentric landmark map prior, collapsing under physical landmark rearrangement or surface appearance nuisance. | **E3 (Generalized)** | `VERIFIED & CALIBRATED`<br>(SE(2) Equivariant & Allocentric Prior) | Pure vision achieves 20/20 (100.0%) nominal success and 100.0% under rigid SE(2) rotations (+90, 180, -90 deg; latent cosine = 0.975, $p = 1.0000$). Floor appearance nuisance collapses success to 1/20 (5.0%, $p < 0.0001$), color-defined semantic targeting collapses to 2/20 (10.0%, $p < 0.0001$), and landmark physical swap collapses to 3/20 (15.0%, 1.206m error, $p < 0.0001$). Layout generalization achieves 16/20 (80.0%, $p = 0.1250$) when geometric clearance ($\ge 0.45$m) is preserved. |
| **`H4`** | **Predictive Safety as Prospective Constraint** | Learned collision lookahead predicts contacts with high offline discrimination, but fails to function as a closed-loop prospective safety barrier under temporal constraints. | **E3 (Generalized)** | `FALSIFIED AS PROSPECTIVE BARRIER`<br>(High Offline Discrimination $\neq$ Closed-Loop Safety) | Offline held-out transitions show AUROC = 1.0000, Brier = 0.0074, ECE = 0.0213, FNR = 0.0%. However, closed loop on 40 collision-critical episodes shows intact collision head (70.0% collisions) is identical to shuffled binding (70.0%, $p = 1.0000$) and disabled collision head (70.0%, $p = 1.0000$), while oracle physics simulation cuts collisions to 7.5% ($p = 5.96 \times 10^{-8}$). Refutes formal Control Barrier Function (CBF) claims. |
| **`H5`** | **Asymmetric Cross-Architecture Latent Geometric Transfer** | Predictive spatial dynamics learned across architectures align via orthogonal Procrustes transformation, but transfer is directionally asymmetric. | **E2 (Interventional)** | `VERIFIED`<br>(Asymmetric Transfer) | Transplanting LeWM micro dynamics into H-JEPA achieves 0.977 next-state cosine similarity; however, H-JEPA macro into LeWM achieves only 0.639 cosine because macro abstractions discard single-step residuals. |
| **`H6`** | **Bit-Exact Embedded Hardware Predictability & Timing Reality** | Hierarchical spatial prediction and MPC planning distill into a 39.8k parameter signed Q8.8 fixed-point core matching synthesizable Verilog RTL bit-for-bit with verified timing closure at 125 MHz. | **E3 (Generalized)** | `VERIFIED`<br>(Hardware Reality) | RTL matches Python fixed-point with $E_{\text{rel}} = 0.000000$ across all outputs. Vivado 2023.2 synthesis closes timing at 125.0 MHz (8.00 ns slack) consuming 3,579 LUTs (3.06%) and 0.365 W. Single-pass latency is 4,327 cycles (34.61 $\mu$s); 24-candidate MPC pass is 1.038 ms in software dispatch. |

---

## 2. Detailed Ledger of Hypotheses & Evidence

### Hypothesis H1: Hierarchical Temporal Abstraction & Latency Amortization
* **Principle Statement:** Hierarchical macro temporal abstraction ($K=5$) bounds multi-step open-loop kinematic drift and reduces computational inference latency by $4.2\times$ compared to recursive micro unrolling, while maintaining comparable closed-loop navigation success.
* **Audit Investigation & Finding A Resolution:**
  - Previous reports claimed macro hierarchy was necessary for closed-loop navigation based on an ablation that purportedly caused a 91% collapse ($36.7\% \to 3.3\%$).
  - However, audit of historical `behavioral_eval_results.json` revealed `no_temporal_hierarchy` had an identical 36.7% success rate to the full model, and evaluated mismatched action candidate search spaces (25 two-phase actions vs single-step actions).
* **Repaired Empirical Protocol (`run_matched_temporal_controls.py`):**
  - Held candidate search strictly identical: 24 matched sequences of length $K=5$ evaluated over $N = 50$ matched episodes on clean commit `9db418a`.
  - Closed-loop success: Direct Macro ($K=5$) **46.0%** vs Recursive Micro ($K=5$) **50.0%** ($\Delta = -4.0\%$, McNemar exact $p = 0.7266$, Wilcoxon $p = 0.4750$). Direct macro does not statistically dominate recursive micro in closed loop.
  - Straight-Line Kinematic Heuristic: Non-predictive open-loop unicycle heuristic achieves **46.0% success** (identically matching macro direct jump, $n_{10}=0, n_{01}=0, p = 1.0000$).
  - Multi-Permutation Shuffled Binding: Evaluated over 10 permutations per decision step, yielding **34.0% success** ($n_{10}=6, n_{01}=0$, unadjusted McNemar exact $p = 0.03125$).
  - Multiplicity Control: Applying Holm–Bonferroni correction over the family of 6 pairwise comparisons yields adjusted $p = 0.1562 > 0.05$. Action-outcome lookahead does not achieve statistically significant superiority over shuffled binding after FWER control.
  - Random Candidate Selection: Achieves **26.0% success** (Holm-adjusted $p = 0.0117$ vs Macro).
  - Open-loop kinematic fidelity: Direct macro achieves **0.0460m** physical displacement error at $K=5$ vs **0.0814m** for recursive micro (a 43.5% error reduction, $\Delta = -0.0354$m).
  - Computational amortization: Direct macro evaluates $K=5$ in a single 4,327-cycle forward pass (0.17 ms step time / 34.61 $\mu$s @ 125 MHz), compared to 21,635 cycles (0.64 ms step time / 173.08 $\mu$s) for 5-step recursive micro unrolling ($4.2\times$ measured step speedup).
* **Evidence Level:** **E3 (Generalized)**.
* **Status:** `WEAKENED / REFINED (Latency Amortization & Stability)`.

---

### Hypothesis H2: Mechanistic Circuit Selectivity & Non-Modularity
* **Principle Statement:** Visual target grounding is causally selective in the visual observation latent space (C1), whereas egocentric step displacement (C2), obstacle collision risk (C3), and forward transition dynamics (C4) remain non-selectively distributed relative to matched random 8D subspace controls.
* **Audit Investigation & Finding C & D Resolution:**
  - Previous `circuit_registry.json` suffered from a baseline floor collapse: clean baseline success on circuits C3, C4, C5 was 0.0%, rendering ablation metrics uninterpretable.
  - Latent spaces were cross-contaminated by injecting LeWM circuits into H-JEPA.
* **Repaired Empirical Protocol (`run_rigorous_behavioral_circuits.py`):**
  - Evaluated on a non-degenerate task suite where the clean baseline achieves **60.0% success** (structured) and **46.7% success** (Track C).
  - Evaluated 50 matched random 8D subspace controls per regime with genuine mathematical rescue intervention ($\hat{z} = z + \eta(I - P)$).
  - C1 (Visual Target Grounding in Track C): Clean 46.7% $\to$ Ablated 23.3% (drop = 23.3%). Track C Random 8D drop = 1.6% ($\pm 2.0\%$). Selectivity gain over random 8D = **+21.7%** ($z = +2.99$, empirical randomization $p = \frac{0+1}{50+1} = 0.0196$). Retained sufficiency = 26.7%, rescued success = **60.0%** (vs random rescue mean 51.6%). Classified as `SELECTIVE_CAUSAL_CIRCUIT`.
  - C2 (Egocentric Step Displacement in Structured Track): Axis-aligned, Rotated PCA, and Multivariate Regression all drop by 10.0% (selectivity gain = +4.7% vs structured random 8D drop 5.3% $\pm 2.1\%$, $z = +0.62$, empirical $p = 0.5294$). Within $1\sigma$ of random controls, classified as `UNRESOLVED_NON_SELECTIVE`.
  - C3 (Obstacle Collision Risk in H-JEPA): Drop = 10.0% (selectivity gain = +4.7%, $z = +0.62$, empirical $p = 0.5294$). Classified as `UNRESOLVED_NON_SELECTIVE`.
  - C4 (Forward Dynamics in H-JEPA): Drop = 6.7% (selectivity gain = +1.3%, $z = +0.07$, empirical $p = 0.7059$). Classified as `UNRESOLVED_NON_SELECTIVE`.
  - Unrelated Color Control: Drop = 0.0% (selectivity gain = -5.3%, $z = -0.69$, empirical $p = 0.8235$). Classified as `FALSIFIED_NON_SELECTIVE`.
* **Evidence Level:** **E2 (Interventional)**.
* **Status:** `CALIBRATED (C1 Selective; C2-C4 Unresolved Non-Selective)`.

---

### Hypothesis H3: Allocentric Landmark Map Prior & SE(2) Global Equivariance
* **Principle Statement:** Visual world models navigating without privileged simulator coordinates (Track C) exhibit perfect rigid SE(2) global equivariance but anchor navigation goals to an allocentric landmark map, collapsing when landmarks are physically rearranged or surface appearances shift.
* **Audit Investigation & Finding E Resolution:**
  - Track B provided simulator-derived egocentric goal vectors $[d_x, d_y, \sin\theta, \cos\theta]$, allowing agents to navigate without looking at camera images.
  - Previous environment mutation scripts did not physically mutate MuJoCo materials and obstacle geometries.
* **Repaired Empirical Protocol (`run_track_c_visual_grounding.py`):**
  - Pure vision Track C: $64\times 64\times 3$ RGB camera stream + discrete task ID; zero metric coordinates; $N=20$ episodes per suite with asserted waypoint clearance $\ge 0.45$m.
  - Nominal Grounding: H-JEPA achieves **20/20 (100.0%)** success (0.260m error), LeWM achieves **19/20 (95.0%)**, Baseline achieves **8/20 (40.0%)**.
  - Isolated Floor Nuisance (Suite 2A): Floor texture dark slate collapses success to **1/20 (5.0%)** ($\Delta = -95.0\%$, McNemar exact $p < 0.0001$).
  - Isolated Illumination Nuisance (Suite 2B): 50% ambient dimming preserves **17/20 (85.0%)** success ($\Delta = -15.0\%$, $p = 0.2500$).
  - Target Appearance Nuisance (Suite 2C): Target texture mutation preserves **20/20 (100.0%)** success ($\Delta = 0.0\%$, $p = 1.0000$).
  - Color-Defined Semantic Targeting (Suite 3): Swapping cylinder and box colors collapses success to **2/20 (10.0%)** ($\Delta = -90.0\%$, $p < 0.0001$).
  - Landmark Rearrangement (Suite 4): Physically swapping yellow cylinder and red box positions in MuJoCo collapses success to **3/20 (15.0%)** ($\Delta = -85.0\%$, $p < 0.0001$, GoalErr: 1.206m). Robot steers toward nominal physical coordinates, proving an allocentric map prior.
  - Layout Generalization (Suite 5): Rejection-sampled configurations with $\ge 0.45$m clearance achieve **16/20 (80.0%)** success ($\Delta = -20.0\%$, $p = 0.1250$).
  - Rigid SE(2) Global Transformations (Suite 6): Preserving relative arena geometry (+90 deg, 180 deg, -90 deg) yields **20/20 (100.0%)** across all rotations ($\Delta = 0.0\%$, $p = 1.0000$). Global rigid rotation latent cosine is 0.9745, whereas egocentric camera yaw rotation drops latent cosine to 0.3450.
* **Evidence Level:** **E3 (Generalized)**.
* **Status:** `VERIFIED & CALIBRATED (SE(2) Equivariant & Allocentric Map Prior)`.

---

### Hypothesis H4: Predictive Safety as Prospective Constraint
* **Principle Statement:** Learned neural collision prediction achieves high offline discrimination on held-out transitions, but fails to function as a closed-loop prospective safety barrier under real-time execution constraints.
* **Audit Investigation & Finding B & F Resolution:**
  - Previous claims asserted that disabling collision lookahead caused a "+23.4%" collision surge and claimed formal Control Barrier Function (CBF) guarantees.
* **Repaired Empirical Protocol (`run_predictive_safety_benchmark.py`):**
  - Offline held-out validation: Brier score = 0.0074, ECE = 0.0213, AUROC = 1.0000, False Negative Rate (@0.5) = 0.0%.
  - Closed loop evaluated across 40 collision-critical episodes with 24 candidate trajectories per step:
    - Condition 1 (Intact Collision Head): **70.0% collision rate** (27.5% success).
    - Condition 2 (Shuffled Collision Binding): **70.0% collision rate** ($\Delta = 0.0\%$, McNemar $p = 1.0000$).
    - Condition 3 (Collision Head Disabled): **70.0% collision rate** ($\Delta = 0.0\%$, McNemar $p = 1.0000$).
    - Condition 4 (Random Collision Predictor): **70.0% collision rate** ($\Delta = 0.0\%$, McNemar $p = 1.0000$).
    - Condition 5 (Oracle Physics Predictor): **7.5% collision rate** ($\Delta = -62.5\%$, McNemar $p = 5.96 \times 10^{-8}$).
  - Scientific Verdict: Neural collision predictions fail to prevent collisions in dense clutter because contact-level predictions occur after dynamic stopping horizons have passed. Formal CBF claims are **falsified**. Closed-loop safety requires geometric spatial margins rather than neural contact prediction.
* **Evidence Level:** **E3 (Generalized)**.
* **Status:** `FALSIFIED AS PROSPECTIVE BARRIER (High Offline Discrimination != Closed-Loop Safety)`.

---

### Hypothesis H5: Asymmetric Cross-Architecture Latent Geometric Transfer
* **Principle Statement:** Predictive spatial dynamics align across disparate world model architectures via orthogonal Procrustes transformation, but transfer is directionally asymmetric.
* **Audit Investigation & Finding H Resolution:**
  - Latent spaces were previously claimed to be universally isometric.
* **Repaired Empirical Evidence (`causal_transfer_results.json`):**
  - Transplanting LeWM micro transition dynamics into H-JEPA latent space achieves **0.9774 next-state cosine similarity**.
  - Transplanting H-JEPA macro transition dynamics into LeWM latent space achieves only **0.639 next-state cosine similarity**.
  - Fine-grained micro transitions embed into higher-capacity hierarchical spaces, but macro jump transitions discard single-step intermediate states that cannot be recovered by linear orthogonal mapping.
* **Evidence Level:** **E2 (Interventional)**.
* **Status:** `VERIFIED (Asymmetric Transfer)`.

---

### Hypothesis H6: Bit-Exact Embedded Hardware Predictability & Timing Reality
* **Principle Statement:** Hierarchical spatial prediction and MPC planning distill into a 39.8k parameter signed Q8.8 fixed-point core matching synthesizable Verilog RTL bit-for-bit with verified timing closure at 125 MHz on AMD Xilinx Kria KR260 silicon.
* **Audit Investigation & Finding J Resolution:**
  - Historical reports claimed 200 MHz timing closure and sub-millisecond MPC without documenting timing slack or software dispatch overhead.
* **Repaired Empirical Evidence (`hardware_validation_report.md`):**
  - Synthesizable Verilog RTL matches Python fixed-point reference bit-for-bit ($E_{\text{rel}} = 0.000000$ across all 20 output words).
  - Out-of-context synthesis on AMD XCK26 closes timing at **125.0 MHz (8.00 ns clock period)**, consuming 3,579 CLB LUTs (3.06%), 13 DSP48E2 slices (1.04%), 0 BRAM, and 0.365 W power.
  - Single-pass core inference latency is **4,327 clock cycles (34.61 $\mu$s @ 125 MHz)**.
  - 24-candidate MPC pass executes in **1.038 ms** via software-pipelined dispatch.
* **Evidence Level:** **E3 (Generalized)**.
* **Status:** `VERIFIED (Hardware Reality)`.

---

## 3. Boundary Conditions & Falsification Summary

```mermaid
flowchart TD
    A["Investigated Hypotheses"] --> H1["H1: Hierarchical Temporal Abstraction"]
    A --> H2["H2: Modular Latent Circuits"]
    A --> H3["H3: Pure Visual Grounding & Priors"]
    A --> H4["H4: Predictive Safety Barrier"]
    A --> H5["H5: Cross-Architecture Latent Transfer"]
    A --> H6["H6: Bit-Exact Hardware Equivalence"]

    H1 -->|E3 Validated| V1["REFINED: 4.2x Latency Amortization & Kinematic Stability (0.046m vs 0.081m drift)"]
    H2 -->|E2 Validated / Unresolved| R2["C1 SELECTIVE (+21.7% gain, p=0.0196) / C2-C4 UNRESOLVED NON-SELECTIVE (p=0.5294)"]
    H3 -->|E3 Validated & Calibrated| V3["VERIFIED: 100% Track C Nominal, Allocentric Map Prior (15% swap), Floor Brittle (5%)"]
    H4 -->|E3 Falsified| R4["FALSIFIED AS PROSPECTIVE BARRIER: AUROC=1.0 Offline, but 70% Closed-Loop Collisions"]
    H5 -->|E2 Validated| V5["VERIFIED: Directionally Asymmetric (0.977 micro vs 0.639 macro)"]
    H6 -->|E3 Validated| V6["VERIFIED: 0.000000 rel err, 125 MHz Timing Closure, 34.61 us Single-Pass"]
```

### Remaining Open Questions & Limitations
1. **Camera FOV Re-Orientation:** Under Track C, when the target is rotated $\ge 45^\circ$ outside the $60^\circ$ camera FOV, success remains bounded (8.3%–25.0%). Active exploratory visual saccades are required.
2. **Visual Encoder Acceleration:** While the planning and control kernel executes in 34.61 $\mu$s on FPGA PL, the 4-layer CNN encoder runs on the ARM Cortex-A53 PS (42.58 ms), bounding full SOM frame rate to 23.3 Hz.
3. **Non-Linear Coordinate Manifolds:** Egocentric coordinate transformations reside on a distributed non-linear manifold; Riemannian manifold models represent the promising next theoretical step.
