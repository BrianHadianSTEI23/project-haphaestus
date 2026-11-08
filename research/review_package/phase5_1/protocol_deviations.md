# Phase 5.1 Protocol Deviations & Methodological Corrections

**Repository:** `BrianHadianSTEI23/project-haphaestus`  
**Branch:** `fix/interpretability`  
**Execution Commit:** `1b104decca63d95600d3265a1417d6377e4fb7b6`  
**Date:** 2026-11-08  

---

## 1. Summary of Protocol Deviations

| Domain | Historical Protocol (Phase 4 / Phase 5 v4) | Corrected Protocol (Phase 5.1 v5) | Methodological Justification |
| :--- | :--- | :--- | :--- |
| **Simulator Initialization** | Called `env.reset(options={"task_spec": spec})` which was unhandled by `JetAutoEnv.reset()`. | Enforced 5-step initialization: `restore_physical_baseline()`, `apply_task_spec()`, `reset(options={"initial_pos": ...})`, followed by explicit runtime assertions on robot pose ($\le 10^{-3}$), obstacle geoms ($\le 10^{-3}$), and visibility ($\ge 100$ px). | Guarantees that intended obstacles and robot poses are physically present in MuJoCo. |
| **Instance Binding Scope** | Evaluated `yellow_cylinder` vs `cyan_cylinder` and interpreted 5.0% dual discrimination as instance binding failure. | Audited `encode_task_id()` vocabulary (6 color-specific targets $\times$ 8 relations); documented that instance distinction without color labels is architecturally unrepresentable; invalidated 5.0% result. | Prevents confusing category discrimination with instance binding. |
| **Permutation Testing** | Flipped token assignments independently in Scene A and Scene B. | Pair-clustered permutation test (100,000 resamples): 1 random coin flip $s_i \in \{0, 1\}$ per pair applied consistently across both Scene A and Scene B. | Preserves paired factorial counterfactual structure. |
| **Interaction Alignment** | Reported one-sided $p=0.9997$ in positive direction and interpreted as negative alignment. | Evaluated explicit positive ($p_{\text{pos}}$), negative ($p_{\text{neg}}$), and two-sided ($p_{\text{two}}$) sign-flip tests. | Clarifies test direction and avoids misinterpreting large positive p-values as significant negative alignment. |
| **Causal Metrics** | Computed 4D Euclidean norm of $[dx, dy, \sin\theta, \cos\theta]$ and labeled in meters. | Separated Translation ($d_{xy} = \sqrt{dx^2 + dy^2}$ in meters) from Orientation (wrapped angular difference in radians/degrees). | Prevents combining meters and dimensionless trigonometric outputs into an invalid metric. |
| **Representation Profiling** | Pooled all units into a single scalar for Treves-Rolls sparseness. | Computed lifetime sparseness across observations per unit, population sparseness across units per observation, and participation ratio of covariance eigenvalues. | Conforms to standard computational neuroscience definitions (Treves & Rolls 1991). |
| **Ego-Motion Alignment** | Compared forward transition $\hat{s}_{t+1}(s_t, a_t)$ against previous step's displacement ($t-1 \to t$). | Aligned forward prediction with the exact transition target ($t \to t+1$) caused by `action_t`. | Resolves off-by-one temporal misalignment in offline calibration. |
| **Memory Replay Comparison** | Claimed hold policy outperformed fixed blending despite higher mean tracking error. | Reported exact paired difference $E_{\text{fixed}} - E_{\text{hold}}$, two-sided Wilcoxon tests, bootstrap CIs, and TOST equivalence between Action Hold and Oracle Hold. | Provides transparent reporting of empirical data and mechanisms. |

---

## 2. Mathematical Formulations & Invariants

### 2.1 Pair-Clustered Permutation Test
For independent pair $i \in \{1, \dots, N\}$:
- Scene A dual discrimination indicators:
  - Intact: $y_{i,A} = \mathbf{1}[d(g_{aa}, gt_a^A) < d(g_{aa}, gt_b^A) \land d(g_{ab}, gt_b^A) < d(g_{ab}, gt_a^A)]$
  - Swapped: $y_{i,A}^{\text{swap}} = \mathbf{1}[d(g_{ab}, gt_a^A) < d(g_{ab}, gt_b^A) \land d(g_{aa}, gt_b^A) < d(g_{aa}, gt_a^A)]$
- Scene B dual discrimination indicators:
  - Intact: $y_{i,B} = \mathbf{1}[d(g_{ba}, gt_a^B) < d(g_{ba}, gt_b^B) \land d(g_{bb}, gt_b^B) < d(g_{bb}, gt_a^B)]$
  - Swapped: $y_{i,B}^{\text{swap}} = \mathbf{1}[d(g_{bb}, gt_a^B) < d(g_{bb}, gt_b^B) \land d(g_{ba}, gt_b^B) < d(g_{ba}, gt_a^B)]$

In resample $m \in \{1, \dots, M\}$:
Draw $s_i \sim \text{Bernoulli}(0.5)$ for each pair $i$:
$$T^{(m)} = \frac{1}{2N} \sum_{i=1}^N \left[ (1 - s_i) y_{i,A} + s_i y_{i,A}^{\text{swap}} + (1 - s_i) y_{i,B} + s_i y_{i,B}^{\text{swap}} \right]$$
Observed statistic:
$$T_{\text{obs}} = \frac{1}{2N} \sum_{i=1}^N (y_{i,A} + y_{i,B})$$
P-values:
$$p_{\text{one\_sided}} = \frac{1}{M + 1} \left( 1 + \sum_{m=1}^M \mathbf{1}[T^{(m)} \ge T_{\text{obs}}] \right)$$
$$p_{\text{two\_sided}} = \frac{1}{M + 1} \left( 1 + \sum_{m=1}^M \mathbf{1}[|T^{(m)} - \mu_{\text{null}}| \ge |T_{\text{obs}} - \mu_{\text{null}}|] \right)$$

### 2.2 Sign-Flip Permutation Test for Cosine Alignment
For each pair $i$, interaction vector $I_i = (g_{bb} - g_{ba}) - (g_{ab} - g_{aa})$ and physical target swap displacement $\Delta P_i = gt_{b}^A - gt_{a}^A$:
$$\cos_i = \frac{I_i \cdot \Delta P_i}{\|I_i\| \|\Delta P_i\|}$$
Observed mean: $\overline{\cos} = \frac{1}{N} \sum_{i=1}^N \cos_i$.
Under $H_0$, draw signs $\epsilon_i \in \{-1, +1\}$ with probability 0.5:
$$\overline{\cos}^{(m)} = \frac{1}{N} \sum_{i=1}^N \epsilon_i \cos_i$$
1. Positive Alignment ($H_1: \mu > 0$):
   $$p_{\text{pos}} = \frac{1}{M + 1} \left( 1 + \sum_{m=1}^M \mathbf{1}[\overline{\cos}^{(m)} \ge \overline{\cos}] \right)$$
2. Negative Alignment ($H_1: \mu < 0$):
   $$p_{\text{neg}} = \frac{1}{M + 1} \left( 1 + \sum_{m=1}^M \mathbf{1}[\overline{\cos}^{(m)} \le \overline{\cos}] \right)$$
3. Two-Sided ($H_1: |\mu| > 0$):
   $$p_{\text{two}} = \frac{1}{M + 1} \left( 1 + \sum_{m=1}^M \mathbf{1}[|\overline{\cos}^{(m)}| \ge |\overline{\cos}|] \right)$$

### 2.3 Treves-Rolls Sparseness Formulations
For activation matrix $A \in \mathbb{R}^{N \times D}$ where $A_{ij} \ge 0$:
1. **Lifetime Sparseness** of neuron $d$:
   $$S_L(d) = \frac{1 - \frac{(\frac{1}{N} \sum_{i=1}^N A_{id})^2}{\frac{1}{N} \sum_{i=1}^N A_{id}^2}}{1 - \frac{1}{N}}$$
2. **Population Sparseness** of observation $n$:
   $$S_P(n) = \frac{1 - \frac{(\frac{1}{D} \sum_{j=1}^D A_{nj})^2}{\frac{1}{D} \sum_{j=1}^D A_{nj}^2}}{1 - \frac{1}{D}}$$

### 2.4 TOST Equivalence Procedure
For paired differences $d_i = E_{i, \text{action}} - E_{i, \text{oracle}}$ with practical margin $\delta = 0.05$ m:
$$t_1 = \frac{\bar{d} + \delta}{\text{SE}}, \quad p_1 = 1 - F_{t}(t_1; n-1)$$
$$t_2 = \frac{\delta - \bar{d}}{\text{SE}}, \quad p_2 = 1 - F_{t}(t_2; n-1)$$
$$p_{\text{TOST}} = \max(p_1, p_2)$$
Equivalence is declared if $p_{\text{TOST}} < 0.05$.
