# Phase 5.2 Protocol Deviations & Formal Methodological Definitions

**Repository:** `BrianHadianSTEI23/project-haphaestus`  
**Branch:** `fix/interpretability`  
**Commit:** `bc568052e8be10fba4f4a466e913e3dbd8bb4fd6`  
**Role:** Experimentalist and implementation engineer  
**Auditor:** ChatGPT (Senior Independent Scientific Researcher)  

---

## 1. Protocol Comparison: Phase 5.1 vs Phase 5.2

| Component | Phase 5.1 Protocol (`v5`) | Phase 5.2 Upgraded Protocol (`v5.2`) | Methodological Justification |
| :--- | :--- | :--- | :--- |
| **Candidate Ranking** | Single-context sensitivity on discovery split: $s_i = \|h_i(z_A, e_B) - h_i(z_A, e_A)\|$ | Both-context sensitivity on discovery split: $s_i = 0.5 \times (\|\Delta h_i^A\| + \|\Delta h_i^B\|)$ | Prevents context-specific overfitting to Scene A geometry; ensures candidate units are robust across visual contexts. |
| **Evaluation Set** | 25 held-out scene pairs (50 scenes), nominal lighting only | 60 independent scene pairs (120 scenes): 30 nominal ($\text{scale}=1.0$) + 30 illumination shifted ($\text{scale}=0.60$) | Rejection-sampled with verified 5-step simulator protocol to test OOD visual robustness and statistical power. |
| **Random Circuit Controls** | Unconstrained dimension-matched random subsets (100 draws/pair) | Dual controls: (1) Activation-norm-matched ($\le 5\%$ tolerance, closest 100 of $\ge 5,000$ draws) + (2) Dimension-matched | Isolates true functional wiring selectivity from activation-magnitude perturbation confounders. |
| **Transplantation Direction** | Unidirectional ($A \to B$) | Bidirectional ($A \to B$ and $B \to A$) | Tests directional symmetry of causal sufficiency and eliminates recipient-target directional bias. |
| **Representation Statistics** | Treves-Rolls applied across all layers regardless of activation sign | Non-negative post-ReLU: true Treves-Rolls; Signed stages: positive-part sparseness, excess kurtosis, negative fraction | Fixes mathematical violation of applying non-negative sparseness equations to zero-mean signed activations. |
| **Ego-Motion Shuffling** | Cross-trajectory random action sampling (pooled across all 50 trajectories) | Within-trajectory temporal permutation (50 perms/trajectory during 10-step blackout) | Controls for trajectory-specific action distribution marginals; strictly isolates temporal sequence sensitivity. |
| **Displacement $R^2$ Reporting** | Single scalar magnitude regression $R^2 = 0.2549$ | Decoupled: Scalar magnitude $R^2$, 2D vector coordinate $R^2$, component $dx/dy/d\theta$ $R^2$, MAE, RMSE across all regimes | Eliminates ambiguity between scalar speed tracking and physical coordinate vector prediction. |

---

## 2. Formal Mathematical Definitions

### 2.1 Both-Context Candidate Sensitivity
For discovery scene pairs $p \in \{0, \dots, 24\}$, intermediate activations at hidden layer $l$ under visual latents $z_A, z_B$ and task embeddings $e_A, e_B$:
$$s_i = \frac{1}{25} \sum_{p=0}^{24} \frac{1}{2} \left( |h_i(z_A^{(p)}, e_B) - h_i(z_A^{(p)}, e_A)| + |h_i(z_B^{(p)}, e_B) - h_i(z_B^{(p)}, e_A)| \right)$$
Candidate units of cardinality $k$ are defined by the top-$k$ indices:
$$U_{\text{cand}}(k) = \operatorname{arg\,top-}k (s_i)$$

### 2.2 Activation Perturbation Norm Matching
For a given scene, let $\Delta h_i = h_i(z, e_B) - h_i(z, e_A)$. The perturbation norm of any subset $U \subset \{1, \dots, D\}$ of size $k$ is:
$$\|\Delta h_U\|_2 = \sqrt{\sum_{i \in U} (\Delta h_i)^2}$$
Random subset draws $U_{\text{rand}}^{(j)}$ ($j = 1, \dots, M$, with $M \ge 5000$) are filtered by relative error:
$$\epsilon_{\text{rel}}(U) = \frac{|\|\Delta h_U\|_2 - \|\Delta h_{U_{\text{cand}}}\|_2|}{\|\Delta h_{U_{\text{cand}}}\|_2}$$
The 100 subsets with the smallest $\epsilon_{\text{rel}} \le 0.05$ are selected as norm-matched controls.

### 2.3 Projected Transfer Efficiency ($\eta$)
Let $\mathbf{g}_{\text{recip}}, \mathbf{g}_{\text{donor}} \in \mathbb{R}^2$ be the unperturbed 2D body-relative waypoint goals. The intended task contrast vector is:
$$\Delta \mathbf{g}^* = \mathbf{g}_{\text{donor}} - \mathbf{g}_{\text{recip}}$$
Under hidden layer transplantation $h_{\text{trans}} = h_{\text{recip}}$, with $h_{\text{trans}}[U] = h_{\text{donor}}[U]$, the intervened output is $\mathbf{g}_{\text{trans}}$. The transferred displacement vector is:
$$\Delta \mathbf{g}_{\text{trans}} = \mathbf{g}_{\text{trans}} - \mathbf{g}_{\text{recip}}$$
The projected transfer efficiency $\eta$ is the normalized projection along the intended contrast:
$$\eta = \frac{\Delta \mathbf{g}_{\text{trans}} \cdot \Delta \mathbf{g}^*}{\|\Delta \mathbf{g}^*\|_2^2}$$
- $\eta = 1.0$: Complete task contrast transfer.
- $\eta = 0.0$: Zero causal effect.
- $\eta < 0.0$: Counterproductive/inverted transfer.

### 2.4 Orthogonal Residual ($\mathbf{r}_\perp$)
The unintended distortion orthogonal to the intended contrast vector:
$$\mathbf{r}_\perp = \Delta \mathbf{g}_{\text{trans}} - \eta \Delta \mathbf{g}^*$$
$$\|\mathbf{r}_\perp\|_2 = \sqrt{\|\Delta \mathbf{g}_{\text{trans}}\|_2^2 - \eta^2 \|\Delta \mathbf{g}^*\|_2^2}$$

### 2.5 Necessity: Condition-Averaged Contrast Retention Ratio
Under condition-averaged ablation on candidate subset $U_{\text{cand}}$:
$$h_i^{\text{abl}} = \frac{h_i(z, e_A) + h_i(z, e_B)}{2}, \quad \forall i \in U_{\text{cand}}$$
The retention ratio $\rho$ and drop ratio $\delta$ are:
$$\rho = \frac{\|\mathbf{g}_{\text{abl}}(e_B) - \mathbf{g}_{\text{abl}}(e_A)\|_2}{\|\mathbf{g}(e_B) - \mathbf{g}(e_A)\|_2}, \quad \delta = 1 - \rho$$
If candidate units are causally necessary for task discrimination, $\rho \to 0$ ($\delta \to 1$).
