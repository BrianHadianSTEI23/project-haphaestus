# Phase 5.3 Data Schema Documentation

## 1. Primary Output Files

- `task_conditioned_circuits_v5_3_summary.json`: Top-level experimental configuration, provenance hashes, aggregate statistics across 60 independent pairs, Holm-Bonferroni adjusted p-values, match coverage table, representation profiling across all 9 stages, and machine-readable mechanistic verdict.
- `task_conditioned_circuits_v5_3_pairs.jsonl`: One record per independent scene pair (N = 60), containing paired sufficiency metrics, necessity drops, contrast retentions, heading/bearing shifts, and common-support eligibility status.
- `task_conditioned_circuits_v5_3_activations.jsonl`: One record per independent pair and scene (120 records total), containing raw Layer 1 and Layer 2 post-ReLU 64D activations under Task A and Task B, baseline outputs, and ground-truth goal vectors.
- `task_conditioned_circuits_v5_3_interventions.jsonl.gz`: Gzipped records of all individual causal interventions across candidate, strict norm-matched random, unconstrained dimension-matched random, overlap-stratified, and disjoint controls.
- `task_conditioned_circuits_v5_3_control_audit.jsonl`: Granular audit records for every setting (pair × scene × layer × k × stratum), detailing random seed, attempted draws, valid norm matches within 5%, retained count, min/median/max relative error, and common-support eligibility status.

## 2. Statistical Unit & Conventions

- **Statistical Unit**: Independent scene pair ($N = 60$). Each pair features two visual scenes (Scene A and Scene B) with target and distractor positions swapped.
- **Pair Aggregation**: Scene A and Scene B are averaged. Transplantation directions ($A \to B$ and $B \to A$) are averaged relative to their respective contrast directions $\Delta \mathbf{g}^*$.
- **Strict Norm-Matching Filter**:
  $$\text{rel\_error} = \frac{|\|\Delta \mathbf{h}[U_{\text{ctrl}}]\| - \|\Delta \mathbf{h}[U_{\text{cand}}]\||}{\max(10^{-8}, \|\Delta \mathbf{h}[U_{\text{cand}}]\|)} \le 0.05$$
  If $< 100$ valid subsets exist after adaptive sampling up to 200,000 draws, the setting is marked `INSUFFICIENT_COMMON_SUPPORT`, retained controls are set to `None`, and no confirmatory p-value is computed.
- **Overlap Recording**:
  $$O = |U_{\text{cand}} \cap U_{\text{rand}}|$$
  $$P(O = o) = \frac{\binom{k}{o} \binom{D - k}{k - o}}{\binom{D}{k}}, \quad D = 64$$
- **Projected Transfer Efficiency**:
  $$\eta = \frac{(\mathbf{g}_{\text{trans}} - \mathbf{g}_{\text{recip}})_{:2} \cdot \Delta \mathbf{g}^*}{\|\Delta \mathbf{g}^*\|^2}$$
- **Condition-Averaged Necessity Ablation**:
  $$h_i^A, h_i^B \longrightarrow \frac{h_i^A + h_i^B}{2} \quad \text{for } i \in U$$
  $$\text{retention\_ratio} = \frac{\|\Delta \tilde{\mathbf{g}}_{\text{xy}}\|}{\|\Delta \mathbf{g}^*_{\text{xy}}\|}, \quad \text{drop\_ratio} = 1 - \text{retention\_ratio}$$
