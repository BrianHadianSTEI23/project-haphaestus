# Phase 5.4.1 Schema Documentation: Forensic Reconciliation & Statistical Reporting

## 1. Scope & Purpose

Phase 5.4.1 establishes full cryptographic and mathematical reconciliation of the Phase 5.4 ground-truth spatial reanalysis:
1. Reconciles exact record counts (447,800 rows total: 2,400 candidate, 240,000 dimension-matched, 205,400 norm-matched).
2. Clarifies ground-truth coordinates as task-defined standoff goal waypoints (0.60 m standoff toward robot), not object centers.
3. Corrects physical scene-swap reversal cosine interpretation (-0.7119 signifies non-reversal / orientation retention).
4. Establishes pair-level pooled candidate statistics where the independent scene pair (N=60) is the unit of inference.
5. Populates cohort-matched candidate vs norm-matched tables strictly on identical eligible pair IDs.
6. Designates all post-hoc ground-truth reanalysis tests as `EXPLORATORY_COMPLETE_SUPPORT` or `EXPLORATORY_INCOMPLETE_SUPPORT`.

## 2. Primary Output Files

- `task_conditioned_circuits_v5_4_1_audit_summary.json`: Complete audit records, hashes, waypoint geometry, pooled candidate statistics, cohort-matched comparison metrics, Holm-adjusted p-values, and claims ledger.
- `task_conditioned_circuits_v5_4_1_condition_counts.csv`: Exact condition-level row accounting (expected vs observed).
- `task_conditioned_circuits_v5_4_1_pair_metrics.jsonl`: Pair-level aggregated metrics across all 60 independent pairs and 10 settings.
- `task_conditioned_circuits_v5_4_1_schema.md`: This documentation.

## 3. Mathematical Definitions

### A. Task-Defined Standoff Waypoint
$$\mathbf{w} = \mathbf{p}_{\text{obj}} - 0.60 \cdot \frac{\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{robot}}}{\|\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{robot}}\|}$$

### B. Scene-Swap Reversal Cosine
$$\cos_{\text{swap}} = \frac{\Delta\mathbf{g}_{\text{model}}^A \cdot (-\Delta\mathbf{g}_{\text{model}}^B)}{\|\Delta\mathbf{g}_{\text{model}}^A\| \|\Delta\mathbf{g}_{\text{model}}^B\|}$$
- $\cos_{\text{swap}} = +1.0$: Ideal reversal tracking ground-truth inversion ($\Delta\mathbf{g}_{\text{model}}^B = -\Delta\mathbf{g}_{\text{model}}^A$).
- $\cos_{\text{swap}} = -1.0$: Invariant contrast retaining identical orientation ($\Delta\mathbf{g}_{\text{model}}^B = \Delta\mathbf{g}_{\text{model}}^A$).

### C. Cohort-Matched Paired Difference
$$\Delta\eta_{\text{gt}}(L, k) = \frac{1}{|\mathcal{P}_{\text{elig}}|} \sum_{p \in \mathcal{P}_{\text{elig}}} \eta_{\text{cand}}(p, L, k) - \frac{1}{|\mathcal{P}_{\text{elig}}|} \sum_{p \in \mathcal{P}_{\text{elig}}} \eta_{\text{norm}}(p, L, k)$$
Ensures candidate and control means are evaluated strictly on the exact same pair IDs $\mathcal{P}_{\text{elig}}$.
