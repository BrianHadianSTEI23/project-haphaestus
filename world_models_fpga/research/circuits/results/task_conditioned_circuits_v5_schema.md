# Schema: Task-Conditioned Circuits Revalidation v5 (`task_conditioned_circuits_v5`)

## 1. Primary Artifacts
- **Summary JSON**: [`task_conditioned_circuits_v5_summary.json`](./task_conditioned_circuits_v5_summary.json)
- **Pairs JSONL**: [`task_conditioned_circuits_v5_pairs.jsonl`](./task_conditioned_circuits_v5_pairs.jsonl)
- **Interventions JSONL**: [`task_conditioned_circuits_v5_interventions.jsonl`](./task_conditioned_circuits_v5_interventions.jsonl)

## 2. Invariants & Corrections
1. **Unit Separation**:
   - Translation reported strictly in meters (d_xy = sqrt(dx^2 + dy^2)).
   - Orientation reported in radians / degrees from trigonometric components.
   - Zero mixing into 4D Euclidean distance.
2. **Representation Metrics**:
   - Participation ratio and eigenvalue spectrum for all 9 stages.
   - Lifetime sparseness computed per unit across observations, then aggregated.
   - Population sparseness reported separately per observation across units.
3. **Causal Controls**:
   - 100 independent random draws per held-out pair for each dimension sweep k in {2, 4, 8, 16, 32, 64}.
   - Holm-Bonferroni correction applied across the 6 hypothesis tests.
   - Separate rescue claims eliminated (classified as donor sufficiency).
