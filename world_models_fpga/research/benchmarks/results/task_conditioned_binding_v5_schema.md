# Schema: Task-Conditioned Binding Revalidation v5 (`task_conditioned_binding_v5`)

## 1. Primary Artifacts
- **Summary JSON**: [`task_conditioned_binding_v5_summary.json`](./task_conditioned_binding_v5_summary.json)
- **Pairs JSONL**: [`task_conditioned_binding_v5_pairs.jsonl`](./task_conditioned_binding_v5_pairs.jsonl)
- **Fold Predictions JSONL**: [`task_conditioned_binding_v5_fold_predictions.jsonl`](./task_conditioned_binding_v5_fold_predictions.jsonl)

## 2. Statistical Invariants & Protocol Corrections
1. **Pair-Clustered Permutation Null**:
   - Resamples: 100,000
   - One swap decision per pair applied identically across Scene A and Scene B.
   - Evaluates whether dual discrimination exceeds paired chance under preserved factorial structure.
2. **Sign-Flip Alignment Tests**:
   - Explicitly evaluates positive (p_pos), negative (p_neg), and two-sided (p_two) alignment of interaction vector I with physical swap vector Delta P.
3. **Stratified Grouped Cross-Validation**:
   - 5-fold CV grouped by pair_id and stratified by target pair configuration (key_a, key_b).
   - Records full per-fold predictions in task_conditioned_binding_v5_fold_predictions.jsonl.
4. **Instance Binding Limitation**:
   - Documents that encode_task_id() cannot represent instance distinctions between identical appearances.
