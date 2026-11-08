# Schema: Ego-Motion Memory Revalidation v5 (`ego_memory_mechanism_v5`)

## 1. Primary Artifacts
- **Summary JSON**: [`ego_memory_mechanism_v5_summary.json`](./ego_memory_mechanism_v5_summary.json)
- **Pairs JSONL**: [`ego_memory_mechanism_v5_pairs.jsonl`](./ego_memory_mechanism_v5_pairs.jsonl)
- **Steps JSONL**: [`ego_memory_mechanism_v5_steps.jsonl`](./ego_memory_mechanism_v5_steps.jsonl)

## 2. Invariants & Corrections
1. **Simulator Initialization**:
   - `apply_task_spec` called prior to `reset(options={"initial_pos": ...})`
   - Verified robot pose and obstacle geometry positions with runtime assertions.
2. **Action / Ground-Truth Time Alignment**:
   - Transition $t \to t+1$ uses commanded action $t$ and physical displacement from $t$ to $t+1$.
3. **Blackout Replay Information Path**:
   - Step 5 transition uses clean latent 4 $\to$ black frame.
   - Subsequent steps use black frame $\to$ black frame.
   - Zero future or intermediate visual information during blackout.
4. **Statistical Reporting**:
   - Exact paired differences reported between Fixed Blending and Hold policy.
   - TOST equivalence test between Action Hold and Oracle Hold.
   - All 50 trajectories stored with complete per-step records in `steps.jsonl`.
