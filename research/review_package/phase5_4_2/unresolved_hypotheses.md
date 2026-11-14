# Phase 5.4.2 Unresolved Hypotheses

1. **Selectivity vs Perturbation Magnitude in 8/10 Incomplete-Support Settings:** For Layer 1 ($k \in \{8, 16, 32\}$) and Layer 2 ($k \in \{2, 4, 8, 16, 32\}$), standard coordinate random draws cannot match candidate perturbation norms within 5% tolerance across all 60 pairs (coverage 51.7%–93.3%), leaving candidate selectivity unidentifiable in those regimes.
2. **Closed-Loop Resolution of Static Dissociation:** Phase 5.4.2 confirms static `t=0` internal contrast interventions do not selectively transfer true donor waypoint alignment (`eta_gt <= 0` in Layer 1), which leaves open how the identical checkpoint navigates in closed loop — investigated directly in Phase 5.5 (`task_conditioned_closed_loop_grounding_v5_5`).
3. **Checkpoint Scope:** Findings strictly apply to the tested H-JEPA checkpoint (`ac69688eae1f4128719dfce912da47c8ae55cbf2412fc8c10feba53d93f1f442`) and dual-visible paired benchmark.
