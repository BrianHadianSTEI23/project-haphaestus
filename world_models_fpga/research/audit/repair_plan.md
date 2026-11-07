# Sequential repair plan

1. **Environment validity — in progress.** Real E5 geometry, E6 type mutation, E7 camera quaternion mutation, rendered-state checks, collision-distance fallback, and a deterministic privileged geometric expert are implemented. Generate versioned data and reject legacy data for retraining.
2. **Training validity — next.** Replace absolute paths and transition-level splitting; add goal provenance, macro policy supervision, validation checkpoint selection, configs, histories, and checksums. Rename or verify the SIGReg approximation.
3. **Behavioral validity.** Replace pose-derived learned-agent actions with explicit A/B/C information boundaries and candidate scoring from learned predictions; retain an explicitly labeled oracle only as an upper bound.
4. **Mechanistic validity.** Verify hooks on the actual action path; rebuild patching with episode-pair indexing, behavioral outcomes, reverse/interchange interventions, and controls.
5. **Principles and synthesis.** Retire literal scores; run independent multi-seed experiments before making E0--E3 classifications. No architecture is called principle-derived until that gate passes.
6. **Hardware.** Keep RTL prototype, label it feature-conditioned, and only report board/synthesis values accompanied by raw files. No KR260 board measurement has been performed in this repair pass.

Historical reports remain in place as historical records. New reports will distinguish `MEASURED`, `SIMULATED`, `ESTIMATED`, `SYNTHETIC`, and `NOT PERFORMED`.
