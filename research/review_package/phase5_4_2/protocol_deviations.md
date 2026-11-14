# Phase 5.4.2 Protocol Deviations

1. **Historical Raw Artifact Preservation:** No Phase 5.1, 5.2, 5.3, 5.4, or 5.4.1 raw JSON/JSONL/NPZ file was modified, overwritten, or regenerated.
2. **Exploratory Labeling:** Because Phase 5.4.2 re-audits existing Phase 5.3 intervention outputs using corrected single-target estimands, all complete-support statistical tests (Layer 1 $k=2, 4$) are designated `EXPLORATORY_COMPLETE_SUPPORT` and incomplete-support settings are designated `EXPLORATORY_INCOMPLETE_SUPPORT`.
3. **No Equivalence Inference:** All non-significant $p$-values are interpreted strictly as lack of evidence for a difference, with full paired bootstrap 95% confidence intervals reported.
4. **No Checkpoint Modification or Re-simulation:** Strictly zero model weights were altered and zero new simulator runs were performed in Phase 5.4.2.
