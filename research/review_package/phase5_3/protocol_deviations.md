# Phase 5.3 Protocol Deviations & Compliance Log

**Experiment:** Phase 5.3 Valid Norm-Matched Causal Mediation & Necessity Revalidation  
**Execution Date:** 2026-11-08  
**Repository Branch:** `fix/interpretability`  
**Execution Commit SHA:** `7c30728179b2bd04778219f1f2b9746ba9dcedd7`

---

## 1. Intentional Methodological Upgrades (Pre-Registered Deviations from Phase 5.2)

1. **Strict Tolerance Filtering**:
   - *Phase 5.2*: Retained the 100 closest random subsets from the pool regardless of whether they satisfied the 5% tolerance threshold.
   - *Phase 5.3*: Implemented strict filtering (`rel_error <= 0.05`). Subsets with `rel_error > 0.05` are NEVER selected. If $< 100$ valid subsets exist after adaptive sampling, the setting is marked `INSUFFICIENT_COMMON_SUPPORT`, retained controls are set to `None`, and no confirmatory p-value is computed.

2. **Adaptive Control Sampling**:
   - *Phase 5.2*: Sampled a fixed pool of 5,000 random subsets per setting.
   - *Phase 5.3*: Sampled an initial pool of 5,000 subsets, and if $< 100$ valid matches were found, adaptively drew additional batches of 20,000 up to a hard maximum of 200,000 draws per setting.

3. **Overlap Recording & Hypergeometric Stratification**:
   - *Phase 5.2*: Overlap between candidate and random subsets was unrecorded.
   - *Phase 5.3*: Overlap $O = |U_{\text{cand}} \cap U_{\text{rand}}|$ and overlap fraction $O/k$ recorded for all draws. Theoretical hypergeometric distribution probabilities $w_o = \frac{\binom{k}{o}\binom{64-k}{k-o}}{\binom{64}{k}}$ calculated and preserved.

4. **Matched-Control Necessity Comparison**:
   - *Phase 5.2*: Condition-averaged ablation ($h_i \to \frac{h_i^A + h_i^B}{2}$) was applied only to the candidate subset.
   - *Phase 5.3*: Condition-averaged ablation applied to BOTH the candidate subset AND the eligible norm-matched random controls, providing an empirical null distribution for causal necessity.

5. **Disjoint Controls Reference**:
   - *Phase 5.2*: Disjoint controls were not evaluated.
   - *Phase 5.3*: Evaluated disjoint controls ($O=0$) sampled strictly from $U_{\text{cand}}^c$. At $k=32$, because $|U_{\text{cand}}^c| = 32$, evaluated as a single deterministic reference ($\binom{32}{32} = 1$).

---

## 2. Runtime Engineering Discoveries & Adjustments

1. **MuJoCo Renderer Buffer In-Place Mutation**:
   - *Observation*: During observation regeneration, individually calling `env.apply_task_spec` and `env.reset` produced observation hash mismatches for some scenes in `task_conditioned_circuits_v5_2_scene_specs.jsonl` (31/120 scenes mismatched).
   - *Root Cause Analysis*: MuJoCo's `Renderer.render()` returns an in-place view into its internal framebuffer. In Phase 5.2, `generate_dual_visible_scene_pair` retained the environment across trials, and segmentation rendering for visibility checks touched the framebuffer before hashing.
   - *Resolution*: Ported `generate_dual_visible_scene_pair` and executed sequentially with `rng = np.random.default_rng(2026)`. This achieved **120/120 (100.0%) exact bit-for-bit SHA-256 hash matches** against Phase 5.2 specs.

2. **Gzip Compression for Interventions Artifact**:
   - *Observation*: The raw intervention records across all individual draws (447,800 records) would produce an uncompressed JSONL of ~270MB, exceeding GitHub's 100MB file limit.
   - *Resolution*: Directly streamed into `task_conditioned_circuits_v5_3_interventions.jsonl.gz` using Python's `gzip` module, producing a 69MB compressed artifact that preserves 100% of individual draw records while remaining within GitHub push constraints.

---

## 3. Protocol Compliance Verification

- [x] All 60 scene pairs from Phase 5.2 reused without modification.
- [x] 120/120 observation hashes verified bit-for-bit against Phase 5.2 records.
- [x] Frozen candidate rankings from Phase 5.2 discovery split used without re-ranking.
- [x] Checkpoint `world_models_fpga/checkpoints_v4/hjepa_best.pt` remained strictly frozen.
- [x] Independent scene pair ($N = 60$) maintained as the paired statistical unit.
- [x] Holm-Bonferroni correction applied across the 10 planned primary tests.
- [x] No invalid controls substituted outside 5% tolerance.
- [x] All historical Phase 5.1 and Phase 5.2 files preserved untouched.
