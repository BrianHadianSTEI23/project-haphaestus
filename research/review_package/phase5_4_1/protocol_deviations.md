# Phase 5.4.1 Protocol Deviations & Methodological Compliance

**Role:** Implementation Engineer & Experimentalist  
**Target:** Independent review by ChatGPT (Senior Independent Scientific Researcher)  
**Baseline Commit SHA:** `0cafb12240dcd749a7c3e0c8c9612f4ddeca5ea2`  
**Execution / Source Commit SHA:** `d668ce5c0f552eb90a8e26307409b813efc5fc08`  

---

## 1. Methodological Compliance & Integrity Invariants

Phase 5.4.1 was conducted under strict experimental constraints. The following execution invariants were maintained:

1. **Role Boundary Adherence**:
   - The implementation engineer conducted data collection, code execution, unit testing, vector derivations, and statistical reporting.
   - All high-level computational interpretations and decisions regarding intelligence principles are left to ChatGPT (Senior Independent Scientific Researcher).

2. **Preservation of Historical Artifacts**:
   - Zero historical raw results from Phase 5.1, Phase 5.2, Phase 5.3, or Phase 5.4 were deleted, regenerated, or overwritten.
   - All Phase 5.4.1 outputs are stored in dedicated new artifact files (`task_conditioned_circuits_v5_4_1_*`).
   - The single forensic fix to a historical file was the correction of stale machine-readable claim text in `task_conditioned_circuits_v5_4_ground_truth_summary.json` (replacing stale values `-0.0620` and `93.5 deg` with verified values `-0.7119` and `143.50 deg`).

3. **No Retraining or Simulator Regeneration**:
   - No neural network weights were retrained.
   - No simulator scenes were regenerated.
   - All analyses operate directly on the immutable Phase 5.3 dataset (60 independent pairs, 120 scenes, 447,800 intervention records).

4. **Clean Commit Sequence**:
   - Source code and unit tests were staged and committed in a dedicated clean commit prior to execution: `d668ce5c0f552eb90a8e26307409b813efc5fc08`.
   - The reporting audit script was executed directly from this clean source commit.

---

## 2. Protocol Deviations & Forensic Resolutions

| Audit Finding | Prior State | Rectified State in Phase 5.4.1 | Rationale |
| :--- | :--- | :--- | :--- |
| **Scene-Swap Reversal Directionality** | Described as "reversing with cosine -0.7119" or "confirming directional sensitivity". Stale values `-0.0620` and `93.5 deg` present. | Formally proved that $\cos(\Delta\mathbf{g}_{\text{model}}^A, -\Delta\mathbf{g}_{\text{model}}^B) = -0.7119$ means $\Delta\mathbf{g}_{\text{model}}^A \cdot \Delta\mathbf{g}_{\text{model}}^B = +0.7119 > 0$. Model contrast retains substantially the *same* orientation across scenes, **failing to reverse**. Stale values eliminated. | Mathematical vector algebra: reversal requires $\Delta\mathbf{g}_{\text{model}}^B = -\Delta\mathbf{g}_{\text{model}}^A$ ($\cos = +1.0$). A negative value indicates orientation retention, not reversal. |
| **Ground-Truth Target Definition** | Variously referred to as "ground truth target" or "object center". | Standardized to "ground-truth task waypoint" located at standoff distance $0.60$ m along line of sight: $\mathbf{w} = \mathbf{p}_{\text{obj}} - 0.60 \frac{\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{robot}}}{\|\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{robot}}\|}$. | Accurately reflects the agent's navigation training target and prevents conflation with physical object centers. |
| **Intervention Record Count** | Stated as "4,800 candidate interventions". | Formally derived exact count: 2,400 candidate rows ($60 \times 2 \times 2 \times 5 \times 2$), 240,000 dimension-matched, and 205,400 norm-matched (total 447,800). Verified 0 missing, 0 extra, 0 duplicates. | Rigorous accounting: each setting had 1 candidate draw, 100 dimension-matched draws, and 100 norm-matched draws for the 2,054 settings with valid common support. |
| **Statistical Status of Complete-Support Tests** | Previously labeled `CONFIRMATORY_EVALUATED`. | Formally classified as `EXPLORATORY_COMPLETE_SUPPORT`. Holm-Bonferroni correction applied over family of 4 comparisons. | Because ground-truth spatial accuracy was evaluated post-hoc on existing intervention data, standard practice requires classifying post-hoc reanalyses as exploratory. |
| **Paired Difference Invariant** | Reporting inconsistencies in incomplete-support settings. | Enforced strict identical pair ID matching: `paired_diff = cand_eligible - norm_eligible` across all 10 settings. | Eliminates sampling bias when comparing candidate units against norm-matched controls. |
| **Diffuse Population Coding Principle** | Promoted as "Principle 4: Diffuse Population Coding (Modularity Falsified)". | Removed from established principles; updated to `UNRESOLVED: tested candidate latent subspace was not selective relative to the tested random controls; the evidence does not establish a diffuse population code.` | Non-selectivity relative to random controls disconfirms modular circuit localization, but does not affirmatively establish a diffuse population code. |
