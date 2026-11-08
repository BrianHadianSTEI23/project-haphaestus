# Phase 5.4 Data Schema Documentation: Ground-Truth Spatial Grounding

## 1. Overview and Primary Files

Phase 5.4 evaluates H-JEPA's full-model predictions and hidden-layer causal interventions against the actual physical target coordinates in egocentric body frame, separating model-contrast mediation from true spatial mediation.

- `task_conditioned_circuits_v5_4_ground_truth_summary.json`: Top-level experimental summary, input file provenance hashes, reconciliation audit metrics, context and object-pair stratifications, pair-clustered bootstrap confidence intervals, and machine-readable claims ledger.
- `task_conditioned_circuits_v5_4_scene_metrics.jsonl`: One record per independent pair and scene (120 records total), containing body-frame ground-truth targets, predicted outputs, spatial errors, contrast alignment, bias vectors, heading errors, and physical scene-swap consistency.
- `task_conditioned_circuits_v5_4_intervention_metrics.jsonl.gz`: Streaming records for all 447,800 causal interventions, augmented with donor ground-truth targets, recipient baseline errors, donor-target transplant errors, error improvements, and signed ground-truth contrast transfer metrics.

## 2. Mathematical Definitions & Coordinate Frames

- **Coordinate System**: Body-relative 2D Cartesian frame ($m$). $+X$ points directly forward along the robot's heading; $+Y$ points lateral left.
- **Goal Displacements**: $\mathbf{g}_{\text{model}} = [x, y]$, $\mathbf{g}_{\text{gt}} = [x_{\text{gt}}, y_{\text{gt}}]$.
- **Spatial Accuracy**:
  $$\text{error}_A = \|\mathbf{g}_{\text{model}}^A - \mathbf{g}_{\text{gt}}^A\|_2, \quad \text{error}_B = \|\mathbf{g}_{\text{model}}^B - \mathbf{g}_{\text{gt}}^B\|_2$$
- **Ground-Truth Task Contrast**:
  $$\Delta \mathbf{g}_{\text{gt}} = \mathbf{g}_{\text{gt}}^B - \mathbf{g}_{\text{gt}}^A, \quad \Delta \mathbf{g}_{\text{model}} = \mathbf{g}_{\text{model}}^B - \mathbf{g}_{\text{model}}^A$$
  $$\eta_{\text{gt}} = \frac{\Delta \mathbf{g}_{\text{model}} \cdot \Delta \mathbf{g}_{\text{gt}}}{\|\Delta \mathbf{g}_{\text{gt}}\|^2}$$
- **Intervention Error Improvement**:
  $$\text{baseline\_error} = \|\mathbf{g}_{\text{base}} - \mathbf{g}_{\text{recip\_gt}}\|_2$$
  $$\text{transplant\_error} = \|\mathbf{g}_{\text{trans}} - \mathbf{g}_{\text{donor\_gt}}\|_2$$
  $$\text{error\_improvement} = \text{baseline\_error} - \text{transplant\_error}$$
- **Intervention Ground-Truth Contrast Transfer**:
  $$\mathbf{d}_{\text{true}} = \mathbf{g}_{\text{donor\_gt}} - \mathbf{g}_{\text{recip\_gt}}$$
  $$\mathbf{d}_{\text{intervention}} = \mathbf{g}_{\text{trans}} - \mathbf{g}_{\text{base}}$$
  $$\eta_{\text{gt}} = \frac{\mathbf{d}_{\text{intervention}} \cdot \mathbf{d}_{\text{true}}}{\|\mathbf{d}_{\text{true}}\|^2}$$
- **Physical Scene Swap Reversal**:
  Between Scene A and Scene B, physical objects swap positions, ensuring $\Delta \mathbf{g}_{\text{gt}}^B = -\Delta \mathbf{g}_{\text{gt}}^A$. The model reversal cosine is computed as:
  $$\cos_{\text{swap}} = \frac{\Delta \mathbf{g}_{\text{model}}^A \cdot (-\Delta \mathbf{g}_{\text{model}}^B)}{\|\Delta \mathbf{g}_{\text{model}}^A\| \|\Delta \mathbf{g}_{\text{model}}^B\|}$$
