"""
Unit tests for Phase 5 Experiment B: Targeted Internal Subspace Circuit Interventions (v4).
Validates 9-stage extraction, sparsity, effective dimensionality, and causal intervention logic.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "world_models_fpga"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
import torch

from models.h_jepa import HJEPAModel
from research.benchmarks.model_wrappers import HJEPAAgent
from research.datasets.dataset_config import CKPT_DIR
from research.circuits.run_task_conditioned_circuits_v4 import (
    extract_9_stages,
    compute_treves_rolls_sparsity,
    compute_effective_dimensionality,
    run_grounding_head_forward,
)


@pytest.fixture(scope="module")
def model():
    ckpt = CKPT_DIR / "hjepa_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = HJEPAAgent(str(ckpt), device=device, track="observation")
    return agent.model


def test_9_stages_extraction_shapes(model):
    """Verifies that all 9 stages are extracted with exact mathematical tensor dimensions."""
    device = next(model.parameters()).device
    z = torch.randn(1, 32, device=device)
    tid = torch.tensor([0], dtype=torch.long, device=device)

    stages = extract_9_stages(model, z, tid)
    assert stages["visual_latent_z"].shape == (1, 32)
    assert stages["task_embedding_e"].shape == (1, 16)
    assert stages["input_concat_48d"].shape == (1, 48)
    assert stages["layer1_pre_relu_64d"].shape == (1, 64)
    assert stages["layer1_post_relu_64d"].shape == (1, 64)
    assert stages["layer2_pre_relu_64d"].shape == (1, 64)
    assert stages["layer2_post_relu_64d"].shape == (1, 64)
    assert stages["raw_output_4d"].shape == (1, 4)
    assert stages["postprocessed_goal_4d"].shape == (1, 4)


def test_sparsity_and_effective_dimensionality():
    """Verifies that Treves-Rolls sparsity and participation ratio satisfy mathematical bounds."""
    # 1. Perfectly sparse vector (one unit active): Sparsity = 1/D
    D = 64
    sparse_acts = np.zeros((10, D))
    sparse_acts[:, 0] = 1.0
    sp = compute_treves_rolls_sparsity(sparse_acts)
    assert np.isclose(sp, 1.0 / D, atol=1e-3)

    # 2. Dense uniform vector: Sparsity = 1.0
    dense_acts = np.ones((10, D))
    sp_dense = compute_treves_rolls_sparsity(dense_acts)
    assert np.isclose(sp_dense, 1.0, atol=1e-3)

    # 3. Effective dimensionality on isotropic gaussian: ~ min(N, D)
    rng = np.random.default_rng(42)
    gauss_acts = rng.normal(size=(200, 20))
    ed = compute_effective_dimensionality(gauss_acts)
    assert 15.0 <= ed <= 21.0


def test_circuits_v4_artifacts_integrity():
    """Verifies that circuit v4 artifacts exist and parse cleanly."""
    results_dir = REPO_ROOT / "world_models_fpga/research/circuits/results"
    summary_path = results_dir / "task_conditioned_circuits_v4_summary.json"
    pairs_path = results_dir / "task_conditioned_circuits_v4_pairs.jsonl"
    activations_path = results_dir / "task_conditioned_circuits_v4_activations.jsonl"
    schema_path = results_dir / "task_conditioned_circuits_v4_schema.md"

    assert summary_path.exists(), "Summary JSON missing"
    assert pairs_path.exists(), "Pairs JSONL missing"
    assert activations_path.exists(), "Activations JSONL missing"
    assert schema_path.exists(), "Schema Markdown missing"

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    # Must be 50 total pairs (25 discovery, 25 held-out)
    assert summary["sample_counts"]["total_pairs_instrumented"] == 50
    assert summary["sample_counts"]["discovery_split_pairs"] == 25
    assert summary["sample_counts"]["held_out_evaluation_pairs"] == 25

    # Check that activations JSONL contains exact 50 lines with all 9 stages
    line_count = 0
    with activations_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assert "conditions" in rec
            assert "cond_aa" in rec["conditions"]
            assert len(rec["conditions"]["cond_aa"]["layer1_post_relu_64d"]) == 64
            line_count += 1
    assert line_count == 50
