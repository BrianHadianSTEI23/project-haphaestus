"""
Unit tests for Phase 4 Experiment B: Internal Mechanistic Circuits Benchmark (v3).
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "world_models_fpga"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
import torch

from env.amr_env import JetAutoEnv
from research.benchmarks.model_wrappers import HJEPAAgent
from research.datasets.dataset_config import CKPT_DIR
from research.circuits.run_task_conditioned_circuits_v3 import (
    decompose_grounding_forward,
)


@pytest.fixture(scope="module")
def agent():
    ckpt = CKPT_DIR / "hjepa_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HJEPAAgent(str(ckpt), device=device, track="observation")


def test_decompose_grounding_forward(agent):
    """Verifies that decompose_grounding_forward unrolls all 9 intermediate representations."""
    device = agent.device
    model = agent.model

    z = torch.randn(1, 32, device=device)
    e = torch.randn(1, 16, device=device)

    res = decompose_grounding_forward(model, z, e)

    expected_keys = ["z", "e", "inp48", "pre1", "post1", "pre2", "post2", "raw4", "goal4"]
    for k in expected_keys:
        assert k in res, f"Missing key {k} in unrolled grounding decomposition"

    assert res["inp48"].shape == (1, 48)
    assert res["pre1"].shape == (1, 64)
    assert res["post1"].shape == (1, 64)
    assert res["pre2"].shape == (1, 64)
    assert res["post2"].shape == (1, 64)
    assert res["raw4"].shape == (1, 4)
    assert res["goal4"].shape == (1, 4)

    # Verify that hooking layer 1 works
    donor_post1 = torch.zeros(1, 64, device=device)
    hook_l1 = lambda p1: donor_post1
    res_hooked = decompose_grounding_forward(model, z, e, hook_layer1=hook_l1)
    np.testing.assert_allclose(res_hooked["post1"].cpu().numpy(), donor_post1.cpu().numpy())
    assert not torch.allclose(res["goal4"], res_hooked["goal4"])
