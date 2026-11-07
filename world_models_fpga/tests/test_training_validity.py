import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.baseline_policy import BaselineNavigationPolicy, encode_task_id
from models.h_jepa import HJEPAModel
from models.le_world_model import LeWorldModel, SIGRegLoss


def test_sigreg_determinism_and_collapse_penalty():
    sig = SIGRegLoss(latent_dim=32, num_projections=32)
    sig.eval()
    gen = torch.Generator().manual_seed(42)
    z_gauss = torch.randn(128, 32, generator=gen)
    z_collapsed = torch.randn(128, 32, generator=gen) * 0.01

    l1 = sig(z_gauss)
    l2 = sig(z_gauss)
    assert torch.allclose(l1, l2, atol=1e-7), "SIGRegLoss must be deterministic in eval mode"
    l_col = sig(z_collapsed)
    assert float(l_col) > float(l1) + 0.5, "SIGRegLoss must penalize variance collapse"


def test_hjepa_and_lewm_gradient_flow():
    device = torch.device("cpu")
    lewm = LeWorldModel().to(device)
    img_t = torch.rand(8, 3, 64, 64)
    img_next = torch.rand(8, 3, 64, 64)
    act_t = torch.randn(8, 3).clamp(-1.0, 1.0)
    col_t = torch.zeros(8, 1)
    ego_t = torch.randn(8, 3) * 0.05
    goal_t = torch.randn(8, 4)
    task_ids = torch.tensor([encode_task_id("yellow_cylinder", "behind")] * 8, dtype=torch.long)

    loss_lewm, _ = lewm.compute_loss(
        img_t, act_t, img_next, col_t, ego_disp_target=ego_t, goal_vec=goal_t, task_ids=task_ids
    )
    loss_lewm.backward()
    for name, param in lewm.named_parameters():
        assert param.grad is not None, f"LeWorldModel parameter {name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"

    hjepa = HJEPAModel(stride_k=5).to(device)
    img_seq = torch.rand(8, 6, 3, 64, 64)
    act_seq = torch.randn(8, 6, 3).clamp(-1.0, 1.0)
    col_seq = torch.zeros(8, 6, 1)
    macro_ego = torch.randn(8, 3) * 0.2

    loss_hjepa, _ = hjepa.compute_loss(
        img_seq,
        act_seq,
        col_seq,
        micro_ego_target=ego_t,
        macro_ego_target=macro_ego,
        goal_vec=goal_t,
        task_ids=task_ids,
    )
    loss_hjepa.backward()
    for name, param in hjepa.named_parameters():
        assert param.grad is not None, f"HJEPAModel parameter {name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"

    # Verify MacroPredictor accepts both (B, 15) and (B, 3)
    s_dummy = torch.randn(4, 32)
    next_s_15, _ = hjepa.macro_predictor(s_dummy, torch.randn(4, 15))
    next_s_3, _ = hjepa.macro_predictor(s_dummy, torch.randn(4, 3))
    assert next_s_15.shape == (4, 32) and next_s_3.shape == (4, 32)


class TestTrainingValidity(unittest.TestCase):
    def test_sigreg(self):
        test_sigreg_determinism_and_collapse_penalty()

    def test_gradients_and_shapes(self):
        test_hjepa_and_lewm_gradient_flow()


if __name__ == "__main__":
    unittest.main()
