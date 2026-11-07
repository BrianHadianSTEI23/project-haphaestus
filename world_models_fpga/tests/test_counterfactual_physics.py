from research.datasets.counterfactual.validate_counterfactuals import validate_all


def test_all_counterfactuals_change_real_simulator_state():
    assert all(item["status"] == "VALID" for item in validate_all(pair_id=3))
