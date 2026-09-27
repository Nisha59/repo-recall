import pytest

from repo_recall.generation.cost import estimate_usage


def test_cost_uses_per_million_token_prices():
    usage = estimate_usage(2_000, 500, input_cost_per_mtok=0.5, output_cost_per_mtok=2.0)
    assert usage.estimated_cost_usd == pytest.approx(0.002)


def test_unknown_token_counts_give_unknown_cost():
    assert estimate_usage(None, 10, 1.0, 1.0).estimated_cost_usd is None


def test_free_local_model_costs_zero():
    assert estimate_usage(3_000, 400, 0.0, 0.0).estimated_cost_usd == 0.0
