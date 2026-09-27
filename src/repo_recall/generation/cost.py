"""Estimating what one LLM call cost."""

from repo_recall.generation.models import TokenUsage

TOKENS_PER_PRICE_UNIT = 1_000_000


def estimate_usage(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    input_cost_per_mtok: float,
    output_cost_per_mtok: float,
) -> TokenUsage:
    """Token counts plus cost, from prices in USD per million tokens.

    Prices come from config because they differ per provider and change
    over time; a local Ollama model costs 0. Cost is None if either
    count is unknown, rather than a misleading partial figure.
    """
    cost = None
    if prompt_tokens is not None and completion_tokens is not None:
        cost = (
            prompt_tokens * input_cost_per_mtok + completion_tokens * output_cost_per_mtok
        ) / TOKENS_PER_PRICE_UNIT
    return TokenUsage(prompt_tokens, completion_tokens, cost)
