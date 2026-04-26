"""Token pricing snapshot.

Prices drift; this module is a single point of update. ``run_eval.py``
writes the active pricing to ``results/run_metadata.json`` so the README
table is reproducible from a snapshot.

Prices in USD per 1M tokens. Verify against provider docs at run time.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float


PRICING: dict[str, dict[str, ModelPrice]] = {
    "anthropic": {
        # TODO: verify exact IDs and prices at run time
        "claude-haiku-4-5": ModelPrice(1.00, 5.00),
        "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
        "claude-opus-4-7": ModelPrice(15.00, 75.00),
    },
    "openai": {
        "gpt-5-mini": ModelPrice(0.25, 2.00),
        "gpt-5": ModelPrice(5.00, 15.00),
    },
    "gemini": {
        "gemini-2.5-flash": ModelPrice(0.30, 2.50),
        "gemini-2.5-pro": ModelPrice(1.25, 10.00),
    },
}


def price_completion(
    provider: str, model: str, input_tokens: int, output_tokens: int
) -> float:
    """USD cost of a single completion at current snapshot prices."""
    try:
        price = PRICING[provider][model]
    except KeyError as exc:
        raise KeyError(
            f"No price for {provider!r}/{model!r}; add it to infra.pricing.PRICING."
        ) from exc
    return (
        input_tokens / 1_000_000 * price.input_per_mtok
        + output_tokens / 1_000_000 * price.output_per_mtok
    )