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
    # ``None`` falls back to ``input_per_mtok`` (no discount/surcharge applied).
    cached_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None


PRICING: dict[str, dict[str, ModelPrice]] = {
    "anthropic": {
        # Cache write (5-min ephemeral) = input × 1.25; cache read = input × 0.10.
        "claude-haiku-4-5": ModelPrice(1.00, 5.00, 0.10, 1.25),
        "claude-sonnet-4-6": ModelPrice(3.00, 15.00, 0.30, 3.75),
        "claude-opus-4-7": ModelPrice(15.00, 75.00, 1.50, 18.75),
    },
    "openai": {
        # GPT-5 auto-cache: cached input = input × 0.10. No write surcharge.
        "gpt-5-mini": ModelPrice(0.25, 2.00, 0.025),
        "gpt-5": ModelPrice(5.00, 15.00, 0.50),
    },
    "gemini": {
        "gemini-2.5-flash": ModelPrice(0.30, 2.50),
        "gemini-2.5-pro": ModelPrice(1.25, 10.00),
    },
}


def price_completion(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cached_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """USD cost of a single completion at current snapshot prices.

    ``input_tokens`` is the uncached prompt portion. ``cached_read_tokens``
    and ``cache_write_tokens`` are billed at the model's cache rates if set,
    else at the base input rate.
    """
    try:
        price = PRICING[provider][model]
    except KeyError as exc:
        raise KeyError(
            f"No price for {provider!r}/{model!r}; add it to infra.pricing.PRICING."
        ) from exc
    cached_rate = (
        price.cached_read_per_mtok
        if price.cached_read_per_mtok is not None
        else price.input_per_mtok
    )
    write_rate = (
        price.cache_write_per_mtok
        if price.cache_write_per_mtok is not None
        else price.input_per_mtok
    )
    return (
        input_tokens / 1_000_000 * price.input_per_mtok
        + cached_read_tokens / 1_000_000 * cached_rate
        + cache_write_tokens / 1_000_000 * write_rate
        + output_tokens / 1_000_000 * price.output_per_mtok
    )