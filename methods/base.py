"""Common interface for classification methods.

Every method (kNN, zero-shot LLM, retrieval-augmented LLM, few-shot LLM,
fine-tuned SetFit) implements ``ClassificationMethod`` and is therefore
interchangeable behind the same call site in ``eval/run_eval.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from data.banking77 import LabelledExample


@dataclass(frozen=True, slots=True)
class Prediction:
    """Immutable result of a single classification call."""

    query: str
    predicted_label: str
    top_k: list[tuple[str, float]] = field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    raw_response: str | None = None


@runtime_checkable
class ClassificationMethod(Protocol):
    """Interchangeable classification algorithms.

    Implementations may be stateless (zero-shot LLM) or stateful (kNN,
    fine-tuned SetFit). ``fit`` is a no-op for stateless methods.
    """

    name: str

    def fit(self, train: list[LabelledExample]) -> None: ...

    def predict(self, query: str) -> Prediction: ...

    def predict_batch(self, queries: list[str]) -> list[Prediction]:
        """Default sequential implementation; override for true batching."""
        ...