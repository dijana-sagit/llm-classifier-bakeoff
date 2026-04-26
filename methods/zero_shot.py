"""Zero-shot LLM classifier: ask the model to pick from all 77 labels.

The naive baseline against which the cleverer methods (retrieval-augmented,
few-shot, fine-tuned) must justify their extra complexity. Composition: the
method *has-a* provider rather than inheriting — so the same class works
with Anthropic, OpenAI, or Gemini behind the ``LLMProvider`` Protocol.

The full label list goes into the system prompt and is reused on every
call, making it a perfect prompt-caching target on Anthropic.
"""
from __future__ import annotations

import time

from data.banking77 import LabelledExample
from infra.registry import register_method
from methods.base import Prediction
from providers.base import LLMProvider
from providers.cached import cached_complete

UNPARSED_LABEL = "_unparsed"

_SYSTEM_TEMPLATE = """\
You are a banking-customer-support intent classifier.

Given a customer query, return EXACTLY ONE intent label from the list below \
that best matches. Output ONLY the label, with no preamble, punctuation, \
quotes, or explanation. Use the exact spelling and underscores shown.

Allowed labels:
{labels}
"""


@register_method("zero_shot")
class ZeroShotLLM:
    """Zero-shot intent classification via an LLM provider."""

    name = "zero_shot"

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        *,
        max_tokens: int = 32,
        temperature: float = 0.0,
        cache_system: bool = True,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_system = cache_system
        self._label_list: list[str] | None = None
        self._system: str | None = None

    def fit(self, train: list[LabelledExample]) -> None:
        labels = sorted({ex.label for ex in train})
        self._label_list = labels
        self._system = _SYSTEM_TEMPLATE.format(labels="\n".join(labels))

    def predict(self, query: str) -> Prediction:
        if self._label_list is None or self._system is None:
            raise RuntimeError("ZeroShotLLM.fit must be called before predict.")

        t0 = time.perf_counter()
        response = cached_complete(
            provider_name=self.provider.name,
            model=self.model,
            system=self._system,
            user=query,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            cache_system=self.cache_system,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        label = parse_label(response.text, self._label_list)
        return Prediction(
            query=query,
            predicted_label=label,
            top_k=[(label, 1.0)],
            latency_ms=latency_ms,
            cost_usd=response.cost_usd,
            raw_response=response.text,
        )

    def predict_batch(self, queries: list[str]) -> list[Prediction]:
        return [self.predict(q) for q in queries]


def parse_label(text: str, allowed: list[str]) -> str:
    """Map a free-form LLM response to one of the allowed labels.

    Returns ``UNPARSED_LABEL`` when no allowed label can be located —
    deliberately a non-matching sentinel so it counts as wrong in metrics
    and shows up in ``hardest_classes`` for diagnosis instead of being
    silently swallowed.
    """
    if not text:
        return UNPARSED_LABEL
    cleaned = text.strip().strip("\"'`. ").lower()
    for prefix in ("label:", "intent:", "answer:", "category:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break
    lower_to_label = {label.lower(): label for label in allowed}
    if cleaned in lower_to_label:
        return lower_to_label[cleaned]
    cleaned_underscored = cleaned.replace(" ", "_")
    if cleaned_underscored in lower_to_label:
        return lower_to_label[cleaned_underscored]
    for low, label in lower_to_label.items():
        if low in cleaned:
            return label
    return UNPARSED_LABEL
