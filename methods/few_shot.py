"""Few-shot LLM: top-N retrieved labelled examples + candidate labels in the prompt.

Same retrieval as ``RetrievalAugmentedLLM``, plus N nearest training examples
shown as in-context demonstrations. The hypothesis is that real query-to-label
phrasings give the LLM more signal than label names alone — particularly
helpful where labels are near-synonyms (e.g. ``top_up_by_card_charge`` vs
``topping_up_by_card``).
"""
from __future__ import annotations

import time

import numpy as np
from sentence_transformers import SentenceTransformer

from data.banking77 import LabelledExample
from infra.registry import register_method
from methods.base import Prediction
from methods.zero_shot import UNPARSED_LABEL, parse_label
from providers.base import LLMProvider
from providers.cached import cached_complete

_SYSTEM = """\
You are a banking-customer-support intent classifier.

You will see a few examples of customer queries with their correct intent \
labels, then a new query and a list of candidate labels. Return EXACTLY ONE \
label from the candidate list that best matches the new query. Output ONLY \
the label, with no preamble, punctuation, quotes, or explanation. Use the \
exact spelling and underscores shown.\
"""

_USER_TEMPLATE = """\
Examples:
{examples}

Now classify:
Query: {query}

Candidates:
{candidates}\
"""


@register_method("few_shot")
class FewShotLLM:
    """Top-N labelled examples + top-K candidate labels in the prompt."""

    name = "few_shot"

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        *,
        encoder: str = "BAAI/bge-small-en-v1.5",
        n_examples: int = 5,
        k_candidates: int = 10,
        max_tokens: int = 32,
        temperature: float = 0.0,
        cache_system: bool = True,
    ) -> None:
        self.provider = provider
        self.model = model
        self.n_examples = n_examples
        self.k_candidates = k_candidates
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_system = cache_system
        self._encoder = SentenceTransformer(encoder)
        self._train_emb: np.ndarray | None = None
        self._train_texts: list[str] | None = None
        self._train_labels: list[str] | None = None
        self._all_labels: list[str] | None = None

    def fit(self, train: list[LabelledExample]) -> None:
        texts = [ex.text for ex in train]
        self._train_emb = self._encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        self._train_texts = texts
        self._train_labels = [ex.label for ex in train]
        self._all_labels = sorted({ex.label for ex in train})

    def _rank(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (sims, sorted_indices) for the query against training set."""
        assert self._train_emb is not None
        q_emb = self._encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        sims = self._train_emb @ q_emb
        order = np.argsort(-sims)
        return sims, order

    def _top_examples(
        self, order: np.ndarray, n: int
    ) -> list[tuple[str, str]]:
        """Top-n nearest training examples (duplicates allowed)."""
        assert self._train_texts is not None and self._train_labels is not None
        return [(self._train_texts[i], self._train_labels[i]) for i in order[:n]]

    def _top_unique_labels(
        self, sims: np.ndarray, order: np.ndarray, k: int
    ) -> list[tuple[str, float]]:
        """Top-k unique labels by max-pool similarity."""
        assert self._train_labels is not None
        seen: dict[str, float] = {}
        for idx in order:
            label = self._train_labels[idx]
            if label not in seen:
                seen[label] = float(sims[idx])
                if len(seen) >= k:
                    break
        return list(seen.items())

    def predict(self, query: str) -> Prediction:
        if self._train_emb is None or self._all_labels is None:
            raise RuntimeError("FewShotLLM.fit must be called before predict.")

        t0 = time.perf_counter()
        sims, order = self._rank(query)
        examples = self._top_examples(order, self.n_examples)
        candidates = self._top_unique_labels(sims, order, self.k_candidates)
        candidate_labels = [lbl for lbl, _ in candidates]

        examples_block = "\n\n".join(
            f"Q: {q}\nA: {lbl}" for q, lbl in examples
        )
        user_msg = _USER_TEMPLATE.format(
            examples=examples_block,
            query=query,
            candidates="\n".join(f"- {lbl}" for lbl in candidate_labels),
        )

        response = cached_complete(
            provider_name=self.provider.name,
            model=self.model,
            system=_SYSTEM,
            user=user_msg,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            cache_system=self.cache_system,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        label = parse_label(response.text, candidate_labels)
        if label == UNPARSED_LABEL:
            label = parse_label(response.text, self._all_labels)

        return Prediction(
            query=query,
            predicted_label=label,
            top_k=candidates,
            latency_ms=latency_ms,
            cost_usd=response.cost_usd,
            raw_response=response.text,
        )

    def predict_batch(self, queries: list[str]) -> list[Prediction]:
        return [self.predict(q) for q in queries]
