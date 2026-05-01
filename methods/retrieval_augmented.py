"""Retrieval-augmented LLM: embed query → top-K candidate labels → LLM picks.

The zero-shot baseline dumps all 77 labels in every prompt. Here the embedding
step pre-filters to ~10 plausible labels and the LLM only has to disambiguate
among those — much easier task, much smaller prompt.
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

The user message contains a customer query and a list of candidate intent \
labels. Return EXACTLY ONE label from the candidate list — the one that \
best matches the query. Output ONLY the label, with no preamble, \
punctuation, quotes, or explanation. Use the exact spelling and \
underscores shown.\
"""

_USER_TEMPLATE = """\
Query: {query}

Candidates:
{candidates}\
"""


@register_method("retrieval_augmented")
class RetrievalAugmentedLLM:
    """Embedding retrieval narrows the label set; the LLM picks among candidates.

    Stateful: ``fit`` encodes the training set once; ``predict`` ranks labels by
    max cosine similarity of the query against each label's training examples,
    keeps the top-K labels, and asks the LLM to pick.
    """

    name = "retrieval_augmented"

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        *,
        encoder: str = "BAAI/bge-small-en-v1.5",
        k_candidates: int = 10,
        max_tokens: int = 32,
        temperature: float = 0.0,
        cache_system: bool = True,
    ) -> None:
        self.provider = provider
        self.model = model
        self.k_candidates = k_candidates
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_system = cache_system
        self._encoder = SentenceTransformer(encoder)
        self._train_emb: np.ndarray | None = None
        self._train_labels: list[str] | None = None
        self._all_labels: list[str] | None = None

    def fit(self, train: list[LabelledExample]) -> None:
        texts = [ex.text for ex in train]
        self._train_emb = self._encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        self._train_labels = [ex.label for ex in train]
        self._all_labels = sorted({ex.label for ex in train})

    def _retrieve(self, query: str) -> list[tuple[str, float]]:
        """Top-K unique labels by max cosine sim of query to each label's examples."""
        assert self._train_emb is not None and self._train_labels is not None
        q_emb = self._encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        sims = self._train_emb @ q_emb
        order = np.argsort(-sims)
        seen: dict[str, float] = {}
        for idx in order:
            label = self._train_labels[idx]
            if label not in seen:
                seen[label] = float(sims[idx])
                if len(seen) >= self.k_candidates:
                    break
        return list(seen.items())

    def predict(self, query: str) -> Prediction:
        if self._train_emb is None or self._all_labels is None:
            raise RuntimeError("RetrievalAugmentedLLM.fit must be called before predict.")

        t0 = time.perf_counter()
        candidates = self._retrieve(query)
        candidate_labels = [lbl for lbl, _ in candidates]
        user_msg = _USER_TEMPLATE.format(
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
            top_k=[(label, 1.0)],
            latency_ms=latency_ms,
            cost_usd=response.cost_usd,
            raw_response=response.text,
        )

    def predict_batch(self, queries: list[str]) -> list[Prediction]:
        return [self.predict(q) for q in queries]
