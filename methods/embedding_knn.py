"""kNN baseline: sentence-transformer embedding + cosine similarity.

No external API calls; runs on CPU. The ``$0/prediction`` data point on the
cost-vs-accuracy plot.
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer

from data.banking77 import LabelledExample
from infra.registry import register_method
from methods.base import Prediction


@register_method("knn")
class EmbeddingKNN:
    """kNN over sentence-transformer embeddings.

    Stateful: ``fit`` encodes the training set once. ``predict`` does a
    cosine-similarity lookup against the cached matrix.
    """

    name = "knn"

    def __init__(
        self,
        encoder: str = "BAAI/bge-small-en-v1.5",
        k: int = 7,
    ) -> None:
        self._encoder = SentenceTransformer(encoder)
        self._k = k
        self._train_emb: np.ndarray | None = None
        self._train_labels: list[str] | None = None

    def fit(self, train: list[LabelledExample]) -> None:
        texts = [ex.text for ex in train]
        self._train_emb = self._encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        self._train_labels = [ex.label for ex in train]

    def predict(self, query: str) -> Prediction:
        if self._train_emb is None or self._train_labels is None:
            raise RuntimeError("EmbeddingKNN.fit must be called before predict.")
        t0 = time.perf_counter()
        q_emb = self._encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        sims = self._train_emb @ q_emb
        top_idx = np.argpartition(-sims, self._k)[: self._k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        votes = Counter(self._train_labels[i] for i in top_idx)
        predicted, _ = votes.most_common(1)[0]
        top_k = [(label, count / self._k) for label, count in votes.most_common()]
        latency_ms = (time.perf_counter() - t0) * 1000
        return Prediction(
            query=query,
            predicted_label=predicted,
            top_k=top_k,
            latency_ms=latency_ms,
            cost_usd=0.0,
        )

    def predict_batch(self, queries: list[str]) -> list[Prediction]:
        return [self.predict(q) for q in queries]