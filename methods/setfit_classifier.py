"""SetFit fine-tune: contrastive sentence-transformer + linear classification head.

Two-phase training:
  1. Fine-tune the sentence-transformer body with contrastive pairs (same-label
     positives, different-label negatives) to specialise embeddings for the
     class boundaries we care about.
  2. Train a logistic-regression head on the (now class-aware) embeddings.

The fitted model is cached under ``results/cache/setfit/`` keyed on the train
set + hyperparameters, so subsequent runs skip the ~minutes of training.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

from data.banking77 import LabelledExample
from infra.registry import register_method
from methods.base import Prediction

CACHE_DIR = Path("results/cache/setfit")


@register_method("setfit")
class SetFitClassifier:
    """Sentence-transformer fine-tuned contrastively + sklearn linear head."""

    name = "setfit"

    def __init__(
        self,
        encoder: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        num_epochs: int = 1,
        # Contrastive pairs sampled per training example. SetFit's default is
        # 20; we drop to 10 to keep CPU training under ~10 min on Banking77.
        num_iterations: int = 10,
        batch_size: int = 32,
    ) -> None:
        self.encoder = encoder
        self.num_epochs = num_epochs
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self._model: SetFitModel | None = None
        self._labels: list[str] | None = None

    def _cache_key(self, train: list[LabelledExample]) -> str:
        h = hashlib.sha256()
        for part in (
            self.encoder,
            str(self.num_epochs),
            str(self.num_iterations),
            str(self.batch_size),
        ):
            h.update(part.encode())
            h.update(b"\0")
        for ex in train:
            h.update(ex.text.encode())
            h.update(b"\0")
            h.update(ex.label.encode())
            h.update(b"\0")
        return h.hexdigest()[:16]

    def fit(self, train: list[LabelledExample]) -> None:
        labels = sorted({ex.label for ex in train})
        self._labels = labels
        cache_path = CACHE_DIR / self._cache_key(train)

        if cache_path.exists():
            self._model = SetFitModel.from_pretrained(str(cache_path))
            return

        label_to_id = {lbl: i for i, lbl in enumerate(labels)}
        train_ds = Dataset.from_dict(
            {
                "text": [ex.text for ex in train],
                "label": [label_to_id[ex.label] for ex in train],
            }
        )
        self._model = SetFitModel.from_pretrained(self.encoder, labels=labels)
        args = TrainingArguments(
            num_epochs=self.num_epochs,
            num_iterations=self.num_iterations,
            batch_size=self.batch_size,
        )
        trainer = Trainer(model=self._model, args=args, train_dataset=train_ds)
        trainer.train()

        cache_path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(cache_path))

    def predict(self, query: str) -> Prediction:
        if self._model is None or self._labels is None:
            raise RuntimeError("SetFitClassifier.fit must be called before predict.")

        t0 = time.perf_counter()
        probs = self._model.predict_proba([query])[0]
        # predict_proba returns a torch tensor or numpy array depending on
        # backend; coerce to a flat python list either way.
        probs = np.asarray(probs).tolist()
        latency_ms = (time.perf_counter() - t0) * 1000

        ranked = sorted(zip(self._labels, probs, strict=True), key=lambda x: -x[1])
        predicted = ranked[0][0]
        return Prediction(
            query=query,
            predicted_label=predicted,
            top_k=ranked,
            latency_ms=latency_ms,
            cost_usd=0.0,
        )

    def predict_batch(self, queries: list[str]) -> list[Prediction]:
        return [self.predict(q) for q in queries]
