"""Banking77 dataset loader.

77-class banking customer-service intent classification benchmark.
HuggingFace: https://huggingface.co/datasets/PolyAI/banking77
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from datasets import load_dataset


@dataclass(frozen=True, slots=True)
class LabelledExample:
    """A labelled example flowing through methods and metrics."""

    text: str
    label: str


@dataclass(frozen=True, slots=True)
class Banking77Splits:
    train: list[LabelledExample]
    test: list[LabelledExample]
    label_names: list[str]


def load_banking77(seed: int = 0) -> Banking77Splits:
    """Load Banking77 train/test splits as ``LabelledExample`` lists."""
    # PolyAI/banking77 still ships a script-based loader, which `datasets`>=3
    # refuses to run. Load from the auto-generated parquet ref instead.
    ds = load_dataset("PolyAI/banking77", revision="refs/convert/parquet")
    label_names: list[str] = ds["train"].features["label"].names

    def to_examples(split) -> list[LabelledExample]:
        return [
            LabelledExample(text=row["text"], label=label_names[row["label"]])
            for row in split
        ]

    train = to_examples(ds["train"])
    test = to_examples(ds["test"])
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(test)
    return Banking77Splits(train=train, test=test, label_names=label_names)