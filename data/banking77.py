"""Banking77 dataset loader.

77-class banking customer-service intent classification benchmark.
HuggingFace: https://huggingface.co/datasets/PolyAI/banking77
"""
from __future__ import annotations

import random
from collections import defaultdict
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
    ds = load_dataset("PolyAI/banking77")
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


def stratified_subset(
    examples: list[LabelledExample], n: int, seed: int = 0
) -> list[LabelledExample]:
    """Stratified subset preserving per-class proportions.

    Used for the Tier-2 (premium) model runs to keep total spend low while
    still getting a representative score per class.
    """
    by_label: dict[str, list[LabelledExample]] = defaultdict(list)
    for ex in examples:
        by_label[ex.label].append(ex)
    rng = random.Random(seed)
    total = len(examples)
    out: list[LabelledExample] = []
    for label, items in by_label.items():
        take = max(1, round(n * len(items) / total))
        rng.shuffle(items)
        out.extend(items[:take])
    rng.shuffle(out)
    return out[:n]