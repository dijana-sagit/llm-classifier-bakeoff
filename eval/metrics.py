"""Classification metrics: top-1 accuracy, macro-F1, top-3 accuracy.

Pure functions over ``Prediction`` lists — no sklearn dependency for the
metric maths so the dependency footprint stays small.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

from data.banking77 import LabelledExample
from methods.base import Prediction


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Frozen value object summarising a method's run on a test set."""

    method: str
    model: str
    n: int
    top1_accuracy: float
    macro_f1: float
    top3_accuracy: float
    total_cost_usd: float
    p50_latency_ms: float
    p95_latency_ms: float


def top1_accuracy(preds: list[Prediction], gold: list[LabelledExample]) -> float:
    correct = sum(1 for p, g in zip(preds, gold, strict=True) if p.predicted_label == g.label)
    return correct / len(gold)


def top_k_accuracy(preds: list[Prediction], gold: list[LabelledExample], k: int) -> float:
    hits = 0
    for p, g in zip(preds, gold, strict=True):
        candidates = [label for label, _ in p.top_k[:k]] or [p.predicted_label]
        if g.label in candidates:
            hits += 1
    return hits / len(gold)


def macro_f1(preds: list[Prediction], gold: list[LabelledExample]) -> float:
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    labels: set[str] = set()
    for p, g in zip(preds, gold, strict=True):
        labels.add(g.label)
        labels.add(p.predicted_label)
        if p.predicted_label == g.label:
            tp[g.label] += 1
        else:
            fp[p.predicted_label] += 1
            fn[g.label] += 1
    f1s: list[float] = []
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return sum(f1s) / len(f1s) if f1s else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(pct / 100 * (len(s) - 1))))
    return s[idx]


def summarise(
    method_name: str,
    model: str,
    preds: list[Prediction],
    gold: list[LabelledExample],
) -> EvalResult:
    latencies = [p.latency_ms for p in preds]
    return EvalResult(
        method=method_name,
        model=model,
        n=len(gold),
        top1_accuracy=top1_accuracy(preds, gold),
        macro_f1=macro_f1(preds, gold),
        top3_accuracy=top_k_accuracy(preds, gold, k=3),
        total_cost_usd=sum(p.cost_usd for p in preds),
        p50_latency_ms=statistics.median(latencies) if latencies else 0.0,
        p95_latency_ms=_percentile(latencies, 95),
    )