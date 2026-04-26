"""Metric-function correctness tests."""
from __future__ import annotations

from data.banking77 import LabelledExample
from eval.metrics import macro_f1, summarise, top1_accuracy, top_k_accuracy
from methods.base import Prediction


def _pred(query: str, label: str, top_k: list[tuple[str, float]] | None = None) -> Prediction:
    return Prediction(query=query, predicted_label=label, top_k=top_k or [(label, 1.0)])


def test_top1_all_correct():
    gold = [LabelledExample("a", "x"), LabelledExample("b", "y")]
    preds = [_pred("a", "x"), _pred("b", "y")]
    assert top1_accuracy(preds, gold) == 1.0


def test_top1_half_correct():
    gold = [LabelledExample("a", "x"), LabelledExample("b", "y")]
    preds = [_pred("a", "x"), _pred("b", "z")]
    assert top1_accuracy(preds, gold) == 0.5


def test_top_k_accuracy_picks_from_top_k():
    gold = [LabelledExample("a", "y")]
    preds = [_pred("a", "x", top_k=[("x", 0.6), ("y", 0.4)])]
    assert top_k_accuracy(preds, gold, k=2) == 1.0
    assert top_k_accuracy(preds, gold, k=1) == 0.0


def test_macro_f1_perfect_classifier():
    gold = [LabelledExample("a", "x"), LabelledExample("b", "y")]
    preds = [_pred("a", "x"), _pred("b", "y")]
    assert macro_f1(preds, gold) == 1.0


def test_macro_f1_treats_classes_equally():
    # Heavy class 'x' (3 right) + minority class 'y' (0 right).
    # Top-1 acc ignores imbalance; macro-F1 should call this out.
    gold = [
        LabelledExample("a", "x"),
        LabelledExample("b", "x"),
        LabelledExample("c", "x"),
        LabelledExample("d", "y"),
    ]
    preds = [_pred("a", "x"), _pred("b", "x"), _pred("c", "x"), _pred("d", "x")]
    assert top1_accuracy(preds, gold) == 0.75
    assert macro_f1(preds, gold) < 0.75


def test_summarise_packages_into_eval_result():
    gold = [LabelledExample("a", "x")]
    preds = [Prediction(query="a", predicted_label="x", latency_ms=10.0, cost_usd=0.001)]
    result = summarise("knn", "-", preds, gold)
    assert result.method == "knn"
    assert result.n == 1
    assert result.top1_accuracy == 1.0
    assert result.total_cost_usd == 0.001