"""Tests for per-class statistics."""
from __future__ import annotations

from analysis.per_label import (
    confusion_pairs,
    hardest_classes,
    per_class_f1_with_ci,
    rank_correlation_between_methods,
)


def test_per_class_f1_perfect_predictor():
    preds = ["x", "y", "x", "y"]
    gold = ["x", "y", "x", "y"]
    stats = per_class_f1_with_ci(preds, gold, n_resamples=200, seed=0)
    by_label = {s.label: s for s in stats}
    assert by_label["x"].f1 == 1.0
    assert by_label["y"].f1 == 1.0
    assert by_label["x"].support == 2


def test_per_class_f1_isolates_failing_class():
    preds = ["x"] * 4 + ["x"] * 4  # always says x
    gold = ["x"] * 4 + ["y"] * 4
    stats = per_class_f1_with_ci(preds, gold, n_resamples=200, seed=0)
    by_label = {s.label: s for s in stats}
    assert by_label["x"].f1 > 0.0
    assert by_label["y"].f1 == 0.0  # never predicted, support exists


def test_hardest_classes_returns_lowest_f1():
    preds = ["x", "x", "z", "y"]
    gold = ["x", "y", "z", "y"]
    stats = per_class_f1_with_ci(preds, gold, n_resamples=200, seed=0)
    worst = hardest_classes(stats, k=1)
    assert len(worst) == 1
    assert all(worst[0].f1 <= s.f1 for s in stats)


def test_rank_correlation_identical_methods_is_one():
    preds = ["x", "y", "x", "y", "z", "z"]
    gold = ["x", "y", "x", "y", "z", "z"]
    stats = per_class_f1_with_ci(preds, gold, n_resamples=200, seed=0)
    rho = rank_correlation_between_methods({"a": stats, "b": stats})
    assert rho[("a", "b")] == 1.0


def test_confusion_pairs_orders_by_count():
    preds = ["y", "y", "z", "x"]
    gold = ["x", "x", "x", "x"]
    pairs = confusion_pairs(preds, gold, top_n=3)
    assert pairs[0] == ("x", "y", 2)
    assert {p[1] for p in pairs} == {"y", "z"}
