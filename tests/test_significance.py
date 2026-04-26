"""Tests for significance tests against known results."""
from __future__ import annotations

from analysis.significance import mcnemar_test, paired_bootstrap_accuracy_diff


def _build_paired_predictions(
    n_both_right: int,
    n_a_only: int,
    n_b_only: int,
    n_both_wrong: int,
) -> tuple[list[str], list[str], list[str]]:
    """Construct paired predictions with the requested confusion counts.

    For each example the gold label is 'x'. Predictions are either 'x'
    (correct) or 'y' (wrong).
    """
    gold = ["x"] * (n_both_right + n_a_only + n_b_only + n_both_wrong)
    preds_a, preds_b = [], []
    for _ in range(n_both_right):
        preds_a.append("x")
        preds_b.append("x")
    for _ in range(n_a_only):
        preds_a.append("x")
        preds_b.append("y")
    for _ in range(n_b_only):
        preds_a.append("y")
        preds_b.append("x")
    for _ in range(n_both_wrong):
        preds_a.append("y")
        preds_b.append("y")
    return preds_a, preds_b, gold


def test_mcnemar_no_disagreement_returns_unity_p():
    a, b, g = _build_paired_predictions(50, 0, 0, 0)
    res = mcnemar_test(a, b, g)
    assert res.p_value == 1.0
    assert not res.significant


def test_mcnemar_lopsided_disagreement_is_significant():
    # 30 cases where A is right and B is wrong, 5 the other way.
    a, b, g = _build_paired_predictions(40, 30, 5, 0)
    res = mcnemar_test(a, b, g)
    assert res.discordant_a_only == 30
    assert res.discordant_b_only == 5
    assert res.p_value < 0.001
    assert res.significant


def test_mcnemar_balanced_disagreement_is_not_significant():
    # Equal disagreement in both directions — should not reject.
    a, b, g = _build_paired_predictions(40, 10, 10, 0)
    res = mcnemar_test(a, b, g)
    assert res.p_value > 0.05
    assert not res.significant


def test_paired_bootstrap_ci_contains_zero_when_methods_agree():
    a, b, g = _build_paired_predictions(40, 5, 5, 10)
    ci = paired_bootstrap_accuracy_diff(a, b, g, n_resamples=2000, seed=0)
    assert ci.lower <= 0.0 <= ci.upper


def test_paired_bootstrap_ci_excludes_zero_when_methods_disagree_strongly():
    a, b, g = _build_paired_predictions(40, 30, 1, 0)
    ci = paired_bootstrap_accuracy_diff(a, b, g, n_resamples=2000, seed=0)
    assert ci.lower > 0.0
    assert ci.point_estimate > 0.0
