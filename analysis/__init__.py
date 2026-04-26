"""Statistical analysis layer.

Consumes per-prediction traces written by ``eval/run_eval.py`` and produces
significance tests, per-class statistics, and plots — the depth that
distinguishes a portfolio benchmark from "I ran a notebook."
"""
from analysis.per_label import PerClassStats, per_class_f1_with_ci
from analysis.significance import (
    SignificanceResult,
    mcnemar_test,
    paired_bootstrap_accuracy_diff,
)

__all__ = [
    "PerClassStats",
    "per_class_f1_with_ci",
    "SignificanceResult",
    "mcnemar_test",
    "paired_bootstrap_accuracy_diff",
]
