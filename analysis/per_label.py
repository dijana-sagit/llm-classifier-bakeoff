"""Per-class statistics with bootstrap confidence intervals.

77 classes × headline metric obscures which intents are easy and which are
disasters. This module reports per-class precision/recall/F1 with CIs,
plus rank correlation between methods (do methods agree on which classes
are hard?).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True, slots=True)
class PerClassStats:
    """Frozen value object: precision/recall/F1 with bootstrap CI."""

    label: str
    support: int
    precision: float
    recall: float
    f1: float
    f1_lower: float
    f1_upper: float


def _f1_from_arrays(pred: np.ndarray, gold: np.ndarray, label: str) -> float:
    tp = int(((pred == label) & (gold == label)).sum())
    fp = int(((pred == label) & (gold != label)).sum())
    fn = int(((pred != label) & (gold == label)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def per_class_f1_with_ci(
    preds: list[str],
    gold: list[str],
    *,
    labels: list[str] | None = None,
    n_resamples: int = 2_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[PerClassStats]:
    """Per-class precision/recall/F1 with bootstrap CI on F1.

    Bootstrap by resampling test indices — propagates the same
    inter-example correlation as ``paired_bootstrap_accuracy_diff``.
    """
    if len(preds) != len(gold):
        raise ValueError("preds and gold must have equal length.")
    pred_arr = np.array(preds)
    gold_arr = np.array(gold)
    label_set = labels if labels is not None else sorted(set(gold) | set(preds))

    rng = np.random.default_rng(seed)
    n = len(gold_arr)
    boot_idx = rng.integers(0, n, size=(n_resamples, n))

    out: list[PerClassStats] = []
    for label in label_set:
        tp = int(((pred_arr == label) & (gold_arr == label)).sum())
        fp = int(((pred_arr == label) & (gold_arr != label)).sum())
        fn = int(((pred_arr != label) & (gold_arr == label)).sum())
        support = int((gold_arr == label).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        boot_f1s = np.empty(n_resamples, dtype=np.float64)
        for i in range(n_resamples):
            idx = boot_idx[i]
            boot_f1s[i] = _f1_from_arrays(pred_arr[idx], gold_arr[idx], label)
        lo = float(np.quantile(boot_f1s, alpha / 2))
        hi = float(np.quantile(boot_f1s, 1 - alpha / 2))

        out.append(
            PerClassStats(
                label=label,
                support=support,
                precision=precision,
                recall=recall,
                f1=f1,
                f1_lower=lo,
                f1_upper=hi,
            )
        )
    return out


def rank_correlation_between_methods(
    per_class_by_method: dict[str, list[PerClassStats]],
) -> dict[tuple[str, str], float]:
    """Spearman rank correlation of per-class F1 between every method pair.

    High correlation → methods agree on which classes are easy/hard
    (the dataset's class difficulty dominates). Low correlation →
    methods have genuinely different failure modes — worth investigating.
    """
    method_to_f1: dict[str, dict[str, float]] = {
        m: {s.label: s.f1 for s in stats} for m, stats in per_class_by_method.items()
    }
    methods = sorted(method_to_f1)
    results: dict[tuple[str, str], float] = {}
    for i, ma in enumerate(methods):
        for mb in methods[i + 1 :]:
            shared = sorted(set(method_to_f1[ma]) & set(method_to_f1[mb]))
            if len(shared) < 3:
                results[(ma, mb)] = float("nan")
                continue
            xs = [method_to_f1[ma][lab] for lab in shared]
            ys = [method_to_f1[mb][lab] for lab in shared]
            rho, _ = spearmanr(xs, ys)
            results[(ma, mb)] = float(rho)
    return results


def hardest_classes(stats: list[PerClassStats], k: int = 5) -> list[PerClassStats]:
    """Return the ``k`` classes with the lowest F1 — the error-analysis seed list."""
    return sorted(stats, key=lambda s: s.f1)[:k]


def confusion_pairs(
    preds: list[str], gold: list[str], top_n: int = 10
) -> list[tuple[str, str, int]]:
    """Top ``top_n`` (gold_label, predicted_label, count) confusions for error analysis."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for p, g in zip(preds, gold):
        if p != g:
            counts[(g, p)] += 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return [(gold_lab, pred_lab, n) for (gold_lab, pred_lab), n in ranked[:top_n]]
