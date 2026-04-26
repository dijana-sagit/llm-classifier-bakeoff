"""Pairwise significance tests for paired classifiers.

Two complementary tools:

* **McNemar's test** — exact test for whether two classifiers' error patterns
  differ on the same test set. The right test for "did A beat B, or are
  the differences within sampling noise?".
* **Paired bootstrap** — distribution-free CI on the metric difference.
  Universal: works for accuracy, F1, top-k, anything.

Both consume aligned label sequences (one prediction per test example,
same order).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, chi2


@dataclass(frozen=True, slots=True)
class SignificanceResult:
    """Frozen value object describing a paired-classifier comparison."""

    test: str
    statistic: float
    p_value: float
    discordant_a_only: int
    discordant_b_only: int
    n: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def mcnemar_test(
    preds_a: list[str],
    preds_b: list[str],
    gold: list[str],
    *,
    exact_threshold: int = 25,
) -> SignificanceResult:
    """McNemar's test on the discordant pairs.

    For small samples (b + c < ``exact_threshold``) uses the exact binomial
    test on min(b, c) under H0: p=0.5. For larger samples uses the
    chi-squared statistic with Edwards' continuity correction:
    ``stat = (|b - c| - 1)^2 / (b + c)``.
    """
    if not (len(preds_a) == len(preds_b) == len(gold)):
        raise ValueError("preds_a, preds_b, and gold must have equal length.")

    b = sum(1 for pa, pb, g in zip(preds_a, preds_b, gold) if pa == g and pb != g)
    c = sum(1 for pa, pb, g in zip(preds_a, preds_b, gold) if pa != g and pb == g)
    n_disc = b + c

    if n_disc == 0:
        return SignificanceResult(
            test="mcnemar_exact",
            statistic=0.0,
            p_value=1.0,
            discordant_a_only=b,
            discordant_b_only=c,
            n=len(gold),
        )

    if n_disc < exact_threshold:
        result = binomtest(min(b, c), n_disc, p=0.5, alternative="two-sided")
        return SignificanceResult(
            test="mcnemar_exact",
            statistic=float(min(b, c)),
            p_value=float(result.pvalue),
            discordant_a_only=b,
            discordant_b_only=c,
            n=len(gold),
        )

    stat = (abs(b - c) - 1) ** 2 / n_disc
    p = float(chi2.sf(stat, df=1))
    return SignificanceResult(
        test="mcnemar_chi2",
        statistic=float(stat),
        p_value=p,
        discordant_a_only=b,
        discordant_b_only=c,
        n=len(gold),
    )


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Frozen CI on a paired metric difference."""

    point_estimate: float
    lower: float
    upper: float
    alpha: float
    n_resamples: int


def paired_bootstrap_accuracy_diff(
    preds_a: list[str],
    preds_b: list[str],
    gold: list[str],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapInterval:
    """(1 - alpha) CI on accuracy(A) - accuracy(B) by paired bootstrap.

    Pairing matters: resample the *test indices* with replacement, then
    score both classifiers on the resampled set. Naive unpaired bootstrap
    discards the per-example correlation and gives a wider, less-powerful CI.
    """
    if not (len(preds_a) == len(preds_b) == len(gold)):
        raise ValueError("preds_a, preds_b, and gold must have equal length.")

    a = np.array([pa == g for pa, g in zip(preds_a, gold)], dtype=np.float64)
    b = np.array([pb == g for pb, g in zip(preds_b, gold)], dtype=np.float64)
    diff = a - b

    rng = np.random.default_rng(seed)
    n = len(diff)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = diff[idx].mean()

    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return BootstrapInterval(
        point_estimate=float(diff.mean()),
        lower=lo,
        upper=hi,
        alpha=alpha,
        n_resamples=n_resamples,
    )


def _stable_seed(label: str) -> int:
    rng = random.Random(label)
    return rng.randint(0, 2**32 - 1)
