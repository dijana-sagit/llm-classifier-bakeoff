"""Plots for the analysis report.

Each function takes structured input (predictions, eval results, per-class
stats) and writes a single PNG. Plot functions never load data themselves
— composition stays in ``reports.py``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from analysis.per_label import PerClassStats
from eval.metrics import EvalResult


def plot_per_class_f1_boxplot(
    per_class_by_method: dict[str, list[PerClassStats]],
    out_path: Path,
) -> None:
    """Boxplot of per-class F1 scores, one box per method.

    Reveals distribution: a method with high mean F1 but a long lower tail
    is risky in production (classes it can't handle at all).
    """
    methods = sorted(per_class_by_method)
    data = [[s.f1 for s in per_class_by_method[m]] for m in methods]

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(methods)), 5))
    bp = ax.boxplot(data, tick_labels=methods, showmeans=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfd8ff")
    ax.set_ylabel("Per-class F1")
    ax.set_title("Per-class F1 distribution by method")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_cost_vs_accuracy(
    results: list[EvalResult],
    out_path: Path,
) -> None:
    """Cost-per-prediction vs accuracy — the "production decision" plot."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results:
        cost_per_pred = r.total_cost_usd / r.n if r.n else 0.0
        ax.scatter(cost_per_pred, r.top1_accuracy, s=80)
        ax.annotate(
            f"{r.method}/{r.model}",
            (cost_per_pred, r.top1_accuracy),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    ax.set_xscale("symlog", linthresh=1e-5)
    ax.set_xlabel("Cost per prediction (USD, symlog)")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Cost vs accuracy")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_confusion_heatmap(
    preds: list[str],
    gold: list[str],
    labels: list[str],
    out_path: Path,
    *,
    top_n: int = 20,
) -> None:
    """Confusion matrix heatmap, optionally restricted to the most-confused labels.

    ``top_n`` keeps the plot readable when there are 77 classes — picks
    labels involved in the most off-diagonal mass.
    """
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    for g, p in zip(gold, preds):
        if g in label_to_idx and p in label_to_idx:
            cm[label_to_idx[g], label_to_idx[p]] += 1

    if top_n < n_labels:
        off_diag = cm.copy()
        np.fill_diagonal(off_diag, 0)
        scores = off_diag.sum(axis=0) + off_diag.sum(axis=1)
        keep = np.argsort(-scores)[:top_n]
        keep.sort()
        cm = cm[np.ix_(keep, keep)]
        labels = [labels[i] for i in keep]

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(labels)), max(5, 0.4 * len(labels))))
    sns.heatmap(
        cm, ax=ax, cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=True
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title("Confusion matrix (top-N most confused labels)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
