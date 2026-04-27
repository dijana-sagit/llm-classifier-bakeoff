"""Assemble a markdown analysis report from result traces.

Loads per-method prediction CSVs written by ``eval/run_eval.py``,
runs significance tests, builds per-class stats, generates plots, and
emits ``results/analysis_report.md``.

Usage::

    python -m analysis.reports
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from analysis.per_label import (
    PerClassStats,
    confusion_pairs,
    hardest_classes,
    per_class_f1_with_ci,
    rank_correlation_between_methods,
)
from analysis.plots import (
    plot_confusion_heatmap,
    plot_cost_vs_accuracy,
    plot_per_class_f1_boxplot,
)
from analysis.significance import (
    SignificanceResult,
    mcnemar_test,
    paired_bootstrap_accuracy_diff,
)
from eval.metrics import EvalResult

PREDICTIONS_DIR = Path("results/predictions")
PLOTS_DIR = Path("results/plots")
REPORT_PATH = Path("results/analysis_report.md")


@dataclass
class _MethodTrace:
    method: str
    model: str
    queries: list[str]
    preds: list[str]
    gold: list[str]


def _load_trace(method: str) -> _MethodTrace:
    path = PREDICTIONS_DIR / f"{method}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction trace for {method!r} at {path}")
    queries, preds, gold = [], [], []
    model = "-"
    with path.open() as f:
        for row in csv.DictReader(f):
            queries.append(row["query"])
            preds.append(row["predicted_label"])
            gold.append(row["gold_label"])
            model = row.get("model", "-")
    return _MethodTrace(method=method, model=model, queries=queries, preds=preds, gold=gold)


class ReportBuilder:
    """Chain analyses, then ``.build()`` writes the report.

    Keeps each analysis step independent and lets the caller pick subsets
    (e.g. skip plots in CI). Each method returns ``self`` for chaining.
    """

    def __init__(self, traces: list[_MethodTrace], eval_results: list[EvalResult]) -> None:
        self._traces = traces
        self._results = eval_results
        self._sections: list[str] = []
        self._per_class: dict[str, list[PerClassStats]] = {}

    def with_headline_table(self) -> Self:
        rows = ["| Method | Model | n | Top-1 | Macro-F1 | Top-3 | $/1k | p50 ms | p95 ms |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in self._results:
            cost_per_1k = (r.total_cost_usd / r.n * 1000) if r.n else 0.0
            rows.append(
                f"| {r.method} | {r.model} | {r.n} | {r.top1_accuracy:.3f} | "
                f"{r.macro_f1:.3f} | {r.top3_accuracy:.3f} | ${cost_per_1k:.2f} | "
                f"{r.p50_latency_ms:.0f} | {r.p95_latency_ms:.0f} |"
            )
        self._sections.append("## Headline\n\n" + "\n".join(rows))
        return self

    def with_significance_matrix(self) -> Self:
        lines = ["## Pairwise significance (McNemar + paired bootstrap)\n"]
        traces = self._traces
        for i, ta in enumerate(traces):
            for tb in traces[i + 1 :]:
                mc: SignificanceResult = mcnemar_test(ta.preds, tb.preds, ta.gold)
                ci = paired_bootstrap_accuracy_diff(ta.preds, tb.preds, ta.gold)
                star = "**" if mc.significant else ""
                lines.append(
                    f"- {star}{ta.method} vs {tb.method}{star}: "
                    f"acc Δ = {ci.point_estimate:+.3f} "
                    f"[{ci.lower:+.3f}, {ci.upper:+.3f}] (95% CI), "
                    f"McNemar p = {mc.p_value:.4f} "
                    f"(b={mc.discordant_a_only}, c={mc.discordant_b_only})"
                )
        self._sections.append("\n".join(lines))
        return self

    def with_per_class_analysis(self) -> Self:
        for t in self._traces:
            self._per_class[t.method] = per_class_f1_with_ci(t.preds, t.gold)

        lines = ["## Per-class analysis\n"]
        for method, stats in self._per_class.items():
            worst = hardest_classes(stats, k=5)
            lines.append(f"### {method} — 5 hardest classes")
            lines.append("| Label | Support | F1 | 95% CI |")
            lines.append("|---|---|---|---|")
            for s in worst:
                lines.append(
                    f"| {s.label} | {s.support} | {s.f1:.3f} | "
                    f"[{s.f1_lower:.3f}, {s.f1_upper:.3f}] |"
                )
            lines.append("")
        if len(self._per_class) >= 2:
            lines.append("### Rank correlation of per-class F1 between methods")
            lines.append("(High → methods agree on which classes are hard.)\n")
            for (a, b), rho in rank_correlation_between_methods(self._per_class).items():
                lines.append(f"- {a} vs {b}: ρ = {rho:.3f}")
        self._sections.append("\n".join(lines))
        return self

    def with_top_confusions(self) -> Self:
        lines = ["## Top confusions (gold → predicted) per method\n"]
        for t in self._traces:
            lines.append(f"### {t.method}")
            lines.append("| Gold | Predicted | Count |")
            lines.append("|---|---|---|")
            for gold_lab, pred_lab, n in confusion_pairs(t.preds, t.gold, top_n=10):
                lines.append(f"| {gold_lab} | {pred_lab} | {n} |")
            lines.append("")
        self._sections.append("\n".join(lines))
        return self

    def with_plots(self) -> Self:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        if not self._per_class:
            for t in self._traces:
                self._per_class[t.method] = per_class_f1_with_ci(t.preds, t.gold)

        boxplot_path = PLOTS_DIR / "per_class_f1_boxplot.png"
        plot_per_class_f1_boxplot(self._per_class, boxplot_path)

        scatter_path = PLOTS_DIR / "cost_vs_accuracy.png"
        plot_cost_vs_accuracy(self._results, scatter_path)

        if self._traces:
            labels = sorted(set(self._traces[0].gold))
            for t in self._traces:
                cm_path = PLOTS_DIR / f"confusion_{t.method}.png"
                plot_confusion_heatmap(t.preds, t.gold, labels, cm_path)

        self._sections.append(
            "## Plots\n\n"
            f"![cost vs accuracy](plots/{scatter_path.name})\n"
            f"![per-class F1 boxplot](plots/{boxplot_path.name})\n"
        )
        return self

    def build(self, out_path: Path = REPORT_PATH) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("# Banking77 Analysis Report\n\n" + "\n\n".join(self._sections))
        return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the analysis report")
    parser.add_argument(
        "--methods", default="auto", help="Comma-separated method names, or 'auto'."
    )
    args = parser.parse_args()

    if args.methods == "auto":
        method_names = sorted(p.stem for p in PREDICTIONS_DIR.glob("*.csv"))
    else:
        method_names = args.methods.split(",")
    if not method_names:
        raise SystemExit(f"No prediction traces found under {PREDICTIONS_DIR}")

    traces = [_load_trace(m) for m in method_names]
    # Synthesise an EvalResult per trace so the headline table has something to show
    # even when run independently of run_eval.py.
    from eval.metrics import summarise
    from data.banking77 import LabelledExample
    from methods.base import Prediction

    eval_results = []
    for t in traces:
        gold = [LabelledExample(text=q, label=g) for q, g in zip(t.queries, t.gold)]
        preds = [Prediction(query=q, predicted_label=p) for q, p in zip(t.queries, t.preds)]
        eval_results.append(summarise(t.method, t.model, preds, gold))

    out = (
        ReportBuilder(traces, eval_results)
        .with_headline_table()
        .with_significance_matrix()
        .with_per_class_analysis()
        .with_top_confusions()
        .with_plots()
        .build()
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
