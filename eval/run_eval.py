"""CLI entry point: orchestrates method × provider × metrics.

Usage::

    python -m eval.run_eval --methods knn                       # no API cost
    python -m eval.run_eval --methods knn,zero_shot_haiku        # specific methods
    python -m eval.run_eval --methods all                        # full bake-off

Method *variants* (``zero_shot_haiku``, ``zero_shot_sonnet``, …) are
declared in ``METHOD_VARIANTS`` below — each entry is a thunk that
produces a fully-wired ``ClassificationMethod``. Adding a model is one line.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from data.banking77 import load_banking77
from eval.metrics import EvalResult, summarise
from infra.registry import METHOD_REGISTRY
from methods.base import ClassificationMethod

# Importing the modules registers them via @register_method decorators.
import methods.embedding_knn  # noqa: F401
import methods.retrieval_augmented  # noqa: F401
import methods.zero_shot  # noqa: F401
import providers.anthropic_provider  # noqa: F401
import providers.gemini_provider  # noqa: F401
import providers.openai_provider  # noqa: F401
from methods.embedding_knn import EmbeddingKNN
from methods.retrieval_augmented import RetrievalAugmentedLLM
from providers.anthropic_provider import AnthropicProvider
from providers.gemini_provider import GeminiProvider
from providers.openai_provider import OpenAIProvider

RESULTS_DIR = Path("results")
RESULTS_CSV = RESULTS_DIR / "results.csv"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


# Variant table: name → factory. The method is fixed (``ZeroShotLLM``);
# the configuration (provider × model) varies. One line per benchmarked
# model; eval CLI accepts these names directly.
METHOD_VARIANTS: dict[str, Callable[[], ClassificationMethod]] = {
    "knn": EmbeddingKNN,
    "retrieval_haiku": lambda: RetrievalAugmentedLLM(
        provider=AnthropicProvider(), model="claude-haiku-4-5"
    ),
    "retrieval_sonnet": lambda: RetrievalAugmentedLLM(
        provider=AnthropicProvider(), model="claude-sonnet-4-6"
    ),
    "retrieval_gpt5_mini": lambda: RetrievalAugmentedLLM(
        provider=OpenAIProvider(), model="gpt-5-mini"
    ),
    "retrieval_gemini_flash": lambda: RetrievalAugmentedLLM(
        provider=GeminiProvider(), model="gemini-2.5-flash"
    ),
}


def _build_method(name: str) -> ClassificationMethod:
    """Factory: instantiate a method variant by name."""
    if name not in METHOD_VARIANTS:
        raise SystemExit(
            f"Unknown variant {name!r}. Available: {sorted(METHOD_VARIANTS)}"
        )
    return METHOD_VARIANTS[name]()


def _write_results(results: list[EvalResult]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    with RESULTS_CSV.open("w", newline="") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_predictions(method_name: str, model: str, preds, gold) -> None:
    """Persist per-prediction traces for downstream significance testing."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"{method_name}.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["query", "predicted_label", "gold_label", "method", "model", "latency_ms", "cost_usd"]
        )
        for p, g in zip(preds, gold, strict=True):
            writer.writerow(
                [p.query, p.predicted_label, g.label, method_name, model, p.latency_ms, p.cost_usd]
            )


def _write_metadata(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "args": vars(args),
        "registered_methods": sorted(METHOD_REGISTRY),
        "registered_variants": sorted(METHOD_VARIANTS),
    }
    (RESULTS_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Banking77 LLM bake-off")
    parser.add_argument(
        "--methods",
        default="knn",
        help="Comma-separated variant names, or 'all'.",
    )
    parser.add_argument("--n-test", type=int, default=200, help="Test set size cap.")
    args = parser.parse_args()

    requested = (
        sorted(METHOD_VARIANTS) if args.methods == "all" else args.methods.split(",")
    )

    splits = load_banking77()
    test = splits.test[: args.n_test]

    results: list[EvalResult] = []
    for name in requested:
        method = _build_method(name)
        method.fit(splits.train)
        preds = [method.predict(ex.text) for ex in tqdm(test, desc=name)]
        model_label = getattr(method, "model", "-")
        _write_predictions(name, model_label, preds, test)
        results.append(summarise(name, model=model_label, preds=preds, gold=test))

    _write_results(results)
    _write_metadata(args)
    for r in results:
        print(
            f"{r.method:<22} acc={r.top1_accuracy:.3f}  "
            f"macroF1={r.macro_f1:.3f}  top3={r.top3_accuracy:.3f}  "
            f"cost=${r.total_cost_usd:.2f}  p50={r.p50_latency_ms:.0f}ms"
        )


if __name__ == "__main__":
    main()
