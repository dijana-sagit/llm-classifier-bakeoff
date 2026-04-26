# Banking77 LLM Classifier Bake-Off

> **TL;DR (placeholder until results are in):** Comparing zero-shot LLMs, retrieval-augmented LLMs, few-shot LLMs, fine-tuned small models, and a plain embedding+kNN baseline on the 77-class Banking77 intent classification benchmark. Optimised for the actual production tradeoff: **accuracy vs. cost-per-prediction vs. latency.**

![results placeholder](results/plots/cost_vs_accuracy.png)

## Why this benchmark

Banking77 is 13k customer queries across 77 fine-grained banking intents — a realistic stand-in for production support-ticket triage. The 77-class size matters: you can't fit every label in a cheap prompt, which forces real engineering decisions (retrieval over candidates, prompt caching, fine-tune-vs-prompt) — exactly the decisions that come up in real deployments.

## Methods compared

| # | Method | Idea |
|---|--------|------|
| 1 | Embedding + kNN | Sentence-transformers + cosine similarity. Cheap baseline. |
| 2 | Zero-shot LLM | Dump all 77 intents in the prompt, ask LLM to pick. |
| 3 | Retrieval-augmented LLM | Embed query → top-10 candidates → LLM picks among 10. |
| 4 | Few-shot LLM | k=5 retrieved labelled examples in the prompt. |
| 5 | SetFit fine-tune | Sentence-transformer + classification head, ~minutes to train. |

LLM providers: **Claude Haiku 4.5, Claude Sonnet 4.6, Gemini 2.5 Flash, Gemini 2.5 Pro, GPT-5 mini, GPT-5.** Tier-2 models are evaluated on a stratified 500-example subset to keep total spend under $20.

## Results

_Filled in by `eval/run_eval.py`. Date-stamped; see `results/run_metadata.json` for exact model IDs and prices used._

| Method | Model | Top-1 Acc | Macro F1 | Top-3 Acc | $/1k preds | p50 latency | p95 latency |
|--------|-------|-----------|----------|-----------|------------|-------------|-------------|
| _TBD_  |       |           |          |           |            |             |             |

## Reproduce

```bash
uv sync
cp .env.example .env  # fill in API keys for LLM methods
python -m eval.run_eval --methods knn  # no API cost
python -m eval.run_eval --methods all  # ~$10-15 with caching
```

All LLM responses are cached on disk (`results/cache/`), so reruns are free.

## Key findings

_To be written after the eval runs._

1. _..._
2. _..._
3. _..._

## When to use which (the production recommendation)

_Opinionated paragraph after results are in. Don't sit on the fence._

## Limitations

- Banking77 is English-only and skewed to retail banking — generalisation to other domains untested.
- LLM prices and model quality drift; numbers reflect a snapshot, not a permanent ranking.
- Fine-tuned SetFit benefits from clean training data; production noise would degrade it more than the LLM methods.

## Code organisation

The code uses classic patterns deliberately, not decoratively — the file tree is the architecture diagram:

```
methods/         # Strategy: each ClassificationMethod is interchangeable
providers/       # Adapter: three LLM SDKs behind one Protocol
infra/cache.py   # Decorator: @disk_cache wraps any predict callable
infra/registry.py # Registry: @register("zero_shot") decorator → CLI by name
eval/            # Orchestration: composes methods × providers × metrics
analysis/        # Builder: ReportBuilder().with_*().build() — significance + plots
```

Method has-a Provider (composition); methods register themselves; providers conform to a Protocol; results are immutable dataclasses. `typing.Protocol` over ABC throughout.

## Statistical analysis (the depth past headline accuracy)

`eval/run_eval.py` writes per-prediction traces to `results/predictions/{method}.csv`. `analysis/reports.py` consumes them and produces `results/analysis_report.md` containing:

- **Pairwise McNemar tests** + **paired bootstrap CIs** on accuracy differences — does method A really beat method B, or is it within sampling noise?
- **Per-class F1 with bootstrap CIs** — uniform performance vs. methods with brittle long tails
- **Spearman rank correlation between methods on per-class F1** — do methods agree on which classes are hard?
- **Top confusions per method** — error-analysis seed list
- **Plots:** per-class F1 boxplots, cost-vs-accuracy scatter, confusion-matrix heatmaps

```bash
python -m eval.run_eval --methods knn,zero_shot   # writes traces
python -m analysis.reports                        # writes report + plots
```

## License

MIT.