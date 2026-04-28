# Banking77 LLM Classifier Bake-Off

> **TL;DR (preliminary, n=200 of 3,080):** Few-shot LLM with retrieved labelled examples leads at **94.5% top-1** (Gemini 2.5 Flash, k=5 demos + top-10 candidate labels), narrowly beating a plain embedding+kNN baseline at **93.0%**. Zero-shot LLM (all 77 labels in the prompt) is dominated and was dropped from the runner. SetFit reaches 89.0% but is the latency leader (28ms p50 — ~4× faster than kNN). Full 3,080-example numbers and significance tests pending.

![results placeholder](results/plots/cost_vs_accuracy.png)

## Why this benchmark

Banking77 is 13k customer queries across 77 fine-grained banking intents — a realistic stand-in for production support-ticket triage. The 77-class size matters: you can't fit every label in a cheap prompt, which forces real engineering decisions (retrieval over candidates, prompt caching, fine-tune-vs-prompt) — exactly the decisions that come up in real deployments.

## Methods compared

| # | Method | Idea | Status |
|---|--------|------|--------|
| 1 | Embedding + kNN | `bge-small-en-v1.5` + cosine similarity, k=7 majority vote. | ✅ |
| 2 | Zero-shot LLM | Dump all 77 intents in the prompt, ask LLM to pick. | Ruled out (dominated by retrieval-augmented; kept on disk for reference) |
| 3 | Retrieval-augmented LLM | Embed query → top-10 candidate labels → LLM picks among 10. | ✅ |
| 4 | Few-shot LLM | Top-5 retrieved labelled `(query, label)` pairs as in-context demos + top-10 candidate labels. | ✅ |
| 5 | SetFit fine-tune | `paraphrase-mpnet-base-v2` + contrastive fine-tune + sklearn linear head. ~7h on CPU; cached afterwards. | ✅ |

**LLM providers tested in this preliminary run:** GPT-5 mini, Gemini 2.5 Flash. Anthropic (Haiku 4.5, Sonnet 4.6) and frontier OpenAI/Gemini variants are wired up in `METHOD_VARIANTS` but unrun pending billing.

## Results

_Preliminary — 200 stratified test examples (of 3,080); only OpenAI gpt-5-mini and Gemini 2.5 Flash variants run. Latencies are wall-clock end-to-end including local embedding for the LLM methods. p95 captured only where available from the original (uncached) run. See `results/results.csv` for the canonical numbers and `results/run_metadata.json` for exact model IDs._

| Method | Model | Top-1 Acc | Macro F1 | Top-3 Acc | $/1k preds | p50 latency | p95 latency |
|--------|-------|-----------|----------|-----------|------------|-------------|-------------|
| **few_shot_gemini_flash** | gemini-2.5-flash | **0.945** | **0.920** | 0.985 | $0.106 | 552ms | 680ms |
| few_shot_gpt5_mini | gpt-5-mini | 0.935 | 0.918 | 0.985 | $0.094 | 1161ms | 2931ms |
| knn | bge-small-en-v1.5 | 0.930 | 0.900 | 0.975 | $0.00 | 122ms | — |
| setfit | paraphrase-mpnet-base-v2 | 0.890 | 0.878 | 0.985 | $0.00 | **28ms** | — |
| retrieval_gpt5_mini | gpt-5-mini | 0.835 | 0.775 | 0.985 | $0.045 | 957ms | — |
| retrieval_gemini_flash | gemini-2.5-flash | 0.835 | 0.792 | 0.985 | $0.057 | 512ms | — |

## Reproduce

```bash
uv sync
cp .env.example .env  # fill in API keys for LLM methods
python -m eval.run_eval --methods knn                                            # no API cost
python -m eval.run_eval --methods knn,setfit,few_shot_gemini_flash,few_shot_gpt5_mini   # ~$0.05 on n=200
python -m eval.run_eval --methods all                                            # full bake-off
```

Or with Docker (mounts `data/` and `results/` from the host so caches persist across runs):

```bash
cp .env.example .env  # fill in API keys
docker compose build
docker compose run --rm bakeoff python -m eval.run_eval --methods knn
docker compose run --rm bakeoff python -m eval.run_eval --methods all
docker compose run --rm bakeoff python -m analysis.reports
```

All LLM responses are cached on disk (`results/cache/`), so reruns are free.

## Key findings (preliminary)

1. **In-context demonstrations carried the LLM methods, not stronger models.** Adding 5 retrieved labelled examples to the prompt took both LLMs from 83.5% (worse than kNN) to 93.5–94.5% (beating it). Same encoder, same retrieval, same candidate label set — the only change was showing the model concrete query-to-label phrasings. Label names alone weren't enough signal, especially for near-synonyms like `top_up_by_card_charge` vs `topping_up_by_card`.
2. **Zero-shot with 77 labels is dominated on every axis.** GPT-5 mini at 79.0% top-1 cost more than retrieval-augmented (which got 83.5%) and was 14 points behind kNN. The naive prompt is the wrong shape: the embedding step pre-filtering to 5–10 candidates makes the LLM's job tractable.
3. **Retrieval recall on Banking77 saturates fast.** With `bge-small-en-v1.5`, recall@5 = 99.1% and recall@10 = 99.8% on the full test set — meaning retrieval is essentially never the bottleneck. The interesting variable is how well the disambiguation step (LLM, kNN vote, or SetFit head) handles 5-10 already-relevant candidates.

## When to use which (the production recommendation, preliminary)

For a 77-class intent triage system at this scale, the picture is:

- **kNN is the default.** 93.0% top-1, $0/prediction, 122ms p50, no API dependency, no rate limits, fully reproducible. If 93% is good enough — and for many production triage flows it is — there is no honest reason to reach for an LLM.
- **Few-shot LLM (Gemini 2.5 Flash) is the upgrade.** ~1.5pp accuracy gain over kNN for ~$0.10/1k predictions and ~5× the latency. Worth it if those points are revenue-relevant (e.g., misroutes are expensive) and the latency budget allows.
- **SetFit is the latency play.** 4× faster than kNN at the cost of 4 accuracy points. Pick this only if 28ms p50 actually matters — high-QPS frontends, on-device, etc. — otherwise kNN dominates it.
- **Retrieval-augmented LLM (label-only, no demos) and zero-shot are not on the Pareto frontier here.** Don't ship them.

The narrow-but-real margin between few-shot LLM and kNN means the right choice is sensitive to your cost-of-error and latency budget — neither is universally correct.

## Limitations

- Banking77 is English-only and skewed to retail banking — generalisation to other domains untested.
- LLM prices and model quality drift; numbers reflect a snapshot, not a permanent ranking.
- Fine-tuned SetFit benefits from clean training data; production noise would degrade it more than the LLM methods.

## Code organisation

```
methods/         # Each ClassificationMethod is interchangeable
providers/       # Three LLM SDKs behind one common interface
infra/cache.py   # @disk_cache wraps any predict callable
infra/registry.py # @register("zero_shot") → exposes the method by name to the CLI
eval/            # Composes methods × providers × metrics
analysis/        # ReportBuilder().with_*().build() — significance + plots
```

## Statistical analysis (the depth past headline accuracy)

`eval/run_eval.py` writes per-prediction traces to `results/predictions/{method}.csv`. `analysis/reports.py` consumes them and produces `results/analysis_report.md` containing:

- **Pairwise McNemar tests** + **paired bootstrap CIs** on accuracy differences — does method A really beat method B, or is it within sampling noise?
- **Per-class F1 with bootstrap CIs** — uniform performance vs. methods with brittle long tails
- **Spearman rank correlation between methods on per-class F1** — do methods agree on which classes are hard?
- **Top confusions per method** — error-analysis seed list
- **Plots:** per-class F1 boxplots, cost-vs-accuracy scatter, confusion-matrix heatmaps

```bash
python -m eval.run_eval --methods knn,few_shot_gemini_flash   # writes traces
python -m analysis.reports                                    # writes report + plots
```

## License

MIT.