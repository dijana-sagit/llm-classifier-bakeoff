"""OpenAI provider — wraps the ``openai`` SDK behind ``LLMProvider``."""
from __future__ import annotations

import time

import openai

from infra.pricing import price_completion
from infra.registry import register_provider
from providers.base import ProviderResponse


@register_provider("openai")
class OpenAIProvider:
    name = "openai"

    def __init__(self) -> None:
        self._client = openai.OpenAI()

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        cache_system: bool = False,  # OpenAI auto-caches >1024-token prefixes
    ) -> ProviderResponse:
        # GPT-5 family quirks: only supports temperature=1 (so we omit the
        # parameter), uses chain-of-thought reasoning by default which eats
        # the output-token budget — keep it minimal for classification.
        is_gpt5 = model.startswith("gpt-5")
        params: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": max_tokens,
        }
        if is_gpt5:
            params["reasoning_effort"] = "minimal"
        else:
            params["temperature"] = temperature

        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(**params)
        latency_ms = (time.perf_counter() - t0) * 1000
        text = resp.choices[0].message.content or ""
        in_tok = resp.usage.prompt_tokens
        out_tok = resp.usage.completion_tokens
        details = getattr(resp.usage, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        return ProviderResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=price_completion(
                self.name,
                model,
                in_tok - cached,
                out_tok,
                cached_read_tokens=cached,
            ),
            latency_ms=latency_ms,
            model=model,
        )