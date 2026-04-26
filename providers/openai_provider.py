"""OpenAI adapter — wraps the ``openai`` SDK behind ``LLMProvider``."""
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
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = resp.choices[0].message.content or ""
        in_tok = resp.usage.prompt_tokens
        out_tok = resp.usage.completion_tokens
        return ProviderResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=price_completion(self.name, model, in_tok, out_tok),
            latency_ms=latency_ms,
            model=model,
        )