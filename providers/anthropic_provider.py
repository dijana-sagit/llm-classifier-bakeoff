"""Anthropic adapter — wraps the ``anthropic`` SDK behind ``LLMProvider``.

Supports prompt caching via ``cache_control`` on the system block. For a
zero-shot classifier where the same 77-label system prompt is reused every
call, caching cuts input cost ~10× after the first request.
"""
from __future__ import annotations

import time

import anthropic

from infra.pricing import price_completion
from infra.registry import register_provider
from providers.base import ProviderResponse


@register_provider("anthropic")
class AnthropicProvider:
    name = "anthropic"

    def __init__(self) -> None:
        self._client = anthropic.Anthropic()

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        cache_system: bool = False,
    ) -> ProviderResponse:
        if cache_system:
            system_param: str | list[dict] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            system_param = system

        t0 = time.perf_counter()
        resp = self._client.messages.create(
            model=model,
            system=system_param,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = resp.content[0].text if resp.content else ""
        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        return ProviderResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=price_completion(self.name, model, in_tok, out_tok),
            latency_ms=latency_ms,
            model=model,
        )
