"""Gemini provider — wraps ``google-genai`` behind ``LLMProvider``."""
from __future__ import annotations

import time

from google import genai
from google.genai import types

from infra.pricing import price_completion
from infra.registry import register_provider
from providers.base import ProviderResponse


@register_provider("gemini")
class GeminiProvider:
    name = "gemini"

    def __init__(self) -> None:
        self._client = genai.Client()

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        cache_system: bool = False,  # Gemini caching uses a separate API; ignored here
    ) -> ProviderResponse:
        t0 = time.perf_counter()
        resp = self._client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=temperature,
                # Gemini 2.5 enables "thinking" by default, which consumes the
                # output-token budget before any answer text is emitted. For a
                # single-label classification task we want the label, not a
                # reasoning trace.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text = resp.text or ""
        # Either token count can be None (e.g. empty / refused responses); the
        # pricing module multiplies by these, so coerce to int.
        in_tok = resp.usage_metadata.prompt_token_count or 0
        out_tok = resp.usage_metadata.candidates_token_count or 0
        return ProviderResponse(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=price_completion(self.name, model, in_tok, out_tok),
            latency_ms=latency_ms,
            model=model,
        )