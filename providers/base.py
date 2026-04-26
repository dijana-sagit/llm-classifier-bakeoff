"""Adapter interface for LLM providers.

Three providers (Anthropic, OpenAI, Google) ship three different SDKs with
three different request/response shapes. ``LLMProvider`` is the uniform
interface every method talks to; the concrete adapters in this package
translate to and from each SDK.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Provider-agnostic completion result."""

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    model: str


@runtime_checkable
class LLMProvider(Protocol):
    """Adapter Protocol: every concrete adapter exposes the same shape.

    ``cache_system=True`` is a hint that the system prompt is reused across
    many calls and should be cached on the provider side (e.g. Anthropic
    cache_control). Adapters that don't support it ignore the flag.
    """

    name: str

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        cache_system: bool = False,
    ) -> ProviderResponse: ...
