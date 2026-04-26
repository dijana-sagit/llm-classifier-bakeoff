"""Cached completions via the Decorator pattern.

``@disk_cache`` keys on the function arguments — but ``LLMProvider``
instances aren't hashable in a stable way (their SDK client is opaque).
We split the cache: the *payload* (text + tokens + cost) is cached; the
*latency* is recomputed on every call so cache hits show their true
sub-millisecond timing instead of replaying the original API latency.

Usage::

    from providers.cached import cached_complete
    response = cached_complete(
        provider_name="anthropic",
        model="claude-haiku-4-5",
        system=system_prompt,
        user=query,
        cache_system=True,
    )
"""
from __future__ import annotations

import time
from typing import TypedDict

from infra.cache import disk_cache
from infra.registry import PROVIDER_REGISTRY
from providers.base import LLMProvider, ProviderResponse

_singletons: dict[str, LLMProvider] = {}


def _get_provider(name: str) -> LLMProvider:
    if name not in _singletons:
        if name not in PROVIDER_REGISTRY:
            raise KeyError(
                f"Unknown provider {name!r}. Registered: {sorted(PROVIDER_REGISTRY)}"
            )
        _singletons[name] = PROVIDER_REGISTRY[name]()
    return _singletons[name]


class _CachedPayload(TypedDict):
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str


@disk_cache
def _cached_payload(
    *,
    provider_name: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    cache_system: bool,
) -> _CachedPayload:
    response = _get_provider(provider_name).complete(
        model=model,
        system=system,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
        cache_system=cache_system,
    )
    return _CachedPayload(
        text=response.text,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
        model=response.model,
    )


def cached_complete(
    *,
    provider_name: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 64,
    temperature: float = 0.0,
    cache_system: bool = False,
) -> ProviderResponse:
    """Disk-cached LLM completion. Latency is measured fresh each call."""
    t0 = time.perf_counter()
    payload = _cached_payload(
        provider_name=provider_name,
        model=model,
        system=system,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
        cache_system=cache_system,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return ProviderResponse(latency_ms=latency_ms, **payload)
