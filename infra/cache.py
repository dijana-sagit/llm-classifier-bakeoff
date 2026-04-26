"""Decorator pattern: ``@disk_cache`` persists any callable's return value.

Used to cache LLM completions so re-running the eval is free. Cache key is
derived from the function name and a SHA-256 of the arguments — keep the
arguments serialisable (str / numbers / dataclasses).
"""
from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import diskcache

CACHE_DIR = Path("results/cache")
_cache: diskcache.Cache | None = None


def _get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache = diskcache.Cache(str(CACHE_DIR))
    return _cache


def _serialise(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def _key(func_name: str, args: tuple, kwargs: dict) -> str:
    payload = {
        "func": func_name,
        "args": [_serialise(a) for a in args],
        "kwargs": {k: _serialise(v) for k, v in sorted(kwargs.items())},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


F = TypeVar("F", bound=Callable[..., Any])


def disk_cache(func: F) -> F:
    """Cache ``func`` return values on disk, keyed by its arguments."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cache = _get_cache()
        k = _key(func.__qualname__, args, kwargs)
        if k in cache:
            return cache[k]
        result = func(*args, **kwargs)
        cache[k] = result
        return result

    return wrapper  # type: ignore[return-value]