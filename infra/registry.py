"""Decorator-based string→class lookup.

Lets the CLI accept ``--methods knn,zero_shot`` and the eval loop instantiate
the right method without a giant if/elif.
"""
from __future__ import annotations

from typing import TypeVar

T = TypeVar("T", bound=type)

METHOD_REGISTRY: dict[str, type] = {}
PROVIDER_REGISTRY: dict[str, type] = {}


def _register(target: dict[str, type], name: str):
    def decorator(cls: T) -> T:
        if name in target:
            raise ValueError(f"{name!r} already registered as {target[name].__name__}")
        target[name] = cls
        return cls

    return decorator


def register_method(name: str):
    """Register a ``ClassificationMethod`` under ``name``."""
    return _register(METHOD_REGISTRY, name)


def register_provider(name: str):
    """Register an ``LLMProvider`` under ``name``."""
    return _register(PROVIDER_REGISTRY, name)