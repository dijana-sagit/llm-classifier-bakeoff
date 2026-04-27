"""Smoke tests for the method/provider registry."""
from __future__ import annotations

import pytest

from infra.registry import METHOD_REGISTRY, register_method


def test_method_can_be_registered_and_retrieved():
    @register_method("dummy_test_method")
    class Dummy:
        name = "dummy_test_method"

    try:
        assert METHOD_REGISTRY["dummy_test_method"] is Dummy
    finally:
        METHOD_REGISTRY.pop("dummy_test_method", None)


def test_duplicate_registration_raises():
    @register_method("dummy_dup")
    class A:
        name = "dummy_dup"

    try:
        with pytest.raises(ValueError, match="already registered"):

            @register_method("dummy_dup")
            class B:
                name = "dummy_dup"
    finally:
        METHOD_REGISTRY.pop("dummy_dup", None)


def test_knn_is_registered_on_import():
    import methods.embedding_knn  # noqa: F401

    assert "knn" in METHOD_REGISTRY