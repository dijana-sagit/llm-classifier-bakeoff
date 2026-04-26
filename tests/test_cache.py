"""Decorator-pattern test for the disk cache."""
from __future__ import annotations

import infra.cache as cache_mod
from infra.cache import disk_cache


def test_disk_cache_is_called_once_per_unique_arg(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cache_mod, "_cache", None)

    call_count = {"n": 0}

    @disk_cache
    def expensive(x: int) -> int:
        call_count["n"] += 1
        return x * 2

    assert expensive(3) == 6
    assert expensive(3) == 6  # cached
    assert call_count["n"] == 1

    assert expensive(4) == 8  # different arg → recomputed
    assert call_count["n"] == 2