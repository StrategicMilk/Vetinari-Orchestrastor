"""Tests for the plan-tracking memory store."""

from __future__ import annotations

from vetinari.memory.plan_tracking import MemoryStore


def test_memory_store_del_safe_when_contextlib_unavailable(monkeypatch, tmp_path) -> None:
    """__del__ stays quiet during interpreter-style module teardown."""
    store = MemoryStore(db_path=str(tmp_path / "plan_memory.db"), use_json_fallback=True)

    monkeypatch.setattr("vetinari.memory.plan_tracking.contextlib", None)

    store.__del__()
    assert store.get_memory_stats()["storage_type"] == "json"
