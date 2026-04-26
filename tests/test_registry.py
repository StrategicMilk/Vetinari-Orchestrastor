"""Tests for vetinari/utils/registry.py — BaseRegistry utility."""

from __future__ import annotations

from vetinari.utils.registry import BaseRegistry


class TestBaseRegistry:
    def test_register_and_get(self):
        reg: BaseRegistry[str, int] = BaseRegistry()
        reg.register("a", 1)
        assert reg.get("a") == 1

    def test_get_missing_returns_none(self):
        reg: BaseRegistry[str, int] = BaseRegistry()
        assert reg.get("missing") is None

    def test_unregister(self):
        reg: BaseRegistry[str, int] = BaseRegistry()
        reg.register("a", 1)
        removed = reg.unregister("a")
        assert removed == 1
        assert reg.get("a") is None

    def test_unregister_missing_returns_none(self):
        reg: BaseRegistry[str, int] = BaseRegistry()
        assert reg.unregister("missing") is None

    def test_list_all(self):
        reg: BaseRegistry[str, str] = BaseRegistry()
        reg.register("x", "hello")
        reg.register("y", "world")
        assert sorted(reg.list_all()) == ["hello", "world"]

    def test_list_keys(self):
        reg: BaseRegistry[str, int] = BaseRegistry()
        reg.register("a", 1)
        reg.register("b", 2)
        assert sorted(reg.list_keys()) == ["a", "b"]

    def test_clear(self):
        reg: BaseRegistry[str, int] = BaseRegistry()
        reg.register("a", 1)
        reg.clear()
        assert len(reg) == 0

    def test_len(self):
        reg: BaseRegistry[str, int] = BaseRegistry()
        assert len(reg) == 0
        reg.register("a", 1)
        assert len(reg) == 1

    def test_contains(self):
        reg: BaseRegistry[str, int] = BaseRegistry()
        reg.register("a", 1)
        assert "a" in reg
        assert "b" not in reg

    def test_repr(self):
        reg: BaseRegistry[str, int] = BaseRegistry()
        reg.register("a", 1)
        assert "BaseRegistry" in repr(reg)
        assert "1 items" in repr(reg)

    def test_replace_on_register(self):
        reg: BaseRegistry[str, int] = BaseRegistry()
        reg.register("a", 1)
        reg.register("a", 2)
        assert reg.get("a") == 2
        assert len(reg) == 1
