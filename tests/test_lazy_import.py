"""Tests for vetinari/utils/lazy_import.py — lazy import utility."""

from __future__ import annotations

import pytest

from vetinari.utils.lazy_import import lazy_import, require_import


class TestLazyImport:
    def test_existing_module(self):
        mod, available = lazy_import("os")
        assert available is True
        assert mod is not None
        assert hasattr(mod, "path")

    def test_missing_module(self):
        mod, available = lazy_import("nonexistent_module_xyz_123")
        assert available is False
        assert mod is None

    def test_broken_optional_module_is_bounded(self, monkeypatch):
        def _raise_runtime_error(*_args, **_kwargs):
            raise RuntimeError("broken optional dependency")

        monkeypatch.setattr("importlib.import_module", _raise_runtime_error)

        mod, available = lazy_import("broken_optional_dependency")

        assert available is False
        assert mod is not None
        assert not mod

    def test_nested_missing_dependency_is_distinguishable_from_missing_module(self, monkeypatch):
        err = ModuleNotFoundError("No module named 'nested_dep'")
        err.name = "nested_dep"

        def _raise_nested_missing(*_args, **_kwargs):
            raise err

        monkeypatch.setattr("importlib.import_module", _raise_nested_missing)

        mod, available = lazy_import("package_that_exists_but_breaks")

        assert available is False
        assert mod is not None
        assert not mod

    def test_submodule(self):
        mod, available = lazy_import("os.path")
        assert available is True
        assert hasattr(mod, "join")


class TestRequireImport:
    def test_existing_module(self):
        mod = require_import("os")
        assert hasattr(mod, "path")

    def test_missing_module_raises(self):
        with pytest.raises(ImportError, match="nonexistent_module"):
            require_import("nonexistent_module_xyz_123")

    def test_missing_module_with_feature(self):
        with pytest.raises(ImportError, match="for LLM scanning"):
            require_import("nonexistent_module_xyz_123", feature="LLM scanning")
