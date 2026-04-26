"""Tests for ContextWindowManager paging behavior."""

from __future__ import annotations

from types import SimpleNamespace

from vetinari.context.window_manager import ContextWindowManager


class _Store:
    def search(self, query: str, limit: int = 5) -> list[SimpleNamespace]:
        return [SimpleNamespace(content="already injected"), SimpleNamespace(content="new snippet")]


def test_page_in_reports_only_newly_injected_snippets() -> None:
    manager = ContextWindowManager(auto_compress=False)
    manager.inject_context(["already injected"])

    injected = manager.page_in("query", memory_store=_Store())

    assert injected == ["new snippet"]
