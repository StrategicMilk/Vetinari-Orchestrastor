"""Tests for vetinari.context.tool_persistence.ToolResultStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from vetinari.context.tool_persistence import LARGE_RESULT_THRESHOLD, PREVIEW_SIZE, ToolResultStore


class TestSmallContent:
    """Content at or below the threshold is returned as-is without disk writes."""

    def test_small_content_not_persisted(self, tmp_path: Path) -> None:
        """Content <= LARGE_RESULT_THRESHOLD returns (content, '') without writing to disk."""
        store = ToolResultStore(cache_dir=tmp_path / "tool_results")
        content = "x" * LARGE_RESULT_THRESHOLD
        preview, key = store.store(content)

        assert preview == content
        assert key == ""
        assert not any(tmp_path.rglob("*.txt"))

    def test_empty_content_not_persisted(self, tmp_path: Path) -> None:
        """Empty string returns (empty, '') without writing to disk."""
        store = ToolResultStore(cache_dir=tmp_path / "tool_results")
        preview, key = store.store("")

        assert preview == ""
        assert key == ""

    def test_exactly_threshold_not_persisted(self, tmp_path: Path) -> None:
        """Content exactly at the threshold boundary is NOT persisted (threshold is exclusive)."""
        store = ToolResultStore(cache_dir=tmp_path / "tool_results")
        content = "a" * LARGE_RESULT_THRESHOLD
        preview, key = store.store(content)

        assert key == ""
        assert preview == content


class TestLargeContent:
    """Content above the threshold is offloaded to disk with a preview returned."""

    def test_large_content_persisted_with_preview(self, tmp_path: Path) -> None:
        """Content > LARGE_RESULT_THRESHOLD returns truncated preview + non-empty key."""
        store = ToolResultStore(cache_dir=tmp_path / "tool_results")
        content = "y" * (LARGE_RESULT_THRESHOLD + 1000)
        preview, key = store.store(content, tool_name="grep")

        assert key != ""
        assert len(preview) <= PREVIEW_SIZE + 200  # preview + truncation notice
        assert preview.startswith("y" * min(PREVIEW_SIZE, len(content)))
        assert "truncated" in preview

    def test_large_content_writes_file(self, tmp_path: Path) -> None:
        """Persisted content creates exactly one file in cache_dir."""
        cache_dir = tmp_path / "tool_results"
        store = ToolResultStore(cache_dir=cache_dir)
        content = "z" * (LARGE_RESULT_THRESHOLD + 500)
        _, key = store.store(content, tool_name="search")

        files = list(cache_dir.glob("*.txt"))
        assert len(files) == 1
        assert key in files[0].name

    def test_tool_name_in_filename(self, tmp_path: Path) -> None:
        """When tool_name is given, the filename contains it for readability."""
        cache_dir = tmp_path / "tool_results"
        store = ToolResultStore(cache_dir=cache_dir)
        content = "q" * (LARGE_RESULT_THRESHOLD + 100)
        _, _ = store.store(content, tool_name="my_tool")

        files = list(cache_dir.glob("*.txt"))
        assert len(files) == 1
        assert "my_tool" in files[0].name


class TestLoadRoundtrip:
    """store() then load() recovers the original full content."""

    def test_load_roundtrip(self, tmp_path: Path) -> None:
        """store() then load(key) returns the original content."""
        store = ToolResultStore(cache_dir=tmp_path / "tool_results")
        original = ("hello world " * 5000)[: LARGE_RESULT_THRESHOLD + 100]
        _, key = store.store(original, tool_name="roundtrip")

        recovered = store.load(key)
        assert recovered == original

    def test_load_empty_key_returns_none(self, tmp_path: Path) -> None:
        """load('') returns None without raising."""
        store = ToolResultStore(cache_dir=tmp_path / "tool_results")
        assert store.load("") is None

    def test_load_unknown_key_returns_none(self, tmp_path: Path) -> None:
        """load() with a key that has no matching file returns None."""
        store = ToolResultStore(cache_dir=tmp_path / "tool_results")
        assert store.load("nonexistentkey12") is None

    def test_load_missing_cache_dir_returns_none(self, tmp_path: Path) -> None:
        """load() when cache_dir doesn't exist yet returns None gracefully."""
        store = ToolResultStore(cache_dir=tmp_path / "missing_dir")
        assert store.load("somekey") is None


class TestTraversalRejection:
    """store() rejects tool_name values that contain path-traversal characters."""

    def test_tool_result_store_rejects_slash_in_tool_name(self, tmp_path: Path) -> None:
        """store() raises ValueError when tool_name contains a forward slash."""
        store = ToolResultStore(cache_dir=tmp_path / "tool_results")
        content = "x" * (LARGE_RESULT_THRESHOLD + 100)

        with pytest.raises(ValueError, match="forbidden"):
            store.store(content, tool_name="../../etc/passwd")

    def test_tool_result_store_rejects_backslash_in_tool_name(self, tmp_path: Path) -> None:
        """store() raises ValueError when tool_name contains a backslash."""
        store = ToolResultStore(cache_dir=tmp_path / "tool_results")
        content = "x" * (LARGE_RESULT_THRESHOLD + 100)

        with pytest.raises(ValueError, match="forbidden"):
            store.store(content, tool_name="..\\..\\evil")

    def test_tool_name_without_traversal_is_accepted(self, tmp_path: Path) -> None:
        """store() accepts a normal tool_name with no special characters."""
        store = ToolResultStore(cache_dir=tmp_path / "tool_results")
        content = "y" * (LARGE_RESULT_THRESHOLD + 100)

        preview, key = store.store(content, tool_name="safe_tool_name")
        assert key != ""
