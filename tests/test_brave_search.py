"""Tests for vetinari.tools.brave_search_tool — Brave Search provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.tools.brave_search_tool import BraveSearchTool

_TEST_API_KEY = "test-key-for-unit-tests"


class TestBraveSearchTool:
    """Tests for the BraveSearchTool class."""

    def test_unavailable_without_api_key(self):
        """Tool should not be available without an API key."""
        with patch.dict("os.environ", {}, clear=True):
            tool = BraveSearchTool(api_key="")
            assert not tool.is_available

    def test_unavailable_without_client(self):
        """Tool reports unavailability when client package is missing."""
        with patch("vetinari.tools.brave_search_tool._BRAVE_AVAILABLE", False):
            tool = BraveSearchTool(api_key=_TEST_API_KEY)
            assert not tool.is_available

    def test_search_raises_when_unavailable(self):
        """search() should raise RuntimeError when client is not available."""
        with patch("vetinari.tools.brave_search_tool._BRAVE_AVAILABLE", False):
            tool = BraveSearchTool(api_key="")
            with pytest.raises(RuntimeError, match="not available"):
                tool.search("test query")

    def test_parse_results_dict_format(self):
        """_parse_results should handle dict-style API responses."""
        tool = BraveSearchTool.__new__(BraveSearchTool)
        tool._api_key = _TEST_API_KEY
        tool._country = "US"
        tool._language = "en"
        tool._client = None

        raw = {
            "web": {
                "results": [
                    {
                        "title": "Python Docs",
                        "url": "https://docs.python.org",
                        "description": "Official Python documentation",
                    },
                    {
                        "title": "Real Python",
                        "url": "https://realpython.com",
                        "description": "Python tutorials",
                        "published_at": "2025-01-15",
                    },
                ]
            }
        }
        results = tool._parse_results(raw, "python docs")
        assert len(results) == 2
        assert results[0].title == "Python Docs"
        assert results[0].url == "https://docs.python.org"
        assert results[0].query_used == "python docs"
        assert results[0].source_reliability == 0.7
        assert results[1].published_at == "2025-01-15"

    def test_parse_results_empty(self):
        """_parse_results should handle empty responses gracefully."""
        tool = BraveSearchTool.__new__(BraveSearchTool)
        tool._api_key = _TEST_API_KEY
        tool._country = "US"
        tool._language = "en"
        tool._client = None

        assert tool._parse_results({}, "query") == []
        assert tool._parse_results({"web": {}}, "query") == []
        assert tool._parse_results({"web": {"results": []}}, "query") == []

    def test_get_stats(self):
        """get_stats should return configuration info."""
        with patch("vetinari.tools.brave_search_tool._BRAVE_AVAILABLE", False):
            tool = BraveSearchTool(api_key="")
            stats = tool.get_stats()
            assert stats["available"] is False
            assert stats["client_installed"] is False
            assert stats["api_key_set"] is False
            assert stats["country"] == "US"

    def test_result_format_matches_websearchtool(self):
        """Results should use the same SearchResult format as WebSearchTool."""
        tool = BraveSearchTool.__new__(BraveSearchTool)
        tool._api_key = _TEST_API_KEY
        tool._country = "US"
        tool._language = "en"
        tool._client = None

        raw = {
            "web": {
                "results": [
                    {"title": "Test", "url": "https://example.com", "description": "Desc"},
                ]
            }
        }
        results = tool._parse_results(raw, "test")
        result = results[0]
        # Verify SearchResult fields exist
        assert hasattr(result, "title")
        assert hasattr(result, "url")
        assert hasattr(result, "snippet")
        assert hasattr(result, "source_reliability")
        assert hasattr(result, "source_type")
        assert hasattr(result, "query_used")


class TestBraveSearchImports:
    """Test that Brave Search tool is properly wired."""

    def test_import_from_tools_package(self):
        """BraveSearchTool should be importable from vetinari.tools."""
        from vetinari.tools import BraveSearchTool as BSTool

        assert BSTool is BraveSearchTool

    def test_direct_import(self):
        """Direct import should work."""
        from vetinari.tools.brave_search_tool import BraveSearchTool

        assert BraveSearchTool is not None
        assert callable(BraveSearchTool)
