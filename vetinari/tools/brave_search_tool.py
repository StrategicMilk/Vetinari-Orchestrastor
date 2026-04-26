"""Brave Search Tool for Vetinari.

Provides web search capabilities via the Brave Search API, complementing
the existing DuckDuckGo and SearXNG backends in WebSearchTool.

Requires the ``brave-search-python-client`` package and a valid API key
in the ``BRAVE_SEARCH_API_KEY`` environment variable.  When either is
unavailable, all methods raise informative errors rather than silently
returning empty results.

Usage::

    from vetinari.tools.brave_search_tool import BraveSearchTool

    tool = BraveSearchTool()
    results = tool.search("Python async best practices", max_results=5)
    for r in results:
        logger.debug("%s — %s", r.title, r.url)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from vetinari.constants import MAX_RETRIES, WEB_SEARCH_SHORT_TIMEOUT
from vetinari.tools.web_search_types import SearchResult

logger = logging.getLogger(__name__)

# Attempt to import the Brave Search client
try:
    from brave_search import BraveSearch as _BraveSearchClient

    _BRAVE_AVAILABLE = True
except ImportError:
    _BraveSearchClient = None  # type: ignore[assignment,misc]
    _BRAVE_AVAILABLE = False


# ── Constants ────────────────────────────────────────────────────────

_DEFAULT_MAX_RESULTS = 10
_DEFAULT_COUNTRY = "US"
_DEFAULT_LANGUAGE = "en"
_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
_NEWS_SEARCH_URL = "https://api.search.brave.com/res/v1/news/search"

# Brave Search tends to surface high-quality results for technical queries
_BASE_RELIABILITY_SCORE = 0.7


def _redact_query(query: str) -> str:
    return f"<redacted:{len(query)} chars>"


# ── BraveSearchTool ──────────────────────────────────────────────────


class BraveSearchTool:
    """Web search via the Brave Search API.

    Wraps the ``brave-search-python-client`` library to provide search
    results in the same ``SearchResult`` format used by ``WebSearchTool``.

    Args:
        api_key: Brave Search API key.  Defaults to the
            ``BRAVE_SEARCH_API_KEY`` environment variable.
        country: Country code for search localisation.
        language: Language code for results.
    """

    def __init__(
        self,
        api_key: str | None = None,
        country: str = _DEFAULT_COUNTRY,
        language: str = _DEFAULT_LANGUAGE,
    ) -> None:
        self._api_key = api_key or os.environ.get("BRAVE_SEARCH_API_KEY", "")
        self._country = country
        self._language = language
        self._client: Any = None

        if _BRAVE_AVAILABLE and self._api_key:
            try:
                self._client = _BraveSearchClient(api_key=self._api_key)
                logger.info("BraveSearchTool: initialized with API key")
            except Exception as exc:
                logger.warning("BraveSearchTool: client init failed: %s", exc)
        elif not _BRAVE_AVAILABLE:
            logger.debug(
                "BraveSearchTool: brave-search-python-client not installed; "
                "install with: pip install 'brave-search-python-client>=0.4.0'",  # noqa: VET301 — user guidance string
            )
        elif not self._api_key:
            logger.debug(
                "BraveSearchTool: no API key found; set BRAVE_SEARCH_API_KEY environment variable",
            )

    @property
    def is_available(self) -> bool:
        """Whether the Brave Search client is configured and ready.

        Returns:
            True if the client is initialised with a valid API key.
        """
        return self._client is not None

    def search(
        self,
        query: str,
        max_results: int = _DEFAULT_MAX_RESULTS,
    ) -> list[SearchResult]:
        """Execute a web search and return structured results.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of SearchResult objects with title, URL, snippet, and
            provenance metadata.

        Raises:
            RuntimeError: If the client is not available (missing package
                or API key).
        """
        if not self.is_available:
            reason = (
                "brave-search-python-client not installed" if not _BRAVE_AVAILABLE else "BRAVE_SEARCH_API_KEY not set"
            )
            raise RuntimeError(
                f"BraveSearchTool not available: {reason}. "
                f"Install: pip install 'brave-search-python-client>=0.4.0' "  # noqa: VET301 — user guidance string
                f"and set BRAVE_SEARCH_API_KEY.",
            )

        try:
            raw_results = self._request(_WEB_SEARCH_URL, query, max_results)
            return self._parse_results(raw_results, query)
        except Exception as exc:
            logger.warning("BraveSearchTool: search failed for %s: %s", _redact_query(query), exc)
            return []

    def search_news(
        self,
        query: str,
        max_results: int = _DEFAULT_MAX_RESULTS,
    ) -> list[SearchResult]:
        """Search for recent news articles.

        Args:
            query: The news search query.
            max_results: Maximum number of results.

        Returns:
            List of SearchResult objects from news sources.

        Raises:
            RuntimeError: If the client is not available.
        """
        if not self.is_available:
            raise RuntimeError("BraveSearchTool not available for news search")

        try:
            raw_results = self._request(_NEWS_SEARCH_URL, query, max_results)
            return self._parse_results(raw_results, query, source_type="news")
        except Exception as exc:
            logger.warning("BraveSearchTool: news search failed for %s: %s", _redact_query(query), exc)
            return []

    def _request(self, url: str, query: str, max_results: int) -> dict[str, Any]:
        """Call Brave's HTTPS API with bounded timeout and retry behavior."""
        import requests

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key,
        }
        params = {
            "q": query,
            "count": max(1, min(max_results, 20)),
            "country": self._country,
            "search_lang": self._language,
        }
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=WEB_SEARCH_SHORT_TIMEOUT,
                    allow_redirects=False,
                )
                resp.raise_for_status()
                data = resp.json()
                return data if isinstance(data, dict) else {}
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "BraveSearchTool: request attempt %d/%d failed for %s: %s",
                    attempt,
                    MAX_RETRIES,
                    _redact_query(query),
                    type(exc).__name__,
                )
        raise RuntimeError(f"Brave Search request failed after {MAX_RETRIES} attempts: {last_exc}")

    def _parse_results(
        self,
        raw: Any,
        query: str,
        source_type: str = "web",
    ) -> list[SearchResult]:
        """Parse raw Brave API response into SearchResult objects.

        Args:
            raw: Raw response from the Brave Search client.
            query: The original query (for provenance).
            source_type: Type of source (web, news).

        Returns:
            Parsed list of SearchResult objects.
        """
        results: list[SearchResult] = []

        # Brave API returns results in web.results or news.results
        items: list[dict[str, Any]] = []
        if isinstance(raw, dict):
            web_results = raw.get("web", {})
            if isinstance(web_results, dict):
                items = web_results.get("results", [])
            if not items:
                news_results = raw.get("news", {})
                if isinstance(news_results, dict):
                    items = news_results.get("results", [])
        elif hasattr(raw, "web_results"):
            # Object-style response from newer client versions
            items = getattr(raw, "web_results", []) or []

        for item in items:
            if isinstance(item, dict):
                title = item.get("title", "")
                url = item.get("url", "")
                snippet = item.get("description", "") or item.get("snippet", "")
                published = item.get("published_at") or item.get("age")
            else:
                # Object-style item
                title = getattr(item, "title", "")
                url = getattr(item, "url", "")
                snippet = getattr(item, "description", "") or getattr(item, "snippet", "")
                published = getattr(item, "published_at", None) or getattr(item, "age", None)

            if title and url:
                results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        published_at=str(published) if published else None,
                        source_reliability=_BASE_RELIABILITY_SCORE,
                        source_type=source_type,
                        query_used=query,
                    ),
                )

        return results

    def get_stats(self) -> dict[str, Any]:
        """Return tool status information.

        Returns:
            Dictionary with availability status and configuration.
        """
        return {
            "available": self.is_available,
            "client_installed": _BRAVE_AVAILABLE,
            "api_key_set": bool(self._api_key),
            "country": self._country,
            "language": self._language,
        }
