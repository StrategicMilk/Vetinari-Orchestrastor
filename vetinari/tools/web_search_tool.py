"""
Web Search Tool for Vetinari

Provides comprehensive web search capabilities with provenance tracking,
source credibility scoring, and citation support.

Supports multiple backends:
- DuckDuckGo (free, no API key required)
- SerpAPI (Google)
- Tavily (AI-optimized search)
- Wikipedia (direct API)
- arXiv (academic papers)
- Custom SearxNG instance
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

logger = logging.getLogger(__name__)


class SearchBackend(Enum):
    DUCKDUCKGO = "duckduckgo"
    SERPAPI = "serpapi"
    TAVILY = "tavily"
    WIKIPEDIA = "wikipedia"
    ARXIV = "arxiv"
    SEARXNG = "searxng"
    LOCAL = "local"  # Offline/local knowledge base


@dataclass
class SearchResult:
    """Single search result with provenance tracking."""
    title: str
    url: str
    snippet: str
    published_at: Optional[str] = None
    source_reliability: float = 0.5  # 0.0 - 1.0
    source_type: str = "web"
    query_used: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "published_at": self.published_at,
            "source_reliability": self.source_reliability,
            "source_type": self.source_type,
            "query_used": self.query_used,
            "timestamp": self.timestamp
        }


@dataclass
class SearchResponse:
    """Complete search response with metadata."""
    results: List[SearchResult]
    query: str
    backend: str
    total_results: int
    execution_time_ms: int
    citations: List[str] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "query": self.query,
            "backend": self.backend,
            "total_results": self.total_results,
            "execution_time_ms": self.execution_time_ms,
            "citations": self.citations,
            "provenance": self.provenance
        }
    
    def get_citations(self) -> List[str]:
        """Get formatted citations for all results."""
        citations = []
        for i, r in enumerate(self.results, 1):
            citations.append(f"[{i}] {r.title}. {r.url}")
        return citations


class SourceCredibility:
    """Source credibility scoring for search results."""
    
    # Credibility scores by domain type
    DOMAIN_SCORES = {
        # High credibility
        "arxiv.org": 0.95,
        "wikipedia.org": 0.9,
        "github.com": 0.85,
        "stackoverflow.com": 0.8,
        "medium.com": 0.7,
        "docs.python.org": 0.95,
        "docs.ruby-lang.org": 0.95,
        "developer.mozilla.org": 0.95,
        # Government/Academic
        ".edu": 0.9,
        ".gov": 0.95,
        ".org": 0.7,
        # Lower credibility
        ".com": 0.5,
        ".io": 0.5,
    }
    
    @classmethod
    def score_url(cls, url: str) -> float:
        """Score a URL based on domain credibility."""
        url_lower = url.lower()
        
        # Check exact matches first
        for domain, score in cls.DOMAIN_SCORES.items():
            if domain.startswith("."):
                if url_lower.endswith(domain):
                    return score
            elif domain in url_lower:
                return score
        
        return 0.5  # Default score


class WebSearchTool:
    """
    Comprehensive web search tool with multi-backend support.
    
    Features:
    - Multiple search backends
    - Source credibility scoring
    - Provenance tracking
    - Citation generation
    - Rate limiting and caching
    """
    
    def __init__(self, 
                 backend: str = None,
                 serapi_key: str = None,
                 tavily_key: str = None,
                 searxng_url: str = None,
                 cache_ttl: int = 3600):
        """
        Initialize web search tool.
        
        Args:
            backend: Search backend to use (default: duckduckgo)
            serapi_key: SerpAPI key for Google search
            tavily_key: Tavily API key
            searxng_url: Custom SearxNG instance URL
            cache_ttl: Cache TTL in seconds
        """
        self.backend_name = backend or os.environ.get("VETINARI_SEARCH_BACKEND", "duckduckgo")
        self.serapi_key = serapi_key or os.environ.get("SERPAPI_KEY", "")
        self.tavily_key = tavily_key or os.environ.get("TAVILY_API_KEY", "")
        self.searxng_url = searxng_url or os.environ.get("SEARXNG_URL", "")
        self.cache_ttl = cache_ttl
        
        # Simple in-memory cache
        self._cache: Dict[str, tuple] = {}
        
        # Rate limiting
        self._request_times: List[float] = []
        self._min_request_interval = 1.0  # seconds
        
        # Setup backend
        self._setup_backend()
        
        logger.info(f"WebSearchTool initialized with backend: {self.backend_name}")
    
    def _setup_backend(self):
        """Setup the search backend."""
        if self.backend_name == SearchBackend.DUCKDUCKGO.value:
            self._search_func = self._search_duckduckgo
        elif self.backend_name == SearchBackend.SERPAPI.value:
            self._search_func = self._search_serpapi
        elif self.backend_name == SearchBackend.TAVILY.value:
            self._search_func = self._search_tavily
        elif self.backend_name == SearchBackend.WIKIPEDIA.value:
            self._search_func = self._search_wikipedia
        elif self.backend_name == SearchBackend.ARXIV.value:
            self._search_func = self._search_arxiv
        elif self.backend_name == SearchBackend.SEARXNG.value:
            self._search_func = self._search_searxng
        else:
            logger.warning(f"Unknown backend {self.backend_name}, using DuckDuckGo")
            self._search_func = self._search_duckduckgo
    
    def search(self, 
               query: str, 
               max_results: int = 5,
               language: str = "en",
               time_range: str = None) -> SearchResponse:
        """
        Perform a web search.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            language: Language code
            time_range: Time range (day, week, month, year)
            
        Returns:
            SearchResponse with results and metadata
        """
        start_time = time.time()
        
        # Check cache
        cache_key = f"{query}:{max_results}:{language}:{time_range}"
        if cache_key in self._cache:
            cached_time, cached_response = self._cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                logger.debug(f"Cache hit for query: {query}")
                return cached_response
        
        # Rate limiting
        self._apply_rate_limit()
        
        # Perform search
        try:
            results = self._search_func(query, max_results, language, time_range)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            results = []
        
        # Build response
        response = SearchResponse(
            results=results,
            query=query,
            backend=self.backend_name,
            total_results=len(results),
            execution_time_ms=int((time.time() - start_time) * 1000),
            citations=[f"[{i+1}] {r.title}. {r.url}" for i, r in enumerate(results[:max_results])] if results else [],
            provenance=[{
                "backend": self.backend_name,
                "timestamp": datetime.now().isoformat(),
                "query": query
            }]
        )
        
        # Update cache
        self._cache[cache_key] = (time.time(), response)
        
        return response
    
    def _apply_rate_limit(self):
        """Apply rate limiting between requests."""
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        if self._request_times:
            time_since_last = now - self._request_times[-1]
            if time_since_last < self._min_request_interval:
                sleep_time = self._min_request_interval - time_since_last
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        self._request_times.append(now)
    
    def _search_duckduckgo(self, query: str, max_results: int, 
                           language: str, time_range: str) -> List[SearchResult]:
        """Search using DuckDuckGo (via ddg library or HTTP)."""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                for r in ddgs.search(query, max_results=max_results, region=language):
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                        source_reliability=SourceCredibility.score_url(r.get("href", "")),
                        source_type="web",
                        query_used=query
                    ))
            return results
        except ImportError:
            logger.warning("duckduckgo_search not installed, trying alternative method")
            return self._search_duckduckgo_http(query, max_results, language)
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    def _search_duckduckgo_http(self, query: str, max_results: int,
                                 language: str) -> List[SearchResult]:
        """Fallback DuckDuckGo search using HTTP requests."""
        try:
            import requests
            
            url = "https://html.duckduckgo.com/html/"
            data = {"q": query, "b": ""}
            
            resp = requests.post(url, data=data, timeout=10)
            resp.raise_for_status()
            
            results = []
            # Simple regex parsing (not ideal but works for basic needs)
            import re
            pattern = r'<a class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>.*?<a class="result__snippet"[^>]*>([^<]+)</a>'
            
            for match in re.findall(pattern, resp.text, re.DOTALL)[:max_results]:
                url, title, snippet = match
                results.append(SearchResult(
                    title=title.strip(),
                    url=url,
                    snippet=snippet.strip(),
                    source_reliability=SourceCredibility.score_url(url),
                    source_type="web",
                    query_used=query
                ))
            
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo HTTP search failed: {e}")
            return []
    
    def _search_serpapi(self, query: str, max_results: int,
                       language: str, time_range: str) -> List[SearchResult]:
        """Search using SerpAPI (Google)."""
        if not self.serapi_key:
            logger.warning("SerpAPI key not set, falling back to DuckDuckGo")
            return self._search_duckduckgo(query, max_results, language, time_range)
        
        try:
            import requests
            
            params = {
                "q": query,
                "api_key": self.serapi_key,
                "num": max_results,
                "engine": "google"
            }
            if time_range:
                params["tbs"] = f"qdr:{time_range}"
            
            resp = requests.get("https://serpapi.com/search", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            results = []
            for item in data.get("organic_results", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    published_at=item.get("date"),
                    source_reliability=SourceCredibility.score_url(item.get("link", "")),
                    source_type="web",
                    query_used=query
                ))
            
            return results
        except Exception as e:
            logger.error(f"SerpAPI search failed: {e}")
            return []
    
    def _search_tavily(self, query: str, max_results: int,
                      language: str, time_range: str) -> List[SearchResult]:
        """Search using Tavily (AI-optimized)."""
        if not self.tavily_key:
            logger.warning("Tavily key not set, falling back to DuckDuckGo")
            return self._search_duckduckgo(query, max_results, language, time_range)
        
        try:
            import requests
            
            headers = {"Authorization": f"Bearer {self.tavily_key}"}
            data = {
                "query": query,
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": False
            }
            
            resp = requests.post(
                "https://api.tavily.com/search",
                json=data,
                headers=headers,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source_reliability=SourceCredibility.score_url(item.get("url", "")),
                    source_type="web",
                    query_used=query
                ))
            
            return results
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []
    
    def _search_wikipedia(self, query: str, max_results: int,
                         language: str, time_range: str) -> List[SearchResult]:
        """Search Wikipedia directly."""
        try:
            import requests
            
            lang_code = language.split("-")[0]
            url = f"https://{lang_code}.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
                "origin": "*"
            }
            
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            results = []
            for item in data.get("query", {}).get("search", [])[:max_results]:
                page_id = item.get("pageid")
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=f"https://{lang_code}.wikipedia.org/wiki?curid={page_id}",
                    snippet=re.sub(r'<[^>]+>', '', item.get("snippet", "")),
                    source_reliability=0.9,
                    source_type="wikipedia",
                    query_used=query
                ))
            
            return results
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
            return []
    
    def _search_arxiv(self, query: str, max_results: int,
                     language: str, time_range: str) -> List[SearchResult]:
        """Search arXiv for academic papers."""
        try:
            import requests
            
            url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.content)
            
            results = []
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            
            for entry in root.findall("atom:entry", ns)[:max_results]:
                title = entry.find("atom:title", ns).text.strip()
                summary = entry.find("atom:summary", ns).text.strip()
                link = entry.find("atom:id", ns).text
                published = entry.find("atom:published", ns).text
                
                results.append(SearchResult(
                    title=title,
                    url=link,
                    snippet=summary[:300] + "..." if len(summary) > 300 else summary,
                    published_at=published,
                    source_reliability=0.95,
                    source_type="arxiv",
                    query_used=query
                ))
            
            return results
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return []
    
    def _search_searxng(self, query: str, max_results: int,
                       language: str, time_range: str) -> List[SearchResult]:
        """Search using custom SearxNG instance."""
        if not self.searxng_url:
            logger.warning("SearxNG URL not set, falling back to DuckDuckGo")
            return self._search_duckduckgo(query, max_results, language, time_range)
        
        try:
            import requests
            
            url = f"{self.searxng_url.rstrip('/')}/search"
            params = {
                "q": query,
                "format": "json",
                "engines": "google,duckduckgo,bing",
                "language": language
            }
            
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source_reliability=SourceCredibility.score_url(item.get("url", "")),
                    source_type="searxng",
                    query_used=query
                ))
            
            return results
        except Exception as e:
            logger.error(f"SearxNG search failed: {e}")
            return []
    
    def search_multiple_queries(self, queries: List[str], 
                                 max_results_per_query: int = 3) -> List[SearchResponse]:
        """
        Search multiple queries and aggregate results.
        
        Useful for comprehensive research on a topic.
        """
        all_responses = []
        
        for query in queries:
            try:
                response = self.search(query, max_results=max_results_per_query)
                all_responses.append(response)
            except Exception as e:
                logger.error(f"Query '{query}' failed: {e}")
        
        return all_responses
    
    def research_topic(self, topic: str, 
                      aspects: List[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive research on a topic.
        
        Args:
            topic: Main topic to research
            aspects: Specific aspects to investigate
            
        Returns:
            Comprehensive research results with citations
        """
        if aspects is None:
            aspects = [
                f"{topic} overview",
                f"{topic} implementation",
                f"{topic} best practices",
                f"{topic} tutorials"
            ]
        
        # Search multiple aspects
        responses = self.search_multiple_queries(aspects, max_results_per_query=3)
        
        # Aggregate results
        all_results = []
        for response in responses:
            all_results.extend(response.results)
        
        # Remove duplicates by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)
        
        # Sort by credibility
        unique_results.sort(key=lambda x: x.source_reliability, reverse=True)
        
        # Generate summary
        summary = {
            "topic": topic,
            "total_sources": len(unique_results),
            "high_credibility_sources": sum(1 for r in unique_results if r.source_reliability >= 0.8),
            "results": [r.to_dict() for r in unique_results],
            "citations": [f"[{i+1}] {r.title}: {r.url}" for i, r in enumerate(unique_results[:10])],
            "research_queries": aspects
        }
        
        return summary
    
    def clear_cache(self):
        """Clear the search cache."""
        self._cache.clear()
        logger.info("Search cache cleared")

    def multi_source_search(
        self,
        query: str,
        max_results: int = 5,
        backends: Optional[List[str]] = None,
    ) -> "SearchResponse":
        """Search across multiple backends and merge results.

        Queries DuckDuckGo, Wikipedia and arXiv in parallel (sequentially
        here for simplicity) and cross-references results to increase
        reliability.  Deduplicated by URL; sorted by credibility.

        Args:
            query: The search query.
            max_results: Max results per backend.
            backends: List of backend names to query. Defaults to
                      ["duckduckgo", "wikipedia", "arxiv"].

        Returns:
            Merged SearchResponse with combined results.
        """
        if backends is None:
            backends = ["duckduckgo", "wikipedia", "arxiv"]

        start_time = time.time()
        all_results: List[SearchResult] = []
        seen_urls: set = set()
        provenance: List[Dict[str, Any]] = []

        original_backend = self.backend_name
        for backend in backends:
            try:
                self.backend_name = backend
                response = self.search(query, max_results=max_results)
                for r in response.results:
                    if r.url not in seen_urls:
                        seen_urls.add(r.url)
                        all_results.append(r)
                provenance.append({
                    "backend": backend,
                    "result_count": len(response.results),
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                logger.warning(f"Backend {backend} failed for query '{query}': {e}")
        self.backend_name = original_backend

        # Sort by credibility, then trim
        all_results.sort(key=lambda x: x.source_reliability, reverse=True)
        all_results = all_results[:max_results * len(backends)]

        return SearchResponse(
            results=all_results,
            query=query,
            backend=f"multi:{','.join(backends)}",
            total_results=len(all_results),
            execution_time_ms=int((time.time() - start_time) * 1000),
            citations=[f"[{i+1}] {r.title}. {r.url}" for i, r in enumerate(all_results)],
            provenance=provenance,
        )

    def verify_claim(
        self,
        claim: str,
        min_sources: int = 2,
        max_search_results: int = 5,
    ) -> Dict[str, Any]:
        """Verify a factual claim against multiple web sources.

        Implements the "two-source rule" from web browsing agent best
        practices: a claim is only considered verified when at least
        ``min_sources`` independent sources corroborate it.

        Args:
            claim: The factual statement to verify.
            min_sources: Minimum number of sources needed for verification.
            max_search_results: Max results per search query.

        Returns:
            Dict with keys: verified (bool), confidence (float),
            supporting_sources (list), contradicting_sources (list),
            verdict (str).
        """
        # Search for the claim and its negation
        supporting_results = self.multi_source_search(
            claim, max_results=max_search_results
        ).results
        negation_query = f"NOT ({claim}) OR problems with {claim}"
        contradicting_results = self.search(
            negation_query, max_results=3
        ).results

        claim_lower = claim.lower()
        claim_words = set(claim_lower.split())

        supporting: List[Dict] = []
        contradicting: List[Dict] = []

        for r in supporting_results:
            text = f"{r.title} {r.snippet}".lower()
            # Simple overlap scoring
            overlap = len(claim_words & set(text.split())) / max(len(claim_words), 1)
            entry = {"title": r.title, "url": r.url, "overlap": round(overlap, 2),
                     "reliability": r.source_reliability}
            if overlap > 0.2:
                supporting.append(entry)

        for r in contradicting_results:
            text = f"{r.title} {r.snippet}".lower()
            if any(w in text for w in ["not", "false", "wrong", "incorrect", "debunked", "myth"]):
                contradicting.append({"title": r.title, "url": r.url,
                                      "reliability": r.source_reliability})

        verified = len(supporting) >= min_sources
        confidence = min(1.0, len(supporting) / max(min_sources, 1)) * 0.9
        if contradicting:
            confidence *= 0.7  # Reduce confidence when contradictions exist

        verdict = "VERIFIED" if verified else ("UNVERIFIED" if not supporting else "UNCERTAIN")

        return {
            "claim": claim,
            "verified": verified,
            "confidence": round(confidence, 2),
            "verdict": verdict,
            "supporting_sources": supporting[:5],
            "contradicting_sources": contradicting[:3],
            "min_sources_required": min_sources,
        }


# Global instance
_search_tool: Optional[WebSearchTool] = None


def get_search_tool() -> WebSearchTool:
    """Get or create the global search tool instance."""
    global _search_tool
    if _search_tool is None:
        _search_tool = WebSearchTool()
    return _search_tool


def init_search_tool(backend: str = None, **kwargs) -> WebSearchTool:
    """Initialize a new search tool instance."""
    global _search_tool
    _search_tool = WebSearchTool(backend=backend, **kwargs)
    return _search_tool


if __name__ == "__main__":
    # Test the search tool
    logging.basicConfig(level=logging.DEBUG)
    
    tool = WebSearchTool(backend="duckduckgo")
    
    # Simple search
    print("=== Simple Search ===")
    result = tool.search("Python async programming", max_results=3)
    print(f"Found {result.total_results} results in {result.execution_time_ms}ms")
    for r in result.results:
        print(f"  - {r.title}: {r.url}")
    
    # Research topic
    print("\n=== Research Topic ===")
    research = tool.research_topic("LLM agents orchestration")
    print(f"Topic: {research['topic']}")
    print(f"Total sources: {research['total_sources']}")
    print(f"High credibility: {research['high_credibility_sources']}")
    print("Citations:")
    for cite in research['citations'][:5]:
        print(f"  {cite}")
