"""Unified model discovery module.

Merges ``vetinari.model_search`` and ``vetinari.live_model_search`` into a
single implementation.  Both legacy modules are now thin shims that re-export
from here.
"""
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from vetinari.credentials import credential_manager
from vetinari.utils import estimate_model_memory_gb

logger = logging.getLogger(__name__)

HF_API_URL = "https://huggingface.co/api"
PWC_API_URL = "https://paperswithcode.com/api/v1"
GITHUB_API_URL = "https://api.github.com"

_CACHE_TTL_DAYS = 7


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class ModelSource:
    """Provenance record – kept for backward compat with model_search callers."""
    source_type: str
    url: str
    last_checked: str = ""
    confidence: float = 0.0


@dataclass
class ModelCandidate:
    id: str
    name: str
    source_type: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    memory_gb: int = 2
    context_len: int = 2048
    version: str = ""
    last_updated: str = ""
    hard_data_score: float = 0.0
    benchmark_score: float = 0.0
    sentiment_score: float = 0.0
    recency_score: float = 1.0
    final_score: float = 0.0
    provenance: List[Any] = field(default_factory=list)  # List[Dict] or List[ModelSource]
    short_rationale: str = ""

    def to_dict(self) -> Dict:
        provenance_out = []
        for p in self.provenance:
            if isinstance(p, dict):
                provenance_out.append(p)
            else:
                provenance_out.append({
                    "source_type": p.source_type,
                    "url": p.url,
                    "last_checked": p.last_checked,
                    "confidence": p.confidence,
                })
        return {
            "id": self.id,
            "name": self.name,
            "source_type": self.source_type,
            "metrics": self.metrics,
            "memory_gb": self.memory_gb,
            "context_len": self.context_len,
            "version": self.version,
            "last_updated": self.last_updated,
            "hard_data_score": self.hard_data_score,
            "benchmark_score": self.benchmark_score,
            "sentiment_score": self.sentiment_score,
            "recency_score": self.recency_score,
            "final_score": self.final_score,
            "provenance": provenance_out,
            "short_rationale": self.short_rationale,
        }


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_from_cache(cache_file: Path) -> Optional[List[ModelCandidate]]:
    """Return cached candidates if file exists and is fresh, else None."""
    if not cache_file.exists():
        return None
    age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
    if age >= timedelta(days=_CACHE_TTL_DAYS):
        return None
    try:
        with open(cache_file) as f:
            data = json.load(f)
        return [ModelCandidate(**c) for c in data]
    except Exception as e:
        logger.debug("Cache load failed for %s: %s", cache_file, e)
        return None


def _save_to_cache(cache_file: Path, candidates: List[ModelCandidate]) -> None:
    try:
        with open(cache_file, "w") as f:
            json.dump([c.to_dict() for c in candidates], f)
    except Exception as e:
        logger.debug("Cache save failed for %s: %s", cache_file, e)


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------

class HuggingFaceAdapter:
    def __init__(self):
        self.api_url = HF_API_URL
        self.token = credential_manager.get_token("huggingface")
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        self.session.headers.update({"User-Agent": "Vetinari/1.0"})

    def search_models(self, query: str, limit: int = 10) -> List[ModelCandidate]:
        candidates = []
        try:
            response = self.session.get(
                f"{self.api_url}/models",
                params={"search": query, "limit": limit, "full": "false"},
                timeout=30,
            )
            if response.status_code == 200:
                for model in response.json()[:limit]:
                    try:
                        candidates.append(self._parse_model(model))
                    except Exception as e:
                        logger.warning("Error parsing HF model: %s", e)
        except Exception as e:
            logger.error("HF search error: %s", e)
        return candidates

    def _parse_model(self, model: Dict) -> ModelCandidate:
        model_id = model.get("id", "")
        last_modified = model.get("lastModified", "")
        metrics = model.get("metrics", [])
        hard_data = 0.8 if metrics else 0.6
        rationale_parts = []
        if metrics:
            rationale_parts.append(f"Benchmarks: {len(metrics)} metrics available")
        rationale_parts.append(f"Updated: {last_modified[:10] if last_modified else 'unknown'}")
        return ModelCandidate(
            id=model_id,
            name=model_id.split("/")[-1] if "/" in model_id else model_id,
            source_type="huggingface",
            metrics={
                "downloads": model.get("downloads", 0),
                "likes": model.get("likes", 0),
                "tags": model.get("tags", [])[:5],
            },
            memory_gb=estimate_model_memory_gb(model_id),
            context_len=8192,
            version=model.get("sha", "latest")[:8],
            last_updated=last_modified,
            hard_data_score=hard_data,
            benchmark_score=0.7,
            sentiment_score=min(0.5 + (model.get("likes", 0) / 1000), 1.0),
            provenance=[{
                "source_type": "huggingface",
                "url": f"https://huggingface.co/{model_id}",
                "last_checked": datetime.now().isoformat(),
                "confidence": 0.9,
            }],
            short_rationale=(
                f"HF model with {model.get('likes', 0)} likes, "
                f"{model.get('downloads', 0)} downloads. "
                f"{rationale_parts[0] if rationale_parts else ''}"
            ),
        )


class RedditAdapter:
    def __init__(self):
        self.token = credential_manager.get_token("reddit")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Vetinari/1.0"})
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    def search_local_llm_posts(self, query: str, limit: int = 10) -> List[ModelCandidate]:
        candidates: List[ModelCandidate] = []
        for subreddit in ["LocalLLaMA", "MachineLearning", "LLM", "LanguageTechnology"]:
            try:
                response = self.session.get(
                    f"https://www.reddit.com/r/{subreddit}/search.json",
                    params={"q": query, "limit": limit, "sort": "relevance", "t": "year"},
                    timeout=30,
                )
                if response.status_code == 200:
                    posts = response.json().get("data", {}).get("children", [])
                    for post in posts[:5]:
                        candidate = self._parse_post(post.get("data", {}), subreddit)
                        if candidate:
                            candidates.append(candidate)
            except Exception as e:
                logger.warning("Reddit search error for %s: %s", subreddit, e)
        return candidates[:limit]

    def _parse_post(self, post: Dict, subreddit: str) -> Optional[ModelCandidate]:
        title = post.get("title", "")
        score = post.get("score", 0)
        num_comments = post.get("num_comments", 0)
        model_mentions = self._extract_model_mentions(title)
        if not model_mentions:
            return None
        best_model = model_mentions[0]
        sentiment = min(0.5 + (score / 500), 1.0)
        return ModelCandidate(
            id=best_model.lower().replace(" ", "-"),
            name=best_model,
            source_type="reddit",
            metrics={
                "subreddit": subreddit,
                "upvotes": score,
                "comments": num_comments,
                "title": title[:100],
                "all_mentions": model_mentions,
            },
            memory_gb=estimate_model_memory_gb(best_model),
            context_len=4096,
            version="latest",
            last_updated=datetime.now().isoformat(),
            hard_data_score=0.3,
            benchmark_score=0.4,
            sentiment_score=sentiment,
            provenance=[{
                "source_type": "reddit",
                "url": f"https://reddit.com{post.get('permalink', '')}",
                "last_checked": datetime.now().isoformat(),
                "confidence": 0.6,
            }],
            short_rationale=f"Mentioned in r/{subreddit}: {score} upvotes, {num_comments} comments",
        )

    def _extract_model_mentions(self, text: str) -> List[str]:
        known_models = [
            "Qwen", "Llama", "Mistral", "DeepSeek", "CodeLlama", "Gemma",
            "Phi", "Yi", "StarCoder", "Mixtral", "Command-R", "Haiku",
        ]
        text_lower = text.lower()
        return [m for m in known_models if m.lower() in text_lower]


class GitHubAdapter:
    def __init__(self):
        self.token = credential_manager.get_token("github")
        self.api_url = GITHUB_API_URL
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"token {self.token}"})
        self.session.headers.update({"User-Agent": "Vetinari/1.0"})

    def search_repos(self, query: str, limit: int = 10) -> List[ModelCandidate]:
        candidates = []
        try:
            response = self.session.get(
                f"{self.api_url}/search/repositories",
                params={"q": f"{query} language:python stars:>100", "per_page": limit, "sort": "stars"},
                timeout=30,
            )
            if response.status_code == 200:
                for repo in response.json().get("items", [])[:limit]:
                    try:
                        candidates.append(self._parse_repo(repo))
                    except Exception as e:
                        logger.warning("Error parsing GitHub repo: %s", e)
        except Exception as e:
            logger.error("GitHub search error: %s", e)
        return candidates

    def _parse_repo(self, repo: Dict) -> ModelCandidate:
        full_name = repo.get("full_name", "")
        name = full_name.split("/")[-1] if "/" in full_name else repo.get("name", "")
        stars = repo.get("stargazers_count", 0)
        updated = repo.get("updated_at", "")
        return ModelCandidate(
            id=f"github/{full_name}",
            name=name,
            source_type="github",
            metrics={
                "stars": stars,
                "forks": repo.get("forks_count", 0),
                "description": repo.get("description", "")[:100],
            },
            memory_gb=4,
            context_len=4096,
            version="latest",
            last_updated=updated,
            hard_data_score=0.4,
            benchmark_score=0.5,
            sentiment_score=min(0.4 + (stars / 5000), 1.0),
            provenance=[{
                "source_type": "github",
                "url": repo.get("html_url", ""),
                "last_checked": datetime.now().isoformat(),
                "confidence": 0.7,
            }],
            short_rationale=f"GitHub repo with {stars} stars, updated {updated[:10] if updated else 'recently'}",
        )


class PapersWithCodeAdapter:
    def __init__(self):
        self.api_url = PWC_API_URL
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Vetinari/1.0"})

    def search_papers(self, query: str, limit: int = 10) -> List[ModelCandidate]:
        candidates = []
        try:
            response = self.session.get(
                f"{self.api_url}/papers/", params={"search": query}, timeout=30
            )
            if response.status_code == 200:
                for paper in response.json().get("results", [])[:limit]:
                    try:
                        candidates.append(self._parse_paper(paper))
                    except Exception as e:
                        logger.warning("Error parsing PWC paper: %s", e)
        except Exception as e:
            logger.error("PapersWithCode search error: %s", e)
        return candidates

    def _parse_paper(self, paper: Dict) -> ModelCandidate:
        title = paper.get("title", "")
        arxiv_id = paper.get("arxiv_id", "")
        benchmarks = paper.get("benchmarks", [])
        rationale = "Paper on PapersWithCode"
        if benchmarks:
            rationale += f", {len(benchmarks)} benchmarks"
        return ModelCandidate(
            id=f"pwcode/{arxiv_id}" if arxiv_id else f"pwcode/{title[:20]}",
            name=title[:40],
            source_type="paperswithcode",
            metrics={
                "title": title,
                "abstract": paper.get("abstract", "")[:200],
                "benchmarks_count": len(benchmarks),
            },
            memory_gb=4,
            context_len=8192,
            version="latest",
            last_updated=paper.get("published", ""),
            hard_data_score=0.9,
            benchmark_score=0.85,
            sentiment_score=0.7,
            provenance=[{
                "source_type": "paperswithcode",
                "url": f"https://paperswithcode.com/paper/{arxiv_id}" if arxiv_id else "",
                "last_checked": datetime.now().isoformat(),
                "confidence": 0.9,
            }],
            short_rationale=rationale,
        )


# ---------------------------------------------------------------------------
# Scoring weights (shared)
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "hard_data": 0.55,
    "benchmarks": 0.25,
    "sentiment": 0.15,
    "recency": 0.05,
}


def _calculate_score(candidate: ModelCandidate) -> float:
    recency_days = 30
    if candidate.last_updated:
        try:
            updated = datetime.fromisoformat(candidate.last_updated)
            recency_days = (datetime.now() - updated).days
        except (ValueError, TypeError):
            pass
    recency_score = max(0.5, 1.0 - (recency_days / 365))
    candidate.recency_score = recency_score
    return round(
        candidate.hard_data_score * _WEIGHTS["hard_data"]
        + candidate.benchmark_score * _WEIGHTS["benchmarks"]
        + candidate.sentiment_score * _WEIGHTS["sentiment"]
        + recency_score * _WEIGHTS["recency"],
        3,
    )


def _generate_rationale(candidate: ModelCandidate, recency_score: float) -> str:
    parts = []
    if candidate.hard_data_score >= 0.7:
        parts.append("Strong hard data")
    elif candidate.hard_data_score >= 0.5:
        parts.append("Moderate hard data")
    if candidate.benchmark_score >= 0.7:
        parts.append("High benchmark score")
    if candidate.sentiment_score >= 0.7:
        parts.append("Positive community")
    elif candidate.sentiment_score >= 0.5:
        parts.append("Mixed sentiment")
    if recency_score >= 0.9:
        parts.append("Recently updated")
    elif recency_score < 0.7:
        parts.append("May be outdated")
    if not parts:
        parts.append("General purpose")
    return "; ".join(parts[:2]) + "."


def _lmstudio_candidate(model: Dict) -> ModelCandidate:
    return ModelCandidate(
        id=model.get("id", model.get("name", "")),
        name=model.get("name", model.get("id", "")),
        source_type="lm_studio",
        metrics=model.get("capabilities", []),
        memory_gb=model.get("memory_gb", 2),
        context_len=model.get("context_len", 2048),
        version=model.get("version", ""),
        last_updated=datetime.now().isoformat(),
        hard_data_score=0.8,
        benchmark_score=0.7,
        sentiment_score=0.7,
        provenance=[{
            "source_type": "lm_studio",
            "url": f"lmstudio://model/{model.get('id', '')}",
            "last_checked": datetime.now().isoformat(),
            "confidence": 0.9,
        }],
        short_rationale="Local model available in LM Studio",
    )


# ---------------------------------------------------------------------------
# Unified discovery class
# ---------------------------------------------------------------------------

class ModelDiscovery:
    """Unified model discovery.  Replaces both ``ModelSearchEngine`` and
    ``LiveModelSearchAdapter``.

    Compatible API surface:
    - ``search(query, lm_studio_models)`` – main entry point
    - ``search_for_task(task_description, lm_studio_models)`` – alias (ModelSearchEngine compat)
    - ``refresh_all_caches()``
    - ``get_cached_candidates()``
    """

    def __init__(self, cache_dir: str = None):
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path.home() / ".lmstudio" / "projects" / "Vetinari" / "model_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_adapter = HuggingFaceAdapter()
        self.reddit_adapter = RedditAdapter()
        self.github_adapter = GitHubAdapter()
        self.pwc_adapter = PapersWithCodeAdapter()

    # -- main search ---------------------------------------------------------

    def search(self, query: str, lm_studio_models: List[Dict] = None) -> List[ModelCandidate]:
        all_candidates: List[ModelCandidate] = []

        if lm_studio_models:
            all_candidates.extend(_lmstudio_candidate(m) for m in lm_studio_models)

        logger.info("Searching for: %s", query)

        external = self._search_external_sources(query)
        all_candidates.extend(external)

        # cloud providers (no cache – key-gated)
        all_candidates.extend(self._search_cloud_providers(query))

        # de-duplicate
        seen: set = set()
        unique: List[ModelCandidate] = []
        for c in all_candidates:
            if c.id not in seen:
                seen.add(c.id)
                unique.append(c)

        for c in unique:
            prev_rationale = c.short_rationale
            c.final_score = _calculate_score(c)
            if not prev_rationale:
                c.short_rationale = _generate_rationale(c, c.recency_score)

        unique.sort(key=lambda x: x.final_score, reverse=True)
        return unique[:15]

    # Backward-compat alias for ModelSearchEngine callers
    def search_for_task(
        self, task_description: str, lm_studio_models: List[Dict] = None
    ) -> List[ModelCandidate]:
        return self.search(task_description, lm_studio_models)

    # -- source groupings (kept for backward compat / patchability) ----------

    def _search_external_sources(self, query: str) -> List[ModelCandidate]:
        """Search HF, Reddit, GitHub, and PapersWithCode with caching."""
        candidates: List[ModelCandidate] = []

        hf = self._cached_search("hf", query, lambda q: self.hf_adapter.search_models(q, limit=8))
        candidates.extend(hf)
        logger.info("HF found %d candidates", len(hf))

        reddit = self._cached_search("reddit", query, lambda q: self.reddit_adapter.search_local_llm_posts(q, limit=5))
        candidates.extend(reddit)
        logger.info("Reddit found %d candidates", len(reddit))

        github = self._cached_search("github", query, lambda q: self.github_adapter.search_repos(q, limit=5))
        candidates.extend(github)
        logger.info("GitHub found %d candidates", len(github))

        pwc = self._cached_search("pwc", query, lambda q: self.pwc_adapter.search_papers(q, limit=3))
        candidates.extend(pwc)
        logger.info("PapersWithCode found %d candidates", len(pwc))

        return candidates

    # -- cache helpers -------------------------------------------------------

    def _cached_search(self, prefix: str, query: str, fetch_fn) -> List[ModelCandidate]:
        cache_key = hashlib.md5(query.encode("utf-8")).hexdigest()
        cache_file = self.cache_dir / f"{prefix}_{cache_key}.json"
        cached = _load_from_cache(cache_file)
        if cached is not None:
            return cached
        results = fetch_fn(query)
        if results:
            _save_to_cache(cache_file, results)
        return results

    def refresh_all_caches(self):
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info("Model cache cleared")

    def get_cached_candidates(self) -> List[ModelCandidate]:
        candidates: List[ModelCandidate] = []
        for cache_file in self.cache_dir.glob("*.json"):
            cached = _load_from_cache(cache_file)
            if cached:
                candidates.extend(cached)
        return candidates

    # -- cloud providers -----------------------------------------------------

    def _search_cloud_providers(self, query: str) -> List[ModelCandidate]:
        candidates: List[ModelCandidate] = []
        candidates.extend(self._search_claude(query))
        candidates.extend(self._search_gemini(query))
        return candidates

    def _search_claude(self, query: str) -> List[ModelCandidate]:
        import os
        api_key = os.environ.get("CLAUDE_API_KEY")
        if not api_key:
            return []
        keywords = _extract_keywords(query)
        models = [
            ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet", 200000),
            ("claude-3-opus-20240229", "Claude 3 Opus", 200000),
            ("claude-3-haiku-20240307", "Claude 3 Haiku", 200000),
        ]
        candidates = []
        for model_id, model_name, context_len in models:
            relevance = 0.85 if any(k in ["reasoning", "think", "analyze"] for k in keywords) \
                else 0.75 if any(k in ["code", "program", "develop"] for k in keywords) \
                else 0.5
            candidates.append(ModelCandidate(
                id=f"claude:{model_id}",
                name=model_name,
                source_type="claude",
                metrics={"query": query, "relevance": relevance},
                memory_gb=0,
                context_len=context_len,
                version=model_id,
                last_updated=datetime.now().isoformat(),
                hard_data_score=0.9,
                benchmark_score=0.92,
                sentiment_score=0.88,
                provenance=[{
                    "source_type": "claude",
                    "url": f"https://console.anthropic.com/{model_id}",
                    "last_checked": datetime.now().isoformat(),
                    "confidence": 0.9,
                }],
            ))
        return candidates

    def _search_gemini(self, query: str) -> List[ModelCandidate]:
        import os
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return []
        keywords = _extract_keywords(query)
        models = [
            ("gemini-2.0-flash-exp", "Gemini 2.0 Flash", 1000000),
            ("gemini-1.5-pro", "Gemini 1.5 Pro", 2000000),
            ("gemini-1.5-flash", "Gemini 1.5 Flash", 1000000),
        ]
        candidates = []
        for model_id, model_name, context_len in models:
            relevance = 0.8 if any(k in ["reasoning", "think", "analyze"] for k in keywords) \
                else 0.82 if any(k in ["creative", "write", "story"] for k in keywords) \
                else 0.75 if any(k in ["code", "program", "develop"] for k in keywords) \
                else 0.5
            candidates.append(ModelCandidate(
                id=f"gemini:{model_id}",
                name=model_name,
                source_type="gemini",
                metrics={"query": query, "relevance": relevance},
                memory_gb=0,
                context_len=context_len,
                version=model_id,
                last_updated=datetime.now().isoformat(),
                hard_data_score=0.85,
                benchmark_score=0.88,
                sentiment_score=0.82,
                provenance=[{
                    "source_type": "gemini",
                    "url": f"https://aistudio.google.com/app/{model_id}",
                    "last_checked": datetime.now().isoformat(),
                    "confidence": 0.85,
                }],
            ))
        return candidates


# ---------------------------------------------------------------------------
# Keyword helper (shared)
# ---------------------------------------------------------------------------

def _extract_keywords(text: str) -> List[str]:
    keywords = []
    text_lower = text.lower()
    categories = {
        "code": ["code", "program", "develop", "implement", "software", "app", "web", "script", "function", "api"],
        "reasoning": ["reason", "think", "analyze", "solve", "problem", "logic", "math"],
        "chat": ["chat", "conversation", "talk", "message", "respond"],
        "creative": ["write", "story", "creative", "article", "content"],
        "data": ["data", "database", "sql", "query", "analyze", "etl"],
    }
    for category, words in categories.items():
        if any(w in text_lower for w in words):
            keywords.append(category)
    return keywords


# ---------------------------------------------------------------------------
# Backward-compat aliases
# ---------------------------------------------------------------------------

# model_search.ModelSearchEngine -> ModelDiscovery (same API)
ModelSearchEngine = ModelDiscovery

# live_model_search.LiveModelSearchAdapter -> ModelDiscovery
LiveModelSearchAdapter = ModelDiscovery
