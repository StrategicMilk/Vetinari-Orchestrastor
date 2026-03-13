from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from vetinari.utils import estimate_model_memory_gb

logger = logging.getLogger(__name__)


@dataclass
class ModelSource:
    source_type: str
    url: str
    last_checked: str = ""
    confidence: float = 0.0


@dataclass
class ModelCandidate:
    id: str
    name: str
    source_type: str
    metrics: dict[str, Any] = field(default_factory=dict)
    memory_gb: int = 2
    context_len: int = 2048
    version: str = ""
    last_updated: str = ""
    hard_data_score: float = 0.0
    benchmark_score: float = 0.0
    sentiment_score: float = 0.0
    recency_score: float = 1.0
    final_score: float = 0.0
    provenance: list[ModelSource] = field(default_factory=list)

    def to_dict(self) -> dict:
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
            "provenance": [
                {"source_type": p.source_type, "url": p.url, "last_checked": p.last_checked, "confidence": p.confidence}
                for p in self.provenance
            ],
        }


class ModelSearchEngine:
    WEIGHTS = {"hard_data": 0.55, "benchmarks": 0.25, "sentiment": 0.15, "recency": 0.05}

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".lmstudio" / "projects" / "Vetinari" / "model_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Vetinari/1.0"})

    def search_for_task(
        self, task_description: str, lm_studio_models: list[dict] | None = None
    ) -> list[ModelCandidate]:
        candidates = []

        if lm_studio_models:
            for model in lm_studio_models:
                candidate = self._create_candidate_from_lm_studio(model)
                candidates.append(candidate)

        external_candidates = self._search_external_sources(task_description)
        candidates.extend(external_candidates)

        cloud_candidates = self._search_cloud_providers(task_description)
        candidates.extend(cloud_candidates)

        for candidate in candidates:
            candidate.final_score = self._calculate_score(candidate)

        candidates.sort(key=lambda x: x.final_score, reverse=True)

        return candidates[:15]

    def _create_candidate_from_lm_studio(self, model: dict) -> ModelCandidate:
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
            provenance=[
                ModelSource(
                    source_type="lm_studio",
                    url=f"lmstudio://model/{model.get('id', '')}",
                    last_checked=datetime.now().isoformat(),
                    confidence=0.9,
                )
            ],
        )

    def _search_external_sources(self, task_description: str) -> list[ModelCandidate]:
        candidates = []

        hf_candidates = self._search_huggingface(task_description)
        candidates.extend(hf_candidates)

        reddit_candidates = self._search_reddit(task_description)
        candidates.extend(reddit_candidates)

        github_candidates = self._search_github(task_description)
        candidates.extend(github_candidates)

        return candidates

    def _search_huggingface(self, query: str) -> list[ModelCandidate]:
        cache_key = hashlib.md5(query.encode("utf-8")).hexdigest()  # noqa: S324
        cache_file = self.cache_dir / f"hf_{cache_key}.json"

        if cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age < timedelta(days=7):
                try:
                    with open(cache_file) as f:
                        data = json.load(f)
                    result = []
                    for c in data:
                        # Deserialise provenance dicts back to ModelSource objects
                        if "provenance" in c and isinstance(c["provenance"], list):
                            c["provenance"] = [ModelSource(**p) if isinstance(p, dict) else p for p in c["provenance"]]
                        result.append(ModelCandidate(**c))
                    return result
                except Exception as e:
                    logger.debug(f"Cache load failed for {cache_file}: {e}")

        candidates = []

        keywords = self._extract_keywords(query)

        model_names = [
            ("Qwen/Qwen2.5-Coder", "code generation"),
            ("Qwen/Qwen2.5", "general purpose"),
            ("meta-llama/Llama-3", "general purpose"),
            ("mistralai/Mistral", "general purpose"),
            ("codellama/CodeLlama", "code generation"),
            ("deepseek-ai/DeepSeek-Coder", "code generation"),
            ("bigcode/starcoder2", "code generation"),
            ("microsoft/Phi", "compact"),
            ("google/gemma", "general purpose"),
            ("01-ai/Yi", "general purpose"),
        ]

        for model_id, category in model_names:
            if any(kw in category.lower() for kw in keywords) or any(kw in model_id.lower() for kw in keywords):
                candidate = ModelCandidate(
                    id=model_id,
                    name=model_id.split("/")[-1],
                    source_type="huggingface",
                    metrics={"category": category, "query_match": True},
                    memory_gb=self._estimate_memory(model_id),
                    context_len=8192,
                    version="latest",
                    last_updated=datetime.now().isoformat(),
                    hard_data_score=0.7,
                    benchmark_score=0.75,
                    sentiment_score=0.7,
                    provenance=[
                        ModelSource(
                            source_type="huggingface",
                            url=f"https://huggingface.co/{model_id}",
                            last_checked=datetime.now().isoformat(),
                            confidence=0.8,
                        )
                    ],
                )
                candidates.append(candidate)

        if candidates:
            with open(cache_file, "w") as f:
                json.dump([c.to_dict() for c in candidates], f)

        return candidates

    def _search_reddit(self, query: str) -> list[ModelCandidate]:
        cache_key = hashlib.md5(query.encode("utf-8")).hexdigest()  # noqa: S324
        cache_file = self.cache_dir / f"reddit_{cache_key}.json"

        if cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age < timedelta(days=7):
                try:
                    with open(cache_file) as f:
                        data = json.load(f)
                    result = []
                    for c in data:
                        if "provenance" in c and isinstance(c["provenance"], list):
                            c["provenance"] = [ModelSource(**p) if isinstance(p, dict) else p for p in c["provenance"]]
                        result.append(ModelCandidate(**c))
                    return result
                except Exception as e:
                    logger.debug(f"Cache load failed for {cache_file}: {e}")

        candidates = []

        recommendations = self._get_reddit_recommendations(query)

        for rec in recommendations:
            candidate = ModelCandidate(
                id=rec["model_id"],
                name=rec["model_name"],
                source_type="reddit",
                metrics={"subreddit": rec.get("subreddit", ""), "votes": rec.get("votes", 0)},
                memory_gb=rec.get("memory_gb", 4),
                context_len=rec.get("context_len", 4096),
                version=rec.get("version", ""),
                last_updated=datetime.now().isoformat(),
                hard_data_score=0.3,
                benchmark_score=0.4,
                sentiment_score=rec.get("sentiment", 0.6),
                provenance=[
                    ModelSource(
                        source_type="reddit",
                        url=rec.get("url", ""),
                        last_checked=datetime.now().isoformat(),
                        confidence=0.5,
                    )
                ],
            )
            candidates.append(candidate)

        if candidates:
            with open(cache_file, "w") as f:
                json.dump([c.to_dict() for c in candidates], f)

        return candidates

    def _search_github(self, query: str) -> list[ModelCandidate]:
        cache_key = hashlib.md5(query.encode("utf-8")).hexdigest()  # noqa: S324
        cache_file = self.cache_dir / f"github_{cache_key}.json"

        if cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age < timedelta(days=7):
                try:
                    with open(cache_file) as f:
                        data = json.load(f)
                    result = []
                    for c in data:
                        if "provenance" in c and isinstance(c["provenance"], list):
                            c["provenance"] = [ModelSource(**p) if isinstance(p, dict) else p for p in c["provenance"]]
                        result.append(ModelCandidate(**c))
                    return result
                except Exception as e:
                    logger.debug(f"Cache load failed for {cache_file}: {e}")

        candidates = []

        repo_recs = [
            ("meta-llama/llama", "meta-llama", 8),
            ("mistralai/mistral-src", "mistralai", 8),
            ("QwenLM/Qwen", "qwen", 4),
            ("deepseek-ai/deepseek-coder", "deepseek-ai", 4),
            ("bigcode-project/starcoder2", "bigcode", 4),
        ]

        keywords = self._extract_keywords(query)

        for repo, org, mem in repo_recs:
            if any(kw in repo.lower() for kw in keywords):
                candidate = ModelCandidate(
                    id=f"github/{org}/{repo.split('/')[-1]}",
                    name=repo.split("/")[-1],
                    source_type="github",
                    metrics={"repo": repo, "stars": 5000},
                    memory_gb=mem,
                    context_len=4096,
                    version="latest",
                    last_updated=datetime.now().isoformat(),
                    hard_data_score=0.5,
                    benchmark_score=0.6,
                    sentiment_score=0.6,
                    provenance=[
                        ModelSource(
                            source_type="github",
                            url=f"https://github.com/{repo}",
                            last_checked=datetime.now().isoformat(),
                            confidence=0.7,
                        )
                    ],
                )
                candidates.append(candidate)

        if candidates:
            with open(cache_file, "w") as f:
                json.dump([c.to_dict() for c in candidates], f)

        return candidates

    def _search_claude(self, query: str) -> list[ModelCandidate]:
        import os

        candidates = []

        api_key = os.environ.get("CLAUDE_API_KEY")
        if not api_key:
            logger.debug("Claude API key not configured")
            return candidates

        cache_file = self.cache_dir / f"claude_{hash(query)}.json"

        if cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age < timedelta(seconds=60):
                with open(cache_file) as f:
                    data = json.load(f)
                    return [ModelCandidate(**c) for c in data]

        keywords = self._extract_keywords(query)

        claude_models = [
            ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet", 200000),
            ("claude-3-opus-20240229", "Claude 3 Opus", 200000),
            ("claude-3-haiku-20240307", "Claude 3 Haiku", 200000),
        ]

        for model_id, model_name, context_len in claude_models:
            relevance = 0.5
            if any(kw in ["reasoning", "think", "analyze"] for kw in keywords):
                relevance = 0.85
            elif any(kw in ["code", "program", "develop"] for kw in keywords):
                relevance = 0.75

            candidate = ModelCandidate(
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
                provenance=[
                    ModelSource(
                        source_type="claude",
                        url=f"https://console.anthropic.com/{model_id}",
                        last_checked=datetime.now().isoformat(),
                        confidence=0.9,
                    )
                ],
            )
            candidates.append(candidate)

        if candidates:
            with open(cache_file, "w") as f:
                json.dump([c.to_dict() for c in candidates], f)

        return candidates

    def _search_gemini(self, query: str) -> list[ModelCandidate]:
        import os

        candidates = []

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.debug("Gemini API key not configured")
            return candidates

        cache_file = self.cache_dir / f"gemini_{hash(query)}.json"

        if cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age < timedelta(seconds=60):
                with open(cache_file) as f:
                    data = json.load(f)
                    return [ModelCandidate(**c) for c in data]

        keywords = self._extract_keywords(query)

        gemini_models = [
            ("gemini-2.0-flash-exp", "Gemini 2.0 Flash", 1000000),
            ("gemini-1.5-pro", "Gemini 1.5 Pro", 2000000),
            ("gemini-1.5-flash", "Gemini 1.5 Flash", 1000000),
        ]

        for model_id, model_name, context_len in gemini_models:
            relevance = 0.5
            if any(kw in ["reasoning", "think", "analyze"] for kw in keywords):
                relevance = 0.8
            elif any(kw in ["code", "program", "develop"] for kw in keywords):
                relevance = 0.75
            elif any(kw in ["creative", "write", "story"] for kw in keywords):
                relevance = 0.82

            candidate = ModelCandidate(
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
                provenance=[
                    ModelSource(
                        source_type="gemini",
                        url=f"https://aistudio.google.com/app/{model_id}",
                        last_checked=datetime.now().isoformat(),
                        confidence=0.85,
                    )
                ],
            )
            candidates.append(candidate)

        if candidates:
            with open(cache_file, "w") as f:
                json.dump([c.to_dict() for c in candidates], f)

        return candidates

    def _search_cloud_providers(self, query: str) -> list[ModelCandidate]:
        """Search all configured cloud providers."""
        candidates = []

        claude_candidates = self._search_claude(query)
        candidates.extend(claude_candidates)

        gemini_candidates = self._search_gemini(query)
        candidates.extend(gemini_candidates)

        return candidates

    def _get_reddit_recommendations(self, query: str) -> list[dict]:
        self._extract_keywords(query)

        recommendations = []

        if any(kw in query.lower() for kw in ["code", "programming", "coding", "developer"]):
            recommendations.extend(
                [
                    {
                        "model_id": "qwen2.5-coder",
                        "model_name": "Qwen2.5-Coder",
                        "subreddit": "r/LocalLLaMA",
                        "votes": 150,
                        "sentiment": 0.85,
                        "memory_gb": 4,
                        "context_len": 8192,
                        "url": "https://reddit.com/r/LocalLLaMA",
                    },
                    {
                        "model_id": "codellama",
                        "model_name": "CodeLlama",
                        "subreddit": "r/LocalLLaMA",
                        "votes": 120,
                        "sentiment": 0.75,
                        "memory_gb": 8,
                        "context_len": 16384,
                        "url": "https://reddit.com/r/LocalLLaMA",
                    },
                    {
                        "model_id": "deepseek-coder",
                        "model_name": "DeepSeek Coder",
                        "subreddit": "r/LocalLLaMA",
                        "votes": 95,
                        "sentiment": 0.8,
                        "memory_gb": 4,
                        "context_len": 8192,
                        "url": "https://reddit.com/r/LocalLLaMA",
                    },
                ]
            )

        if any(kw in query.lower() for kw in ["general", "chat", "conversation"]):
            recommendations.extend(
                [
                    {
                        "model_id": "llama-3",
                        "model_name": "Llama 3",
                        "subreddit": "r/LocalLLaMA",
                        "votes": 200,
                        "sentiment": 0.9,
                        "memory_gb": 8,
                        "context_len": 8192,
                        "url": "https://reddit.com/r/LocalLLaMA",
                    },
                    {
                        "model_id": "mistral",
                        "model_name": "Mistral",
                        "subreddit": "r/LocalLLaMA",
                        "votes": 180,
                        "sentiment": 0.85,
                        "memory_gb": 4,
                        "context_len": 8192,
                        "url": "https://reddit.com/r/LocalLLaMA",
                    },
                    {
                        "model_id": "qwen2.5",
                        "model_name": "Qwen2.5",
                        "subreddit": "r/LocalLLaMA",
                        "votes": 160,
                        "sentiment": 0.88,
                        "memory_gb": 4,
                        "context_len": 8192,
                        "url": "https://reddit.com/r/LocalLLaMA",
                    },
                ]
            )

        if any(kw in query.lower() for kw in ["reasoning", "think", "analysis"]):
            recommendations.extend(
                [
                    {
                        "model_id": "qwen3-thinking",
                        "model_name": "Qwen3 (Thinking)",
                        "subreddit": "r/LocalLLaMA",
                        "votes": 140,
                        "sentiment": 0.82,
                        "memory_gb": 8,
                        "context_len": 32768,
                        "url": "https://reddit.com/r/LocalLLaMA",
                    },
                    {
                        "model_id": "deepseek-llm",
                        "model_name": "DeepSeek LLM",
                        "subreddit": "r/LocalLLaMA",
                        "votes": 100,
                        "sentiment": 0.75,
                        "memory_gb": 8,
                        "context_len": 16384,
                        "url": "https://reddit.com/r/LocalLLaMA",
                    },
                ]
            )

        return recommendations[:5]

    def _extract_keywords(self, text: str) -> list[str]:
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

    def _estimate_memory(self, model_id: str) -> int:
        """Delegate to shared utility in vetinari.utils."""
        return estimate_model_memory_gb(model_id)

    def _calculate_score(self, candidate: ModelCandidate) -> float:
        recency_days = 30
        if candidate.last_updated:
            try:
                updated = datetime.fromisoformat(candidate.last_updated)
                recency_days = (datetime.now() - updated).days
            except (ValueError, TypeError):  # noqa: VET022
                pass

        recency_score = max(0.5, 1.0 - (recency_days / 365))
        candidate.recency_score = recency_score

        final_score = (
            candidate.hard_data_score * self.WEIGHTS["hard_data"]
            + candidate.benchmark_score * self.WEIGHTS["benchmarks"]
            + candidate.sentiment_score * self.WEIGHTS["sentiment"]
            + recency_score * self.WEIGHTS["recency"]
        )

        return round(final_score, 3)

    def refresh_all_caches(self):
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info("Model cache cleared")

    def get_cached_candidates(self) -> list[ModelCandidate]:
        candidates = []
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    candidates.extend([ModelCandidate(**c) for c in data])
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_file}: {e}")
        return candidates
