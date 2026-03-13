from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import requests

from vetinari.credentials import credential_manager
from vetinari.utils import estimate_model_memory_gb

logger = logging.getLogger(__name__)

HF_API_URL = "https://huggingface.co/api"
PWC_API_URL = "https://paperswithcode.com/api/v1"
GITHUB_API_URL = "https://api.github.com"


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
    provenance: list[dict] = field(default_factory=list)
    short_rationale: str = ""

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
            "provenance": self.provenance,
            "short_rationale": self.short_rationale,
        }


class HuggingFaceAdapter:
    def __init__(self):
        self.api_url = HF_API_URL
        self.token = credential_manager.get_token("huggingface")
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        self.session.headers.update({"User-Agent": "Vetinari/1.0"})

    def search_models(self, query: str, limit: int = 10) -> list[ModelCandidate]:
        candidates = []

        try:
            params = {"search": query, "limit": limit, "full": "false"}

            response = self.session.get(f"{self.api_url}/models", params=params, timeout=30)

            if response.status_code == 200:
                models = response.json()

                for model in models[:limit]:
                    try:
                        candidate = self._parse_model(model)
                        candidates.append(candidate)
                    except Exception as e:
                        logger.warning(f"Error parsing HF model: {e}")

        except Exception as e:
            logger.error(f"HF search error: {e}")

        return candidates

    def _parse_model(self, model: dict) -> ModelCandidate:
        model_id = model.get("id", "")
        last_modified = model.get("lastModified", "")

        memory = self._estimate_memory(model_id, model.get("params", ""))

        metrics = model.get("metrics", [])

        hard_data = 0.6
        if metrics:
            hard_data = 0.8

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
            memory_gb=memory,
            context_len=8192,
            version=model.get("sha", "latest")[:8],
            last_updated=last_modified,
            hard_data_score=hard_data,
            benchmark_score=0.7,
            sentiment_score=min(0.5 + (model.get("likes", 0) / 1000), 1.0),
            provenance=[
                {
                    "source_type": "huggingface",
                    "url": f"https://huggingface.co/{model_id}",
                    "last_checked": datetime.now().isoformat(),
                    "confidence": 0.9,
                }
            ],
            short_rationale=f"HF model with {model.get('likes', 0)} likes, {model.get('downloads', 0)} downloads. {rationale_parts[0] if rationale_parts else ''}",
        )

    def _estimate_memory(self, model_id: str, params: str = "") -> int:
        """Delegate to shared utility in vetinari.utils."""
        return estimate_model_memory_gb(model_id)


class RedditAdapter:
    def __init__(self):
        self.token = credential_manager.get_token("reddit")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Vetinari/1.0"})

        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    def search_local_llm_posts(self, query: str, limit: int = 10) -> list[ModelCandidate]:
        candidates = []

        subreddits = ["LocalLLaMA", "MachineLearning", "LLM", "LanguageTechnology"]

        for subreddit in subreddits:
            try:
                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {"q": query, "limit": limit, "sort": "relevance", "t": "year"}

                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    posts = data.get("data", {}).get("children", [])

                    for post in posts[:5]:
                        post_data = post.get("data", {})
                        candidate = self._parse_post(post_data, subreddit)
                        if candidate:
                            candidates.append(candidate)

            except Exception as e:
                logger.warning(f"Reddit search error for {subreddit}: {e}")

        return candidates[:limit]

    def _parse_post(self, post: dict, subreddit: str) -> ModelCandidate | None:
        title = post.get("title", "")
        score = post.get("score", 0)
        num_comments = post.get("num_comments", 0)

        model_mentions = self._extract_model_mentions(title)

        if not model_mentions:
            return None

        # Return the best-mentioned model (highest upvote sentiment)
        best_model = model_mentions[0] if model_mentions else None
        if not best_model:
            return None

        sentiment = min(0.5 + (score / 500), 1.0)
        rationale = f"Mentioned in r/{subreddit}: {score} upvotes, {num_comments} comments"

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
            memory_gb=self._estimate_memory(best_model),
            context_len=4096,
            version="latest",
            last_updated=datetime.now().isoformat(),
            hard_data_score=0.3,
            benchmark_score=0.4,
            sentiment_score=sentiment,
            provenance=[
                {
                    "source_type": "reddit",
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                    "last_checked": datetime.now().isoformat(),
                    "confidence": 0.6,
                }
            ],
            short_rationale=rationale,
        )

    def _extract_model_mentions(self, text: str) -> list[str]:
        known_models = [
            "Qwen",
            "Llama",
            "Mistral",
            "DeepSeek",
            "CodeLlama",
            "Gemma",
            "Phi",
            "Yi",
            "StarCoder",
            "Mixtral",
            "Command-R",
            "Haiku",
        ]

        found = []
        text_lower = text.lower()
        for model in known_models:
            if model.lower() in text_lower:
                found.append(model)

        return found

    def _estimate_memory(self, model_name: str) -> int:
        model_lower = model_name.lower()

        if "70b" in model_lower or "65b" in model_lower:
            return 80
        elif "34b" in model_lower or "30b" in model_lower:
            return 32
        elif "13b" in model_lower:
            return 16
        elif "8b" in model_lower:
            return 8
        elif "3b" in model_lower:
            return 4
        else:
            return 4


class GitHubAdapter:
    def __init__(self):
        self.token = credential_manager.get_token("github")
        self.api_url = GITHUB_API_URL
        self.session = requests.Session()

        if self.token:
            self.session.headers.update({"Authorization": f"token {self.token}"})
        self.session.headers.update({"User-Agent": "Vetinari/1.0"})

    def search_repos(self, query: str, limit: int = 10) -> list[ModelCandidate]:
        candidates = []

        try:
            params = {"q": f"{query} language:python stars:>100", "per_page": limit, "sort": "stars"}

            response = self.session.get(f"{self.api_url}/search/repositories", params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                repos = data.get("items", [])

                for repo in repos[:limit]:
                    try:
                        candidate = self._parse_repo(repo)
                        candidates.append(candidate)
                    except Exception as e:
                        logger.warning(f"Error parsing GitHub repo: {e}")

        except Exception as e:
            logger.error(f"GitHub search error: {e}")

        return candidates

    def _parse_repo(self, repo: dict) -> ModelCandidate:
        full_name = repo.get("full_name", "")
        name = full_name.split("/")[-1] if "/" in full_name else repo.get("name", "")
        stars = repo.get("stargazers_count", 0)
        updated = repo.get("updated_at", "")

        rationale = f"GitHub repo with {stars} stars, updated {updated[:10] if updated else 'recently'}"

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
            provenance=[
                {
                    "source_type": "github",
                    "url": repo.get("html_url", ""),
                    "last_checked": datetime.now().isoformat(),
                    "confidence": 0.7,
                }
            ],
            short_rationale=rationale,
        )


class PapersWithCodeAdapter:
    def __init__(self):
        self.api_url = PWC_API_URL
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Vetinari/1.0"})

    def search_papers(self, query: str, limit: int = 10) -> list[ModelCandidate]:
        candidates = []

        try:
            params = {"search": query}

            response = self.session.get(f"{self.api_url}/papers/", params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()
                papers = data.get("results", [])

                for paper in papers[:limit]:
                    try:
                        candidate = self._parse_paper(paper)
                        candidates.append(candidate)
                    except Exception as e:
                        logger.warning(f"Error parsing PWC paper: {e}")

        except Exception as e:
            logger.error(f"PapersWithCode search error: {e}")

        return candidates

    def _parse_paper(self, paper: dict) -> ModelCandidate:
        title = paper.get("title", "")
        arxiv_id = paper.get("arxiv_id", "")
        abstract = paper.get("abstract", "")[:200]

        benchmarks = paper.get("benchmarks", [])

        rationale = "Paper on PapersWithCode"
        if benchmarks:
            rationale += f", {len(benchmarks)} benchmarks"

        return ModelCandidate(
            id=f"pwcode/{arxiv_id}" if arxiv_id else f"pwcode/{title[:20]}",
            name=title[:40],
            source_type="paperswithcode",
            metrics={"title": title, "abstract": abstract, "benchmarks_count": len(benchmarks)},
            memory_gb=4,
            context_len=8192,
            version="latest",
            last_updated=paper.get("published", ""),
            hard_data_score=0.9,
            benchmark_score=0.85,
            sentiment_score=0.7,
            provenance=[
                {
                    "source_type": "paperswithcode",
                    "url": f"https://paperswithcode.com/paper/{arxiv_id}" if arxiv_id else "",
                    "last_checked": datetime.now().isoformat(),
                    "confidence": 0.9,
                }
            ],
            short_rationale=rationale,
        )


class LiveModelSearchAdapter:
    WEIGHTS = {"hard_data": 0.55, "benchmarks": 0.25, "sentiment": 0.15, "recency": 0.05}

    def __init__(self):
        self.hf_adapter = HuggingFaceAdapter()
        self.reddit_adapter = RedditAdapter()
        self.github_adapter = GitHubAdapter()
        self.pwc_adapter = PapersWithCodeAdapter()

    def search(self, query: str, lm_studio_models: list[dict] | None = None) -> list[ModelCandidate]:
        all_candidates = []

        if lm_studio_models:
            for model in lm_studio_models:
                candidate = self._create_lmstudio_candidate(model)
                all_candidates.append(candidate)

        logger.info(f"Searching for: {query}")

        hf_candidates = self.hf_adapter.search_models(query, limit=8)
        all_candidates.extend(hf_candidates)
        logger.info(f"HF found {len(hf_candidates)} candidates")

        reddit_candidates = self.reddit_adapter.search_local_llm_posts(query, limit=5)
        all_candidates.extend(reddit_candidates)
        logger.info(f"Reddit found {len(reddit_candidates)} candidates")

        github_candidates = self.github_adapter.search_repos(query, limit=5)
        all_candidates.extend(github_candidates)
        logger.info(f"GitHub found {len(github_candidates)} candidates")

        pwc_candidates = self.pwc_adapter.search_papers(query, limit=3)
        all_candidates.extend(pwc_candidates)
        logger.info(f"PapersWithCode found {len(pwc_candidates)} candidates")

        seen = set()
        unique_candidates = []
        for c in all_candidates:
            if c.id not in seen:
                seen.add(c.id)
                unique_candidates.append(c)

        for candidate in unique_candidates:
            candidate.final_score = self._calculate_score(candidate)

        unique_candidates.sort(key=lambda x: x.final_score, reverse=True)

        return unique_candidates[:15]

    def _create_lmstudio_candidate(self, model: dict) -> ModelCandidate:
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
                {
                    "source_type": "lm_studio",
                    "url": f"lmstudio://model/{model.get('id', '')}",
                    "last_checked": datetime.now().isoformat(),
                    "confidence": 0.9,
                }
            ],
            short_rationale="Local model available in LM Studio",
        )

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

        candidate.short_rationale = self._generate_rationale(candidate, recency_score)

        return round(final_score, 3)

    def _generate_rationale(self, candidate: ModelCandidate, recency_score: float) -> str:
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
