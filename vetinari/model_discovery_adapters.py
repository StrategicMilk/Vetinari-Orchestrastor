"""Model discovery source adapters — fetch model candidates from external APIs.

Each adapter queries one external source and returns a list of
``ModelCandidate`` objects. Adapters are used by ``ModelDiscovery`` in
``vetinari.model_discovery`` to fan out searches concurrently.

Sources covered:
- ``HuggingFaceAdapter`` — HuggingFace Hub API
- ``RedditAdapter`` — r/LocalLLaMA and related subreddits
- ``GitHubAdapter`` — GitHub repository search
- ``PapersWithCodeAdapter`` — PapersWithCode research papers
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

from vetinari.constants import MODEL_DISCOVERY_TIMEOUT
from vetinari.credentials import get_credential_manager
from vetinari.model_discovery import ModelCandidate
from vetinari.utils import estimate_model_memory_gb

logger = logging.getLogger(__name__)

HF_API_URL = "https://huggingface.co/api"
PWC_API_URL = "https://paperswithcode.com/api/v1"
GITHUB_API_URL = "https://api.github.com"

# Maps HuggingFace pipeline tags and model tags to user-facing use-case labels.
_HF_TAG_TO_USE: dict[str, str] = {
    "text-generation": "general",
    "text2text-generation": "general",
    "conversational": "chat",
    "image-text-to-text": "vision",
    "visual-question-answering": "vision",
    "image-to-text": "vision",
    "feature-extraction": "embeddings",
    "question-answering": "reasoning",
    "summarization": "documentation",
    "translation": "translation",
    "fill-mask": "general",
    "text-classification": "classification",
    "token-classification": "extraction",
    "code": "coding",
    "math": "reasoning",
}

# Tags found in HF model card tags (not pipeline tags) that hint at use-case.
_HF_KEYWORD_TO_USE: dict[str, str] = {
    "code": "coding",
    "coder": "coding",
    "math": "reasoning",
    "vision": "vision",
    "vl": "vision",
    "instruct": "general",
    "chat": "chat",
    "uncensored": "creative",
}


def _infer_uses_from_hf_tags(tags: list[str], pipeline_tag: str) -> list[str]:
    """Derive human-friendly use-case labels from HuggingFace metadata.

    Args:
        tags: Raw tag list from the HF model card (e.g. ``["transformers",
            "safetensors", "gemma4", "text-generation"]``).
        pipeline_tag: The HF pipeline tag (e.g. ``"text-generation"``).

    Returns:
        Deduplicated, sorted list of use-case strings like
        ``["chat", "general", "vision"]``.
    """
    uses: set[str] = set()

    # Map pipeline tag
    if pipeline_tag and pipeline_tag in _HF_TAG_TO_USE:
        uses.add(_HF_TAG_TO_USE[pipeline_tag])

    # Map raw tags
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in _HF_TAG_TO_USE:
            uses.add(_HF_TAG_TO_USE[tag_lower])
        # Check keyword substrings in tag names (e.g. "qwen2.5-coder" -> coding)
        for keyword, use in _HF_KEYWORD_TO_USE.items():
            if keyword in tag_lower:
                uses.add(use)

    # Fallback: if nothing matched, at least say "general" for text models
    if not uses and any(t in tags for t in ("transformers", "pytorch", "safetensors")):
        uses.add("general")

    return sorted(uses)


# Maps HuggingFace pipeline tags and model tags to user-facing use-case labels.
_HF_TAG_TO_USE: dict[str, str] = {
    "text-generation": "general",
    "text2text-generation": "general",
    "conversational": "chat",
    "image-text-to-text": "vision",
    "visual-question-answering": "vision",
    "image-to-text": "vision",
    "feature-extraction": "embeddings",
    "question-answering": "reasoning",
    "summarization": "documentation",
    "translation": "translation",
    "fill-mask": "general",
    "text-classification": "classification",
    "token-classification": "extraction",
    "code": "coding",
    "math": "reasoning",
}

# Tags found in HF model card tags (not pipeline tags) that hint at use-case.
_HF_KEYWORD_TO_USE: dict[str, str] = {
    "code": "coding",
    "coder": "coding",
    "math": "reasoning",
    "vision": "vision",
    "vl": "vision",
    "instruct": "general",
    "chat": "chat",
    "uncensored": "creative",
}


def _infer_uses_from_hf_tags(tags: list[str], pipeline_tag: str) -> list[str]:
    """Derive human-friendly use-case labels from HuggingFace metadata.

    Args:
        tags: Raw tag list from the HF model card (e.g. ``["transformers",
            "safetensors", "gemma4", "text-generation"]``).
        pipeline_tag: The HF pipeline tag (e.g. ``"text-generation"``).

    Returns:
        Deduplicated, sorted list of use-case strings like
        ``["chat", "general", "vision"]``.
    """
    uses: set[str] = set()

    # Map pipeline tag
    if pipeline_tag and pipeline_tag in _HF_TAG_TO_USE:
        uses.add(_HF_TAG_TO_USE[pipeline_tag])

    # Map raw tags
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in _HF_TAG_TO_USE:
            uses.add(_HF_TAG_TO_USE[tag_lower])
        # Check keyword substrings in tag names (e.g. "qwen2.5-coder" -> coding)
        for keyword, use in _HF_KEYWORD_TO_USE.items():
            if keyword in tag_lower:
                uses.add(use)

    # Fallback: if nothing matched, at least say "general" for text models
    if not uses and any(t in tags for t in ("transformers", "pytorch", "safetensors")):
        uses.add("general")

    return sorted(uses)


class HuggingFaceAdapter:
    """Searches HuggingFace Hub for model candidates matching a query string.

    Uses the HF API with an optional Bearer token for authenticated requests.
    Unauthenticated requests are rate-limited but still functional.
    """

    def __init__(self) -> None:
        self.api_url = HF_API_URL
        self.token = get_credential_manager().get_token("huggingface")
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        self.session.headers.update({"User-Agent": "Vetinari/1.0"})

    def search_models(self, query: str, limit: int = 10) -> list[ModelCandidate]:
        """Fetch model candidates from HuggingFace Hub matching the query.

        Args:
            query: Search query string.
            limit: Maximum number of candidates to return.

        Returns:
            List of ModelCandidate objects ranked by HF popularity signals.
        """
        candidates = []
        try:
            response = self.session.get(
                f"{self.api_url}/models",
                params={"search": query, "limit": limit, "full": "false"},
                timeout=MODEL_DISCOVERY_TIMEOUT,
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

    def _parse_model(self, model: dict) -> ModelCandidate:
        model_id = model.get("id", "")
        last_modified = model.get("lastModified", "")
        metrics = model.get("metrics", [])
        hard_data = 0.8 if metrics else 0.6
        rationale_parts = []
        if metrics:
            rationale_parts.append(f"Benchmarks: {len(metrics)} metrics available")
        rationale_parts.append(f"Updated: {last_modified[:10] if last_modified else 'unknown'}")
        tags = model.get("tags", [])
        recommended_for = _infer_uses_from_hf_tags(tags, model.get("pipeline_tag", ""))
        return ModelCandidate(
            id=model_id,
            name=model_id.split("/")[-1] if "/" in model_id else model_id,
            source_type="huggingface",
            recommended_for=recommended_for,
            metrics={
                "downloads": model.get("downloads", 0),
                "likes": model.get("likes", 0),
                "tags": tags[:5],
            },
            memory_gb=estimate_model_memory_gb(model_id),
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
                    "last_checked": datetime.now(timezone.utc).isoformat(),
                    "confidence": 0.9,
                },
            ],
            short_rationale=(
                f"HF model with {model.get('likes', 0)} likes, "
                f"{model.get('downloads', 0)} downloads. "
                f"{rationale_parts[0] if rationale_parts else ''}"
            ),
        )


class RedditAdapter:
    """Scrapes r/LocalLLaMA and related subreddits for model recommendations.

    Extracts model names from post titles and uses upvote/comment counts as
    a community sentiment signal.
    """

    def __init__(self) -> None:
        self.token = get_credential_manager().get_token("reddit")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Vetinari/1.0"})
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    def search_local_llm_posts(self, query: str, limit: int = 10) -> list[ModelCandidate]:
        """Search r/LocalLLaMA and related subreddits for model discussion posts.

        Args:
            query: Search query string.
            limit: Maximum number of candidates to return.

        Returns:
            List of ModelCandidate objects extracted from matching posts.
        """
        candidates: list[ModelCandidate] = []
        for subreddit in ["LocalLLaMA", "MachineLearning", "LLM", "LanguageTechnology"]:
            try:
                response = self.session.get(
                    f"https://www.reddit.com/r/{subreddit}/search.json",
                    params={"q": query, "limit": limit, "sort": "relevance", "t": "year"},
                    timeout=MODEL_DISCOVERY_TIMEOUT,
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

    def _parse_post(self, post: dict, subreddit: str) -> ModelCandidate | None:
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
            context_len=8192,
            version="latest",
            last_updated=datetime.now(timezone.utc).isoformat(),
            hard_data_score=0.3,
            benchmark_score=0.4,
            sentiment_score=sentiment,
            provenance=[
                {
                    "source_type": "reddit",
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                    "last_checked": datetime.now(timezone.utc).isoformat(),
                    "confidence": 0.6,
                },
            ],
            short_rationale=f"Mentioned in r/{subreddit}: {score} upvotes, {num_comments} comments",
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
        text_lower = text.lower()
        return [m for m in known_models if m.lower() in text_lower]


class GitHubAdapter:
    """Searches GitHub for model repositories using the GitHub REST API.

    Uses star count and fork count as proxy signals for model quality.
    """

    def __init__(self) -> None:
        self.token = get_credential_manager().get_token("github")
        self.api_url = GITHUB_API_URL
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"token {self.token}"})
        self.session.headers.update({"User-Agent": "Vetinari/1.0"})

    def search_repos(self, query: str, limit: int = 10) -> list[ModelCandidate]:
        """Search GitHub repositories for model implementations matching the query.

        Args:
            query: Search query string.
            limit: Maximum number of candidates to return.

        Returns:
            List of ModelCandidate objects built from repository metadata.
        """
        candidates = []
        try:
            response = self.session.get(
                f"{self.api_url}/search/repositories",
                params={"q": f"{query} language:python stars:>100", "per_page": limit, "sort": "stars"},
                timeout=MODEL_DISCOVERY_TIMEOUT,
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

    def _parse_repo(self, repo: dict) -> ModelCandidate:
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
            context_len=8192,
            version="latest",
            last_updated=updated,
            hard_data_score=0.4,
            benchmark_score=0.5,
            sentiment_score=min(0.4 + (stars / 5000), 1.0),
            provenance=[
                {
                    "source_type": "github",
                    "url": repo.get("html_url", ""),
                    "last_checked": datetime.now(timezone.utc).isoformat(),
                    "confidence": 0.7,
                },
            ],
            short_rationale=f"GitHub repo with {stars} stars, updated {updated[:10] if updated else 'recently'}",
        )


class PapersWithCodeAdapter:
    """Searches PapersWithCode for peer-reviewed model papers.

    Papers with benchmark results get higher hard_data_score than community sources.
    """

    def __init__(self) -> None:
        self.api_url = PWC_API_URL
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Vetinari/1.0"})

    def search_papers(self, query: str, limit: int = 10) -> list[ModelCandidate]:
        """Search PapersWithCode for model papers matching the query.

        Args:
            query: Search query string.
            limit: Maximum number of candidates to return.

        Returns:
            List of ModelCandidate objects with benchmark-backed hard_data_score.
        """
        candidates = []
        try:
            response = self.session.get(
                f"{self.api_url}/papers/", params={"search": query}, timeout=MODEL_DISCOVERY_TIMEOUT
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

    def _parse_paper(self, paper: dict) -> ModelCandidate:
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
            provenance=[
                {
                    "source_type": "paperswithcode",
                    "url": f"https://paperswithcode.com/paper/{arxiv_id}" if arxiv_id else "",
                    "last_checked": datetime.now(timezone.utc).isoformat(),
                    "confidence": 0.9,
                },
            ],
            short_rationale=rationale,
        )
