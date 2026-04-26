"""Entailment Checker — Tier 2 of the verification cascade.

Checks whether a response semantically entails the task requirements using
lightweight NLP heuristics (keyword overlap, requirement coverage) — no LLM
needed for most cases.

Pipeline role: Called by CascadeOrchestrator when Tier 1 (StaticVerifier)
passes but score confidence is still uncertain.  Avoids an LLM call by using
token overlap and structural coverage as a proxy for entailment.

When sentence-transformers is available, an optional cosine-similarity check
is added for higher accuracy (see _semantic_similarity).
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Stop-words filtered out during keyword extraction — common English words that carry no domain signal
_STOP_WORDS = frozenset({
    "that",
    "this",
    "with",
    "from",
    "have",
    "will",
    "should",
    "would",
    "could",
    "their",
    "there",
    "which",
    "when",
    "then",
    "than",
    "also",
    "into",
    "more",
    "some",
    "each",
    "been",
    "were",
    "they",
    "them",
    "make",
    "made",
    "does",
    "where",
    "what",
    "your",
    "just",
    "only",
    "very",
    "about",
    "after",
})

# Minimum fraction of task requirement keywords that must appear in the response
_MIN_KEYWORD_COVERAGE = 0.4  # 40 % coverage required for PASS

# Cosine similarity threshold when sentence-transformers is available
_SEMANTIC_SIMILARITY_THRESHOLD = 0.55


@dataclass
class EntailmentResult:
    """Result of the entailment check.

    Attributes:
        entailed: True when the response adequately covers the task requirements.
        coverage: Fraction of task keywords found in the response (0.0-1.0).
        similarity: Cosine similarity score if sentence-transformers was used,
            otherwise None.
        missing_keywords: Keywords from the task that were absent in the response.
    """

    entailed: bool
    coverage: float
    similarity: float | None = None
    missing_keywords: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        """Show key fields for debugging."""
        return (
            f"EntailmentResult(entailed={self.entailed!r}, "
            f"coverage={self.coverage:.3f}, similarity={self.similarity!r})"
        )


class EntailmentChecker:
    """Tier 2 verifier — semantic coverage check without LLM calls.

    Uses keyword overlap as the primary signal and optional sentence-transformer
    cosine similarity as a secondary signal when the library is installed.

    Example::

        checker = EntailmentChecker()
        result = checker.check(
            task_description="Implement a binary search function",
            response_text="def binary_search(arr, target): ...",
        )
        assert result.entailed
    """

    def check(self, task_description: str, response_text: str) -> EntailmentResult:
        """Check whether *response_text* entails the requirements in *task_description*.

        Args:
            task_description: The original task or requirement text.
            response_text: The response to evaluate.

        Returns:
            :class:`EntailmentResult` with entailment verdict and evidence scores.
        """
        if not task_description or not response_text:
            return EntailmentResult(entailed=False, coverage=0.0, missing_keywords=[])

        # ── Step 1: keyword coverage ─────────────────────────────────────────
        task_keywords = self._extract_keywords(task_description)
        resp_lower = response_text.lower()

        if not task_keywords:
            # No keywords extracted means we cannot verify the response satisfies the task.
            # Returning entailed=True here would certify arbitrary content as valid —
            # the "default-pass verifier" anti-pattern.  Return score=0.0 instead.
            return EntailmentResult(entailed=False, coverage=0.0, missing_keywords=[])

        # Use whole-word matching to prevent substring false positives (e.g. "search"
        # must NOT match inside "research").  Two complementary strategies:
        #
        # 1. Word-boundary regex (\b) — catches keywords in prose where word boundaries
        #    are whitespace / punctuation.
        # 2. Identifier-token set — splits the response on non-alpha chars so that
        #    code identifiers like "binary_search" contribute tokens "binary" and
        #    "search".  This prevents \b from missing keywords embedded in snake_case
        #    or camelCase identifiers while still rejecting "research" → "search".
        resp_tokens: set[str] = set(re.split(r"[^a-z]+", resp_lower))
        resp_tokens.discard("")

        def _matches(kw: str) -> bool:
            return bool(re.search(rf"\b{re.escape(kw)}\b", resp_lower)) or kw in resp_tokens

        found = [kw for kw in task_keywords if _matches(kw)]
        missing = [kw for kw in task_keywords if not _matches(kw)]
        coverage = len(found) / len(task_keywords)

        logger.debug(
            "EntailmentChecker: coverage=%.3f (%d/%d keywords found)",
            coverage,
            len(found),
            len(task_keywords),
        )

        # ── Step 2: optional semantic similarity ─────────────────────────────
        similarity: float | None = None
        try:
            similarity = self._semantic_similarity(task_description, response_text)
            if similarity is not None:
                logger.debug("EntailmentChecker: semantic similarity=%.3f", similarity)
        except Exception as exc:
            logger.warning(
                "EntailmentChecker: semantic similarity check failed (%s) — falling back to keyword coverage only", exc
            )

        # ── Step 3: verdict ──────────────────────────────────────────────────
        if similarity is not None:
            # When we have semantic similarity, require BOTH coverage and similarity
            entailed = coverage >= _MIN_KEYWORD_COVERAGE and similarity >= _SEMANTIC_SIMILARITY_THRESHOLD
        else:
            entailed = coverage >= _MIN_KEYWORD_COVERAGE

        return EntailmentResult(
            entailed=entailed,
            coverage=round(coverage, 3),
            similarity=round(similarity, 3) if similarity is not None else None,
            missing_keywords=missing,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful content words from *text* for overlap checking.

        Filters out stop-words and short tokens to focus on domain vocabulary.

        Args:
            text: Input text to tokenise.

        Returns:
            Deduplicated list of lowercase content words (length >= 4).
        """
        words = re.findall(r"\b[a-z]{4,}\b", text.lower())
        return list(dict.fromkeys(w for w in words if w not in _STOP_WORDS))

    def _semantic_similarity(self, text_a: str, text_b: str) -> float | None:
        """Compute cosine similarity between two texts using sentence-transformers.

        Returns None when the library is not installed or an error occurs —
        callers fall back to keyword coverage only.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Cosine similarity in [0.0, 1.0] or None if unavailable.
        """
        model = _get_st_model()
        if model is None:
            return None
        try:
            from sentence_transformers import util  # type: ignore[import]
        except ImportError:
            logger.debug("EntailmentChecker: sentence_transformers.util not available — skipping similarity")
            return None
        try:
            embeddings = model.encode([text_a[:512], text_b[:512]], convert_to_tensor=True)
            score = float(util.cos_sim(embeddings[0], embeddings[1]))
            return max(0.0, min(1.0, score))
        except Exception as exc:
            logger.warning(
                "EntailmentChecker: similarity computation failed (%s) — falling back to keyword coverage",
                exc,
            )
            return None


# ── sentence-transformers singleton ──────────────────────────────────────────

_st_model = None
_st_model_loaded = False  # True only when a real SentenceTransformer was successfully cached
_st_lock = threading.Lock()


def _get_st_model() -> object | None:
    """Lazily load and cache a lightweight sentence-transformers model.

    Uses double-checked locking so concurrent first calls are safe.  Returns
    None (and logs a debug message) when sentence-transformers is not installed
    or failed to load.  After the singleton is initialized, no further imports
    of sentence_transformers are performed — this avoids torch re-import errors
    in test environments where torch was partially loaded by an earlier import.

    Returns:
        A ``SentenceTransformer`` instance or None if unavailable.
    """
    global _st_model, _st_model_loaded
    if _st_model is None:
        with _st_lock:
            if _st_model is None:
                try:
                    from sentence_transformers import SentenceTransformer  # type: ignore[import]

                    # Use local_files_only to avoid network downloads in production;
                    # falls back gracefully if model not cached locally.
                    _st_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
                    _st_model_loaded = True
                    logger.info("EntailmentChecker: loaded sentence-transformers model all-MiniLM-L6-v2")
                except Exception as exc:
                    logger.warning(
                        "EntailmentChecker: sentence-transformers load failed (%s) — keyword-only mode",
                        exc,
                    )
                    # Cache a sentinel to avoid retrying on every call
                    _st_model = object()
                    _st_model_loaded = False
    # Return the real model only when load succeeded; sentinel maps to None
    return _st_model if _st_model_loaded else None
