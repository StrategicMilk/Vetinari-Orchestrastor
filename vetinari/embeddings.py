"""Unified embedding singleton for Vetinari.

Provides a single ``get_embedder()`` function that returns a shared
``SentenceTransformerEmbedder`` instance backed by ``all-MiniLM-L6-v2``
(384-dimensional, CPU, no GPU required).  When ``sentence-transformers``
is not installed, falls back to a deterministic n-gram hash embedder that
always produces 384-dimensional unit vectors.

Decision: replace scattered LLM-based embedding calls with a single
lightweight sentence-transformers model (ADR-0078).
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Dimensionality contract — callers depend on this being 384.
EMBEDDING_DIMENSIONS = 384
_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# N-gram fallback embedder
# ---------------------------------------------------------------------------


def _ngram_hash_embed(text: str, dims: int = EMBEDDING_DIMENSIONS) -> list[float]:
    """Produce a deterministic unit vector via character n-gram hashing.

    Uses SHA-256 seeded with overlapping 4-grams to fill a float vector,
    then L2-normalises.  Always returns a ``dims``-dimensional vector.

    Args:
        text: Input string to embed.
        dims: Number of output dimensions (must match EMBEDDING_DIMENSIONS).

    Returns:
        A normalised list of ``dims`` floats.
    """
    vec = [0.0] * dims
    if not text:
        return vec
    # 4-character n-grams; fallback to bigrams for very short text
    n = 4 if len(text) >= 4 else 2
    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)] or [text]
    for gram in ngrams:
        digest = hashlib.sha256(gram.encode("utf-8", errors="replace")).digest()
        # Unpack 8 floats per digest (32 bytes / 4 bytes per float = 8)
        floats = struct.unpack("8f", digest[:32])
        # Distribute across dims using modulo
        for j, v in enumerate(floats):
            vec[(j * len(ngrams) + ngrams.index(gram)) % dims] += v
    # L2 normalise
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


# ---------------------------------------------------------------------------
# Sentence-transformers embedder
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder:
    """Thin wrapper around ``sentence_transformers.SentenceTransformer``.

    Falls back to n-gram hashing when sentence-transformers is unavailable.
    All output vectors are 384-dimensional and L2-normalised.

    Model loading is deferred until first use (lazy initialization) to avoid
    blocking during import. This is critical for test startup performance.
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._available: bool | None = None  # None = not yet tried, True/False = result
        self._lock = threading.Lock()

    def _try_load(self) -> None:
        """Attempt to load the sentence-transformers model (lazy, on first use)."""
        if self._available is not None:
            return  # Already attempted
        with self._lock:
            if self._available is not None:
                return  # Already attempted (double-checked locking)
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import]

                # device="cpu" — no GPU dependency; model is small enough for CPU
                self._model = SentenceTransformer(self._model_name, device="cpu", local_files_only=True)
                self._available = True
                logger.info("sentence-transformers embedder loaded: %s (384-dim)", self._model_name)
            except Exception as exc:
                logger.warning(
                    "sentence-transformers unavailable (%s) — using n-gram fallback embedder",
                    exc,
                )
                self._available = False

    @property
    def available(self) -> bool:
        """Return True when the sentence-transformers model is loaded.

        Triggers lazy model loading on first access.
        """
        self._try_load()
        return self._available or False

    def embed(self, text: str) -> list[float]:
        """Embed a single string into a 384-dimensional unit vector.

        Args:
            text: Input text to embed.

        Returns:
            384-dimensional L2-normalised float list.
        """
        if not text:
            return [0.0] * EMBEDDING_DIMENSIONS
        if self._available and self._model is not None:
            try:
                vec = self._model.encode(text, normalize_embeddings=True, show_progress_bar=False)
                return [float(x) for x in vec]
            except Exception as exc:
                logger.warning("sentence-transformers encode failed: %s; using fallback", exc)
        return _ngram_hash_embed(text, EMBEDDING_DIMENSIONS)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings efficiently.

        Args:
            texts: List of input strings to embed.

        Returns:
            List of 384-dimensional unit vectors, one per input string.
        """
        if not texts:
            return []
        if self._available and self._model is not None:
            try:
                vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                return [[float(x) for x in v] for v in vecs]
            except Exception as exc:
                logger.warning("sentence-transformers batch encode failed: %s; using fallback", exc)
        return [_ngram_hash_embed(t, EMBEDDING_DIMENSIONS) for t in texts]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_embedder: SentenceTransformerEmbedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> SentenceTransformerEmbedder:
    """Return the process-wide embedding singleton (thread-safe).

    Uses double-checked locking so the model is loaded at most once.

    Returns:
        The shared :class:`SentenceTransformerEmbedder` instance.
    """
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = SentenceTransformerEmbedder()
    return _embedder
