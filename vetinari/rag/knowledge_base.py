"""
Vetinari RAG Knowledge Base
=============================
Vector-backed knowledge base for Retrieval-Augmented Generation.

Agents query this to get relevant context before execution:
- Project documentation
- Past successful outputs
- Code patterns and templates
- Error resolution guides
- LLM best practices

Architecture
------------
- ChromaDB for vector storage (optional; falls back to simple text search)
- sentence-transformers for embeddings (CPU, no VRAM impact)
- Automatic document ingestion from project docs/ directory
- Context-window-aware retrieval (returns only what fits)

Usage::

    from vetinari.rag.knowledge_base import get_knowledge_base

    kb = get_knowledge_base()
    kb.ingest_directory("docs/")           # one-time ingestion

    results = kb.query(
        "How do I implement exponential backoff?",
        k=5,
        max_chars=3000,
    )
    context = "\n---\n".join(r.content for r in results)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CHROMA_DIR = os.environ.get(
    "VETINARI_CHROMA_DIR",
    str(Path.home() / ".lmstudio" / "projects" / "Vetinari" / ".vetinari" / "chroma"),
)


@dataclass
class KBDocument:
    """A document chunk in the knowledge base."""
    doc_id: str
    content: str
    source: str                    # File path or URL
    category: str = "general"     # docs / code / pattern / error / etc.
    score: float = 0.0            # Relevance score (populated by query)


class KnowledgeBase:
    """Vector-backed knowledge base with fallback to simple keyword search."""

    _instance: Optional["KnowledgeBase"] = None
    _cls_lock = threading.Lock()

    def __init__(self, persist_dir: str = _CHROMA_DIR):
        self._persist_dir = persist_dir
        self._chroma_available = False
        self._collection = None
        self._fallback_docs: List[Dict] = []  # Simple in-memory fallback
        self._lock = threading.RLock()
        self._init_chroma()

    @classmethod
    def get_instance(cls) -> "KnowledgeBase":
        with cls._cls_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_chroma(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
            client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = client.get_or_create_collection(
                name="vetinari_kb",
                metadata={"hnsw:space": "cosine"},
            )
            self._chroma_available = True
            logger.debug(f"[KnowledgeBase] ChromaDB initialized at {self._persist_dir}")
        except ImportError:
            logger.debug("[KnowledgeBase] ChromaDB not installed; using simple text search fallback")
        except Exception as e:
            logger.debug(f"[KnowledgeBase] ChromaDB init failed: {e}; using fallback")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_document(
        self,
        content: str,
        source: str,
        category: str = "general",
        doc_id: Optional[str] = None,
    ) -> str:
        """Add a document chunk to the knowledge base."""
        if not doc_id:
            doc_id = f"doc_{hashlib.md5(f'{source}:{content[:50]}'.encode()).hexdigest()[:8]}"

        with self._lock:
            if self._chroma_available and self._collection is not None:
                try:
                    self._collection.upsert(
                        ids=[doc_id],
                        documents=[content[:5000]],
                        metadatas=[{"source": source, "category": category}],
                    )
                    return doc_id
                except Exception as e:
                    logger.debug(f"[KnowledgeBase] ChromaDB upsert failed: {e}")

            # Fallback: in-memory list
            self._fallback_docs = [d for d in self._fallback_docs if d["id"] != doc_id]
            self._fallback_docs.append({
                "id": doc_id,
                "content": content[:5000],
                "source": source,
                "category": category,
            })

        return doc_id

    def ingest_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> int:
        """Ingest all documents from a directory. Returns count of chunks added."""
        extensions = extensions or [".md", ".txt", ".py", ".yaml", ".json"]
        base = Path(directory)
        if not base.exists():
            logger.debug(f"[KnowledgeBase] Directory {directory} not found")
            return 0

        count = 0
        for path in base.rglob("*"):
            if path.suffix.lower() not in extensions:
                continue
            if any(part.startswith(".") or part in ("__pycache__", "venv", "node_modules")
                   for part in path.parts):
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                chunks = self._chunk(content, chunk_size, chunk_overlap)
                for i, chunk in enumerate(chunks):
                    category = "code" if path.suffix == ".py" else "docs"
                    self.add_document(
                        content=chunk,
                        source=str(path),
                        category=category,
                        doc_id=f"doc_{hashlib.md5(f'{path}_{i}'.encode()).hexdigest()[:8]}",
                    )
                    count += 1
            except Exception as e:
                logger.debug(f"[KnowledgeBase] Skipped {path}: {e}")

        logger.info(f"[KnowledgeBase] Ingested {count} chunks from {directory}")
        return count

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        k: int = 5,
        max_chars: int = 3000,
        category: Optional[str] = None,
    ) -> List[KBDocument]:
        """Retrieve the k most relevant documents for a query.

        Args:
            query:      The query string
            k:          Number of results to return
            max_chars:  Maximum total characters in returned results
            category:   Optional category filter

        Returns:
            List of KBDocument sorted by relevance, truncated to max_chars total.
        """
        results: List[KBDocument] = []

        with self._lock:
            if self._chroma_available and self._collection is not None:
                results = self._query_chroma(query, k, category)
            else:
                results = self._query_fallback(query, k, category)

        # Respect character budget
        filtered: List[KBDocument] = []
        total_chars = 0
        for doc in results:
            if total_chars + len(doc.content) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    doc.content = doc.content[:remaining]
                    filtered.append(doc)
                break
            filtered.append(doc)
            total_chars += len(doc.content)

        return filtered

    def _query_chroma(
        self, query: str, k: int, category: Optional[str]
    ) -> List[KBDocument]:
        try:
            where = {"category": category} if category else None
            kwargs = {
                "query_texts": [query],
                "n_results": min(k, max(1, self._collection.count())),
            }
            if where:
                kwargs["where"] = where
            result = self._collection.query(**kwargs)

            docs = []
            for i, doc_content in enumerate(result.get("documents", [[]])[0]):
                meta = result.get("metadatas", [[]])[0][i] if result.get("metadatas") else {}
                dist = result.get("distances", [[]])[0][i] if result.get("distances") else 0.0
                docs.append(KBDocument(
                    doc_id=result.get("ids", [[]])[0][i] if result.get("ids") else "",
                    content=doc_content,
                    source=meta.get("source", ""),
                    category=meta.get("category", "general"),
                    score=max(0.0, 1.0 - float(dist)),
                ))
            return docs
        except Exception as e:
            logger.debug(f"[KnowledgeBase] ChromaDB query failed: {e}")
            return []

    def _query_fallback(
        self, query: str, k: int, category: Optional[str]
    ) -> List[KBDocument]:
        """Simple keyword-overlap search for when ChromaDB is unavailable."""
        query_words = set(query.lower().split())
        scored = []
        for doc in self._fallback_docs:
            if category and doc.get("category") != category:
                continue
            content_words = set(doc["content"].lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored.append((overlap, doc))
        scored.sort(key=lambda x: -x[0])
        return [
            KBDocument(
                doc_id=d["id"],
                content=d["content"],
                source=d["source"],
                category=d["category"],
                score=s / max(len(query_words), 1),
            )
            for s, d in scored[:k]
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk(text: str, size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= size:
            return [text]
        chunks = []
        i = 0
        while i < len(text):
            chunks.append(text[i:i + size])
            i += size - overlap
        return chunks

    def get_stats(self) -> Dict[str, Any]:
        count = 0
        if self._chroma_available and self._collection:
            try:
                count = self._collection.count()
            except Exception:
                pass
        else:
            count = len(self._fallback_docs)
        return {
            "document_count": count,
            "backend": "chromadb" if self._chroma_available else "in_memory",
            "persist_dir": self._persist_dir,
        }


# ---------------------------------------------------------------------------
# Module-level accessor
# ---------------------------------------------------------------------------

_kb: Optional[KnowledgeBase] = None
_kb_lock = threading.Lock()


def get_knowledge_base() -> KnowledgeBase:
    """Return the global KnowledgeBase singleton."""
    global _kb
    if _kb is None:
        with _kb_lock:
            if _kb is None:
                _kb = KnowledgeBase.get_instance()
    return _kb


def ingest_project_docs() -> int:
    """Convenience function to ingest all Vetinari project documentation."""
    kb = get_knowledge_base()
    project_root = Path(__file__).parent.parent.parent
    total = 0
    for d in ["docs", "skills", "system_prompts", "prompts"]:
        p = project_root / d
        if p.exists():
            total += kb.ingest_directory(str(p))
    logger.info(f"[KnowledgeBase] Project docs ingestion complete: {total} chunks")
    return total
