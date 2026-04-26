"""Retrieval-Augmented Generation: knowledge base, chunking, and vector search."""

from __future__ import annotations

from vetinari.rag.knowledge_base import (  # noqa: VET123 — ingest_project_docs has no external callers but removing causes VET120
    KBDocument,
    KnowledgeBase,
    get_knowledge_base,
    ingest_project_docs,
)

__all__ = [
    "KBDocument",
    "KnowledgeBase",
    "get_knowledge_base",
    "ingest_project_docs",
]
