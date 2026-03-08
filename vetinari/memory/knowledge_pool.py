"""
Cross-Agent Knowledge Pool
============================

A shared knowledge store where agents can publish discovered solutions,
patterns, code snippets, and API usage for other agents to query before
generating new solutions from scratch.

Usage:
    from vetinari.memory.knowledge_pool import KnowledgePool, get_knowledge_pool

    pool = get_knowledge_pool()
    entry_id = pool.share("BUILDER", "api_pattern", {"pattern": "retry with backoff"})
    results = pool.query("how to handle API retries", limit=5)
"""

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_POOL_PATH = ".vetinari/knowledge_pool.json"


@dataclass
class KnowledgeEntry:
    """A single knowledge entry shared by an agent."""
    entry_id: str = field(default_factory=lambda: f"kp_{uuid.uuid4().hex[:8]}")
    agent_type: str = ""
    key: str = ""                          # Semantic key (e.g., "retry_pattern")
    knowledge: Any = None                  # The actual knowledge (dict, str, code snippet, etc.)
    category: str = "general"             # "pattern", "snippet", "api_usage", "solution", "warning"
    tags: List[str] = field(default_factory=list)
    description: str = ""                 # Human-readable description
    confidence: float = 0.8              # How confident the sharing agent is
    usage_count: int = 0                  # How many times this was retrieved
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    content_hash: str = ""                # Deduplication hash
    superseded_by: str = ""              # If this entry was replaced by a newer one
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Ensure knowledge is serializable
        if not isinstance(self.knowledge, (str, int, float, bool, list, dict, type(None))):
            d["knowledge"] = str(self.knowledge)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class KnowledgePool:
    """
    Shared knowledge pool for cross-agent solution reuse.

    Agents publish discoveries, patterns, and solutions into the pool.
    Before generating new solutions, agents query the pool for existing
    knowledge that might be applicable.

    Features:
    - Content deduplication via hashing
    - Keyword and tag-based search
    - Usage tracking (popular entries are ranked higher)
    - Persistence to JSON file
    - Supersession: newer entries can replace older ones
    - Category-based filtering
    """

    def __init__(self, pool_path: Optional[str] = None, auto_persist: bool = True):
        """
        Args:
            pool_path: Path to the JSON persistence file.
            auto_persist: Whether to auto-save after modifications.
        """
        self._path = Path(pool_path or DEFAULT_POOL_PATH)
        self._auto_persist = auto_persist
        self._lock = threading.Lock()
        self._entries: Dict[str, KnowledgeEntry] = {}
        self._hash_index: Dict[str, str] = {}  # content_hash -> entry_id
        self._tag_index: Dict[str, List[str]] = {}  # tag -> [entry_ids]

        self._load()
        logger.info("KnowledgePool initialized: %d entries (path=%s)", len(self._entries), self._path)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def share(
        self,
        agent_type: str,
        key: str,
        knowledge: Any,
        category: str = "general",
        tags: Optional[List[str]] = None,
        description: str = "",
        confidence: float = 0.8,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Share knowledge into the pool.

        Args:
            agent_type: The agent sharing the knowledge.
            key: Semantic key for the knowledge (e.g., "retry_pattern").
            knowledge: The knowledge payload (dict, str, code snippet, etc.).
            category: Category like "pattern", "snippet", "api_usage", "solution".
            tags: Searchable tags.
            description: Human-readable description.
            confidence: How confident the agent is (0.0 to 1.0).
            metadata: Additional metadata.

        Returns:
            entry_id: Unique identifier for the stored knowledge.
        """
        with self._lock:
            # Compute content hash for deduplication
            content_str = json.dumps({"key": key, "knowledge": knowledge}, sort_keys=True, default=str)
            content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]

            # Check for duplicates
            existing_id = self._hash_index.get(content_hash)
            if existing_id and existing_id in self._entries:
                existing = self._entries[existing_id]
                # Update usage and confidence if duplicate
                if confidence > existing.confidence:
                    existing.confidence = confidence
                    existing.updated_at = datetime.now(timezone.utc).isoformat()
                logger.debug("Duplicate knowledge detected, updating entry %s", existing_id)
                if self._auto_persist:
                    self._save()
                return existing_id

            entry = KnowledgeEntry(
                agent_type=agent_type,
                key=key,
                knowledge=knowledge,
                category=category,
                tags=tags or [],
                description=description or key,
                confidence=confidence,
                content_hash=content_hash,
                metadata=metadata or {},
            )

            self._entries[entry.entry_id] = entry
            self._hash_index[content_hash] = entry.entry_id

            # Update tag index
            for tag in entry.tags:
                tag_lower = tag.lower()
                if tag_lower not in self._tag_index:
                    self._tag_index[tag_lower] = []
                self._tag_index[tag_lower].append(entry.entry_id)

            if self._auto_persist:
                self._save()

            logger.info(
                "Knowledge shared: id=%s agent=%s key=%s category=%s",
                entry.entry_id, agent_type, key, category,
            )
            return entry.entry_id

    def query(
        self,
        query_text: str,
        limit: int = 5,
        category: Optional[str] = None,
        agent_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> List[KnowledgeEntry]:
        """
        Query the knowledge pool for relevant entries.

        Uses keyword matching against keys, descriptions, tags, and knowledge
        content. Results are ranked by relevance score and usage popularity.

        Args:
            query_text: Natural language or keyword query.
            limit: Maximum number of results.
            category: Filter by category.
            agent_type: Filter by sharing agent type.
            tags: Filter by tags (any match).
            min_confidence: Minimum confidence threshold.

        Returns:
            List of matching KnowledgeEntry objects, ranked by relevance.
        """
        with self._lock:
            candidates = list(self._entries.values())

            # Filter out superseded entries
            candidates = [e for e in candidates if not e.superseded_by]

            # Apply filters
            if category:
                candidates = [e for e in candidates if e.category == category]
            if agent_type:
                candidates = [e for e in candidates if e.agent_type == agent_type]
            if min_confidence > 0:
                candidates = [e for e in candidates if e.confidence >= min_confidence]
            if tags:
                tag_set = {t.lower() for t in tags}
                candidates = [e for e in candidates if tag_set & {t.lower() for t in e.tags}]

            # Score candidates by keyword relevance
            query_words = set(query_text.lower().split())
            scored: List[tuple] = []

            for entry in candidates:
                score = self._compute_relevance(entry, query_words)
                if score > 0:
                    scored.append((score, entry))

            # Sort by score (descending), then by usage count
            scored.sort(key=lambda x: (x[0], x[1].usage_count), reverse=True)

            # Update usage counts for returned entries
            results = []
            for score, entry in scored[:limit]:
                entry.usage_count += 1
                results.append(entry)

            if results and self._auto_persist:
                self._save()

            logger.debug("Knowledge query '%s': %d results from %d candidates",
                         query_text[:50], len(results), len(candidates))
            return results

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Retrieve a specific knowledge entry by ID."""
        return self._entries.get(entry_id)

    def supersede(self, old_entry_id: str, new_entry_id: str) -> bool:
        """
        Mark an entry as superseded by a newer one.

        The old entry remains queryable but is ranked lower.

        Returns:
            True if the supersession was recorded.
        """
        with self._lock:
            old = self._entries.get(old_entry_id)
            new = self._entries.get(new_entry_id)
            if old is None or new is None:
                logger.warning("Cannot supersede: entry not found")
                return False

            old.superseded_by = new_entry_id
            old.updated_at = datetime.now(timezone.utc).isoformat()

            if self._auto_persist:
                self._save()

            logger.info("Entry %s superseded by %s", old_entry_id, new_entry_id)
            return True

    def remove(self, entry_id: str) -> bool:
        """Remove an entry from the pool."""
        with self._lock:
            entry = self._entries.pop(entry_id, None)
            if entry is None:
                return False

            self._hash_index.pop(entry.content_hash, None)
            for tag in entry.tags:
                tag_lower = tag.lower()
                if tag_lower in self._tag_index:
                    self._tag_index[tag_lower] = [
                        eid for eid in self._tag_index[tag_lower] if eid != entry_id
                    ]

            if self._auto_persist:
                self._save()

            logger.info("Knowledge entry removed: %s", entry_id)
            return True

    def list_by_category(self, category: str) -> List[KnowledgeEntry]:
        """List all entries in a category."""
        with self._lock:
            return [e for e in self._entries.values() if e.category == category and not e.superseded_by]

    def list_by_agent(self, agent_type: str) -> List[KnowledgeEntry]:
        """List all entries shared by a specific agent type."""
        with self._lock:
            return [e for e in self._entries.values() if e.agent_type == agent_type]

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            entries = list(self._entries.values())
            categories = {}
            agents = {}
            for e in entries:
                categories[e.category] = categories.get(e.category, 0) + 1
                agents[e.agent_type] = agents.get(e.agent_type, 0) + 1

            return {
                "total_entries": len(entries),
                "active_entries": sum(1 for e in entries if not e.superseded_by),
                "superseded_entries": sum(1 for e in entries if e.superseded_by),
                "total_queries_served": sum(e.usage_count for e in entries),
                "by_category": categories,
                "by_agent": agents,
                "unique_tags": len(self._tag_index),
            }

    # ------------------------------------------------------------------
    # Relevance scoring
    # ------------------------------------------------------------------

    def _compute_relevance(self, entry: KnowledgeEntry, query_words: set) -> float:
        """Compute a relevance score for an entry against query words."""
        score = 0.0

        # Match against key
        key_words = set(entry.key.lower().replace("_", " ").split())
        key_overlap = len(query_words & key_words)
        score += key_overlap * 3.0

        # Match against description
        desc_words = set(entry.description.lower().split())
        desc_overlap = len(query_words & desc_words)
        score += desc_overlap * 2.0

        # Match against tags
        tag_words = {t.lower() for t in entry.tags}
        tag_overlap = len(query_words & tag_words)
        score += tag_overlap * 2.5

        # Match against knowledge content (if string)
        if isinstance(entry.knowledge, str):
            content_words = set(entry.knowledge.lower().split()[:100])  # Cap for performance
            content_overlap = len(query_words & content_words)
            score += content_overlap * 1.0
        elif isinstance(entry.knowledge, dict):
            content_str = " ".join(str(v) for v in entry.knowledge.values())
            content_words = set(content_str.lower().split()[:100])
            content_overlap = len(query_words & content_words)
            score += content_overlap * 1.0

        # Boost by confidence and usage
        score *= (0.5 + entry.confidence * 0.5)
        score += min(entry.usage_count * 0.1, 2.0)  # Popularity boost, capped

        return score

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load entries from disk."""
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry_data in data.get("entries", []):
                entry = KnowledgeEntry.from_dict(entry_data)
                self._entries[entry.entry_id] = entry
                if entry.content_hash:
                    self._hash_index[entry.content_hash] = entry.entry_id
                for tag in entry.tags:
                    tag_lower = tag.lower()
                    if tag_lower not in self._tag_index:
                        self._tag_index[tag_lower] = []
                    self._tag_index[tag_lower].append(entry.entry_id)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Failed to load knowledge pool from %s: %s", self._path, exc)

    def _save(self) -> None:
        """Persist entries to disk."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": "1.0",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "entries": [e.to_dict() for e in self._entries.values()],
            }
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as exc:
            logger.error("Failed to save knowledge pool: %s", exc)


# ------------------------------------------------------------------
# Singleton accessor
# ------------------------------------------------------------------

_pool_instance: Optional[KnowledgePool] = None
_pool_lock = threading.Lock()


def get_knowledge_pool() -> KnowledgePool:
    """Get or create the global KnowledgePool singleton."""
    global _pool_instance
    if _pool_instance is None:
        with _pool_lock:
            if _pool_instance is None:
                _pool_instance = KnowledgePool()
    return _pool_instance


def reset_knowledge_pool() -> None:
    """Reset the knowledge pool singleton (mainly for testing)."""
    global _pool_instance
    _pool_instance = None
