"""DualMemoryStore - Coordinates between OcMemoryStore and MnemosyneMemoryStore.

This store writes to both backends and merges reads according to the
merge policy defined in docs/memory_merge_policy.md.

Security: All entries are scanned for secrets before being written to backends.
"""

from __future__ import annotations

import functools
import logging
import operator
from datetime import datetime

from vetinari.security import get_secret_scanner

from .interfaces import (
    MEMORY_DEDUP_ENABLED,
    MEMORY_PRIMARY_READ,
    IMemoryStore,
    MemoryEntry,
    MemoryStats,
    content_hash,
)
from .mnemosyne_memory import MnemosyneMemoryStore
from .oc_memory import OcMemoryStore

logger = logging.getLogger(__name__)


class DualMemoryStore(IMemoryStore):
    """Coordinates between OcMemoryStore and MnemosyneMemoryStore.

    Write behavior: writes to both backends simultaneously
    Read behavior: merges results from both backends with deduplication
    """

    def __init__(self, oc_path: str | None = None, mnemosyne_path: str | None = None):
        self.oc_store = OcMemoryStore(oc_path) if oc_path else OcMemoryStore()
        self.mnemo_store = MnemosyneMemoryStore(mnemosyne_path) if mnemosyne_path else MnemosyneMemoryStore()
        self.primary_backend = MEMORY_PRIMARY_READ
        self.dedup_enabled = MEMORY_DEDUP_ENABLED

        logger.info("DualMemoryStore initialized with primary=%s", self.primary_backend)

    def remember(self, entry: MemoryEntry) -> str:
        """Write to both backends with secret sanitization."""
        # Filter secrets before persisting
        entry = self._filter_secrets(entry)
        entry.source_backends = []

        oc_id = None
        mnemo_id = None

        try:
            oc_id = self.oc_store.remember(entry)
            entry.source_backends.append("oc")
        except Exception as e:
            logger.warning("OcMemoryStore write failed: %s", e)

        try:
            mnemo_id = self.mnemo_store.remember(entry)
            entry.source_backends.append("mnemosyne")
        except Exception as e:
            logger.warning("MnemosyneStore write failed: %s", e)

        if not entry.source_backends:
            raise RuntimeError("Both backends failed to write")

        primary_id = oc_id if oc_id else mnemo_id
        logger.debug("DualMemoryStore: wrote entry to %s", entry.source_backends)

        return primary_id

    def _filter_secrets(self, entry: MemoryEntry) -> MemoryEntry:
        """Sanitize entry content to remove secrets before storage."""
        scanner = get_secret_scanner()

        # Sanitize the main content
        if entry.content:
            sanitized_content = scanner.sanitize(entry.content)
            if sanitized_content != entry.content:
                logger.debug("Entry content contained secrets - sanitized")
                entry.content = sanitized_content

        # Sanitize metadata if it's a dictionary
        if entry.metadata and isinstance(entry.metadata, dict):
            sanitized_metadata = scanner.sanitize_dict(entry.metadata)
            if sanitized_metadata != entry.metadata:
                logger.debug("Entry metadata contained secrets - sanitized")
                entry.metadata = sanitized_metadata

        return entry

    def search(
        self, query: str, agent: str | None = None, entry_types: list[str] | None = None, limit: int = 10
    ) -> list[MemoryEntry]:
        """Search with merged results from both backends."""
        oc_results = []
        mnemo_results = []

        try:
            oc_results = self.oc_store.search(query, agent, entry_types, limit)
        except Exception as e:
            logger.warning("OcMemoryStore search failed: %s", e)

        try:
            mnemo_results = self.mnemo_store.search(query, agent, entry_types, limit)
        except Exception as e:
            logger.warning("MnemosyneStore search failed: %s", e)

        all_results = oc_results + mnemo_results

        if self.dedup_enabled:
            return self._merge_and_dedupe(all_results, limit)

        all_results.sort(key=lambda e: e.timestamp, reverse=True)
        return all_results[:limit]

    def timeline(
        self, agent: str | None = None, start_time: int | None = None, end_time: int | None = None, limit: int = 100
    ) -> list[MemoryEntry]:
        """Timeline with merged results from both backends."""
        oc_results = []
        mnemo_results = []

        try:
            oc_results = self.oc_store.timeline(agent, start_time, end_time, limit)
        except Exception as e:
            logger.warning("OcMemoryStore timeline failed: %s", e)

        try:
            mnemo_results = self.mnemo_store.timeline(agent, start_time, end_time, limit)
        except Exception as e:
            logger.warning("MnemosyneStore timeline failed: %s", e)

        all_results = oc_results + mnemo_results

        if self.dedup_enabled:
            return self._merge_and_dedupe(all_results, limit)

        all_results.sort(key=lambda e: e.timestamp, reverse=True)
        return all_results[:limit]

    def ask(self, question: str, agent: str | None = None) -> list[MemoryEntry]:
        """Ask with merged results from both backends."""
        oc_results = []
        mnemo_results = []

        try:
            oc_results = self.oc_store.ask(question, agent)
        except Exception as e:
            logger.warning("OcMemoryStore ask failed: %s", e)

        try:
            mnemo_results = self.mnemo_store.ask(question, agent)
        except Exception as e:
            logger.warning("MnemosyneStore ask failed: %s", e)

        all_results = oc_results + mnemo_results

        if self.dedup_enabled:
            return self._merge_and_dedupe(all_results, 5)

        all_results.sort(key=lambda e: e.timestamp, reverse=True)
        return all_results[:5]

    def export(self, path: str) -> bool:
        """Export from both backends to a combined JSON file."""
        import json

        oc_path = path.replace(".json", "_oc.json")
        mnemo_path = path.replace(".json", "_mnemo.json")

        oc_success = False
        mnemo_success = False

        try:
            oc_success = self.oc_store.export(oc_path)
        except Exception as e:
            logger.warning("OcMemoryStore export failed: %s", e)

        try:
            mnemo_success = self.mnemo_store.export(mnemo_path)
        except Exception as e:
            logger.warning("MnemosyneStore export failed: %s", e)

        combined = {
            "oc_export": oc_path if oc_success else None,
            "mnemo_export": mnemo_path if mnemo_success else None,
            "exported_at": datetime.now().isoformat(),
            "note": "DualMemoryStore writes to both backends separately",
        }

        with open(path, "w") as f:
            json.dump(combined, f, indent=2)

        return oc_success or mnemo_success

    def forget(self, entry_id: str, reason: str) -> bool:
        """Forget in both backends."""
        oc_success = False
        mnemo_success = False

        try:
            oc_success = self.oc_store.forget(entry_id, reason)
        except Exception as e:
            logger.warning("OcMemoryStore forget failed: %s", e)

        try:
            mnemo_success = self.mnemo_store.forget(entry_id, reason)
        except Exception as e:
            logger.warning("MnemosyneStore forget failed: %s", e)

        return oc_success or mnemo_success

    def compact(self, max_age_days: int | None = None) -> int:
        """Compact both backends."""
        oc_deleted = 0
        mnemo_deleted = 0

        try:
            oc_deleted = self.oc_store.compact(max_age_days)
        except Exception as e:
            logger.warning("OcMemoryStore compact failed: %s", e)

        try:
            mnemo_deleted = self.mnemo_store.compact(max_age_days)
        except Exception as e:
            logger.warning("MnemosyneStore compact failed: %s", e)

        return oc_deleted + mnemo_deleted

    def stats(self) -> MemoryStats:
        """Get combined stats from both backends."""
        oc_stats = MemoryStats()
        mnemo_stats = MemoryStats()

        try:
            oc_stats = self.oc_store.stats()
        except Exception as e:
            logger.warning("OcMemoryStore stats failed: %s", e)

        try:
            mnemo_stats = self.mnemo_store.stats()
        except Exception as e:
            logger.warning("MnemosyneStore stats failed: %s", e)

        combined = MemoryStats(
            total_entries=oc_stats.total_entries + mnemo_stats.total_entries,
            file_size_bytes=oc_stats.file_size_bytes + mnemo_stats.file_size_bytes,
            oldest_entry=min(oc_stats.oldest_entry, mnemo_stats.oldest_entry)
            if oc_stats.oldest_entry and mnemo_stats.oldest_entry
            else (oc_stats.oldest_entry or mnemo_stats.oldest_entry),
            newest_entry=max(oc_stats.newest_entry, mnemo_stats.newest_entry)
            if oc_stats.newest_entry and mnemo_stats.newest_entry
            else (oc_stats.newest_entry or mnemo_stats.newest_entry),
            entries_by_agent=self._merge_dicts(oc_stats.entries_by_agent, mnemo_stats.entries_by_agent),
            entries_by_type=self._merge_dicts(oc_stats.entries_by_type, mnemo_stats.entries_by_type),
        )

        return combined

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        """Get entry from primary backend first, then fallback."""
        try:
            if self.primary_backend == "oc":
                entry = self.oc_store.get_entry(entry_id)
                if entry:
                    return entry
                return self.mnemo_store.get_entry(entry_id)
            else:
                entry = self.mnemo_store.get_entry(entry_id)
                if entry:
                    return entry
                return self.oc_store.get_entry(entry_id)
        except Exception as e:
            logger.warning("get_entry failed: %s", e)
            return None

    def _merge_and_dedupe(self, entries: list[MemoryEntry], limit: int) -> list[MemoryEntry]:
        """Merge entries and deduplicate by content hash."""
        if not entries:
            return []

        by_hash: dict[str, list[MemoryEntry]] = {}

        for entry in entries:
            h = content_hash(entry.content)
            if h not in by_hash:
                by_hash[h] = []
            by_hash[h].append(entry)

        resolved = []
        for hash_group in by_hash.values():
            winner = self._pick_winner(hash_group)
            resolved.append(winner)

        resolved.sort(key=lambda e: e.timestamp, reverse=True)
        return resolved[:limit]

    def _pick_winner(self, entries: list[MemoryEntry]) -> MemoryEntry:
        """Pick the winning entry from duplicates."""
        if len(entries) == 1:
            return entries[0]

        sorted_entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)

        if len(sorted_entries) > 1 and sorted_entries[0].timestamp != sorted_entries[1].timestamp:
            winner = sorted_entries[0]
            winner.source_backends = list(
                set(functools.reduce(operator.iadd, [e.source_backends for e in entries], []))
            )
            return winner

        for entry in sorted_entries:
            if self.primary_backend == "oc" and "oc" in entry.source_backends:
                entry.source_backends = list(
                    set(functools.reduce(operator.iadd, [e.source_backends for e in entries], []))
                )
                return entry
            if self.primary_backend == "mnemosyne" and "mnemosyne" in entry.source_backends:
                entry.source_backends = list(
                    set(functools.reduce(operator.iadd, [e.source_backends for e in entries], []))
                )
                return entry

        return sorted_entries[0]

    def _merge_dicts(self, d1: dict[str, int], d2: dict[str, int]) -> dict[str, int]:
        """Merge two dictionaries by adding values."""
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) + v
        return result

    def close(self):
        """Close both backends."""
        try:
            self.oc_store.close()
        except Exception as e:
            logger.warning("Failed to close OcMemoryStore: %s", e)

        logger.info("DualMemoryStore closed")


_dual_store: DualMemoryStore | None = None


def get_dual_memory_store() -> DualMemoryStore:
    """Get or create the global dual memory store instance."""
    global _dual_store
    if _dual_store is None:
        _dual_store = DualMemoryStore()
    return _dual_store


def init_dual_memory_store(oc_path: str | None = None, mnemosyne_path: str | None = None) -> DualMemoryStore:
    """Initialize a new dual memory store instance."""
    global _dual_store
    _dual_store = DualMemoryStore(oc_path=oc_path, mnemosyne_path=mnemosyne_path)
    return _dual_store
