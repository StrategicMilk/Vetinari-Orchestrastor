"""
DualMemoryStore - Coordinates between OcMemoryStore and MnemosyneMemoryStore.

This store writes to both backends and merges reads according to the
merge policy defined in docs/memory_merge_policy.md.

Security: All entries are scanned for secrets before being written to backends.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from .interfaces import (
    IMemoryStore, MemoryEntry, MemoryStats, MemoryEntryType,
    content_hash, MEMORY_PRIMARY_READ, MEMORY_DEDUP_ENABLED
)
from .oc_memory import OcMemoryStore
from .mnemosyne_memory import MnemosyneMemoryStore
from ..security import get_secret_scanner

logger = logging.getLogger(__name__)


class DualMemoryStore(IMemoryStore):
    """Coordinates between OcMemoryStore and MnemosyneMemoryStore.
    
    Write behavior: writes to both backends simultaneously
    Read behavior: merges results from both backends with deduplication
    """
    
    def __init__(self, oc_path: str = None, mnemosyne_path: str = None):
        self.oc_store = OcMemoryStore(oc_path) if oc_path else OcMemoryStore()
        self.mnemo_store = MnemosyneMemoryStore(mnemosyne_path) if mnemosyne_path else MnemosyneMemoryStore()
        self.primary_backend = MEMORY_PRIMARY_READ
        self.dedup_enabled = MEMORY_DEDUP_ENABLED
        
        logger.info(f"DualMemoryStore initialized with primary={self.primary_backend}")
    
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
            logger.warning(f"OcMemoryStore write failed: {e}")
        
        try:
            mnemo_id = self.mnemo_store.remember(entry)
            entry.source_backends.append("mnemosyne")
        except Exception as e:
            logger.warning(f"MnemosyneStore write failed: {e}")
        
        if not entry.source_backends:
            raise RuntimeError("Both backends failed to write")
        
        primary_id = oc_id if oc_id else mnemo_id
        logger.debug(f"DualMemoryStore: wrote entry to {entry.source_backends}")
        
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
    
    def search(self, query: str, agent: Optional[str] = None,
               entry_types: Optional[List[str]] = None,
               limit: int = 10) -> List[MemoryEntry]:
        """Search with merged results from both backends."""
        oc_results = []
        mnemo_results = []
        
        try:
            oc_results = self.oc_store.search(query, agent, entry_types, limit)
        except Exception as e:
            logger.warning(f"OcMemoryStore search failed: {e}")
        
        try:
            mnemo_results = self.mnemo_store.search(query, agent, entry_types, limit)
        except Exception as e:
            logger.warning(f"MnemosyneStore search failed: {e}")
        
        all_results = oc_results + mnemo_results
        
        if self.dedup_enabled:
            return self._merge_and_dedupe(all_results, limit)
        
        all_results.sort(key=lambda e: e.timestamp, reverse=True)
        return all_results[:limit]
    
    def timeline(self, agent: Optional[str] = None,
                 start_time: Optional[int] = None,
                 end_time: Optional[int] = None,
                 limit: int = 100) -> List[MemoryEntry]:
        """Timeline with merged results from both backends."""
        oc_results = []
        mnemo_results = []
        
        try:
            oc_results = self.oc_store.timeline(agent, start_time, end_time, limit)
        except Exception as e:
            logger.warning(f"OcMemoryStore timeline failed: {e}")
        
        try:
            mnemo_results = self.mnemo_store.timeline(agent, start_time, end_time, limit)
        except Exception as e:
            logger.warning(f"MnemosyneStore timeline failed: {e}")
        
        all_results = oc_results + mnemo_results
        
        if self.dedup_enabled:
            return self._merge_and_dedupe(all_results, limit)
        
        all_results.sort(key=lambda e: e.timestamp, reverse=True)
        return all_results[:limit]
    
    def ask(self, question: str, agent: Optional[str] = None) -> List[MemoryEntry]:
        """Ask with merged results from both backends."""
        oc_results = []
        mnemo_results = []
        
        try:
            oc_results = self.oc_store.ask(question, agent)
        except Exception as e:
            logger.warning(f"OcMemoryStore ask failed: {e}")
        
        try:
            mnemo_results = self.mnemo_store.ask(question, agent)
        except Exception as e:
            logger.warning(f"MnemosyneStore ask failed: {e}")
        
        all_results = oc_results + mnemo_results
        
        if self.dedup_enabled:
            return self._merge_and_dedupe(all_results, 5)
        
        all_results.sort(key=lambda e: e.timestamp, reverse=True)
        return all_results[:5]
    
    def export(self, path: str) -> bool:
        """Export from both backends to a combined JSON file."""
        import json
        from datetime import datetime
        
        oc_path = path.replace(".json", "_oc.json")
        mnemo_path = path.replace(".json", "_mnemo.json")
        
        oc_success = False
        mnemo_success = False
        
        try:
            oc_success = self.oc_store.export(oc_path)
        except Exception as e:
            logger.warning(f"OcMemoryStore export failed: {e}")
        
        try:
            mnemo_success = self.mnemo_store.export(mnemo_path)
        except Exception as e:
            logger.warning(f"MnemosyneStore export failed: {e}")
        
        combined = {
            "oc_export": oc_path if oc_success else None,
            "mnemo_export": mnemo_path if mnemo_success else None,
            "exported_at": datetime.now().isoformat(),
            "note": "DualMemoryStore writes to both backends separately"
        }
        
        with open(path, 'w') as f:
            json.dump(combined, f, indent=2)
        
        return oc_success or mnemo_success
    
    def forget(self, entry_id: str, reason: str) -> bool:
        """Forget in both backends."""
        oc_success = False
        mnemo_success = False
        
        try:
            oc_success = self.oc_store.forget(entry_id, reason)
        except Exception as e:
            logger.warning(f"OcMemoryStore forget failed: {e}")
        
        try:
            mnemo_success = self.mnemo_store.forget(entry_id, reason)
        except Exception as e:
            logger.warning(f"MnemosyneStore forget failed: {e}")
        
        return oc_success or mnemo_success
    
    def compact(self, max_age_days: Optional[int] = None) -> int:
        """Compact both backends."""
        oc_deleted = 0
        mnemo_deleted = 0
        
        try:
            oc_deleted = self.oc_store.compact(max_age_days)
        except Exception as e:
            logger.warning(f"OcMemoryStore compact failed: {e}")
        
        try:
            mnemo_deleted = self.mnemo_store.compact(max_age_days)
        except Exception as e:
            logger.warning(f"MnemosyneStore compact failed: {e}")
        
        return oc_deleted + mnemo_deleted
    
    def stats(self) -> MemoryStats:
        """Get combined stats from both backends."""
        oc_stats = MemoryStats()
        mnemo_stats = MemoryStats()
        
        try:
            oc_stats = self.oc_store.stats()
        except Exception as e:
            logger.warning(f"OcMemoryStore stats failed: {e}")
        
        try:
            mnemo_stats = self.mnemo_store.stats()
        except Exception as e:
            logger.warning(f"MnemosyneStore stats failed: {e}")
        
        combined = MemoryStats(
            total_entries=oc_stats.total_entries + mnemo_stats.total_entries,
            file_size_bytes=oc_stats.file_size_bytes + mnemo_stats.file_size_bytes,
            oldest_entry=min(oc_stats.oldest_entry, mnemo_stats.oldest_entry) if oc_stats.oldest_entry and mnemo_stats.oldest_entry else (oc_stats.oldest_entry or mnemo_stats.oldest_entry),
            newest_entry=max(oc_stats.newest_entry, mnemo_stats.newest_entry) if oc_stats.newest_entry and mnemo_stats.newest_entry else (oc_stats.newest_entry or mnemo_stats.newest_entry),
            entries_by_agent=self._merge_dicts(oc_stats.entries_by_agent, mnemo_stats.entries_by_agent),
            entries_by_type=self._merge_dicts(oc_stats.entries_by_type, mnemo_stats.entries_by_type)
        )
        
        return combined
    
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
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
            logger.warning(f"get_entry failed: {e}")
            return None
    
    def _merge_and_dedupe(self, entries: List[MemoryEntry], limit: int) -> List[MemoryEntry]:
        """Merge entries and deduplicate by content hash."""
        if not entries:
            return []
        
        by_hash: Dict[str, List[MemoryEntry]] = {}
        
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
    
    def _pick_winner(self, entries: List[MemoryEntry]) -> MemoryEntry:
        """Pick the winning entry from duplicates."""
        if len(entries) == 1:
            return entries[0]
        
        sorted_entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)
        
        if len(sorted_entries) > 1 and sorted_entries[0].timestamp != sorted_entries[1].timestamp:
            winner = sorted_entries[0]
            winner.source_backends = list(set(sum([e.source_backends for e in entries], [])))
            return winner
        
        for entry in sorted_entries:
            if self.primary_backend == "oc" and "oc" in entry.source_backends:
                entry.source_backends = list(set(sum([e.source_backends for e in entries], [])))
                return entry
            if self.primary_backend == "mnemosyne" and "mnemosyne" in entry.source_backends:
                entry.source_backends = list(set(sum([e.source_backends for e in entries], [])))
                return entry
        
        return sorted_entries[0]
    
    def _merge_dicts(self, d1: Dict[str, int], d2: Dict[str, int]) -> Dict[str, int]:
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
            logger.warning(f"Failed to close OcMemoryStore: {e}")
        
        logger.info("DualMemoryStore closed")


_dual_store: Optional[DualMemoryStore] = None


def get_dual_memory_store() -> DualMemoryStore:
    """Get or create the global dual memory store instance."""
    global _dual_store
    if _dual_store is None:
        _dual_store = DualMemoryStore()
    return _dual_store


def init_dual_memory_store(oc_path: str = None, mnemosyne_path: str = None) -> DualMemoryStore:
    """Initialize a new dual memory store instance."""
    global _dual_store
    _dual_store = DualMemoryStore(oc_path=oc_path, mnemosyne_path=mnemosyne_path)
    return _dual_store
