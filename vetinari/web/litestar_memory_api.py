"""UnifiedMemoryStore REST endpoints as native Litestar handlers.

Native Litestar equivalents of the routes previously registered by
``memory_api``. Part of the Flask->Litestar migration (ADR-0066).
URL paths are identical to the Flask originals.

Endpoints
---------
    GET    /api/v1/memory            — List memories with pagination
    GET    /api/v1/memory/search     — Full-text search
    POST   /api/v1/memory            — Store a new memory entry
    PUT    /api/v1/memory/<entry_id> — Update an existing memory entry
    DELETE /api/v1/memory/<entry_id> — Soft-delete (forget) a memory
    GET    /api/v1/memory/sessions   — Session context state
    GET    /api/v1/memory/stats      — Store statistics
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from litestar import MediaType, Response, delete, get, post, put
    from litestar.params import Parameter

    _LITESTAR_AVAILABLE = True
except ImportError:
    _LITESTAR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_store() -> Any:
    """Return the global UnifiedMemoryStore singleton.

    Deferred import avoids circular dependencies at module load time.

    Returns:
        The UnifiedMemoryStore singleton instance.
    """
    from vetinari.memory.unified import get_unified_memory_store

    return get_unified_memory_store()


def _entry_type_from_str(raw: str | None) -> Any:
    """Convert a string to a MemoryType enum value.

    Args:
        raw: String representation of the entry type, or ``None``.

    Returns:
        The corresponding ``MemoryType`` enum member, or ``None`` if
        *raw* is falsy.

    Raises:
        ValueError: If *raw* is non-empty but does not match any enum value.
    """
    if not raw:
        return None
    from vetinari.memory.interfaces import MemoryType

    return MemoryType(raw)


def create_memory_handlers() -> list[Any]:
    """Create and return all memory API route handlers.

    Returns an empty list when Litestar is not installed so the caller can
    safely extend its handler list without guarding the call.

    Returns:
        List of Litestar route handler objects for all memory endpoints.
    """
    if not _LITESTAR_AVAILABLE:
        return []

    from vetinari.web.responses import litestar_error_response

    # -- List memories -------------------------------------------------------

    @get("/api/v1/memory")
    async def list_memories(
        page: int = Parameter(query="page", default=1, ge=1, le=10_000),
        per_page: int = Parameter(query="per_page", default=20, ge=1, le=200),
        type_filter: str | None = Parameter(query="type", default=None),
        agent: str | None = Parameter(query="agent", default=None),
    ) -> Any:
        """List memory entries in reverse-chronological order with pagination.

        Args:
            page: Page number, 1-based (default 1).
            per_page: Entries per page (default 20, max 200).
            type_filter: Filter by ``MemoryType`` string value (optional).
            agent: Filter by agent name (optional).

        Returns:
            JSON object with keys ``items`` (list of serialised entries),
            ``total`` (count of returned items), ``page``, and ``per_page``.
            Returns 400 when ``type`` is an unknown enum value.
            Returns 500 when the memory store is unavailable.
        """
        try:
            entry_type = _entry_type_from_str(type_filter)
        except ValueError as exc:
            logger.warning("Unknown memory type filter %r: %s", type_filter, exc)
            return litestar_error_response(f"Unknown type: {type_filter!r}", code=400)

        try:
            store = _get_store()
            offset = (page - 1) * per_page
            raw_limit = offset + per_page

            entries = store.timeline(agent=agent or None, limit=raw_limit)

            if entry_type is not None:
                entries = [e for e in entries if e.entry_type == entry_type]

            page_entries = entries[offset : offset + per_page]
        except Exception:
            logger.warning(
                "Failed to list memories (page=%d, per_page=%d) — memory system unavailable, returning 500",
                page,
                per_page,
                exc_info=True,
            )
            return litestar_error_response("Memory system unavailable", code=500)

        return {
            "items": [e.to_dict() for e in page_entries],
            "total": len(page_entries),
            "page": page,
            "per_page": per_page,
        }

    # -- Search memories -----------------------------------------------------

    @get("/api/v1/memory/search")
    async def search_memories(
        q: str = Parameter(query="q", default=""),
        limit: int = Parameter(query="limit", default=10, ge=1, le=100),
        type_filter: str | None = Parameter(query="type", default=None),
        agent: str | None = Parameter(query="agent", default=None),
    ) -> Any:
        """Search memory entries using full-text (BM25) matching.

        Args:
            q: Search query string (required).
            limit: Maximum number of results to return (default 10, max 100).
            type_filter: Filter results by ``MemoryType`` string value (optional).
            agent: Filter results by agent name (optional).

        Returns:
            JSON object with keys ``items`` (list of serialised entries),
            ``total`` (count of returned items), and ``query``
            (the original query).
            Returns 400 when ``q`` is absent or ``type`` is unknown.
            Returns 500 when the memory store is unavailable.
        """
        query = q.strip()
        if not query:
            return litestar_error_response("'q' query parameter is required", code=400)

        try:
            entry_type = _entry_type_from_str(type_filter)
        except ValueError as exc:
            logger.warning("Unknown memory type filter %r: %s", type_filter, exc)
            return litestar_error_response(f"Unknown type: {type_filter!r}", code=400)

        entry_types_arg: list[str] | None = None
        if entry_type is not None:
            entry_types_arg = [entry_type.value]

        try:
            store = _get_store()
            entries = store.search(query, agent=agent or None, entry_types=entry_types_arg, limit=limit)
        except Exception:
            logger.warning(
                "Failed to search memories (q=%r) — memory system unavailable, returning 500",
                query,
                exc_info=True,
            )
            return litestar_error_response("Memory system unavailable", code=500)

        return {
            "items": [e.to_dict() for e in entries],
            "total": len(entries),
            "query": query,
        }

    # -- Session info --------------------------------------------------------

    @get("/api/v1/memory/sessions")
    async def get_session_info() -> Any:
        """Return current session context state.

        The session context is the short-term in-memory LRU store that holds
        entries between consolidation runs.

        Returns:
            JSON object with ``active`` (always ``true`` while the server is
            running) and ``entries_count`` (number of entries currently held in
            the session store).
            Returns 500 when the memory store is unavailable.
        """
        try:
            store = _get_store()
            session = store.session
            return {
                "active": True,
                "entries_count": len(session),
            }
        except Exception:
            logger.warning(
                "Failed to get session info — memory system unavailable, returning 500",
                exc_info=True,
            )
            return litestar_error_response("Memory system unavailable", code=500)

    # -- Memory stats --------------------------------------------------------

    @get("/api/v1/memory/stats")
    async def get_memory_stats() -> Any:
        """Return aggregate statistics for the unified memory store.

        Returns:
            JSON object serialised from ``MemoryStats``, containing
            ``total_entries``, ``file_size_bytes``, ``oldest_entry``,
            ``newest_entry``, ``entries_by_agent``, and ``entries_by_type``.
            Returns 500 when the memory store is unavailable.
        """
        try:
            store = _get_store()
            stats = store.stats()
            return stats.to_dict()
        except Exception:
            logger.warning(
                "Failed to get memory stats — memory system unavailable, returning 500",
                exc_info=True,
            )
            return litestar_error_response("Memory system unavailable", code=500)

    # -- Store memory --------------------------------------------------------

    @post("/api/v1/memory")
    async def store_memory(data: dict[str, Any]) -> Any:
        """Store a new memory entry in the unified store.

        Args:
            data: JSON request body with the following fields:
                content: Memory content string (required).
                entry_type: ``MemoryType`` string value (required).
                agent: Agent identifier (optional, defaults to empty string).
                summary: Short summary of the content (optional).
                provenance: Source or provenance string (optional).

        Returns:
            JSON object with ``ok`` (``true``) and ``id`` (the assigned entry
            ID). Returns 400 when required fields are absent or
            ``entry_type`` is invalid. Returns 500 when the memory store
            is unavailable.
        """
        content_raw = data.get("content", "")
        if not isinstance(content_raw, str):
            return litestar_error_response("'content' must be a string", code=422)
        content = content_raw.strip()
        if not content:
            return litestar_error_response("'content' is required", code=400)

        raw_type_val = data.get("entry_type", "")
        if not isinstance(raw_type_val, str):
            return litestar_error_response("'entry_type' must be a string", code=422)
        raw_type = raw_type_val.strip()
        if not raw_type:
            return litestar_error_response("'entry_type' is required", code=400)

        try:
            entry_type = _entry_type_from_str(raw_type)
        except ValueError as exc:
            logger.warning("Invalid entry_type %r: %s", raw_type, exc)
            return litestar_error_response(f"Invalid entry_type: {raw_type!r}", code=400)

        from vetinari.memory.interfaces import MemoryEntry

        entry = MemoryEntry(
            agent=data.get("agent", ""),
            entry_type=entry_type,
            content=content,
            summary=data.get("summary", ""),
            provenance=data.get("provenance", ""),
        )

        try:
            store = _get_store()
            entry_id = store.remember(entry)
        except Exception:
            logger.warning(
                "Failed to store memory entry — memory system unavailable, returning 500",
                exc_info=True,
            )
            return litestar_error_response("Memory system unavailable", code=500)

        logger.info("Stored memory entry %s via REST API", entry_id)
        return Response(
            content={"ok": True, "id": entry_id},
            status_code=201,
            media_type=MediaType.JSON,
        )

    # -- Update memory -------------------------------------------------------

    @put("/api/v1/memory/{entry_id:str}")
    async def update_memory(entry_id: str, data: dict[str, Any]) -> Any:
        """Update the content of an existing memory entry.

        Replaces the content of the entry identified by *entry_id*. The entry
        must exist and not be forgotten.

        Args:
            entry_id: The unique identifier of the memory entry to update.
            data: JSON request body with ``content`` (new content string,
                required and non-empty).

        Returns:
            JSON object with ``ok`` (``true``) on success.
            Returns 400 when ``content`` is absent or empty.
            Returns 404 when the entry does not exist.
            Returns 500 when the memory store is unavailable.
        """
        content_raw = data.get("content", "")
        if not isinstance(content_raw, str):
            return litestar_error_response("'content' must be a string", code=422)
        content = content_raw.strip()
        if not content:
            return litestar_error_response("'content' is required", code=400)

        try:
            store = _get_store()
            success = store.update_content(entry_id, content)
        except Exception:
            logger.warning(
                "Failed to update memory entry %r — memory system unavailable, returning 500",
                entry_id,
                exc_info=True,
            )
            return litestar_error_response("Memory system unavailable", code=500)

        if not success:
            return litestar_error_response(f"Entry '{entry_id}' not found", code=404)

        logger.info("Updated memory entry %s via REST API", entry_id)
        return {"ok": True}

    # -- Forget memory -------------------------------------------------------

    @delete("/api/v1/memory/{entry_id:str}", status_code=200)
    async def forget_memory(entry_id: str, data: dict[str, Any] | None = None) -> Any:
        """Soft-delete a memory entry by marking it as forgotten (tombstone).

        Args:
            entry_id: The unique identifier of the memory entry to forget.
            data: Optional JSON request body with a ``reason`` field
                (defaults to ``"deleted via API"``).

        Returns:
            JSON object with ``ok`` (``true``) on success.
            Returns 404 when the entry does not exist or was already forgotten.
            Returns 500 when the memory store is unavailable.
        """
        from vetinari.web.request_validation import json_object_body

        body = json_object_body(data)
        if body is None:
            return litestar_error_response("Request body must be a JSON object", code=400)
        reason = body.get("reason", "deleted via API")

        try:
            store = _get_store()
            success = store.forget(entry_id, reason)
        except Exception:
            logger.warning(
                "Failed to forget memory entry %r — memory system unavailable, returning 500",
                entry_id,
                exc_info=True,
            )
            return litestar_error_response("Memory system unavailable", code=500)

        if not success:
            return litestar_error_response(f"Entry '{entry_id}' not found or already forgotten", code=404)

        logger.info("Forgot memory entry %s via REST API (reason=%s)", entry_id, reason)
        return {"ok": True}

    return [
        list_memories,
        search_memories,
        get_session_info,
        get_memory_stats,
        store_memory,
        update_memory,
        forget_memory,
    ]
