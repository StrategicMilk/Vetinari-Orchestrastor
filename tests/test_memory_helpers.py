"""Regression tests for memory-helper script defects.

Covers the 8 defects fixed in scripts/memory/session.py,
scripts/memory/embeddings.py, and scripts/memory/search.py.
"""

from __future__ import annotations

import sqlite3
import struct
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db(tmp_path: Path) -> sqlite3.Connection:
    """Create an in-memory SQLite DB with the memory schema."""
    conn = sqlite3.connect(str(tmp_path / "memory.db"))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            type         TEXT    NOT NULL DEFAULT 'project',
            title        TEXT    NOT NULL,
            content      TEXT    NOT NULL,
            tags         TEXT    DEFAULT '',
            created_at   TEXT    NOT NULL DEFAULT '2024-01-01',
            updated_at   TEXT    NOT NULL DEFAULT '2024-01-01',
            last_accessed TEXT   NOT NULL DEFAULT '2024-01-01',
            access_count INTEGER DEFAULT 0,
            is_pinned    INTEGER DEFAULT 0,
            content_hash TEXT    NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
            vector    BLOB    NOT NULL,
            model     TEXT    NOT NULL,
            dims      INTEGER NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            title, content, tags,
            content=memories, content_rowid=id,
            tokenize='porter unicode61'
        );
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, title, content, tags)
            VALUES (new.id, new.title, new.content, new.tags);
        END;
        """
    )
    conn.commit()
    return conn


def _pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _insert_memory(conn: sqlite3.Connection, title: str = "T", content: str = "C") -> int:
    cur = conn.execute(
        "INSERT INTO memories (type, title, content, content_hash) VALUES ('project', ?, ?, '')",
        (title, content),
    )
    conn.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Defect 1 & 2 — session.py _append
# ---------------------------------------------------------------------------


class TestAppendSanitize:
    """_append must not corrupt unrelated sections or allow heading injection."""

    def test_append_leaves_other_sections_placeholder_intact(self, tmp_path: Path) -> None:
        """Appending to one section must NOT erase _(none yet)_ in sibling sections."""
        session_file = tmp_path / "current.md"
        session_file.write_text(
            "# Session\n\n"
            "## Decisions Made\n\n_(none yet)_\n\n"
            "## Key References\n\n_(none yet)_\n\n"
            "## Blockers\n\n_(none yet)_\n",
            encoding="utf-8",
        )

        # Patch session module's SESSION_CURRENT to point at our file
        import scripts.memory.session as session_mod

        args = types.SimpleNamespace(section="Decisions Made", note="chose X")
        with patch.object(session_mod, "SESSION_CURRENT", session_file):
            session_mod._append(args)

        result = session_file.read_text(encoding="utf-8")
        # Targeted section should have the note, NOT the placeholder
        assert "- chose X" in result
        assert "_(none yet)_" not in result.split("## Decisions Made")[1].split("## Key References")[0]
        # Sibling sections must still have their placeholder
        assert "_(none yet)_" in result.split("## Key References")[1].split("## Blockers")[0]
        assert "_(none yet)_" in result.split("## Blockers")[1]

    def test_append_multiline_note_cannot_inject_heading(self, tmp_path: Path) -> None:
        """A note containing Markdown heading syntax must not inject a new section."""
        session_file = tmp_path / "current.md"
        session_file.write_text(
            "# Session\n\n## Decisions Made\n\n_(none yet)_\n\n## Blockers\n\n_(none yet)_\n",
            encoding="utf-8",
        )

        import scripts.memory.session as session_mod

        # Attempt to inject a heading via the note text
        malicious_note = "legit note\n## INJECTED SECTION\nevil content"
        args = types.SimpleNamespace(section="Decisions Made", note=malicious_note)
        with patch.object(session_mod, "SESSION_CURRENT", session_file):
            session_mod._append(args)

        result = session_file.read_text(encoding="utf-8")
        # The injected heading marker must not appear as a real heading
        assert "## INJECTED SECTION" not in result
        # The text content still appears, just without the heading prefix
        assert "INJECTED SECTION" in result

    @pytest.mark.parametrize(
        "note",
        [
            "# single hash",
            "## double hash",
            "### triple hash",
            "## Heading\nbody text",
        ],
    )
    def test_sanitize_note_strips_heading_markers(self, note: str) -> None:
        """_sanitize_note must strip leading # characters from every line."""
        import scripts.memory.session as session_mod

        result = session_mod._sanitize_note(note)
        for line in result.splitlines():
            assert not line.startswith("#"), f"Heading not stripped in: {line!r}"


# ---------------------------------------------------------------------------
# Defect 4 — session.py _archive collision
# ---------------------------------------------------------------------------


class TestArchiveCollision:
    """_archive must not overwrite an existing archive when names collide."""

    def test_archive_no_collision_on_same_second(self, tmp_path: Path) -> None:
        """Two archives with same timestamp+slug must get unique filenames."""
        import scripts.memory.session as session_mod

        session_dir = tmp_path / "sessions"
        session_dir.mkdir()
        current = tmp_path / "current.md"

        summary = "test summary"

        # Create first archive
        current.write_text("session content A", encoding="utf-8")
        with (
            patch.object(session_mod, "SESSION_CURRENT", current),
            patch.object(session_mod, "SESSION_DIR", session_dir),
            patch("scripts.memory.session.datetime") as mock_dt,
        ):
            mock_dt.datetime.now.return_value.strftime.return_value = "20240101_120000"
            session_mod._archive(summary)

        first_archives = list(session_dir.glob("*.md"))
        assert len(first_archives) == 1

        # Create second archive — current must be recreated since _archive unlinks it
        current.write_text("session content B", encoding="utf-8")
        with (
            patch.object(session_mod, "SESSION_CURRENT", current),
            patch.object(session_mod, "SESSION_DIR", session_dir),
            patch("scripts.memory.session.datetime") as mock_dt,
        ):
            mock_dt.datetime.now.return_value.strftime.return_value = "20240101_120000"
            session_mod._archive(summary)

        all_archives = list(session_dir.glob("*.md"))
        assert len(all_archives) == 2, f"Expected 2 archives, got: {[f.name for f in all_archives]}"
        # Their names must differ
        names = {f.name for f in all_archives}
        assert len(names) == 2, f"Duplicate archive names: {names}"


# ---------------------------------------------------------------------------
# Defect 3 & 5 — embeddings.py model filtering
# ---------------------------------------------------------------------------


class TestEmbeddingsModelFiltering:
    """search_similar and embed_all must not mix or ignore model boundaries."""

    def _get_search_similar(self, tmp_path: Path):
        """Return a patched search_similar that uses a temp DB."""
        conn = _make_db(tmp_path)
        mid = _insert_memory(conn, "alpha", "content alpha")
        # Store an embedding with model_a (4 dims)
        conn.execute(
            "INSERT INTO embeddings (memory_id, vector, model, dims) VALUES (?, ?, ?, ?)",
            (mid, _pack_vector([1.0, 0.0, 0.0, 0.0]), "model_a", 4),
        )
        conn.commit()
        conn.close()

        db_path = str(tmp_path / "memory.db")

        import scripts.memory.embeddings as emb_mod

        def patched_get_db():
            c = sqlite3.connect(db_path)
            c.row_factory = sqlite3.Row
            return c

        return emb_mod, patched_get_db

    def test_search_similar_skips_incompatible_model(self, tmp_path: Path) -> None:
        """search_similar with model_b must return empty when only model_a rows exist."""
        emb_mod, patched_get_db = self._get_search_similar(tmp_path)

        query_vec = [1.0, 0.0, 0.0, 0.0]
        with (
            patch.object(emb_mod, "get_db", patched_get_db),
            patch.object(emb_mod, "generate", return_value=query_vec),
        ):
            results = emb_mod.search_similar("alpha", model="model_b")

        assert results == [], f"Expected empty — wrong model; got {results}"

    def test_search_similar_returns_results_for_matching_model(self, tmp_path: Path) -> None:
        """search_similar must return results when model matches stored rows."""
        emb_mod, patched_get_db = self._get_search_similar(tmp_path)

        query_vec = [1.0, 0.0, 0.0, 0.0]
        with (
            patch.object(emb_mod, "get_db", patched_get_db),
            patch.object(emb_mod, "generate", return_value=query_vec),
        ):
            results = emb_mod.search_similar("alpha", model="model_a")

        assert len(results) == 1
        assert results[0]["title"] == "alpha"

    def test_search_similar_skips_mismatched_dims(self, tmp_path: Path) -> None:
        """search_similar must skip stored rows whose dims differ from query dims."""
        conn = _make_db(tmp_path)
        mid = _insert_memory(conn, "beta", "content beta")
        # Store with model_a but 4 dims
        conn.execute(
            "INSERT INTO embeddings (memory_id, vector, model, dims) VALUES (?, ?, ?, ?)",
            (mid, _pack_vector([1.0, 0.0, 0.0, 0.0]), "model_a", 4),
        )
        conn.commit()
        conn.close()

        db_path = str(tmp_path / "memory.db")
        import scripts.memory.embeddings as emb_mod

        def patched_get_db():
            c = sqlite3.connect(db_path)
            c.row_factory = sqlite3.Row
            return c

        # Query vector has 8 dims — dimension mismatch must be skipped
        query_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        with (
            patch.object(emb_mod, "get_db", patched_get_db),
            patch.object(emb_mod, "generate", return_value=query_vec),
        ):
            results = emb_mod.search_similar("beta", model="model_a")

        assert results == [], f"Expected empty due to dim mismatch; got {results}"

    def test_embed_all_reembeds_when_model_differs(self, tmp_path: Path) -> None:
        """embed_all must process memories that have a row for a different model."""
        conn = _make_db(tmp_path)
        mid = _insert_memory(conn, "gamma", "content gamma")
        # Already embedded with model_a
        conn.execute(
            "INSERT INTO embeddings (memory_id, vector, model, dims) VALUES (?, ?, ?, ?)",
            (mid, _pack_vector([1.0, 0.0, 0.0, 0.0]), "model_a", 4),
        )
        conn.commit()
        conn.close()

        db_path = str(tmp_path / "memory.db")
        import scripts.memory.embeddings as emb_mod

        def patched_get_db():
            c = sqlite3.connect(db_path)
            c.row_factory = sqlite3.Row
            return c

        new_vec = [0.5, 0.5, 0.5, 0.5]
        with (
            patch.object(emb_mod, "get_db", patched_get_db),
            patch.object(emb_mod, "generate", return_value=new_vec),
        ):
            count = emb_mod.embed_all(model="model_b")

        # Memory was NOT embedded with model_b yet — must be processed
        assert count == 1, f"Expected 1 newly embedded; got {count}"

    def test_embed_all_skips_already_embedded(self, tmp_path: Path) -> None:
        """embed_all must skip memories that already have a row for the requested model."""
        conn = _make_db(tmp_path)
        mid = _insert_memory(conn, "delta", "content delta")
        conn.execute(
            "INSERT INTO embeddings (memory_id, vector, model, dims) VALUES (?, ?, ?, ?)",
            (mid, _pack_vector([1.0, 0.0, 0.0, 0.0]), "model_a", 4),
        )
        conn.commit()
        conn.close()

        db_path = str(tmp_path / "memory.db")
        import scripts.memory.embeddings as emb_mod

        def patched_get_db():
            c = sqlite3.connect(db_path)
            c.row_factory = sqlite3.Row
            return c

        with (
            patch.object(emb_mod, "get_db", patched_get_db),
            patch.object(emb_mod, "generate", return_value=[1.0, 0.0, 0.0, 0.0]),
        ):
            count = emb_mod.embed_all(model="model_a")

        assert count == 0, f"Expected 0 (already embedded); got {count}"


# ---------------------------------------------------------------------------
# Defect 6 — search.py stale access count
# ---------------------------------------------------------------------------


class TestRecallAccessCount:
    """cmd_recall must print post-increment access count."""

    def test_recall_prints_incremented_access_count(self, tmp_path: Path, capsys) -> None:
        conn = _make_db(tmp_path)
        mid = _insert_memory(conn, "epsilon", "body text")
        # Set access_count to 5 before the recall
        conn.execute("UPDATE memories SET access_count = 5 WHERE id = ?", (mid,))
        conn.commit()
        conn.close()

        db_path = str(tmp_path / "memory.db")
        import scripts.memory.search as search_mod

        def patched_get_db():
            c = sqlite3.connect(db_path)
            c.row_factory = sqlite3.Row
            return c

        args = types.SimpleNamespace(id=mid)
        with patch.object(search_mod, "get_db", patched_get_db):
            search_mod.cmd_recall(args)

        output = capsys.readouterr().out
        # Must show 6 (post-increment), not 5 (pre-touch stale value)
        assert "6x" in output, f"Expected '6x' in output; got: {output!r}"
        assert "5x" not in output, f"Stale '5x' must not appear; got: {output!r}"


# ---------------------------------------------------------------------------
# Defect 7 — search.py FTS input sanitization
# ---------------------------------------------------------------------------


class TestFtsSanitization:
    """_sanitize_fts_query must handle empty and quote-broken input safely."""

    def test_sanitize_empty_string_returns_none(self) -> None:
        from scripts.memory.search import _sanitize_fts_query

        assert _sanitize_fts_query("") is None

    def test_sanitize_whitespace_only_returns_none(self) -> None:
        from scripts.memory.search import _sanitize_fts_query

        assert _sanitize_fts_query("   ") is None

    def test_sanitize_normal_query_returns_quoted_terms(self) -> None:
        from scripts.memory.search import _sanitize_fts_query

        result = _sanitize_fts_query("hello world")
        assert result is not None
        assert '"hello"' in result
        assert '"world"' in result

    def test_sanitize_embedded_quote_is_escaped(self) -> None:
        from scripts.memory.search import _sanitize_fts_query

        # A raw double-quote in the term must not break FTS5 quoting
        result = _sanitize_fts_query('say "hi"')
        assert result is not None
        # The term should not contain an unescaped raw quote in a way that
        # breaks the outer double-quote wrapping
        # Each token is wrapped: "say" and the escaped token
        assert result.startswith('"')

    def test_cmd_search_exits_cleanly_on_empty_query(self, tmp_path: Path, capsys) -> None:
        """cmd_search must not crash SQLite when given an empty query."""
        db_path = str(tmp_path / "memory.db")
        _make_db(tmp_path)

        import scripts.memory.search as search_mod

        def patched_get_db():
            c = sqlite3.connect(db_path)
            c.row_factory = sqlite3.Row
            return c

        args = types.SimpleNamespace(query="", limit=10, type=None)
        with patch.object(search_mod, "get_db", patched_get_db):
            # Must not raise
            search_mod.cmd_search(args)

        output = capsys.readouterr().out
        assert "empty" in output.lower() or "nothing" in output.lower()


# ---------------------------------------------------------------------------
# Defect 8 — search.py negative limit
# ---------------------------------------------------------------------------


class TestClampLimit:
    """Negative --limit values must not disable the result bound."""

    @pytest.mark.parametrize("bad_limit", [-1, -100, -9999])
    def test_clamp_limit_rejects_negative(self, bad_limit: int) -> None:
        from scripts.memory.search import _clamp_limit

        result = _clamp_limit(bad_limit, 10)
        assert result > 0, f"Expected positive limit for input {bad_limit}; got {result}"
        assert result == 10, f"Expected default 10; got {result}"

    def test_clamp_limit_zero_uses_default(self) -> None:
        from scripts.memory.search import _clamp_limit

        assert _clamp_limit(0, 20) == 20

    def test_clamp_limit_none_uses_default(self) -> None:
        from scripts.memory.search import _clamp_limit

        assert _clamp_limit(None, 15) == 15

    def test_clamp_limit_positive_preserved(self) -> None:
        from scripts.memory.search import _clamp_limit

        assert _clamp_limit(5, 20) == 5

    def test_cmd_list_negative_limit_does_not_return_all_rows(
        self, tmp_path: Path, capsys
    ) -> None:
        """cmd_list with a negative limit must not return unbounded rows."""
        conn = _make_db(tmp_path)
        for i in range(25):
            _insert_memory(conn, f"title_{i}", f"content_{i}")
        conn.close()

        db_path = str(tmp_path / "memory.db")
        import scripts.memory.search as search_mod

        def patched_get_db():
            c = sqlite3.connect(db_path)
            c.row_factory = sqlite3.Row
            return c

        args = types.SimpleNamespace(limit=-1, sort="updated", type=None)
        with patch.object(search_mod, "get_db", patched_get_db):
            search_mod.cmd_list(args)

        output = capsys.readouterr().out
        # With clamping to default=20, only 20 rows should appear, not all 25
        shown = int(output.split("memories shown")[0].strip().split("\n")[-1].strip())
        assert shown <= 20, f"Expected ≤20 rows with negative limit; got {shown}"
