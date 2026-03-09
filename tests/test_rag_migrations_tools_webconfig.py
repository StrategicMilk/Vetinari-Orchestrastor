"""
Comprehensive tests for five uncovered Vetinari modules:

1. vetinari.rag.knowledge_base     -- RAG knowledge base (fallback path)
2. vetinari.migrations.upgrade_subtask_schema  -- subtask JSON migration
3. vetinari.tools.tool_registry_integration    -- deprecated tool wrappers
4. vetinari.web.config             -- centralised app configuration
5. vetinari.skills.data_engineer   -- deprecated data engineer skill

60+ tests organised into clearly-named test classes.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Clean stubs that may have been created by other test files
# ---------------------------------------------------------------------------
for _stubname in (
    "vetinari.rag.knowledge_base",
    "vetinari.rag",
    "vetinari.migrations.upgrade_subtask_schema",
    "vetinari.migrations",
    "vetinari.tools.tool_registry_integration",
    "vetinari.web.config",
    "vetinari.web",
    "vetinari.skills.data_engineer",
):
    sys.modules.pop(_stubname, None)


# =========================================================================
# 1. vetinari.rag.knowledge_base
# =========================================================================
# ChromaDB is NOT installed in CI, so the fallback (in-memory) path is
# exercised automatically.

from vetinari.rag.knowledge_base import KBDocument, KnowledgeBase


@pytest.fixture(autouse=True)
def _reset_kb_singleton():
    """Ensure every test starts with a fresh KnowledgeBase singleton."""
    KnowledgeBase._instance = None
    # Also reset the module-level _kb singleton
    import vetinari.rag.knowledge_base as _kb_mod
    _kb_mod._kb = None
    yield
    KnowledgeBase._instance = None
    _kb_mod._kb = None


# -- KBDocument dataclass --------------------------------------------------

class TestKBDocument:
    """Tests for the KBDocument dataclass."""

    def test_create_with_defaults(self):
        doc = KBDocument(doc_id="d1", content="hello", source="test.py")
        assert doc.doc_id == "d1"
        assert doc.content == "hello"
        assert doc.source == "test.py"
        assert doc.category == "general"
        assert doc.score == 0.0

    def test_create_with_all_fields(self):
        doc = KBDocument(
            doc_id="d2",
            content="body",
            source="s.md",
            category="code",
            score=0.95,
        )
        assert doc.category == "code"
        assert doc.score == 0.95

    def test_equality(self):
        a = KBDocument("x", "y", "z")
        b = KBDocument("x", "y", "z")
        assert a == b

    def test_different_docs_not_equal(self):
        a = KBDocument("x", "y", "z")
        b = KBDocument("x2", "y", "z")
        assert a != b

    def test_score_mutable(self):
        doc = KBDocument("id", "content", "src")
        doc.score = 1.0
        assert doc.score == 1.0

    def test_empty_content(self):
        doc = KBDocument(doc_id="e", content="", source="")
        assert doc.content == ""

    def test_long_content(self):
        long_text = "a" * 10_000
        doc = KBDocument(doc_id="long", content=long_text, source="big.txt")
        assert len(doc.content) == 10_000

    def test_category_custom_value(self):
        doc = KBDocument(doc_id="c", content="x", source="y", category="error")
        assert doc.category == "error"

    def test_repr_contains_id(self):
        doc = KBDocument(doc_id="myid", content="x", source="y")
        assert "myid" in repr(doc)

    def test_fields_are_accessible(self):
        doc = KBDocument(doc_id="f", content="c", source="s", category="docs", score=0.5)
        assert doc.__dataclass_fields__  # it's a dataclass

    def test_from_keyword_args(self):
        kwargs = {"doc_id": "kw", "content": "stuff", "source": "file.py"}
        doc = KBDocument(**kwargs)
        assert doc.doc_id == "kw"

    def test_score_defaults_to_zero(self):
        doc = KBDocument(doc_id="z", content="c", source="s")
        assert doc.score == pytest.approx(0.0)


# -- KnowledgeBase core operations ----------------------------------------

class TestKnowledgeBase:
    """Tests for KnowledgeBase add_document, get_stats, singleton, etc."""

    def test_init_uses_fallback(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path / "chroma"))
        assert kb._chroma_available is False
        assert kb._fallback_docs == []

    def test_add_document_returns_id(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        doc_id = kb.add_document("test content", source="test.py")
        assert doc_id.startswith("doc_")

    def test_add_document_custom_id(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        doc_id = kb.add_document("hello", source="s.py", doc_id="custom_1")
        assert doc_id == "custom_1"

    def test_add_document_with_category(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        kb.add_document("code here", source="a.py", category="code")
        assert kb._fallback_docs[0]["category"] == "code"

    def test_add_document_truncates_at_5000(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        long = "x" * 10_000
        kb.add_document(long, source="big.txt")
        assert len(kb._fallback_docs[0]["content"]) == 5000

    def test_add_document_upsert_replaces_existing(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        kb.add_document("v1", source="s.py", doc_id="same_id")
        kb.add_document("v2", source="s.py", doc_id="same_id")
        assert len(kb._fallback_docs) == 1
        assert kb._fallback_docs[0]["content"] == "v2"

    def test_get_stats_empty(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        stats = kb.get_stats()
        assert stats["document_count"] == 0
        assert stats["backend"] == "in_memory"
        assert "persist_dir" in stats

    def test_get_stats_after_adds(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        kb.add_document("a", source="a.txt")
        kb.add_document("b", source="b.txt")
        stats = kb.get_stats()
        assert stats["document_count"] == 2

    def test_singleton_get_instance(self, tmp_path):
        KnowledgeBase._instance = None
        inst1 = KnowledgeBase.get_instance()
        inst2 = KnowledgeBase.get_instance()
        assert inst1 is inst2

    def test_module_level_get_knowledge_base(self, tmp_path):
        from vetinari.rag.knowledge_base import get_knowledge_base
        kb = get_knowledge_base()
        assert isinstance(kb, KnowledgeBase)

    def test_auto_generated_doc_id_is_deterministic(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        id1 = kb.add_document("same content", source="same.py")
        # Remove it so we can re-add
        kb._fallback_docs.clear()
        id2 = kb.add_document("same content", source="same.py")
        assert id1 == id2

    def test_add_multiple_documents_different_ids(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        id1 = kb.add_document("alpha content", source="a.py")
        id2 = kb.add_document("beta content", source="b.py")
        assert id1 != id2
        assert len(kb._fallback_docs) == 2


# -- KnowledgeBase query and ingest ----------------------------------------

class TestKnowledgeBaseQuery:
    """Tests for query (fallback keyword-overlap), ingest_directory, _chunk."""

    def test_query_returns_matching_docs(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        kb.add_document("python flask web framework", source="a.md", doc_id="d1")
        kb.add_document("java spring backend", source="b.md", doc_id="d2")
        results = kb.query("python flask")
        assert len(results) >= 1
        assert results[0].doc_id == "d1"

    def test_query_no_match(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        kb.add_document("alpha beta gamma", source="x.md", doc_id="d1")
        results = kb.query("zebra unicorn")
        assert results == []

    def test_query_category_filter(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        kb.add_document("python code example", source="a.py", category="code", doc_id="c1")
        kb.add_document("python docs tutorial", source="a.md", category="docs", doc_id="c2")
        results = kb.query("python", category="code")
        assert all(r.category == "code" for r in results)

    def test_query_k_limit(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        for i in range(10):
            kb.add_document(f"common word number {i}", source=f"{i}.md", doc_id=f"d{i}")
        results = kb.query("common word", k=3)
        assert len(results) <= 3

    def test_query_max_chars_truncates(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        # Each doc is 200 chars; budget is 350 so should get ~2
        for i in range(5):
            content = f"keyword {'a' * 190}"
            kb.add_document(content, source=f"{i}.md", doc_id=f"d{i}")
        results = kb.query("keyword", k=5, max_chars=350)
        total = sum(len(r.content) for r in results)
        assert total <= 350

    def test_query_score_is_positive(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        kb.add_document("hello world test", source="s.md", doc_id="d1")
        results = kb.query("hello world")
        assert len(results) == 1
        assert results[0].score > 0.0

    def test_query_sorted_by_relevance(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        kb.add_document("apple", source="a.md", doc_id="d1")
        kb.add_document("apple orange banana", source="b.md", doc_id="d2")
        results = kb.query("apple orange banana")
        # d2 should have higher overlap
        assert results[0].doc_id == "d2"

    def test_query_empty_string(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        kb.add_document("some content", source="a.md", doc_id="d1")
        # Empty query - no words overlap
        results = kb.query("")
        assert results == []

    # -- _chunk static method -----------------------------------------------

    def test_chunk_short_text(self):
        chunks = KnowledgeBase._chunk("short", 100, 10)
        assert chunks == ["short"]

    def test_chunk_exact_size(self):
        text = "a" * 100
        chunks = KnowledgeBase._chunk(text, 100, 10)
        assert chunks == [text]

    def test_chunk_splits_long_text(self):
        text = "a" * 250
        chunks = KnowledgeBase._chunk(text, 100, 20)
        assert len(chunks) >= 3
        # First chunk is 100 chars
        assert len(chunks[0]) == 100

    def test_chunk_overlap(self):
        text = "0123456789" * 5  # 50 chars
        chunks = KnowledgeBase._chunk(text, 20, 5)
        # Overlapping: chunk boundaries at 0, 15, 30, 45
        assert len(chunks) >= 3
        # Check overlap: last 5 of chunk[0] == first 5 of chunk[1]
        assert chunks[0][-5:] == chunks[1][:5]

    def test_chunk_zero_overlap(self):
        text = "abcdefghij" * 3  # 30 chars
        chunks = KnowledgeBase._chunk(text, 10, 0)
        assert len(chunks) == 3
        assert "".join(chunks) == text

    # -- ingest_directory ---------------------------------------------------

    def test_ingest_directory_creates_chunks(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path / "chroma"))
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "readme.md").write_text("hello world documentation", encoding="utf-8")
        count = kb.ingest_directory(str(docs_dir))
        assert count >= 1

    def test_ingest_directory_filters_by_extension(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path / "chroma"))
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "readme.md").write_text("markdown content", encoding="utf-8")
        (docs_dir / "data.csv").write_text("a,b,c", encoding="utf-8")
        count = kb.ingest_directory(str(docs_dir), extensions=[".md"])
        assert count >= 1
        # csv should not be ingested
        assert all("csv" not in d["source"] for d in kb._fallback_docs)

    def test_ingest_directory_recursive(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path / "chroma"))
        docs_dir = tmp_path / "docs"
        sub = docs_dir / "sub"
        sub.mkdir(parents=True)
        (docs_dir / "top.md").write_text("top level", encoding="utf-8")
        (sub / "nested.md").write_text("nested level", encoding="utf-8")
        count = kb.ingest_directory(str(docs_dir))
        assert count >= 2

    def test_ingest_directory_nonexistent_returns_zero(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path))
        count = kb.ingest_directory(str(tmp_path / "nope"))
        assert count == 0

    def test_ingest_directory_skips_pycache(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path / "chroma"))
        docs_dir = tmp_path / "project"
        cache = docs_dir / "__pycache__"
        cache.mkdir(parents=True)
        (cache / "cached.py").write_text("# cached", encoding="utf-8")
        (docs_dir / "real.py").write_text("# real code", encoding="utf-8")
        count = kb.ingest_directory(str(docs_dir))
        # Only real.py should be ingested
        sources = [d["source"] for d in kb._fallback_docs]
        assert all("__pycache__" not in s for s in sources)

    def test_ingest_directory_py_files_get_code_category(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path / "chroma"))
        docs_dir = tmp_path / "src"
        docs_dir.mkdir()
        (docs_dir / "main.py").write_text("print('hello')", encoding="utf-8")
        kb.ingest_directory(str(docs_dir))
        assert kb._fallback_docs[0]["category"] == "code"

    def test_ingest_directory_md_files_get_docs_category(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path / "chroma"))
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "guide.md").write_text("# Guide", encoding="utf-8")
        kb.ingest_directory(str(docs_dir))
        assert kb._fallback_docs[0]["category"] == "docs"

    def test_ingest_directory_with_chunk_size(self, tmp_path):
        kb = KnowledgeBase(persist_dir=str(tmp_path / "chroma"))
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        # Write a long file that needs chunking
        (docs_dir / "long.md").write_text("word " * 500, encoding="utf-8")
        count = kb.ingest_directory(str(docs_dir), chunk_size=100, chunk_overlap=10)
        assert count > 1


# =========================================================================
# 2. vetinari.migrations.upgrade_subtask_schema
# =========================================================================

from vetinari.migrations.upgrade_subtask_schema import migrate


class TestMigration:
    """Tests for the subtask schema migration function."""

    def _write_subtask_file(self, directory: Path, name: str, subtasks: list) -> Path:
        fp = directory / name
        fp.write_text(json.dumps({"subtasks": subtasks}, indent=2), encoding="utf-8")
        return fp

    def test_migrate_adds_ponder_fields(self, tmp_path):
        self._write_subtask_file(tmp_path, "task1.json", [
            {"name": "sub1", "status": "pending"},
        ])
        migrate(str(tmp_path))
        data = json.loads((tmp_path / "task1.json").read_text(encoding="utf-8"))
        st = data["subtasks"][0]
        assert st["ponder_ranking"] == []
        assert st["ponder_scores"] == {}
        assert st["ponder_used"] is False
        assert st["schema_version"] == 1

    def test_migrate_skips_already_migrated(self, tmp_path, capsys):
        self._write_subtask_file(tmp_path, "done.json", [
            {
                "name": "sub1",
                "ponder_ranking": [1],
                "ponder_scores": {"a": 1},
                "ponder_used": True,
                "schema_version": 1,
            },
        ])
        migrate(str(tmp_path))
        captured = capsys.readouterr()
        assert "Skipped (up to date)" in captured.out

    def test_migrate_multiple_subtasks(self, tmp_path):
        self._write_subtask_file(tmp_path, "multi.json", [
            {"name": "a"},
            {"name": "b"},
            {"name": "c"},
        ])
        migrate(str(tmp_path))
        data = json.loads((tmp_path / "multi.json").read_text(encoding="utf-8"))
        for st in data["subtasks"]:
            assert "ponder_ranking" in st
            assert "ponder_scores" in st
            assert "ponder_used" in st
            assert "schema_version" in st

    def test_migrate_multiple_files(self, tmp_path, capsys):
        for i in range(3):
            self._write_subtask_file(tmp_path, f"file{i}.json", [{"name": f"s{i}"}])
        migrate(str(tmp_path))
        captured = capsys.readouterr()
        # Per-file lines like "Migrated: file0.json"
        assert captured.out.count("Migrated: file") == 3
        # Summary line
        assert "Migrated: 3 files" in captured.out

    def test_migrate_empty_subtasks_list(self, tmp_path):
        self._write_subtask_file(tmp_path, "empty.json", [])
        migrate(str(tmp_path))
        data = json.loads((tmp_path / "empty.json").read_text(encoding="utf-8"))
        assert data["subtasks"] == []

    def test_migrate_nonexistent_directory(self, tmp_path, capsys):
        migrate(str(tmp_path / "nope"))
        captured = capsys.readouterr()
        assert "does not exist" in captured.out

    def test_migrate_counts_migrated(self, tmp_path, capsys):
        self._write_subtask_file(tmp_path, "a.json", [{"name": "x"}])
        self._write_subtask_file(tmp_path, "b.json", [{"name": "y"}])
        migrate(str(tmp_path))
        captured = capsys.readouterr()
        assert "Migrated: 2 files" in captured.out

    def test_migrate_error_count_on_bad_json(self, tmp_path, capsys):
        (tmp_path / "bad.json").write_text("not valid json{{{", encoding="utf-8")
        migrate(str(tmp_path))
        captured = capsys.readouterr()
        assert "Errors: 1 files" in captured.out

    def test_migrate_preserves_existing_fields(self, tmp_path):
        self._write_subtask_file(tmp_path, "keep.json", [
            {"name": "s1", "status": "done", "extra": 42},
        ])
        migrate(str(tmp_path))
        data = json.loads((tmp_path / "keep.json").read_text(encoding="utf-8"))
        st = data["subtasks"][0]
        assert st["name"] == "s1"
        assert st["status"] == "done"
        assert st["extra"] == 42

    def test_migrate_partial_fields_adds_missing(self, tmp_path):
        self._write_subtask_file(tmp_path, "partial.json", [
            {"name": "s", "ponder_ranking": [1, 2]},
        ])
        migrate(str(tmp_path))
        data = json.loads((tmp_path / "partial.json").read_text(encoding="utf-8"))
        st = data["subtasks"][0]
        assert st["ponder_ranking"] == [1, 2]  # preserved
        assert st["ponder_scores"] == {}  # added
        assert st["ponder_used"] is False  # added
        assert st["schema_version"] == 1  # added

    def test_migrate_ignores_non_json_files(self, tmp_path):
        (tmp_path / "readme.txt").write_text("not a subtask file", encoding="utf-8")
        self._write_subtask_file(tmp_path, "real.json", [{"name": "x"}])
        migrate(str(tmp_path))
        data = json.loads((tmp_path / "real.json").read_text(encoding="utf-8"))
        assert "ponder_ranking" in data["subtasks"][0]
        # txt file unchanged
        assert (tmp_path / "readme.txt").read_text(encoding="utf-8") == "not a subtask file"

    def test_migrate_default_storage_root_type(self):
        # When called with None, it should construct a Path; just verify it
        # doesn't crash (the directory won't exist, so it prints a message).
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            migrate(None)
        # Should print "does not exist" unless the default dir happens to exist
        # In test env it likely doesn't exist, but this should not raise

    def test_migrate_writes_indented_json(self, tmp_path):
        self._write_subtask_file(tmp_path, "indent.json", [{"name": "x"}])
        migrate(str(tmp_path))
        raw = (tmp_path / "indent.json").read_text(encoding="utf-8")
        # json.dump with indent=2 produces formatted output
        assert "\n" in raw
        assert "  " in raw


# =========================================================================
# 3. vetinari.tools.tool_registry_integration
# =========================================================================
# This module auto-registers on import and imports from vetinari.tool_interface
# and vetinari.execution_context.  We mock those to isolate the tests.


def _make_mock_tool_interface():
    """Create a mock vetinari.tool_interface module with real-enough classes."""
    mod = MagicMock()

    class _ToolCategory:
        SEARCH_ANALYSIS = "search_analysis"
        CODE_EXECUTION = "code_execution"
        MODEL_INFERENCE = "model_inference"
        SYSTEM_OPERATIONS = "system_operations"
        IMPLEMENTATION = "implementation"

    class _ToolParameter:
        def __init__(self, name="", type=str, description="", required=True, default=None, allowed_values=None):
            self.name = name
            self.type = type
            self.description = description
            self.required = required
            self.default = default
            self.allowed_values = allowed_values

    class _ToolMetadata:
        def __init__(self, name="", description="", category=None, version="1.0.0",
                     author="Vetinari", parameters=None, required_permissions=None,
                     allowed_modes=None, tags=None):
            self.name = name
            self.description = description
            self.category = category
            self.version = version
            self.author = author
            self.parameters = parameters or []
            self.required_permissions = required_permissions or []
            self.allowed_modes = allowed_modes or []
            self.tags = tags or []

    class _ToolResult:
        def __init__(self, success=False, output=None, error=None, execution_time_ms=0, metadata=None):
            self.success = success
            self.output = output
            self.error = error
            self.execution_time_ms = execution_time_ms
            self.metadata = metadata or {}

        def to_dict(self):
            return {"success": self.success, "output": self.output, "error": self.error}

    class _Tool:
        def __init__(self, metadata):
            self.metadata = metadata

        def execute(self, **kwargs):
            raise NotImplementedError

    class _ToolRegistry:
        def __init__(self):
            self._tools = {}

        def register(self, tool):
            self._tools[tool.metadata.name] = tool

        def list_tools(self):
            return list(self._tools.values())

    _registry_instance = _ToolRegistry()

    mod.Tool = _Tool
    mod.ToolMetadata = _ToolMetadata
    mod.ToolParameter = _ToolParameter
    mod.ToolResult = _ToolResult
    mod.ToolCategory = _ToolCategory
    mod.get_tool_registry = lambda: _registry_instance
    mod._registry_instance = _registry_instance
    return mod


def _make_mock_exec_context():
    """Create a mock vetinari.execution_context module."""
    mod = MagicMock()

    class _ToolPermission:
        WEB_ACCESS = "web_access"
        CODE_EXECUTION = "code_execution"
        MODEL_INFERENCE = "model_inference"
        PLANNING = "planning"
        FILE_READ = "file_read"
        FILE_WRITE = "file_write"

    class _ExecutionMode:
        EXECUTION = "execution"
        SANDBOX = "sandbox"
        PLANNING = "planning"

    mod.ToolPermission = _ToolPermission
    mod.ExecutionMode = _ExecutionMode
    return mod


@pytest.fixture
def tool_registry_env():
    """Set up mocked modules, import tool_registry_integration, then clean up."""
    mock_ti = _make_mock_tool_interface()
    mock_ec = _make_mock_exec_context()

    # Remove cached module so we can re-import with mocks
    sys.modules.pop("vetinari.tools.tool_registry_integration", None)

    saved = {}
    for name in ("vetinari.tool_interface", "vetinari.execution_context"):
        saved[name] = sys.modules.get(name)

    sys.modules["vetinari.tool_interface"] = mock_ti
    sys.modules["vetinari.execution_context"] = mock_ec

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import vetinari.tools.tool_registry_integration as tri_mod

    yield tri_mod, mock_ti

    # Restore
    sys.modules.pop("vetinari.tools.tool_registry_integration", None)
    for name, val in saved.items():
        if val is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = val


class TestToolWrappers:
    """Tests for each of the 7 Tool wrapper classes in tool_registry_integration."""

    def test_deprecation_warning_on_import(self):
        mock_ti = _make_mock_tool_interface()
        mock_ec = _make_mock_exec_context()
        sys.modules.pop("vetinari.tools.tool_registry_integration", None)
        saved_ti = sys.modules.get("vetinari.tool_interface")
        saved_ec = sys.modules.get("vetinari.execution_context")
        sys.modules["vetinari.tool_interface"] = mock_ti
        sys.modules["vetinari.execution_context"] = mock_ec
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                import vetinari.tools.tool_registry_integration  # noqa: F811
                # Module is no longer deprecated — verify no deprecation warning
                dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)
                                and "tool_registry_integration" in str(x.message)]
                assert len(dep_warnings) == 0
        finally:
            sys.modules.pop("vetinari.tools.tool_registry_integration", None)
            if saved_ti is None:
                sys.modules.pop("vetinari.tool_interface", None)
            else:
                sys.modules["vetinari.tool_interface"] = saved_ti
            if saved_ec is None:
                sys.modules.pop("vetinari.execution_context", None)
            else:
                sys.modules["vetinari.execution_context"] = saved_ec

    # -- WebSearchToolWrapper -----------------------------------------------

    def test_web_search_wrapper_metadata(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.WebSearchToolWrapper()
        assert w.metadata.name == "web_search"
        assert "search" in w.metadata.tags

    def test_web_search_execute_success(self, tool_registry_env):
        tri_mod, mock_ti = tool_registry_env
        w = tri_mod.WebSearchToolWrapper()
        mock_search_tool = MagicMock()
        mock_response = MagicMock()
        mock_response.results = []
        mock_response.query = "test"
        mock_response.total_results = 0
        mock_response.get_citations.return_value = []
        mock_response.execution_time_ms = 10
        mock_search_tool.search.return_value = mock_response
        w._search_tool = mock_search_tool

        result = w.execute(query="test", max_results=3)
        assert result.success is True
        mock_search_tool.search.assert_called_once_with("test", max_results=3)

    def test_web_search_execute_failure(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.WebSearchToolWrapper()
        w._search_tool = MagicMock(side_effect=Exception("network error"))
        # The _get_search_tool won't be called since _search_tool is set,
        # but search itself will fail
        mock_st = MagicMock()
        mock_st.search.side_effect = Exception("network error")
        w._search_tool = mock_st
        result = w.execute(query="fail")
        assert result.success is False
        assert "network error" in result.error

    # -- ResearchTopicToolWrapper -------------------------------------------

    def test_research_topic_wrapper_metadata(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.ResearchTopicToolWrapper()
        assert w.metadata.name == "research_topic"

    def test_research_topic_execute_success(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.ResearchTopicToolWrapper()
        mock_st = MagicMock()
        mock_st.research_topic.return_value = {"summary": "findings"}
        w._search_tool = mock_st
        result = w.execute(topic="AI safety")
        assert result.success is True
        assert result.output == {"summary": "findings"}

    def test_research_topic_execute_failure(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.ResearchTopicToolWrapper()
        mock_st = MagicMock()
        mock_st.research_topic.side_effect = RuntimeError("api down")
        w._search_tool = mock_st
        result = w.execute(topic="nothing")
        assert result.success is False
        assert "api down" in result.error

    # -- CodeExecutionToolWrapper -------------------------------------------

    def test_code_execution_wrapper_metadata(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.CodeExecutionToolWrapper()
        assert w.metadata.name == "execute_code"
        assert "code" in w.metadata.tags

    def test_code_execution_execute_success(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.CodeExecutionToolWrapper()
        mock_exec = MagicMock()
        mock_exec.run.return_value = {"success": True, "stdout": "hello"}
        w._executor = mock_exec
        result = w.execute(code="print('hello')", language="python", timeout=30)
        assert result.success is True
        mock_exec.run.assert_called_once_with("print('hello')", language="python", timeout=30)

    def test_code_execution_execute_failure(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.CodeExecutionToolWrapper()
        mock_exec = MagicMock()
        mock_exec.run.side_effect = Exception("sandbox error")
        w._executor = mock_exec
        result = w.execute(code="bad")
        assert result.success is False

    # -- MemoryRecallToolWrapper --------------------------------------------

    def test_memory_recall_wrapper_metadata(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.MemoryRecallToolWrapper()
        assert w.metadata.name == "recall_memory"

    def test_memory_recall_execute_success(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.MemoryRecallToolWrapper()
        mock_mem = MagicMock()
        entry = MagicMock()
        entry.to_dict.return_value = {"id": "1", "content": "recalled"}
        mock_mem.search.return_value = [entry]
        w._memory = mock_mem
        result = w.execute(query="test", limit=3)
        assert result.success is True
        assert result.output["count"] == 1

    def test_memory_recall_execute_failure(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.MemoryRecallToolWrapper()
        mock_mem = MagicMock()
        mock_mem.search.side_effect = Exception("db error")
        w._memory = mock_mem
        result = w.execute(query="fail")
        assert result.success is False

    # -- MemoryRememberToolWrapper ------------------------------------------

    def test_memory_remember_wrapper_metadata(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.MemoryRememberToolWrapper()
        assert w.metadata.name == "remember"

    def test_memory_remember_execute_success(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.MemoryRememberToolWrapper()
        mock_mem = MagicMock()
        mock_mem.remember.return_value = "entry_123"
        w._memory = mock_mem

        result = w.execute(content="remember this", memory_type="context", tags=["test"])
        assert result.success is True
        assert result.output["entry_id"] == "entry_123"

    def test_memory_remember_execute_failure(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.MemoryRememberToolWrapper()
        mock_mem = MagicMock()
        mock_mem.remember.side_effect = ValueError("bad type")
        w._memory = mock_mem
        result = w.execute(content="x")
        assert result.success is False

    # -- ModelSelectToolWrapper ---------------------------------------------

    def test_model_select_wrapper_metadata(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.ModelSelectToolWrapper()
        assert w.metadata.name == "select_model"

    def test_model_select_execute_success(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.ModelSelectToolWrapper()
        mock_router = MagicMock()
        selection = MagicMock()
        selection.model.id = "qwen-32b"
        selection.model.name = "Qwen 32B"
        selection.reasoning = "best for code"
        selection.confidence = 0.95
        selection.alternatives = []
        mock_router.select_model.return_value = selection
        w._router = mock_router

        mock_dmr = MagicMock()
        mock_dmr.TaskType.return_value = "coding"
        with patch.dict(sys.modules, {"vetinari.dynamic_model_router": mock_dmr}):
            result = w.execute(task_type="coding", task_description="write code")
        assert result.success is True
        assert result.output["model_id"] == "qwen-32b"

    def test_model_select_execute_no_model(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.ModelSelectToolWrapper()
        mock_router = MagicMock()
        mock_router.select_model.return_value = None
        w._router = mock_router
        mock_dmr = MagicMock()
        mock_dmr.TaskType.return_value = "general"
        with patch.dict(sys.modules, {"vetinari.dynamic_model_router": mock_dmr}):
            result = w.execute(task_type="general")
        assert result.success is False
        assert "No suitable model" in result.error

    # -- GeneratePlanToolWrapper --------------------------------------------

    def test_generate_plan_wrapper_metadata(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.GeneratePlanToolWrapper()
        assert w.metadata.name == "generate_plan"
        assert "planning" in w.metadata.tags

    def test_generate_plan_execute_success(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.GeneratePlanToolWrapper()
        mock_orch = MagicMock()
        mock_graph = MagicMock()
        mock_graph.to_dict.return_value = {"nodes": [], "edges": []}
        mock_orch.generate_plan_only.return_value = mock_graph
        w._orchestrator = mock_orch
        result = w.execute(goal="build a web app")
        assert result.success is True
        assert result.output == {"nodes": [], "edges": []}

    def test_generate_plan_execute_failure(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.GeneratePlanToolWrapper()
        mock_orch = MagicMock()
        mock_orch.generate_plan_only.side_effect = Exception("plan failed")
        w._orchestrator = mock_orch
        result = w.execute(goal="impossible")
        assert result.success is False
        assert "plan failed" in result.error

    # -- Lazy import patterns -----------------------------------------------

    def test_web_search_lazy_import(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.WebSearchToolWrapper()
        assert w._search_tool is None
        mock_module = MagicMock()
        mock_module.get_search_tool.return_value = MagicMock()
        with patch.dict(sys.modules, {"vetinari.tools.web_search_tool": mock_module}):
            tool = w._get_search_tool()
            assert tool is not None
            assert w._search_tool is not None

    def test_code_execution_lazy_import(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.CodeExecutionToolWrapper()
        assert w._executor is None
        mock_module = MagicMock()
        mock_module.get_code_executor.return_value = MagicMock()
        with patch.dict(sys.modules, {"vetinari.sandbox": mock_module}):
            ex = w._get_executor()
            assert ex is not None

    def test_memory_recall_lazy_import(self, tool_registry_env):
        tri_mod, _ = tool_registry_env
        w = tri_mod.MemoryRecallToolWrapper()
        assert w._memory is None
        mock_module = MagicMock()
        mock_module.get_dual_memory_store.return_value = MagicMock()
        with patch.dict(sys.modules, {"vetinari.memory": mock_module}):
            mem = w._get_memory()
            assert mem is not None


class TestRegisterAllTools:
    """Tests for the register_all_tools() function."""

    def test_register_all_tools_returns_count(self, tool_registry_env):
        tri_mod, mock_ti = tool_registry_env
        count = tri_mod.register_all_tools()
        assert count == 7

    def test_register_all_tools_populates_registry(self, tool_registry_env):
        tri_mod, mock_ti = tool_registry_env
        # Clear and re-register
        registry = mock_ti.get_tool_registry()
        registry._tools.clear()
        tri_mod.register_all_tools()
        tools = registry.list_tools()
        assert len(tools) == 7

    def test_registered_tool_names(self, tool_registry_env):
        tri_mod, mock_ti = tool_registry_env
        registry = mock_ti.get_tool_registry()
        registry._tools.clear()
        tri_mod.register_all_tools()
        names = {t.metadata.name for t in registry.list_tools()}
        expected = {
            "web_search", "research_topic", "execute_code",
            "recall_memory", "remember", "select_model", "generate_plan",
        }
        assert names == expected

    def test_register_all_tools_handles_exception(self, tool_registry_env):
        tri_mod, mock_ti = tool_registry_env
        # Make registry.register raise for one tool
        original_register = mock_ti.get_tool_registry().register

        call_count = [0]

        def flaky_register(tool):
            call_count[0] += 1
            if call_count[0] == 3:
                raise RuntimeError("boom")
            original_register(tool)

        mock_ti.get_tool_registry().register = flaky_register
        # Should not raise
        count = tri_mod.register_all_tools()
        assert count == 7  # count is len(tools), not successful registrations


# =========================================================================
# 4. vetinari.web.config
# =========================================================================

# Reset module-level singleton before each import
sys.modules.pop("vetinari.web.config", None)
sys.modules.pop("vetinari.web", None)

from vetinari.web.config import VetinariConfig, get_config
import vetinari.web.config as _web_config_mod


class TestVetinariConfig:
    """Tests for the VetinariConfig dataclass and from_env()."""

    @pytest.fixture(autouse=True)
    def _reset_config_singleton(self):
        _web_config_mod._config = None
        yield
        _web_config_mod._config = None

    def test_default_values(self):
        cfg = VetinariConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 5000
        assert cfg.debug is False
        assert cfg.lm_studio_host == "http://localhost:1234"
        assert cfg.api_token == ""
        assert cfg.memory_budget_gb == 48
        assert cfg.max_concurrent_tasks == 4
        assert cfg.default_timeout == 120
        assert cfg.llm_timeout == 300
        assert cfg.enable_external_discovery is True

    def test_default_models_list(self):
        cfg = VetinariConfig()
        assert isinstance(cfg.default_models, list)
        assert len(cfg.default_models) >= 1

    def test_fallback_models_list(self):
        cfg = VetinariConfig()
        assert isinstance(cfg.fallback_models, list)
        assert len(cfg.fallback_models) >= 1

    def test_uncensored_fallback_models_list(self):
        cfg = VetinariConfig()
        assert isinstance(cfg.uncensored_fallback_models, list)

    def test_from_env_defaults(self, monkeypatch):
        # Clear all relevant env vars
        for var in ("VETINARI_WEB_HOST", "VETINARI_WEB_PORT", "FLASK_DEBUG",
                     "LM_STUDIO_HOST", "LM_STUDIO_API_TOKEN", "VETINARI_API_TOKEN",
                     "VETINARI_CONFIG", "VETINARI_PROJECT_DIR", "VETINARI_OUTPUT_DIR",
                     "VETINARI_MEMORY_GB", "VETINARI_MAX_CONCURRENT", "VETINARI_TIMEOUT",
                     "VETINARI_LLM_TIMEOUT", "ENABLE_EXTERNAL_DISCOVERY"):
            monkeypatch.delenv(var, raising=False)
        cfg = VetinariConfig.from_env()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 5000
        assert cfg.debug is False

    def test_from_env_custom_host_port(self, monkeypatch):
        monkeypatch.setenv("VETINARI_WEB_HOST", "0.0.0.0")
        monkeypatch.setenv("VETINARI_WEB_PORT", "8080")
        cfg = VetinariConfig.from_env()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8080

    def test_from_env_debug_true(self, monkeypatch):
        monkeypatch.setenv("FLASK_DEBUG", "1")
        cfg = VetinariConfig.from_env()
        assert cfg.debug is True

    def test_from_env_debug_yes(self, monkeypatch):
        monkeypatch.setenv("FLASK_DEBUG", "yes")
        cfg = VetinariConfig.from_env()
        assert cfg.debug is True

    def test_from_env_debug_false(self, monkeypatch):
        monkeypatch.setenv("FLASK_DEBUG", "0")
        cfg = VetinariConfig.from_env()
        assert cfg.debug is False

    def test_from_env_lm_studio_host(self, monkeypatch):
        monkeypatch.setenv("LM_STUDIO_HOST", "http://gpu-box:1234")
        cfg = VetinariConfig.from_env()
        assert cfg.lm_studio_host == "http://gpu-box:1234"

    def test_from_env_api_token_primary(self, monkeypatch):
        monkeypatch.setenv("LM_STUDIO_API_TOKEN", "tok_primary")
        monkeypatch.delenv("VETINARI_API_TOKEN", raising=False)
        cfg = VetinariConfig.from_env()
        assert cfg.api_token == "tok_primary"

    def test_from_env_api_token_fallback(self, monkeypatch):
        monkeypatch.delenv("LM_STUDIO_API_TOKEN", raising=False)
        monkeypatch.setenv("VETINARI_API_TOKEN", "tok_fallback")
        cfg = VetinariConfig.from_env()
        assert cfg.api_token == "tok_fallback"

    def test_from_env_api_token_primary_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("LM_STUDIO_API_TOKEN", "primary")
        monkeypatch.setenv("VETINARI_API_TOKEN", "fallback")
        cfg = VetinariConfig.from_env()
        assert cfg.api_token == "primary"

    def test_from_env_memory_budget(self, monkeypatch):
        monkeypatch.setenv("VETINARI_MEMORY_GB", "16")
        cfg = VetinariConfig.from_env()
        assert cfg.memory_budget_gb == 16

    def test_from_env_max_concurrent(self, monkeypatch):
        monkeypatch.setenv("VETINARI_MAX_CONCURRENT", "8")
        cfg = VetinariConfig.from_env()
        assert cfg.max_concurrent_tasks == 8

    def test_from_env_timeouts(self, monkeypatch):
        monkeypatch.setenv("VETINARI_TIMEOUT", "60")
        monkeypatch.setenv("VETINARI_LLM_TIMEOUT", "600")
        cfg = VetinariConfig.from_env()
        assert cfg.default_timeout == 60
        assert cfg.llm_timeout == 600

    def test_from_env_external_discovery_false(self, monkeypatch):
        monkeypatch.setenv("ENABLE_EXTERNAL_DISCOVERY", "false")
        cfg = VetinariConfig.from_env()
        assert cfg.enable_external_discovery is False

    def test_from_env_external_discovery_zero(self, monkeypatch):
        monkeypatch.setenv("ENABLE_EXTERNAL_DISCOVERY", "0")
        cfg = VetinariConfig.from_env()
        assert cfg.enable_external_discovery is False

    def test_from_env_config_path(self, monkeypatch):
        monkeypatch.setenv("VETINARI_CONFIG", "/custom/path.yaml")
        cfg = VetinariConfig.from_env()
        assert cfg.config_path == "/custom/path.yaml"

    def test_from_env_project_and_output_dir(self, monkeypatch):
        monkeypatch.setenv("VETINARI_PROJECT_DIR", "/proj")
        monkeypatch.setenv("VETINARI_OUTPUT_DIR", "/out")
        cfg = VetinariConfig.from_env()
        assert cfg.project_dir == "/proj"
        assert cfg.output_dir == "/out"

    def test_to_dict(self):
        cfg = VetinariConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["host"] == "127.0.0.1"
        assert d["port"] == 5000
        assert "default_models" in d
        assert "fallback_models" in d
        assert "memory_budget_gb" in d

    def test_to_dict_includes_all_fields(self):
        cfg = VetinariConfig()
        d = cfg.to_dict()
        field_names = {f.name for f in cfg.__dataclass_fields__.values()}
        assert set(d.keys()) == field_names

    def test_get_config_singleton(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_get_config_returns_vetinari_config(self):
        cfg = get_config()
        assert isinstance(cfg, VetinariConfig)

    def test_custom_construction(self):
        cfg = VetinariConfig(host="10.0.0.1", port=9999, debug=True)
        assert cfg.host == "10.0.0.1"
        assert cfg.port == 9999
        assert cfg.debug is True

    def test_model_lists_are_independent_across_instances(self):
        a = VetinariConfig()
        b = VetinariConfig()
        a.default_models.append("new_model")
        assert "new_model" not in b.default_models


# =========================================================================
# 5. vetinari.skills.data_engineer
# =========================================================================
# This module imports from vetinari.tool_interface and
# vetinari.execution_context at module level. We need those mocks active.


@pytest.fixture
def data_engineer_env():
    """Set up mocked modules, import data_engineer skill, then clean up."""
    mock_ti = _make_mock_tool_interface()
    mock_ec = _make_mock_exec_context()

    sys.modules.pop("vetinari.skills.data_engineer", None)

    saved = {}
    for name in ("vetinari.tool_interface", "vetinari.execution_context"):
        saved[name] = sys.modules.get(name)

    sys.modules["vetinari.tool_interface"] = mock_ti
    sys.modules["vetinari.execution_context"] = mock_ec

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import vetinari.skills.data_engineer as de_mod

    yield de_mod, mock_ti, mock_ec

    sys.modules.pop("vetinari.skills.data_engineer", None)
    for name, val in saved.items():
        if val is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = val


class TestDataEngineerSkill:
    """Tests for the deprecated DataEngineerSkill."""

    def test_deprecation_warning(self):
        mock_ti = _make_mock_tool_interface()
        mock_ec = _make_mock_exec_context()
        sys.modules.pop("vetinari.skills.data_engineer", None)
        saved_ti = sys.modules.get("vetinari.tool_interface")
        saved_ec = sys.modules.get("vetinari.execution_context")
        sys.modules["vetinari.tool_interface"] = mock_ti
        sys.modules["vetinari.execution_context"] = mock_ec
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                from vetinari.skills.data_engineer import DataEngineerSkill
                skill = DataEngineerSkill()
                dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
                assert len(dep) >= 1
                assert "deprecated" in str(dep[0].message).lower()
        finally:
            sys.modules.pop("vetinari.skills.data_engineer", None)
            if saved_ti is None:
                sys.modules.pop("vetinari.tool_interface", None)
            else:
                sys.modules["vetinari.tool_interface"] = saved_ti
            if saved_ec is None:
                sys.modules.pop("vetinari.execution_context", None)
            else:
                sys.modules["vetinari.execution_context"] = saved_ec

    def test_metadata_name(self, data_engineer_env):
        de_mod, _, _ = data_engineer_env
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            skill = de_mod.DataEngineerSkill()
        assert skill.metadata.name == "data_engineer"

    def test_metadata_category(self, data_engineer_env):
        de_mod, mock_ti, _ = data_engineer_env
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            skill = de_mod.DataEngineerSkill()
        assert skill.metadata.category == mock_ti.ToolCategory.IMPLEMENTATION

    def test_metadata_tags(self, data_engineer_env):
        de_mod, _, _ = data_engineer_env
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            skill = de_mod.DataEngineerSkill()
        assert "data" in skill.metadata.tags
        assert "pipeline" in skill.metadata.tags

    def test_metadata_has_parameters(self, data_engineer_env):
        de_mod, _, _ = data_engineer_env
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            skill = de_mod.DataEngineerSkill()
        param_names = [p.name for p in skill.metadata.parameters]
        assert "task" in param_names

    def test_execute_agent_not_available(self, data_engineer_env):
        de_mod, _, _ = data_engineer_env
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            skill = de_mod.DataEngineerSkill()
        # _get_agent will fail since the real agent module isn't available
        skill._agent = None
        # Mock the import to fail
        with patch.dict(sys.modules, {"vetinari.agents.data_engineer_agent": None}):
            result = skill.execute(task="build pipeline")
        assert result.success is False
        assert "not available" in result.error.lower()

    def test_execute_success_with_mocked_agent(self, data_engineer_env):
        de_mod, _, _ = data_engineer_env
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            skill = de_mod.DataEngineerSkill()

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {"pipeline": "created"}
        mock_result.errors = []
        mock_agent.execute.return_value = mock_result
        skill._agent = mock_agent

        # Mock the contracts import inside execute
        mock_contracts = MagicMock()
        mock_contracts.AgentTask.return_value = MagicMock()
        mock_contracts.AgentType.DATA_ENGINEER = "data_engineer"
        with patch.dict(sys.modules, {"vetinari.agents.contracts": mock_contracts}):
            result = skill.execute(task="build ETL pipeline")
        assert result.success is True
        assert result.output == {"pipeline": "created"}

    def test_execute_with_errors(self, data_engineer_env):
        de_mod, _, _ = data_engineer_env
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            skill = de_mod.DataEngineerSkill()

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.output = None
        mock_result.errors = ["schema invalid", "missing table"]
        mock_agent.execute.return_value = mock_result
        skill._agent = mock_agent

        mock_contracts = MagicMock()
        mock_contracts.AgentTask.return_value = MagicMock()
        mock_contracts.AgentType.DATA_ENGINEER = "data_engineer"
        with patch.dict(sys.modules, {"vetinari.agents.contracts": mock_contracts}):
            result = skill.execute(task="bad task")
        assert result.success is False
        assert "schema invalid" in result.error
        assert "missing table" in result.error

    def test_execute_exception_in_agent(self, data_engineer_env):
        de_mod, _, _ = data_engineer_env
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            skill = de_mod.DataEngineerSkill()

        mock_agent = MagicMock()
        mock_agent.execute.side_effect = RuntimeError("agent crashed")
        skill._agent = mock_agent

        mock_contracts = MagicMock()
        mock_contracts.AgentTask.return_value = MagicMock()
        mock_contracts.AgentType.DATA_ENGINEER = "data_engineer"
        with patch.dict(sys.modules, {"vetinari.agents.contracts": mock_contracts}):
            result = skill.execute(task="crash")
        assert result.success is False
        assert "agent crashed" in result.error

    def test_lazy_agent_loading(self, data_engineer_env):
        de_mod, _, _ = data_engineer_env
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            skill = de_mod.DataEngineerSkill()
        assert skill._agent is None

    def test_get_agent_caches(self, data_engineer_env):
        de_mod, _, _ = data_engineer_env
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            skill = de_mod.DataEngineerSkill()
        mock_agent = MagicMock()
        mock_de_mod = MagicMock()
        mock_de_mod.get_data_engineer_agent.return_value = mock_agent
        with patch.dict(sys.modules, {"vetinari.agents.data_engineer_agent": mock_de_mod}):
            a1 = skill._get_agent()
            a2 = skill._get_agent()
        assert a1 is a2
        assert a1 is mock_agent
        # get_data_engineer_agent called only once due to caching
        assert mock_de_mod.get_data_engineer_agent.call_count == 1

    def test_get_agent_returns_none_on_import_error(self, data_engineer_env):
        de_mod, _, _ = data_engineer_env
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            skill = de_mod.DataEngineerSkill()
        # Force a fresh attempt by ensuring _agent is None
        skill._agent = None
        # Simulate import failure
        with patch.dict(sys.modules, {"vetinari.agents.data_engineer_agent": None}):
            agent = skill._get_agent()
        assert agent is None

    def test_execute_passes_context_kwargs(self, data_engineer_env):
        de_mod, _, _ = data_engineer_env
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            skill = de_mod.DataEngineerSkill()

        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = {}
        mock_result.errors = []
        mock_agent.execute.return_value = mock_result
        skill._agent = mock_agent

        mock_contracts = MagicMock()
        # Capture the AgentTask construction args
        captured_tasks = []
        original_task_cls = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
        mock_contracts.AgentTask = original_task_cls
        mock_contracts.AgentType.DATA_ENGINEER = "data_engineer"

        with patch.dict(sys.modules, {"vetinari.agents.contracts": mock_contracts}):
            skill.execute(task="build pipeline", data_source="postgres", target="s3")

        # Verify AgentTask was called with context containing kwargs
        call_kwargs = original_task_cls.call_args[1]
        assert call_kwargs["task_id"] == "data-eng"
        assert call_kwargs["description"] == "build pipeline"
        ctx = call_kwargs["context"]
        assert ctx["data_source"] == "postgres"
        assert ctx["target"] == "s3"
