"""Regression proofs for SESSION-34F3 boundary hardening."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vetinari.agents.builder_agent import BuilderAgent
from vetinari.code_sandbox import CodeSandbox
from vetinari.rag import knowledge_base_helpers as kb_helpers
from vetinari.rag.knowledge_base import KnowledgeBase
from vetinari.tools.web_search_backends import search_wikipedia
from vetinari.verification.sandbox_verifier import verify_code


def test_embedding_remote_endpoint_blocked_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VETINARI_ALLOW_REMOTE_EMBEDDINGS", raising=False)
    monkeypatch.delenv("VETINARI_EMBEDDING_API_ALLOWLIST", raising=False)
    monkeypatch.setattr(kb_helpers, "_EMBEDDING_API_URL", "https://embeddings.example.com")
    with patch("httpx.post") as mock_post:
        assert kb_helpers.kb_embed("private document text") is None
    mock_post.assert_not_called()


def test_ingest_directory_skips_symlink_escape(tmp_path: Path) -> None:
    kb = KnowledgeBase(db_path=str(tmp_path / "kb.db"))
    docs = tmp_path / "docs"
    docs.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside secret", encoding="utf-8")
    link = docs / "linked.md"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable on this platform: {exc}")

    with patch("vetinari.rag.knowledge_base._embed", return_value=None):
        assert kb.ingest_directory(str(docs)) == 0


def test_ingest_url_requires_https_after_ssrf_validation(tmp_path: Path) -> None:
    kb = KnowledgeBase(db_path=str(tmp_path / "kb.db"))
    insecure_url = "http://example.com/doc"  # noqa: VET303 - verifies rejection of insecure RAG ingestion input.
    with patch("vetinari.security.validate_url_no_ssrf", return_value=insecure_url):
        with pytest.raises(ValueError, match="requires https"):
            kb.ingest_url(insecure_url)


def test_wikipedia_language_host_is_allowlisted() -> None:
    with patch("requests.get") as mock_get:
        assert search_wikipedia("query", max_results=5, language="en.evil", time_range=None) == []
    mock_get.assert_not_called()


def test_wikipedia_uses_fixed_host_and_disables_redirects() -> None:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"query": {"search": [{"pageid": 1, "title": "T", "snippet": "S"}]}}
    with patch("requests.get", return_value=resp) as mock_get:
        results = search_wikipedia("query", max_results=1, language="en-US", time_range=None)

    assert results[0].url == "https://en.wikipedia.org/wiki?curid=1"
    assert mock_get.call_args.args[0] == "https://en.wikipedia.org/w/api.php"
    assert mock_get.call_args.kwargs["allow_redirects"] is False


def test_run_tests_does_not_forward_parent_secret_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    sandbox = CodeSandbox(working_dir=str(tmp_path))
    captured: dict[str, object] = {}

    def capture_run(*args: object, **kwargs: object) -> object:
        captured["env"] = kwargs.get("env")
        return MagicMock(returncode=0, stdout="passed", stderr="")

    with patch("subprocess.run", side_effect=capture_run):
        assert sandbox.run_tests().success is True

    assert isinstance(captured["env"], dict)
    assert "AWS_SECRET_ACCESS_KEY" not in captured["env"]


def test_sandbox_verifier_rejects_blocked_import_without_execution() -> None:
    result = verify_code("import subprocess\n", run_tests=False)
    assert result.passed is False
    assert result.imports_valid is False
    assert "blocked by sandbox policy" in (result.error_message or "")


def test_builder_rejects_generated_path_traversal(tmp_path: Path) -> None:
    agent = BuilderAgent.__new__(BuilderAgent)
    agent._has_tool = lambda _name: False  # type: ignore[method-assign]
    scaffold = {
        "scaffold_code": "",
        "tests": [{"filename": "../escape.py", "content": "x = 1"}],
        "artifacts": [],
    }

    assert agent._write_scaffold_to_disk(scaffold, str(tmp_path)) == []
    assert not (tmp_path.parent / "escape.py").exists()


def test_builder_blocks_direct_write_after_file_tool_failure(tmp_path: Path) -> None:
    agent = BuilderAgent.__new__(BuilderAgent)
    agent._has_tool = lambda _name: True  # type: ignore[method-assign]
    agent._use_tool = lambda *args, **kwargs: {"success": False}  # type: ignore[method-assign]
    target = tmp_path / "blocked.py"

    assert agent._tool_write_file(str(target), "x = 1") is False
    assert not target.exists()
