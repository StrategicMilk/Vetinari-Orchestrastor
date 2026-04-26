"""Session 34F1 auth, redaction, and Markdown boundary regressions."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from litestar import Litestar
from litestar.testing import TestClient

from vetinari.security.redaction import redact_route_payload, redact_text
from vetinari.web.litestar_chat_api import _content_disposition, _render_conversation_markdown
from vetinari.web.litestar_projects_formatters import build_final_report


def test_final_report_fence_cannot_be_broken_by_task_output() -> None:
    payload = "safe before\n```\n# injected heading\n<script>alert(1)</script>\n```\nsafe after"

    report = build_final_report(
        "probe-project",
        {"goal": "probe"},
        [{"id": "t1", "description": "unsafe export", "assigned_model": "worker", "output": payload}],
    )

    lines = report.splitlines()
    opening = lines.index("````text")
    injected = lines.index("# injected heading")
    closing = max(index for index, line in enumerate(lines) if line == "````")
    assert opening < injected < closing


def test_markdown_exports_redact_secrets_paths_and_provider_urls() -> None:
    payload = (
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456 "
        r"C:\Users\alice\.vetinari\secret.txt "
        "https://api.openai.com/v1/chat/completions?api_key=top-secret"
    )

    report = build_final_report(
        "probe-project",
        {"goal": "probe"},
        [{"id": "t1", "description": "secret export", "assigned_model": "worker", "output": payload}],
    )

    assert "abcdefghijklmnopqrstuvwxyz123456" not in report
    assert r"C:\Users\alice" not in report
    assert "api.openai.com" not in report
    assert "[REDACTED]" in report
    assert "[REDACTED_PATH]" in report
    assert "[REDACTED_URL]" in report


def test_chat_markdown_export_fences_untrusted_message_content() -> None:
    export = _render_conversation_markdown(
        [
            {
                "role": "assistant",
                "agent": "worker",
                "timestamp": "2026-04-22T00:00:00Z",
                "content": "```\n# injected\n<script>alert(1)</script>\n```",
            }
        ],
        "proj",
    )

    assert "\n````text\n```\n# injected\n<script>alert(1)</script>\n```\n````" in export


def test_redaction_covers_label_variants_paths_and_urls() -> None:
    payload = {
        "webhookUrl": "https://hooks.example.test/path?token=secret-token",
        "trace.logDetails": r"failed at C:\Users\alice\repo\file.py",
        "tool output": "provider URL: https://api.anthropic.com/v1/messages?key=secret",
        "safe": "visible",
    }

    redacted = redact_route_payload(payload)

    assert redacted["webhookUrl"] == "[REDACTED]"
    assert redacted["trace.logDetails"] == "[REDACTED]"
    assert redacted["tool output"] == "[REDACTED]"
    assert redacted["safe"] == "visible"
    assert "C:\\Users\\alice" not in redact_text(str(payload))


def test_content_disposition_encodes_hostile_filenames() -> None:
    header = _content_disposition("attachment", 'bad";\r\nx=.md')

    assert "\r" not in header
    assert "\n" not in header
    assert ";" in header
    assert 'filename="bad_x=.md"' in header
    assert "filename*=UTF-8''bad_x%3D.md" in header


def test_frontend_markdown_renderer_blocks_xss_vectors() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    svelte_root = Path(__file__).resolve().parents[1] / "ui" / "svelte"
    if not (svelte_root / "node_modules").is_dir():
        pytest.skip(
            "ui/svelte/node_modules missing — run `npm install` in ui/svelte to enable the XSS-vector probe",
        )

    script = r"""
import assert from 'node:assert/strict';
import { renderSafeMarkdown } from './src/lib/markdown.js';

const cases = [
  '<script>alert(1)</script>',
  '<img src=x onerror=alert(1)>',
  '[x](javascript:alert(1))',
  '[x](data:text/html,<script>alert(1)</script>)',
  '<a href="javascript:alert(1)" onclick="alert(1)">x</a>',
];

for (const input of cases) {
  const html = renderSafeMarkdown(input).toLowerCase();
  assert(!html.includes('<script'), html);
  assert(!html.includes('href="javascript:'), html);
  assert(!html.includes('href="data:'), html);
  assert(!html.includes('src="data:'), html);
  assert(!html.includes('<img'), html);
  assert(!html.includes('<a href="javascript:'), html);
}
"""
    result = subprocess.run(  # noqa: S603 - node path is resolved from the local toolchain for this renderer probe.
        [node, "--input-type=module"],
        cwd=Path(__file__).resolve().parents[1] / "ui" / "svelte",
        input=script,
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr + result.stdout


_ADMIN_TOKEN = "session34f1-admin-token"
_ADMIN_HEADERS = {"X-Admin-Token": _ADMIN_TOKEN}


def _projects_app() -> Litestar:
    import vetinari.web.litestar_projects_api
    from vetinari.web.litestar_projects_api import create_projects_api_handlers

    return Litestar(route_handlers=create_projects_api_handlers(), debug=True)


def _admin_app() -> Litestar:
    import vetinari.web.litestar_admin_routes
    from vetinari.web.litestar_admin_routes import create_admin_handlers

    return Litestar(route_handlers=create_admin_handlers(), debug=True)


def test_project_list_returns_relative_path_not_host_path(tmp_path: Path) -> None:
    project_dir = tmp_path / "projects" / "demo"
    project_dir.mkdir(parents=True)
    (project_dir / "project.yaml").write_text(yaml.safe_dump({"project_name": "Demo"}), encoding="utf-8")

    with patch("vetinari.web.shared.PROJECT_ROOT", tmp_path), TestClient(_projects_app()) as client:
        response = client.get("/api/projects")

    assert response.status_code == 200
    project = response.json()["projects"][0]
    assert project["path"] == "demo"
    assert str(tmp_path) not in response.text


def test_project_review_redacts_output_and_uses_relative_generated_paths(tmp_path: Path) -> None:
    project_dir = tmp_path / "projects" / "demo"
    output_dir = project_dir / "outputs" / "t1"
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True)
    (project_dir / "project.yaml").write_text(
        yaml.safe_dump({"high_level_goal": "demo", "tasks": [{"id": "t1", "description": "task"}]}),
        encoding="utf-8",
    )
    (output_dir / "output.txt").write_text(
        r"token=ghp_abcdefghijklmnopqrstuvwxyz C:\Users\alice\secret.txt",
        encoding="utf-8",
    )
    (generated_dir / "result.txt").write_text("ok", encoding="utf-8")

    with (
        patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}),
        patch("vetinari.web.shared.PROJECT_ROOT", tmp_path),
        TestClient(_projects_app()) as client,
    ):
        response = client.get("/api/project/demo/review", headers=_ADMIN_HEADERS)

    assert response.status_code == 200
    body = response.json()
    task = body["tasks"][0]
    assert "ghp_" not in task["output"]
    assert r"C:\Users\alice" not in task["output"]
    assert task["files"] == [{"name": "result.txt", "path": "outputs/t1/generated/result.txt"}]


def test_project_file_reads_require_admin_token(tmp_path: Path) -> None:
    project_dir = tmp_path / "projects" / "demo" / "workspace"
    project_dir.mkdir(parents=True)
    (project_dir / "safe.txt").write_text("ok", encoding="utf-8")

    with (
        patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}),
        patch("vetinari.web.shared.PROJECT_ROOT", tmp_path),
        TestClient(_projects_app()) as client,
    ):
        read_response = client.post("/api/project/demo/files/read", json={"path": "safe.txt"})
        list_response = client.get("/api/project/demo/files/list")

    assert read_response.status_code in {401, 403}
    assert list_response.status_code in {401, 403}


def test_admin_permissions_is_not_token_oracle() -> None:
    with patch.dict(os.environ, {"VETINARI_ADMIN_TOKEN": _ADMIN_TOKEN}), TestClient(_admin_app()) as client:
        unauthenticated = client.get("/api/admin/permissions")
        authenticated = client.get("/api/admin/permissions", headers=_ADMIN_HEADERS)

    assert unauthenticated.status_code in {401, 403}
    assert authenticated.status_code == 200
    assert authenticated.json() == {"admin": True}
