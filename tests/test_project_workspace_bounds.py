"""Focused bounds tests for project workspace and export handlers."""

from __future__ import annotations

import asyncio

import vetinari.web.litestar_projects_api as projects_api


def _handler(path: str):
    for handler in projects_api.create_projects_api_handlers():
        if path in handler.paths:
            return handler.fn
    raise AssertionError(f"handler not found for {path}")


def test_workspace_read_rejects_oversized_text_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(projects_api, "_WORKSPACE_READ_MAX_BYTES", 4)
    from vetinari.web import shared

    monkeypatch.setattr(shared, "PROJECT_ROOT", tmp_path)
    workspace = tmp_path / "projects" / "p1" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "large.txt").write_text("too large", encoding="utf-8")

    response = asyncio.run(_handler("/api/project/{project_id:str}/files/read")("p1", {"path": "large.txt"}))

    assert response.status_code == 413
    assert response.content["details"]["max_bytes"] == 4


def test_workspace_list_reports_truncation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(projects_api, "_WORKSPACE_LIST_MAX_FILES", 2)
    from vetinari.web import shared

    monkeypatch.setattr(shared, "PROJECT_ROOT", tmp_path)
    workspace = tmp_path / "projects" / "p1" / "workspace"
    workspace.mkdir(parents=True)
    for index in range(3):
        (workspace / f"{index}.txt").write_text(str(index), encoding="utf-8")

    response = asyncio.run(_handler("/api/project/{project_id:str}/files/list")("p1"))

    assert response.status_code == 200
    assert response.content["truncated"] is True
    assert len(response.content["files"]) == 2


def test_project_download_rejects_export_byte_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(projects_api, "_PROJECT_EXPORT_MAX_BYTES", 4)
    from vetinari.web import shared

    monkeypatch.setattr(shared, "PROJECT_ROOT", tmp_path)
    outputs = tmp_path / "projects" / "p1" / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "large.txt").write_text("too large", encoding="utf-8")

    response = asyncio.run(_handler("/api/project/{project_id:str}/download")("p1"))

    assert response.status_code == 413
    assert response.content["details"]["max_bytes"] == 4
