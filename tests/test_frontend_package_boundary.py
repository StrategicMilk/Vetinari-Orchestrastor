"""Frontend package-boundary proof tests."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None  # type: ignore[assignment]

import pytest

from vetinari.web.litestar_app import create_app

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
MANIFEST = ROOT / "MANIFEST.in"
SOURCES = ROOT / "vetinari.egg-info" / "SOURCES.txt"
PROOF_NOTES = (
    ROOT / "docs" / "runbooks" / "dashboard-guide.md",
    ROOT / "docs" / "architecture" / "decisions.md",
    ROOT / "docs" / "security" / "license-notes.md",
)


def _pyproject() -> dict:
    if tomllib is None:
        pytest.skip("tomllib is required to parse pyproject.toml")
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_package_boundary_excludes_frontend_parent_globs() -> None:
    """The release package must not sweep checked-in UI workspaces or bundles."""
    patterns = _pyproject()["tool"]["setuptools"]["package-data"]["vetinari"]
    joined_patterns = "\n".join(patterns)
    manifest = MANIFEST.read_text(encoding="utf-8")

    assert "../ui/**/*" not in patterns
    assert "../config/**/*" not in patterns
    assert "ui/" not in joined_patterns.replace("\\", "/")
    assert "prune ui" in manifest
    for forbidden in ("*.map", "*.safetensors", "esbuild.exe"):
        assert forbidden in manifest


def test_package_boundary_sources_manifest_excludes_frontend_artifacts() -> None:
    """Generated package metadata must not include dormant UI or local frontend state."""
    lines = [line.strip().replace("\\", "/") for line in SOURCES.read_text(encoding="utf-8").splitlines()]
    forbidden_fragments = (
        "ui/",
        "vetinari/../ui/",
        "node_modules/",
        "vendor-chart",
        "vendor-hljs",
        "vendor-marked",
        "esbuild.exe",
    )
    forbidden_suffixes = (".map", ".safetensors", ".gguf", ".bin", ".onnx", ".pt", ".pth", ".ckpt")

    offenders = [
        line
        for line in lines
        if any(fragment in line for fragment in forbidden_fragments) or line.endswith(forbidden_suffixes)
    ]

    assert offenders == []


def test_mounted_route_inventory_keeps_legacy_frontend_dormant() -> None:
    """The current mounted route table must not include excluded UI entry points."""
    app = create_app(debug=True)
    route_paths = {route.path for route in app.routes}

    assert "/" not in route_paths
    assert "/dashboard" not in route_paths
    assert not any(path.startswith("/static") for path in route_paths)


def test_frontend_public_docs_record_required_release_decisions() -> None:
    """Public docs must name asset status, route proof, and release contract decisions."""
    text = "\n".join(path.read_text(encoding="utf-8") for path in PROOF_NOTES)

    for required in (
        "`ui/templates`",
        "`ui/static/svelte/js/main.js`",
        "legacy/dormant",
        "excluded from Python package artifacts",
        "license",
        "CSP",
        "`node_modules/`",
        "`ui/svelte/models/`",
        "Model weights",
        "Svelte app",
    ):
        assert required in text
