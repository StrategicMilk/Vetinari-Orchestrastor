"""Fail-closed release certifier coverage."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.release import release_certifier as cert

ROOT = Path(__file__).resolve().parents[1]


def _write_package_fixture(root: Path) -> None:
    (root / "vetinari.egg-info").mkdir()
    (root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "vetinari"',
                'description = "Correct description"',
                'classifiers = ["Programming Language :: Python :: 3.12"]',
                'dependencies = ["requests>=2.28.0"]',
            ]
        ),
        encoding="utf-8",
    )
    (root / "setup.py").write_text(
        "\n".join(
            [
                "from setuptools import setup",
                "",
                'setup(name="wrong", install_requires=["flask"], '
                'classifiers=["Programming Language :: Python :: 3.14"])',
            ]
        ),
        encoding="utf-8",
    )
    (root / "vetinari.egg-info" / "PKG-INFO").write_text(
        "\n".join(
            [
                "Metadata-Version: 2.1",
                "Name: wrong",
                "Summary: Stale description",
                "Classifier: Programming Language :: Python :: 3.14",
                "Requires-Dist: flask>=2",
            ]
        ),
        encoding="utf-8",
    )
    (root / "vetinari.egg-info" / "SOURCES.txt").write_text(
        "\n".join(
            [
                "pyproject.toml",
                "missing.py",
                "../outside.py",
                "ui/src/App.svelte",
                "ui/legacy/static/js/app.js.map",
                "ui/node_modules/pkg/index.js",
                "models/native/model.safetensors",
                "vetinari/native/extension.node",
            ]
        ),
        encoding="utf-8",
    )


def test_release_certifier_detects_package_metadata_parity_failures(tmp_path: Path) -> None:
    _write_package_fixture(tmp_path)

    result = cert.certify_package_metadata(tmp_path)

    checks = {failure.check for failure in result.failures}
    messages = "\n".join(failure.message for failure in result.failures)
    assert "setup-py" in checks
    assert "metadata-parity" in checks
    assert "release-manifest" in checks
    assert "setup.py must delegate" in messages
    assert "PKG-INFO Name" in messages
    assert "missing file" in messages
    assert "frontend workspace" in messages
    assert "forbidden generated/binary/model artifact" in messages


def test_release_manifest_rejects_workspace_and_binary_sweeps() -> None:
    entries = [
        "ui/node_modules/vite/index.js",
        "ui/src/App.svelte",
        "ui/legacy/static/js/main.js.map",
        "vetinari/native/addon.pyd",
        "models/native/shard-00001.safetensors",
        ".ai-codex/wiki.md",
    ]

    result = cert.certify_release_manifest(entries)

    assert not result.ok
    assert len(result.failures) >= len(entries)


def test_release_evidence_rejects_false_green_classes() -> None:
    payloads = [
        {"status": "clean", "tools_run": 0},
        {"status": "passed", "coverage": {"files": {}}},
        {"status": "passed", "tests": {"total": 5, "skipped": 5}},
        {"status": "passed", "continue_on_error": True},
        {"status": "passed", "xfail_count": 1},
    ]

    for payload in payloads:
        result = cert.certify_release_evidence(payload)
        assert not result.ok, payload


def test_model_provenance_rejects_mutable_revision_and_display_only_hash() -> None:
    source = """
def download(hf_hub_download, dest, filename):
    filename = "model.gguf"
    if dest.exists():
        return 0
    local = hf_hub_download(repo_id="owner/model", filename=filename, revision="main")
    digest = "abc123"
    print(f"SHA-256: {digest}")
    model_id = Path(filename).stem
    return local, model_id
"""

    result = cert.certify_model_provenance_source(source, path="fixture.py")

    messages = "\n".join(failure.message for failure in result.failures)
    assert "mutable model/download revision" in messages
    assert "display-only hash" in messages
    assert "existing-file trust" in messages
    assert "stem-only GGUF identity" in messages


def test_plan_index_requires_child_shards_and_rejects_coordination_parent() -> None:
    text = "Release proof routes to SESSION-34D.md for accelerator certifier evidence."

    result = cert.certify_plan_index(text, required_children=["SESSION-34D5.md"])

    messages = "\n".join(failure.message for failure in result.failures)
    assert "omits child shard: SESSION-34D5.md" in messages
    assert "coordination-only parent: SESSION-34D.md" in messages


def test_release_certifier_cli_fails_bad_evidence_json(tmp_path: Path, capsys) -> None:
    payload = tmp_path / "evidence.json"
    payload.write_text(json.dumps({"status": "clean", "tools_run": 0}), encoding="utf-8")

    exit_code = cert.main(["evidence-json", str(payload)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "zero-tool project scan" in captured.err
