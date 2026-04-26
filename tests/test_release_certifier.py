"""Fail-closed release certifier coverage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import analysis_router, check_agent_capabilities
from scripts import release_certifier as cert

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


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
                "ui/static/js/app.js.map",
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
        "ui/static/js/main.js.map",
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


def test_analysis_router_cfg_dfg_tools_are_executable_or_labeled() -> None:
    valid_result = cert.certify_analysis_router_tools(analysis_router.merge_tools(["cfg", "dfg"]))
    invalid_result = cert.certify_analysis_router_tools(
        ["scripts/cfg_dfg_analysis.py analyze --category unused-var,shadowed-var"]
    )

    assert valid_result.ok
    assert not invalid_result.ok


def test_agent_capability_audit_rejects_order_instability(monkeypatch) -> None:
    class AlternatingAgent:
        def __init__(self) -> None:
            self.calls = 0

        def get_capabilities(self) -> list[str]:
            self.calls += 1
            if self.calls == 1:
                return ["b", "a"]
            return ["a", "b"]

    monkeypatch.setattr(check_agent_capabilities, "_load", lambda _dotpath: AlternatingAgent())

    result = check_agent_capabilities.audit_agent("WORKER", "fake.Worker", verbose=False)

    assert result["ok"] is False
    assert "non-deterministic order" in str(result["error"])


def test_accelerator_status_detects_empty_sources_missing_outputs_and_digest_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import refresh_ai_accelerators as refresh
    from ai_accelerators import ArtifactSpec

    monkeypatch.setattr(refresh, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(refresh, "SOURCE_SNAPSHOT_PATH", tmp_path / ".ai-codex" / "accelerator-source-snapshot.json")

    source_dir = tmp_path / "src"
    source_dir.mkdir()
    source = source_dir / "input.py"
    source.write_text("print('one')\n", encoding="utf-8")
    output = tmp_path / ".ai-codex" / "out.md"
    spec = ArtifactSpec(
        name="fixture",
        description="fixture",
        outputs=(output,),
        source_roots=(source_dir,),
        source_suffixes=(".py",),
    )

    missing_output = refresh._status_for(spec)
    assert missing_output.stale
    assert "missing outputs" in missing_output.reason

    output.parent.mkdir(parents=True)
    output.write_text("generated\n", encoding="utf-8")
    no_snapshot = refresh._status_for(spec)
    assert no_snapshot.stale
    assert "no recorded refresh snapshot" in no_snapshot.reason

    snapshot = {"fixture": refresh._source_signature(refresh._iter_source_files(spec))}
    refresh._write_source_snapshot(snapshot)
    assert refresh._status_for(spec).stale is False

    source.write_text("print('two')\n", encoding="utf-8")
    changed_source = refresh._status_for(spec)
    assert changed_source.stale
    assert "source input set changed" in changed_source.reason

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    empty_spec = ArtifactSpec(
        name="empty",
        description="empty",
        outputs=(output,),
        source_roots=(empty_dir,),
        source_suffixes=(".py",),
    )
    empty_sources = refresh._status_for(empty_spec)
    assert empty_sources.stale
    assert empty_sources.source_count == 0
    assert "no source files discovered" in empty_sources.reason


def test_release_certifier_cli_fails_bad_evidence_json(tmp_path: Path, capsys) -> None:
    payload = tmp_path / "evidence.json"
    payload.write_text(json.dumps({"status": "clean", "tools_run": 0}), encoding="utf-8")

    exit_code = cert.main(["evidence-json", str(payload)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "zero-tool project scan" in captured.err
