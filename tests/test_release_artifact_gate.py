"""Fail-closed tests for built release artifact hygiene checks."""

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "release" / "check_release_artifacts.py"


def _load_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_release_artifacts", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_wheel(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, mode="w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)


def _write_sdist(path: Path, entries: dict[str, bytes]) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for name, payload in entries.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_clean_artifacts_pass_hygiene_gate(tmp_path: Path, capsys) -> None:
    checker = _load_checker()
    dist = tmp_path / "dist"
    dist.mkdir()
    _write_wheel(
        dist / "vetinari-0.1.0-py3-none-any.whl",
        {
            "vetinari/__init__.py": b"__version__ = '0.1.0'\n",
            "vetinari/config/runtime/models.yaml": b"models: []\n",
        },
    )
    _write_sdist(
        dist / "vetinari-0.1.0.tar.gz",
        {
            "vetinari-0.1.0/pyproject.toml": b"[project]\nname = 'vetinari'\n",
            "vetinari-0.1.0/vetinari.egg-info/PKG-INFO": b"Name: vetinari\n",
            "vetinari-0.1.0/vetinari/__init__.py": b"__version__ = '0.1.0'\n",
        },
    )

    exit_code = checker.main(["--dist-dir", str(dist)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Release artifact hygiene check passed." in captured.out
    assert "vetinari-0.1.0-py3-none-any.whl" in captured.out
    assert "vetinari-0.1.0.tar.gz" in captured.out


def test_internal_trees_fail_with_actionable_samples(tmp_path: Path, capsys) -> None:
    checker = _load_checker()
    dist = tmp_path / "dist"
    dist.mkdir()
    _write_wheel(
        dist / "vetinari-0.1.0-py3-none-any.whl",
        {"vetinari/__init__.py": b"__version__ = '0.1.0'\n"},
    )
    _write_sdist(
        dist / "vetinari-0.1.0.tar.gz",
        {
            "vetinari-0.1.0/vetinari/__init__.py": b"__version__ = '0.1.0'\n",
            "vetinari-0.1.0/.ai-codex/wiki.md": b"internal\n",
            "vetinari-0.1.0/.claude/settings.json": b"{}\n",
        },
    )

    exit_code = checker.main(["--dist-dir", str(dist)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "forbidden workspace path in release manifest" in captured.err
    assert ".ai-codex/wiki.md" in captured.err
    assert ".claude/settings.json" in captured.err


def test_oversized_artifacts_fail_with_byte_limit_in_output(tmp_path: Path, capsys) -> None:
    checker = _load_checker()
    dist = tmp_path / "dist"
    dist.mkdir()
    _write_wheel(
        dist / "vetinari-0.1.0-py3-none-any.whl",
        {
            "vetinari/__init__.py": b"__version__ = '0.1.0'\n",
            "vetinari/large.bin": b"x" * 512,
        },
    )
    _write_sdist(
        dist / "vetinari-0.1.0.tar.gz",
        {"vetinari-0.1.0/vetinari/__init__.py": b"__version__ = '0.1.0'\n"},
    )

    exit_code = checker.main(["--dist-dir", str(dist), "--max-wheel-bytes", "128"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "artifact size" in captured.err
    assert "exceeds wheel limit 128 bytes" in captured.err
    assert "vetinari-0.1.0-py3-none-any.whl" in captured.err
