from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows temp-root contract")
def test_tempfile_uses_safe_windows_root() -> None:
    temp_root = Path(tempfile.gettempdir())
    created = Path(tempfile.mkdtemp())

    try:
        assert "vetinari-pytest" in str(temp_root).lower()
        assert created.is_dir()
        assert str(created).lower().startswith(str(temp_root).lower())

        probe = created / "probe.txt"
        probe.write_text("ok", encoding="utf-8")
        assert probe.read_text(encoding="utf-8") == "ok"
        probe.unlink()
        assert not probe.exists()
    finally:
        shutil.rmtree(created, ignore_errors=True)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows temp-root contract")
def test_tmp_path_uses_safe_windows_root(tmp_path: Path) -> None:
    assert "vetinari-pytest" in str(tmp_path).lower()
