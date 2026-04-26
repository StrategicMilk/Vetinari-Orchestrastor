"""Regression checks for the supported startup and pytest entrypoints."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_windows_startup_uses_canonical_environment() -> None:
    """Windows startup must target .venv312 instead of the stale venv/ path."""
    content = (PROJECT_ROOT / "start.bat").read_text(encoding="utf-8")

    assert r'\.venv312"' in content
    assert r"venv\Scripts\activate.bat" not in content
    assert 'set "VETINARI_PYTHON=' in content
    assert '"%VETINARI_PYTHON%" -m vetinari' in content
    assert "sys.executable" in content
    assert "vetinari.__version__" in content


def test_posix_startup_uses_canonical_environment() -> None:
    """POSIX startup must target .venv312 instead of the stale venv/ path."""
    content = (PROJECT_ROOT / "start.sh").read_text(encoding="utf-8")

    assert '.venv312"' in content
    assert "venv/bin/activate" not in content
    assert 'PYTHON_EXE="$VENV_DIR/bin/python"' in content
    assert '"$PYTHON_EXE" -m vetinari' in content
    assert "sys.executable" in content
    assert "vetinari.__version__" in content
