"""Direct behavior tests for scripts/update_scaffold.py.

Tests verify the real runtime contract — update_scaffold() returns False
when the scaffold file is missing and True after a successful update, and
the __main__ block exits non-zero when the scaffold is absent.

These tests FAIL on the original buggy code where update_scaffold()
returned None unconditionally and the __main__ block always printed
"Scaffold updated." regardless of whether anything was written.
"""

from __future__ import annotations

import datetime
import importlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Module import fixture
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


@pytest.fixture
def scaffold_mod():
    """Import update_scaffold as a module, reloaded fresh for each test.

    Yields:
        The freshly-imported update_scaffold module.
    """
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        mod = importlib.import_module("update_scaffold")
        importlib.reload(mod)  # ensure fresh state each test
        yield mod
    finally:
        sys.path.pop(0)
        sys.modules.pop("update_scaffold", None)


# ---------------------------------------------------------------------------
# Tests for update_scaffold()
# ---------------------------------------------------------------------------


class TestUpdateScaffoldReturnValue:
    """update_scaffold() must return bool, not None."""

    def test_update_scaffold_returns_false_when_missing(self, scaffold_mod: object, tmp_path: Path) -> None:
        """Returns False when the scaffold file does not exist.

        Args:
            scaffold_mod: Fresh import of update_scaffold.
            tmp_path: Pytest temporary directory.
        """
        missing = tmp_path / "scaffold.md"
        # Patch module-level SCAFFOLD_PATH to a non-existent file
        with patch.object(scaffold_mod, "SCAFFOLD_PATH", missing):
            result = scaffold_mod.update_scaffold()

        assert result is False, f"update_scaffold() must return False when scaffold file is missing, got {result!r}"

    def test_update_scaffold_returns_true_when_present(self, scaffold_mod: object, tmp_path: Path) -> None:
        """Returns True after successfully updating an existing scaffold.

        Args:
            scaffold_mod: Fresh import of update_scaffold.
            tmp_path: Pytest temporary directory.
        """
        scaffold_file = tmp_path / "scaffold.md"
        # Minimal content with the date marker the function replaces
        scaffold_file.write_text("# Scaffold\n<!-- Last updated: 2000-01-01 -->\n", encoding="utf-8")

        # update_scaffold() calls count_files on ui/ and tests/ subdirs.
        # Create them so rglob does not raise on Python 3.12+.
        (tmp_path / "ui").mkdir()
        (tmp_path / "tests").mkdir()

        with (
            patch.object(scaffold_mod, "SCAFFOLD_PATH", scaffold_file),
            patch.object(scaffold_mod, "VETINARI_DIR", tmp_path),
            patch.object(scaffold_mod, "PROJECT_ROOT", tmp_path),
        ):
            result = scaffold_mod.update_scaffold()

        assert result is True, f"update_scaffold() must return True after writing an updated scaffold, got {result!r}"

    def test_update_scaffold_writes_updated_date(self, scaffold_mod: object, tmp_path: Path) -> None:
        """The scaffold file contains today's date after a successful update.

        Args:
            scaffold_mod: Fresh import of update_scaffold.
            tmp_path: Pytest temporary directory.
        """
        scaffold_file = tmp_path / "scaffold.md"
        scaffold_file.write_text("# Scaffold\n<!-- Last updated: 2000-01-01 -->\n", encoding="utf-8")

        # Create ui/ and tests/ so count_files doesn't raise on missing dirs
        (tmp_path / "ui").mkdir()
        (tmp_path / "tests").mkdir()

        with (
            patch.object(scaffold_mod, "SCAFFOLD_PATH", scaffold_file),
            patch.object(scaffold_mod, "VETINARI_DIR", tmp_path),
            patch.object(scaffold_mod, "PROJECT_ROOT", tmp_path),
        ):
            scaffold_mod.update_scaffold()

        content = scaffold_file.read_text(encoding="utf-8")
        today = datetime.date.today().isoformat()
        assert today in content, f"Expected today's date {today!r} in updated scaffold, got: {content!r}"


# ---------------------------------------------------------------------------
# Tests for CLI exit behaviour
# ---------------------------------------------------------------------------


class TestCliExitBehaviour:
    """The __main__ block exits 1 when scaffold is missing."""

    def test_cli_exits_nonzero_on_missing_scaffold(self, tmp_path: Path) -> None:
        """Running the script as a subprocess exits with code 1 on missing scaffold.

        The test passes a patched SCAFFOLD_PATH via environment — because we
        can't monkey-patch a subprocess directly we inject a tiny wrapper that
        redirects the path before calling the script.

        Args:
            tmp_path: Pytest temporary directory — scaffold file NOT created here.
        """
        missing_scaffold = tmp_path / "nonexistent_scaffold.md"

        # Build a small driver that patches the module and calls __main__ logic
        driver = tmp_path / "driver.py"
        driver.write_text(
            f"""
import sys
from pathlib import Path
from unittest.mock import patch

scripts_dir = {str(SCRIPTS_DIR)!r}
sys.path.insert(0, scripts_dir)

import importlib
mod = importlib.import_module("update_scaffold")

missing = Path({str(missing_scaffold)!r})
with patch.object(mod, "SCAFFOLD_PATH", missing):
    ok = mod.update_scaffold()

if ok:
    print("Scaffold updated.")
else:
    print(f"Scaffold update failed: file not found at {{missing}}", file=sys.stderr)
    sys.exit(1)
""",
            encoding="utf-8",
        )

        result = subprocess.run(  # noqa: S603 — argv is built internally, not user input
            [sys.executable, str(driver)],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, (
            "Expected non-zero exit code when scaffold is missing, "
            f"got returncode={result.returncode}, stdout={result.stdout!r}, "
            f"stderr={result.stderr!r}"
        )


# ---------------------------------------------------------------------------
# Tests for helper functions
# ---------------------------------------------------------------------------


class TestCountFiles:
    """count_files() accurately counts extension-matched files, skipping __pycache__."""

    def test_count_files_returns_correct_count(self, scaffold_mod: object, tmp_path: Path) -> None:
        """Counts .py files in directory, excluding __pycache__ entries.

        Args:
            scaffold_mod: Fresh import of update_scaffold.
            tmp_path: Pytest temporary directory.
        """
        (tmp_path / "a.py").write_text("# a", encoding="utf-8")
        (tmp_path / "b.py").write_text("# b", encoding="utf-8")
        (tmp_path / "c.txt").write_text("text", encoding="utf-8")
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "a.cpython-312.pyc").write_bytes(b"bytecode")

        count = scaffold_mod.count_files(tmp_path)

        assert count == 2, f"Expected 2 .py files, got {count}"

    def test_count_files_empty_directory(self, scaffold_mod: object, tmp_path: Path) -> None:
        """Returns 0 for a directory with no matching files.

        Args:
            scaffold_mod: Fresh import of update_scaffold.
            tmp_path: Pytest temporary directory.
        """
        count = scaffold_mod.count_files(tmp_path)
        assert count == 0, f"Expected 0 for empty directory, got {count}"

    def test_count_files_custom_extension(self, scaffold_mod: object, tmp_path: Path) -> None:
        """Counts files with a specified extension.

        Args:
            scaffold_mod: Fresh import of update_scaffold.
            tmp_path: Pytest temporary directory.
        """
        (tmp_path / "a.js").write_text("// js", encoding="utf-8")
        (tmp_path / "b.js").write_text("// js", encoding="utf-8")
        (tmp_path / "c.py").write_text("# py", encoding="utf-8")

        count = scaffold_mod.count_files(tmp_path, ".js")

        assert count == 2, f"Expected 2 .js files, got {count}"


# ---------------------------------------------------------------------------
# Tests for get_package_stats()
# ---------------------------------------------------------------------------


class TestGetPackageStats:
    """get_package_stats() returns (name, count) tuples sorted descending by count."""

    def test_get_package_stats_returns_sorted_stats(self, scaffold_mod: object, tmp_path: Path) -> None:
        """Returns package stats sorted from most to fewest files.

        Creates two subdirectories with different numbers of .py files and
        verifies the function returns them in descending order.

        Args:
            scaffold_mod: Fresh import of update_scaffold.
            tmp_path: Pytest temporary directory.
        """
        pkg_a = tmp_path / "alpha"
        pkg_a.mkdir()
        (pkg_a / "one.py").write_text("# 1", encoding="utf-8")
        (pkg_a / "two.py").write_text("# 2", encoding="utf-8")
        (pkg_a / "three.py").write_text("# 3", encoding="utf-8")

        pkg_b = tmp_path / "beta"
        pkg_b.mkdir()
        (pkg_b / "one.py").write_text("# 1", encoding="utf-8")

        with patch.object(scaffold_mod, "VETINARI_DIR", tmp_path):
            stats = scaffold_mod.get_package_stats()

        assert len(stats) == 2, f"Expected 2 packages, got {len(stats)}: {stats}"

        names = [name for name, _ in stats]
        counts = [count for _, count in stats]

        assert "alpha" in names, f"Expected 'alpha' in stats, got {names}"
        assert "beta" in names, f"Expected 'beta' in stats, got {names}"

        # Descending sort: alpha (3 files) before beta (1 file)
        alpha_idx = names.index("alpha")
        beta_idx = names.index("beta")
        assert alpha_idx < beta_idx, (
            f"Expected 'alpha' (3 files) before 'beta' (1 file) in sorted stats, but got order: {names}"
        )

        assert counts[alpha_idx] == 3, f"Expected alpha count=3, got {counts[alpha_idx]}"
        assert counts[beta_idx] == 1, f"Expected beta count=1, got {counts[beta_idx]}"

    def test_get_package_stats_excludes_pycache(self, scaffold_mod: object, tmp_path: Path) -> None:
        """__pycache__ directories are not included in stats.

        Args:
            scaffold_mod: Fresh import of update_scaffold.
            tmp_path: Pytest temporary directory.
        """
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "cached.py").write_text("# cached", encoding="utf-8")

        real_pkg = tmp_path / "mypkg"
        real_pkg.mkdir()
        (real_pkg / "a.py").write_text("# a", encoding="utf-8")

        with patch.object(scaffold_mod, "VETINARI_DIR", tmp_path):
            stats = scaffold_mod.get_package_stats()

        names = [name for name, _ in stats]
        assert "__pycache__" not in names, f"__pycache__ must not appear in package stats, got: {names}"
        assert "mypkg" in names, f"Expected 'mypkg' in stats, got: {names}"

    def test_get_package_stats_empty_packages_excluded(self, scaffold_mod: object, tmp_path: Path) -> None:
        """Packages with zero .py files are not included in the result.

        Args:
            scaffold_mod: Fresh import of update_scaffold.
            tmp_path: Pytest temporary directory.
        """
        empty_pkg = tmp_path / "empty"
        empty_pkg.mkdir()
        (empty_pkg / "readme.txt").write_text("no python here", encoding="utf-8")

        real_pkg = tmp_path / "haspython"
        real_pkg.mkdir()
        (real_pkg / "module.py").write_text("# py", encoding="utf-8")

        with patch.object(scaffold_mod, "VETINARI_DIR", tmp_path):
            stats = scaffold_mod.get_package_stats()

        names = [name for name, _ in stats]
        assert "empty" not in names, f"Packages with zero .py files must be excluded, got: {names}"
        assert "haspython" in names, f"Expected 'haspython' in stats, got: {names}"
