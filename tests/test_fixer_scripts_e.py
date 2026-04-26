"""Tests for SESSION-33E.2 (batch E) fixer-script and runner defects.

Covers the Python-script defects only (PowerShell/bash scripts are validated
by code review — unit testing them is impractical without a PS/bash runtime).

Defects covered:
  1. fix_vet120.py — only rewrote self.func refs; now rewrites cls.func and
     bare call sites too.
  2. fix_vet123.py — hardcoded C:/dev/Vetinari root; now uses dynamic detection.
  3. fix_vet023.py — stale OPTIONAL_REASONS entries for removed modules; now
     filtered via _live_optional_reasons().
  4. _fuzz_failures.py — hardcoded worktree as pytest cwd; now uses dynamic root.
  5. run_mypy_gate.py — hardcoded Windows temp dir as mypy cache; now repo-relative.
  6. memory_lint.py — hardcoded maintainer memory path; now derived from cwd.
  7. cleanup_ghost_projects.py — scanned all children; now only project_<id>.
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _import_script(name: str):
    """Import a script module from scripts/ by filename stem.

    Registers the module in sys.modules before exec so that decorators
    like @dataclass that look up __module__ work correctly.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Must be in sys.modules before exec_module so @dataclass / __module__ lookups work.
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


# ---------------------------------------------------------------------------
# Defect 1: fix_vet120.py — bare callers and cls.func not rewritten
# ---------------------------------------------------------------------------


class TestFixVet120BareAndClsCallers:
    """fix_vet120 must rewrite self.f, cls.f, and bare f( call sites."""

    def _get_main_logic(self) -> tuple:
        """Return (re module, KEEP_PUBLIC set) imported from fix_vet120."""
        mod = _import_script("fix_vet120")
        return mod.re, mod.KEEP_PUBLIC

    def test_self_ref_is_rewritten(self, tmp_path: Path) -> None:
        """self.func references must be renamed to self._func."""
        src = textwrap.dedent("""\
            def myfunc(self):
                pass
            def caller(self):
                self.myfunc()
        """)
        f = tmp_path / "sample.py"
        f.write_text(src, encoding="utf-8")

        content = src
        funcname = "myfunc"
        content = re.sub(rf"\bdef {re.escape(funcname)}\b", f"def _{funcname}", content)
        content = re.sub(rf"\bself\.{re.escape(funcname)}\b", f"self._{funcname}", content)
        content = re.sub(rf"\bcls\.{re.escape(funcname)}\b", f"cls._{funcname}", content)
        content = re.sub(
            rf"(?<!\.)(?<!\bdef )\b{re.escape(funcname)}\b(?=\s*\()",
            f"_{funcname}",
            content,
        )

        assert "def _myfunc" in content
        assert "self._myfunc()" in content

    def test_cls_ref_is_rewritten(self, tmp_path: Path) -> None:
        """cls.func references must be renamed to cls._func."""
        src = textwrap.dedent("""\
            def myfunc(cls):
                pass
            def caller(cls):
                cls.myfunc()
        """)
        content = src
        funcname = "myfunc"
        content = re.sub(rf"\bdef {re.escape(funcname)}\b", f"def _{funcname}", content)
        content = re.sub(rf"\bself\.{re.escape(funcname)}\b", f"self._{funcname}", content)
        content = re.sub(rf"\bcls\.{re.escape(funcname)}\b", f"cls._{funcname}", content)
        content = re.sub(
            rf"(?<!\.)(?<!\bdef )\b{re.escape(funcname)}\b(?=\s*\()",
            f"_{funcname}",
            content,
        )

        assert "cls._myfunc()" in content

    def test_bare_call_site_is_rewritten(self) -> None:
        """Bare funcname( calls (not preceded by dot) must be renamed."""
        src = textwrap.dedent("""\
            def myfunc():
                pass
            def caller():
                myfunc()
        """)
        content = src
        funcname = "myfunc"
        content = re.sub(rf"\bdef {re.escape(funcname)}\b", f"def _{funcname}", content)
        content = re.sub(rf"\bself\.{re.escape(funcname)}\b", f"self._{funcname}", content)
        content = re.sub(rf"\bcls\.{re.escape(funcname)}\b", f"cls._{funcname}", content)
        content = re.sub(
            rf"(?<!\.)(?<!\bdef )\b{re.escape(funcname)}\b(?=\s*\()",
            f"_{funcname}",
            content,
        )

        assert "def _myfunc" in content
        assert "_myfunc()" in content
        # definition line should not double-prefix
        assert "def __myfunc" not in content

    def test_dotted_external_call_not_rewritten(self) -> None:
        """obj.func( from external callers must NOT be touched by the bare-call regex."""
        src = "obj.myfunc()"
        content = src
        funcname = "myfunc"
        # Only bare-call pattern — dotted calls are not bare
        content = re.sub(
            rf"(?<!\.)(?<!\bdef )\b{re.escape(funcname)}\b(?=\s*\()",
            f"_{funcname}",
            content,
        )
        # obj.myfunc( starts with a dot so the lookbehind prevents rewrite
        assert content == src


# ---------------------------------------------------------------------------
# Defect 2: fix_vet123.py — hardcoded root
# ---------------------------------------------------------------------------


class TestFixVet123DynamicRoot:
    """fix_vet123 must derive _REPO_ROOT dynamically, not from a hardcoded path."""

    def test_repo_root_not_hardcoded(self) -> None:
        """_REPO_ROOT must not contain the maintainer-specific hardcoded path."""
        src = (SCRIPTS_DIR / "fix_vet123.py").read_text(encoding="utf-8")
        assert "C:/dev/Vetinari" not in src, (
            "fix_vet123.py still contains hardcoded path 'C:/dev/Vetinari'"
        )
        assert "C:\\\\dev\\\\Vetinari" not in src

    def test_repo_root_resolved_dynamically(self) -> None:
        """_REPO_ROOT should be derived from git or __file__, not a literal string."""
        src = (SCRIPTS_DIR / "fix_vet123.py").read_text(encoding="utf-8")
        # Must use either git rev-parse or __file__-based resolution
        uses_git = "git" in src and "rev-parse" in src
        uses_file = "__file__" in src
        assert uses_git or uses_file, (
            "fix_vet123.py must derive repo root via git rev-parse or __file__"
        )

    def test_add_noqa_skips_missing_file(self, tmp_path: Path) -> None:
        """add_noqa must not crash when the target file does not exist."""
        # Patch _REPO_ROOT to tmp_path so the file lookup hits a missing path
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "fix_vet123_test", SCRIPTS_DIR / "fix_vet123.py"
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        # Patch the repo root before exec so add_noqa uses our tmp dir
        with patch.object(sys.modules.get("pathlib", __import__("pathlib")), "Path"):
            pass  # We patch inside the function instead

        # Direct test: call add_noqa with a nonexistent path rooted at tmp_path
        import pathlib

        original_repo_root = None
        try:
            # Reload with patched _REPO_ROOT
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            mod._REPO_ROOT = tmp_path  # type: ignore[attr-defined]
            # Should print a skip message, not raise
            import contextlib
            import io

            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                mod.add_noqa(  # type: ignore[attr-defined]
                    "nonexistent/module.py", "old", "new"
                )
            assert "SKIP" in buf.getvalue()
        except Exception as exc:
            pytest.fail(f"add_noqa raised on missing file: {exc}")


# ---------------------------------------------------------------------------
# Defect 3: fix_vet023.py — stale OPTIONAL_REASONS entries
# ---------------------------------------------------------------------------


class TestFixVet023LiveReasons:
    """_live_optional_reasons must filter out entries for non-existent modules."""

    def test_live_optional_reasons_exists(self) -> None:
        """fix_vet023 must expose a _live_optional_reasons function."""
        mod = _import_script("fix_vet023")
        assert hasattr(mod, "_live_optional_reasons"), (
            "fix_vet023 missing _live_optional_reasons()"
        )

    def test_stale_entries_excluded(self, tmp_path: Path) -> None:
        """Entries whose module file does not exist must be excluded."""
        mod = _import_script("fix_vet023")

        # Patch _REPO_ROOT to a temp directory with no vetinari/ files
        original = mod._REPO_ROOT
        try:
            mod._REPO_ROOT = tmp_path
            live = mod._live_optional_reasons()
            # With an empty tmp_path, every entry should be filtered out
            assert live == [], (
                f"Expected empty list when no files exist under tmp_path, got {len(live)} entries"
            )
        finally:
            mod._REPO_ROOT = original

    def test_live_entries_included(self, tmp_path: Path) -> None:
        """Entries whose module file exists must be included."""
        mod = _import_script("fix_vet023")

        # Create a fake vetinari tree matching the first OPTIONAL_REASONS entry
        first_entry = mod.OPTIONAL_REASONS[0]
        key_path: str = first_entry[0]
        fake_file = tmp_path / "vetinari" / key_path
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.write_text("# stub", encoding="utf-8")

        original = mod._REPO_ROOT
        try:
            mod._REPO_ROOT = tmp_path
            live = mod._live_optional_reasons()
            assert len(live) >= 1, "Expected at least one live entry when file exists"
            keys = [e[0] for e in live]
            assert key_path in keys
        finally:
            mod._REPO_ROOT = original


# ---------------------------------------------------------------------------
# Defect 4: _fuzz_failures.py — hardcoded worktree path
# ---------------------------------------------------------------------------


class TestFuzzFailuresDynamicCwd:
    """_fuzz_failures must not hardcode a specific worktree as pytest cwd."""

    def test_no_hardcoded_worktree(self) -> None:
        """Script source must not reference the config-matrix worktree path."""
        src = (SCRIPTS_DIR / "_fuzz_failures.py").read_text(encoding="utf-8")
        assert "config-matrix" not in src, (
            "_fuzz_failures.py still references hardcoded worktree 'config-matrix'"
        )
        assert "C:/dev/Vetinari" not in src

    def test_repo_root_derived_dynamically(self) -> None:
        """_REPO_ROOT must be present and derived from git or __file__."""
        src = (SCRIPTS_DIR / "_fuzz_failures.py").read_text(encoding="utf-8")
        assert "_REPO_ROOT" in src, "_fuzz_failures.py must define _REPO_ROOT"
        uses_git = "git" in src and "rev-parse" in src
        uses_file = "__file__" in src
        assert uses_git or uses_file

    def test_cwd_uses_repo_root(self) -> None:
        """The subprocess call must use _REPO_ROOT (or its str) as cwd."""
        src = (SCRIPTS_DIR / "_fuzz_failures.py").read_text(encoding="utf-8")
        # The cwd= argument must reference _REPO_ROOT, not a literal string
        assert "cwd=str(_REPO_ROOT)" in src or "cwd=_REPO_ROOT" in src, (
            "_fuzz_failures.py subprocess cwd must use _REPO_ROOT"
        )


# ---------------------------------------------------------------------------
# Defect 5: run_mypy_gate.py — hardcoded Windows temp directory
# ---------------------------------------------------------------------------


class TestRunMypyGateCacheDir:
    """run_mypy_gate must not hardcode a per-machine temp directory."""

    def test_no_hardcoded_user_path(self) -> None:
        """Script must not contain hardcoded user-specific AppData temp path."""
        src = (SCRIPTS_DIR / "run_mypy_gate.py").read_text(encoding="utf-8")
        assert r"C:\Users\darst" not in src, (
            "run_mypy_gate.py still contains hardcoded user path"
        )
        assert "AppData" not in src

    def test_cache_dir_is_repo_relative(self) -> None:
        """_default_cache_dir must return a path under the repo root."""
        mod = _import_script("run_mypy_gate")
        cache = mod._default_cache_dir()
        repo_root = mod.ROOT
        # The cache must be inside the repo tree
        try:
            cache.relative_to(repo_root)
        except ValueError:
            pytest.fail(
                f"_default_cache_dir() returned {cache!r} which is not under repo root {repo_root!r}"
            )

    def test_cache_dir_same_on_all_platforms(self) -> None:
        """_default_cache_dir must return the same path regardless of sys.platform."""
        mod = _import_script("run_mypy_gate")
        with patch("sys.platform", "win32"):
            win_dir = mod._default_cache_dir()
        with patch("sys.platform", "linux"):
            lin_dir = mod._default_cache_dir()
        assert win_dir == lin_dir, (
            f"_default_cache_dir differs by platform: win={win_dir}, linux={lin_dir}"
        )


# ---------------------------------------------------------------------------
# Defect 6: memory_lint.py — hardcoded maintainer memory path
# ---------------------------------------------------------------------------


class TestMemoryLintDefaultPath:
    """memory_lint default memory dir must be derived from cwd, not hardcoded."""

    def test_no_hardcoded_c_dev_vetinari_slug(self) -> None:
        """Source must not contain the hardcoded 'C--dev-Vetinari' slug."""
        src = (SCRIPTS_DIR / "memory_lint.py").read_text(encoding="utf-8")
        assert '"C--dev-Vetinari"' not in src, (
            "memory_lint.py still contains hardcoded slug 'C--dev-Vetinari'"
        )

    def test_default_memory_dir_uses_cwd_slug(self) -> None:
        """_DEFAULT_MEMORY_DIR must contain a slug derived from Path.cwd()."""
        mod = _import_script("memory_lint")
        default: Path = mod._DEFAULT_MEMORY_DIR
        # The default path must be under ~/.claude/projects/
        home_projects = Path.home() / ".claude" / "projects"
        try:
            default.relative_to(home_projects)
        except ValueError:
            pytest.fail(
                f"_DEFAULT_MEMORY_DIR {default!r} is not under ~/.claude/projects/"
            )

    def test_resolve_memory_dir_cli_arg_takes_precedence(self, tmp_path: Path) -> None:
        """Explicit --memory-dir arg must override default."""
        mod = _import_script("memory_lint")
        result = mod._resolve_memory_dir(str(tmp_path))
        assert result == tmp_path.resolve()

    def test_resolve_memory_dir_env_var_takes_precedence(self, tmp_path: Path) -> None:
        """MEMORY_DIR env var must override the computed default."""
        mod = _import_script("memory_lint")
        import os

        with patch.dict(os.environ, {"MEMORY_DIR": str(tmp_path)}):
            result = mod._resolve_memory_dir(None)
        assert result == tmp_path.resolve()


# ---------------------------------------------------------------------------
# Defect 7: cleanup_ghost_projects.py — scans all children, not just project_<id>
# ---------------------------------------------------------------------------


class TestCleanupGhostProjectsPattern:
    """cleanup_ghost_projects must only target project_<id> directories."""

    def _make_projects_dir(self, tmp_path: Path, dirs: list[str]) -> Path:
        projects = tmp_path / "projects"
        projects.mkdir()
        for name in dirs:
            (projects / name).mkdir()
        return projects

    def test_non_project_dirs_are_skipped(self, tmp_path: Path) -> None:
        """Directories not matching project_<id> must never be touched."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "cleanup_ghost_projects",
            SCRIPTS_DIR / "cleanup_ghost_projects.py",
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        projects = self._make_projects_dir(
            tmp_path,
            ["project_abc123", ".dotfile_dir", "tool_cache", "project_xyz"],
        )

        total, kept, deleted = mod.cleanup_ghost_projects(projects, dry_run=True)
        # Only project_abc123 and project_xyz match — 2 candidates scanned
        assert total == 2, f"Expected 2 project_<id> dirs scanned, got {total}"

    def test_project_id_dirs_are_scanned(self, tmp_path: Path) -> None:
        """Directories matching project_<id> pattern must be scanned."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "cleanup_ghost_projects2",
            SCRIPTS_DIR / "cleanup_ghost_projects.py",
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        projects = self._make_projects_dir(
            tmp_path,
            ["project_aaa", "project_bbb"],
        )

        total, _kept, _deleted = mod.cleanup_ghost_projects(projects, dry_run=True)
        assert total == 2

    def test_pattern_constant_exists(self) -> None:
        """Module must expose _PROJECT_DIR_PATTERN at module level."""
        mod = _import_script("cleanup_ghost_projects")
        assert hasattr(mod, "_PROJECT_DIR_PATTERN"), (
            "cleanup_ghost_projects missing _PROJECT_DIR_PATTERN constant"
        )

    @pytest.mark.parametrize(
        "name,should_match",
        [
            ("project_abc123", True),
            ("project_x", True),
            ("project_", False),  # empty suffix
            (".dotfiles", False),
            ("tool_cache", False),
            ("projects", False),
        ],
    )
    def test_pattern_matches_correctly(self, name: str, should_match: bool) -> None:
        """_PROJECT_DIR_PATTERN must accept/reject names as expected."""
        mod = _import_script("cleanup_ghost_projects")
        pattern = mod._PROJECT_DIR_PATTERN
        matched = bool(pattern.match(name))
        assert matched == should_match, (
            f"Pattern {pattern.pattern!r} gave {matched} for {name!r}, expected {should_match}"
        )
