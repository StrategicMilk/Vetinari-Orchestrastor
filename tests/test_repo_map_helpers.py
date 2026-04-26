"""Regression tests for repo-map and impact-analysis helper fixes.

Each test class targets one of the eight defects fixed in SESSION-33E.1:

  Defect 1  — get_ast_indexer: per-root dict-keyed singleton (no cross-root pollution)
  Defect 2  — RepoMap.generate: focus_paths ordering preserved, not alphabetical
  Defect 3  — generate_architecture._parse_imports: relative imports resolved
  Defect 4  — impact_analysis._parse_imports: relative imports resolved
  Defect 5  — impact_analysis._parse_imports: ``from . import foo`` (module=None) handled
  Defect 6  — build_graph / build_forward_graph: package ids from __init__.py in ``known``
  Defect 7  — codebase_graph.cmd_scan: returns skip count; main exits non-zero on skips
  Defect 8  — ASTIndexer.find_usages: detects live code-body references via name_refs

NOTE on current_module convention:
  Both generate_architecture._parse_imports and impact_analysis._parse_imports
  receive current_module as the SHORT dotted id relative to the vetinari/ package,
  e.g. "agents.base" (NOT "vetinari.agents.base").  Tests use this convention.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to load script modules by path
# ---------------------------------------------------------------------------


def _load_script(name: str, script_path: str):
    """Load a script file as a module by absolute path."""
    spec = importlib.util.spec_from_file_location(name, script_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Defect 1: get_ast_indexer returns isolated per-root instances
# ---------------------------------------------------------------------------


class TestGetAstIndexerPerRoot:
    """get_ast_indexer must return distinct ASTIndexer instances for distinct roots."""

    def test_different_roots_return_different_instances(self, tmp_path: Path) -> None:
        """Two distinct root paths must each get their own ASTIndexer."""
        root_a = tmp_path / "project_a"
        root_b = tmp_path / "project_b"
        root_a.mkdir()
        root_b.mkdir()

        import vetinari.repo_map as rm

        with patch.object(rm, "_indexer_cache", {}):
            indexer_a = rm.get_ast_indexer(str(root_a))
            indexer_b = rm.get_ast_indexer(str(root_b))

        assert indexer_a is not indexer_b, (
            "Different roots must produce separate ASTIndexer instances — "
            "the old single-global implementation returned the same object."
        )

    def test_same_root_returns_same_instance(self, tmp_path: Path) -> None:
        """The same resolved root path must be cached and return the same instance."""
        root = tmp_path / "project"
        root.mkdir()

        import vetinari.repo_map as rm

        with patch.object(rm, "_indexer_cache", {}):
            first = rm.get_ast_indexer(str(root))
            second = rm.get_ast_indexer(str(root))

        assert first is second, "Same root must return the cached ASTIndexer."

    def test_symbol_tables_do_not_bleed_across_roots(self, tmp_path: Path) -> None:
        """Symbols indexed from root_a must not appear when querying root_b."""
        root_a = tmp_path / "root_a"
        root_b = tmp_path / "root_b"
        root_a.mkdir()
        root_b.mkdir()

        (root_a / "alpha.py").write_text(
            "def only_in_alpha(): pass\n", encoding="utf-8"
        )

        import vetinari.repo_map as rm

        with patch.object(rm, "_indexer_cache", {}):
            idx_a = rm.get_ast_indexer(str(root_a))
            idx_a.index_project()

            idx_b = rm.get_ast_indexer(str(root_b))
            idx_b.index_project()

        assert idx_a.find_symbol("only_in_alpha"), (
            "Symbol must be found in root_a's indexer."
        )
        assert not idx_b.find_symbol("only_in_alpha"), (
            "Symbol from root_a must NOT appear in root_b's isolated indexer."
        )


# ---------------------------------------------------------------------------
# Defect 2: RepoMap.generate preserves focus_paths order
# ---------------------------------------------------------------------------


class TestRepoMapFocusPathsOrder:
    """generate() must output focus_paths-ordered modules, not alphabetical."""

    def test_focus_paths_order_preserved(self, tmp_path: Path) -> None:
        """Modules listed in focus_paths must appear in that order in the output."""
        for name in ("aaa.py", "bbb.py", "zzz.py"):
            (tmp_path / name).write_text(
                f'"""Module {name}."""\ndef func_{name[:-3]}(): pass\n',
                encoding="utf-8",
            )

        from vetinari.repo_map import RepoMap

        rm = RepoMap()

        # Request zzz first, then aaa — opposite of alphabetical.
        output = rm.generate(
            str(tmp_path),
            focus_paths=["zzz.py", "aaa.py"],
            include_private=False,
        )

        pos_zzz = output.find("zzz")
        pos_aaa = output.find("aaa")

        assert pos_zzz != -1, "zzz.py must appear in output"
        assert pos_aaa != -1, "aaa.py must appear in output"
        assert pos_zzz < pos_aaa, (
            "zzz must appear before aaa when focus_paths=[zzz, aaa] — "
            "old code always sorted alphabetically."
        )


# ---------------------------------------------------------------------------
# Defect 3: generate_architecture._parse_imports resolves relative imports
# ---------------------------------------------------------------------------


class TestGenerateArchitectureRelativeImports:
    """_parse_imports in generate_architecture.py must resolve relative imports.

    current_module is the short dotted id relative to the package root,
    e.g. "agents.base" (without the leading "vetinari." prefix).
    """

    @pytest.fixture
    def parse_imports(self):
        mod = _load_script(
            "generate_architecture",
            "C:/dev/Vetinari/scripts/inspect/generate_architecture.py",
        )
        return mod._parse_imports

    def test_single_dot_import_resolved_to_package(self, parse_imports) -> None:
        """``from . import sibling`` in agents/base.py resolves anchor to 'agents'."""
        # current_module="agents.base" → level=1 strips 1 component → anchor="agents"
        source = "from . import sibling\n"
        result = parse_imports(source, "vetinari", current_module="agents.base")
        assert "agents" in result, (
            "Single-dot import from agents.base must resolve anchor to 'agents'."
        )

    def test_double_dot_import_resolved_to_root(self, parse_imports) -> None:
        """``from .. import top`` in agents/base.py resolves anchor to '' (root)."""
        # current_module="agents.base" → level=2 strips 2 components → anchor=[]
        # With no module component, resolved="" which is filtered out.
        # This is expected — the root package itself has no sub-id.
        source = "from .. import top\n"
        result = parse_imports(source, "vetinari", current_module="agents.base")
        # Result may be empty (root pkg filtered) — must not raise.
        assert isinstance(result, list), "Must return a list without raising."

    def test_relative_import_with_module_resolved(self, parse_imports) -> None:
        """``from .utils import helper`` in agents/base.py resolves to 'agents.utils'."""
        source = "from .utils import helper\n"
        result = parse_imports(source, "vetinari", current_module="agents.base")
        assert "agents.utils" in result, (
            "from .utils in agents.base must resolve to 'agents.utils'."
        )

    def test_absolute_import_still_works(self, parse_imports) -> None:
        """Absolute intra-package import must still be captured correctly."""
        source = "from vetinari.types import AgentType\n"
        result = parse_imports(source, "vetinari", current_module="agents.base")
        assert "types" in result, (
            "Absolute from vetinari.types must produce 'types'."
        )


# ---------------------------------------------------------------------------
# Defect 4: impact_analysis._parse_imports resolves relative imports
# ---------------------------------------------------------------------------


class TestImpactAnalysisRelativeImports:
    """_parse_imports in impact_analysis.py must resolve relative imports."""

    @pytest.fixture
    def parse_imports(self):
        mod = _load_script(
            "impact_analysis",
            "C:/dev/Vetinari/scripts/inspect/impact_analysis.py",
        )
        return mod._parse_imports

    def test_relative_with_module(self, parse_imports) -> None:
        """``from .types import AgentType`` in agents/base.py resolves to 'agents.types'."""
        source = "from .types import AgentType\n"
        result = parse_imports(source, "vetinari", current_module="agents.base")
        assert "agents.types" in result, (
            "from .types in agents.base must resolve to 'agents.types'."
        )

    def test_two_level_relative(self, parse_imports) -> None:
        """``from ..core import X`` in agents/sub/worker.py resolves to 'agents.core'."""
        source = "from ..core import X\n"
        result = parse_imports(source, "vetinari", current_module="agents.sub.worker")
        assert "agents.core" in result, (
            "from ..core in agents.sub.worker must resolve to 'agents.core'."
        )


# ---------------------------------------------------------------------------
# Defect 5: impact_analysis._parse_imports handles ``from . import foo`` (module=None)
# ---------------------------------------------------------------------------


class TestImpactAnalysisNoneModule:
    """``from . import foo`` has node.module=None — must not crash or silently drop."""

    @pytest.fixture
    def parse_imports(self):
        mod = _load_script(
            "impact_analysis_d5",
            "C:/dev/Vetinari/scripts/inspect/impact_analysis.py",
        )
        return mod._parse_imports

    def test_from_dot_import_no_crash(self, parse_imports) -> None:
        """``from . import foo`` must not raise and must return the package anchor."""
        source = "from . import foo\n"
        # Must not raise
        result = parse_imports(source, "vetinari", current_module="agents.base")
        assert isinstance(result, list), "Must return a list"
        # level=1 from "agents.base" → anchor = "agents"
        assert "agents" in result, (
            "from . import foo in agents.base must resolve anchor to 'agents'."
        )

    def test_from_dot_import_at_top_level_no_crash(self, parse_imports) -> None:
        """``from . import foo`` in a top-level module must not crash."""
        source = "from . import foo\n"
        # current_module has only 1 part → stripping 1 → empty anchor → filtered out
        result = parse_imports(source, "vetinari", current_module="core")
        assert isinstance(result, list), "Must return a list even when anchor resolves empty."


# ---------------------------------------------------------------------------
# Defect 6: build_forward_graph includes package ids from __init__.py in ``known``
# ---------------------------------------------------------------------------


class TestPackageIdsInKnown:
    """Package-level ids (e.g. 'agents') must appear in ``known`` so relative imports
    targeting a package node are not silently dropped from the dependency graph."""

    @pytest.fixture
    def impact_mod(self):
        return _load_script(
            "impact_analysis_d6",
            "C:/dev/Vetinari/scripts/inspect/impact_analysis.py",
        )

    def test_package_dep_is_included_in_graph(self, tmp_path: Path, impact_mod) -> None:
        """A module that imports a sibling package should have that package in its deps.

        Layout:
            tmp_path/vetinari/__init__.py       (root package)
            tmp_path/vetinari/agents/__init__.py (sub-package)
            tmp_path/vetinari/core.py            (imports from .agents)
        """
        pkg = tmp_path / "vetinari"
        agents = pkg / "agents"
        agents.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (agents / "__init__.py").write_text("", encoding="utf-8")
        # core.py uses a relative import targeting the agents package
        (pkg / "core.py").write_text(
            "from .agents import something\n", encoding="utf-8"
        )

        # build_forward_graph receives the vetinari/ directory directly
        graph = impact_mod.build_forward_graph(pkg)

        core_deps = graph.get("core", set())
        assert "agents" in core_deps, (
            "core.py imports from .agents — 'agents' package id must appear in deps. "
            "Old code excluded __init__.py ids from known, so the edge was silently dropped."
        )


# ---------------------------------------------------------------------------
# Defect 7: codebase_graph.cmd_scan returns skip count; main exits non-zero
# ---------------------------------------------------------------------------


class TestCmdScanSkipCount:
    """cmd_scan must return the number of skipped files, not None."""

    @pytest.fixture
    def cg_mod(self):
        return _load_script(
            "codebase_graph",
            "C:/dev/Vetinari/scripts/codebase_graph.py",
        )

    def test_cmd_scan_returns_int(self, tmp_path: Path, cg_mod) -> None:
        """cmd_scan must return an int (skip count), not None."""
        conn = sqlite3.connect(":memory:")
        cg_mod.init_db(conn)

        result = cg_mod.cmd_scan(conn)

        assert isinstance(result, int), (
            f"cmd_scan must return int, got {type(result).__name__}. "
            "Old code returned None implicitly."
        )

    def test_cmd_scan_returns_zero_on_empty_project(self, tmp_path: Path, cg_mod) -> None:
        """cmd_scan returns 0 when no files are found."""
        conn = sqlite3.connect(":memory:")
        cg_mod.init_db(conn)

        with patch.object(cg_mod, "_iter_py_files", return_value=iter([])):
            result = cg_mod.cmd_scan(conn)

        assert result == 0

    def test_cmd_scan_returns_nonzero_on_parse_error(self, tmp_path: Path, cg_mod) -> None:
        """cmd_scan increments and returns the skip count when a file cannot be parsed."""
        conn = sqlite3.connect(":memory:")
        cg_mod.init_db(conn)

        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def (:", encoding="utf-8")  # syntax error

        with (
            patch.object(cg_mod, "_iter_py_files", return_value=iter([bad_file])),
            patch.object(cg_mod, "_scan_file", return_value=None),
        ):
            result = cg_mod.cmd_scan(conn)

        assert result > 0, (
            "cmd_scan must return a positive skip count when _scan_file returns None."
        )


# ---------------------------------------------------------------------------
# Defect 8: ASTIndexer.find_usages detects live code-body references
# ---------------------------------------------------------------------------


class TestFindUsagesNameRefs:
    """find_usages must find call sites in code bodies, not just import lines."""

    def _make_indexer(self, tmp_path: Path):
        from vetinari.repo_map import ASTIndexer

        return ASTIndexer(str(tmp_path))

    def test_finds_call_site_not_in_imports(self, tmp_path: Path) -> None:
        """A function called in code body but not imported must still be found."""
        (tmp_path / "caller.py").write_text(
            textwrap.dedent("""\
                def run():
                    my_func()
            """),
            encoding="utf-8",
        )

        idx = self._make_indexer(tmp_path)
        idx.index_project()

        usages = idx.find_usages("my_func")
        assert any("caller" in u for u in usages), (
            "find_usages must detect 'my_func' referenced in caller.py code body. "
            "Old implementation only checked imports and docstrings."
        )

    def test_does_not_find_unreferenced_name(self, tmp_path: Path) -> None:
        """A name that appears nowhere must not be returned."""
        (tmp_path / "module.py").write_text(
            "def some_func(): pass\n", encoding="utf-8"
        )

        idx = self._make_indexer(tmp_path)
        idx.index_project()

        usages = idx.find_usages("totally_absent_name_xyz123")
        assert usages == [], "No references to an absent name must return empty list."

    def test_name_refs_populated_on_index(self, tmp_path: Path) -> None:
        """FileIndex.name_refs must be non-empty after indexing a file with code."""
        (tmp_path / "worker.py").write_text(
            textwrap.dedent("""\
                def work():
                    result = compute()
                    return result
            """),
            encoding="utf-8",
        )

        from vetinari.repo_map import ASTIndexer

        idx = ASTIndexer(str(tmp_path))
        idx.index_project()

        file_idx = idx._index.get("worker.py")
        assert file_idx is not None, "worker.py must be indexed"
        assert "compute" in file_idx.name_refs, (
            "name_refs must include 'compute' which is called in the function body."
        )

    def test_name_refs_round_trips_through_cache(self, tmp_path: Path) -> None:
        """name_refs must survive serialization/deserialization through the JSON cache."""
        (tmp_path / "app.py").write_text(
            "def start():\n    launch_server()\n",
            encoding="utf-8",
        )

        from vetinari.repo_map import ASTIndexer

        # Index and save cache.
        idx1 = ASTIndexer(str(tmp_path))
        idx1.index_project()
        idx1._save_cache()

        # Load fresh indexer from cache only.
        idx2 = ASTIndexer(str(tmp_path))
        idx2._load_cache()

        file_idx = idx2._index.get("app.py")
        assert file_idx is not None, "app.py must be present after cache load"
        assert "launch_server" in file_idx.name_refs, (
            "name_refs must be preserved after round-trip through JSON cache."
        )
