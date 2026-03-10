"""Tests for ASTIndexer in vetinari.repo_map."""
import json
import os
import pytest
from pathlib import Path

from vetinari.repo_map import ASTIndexer, SymbolInfo, FileIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_py(tmp_path: Path, name: str, source: str) -> Path:
    """Write a Python file to tmp_path and return its path."""
    p = tmp_path / name
    p.write_text(source, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_index_simple_file(tmp_path):
    """Indexes a Python file and finds a class and a top-level function."""
    write_py(tmp_path, "simple.py", """
class MyClass:
    pass

def my_func():
    pass
""")
    indexer = ASTIndexer(str(tmp_path))
    count = indexer.index_project(force=True)
    assert count >= 1

    symbols = indexer.get_file_symbols("simple.py")
    kinds = {s.kind for s in symbols}
    names = {s.name for s in symbols}
    assert "class" in kinds
    assert "function" in kinds
    assert "MyClass" in names
    assert "my_func" in names


def test_index_class_with_methods(tmp_path):
    """Finds methods with the correct parent class name."""
    write_py(tmp_path, "cls.py", """
class Animal:
    def speak(self):
        pass

    def move(self):
        pass
""")
    indexer = ASTIndexer(str(tmp_path))
    indexer.index_project(force=True)

    symbols = indexer.get_file_symbols("cls.py")
    methods = [s for s in symbols if s.kind == "method"]
    assert len(methods) == 2
    for m in methods:
        assert m.parent == "Animal"
    method_names = {m.name for m in methods}
    assert method_names == {"speak", "move"}


def test_index_imports(tmp_path):
    """Extracts import statements from a Python file."""
    write_py(tmp_path, "imports.py", """
import os
import sys
from pathlib import Path
from collections import defaultdict
""")
    indexer = ASTIndexer(str(tmp_path))
    indexer.index_project(force=True)

    fi = indexer._index.get("imports.py")
    assert fi is not None
    assert "os" in fi.imports
    assert "sys" in fi.imports
    assert "pathlib" in fi.imports
    assert "collections" in fi.imports


def test_index_docstrings(tmp_path):
    """Captures docstrings for classes and functions."""
    write_py(tmp_path, "docs.py", '''
class Documented:
    """This is a class docstring."""
    pass

def helper():
    """Helper function docstring."""
    pass
''')
    indexer = ASTIndexer(str(tmp_path))
    indexer.index_project(force=True)

    symbols = {s.name: s for s in indexer.get_file_symbols("docs.py")}
    assert "This is a class docstring." in symbols["Documented"].docstring
    assert "Helper function docstring." in symbols["helper"].docstring


def test_index_decorators(tmp_path):
    """Captures decorator names for classes and functions."""
    write_py(tmp_path, "decs.py", """
def mydecorator(f):
    return f

@mydecorator
def decorated():
    pass

class Base:
    @staticmethod
    def static_method():
        pass
""")
    indexer = ASTIndexer(str(tmp_path))
    indexer.index_project(force=True)

    symbols = {s.name: s for s in indexer.get_file_symbols("decs.py")}
    assert "mydecorator" in symbols["decorated"].decorators
    assert "staticmethod" in symbols["static_method"].decorators


def test_find_symbol(tmp_path):
    """find_symbol returns SymbolInfo objects matching the given name."""
    write_py(tmp_path, "a.py", """
class Foo:
    pass
""")
    write_py(tmp_path, "b.py", """
class Foo:
    pass
""")
    indexer = ASTIndexer(str(tmp_path))
    indexer.index_project(force=True)

    results = indexer.find_symbol("Foo")
    assert len(results) == 2
    assert all(s.name == "Foo" for s in results)
    assert all(s.kind == "class" for s in results)


def test_find_usages(tmp_path):
    """find_usages returns files that import a module by name."""
    write_py(tmp_path, "mylib.py", """
class Engine:
    pass
""")
    write_py(tmp_path, "consumer.py", """
from mylib import Engine

class MyEngine:
    pass
""")
    indexer = ASTIndexer(str(tmp_path))
    indexer.index_project(force=True)

    # find_usages searches import strings; consumer.py imports "mylib"
    usages = indexer.find_usages("mylib")
    assert any("consumer.py" in u for u in usages)


def test_get_file_symbols(tmp_path):
    """get_file_symbols returns only symbols from the requested file."""
    write_py(tmp_path, "only.py", """
class Alpha:
    def method_a(self):
        pass

def standalone():
    pass
""")
    write_py(tmp_path, "other.py", """
class Beta:
    pass
""")
    indexer = ASTIndexer(str(tmp_path))
    indexer.index_project(force=True)

    symbols = indexer.get_file_symbols("only.py")
    names = {s.name for s in symbols}
    assert "Alpha" in names
    assert "method_a" in names
    assert "standalone" in names
    assert "Beta" not in names


def test_get_import_graph(tmp_path):
    """get_import_graph returns a dict keyed by file path."""
    write_py(tmp_path, "mod.py", """
import os
from vetinari.repo_map import get_repo_map
""")
    indexer = ASTIndexer(str(tmp_path))
    indexer.index_project(force=True)

    graph = indexer.get_import_graph()
    assert "mod.py" in graph
    # Only vetinari.* imports appear in the graph values
    assert any("vetinari" in imp for imp in graph["mod.py"])


def test_get_stats(tmp_path):
    """get_stats returns correct counts for files, classes, functions."""
    write_py(tmp_path, "s1.py", """
class A:
    def m(self): pass

def f(): pass
""")
    write_py(tmp_path, "s2.py", """
class B:
    pass
""")
    indexer = ASTIndexer(str(tmp_path))
    indexer.index_project(force=True)

    stats = indexer.get_stats()
    assert stats["files_indexed"] == 2
    assert stats["total_classes"] == 2
    # m (method) + f (function) = 2
    assert stats["total_functions"] >= 2
    assert stats["total_symbols"] >= 4


def test_cache_persistence(tmp_path):
    """Index saved to cache can be reloaded without re-scanning files."""
    write_py(tmp_path, "cached.py", """
class Cached:
    pass
""")
    indexer1 = ASTIndexer(str(tmp_path))
    indexer1.index_project(force=True)

    # Cache file should exist
    cache_path = tmp_path / ".vetinari" / "ast_index.json"
    assert cache_path.exists()

    # Second indexer loads from cache
    indexer2 = ASTIndexer(str(tmp_path))
    indexer2._load_cache()
    assert "cached.py" in indexer2._index
    symbols = indexer2.get_file_symbols("cached.py")
    assert any(s.name == "Cached" for s in symbols)


def test_skip_hidden_dirs(tmp_path):
    """Hidden directories and __pycache__ are not indexed."""
    # Create files inside skipped dirs
    hidden = tmp_path / ".hidden_dir"
    hidden.mkdir()
    write_py(hidden, "secret.py", "class Secret: pass")

    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    write_py(pycache, "compiled.py", "class Compiled: pass")

    venv_dir = tmp_path / "venv"
    venv_dir.mkdir()
    write_py(venv_dir, "venv_mod.py", "class VenvMod: pass")

    # A normal file that should be indexed
    write_py(tmp_path, "visible.py", "class Visible: pass")

    indexer = ASTIndexer(str(tmp_path))
    count = indexer.index_project(force=True)

    indexed_paths = set(indexer._index.keys())
    assert "visible.py" in indexed_paths
    assert not any("secret.py" in p for p in indexed_paths)
    assert not any("compiled.py" in p for p in indexed_paths)
    assert not any("venv_mod.py" in p for p in indexed_paths)
