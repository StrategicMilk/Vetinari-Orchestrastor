"""Vetinari RepoMap.

Inspired by Aider's repository mapping technique. Instead of sending entire
codebases to LLMs, generates a concise structural summary:
- Module names, class names, function signatures
- Imports and dependencies
- File relationships

This dramatically reduces token usage when giving LLMs codebase context,
while preserving structural understanding.

Usage:
    from vetinari.repo_map import get_repo_map

    mapper = get_repo_map()
    summary = mapper.generate(root_path="/path/to/project", max_tokens=2000)
    # summary is a concise string representing the codebase structure
"""

from __future__ import annotations

import ast
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Files/dirs to always skip
_SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    "*.egg-info",
    "model_cache",
    "vetinari_checkpoints",
    "logs",
    "outputs",
    "projects",
}
_SKIP_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".exe",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".lock",
    ".min.js",
    ".min.css",
}
_SKIP_FILES = {"__pycache__", ".DS_Store", "Thumbs.db"}


@dataclass
class ModuleInfo:
    """Structural information about a Python module."""

    path: str
    name: str
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    docstring: str = ""
    line_count: int = 0


class RepoMap:
    """Generates compact structural maps of codebases for LLM consumption.

    The output is a text representation showing:
    - Module hierarchy
    - Class names with their methods
    - Top-level function signatures
    - Key imports

    Designed to give LLMs structural awareness in ~500-2000 tokens instead
    of 10,000+ tokens for raw file contents.
    """

    def __init__(self):
        self._cache: dict[str, str] = {}  # path -> cached map

    def generate(
        self,
        root_path: str,
        max_tokens: int = 2000,
        include_private: bool = False,
        focus_paths: list[str] | None = None,
    ) -> str:
        """Generate a repository structure map.

        Args:
            root_path: Root directory to map.
            max_tokens: Approximate token limit (~4 chars/token).
            include_private: Include private (_name) members.
            focus_paths: If provided, only map these specific paths/modules.

        Returns:
            A string representation of the repository structure.
        """
        max_chars = max_tokens * 4
        root = Path(root_path)

        if not root.exists():
            return f"[RepoMap] Path not found: {root_path}"

        modules = self._scan_directory(root, focus_paths, include_private)

        if not modules:
            return f"[RepoMap] No Python files found in: {root_path}"

        lines = [f"# Repository Structure: {root.name}", ""]
        chars_used = sum(len(l) + 1 for l in lines)  # noqa: E741

        for mod in sorted(modules, key=lambda m: m.path):
            mod_lines = self._format_module(mod, include_private)
            mod_str = "\n".join(mod_lines) + "\n"

            if chars_used + len(mod_str) > max_chars:
                remaining = max_chars - chars_used
                if remaining > 100:
                    lines.append(mod_str[:remaining] + "\n  [... truncated]")
                lines.append(f"\n[{len(modules) - modules.index(mod)} more modules not shown — token limit]")
                break

            lines.extend(mod_lines)
            lines.append("")
            chars_used += len(mod_str)

        return "\n".join(lines)

    def generate_for_task(
        self,
        root_path: str,
        task_description: str,
        max_tokens: int = 1500,
    ) -> str:
        """Generate a task-focused repo map that emphasises relevant modules.

        Uses keyword matching to prioritise files likely relevant to the task.
        """
        root = Path(root_path)
        if not root.exists():
            return ""

        modules = self._scan_directory(root, None, False)
        if not modules:
            return ""

        # Score modules by relevance to task
        task_keywords = set(task_description.lower().split())
        scored = []
        for mod in modules:
            score = 0
            mod_text = (mod.name + " " + " ".join(mod.classes) + " " + " ".join(mod.functions)).lower()
            for kw in task_keywords:
                if kw in mod_text:
                    score += 1
            scored.append((score, mod))

        # Sort by relevance, then alphabetically
        scored.sort(key=lambda x: (-x[0], x[1].path))
        prioritised = [m for _, m in scored]

        return self.generate(root_path, max_tokens, False, [m.path for m in prioritised[:20]])

    def _scan_directory(
        self,
        root: Path,
        focus_paths: list[str] | None,
        include_private: bool,
    ) -> list[ModuleInfo]:
        """Scan directory and extract module information."""
        modules = []
        focus_set: set[str] | None = set(focus_paths) if focus_paths else None

        for py_file in self._iter_python_files(root):
            if focus_set and str(py_file) not in focus_set and py_file.name not in focus_set:
                # Check if the stem matches
                rel = str(py_file.relative_to(root))
                if rel not in focus_set and py_file.stem not in focus_set:
                    continue

            try:
                mod = self._parse_file(py_file, root)
                if mod:
                    modules.append(mod)
            except Exception as e:
                logger.debug("[RepoMap] Could not parse %s: %s", py_file, e)

        return modules

    def _iter_python_files(self, root: Path):
        """Yield Python files, respecting skip lists."""
        for item in root.rglob("*.py"):
            # Check if any parent directory should be skipped
            skip = False
            for part in item.parts:
                if part in _SKIP_DIRS or any(part.endswith(s.lstrip("*")) for s in _SKIP_DIRS if "*" in s):
                    skip = True
                    break
            if skip:
                continue
            if item.name in _SKIP_FILES:
                continue
            if item.suffix in _SKIP_EXTENSIONS:
                continue
            yield item

    def _parse_file(self, path: Path, root: Path) -> ModuleInfo | None:
        """Parse a Python file and extract structural information."""
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
            if len(source) > 100_000:  # Skip huge generated files
                return None

            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            return None
        except Exception:
            return None

        rel_path = str(path.relative_to(root))
        module_name = rel_path.replace(os.sep, ".").rstrip(".py")[:-3] if rel_path.endswith(".py") else rel_path

        mod = ModuleInfo(
            path=rel_path,
            name=module_name,
            line_count=len(source.splitlines()),
        )

        # Extract docstring
        if (
            isinstance(tree.body, list)
            and tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, (ast.Constant, ast.Str))
        ):
            doc = tree.body[0].value
            mod.docstring = (doc.s if isinstance(doc, ast.Str) else doc.value or "")[:80]

        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names[:2]:
                    mod.imports.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                mod.imports.append(node.module.split(".")[0])

        mod.imports = list(dict.fromkeys(mod.imports))[:8]

        # Extract classes and functions
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                mod.classes.append(self._format_class(node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = self._format_function(node)
                if sig:
                    mod.functions.append(sig)

        return mod

    def _format_class(self, node: ast.ClassDef) -> str:
        """Format a class definition as a concise string."""
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)

        base_names = []
        for base in node.bases[:2]:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)

        bases = f"({', '.join(base_names)})" if base_names else ""
        method_str = f" [{', '.join(methods[:6])}{'...' if len(methods) > 6 else ''}]" if methods else ""
        return f"{node.name}{bases}{method_str}"

    def _format_function(self, node) -> str | None:
        """Format a function signature as a concise string."""
        args = []
        for arg in node.args.args[:4]:
            args.append(arg.arg)
        if len(node.args.args) > 4:
            args.append("...")
        prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        return f"{prefix}{node.name}({', '.join(args)})"

    def _format_module(self, mod: ModuleInfo, include_private: bool) -> list[str]:
        """Format module info as a list of lines."""
        lines = [f"## {mod.path}"]
        if mod.docstring:
            lines.append(f"  # {mod.docstring[:60]}")

        if mod.classes:
            visible = [c for c in mod.classes if include_private or not c.startswith("_")]
            if visible:
                lines.append(f"  classes: {', '.join(visible[:5])}")

        if mod.functions:
            visible = [f for f in mod.functions if include_private or not f.startswith("_")]
            if visible:
                lines.append(f"  functions: {', '.join(visible[:8])}")

        return lines


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_repo_map: RepoMap | None = None


def get_repo_map() -> RepoMap:
    global _repo_map
    if _repo_map is None:
        _repo_map = RepoMap()
    return _repo_map


# ---------------------------------------------------------------------------
# AST Indexer
# ---------------------------------------------------------------------------


@dataclass
class SymbolInfo:
    """Information about a code symbol (class, function, variable)."""

    name: str
    kind: str  # "class", "function", "method", "variable", "import"
    file_path: str
    line_start: int
    line_end: int
    docstring: str = ""
    parent: str = ""  # parent class/function name
    decorators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FileIndex:
    """Index of a single Python file."""

    file_path: str
    mtime: float
    symbols: list[SymbolInfo] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "mtime": self.mtime,
            "symbols": [s.to_dict() for s in self.symbols],
            "imports": self.imports,
        }


class ASTIndexer:
    """AST-based Python code indexer.

    Parses Python files with the ast module to extract:
    - Classes and their methods
    - Top-level functions
    - Imports (what modules are used)
    - Docstrings

    Caches index to disk, invalidates on file mtime change.
    """

    CACHE_FILE = ".vetinari/ast_index.json"

    def __init__(self, root_path: str = "."):
        self._root = Path(root_path)
        self._index: dict[str, FileIndex] = {}
        self._symbol_table: dict[str, list[SymbolInfo]] = {}  # name -> locations
        self._loaded = False

    def index_project(self, force: bool = False) -> int:
        """Index all Python files in the project. Returns count of indexed files."""
        if not force:
            self._load_cache()

        indexed_count = 0
        for py_file in self._iter_python_files():
            rel_path = str(py_file.relative_to(self._root))
            mtime = py_file.stat().st_mtime

            # Skip if cached and not modified
            if not force and rel_path in self._index and self._index[rel_path].mtime >= mtime:
                continue

            file_index = self._index_file(py_file, rel_path, mtime)
            if file_index:
                self._index[rel_path] = file_index
                indexed_count += 1

        # Build symbol table
        self._build_symbol_table()

        # Save cache
        self._save_cache()

        return indexed_count

    def _iter_python_files(self):
        """Iterate over Python files, skipping hidden dirs and venvs."""
        skip_dirs = {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            "node_modules",
            ".tox",
            ".eggs",
            "build",
            "dist",
        }
        for py_file in self._root.rglob("*.py"):
            parts = py_file.relative_to(self._root).parts
            if any(p in skip_dirs or p.startswith(".") for p in parts[:-1]):
                continue
            yield py_file

    def _index_file(self, file_path: Path, rel_path: str, mtime: float) -> FileIndex | None:
        """Parse a single Python file and extract symbols."""
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, Exception):
            return None

        symbols = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                symbols.append(
                    SymbolInfo(
                        name=node.name,
                        kind="class",
                        file_path=rel_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        docstring=ast.get_docstring(node) or "",
                        decorators=[self._decorator_name(d) for d in node.decorator_list],
                    )
                )
                # Index methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append(
                            SymbolInfo(
                                name=item.name,
                                kind="method",
                                file_path=rel_path,
                                line_start=item.lineno,
                                line_end=item.end_lineno or item.lineno,
                                docstring=ast.get_docstring(item) or "",
                                parent=node.name,
                                decorators=[self._decorator_name(d) for d in item.decorator_list],
                            )
                        )

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only top-level functions (not methods — those are caught above)
                if not any(
                    isinstance(parent, ast.ClassDef)
                    for parent in ast.walk(tree)
                    if hasattr(parent, "body") and node in getattr(parent, "body", [])
                ):
                    symbols.append(
                        SymbolInfo(
                            name=node.name,
                            kind="function",
                            file_path=rel_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                            docstring=ast.get_docstring(node) or "",
                            decorators=[self._decorator_name(d) for d in node.decorator_list],
                        )
                    )

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return FileIndex(
            file_path=rel_path,
            mtime=mtime,
            symbols=symbols,
            imports=list(set(imports)),
        )

    def _decorator_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._decorator_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._decorator_name(node.func)
        return ""

    def _build_symbol_table(self):
        """Build a reverse lookup: symbol name -> list of SymbolInfo."""
        self._symbol_table.clear()
        for file_index in self._index.values():
            for symbol in file_index.symbols:
                self._symbol_table.setdefault(symbol.name, []).append(symbol)

    def find_symbol(self, name: str) -> list[SymbolInfo]:
        """Find all definitions of a symbol by name."""
        if not self._loaded and not self._index:
            self.index_project()
        return self._symbol_table.get(name, [])

    def find_usages(self, name: str) -> list[str]:
        """Find files that import or reference a symbol name."""
        if not self._loaded and not self._index:
            self.index_project()
        files: set[str] = set()
        for file_path, file_index in self._index.items():
            # Check imports
            for imp in file_index.imports:
                if name in imp:
                    files.add(file_path)
            # Check symbol references
            for sym in file_index.symbols:
                if name in sym.docstring or name == sym.parent:
                    files.add(file_path)
        return sorted(files)

    def get_file_symbols(self, file_path: str) -> list[SymbolInfo]:
        """Get all symbols in a specific file."""
        if not self._loaded and not self._index:
            self.index_project()
        fi = self._index.get(file_path)
        return fi.symbols if fi else []

    def get_import_graph(self) -> dict[str, list[str]]:
        """Get the import dependency graph."""
        if not self._loaded and not self._index:
            self.index_project()
        graph = {}
        for file_path, file_index in self._index.items():
            graph[file_path] = [imp for imp in file_index.imports if imp.startswith("vetinari")]
        return graph

    def get_stats(self) -> dict[str, int]:
        return {
            "files_indexed": len(self._index),
            "total_symbols": sum(len(fi.symbols) for fi in self._index.values()),
            "total_classes": sum(1 for fi in self._index.values() for s in fi.symbols if s.kind == "class"),
            "total_functions": sum(
                1 for fi in self._index.values() for s in fi.symbols if s.kind in ("function", "method")
            ),
        }

    def _load_cache(self):
        cache_file = self._root / self.CACHE_FILE
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                for entry in data:
                    symbols = [SymbolInfo(**s) for s in entry.get("symbols", [])]
                    fi = FileIndex(
                        file_path=entry["file_path"],
                        mtime=entry["mtime"],
                        symbols=symbols,
                        imports=entry.get("imports", []),
                    )
                    self._index[fi.file_path] = fi
                self._build_symbol_table()
                self._loaded = True
            except Exception as e:
                logger.debug(f"AST index cache load error: {e}")

    def _save_cache(self):
        try:
            cache_file = self._root / self.CACHE_FILE
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            data = [fi.to_dict() for fi in self._index.values()]
            cache_file.write_text(json.dumps(data), encoding="utf-8")
        except Exception as e:
            logger.debug(f"AST index cache save error: {e}")


_indexer: ASTIndexer | None = None


def get_ast_indexer(root_path: str = ".") -> ASTIndexer:
    global _indexer
    if _indexer is None:
        _indexer = ASTIndexer(root_path)
    return _indexer
