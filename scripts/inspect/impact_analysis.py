#!/usr/bin/env python3
"""Transitive impact analysis for vetinari/ modules.

Given a file path, builds the full import graph of vetinari/ and shows
either which modules depend on the given file (forward impact), or which
modules the given file depends on (reverse mode).

Usage:
    python scripts/inspect/impact_analysis.py FILE [--depth N] [--reverse] [--output FILE]

Exit codes:
    0   Success.
    1   Fatal error.
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
from collections import defaultdict, deque
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VETINARI_DIR = PROJECT_ROOT / "vetinari"

_SKIP_NAMES: frozenset[str] = frozenset({"__init__", "__main__"})


# ---------------------------------------------------------------------------
# Graph construction (shared with generate_architecture.py but standalone)
# ---------------------------------------------------------------------------


def _module_id(rel: Path) -> str:
    """Convert a path relative to vetinari/ to a dotted module identifier.

    Args:
        rel: Path relative to the vetinari/ directory, e.g. ``agents/base_agent.py``.

    Returns:
        Dotted module string, e.g. ``agents.base_agent``.
    """
    return ".".join(rel.with_suffix("").parts)


def _parse_imports(source: str, pkg_prefix: str, current_module: str = "") -> list[str]:
    """Extract intra-package import targets from Python source code.

    Relative imports (``from . import foo``, ``from .utils import helper``,
    ``from .. import top``) are resolved against *current_module*, the short
    dotted module id relative to the package root (e.g. ``"agents.base"``
    — **not** ``"vetinari.agents.base"``).

    Args:
        source: Python source code string.
        pkg_prefix: Top-level package name to filter on (e.g. ``"vetinari"``).
        current_module: Short dotted id of the file being parsed, relative to
            the package root.  Required for relative-import resolution; empty
            string silently skips relative imports.

    Returns:
        List of relative module ids (without the leading package prefix).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    deps: list[str] = []
    prefix = pkg_prefix + "."

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            deps.extend(alias.name[len(prefix) :] for alias in node.names if alias.name.startswith(prefix))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # Relative import — resolve against current_module.
                # level=1 → from . (same package), level=2 → from .. (parent), etc.
                if not current_module:
                    continue
                parts = current_module.split(".")
                anchor_parts = parts[: max(0, len(parts) - node.level)]
                if node.module:
                    resolved = ".".join(anchor_parts + [node.module]) if anchor_parts else node.module
                else:
                    # ``from . import foo`` — target is the anchor package itself.
                    resolved = ".".join(anchor_parts)
                if resolved:
                    deps.append(resolved)
            else:
                mod = node.module or ""
                if mod.startswith(prefix):
                    deps.append(mod[len(prefix) :])

    return [d for d in deps if d]


def _collect_py_files(root: Path) -> list[Path]:
    """Collect all .py files under root, skipping pycache and init/main files.

    Args:
        root: Directory to walk.

    Returns:
        Sorted list of absolute .py paths.
    """
    return sorted(p for p in root.rglob("*.py") if p.stem not in _SKIP_NAMES and "__pycache__" not in p.parts)


def build_forward_graph(vetinari_dir: Path) -> dict[str, set[str]]:
    """Build a graph mapping each module to the set of modules it imports.

    Args:
        vetinari_dir: Absolute path to vetinari/ package directory.

    Returns:
        Dict of ``{module_id: {imported_module_id, ...}}``.
    """
    graph: dict[str, set[str]] = defaultdict(set)
    py_files = _collect_py_files(vetinari_dir)

    known: set[str] = set()
    for path in py_files:
        mod_id = _module_id(path.relative_to(vetinari_dir))
        known.add(mod_id)
        graph[mod_id]  # ensure every node exists
    # Also register every directory that has an __init__.py as a known package id
    # so that ``from .agents import X`` edges are not silently dropped.
    for init in vetinari_dir.rglob("__init__.py"):
        pkg_rel = init.parent.relative_to(vetinari_dir)
        pkg_id = ".".join(pkg_rel.parts) if pkg_rel.parts else ""
        if pkg_id:
            known.add(pkg_id)

    for path in py_files:
        rel = path.relative_to(vetinari_dir)
        mod_id = _module_id(rel)
        source = path.read_text(encoding="utf-8")
        for dep in _parse_imports(source, "vetinari", current_module=mod_id):
            if dep in known and dep != mod_id:
                graph[mod_id].add(dep)

    return dict(graph)


def invert_graph(forward: dict[str, set[str]]) -> dict[str, set[str]]:
    """Reverse a directed graph so edges point from dependency to dependant.

    Args:
        forward: Forward graph (A imports B -> A -> B).

    Returns:
        Reverse graph (B -> {A, ...}) where A imports B.
    """
    reverse: dict[str, set[str]] = defaultdict(set)
    for mod_id in forward:
        reverse[mod_id]  # ensure every node exists
    for mod_id, deps in forward.items():
        for dep in deps:
            reverse[dep].add(mod_id)
    return dict(reverse)


# ---------------------------------------------------------------------------
# BFS traversal with depth limit
# ---------------------------------------------------------------------------


def bfs_dependents(
    start: str,
    graph: dict[str, set[str]],
    max_depth: int,
) -> dict[str, int]:
    """Find all reachable nodes from start up to max_depth hops.

    Args:
        start: Starting module id.
        graph: Directed graph to traverse (edges represent "depends on / imports").
        max_depth: Maximum BFS depth (number of hops).

    Returns:
        Dict mapping each reachable module id (excluding start) to its depth.
    """
    visited: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        if node in visited:
            continue
        if node != start:
            visited[node] = depth
        if depth < max_depth:
            for neighbour in sorted(graph.get(node, set())):
                if neighbour not in visited:
                    queue.append((neighbour, depth + 1))
    return visited


# ---------------------------------------------------------------------------
# Tree rendering
# ---------------------------------------------------------------------------


def _render_tree(
    start: str,
    graph: dict[str, set[str]],
    max_depth: int,
    prefix: str = "",
    depth: int = 0,
    visited: set[str] | None = None,
) -> list[str]:
    """Recursively render a dependency tree as indented text lines.

    Uses ASCII-only box-drawing characters so output works on Windows
    consoles whose active code page does not support Unicode line-drawing.

    Args:
        start: Current node being rendered.
        graph: Directed graph for traversal.
        max_depth: Maximum recursion depth.
        prefix: Current line prefix for branch drawing.
        depth: Current depth in the recursion.
        visited: Set of already-visited nodes to prevent cycles.

    Returns:
        List of strings, one per line of the rendered tree.
    """
    if visited is None:
        visited = set()

    lines: list[str] = []
    children = sorted(graph.get(start, set()))

    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        connector = "`-- " if is_last else "+-- "
        child_prefix = prefix + ("    " if is_last else "|   ")

        if child in visited:
            lines.append(f"{prefix}{connector}{child} (cycle)")
            continue

        lines.append(f"{prefix}{connector}{child}")

        if depth < max_depth - 1:
            visited.add(child)
            lines.extend(_render_tree(child, graph, max_depth, child_prefix, depth + 1, visited))
            visited.discard(child)

    return lines


def format_tree(
    target_mod: str,
    graph: dict[str, set[str]],
    max_depth: int,
    is_reverse: bool,
) -> str:
    """Format the impact analysis as a readable tree.

    Args:
        target_mod: The module being analysed.
        graph: Graph to traverse (already oriented for the desired direction).
        max_depth: Maximum traversal depth.
        is_reverse: If ``True``, the graph shows dependencies; if ``False``,
            it shows dependants.

    Returns:
        Multi-line string representing the tree.
    """
    direction = "depends on" if is_reverse else "depends on (transitive impact)"
    header = f"{target_mod}\n  [{direction}, depth={max_depth}]"

    tree_lines = _render_tree(target_mod, graph, max_depth)

    if not tree_lines:
        return f"{header}\n  (none)"

    # Summary BFS count
    reachable = bfs_dependents(target_mod, graph, max_depth)
    summary = f"\n  {len(reachable)} module(s) reachable within depth {max_depth}"

    return header + "\n" + "\n".join(tree_lines) + summary


# ---------------------------------------------------------------------------
# Module id resolution
# ---------------------------------------------------------------------------


def _resolve_module_id(file_arg: str, vetinari_dir: Path) -> str:
    """Resolve a user-supplied file path to a vetinari module id.

    Accepts absolute paths, paths relative to cwd, or dotted module ids.

    Args:
        file_arg: User-supplied argument (file path or module id).
        vetinari_dir: Absolute path to vetinari/ directory.

    Returns:
        Dotted module id relative to vetinari/ (e.g. ``"types"``).

    Raises:
        ValueError: If the path cannot be resolved to a vetinari module.
    """
    # Try as a filesystem path first
    candidate = Path(file_arg)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate

    if candidate.exists():
        try:
            rel = candidate.relative_to(vetinari_dir)
            return _module_id(rel)
        except ValueError:
            rel = None

    # Try dotted module id: strip leading "vetinari."
    mod_id = file_arg.replace("/", ".").replace("\\", ".")
    mod_id = mod_id.removeprefix("vetinari.")
    mod_id = mod_id.removesuffix(".py")

    return mod_id


def write_output(content: str, output_path: Path | None) -> None:
    """Write content to a file or stdout.

    Args:
        content: Text to write.
        output_path: Destination file, or ``None`` for stdout.
    """
    if output_path is None:
        sys.stdout.write(content)
        sys.stdout.write("\n")
    else:
        output_path.write_text(content + "\n", encoding="utf-8")
        logger.info("Wrote impact analysis to %s", output_path)


def main() -> int:
    """Entry point for the impact analysis tool.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Show what depends on a given vetinari file, transitively.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/inspect/impact_analysis.py vetinari/types.py\n"
            "  python scripts/inspect/impact_analysis.py vetinari/types.py --depth 2\n"
            "  python scripts/inspect/impact_analysis.py vetinari/types.py --reverse\n"
            "  python scripts/inspect/impact_analysis.py vetinari/agents/contracts.py --depth 1 --output impact.txt"
        ),
    )
    parser.add_argument(
        "file",
        metavar="FILE",
        help=(
            "File to analyse. Accepts a filesystem path (e.g. vetinari/types.py) "
            "or a dotted module id (e.g. types or agents.contracts)."
        ),
    )
    parser.add_argument(
        "--depth",
        "-d",
        type=int,
        default=3,
        metavar="N",
        help="Maximum transitive depth to traverse (default: 3).",
    )
    parser.add_argument(
        "--reverse",
        "-r",
        action="store_true",
        help=("Show what the given file depends ON (its imports), instead of what depends on it."),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write output to FILE instead of stdout.",
    )
    parser.add_argument(
        "--vetinari-dir",
        type=Path,
        default=VETINARI_DIR,
        metavar="DIR",
        help="Path to vetinari/ source directory (default: auto-detected).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    vetinari_dir: Path = args.vetinari_dir
    if not vetinari_dir.is_dir():
        logger.error("vetinari directory not found: %s", vetinari_dir)
        return 1

    try:
        target_mod = _resolve_module_id(args.file, vetinari_dir)
    except ValueError as exc:
        logger.error("Cannot resolve module id: %s", exc)
        return 1

    try:
        forward = build_forward_graph(vetinari_dir)
    except Exception as exc:
        logger.error("Failed to build import graph: %s", exc)
        return 1

    if target_mod not in forward:
        logger.error(
            "Module '%s' not found in vetinari/. Known modules: %d",
            target_mod,
            len(forward),
        )
        return 1

    if args.reverse:
        # Show what target_mod imports (forward graph)
        traversal_graph = forward
    else:
        # Show what imports target_mod (reverse/inverted graph)
        traversal_graph = invert_graph(forward)

    content = format_tree(
        target_mod=target_mod,
        graph=traversal_graph,
        max_depth=args.depth,
        is_reverse=args.reverse,
    )

    write_output(content, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
