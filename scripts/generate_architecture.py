#!/usr/bin/env python3
"""Generate a dependency/import graph of vetinari/ modules.

Walks all .py files under vetinari/, parses imports using the ast module,
and produces a diagram or list showing which modules depend on which other
vetinari modules.

Usage:
    python scripts/generate_architecture.py [--output FILE] [--format FORMAT]

Exit codes:
    0   Success.
    1   Fatal error during graph generation.
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
VETINARI_DIR = PROJECT_ROOT / "vetinari"

# Top-level package groups to cluster in the Mermaid diagram
_PACKAGE_GROUPS: list[str] = [
    "agents",
    "web",
    "learning",
    "orchestration",
    "analytics",
    "planning",
    "training",
    "memory",
    "dashboard",
    "skills",
    "config",
    "constraints",
    "context",
    "drift",
    "inference",
    "ml",
    "models",
    "resilience",
    "routing",
    "safety",
    "validation",
    "workflow",
    "optimization",
    "coding_agent",
    "benchmarks",
    "async_support",
    "image",
    "rag",
    "kaizen",
    "observability",
    "prompts",
    "enforcement",
]

_SKIP_NAMES: frozenset[str] = frozenset({"__init__", "__main__"})


def _module_id(rel: Path) -> str:
    """Convert a relative path inside vetinari/ to a dotted module id.

    Args:
        rel: Path relative to vetinari/ (e.g. ``agents/base_agent.py``).

    Returns:
        Dotted module string, e.g. ``agents.base_agent``.
    """
    parts = list(rel.with_suffix("").parts)
    return ".".join(parts)


def _mermaid_safe(name: str) -> str:
    """Convert a dotted module name to a Mermaid-safe node identifier.

    Args:
        name: Dotted module name.

    Returns:
        String with dots replaced by underscores.
    """
    return name.replace(".", "_")


def _collect_py_files(root: Path) -> list[Path]:
    """Recursively collect all .py files under root, excluding skipped names.

    Args:
        root: Directory to walk.

    Returns:
        Sorted list of absolute paths to .py files.
    """
    result: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if path.stem in _SKIP_NAMES:
            continue
        if "__pycache__" in path.parts:
            continue
        result.append(path)
    return result


def _parse_imports(source: str, pkg_prefix: str, current_module: str = "") -> list[str]:
    """Extract intra-package imports from Python source.

    Parses the AST and collects any ``import vetinari.X`` or
    ``from vetinari.X import ...`` statements, returning the imported module
    names without the leading ``vetinari.`` prefix.

    Relative imports (``from . import foo``, ``from .utils import helper``) are
    resolved against *current_module*, which must be the short dotted id of the
    file being parsed relative to the ``vetinari/`` directory (e.g.
    ``"agents.base"`` — **not** ``"vetinari.agents.base"``).

    Args:
        source: Python source code.
        pkg_prefix: Package prefix to filter (e.g. ``"vetinari"``).
        current_module: Short dotted module id of the file being parsed,
            relative to the package root.  Required for relative-import
            resolution; empty string silently skips relative imports.

    Returns:
        List of relative module identifiers (e.g. ``["agents.base_agent"]``).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    deps: list[str] = []
    prefix = pkg_prefix + "."

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(prefix):
                    deps.append(alias.name[len(prefix) :])
                elif alias.name == pkg_prefix:
                    deps.append("")
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # Relative import — resolve against current_module.
                # level=1 → from . (same package), level=2 → from .. (parent), etc.
                if not current_module:
                    continue
                parts = current_module.split(".")
                # Strip (level) components from the tail to find the anchor package.
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
                elif mod == pkg_prefix:
                    deps.append("")

    return [d for d in deps if d]


def build_graph(vetinari_dir: Path) -> dict[str, set[str]]:
    """Build the full module dependency graph.

    Args:
        vetinari_dir: Absolute path to the vetinari/ package directory.

    Returns:
        Dict mapping each module id to the set of module ids it imports.
    """
    graph: dict[str, set[str]] = defaultdict(set)
    py_files = _collect_py_files(vetinari_dir)

    # Build a set of known module ids so we only link to real modules.
    # Include __init__.py files so that package imports (e.g. "from .agents
    # import X") are not silently dropped — the package node "agents" must
    # appear in known even though __init__.py files are skipped as graph sources.
    known: set[str] = set()
    for path in py_files:
        rel = path.relative_to(vetinari_dir)
        mod_id = _module_id(rel)
        known.add(mod_id)
        graph[mod_id]  # ensure node exists even with no deps
    # Also register every directory that has an __init__.py as a known package id.
    for init in vetinari_dir.rglob("__init__.py"):
        pkg_rel = init.parent.relative_to(vetinari_dir)
        pkg_id = ".".join(pkg_rel.parts) if pkg_rel.parts else ""
        if pkg_id:
            known.add(pkg_id)

    for path in py_files:
        rel = path.relative_to(vetinari_dir)
        mod_id = _module_id(rel)
        source = path.read_text(encoding="utf-8")
        current_module = mod_id  # short id, e.g. "agents.base_agent"
        for dep in _parse_imports(source, "vetinari", current_module=current_module):
            if dep in known and dep != mod_id:
                graph[mod_id].add(dep)

    return dict(graph)


def _group_of(mod_id: str) -> str | None:
    """Return the top-level package group for a module id, or None.

    Args:
        mod_id: Dotted module id, e.g. ``agents.base_agent``.

    Returns:
        Group name string, or ``None`` if the module is at the root level.
    """
    parts = mod_id.split(".")
    if len(parts) > 1:
        return parts[0]
    return None


def format_mermaid(graph: dict[str, set[str]]) -> str:
    """Render the graph as a Mermaid TD diagram with subgraphs.

    Args:
        graph: Module dependency graph.

    Returns:
        Mermaid diagram string.
    """
    lines: list[str] = ["```mermaid", "graph TD"]

    # Collect modules by group
    groups: dict[str, list[str]] = defaultdict(list)
    ungrouped: list[str] = []
    for mod_id in sorted(graph):
        grp = _group_of(mod_id)
        if grp and grp in _PACKAGE_GROUPS:
            groups[grp].append(mod_id)
        else:
            ungrouped.append(mod_id)

    # Emit subgraphs
    for grp in _PACKAGE_GROUPS:
        if grp not in groups:
            continue
        lines.append(f"    subgraph {grp}")
        for mod_id in sorted(groups[grp]):
            node = _mermaid_safe(mod_id)
            label = mod_id.split(".")[-1]
            lines.append(f'        {node}["{label}"]')
        lines.append("    end")

    # Ungrouped root-level modules
    for mod_id in sorted(ungrouped):
        node = _mermaid_safe(mod_id)
        lines.append(f'    {node}["{mod_id}"]')

    # Emit edges (limit to avoid enormous diagrams)
    edge_count = 0
    for mod_id in sorted(graph):
        src = _mermaid_safe(mod_id)
        for dep in sorted(graph[mod_id]):
            dst = _mermaid_safe(dep)
            lines.append(f"    {src} --> {dst}")
            edge_count += 1

    lines.extend(["```", f"\n<!-- {len(graph)} nodes, {edge_count} edges -->"])
    return "\n".join(lines)


def format_dot(graph: dict[str, set[str]]) -> str:
    """Render the graph in Graphviz DOT format with cluster subgraphs.

    Args:
        graph: Module dependency graph.

    Returns:
        DOT language string.
    """
    lines: list[str] = ["digraph vetinari {", '    rankdir="LR";', "    node [shape=box];"]

    groups: dict[str, list[str]] = defaultdict(list)
    ungrouped: list[str] = []
    for mod_id in sorted(graph):
        grp = _group_of(mod_id)
        if grp and grp in _PACKAGE_GROUPS:
            groups[grp].append(mod_id)
        else:
            ungrouped.append(mod_id)

    for i, grp in enumerate(_PACKAGE_GROUPS):
        if grp not in groups:
            continue
        lines.extend([f"    subgraph cluster_{i} {{", f'        label="{grp}";'])
        for mod_id in sorted(groups[grp]):
            node = _mermaid_safe(mod_id)
            label = mod_id.split(".")[-1]
            lines.append(f'        {node} [label="{label}"];')
        lines.append("    }")

    for mod_id in sorted(ungrouped):
        node = _mermaid_safe(mod_id)
        lines.append(f'    {node} [label="{mod_id}"];')

    for mod_id in sorted(graph):
        src = _mermaid_safe(mod_id)
        for dep in sorted(graph[mod_id]):
            dst = _mermaid_safe(dep)
            lines.append(f"    {src} -> {dst};")

    lines.append("}")
    return "\n".join(lines)


def format_text(graph: dict[str, set[str]]) -> str:
    """Render the graph as a plain list of ``A -> B`` dependency lines.

    Args:
        graph: Module dependency graph.

    Returns:
        Plain-text string with one edge per line, or ``(no dependencies)`` for
        isolated nodes.
    """
    lines: list[str] = []
    for mod_id in sorted(graph):
        deps = sorted(graph[mod_id])
        if deps:
            lines.extend(f"{mod_id} -> {dep}" for dep in deps)
        else:
            lines.append(f"{mod_id} (no dependencies)")
    return "\n".join(lines)


def write_output(content: str, output_path: Path | None) -> None:
    """Write content to a file or stdout.

    Args:
        content: Text to write.
        output_path: File path to write to, or ``None`` for stdout.
    """
    if output_path is None:
        sys.stdout.write(content)
        sys.stdout.write("\n")
    else:
        output_path.write_text(content + "\n", encoding="utf-8")
        logger.info("Wrote architecture graph to %s", output_path)


def main() -> int:
    """Entry point for the architecture graph generator.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Generate a dependency/import graph of vetinari/ modules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/generate_architecture.py\n"
            "  python scripts/generate_architecture.py --format dot --output graph.dot\n"
            "  python scripts/generate_architecture.py --format text"
        ),
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
        "--format",
        "-f",
        choices=["mermaid", "dot", "text"],
        default="mermaid",
        help="Output format (default: mermaid).",
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
        graph = build_graph(vetinari_dir)
    except Exception as exc:
        logger.error("Failed to build graph: %s", exc)
        return 1

    fmt: str = args.format
    if fmt == "mermaid":
        content = format_mermaid(graph)
    elif fmt == "dot":
        content = format_dot(graph)
    else:
        content = format_text(graph)

    write_output(content, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
