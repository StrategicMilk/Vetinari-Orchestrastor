#!/usr/bin/env python3
"""Document all config keys from config/*.yaml files.

Reads every YAML file in the project's config/ directory, recursively
extracts all keys with their full dot-separated paths, infers the value
type, records the current value, and searches vetinari/ source files for
references to each key name.  Output can be a Markdown table or JSON.

Usage:
    python scripts/inspect/generate_config_docs.py [--output FILE] [--format FORMAT]

Exit codes:
    0   Success.
    1   Fatal error during generation.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
VETINARI_DIR = PROJECT_ROOT / "vetinari"


# ---------------------------------------------------------------------------
# YAML parsing (stdlib only — avoid adding optional deps)
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Any:
    """Load a YAML file using PyYAML if available, else a minimal fallback.

    Args:
        path: Absolute path to the YAML file.

    Returns:
        Parsed Python object (dict / list / scalar).

    Raises:
        ImportError: If PyYAML is not installed and the fallback cannot parse.
        OSError: If the file cannot be read.
    """
    try:
        import yaml  # type: ignore[import-untyped]

        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except ImportError as exc:
        raise ImportError("PyYAML is required: pip install pyyaml") from exc  # noqa: VET301 — user guidance string


# ---------------------------------------------------------------------------
# Key extraction
# ---------------------------------------------------------------------------


def _infer_type(value: Any) -> str:
    """Return a human-readable type label for a YAML value.

    Args:
        value: Python value decoded from YAML.

    Returns:
        Type label string such as ``"str"``, ``"int"``, ``"float"``,
        ``"bool"``, ``"list"``, ``"dict"``, or ``"null"``.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _truncate(value: Any, max_len: int = 60) -> str:
    """Render a value as a truncated string suitable for a table cell.

    Args:
        value: Any Python value.
        max_len: Maximum character length before truncation.

    Returns:
        String representation, truncated with ``...`` if necessary.
    """
    raw = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    if len(raw) > max_len:
        return raw[:max_len] + "..."
    return raw


def _extract_keys(
    data: Any,
    prefix: str = "",
    results: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Recursively extract all leaf keys from a nested YAML structure.

    Args:
        data: Parsed YAML value (dict, list, or scalar).
        prefix: Dot-separated key path accumulated so far.
        results: Accumulator list; created on first call.

    Returns:
        List of dicts with keys ``path``, ``type``, and ``value``.
    """
    if results is None:
        results = []

    if isinstance(data, dict):
        for key, val in data.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(val, (dict, list)):
                # Recurse into nested structures but also record dicts/lists
                results.append({
                    "path": child_path,
                    "type": _infer_type(val),
                    "value": val,
                })
                _extract_keys(val, child_path, results)
            else:
                results.append({
                    "path": child_path,
                    "type": _infer_type(val),
                    "value": val,
                })
    elif isinstance(data, list):
        for i, item in enumerate(data):
            child_path = f"{prefix}[{i}]"
            if isinstance(item, (dict, list)):
                _extract_keys(item, child_path, results)
            else:
                results.append({
                    "path": child_path,
                    "type": _infer_type(item),
                    "value": item,
                })

    return results


# ---------------------------------------------------------------------------
# Reference search
# ---------------------------------------------------------------------------


def _collect_source_files(vetinari_dir: Path) -> list[Path]:
    """Return all .py files under vetinari/, excluding __pycache__.

    Args:
        vetinari_dir: Root of the vetinari package.

    Returns:
        Sorted list of absolute .py paths.
    """
    return sorted(p for p in vetinari_dir.rglob("*.py") if "__pycache__" not in p.parts)


def _build_source_index(source_files: list[Path]) -> dict[str, str]:
    """Read all source files into memory keyed by path string.

    Args:
        source_files: List of absolute .py paths.

    Returns:
        Dict mapping path string to file content.
    """
    index: dict[str, str] = {}
    for path in source_files:
        try:
            index[str(path)] = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", path, exc)
    return index


def _find_references(
    key_name: str,
    source_index: dict[str, str],
    vetinari_dir: Path,
) -> list[str]:
    """Search vetinari source files for references to a config key name.

    Looks for the bare key name (last segment of the dotted path) as a
    whole-word match in string literals or dictionary key accesses.

    Args:
        key_name: The leaf key name to search for (e.g. ``"default_temperature"``).
        source_index: Dict of path -> content for all source files.
        vetinari_dir: Root path, used to compute relative file names.

    Returns:
        List of relative file paths that reference the key.
    """
    pattern = re.compile(r"\b" + re.escape(key_name) + r"\b")
    refs: list[str] = []
    for path_str, content in source_index.items():
        if pattern.search(content):
            rel = Path(path_str).relative_to(vetinari_dir.parent)
            refs.append(str(rel))
    return sorted(refs)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_markdown(
    file_records: list[dict[str, Any]],
) -> str:
    """Render config documentation as a Markdown document with one table per file.

    Args:
        file_records: List of per-file dicts with keys ``file``, ``keys``
            (list of key dicts including ``references``).

    Returns:
        Markdown string.
    """
    sections: list[str] = ["# Vetinari Config Documentation\n"]

    for record in file_records:
        sections.append(f"## `{record['file']}`\n")
        if not record["keys"]:
            sections.append("_No keys found._\n")
            continue

        sections.extend([
            "| Key Path | Type | Value | References in vetinari/ |",
            "|---|---|---|---|",
        ])

        for entry in record["keys"]:
            path = entry["path"]
            typ = entry["type"]
            val = _truncate(entry["value"])
            refs = entry.get("references", [])
            refs_cell = "<br>".join(refs) if refs else "_none_"
            # Escape pipe characters inside table cells
            val = val.replace("|", "\\|")
            refs_cell = refs_cell.replace("|", "\\|")
            sections.append(f"| `{path}` | {typ} | `{val}` | {refs_cell} |")

        sections.append("")

    return "\n".join(sections)


def format_json(file_records: list[dict[str, Any]]) -> str:
    """Render config documentation as JSON.

    Args:
        file_records: List of per-file dicts.

    Returns:
        JSON string with 2-space indentation.
    """
    # Make values JSON-serialisable
    serialisable: list[dict[str, Any]] = []
    for record in file_records:
        keys_out: list[dict[str, Any]] = []
        for entry in record["keys"]:
            val = entry["value"]
            # Convert non-serialisable types to their repr
            if not isinstance(val, (str, int, float, bool, list, dict, type(None))):
                val = str(val)
            keys_out.append({
                "path": entry["path"],
                "type": entry["type"],
                "value": val,
                "references": entry.get("references", []),
            })
        serialisable.append({"file": record["file"], "keys": keys_out})

    return json.dumps(serialisable, indent=2)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def collect_config_docs(
    config_dir: Path,
    vetinari_dir: Path,
) -> list[dict[str, Any]]:
    """Load all YAML config files and build the documentation records.

    Args:
        config_dir: Directory containing *.yaml config files.
        vetinari_dir: Vetinari source root for reference searching.

    Returns:
        List of per-file documentation dicts.

    Raises:
        FileNotFoundError: If config_dir does not exist.
    """
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    yaml_files = sorted(config_dir.glob("*.yaml"))
    if not yaml_files:
        logger.warning("No .yaml files found in %s", config_dir)
        return []

    source_files = _collect_source_files(vetinari_dir)
    source_index = _build_source_index(source_files)

    records: list[dict[str, Any]] = []
    for yaml_path in yaml_files:
        rel_name = yaml_path.name
        try:
            data = _load_yaml(yaml_path)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", yaml_path, exc)
            records.append({"file": rel_name, "keys": []})
            continue

        keys = _extract_keys(data)
        for entry in keys:
            leaf_name = entry["path"].split(".")[-1].lstrip("[").rstrip("]")
            entry["references"] = _find_references(leaf_name, source_index, vetinari_dir)

        records.append({"file": rel_name, "keys": keys})
        logger.info("Processed %s: %d keys", rel_name, len(keys))

    return records


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
        logger.info("Wrote config docs to %s", output_path)


def main() -> int:
    """Entry point for the config documentation generator.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Document all config keys from config/*.yaml files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/inspect/generate_config_docs.py\n"
            "  python scripts/inspect/generate_config_docs.py --format json --output docs/config.json\n"
            "  python scripts/inspect/generate_config_docs.py --format markdown"
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
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        metavar="DIR",
        help="Path to config/ directory (default: auto-detected).",
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

    try:
        records = collect_config_docs(args.config_dir, args.vetinari_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except ImportError as exc:
        logger.error("%s", exc)
        return 1

    fmt: str = args.format
    if fmt == "json":
        content = format_json(records)
    else:
        content = format_markdown(records)

    write_output(content, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
