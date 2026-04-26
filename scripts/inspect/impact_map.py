#!/usr/bin/env python3
"""Target-specific blast-radius queries backed by the code graph and AI indexes."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from ai_accelerators import CLAUDE_DIR, collect_change_risk, collect_env_surface, collect_test_impact

DB_PATH = CLAUDE_DIR / "codebase-graph.db"


def _short(items: list[str] | set[str], limit: int = 8) -> list[str]:
    values = sorted({item for item in items if item})
    return values[:limit]


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _resolve_module(conn: sqlite3.Connection, query: str) -> sqlite3.Row | None:
    normalized = query.replace("\\", "/")
    rows = conn.execute(
        """
        SELECT path, qualified_name
        FROM modules
        WHERE path = ?
           OR qualified_name = ?
           OR path LIKE ?
           OR qualified_name LIKE ?
        ORDER BY
            CASE
                WHEN path = ? OR qualified_name = ? THEN 0
                WHEN path LIKE ? OR qualified_name LIKE ? THEN 1
                ELSE 2
            END,
            LENGTH(path)
        LIMIT 1
        """,
        (
            normalized,
            query,
            f"%{normalized}",
            f"%{query}%",
            normalized,
            query,
            f"%/{Path(normalized).name}",
            f"%.{query.split('.')[-1]}",
        ),
    ).fetchone()
    return rows


def _resolve_symbol(conn: sqlite3.Connection, query: str) -> sqlite3.Row | None:
    rows = conn.execute(
        """
        SELECT qualified_name, module_path, name, kind
        FROM symbols
        WHERE qualified_name = ?
           OR name = ?
           OR qualified_name LIKE ?
        ORDER BY
            CASE
                WHEN qualified_name = ? THEN 0
                WHEN name = ? THEN 1
                ELSE 2
            END,
            LENGTH(qualified_name)
        LIMIT 1
        """,
        (query, query, f"%{query}%", query, query),
    ).fetchone()
    return rows


def _reverse_importers(conn: sqlite3.Connection, module_qname: str) -> set[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT importer_path
        FROM imports
        WHERE imported_from = ? OR imported_name = ?
        """,
        (module_qname, module_qname),
    ).fetchall()
    return {row["importer_path"] for row in rows}


def _caller_modules(conn: sqlite3.Connection, symbol_qname: str) -> tuple[set[str], list[str]]:
    rows = conn.execute(
        """
        SELECT DISTINCT s.module_path, e.source
        FROM edges e
        JOIN symbols s ON s.qualified_name = e.source
        WHERE e.relation = 'calls' AND e.target = ?
        ORDER BY e.source
        """,
        (symbol_qname,),
    ).fetchall()
    modules = {row["module_path"] for row in rows}
    callers = [row["source"] for row in rows]
    return modules, callers


def _build_response(query: str) -> dict[str, object]:
    if not DB_PATH.exists():
        return {
            "query": query,
            "error": f"Code graph database missing at {DB_PATH}",
        }

    env_data = collect_env_surface()
    risk_rows = {row["path"]: row for row in collect_change_risk(limit=None)["rows"]}
    test_data = collect_test_impact()
    route_rows = test_data["route_rows"]
    source_to_tests = test_data["source_to_tests"]

    with _open_db() as conn:
        module_row = _resolve_module(conn, query)
        symbol_row = None
        caller_modules: set[str] = set()
        caller_symbols: list[str] = []

        if module_row is None:
            symbol_row = _resolve_symbol(conn, query)
        if symbol_row is not None:
            module_row = _resolve_module(conn, symbol_row["module_path"])
            caller_modules, caller_symbols = _caller_modules(conn, symbol_row["qualified_name"])

        if module_row is None:
            return {"query": query, "error": "No matching module or symbol found"}

        target_module_path = module_row["path"]
        target_module_qname = module_row["qualified_name"]
        impacted_modules = {target_module_path}
        reverse_importers = _reverse_importers(conn, target_module_qname)
        impacted_modules.update(reverse_importers)
        impacted_modules.update(caller_modules)

    transitive_modules = (reverse_importers | caller_modules) - {target_module_path}
    direct_routes = [
        f"{row.method} {row.path}"
        for row in route_rows
        if row.module == target_module_path
    ]
    transitive_routes = [
        f"{row.method} {row.path}"
        for row in route_rows
        if row.module in transitive_modules
    ]

    direct_tests: set[str] = set(source_to_tests.get(target_module_path, []))
    transitive_tests: set[str] = set()
    for module_path in transitive_modules:
        transitive_tests.update(source_to_tests.get(module_path, []))
    for row in route_rows:
        if row.request_tests == "-":
            continue
        tests = {item.strip() for item in row.request_tests.split(",") if item.strip() and item.strip() != "-"}
        if row.module == target_module_path:
            direct_tests.update(tests)
        elif row.module in transitive_modules:
            transitive_tests.update(tests)

    direct_env = [
        name
        for name, record in env_data["env_vars"].items()
        if target_module_path in record["files"]
    ]
    transitive_env = [
        name
        for name, record in env_data["env_vars"].items()
        if name not in direct_env and transitive_modules.intersection(record["files"])
    ]
    risk_row = risk_rows.get(target_module_path, {})

    payload = {
        "query": query,
        "kind": "symbol" if symbol_row is not None else "module",
        "module": target_module_path,
        "symbol": symbol_row["qualified_name"] if symbol_row is not None else None,
        "impact_score": int(risk_row.get("composite", 0)),
        "dimension_breakdown": {
            "history": int(risk_row.get("history", 0)),
            "fragility": int(risk_row.get("fragility", 0)),
            "blast_radius": int(risk_row.get("blast_radius", 0)),
            "complexity": int(risk_row.get("complexity", 0)),
            "env_surface": int(risk_row.get("env_surface", 0)),
            "test_gap": int(risk_row.get("test_gap", 0)),
        },
        "why_risky": list(risk_row.get("reasons", [])),
        "reverse_importers": sorted(reverse_importers),
        "caller_symbols": caller_symbols,
        "routes": sorted(set(direct_routes) | set(transitive_routes)),
        "direct_routes": sorted(direct_routes),
        "transitive_routes": sorted(transitive_routes),
        "tests": sorted(direct_tests | transitive_tests),
        "direct_tests": sorted(direct_tests),
        "transitive_tests": sorted(transitive_tests - direct_tests),
        "env_vars": sorted(set(direct_env) | set(transitive_env)),
        "direct_env_vars": sorted(direct_env),
        "transitive_env_vars": sorted(transitive_env),
    }
    return payload


def _format_text(payload: dict[str, object]) -> str:
    if payload.get("error"):
        return f"Impact query failed: {payload['error']}"

    lines = [
        f"Target: {payload['query']}",
        f"Kind: {payload['kind']}",
        f"Module: {payload['module']}",
    ]
    if payload.get("symbol"):
        lines.append(f"Symbol: {payload['symbol']}")
    lines.append(f"Impact Score: {payload.get('impact_score', 0)}")
    breakdown = payload.get("dimension_breakdown", {})
    if isinstance(breakdown, dict):
        lines.append(
            "Risk Breakdown: "
            f"history={breakdown.get('history', 0)}, "
            f"fragility={breakdown.get('fragility', 0)}, "
            f"blast_radius={breakdown.get('blast_radius', 0)}, "
            f"complexity={breakdown.get('complexity', 0)}, "
            f"env={breakdown.get('env_surface', 0)}, "
            f"test_gap={breakdown.get('test_gap', 0)}"
        )
    why_risky = payload.get("why_risky", [])
    if why_risky:
        lines.append("Why Risky: " + ", ".join(str(item) for item in why_risky[:4]))
    lines.append("")
    lines.append("Reverse Importers:")
    reverse_importers = payload.get("reverse_importers", [])
    if reverse_importers:
        for item in _short(reverse_importers):
            lines.append(f"  - {item}")
    else:
        lines.append("  - none")
    if payload.get("caller_symbols"):
        lines.append("")
        lines.append("Callers:")
        for item in _short(payload["caller_symbols"]):
            lines.append(f"  - {item}")
    lines.append("")
    lines.append("Direct Routes:")
    direct_routes = payload.get("direct_routes", [])
    if direct_routes:
        for item in _short(direct_routes):
            lines.append(f"  - {item}")
    else:
        lines.append("  - none")
    lines.append("")
    lines.append("Transitive Routes:")
    transitive_routes = payload.get("transitive_routes", [])
    if transitive_routes:
        for item in _short(transitive_routes):
            lines.append(f"  - {item}")
    else:
        lines.append("  - none")
    lines.append("")
    lines.append("Direct Tests:")
    direct_tests = payload.get("direct_tests", [])
    if direct_tests:
        for item in _short(direct_tests):
            lines.append(f"  - {item}")
    else:
        lines.append("  - none")
    lines.append("")
    lines.append("Transitive Tests:")
    transitive_tests = payload.get("transitive_tests", [])
    if transitive_tests:
        for item in _short(transitive_tests):
            lines.append(f"  - {item}")
    else:
        lines.append("  - none")
    lines.append("")
    lines.append("Direct Env / External Surface:")
    direct_env = payload.get("direct_env_vars", [])
    if direct_env:
        for item in _short(direct_env):
            lines.append(f"  - {item}")
    else:
        lines.append("  - none")
    lines.append("")
    lines.append("Transitive Env / External Surface:")
    transitive_env = payload.get("transitive_env_vars", [])
    if transitive_env:
        for item in _short(transitive_env):
            lines.append(f"  - {item}")
    else:
        lines.append("  - none")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query blast radius for a module or symbol")
    parser.add_argument("query", help="Module path, qualified module name, or symbol")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = _build_response(args.query)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(_format_text(payload))


if __name__ == "__main__":
    main()
