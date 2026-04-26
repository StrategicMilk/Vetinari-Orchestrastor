"""Route-to-test coverage matrix generator — scans Litestar handler modules
and maps each route to its test file(s), producing a markdown audit report.

Post-Flask-cutover version: all routes are Litestar. Flask code paths removed.
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LITESTAR_APP = ROOT / "vetinari" / "web" / "litestar_app.py"
OUTPUT = ROOT / "docs" / "audit" / "ROUTE-TO-TEST-MATRIX.md"
TESTS_ROOT = ROOT / "tests"

HTTP_DECORATORS = {"get", "post", "put", "delete", "patch"}
MUTATING_METHODS = {"POST", "PUT", "DELETE", "PATCH"}
PATH_PARAM_RE = re.compile(r"{(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)(?::[^}]+)?}")


@dataclass(frozen=True)
class RouteRow:
    method: str
    path: str
    module: str
    auth: str
    streaming: str
    request_tests: str
    coverage_state: str
    body_schema: str
    evidence_type: str


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _detect_auth(decorators: list[str]) -> str:
    if any("@require_admin_token" in line for line in decorators):
        return "admin-token"
    if any("@require_admin" in line for line in decorators):
        return "admin/local"
    if any("admin_guard" in line for line in decorators):
        return "admin-guard"
    return "unspecified"


def _detect_streaming(path: str, decorators: list[str], function_line: str) -> str:
    lowered = f"{path} {' '.join(decorators)} {function_line}".lower()
    return "yes" if "stream" in lowered else "no"


def _decorator_name(decorator: ast.expr) -> str | None:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _decorator_path(decorator: ast.expr) -> str | None:
    if not isinstance(decorator, ast.Call) or not decorator.args:
        return None
    first_arg = decorator.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        return first_arg.value
    return None


def _annotation_label(annotation: ast.expr | None, source: str) -> str:
    if annotation is None:
        return ""
    return ast.get_source_segment(source, annotation) or ""


def _body_schema_for(function: ast.FunctionDef | ast.AsyncFunctionDef, method: str, path_value: str, source: str) -> str:
    path_params = {match.group("name") for match in PATH_PARAM_RE.finditer(path_value)}
    args = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
    body_candidates: list[str] = []
    for arg in args:
        if arg.arg == "request" and method in MUTATING_METHODS:
            return "raw-request"
        if arg.arg in {"self", "request", "socket", "scope", "state"} or arg.arg in path_params:
            continue
        annotation = _annotation_label(arg.annotation, source)
        body_candidates.append(f"{arg.arg}:{annotation}")
        if "UploadFile" in annotation:
            return "file-upload"
        if "dict" in annotation or "BaseModel" in annotation or arg.arg in {"data", "payload", "body"}:
            return "dict"
    if method in MUTATING_METHODS and body_candidates:
        return "unknown"
    return "none"


def _iter_routes(path: Path) -> list[tuple[list[str], str, str, str, str]]:
    """Extract Litestar route definitions from a handler module.

    Returns:
        List of (decorators, function_line, HTTP method, path, body_schema) tuples.
    """
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    routes: list[tuple[list[str], str, str, str, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorator_lines = [
            (ast.get_source_segment(source, decorator) or "").strip()
            for decorator in node.decorator_list
        ]
        function_line = lines[node.lineno - 1].strip()
        for decorator in node.decorator_list:
            name = _decorator_name(decorator)
            if name not in HTTP_DECORATORS:
                continue
            path_value = _decorator_path(decorator)
            if path_value is None:
                continue
            method = name.upper()
            body_schema = _body_schema_for(node, method, path_value, source)
            routes.append((decorator_lines, function_line, method, path_value, body_schema))
    return routes


def _test_inventory() -> set[str]:
    return {path.relative_to(ROOT).as_posix() for path in TESTS_ROOT.rglob("test_*.py")}


def _find_dedicated_tests(module_stem: str) -> list[str]:
    """Find dedicated test files for a module by naming convention."""
    special = {
        "litestar_analytics": ["tests/test_litestar_wave1_parity.py"],
        "litestar_dashboard_metrics": ["tests/test_litestar_wave1_parity.py"],
    }
    if module_stem in special:
        return [path for path in special[module_stem] if (ROOT / path).exists()]

    candidates = [
        f"tests/test_{module_stem}.py",
        f"tests/test_{module_stem.replace('_routes', '_api')}.py",
        f"tests/test_{module_stem.replace('_routes', '')}.py",
        f"tests/test_{module_stem.replace('_api', '')}.py",
    ]
    inventory = _test_inventory()
    return list(dict.fromkeys(candidate for candidate in candidates if candidate in inventory))


# Maps module stems to broad test files when no dedicated test exists
BROAD_TEST_MAP: dict[str, list[str]] = {
    # Wave 1 parity modules
    "litestar_analytics": ["tests/test_litestar_wave1_parity.py"],
    "litestar_dashboard_metrics": ["tests/test_litestar_wave1_parity.py"],
    "litestar_project_git": ["tests/test_litestar_wave1_parity.py"],
    "litestar_model_mgmt": ["tests/test_litestar_wave1_parity.py"],
    "litestar_models_catalog": ["tests/test_litestar_wave1_parity.py"],
    "litestar_models_discovery": ["tests/test_litestar_wave1_parity.py"],
    "litestar_system_status": ["tests/test_litestar_wave1_parity.py"],
    "litestar_system_hardware": ["tests/test_litestar_wave1_parity.py"],
    "litestar_system_content": ["tests/test_litestar_wave1_parity.py"],
    "litestar_log_stream": ["tests/test_litestar_log_stream.py"],
    "litestar_app": ["tests/test_litestar_app.py"],
    # Migrated modules — covered by broad harness
    "litestar_admin_routes": ["tests/test_web_apis.py"],
    "litestar_adr_routes": ["tests/test_web_apis.py"],
    "litestar_agents_api": ["tests/test_web_apis.py"],
    "litestar_analytics_routes": ["tests/test_web_apis.py"],
    "litestar_audit_api": ["tests/test_web_apis.py"],
    "litestar_chat_api": ["tests/test_web_apis.py"],
    "litestar_decomposition_routes": ["tests/test_web_apis.py"],
    "litestar_learning_api": ["tests/test_web_apis.py"],
    "litestar_memory_api": ["tests/test_web_apis.py"],
    "litestar_plans_api": ["tests/test_web_apis.py"],
    "litestar_ponder_routes": ["tests/test_web_apis.py"],
    "litestar_sandbox_api": ["tests/test_web_apis.py"],
    "litestar_search_api": ["tests/test_web_apis.py"],
    "litestar_skills_api": ["tests/test_web_apis.py"],
    "litestar_subtasks_api": ["tests/test_web_apis.py"],
    "litestar_tasks_api": ["tests/test_web_apis.py"],
    "litestar_training_api": ["tests/test_web_apis.py"],
    "litestar_training_routes": ["tests/test_web_apis.py"],
    "litestar_rules_routes": ["tests/test_web_apis.py"],
    "litestar_projects_api": ["tests/test_web_ui.py", "tests/test_web_apis.py"],
    "litestar_plan_api": ["tests/test_plan_api.py"],
    "litestar_mcp_transport": ["tests/test_mcp.py"],
}


def _evidence_type_for_test(test_path: str) -> str:
    text = (ROOT / test_path).read_text(encoding="utf-8", errors="replace")
    if "TestClient" in text and re.search(r"\bclient\.(get|post|put|patch|delete|request)\(", text):
        return "request-level"
    if ".fn(" in text or "handler.fn" in text:
        return "handler-direct"
    if "create_" in text and "_handlers" in text:
        return "route-registration"
    return "metadata"


def _coverage_for_module(module_stem: str) -> tuple[str, str, str]:
    dedicated = _find_dedicated_tests(module_stem)
    if dedicated:
        evidence_types = {_evidence_type_for_test(path) for path in dedicated}
        evidence = "request-level" if "request-level" in evidence_types else min(evidence_types)
        return (", ".join(dedicated), f"dedicated {evidence}", evidence)

    broad = [path for path in BROAD_TEST_MAP.get(module_stem, []) if (ROOT / path).exists()]
    if broad:
        evidence_types = {_evidence_type_for_test(path) for path in broad}
        evidence = "request-level" if "request-level" in evidence_types else min(evidence_types)
        return (", ".join(broad), f"broad {evidence} harness", evidence)

    return ("-", "unmapped", "none")


def _module_label(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _litestar_handler_modules() -> list[Path]:
    """Discover mounted Litestar handler modules in the web package.

    The app factory is the mounting source of truth, and the litestar_*.py glob
    keeps indirect factory modules such as litestar_training_api_part3 covered.
    """
    web_dir = ROOT / "vetinari" / "web"
    infra = {"litestar_app.py", "litestar_guards.py", "litestar_exceptions.py", "litestar_middleware.py"}
    modules = {
        p for p in web_dir.glob("litestar_*.py")
        if p.name not in infra
    }
    modules.add(web_dir / "approvals_api.py")
    return sorted(path for path in modules if path.exists())


def build_rows() -> list[RouteRow]:
    """Build route inventory from litestar_app.py and all handler modules."""
    litestar_modules = _litestar_handler_modules()
    route_files = [LITESTAR_APP] + litestar_modules
    rows: list[RouteRow] = []

    for file_path in route_files:
        module_stem = file_path.stem
        request_tests, coverage_state, evidence_type = _coverage_for_module(module_stem)
        for decorators, function_line, method, path_value, body_schema in _iter_routes(file_path):
            rows.append(
                RouteRow(
                    method=method,
                    path=path_value,
                    module=_module_label(file_path),
                    auth=_detect_auth(decorators),
                    streaming=_detect_streaming(path_value, decorators, function_line),
                    request_tests=request_tests,
                    coverage_state=coverage_state,
                    body_schema=body_schema,
                    evidence_type=evidence_type,
                )
            )

    rows.sort(key=lambda row: (row.module, row.path, row.method))
    return rows


def render(rows: list[RouteRow]) -> str:
    counts = Counter(row.coverage_state for row in rows)
    evidence_counts = Counter(row.evidence_type for row in rows)
    body_counts = Counter(row.body_schema for row in rows)
    lines = [
        "# Route To Test Matrix",
        "",
        "Generated from the live codebase by [generate_route_to_test_matrix.py](../../scripts/generate_route_to_test_matrix.py).",
        "",
        "This is a route inventory and evidence matrix, not a fuzz-proof certificate.",
        "Coverage states distinguish request-level, handler-direct, route-registration, metadata, and unmapped evidence.",
        "Platform-skipped fuzz suites and smoke checks must not be counted as schema/random fuzz proof.",
        "",
        "## Summary",
        "",
        f"- total routes inventoried: `{len(rows)}`",
        f"- body-bearing routes: `{len([row for row in rows if row.body_schema != 'none'])}`",
        "",
        "## Coverage States",
        "",
        "| Coverage State | Routes |",
        "| --- | ---: |",
    ]
    for coverage_state, count in sorted(counts.items()):
        lines.append(f"| `{coverage_state}` | {count} |")
    lines.extend([
        "",
        "## Evidence Types",
        "",
        "| Evidence Type | Routes |",
        "| --- | ---: |",
    ])
    for evidence_type, count in sorted(evidence_counts.items()):
        lines.append(f"| `{evidence_type}` | {count} |")
    lines.extend([
        "",
        "## Body Schemas",
        "",
        "| Body Schema | Routes |",
        "| --- | ---: |",
    ])
    for body_schema, count in sorted(body_counts.items()):
        lines.append(f"| `{body_schema}` | {count} |")
    lines.extend([
        "",
        "## Matrix",
        "",
        "| Method | Path | Module | Auth | Streaming | Body Schema | Tests | Coverage state | Evidence type |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    for row in rows:
        lines.append(
            f"| `{row.method}` | `{row.path}` | [{Path(row.module).name}](../../{row.module}) | "
            f"`{row.auth}` | `{row.streaming}` | `{row.body_schema}` | {row.request_tests} | "
            f"`{row.coverage_state}` | `{row.evidence_type}` |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- All routes are served by Litestar (ASGI). Flask was removed in Session 22.",
        "- `auth=admin-guard` means the route uses the Litestar `admin_guard` dependency.",
        "- `auth=admin-token` means the route is decorated with `@require_admin_token`.",
        "- `auth=admin/local` means the route is decorated with `@require_admin`.",
        "- `auth=unspecified` means no direct auth wrapper was found; middleware or app-level policy may still apply.",
        "- `streaming=yes` is a heuristic based on route path/function naming and should be verified.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = build_rows()
    OUTPUT.write_text(render(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} routes to {OUTPUT}")


if __name__ == "__main__":
    main()
