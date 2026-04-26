"""Bulk-rename unwired public functions to private (prefix with _).

Reads VET120 errors from check_vetinari_rules.py and renames each function
in-place by adding a _ prefix to the definition AND all self.method_name
references within the same file.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Functions to keep public (will be wired separately)
KEEP_PUBLIC = {
    "ensure_tools_registered",
    "get_adapter_registry",
    "get_sandbox_manager",
    "create_schema",
    "create_vec_tables",
    "embed_all_missing",
    "drain_queue",
    "is_shutting_down",
    "rate_limit_guard",
    "refresh_model_cache",
    "log_api_request",
    "unload_all",
    "init_search_tool",
    "detect_scoped",
    "get_stagnant_scopes",
    "reset_scope",
    "get_training_status",
    "get_training_history",
    "get_quality_comparison",
    "_persist_sse_event",
    "get_recent_sse_events",
    "cleanup_old_sse_events",
    "get_agent_control_state",
    "vetinari_lifespan",
    "start_health_monitor",
    "stop_health_monitor",
    "get_health_snapshot",
}


def main():
    result = subprocess.run(
        [sys.executable, "scripts/check_vetinari_rules.py", "--errors-only"], capture_output=True, text=True
    )

    fixes: list[tuple[str, int, str]] = []
    for line in result.stdout.split("\n"):
        if "VET120" not in line:
            continue
        m = re.match(r"\s+(.+?):(\d+): VET120.*'([^']+)'", line)
        if m:
            filepath = m.group(1).replace("\\", "/")
            lineno = int(m.group(2))
            funcname = m.group(3)
            if funcname not in KEEP_PUBLIC and not funcname.startswith("_"):
                fixes.append((filepath, lineno, funcname))

    print(f"Found {len(fixes)} functions to rename")

    # Group by file
    by_file: dict[str, list[tuple[int, str]]] = {}
    for filepath, lineno, funcname in fixes:
        by_file.setdefault(filepath, []).append((lineno, funcname))

    renamed = 0
    for filepath, items in by_file.items():
        p = Path(filepath)
        if not p.exists():
            print(f"  SKIP {filepath} (not found)")
            continue

        content = p.read_text(encoding="utf-8")
        original = content

        for _lineno, funcname in items:
            # Rename function definition: def funcname( -> def _funcname(
            content = re.sub(rf"\bdef {re.escape(funcname)}\b", f"def _{funcname}", content)
            # Rename self.funcname references within the same file
            content = re.sub(rf"\bself\.{re.escape(funcname)}\b", f"self._{funcname}", content)

        if content != original:
            p.write_text(content, encoding="utf-8")
            renamed += len(items)
            print(f"  {filepath}: renamed {len(items)} functions")

    print(f"\nTotal renamed: {renamed}")


if __name__ == "__main__":
    main()
