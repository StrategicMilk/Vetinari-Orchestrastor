"""DevOps and maintenance CLI commands for Vetinari.

Handles system health, model management, contract drift, MCP server
integration, diagnostics, and the self-improvement review cycle.

This module is part of the CLI pipeline:
argument parsing (cli.py) -> **devops commands** (cli_devops.py).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from vetinari.types import AgentType

logger = logging.getLogger(__name__)


def cmd_upgrade(args: Any) -> int:
    """Check for model upgrades by discovering available local models.

    Queries each configured adapter (llama-cpp, LiteLLM, etc.) for its
    available models and prints a summary. This is used to verify that
    newly downloaded GGUF files or newly configured cloud providers are
    visible to the system.

    Args:
        args: Parsed CLI arguments with config, mode, verbose.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from vetinari.cli_startup import _setup_logging

    _setup_logging(args.verbose)

    try:
        from vetinari.adapter_manager import get_adapter_manager

        mgr = get_adapter_manager()
        # discover_models() returns dict[provider_name, list[ModelInfo]]
        discovered = mgr.discover_models()
        total = sum(len(v) for v in discovered.values())
        print(f"[Vetinari] Discovered {total} models across {len(discovered)} provider(s)")
        for provider, models in discovered.items():
            if models:
                print(f"  [{provider}]")
                for m in models:
                    size = f"{m.memory_gb} GB" if m.memory_gb else "size unknown"
                    print(f"    - {m.name or m.id} ({size})")
        print("[Vetinari] Upgrade check complete")
        return 0
    except Exception as exc:
        print(f"[Vetinari] Upgrade check failed: {exc}")
        logger.warning(
            "cmd_upgrade failed during model discovery — check adapter configuration: %s",
            exc,
        )
        return 1


def cmd_review(args: Any) -> int:
    """Run the self-improvement agent to generate performance recommendations.

    Args:
        args: Parsed CLI arguments with verbose.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from vetinari.cli_startup import _setup_logging

    _setup_logging(args.verbose)

    print("[Vetinari] Running self-improvement review...")
    try:
        from vetinari.adapter_manager import get_adapter_manager
        from vetinari.agents import get_worker_agent
        from vetinari.agents.contracts import AgentTask

        agent = get_worker_agent()
        try:
            agent.initialize({"adapter_manager": get_adapter_manager()})
        except Exception:
            logger.warning("Could not initialize improvement agent with adapter manager", exc_info=True)

        task = AgentTask(
            task_id="review-cli",
            agent_type=AgentType.WORKER,
            description="Run system performance review",
            context={"review_type": "full"},
        )
        result = agent.execute(task)

        if result.success and result.output:
            recs = result.output.get("recommendations", [])
            applied = result.output.get("auto_applied", [])
            print(f"\n[Vetinari] Found {len(recs)} recommendations, auto-applied {len(applied)}")
            for rec in recs[:5]:
                priority = rec.get("priority", "?").upper()
                print(f"  [{priority}] {rec.get('action', '?')}")
                print(f"         Rationale: {rec.get('rationale', '')[:80]}")
        return 0
    except Exception as exc:
        print(f"[Vetinari] Review failed: {exc}")
        logger.warning("Self-improvement review command failed: %s — CLI returns exit code 1", exc)
        return 1


def cmd_benchmark(args: Any) -> int:
    """Run agent benchmarks and report any performance regressions.

    Args:
        args: Parsed CLI arguments with optional agents filter and verbose.

    Returns:
        Exit code (0 if no regressions, 1 if regressions detected or on error).
    """
    from vetinari.cli_startup import _setup_logging

    _setup_logging(args.verbose)

    # Single-case mode: run one benchmark case by "suite:case_id" composite ID
    single_case = getattr(args, "case", None)
    if single_case:
        print(f"[Vetinari] Running single benchmark case: {single_case}")
        try:
            from vetinari.benchmarks.runner import get_default_runner

            runner = get_default_runner()
            result = runner.run_single(single_case)
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.case_id}  score={result.score:.3f}  latency={result.latency_ms:.0f}ms")
            if result.error:
                print(f"  Error: {result.error}")
            return 0 if result.passed else 1
        except Exception as exc:
            print(f"[Vetinari] Single-case benchmark failed: {exc}")
            logger.warning("Single-case benchmark '%s' failed: %s", single_case, exc)
            return 1

    print("[Vetinari] Running agent benchmarks...")
    try:
        from vetinari.benchmarks.suite import BenchmarkSuite

        suite = BenchmarkSuite()
        agent_filter = getattr(args, "agents", None)
        results = suite.run_all(agent_types=agent_filter)
        suite.print_report(results)
        regressions = suite.check_regression(results)
        if regressions:
            print("\nREGRESSIONS DETECTED:")
            for r in regressions:
                print(f"  {r}")
            return 1
        return 0
    except Exception as exc:
        print(f"[Vetinari] Benchmark failed: {exc}")
        logger.warning("Benchmark suite failed to run: %s — no results available, CLI returns exit code 1", exc)
        return 1


def cmd_mcp(args: Any) -> int:
    """Start the MCP server for editor integration (stdio or http transport).

    For stdio transport, runs the JSON-RPC message loop on stdin/stdout.
    For http transport, the Litestar web server provides JSON-RPC over HTTP
    at POST /mcp/message — start with 'python -m vetinari' instead.

    Args:
        args: Parsed CLI arguments with transport, mcp_port, mcp_host,
              verbose.

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    from vetinari.cli_startup import _setup_logging

    _setup_logging(args.verbose)
    transport = getattr(args, "transport", "stdio")

    try:
        from vetinari.mcp.server import get_mcp_server

        server = get_mcp_server()

        if transport == "http":
            logger.warning(
                "HTTP transport is provided by the Litestar web server — "
                "start the server with 'python -m vetinari' and send JSON-RPC requests to POST /mcp/message"
            )
            return 0
        else:
            from vetinari.mcp.transport import StdioTransport

            print("[Vetinari] MCP stdio server ready", file=sys.stderr)
            stdio = StdioTransport(server)
            stdio.run()

        return 0
    except KeyboardInterrupt:
        logger.warning("MCP server interrupted by user — shutting down cleanly")
        return 0
    except Exception as exc:
        print(f"[Vetinari] MCP server failed: {exc}", file=sys.stderr)
        logger.warning("MCP server encountered a fatal error and will exit: %s — editor integration unavailable", exc)
        return 1


def cmd_diagnose(args: Any) -> int:
    """Trace execution history for a project and show what happened.

    Reads the project state, event log, and SSE events to produce a
    diagnostic timeline showing the sequence of agent actions, model
    selections, quality gate results, and any errors or anomalies.

    Args:
        args: Parsed CLI arguments — requires ``args.project_id``.

    Returns:
        0 on success, 1 if the project cannot be found or on error.
    """
    from vetinari.cli_startup import _setup_logging

    _setup_logging(args.verbose)
    project_id = args.project_id
    print(f"[Vetinari] Diagnosing project: {project_id}")
    print("=" * 60)

    try:
        from vetinari.constants import PROJECTS_DIR, VETINARI_STATE_DIR

        project_dir = PROJECTS_DIR / project_id
        if not project_dir.exists():
            print(f"  Project directory not found: {project_dir}")
            return 1

        # 1. Project metadata
        project_meta = project_dir / "project.yaml"
        if project_meta.exists():
            import yaml

            with project_meta.open(encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
            print(f"  Project: {meta.get('name', project_id)}")
            print(f"  Category: {meta.get('category', 'unknown')}")
            print(f"  Status: {meta.get('status', 'unknown')}")
            print(f"  Created: {meta.get('created_at', 'unknown')}")
        else:
            print(f"  No project.yaml found in {project_dir}")

        # 2. Plan state
        plan_file = project_dir / "plan.json"
        if plan_file.exists():
            import json

            plan_data = json.loads(plan_file.read_text(encoding="utf-8"))
            task_count = len(plan_data.get("tasks", []))
            print(f"\n  Plan: {plan_data.get('plan_id', 'unknown')}")
            print(f"  Goal: {plan_data.get('goal', 'N/A')[:80]}")
            print(f"  Phase: {plan_data.get('phase', 0)}")
            print(f"  Tasks: {task_count}")
            for task in plan_data.get("tasks", []):
                status = task.get("status", "unknown")
                agent = task.get("assigned_agent", "?")
                desc = task.get("description", "")[:60]
                print(f"    [{status:>10}] ({agent}) {desc}")
        else:
            print("\n  No plan.json found")

        # 3. Execution log
        exec_log = project_dir / "execution.log"
        if exec_log.exists():
            log_lines = exec_log.read_text(encoding="utf-8").splitlines()
            print(f"\n  Execution log: {len(log_lines)} entries")
            # Show last 10 entries
            for line in log_lines[-10:]:
                print(f"    {line}")
        else:
            print("\n  No execution.log found")

        # 4. Database state
        db_path = VETINARI_STATE_DIR / "vetinari.db"
        if db_path.exists():
            import sqlite3

            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                print(f"\n  Database: {len(tables)} tables")
                # Report row counts for unified-schema tables (schema.sql).
                if "PlanHistory" in tables:
                    cursor = conn.execute("SELECT COUNT(*) FROM PlanHistory")
                    count = cursor.fetchone()[0]
                    print(f"  Plan history entries: {count}")
                if "SubtaskMemory" in tables:
                    cursor = conn.execute("SELECT COUNT(*) FROM SubtaskMemory")
                    count = cursor.fetchone()[0]
                    print(f"  Subtask memory entries: {count}")
                if "ModelPerformance" in tables:
                    cursor = conn.execute("SELECT COUNT(*) FROM ModelPerformance")
                    count = cursor.fetchone()[0]
                    print(f"  Model performance records: {count}")
        else:
            print("\n  No database found")

        # 5. Output artefacts
        from vetinari.constants import OUTPUTS_DIR

        output_dir = OUTPUTS_DIR / project_id
        if output_dir.exists():
            artefacts = list(output_dir.iterdir())
            print(f"\n  Output artefacts: {len(artefacts)}")
            for art in artefacts[:10]:
                size = art.stat().st_size if art.is_file() else 0
                print(f"    {art.name} ({size:,} bytes)")
        else:
            print("\n  No output artefacts found")

        # 6. Training batch queue stats
        try:
            from vetinari.adapters.batch_processor import get_batch_processor

            queue_stats = get_batch_processor().get_queue_stats()
            print("\n  Training batch queue:")
            print(f"    enabled: {queue_stats['enabled']}")
            print(f"    total_queued: {queue_stats['total_queued']}")
            print(f"    flush_thread_active: {queue_stats['flush_thread_active']}")
        except (ImportError, AttributeError):
            logger.debug("Batch processor unavailable — skipping queue stats in diagnosis")

        print(f"\n{'=' * 60}")
        print("  Diagnosis complete.")
        return 0

    except Exception as exc:
        print(f"[Vetinari] Diagnosis failed: {exc}")
        logger.exception("Diagnosis failed for project %s", project_id)
        return 1


def cmd_drift_check(args: Any) -> int:
    """Run a full drift audit using DriftMonitor.

    Uses the DriftMonitor to check contract fingerprints, capability
    coverage, and schema validation.  Reports all detected drifts and
    exits 1 if any issues are found.

    Args:
        args: Parsed CLI arguments.  Recognises ``args.update`` (bool)
            to regenerate the drift baseline instead of checking.

    Returns:
        Exit code (0 if no drift, 1 if drift detected or on error).
    """
    from vetinari.cli_startup import _setup_logging

    _setup_logging(args.verbose)

    # Handle 'drift update' subcommand
    if getattr(args, "update", False):
        return _drift_update_baseline()

    print("[Vetinari] Running full drift audit...")
    try:
        from vetinari.drift.monitor import get_drift_monitor

        monitor = get_drift_monitor()
        report = monitor.run_full_audit()

        if report.is_clean:
            print(f"[Vetinari] No drift detected. ({report.duration_ms:.0f}ms)")
            return 0

        print(f"[Vetinari] Drift detected ({report.duration_ms:.0f}ms):")
        if report.contract_drifts:
            print(f"  Contract drifts: {len(report.contract_drifts)}")
            for name, info in report.contract_drifts.items():
                print(f"    - {name}: was {info.get('previous', '?')[:12]}.. now {info.get('current', '?')[:12]}..")
        if report.capability_drifts:
            print(f"  Capability drifts: {len(report.capability_drifts)}")
            for item in report.capability_drifts:
                print(f"    - {item}")
        if report.schema_errors:
            print(f"  Schema errors: {len(report.schema_errors)}")
            for schema_name, errors in report.schema_errors.items():
                for err in errors:
                    print(f"    - {schema_name}: {err}")
        for issue in report.issues:
            print(f"  - {issue}")
        return 1
    except Exception as exc:
        print(f"[Vetinari] Drift check failed: {exc}")
        logger.exception("Drift check failed")
        return 1


def _drift_update_baseline() -> int:
    """Regenerate the drift baseline snapshot by re-registering core contracts.

    Returns:
        0 on success, 1 on failure.
    """
    print("[Vetinari] Updating drift baseline...")
    try:
        from vetinari.drift.contract_registry import get_contract_registry

        registry = get_contract_registry()

        # Register core contracts to establish baseline
        from vetinari.agents.contracts import AgentResult, AgentSpec, ExecutionPlan, Task, VerificationResult

        for name, cls in [
            ("AgentSpec", AgentSpec),
            ("Task", Task),
            ("ExecutionPlan", ExecutionPlan),
            ("AgentResult", AgentResult),
            ("VerificationResult", VerificationResult),
        ]:
            # Register a default instance as the contract fingerprint
            try:
                instance = cls.__new__(cls)
                registry.register(name, instance)
            except Exception:
                logger.warning("Could not register %s for baseline", name)

        registry.snapshot()
        print("[Vetinari] Drift baseline updated successfully.")
        return 0
    except Exception as exc:
        print(f"[Vetinari] Baseline update failed: {exc}")
        logger.exception("Baseline update failed")
        return 1


def _register_devops_commands(subparsers: Any) -> None:
    """Register DevOps commands with the CLI argument parser.

    Args:
        subparsers: The argparse subparsers action group from the main parser.
    """
    subparsers.add_parser("upgrade", help="Check for model upgrades")
    subparsers.add_parser("review", help="Run self-improvement agent review")

    p_bench = subparsers.add_parser("benchmark", help="Run agent benchmarks")
    p_bench.add_argument("--agents", nargs="*", help="Specific agent types to benchmark")
    p_bench.add_argument("--case", metavar="SUITE:CASE_ID", help="Run a single benchmark case (e.g. toolbench:tb-l1-001)")

    p_mcp = subparsers.add_parser("mcp", help="Start MCP server for editor integration")
    p_mcp.add_argument(
        "--transport", default="stdio", choices=["stdio", "http"], help="Transport mode (default: stdio)"
    )
    p_mcp.add_argument(
        "--mcp-port", type=int, default=8765, help="HTTP transport port (default: 8765; http transport only)"
    )
    p_mcp.add_argument("--mcp-host", default="127.0.0.1", help="HTTP transport bind address (http transport only)")

    p_drift = subparsers.add_parser("drift-check", help="Check for contract drift across agents")
    p_drift.add_argument("--update", action="store_true", help="Regenerate drift baseline instead of checking")

    p_diagnose = subparsers.add_parser("diagnose", help="Trace execution history for a project")
    p_diagnose.add_argument("project_id", help="The project ID to diagnose")


__all__ = [
    "_register_devops_commands",
    "cmd_benchmark",
    "cmd_diagnose",
    "cmd_drift_check",
    "cmd_mcp",
    "cmd_review",
    "cmd_upgrade",
]
