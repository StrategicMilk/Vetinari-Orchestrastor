"""Core CLI command implementations for Vetinari.

Handles the fundamental operational commands: run, serve, start, status,
health, interactive, prompt versioning, and database migration.

This is step 2 of the CLI pipeline: argument parsing (cli.py) ->
**command execution** (cli_commands.py / cli_devops.py / cli_training.py).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any

from vetinari.constants import (
    MAIN_LOOP_POLL_INTERVAL,
    SHUTDOWN_TIMEOUT,
    THREAD_JOIN_TIMEOUT,
    THREAD_JOIN_TIMEOUT_SHORT,
    TRUNCATE_OUTPUT_PREVIEW,
    VETINARI_STARTUP_DELAY,
)

logger = logging.getLogger(__name__)

# Optional ASGI server — imported at module level so tests can patch
# ``vetinari.cli_commands.uvicorn`` without relying on sys.modules injection.
try:
    import uvicorn  # type: ignore[import-untyped]
except ImportError:
    uvicorn = None  # type: ignore[assignment]


def _resolve_web_port(value: int | str | None, *, env_var: str = "VETINARI_WEB_PORT") -> int:
    """Resolve and validate the dashboard port from CLI or environment."""
    raw_value: int | str = value if value is not None else os.environ.get(env_var, "5000")
    try:
        port = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_var} must be an integer between 1 and 65535, got {raw_value!r}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{env_var} must be between 1 and 65535, got {port}")
    return port


def _resolve_web_host(value: str | None, *, env_var: str = "VETINARI_WEB_HOST") -> str:
    """Resolve the dashboard bind address from CLI or environment."""
    return (value or os.environ.get(env_var, "127.0.0.1")).strip() or "127.0.0.1"


def cmd_run(args: Any) -> int:
    """Execute a goal or manifest task.

    When ``args.goal`` is set, routes through the two-layer orchestrator.
    Otherwise, uses the manifest-based orchestrator for task or full run.

    Args:
        args: Parsed CLI arguments with goal, task, config, mode, verbose.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from vetinari.cli_startup import _build_orchestrator, _check_drift_at_startup, _setup_logging

    _setup_logging(args.verbose)
    _check_drift_at_startup()

    if args.goal:
        # High-level goal → assembly-line pipeline
        import uuid as _uuid

        trace_id = str(_uuid.uuid4())[:12]
        print(f"[Vetinari] Running goal: {args.goal[:80]}")
        print(f"[Vetinari] Trace ID: {trace_id}")
        # Propagate trace_id to all log records for this goal execution
        _trace_logger = logging.LoggerAdapter(logger, {"trace_id": trace_id})
        _trace_logger.info("Starting goal execution with trace_id=%s", trace_id)
        try:
            from vetinari.orchestration.two_layer import get_two_layer_orchestrator

            orch = get_two_layer_orchestrator()
            # Wire agent context if orchestrator is available
            try:
                base_orch = _build_orchestrator(args.config, args.mode)
                orch.set_agent_context(base_orch._agent_context)
            except Exception:
                logger.warning("Could not wire agent context from base orchestrator", exc_info=True)
            results = orch.generate_and_execute(
                goal=args.goal,
                constraints={"mode": args.mode, "trace_id": trace_id},
            )
            print(f"\n[Vetinari] Completed: {results.get('completed', 0)} tasks")
            print(f"[Vetinari] Failed:    {results.get('failed', 0)} tasks")
            if results.get("final_output"):
                print("\n--- Final Output ---")
                print(str(results["final_output"])[:TRUNCATE_OUTPUT_PREVIEW])
            return 0
        except Exception as e:
            print(f"[Vetinari] Error: {e}")
            logger.exception("Goal execution failed")
            return 1

    # Manifest-based task execution
    try:
        orch = _build_orchestrator(args.config, args.mode)
        if args.task:
            print(f"[Vetinari] Running task: {args.task}")
            orch.run_task(args.task)
        else:
            print("[Vetinari] Running all manifest tasks...")
            orch.run_all()
        return 0
    except Exception as e:
        print(f"[Vetinari] Error: {e}")
        logger.exception("Run failed")
        return 1


def cmd_serve(args: Any) -> int:
    """Start the web dashboard.

    Args:
        args: Parsed CLI arguments with port, web_host, debug, verbose.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from vetinari.cli_startup import _setup_logging, _wire_subsystems

    _setup_logging(args.verbose)
    try:
        port = _resolve_web_port(args.port)
    except ValueError as exc:
        logger.warning("Invalid web port for serve command: %s", exc)
        print(f"[Vetinari] Invalid web port: {exc}")
        return 1
    web_host = _resolve_web_host(getattr(args, "web_host", None))

    print(f"[Vetinari] Starting web dashboard on {web_host}:{port}")
    print(f"[Vetinari] Dashboard URL: http://{web_host}:{port}")

    try:
        _wire_subsystems()
    except Exception as exc:
        logger.warning("Non-fatal: subsystem wiring failed: %s", exc)

    if uvicorn is None:
        print("[Vetinari] Web UI not available: uvicorn is not installed")
        print("[Vetinari] Install dependencies: pip install 'litestar>=2.12' 'uvicorn>=0.30'")  # noqa: VET301 — user guidance string
        logger.warning("uvicorn not installed — web dashboard unavailable")
        return 1

    try:
        from vetinari.web.litestar_app import create_app as create_litestar_app

        litestar_app = create_litestar_app(debug=args.debug)
        uvicorn.run(litestar_app, host=web_host, port=port, log_level="info")
        return 0
    except ImportError as e:
        print(f"[Vetinari] Web UI not available: {e}")
        print("[Vetinari] Install dependencies: pip install 'litestar>=2.12' 'uvicorn>=0.30'")  # noqa: VET301 — user guidance string
        logger.warning("Litestar not installed — web dashboard unavailable: %s", e)
        return 1
    except Exception as e:
        print(f"[Vetinari] Dashboard error: {e}")
        logger.warning(
            "Web dashboard failed to start on port %d: %s — web UI unavailable, CLI still functional",
            port,
            e,
        )
        return 1


def cmd_start(args: Any) -> int:
    """Start CLI + optional web dashboard (recommended entry point).

    Wires all subsystems, optionally launches the Litestar dashboard in a
    background thread via uvicorn, starts the AutoTuner cycle, then either
    executes a provided goal/task or blocks in interactive REPL mode.

    Args:
        args: Parsed CLI arguments with goal, task, port, no_dashboard,
              web_host, mode, verbose.

    Returns:
        Exit code (0 for success, 1 for error).

    Raises:
        KeyboardInterrupt: Propagated from the interactive REPL loop when the
            user presses Ctrl+C while no goal or task was provided and the
            dashboard is not running (handled by the top-level CLI entry point).
    """
    from vetinari.cli_startup import (
        _check_drift_at_startup,
        _print_banner,
        _setup_logging,
        _wire_subsystems,
    )

    _setup_logging(args.verbose)
    _check_drift_at_startup()
    _print_banner(args.mode)
    try:
        port = _resolve_web_port(args.port)
    except ValueError as exc:
        logger.warning("Invalid web port for start command: %s", exc)
        print(f"[Vetinari] Invalid web port: {exc}")
        return 1

    # Preflight: detect hardware, report missing deps, offer to install
    if not getattr(args, "skip_preflight", False):
        try:
            from vetinari.preflight import run_preflight

            report = run_preflight(interactive=sys.stdin.isatty())
            missing_required = [
                item.package
                for item in getattr(report, "dependency_matrix", [])
                if getattr(item, "status", "") == "missing-required"
            ]
            if missing_required:
                print(
                    "[Vetinari] Startup blocked: missing required dependencies: "
                    + ", ".join(missing_required)
                )
                print("[Vetinari] Install the required packages or rerun with --skip-preflight for diagnostics only.")
                return 1
        except Exception as exc:
            logger.warning("Preflight check failed before startup", exc_info=True)
            print(f"[Vetinari] Startup preflight failed: {exc}")
            return 1

    _wire_subsystems()

    # Start web dashboard in background thread
    # Default to loopback — require explicit opt-in for network binding
    web_host = _resolve_web_host(getattr(args, "web_host", None))
    dashboard_started = False
    dashboard_thread: threading.Thread | None = None

    if not args.no_dashboard:
        try:
            if uvicorn is None:
                raise ImportError("uvicorn is not installed")

            from vetinari.web.litestar_app import create_app as create_litestar_app

            _litestar_app = create_litestar_app()

            def _run_dashboard():
                uvicorn.run(_litestar_app, host=web_host, port=port, log_level="warning")

            dashboard_thread = threading.Thread(target=_run_dashboard, daemon=True, name="dashboard")
            dashboard_thread.start()
            time.sleep(VETINARI_STARTUP_DELAY)  # Give Litestar/uvicorn time to start
            if not dashboard_thread.is_alive():
                print(f"[Vetinari] Dashboard startup failed — uvicorn thread exited (port {port} may be in use)")
            else:
                print(f"[Vetinari] Dashboard started: http://{web_host}:{port}")
                dashboard_started = True
        except Exception as e:
            print(f"[Vetinari] Dashboard unavailable: {e}")

    # Run health check
    print("\n[Vetinari] Running startup health checks...")
    if not _health_check_quiet():
        print("[Vetinari] Startup health checks reported a degraded or failed subsystem.")

    # Start AutoTuner background cycle (every 15 minutes while running)
    _shutdown_event = threading.Event()

    def _auto_tuner_loop():
        while not _shutdown_event.is_set():
            _shutdown_event.wait(timeout=SHUTDOWN_TIMEOUT)  # 15 min, interruptible
            if _shutdown_event.is_set():
                break
            try:
                from vetinari.learning.auto_tuner import get_auto_tuner

                get_auto_tuner().run_cycle()
                logger.debug("[AutoTuner] Periodic cycle complete")
            except Exception as _at_err:
                logger.warning("[AutoTuner] Cycle error (non-fatal): %s", _at_err)

    _tuner_thread = threading.Thread(target=_auto_tuner_loop, daemon=True, name="auto-tuner")
    _tuner_thread.start()

    # Execute goal if provided
    if args.goal:
        return cmd_run(args)

    if args.task:
        return cmd_run(args)

    # Enter interactive REPL if no task specified
    if dashboard_started:
        print(f"\n[Vetinari] Dashboard running at http://{web_host}:{port}")
        print("[Vetinari] Press Ctrl+C to exit")
        try:
            while True:
                time.sleep(MAIN_LOOP_POLL_INTERVAL)
        except KeyboardInterrupt:
            print("\n[Vetinari] Shutting down...")
            _shutdown_event.set()
            _tuner_thread.join(timeout=THREAD_JOIN_TIMEOUT)
            if dashboard_thread is not None and dashboard_thread.is_alive():
                dashboard_thread.join(timeout=THREAD_JOIN_TIMEOUT_SHORT)
    else:
        return cmd_interactive(args)

    return 0


def cmd_status(args: Any) -> int:
    """Show system status: models loaded, providers, and learning state.

    Args:
        args: Parsed CLI arguments with config, verbose.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from vetinari.cli_startup import _setup_logging

    _setup_logging(args.verbose)

    print("\n[Vetinari] System Status")
    print(f"  Config:         {args.config}")

    # Check local inference adapter
    try:
        from vetinari.adapters.llama_cpp_local_adapter import LocalInferenceAdapter

        adapter = LocalInferenceAdapter()
        models = adapter.list_loaded_models()
        print(f"  Models loaded:  {len(models)}")
        for m in models[:5]:
            mid = m.get("id", m.get("model", "unknown")) if isinstance(m, dict) else str(m)
            print(f"    - {mid}")
    except Exception as e:
        print(f"  Local inference: UNREACHABLE ({e})")
        try:
            from vetinari.errors import find_remediation

            hint = find_remediation(str(e))
            if hint:
                print(f"    Hint: {hint.suggested_action}")
        except Exception:
            logger.warning(
                "Remediation hint lookup failed for error %r — status output continues without hint",
                str(e),
                exc_info=True,
            )

    # Adapter manager status
    try:
        from vetinari.adapter_manager import get_adapter_manager

        mgr = get_adapter_manager()
        status = mgr.get_status()
        providers = status.get("providers", {})
        print(f"\n  Providers: {len(providers)}")
        for pname, pinfo in list(providers.items())[:5]:
            health = pinfo.get("health", "unknown")
            print(f"    - {pname}: {health}")
    except Exception as e:
        print(f"  Adapter Manager: {e}")

    # Learning system status
    try:
        from vetinari.learning.model_selector import get_thompson_selector

        selector = get_thompson_selector()
        total_arms = len(selector._arms)
        total_pulls = sum(a.total_pulls for a in selector._arms.values())
        print(f"\n  Thompson Sampling: {total_arms} arms, {total_pulls} total pulls")
    except Exception as e:
        print(f"  Learning System: {e}")

    return 0


def cmd_health(args: Any) -> int:
    """Run health checks on all providers and print a summary.

    Args:
        args: Parsed CLI arguments with verbose.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from vetinari.cli_startup import _setup_logging

    _setup_logging(args.verbose)

    print("[Vetinari] Running health checks...")
    return 0 if _health_check_quiet() else 1


def cmd_interactive(args: Any) -> int:
    """Enter interactive REPL mode for iterative goal execution.

    Accepts goals via stdin and dispatches them through the two-layer
    orchestrator.  Special commands: /quit, /status, /review, /help.

    Args:
        args: Parsed CLI arguments with config, mode, verbose.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    from vetinari.cli_startup import _build_orchestrator, _setup_logging

    _setup_logging(args.verbose)

    print("[Vetinari] Interactive mode. Type your goal and press Enter.")
    print("Commands: /quit, /status, /review, /help")
    print("-" * 50)

    try:
        from vetinari.orchestration.two_layer import get_two_layer_orchestrator

        orch = get_two_layer_orchestrator()
        try:
            base_orch = _build_orchestrator(args.config, args.mode)
            orch.set_agent_context(base_orch._agent_context)
        except Exception:
            logger.warning("Could not wire agent context from base orchestrator", exc_info=True)
    except Exception:
        logger.warning("Two-layer orchestrator unavailable for interactive mode", exc_info=True)
        orch = None

    while True:
        try:
            goal = input("\nGoal> ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.warning("Interactive mode interrupted by user — exiting")
            print("\n[Vetinari] Exiting interactive mode.")
            return 0

        if not goal:
            continue
        if goal.lower() in ("/quit", "/exit", "quit", "exit"):
            print("[Vetinari] Goodbye.")
            return 0
        if goal.lower() == "/status":
            cmd_status(args)
            continue
        if goal.lower() == "/review":
            from vetinari.cli_devops import cmd_review

            cmd_review(args)
            continue
        if goal.lower() == "/help":
            print("  /quit   - Exit")
            print("  /status - Show system status")
            print("  /review - Run self-improvement review")
            print("  Any other text - Execute as a goal")
            continue

        print(f"\n[Vetinari] Working on: {goal[:60]}...")
        try:
            if orch:
                results = orch.generate_and_execute(goal=goal, constraints={"mode": args.mode})
                print(f"\n  Completed: {results.get('completed', 0)} tasks")
                if results.get("final_output"):
                    print("\n--- Output ---")
                    print(str(results["final_output"])[:1500])
            else:
                print("[Vetinari] Orchestrator not available. Check local inference adapter.")
        except Exception as e:
            print(f"[Vetinari] Error: {e}")
            logger.warning("Interactive execution error", exc_info=True)


def cmd_prompt(args: Any) -> int:
    """Manage agent prompt versions — history and rollback.

    Args:
        args: Parsed CLI arguments with action, agent, mode, version.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    # Reject path-like traversal strings before reaching the persistence layer.
    # Agent names are simple identifiers like "WORKER" or "FOREMAN" — slashes,
    # dots, and null bytes have no valid meaning here.
    agent_name = args.agent
    if any(ch in agent_name for ch in ("/", "\\", "..", "\x00")) or agent_name.startswith("."):
        print(f"Error: invalid agent name {agent_name!r} — must be a plain identifier")
        return 1

    from vetinari.prompts import get_version_manager

    mgr = get_version_manager()

    if args.action == "history":
        try:
            history = mgr.get_history(args.agent.upper(), args.mode)
        except ValueError as exc:
            # Persistence layer rejects traversal-style agent names before touching disk.
            logger.warning("Could not retrieve prompt history for agent %s — invalid agent name", args.agent)
            print(f"Error: invalid agent name — {exc}")
            return 1
        if not history:
            print(f"No prompt versions found for {args.agent}:{args.mode}")
            return 0
        print(f"Prompt history for {args.agent}:{args.mode}:")
        for v in history:
            score_str = f" (score: {v.quality_score:.3f})" if v.quality_score is not None else ""
            print(f"  {v.version}  {v.timestamp[:19]}  {v.checksum[:12]}...{score_str}  {v.notes}")
        return 0

    if args.action == "rollback":
        if not args.version:
            print("Error: --version is required for rollback")
            return 1
        try:
            result = mgr.rollback(args.agent.upper(), args.mode, args.version)
        except ValueError as exc:
            # Persistence layer rejects traversal-style agent names before touching disk.
            logger.warning("Could not rollback prompt version for agent %s — invalid agent name", args.agent)
            print(f"Error: invalid agent name — {exc}")
            return 1
        if result:
            print(f"Rolled back {args.agent}:{args.mode} to version {args.version} (new version: {result.version})")
            return 0
        print(f"Version {args.version} not found for {args.agent}:{args.mode}")
        return 1

    return 0


def cmd_migrate(args: Any) -> int:
    """Apply database migrations to initialise or upgrade storage schemas.

    Args:
        args: Parsed CLI arguments with db_path and optional verbose.

    Returns:
        0 on success, 1 on failure.
    """
    from vetinari.cli_startup import _setup_logging
    from vetinari.migrations import run_migrations

    db_path = args.db_path or os.environ.get("VETINARI_DB_PATH", ".vetinari/vetinari.db")
    _setup_logging(getattr(args, "verbose", False))
    logger.info("Running migrations on %s", db_path)
    try:
        applied = run_migrations(db_path)
        print(f"Migrations complete — {applied} applied to {db_path}")
        return 0
    except Exception:
        logger.exception("Migration failed")
        return 1


def _health_check_quiet() -> bool:
    """Run health checks on all providers and print results to stdout."""
    healthy = True
    try:
        from vetinari.adapters.llama_cpp_local_adapter import LocalInferenceAdapter

        adapter = LocalInferenceAdapter()
        is_healthy = adapter.is_healthy()
        if is_healthy:
            models = adapter.list_loaded_models()
            print(f"  Local inference: OK ({len(models)} models)")
        else:
            print("  Local inference: FAIL (unhealthy)")
            healthy = False
    except Exception as e:
        print(f"  Local inference: FAIL ({e})")
        healthy = False

    try:
        from vetinari.adapter_manager import get_adapter_manager

        mgr = get_adapter_manager()
        results = mgr.health_check()
        for name, info in results.items():
            status = "OK" if info.get("healthy") else "FAIL"
            print(f"  {name:20s}: {status}")
            if not info.get("healthy"):
                healthy = False
    except Exception:
        logger.warning("Adapter manager health check unavailable", exc_info=True)
        healthy = False
    return healthy


__all__ = [
    "_health_check_quiet",
    "_resolve_web_host",
    "_resolve_web_port",
    "cmd_health",
    "cmd_interactive",
    "cmd_migrate",
    "cmd_prompt",
    "cmd_run",
    "cmd_serve",
    "cmd_start",
    "cmd_status",
]
