"""Vetinari Unified CLI - thin facade over split command modules.

Delegates every subcommand to one of the implementation modules:
- ``cli_startup``   - logging, config, orchestrator, subsystem wiring
- ``cli_commands``  - run, serve, start, status, health, interactive, prompt, migrate
- ``cli_devops``    - upgrade, review, benchmark, mcp, diagnose, drift-check
- ``cli_training``  - kaizen, train, watch
- ``cli_packaging`` - init, doctor, models, forget, config, resume, quick-action verbs

Global flags: --config PATH  --mode MODE  --verbose
"""

from __future__ import annotations

import argparse
import logging  # noqa: F401  - kept so patch("vetinari.cli.logging...") works in tests
import sys

from vetinari.cli_commands import (
    _health_check_quiet,  # noqa: F401  - re-exported for tests
    cmd_health,
    cmd_interactive,
    cmd_migrate,
    cmd_prompt,
    cmd_run,
    cmd_serve,
    cmd_start,
    cmd_status,
)
from vetinari.cli_devops import (
    _register_devops_commands,
    cmd_benchmark,
    cmd_diagnose,
    cmd_drift_check,
    cmd_mcp,
    cmd_review,
    cmd_upgrade,
)
from vetinari.cli_packaging import (
    _register_packaging_commands,
    cmd_config_reload,
    cmd_doctor,
    cmd_forget,
    cmd_init,
    cmd_models,
    cmd_quick_action,
    cmd_resume,
)
from vetinari.cli_startup import (
    _build_orchestrator,  # noqa: F401  - re-exported for tests
    _load_config,  # noqa: F401  - re-exported for tests
    _print_banner,  # noqa: F401  - re-exported for tests
    _setup_logging,
)
from vetinari.cli_training import (
    _register_kaizen_commands,
    _register_training_commands,
    _register_watch_commands,
    cmd_kaizen,
    cmd_train,
    cmd_watch,
)

# Re-exported so ``from vetinari.cli import cmd_kaizen`` keeps working for tests.
__all__ = ["cmd_kaizen", "main"]


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command handler.

    Reads global flags (--config, --mode, --verbose), registers all subcommand
    parsers from the split command modules, then routes to the matching handler.
    Defaults to the ``start`` command when no subcommand is given.
    """
    parser = argparse.ArgumentParser(
        prog="vetinari",
        description="Vetinari: Comprehensive AI Orchestration System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vetinari start --goal "Build a Python REST API with JWT auth"
  vetinari run --task t1 --config manifest/vetinari.yaml
  vetinari serve --port 5001
  vetinari status
  vetinari review
  vetinari interactive
""",
    )

    # Global flags
    parser.add_argument("--config", default="manifest/vetinari.yaml", help="Path to manifest file")
    parser.add_argument(
        "--mode",
        default="execution",
        choices=["planning", "execution", "sandbox"],
        help="Execution mode",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run
    p_run = subparsers.add_parser("run", help="Execute a goal or manifest task")
    p_run.add_argument("--goal", "-g", help="High-level goal string")
    p_run.add_argument("--task", "-t", help="Specific task ID from manifest")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start the web dashboard")
    p_serve.add_argument("--port", type=int, default=None, help="Web server port (default 5000)")
    p_serve.add_argument("--web-host", default=None, help="Web server bind address")
    p_serve.add_argument("--debug", action="store_true", help="Enable debug mode")

    # start
    p_start = subparsers.add_parser("start", help="Start Vetinari (CLI + optional dashboard)")
    p_start.add_argument("--goal", "-g", help="Execute this goal on startup")
    p_start.add_argument("--task", "-t", help="Execute this task on startup")
    p_start.add_argument("--port", type=int, default=None, help="Dashboard port")
    p_start.add_argument("--web-host", default=None, help="Dashboard bind address")
    p_start.add_argument("--no-dashboard", action="store_true", help="Disable web dashboard")
    p_start.add_argument("--skip-preflight", action="store_true", help="Skip dependency preflight check")

    # status
    subparsers.add_parser("status", help="Show system status")

    # health
    subparsers.add_parser("health", help="Health check all providers")

    # interactive
    subparsers.add_parser("interactive", help="Enter interactive REPL mode")

    # prompt versioning
    p_prompt = subparsers.add_parser("prompt", help="Manage agent prompt versions")
    p_prompt.add_argument("action", choices=["history", "rollback"], help="Action to perform")
    p_prompt.add_argument("agent", help="Agent type (e.g. WORKER)")
    p_prompt.add_argument("--mode", default="build", help="Agent mode (default: build)")
    p_prompt.add_argument("--version", help="Version to rollback to (required for rollback)")

    # migrate
    p_migrate = subparsers.add_parser("migrate", help="Apply database schema migrations")
    p_migrate.add_argument(
        "--db-path",
        default=None,
        help="Path to the SQLite database (default: VETINARI_DB_PATH env var or .vetinari/vetinari.db)",
    )

    # Register command groups from split modules
    _register_devops_commands(subparsers)
    _register_kaizen_commands(subparsers)
    _register_training_commands(subparsers)
    _register_watch_commands(subparsers)
    _register_packaging_commands(subparsers)

    args = parser.parse_args()

    _setup_logging(getattr(args, "verbose", False))

    # Default command when none is given: start (interactive)
    if args.command is None:
        args.command = "start"
        if not hasattr(args, "goal"):
            args.goal = None
        if not hasattr(args, "task"):
            args.task = None
        if not hasattr(args, "port"):
            args.port = None
        if not hasattr(args, "web_host"):
            args.web_host = None
        if not hasattr(args, "no_dashboard"):
            args.no_dashboard = False
        if not hasattr(args, "skip_preflight"):
            args.skip_preflight = False

    dispatch: dict[str, object] = {
        "run": cmd_run,
        "serve": cmd_serve,
        "start": cmd_start,
        "status": cmd_status,
        "health": cmd_health,
        "upgrade": cmd_upgrade,
        "review": cmd_review,
        "interactive": cmd_interactive,
        "benchmark": cmd_benchmark,
        "mcp": cmd_mcp,
        "drift-check": cmd_drift_check,
        "diagnose": cmd_diagnose,
        "prompt": cmd_prompt,
        "kaizen": cmd_kaizen,
        "train": cmd_train,
        "watch": cmd_watch,
        "migrate": cmd_migrate,
        "init": cmd_init,
        "doctor": cmd_doctor,
        "models": cmd_models,
        "forget": cmd_forget,
        "config": cmd_config_reload,
        "resume": cmd_resume,
        "explain": cmd_quick_action,
        "test": cmd_quick_action,
        "fix": cmd_quick_action,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))  # type: ignore[operator]


if __name__ == "__main__":
    main()
