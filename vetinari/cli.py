"""
Vetinari Unified CLI

Single entry point for all Vetinari operations:

  vetinari run       -- Execute a goal or manifest task
  vetinari serve     -- Start the web dashboard
  vetinari start     -- Start CLI + optional web dashboard (recommended)
  vetinari status    -- Show system status (models, agents, context)
  vetinari health    -- Health check all providers
  vetinari upgrade   -- Check for model upgrades
  vetinari review    -- Run the self-improvement agent
  vetinari interactive -- Enter REPL mode

Global flags:
  --config PATH     Manifest file (default: manifest/vetinari.yaml)
  --host URL        LM Studio host URL
  --mode MODE       Execution mode: planning|execution|sandbox
  --verbose         Enable debug logging
"""

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# Shared helpers
# ============================================================

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(config_path: str) -> dict:
    import yaml
    p = Path(config_path)
    if not p.exists():
        # Try relative to package directory
        pkg_root = Path(__file__).resolve().parents[1]
        p = pkg_root / config_path
    if not p.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {"project_name": "vetinari", "tasks": []}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_host(args_host: Optional[str]) -> str:
    """Resolve host from args → env → default."""
    return args_host or os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")


def _build_orchestrator(config_path: str, host: str, mode: str = "execution"):
    from vetinari.orchestrator import Orchestrator
    orch = Orchestrator(config_path, host=host, execution_mode=mode)
    return orch


def _print_banner(mode: str, host: str) -> None:
    print("=" * 60)
    print(" VETINARI AI Orchestration System")
    print(f" Mode: {mode.upper()}  |  Host: {host}")
    print("=" * 60)


# ============================================================
# Subcommand handlers
# ============================================================

def cmd_run(args) -> int:
    """Execute a goal or manifest task."""
    _setup_logging(args.verbose)
    host = _get_host(args.host)

    if args.goal:
        # High-level goal → assembly-line pipeline
        print(f"[Vetinari] Running goal: {args.goal[:80]}")
        try:
            from vetinari.two_layer_orchestration import get_two_layer_orchestrator
            orch = get_two_layer_orchestrator()
            # Wire agent context if orchestrator is available
            try:
                base_orch = _build_orchestrator(args.config, host, args.mode)
                orch.set_agent_context(base_orch._agent_context)
            except Exception:
                pass
            results = orch.generate_and_execute(
                goal=args.goal,
                constraints={"mode": args.mode},
            )
            print(f"\n[Vetinari] Completed: {results.get('completed', 0)} tasks")
            print(f"[Vetinari] Failed:    {results.get('failed', 0)} tasks")
            if results.get("final_output"):
                print("\n--- Final Output ---")
                print(str(results["final_output"])[:2000])
            return 0
        except Exception as e:
            print(f"[Vetinari] Error: {e}")
            logger.exception("Goal execution failed")
            return 1

    # Manifest-based task execution
    try:
        orch = _build_orchestrator(args.config, host, args.mode)
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


def cmd_serve(args) -> int:
    """Start the web dashboard."""
    _setup_logging(args.verbose)
    host = _get_host(args.host)
    port = args.port or int(os.environ.get("VETINARI_WEB_PORT", "5000"))
    web_host = args.web_host or "0.0.0.0"

    print(f"[Vetinari] Starting web dashboard on {web_host}:{port}")
    print(f"[Vetinari] Dashboard URL: http://localhost:{port}")

    try:
        from vetinari.web_ui import app
        # Set global orchestrator config
        app.config["VETINARI_HOST"] = host
        app.run(host=web_host, port=port, debug=args.debug, use_reloader=False)
        return 0
    except ImportError as e:
        print(f"[Vetinari] Web UI not available: {e}")
        print("[Vetinari] Install Flask: pip install flask")
        return 1
    except Exception as e:
        print(f"[Vetinari] Dashboard error: {e}")
        return 1


def cmd_start(args) -> int:
    """Start CLI + optional web dashboard."""
    _setup_logging(args.verbose)
    host = _get_host(args.host)
    _print_banner(args.mode, host)

    # Start web dashboard in background thread
    port = args.port or int(os.environ.get("VETINARI_WEB_PORT", "5000"))
    dashboard_started = False

    if not args.no_dashboard:
        try:
            from vetinari.web_ui import app
            app.config["VETINARI_HOST"] = host

            def _run_dashboard():
                app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

            t = threading.Thread(target=_run_dashboard, daemon=True)
            t.start()
            time.sleep(1)  # Give Flask time to start
            print(f"[Vetinari] Dashboard started: http://localhost:{port}")
            dashboard_started = True
        except Exception as e:
            print(f"[Vetinari] Dashboard unavailable: {e}")

    # Run health check
    print("\n[Vetinari] Running startup health checks...")
    _health_check_quiet(host)

    # Start AutoTuner background cycle (every 15 minutes while running)
    def _auto_tuner_loop():
        import time as _time
        while True:
            _time.sleep(900)  # 15 minutes
            try:
                from vetinari.learning.auto_tuner import get_auto_tuner
                get_auto_tuner().run_cycle()
                logger.debug("[AutoTuner] Periodic cycle complete")
            except Exception as _at_err:
                logger.debug(f"[AutoTuner] Cycle error (non-fatal): {_at_err}")

    _tuner_thread = threading.Thread(target=_auto_tuner_loop, daemon=True, name="auto-tuner")
    _tuner_thread.start()

    # Execute goal if provided
    if args.goal:
        return cmd_run(args)

    if args.task:
        return cmd_run(args)

    # Enter interactive REPL if no task specified
    if dashboard_started:
        print(f"\n[Vetinari] Dashboard running at http://localhost:{port}")
        print("[Vetinari] Press Ctrl+C to exit")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Vetinari] Shutting down...")
    else:
        return cmd_interactive(args)

    return 0


def cmd_status(args) -> int:
    """Show system status."""
    _setup_logging(args.verbose)
    host = _get_host(args.host)

    print(f"\n[Vetinari] System Status")
    print(f"  LM Studio Host: {host}")
    print(f"  Config:         {args.config}")

    # Check LM Studio connection
    try:
        from vetinari.lmstudio_adapter import LMStudioAdapter
        adapter = LMStudioAdapter(host=host)
        models = adapter._get(f"{host}/v1/models")
        model_list = models.get("data", []) if isinstance(models, dict) else []
        print(f"  Models loaded:  {len(model_list)}")
        for m in model_list[:5]:
            mid = m.get("id", m) if isinstance(m, dict) else str(m)
            print(f"    - {mid}")
    except Exception as e:
        print(f"  LM Studio:      UNREACHABLE ({e})")

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


def cmd_health(args) -> int:
    """Run health checks on all providers."""
    _setup_logging(args.verbose)
    host = _get_host(args.host)

    print("[Vetinari] Running health checks...")
    _health_check_quiet(host)
    return 0


def _health_check_quiet(host: str) -> None:
    """Run health checks and print results."""
    try:
        from vetinari.lmstudio_adapter import LMStudioAdapter
        adapter = LMStudioAdapter(host=host)
        result = adapter._get(f"{host}/v1/models")
        models = result.get("data", []) if isinstance(result, dict) else []
        print(f"  LM Studio:   OK ({len(models)} models)")
    except Exception as e:
        print(f"  LM Studio:   FAIL ({e})")

    try:
        from vetinari.adapter_manager import get_adapter_manager
        mgr = get_adapter_manager()
        results = mgr.health_check()
        for name, info in results.items():
            status = "OK" if info.get("healthy") else "FAIL"
            print(f"  {name:20s}: {status}")
    except Exception:
        pass


def cmd_upgrade(args) -> int:
    """Check for model upgrades."""
    _setup_logging(args.verbose)
    host = _get_host(args.host)

    try:
        orch = _build_orchestrator(args.config, host, args.mode)
        orch.check_and_upgrade_models()
        return 0
    except Exception as e:
        print(f"[Vetinari] Upgrade check failed: {e}")
        return 1


def cmd_review(args) -> int:
    """Run the self-improvement agent."""
    _setup_logging(args.verbose)

    print("[Vetinari] Running self-improvement review...")
    try:
        from vetinari.agents.improvement_agent import get_improvement_agent
        from vetinari.agents.contracts import AgentTask, AgentType

        agent = get_improvement_agent()
        try:
            from vetinari.adapter_manager import get_adapter_manager
            agent.initialize({"adapter_manager": get_adapter_manager()})
        except Exception:
            pass

        task = AgentTask(
            task_id="review-cli",
            agent_type=AgentType.IMPROVEMENT,
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
    except Exception as e:
        print(f"[Vetinari] Review failed: {e}")
        return 1


def cmd_train(args) -> int:
    """Manage training data and fine-tuning jobs."""
    _setup_logging(args.verbose)

    from vetinari.learning.training_manager import get_training_manager
    manager = get_training_manager()

    # --status: list all jobs
    if args.status:
        jobs = manager.list_jobs()
        if not jobs:
            print("[Vetinari Train] No training jobs found.")
            return 0
        print(f"[Vetinari Train] {len(jobs)} job(s):")
        for job in jobs:
            result_info = ""
            if job.result:
                result_info = f" | loss={job.result.metrics.get('loss', '?')}"
            print(f"  {job.job_id}  [{job.status}]  {job.provider}/{job.model_id}{result_info}")
        return 0

    # --stats: training data statistics
    if args.stats:
        from vetinari.learning.training_data import get_training_collector
        stats = get_training_collector().get_stats()
        print("[Vetinari Train] Training data statistics:")
        print(f"  Total records  : {stats.get('total', 0)}")
        print(f"  SFT eligible   : {stats.get('sft_eligible', 0)}")
        print(f"  Average score  : {stats.get('avg_score', 0.0)}")
        print(f"  Output path    : {stats.get('output_path', 'N/A')}")
        by_type = stats.get("by_task_type", {})
        if by_type:
            print("  By task type:")
            for tt, info in by_type.items():
                print(f"    {tt}: {info.get('count', 0)} records, avg {info.get('avg_score', 0.0)}")
        return 0

    # --export: export dataset
    if args.export:
        fmt = args.export
        min_score = args.min_score
        dataset = manager.prepare_training_data(min_score=min_score, format=fmt)
        print(f"[Vetinari Train] Exported {dataset.stats['count']} records in '{fmt}' format")
        if dataset.records:
            import json
            print(json.dumps(dataset.records[0], indent=2, ensure_ascii=False))
            if len(dataset.records) > 1:
                print(f"  ... ({len(dataset.records) - 1} more records)")
        return 0

    # --model: start a training run
    if args.model:
        method = args.method or "qlora"
        min_score = args.min_score
        dataset = manager.prepare_training_data(min_score=min_score, format="sft")
        print(f"[Vetinari Train] Prepared {dataset.stats['count']} SFT records")
        print(f"[Vetinari Train] Starting local training: {args.model} ({method})")
        result = manager.train_local(args.model, dataset, method=method)
        if result.success:
            print(f"[Vetinari Train] Training complete: {result.model_path}")
            print(f"  Duration: {result.duration_seconds}s")
        else:
            print(f"[Vetinari Train] Training failed: {result.error}")
            return 1
        return 0

    # No flags — show help
    print("[Vetinari Train] Usage:")
    print("  vetinari train --model <model_id> --method qlora --min-score 0.85")
    print("  vetinari train --status")
    print("  vetinari train --export hf")
    print("  vetinari train --export sft")
    print("  vetinari train --stats")
    return 0


def cmd_interactive(args) -> int:
    """Enter interactive REPL mode."""
    _setup_logging(args.verbose)
    host = _get_host(args.host)

    print("[Vetinari] Interactive mode. Type your goal and press Enter.")
    print("Commands: /quit, /status, /review, /help")
    print("-" * 50)

    try:
        from vetinari.two_layer_orchestration import get_two_layer_orchestrator
        orch = get_two_layer_orchestrator()
        try:
            base_orch = _build_orchestrator(args.config, host, args.mode)
            orch.set_agent_context(base_orch._agent_context)
        except Exception:
            pass
    except Exception:
        orch = None

    while True:
        try:
            goal = input("\nGoal> ").strip()
        except (EOFError, KeyboardInterrupt):
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
                print("[Vetinari] Orchestrator not available. Check LM Studio connection.")
        except Exception as e:
            print(f"[Vetinari] Error: {e}")
            logger.debug("Interactive execution error", exc_info=True)


def cmd_benchmark(args) -> int:
    """Run benchmark suites."""
    from vetinari.benchmarks.runner import get_default_runner
    import tempfile, os

    db_path = os.path.join(tempfile.gettempdir(), "vetinari_benchmarks.db")
    runner = get_default_runner(db_path=db_path)

    action = getattr(args, "action", "list")

    if action == "list":
        suites = runner.list_suites()
        if not suites:
            print("No benchmark suites registered.")
            return 0
        print(f"{'Name':<20} {'Layer':<15} {'Tier':<10} {'Description'}")
        print("-" * 70)
        for s in suites:
            print(f"{s['name']:<20} {s['layer']:<15} {s['tier']:<10} {s['description']}")
        return 0

    elif action == "run":
        suite_name = getattr(args, "suite", None)
        if not suite_name:
            print("Usage: vetinari benchmark run <suite> [--limit N] [--trials K]")
            return 1
        limit = getattr(args, "limit", None)
        trials = getattr(args, "trials", 1)
        print(f"Running benchmark: {suite_name} (limit={limit}, trials={trials})")
        try:
            report = runner.run_suite(suite_name, limit=limit, trials=trials)
            summary = report.summary_dict()
            print(f"\nResults for {suite_name}:")
            print(f"  Run ID:     {summary['run_id']}")
            print(f"  Total:      {summary['total']}")
            print(f"  Passed:     {summary['passed']}")
            print(f"  pass@1:     {summary['pass@1']:.2%}")
            print(f"  pass^k:     {summary['pass^k']:.2%}")
            print(f"  Avg score:  {summary['avg_score']:.4f}")
            print(f"  Avg latency: {summary['avg_latency_ms']:.1f}ms")
            print(f"  Tokens:     {summary['total_tokens']}")
            return 0
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    elif action == "report":
        compare = getattr(args, "compare", None)
        suite_name = getattr(args, "suite", None)
        if compare == "last-2" and suite_name:
            comp = runner.get_last_comparison(suite_name)
            if comp is None:
                print(f"Need at least 2 runs of '{suite_name}' to compare.")
                return 1
            print(f"Comparison: {comp.run_a} vs {comp.run_b}")
            print(f"  Delta pass@1:   {comp.delta_pass_at_1:+.4f}")
            print(f"  Delta score:    {comp.delta_avg_score:+.4f}")
            print(f"  Delta latency:  {comp.delta_avg_latency_ms:+.1f}ms")
            if comp.regressions:
                print(f"  Regressions:    {', '.join(comp.regressions)}")
            if comp.improvements:
                print(f"  Improvements:   {', '.join(comp.improvements)}")
        else:
            runs = runner.list_runs(suite_name=suite_name, limit=10)
            if not runs:
                print("No benchmark runs found.")
                return 0
            print(f"{'Run ID':<30} {'Suite':<20} {'Pass@1':<10} {'Score':<10}")
            print("-" * 70)
            for r in runs:
                print(f"{r['run_id']:<30} {r['suite_name']:<20} {r.get('pass_at_1', 0):<10.4f} {r.get('avg_score', 0):<10.4f}")
        return 0

    return 0


# ============================================================
# Main entry point
# ============================================================

def main():
    """Main CLI entry point."""
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
    parser.add_argument("--config", default="manifest/vetinari.yaml",
                        help="Path to manifest file")
    parser.add_argument("--host", default=None,
                        help="LM Studio host URL (overrides LM_STUDIO_HOST env var)")
    parser.add_argument("--mode", default="execution",
                        choices=["planning", "execution", "sandbox"],
                        help="Execution mode")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run
    p_run = subparsers.add_parser("run", help="Execute a goal or manifest task")
    p_run.add_argument("--goal", "-g", help="High-level goal string")
    p_run.add_argument("--task", "-t", help="Specific task ID from manifest")

    # serve
    p_serve = subparsers.add_parser("serve", help="Start the web dashboard")
    p_serve.add_argument("--port", type=int, default=None, help="Web server port (default 5000)")
    p_serve.add_argument("--web-host", default="0.0.0.0", help="Web server bind address")
    p_serve.add_argument("--debug", action="store_true", help="Enable Flask debug mode")

    # start
    p_start = subparsers.add_parser("start", help="Start Vetinari (CLI + optional dashboard)")
    p_start.add_argument("--goal", "-g", help="Execute this goal on startup")
    p_start.add_argument("--task", "-t", help="Execute this task on startup")
    p_start.add_argument("--port", type=int, default=None, help="Dashboard port")
    p_start.add_argument("--no-dashboard", action="store_true", help="Disable web dashboard")

    # status
    subparsers.add_parser("status", help="Show system status")

    # health
    subparsers.add_parser("health", help="Health check all providers")

    # upgrade
    subparsers.add_parser("upgrade", help="Check for model upgrades")

    # review
    subparsers.add_parser("review", help="Run self-improvement agent review")

    # interactive
    subparsers.add_parser("interactive", help="Enter interactive REPL mode")

    # train
    p_train = subparsers.add_parser("train", help="Manage training data and fine-tuning jobs")
    p_train.add_argument("--model", default=None, help="Model ID to fine-tune")
    p_train.add_argument("--method", default="qlora", choices=["qlora", "full"],
                         help="Training method (default: qlora)")
    p_train.add_argument("--min-score", dest="min_score", type=float, default=0.85,
                         help="Minimum quality score for dataset export (default: 0.85)")
    p_train.add_argument("--status", action="store_true", help="Show training jobs")
    p_train.add_argument("--export", default=None, choices=["hf", "sft", "dpo", "ranking"],
                         help="Export dataset in the given format")
    p_train.add_argument("--stats", action="store_true", help="Show training data statistics")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run benchmark suites")
    p_bench.add_argument("action", nargs="?", default="list",
                         choices=["run", "list", "report"],
                         help="Benchmark action (default: list)")
    p_bench.add_argument("suite", nargs="?", default=None,
                         help="Benchmark suite name (for run/report)")
    p_bench.add_argument("--limit", type=int, default=None,
                         help="Max cases to run")
    p_bench.add_argument("--trials", type=int, default=1,
                         help="Trials per case for pass^k (default: 1)")
    p_bench.add_argument("--compare", default=None,
                         help="Compare mode: 'last-2' or 'RUN_A:RUN_B'")

    args = parser.parse_args()

    # Default command: start (interactive)
    if args.command is None:
        args.command = "start"
        # Set safe defaults for any attributes subparsers would normally define
        if not hasattr(args, "goal"):
            args.goal = None
        if not hasattr(args, "task"):
            args.task = None
        if not hasattr(args, "port"):
            args.port = None
        if not hasattr(args, "no_dashboard"):
            args.no_dashboard = False

    dispatch = {
        "run": cmd_run,
        "serve": cmd_serve,
        "start": cmd_start,
        "status": cmd_status,
        "health": cmd_health,
        "upgrade": cmd_upgrade,
        "review": cmd_review,
        "interactive": cmd_interactive,
        "train": cmd_train,
        "benchmark": cmd_benchmark,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
