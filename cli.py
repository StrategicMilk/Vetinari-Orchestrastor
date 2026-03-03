import os
import sys
import json
import yaml
import argparse
from pathlib import Path
import logging

from vetinari.orchestrator import Orchestrator
from vetinari.utils import setup_logging, load_config
from vetinari.execution_context import (
    get_context_manager,
    ExecutionMode,
    ToolPermission,
)
from vetinari.adapter_manager import get_adapter_manager
from vetinari.tool_interface import get_tool_registry

logger = logging.getLogger(__name__)


def print_mode_banner(mode: ExecutionMode):
    """Print a prominent banner indicating the current execution mode."""
    mode_display = {
        ExecutionMode.PLANNING: "📋 PLANNING MODE (Read-Only)",
        ExecutionMode.EXECUTION: "⚙️  EXECUTION MODE (Full Access)",
        ExecutionMode.SANDBOX: "🔒 SANDBOX MODE (Restricted)",
    }
    
    print("\n" + "=" * 60)
    print(mode_display.get(mode, mode.value))
    print("=" * 60 + "\n")


def show_context_status(context_manager):
    """Display the current execution context status."""
    status = context_manager.get_status()
    print(f"\n📊 Execution Context:")
    print(f"   Mode: {status['mode']}")
    print(f"   Task: {status['task_id'] or 'None'}")
    print(f"   Operations: {status['operations_count']}")
    if status['permissions']:
        print(f"   Permissions: {', '.join(status['permissions'][:3])}")
        if len(status['permissions']) > 3:
            print(f"                ... and {len(status['permissions']) - 3} more")


def show_provider_status(adapter_manager):
    """Display the status of available providers."""
    status = adapter_manager.get_status()
    providers = status.get("providers", {})
    
    if not providers:
        print("\n⚠️  No providers registered")
        return
    
    print(f"\n🔌 Provider Status ({len(providers)} provider(s)):")
    for name, info in providers.items():
        metrics = info.get("metrics", {})
        health = metrics.get("health_status", "unknown")
        icon = "✅" if health == "healthy" else "⚠️ " if health == "degraded" else "❌"
        
        print(f"   {icon} {name}")
        if metrics:
            print(f"      Health: {health}")
            print(f"      Success Rate: {metrics.get('success_rate', 0):.1%}")
            print(f"      Avg Latency: {metrics.get('avg_latency_ms', 0):.0f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="Vetinari: Comprehensive LLM Orchestration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in planning mode (read-only analysis)
  vetinari --mode planning --task t1
  
  # Run in execution mode (full access)
  vetinari --mode execution --task t1
  
  # Show provider status
  vetinari --providers
  
  # Check execution context
  vetinari --context
        """,
    )
    
    # Core arguments
    parser.add_argument(
        "--config",
        default="manifest/vetinari.yaml",
        help="Path to the manifest file",
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["planning", "execution", "sandbox"],
        default="execution",
        help="Execution mode: planning (read-only), execution (full access), or sandbox (restricted)",
    )
    
    # Task selection
    parser.add_argument(
        "--task",
        help="Run a specific task by ID",
    )
    
    # Provider management
    parser.add_argument(
        "--providers",
        action="store_true",
        help="Show provider status and available models",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run health check on all providers",
    )
    
    # Context and status
    parser.add_argument(
        "--context",
        action="store_true",
        help="Show current execution context status",
    )
    
    # Upgrade
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Check for and install model upgrades",
    )
    
    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file {config_path} not found")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Initialize context manager with selected mode
    context_manager = get_context_manager()
    execution_mode = ExecutionMode(args.mode)
    context_manager.switch_mode(execution_mode)
    
    # Print mode banner
    print_mode_banner(execution_mode)
    
    # Initialize orchestrator
    orchestrator = Orchestrator(str(config_path))
    
    # Get adapter manager
    adapter_manager = get_adapter_manager()
    
    # Handle various commands
    if args.context:
        show_context_status(context_manager)
        sys.exit(0)
    
    if args.providers:
        show_provider_status(adapter_manager)
        sys.exit(0)
    
    if args.health_check:
        print("\n🔍 Running health checks...")
        health_results = adapter_manager.health_check()
        for provider, status in health_results.items():
            health_ok = status.get("healthy", False)
            icon = "✅" if health_ok else "❌"
            print(f"   {icon} {provider}: {status.get('reason', 'OK')}")
        sys.exit(0)
    
    if args.upgrade:
        if execution_mode == ExecutionMode.PLANNING:
            print("⚠️  Cannot perform upgrades in planning mode")
            sys.exit(1)
        orchestrator.check_and_upgrade_models()
    elif args.task:
        print(f"▶️  Running task: {args.task}")
        orchestrator.run_task(args.task)
    else:
        print("▶️  Running full workflow")
        orchestrator.run_all()
    
    # Show final context status
    show_context_status(context_manager)
    print("\n✅ Workflow completed\n")


if __name__ == "__main__":
    main()