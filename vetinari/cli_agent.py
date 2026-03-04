#!/usr/bin/env python
"""
Enhanced CLI for Vetinari - Comprehensive AI Orchestration System

This CLI provides access to all Vetinari capabilities:
- Goal-based execution with planning
- Web search and information gathering
- Model selection and routing
- Memory and context management
- Code execution
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Optional

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from vetinari.utils import setup_logging, load_config
from vetinari.execution_context import ExecutionMode, get_context_manager
from vetinari.integrated_agent import get_agent, AgentRequest, IntegratedAgent
from vetinari.dynamic_model_router import TaskType
from vetinari.two_layer_orchestration import get_two_layer_orchestrator
from vetinari.enhanced_memory import get_memory_manager
from vetinari.tools.web_search_tool import get_search_tool
from vetinari.code_sandbox import get_code_executor


def setup_args(parser: argparse.ArgumentParser):
    """Setup command line arguments."""
    parser.add_argument(
        "--goal", "-g",
        type=str,
        help="Goal to achieve"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["planning", "execution", "sandbox"],
        default="execution",
        help="Execution mode"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--config", "-c",
        default="manifest/vetinari.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available tools"
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search the web for information"
    )
    parser.add_argument(
        "--research",
        type=str,
        help="Research a topic comprehensively"
    )
    parser.add_argument(
        "--execute-code",
        type=str,
        help="Execute Python code"
    )
    parser.add_argument(
        "--memory-recall",
        type=str,
        help="Recall from memory"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show system status"
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Generate plan without executing"
    )


def list_models(router):
    """List available models."""
    print("\n=== Available Models ===")
    models = router.get_available_models()
    
    if not models:
        print("No models available")
        return
    
    for m in models:
        print(f"\n{m.id}")
        print(f"  Provider: {m.provider.value}")
        print(f"  Capabilities: {', '.join(m.capabilities.tags) if hasattr(m.capabilities, 'tags') else 'N/A'}")
        print(f"  Context: {m.context_length} tokens")
        print(f"  Memory: {m.memory_gb} GB")


def list_tools(registry):
    """List available tools."""
    print("\n=== Available Tools ===")
    tools = registry.list_tools()
    
    for tool in tools:
        print(f"\n{tool.metadata.name}")
        print(f"  {tool.metadata.description}")
        print(f"  Category: {tool.metadata.category.value}")
        print(f"  Mode: {', '.join(m.value for m in tool.metadata.allowed_modes)}")


def handle_search(query: str):
    """Handle web search."""
    print(f"\n=== Searching: {query} ===")
    tool = get_search_tool()
    result = tool.search(query, max_results=5)
    
    print(f"\nFound {result.total_results} results in {result.execution_time_ms}ms\n")
    
    for i, r in enumerate(result.results, 1):
        print(f"{i}. {r.title}")
        print(f"   {r.url}")
        print(f"   {r.snippet[:150]}...")
        print(f"   Reliability: {r.source_reliability:.2f}")
        print()


def handle_research(topic: str):
    """Handle comprehensive research."""
    print(f"\n=== Researching: {topic} ===")
    tool = get_search_tool()
    result = tool.research_topic(topic)
    
    print(f"\nTotal sources: {result['total_sources']}")
    print(f"High credibility: {result['high_credibility_sources']}")
    print("\nCitations:")
    for cite in result['citations'][:10]:
        print(f"  {cite}")


def handle_execute_code(code: str):
    """Handle code execution."""
    print(f"\n=== Executing Code ===")
    executor = get_code_executor()
    result = executor.run(code, language="python")
    
    print(f"\nSuccess: {result['success']}")
    print(f"\nOutput:")
    print(result.get('output', ''))
    if result.get('error'):
        print(f"\nError:")
        print(result['error'])


def handle_memory_recall(query: str):
    """Handle memory recall."""
    print(f"\n=== Recalling from Memory: {query} ===")
    memory = get_memory_manager()
    results = memory.recall(query=query, limit=5)
    
    if not results:
        print("No results found")
        return
    
    for r in results:
        print(f"\n--- {r.memory_type.value} ---")
        print(r.content[:300])


def show_status(agent: IntegratedAgent):
    """Show system status."""
    print("\n=== Vetinari Status ===")
    status = agent.get_status()
    
    print(f"\nState: {status['state']}")
    if status.get('current_goal'):
        print(f"Current Goal: {status['current_goal']}")
    if status.get('plan_id'):
        print(f"Plan ID: {status['plan_id']}")
    
    print("\nComponents:")
    for comp, available in status['components'].items():
        icon = "✅" if available else "❌"
        print(f"  {icon} {comp}")
    
    # Memory stats
    try:
        memory = get_memory_manager()
        stats = memory.get_stats()
        print(f"\nMemory: {stats['total_entries']} entries")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Vetinari: Comprehensive AI Orchestration System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    setup_args(parser)
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to project root
        project_root = Path(__file__).parent
        config_path = project_root / args.config
    
    config = {}
    if config_path.exists():
        config = load_config(config_path)
    
    # Setup context
    context_manager = get_context_manager()
    execution_mode = ExecutionMode(args.mode)
    context_manager.switch_mode(execution_mode)
    
    # Print banner
    print("\n" + "=" * 60)
    print("Vetinari - Comprehensive AI Orchestration System")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print()
    
    # Handle different commands
    if args.list_models:
        from vetinari.dynamic_model_router import get_model_router
        router = get_model_router()
        list_models(router)
        return
    
    if args.list_tools:
        from vetinari.tool_interface import get_tool_registry
        registry = get_tool_registry()
        list_tools(registry)
        return
    
    if args.search:
        handle_search(args.search)
        return
    
    if args.research:
        handle_research(args.research)
        return
    
    if args.execute_code:
        handle_execute_code(args.execute_code)
        return
    
    if args.memory_recall:
        handle_memory_recall(args.memory_recall)
        return
    
    if args.status:
        agent = get_agent(config)
        show_status(agent)
        return
    
    if args.goal:
        # Run the agent with the goal
        print(f"\n=== Processing Goal: {args.goal} ===\n")
        
        agent = get_agent(config)
        
        request = AgentRequest(
            goal=args.goal,
            constraints={"plan_only": args.plan_only} if args.plan_only else {}
        )
        
        response = agent.execute(request)
        
        print(f"\nSuccess: {response.success}")
        print(f"State: {response.state.value}")
        
        if response.error:
            print(f"\nError: {response.error}")
        else:
            print(f"\n{response.output}")
            
            if response.citations:
                print("\nCitations:")
                for cite in response.citations[:5]:
                    print(f"  {cite}")
        
        print()
    else:
        # Show help
        parser.print_help()
        print("\n" + "=" * 60)
        print("Examples:")
        print("  python vetinari/cli_agent.py --goal 'Build a Python web app'")
        print("  python vetinari/cli_agent.py --search 'Python async tutorial'")
        print("  python vetinari/cli_agent.py --research 'LLM agents'")
        print("  python vetinari/cli_agent.py --list-models")
        print("  python vetinari/cli_agent.py --status")
        print("=" * 60)


if __name__ == "__main__":
    main()
