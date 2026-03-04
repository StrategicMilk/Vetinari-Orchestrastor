#!/usr/bin/env python
"""
Example: Using the Explorer Skill Tool

This script demonstrates how to use the migrated explorer skill through the
Tool interface. It shows:

1. Basic initialization
2. Different search capabilities
3. Advanced search strategies
4. Execution mode differences
5. Real-world use cases

To run this example:
    python examples/explorer_skill_example.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vetinari.tools.explorer_skill import ExplorerSkillTool, ExplorerCapability, ThinkingMode, SearchStrategy
from vetinari.execution_context import ExecutionMode, ExecutionContext, get_context_manager


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def print_result(result) -> None:
    """Print tool execution result."""
    if result.success:
        print("[OK] Success")
    else:
        print("[FAIL] Failed")
    
    if result.output:
        if isinstance(result.output, dict):
            query = result.output.get("query", "")
            capability = result.output.get("capability", "")
            total_found = result.output.get("total_found", 0)
            files_searched = result.output.get("files_searched", 0)
            project_type = result.output.get("project_type")
            
            print(f"\nQuery: {query}")
            print(f"Capability: {capability}")
            print(f"Results found: {total_found}")
            print(f"Files searched: {files_searched}")
            if project_type:
                print(f"Project type: {project_type}")
            
            results = result.output.get("results", [])
            if results:
                print(f"\nTop results:")
                for res in results[:3]:
                    print(f"  - {res.get('file_path')}:{res.get('line_number')}")
    
    if result.error:
        print(f"Error: {result.error}")


def example_metadata() -> None:
    """Example: Inspecting tool metadata."""
    print_section("Example 1: Tool Metadata")
    
    tool = ExplorerSkillTool()
    
    print(f"Tool Name: {tool.metadata.name}")
    print(f"Description: {tool.metadata.description}")
    print(f"Version: {tool.metadata.version}")
    print(f"Category: {tool.metadata.category.value}")
    
    print(f"\nCapabilities:")
    for cap in ExplorerCapability:
        print(f"  - {cap.value}")
    
    print(f"\nThinking Modes:")
    for mode in ThinkingMode:
        print(f"  - {mode.value}")
    
    print(f"\nSearch Strategies:")
    for strat in SearchStrategy:
        print(f"  - {strat.value}")


def example_grep_search() -> None:
    """Example 1: Grep-based text search."""
    print_section("Example 2: Grep Search")
    
    tool = ExplorerSkillTool()
    
    print("\n2a. Quick grep search (low thinking mode):")
    result = tool.run(
        capability=ExplorerCapability.GREP_SEARCH.value,
        query="def authenticate",
        thinking_mode=ThinkingMode.LOW.value,
        search_strategy=SearchStrategy.PARTIAL.value,
    )
    print_result(result)
    
    print("\n2b. Regex pattern search:")
    result = tool.run(
        capability=ExplorerCapability.GREP_SEARCH.value,
        query=r"^\s*async\s+def\s+\w+",
        search_strategy=SearchStrategy.REGEX.value,
        file_extensions=[".py"],
    )
    print_result(result)
    
    print("\n2c. Exact match search:")
    result = tool.run(
        capability=ExplorerCapability.GREP_SEARCH.value,
        query="SECRET_KEY",
        search_strategy=SearchStrategy.EXACT.value,
        max_results=10,
    )
    print_result(result)


def example_file_discovery() -> None:
    """Example 2: File discovery."""
    print_section("Example 3: File Discovery")
    
    tool = ExplorerSkillTool()
    
    print("\n3a. Find Python files:")
    result = tool.run(
        capability=ExplorerCapability.FILE_DISCOVERY.value,
        query="**/*.py",
        file_extensions=[".py"],
    )
    print_result(result)
    
    print("\n3b. Find configuration files:")
    result = tool.run(
        capability=ExplorerCapability.FILE_DISCOVERY.value,
        query="config/**/*",
    )
    print_result(result)


def example_pattern_matching() -> None:
    """Example 3: Pattern matching."""
    print_section("Example 4: Pattern Matching")
    
    tool = ExplorerSkillTool()
    
    print("\n4a. Find class definitions:")
    result = tool.run(
        capability=ExplorerCapability.PATTERN_MATCHING.value,
        query=r"^class\s+\w+",
        search_strategy=SearchStrategy.REGEX.value,
        thinking_mode=ThinkingMode.MEDIUM.value,
    )
    print_result(result)
    
    print("\n4b. Find error handling:")
    result = tool.run(
        capability=ExplorerCapability.PATTERN_MATCHING.value,
        query="except.*Error",
        search_strategy=SearchStrategy.REGEX.value,
        context_lines=3,
    )
    print_result(result)


def example_symbol_lookup() -> None:
    """Example 4: Symbol lookup."""
    print_section("Example 5: Symbol Lookup")
    
    tool = ExplorerSkillTool()
    
    print("\n5a. Find function definition:")
    result = tool.run(
        capability=ExplorerCapability.SYMBOL_LOOKUP.value,
        query="create_user",
        thinking_mode=ThinkingMode.MEDIUM.value,
    )
    print_result(result)
    
    print("\n5b. Find class with deep analysis:")
    result = tool.run(
        capability=ExplorerCapability.SYMBOL_LOOKUP.value,
        query="DatabaseConnection",
        thinking_mode=ThinkingMode.HIGH.value,
    )
    print_result(result)


def example_import_analysis() -> None:
    """Example 5: Import analysis."""
    print_section("Example 6: Import Analysis")
    
    tool = ExplorerSkillTool()
    
    print("\n6a. Analyze imports in file:")
    result = tool.run(
        capability=ExplorerCapability.IMPORT_ANALYSIS.value,
        query="src/services/auth.py",
    )
    print_result(result)
    
    print("\n6b. Trace dependencies for module:")
    result = tool.run(
        capability=ExplorerCapability.IMPORT_ANALYSIS.value,
        query="utils/database",
        thinking_mode=ThinkingMode.HIGH.value,
    )
    print_result(result)


def example_project_mapping() -> None:
    """Example 6: Project mapping."""
    print_section("Example 7: Project Mapping")
    
    tool = ExplorerSkillTool()
    
    print("\n7a. Quick project scan:")
    result = tool.run(
        capability=ExplorerCapability.PROJECT_MAPPING.value,
        query="overall structure",
        thinking_mode=ThinkingMode.LOW.value,
    )
    print_result(result)
    
    print("\n7b. Comprehensive project map:")
    result = tool.run(
        capability=ExplorerCapability.PROJECT_MAPPING.value,
        query="authentication feature",
        thinking_mode=ThinkingMode.MEDIUM.value,
    )
    print_result(result)
    
    print("\n7c. Deep architecture analysis:")
    result = tool.run(
        capability=ExplorerCapability.PROJECT_MAPPING.value,
        query="payment processing system",
        thinking_mode=ThinkingMode.HIGH.value,
    )
    print_result(result)


def example_advanced_search() -> None:
    """Example 7: Advanced search scenarios."""
    print_section("Example 8: Advanced Searches")
    
    tool = ExplorerSkillTool()
    
    print("\n8a. Find security-sensitive code:")
    result = tool.run(
        capability=ExplorerCapability.GREP_SEARCH.value,
        query="password|token|secret|apikey",
        search_strategy=SearchStrategy.REGEX.value,
        max_results=50,
    )
    print_result(result)
    
    print("\n8b. Find TODO/FIXME comments:")
    result = tool.run(
        capability=ExplorerCapability.GREP_SEARCH.value,
        query="(TODO|FIXME|BUG|HACK)",
        search_strategy=SearchStrategy.REGEX.value,
        context_lines=1,
    )
    print_result(result)
    
    print("\n8c. Find logging statements:")
    result = tool.run(
        capability=ExplorerCapability.GREP_SEARCH.value,
        query="logger\\.|(log\\.|console\\.log|print\\()",
        search_strategy=SearchStrategy.REGEX.value,
        file_extensions=[".py", ".js", ".ts"],
    )
    print_result(result)


def example_execution_modes() -> None:
    """Example 8: Execution modes."""
    print_section("Example 9: Execution Modes")
    
    tool = ExplorerSkillTool()
    context_manager = get_context_manager()
    
    print(f"\nCurrent mode: {context_manager.current_mode.value}")
    
    print("\n9a. EXECUTION mode search:")
    context_manager.switch_mode(ExecutionMode.EXECUTION)
    result = tool.run(
        capability=ExplorerCapability.GREP_SEARCH.value,
        query="import",
    )
    print_result(result)
    
    print("\n9b. PLANNING mode search:")
    context_manager.switch_mode(ExecutionMode.PLANNING)
    result = tool.run(
        capability=ExplorerCapability.PROJECT_MAPPING.value,
        query="architecture",
    )
    print_result(result)


def main() -> None:
    """Run all examples."""
    print("\n" + "="*70)
    print("  Explorer Skill Tool - Examples")
    print("="*70)
    
    try:
        # Set up proper execution context
        context_manager = get_context_manager()
        context_manager.switch_mode(ExecutionMode.EXECUTION)
        
        # Run all examples
        example_metadata()
        example_grep_search()
        example_file_discovery()
        example_pattern_matching()
        example_symbol_lookup()
        example_import_analysis()
        example_project_mapping()
        example_advanced_search()
        example_execution_modes()
        
        print_section("Examples Complete")
        print("\nAll examples completed successfully!")
        print("\nKey Takeaways:")
        print("  1. ExplorerSkillTool provides 6 major capabilities")
        print("  2. Thinking modes control search depth and comprehensiveness")
        print("  3. Search strategies allow exact, regex, or partial matching")
        print("  4. File extensions help narrow search scope")
        print("  5. Context lines provide surrounding code")
        print("  6. Thinking modes affect what gets returned")
        
    except Exception as e:
        print(f"\n[ERROR] Error running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
