#!/usr/bin/env python
"""Example: Using the Librarian Skill Tool"""

from vetinari.tools.librarian_skill import LibrarianSkillTool, LibrarianCapability, ThinkingMode
from vetinari.execution_context import ExecutionMode
from unittest.mock import Mock


def setup_tool():
    tool = LibrarianSkillTool()
    mock_ctx = Mock()
    mock_ctx.mode = ExecutionMode.EXECUTION
    tool._context_manager = Mock(current_context=mock_ctx)
    return tool


def main():
    print("=== Librarian Skill Tool Examples ===\n")
    tool = setup_tool()

    # Example 1: Documentation Lookup
    print("1. Documentation Lookup:")
    result = tool.execute(capability="docs_lookup", query="React Query", thinking_mode="medium")
    print(f"   Success: {result.success}")
    print(f"   Summary: {result.output.get('summary', 'N/A')[:60]}...")

    # Example 2: GitHub Examples
    print("\n2. GitHub Examples:")
    result = tool.execute(capability="github_examples", query="Python async", thinking_mode="high")
    print(f"   Success: {result.success}")
    print(f"   Code Example: {result.output.get('code_example', 'N/A')[:40]}...")

    # Example 3: Best Practices
    print("\n3. Best Practices:")
    result = tool.execute(capability="best_practices", query="API security", focus_areas=["auth"])
    print(f"   Success: {result.success}")
    print(f"   Practices: {result.output.get('best_practices', [])}")

    # Example 4: Package Info
    print("\n4. Package Info:")
    result = tool.execute(capability="package_info", query="requests")
    print(f"   Success: {result.success}")
    print(f"   URL: {result.output.get('documentation_url', 'N/A')}")

    print("\n=== All examples completed! ===")


if __name__ == "__main__":
    main()
