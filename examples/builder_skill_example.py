#!/usr/bin/env python
"""
Example: Using the Builder Skill Tool

This script demonstrates how to use the migrated builder skill through the
Tool interface. It shows:

1. Basic initialization
2. Feature implementation with different thinking modes
3. Refactoring existing code
4. Writing tests
5. Permission enforcement with execution modes
6. Error handling

To run this example:
    python examples/builder_skill_example.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vetinari.skills.builder import BuilderSkillTool, BuilderCapability, ThinkingMode
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
            explanation = result.output.get("explanation", "")
            if explanation:
                print(f"\nExplanation:\n{explanation}")
            
            files = result.output.get("files_affected", [])
            if files:
                print(f"\nFiles affected: {', '.join(files)}")
            
            tests = result.output.get("tests_added", 0)
            if tests > 0:
                print(f"Tests added: {tests}")
            
            warnings = result.output.get("warnings", [])
            if warnings:
                print(f"\nWarnings:")
                for w in warnings:
                    print(f"  - {w}")
    
    if result.error:
        print(f"Error: {result.error}")


def example_feature_implementation() -> None:
    """Example 1: Implementing a new feature."""
    print_section("Example 1: Feature Implementation")
    
    tool = BuilderSkillTool()
    
    # Quick implementation
    print("\n1a. Quick implementation (low thinking mode):")
    result = tool.run(
        capability=BuilderCapability.FEATURE_IMPLEMENTATION.value,
        description="Create a REST API endpoint to fetch user profile",
        thinking_mode=ThinkingMode.LOW.value,
    )
    print_result(result)
    
    # Full feature with tests
    print("\n1b. Full feature with tests (medium thinking mode):")
    result = tool.run(
        capability=BuilderCapability.FEATURE_IMPLEMENTATION.value,
        description="Create a user authentication system with JWT tokens",
        thinking_mode=ThinkingMode.MEDIUM.value,
        requirements=["async support", "password hashing", "refresh tokens"],
    )
    print_result(result)
    
    # Production-ready
    print("\n1c. Production-ready implementation (xhigh thinking mode):")
    result = tool.run(
        capability=BuilderCapability.FEATURE_IMPLEMENTATION.value,
        description="Implement a rate-limiting middleware",
        thinking_mode=ThinkingMode.XHIGH.value,
        requirements=["configurable limits", "distributed support", "metrics"],
    )
    print_result(result)


def example_refactoring() -> None:
    """Example 2: Refactoring existing code."""
    print_section("Example 2: Code Refactoring")
    
    tool = BuilderSkillTool()
    
    # Refactoring with context
    print("\n2a. Refactor for readability:")
    code_to_refactor = """
def process_data(d):
    result = []
    for i in d:
        if i > 0:
            result.append(i * 2)
    return result
"""
    
    result = tool.run(
        capability=BuilderCapability.REFACTORING.value,
        description="Improve readability and add type hints",
        context=code_to_refactor,
        thinking_mode=ThinkingMode.MEDIUM.value,
    )
    print_result(result)
    
    # Refactoring without context (should fail)
    print("\n2b. Refactoring without context (should show warning):")
    result = tool.run(
        capability=BuilderCapability.REFACTORING.value,
        description="Optimize performance",
    )
    print_result(result)


def example_test_writing() -> None:
    """Example 3: Writing tests."""
    print_section("Example 3: Test Writing")
    
    tool = BuilderSkillTool()
    
    print("\n3a. Write unit tests:")
    code_to_test = """
def calculate_discount(price: float, quantity: int) -> float:
    if quantity >= 10:
        return price * 0.9
    elif quantity >= 5:
        return price * 0.95
    else:
        return price
"""
    
    result = tool.run(
        capability=BuilderCapability.TEST_WRITING.value,
        description="Write comprehensive unit tests",
        context=code_to_test,
        thinking_mode=ThinkingMode.HIGH.value,
        requirements=["edge cases", "boundary testing", "100% coverage"],
    )
    print_result(result)


def example_error_handling() -> None:
    """Example 4: Adding error handling."""
    print_section("Example 4: Error Handling")
    
    tool = BuilderSkillTool()
    
    print("\n4a. Add error handling to code:")
    code = """
def fetch_user(user_id: int) -> dict:
    db = connect_to_database()
    user = db.query(f"SELECT * FROM users WHERE id={user_id}")
    return user.to_dict()
"""
    
    result = tool.run(
        capability=BuilderCapability.ERROR_HANDLING.value,
        description="Add proper error handling and validation",
        context=code,
        thinking_mode=ThinkingMode.HIGH.value,
    )
    print_result(result)


def example_code_generation() -> None:
    """Example 5: Code generation."""
    print_section("Example 5: Code Generation")
    
    tool = BuilderSkillTool()
    
    print("\n5a. Generate CRUD operations:")
    result = tool.run(
        capability=BuilderCapability.CODE_GENERATION.value,
        description="Generate CRUD endpoints for a Product model",
        thinking_mode=ThinkingMode.MEDIUM.value,
        requirements=["FastAPI", "async", "SQLAlchemy ORM"],
    )
    print_result(result)


def example_debugging() -> None:
    """Example 6: Debugging code."""
    print_section("Example 6: Debugging")
    
    tool = BuilderSkillTool()
    
    print("\n6a. Debug and fix code:")
    buggy_code = """
def calculate_sum(numbers: list) -> int:
    total = 0
    for num in numbers:
        total =+ num  # Bug: should be +=
    return total
"""
    
    result = tool.run(
        capability=BuilderCapability.DEBUGGING.value,
        description="Identify and fix the bug",
        context=buggy_code,
    )
    print_result(result)


def example_execution_modes() -> None:
    """Example 7: Using different execution modes."""
    print_section("Example 7: Execution Modes")
    
    tool = BuilderSkillTool()
    context_manager = get_context_manager()
    
    print("\n7a. Running in EXECUTION mode (default):")
    # Default is EXECUTION mode
    result = tool.run(
        capability=BuilderCapability.FEATURE_IMPLEMENTATION.value,
        description="Build a feature",
        thinking_mode=ThinkingMode.MEDIUM.value,
    )
    print(f"Mode: {context_manager.current_mode.value}")
    print_result(result)
    
    # Note: To actually switch modes, you would need to:
    # context_manager.set_execution_mode(ExecutionMode.PLANNING)
    # But we'll just demonstrate the concept


def example_metadata() -> None:
    """Example 8: Inspecting tool metadata."""
    print_section("Example 8: Tool Metadata")
    
    tool = BuilderSkillTool()
    
    print(f"\nTool Name: {tool.metadata.name}")
    print(f"Description: {tool.metadata.description}")
    print(f"Version: {tool.metadata.version}")
    print(f"Category: {tool.metadata.category.value}")
    
    print(f"\nAllowed Execution Modes:")
    for mode in tool.metadata.allowed_modes:
        print(f"  - {mode.value}")
    
    print(f"\nRequired Permissions:")
    for perm in tool.metadata.required_permissions:
        print(f"  - {perm.value}")
    
    print(f"\nCapabilities:")
    for cap in BuilderCapability:
        print(f"  - {cap.value}")
    
    print(f"\nThinking Modes:")
    for mode in ThinkingMode:
        print(f"  - {mode.value}")
    
    print(f"\nParameters:")
    for param in tool.metadata.parameters:
        required_str = "required" if param.required else "optional"
        print(f"  - {param.name} ({required_str}): {param.description}")


def main() -> None:
    """Run all examples."""
    print("\n" + "="*70)
    print("  Builder Skill Tool - Examples")
    print("="*70)
    
    try:
        # First, set up proper execution context with required permissions
        context_manager = get_context_manager()
        context_manager.switch_mode(ExecutionMode.EXECUTION)
        
        # Run all examples
        example_metadata()
        example_feature_implementation()
        example_refactoring()
        example_test_writing()
        example_error_handling()
        example_code_generation()
        example_debugging()
        example_execution_modes()
        
        print_section("Examples Complete")
        print("\nAll examples completed successfully!")
        print("\nKey Takeaways:")
        print("  1. BuilderSkillTool provides 6 major capabilities")
        print("  2. Thinking modes control implementation approach")
        print("  3. Some capabilities require code context")
        print("  4. Execution modes affect availability (PLANNING vs EXECUTION)")
        print("  5. All tool interaction goes through the standardized Tool interface")
        print("  6. Results include structured output for programmatic use")
        
    except Exception as e:
        print(f"\n[ERROR] Error running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
