#!/usr/bin/env python
"""
Example: Using the Evaluator Skill Tool

Demonstrates all capabilities of the evaluator skill tool including:
- Code review
- Quality assessment
- Security auditing
- Test strategy planning
- Performance review
- Best practices checking
"""

from vetinari.tools.evaluator_skill import (
    EvaluatorSkillTool,
    EvaluatorCapability,
    ThinkingMode,
)
from vetinari.execution_context import ExecutionMode, get_context_manager
from unittest.mock import Mock


def setup_tool():
    """Set up evaluator tool with mocked context."""
    tool = EvaluatorSkillTool()
    mock_ctx_manager = Mock()
    mock_context = Mock()
    mock_context.mode = ExecutionMode.EXECUTION
    mock_ctx_manager.current_context = mock_context
    tool._context_manager = mock_ctx_manager
    return tool, mock_ctx_manager, mock_context


def example_tool_metadata():
    """Demonstrate tool metadata inspection."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Tool Metadata Inspection")
    print("="*60)
    
    tool = EvaluatorSkillTool()
    print(f"\nTool Name: {tool.metadata.name}")
    print(f"Description: {tool.metadata.description}")
    print(f"Version: {tool.metadata.version}")
    print(f"Author: {tool.metadata.author}")
    print(f"\nAvailable Capabilities:")
    for cap in EvaluatorCapability:
        print(f"  - {cap.value}")
    print(f"\nThinking Modes:")
    for mode in ThinkingMode:
        print(f"  - {mode.value}")
    print(f"\nRequired Permissions:")
    for perm in tool.metadata.required_permissions:
        print(f"  - {perm.value}")


def example_code_review():
    """Demonstrate code review capability."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Code Review")
    print("="*60)
    
    tool, _, _ = setup_tool()
    
    sample_code = """
def calculate_total(items):
    '''Calculate the total of items.'''
    total = 0
    for item in items:
        total += item
    return total
"""
    
    print("\nCode to review:")
    print(sample_code)
    
    result = tool.execute(
        capability="code_review",
        code=sample_code,
        thinking_mode="medium",
    )
    
    print(f"\nResult Success: {result.success}")
    if result.output:
        print(f"Quality Score: {result.output.get('quality_score')}")
        print(f"Issues Found: {len(result.output.get('issues', []))}")
        if result.output.get('summary'):
            print(f"Summary: {result.output['summary']}")


def example_quality_assessment():
    """Demonstrate quality assessment capability."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Quality Assessment")
    print("="*60)
    
    tool, _, _ = setup_tool()
    
    sample_code = """
def process_data(x, y):
    a = x + y
    b = a * 2
    c = b - x
    d = c / y
    return d
"""
    
    print("\nCode to assess:")
    print(sample_code)
    
    result = tool.execute(
        capability="quality_assessment",
        code=sample_code,
        thinking_mode="high",
    )
    
    print(f"\nResult Success: {result.success}")
    if result.output:
        print(f"Quality Score: {result.output.get('quality_score')}")
        print(f"Issues Found: {len(result.output.get('issues', []))}")
        print(f"\nRecommendations:")
        for i, rec in enumerate(result.output.get('recommendations', []), 1):
            print(f"  {i}. {rec}")


def example_security_audit():
    """Demonstrate security audit capability."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Security Audit")
    print("="*60)
    
    tool, _, _ = setup_tool()
    
    vulnerable_code = """
import pickle

def load_config(data):
    config = pickle.loads(data)
    return config

api_key = 'sk-1234567890abcdef'
"""
    
    print("\nCode to audit (with intentional vulnerabilities):")
    print(vulnerable_code)
    
    result = tool.execute(
        capability="security_audit",
        code=vulnerable_code,
        thinking_mode="xhigh",
    )
    
    print(f"\nResult Success: {result.success}")
    if result.output:
        issues = result.output.get('issues', [])
        print(f"Security Issues Found: {len(issues)}")
        for issue in issues:
            print(f"\n  [{issue['severity'].upper()}] {issue['title']}")
            if issue.get('description'):
                print(f"    Description: {issue['description']}")
            if issue.get('suggestion'):
                print(f"    Suggestion: {issue['suggestion']}")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(result.output.get('recommendations', []), 1):
            print(f"  {i}. {rec}")


def example_test_strategy():
    """Demonstrate test strategy capability."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Test Strategy Planning")
    print("="*60)
    
    tool, _, _ = setup_tool()
    
    sample_code = """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
"""
    
    print("\nCode to create test strategy for:")
    print(sample_code)
    
    result = tool.execute(
        capability="test_strategy",
        code=sample_code,
        thinking_mode="high",
    )
    
    print(f"\nResult Success: {result.success}")
    if result.output:
        print(f"Test Coverage Target: {result.output.get('test_coverage')}%")
        print(f"\nTest Recommendations:")
        for i, rec in enumerate(result.output.get('recommendations', []), 1):
            print(f"  {i}. {rec}")


def example_performance_review():
    """Demonstrate performance review capability."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Performance Review")
    print("="*60)
    
    tool, _, _ = setup_tool()
    
    performance_code = """
def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j]:
                duplicates.append(arr[i])
    return duplicates
"""
    
    print("\nCode to review for performance:")
    print(performance_code)
    
    result = tool.execute(
        capability="performance_review",
        code=performance_code,
        thinking_mode="medium",
    )
    
    print(f"\nResult Success: {result.success}")
    if result.output:
        issues = result.output.get('issues', [])
        print(f"Performance Issues Found: {len(issues)}")
        for issue in issues:
            print(f"\n  [{issue['severity'].upper()}] {issue['title']}")
            if issue.get('description'):
                print(f"    Description: {issue['description']}")
            if issue.get('suggestion'):
                print(f"    Suggestion: {issue['suggestion']}")


def example_best_practices():
    """Demonstrate best practices checking."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Best Practices Check")
    print("="*60)
    
    tool, _, _ = setup_tool()
    
    sample_code = """
def process_data(data):
    result = []
    for item in data:
        x = item * 2
        result.append(x)
    return result

def format_output(items):
    output = ""
    for item in items:
        output = output + str(item) + " "
    return output
"""
    
    print("\nCode to check against best practices:")
    print(sample_code)
    
    result = tool.execute(
        capability="best_practices",
        code=sample_code,
        thinking_mode="high",
    )
    
    print(f"\nResult Success: {result.success}")
    if result.output:
        print(f"Quality Score: {result.output.get('quality_score')}")
        print(f"\nBest Practices Recommendations:")
        for i, rec in enumerate(result.output.get('recommendations', []), 1):
            print(f"  {i}. {rec}")


def example_execution_modes():
    """Demonstrate different execution modes."""
    print("\n" + "="*60)
    print("EXAMPLE 8: Execution Modes (EXECUTION vs PLANNING)")
    print("="*60)
    
    tool, mock_ctx_manager, mock_context = setup_tool()
    
    sample_code = """
def hello():
    print("Hello, World!")
"""
    
    print("\nSample code:")
    print(sample_code)
    
    # EXECUTION mode
    print("\n--- EXECUTION Mode ---")
    mock_context.mode = ExecutionMode.EXECUTION
    result_exec = tool.execute(
        capability="code_review",
        code=sample_code,
    )
    print(f"Success: {result_exec.success}")
    print(f"Summary: {result_exec.output.get('summary', 'N/A')[:80]}...")
    
    # PLANNING mode
    print("\n--- PLANNING Mode ---")
    mock_context.mode = ExecutionMode.PLANNING
    result_plan = tool.execute(
        capability="code_review",
        code=sample_code,
    )
    print(f"Success: {result_plan.success}")
    print(f"Summary: {result_plan.output.get('summary', 'N/A')}")


def example_thinking_modes():
    """Demonstrate different thinking modes."""
    print("\n" + "="*60)
    print("EXAMPLE 9: Thinking Modes (low/medium/high/xhigh)")
    print("="*60)
    
    tool, _, _ = setup_tool()
    
    sample_code = """
def process():
    x = 1
    y = 2
    z = x + y
    return z
"""
    
    print("\nSample code:")
    print(sample_code)
    print("\nRunning code review at different thinking levels...")
    
    for mode in ThinkingMode:
        result = tool.execute(
            capability="test_strategy",
            code=sample_code,
            thinking_mode=mode.value,
        )
        rec_count = len(result.output.get('recommendations', []))
        print(f"  {mode.value.upper():6} - {rec_count} recommendations")


def example_with_context():
    """Demonstrate code review with additional context."""
    print("\n" + "="*60)
    print("EXAMPLE 10: Code Review with Context")
    print("="*60)
    
    tool, _, _ = setup_tool()
    
    code_snippet = """
def parse_request(request):
    data = json.loads(request.body)
    return data
"""
    
    context = "This is part of a FastAPI endpoint handler"
    
    print("\nCode:")
    print(code_snippet)
    print(f"\nContext: {context}")
    
    result = tool.execute(
        capability="code_review",
        code=code_snippet,
        context=context,
        thinking_mode="medium",
    )
    
    print(f"\nResult Success: {result.success}")
    if result.output:
        print(f"Summary: {result.output.get('summary', 'N/A')}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("EVALUATOR SKILL TOOL - COMPREHENSIVE EXAMPLES")
    print("="*60)
    
    example_tool_metadata()
    example_code_review()
    example_quality_assessment()
    example_security_audit()
    example_test_strategy()
    example_performance_review()
    example_best_practices()
    example_execution_modes()
    example_thinking_modes()
    example_with_context()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
