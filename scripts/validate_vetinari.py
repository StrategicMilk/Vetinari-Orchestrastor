#!/usr/bin/env python3
"""
Vetinari Smoke Validation Script

Run: python validate_vetinari.py
Runs import, wiring, and lightweight smoke checks without requiring live model inference.
This script is not release proof and must not be presented as full runtime coverage.

VET304 Allowlist: This file is a smoke-test script that validates module presence
and basic structural integrity by design. Type-checking assertions are intentional
and serve as integration smoke tests before running full test suites.
"""

from __future__ import annotations

import sys

errors: list[tuple[str, str]] = []
passed: list[str] = []


def check(name: str, fn: object) -> None:
    """Run a validation function and record pass/fail.

    Args:
        name: Human-readable name for the check.
        fn: Callable to invoke; failure is caught and recorded.
    """
    try:
        fn()
        passed.append(name)
        print(f"  [PASS] {name}")
    except Exception as e:
        errors.append((name, str(e)))
        print(f"  [FAIL] {name}: {e}")


# 1. Core imports
def test_imports() -> None:
    from vetinari.agents import ForemanAgent, InspectorAgent, WorkerAgent
    from vetinari.agents.consolidated.worker_agent import WorkerAgent as WorkerAgentDirect

    if ForemanAgent is None:
        raise AssertionError("ForemanAgent is None")
    if WorkerAgent is None:
        raise AssertionError("WorkerAgent is None")
    if InspectorAgent is None:
        raise AssertionError("InspectorAgent is None")
    if WorkerAgentDirect is None:
        raise AssertionError("WorkerAgentDirect is None")


# 2. Learning subsystem imports
def test_learning_imports() -> None:
    from vetinari.learning.feedback_loop import get_feedback_loop
    from vetinari.learning.model_selector import get_thompson_selector
    from vetinari.learning.quality_scorer import get_quality_scorer

    if get_quality_scorer is None:
        raise AssertionError("get_quality_scorer is None")
    if get_feedback_loop is None:
        raise AssertionError("get_feedback_loop is None")
    if get_thompson_selector is None:
        raise AssertionError("get_thompson_selector is None")
    print("  + Learning imports OK")


# 3. Tool imports
def test_tool_imports() -> None:
    from vetinari.code_sandbox import SandboxManager
    from vetinari.tools.web_search_tool import WebSearchTool

    if WebSearchTool is None:
        raise AssertionError("WebSearchTool is None")
    if SandboxManager is None:
        raise AssertionError("SandboxManager is None")
    print("  + Tool imports OK")


# 4. CLI import
def test_cli() -> None:
    from vetinari.cli import main as cli_main

    if not callable(cli_main):
        raise AssertionError("cli_main is not callable")


# 5. Two-layer orchestration
def test_orchestration() -> None:
    from vetinari.orchestration.two_layer import get_two_layer_orchestrator

    orch = get_two_layer_orchestrator()
    graph = orch.generate_plan_only("Build a Python test project")
    if len(graph.nodes) < 2:
        raise AssertionError(f"Expected 2+ tasks, got {len(graph.nodes)}")
    layers = graph.get_next_layer()
    if len(layers) < 1:
        raise AssertionError("Expected at least 1 execution layer")
    # Verify first layer has tasks with no dependencies
    first_layer = layers[0]
    if not all(len(n.depends_on) == 0 for n in first_layer):
        raise AssertionError("First layer should have no deps")


# 6. ForemanAgent execution
def test_planner() -> None:
    from vetinari.agents import get_foreman_agent
    from vetinari.agents.contracts import AgentTask
    from vetinari.types import AgentType

    planner = get_foreman_agent()
    task = AgentTask(
        task_id="t-plan",
        agent_type=AgentType.FOREMAN,
        description="Create a Litestar REST API with JWT auth",
        prompt="Create a Litestar REST API with JWT auth",
    )
    result = planner.execute(task)
    if not result.success:
        raise AssertionError(f"Planner failed: {result.errors}")
    plan = result.output
    if not isinstance(plan.get("tasks"), list):
        raise AssertionError("Plan should have tasks list")
    if len(plan["tasks"]) < 3:
        raise AssertionError(f"Expected 3+ tasks, got {len(plan['tasks'])}")


# 7. Quality scorer
def test_quality_scorer() -> None:
    from vetinari.learning.quality_scorer import get_quality_scorer

    scorer = get_quality_scorer()
    score = scorer.score(
        task_id="test-qs",
        model_id="test-model",
        task_type="coding",
        task_description="Write a Python function",
        output="def add(a, b):\n    '''Add two numbers.'''\n    return a + b\n",
        use_llm=False,
    )
    if not (0.0 <= score.overall_score <= 1.0):
        raise AssertionError(f"Score out of range: {score.overall_score}")
    if score.method not in ("heuristic", "llm", "hybrid"):
        raise AssertionError(f"Invalid method: {score.method}")


# 8. Thompson Sampling
def test_thompson() -> None:
    from vetinari.learning.model_selector import ThompsonSamplingSelector

    sel = ThompsonSamplingSelector()
    # Give model_a much better performance
    for _ in range(10):
        sel.update("model_a", "coding", 0.95, True)
    for _ in range(10):
        sel.update("model_b", "coding", 0.2, False)
    # Model A should be selected most of the time
    wins = sum(1 for _ in range(20) if sel.select_model("coding", ["model_a", "model_b"]) == "model_a")
    if wins < 12:
        raise AssertionError(f"Thompson should prefer model_a, but only won {wins}/20 times")


# 9. Workflow learner
def test_workflow_learner() -> None:
    from vetinari.learning.workflow_learner import WorkflowLearner

    learner = WorkflowLearner()
    learner.record_outcome("Build web app", 4, 3, ["WORKER", "INSPECTOR"], 0.85, True)
    recs = learner.get_recommendations("Build REST API")
    if recs["domain"] != "coding":
        raise AssertionError(f"Expected domain 'coding', got {recs['domain']}")
    if not isinstance(recs["recommended_depth"], int):
        raise AssertionError(f"Expected int recommended_depth, got {type(recs['recommended_depth'])}")


# 10. Feedback loop (no real DB needed)
def test_feedback_loop() -> None:
    from vetinari.learning.feedback_loop import FeedbackLoop

    loop = FeedbackLoop()  # Will gracefully fail without DB
    # Should not raise even without memory store
    loop.record_outcome("t1", "model-x", "coding", 0.8, 1000, 0.01, True)


# 11. Web search tool — validated against the canonical default backend from vetinari.constants
def test_search_tool() -> None:
    from vetinari.constants import DEFAULT_SEARCH_BACKEND
    from vetinari.tools.web_search_tool import WebSearchTool

    tool = WebSearchTool(backend=DEFAULT_SEARCH_BACKEND)
    if tool is None:
        raise AssertionError("WebSearchTool is None")
    if tool.backend_name != DEFAULT_SEARCH_BACKEND:
        raise AssertionError(f"Expected backend '{DEFAULT_SEARCH_BACKEND}', got '{tool.backend_name}'")
    if not hasattr(tool, "search"):
        raise AssertionError("tool missing 'search' method")
    if not hasattr(tool, "multi_source_search"):
        raise AssertionError("tool missing 'multi_source_search' method")
    if not hasattr(tool, "verify_claim"):
        raise AssertionError("tool missing 'verify_claim' method")


# 12. Adapter registry
def test_adapter_registry() -> None:
    from vetinari.adapters.base import ProviderType
    from vetinari.adapters.registry import AdapterRegistry

    AdapterRegistry.clear_instances()
    providers = AdapterRegistry.list_supported_providers()
    if ProviderType.LOCAL not in providers:
        raise AssertionError(f"ProviderType.LOCAL not in {providers}")
    if ProviderType.OPENAI not in providers:
        raise AssertionError(f"ProviderType.OPENAI not in {providers}")
    AdapterRegistry.clear_instances()


# 13. Sandbox fix
def test_sandbox() -> None:
    from vetinari.sandbox_manager import get_sandbox_manager

    mgr = get_sandbox_manager()
    result = mgr.execute("x = 1 + 1", sandbox_type="in_process")
    if not result.success:
        raise AssertionError(f"Sandbox failed: {result.error}")


# 14. Credentials module imports and singleton accessor
def test_credentials() -> None:
    from vetinari.credentials import CredentialManager, get_credential_manager

    if CredentialManager is None:
        raise AssertionError("CredentialManager is None")
    mgr = get_credential_manager()
    if mgr is None:
        raise AssertionError("get_credential_manager() returned None")
    if not isinstance(mgr, CredentialManager):
        raise AssertionError(f"Expected CredentialManager, got {type(mgr)}")
    print("  + Credentials imports OK")


# 15. Structured logging ContextVar fix
def test_structured_logging() -> None:
    from vetinari.structured_logging import CorrelationContext, get_trace_id

    with CorrelationContext(trace_id="test-trace-123"):
        tid = get_trace_id()
        if tid != "test-trace-123":
            raise AssertionError(f"Expected 'test-trace-123', got {tid}")
    if get_trace_id() is not None:
        raise AssertionError(f"Expected None outside context, got {get_trace_id()}")


def main() -> None:
    """Run all validation checks and report results."""
    print("=" * 60)
    print(" VETINARI Smoke Validation")
    print(" Import/wiring smoke only; not release proof")
    print("=" * 60)

    check("Core agent imports", test_imports)
    check("Learning subsystem imports", test_learning_imports)
    check("Tool skill imports", test_tool_imports)
    check("CLI import", test_cli)
    check("Two-layer orchestration + DAG fix", test_orchestration)
    check("ForemanAgent keyword decomposition", test_planner)
    check("Quality scorer (heuristic)", test_quality_scorer)
    check("Thompson Sampling model selection", test_thompson)
    check("Workflow learner domain inference", test_workflow_learner)
    check("Feedback loop graceful degradation", test_feedback_loop)
    check("WebSearchTool canonical backend", test_search_tool)
    check("Adapter registry class-level API", test_adapter_registry)
    check("Sandbox in-process execution", test_sandbox)
    check("Credentials module syntax", test_credentials)
    check("CorrelationContext __exit__ fix", test_structured_logging)

    print()
    print("=" * 60)
    print(f" Results: {len(passed)} passed, {len(errors)} failed")
    print("=" * 60)
    if errors:
        print("\nFailed checks:")
        for name, err in errors:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("\nAll smoke validation checks PASSED. Run targeted and release gates separately.")
        sys.exit(0)


if __name__ == "__main__":
    main()
