"""
Vetinari System Validation Script

Run: python validate_vetinari.py
Tests all core components without requiring a live LM Studio connection.
"""
import sys

print("=" * 60)
print(" VETINARI System Validation")
print("=" * 60)

errors = []
passed = []


def check(name, fn):
    try:
        fn()
        passed.append(name)
        print(f"  [PASS] {name}")
    except Exception as e:
        errors.append((name, str(e)))
        print(f"  [FAIL] {name}: {e}")


# 1. Core imports
def test_imports():
    from vetinari.agents.base_agent import BaseAgent
    from vetinari.agents.planner_agent import PlannerAgent
    from vetinari.agents.researcher_agent import ResearcherAgent
    from vetinari.agents.evaluator_agent import EvaluatorAgent
    from vetinari.agents.builder_agent import BuilderAgent
    from vetinari.agents.oracle_agent import OracleAgent
    from vetinari.agents.synthesizer_agent import SynthesizerAgent
    from vetinari.agents.explorer_agent import ExplorerAgent
    from vetinari.agents.librarian_agent import LibrarianAgent
    from vetinari.agents.improvement_agent import ImprovementAgent
    from vetinari.agents.user_interaction_agent import UserInteractionAgent

check("Core agent imports", test_imports)


# 2. Learning subsystem imports
def test_learning_imports():
    from vetinari.learning.quality_scorer import get_quality_scorer, QualityScorer
    from vetinari.learning.feedback_loop import get_feedback_loop, FeedbackLoop
    from vetinari.learning.model_selector import get_thompson_selector, ThompsonSamplingSelector
    from vetinari.learning.prompt_evolver import get_prompt_evolver, PromptEvolver
    from vetinari.learning.workflow_learner import get_workflow_learner, WorkflowLearner
    from vetinari.learning.cost_optimizer import get_cost_optimizer, CostOptimizer
    from vetinari.learning.auto_tuner import get_auto_tuner, AutoTuner

check("Learning subsystem imports", test_learning_imports)


# 3. Tool imports
def test_tool_imports():
    from vetinari.tools.web_search_tool import WebSearchTool, get_search_tool
    from vetinari.tools.builder_skill import BuilderSkillTool  # class is BuilderSkillTool
    from vetinari.tools.evaluator_skill import EvaluatorSkillTool
    from vetinari.tools.security_auditor_skill import SecurityAuditorSkill
    from vetinari.tools.data_engineer_skill import DataEngineerSkill
    from vetinari.tools.documentation_skill import DocumentationSkill
    from vetinari.tools.cost_planner_skill import CostPlannerSkill
    from vetinari.tools.test_automation_skill import TestAutomationSkill
    from vetinari.tools.experimentation_manager_skill import ExperimentationManagerSkill

check("Tool skill imports", test_tool_imports)


# 4. CLI import
def test_cli():
    from vetinari.cli import main
    assert callable(main)

check("CLI import", test_cli)


# 5. Two-layer orchestration
def test_orchestration():
    from vetinari.two_layer_orchestration import get_two_layer_orchestrator
    orch = get_two_layer_orchestrator()
    graph = orch.generate_plan_only("Build a Python test project")
    assert len(graph.nodes) >= 2, f"Expected 2+ tasks, got {len(graph.nodes)}"
    layers = graph.get_next_layer()
    assert len(layers) >= 1, "Expected at least 1 execution layer"
    # Verify first layer has tasks with no dependencies
    first_layer = layers[0]
    assert all(len(n.depends_on) == 0 for n in first_layer), "First layer should have no deps"

check("Two-layer orchestration + DAG fix", test_orchestration)


# 6. PlannerAgent execution
def test_planner():
    from vetinari.agents.planner_agent import get_planner_agent
    from vetinari.agents.contracts import AgentTask, AgentType
    planner = get_planner_agent()
    task = AgentTask(task_id="t-plan", agent_type=AgentType.PLANNER,
                     description="Create a Flask REST API with JWT auth",
                     prompt="Create a Flask REST API with JWT auth")
    result = planner.execute(task)
    assert result.success, f"Planner failed: {result.errors}"
    plan = result.output
    assert isinstance(plan.get("tasks"), list), "Plan should have tasks list"
    assert len(plan["tasks"]) >= 3, f"Expected 3+ tasks, got {len(plan['tasks'])}"

check("PlannerAgent keyword decomposition", test_planner)


# 7. Quality scorer
def test_quality_scorer():
    from vetinari.learning.quality_scorer import get_quality_scorer
    scorer = get_quality_scorer()
    score = scorer.score(
        task_id="test-qs", model_id="test-model",
        task_type="coding", task_description="Write a Python function",
        output="def add(a, b):\n    '''Add two numbers.'''\n    return a + b\n",
        use_llm=False,
    )
    assert 0.0 <= score.overall_score <= 1.0
    assert score.method in ("heuristic", "llm", "hybrid")

check("Quality scorer (heuristic)", test_quality_scorer)


# 8. Thompson Sampling
def test_thompson():
    from vetinari.learning.model_selector import ThompsonSamplingSelector
    sel = ThompsonSamplingSelector()
    # Give model_a much better performance
    for _ in range(10):
        sel.update("model_a", "coding", 0.95, True)
    for _ in range(10):
        sel.update("model_b", "coding", 0.2, False)
    # Model A should be selected most of the time
    wins = sum(1 for _ in range(20) if sel.select_model("coding", ["model_a", "model_b"]) == "model_a")
    assert wins >= 12, f"Thompson should prefer model_a, but only won {wins}/20 times"

check("Thompson Sampling model selection", test_thompson)


# 9. Workflow learner
def test_workflow_learner():
    from vetinari.learning.workflow_learner import WorkflowLearner
    learner = WorkflowLearner()
    learner.record_outcome("Build web app", 4, 3, ["BUILDER", "EVALUATOR"], 0.85, True)
    recs = learner.get_recommendations("Build REST API")
    assert recs["domain"] == "coding"
    assert isinstance(recs["recommended_depth"], int)

check("Workflow learner domain inference", test_workflow_learner)


# 10. Feedback loop (no real DB needed)
def test_feedback_loop():
    from vetinari.learning.feedback_loop import FeedbackLoop
    loop = FeedbackLoop()  # Will gracefully fail without DB
    # Should not raise even without memory store
    loop.record_outcome("t1", "model-x", "coding", 0.8, 1000, 0.01, True)

check("Feedback loop graceful degradation", test_feedback_loop)


# 11. Web search tool
def test_search_tool():
    from vetinari.tools.web_search_tool import WebSearchTool, SearchResponse
    tool = WebSearchTool(backend="duckduckgo")
    assert tool is not None
    assert hasattr(tool, "search")
    assert hasattr(tool, "multi_source_search")
    assert hasattr(tool, "verify_claim")

check("WebSearchTool initialization", test_search_tool)


# 12. Adapter registry
def test_adapter_registry():
    from vetinari.adapters.registry import AdapterRegistry
    from vetinari.adapters.base import ProviderType, ProviderConfig
    AdapterRegistry.clear_instances()
    providers = AdapterRegistry.list_supported_providers()
    assert ProviderType.LM_STUDIO in providers
    assert ProviderType.OPENAI in providers
    AdapterRegistry.clear_instances()

check("Adapter registry class-level API", test_adapter_registry)


# 13. Sandbox fix
def test_sandbox():
    from vetinari.sandbox import sandbox_manager, SandboxManager
    result = sandbox_manager.execute("x = 1 + 1", sandbox_type="in_process")
    assert result.success, f"Sandbox failed: {result.error}"

check("Sandbox in-process execution", test_sandbox)


# 14. Credentials syntax (was SyntaxError before fix)
def test_credentials():
    from vetinari.credentials import CredentialManager, CredentialVault

check("Credentials module syntax", test_credentials)


# 15. Structured logging ContextVar fix
def test_structured_logging():
    from vetinari.structured_logging import CorrelationContext, get_trace_id
    with CorrelationContext(trace_id="test-trace-123") as ctx:
        tid = get_trace_id()
        assert tid == "test-trace-123"
    assert get_trace_id() is None

check("CorrelationContext __exit__ fix", test_structured_logging)


# Summary
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
    print("\nAll validation checks PASSED!")
    sys.exit(0)
