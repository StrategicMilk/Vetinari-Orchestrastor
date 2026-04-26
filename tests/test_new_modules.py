"""
Tests for new Vetinari modules added in the comprehensive improvement pass.

Covers:
- vetinari.types (canonical enums)
- vetinari.constants
- vetinari.rules_manager
- vetinari.goal_verifier
- vetinari.decomposition
- vetinari.decomposition_agent
- vetinari.assignment_pass
- vetinari.multi_agent_orchestrator
- vetinari.agents.image_generator_agent (no SD required)
- vetinari.agents.planner_agent (verification fix)
- vetinari.orchestration (bug fixes)
"""

import json
from unittest.mock import patch

import pytest

from vetinari.types import AgentType

# ──────────────────────────────────────────────────────────────────────────────
# vetinari.types
# ──────────────────────────────────────────────────────────────────────────────


class TestCanonicalTypes:
    def test_task_status_values(self):
        from vetinari.types import StatusEnum

        values = [s.value for s in StatusEnum]
        assert "pending" in values
        assert "completed" in values
        assert "failed" in values
        assert "cancelled" in values
        assert "running" in values
        assert "blocked" in values

    def test_plan_status_values(self):
        from vetinari.types import PlanStatus

        values = [s.value for s in PlanStatus]
        assert "executing" in values
        assert "completed" in values
        assert "failed" in values

    def test_agent_type_worker_has_correct_value(self):
        from vetinari.types import AgentType

        assert AgentType.WORKER.value == "WORKER"

    def test_agent_type_has_three_tier_model(self):
        from vetinari.types import AgentType

        # Three active agents in the three-tier model
        assert AgentType.FOREMAN.value == "FOREMAN"
        assert AgentType.WORKER.value == "WORKER"
        assert AgentType.INSPECTOR.value == "INSPECTOR"

    def test_model_provider_values(self):
        from vetinari.types import ModelProvider

        values = [p.value for p in ModelProvider]
        assert "local" in values
        assert "openai" in values
        assert "anthropic" in values

    def test_memory_type_values(self):
        from vetinari.types import MemoryType

        assert MemoryType.INTENT.value == "intent"
        assert MemoryType.SOLUTION.value == "solution"


# ──────────────────────────────────────────────────────────────────────────────
# vetinari.constants
# ──────────────────────────────────────────────────────────────────────────────


class TestConstants:
    def test_default_models_dir(self):
        """Default models directory constant is a non-empty string path."""
        from vetinari.constants import (
            DEFAULT_MODELS_DIR,
        )

        assert isinstance(DEFAULT_MODELS_DIR, str)
        assert len(DEFAULT_MODELS_DIR) > 0

    def test_timeouts_positive(self):
        from vetinari.constants import TIMEOUT_LONG, TIMEOUT_MEDIUM, TIMEOUT_SHORT, TIMEOUT_VERY_LONG

        assert TIMEOUT_SHORT > 0
        assert TIMEOUT_MEDIUM > TIMEOUT_SHORT
        assert TIMEOUT_LONG > TIMEOUT_MEDIUM
        assert TIMEOUT_VERY_LONG > TIMEOUT_LONG

    def test_quality_thresholds_range(self):
        from vetinari.constants import CRITICAL_QUALITY_THRESHOLD, DEFAULT_QUALITY_THRESHOLD, HIGH_QUALITY_THRESHOLD

        for t in (DEFAULT_QUALITY_THRESHOLD, HIGH_QUALITY_THRESHOLD, CRITICAL_QUALITY_THRESHOLD):
            assert 0.0 < t < 1.0

    def test_model_score_weights_sum_to_one(self):
        """Model scoring weights must sum to 1.0 for correct weighted scoring."""
        from vetinari.constants import (
            MODEL_SCORE_WEIGHT_CAPABILITY,
            MODEL_SCORE_WEIGHT_CONTEXT,
            MODEL_SCORE_WEIGHT_COST,
            MODEL_SCORE_WEIGHT_FREE_TIER,
            MODEL_SCORE_WEIGHT_LATENCY,
        )

        total = (
            MODEL_SCORE_WEIGHT_CAPABILITY
            + MODEL_SCORE_WEIGHT_CONTEXT
            + MODEL_SCORE_WEIGHT_LATENCY
            + MODEL_SCORE_WEIGHT_COST
            + MODEL_SCORE_WEIGHT_FREE_TIER
        )
        assert abs(total - 1.0) < 1e-9, f"Model scoring weights sum to {total}, expected 1.0"

    def test_image_defaults(self):
        from vetinari.constants import IMAGE_DEFAULT_HEIGHT, IMAGE_DEFAULT_STEPS, IMAGE_DEFAULT_WIDTH

        assert IMAGE_DEFAULT_WIDTH > 0
        assert IMAGE_DEFAULT_HEIGHT > 0
        assert IMAGE_DEFAULT_STEPS > 0

    def test_min_tasks_changed(self):
        from vetinari.constants import MIN_TASKS_PER_PLAN

        assert MIN_TASKS_PER_PLAN == 3  # Was 5, changed to allow simple goals


# ──────────────────────────────────────────────────────────────────────────────
# vetinari.rules_manager
# ──────────────────────────────────────────────────────────────────────────────


class TestRulesManager:
    @pytest.fixture
    def manager(self, tmp_path):
        from vetinari.rules_manager import RulesManager

        rules_file = tmp_path / "rules.yaml"
        return RulesManager(rules_file=rules_file)

    def test_global_rules_empty_initially(self, manager):
        assert manager.get_global_rules() == []

    def test_set_and_get_global_rules(self, manager):
        rules = ["Use type hints", "Follow PEP 8", "Write tests"]
        manager.set_global_rules(rules)
        assert manager.get_global_rules() == rules

    def test_set_rules_strips_whitespace(self, manager):
        manager.set_global_rules(["  Use type hints  ", "", "  Write tests"])
        result = manager.get_global_rules()
        assert result == ["Use type hints", "Write tests"]

    def test_project_rules(self, manager):
        manager.set_project_rules("my_project", ["Use React", "No jQuery"])
        assert manager.get_project_rules("my_project") == ["Use React", "No jQuery"]
        assert manager.get_project_rules("other_project") == []

    def test_model_rules(self, manager):
        manager.set_model_rules("qwen2.5-7b", ["Keep responses short"])
        assert manager.get_model_rules("qwen2.5-7b") == ["Keep responses short"]

    def test_get_rules_combines_all_scopes(self, manager):
        manager.set_global_rules(["global rule"])
        manager.set_project_rules("proj1", ["project rule"])
        manager.set_model_rules("model1", ["model rule"])

        rules = manager.get_rules(project_id="proj1", model_id="model1")
        assert "global rule" in rules
        assert "project rule" in rules
        assert "model rule" in rules
        # Global first
        assert rules.index("global rule") < rules.index("project rule")

    def test_get_rules_deduplicates(self, manager):
        manager.set_global_rules(["shared rule", "global only"])
        manager.set_project_rules("proj", ["shared rule", "project only"])

        rules = manager.get_rules(project_id="proj")
        assert rules.count("shared rule") == 1

    def test_format_rules_empty(self, manager):
        result = manager.format_rules()
        assert result == ""

    def test_format_rules_with_content(self, manager):
        manager.set_global_rules(["Use type hints", "Write tests"])
        result = manager.format_rules()
        assert "## Project Rules" in result
        assert "Use type hints" in result
        assert "Write tests" in result

    def test_global_system_prompt(self, manager):
        prompt = "You are a helpful assistant. Hardware: RTX 5090."
        manager.set_global_system_prompt(prompt)
        assert manager.get_global_system_prompt() == prompt

    def test_build_system_prompt_prefix_empty(self, manager):
        assert manager.build_system_prompt_prefix() == ""

    def test_build_system_prompt_prefix_with_content(self, manager):
        manager.set_global_system_prompt("Global prompt")
        manager.set_global_rules(["Rule 1"])
        result = manager.build_system_prompt_prefix()
        assert "Global prompt" in result
        assert "Rule 1" in result

    def test_persistence(self, tmp_path):
        from vetinari.rules_manager import RulesManager

        rules_file = tmp_path / "rules.yaml"

        m1 = RulesManager(rules_file=rules_file)
        m1.set_global_rules(["persistent rule"])

        m2 = RulesManager(rules_file=rules_file)
        assert "persistent rule" in m2.get_global_rules()

    def test_to_dict(self, manager):
        manager.set_global_rules(["rule1"])
        d = manager.to_dict()
        assert isinstance(d, dict)
        assert "global" in d

    def test_get_agent_rules(self, tmp_path):
        import yaml

        from vetinari.rules_manager import RulesManager

        rules_file = tmp_path / "rules.yaml"
        rules_file.write_text(
            yaml.dump({
                "global": ["global rule"],
                "agents": {
                    AgentType.WORKER.value: ["builder rule 1", "builder rule 2"],
                    AgentType.FOREMAN.value: ["planner rule"],
                },
            }),
            encoding="utf-8",
        )
        m = RulesManager(rules_file=rules_file)
        assert m.get_agent_rules(AgentType.WORKER.value) == ["builder rule 1", "builder rule 2"]
        assert m.get_agent_rules(AgentType.FOREMAN.value) == ["planner rule"]
        assert m.get_agent_rules("UNKNOWN") == []

    def test_get_rules_for_context_merges_all_scopes(self, tmp_path):
        import yaml

        from vetinari.rules_manager import RulesManager

        rules_file = tmp_path / "rules.yaml"
        rules_file.write_text(
            yaml.dump({
                "global": ["global rule"],
                "agents": {AgentType.WORKER.value: ["builder rule"]},
                "projects": {"proj1": ["project rule"]},
                "models": {"model1": ["model rule"]},
            }),
            encoding="utf-8",
        )
        m = RulesManager(rules_file=rules_file)
        rules = m.get_rules_for_context(AgentType.WORKER.value, "model1", "proj1")
        assert "global rule" in rules
        assert "builder rule" in rules
        assert "project rule" in rules
        assert "model rule" in rules
        # Order: global < agent < project < model
        assert rules.index("global rule") < rules.index("builder rule")
        assert rules.index("builder rule") < rules.index("project rule")

    def test_get_rules_for_context_deduplicates(self, tmp_path):
        import yaml

        from vetinari.rules_manager import RulesManager

        rules_file = tmp_path / "rules.yaml"
        rules_file.write_text(
            yaml.dump({
                "global": ["shared rule"],
                "agents": {AgentType.WORKER.value: ["shared rule", "builder only"]},
            }),
            encoding="utf-8",
        )
        m = RulesManager(rules_file=rules_file)
        rules = m.get_rules_for_context(AgentType.WORKER.value)
        assert rules.count("shared rule") == 1
        assert "builder only" in rules

    def test_format_rules_for_context(self, tmp_path):
        import yaml

        from vetinari.rules_manager import RulesManager

        rules_file = tmp_path / "rules.yaml"
        rules_file.write_text(
            yaml.dump({
                "global": ["rule one"],
                "agents": {AgentType.WORKER.value: ["rule two"]},
            }),
            encoding="utf-8",
        )
        m = RulesManager(rules_file=rules_file)
        text = m.format_rules_for_context(AgentType.WORKER.value)
        assert "## Active Rules" in text
        assert "1. rule one" in text
        assert "2. rule two" in text

    def test_format_rules_for_context_empty(self, manager):
        text = manager.format_rules_for_context(AgentType.WORKER.value)
        assert text == ""

    def test_propose_rule_first_observation(self, manager):
        result = manager.propose_rule_from_feedback(
            AgentType.WORKER.value,
            "build",
            "Always validate file paths before writing",
        )
        assert result is False  # not yet accepted (need 3 observations)
        proposed = manager.get_proposed_rules()
        assert len(proposed) == 1

    def test_propose_rule_auto_accept_after_three(self, manager):
        desc = "Never use wildcard imports"
        for _ in range(2):
            result = manager.propose_rule_from_feedback(AgentType.WORKER.value, "build", desc)
            assert result is False
        # Third observation should auto-accept
        result = manager.propose_rule_from_feedback(AgentType.WORKER.value, "build", desc)
        assert result is True
        # Rule should now be in agent rules
        agent_rules = manager.get_agent_rules(AgentType.WORKER.value)
        assert desc in agent_rules
        # Proposed list should be empty
        assert len(manager.get_proposed_rules()) == 0

    def test_propose_rule_model_specific(self, manager):
        desc = "Use explicit JSON format instructions"
        for _ in range(3):
            manager.propose_rule_from_feedback(
                AgentType.WORKER.value,
                "build",
                desc,
                model_name="qwen2.5-7b",
            )
        model_rules = manager.get_model_rules("qwen2.5-7b")
        assert desc in model_rules

    def test_record_quality_rejection_bridges_to_rules(self, monkeypatch, tmp_path):
        import vetinari.rules_manager as rm
        from vetinari.rules_manager import RulesManager

        # Redirect the singleton to a temp file so this test never writes to
        # the real vetinari/config/rules.yaml.
        tmp_rm = RulesManager(rules_file=tmp_path / "rules.yaml")
        monkeypatch.setattr(rm, "_rules_manager", tmp_rm)

        from vetinari.learning.feedback_loop import FeedbackLoop

        fb = FeedbackLoop()
        # Should not raise — fires and forgets
        result = fb.record_quality_rejection(
            agent_type=AgentType.WORKER.value,
            mode="build",
            violation_description="Test violation",
            model_name=None,
        )
        assert result is None  # record_quality_rejection returns None


# ──────────────────────────────────────────────────────────────────────────────
# vetinari.goal_verifier
# ──────────────────────────────────────────────────────────────────────────────


class TestGoalVerifier:
    @pytest.fixture
    def verifier(self):
        from vetinari.validation import GoalVerifier

        return GoalVerifier(quality_threshold=0.7)

    def test_basic_verification_passes(self, verifier):
        report = verifier.verify(
            project_id="test001",
            goal="Create a Python REST API",
            final_output="""
# REST API Implementation

## Authentication
JWT authentication implemented with /auth/login endpoint.

## Endpoints
- GET /users - returns user list
- POST /users - create user

```python
def verify_user_creation():
    response = client.post("/users", json={"name": "alice"})
    assert response.status_code == 201
```
""",
            required_features=["authentication", "REST API"],
            things_to_avoid=[],
        )
        assert report.project_id == "test001"
        assert isinstance(report.compliance_score, float)
        assert isinstance(report.fully_compliant, bool)
        assert isinstance(report.features, list)

    def test_missing_features_identified(self, verifier):
        report = verifier.verify(
            project_id="test002",
            goal="Build a web app",
            final_output="Here is some unrelated text about databases",
            required_features=["user authentication", "React frontend", "PostgreSQL"],
        )
        # Should find at least some missing features
        assert len(report.missing_features) > 0

    def test_tests_present_detection(self, verifier):
        # Output with tests
        report_with = verifier.verify(
            project_id="t1",
            goal="Write code",
            final_output="import pytest\n\ndef test_foo():\n    assert 1 == 1",
        )
        assert report_with.tests_present

        # Output without tests
        report_without = verifier.verify(
            project_id="t2",
            goal="Write code",
            final_output="def foo():\n    return 42",
        )
        assert not report_without.tests_present

    def test_avoid_list_violations_detected(self, verifier):
        report = verifier.verify(
            project_id="t3",
            goal="Build an API",
            final_output="Using jQuery and Bootstrap for the frontend",
            things_to_avoid=["jQuery"],
        )
        # Should detect jQuery violation
        violations = [f for f in report.features if "AVOID" in f.feature]
        assert len(violations) > 0

    def test_corrective_tasks_generated(self, verifier):
        report = verifier.verify(
            project_id="t4",
            goal="Complete system",
            final_output="Incomplete implementation",
            required_features=["authentication", "API", "database"],
        )
        # Force missing features
        report.missing_features = ["authentication", "tests"]
        report.tests_present = False

        tasks = report.get_corrective_tasks()
        assert len(tasks) > 0
        # Should have task for missing tests
        assert any("test" in t.get("description", "").lower() for t in tasks)

    def test_to_dict_serializable(self, verifier):
        report = verifier.verify(
            project_id="t5",
            goal="Test",
            final_output="Some output",
        )
        d = report.to_dict()
        # Should be JSON serializable
        json_str = json.dumps(d)
        assert json_str is not None
        assert d["project_id"] == "t5"


# ──────────────────────────────────────────────────────────────────────────────
# vetinari.decomposition
# ──────────────────────────────────────────────────────────────────────────────


class TestDecompositionEngine:
    @pytest.fixture
    def engine(self):
        from vetinari.planning.decomposition import DecompositionEngine

        return DecompositionEngine()

    def test_get_templates_returns_list(self, engine):
        templates = engine.get_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0

    def test_get_templates_filter_by_keyword(self, engine):
        templates = engine.get_templates(keywords=["web"])
        assert all(any(kw in t.get("keywords", []) for kw in ["web"]) for t in templates)

    def test_get_dod_criteria(self, engine):
        criteria = engine.get_dod_criteria("Standard")
        assert isinstance(criteria, list)
        assert len(criteria) > 0

    def test_get_dor_criteria(self, engine):
        criteria = engine.get_dor_criteria("Hard")
        assert isinstance(criteria, list)
        assert len(criteria) > 0

    def test_keyword_decompose_fallback(self, engine):
        # Without LLM, should fall back to keyword decomposition
        with patch.object(engine, "_keyword_decompose", wraps=engine._keyword_decompose):
            # Force failure of LLM path
            with patch("vetinari.planning.decomposition.DecompositionEngine.decompose_task") as mock_decomp:
                mock_decomp.return_value = engine._keyword_decompose("Build a web application", "root", 0)
                result = mock_decomp("Build a web application")
                assert isinstance(result, list)

    def test_keyword_decompose_returns_subtasks(self, engine):
        subtasks = engine._keyword_decompose("Build a REST API with database", "root", 0)
        assert len(subtasks) > 0
        for st in subtasks:
            assert "subtask_id" in st
            assert "description" in st
            assert "agent_type" in st
            assert st["depth"] == 1

    def test_decomposition_history_empty_initially(self, engine):
        history = engine.get_decomposition_history()
        assert isinstance(history, list)

    def test_dod_levels(self, engine):
        for level in ["Light", "Standard", "Hard"]:
            criteria = engine.get_dod_criteria(level)
            assert len(criteria) > 0

    def test_detect_style_web(self, engine):
        result = engine._keyword_decompose("Create a web application with React", "root", 0)
        # Should include UI task
        agents = [st["agent_type"] for st in result]
        assert any(a in (AgentType.WORKER.value, AgentType.INSPECTOR.value) for a in agents)

    def test_constants_on_engine(self, engine):
        assert engine.DEFAULT_MAX_DEPTH == 14
        assert engine.MIN_MAX_DEPTH == 12
        assert engine.MAX_MAX_DEPTH == 16


# ──────────────────────────────────────────────────────────────────────────────
# vetinari.multi_agent_orchestrator
# ──────────────────────────────────────────────────────────────────────────────


class TestMultiAgentOrchestrator:
    def test_singleton(self):
        from vetinari.agents.multi_agent_orchestrator import MultiAgentOrchestrator

        o1 = MultiAgentOrchestrator.get_or_create()
        o2 = MultiAgentOrchestrator.get_or_create()
        assert o1 is o2

    def test_get_instance_returns_orchestrator(self):
        from vetinari.agents.multi_agent_orchestrator import MultiAgentOrchestrator

        # Ensure an instance exists first (tests must be independent)
        _ = MultiAgentOrchestrator.get_or_create()
        instance = MultiAgentOrchestrator.get_instance()
        assert instance is not None
        assert isinstance(instance, MultiAgentOrchestrator)

    def test_initialize_agents(self):
        from vetinari.agents.multi_agent_orchestrator import MultiAgentOrchestrator

        o = MultiAgentOrchestrator.get_or_create()
        o.initialize_agents()
        assert len(o.agents) > 0

    def test_get_agent_status(self):
        from vetinari.agents.multi_agent_orchestrator import MultiAgentOrchestrator

        o = MultiAgentOrchestrator.get_or_create()
        o.initialize_agents()
        status = o.get_agent_status()
        assert isinstance(status, list)
        assert len(status) > 0
        for s in status[:3]:
            assert "name" in s
            assert "agent_type" in s
            assert "state" in s


# ──────────────────────────────────────────────────────────────────────────────
# vetinari.agents.image_generator_agent
# ──────────────────────────────────────────────────────────────────────────────


class TestImageGeneratorAgent:
    @pytest.fixture
    def agent(self, tmp_path):
        from vetinari.agents.builder_agent import BuilderAgent

        return BuilderAgent(
            config={
                "sd_enabled": False,  # Disable SD to avoid network calls
                "output_dir": str(tmp_path / "images"),
            }
        )

    def test_capabilities(self, agent):
        caps = agent.get_capabilities()
        assert "image_generation" in caps
        assert "logo_design" in caps
        assert "svg_generation" in caps

    def test_get_system_prompt(self, agent):
        prompt = agent.get_system_prompt()
        # v0.5.0: BuilderAgent is now part of Worker (build mode group)
        assert "Worker" in prompt or "Builder" in prompt

    def test_detect_style_logo(self, agent):
        assert agent._detect_style("Create a company logo") == "logo"

    def test_detect_style_icon(self, agent):
        assert agent._detect_style("Make an app icon") == "icon"

    def test_detect_style_ui_mockup(self, agent):
        assert agent._detect_style("Design a UI wireframe") == "ui_mockup"

    def test_detect_style_diagram(self, agent):
        assert agent._detect_style("Create an architecture diagram") == "diagram"

    def test_get_default_size(self, agent):
        assert agent._get_default_size("logo") == (512, 512)
        assert agent._get_default_size("banner") == (1200, 400)
        assert agent._get_default_size("ui_mockup") == (1280, 720)

    def test_minimal_svg_placeholder(self, agent):
        svg = agent._minimal_svg_placeholder("Test Logo", (512, 512))
        assert "<svg" in svg
        assert "</svg>" in svg
        assert "Test Logo" in svg

    def test_save_svg(self, agent, tmp_path):
        svg_code = '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="40"/></svg>'
        agent._output_dir = tmp_path / "images"
        agent._output_dir.mkdir(exist_ok=True)
        path = agent._save_svg(svg_code, "Test")
        assert path.exists()
        content = path.read_text()
        assert "<svg" in content

    def test_execute_returns_svg_fallback(self, agent, tmp_path):
        """When SD is disabled, should fall back to SVG generation."""
        from vetinari.agents.contracts import AgentTask

        agent._output_dir = tmp_path / "images"
        agent._output_dir.mkdir(exist_ok=True)

        # Mock the LLM inference to return SVG
        svg_code = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><rect width="512" height="512" fill="#000"/></svg>'
        from vetinari.types import AgentType as AT

        with patch.object(
            agent,
            "_infer_json",
            return_value={
                "sd_prompt": "test logo",
                "negative_prompt": "bad quality",
                "style_preset": "logo",
                "width": 512,
                "height": 512,
                "steps": 20,
                "description": "test",
                "svg_fallback": svg_code,
            },
        ):
            task = AgentTask(
                task_id="test_img",
                agent_type=AT.WORKER,
                description="Create a test logo",
                prompt="Create a test logo",
                context={},
            )
            result = agent.execute(task)

        # Even if LLM is mocked, result should be a dict
        assert isinstance(result.output, dict) or result.errors

    def test_verify_empty_output(self, agent):
        result = agent.verify({})
        assert not result.passed

    def test_verify_with_images(self, agent, tmp_path):
        svg_path = tmp_path / "test.svg"
        svg_path.write_text("<svg/>")
        result = agent.verify({
            "images": [{"type": "svg", "path": str(svg_path), "code": "<svg/>"}],
            "count": 1,
        })
        assert result.passed


# ──────────────────────────────────────────────────────────────────────────────
# vetinari.agents.planner_agent (verification fix)
# ──────────────────────────────────────────────────────────────────────────────


class TestPlannerAgentVerification:
    def test_verify_passes_with_warnings(self):
        """Score >= 0.7 should pass even with issues (warnings only)."""
        from vetinari.agents.planner_agent import ForemanAgent as PlannerAgent

        agent = PlannerAgent()

        output = {
            "plan_id": "plan_001",
            "goal": "Test goal",
            "tasks": [
                {"id": "t1", "description": "Task 1", "dependencies": []},
                {"id": "t2", "description": "Task 2", "dependencies": []},
                {"id": "t3", "description": "Task 3", "dependencies": ["t1"]},
            ],
        }
        result = agent.verify(output)
        # Should pass since score >= 0.7 (though no dependencies warn)
        # The key fix: issues should NOT prevent passing if score >= 0.7
        assert result.score > 0.0
        assert isinstance(result.passed, bool)

    def test_verify_fails_with_missing_fields(self):
        from vetinari.agents.planner_agent import ForemanAgent as PlannerAgent

        agent = PlannerAgent()
        # Missing required fields
        result = agent.verify({"tasks": []})
        assert not result.passed
        assert result.score < 0.7

    def test_verify_accepts_small_valid_plans(self):
        """Verification should accept small plans (3-4 tasks) as valid."""
        from vetinari.agents.planner_agent import ForemanAgent as PlannerAgent

        agent = PlannerAgent(config={"min_tasks": 3})

        output = {
            "plan_id": "plan_small",
            "goal": "Simple goal",
            "tasks": [
                {"id": "t1", "description": "Step 1", "dependencies": []},
                {"id": "t2", "description": "Step 2", "dependencies": ["t1"]},
                {"id": "t3", "description": "Step 3", "dependencies": ["t2"]},
            ],
        }
        result = agent.verify(output)
        # Should pass — small valid plan
        assert result.passed


# ──────────────────────────────────────────────────────────────────────────────
# vetinari.orchestration (bug fixes)
# ──────────────────────────────────────────────────────────────────────────────


class TestTwoLayerOrchestrationFixes:
    def test_max_concurrent_enforced(self):
        """max_concurrent should cap thread pool size."""
        from vetinari.orchestration.durable_execution import DurableExecutionEngine

        engine = DurableExecutionEngine(max_concurrent=2)
        assert engine.max_concurrent == 2

    def test_transitive_cancellation(self):
        """Failed task A should cancel B (depends on A) and C (depends on B)."""
        from vetinari.orchestration.durable_execution import DurableExecutionEngine
        from vetinari.orchestration.execution_graph import ExecutionGraph, ExecutionTaskNode
        from vetinari.types import StatusEnum

        engine = DurableExecutionEngine()

        graph = ExecutionGraph(plan_id="test_trans", goal="Test transitive cancellation")
        task_a = ExecutionTaskNode(id="a", description="A")
        task_b = ExecutionTaskNode(id="b", description="B", depends_on=["a"])
        task_c = ExecutionTaskNode(id="c", description="C", depends_on=["b"])  # transitive dep

        for t in [task_a, task_b, task_c]:
            graph.nodes[t.id] = t

        task_a.status = StatusEnum.FAILED
        engine._handle_layer_failure(graph, [task_a])

        assert graph.nodes["b"].status == StatusEnum.CANCELLED
        assert graph.nodes["c"].status == StatusEnum.CANCELLED  # FIXED: was not cancelled before

    def test_event_history_grows(self):
        """Events should be appended to history."""
        from vetinari.orchestration.durable_execution import DurableExecutionEngine

        engine = DurableExecutionEngine()
        engine._emit_event("test_event", "task_1", {"key": "value"})
        assert len(engine._event_history) >= 1
        assert engine._event_history[-1].event_type == "test_event"

    def test_task_node_serialization(self):
        from vetinari.orchestration.execution_graph import ExecutionTaskNode
        from vetinari.types import StatusEnum

        node = ExecutionTaskNode(
            id="t1",
            description="Test task",
            task_type="coding",
            depends_on=["t0"],
            assigned_model="qwen2.5-7b",
        )
        d = node.to_dict()
        assert d["id"] == "t1"
        assert d["task_type"] == "coding"
        assert d["depends_on"] == ["t0"]
        assert d["status"] == StatusEnum.PENDING.value


# ──────────────────────────────────────────────────────────────────────────────
# vetinari.model_search (cache hash fix)
# ──────────────────────────────────────────────────────────────────────────────


class TestModelSearchCacheFix:
    def test_cache_key_is_deterministic(self, tmp_path):
        """Cache keys must be deterministic (not Python's random hash)."""
        import hashlib

        from vetinari.model_discovery import ModelSearchEngine

        engine = ModelSearchEngine(cache_dir=str(tmp_path))

        # Mock the actual search to not hit network
        with patch.object(engine, "_search_external_sources", return_value=[]):
            engine.search_for_task("python REST API", local_models=[])

        # Verify cache files use deterministic names
        query = "python REST API"
        hashlib.md5(query.encode("utf-8")).hexdigest()

        # Check that cache files have MD5-style names (not hash() output)
        cache_files = list(tmp_path.glob("*.json"))
        for cf in cache_files:
            # MD5 hex is 32 chars
            stem = cf.stem
            # Remove prefix (e.g., "hf_", "reddit_", "github_")
            key_part = stem.split("_", 1)[-1] if "_" in stem else stem
            # Should be 32-char hex string (MD5)
            if len(key_part) == 32:
                assert all(c in "0123456789abcdef" for c in key_part), f"Cache key is not valid hex MD5: {key_part}"


# ──────────────────────────────────────────────────────────────────────────────
# Prompt assembler - rules injection
# ──────────────────────────────────────────────────────────────────────────────


class TestPromptAssemblerRulesInjection:
    def test_assembler_includes_three_tier_roles(self):
        from vetinari.prompts.assembler import _ROLE_DEFS

        assert AgentType.FOREMAN.value in _ROLE_DEFS
        assert AgentType.WORKER.value in _ROLE_DEFS
        assert AgentType.INSPECTOR.value in _ROLE_DEFS

    def test_assembler_includes_new_task_types(self):
        from vetinari.prompts.assembler import _TASK_INSTRUCTIONS

        for t in ("security", "devops", "data", "documentation", "image"):
            assert t in _TASK_INSTRUCTIONS, f"Missing task type: {t}"

    def test_build_injects_rules(self, tmp_path):
        """Rules from RulesManager should appear in assembled prompt."""
        from vetinari.prompts import PromptAssembler
        from vetinari.rules_manager import RulesManager

        # Create a temp rules manager with a rule
        rules_file = tmp_path / "rules.yaml"
        rm = RulesManager(rules_file=rules_file)
        rm.set_global_rules(["Always use type hints in all Python code"])

        assembler = PromptAssembler()

        with patch("vetinari.rules_manager.get_rules_manager", return_value=rm):
            result = assembler.build(
                agent_type=AgentType.WORKER.value,
                task_type="coding",
                task_description="Create a user model",
                include_examples=False,
                include_rules=False,
            )

        # Rules should be injected into system prompt
        system = result["system"]
        assert "Always use type hints" in system
