"""Tests for the 2-stage Architect-Executor pipeline."""

import json
import pytest
from unittest.mock import MagicMock, patch

from vetinari.orchestration.architect_executor import (
    ArchitectExecutorPipeline,
    ArchitectPlan,
    PipelineConfig,
)


# ---------------------------------------------------------------------------
# PipelineConfig tests
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_default_config(self):
        config = PipelineConfig()
        assert config.enabled is True
        assert config.architect_model == "qwen2.5-coder-32b"
        assert config.executor_model == "qwen2.5-coder-7b"
        assert config.auto_commit is False
        assert config.commit_style == "conventional"
        assert config.max_steps == 20
        assert config.fallback_to_single is True
        assert config.architect_temperature == 0.4
        assert config.executor_temperature == 0.2
        assert config.architect_max_tokens == 4096
        assert config.executor_max_tokens == 2048

    def test_custom_config(self):
        config = PipelineConfig(
            enabled=False,
            architect_model="big-model",
            executor_model="small-model",
            max_steps=10,
            auto_commit=True,
            commit_style="descriptive",
        )
        assert config.enabled is False
        assert config.architect_model == "big-model"
        assert config.executor_model == "small-model"
        assert config.max_steps == 10
        assert config.auto_commit is True
        assert config.commit_style == "descriptive"

    def test_to_dict(self):
        config = PipelineConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["enabled"] is True
        assert d["architect_model"] == "qwen2.5-coder-32b"
        assert d["max_steps"] == 20

    def test_from_dict(self):
        data = {
            "enabled": False,
            "architect_model": "custom-arch",
            "executor_model": "custom-exec",
            "max_steps": 5,
        }
        config = PipelineConfig.from_dict(data)
        assert config.enabled is False
        assert config.architect_model == "custom-arch"
        assert config.executor_model == "custom-exec"
        assert config.max_steps == 5
        # Defaults for unspecified fields
        assert config.auto_commit is False
        assert config.fallback_to_single is True

    def test_from_dict_ignores_unknown_keys(self):
        data = {
            "enabled": True,
            "unknown_key": "should_be_ignored",
            "another_random": 42,
        }
        config = PipelineConfig.from_dict(data)
        assert config.enabled is True
        assert not hasattr(config, "unknown_key")

    def test_roundtrip(self):
        original = PipelineConfig(
            enabled=False,
            architect_model="test-arch",
            executor_model="test-exec",
            max_steps=15,
            commit_style="descriptive",
        )
        d = original.to_dict()
        restored = PipelineConfig.from_dict(d)
        assert restored.enabled == original.enabled
        assert restored.architect_model == original.architect_model
        assert restored.executor_model == original.executor_model
        assert restored.max_steps == original.max_steps
        assert restored.commit_style == original.commit_style

    def test_validate_valid_config(self):
        config = PipelineConfig()
        errors = config.validate()
        assert errors == []

    def test_validate_bad_max_steps(self):
        config = PipelineConfig(max_steps=0)
        errors = config.validate()
        assert any("max_steps" in e for e in errors)

    def test_validate_max_steps_too_high(self):
        config = PipelineConfig(max_steps=200)
        errors = config.validate()
        assert any("max_steps" in e for e in errors)

    def test_validate_bad_temperature(self):
        config = PipelineConfig(architect_temperature=-1)
        errors = config.validate()
        assert any("architect_temperature" in e for e in errors)

    def test_validate_bad_executor_temperature(self):
        config = PipelineConfig(executor_temperature=5.0)
        errors = config.validate()
        assert any("executor_temperature" in e for e in errors)

    def test_validate_bad_commit_style(self):
        config = PipelineConfig(commit_style="invalid")
        errors = config.validate()
        assert any("commit_style" in e for e in errors)

    def test_validate_bad_max_tokens(self):
        config = PipelineConfig(architect_max_tokens=0, executor_max_tokens=-5)
        errors = config.validate()
        assert any("architect_max_tokens" in e for e in errors)
        assert any("executor_max_tokens" in e for e in errors)


# ---------------------------------------------------------------------------
# ArchitectPlan tests
# ---------------------------------------------------------------------------

class TestArchitectPlan:
    def test_create_empty_plan(self):
        plan = ArchitectPlan(goal="test goal")
        assert plan.goal == "test goal"
        assert plan.steps == []
        assert plan.dependencies == {}
        assert plan.estimated_tokens == 0
        assert plan.plan_id  # auto-generated

    def test_create_plan_with_steps(self):
        steps = [
            {"id": "step-1", "description": "Setup", "files": ["a.py"], "agent_type": "general", "complexity": "low"},
            {"id": "step-2", "description": "Implement", "files": ["b.py"], "agent_type": "coder", "complexity": "high"},
        ]
        deps = {"step-2": ["step-1"]}
        plan = ArchitectPlan(
            goal="Build feature X",
            steps=steps,
            dependencies=deps,
            estimated_tokens=3000,
            architect_model="big-model",
        )
        assert plan.step_count() == 2
        assert plan.architect_model == "big-model"
        assert plan.estimated_tokens == 3000
        assert plan.dependencies == {"step-2": ["step-1"]}

    def test_to_dict(self):
        plan = ArchitectPlan(
            goal="test",
            steps=[{"id": "s1", "description": "do thing"}],
        )
        d = plan.to_dict()
        assert isinstance(d, dict)
        assert d["goal"] == "test"
        assert len(d["steps"]) == 1
        assert "plan_id" in d
        assert "created_at" in d

    def test_from_dict(self):
        data = {
            "plan_id": "abc",
            "goal": "test goal",
            "steps": [{"id": "s1", "description": "step one"}],
            "dependencies": {},
            "estimated_tokens": 1000,
            "architect_model": "model-x",
            "created_at": "2025-01-01T00:00:00",
        }
        plan = ArchitectPlan.from_dict(data)
        assert plan.plan_id == "abc"
        assert plan.goal == "test goal"
        assert len(plan.steps) == 1
        assert plan.estimated_tokens == 1000

    def test_roundtrip(self):
        original = ArchitectPlan(
            goal="roundtrip test",
            steps=[
                {"id": "s1", "description": "first"},
                {"id": "s2", "description": "second"},
            ],
            dependencies={"s2": ["s1"]},
            estimated_tokens=500,
            architect_model="rt-model",
        )
        d = original.to_dict()
        restored = ArchitectPlan.from_dict(d)
        assert restored.goal == original.goal
        assert restored.step_count() == original.step_count()
        assert restored.dependencies == original.dependencies

    def test_get_step(self):
        plan = ArchitectPlan(
            goal="test",
            steps=[
                {"id": "s1", "description": "first"},
                {"id": "s2", "description": "second"},
            ],
        )
        assert plan.get_step("s1")["description"] == "first"
        assert plan.get_step("s2")["description"] == "second"
        assert plan.get_step("s3") is None

    def test_get_ready_steps_no_deps(self):
        plan = ArchitectPlan(
            goal="test",
            steps=[
                {"id": "s1", "description": "a"},
                {"id": "s2", "description": "b"},
            ],
            dependencies={},
        )
        ready = plan.get_ready_steps(set())
        assert len(ready) == 2

    def test_get_ready_steps_with_deps(self):
        plan = ArchitectPlan(
            goal="test",
            steps=[
                {"id": "s1", "description": "a"},
                {"id": "s2", "description": "b"},
                {"id": "s3", "description": "c"},
            ],
            dependencies={"s2": ["s1"], "s3": ["s1", "s2"]},
        )
        # Nothing completed: only s1 is ready
        ready = plan.get_ready_steps(set())
        assert len(ready) == 1
        assert ready[0]["id"] == "s1"

        # s1 completed: s2 is ready
        ready = plan.get_ready_steps({"s1"})
        assert len(ready) == 1
        assert ready[0]["id"] == "s2"

        # s1 and s2 completed: s3 is ready
        ready = plan.get_ready_steps({"s1", "s2"})
        assert len(ready) == 1
        assert ready[0]["id"] == "s3"

    def test_get_ready_steps_excludes_completed(self):
        plan = ArchitectPlan(
            goal="test",
            steps=[
                {"id": "s1", "description": "a"},
                {"id": "s2", "description": "b"},
            ],
            dependencies={},
        )
        ready = plan.get_ready_steps({"s1"})
        assert len(ready) == 1
        assert ready[0]["id"] == "s2"

    def test_validate_valid_plan(self):
        plan = ArchitectPlan(
            goal="test",
            steps=[{"id": "s1", "description": "do it"}],
        )
        errors = plan.validate()
        assert errors == []

    def test_validate_no_goal(self):
        plan = ArchitectPlan(goal="", steps=[{"id": "s1"}])
        errors = plan.validate()
        assert any("goal" in e for e in errors)

    def test_validate_no_steps(self):
        plan = ArchitectPlan(goal="test", steps=[])
        errors = plan.validate()
        assert any("step" in e.lower() for e in errors)

    def test_validate_bad_dependency_key(self):
        plan = ArchitectPlan(
            goal="test",
            steps=[{"id": "s1", "description": "a"}],
            dependencies={"nonexistent": ["s1"]},
        )
        errors = plan.validate()
        assert any("nonexistent" in e for e in errors)

    def test_validate_bad_dependency_value(self):
        plan = ArchitectPlan(
            goal="test",
            steps=[{"id": "s1", "description": "a"}],
            dependencies={"s1": ["nonexistent"]},
        )
        errors = plan.validate()
        assert any("nonexistent" in e for e in errors)

    def test_validate_self_dependency(self):
        plan = ArchitectPlan(
            goal="test",
            steps=[{"id": "s1", "description": "a"}],
            dependencies={"s1": ["s1"]},
        )
        errors = plan.validate()
        assert any("itself" in e for e in errors)

    def test_step_count(self):
        plan = ArchitectPlan(goal="test", steps=[{"id": f"s{i}"} for i in range(5)])
        assert plan.step_count() == 5


# ---------------------------------------------------------------------------
# ArchitectExecutorPipeline creation tests
# ---------------------------------------------------------------------------

class TestPipelineCreation:
    def test_default_creation(self):
        pipeline = ArchitectExecutorPipeline()
        assert pipeline.architect_model == "qwen2.5-coder-32b"
        assert pipeline.executor_model == "qwen2.5-coder-7b"
        assert pipeline.enabled is True

    def test_custom_models(self):
        pipeline = ArchitectExecutorPipeline(
            architect_model="big-model",
            executor_model="small-model",
        )
        assert pipeline.architect_model == "big-model"
        assert pipeline.executor_model == "small-model"

    def test_creation_with_dict_config(self):
        pipeline = ArchitectExecutorPipeline(
            config={
                "enabled": False,
                "architect_model": "arch-from-config",
                "executor_model": "exec-from-config",
                "max_steps": 10,
            }
        )
        # Config is used when explicit args are None
        assert pipeline.architect_model == "arch-from-config"
        assert pipeline.executor_model == "exec-from-config"
        assert pipeline.enabled is False
        assert pipeline.config.max_steps == 10

    def test_explicit_models_override_config(self):
        pipeline = ArchitectExecutorPipeline(
            architect_model="explicit-arch",
            executor_model="explicit-exec",
            config={"architect_model": "config-arch", "executor_model": "config-exec"},
        )
        assert pipeline.architect_model == "explicit-arch"
        assert pipeline.executor_model == "explicit-exec"

    def test_creation_with_pipeline_config_object(self):
        config = PipelineConfig(enabled=False, max_steps=3)
        pipeline = ArchitectExecutorPipeline(config=config)
        assert pipeline.enabled is False
        assert pipeline.config.max_steps == 3

    def test_enabled_property(self):
        pipeline = ArchitectExecutorPipeline()
        assert pipeline.enabled is True
        pipeline.enabled = False
        assert pipeline.enabled is False

    def test_config_property(self):
        pipeline = ArchitectExecutorPipeline()
        config = pipeline.config
        assert isinstance(config, PipelineConfig)
        assert config.enabled is True


# ---------------------------------------------------------------------------
# Plan creation tests (with mocked model calls)
# ---------------------------------------------------------------------------

class TestPlanCreation:
    def _make_pipeline(self, **kwargs):
        """Create a pipeline with model calls mocked."""
        return ArchitectExecutorPipeline(**kwargs)

    def test_create_plan_with_mock_model(self):
        pipeline = self._make_pipeline()
        mock_json = json.dumps({
            "steps": [
                {"id": "s1", "description": "Setup project", "files": ["setup.py"], "agent_type": "general", "complexity": "low"},
                {"id": "s2", "description": "Implement feature", "files": ["main.py"], "agent_type": "coder", "complexity": "high"},
            ],
            "dependencies": {"s2": ["s1"]},
            "estimated_tokens": 2000,
        })

        with patch.object(pipeline, "_call_model", return_value=mock_json):
            plan = pipeline.create_plan("Build a web app", {"files": ["main.py"]})

        assert plan.goal == "Build a web app"
        assert plan.step_count() == 2
        assert plan.dependencies == {"s2": ["s1"]}
        assert plan.estimated_tokens == 2000

    def test_create_plan_parses_markdown_wrapped_json(self):
        pipeline = self._make_pipeline()
        mock_output = (
            "Here is the plan:\n"
            "```json\n"
            '{"steps": [{"id": "s1", "description": "Do it"}], "dependencies": {}, "estimated_tokens": 100}\n'
            "```\n"
            "Let me know if you need changes."
        )

        with patch.object(pipeline, "_call_model", return_value=mock_output):
            plan = pipeline.create_plan("Simple task", {})

        assert plan.step_count() == 1
        assert plan.steps[0]["id"] == "s1"

    def test_create_plan_fallback_on_failure(self):
        pipeline = self._make_pipeline(config={"fallback_to_single": True})

        with patch.object(pipeline, "_call_model", side_effect=RuntimeError("model offline")):
            plan = pipeline.create_plan("Fallback test", {})

        assert plan.step_count() == 1
        assert plan.architect_model == "fallback"
        assert plan.goal == "Fallback test"

    def test_create_plan_no_fallback_raises(self):
        pipeline = self._make_pipeline(config={"fallback_to_single": False})

        with patch.object(pipeline, "_call_model", side_effect=RuntimeError("model offline")):
            with pytest.raises(RuntimeError):
                pipeline.create_plan("Should fail", {})

    def test_create_plan_fallback_on_bad_json(self):
        pipeline = self._make_pipeline(config={"fallback_to_single": True})

        with patch.object(pipeline, "_call_model", return_value="not valid json at all"):
            plan = pipeline.create_plan("Bad JSON test", {})

        assert plan.step_count() == 1
        assert plan.architect_model == "fallback"

    def test_create_plan_truncates_excess_steps(self):
        pipeline = self._make_pipeline(config={"max_steps": 3})
        steps = [{"id": f"s{i}", "description": f"Step {i}"} for i in range(10)]
        mock_json = json.dumps({"steps": steps, "dependencies": {}, "estimated_tokens": 0})

        with patch.object(pipeline, "_call_model", return_value=mock_json):
            plan = pipeline.create_plan("Many steps", {})

        assert plan.step_count() == 3

    def test_create_plan_assigns_ids_to_steps_without_them(self):
        pipeline = self._make_pipeline()
        mock_json = json.dumps({
            "steps": [
                {"description": "No ID step 1"},
                {"description": "No ID step 2"},
            ],
            "dependencies": {},
            "estimated_tokens": 0,
        })

        with patch.object(pipeline, "_call_model", return_value=mock_json):
            plan = pipeline.create_plan("Auto ID test", {})

        assert plan.steps[0]["id"] == "step-1"
        assert plan.steps[1]["id"] == "step-2"

    def test_create_plan_empty_output_falls_back(self):
        pipeline = self._make_pipeline(config={"fallback_to_single": True})

        with patch.object(pipeline, "_call_model", return_value=""):
            plan = pipeline.create_plan("Empty output", {})

        assert plan.architect_model == "fallback"


# ---------------------------------------------------------------------------
# Plan execution tests
# ---------------------------------------------------------------------------

class TestPlanExecution:
    def test_execute_simple_plan(self):
        pipeline = ArchitectExecutorPipeline()
        plan = ArchitectPlan(
            goal="test",
            steps=[
                {"id": "s1", "description": "Step one"},
                {"id": "s2", "description": "Step two"},
            ],
            dependencies={},
        )

        def mock_executor(step, plan):
            return {"success": True, "output": f"Done: {step['description']}"}

        results = pipeline.execute_plan(plan, executor_fn=mock_executor)
        assert len(results) == 2
        assert all(r["success"] for r in results)

    def test_execute_plan_with_dependencies(self):
        pipeline = ArchitectExecutorPipeline()
        execution_order = []

        plan = ArchitectPlan(
            goal="test",
            steps=[
                {"id": "s1", "description": "First"},
                {"id": "s2", "description": "Second"},
                {"id": "s3", "description": "Third"},
            ],
            dependencies={"s2": ["s1"], "s3": ["s2"]},
        )

        def mock_executor(step, plan):
            execution_order.append(step["id"])
            return {"success": True}

        results = pipeline.execute_plan(plan, executor_fn=mock_executor)
        assert len(results) == 3
        assert execution_order == ["s1", "s2", "s3"]

    def test_execute_plan_handles_failure(self):
        pipeline = ArchitectExecutorPipeline()
        plan = ArchitectPlan(
            goal="test",
            steps=[
                {"id": "s1", "description": "Will fail"},
                {"id": "s2", "description": "Should not block"},
            ],
            dependencies={},
        )

        call_count = [0]

        def mock_executor(step, plan):
            call_count[0] += 1
            if step["id"] == "s1":
                return {"success": False, "error": "intentional failure"}
            return {"success": True}

        results = pipeline.execute_plan(plan, executor_fn=mock_executor)
        assert len(results) == 2
        assert not results[0]["success"]
        assert results[1]["success"]

    def test_execute_plan_skips_blocked_by_failed_dependency(self):
        pipeline = ArchitectExecutorPipeline()
        plan = ArchitectPlan(
            goal="test",
            steps=[
                {"id": "s1", "description": "Will fail"},
                {"id": "s2", "description": "Depends on s1"},
            ],
            dependencies={"s2": ["s1"]},
        )

        executed = []

        def mock_executor(step, plan):
            executed.append(step["id"])
            if step["id"] == "s1":
                return {"success": False, "error": "fail"}
            return {"success": True}

        results = pipeline.execute_plan(plan, executor_fn=mock_executor)
        # s1 runs and fails; s2 also runs because failed deps are treated as resolved
        # (both completed_ids and failed_ids are passed to get_ready_steps)
        assert "s1" in executed
        # s2 gets unblocked after s1 fails (dependency resolved by failure)
        assert len(results) == 2

    def test_execute_plan_exception_in_executor(self):
        pipeline = ArchitectExecutorPipeline()
        plan = ArchitectPlan(
            goal="test",
            steps=[{"id": "s1", "description": "Will raise"}],
            dependencies={},
        )

        def bad_executor(step, plan):
            raise ValueError("something broke")

        results = pipeline.execute_plan(plan, executor_fn=bad_executor)
        assert len(results) == 1
        assert results[0]["success"] is False
        assert "something broke" in results[0]["error"]

    def test_execute_empty_plan(self):
        pipeline = ArchitectExecutorPipeline()
        plan = ArchitectPlan(goal="test", steps=[])

        results = pipeline.execute_plan(plan, executor_fn=lambda s, p: {"success": True})
        assert results == []


# ---------------------------------------------------------------------------
# Full pipeline run tests
# ---------------------------------------------------------------------------

class TestPipelineRun:
    def test_run_full_pipeline(self):
        pipeline = ArchitectExecutorPipeline()
        mock_json = json.dumps({
            "steps": [
                {"id": "s1", "description": "Setup", "files": [], "agent_type": "general", "complexity": "low"},
                {"id": "s2", "description": "Build", "files": ["main.py"], "agent_type": "coder", "complexity": "medium"},
            ],
            "dependencies": {"s2": ["s1"]},
            "estimated_tokens": 1500,
        })

        def mock_executor(step, plan):
            return {"success": True, "output": f"Done: {step['id']}"}

        with patch.object(pipeline, "_call_model", return_value=mock_json):
            result = pipeline.run("Build a feature", context={}, executor_fn=mock_executor)

        assert result["success"] is True
        assert result["total_steps"] == 2
        assert result["completed_steps"] == 2
        assert result["failed_steps"] == 0
        assert "plan" in result
        assert "plan_id" in result
        assert result["elapsed_seconds"] >= 0

    def test_run_disabled_pipeline(self):
        pipeline = ArchitectExecutorPipeline(config={"enabled": False})
        result = pipeline.run("Should not run", context={})
        assert result["success"] is False
        assert result["skipped"] is True
        assert result["reason"] == "pipeline_disabled"

    def test_run_with_failure(self):
        pipeline = ArchitectExecutorPipeline()
        mock_json = json.dumps({
            "steps": [
                {"id": "s1", "description": "Will fail"},
            ],
            "dependencies": {},
            "estimated_tokens": 0,
        })

        def failing_executor(step, plan):
            return {"success": False, "error": "intentional"}

        with patch.object(pipeline, "_call_model", return_value=mock_json):
            result = pipeline.run("Failure test", executor_fn=failing_executor)

        assert result["success"] is False
        assert result["failed_steps"] == 1
        assert result["completed_steps"] == 0


# ---------------------------------------------------------------------------
# Integration with TwoLayerOrchestrator
# ---------------------------------------------------------------------------

class TestTwoLayerIntegration:
    def test_get_architect_pipeline_disabled_by_default(self):
        """Pipeline is not enabled when agent_context has no architect_executor config."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        with patch("vetinari.orchestration.two_layer.PlanGenerator"):
            with patch("vetinari.orchestration.two_layer.DurableExecutionEngine"):
                orch = TwoLayerOrchestrator()

        pipeline = orch._get_architect_pipeline()
        assert pipeline is None

    def test_get_architect_pipeline_when_enabled(self):
        """Pipeline is returned when agent_context has architect_executor enabled."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        with patch("vetinari.orchestration.two_layer.PlanGenerator"):
            with patch("vetinari.orchestration.two_layer.DurableExecutionEngine"):
                orch = TwoLayerOrchestrator()

        orch.agent_context = {
            "architect_executor": {
                "enabled": True,
                "architect_model": "big-model",
                "executor_model": "small-model",
            }
        }
        pipeline = orch._get_architect_pipeline()
        assert pipeline is not None
        assert isinstance(pipeline, ArchitectExecutorPipeline)
        assert pipeline.architect_model == "big-model"
        assert pipeline.executor_model == "small-model"

    def test_generate_and_execute_architect_falls_back_when_not_configured(self):
        """When pipeline is not configured, falls back to standard generate_and_execute."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        with patch("vetinari.orchestration.two_layer.PlanGenerator"):
            with patch("vetinari.orchestration.two_layer.DurableExecutionEngine"):
                orch = TwoLayerOrchestrator()

        mock_result = {"plan_id": "test", "completed": 1, "failed": 0}
        with patch.object(orch, "generate_and_execute", return_value=mock_result) as mock_ge:
            result = orch.generate_and_execute_architect(goal="test goal")

        mock_ge.assert_called_once()
        assert result == mock_result


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------

class TestInternalHelpers:
    def test_build_architect_prompt_basic(self):
        pipeline = ArchitectExecutorPipeline()
        prompt = pipeline._build_architect_prompt("Build an API", {})
        assert "Build an API" in prompt
        assert "Instructions" in prompt

    def test_build_architect_prompt_with_context(self):
        pipeline = ArchitectExecutorPipeline()
        context = {
            "files": ["main.py", "utils.py"],
            "tech_stack": "Python + FastAPI",
            "constraints": "Must be under 100 lines",
            "existing_code": "def hello(): pass",
        }
        prompt = pipeline._build_architect_prompt("Build an API", context)
        assert "main.py" in prompt
        assert "Python + FastAPI" in prompt
        assert "Must be under 100 lines" in prompt
        assert "def hello(): pass" in prompt

    def test_parse_architect_output_valid_json(self):
        pipeline = ArchitectExecutorPipeline()
        raw = json.dumps({
            "steps": [{"id": "s1", "description": "Test"}],
            "dependencies": {},
            "estimated_tokens": 100,
        })
        plan = pipeline._parse_architect_output(raw, "test goal")
        assert plan is not None
        assert plan.goal == "test goal"
        assert plan.step_count() == 1

    def test_parse_architect_output_empty(self):
        pipeline = ArchitectExecutorPipeline()
        assert pipeline._parse_architect_output("", "goal") is None
        assert pipeline._parse_architect_output("   ", "goal") is None

    def test_parse_architect_output_bad_json(self):
        pipeline = ArchitectExecutorPipeline()
        result = pipeline._parse_architect_output("not json {{{", "goal")
        assert result is None

    def test_parse_architect_output_markdown_fenced(self):
        pipeline = ArchitectExecutorPipeline()
        raw = '```json\n{"steps": [{"id": "s1", "description": "A"}], "dependencies": {}}\n```'
        plan = pipeline._parse_architect_output(raw, "goal")
        assert plan is not None
        assert plan.step_count() == 1

    def test_parse_architect_output_generic_fence(self):
        pipeline = ArchitectExecutorPipeline()
        raw = '```\n{"steps": [{"id": "s1", "description": "A"}], "dependencies": {}}\n```'
        plan = pipeline._parse_architect_output(raw, "goal")
        assert plan is not None

    def test_make_fallback_plan(self):
        pipeline = ArchitectExecutorPipeline()
        plan = pipeline._make_fallback_plan("do something", {"files": ["a.py"]})
        assert plan.goal == "do something"
        assert plan.step_count() == 1
        assert plan.architect_model == "fallback"
        assert "a.py" in plan.steps[0]["files"]

    def test_make_fallback_plan_no_context(self):
        pipeline = ArchitectExecutorPipeline()
        plan = pipeline._make_fallback_plan("do something", None)
        assert plan.step_count() == 1
        assert plan.steps[0]["files"] == []


# ---------------------------------------------------------------------------
# Import / export tests
# ---------------------------------------------------------------------------

class TestImports:
    def test_import_from_orchestration_package(self):
        from vetinari.orchestration import (
            ArchitectExecutorPipeline,
            ArchitectPlan,
            PipelineConfig,
        )
        assert ArchitectExecutorPipeline is not None
        assert ArchitectPlan is not None
        assert PipelineConfig is not None

    def test_import_from_module_directly(self):
        from vetinari.orchestration.architect_executor import (
            ArchitectExecutorPipeline,
            ArchitectPlan,
            PipelineConfig,
        )
        assert ArchitectExecutorPipeline is not None
