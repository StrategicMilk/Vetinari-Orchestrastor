"""Tests for exception handlers in orchestration files.

Covers all try/except blocks in:
- vetinari/orchestration/architect_executor.py (5 handlers)
- vetinari/orchestration/express_path.py (4 handlers)
- vetinari/orchestration/task_manifest.py (5 handlers)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.orchestration.architect_executor import ArchitectExecutorPipeline, ArchitectPlan, PipelineConfig
from vetinari.orchestration.express_path import ExpressPathMixin
from vetinari.orchestration.task_manifest import ManifestBuilder
from vetinari.types import AgentType

# ---------------------------------------------------------------------------
# architect_executor.py handlers
# ---------------------------------------------------------------------------


class TestArchitectExecutorHandlers:
    """Exception handlers in ArchitectExecutorPipeline."""

    # Handler 1: line ~298 — architect model call raises, fallback_to_single=True
    def test_create_plan_model_failure_falls_back_to_single_step(self) -> None:
        """When the architect model call raises and fallback_to_single is True,
        create_plan returns a single-step fallback plan instead of propagating."""
        pipeline = ArchitectExecutorPipeline(
            config={"fallback_to_single": True},
        )
        with patch.object(pipeline, "_call_model", side_effect=RuntimeError("model offline")):
            plan = pipeline.create_plan("Do something", {})

        assert plan is not None
        assert len(plan.steps) == 1
        assert plan.steps[0]["id"] == "step-1"
        assert plan.architect_model == "fallback"

    def test_create_plan_model_failure_raises_when_fallback_disabled(self) -> None:
        """When fallback_to_single is False and the model call raises, the
        exception propagates out of create_plan."""
        pipeline = ArchitectExecutorPipeline(
            config={"fallback_to_single": False},
        )
        with patch.object(pipeline, "_call_model", side_effect=RuntimeError("model offline")):
            with pytest.raises(RuntimeError, match="model offline"):
                pipeline.create_plan("Do something", {})

    # Handler 2: line ~388 — executor step raises, execution continues
    def test_execute_plan_step_exception_appends_error_result(self) -> None:
        """When an executor step raises an exception, the error is captured into
        a result dict (success=False) and remaining steps continue."""
        pipeline = ArchitectExecutorPipeline()
        plan = ArchitectPlan(
            goal="test goal",
            steps=[
                {"id": "step-1", "description": "step one"},
                {"id": "step-2", "description": "step two"},
            ],
            dependencies={},
        )

        call_count = 0

        def failing_first_then_ok(step: dict, _plan: ArchitectPlan) -> dict:
            nonlocal call_count
            call_count += 1
            if step["id"] == "step-1":
                raise ValueError("step-1 exploded")
            return {"success": True, "output": "done"}

        results = pipeline.execute_plan(plan, executor_fn=failing_first_then_ok)

        step1_result = next(r for r in results if r["step_id"] == "step-1")
        assert step1_result["success"] is False
        assert "step-1 exploded" in step1_result["error"]
        # step-2 has no dependency on step-1, so it should also run
        assert any(r["step_id"] == "step-2" for r in results)

    def test_execute_plan_step_exception_marks_step_as_failed(self) -> None:
        """A step that raises must end up in the failed set, not completed."""
        pipeline = ArchitectExecutorPipeline()
        plan = ArchitectPlan(
            goal="test",
            steps=[{"id": "s1", "description": "task"}],
            dependencies={},
        )

        results = pipeline.execute_plan(
            plan,
            executor_fn=lambda step, _plan: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        assert len(results) == 1
        assert results[0]["success"] is False
        assert results[0]["step_id"] == "s1"

    # Handler 3: line ~523 — adapter_manager inference raises, fallback used
    def test_call_model_adapter_manager_failure_falls_back_to_local_adapter(self) -> None:
        """When adapter_manager.infer() raises, _call_model catches the error
        and falls back to LocalInferenceAdapter."""
        pipeline = ArchitectExecutorPipeline()

        failing_manager = MagicMock()
        failing_manager.infer.side_effect = RuntimeError("adapter exploded")

        context = {"adapter_manager": failing_manager}

        mock_adapter_instance = MagicMock()
        mock_adapter_instance.chat.return_value = {"output": "fallback output"}
        mock_adapter_cls = MagicMock(return_value=mock_adapter_instance)

        mock_llama_module = MagicMock()
        mock_llama_module.LocalInferenceAdapter = mock_adapter_cls

        with patch.dict("sys.modules", {"vetinari.adapters.llama_cpp_local_adapter": mock_llama_module}):
            output = pipeline._call_model(
                model="some-model",
                prompt="hello",
                context=context,
            )

        assert output == "fallback output"
        failing_manager.infer.assert_called_once()

    def test_call_model_adapter_manager_returns_non_ok_status_falls_back(self) -> None:
        """When adapter_manager.infer() returns a non-OK status, _call_model
        does not raise but also does not return that output — falls back instead."""
        pipeline = ArchitectExecutorPipeline()

        mock_response = MagicMock()
        mock_response.status = "error"
        mock_response.output = "bad output"

        mock_manager = MagicMock()
        mock_manager.infer.return_value = mock_response

        context = {"adapter_manager": mock_manager}

        mock_adapter_instance = MagicMock()
        mock_adapter_instance.chat.return_value = {"output": "local output"}
        mock_adapter_cls = MagicMock(return_value=mock_adapter_instance)

        mock_llama_module = MagicMock()
        mock_llama_module.LocalInferenceAdapter = mock_adapter_cls

        with patch.dict("sys.modules", {"vetinari.adapters.llama_cpp_local_adapter": mock_llama_module}):
            output = pipeline._call_model(
                model="some-model",
                prompt="hello",
                context=context,
            )

        assert output == "local output"

    # Handler 4: line ~557 — JSONDecodeError in _parse_architect_output returns None
    def test_parse_architect_output_invalid_json_returns_none(self) -> None:
        """When the model output cannot be parsed as JSON, _parse_architect_output
        returns None instead of raising."""
        pipeline = ArchitectExecutorPipeline()

        result = pipeline._parse_architect_output("this is not json {{{{", "my goal")

        assert result is None

    def test_parse_architect_output_empty_string_returns_none(self) -> None:
        """Empty output is treated the same as invalid JSON — returns None."""
        pipeline = ArchitectExecutorPipeline()

        result = pipeline._parse_architect_output("", "my goal")

        assert result is None

    def test_parse_architect_output_valid_json_returns_plan(self) -> None:
        """Valid JSON is parsed successfully and produces a non-None plan."""
        pipeline = ArchitectExecutorPipeline()
        raw = '{"steps": [{"id": "s1", "description": "do it"}], "dependencies": {}}'

        result = pipeline._parse_architect_output(raw, "my goal")

        assert result is not None
        assert result.goal == "my goal"
        assert len(result.steps) == 1

    # Handler 5: line ~621 — _default_execute_step _call_model raises, returns error dict
    def test_default_execute_step_model_failure_returns_error_dict(self) -> None:
        """When _call_model raises inside _default_execute_step, the method
        returns an error dict with success=False instead of propagating."""
        pipeline = ArchitectExecutorPipeline()
        step = {"id": "s1", "description": "implement feature", "files": ["main.py"]}
        plan = ArchitectPlan(goal="test", steps=[step], dependencies={})

        with patch.object(pipeline, "_call_model", side_effect=OSError("disk full")):
            result = pipeline._default_execute_step(step, plan)

        assert result["success"] is False
        assert "disk full" in result["error"]
        assert result["step_id"] == "s1"

    def test_default_execute_step_success_returns_output(self) -> None:
        """When _call_model succeeds, _default_execute_step returns success dict."""
        pipeline = ArchitectExecutorPipeline()
        step = {"id": "s1", "description": "write tests"}
        plan = ArchitectPlan(goal="test", steps=[step], dependencies={})

        with patch.object(pipeline, "_call_model", return_value="test output"):
            result = pipeline._default_execute_step(step, plan)

        assert result["success"] is True
        assert result["output"] == "test output"


# ---------------------------------------------------------------------------
# express_path.py handlers
# ---------------------------------------------------------------------------


class _ConcreteExpressMixin(ExpressPathMixin):
    """Concrete subclass that provides the attributes ExpressPathMixin expects."""

    def __init__(self) -> None:
        # Do NOT pre-initialize _express_metrics; _record_express_metrics does
        # a hasattr check and initializes it on first call (same as production).
        pass  # noqa: VET031 - empty body is intentional test double behavior

    def _make_default_handler(self):  # type: ignore[override]
        """Return a handler that does nothing."""
        return lambda task: "ok"


class TestExpressPathHandlers:
    """Exception handlers in ExpressPathMixin."""

    # Handler 1: line ~92 — task handler raises, caught and error dict returned
    def test_execute_express_handler_exception_returns_error_dict(self) -> None:
        """When the task handler raises, _execute_express catches the exception
        and returns an error result dict rather than propagating."""
        mixin = _ConcreteExpressMixin()
        stages: dict = {}

        def boom(_task):
            raise ValueError("handler blew up")

        result = mixin._execute_express(
            goal="do work",
            context={},
            stages=stages,
            start_time=0.0,
            corr_ctx=None,
            pipeline_span=None,
            task_handler=boom,
        )

        assert result["tier"] == "express"
        assert result["failed"] == 1
        assert result["completed"] == 0
        assert "handler blew up" in result["error"]
        assert stages["express_execution"]["success"] is False

    def test_execute_express_handler_success_returns_success_dict(self) -> None:
        """Successful handler execution produces completed=1, failed=0."""
        mixin = _ConcreteExpressMixin()

        result = mixin._execute_express(
            goal="do work",
            context={},
            stages={},
            start_time=0.0,
            corr_ctx=None,
            pipeline_span=None,
            task_handler=lambda task: "result text",
        )

        assert result["completed"] == 1
        assert result["failed"] == 0
        assert result["final_output"] == "result text"

    # Handler 2: line ~114 — ImportError/AttributeError closing genai span, does not propagate
    def test_execute_express_span_close_import_error_does_not_propagate(self) -> None:
        """If closing the GenAI span raises ImportError or AttributeError, the
        exception is swallowed and the result is still returned."""
        mixin = _ConcreteExpressMixin()

        # Simulate the module not being installed by setting it to None in sys.modules,
        # which causes `from vetinari.observability.otel_genai import ...` to raise ImportError.
        with patch.dict("sys.modules", {"vetinari.observability.otel_genai": None}):
            # Should not raise despite the ImportError in the finally block
            result = mixin._execute_express(
                goal="work",
                context={},
                stages={},
                start_time=0.0,
                corr_ctx=None,
                pipeline_span=MagicMock(),
                task_handler=lambda task: "done",
            )

        # Result is still valid
        assert result["tier"] == "express"

    def test_execute_express_span_close_attribute_error_does_not_propagate(self) -> None:
        """AttributeError from a broken span object is caught in the finally block."""
        mixin = _ConcreteExpressMixin()

        mock_tracer = MagicMock()
        mock_tracer.end_agent_span.side_effect = AttributeError("no end_agent_span")

        mock_otel_module = MagicMock()
        mock_otel_module.get_genai_tracer.return_value = mock_tracer

        # The finally block does a local import; patch via sys.modules
        with patch.dict("sys.modules", {"vetinari.observability.otel_genai": mock_otel_module}):
            result = mixin._execute_express(
                goal="work",
                context={},
                stages={},
                start_time=0.0,
                corr_ctx=None,
                pipeline_span=MagicMock(),
                task_handler=lambda task: "ok",
            )

        assert result["tier"] == "express"

    # Handler 3: line ~154 — ArithmeticError/AttributeError in _record_express_metrics, no propagation
    def test_record_express_metrics_arithmetic_error_does_not_propagate(self) -> None:
        """ArithmeticError during metrics math is caught; metrics stay unmodified."""
        mixin = _ConcreteExpressMixin()

        # Force ArithmeticError by making time.time raise it inside the try block
        with patch("vetinari.orchestration.express_path.time") as mock_time:
            mock_time.time.side_effect = ArithmeticError("overflow")
            mixin._record_express_metrics(True, 0.0)
        # The error was swallowed — metrics were never initialized because time.time
        # raised before the counter increment, so no _express_metrics dict exists.
        assert not hasattr(mixin, "_express_metrics")

    def test_record_express_metrics_attribute_error_does_not_propagate(self) -> None:
        """AttributeError during metrics update is caught; broken metrics object unchanged."""
        mixin = _ConcreteExpressMixin()

        # Pre-initialize _express_metrics with a broken type so attribute access
        # inside the try block raises AttributeError when it tries to increment.
        broken = MagicMock()
        broken.__getitem__ = MagicMock(side_effect=AttributeError("broken item"))
        mixin._express_metrics = broken  # type: ignore[attr-defined]

        mixin._record_express_metrics(True, 0.0)
        # The broken object was not replaced — the error was swallowed silently.
        assert mixin._express_metrics is broken  # type: ignore[attr-defined]

    def test_record_express_metrics_increments_counters(self) -> None:
        """Success flag increments the success counter; failure increments failed."""
        from vetinari.types import StatusEnum

        mixin = _ConcreteExpressMixin()

        mixin._record_express_metrics(True, 0.0)
        mixin._record_express_metrics(False, 0.0)
        mixin._record_express_metrics(True, 0.0)

        metrics = mixin._express_metrics  # type: ignore[attr-defined]
        assert metrics["total"] == 3
        assert metrics["success"] == 2
        assert metrics[StatusEnum.FAILED.value] == 1

    def test_execute_express_structured_failure_dict_counts_as_failed(self) -> None:
        """A handler that returns {"success": False, ...} must be treated as a failure.

        A non-empty dict is truthy in Python, so naive bool(result) would classify
        this as a success.  The fix inspects the "success" key explicitly so that
        an explicit structured failure is counted as failed=1, completed=0.
        """
        mixin = _ConcreteExpressMixin()
        stages: dict = {}

        def failing_handler(_task):
            return {"success": False, "error": "handler decided not to succeed"}

        result = mixin._execute_express(
            goal="test structured failure",
            context={},
            stages=stages,
            start_time=0.0,
            corr_ctx=None,
            pipeline_span=None,
            task_handler=failing_handler,
        )

        assert result["completed"] == 0
        assert result["failed"] == 1
        assert stages["express_execution"]["success"] is False


# ---------------------------------------------------------------------------
# task_manifest.py handlers
# ---------------------------------------------------------------------------


class TestManifestBuilderHandlers:
    """Exception handlers in ManifestBuilder._get_* methods."""

    # Handler 1: _get_rules — ImportError/AttributeError/KeyError/ValueError → returns []
    def test_get_rules_import_error_returns_empty_list(self) -> None:
        """ImportError from rules_manager is caught; empty list returned."""
        builder = ManifestBuilder()

        with patch(
            "vetinari.orchestration.task_manifest.ManifestBuilder._get_rules",
            wraps=builder._get_rules,
        ):
            with patch.dict("sys.modules", {"vetinari.rules_manager": None}):
                result = builder._get_rules(AgentType.WORKER.value, None)

        assert result == []

    def test_get_rules_attribute_error_returns_empty_list(self) -> None:
        """AttributeError on the rules manager is caught; empty list returned."""
        builder = ManifestBuilder()

        mock_manager = MagicMock()
        mock_manager.get_rules_for_context.side_effect = AttributeError("no method")

        with patch("vetinari.orchestration.task_manifest.get_rules_manager", return_value=mock_manager, create=True):
            with patch("vetinari.orchestration.task_manifest.ManifestBuilder._get_rules", wraps=builder._get_rules):
                # Call directly with the import patched
                with patch("vetinari.rules_manager.get_rules_manager", return_value=mock_manager, create=True):
                    result = builder._get_rules(AgentType.WORKER.value, None)

        assert result == []

    def test_get_rules_key_error_returns_empty_list(self) -> None:
        """KeyError from rules manager is caught; empty list returned."""
        builder = ManifestBuilder()

        mock_manager = MagicMock()
        mock_manager.get_rules_for_context.side_effect = KeyError("missing key")

        result = _call_get_rules_with_mock(builder, mock_manager)

        assert result == []

    def test_get_rules_value_error_returns_empty_list(self) -> None:
        """ValueError from rules manager is caught; empty list returned."""
        builder = ManifestBuilder()

        mock_manager = MagicMock()
        mock_manager.get_rules_for_context.side_effect = ValueError("bad value")

        with patch("vetinari.rules_manager.get_rules_manager", return_value=mock_manager, create=True):
            result = _call_get_rules_with_mock(builder, mock_manager)

        assert result == []

    # Handler 2: _get_constraints — ImportError/AttributeError/KeyError/ValueError → returns {}
    def test_get_constraints_import_error_returns_empty_dict(self) -> None:
        """ImportError from standards_loader is caught; empty dict returned."""
        builder = ManifestBuilder()

        with patch.dict("sys.modules", {"vetinari.config.standards_loader": None}):
            result = builder._get_constraints(AgentType.WORKER.value)

        assert result == {}

    def test_get_constraints_attribute_error_returns_empty_dict(self) -> None:
        """AttributeError on the standards loader is caught; empty dict returned."""
        builder = ManifestBuilder()
        mock_loader = MagicMock()
        mock_loader.get_constraints.side_effect = AttributeError("no method")

        result = _call_get_constraints_with_mock(builder, mock_loader)

        assert result == {}

    def test_get_constraints_key_error_returns_empty_dict(self) -> None:
        """KeyError from standards loader is caught; empty dict returned."""
        builder = ManifestBuilder()
        mock_loader = MagicMock()
        mock_loader.get_constraints.side_effect = KeyError("bad key")

        result = _call_get_constraints_with_mock(builder, mock_loader)

        assert result == {}

    # Handler 3: _get_verification — ImportError/AttributeError/KeyError/ValueError → returns []
    def test_get_verification_import_error_returns_empty_list(self) -> None:
        """ImportError from standards_loader is caught; empty list returned."""
        builder = ManifestBuilder()

        with patch.dict("sys.modules", {"vetinari.config.standards_loader": None}):
            result = builder._get_verification("build")

        assert result == []

    def test_get_verification_value_error_returns_empty_list(self) -> None:
        """ValueError from standards loader is caught; empty list returned."""
        builder = ManifestBuilder()
        mock_loader = MagicMock()
        mock_loader.get_verification_checklist.side_effect = ValueError("bad mode")

        result = _call_get_verification_with_mock(builder, mock_loader)

        assert result == []

    # Handler 4: _get_defect_warnings — ImportError/AttributeError/KeyError/ValueError → returns []
    def test_get_defect_warnings_import_error_returns_empty_list(self) -> None:
        """ImportError from standards_loader is caught; empty list returned."""
        builder = ManifestBuilder()

        with patch.dict("sys.modules", {"vetinari.config.standards_loader": None}):
            result = builder._get_defect_warnings(AgentType.WORKER.value)

        assert result == []

    def test_get_defect_warnings_attribute_error_returns_empty_list(self) -> None:
        """AttributeError from standards_loader is caught; empty list returned."""
        builder = ManifestBuilder()
        mock_loader = MagicMock()
        mock_loader.get_defect_warnings.side_effect = AttributeError("no attr")

        result = _call_get_defect_warnings_with_mock(builder, mock_loader)

        assert result == []

    # Handler 5: _get_episodes — broad Exception → returns []
    def test_get_episodes_runtime_error_returns_empty_list(self) -> None:
        """Any exception from the memory store is caught; empty list returned."""
        builder = ManifestBuilder()

        mock_store = MagicMock()
        mock_store.recall_episodes.side_effect = RuntimeError("DB unavailable")

        with patch("vetinari.memory.unified.get_unified_memory_store", return_value=mock_store, create=True):
            result = _call_get_episodes_with_mock(builder, mock_store)

        assert result == []

    def test_get_episodes_import_error_returns_empty_list(self) -> None:
        """ImportError from unified memory is caught; empty list returned."""
        builder = ManifestBuilder()

        with patch.dict("sys.modules", {"vetinari.memory.unified": None}):
            result = builder._get_episodes("some task description")

        assert result == []

    def test_build_succeeds_when_all_sources_fail(self) -> None:
        """ManifestBuilder.build() returns a valid manifest even when every
        data source raises, because each handler has its own fallback."""
        builder = ManifestBuilder()

        with (
            patch.dict(
                "sys.modules",
                {
                    "vetinari.rules_manager": None,
                    "vetinari.config.standards_loader": None,
                    "vetinari.memory.unified": None,
                },
            ),
        ):
            manifest = builder.build("implement feature X", AgentType.WORKER.value, "build")

        assert manifest.task_spec == "implement feature X"
        assert manifest.relevant_rules == []
        assert manifest.constraints == {}
        assert manifest.verification_checklist == []
        assert manifest.defect_warnings == []
        assert manifest.relevant_episodes == []
        # Escalation triggers are static — always populated
        assert len(manifest.escalation_triggers) > 0
        # Hash is still computed
        assert len(manifest.manifest_hash) == 64


# ---------------------------------------------------------------------------
# Private helpers used to inject mocks into late imports
# ---------------------------------------------------------------------------


def _call_get_rules_with_mock(builder: ManifestBuilder, mock_manager: MagicMock) -> list[str]:
    """Call _get_rules with the rules_manager module patched."""
    mock_module = MagicMock()
    mock_module.get_rules_manager.return_value = mock_manager
    with patch.dict("sys.modules", {"vetinari.rules_manager": mock_module}):
        return builder._get_rules(AgentType.WORKER.value, None)


def _call_get_constraints_with_mock(builder: ManifestBuilder, mock_loader: MagicMock) -> dict:
    """Call _get_constraints with the standards_loader module patched."""
    mock_module = MagicMock()
    mock_module.get_standards_loader.return_value = mock_loader
    with patch.dict("sys.modules", {"vetinari.config.standards_loader": mock_module}):
        return builder._get_constraints(AgentType.WORKER.value)


def _call_get_verification_with_mock(builder: ManifestBuilder, mock_loader: MagicMock) -> list[str]:
    """Call _get_verification with the standards_loader module patched."""
    mock_module = MagicMock()
    mock_module.get_standards_loader.return_value = mock_loader
    with patch.dict("sys.modules", {"vetinari.config.standards_loader": mock_module}):
        return builder._get_verification("build")


def _call_get_defect_warnings_with_mock(builder: ManifestBuilder, mock_loader: MagicMock) -> list[str]:
    """Call _get_defect_warnings with the standards_loader module patched."""
    mock_module = MagicMock()
    mock_module.get_standards_loader.return_value = mock_loader
    with patch.dict("sys.modules", {"vetinari.config.standards_loader": mock_module}):
        return builder._get_defect_warnings(AgentType.WORKER.value)


def _call_get_episodes_with_mock(builder: ManifestBuilder, mock_store: MagicMock) -> list[dict]:
    """Call _get_episodes with the unified memory module patched."""
    mock_module = MagicMock()
    mock_module.get_unified_memory_store.return_value = mock_store
    with patch.dict("sys.modules", {"vetinari.memory.unified": mock_module}):
        return builder._get_episodes("some task")
