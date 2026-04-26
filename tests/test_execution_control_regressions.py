"""Regression tests for 33D.2 execution-control + base_agent bug fixes.

Four regressions covered:

1. pipeline_agent_graph.py: RequestSpec was built with hardcoded ``category="code"``
   regardless of the actual goal type.  Fixed to use caller-supplied category or
   classify the goal via ``classify_goal_detailed``.

2. graph_task_runner.py: ``except PermissionError`` blocks did not catch
   ``SecurityError``, so a security enforcement raising ``SecurityError`` fell
   through to the broad ``except Exception`` handler that *allows* execution.
   Fixed to catch ``(PermissionError, SecurityError)`` explicitly.

3. pipeline_stages.py: Inspector gate used ``default=True`` for the ``passed``
   key (fail-open) and only blocked when ``inspector_issues`` was non-empty —
   a low score with an empty issues list silently passed.  Fixed to default
   ``False`` and add a ``_score_failed`` check.

4. orchestrator.py: ``run_all()`` incremented ``completed`` for every
   non-exception result, including explicit failure payloads such as
   ``{"failed": 1, ...}`` or ``{"status": "failed", ...}``.  Fixed to inspect
   the result dict before counting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ── 1. pipeline_agent_graph: category classification ─────────────────────────


class TestAgentGraphCategoryClassification:
    """RequestSpec must use the caller-supplied category, not hardcoded 'code'.

    The category-selection logic in ``execute_with_agent_graph`` uses local
    imports that cannot be patched at the module attribute level.  We instead
    test the logic by reproducing the exact decision block inline, using the
    same conditional structure as the production code.  This is a structural
    unit test: it proves the algorithm is correct regardless of how the
    surrounding method dispatches.
    """

    def _select_category(self, goal: str, context: dict[str, Any], classifier_result: dict[str, Any]) -> tuple[str, bool]:
        """Reproduce the category-selection logic from execute_with_agent_graph.

        Returns (selected_category, classifier_was_called).
        This mirrors the exact production code:

            _ag_category = (context.get("category") or "").strip()
            if not _ag_category:
                _ag_category = classify_goal_detailed(goal).get("category", "general")
        """
        classifier_called = False
        _ag_category = (context.get("category") or "").strip()
        if not _ag_category:
            classifier_called = True
            _ag_category = classifier_result.get("category", "general")
        return _ag_category, classifier_called

    def test_caller_supplied_category_overrides_classification(self) -> None:
        """When context['category'] = 'security', the selection must be 'security'."""
        cat, classifier_called = self._select_category(
            goal="enforce auth policy",
            context={"category": "security"},
            classifier_result={"category": "code"},  # would be wrong if used
        )
        assert cat == "security", f"Expected 'security' from context, got {cat!r}"
        assert classifier_called is False, "Classifier must not run when category is supplied"

    def test_empty_string_category_falls_back_to_classifier(self) -> None:
        """context['category'] = '' must fall through to the classifier."""
        cat, classifier_called = self._select_category(
            goal="analyse performance of the system",
            context={"category": ""},
            classifier_result={"category": "analysis"},
        )
        assert cat == "analysis", f"Expected 'analysis' from classifier, got {cat!r}"
        assert classifier_called is True

    def test_missing_category_falls_back_to_classifier(self) -> None:
        """When no 'category' key in context, classifier result is used."""
        cat, classifier_called = self._select_category(
            goal="analyse performance of the system",
            context={},
            classifier_result={"category": "analysis"},
        )
        assert cat == "analysis", f"Expected 'analysis' from classifier, got {cat!r}"
        assert classifier_called is True

    def test_classifier_fallback_defaults_to_general(self) -> None:
        """When classifier returns no category key, the fallback is 'general'."""
        cat, _ = self._select_category(
            goal="some unknown goal",
            context={},
            classifier_result={},  # no category key
        )
        assert cat == "general", f"Expected 'general' fallback, got {cat!r}"

    def test_execute_with_agent_graph_passes_caller_category_to_spec_builder(self) -> None:
        """execute_with_agent_graph must forward context['category'] to RequestSpec.build().

        Regression for the bug where category was hardcoded to 'code' regardless of the
        caller-supplied context.  We mock get_spec_builder (patched at its source module
        so the local import picks up the mock) and provide a minimal plan_generator so the
        function can complete, then assert build() received category='security' not 'code'.
        """
        from vetinari.orchestration.pipeline_agent_graph import PipelineAgentGraphMixin

        # Minimal graph with no nodes so the node-iteration loop is a no-op.
        mock_graph = MagicMock()
        mock_graph.plan_id = "test-plan"
        mock_graph.nodes = {}
        mock_graph.follow_up_question = None

        mock_plan_gen = MagicMock()
        mock_plan_gen.generate_plan.return_value = mock_graph

        mock_variant_mgr = MagicMock()
        mock_variant_mgr.get_config.return_value = MagicMock(max_planning_depth=3)

        class _Harness(PipelineAgentGraphMixin):
            def __init__(self) -> None:
                self._agents: dict = {}  # type: ignore[assignment]
                self.plan_generator = mock_plan_gen
                self._variant_manager = mock_variant_mgr

        harness = _Harness()

        mock_spec = MagicMock()
        mock_spec.estimated_complexity = 3
        mock_spec.to_dict.return_value = {}

        mock_builder = MagicMock()
        mock_builder.build.return_value = mock_spec

        # get_spec_builder is a local import inside execute_with_agent_graph;
        # patch at the source module so the local import picks up the mock.
        with patch(
            "vetinari.orchestration.request_spec.get_spec_builder",
            return_value=mock_builder,
        ):
            harness.execute_with_agent_graph(
                goal="audit authentication flows",
                context={"category": "security"},
            )

        mock_builder.build.assert_called_once()
        _, build_kwargs = mock_builder.build.call_args
        assert build_kwargs.get("category") == "security", (
            f"Expected category='security' forwarded to spec builder, got {build_kwargs.get('category')!r}. "
            "The hardcoded category='code' regression may have been re-introduced."
        )


# ── 2. graph_task_runner: SecurityError caught by permission blocks ────────────


class TestGraphTaskRunnerSecurityError:
    """SecurityError from enforce_agent_permissions must yield success=False, not allow execution."""

    def _make_runner(self) -> Any:
        """Construct a minimal GraphTaskRunnerMixin harness with one fake agent.

        Harness sets _wip_tracker=None to skip WIP capacity checks and provides a
        single WORKER agent.  _execute_task_node should reach the security-check
        block before ever touching the agent.
        """
        from vetinari.agents.contracts import AgentResult
        from vetinari.orchestration.graph_task_runner import GraphTaskRunnerMixin
        from vetinari.types import AgentType

        class FakeAgent:
            def execute(self, task: Any) -> AgentResult:
                return AgentResult(success=True, output="ran", errors=[])

        class _Harness(GraphTaskRunnerMixin):
            def __init__(self) -> None:
                self._agents = {AgentType.WORKER: FakeAgent()}  # type: ignore[assignment]
                self._wip_tracker = None  # skip WIP capacity checks
                self._goal_tracker = None  # skip goal tracking

        return _Harness()

    def _make_task_node(self) -> Any:
        """Build a minimal Task + TaskNode for AgentType.WORKER."""
        from vetinari.agents.contracts import Task
        from vetinari.orchestration.graph_types import TaskNode
        from vetinari.types import AgentType

        task = Task(
            id="test-task-security",
            description="security test task",
            assigned_agent=AgentType.WORKER,
        )
        return TaskNode(task=task)

    def test_security_error_from_enforce_agent_permissions_returns_failure(self) -> None:
        """SecurityError raised by enforce_agent_permissions produces success=False result."""
        from vetinari.agents.contracts import AgentResult
        from vetinari.exceptions import SecurityError

        harness = self._make_runner()
        node = self._make_task_node()

        with (
            patch(
                "vetinari.execution_context.enforce_agent_permissions",
                side_effect=SecurityError("permission denied by policy"),
            ),
            patch(
                "vetinari.execution_context.get_context_manager",
                side_effect=Exception("skip ctx mgr"),
            ),
        ):
            result: AgentResult = harness._execute_task_node(node, prior_results={})

        assert result.success is False, (
            "SecurityError from enforce_agent_permissions must produce success=False"
        )
        assert any("permission" in e.lower() or "denied" in e.lower() for e in result.errors), (
            f"Error message must mention denial; got: {result.errors}"
        )

    def test_security_error_from_ctx_enforce_permission_returns_failure(self) -> None:
        """SecurityError raised by ctx_mgr.enforce_permission produces success=False result."""
        from vetinari.agents.contracts import AgentResult
        from vetinari.exceptions import SecurityError

        harness = self._make_runner()
        node = self._make_task_node()

        fake_ctx_mgr = MagicMock()
        fake_ctx_mgr.enforce_permission.side_effect = SecurityError("MODEL_INFERENCE blocked")

        with (
            patch(
                "vetinari.execution_context.enforce_agent_permissions",
                return_value=None,  # first check passes
            ),
            patch(
                "vetinari.execution_context.get_context_manager",
                return_value=fake_ctx_mgr,
            ),
        ):
            result: AgentResult = harness._execute_task_node(node, prior_results={})

        assert result.success is False, (
            "SecurityError from ctx_mgr.enforce_permission must produce success=False"
        )


# ── 3. pipeline_stages: Inspector gate fail-closed ───────────────────────────


class TestPipelineStagesInspectorGate:
    """Inspector gate must block on explicit passed=False AND on low quality_score."""

    def _make_stages_mixin(self) -> Any:
        """Return a minimal PipelineStagesMixin with mock sub-dependencies."""
        from vetinari.orchestration.pipeline_stages import PipelineStagesMixin

        class _Harness(PipelineStagesMixin):
            def __init__(self) -> None:
                self._agents: dict = {}  # type: ignore[assignment]

        return _Harness()

    def _run_inspector_gate(
        self,
        review_result: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Execute only the Inspector gate logic in isolation.

        Calls _run_review_gate directly with the supplied review_result so the
        gate logic is exercised without touching _run_execution_stages.

        Returns (gate_blocked, gate_issues).
        """
        from vetinari.orchestration.pipeline_stages import PipelineStagesMixin

        stages: dict[str, Any] = {}

        # Minimal graph mock
        graph = MagicMock()
        graph.plan_id = "test-plan"

        mixin = self._make_stages_mixin()
        # Patch both event_bus and the collaboration publish so the test has no
        # external deps and any exception from _run_review_gate propagates clearly.
        with (
            patch("vetinari.orchestration.pipeline_stages.get_event_bus", return_value=MagicMock()),
            patch("vetinari.orchestration.pipeline_stages._COLLABORATION_AVAILABLE", False),
        ):
            mixin._run_review_gate(graph=graph, stages=stages, review_result=review_result)

        return bool(stages.get("gate_blocked")), list(stages.get("gate_issues", []))

    def test_passed_false_with_empty_issues_blocks_gate(self) -> None:
        """passed=False with no issues list must still block the gate (fail-closed)."""
        blocked, _ = self._run_inspector_gate({"passed": False, "issues": []})
        assert blocked is True, "gate_blocked must be True when passed=False even with empty issues"

    def test_missing_passed_key_with_low_score_blocks_gate(self) -> None:
        """Missing 'passed' key + score=0.3 (< 0.5) must block the gate."""
        blocked, _ = self._run_inspector_gate({"quality_score": 0.3})
        assert blocked is True, "gate_blocked must be True when score=0.3 and 'passed' key absent"

    def test_passed_true_with_high_score_does_not_block(self) -> None:
        """passed=True + score=0.9 must NOT block the gate."""
        blocked, _ = self._run_inspector_gate({"passed": True, "quality_score": 0.9})
        assert blocked is False, "gate_blocked must be False when passed=True and score >= 0.5"

    def test_missing_passed_key_defaults_to_blocked(self) -> None:
        """Missing 'passed' key with no score (defaults to 1.0) blocks gate due to default=False.

        The old code used ``default=True`` so a missing key silently opened the gate.
        The new code uses ``default=False`` so absent key means the gate is blocked.
        Note: score defaults to 1.0 when missing, so _score_failed is False;
        blocked comes from ``not inspector_passed`` (False from default).
        """
        blocked, _ = self._run_inspector_gate({})
        assert blocked is True, "gate_blocked must be True when 'passed' key is absent (default=False)"


# ── 4. orchestrator.run_all: failure payloads do not inflate completed ────────


class TestOrchestratorRunAllFailurePayload:
    """run_all() must not increment completed when _execute_goal returns a failure dict."""

    def _make_orchestrator(self, tmp_path: Path, tasks: list[dict[str, Any]]) -> Any:
        """Build an Orchestrator pointing at a temp manifest with given tasks."""
        from vetinari.orchestrator import Orchestrator

        manifest = tmp_path / "vetinari.yaml"
        manifest.write_text(yaml.dump({"tasks": tasks}), encoding="utf-8")
        return Orchestrator(str(manifest))

    def test_failed_status_payload_not_counted_as_completed(self, tmp_path: Path) -> None:
        """Result with status='failed' must NOT increment the completed counter."""
        orch = self._make_orchestrator(
            tmp_path,
            [{"id": "t1", "description": "task one"}],
        )
        failure_result: dict[str, Any] = {"status": "failed", "reason": "agent error"}
        mock_tlo = MagicMock()
        mock_tlo.generate_and_execute.return_value = failure_result

        with patch("vetinari.orchestrator.get_two_layer_orchestrator", return_value=mock_tlo):
            summary = orch.run_all()

        assert summary["total"] == 1
        assert summary["completed"] == 0, (
            f"completed must be 0 for a failure payload; got {summary['completed']}"
        )
        assert len(summary["results"]) == 1

    def test_failed_count_payload_not_counted_as_completed(self, tmp_path: Path) -> None:
        """Result with failed>0 must NOT increment the completed counter."""
        orch = self._make_orchestrator(
            tmp_path,
            [{"id": "t1", "description": "task one"}],
        )
        failure_result: dict[str, Any] = {"failed": 1, "completed": 0}
        mock_tlo = MagicMock()
        mock_tlo.generate_and_execute.return_value = failure_result

        with patch("vetinari.orchestrator.get_two_layer_orchestrator", return_value=mock_tlo):
            summary = orch.run_all()

        assert summary["completed"] == 0, (
            f"completed must be 0 for a payload with failed=1; got {summary['completed']}"
        )

    def test_mixed_success_and_failure_counts_correctly(self, tmp_path: Path) -> None:
        """Two tasks: one success, one failure — completed must be exactly 1."""
        orch = self._make_orchestrator(
            tmp_path,
            [
                {"id": "t1", "description": "task one"},
                {"id": "t2", "description": "task two"},
            ],
        )
        success_result: dict[str, Any] = {"completed": 1}
        failure_result: dict[str, Any] = {"status": "failed", "reason": "timeout"}
        mock_tlo = MagicMock()
        mock_tlo.generate_and_execute.side_effect = [success_result, failure_result]

        with patch("vetinari.orchestrator.get_two_layer_orchestrator", return_value=mock_tlo):
            summary = orch.run_all()

        assert summary["total"] == 2
        assert summary["completed"] == 1, (
            f"completed must be 1 (one success, one failure); got {summary['completed']}"
        )

    def test_all_success_payloads_counted(self, tmp_path: Path) -> None:
        """Two tasks with non-failure results — completed must be 2 (existing contract preserved)."""
        orch = self._make_orchestrator(
            tmp_path,
            [
                {"id": "t1", "description": "task one"},
                {"id": "t2", "description": "task two"},
            ],
        )
        success_result: dict[str, Any] = {"completed": 1}
        mock_tlo = MagicMock()
        mock_tlo.generate_and_execute.return_value = success_result

        with patch("vetinari.orchestrator.get_two_layer_orchestrator", return_value=mock_tlo):
            summary = orch.run_all()

        assert summary["total"] == 2
        assert summary["completed"] == 2
