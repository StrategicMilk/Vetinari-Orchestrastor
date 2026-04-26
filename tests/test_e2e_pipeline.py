"""End-to-end pipeline test: project create -> plan -> execute -> quality -> Thompson.

Exercises the full Vetinari product pipeline from goal intake through
learning feedback.  Mocks only the LLM adapter layer so all internal
orchestration, quality scoring, and Thompson Sampling run for real.
"""

from __future__ import annotations

import os
import random
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_mock_llm_output as _make_mock_llm_output
from vetinari.learning.feedback_loop import FeedbackLoop
from vetinari.learning.model_selector import ThompsonSamplingSelector
from vetinari.learning.quality_scorer import QualityScore, QualityScorer
from vetinari.learning.thompson_arms import ThompsonBetaArm
from vetinari.orchestration.execution_graph import ExecutionGraph, ExecutionTaskNode
from vetinari.orchestration.two_layer import TwoLayerOrchestrator
from vetinari.types import PlanStatus, StatusEnum

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MOCK_MODEL_ID = "test-model-7b"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _isolated_thompson(tmp_path):
    """Provide a fresh ThompsonSamplingSelector that persists to tmp_path.

    Patches the VETINARI_STATE_DIR env var so Thompson state files go to
    a temp directory, then constructs a fresh selector with no prior arms.
    """
    state_dir = str(tmp_path / ".vetinari")
    with patch.dict(os.environ, {"VETINARI_STATE_DIR": state_dir}):
        selector = ThompsonSamplingSelector()
        # Clear any arms loaded from prior state
        selector._arms.clear()
        yield selector


@pytest.fixture
def _isolated_scorer():
    """Provide a QualityScorer backed by the per-test isolated database."""
    return QualityScorer(adapter_manager=None)


@pytest.fixture
def _isolated_feedback():
    """Provide a FeedbackLoop with memory/router dependencies disabled."""
    loop = FeedbackLoop()
    # Prevent lazy-init from contacting real memory/router
    loop._memory = MagicMock()
    loop._memory.get_model_performance.return_value = None
    loop._memory.update_model_performance = MagicMock()
    loop._memory.update_subtask_quality = MagicMock()
    loop._router = MagicMock()
    loop._router.get_performance_cache.return_value = {}
    loop._router.update_performance_cache = MagicMock()
    return loop


@pytest.fixture
def _orchestrator(tmp_path):
    """Provide a TwoLayerOrchestrator with a temp checkpoint directory."""
    orch = TwoLayerOrchestrator(checkpoint_dir=str(tmp_path / "checkpoints"))
    return orch


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """End-to-end test covering the full Vetinari product pipeline.

    Exercises: project goal -> plan generation -> task execution ->
    quality scoring -> Thompson Sampling update.

    Only the LLM adapter is mocked; all internal subsystems run for real.
    """

    def test_full_pipeline_create_to_thompson(
        self,
        tmp_path,
        _isolated_thompson,
        _isolated_scorer,
        _isolated_feedback,
        _orchestrator,
    ):
        """Walk a goal through every pipeline stage and verify each output.

        Stages validated:
        1. Plan generation produces an ExecutionGraph with tasks
        2. Task execution populates output_data on every ExecutionTaskNode
        3. Quality scoring produces scores in (0, 1]
        4. FeedbackLoop propagates quality to memory + router caches
        5. Thompson Sampling arms are created and updated
        """
        goal = "Write a Python hello world script"
        thompson = _isolated_thompson
        scorer = _isolated_scorer
        feedback = _isolated_feedback
        orch = _orchestrator

        # -- Stage 1: Plan generation (keyword fallback, no LLM needed) ------
        graph = orch.plan_generator.generate_plan(goal)

        assert isinstance(graph, ExecutionGraph)
        assert graph.plan_id.startswith("plan-")
        assert graph.goal == goal
        assert len(graph.nodes) >= 1, "Plan must decompose goal into at least one task"
        assert graph.status == PlanStatus.DRAFT

        task_ids = list(graph.nodes.keys())
        for tid in task_ids:
            node = graph.nodes[tid]
            assert node.status == StatusEnum.PENDING
            assert node.description, f"Task {tid} must have a description"

        # -- Stage 2: Task execution with mock handler -----------------------
        def _mock_task_handler(task: ExecutionTaskNode) -> dict[str, Any]:
            """Simulate LLM-backed task execution with deterministic output."""
            return {
                "output": _make_mock_llm_output(task.description),
                "model_id": _MOCK_MODEL_ID,
            }

        results = orch.execution_engine.execute_plan(graph, task_handler=_mock_task_handler)

        assert results["plan_id"] == graph.plan_id
        assert results["total_tasks"] == len(task_ids)
        assert results["completed"] == len(task_ids), (
            f"All tasks should complete; got {results['completed']}/{len(task_ids)}"
        )
        assert results["failed"] == 0

        # Verify every node has output_data populated
        for tid in task_ids:
            node = graph.nodes[tid]
            assert node.status == StatusEnum.COMPLETED, f"Task {tid} not completed"
            assert node.output_data, f"Task {tid} has empty output_data"
            assert "output" in node.output_data, f"Task {tid} output_data missing 'output' key"

        # Graph status should reflect successful completion
        assert graph.status == PlanStatus.COMPLETED

        # -- Stage 3: Quality scoring on each task output --------------------
        quality_scores: list[QualityScore] = []

        for tid in task_ids:
            node = graph.nodes[tid]
            output_str = node.output_data.get("output", "")
            if isinstance(output_str, dict):
                output_str = str(output_str)

            q_score = scorer.score(
                task_id=tid,
                model_id=_MOCK_MODEL_ID,
                task_type=node.task_type,
                task_description=node.description,
                output=output_str,
                use_llm=False,
            )
            quality_scores.append(q_score)

            assert isinstance(q_score, QualityScore)
            assert q_score.task_id == tid
            assert q_score.model_id == _MOCK_MODEL_ID
            assert 0.0 < q_score.overall_score <= 1.0, f"Quality score for {tid} out of range: {q_score.overall_score}"
            assert q_score.method == "heuristic"

        # At least one score should have non-empty dimensions
        has_dimensions = any(bool(qs.dimensions) for qs in quality_scores)
        assert has_dimensions, "At least one QualityScore should have dimension breakdown"

        # Verify persistence: scorer should have history
        history = scorer.get_history()
        assert len(history) >= len(task_ids), f"Scorer history should contain at least {len(task_ids)} entries"

        # -- Stage 4: Feedback loop propagates quality -----------------------
        for qs in quality_scores:
            feedback.record_outcome(
                task_id=qs.task_id,
                model_id=qs.model_id,
                task_type=qs.task_type,
                quality_score=qs.overall_score,
                success=True,
            )

        # Verify memory store was updated
        assert feedback._memory.update_model_performance.call_count >= len(quality_scores)
        assert feedback._memory.update_subtask_quality.call_count >= len(quality_scores)

        # -- Stage 5: Thompson Sampling arms are updated ---------------------
        # Record outcomes directly on the isolated selector
        for qs in quality_scores:
            thompson.update(
                model_id=qs.model_id,
                task_type=qs.task_type,
                quality_score=qs.overall_score,
                success=True,
            )

        # Verify arms were created and updated
        assert len(thompson._arms) >= 1, "Thompson should have at least one arm"

        for key, arm in thompson._arms.items():
            assert isinstance(arm, ThompsonBetaArm)
            assert arm.total_pulls >= 1, f"Arm {key} should have been pulled at least once"
            # After a successful update, alpha should increase beyond the default 2.0
            # (or 1.0 for uninformed prior) by the quality_score amount
            assert arm.alpha > 1.0, f"Arm {key} alpha should increase after success"

        # Verify model selection uses the updated arm
        task_types_seen = {qs.task_type for qs in quality_scores}
        for tt in task_types_seen:
            selected = thompson.select_model(tt, [_MOCK_MODEL_ID, "other-model-13b"])
            # Should return a valid model from the candidates
            assert selected in {_MOCK_MODEL_ID, "other-model-13b"}

        # Verify rankings reflect the update
        for tt in task_types_seen:
            rankings = thompson.get_rankings(tt)
            assert len(rankings) >= 1, f"Rankings for {tt} should have at least one entry"
            # The model we updated should appear in rankings
            ranked_models = [r[0] for r in rankings]
            assert _MOCK_MODEL_ID in ranked_models

    def test_plan_with_dependencies_executes_in_order(self, tmp_path):
        """Verify that tasks with dependencies execute after their prerequisites."""
        orch = TwoLayerOrchestrator(checkpoint_dir=str(tmp_path / "checkpoints"))

        # Build a graph manually with explicit dependencies
        graph = ExecutionGraph(plan_id="plan-dep-test", goal="Multi-step task")
        graph.add_task("t1", "First task", task_type="coding")
        graph.add_task("t2", "Second task (depends on t1)", task_type="coding", depends_on=["t1"])
        graph.add_task("t3", "Third task (depends on t2)", task_type="coding", depends_on=["t2"])
        graph.status = PlanStatus.DRAFT

        execution_order: list[str] = []

        def _order_tracking_handler(task: ExecutionTaskNode) -> dict[str, Any]:
            execution_order.append(task.id)
            return {"output": f"Result of {task.id}"}

        results = orch.execution_engine.execute_plan(graph, task_handler=_order_tracking_handler)

        assert results["completed"] == 3
        assert results["failed"] == 0

        # t1 must execute before t2, t2 before t3
        assert execution_order.index("t1") < execution_order.index("t2")
        assert execution_order.index("t2") < execution_order.index("t3")

    def test_quality_scorer_distinguishes_output_quality(self):
        """Verify that the heuristic scorer gives different scores for different outputs."""
        scorer = QualityScorer(adapter_manager=None)

        # Good coding output
        good_output = (
            "def fibonacci(n: int) -> int:\n"
            '    """Calculate the nth Fibonacci number using dynamic programming."""\n'
            "    if n <= 1:\n        return n\n"
            "    a, b = 0, 1\n"
            "    for _ in range(2, n + 1):\n"
            "        a, b = b, a + b\n"
            "    return b\n\n"
            "assert fibonacci(10) == 55\n"
            "assert fibonacci(0) == 0\n"
        )

        # Poor output (empty-ish)
        poor_output = "ok"

        good_score = scorer.score(
            task_id="good",
            model_id=_MOCK_MODEL_ID,
            task_type="coding",
            task_description="Write fibonacci",
            output=good_output,
            use_llm=False,
        )
        poor_score = scorer.score(
            task_id="poor",
            model_id=_MOCK_MODEL_ID,
            task_type="coding",
            task_description="Write fibonacci",
            output=poor_output,
            use_llm=False,
        )

        assert good_score.overall_score > poor_score.overall_score, (
            f"Good output ({good_score.overall_score}) should score higher than poor ({poor_score.overall_score})"
        )

    def test_thompson_arms_converge_with_repeated_feedback(self, tmp_path):
        """Verify that Thompson arms shift toward higher-quality models over time."""
        state_dir = str(tmp_path / ".vetinari")
        with patch.dict(os.environ, {"VETINARI_STATE_DIR": state_dir}):
            thompson = ThompsonSamplingSelector()
            thompson._arms.clear()

        random.seed(42)

        # Simulate: model-A consistently high quality, model-B consistently low
        for _ in range(20):
            thompson.update("model-A", "coding", quality_score=0.9, success=True)
            thompson.update("model-B", "coding", quality_score=0.3, success=False)

        arm_a = thompson.get_arm_state("model-A", "coding")
        arm_b = thompson.get_arm_state("model-B", "coding")

        assert arm_a["mean"] > arm_b["mean"], (
            f"model-A mean ({arm_a['mean']:.3f}) should exceed model-B ({arm_b['mean']:.3f})"
        )
        assert arm_a["total_pulls"] == 20
        assert arm_b["total_pulls"] == 20

        # Selection should strongly prefer model-A after 20 observations
        selections = [thompson.select_model("coding", ["model-A", "model-B"]) for _ in range(50)]
        a_count = selections.count("model-A")
        assert a_count >= 35, f"model-A should be selected most of the time; got {a_count}/50"

    def test_feedback_loop_wires_to_thompson(self, tmp_path):
        """Verify FeedbackLoop._update_thompson_arms actually updates Thompson state."""
        state_dir = str(tmp_path / ".vetinari")
        with patch.dict(os.environ, {"VETINARI_STATE_DIR": state_dir}):
            thompson = ThompsonSamplingSelector()
            thompson._arms.clear()

        feedback = FeedbackLoop()
        feedback._memory = MagicMock()
        feedback._memory.get_model_performance.return_value = None
        feedback._memory.update_model_performance = MagicMock()
        feedback._memory.update_subtask_quality = MagicMock()
        feedback._router = MagicMock()
        feedback._router.get_performance_cache.return_value = {}
        feedback._router.update_performance_cache = MagicMock()

        # Patch get_thompson_selector at its source module so the late
        # import inside _update_thompson_arms picks up our isolated instance
        with patch(
            "vetinari.learning.model_selector.get_thompson_selector",
            return_value=thompson,
        ):
            feedback.record_outcome(
                task_id="e2e-task-1",
                model_id=_MOCK_MODEL_ID,
                task_type="coding",
                quality_score=0.85,
                success=True,
            )

        # Verify the Thompson arm was created and updated
        arm_state = thompson.get_arm_state(_MOCK_MODEL_ID, "coding")
        assert arm_state["total_pulls"] >= 1
        assert arm_state["alpha"] > 1.0, "Alpha should increase after successful outcome"
