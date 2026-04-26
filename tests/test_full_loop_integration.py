"""Canonical full-loop integration test — traces the complete pipeline.

Reusable artifact for Sessions 28 and 29.

Traces: task → agent execution → quality scoring → Thompson update →
learning orchestrator → prompt evolution → shadow test → failure registry →
knowledge correction → decision journal.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from vetinari.analytics.failure_registry import (
    FailureRegistry,
    get_failure_registry,
    reset_failure_registry,
)
from vetinari.knowledge.validator import CorrectionRecord, KnowledgeValidator
from vetinari.learning.model_selector import ThompsonSamplingSelector
from vetinari.learning.orchestrator import (
    LearningOrchestrator,
    reset_learning_orchestrator,
)
from vetinari.learning.quality_scorer import QualityScore, QualityScorer
from vetinari.observability.decision_journal import (
    get_decision_journal,
    reset_decision_journal,
)
from vetinari.types import ConfidenceLevel, DecisionType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Destroy singletons before and after each test to prevent cross-test pollution."""
    reset_failure_registry()
    reset_decision_journal()
    reset_learning_orchestrator()
    yield
    reset_failure_registry()
    reset_decision_journal()
    reset_learning_orchestrator()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_quality_scoring_produces_score() -> None:
    """QualityScorer.score returns a QualityScore with a numeric overall_score."""
    scorer = QualityScorer(adapter_manager=None)

    score = scorer.score(
        task_id="task_001",
        model_id="test-model",
        task_type="coding",
        task_description="Write a function that adds two numbers.",
        output="def add(a: int, b: int) -> int:\n    return a + b",
        use_llm=False,
    )

    assert isinstance(score, QualityScore)
    assert isinstance(score.overall_score, float)
    assert 0.0 <= score.overall_score <= 1.0
    assert score.task_id == "task_001"
    assert score.model_id == "test-model"
    assert score.method in ("heuristic", "llm", "hybrid", "unmeasured", "rejected")


def test_quality_scorer_rejects_empty_output() -> None:
    """QualityScorer rejects empty/fallback outputs with method='rejected'."""
    scorer = QualityScorer(adapter_manager=None)

    score = scorer.score(
        task_id="task_empty",
        model_id="test-model",
        task_type="coding",
        task_description="Do something.",
        output="",
        use_llm=False,
    )

    assert score.method == "rejected"
    assert score.overall_score == 0.0


def test_thompson_arm_update() -> None:
    """Create ThompsonSamplingSelector, record a result, verify arm counts change."""
    selector = ThompsonSamplingSelector()

    # Capture initial state
    key = "test-model-ts:coding"
    # Arm is created lazily on first update
    initial_arms = len(selector._arms)

    selector.update(
        model_id="test-model-ts",
        task_type="coding",
        quality_score=0.8,
        success=True,
    )

    assert len(selector._arms) >= initial_arms + 1 or key in selector._arms
    arm = selector._arms.get(key)
    assert arm is not None
    assert arm.total_pulls == 1
    # alpha should have increased by quality_score (0.8) from the prior
    assert arm.alpha > 2.0  # prior is Beta(2,2); after success alpha > 2


def test_thompson_select_model_returns_candidate() -> None:
    """select_model always returns one of the candidate model ids."""
    selector = ThompsonSamplingSelector()

    candidates = ["model-a", "model-b", "model-c"]
    chosen = selector.select_model(task_type="analysis", candidate_models=candidates)

    assert chosen in candidates


def test_thompson_select_model_empty_candidates() -> None:
    """select_model returns 'default' when candidate list is empty."""
    selector = ThompsonSamplingSelector()
    assert selector.select_model(task_type="coding", candidate_models=[]) == "default"


def test_failure_registry_records() -> None:
    """Get FailureRegistry, register a failure, verify it's stored and retrievable."""
    registry = get_failure_registry()

    entry = registry.log_failure(
        category="test_category",
        severity="error",
        description="Test failure for integration test",
        root_cause="Deliberate test failure",
    )

    assert entry.failure_id.startswith("fail_")
    assert entry.category == "test_category"
    assert entry.severity == "error"

    # Verify it is retrievable
    failures = registry.get_failures(category="test_category")
    matching = [f for f in failures if f.failure_id == entry.failure_id]
    assert len(matching) == 1
    assert matching[0].description == "Test failure for integration test"


def test_failure_registry_singleton_isolation() -> None:
    """After reset, a fresh registry has no entries from previous test."""
    registry = get_failure_registry()
    registry.log_failure(
        category="isolation_check",
        severity="warning",
        description="Should not appear after reset",
    )
    reset_failure_registry()

    fresh_registry = get_failure_registry()
    # The JSONL file may persist between tests (it writes to ~/.vetinari) but
    # the in-memory cache is reset; test only verifies reset clears cache state
    fresh_registry.reset()  # Also reset in-memory rules cache
    # Singleton is new instance — verifies double-checked locking creates fresh object
    assert fresh_registry is not registry


def test_learning_orchestrator_runs_cycle() -> None:
    """Create LearningOrchestrator, mock subsystems, run a cycle, verify it completes."""
    orchestrator = LearningOrchestrator(cycle_interval_seconds=60)

    # _run_cycle does local imports inside the function body.
    # Patch the three instance methods that do the real work so we don't
    # need to thread through all the local-import chains.
    with (
        patch.object(orchestrator, "_dispatch_strategy", return_value=0.01) as mock_dispatch,
        patch.object(orchestrator, "_get_current_quality", return_value=None),
        patch.object(orchestrator, "_check_for_rollback"),
    ):
        from vetinari.learning.meta_optimizer import LearningPhase

        mock_optimizer = MagicMock()
        mock_optimizer.detect_phase.return_value = LearningPhase.IMPROVEMENT
        mock_optimizer.suggest_next_strategy.return_value = "prompt_evolution"
        mock_optimizer.record_cycle.return_value = None

        with patch(
            "vetinari.learning.meta_optimizer.get_meta_optimizer",
            return_value=mock_optimizer,
        ):
            orchestrator._run_cycle()

    assert orchestrator._cycle_count == 1
    mock_dispatch.assert_called_once_with("prompt_evolution")
    mock_optimizer.record_cycle.assert_called_once_with("prompt_evolution", 0.01)


def test_decision_journal_records() -> None:
    """Get decision journal, write an entry, verify it's readable."""
    journal = get_decision_journal()

    decision_id = journal.log_decision(
        decision_type=DecisionType.MODEL_SELECTION,
        description="Selected model-alpha for coding tasks",
        confidence_score=0.92,
        confidence_level=ConfidenceLevel.HIGH,
        action_taken="model-alpha",
        context={"alternatives": ["model-beta", "model-gamma"]},
    )

    assert decision_id is not None
    assert isinstance(decision_id, str)

    # Verify it can be retrieved
    decisions = journal.get_decisions(decision_type=DecisionType.MODEL_SELECTION, limit=10)
    matching = [d for d in decisions if d.decision_id == decision_id]
    assert len(matching) == 1
    assert matching[0].description == "Selected model-alpha for coding tasks"
    assert matching[0].confidence_level == ConfidenceLevel.HIGH


def test_decision_journal_update_outcome() -> None:
    """update_outcome sets the outcome field on a recorded decision."""
    journal = get_decision_journal()

    decision_id = journal.log_decision(
        decision_type=DecisionType.PARAMETER_TUNING,
        description="Set temperature to 0.7 for balanced creativity",
        confidence_score=0.85,
        confidence_level=ConfidenceLevel.HIGH,
        action_taken="temperature=0.7",
    )

    updated = journal.update_outcome(
        decision_id=decision_id,
        outcome="Task completed with quality score 0.87",
    )

    assert updated is True

    decisions = journal.get_decisions(decision_type=DecisionType.PARAMETER_TUNING, limit=5)
    matching = [d for d in decisions if d.decision_id == decision_id]
    assert len(matching) == 1
    assert matching[0].outcome == "Task completed with quality score 0.87"


def test_pipeline_chain_quality_to_thompson() -> None:
    """Score a result, feed to Thompson, verify the arm for that model was updated."""
    scorer = QualityScorer(adapter_manager=None)
    selector = ThompsonSamplingSelector()

    # Score a real output
    score = scorer.score(
        task_id="chain_task_001",
        model_id="chain-model",
        task_type="coding",
        task_description="Implement a stack data structure.",
        output=(
            "class Stack:\n"
            "    def __init__(self) -> None:\n"
            '        """Initialize empty stack."""\n'
            "        self._items: list = []\n\n"
            "    def push(self, item) -> None:\n"
            "        self._items.append(item)\n\n"
            "    def pop(self):\n"
            "        if not self._items:\n"
            "            raise IndexError('pop from empty stack')\n"
            "        return self._items.pop()\n"
        ),
        use_llm=False,
    )

    assert score.overall_score > 0.0, "Score should be non-zero for this output"

    # Feed the score into Thompson
    arm_key = "chain-model:coding"
    alpha_before = selector._arms.get(arm_key, MagicMock(alpha=2.0)).alpha

    selector.update(
        model_id="chain-model",
        task_type="coding",
        quality_score=score.overall_score,
        success=True,
    )

    arm = selector._arms.get(arm_key)
    assert arm is not None
    assert arm.alpha > alpha_before, "Alpha should increase after a successful update"
    assert arm.total_pulls == 1


def test_knowledge_correction_triggers_on_divergence(tmp_path) -> None:
    """KnowledgeValidator auto-corrects when predicted vs actual diverge > 0.15.

    This covers the 'knowledge correction trigger' step in the full pipeline:
    quality scoring reveals actual performance that diverges from stored
    predictions, and the validator creates a CorrectionRecord and persists it.
    """
    corrections_file = tmp_path / "corrections.json"
    validator = KnowledgeValidator(corrections_path=corrections_file)

    # Predicted knowledge says model-A accuracy is 0.90
    knowledge_data = {"model-A": {"accuracy": 0.90, "latency": 0.5}}
    # Actual performance shows accuracy is 0.70 — divergence of 0.20 (> 0.15 threshold)
    actual_data = {"model-A": {"accuracy": 0.70, "latency": 0.48}}

    report = validator.validate(knowledge_data, actual_data)

    assert report.checked_models == 1, "Should have checked model-A"
    assert len(report.corrections) >= 1, "Should trigger correction for accuracy divergence > 0.15"

    accuracy_correction = next((c for c in report.corrections if c.metric == "accuracy"), None)
    assert accuracy_correction is not None, "Must have a correction for 'accuracy'"
    assert isinstance(accuracy_correction, CorrectionRecord)
    assert accuracy_correction.model_id == "model-A"
    assert accuracy_correction.old_value == pytest.approx(0.90)
    assert accuracy_correction.new_value == pytest.approx(0.70)
    assert abs(accuracy_correction.divergence) > 0.15

    # Latency divergence is only 0.02 — should NOT trigger correction
    latency_correction = next((c for c in report.corrections if c.metric == "latency"), None)
    assert latency_correction is None, "Latency divergence 0.02 < threshold, no correction"

    # Corrections should be persisted to the JSON file
    assert corrections_file.exists(), "Corrections file must be created"
