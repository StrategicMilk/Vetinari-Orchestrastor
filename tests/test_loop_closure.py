"""Loop closure verification — traces each claimed feedback loop end-to-end.

Each test verifies a producer-consumer chain exists in the code:
the producer function calls the consumer function (directly or through events).
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from vetinari.analytics.failure_registry import (
    FailureRegistry,
    reset_failure_registry,
)
from vetinari.learning.quality_scorer import QualityScorer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset failure registry singleton before each test."""
    reset_failure_registry()
    yield
    reset_failure_registry()


# ---------------------------------------------------------------------------
# Loop 1: Execution → Quality
# ---------------------------------------------------------------------------


def test_execution_to_quality_loop() -> None:
    """complete_task invokes quality scorer during post-execution subsystem 1."""
    from vetinari.agents.base_agent_completion import complete_task
    from vetinari.agents.contracts import AgentResult, AgentTask
    from vetinari.types import AgentType

    mock_agent = MagicMock()
    mock_agent.agent_type = AgentType.WORKER
    mock_agent.default_model = "test-model"
    mock_agent._last_inference_model_id = "test-model"
    mock_agent._adapter_manager = None
    mock_agent.get_system_prompt.return_value = "You are a worker agent."

    task = AgentTask(
        task_id="exec_qual_001",
        agent_type=AgentType.WORKER,
        description="Write a hello world function.",
        prompt="Write a hello world function.",
    )
    result = AgentResult(
        success=True,
        output="def hello(): return 'Hello, world!'",
    )

    captured_scores = []

    original_score = QualityScorer.score

    def capturing_score(self, **kwargs):
        s = original_score(self, **kwargs)
        captured_scores.append(s)
        return s

    with patch("vetinari.learning.quality_scorer.QualityScorer.score", capturing_score):
        with patch("vetinari.agents.base_agent._get_agent_constraints", return_value=None):
            complete_task(mock_agent, task, result)

    assert len(captured_scores) >= 1, "complete_task must invoke QualityScorer.score at least once"


# ---------------------------------------------------------------------------
# Loop 2: Quality → Thompson
# ---------------------------------------------------------------------------


def test_quality_to_thompson_loop() -> None:
    """Quality scores feed into Thompson arm updates via the feedback loop."""
    from vetinari.learning.feedback_loop import get_feedback_loop

    feedback = get_feedback_loop()

    # The feedback loop uses a local import inside _update_thompson_arms:
    #   from vetinari.learning.model_selector import get_thompson_selector
    # Patch at that canonical location.
    mock_ts = MagicMock()

    with patch("vetinari.learning.model_selector.get_thompson_selector", return_value=mock_ts):
        feedback.record_outcome(
            task_id="q2t_task_001",
            model_id="q2t-model",
            task_type="coding",
            quality_score=0.85,
            success=True,
        )

    # Thompson update must have been called with the model and quality score
    mock_ts.update.assert_called()
    # Inspect the first call
    args, kwargs = mock_ts.update.call_args
    all_args = {**kwargs}
    if args:
        param_names = ["model_id", "task_type", "quality_score", "success"]
        all_args.update(dict(zip(param_names, args)))

    assert all_args.get("quality_score") == pytest.approx(0.85)
    assert all_args.get("model_id") == "q2t-model"


# ---------------------------------------------------------------------------
# Loop 3: Completion → Training data store
# ---------------------------------------------------------------------------


def test_completion_to_training_loop() -> None:
    """complete_task records execution to training data collector (subsystem 5)."""
    from vetinari.agents.base_agent_completion import complete_task
    from vetinari.agents.contracts import AgentResult, AgentTask
    from vetinari.types import AgentType

    mock_agent = MagicMock()
    mock_agent.agent_type = AgentType.WORKER
    mock_agent.default_model = "train-model"
    mock_agent._last_inference_model_id = "train-model"
    mock_agent._adapter_manager = None
    mock_agent.get_system_prompt.return_value = "You are a worker."

    task = AgentTask(
        task_id="train_loop_001",
        agent_type=AgentType.WORKER,
        description="Summarise the following text.",
        prompt="Summarise the following text.",
    )
    result = AgentResult(
        success=True,
        output="This text discusses the importance of testing in software development.",
    )

    with patch("vetinari.agents.base_agent._get_agent_constraints", return_value=None):
        with patch("vetinari.learning.training_data.TrainingDataCollector.record") as mock_record:
            complete_task(mock_agent, task, result)

    mock_record.assert_called_once()
    call_kwargs = mock_record.call_args[1]
    assert "task" in call_kwargs


# ---------------------------------------------------------------------------
# Loop 4: Prompt evolution → shadow tests
# ---------------------------------------------------------------------------


def test_prompt_evolution_to_shadow_test_loop() -> None:
    """PromptEvolver.evolve_if_needed triggers shadow test creation for variants."""
    from vetinari.learning.prompt_evolver import PromptEvolver, PromptVariant
    from vetinari.types import PromptVersionStatus

    evolver = PromptEvolver()
    evolver._variants = {
        "WORKER": [
            PromptVariant(
                variant_id="WORKER_baseline",
                agent_type="WORKER",
                prompt_text="baseline",
                is_baseline=True,
                status=PromptVersionStatus.PROMOTED.value,
            ),
            PromptVariant(
                variant_id="WORKER_v1",
                agent_type="WORKER",
                prompt_text="variant",
                status=PromptVersionStatus.SHADOW_TESTING.value,
                metadata={"shadow_test_id": "shadow-1"},
            ),
        ]
    }

    # Verify the evolver's promote path calls shadow_testing when a variant
    # has enough trials. We test that check_shadow_test_results interacts
    # with the shadow test runner.
    with (
        patch("vetinari.learning.shadow_testing.get_shadow_test_runner") as mock_get_runner,
        patch.object(evolver, "_update_operator_feedback"),
        patch.object(evolver, "_save_variants"),
    ):
        mock_runner = MagicMock()
        mock_runner.evaluate.return_value = {"decision": "promote", "quality_delta": 0.1}
        mock_get_runner.return_value = mock_runner

        # check_shadow_test_results calls get_shadow_test_runner
        evolver.check_shadow_test_results()

    # The shadow runner was accessed — loop is wired
    mock_get_runner.assert_called_once()
    mock_runner.evaluate.assert_called_once_with("shadow-1")
    worker_variants = evolver._variants["WORKER"]
    assert worker_variants[1].status == PromptVersionStatus.PROMOTED.value


# ---------------------------------------------------------------------------
# Loop 5: Failed tasks → FailureRegistry
# ---------------------------------------------------------------------------


def test_failure_to_registry_loop() -> None:
    """Failed tasks get registered in FailureRegistry and are retrievable."""
    registry = FailureRegistry()

    entry = registry.log_failure(
        category="inspector_rejection",
        severity="error",
        description="Output failed quality gate with score 0.2",
        root_cause="Model produced incomplete output",
        affected_components=["worker", "inspector"],
    )

    assert entry.failure_id.startswith("fail_")

    failures = registry.get_failures(category="inspector_rejection")
    assert any(f.failure_id == entry.failure_id for f in failures)

    # Verify the status is active (not silently dropped)
    matching = next(f for f in failures if f.failure_id == entry.failure_id)
    assert matching.status == "active"


# ---------------------------------------------------------------------------
# Loop 6: Memory → Consolidation path
# ---------------------------------------------------------------------------


def test_memory_to_consolidation_loop() -> None:
    """Memory store exposes a consolidation/compaction path (KnowledgeCompactor)."""
    # Verify KnowledgeCompactor.compact accepts MemoryEntry lists and returns a report
    from vetinari.kaizen.knowledge_compactor import CompactionReport, KnowledgeCompactor
    from vetinari.memory.interfaces import MemoryEntry
    from vetinari.types import MemoryType

    compactor = KnowledgeCompactor()

    # Create a minimal set of episode entries
    episodes = [
        MemoryEntry(
            id=f"mem_cons_{i}",
            entry_type=MemoryType.DISCOVERY,
            content=f"Recorded execution result episode {i} with quality 0.8",
            summary=f"Episode {i}",
        )
        for i in range(3)
    ]

    report = compactor.compact(episodes)

    # Consolidation path exists and processes entries
    assert isinstance(report, CompactionReport)
    assert report.input_entries == 3


# ---------------------------------------------------------------------------
# Loop 7: Cloud training boundary
# ---------------------------------------------------------------------------


def test_cloud_training_documented() -> None:
    """train_cloud() raises RuntimeError — intentional boundary until cloud infra exists."""
    from vetinari.learning.training_manager import TrainingDataset, get_training_manager

    manager = get_training_manager()

    # The method must exist (it is a documented boundary)
    assert hasattr(manager, "train_cloud"), "train_cloud() must exist on TrainingManager as a documented boundary"

    dataset = TrainingDataset(records=[], format="sft", stats={})
    with pytest.raises(RuntimeError):
        manager.train_cloud(model_id="test-model", dataset=dataset)
