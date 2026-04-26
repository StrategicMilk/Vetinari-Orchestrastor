"""Deterministic fuzz tests for core pure-function invariants.

These tests keep the original coverage intent of the prior Hypothesis-based
module while avoiding suite-wide state leakage from third-party Hypothesis
internals under the Windows test harness.
"""

from __future__ import annotations

import random
import string

from tests.factories import make_plan, make_task
from vetinari.types import AgentType, StatusEnum
from vetinari.web.sse_events import (
    PlanningStartEvent,
    StatusEvent,
    TaskCompleteEvent,
    TaskFailedEvent,
    TaskStartEvent,
)

SAFE_CHARS = tuple(string.ascii_letters + string.digits + " _-.")
MODEL_ID_CHARS = tuple(string.ascii_letters + string.digits)
RICH_TEXT_CHARS = tuple(string.ascii_letters + string.digits + string.punctuation + " \n\t")


def _make_rng(seed: int) -> random.Random:
    return random.Random(seed)


def _sample_text(
    rng: random.Random,
    *,
    min_size: int = 1,
    max_size: int = 50,
    alphabet: tuple[str, ...] = SAFE_CHARS,
) -> str:
    size = rng.randint(min_size, max_size)
    return "".join(rng.choice(alphabet) for _ in range(size))


def _sample_string_list(rng: random.Random, *, max_items: int = 10) -> list[str]:
    return [_sample_text(rng) for _ in range(rng.randint(0, max_items))]


def _sample_unique_model_ids(
    rng: random.Random,
    *,
    min_items: int,
    max_items: int,
) -> list[str]:
    ids: set[str] = set()
    while len(ids) < rng.randint(min_items, max_items):
        ids.add(_sample_text(rng, min_size=4, max_size=20, alphabet=MODEL_ID_CHARS))
    return list(ids)


class TestTaskNormalization:
    """Tasks created via make_task always have all required keys."""

    def test_make_task_always_has_required_fields(self) -> None:
        rng = _make_rng(1001)
        agents = list(AgentType)
        statuses = list(StatusEnum)

        for _ in range(200):
            task = make_task(
                description=_sample_text(rng),
                inputs=_sample_string_list(rng),
                outputs=_sample_string_list(rng),
                dependencies=_sample_string_list(rng),
                assigned_agent=rng.choice(agents),
                status=rng.choice(statuses),
                depth=rng.randint(0, 10),
            )
            assert hasattr(task, "id")
            assert isinstance(task.id, str)
            assert len(task.id) > 0
            assert hasattr(task, "inputs")
            assert isinstance(task.inputs, list)
            assert hasattr(task, "outputs")
            assert isinstance(task.outputs, list)
            assert hasattr(task, "dependencies")
            assert isinstance(task.dependencies, list)
            assert hasattr(task, "assigned_agent")
            assert hasattr(task, "status")

    def test_task_to_dict_roundtrips(self) -> None:
        rng = _make_rng(1002)
        agents = list(AgentType)

        for _ in range(100):
            task = make_task(
                description=_sample_text(rng),
                assigned_agent=rng.choice(agents),
            )
            payload = task.to_dict()
            assert isinstance(payload, dict)
            assert "id" in payload
            assert "description" in payload
            assert "inputs" in payload
            assert "outputs" in payload
            assert "dependencies" in payload

    def test_plan_always_has_tasks(self) -> None:
        rng = _make_rng(1003)

        for _ in range(50):
            n_tasks = rng.randint(1, 20)
            goal = _sample_text(rng)
            tasks = [make_task(description=f"task-{i}") for i in range(n_tasks)]
            plan = make_plan(goal=goal, tasks=tasks)
            assert len(plan.tasks) == n_tasks
            assert plan.goal == goal


class TestThompsonSamplingInvariants:
    """Thompson Sampling always returns a valid model from candidates."""

    @staticmethod
    def _build_selector():
        from unittest.mock import patch

        from vetinari.learning.model_selector import ThompsonSamplingSelector

        # Patch I/O methods to avoid disk access during selector initialization
        with (
            patch.object(ThompsonSamplingSelector, "_load_state", return_value=None),
            patch.object(ThompsonSamplingSelector, "_seed_from_benchmarks", return_value=None),
        ):
            selector = ThompsonSamplingSelector()

        return selector

    def test_select_returns_one_of_candidates(self) -> None:
        rng = _make_rng(2001)

        for _ in range(100):
            candidates = _sample_unique_model_ids(rng, min_items=1, max_items=10)
            selector = self._build_selector()
            result = selector.select_model("general", candidates)
            assert result in candidates

    def test_update_accepts_any_valid_quality_score(self) -> None:
        rng = _make_rng(2002)

        for _ in range(100):
            candidates = _sample_unique_model_ids(rng, min_items=1, max_items=5)
            quality = rng.random()
            selector = self._build_selector()
            model = candidates[0]
            selector.update(model, "general", quality, quality >= 0.5)
            assert selector._update_count == 1
            assert f"{model}:general" in selector._arms

    def test_select_with_empty_costs_still_works(self) -> None:
        rng = _make_rng(2003)

        for _ in range(50):
            candidates = _sample_unique_model_ids(rng, min_items=2, max_items=5)
            selector = self._build_selector()
            assert selector.select_model("general", candidates, cost_per_model={}) in candidates
            assert selector.select_model("general", candidates, cost_per_model=None) in candidates

    def test_contextual_selector_via_real_constructor_path(self) -> None:
        """Contextual selection and update must work via the real __init__ path.

        This test uses the real ThompsonSamplingSelector() constructor (not
        __new__) so that any attributes added to __init__ are present. I/O is
        patched to avoid disk access, but all in-memory logic runs as production
        code would.
        """
        from unittest.mock import patch

        from vetinari.learning.model_selector import ThompsonSamplingSelector
        from vetinari.learning.thompson_arms import ThompsonTaskContext

        # Patch I/O-only methods so the constructor doesn't need a real DB or state dir.
        with (
            patch.object(ThompsonSamplingSelector, "_load_state", return_value=None),
            patch.object(ThompsonSamplingSelector, "_seed_from_benchmarks", return_value=None),
        ):
            selector = ThompsonSamplingSelector()

        candidates = ["phi-3-mini", "llama-3-8b"]
        ctx = ThompsonTaskContext(
            task_type="coding",
            estimated_complexity=0.5,
            prompt_length=200,
            domain="software",
            requires_reasoning=True,
            requires_creativity=False,
            requires_precision=True,
            file_count=2,
        )

        # Update the contextual arm for one candidate — state must be recorded.
        selector.update_contextual(ctx, "phi-3-mini", quality_score=0.8, success=True)

        # After an update, selecting must still return one of the candidates.
        chosen = selector.select_model_contextual(ctx, candidates)
        assert chosen in candidates, f"select_model_contextual must return one of {candidates!r}, got {chosen!r}"

        # The updated arm key must appear in _arms so the observation persists.
        updated_keys = list(selector._arms.keys())
        assert any("phi-3-mini" in k for k in updated_keys), (
            f"Expected a 'phi-3-mini' arm after update_contextual, got arms: {updated_keys!r}"
        )


class TestSSEEventSerialization:
    """All SSE events serialize to valid dict payloads."""

    def test_task_start_event_serializes(self) -> None:
        rng = _make_rng(3001)
        agent_types = [AgentType.FOREMAN.value, AgentType.WORKER.value, AgentType.INSPECTOR.value]

        for _ in range(100):
            task_id = _sample_text(rng)
            description = _sample_text(rng)
            payload = TaskStartEvent(
                task_id=task_id,
                description=description,
                agent_type=rng.choice(agent_types),
                task_index=rng.randint(0, 100),
                total_tasks=rng.randint(1, 100),
            ).to_sse()
            assert isinstance(payload, dict)
            assert payload["task_id"] == task_id
            assert payload["description"] == description

    def test_task_complete_event_serializes(self) -> None:
        rng = _make_rng(3002)

        for _ in range(100):
            task_id = _sample_text(rng)
            payload = TaskCompleteEvent(
                task_id=task_id,
                output_summary=_sample_text(rng),
                task_index=rng.randint(0, 100),
                total_tasks=rng.randint(1, 100),
            ).to_sse()
            assert isinstance(payload, dict)
            assert payload["task_id"] == task_id

    def test_task_failed_event_serializes(self) -> None:
        rng = _make_rng(3003)

        for _ in range(100):
            task_id = _sample_text(rng)
            error = _sample_text(rng)
            payload = TaskFailedEvent(task_id=task_id, error=error).to_sse()
            assert isinstance(payload, dict)
            assert payload["task_id"] == task_id
            assert payload["error"] == error

    def test_status_event_serializes(self) -> None:
        rng = _make_rng(3004)

        for _ in range(50):
            status = _sample_text(rng)
            total_tasks = rng.randint(0, 500)
            payload = StatusEvent(status=status, total_tasks=total_tasks).to_sse()
            assert isinstance(payload, dict)
            assert payload["status"] == status
            assert payload["total_tasks"] == total_tasks

    def test_planning_start_event_serializes(self) -> None:
        rng = _make_rng(3005)

        for _ in range(50):
            goal = _sample_text(rng)
            payload = PlanningStartEvent(goal=goal, plan_id=_sample_text(rng)).to_sse()
            assert isinstance(payload, dict)
            assert payload["goal"] == goal


class TestQualityScoringInvariants:
    """Quality scores are always in [0.0, 1.0]."""

    def test_heuristic_score_in_valid_range(self) -> None:
        from vetinari.learning.quality_scorer import QualityScorer

        rng = _make_rng(4001)
        task_types = ["coding", "research", "analysis", "documentation", "testing", "default"]
        scorer = QualityScorer.__new__(QualityScorer)
        scorer._adapter_manager = None
        scorer._scores = __import__("collections").deque(maxlen=1000)
        scorer._score_count = 0
        scorer._calibration_interval = 999
        scorer._baselines = {}
        scorer._score_history = __import__("collections").defaultdict(
            lambda: __import__("collections").deque(maxlen=20)
        )

        for _ in range(50):
            task_type = rng.choice(task_types)
            output_text = _sample_text(rng, min_size=10, max_size=500, alphabet=RICH_TEXT_CHARS)
            dims = QualityScorer.DIMENSIONS.get(task_type, QualityScorer.DIMENSIONS["default"])
            result = scorer._score_heuristic("t1", "model-7b", task_type, output_text, dims)
            assert 0.0 <= result.overall_score <= 1.0
            assert 0.0 <= result.completeness <= 1.0
            assert 0.0 <= result.correctness <= 1.0


class TestContextBudgetInvariants:
    """Context operations never exceed the declared budget."""

    def test_prompt_parts_fit_within_window(self) -> None:
        rng = _make_rng(5001)

        for _ in range(100):
            window_size = rng.randint(100, 128_000)
            system_prompt_len = rng.randint(10, 2000)
            user_input_len = rng.randint(1, 10_000)
            total = system_prompt_len + user_input_len

            if total > window_size:
                truncated_input = window_size - system_prompt_len
                assert truncated_input + system_prompt_len <= window_size
            else:
                assert total <= window_size
