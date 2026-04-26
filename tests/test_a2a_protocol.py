"""Tests for A2A dispatch truth fix and agent card validation (US-002 / Session 24).

Covers:
- Dispatch failure (exception) correctly yields STATUS_FAILED, not STATUS_COMPLETED
- Orchestrator unavailable yields STATUS_ACKNOWLEDGED, not STATUS_COMPLETED
- Successful orchestrator dispatch yields STATUS_COMPLETED
- Agent card retrieval via transport JSON-RPC
- Agent card structure validation (all required fields present)
- Unknown task type yields STATUS_FAILED with a descriptive error message
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_a2a_transport
from vetinari.a2a.agent_cards import (
    get_all_cards,
    get_foreman_card,
    get_inspector_card,
    get_worker_card,
)
from vetinari.a2a.executor import (
    STATUS_ACKNOWLEDGED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    A2ATask,
    VetinariA2AExecutor,
)
from vetinari.types import AgentType, StatusEnum

# ── Dispatch truth: exception → STATUS_FAILED ─────────────────────────────────


class TestDispatchFailurePropagates:
    """Verify that orchestrator exceptions reach execute() and yield STATUS_FAILED."""

    def test_orchestrator_exception_yields_failed_not_completed(self) -> None:
        """When the real orchestrator raises, execute() must return STATUS_FAILED.

        This is the critical bug regression: before the fix, _dispatch() caught
        the exception internally and returned an acknowledgement dict, so execute()
        always marked the task STATUS_COMPLETED even when execution failed.
        """
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="build", input_data={"goal": "test"})

        boom = RuntimeError("inference server unreachable")

        # Patch get_two_layer_orchestrator to return a mock that raises on execute_task
        mock_orch = MagicMock()
        mock_orch.execute_task.side_effect = boom

        with patch(
            "vetinari.a2a.executor.get_two_layer_orchestrator",
            return_value=mock_orch,
        ):
            result = executor.execute(task)

        assert result.status == STATUS_FAILED, (
            f"Expected STATUS_FAILED when orchestrator raises, got {result.status!r}. "
            "This indicates _dispatch() is still swallowing exceptions internally."
        )
        assert "inference server unreachable" in result.error

    def test_orchestrator_exception_does_not_yield_completed(self) -> None:
        """STATUS_COMPLETED must NOT be returned when orchestrator raises."""
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="plan")

        mock_orch = MagicMock()
        mock_orch.execute_task.side_effect = ValueError("model not loaded")

        with patch(
            "vetinari.a2a.executor.get_two_layer_orchestrator",
            return_value=mock_orch,
        ):
            result = executor.execute(task)

        assert result.status != STATUS_COMPLETED, (
            "STATUS_COMPLETED returned when orchestrator raised — dispatch is silently swallowing the exception."
        )

    def test_failed_result_has_error_message(self) -> None:
        """A STATUS_FAILED result must carry a non-empty error description."""
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="review")

        mock_orch = MagicMock()
        mock_orch.execute_task.side_effect = ConnectionError("cannot reach model")

        with patch(
            "vetinari.a2a.executor.get_two_layer_orchestrator",
            return_value=mock_orch,
        ):
            result = executor.execute(task)

        assert result.status == STATUS_FAILED
        assert result.error, "STATUS_FAILED result must have a non-empty error field"
        assert "cannot reach model" in result.error


# ── Dispatch truth: orchestrator None → STATUS_ACKNOWLEDGED ───────────────────


class TestOrchestratorUnavailableAcknowledged:
    """Verify that absence of the orchestrator yields STATUS_ACKNOWLEDGED, not STATUS_COMPLETED."""

    def test_no_orchestrator_yields_acknowledged(self) -> None:
        """When get_two_layer_orchestrator() returns None, execute() must return STATUS_ACKNOWLEDGED."""
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="build", input_data={"goal": "hello world"})

        with patch(
            "vetinari.a2a.executor.get_two_layer_orchestrator",
            return_value=None,
        ):
            result = executor.execute(task)

        assert result.status == STATUS_ACKNOWLEDGED, (
            f"Expected STATUS_ACKNOWLEDGED when orchestrator is None, got {result.status!r}. "
            "Tasks must not be falsely reported as completed when no execution occurred."
        )

    def test_no_orchestrator_does_not_yield_completed(self) -> None:
        """STATUS_COMPLETED must NOT be returned when the orchestrator is unavailable."""
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="plan")

        with patch(
            "vetinari.a2a.executor.get_two_layer_orchestrator",
            return_value=None,
        ):
            result = executor.execute(task)

        assert result.status != STATUS_COMPLETED, (
            "STATUS_COMPLETED returned when orchestrator was None — "
            "acknowledgement-only path is being treated as real execution."
        )

    def test_acknowledged_output_has_is_acknowledgement_flag(self) -> None:
        """Acknowledged output must carry '_is_acknowledgement_only': True so callers can detect degraded mode."""
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="architecture")

        with patch(
            "vetinari.a2a.executor.get_two_layer_orchestrator",
            return_value=None,
        ):
            result = executor.execute(task)

        assert result.output_data.get("_is_acknowledgement_only") is True, (
            "Acknowledgement-only results must include '_is_acknowledgement_only': True "
            "so callers can distinguish real execution from degraded acceptance."
        )

    def test_acknowledged_output_has_agent_and_mode(self) -> None:
        """Acknowledgement output must still carry agent and mode for routing diagnostics."""
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="code_review")

        with patch(
            "vetinari.a2a.executor.get_two_layer_orchestrator",
            return_value=None,
        ):
            result = executor.execute(task)

        assert result.output_data["agent"] == AgentType.INSPECTOR.value
        assert result.output_data["mode"] == "code_review"

    @pytest.mark.parametrize("task_type", ["build", "plan", "review", "architecture", "summarise"])
    def test_multiple_task_types_all_acknowledged_without_orchestrator(self, task_type: str) -> None:
        """All recognized task types must return STATUS_ACKNOWLEDGED when orchestrator is None."""
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type=task_type)

        with patch(
            "vetinari.a2a.executor.get_two_layer_orchestrator",
            return_value=None,
        ):
            result = executor.execute(task)

        assert result.status == STATUS_ACKNOWLEDGED, (
            f"task_type={task_type!r} returned {result.status!r} instead of STATUS_ACKNOWLEDGED"
        )


# ── Dispatch truth: successful dispatch → STATUS_COMPLETED ────────────────────


class TestSuccessfulDispatchCompleted:
    """Verify that a real successful orchestrator dispatch yields STATUS_COMPLETED."""

    def _make_successful_orch(self) -> MagicMock:
        """Build a mock orchestrator that returns a successful result."""
        mock_result = MagicMock()
        mock_result.output = "task completed successfully"
        mock_result.success = True

        mock_orch = MagicMock()
        mock_orch.execute_task.return_value = mock_result
        return mock_orch

    def test_successful_orchestrator_yields_completed(self) -> None:
        """When orchestrator.execute_task() succeeds, execute() must return STATUS_COMPLETED."""
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="build", input_data={"goal": "add login"})

        with patch(
            "vetinari.a2a.executor.get_two_layer_orchestrator",
            return_value=self._make_successful_orch(),
        ):
            result = executor.execute(task)

        assert result.status == STATUS_COMPLETED

    def test_successful_dispatch_output_has_no_acknowledgement_flag(self) -> None:
        """Real execution output must NOT have '_is_acknowledgement_only' set."""
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="plan")

        with patch(
            "vetinari.a2a.executor.get_two_layer_orchestrator",
            return_value=self._make_successful_orch(),
        ):
            result = executor.execute(task)

        assert result.output_data.get("_is_acknowledgement_only") is not True

    def test_successful_dispatch_output_has_agent_and_mode(self) -> None:
        """Real execution output must carry agent, mode, and output fields."""
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="build")

        with patch(
            "vetinari.a2a.executor.get_two_layer_orchestrator",
            return_value=self._make_successful_orch(),
        ):
            result = executor.execute(task)

        assert result.output_data["agent"] == AgentType.WORKER.value
        assert result.output_data["mode"] == "build"
        assert "output" in result.output_data


# ── Unknown task type → STATUS_FAILED ────────────────────────────────────────


class TestUnknownTaskType:
    """Unknown task types must fail with a descriptive error message."""

    def test_unknown_task_type_yields_failed(self) -> None:
        """An unrecognised task_type must return STATUS_FAILED, not raise."""
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="not_a_real_task_type_ever")
        result = executor.execute(task)
        assert result.status == STATUS_FAILED

    def test_unknown_task_type_error_names_the_bad_type(self) -> None:
        """The error message must name the unrecognised task type."""
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="flying_unicorn")
        result = executor.execute(task)
        assert "flying_unicorn" in result.error

    def test_unknown_task_type_error_lists_supported_types(self) -> None:
        """The error message should include a list of supported task types."""
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="completely_unknown")
        result = executor.execute(task)
        # Should mention at least one known type so the caller can correct themselves
        assert any(known in result.error for known in ("build", "plan", "review"))


# ── Agent card retrieval via transport JSON-RPC ───────────────────────────────


class TestAgentCardRetrieval:
    """Validate agent card retrieval via the A2ATransport JSON-RPC method."""

    def test_get_agent_card_returns_3_agents(self) -> None:
        """a2a.getAgentCard must return exactly 3 agent cards (Foreman, Worker, Inspector)."""
        transport = make_a2a_transport()
        request = {"jsonrpc": "2.0", "id": 1, "method": "a2a.getAgentCard", "params": {}}
        response = transport.handle_request(request)
        assert "result" in response, f"Expected result, got error: {response.get('error')}"
        assert len(response["result"]["agents"]) == 3

    def test_get_agent_card_has_foreman_worker_inspector(self) -> None:
        """a2a.getAgentCard must include one card per Vetinari agent type."""
        transport = make_a2a_transport()
        request = {"jsonrpc": "2.0", "id": 1, "method": "a2a.getAgentCard", "params": {}}
        response = transport.handle_request(request)
        agent_types = {a["agentType"] for a in response["result"]["agents"]}
        assert AgentType.FOREMAN.value in agent_types
        assert AgentType.WORKER.value in agent_types
        assert AgentType.INSPECTOR.value in agent_types

    def test_get_agent_card_response_is_json_rpc_compliant(self) -> None:
        """a2a.getAgentCard response must be a valid JSON-RPC 2.0 success envelope."""
        transport = make_a2a_transport()
        request = {"jsonrpc": "2.0", "id": 42, "method": "a2a.getAgentCard", "params": {}}
        response = transport.handle_request(request)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 42
        assert "result" in response
        assert "error" not in response


# ── Agent card structure validation ──────────────────────────────────────────

_REQUIRED_CARD_FIELDS = {
    "name",
    "description",
    "url",
    "version",
    "capabilities",
    "supportedInputTypes",
    "supportedOutputTypes",
    "skills",
    "agentType",
}


class TestAgentCardStructure:
    """Validate that each agent card carries the required A2A protocol fields."""

    @pytest.mark.parametrize(
        "card_fn,expected_type",
        [
            (get_foreman_card, AgentType.FOREMAN),
            (get_worker_card, AgentType.WORKER),
            (get_inspector_card, AgentType.INSPECTOR),
        ],
    )
    def test_card_has_all_required_fields(self, card_fn: Any, expected_type: AgentType) -> None:
        """Every agent card must carry all required A2A protocol fields."""
        card = card_fn()
        d = card.to_dict()
        missing = _REQUIRED_CARD_FIELDS - set(d.keys())
        assert not missing, f"{expected_type.value} card missing fields: {missing}"

    @pytest.mark.parametrize(
        "card_fn,expected_type",
        [
            (get_foreman_card, AgentType.FOREMAN),
            (get_worker_card, AgentType.WORKER),
            (get_inspector_card, AgentType.INSPECTOR),
        ],
    )
    def test_card_agent_type_matches(self, card_fn: Any, expected_type: AgentType) -> None:
        """Card agentType field must match the agent's enum value."""
        card = card_fn()
        assert card.agent_type == expected_type
        assert card.to_dict()["agentType"] == expected_type.value

    @pytest.mark.parametrize("card_fn", [get_foreman_card, get_worker_card, get_inspector_card])
    def test_card_has_at_least_one_skill(self, card_fn: Any) -> None:
        """Every agent card must advertise at least one skill."""
        card = card_fn()
        assert card.skills, f"{card.name} must have at least one skill"

    @pytest.mark.parametrize("card_fn", [get_foreman_card, get_worker_card, get_inspector_card])
    def test_card_skills_have_required_keys(self, card_fn: Any) -> None:
        """Every skill in every card must have 'id', 'name', 'description', and 'tags'."""
        card = card_fn()
        for skill in card.skills:
            for key in ("id", "name", "description", "tags"):
                assert key in skill, f"{card.name} skill {skill.get('id', '?')} is missing required key '{key}'"

    @pytest.mark.parametrize("card_fn", [get_foreman_card, get_worker_card, get_inspector_card])
    def test_card_url_is_non_empty_string(self, card_fn: Any) -> None:
        """Each card must have a non-empty URL."""
        card = card_fn()
        assert card.url, f"{card.name} card URL must not be empty"
        assert card.url.startswith("http"), f"{card.name} card URL must start with http"

    def test_get_all_cards_factory_consistent_with_individual_factories(self) -> None:
        """get_all_cards() must return the same data as calling each factory individually."""
        all_cards = get_all_cards()
        individual = [get_foreman_card(), get_worker_card(), get_inspector_card()]
        assert len(all_cards) == len(individual) == 3
        for a, b in zip(all_cards, individual):
            assert a.name == b.name
            assert a.agent_type == b.agent_type
            assert len(a.skills) == len(b.skills)

    def test_card_version_is_semver_like(self) -> None:
        """Card version must follow a semantic version pattern (X.Y.Z)."""
        for card in get_all_cards():
            parts = card.version.split(".")
            assert len(parts) == 3, f"{card.name} version {card.version!r} is not X.Y.Z"
            assert all(p.isdigit() for p in parts), f"{card.name} version {card.version!r} has non-numeric parts"


# ── StatusEnum integrity ──────────────────────────────────────────────────────


class TestStatusEnumAcknowledged:
    """Verify the ACKNOWLEDGED value was added correctly to StatusEnum."""

    def test_acknowledged_value_is_in_status_enum(self) -> None:
        """StatusEnum.ACKNOWLEDGED must exist and equal 'acknowledged'."""
        assert StatusEnum.ACKNOWLEDGED.value == "acknowledged"

    def test_status_acknowledged_constant_matches_enum(self) -> None:
        """The STATUS_ACKNOWLEDGED module constant must equal StatusEnum.ACKNOWLEDGED.value."""
        assert StatusEnum.ACKNOWLEDGED.value == STATUS_ACKNOWLEDGED
