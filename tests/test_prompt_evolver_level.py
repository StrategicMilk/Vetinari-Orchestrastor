"""Tests for PromptEvolver.evolve_per_level and synthesize_scope_guidelines (Gaps 5.14)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vetinari.learning.prompt_evolver import PromptEvolver, PromptVariant
from vetinari.types import AgentType, PromptVersionStatus


@pytest.fixture
def evolver() -> PromptEvolver:
    """Return a PromptEvolver with no adapter and isolated state."""
    with patch.object(PromptEvolver, "_load_variants", return_value=None):
        ev = PromptEvolver(adapter_manager=None)
    # Register a promoted baseline so select_prompt returns something
    ev._variants[AgentType.WORKER.value] = [
        PromptVariant(
            variant_id="WORKER_baseline",
            agent_type=AgentType.WORKER.value,
            prompt_text="You are a Worker agent.",
            is_baseline=True,
            status=PromptVersionStatus.PROMOTED.value,
        )
    ]
    ev._variants[AgentType.INSPECTOR.value] = [
        PromptVariant(
            variant_id="INSPECTOR_baseline",
            agent_type=AgentType.INSPECTOR.value,
            prompt_text="You are an Inspector.",
            is_baseline=True,
            status=PromptVersionStatus.PROMOTED.value,
        )
    ]
    return ev


class TestEvolvePerLevel:
    def test_returns_dict_with_variant_id_and_evolved_prompt(self, evolver: PromptEvolver) -> None:
        """evolve_per_level always returns a dict with the expected keys."""
        with patch.object(evolver, "generate_variant", return_value="improved prompt"):
            result = evolver.evolve_per_level(AgentType.WORKER.value, "build")
        assert "variant_id" in result
        assert "evolved_prompt" in result

    def test_returns_none_when_no_baseline(self, evolver: PromptEvolver) -> None:
        """evolve_per_level returns None values when agent has no registered baseline."""
        result = evolver.evolve_per_level(AgentType.FOREMAN.value, "default")
        assert result["variant_id"] is None
        assert result["evolved_prompt"] is None

    def test_uses_trace_based_evolution_when_traces_provided(self, evolver: PromptEvolver) -> None:
        """evolve_per_level calls generate_variant_from_trace when failed_traces given."""
        trace = {"prompt": "p", "output": "o", "inspector_verdict": {"passed": False}}
        with patch.object(evolver, "generate_variant_from_trace", return_value="trace-evolved prompt") as mock_trace:
            result = evolver.evolve_per_level(AgentType.WORKER.value, "build", failed_traces=[trace])
        mock_trace.assert_called_once()
        assert result["evolved_prompt"] == "trace-evolved prompt"
        assert "evolved" in result["variant_id"]  # type: ignore[operator]

    def test_falls_back_to_blind_mutation_when_trace_returns_same(self, evolver: PromptEvolver) -> None:
        """evolve_per_level falls back to generate_variant when trace returns baseline."""
        baseline = "You are a Worker agent."
        trace = {"prompt": "p", "output": "o"}
        with (
            patch.object(evolver, "generate_variant_from_trace", return_value=baseline),
            patch.object(evolver, "generate_variant", return_value="mutated prompt") as mock_mut,
        ):
            result = evolver.evolve_per_level(AgentType.WORKER.value, "build", failed_traces=[trace])
        mock_mut.assert_called_once()
        assert result["evolved_prompt"] == "mutated prompt"

    def test_worker_hint_contains_mode_name(self, evolver: PromptEvolver) -> None:
        """evolve_per_level injects the mode name into Worker's hint."""
        captured: list[str] = []

        def capture_generate(agent_type: str, prompt: str, mode: str) -> str:
            captured.append(prompt)
            return "evolved"

        with patch.object(evolver, "generate_variant", side_effect=capture_generate):
            evolver.evolve_per_level(AgentType.WORKER.value, "review")

        assert len(captured) == 1
        assert "review" in captured[0]

    def test_inspector_hint_mentions_workers(self, evolver: PromptEvolver) -> None:
        """evolve_per_level injects Inspector-specific hint about Worker failures."""
        captured: list[str] = []

        def capture_generate(agent_type: str, prompt: str, mode: str) -> str:
            captured.append(prompt)
            return "evolved"

        with patch.object(evolver, "generate_variant", side_effect=capture_generate):
            evolver.evolve_per_level(AgentType.INSPECTOR.value, "default")

        assert "Worker" in captured[0]

    def test_mutated_variant_id_includes_agent_and_mode(self, evolver: PromptEvolver) -> None:
        """evolve_per_level encodes agent_type and mode in the fallback variant_id."""
        with patch.object(evolver, "generate_variant", return_value="m"):
            result = evolver.evolve_per_level(AgentType.WORKER.value, "build")
        assert result["variant_id"] == "WORKER_build_mutated"


class TestSynthesizeScopeGuidelines:
    def test_returns_empty_string_when_fewer_than_3_traces(self, evolver: PromptEvolver) -> None:
        """synthesize_scope_guidelines returns '' when < 3 failed traces available."""
        mock_collector = MagicMock()
        mock_collector.get_recent_traces.return_value = [
            {"task_id": "t1", "inspector_verdict": {"passed": False, "issues": ["incomplete"]}}
        ]
        with patch(
            "vetinari.learning.training_data.get_training_collector",
            return_value=mock_collector,
        ):
            result = evolver.synthesize_scope_guidelines(AgentType.WORKER.value, "build")
        assert result == ""

    def test_returns_guidelines_for_sufficient_traces(self, evolver: PromptEvolver) -> None:
        """synthesize_scope_guidelines returns non-empty string with 3+ failed traces."""
        failed_traces = [
            {
                "task_id": f"t{i}",
                "inspector_verdict": {
                    "passed": False,
                    "issues": ["incomplete output"],
                },
            }
            for i in range(4)
        ]
        mock_collector = MagicMock()
        mock_collector.get_recent_traces.return_value = failed_traces
        with patch(
            "vetinari.learning.training_data.get_training_collector",
            return_value=mock_collector,
        ):
            result = evolver.synthesize_scope_guidelines(AgentType.WORKER.value, "build")
        assert result != ""
        assert AgentType.WORKER.value in result
        assert "build" in result

    def test_includes_guideline_for_incomplete_output_failures(self, evolver: PromptEvolver) -> None:
        """synthesize_scope_guidelines maps incomplete_output failures to the correct guideline."""
        failed_traces = [
            {
                "task_id": f"t{i}",
                "inspector_verdict": {
                    "passed": False,
                    "issues": ["incomplete output detected"],
                },
            }
            for i in range(3)
        ]
        mock_collector = MagicMock()
        mock_collector.get_recent_traces.return_value = failed_traces
        with patch(
            "vetinari.learning.training_data.get_training_collector",
            return_value=mock_collector,
        ):
            result = evolver.synthesize_scope_guidelines(AgentType.INSPECTOR.value, "default")
        assert "complete" in result.lower()

    def test_returns_empty_string_when_no_categorised_issues(self, evolver: PromptEvolver) -> None:
        """synthesize_scope_guidelines returns '' when traces have no issues list."""
        failed_traces = [{"task_id": f"t{i}", "inspector_verdict": {"passed": False}} for i in range(5)]
        mock_collector = MagicMock()
        mock_collector.get_recent_traces.return_value = failed_traces
        with patch(
            "vetinari.learning.training_data.get_training_collector",
            return_value=mock_collector,
        ):
            result = evolver.synthesize_scope_guidelines(AgentType.WORKER.value)
        assert result == ""

    def test_handles_import_error_gracefully(self, evolver: PromptEvolver) -> None:
        """synthesize_scope_guidelines returns '' on exception rather than raising."""
        with patch(
            "vetinari.learning.training_data.get_training_collector",
            side_effect=RuntimeError("simulated failure"),
        ):
            result = evolver.synthesize_scope_guidelines(AgentType.WORKER.value, "build")
        assert result == ""
