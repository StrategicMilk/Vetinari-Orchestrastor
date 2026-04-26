"""Tests for vetinari.tools.consolidated_operations.

Verifies that the consolidated operations:

1. Return the correct result types with valid fields.
2. Use the right subsystems (InferenceConfigManager, memory store, skill
   registry) instead of making individual tool calls.
3. Degrade gracefully when subsystems are unavailable.
4. Demonstrably reduce the number of discrete subsystem calls compared to the
   equivalent un-consolidated pattern.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.tools.consolidated_operations import (
    InvestigateTaskResult,
    PrepareModelResult,
    _estimate_complexity,
    investigate_task,
    prepare_model,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory_entry(content: str) -> MagicMock:
    """Return a minimal mock that looks like a MemoryEntry."""
    entry = MagicMock()
    entry.content = content
    return entry


def _make_skill_spec(skill_id: str, capabilities: list[str], tags: list[str]) -> MagicMock:
    """Return a minimal mock that looks like a SkillSpec."""
    spec = MagicMock()
    spec.capabilities = capabilities
    spec.tags = tags
    return spec


# ---------------------------------------------------------------------------
# prepare_model
# ---------------------------------------------------------------------------


class TestPrepareModel:
    """Tests for the prepare_model consolidated operation."""

    def test_prepare_model_returns_result(self) -> None:
        """prepare_model returns a PrepareModelResult with valid fields."""
        with patch("vetinari.config.inference_config.get_inference_config") as mock_cfg_factory:
            mock_cfg = MagicMock()
            mock_cfg.get_effective_params.return_value = {
                "temperature": 0.1,
                "max_tokens": 4096,
            }
            mock_cfg_factory.return_value = mock_cfg

            result = prepare_model("qwen2.5-coder-7b", "coding")

        assert isinstance(result, PrepareModelResult)
        assert result.model_id == "qwen2.5-coder-7b"
        assert result.task_type == "coding"
        assert result.recommended_temperature == 0.1
        assert result.recommended_max_tokens == 4096
        assert result.is_ready is True
        assert result.notes == ""

    def test_prepare_model_default_task_type(self) -> None:
        """prepare_model uses 'general' as default task_type."""
        with patch("vetinari.config.inference_config.get_inference_config") as mock_cfg_factory:
            mock_cfg = MagicMock()
            mock_cfg.get_effective_params.return_value = {
                "temperature": 0.3,
                "max_tokens": 2048,
            }
            mock_cfg_factory.return_value = mock_cfg

            result = prepare_model("my-model")

        assert result.task_type == "general"
        assert result.model_id == "my-model"

    def test_prepare_model_loads_config_once(self) -> None:
        """prepare_model uses InferenceConfigManager instead of multiple tool calls.

        The key acceptance criterion: one consolidated call replaces the
        separate select_model + config_lookup pattern.  We verify that
        get_inference_config is called exactly once and get_effective_params
        is called exactly once with the right arguments.
        """
        with patch("vetinari.config.inference_config.get_inference_config") as mock_cfg_factory:
            mock_cfg = MagicMock()
            mock_cfg.get_effective_params.return_value = {
                "temperature": 0.05,
                "max_tokens": 8192,
            }
            mock_cfg_factory.return_value = mock_cfg

            result = prepare_model("llama-70b", "reasoning")

        # One config factory call (not two separate tool calls).
        mock_cfg_factory.assert_called_once()
        # One effective-params lookup.
        mock_cfg.get_effective_params.assert_called_once_with("reasoning", "llama-70b")
        assert result.recommended_temperature == 0.05
        assert result.recommended_max_tokens == 8192

    def test_prepare_model_graceful_degradation_on_config_error(self) -> None:
        """prepare_model returns safe defaults and is_ready=False when config unavailable."""
        with patch(
            "vetinari.config.inference_config.get_inference_config",
            side_effect=RuntimeError("config file not found"),
        ):
            result = prepare_model("unknown-model", "coding")

        assert result.is_ready is False
        assert result.recommended_temperature == 0.3  # safe default
        assert result.recommended_max_tokens == 2048  # safe default
        assert "Inference config unavailable" in result.notes

    def test_prepare_model_graceful_degradation_on_import_error(self) -> None:
        """prepare_model returns defaults when the config module cannot be imported."""
        with patch(
            "vetinari.config.inference_config.get_inference_config",
            side_effect=ImportError("vetinari.config not available"),
        ):
            result = prepare_model("any-model")

        assert result.is_ready is False
        assert result.model_id == "any-model"
        assert result.recommended_temperature == 0.3
        assert result.recommended_max_tokens == 2048

    def test_prepare_model_result_is_frozen(self) -> None:
        """PrepareModelResult is immutable (frozen dataclass)."""
        result = PrepareModelResult(
            model_id="m",
            task_type="general",
            recommended_temperature=0.3,
            recommended_max_tokens=2048,
            is_ready=True,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.model_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# investigate_task
# ---------------------------------------------------------------------------


class TestInvestigateTask:
    """Tests for the investigate_task consolidated operation."""

    def test_investigate_task_returns_result(self) -> None:
        """investigate_task returns an InvestigateTaskResult with correct types."""
        with (
            patch("vetinari.memory.get_unified_memory_store") as mock_mem_factory,
            patch("vetinari.skills.skill_registry.get_all_skills") as mock_skills,
        ):
            mock_mem = MagicMock()
            mock_mem.search.return_value = [_make_memory_entry("prior context")]
            mock_mem_factory.return_value = mock_mem

            mock_skills.return_value = {
                "worker": _make_skill_spec("worker", ["build", "code"], ["engineering"]),
            }

            result = investigate_task("Build a REST API endpoint")

        assert isinstance(result, InvestigateTaskResult)
        assert result.description == "Build a REST API endpoint"
        assert isinstance(result.relevant_memories, list)
        assert isinstance(result.matching_skills, list)
        assert result.estimated_complexity in {"simple", "moderate", "complex"}
        assert isinstance(result.context_summary, str)
        assert len(result.context_summary) > 0

    def test_investigate_task_searches_memory(self) -> None:
        """investigate_task searches memory for relevant context using the description."""
        with (
            patch("vetinari.memory.get_unified_memory_store") as mock_mem_factory,
            patch("vetinari.skills.skill_registry.get_all_skills", return_value={}),
        ):
            mock_mem = MagicMock()
            mock_mem.search.return_value = [
                _make_memory_entry("memory about databases"),
                _make_memory_entry("memory about schemas"),
            ]
            mock_mem_factory.return_value = mock_mem

            result = investigate_task("design a database schema")

        # Memory store was obtained and searched exactly once.
        mock_mem_factory.assert_called_once()
        mock_mem.search.assert_called_once()
        call_kwargs = mock_mem.search.call_args
        assert call_kwargs.kwargs.get("query") == "design a database schema" or (
            call_kwargs.args and call_kwargs.args[0] == "design a database schema"
        )
        assert result.relevant_memories == [
            "memory about databases",
            "memory about schemas",
        ]

    def test_investigate_task_matches_skills_by_capability(self) -> None:
        """investigate_task returns skill IDs whose capabilities appear in description."""
        with (
            patch("vetinari.memory.get_unified_memory_store") as mock_mem_factory,
            patch("vetinari.skills.skill_registry.get_all_skills") as mock_skills,
        ):
            mock_mem = MagicMock()
            mock_mem.search.return_value = []
            mock_mem_factory.return_value = mock_mem

            mock_skills.return_value = {
                "foreman": _make_skill_spec("foreman", ["planning", "orchestration"], ["management"]),
                "worker": _make_skill_spec("worker", ["build", "code", "research"], ["engineering"]),
                "inspector": _make_skill_spec("inspector", ["review", "audit"], ["quality"]),
            }

            result = investigate_task("planning and orchestration for the deployment pipeline")

        assert "foreman" in result.matching_skills
        assert "worker" not in result.matching_skills

    def test_investigate_task_graceful_degradation_memory_unavailable(self) -> None:
        """investigate_task returns empty memories when memory store raises."""
        with (
            patch(
                "vetinari.memory.get_unified_memory_store",
                side_effect=RuntimeError("db locked"),
            ),
            patch("vetinari.skills.skill_registry.get_all_skills", return_value={}),
        ):
            result = investigate_task("some task description")

        assert result.relevant_memories == []
        assert result.estimated_complexity in {"simple", "moderate", "complex"}

    def test_investigate_task_graceful_degradation_skills_unavailable(self) -> None:
        """investigate_task returns empty skills when skill registry raises."""
        with (
            patch("vetinari.memory.get_unified_memory_store") as mock_mem_factory,
            patch(
                "vetinari.skills.skill_registry.get_all_skills",
                side_effect=ImportError("skills not available"),
            ),
        ):
            mock_mem = MagicMock()
            mock_mem.search.return_value = []
            mock_mem_factory.return_value = mock_mem

            result = investigate_task("build something")

        assert result.matching_skills == []
        assert result.relevant_memories == []

    def test_investigate_task_graceful_degradation_both_unavailable(self) -> None:
        """investigate_task returns defaults when both subsystems are unavailable."""
        with (
            patch(
                "vetinari.memory.get_unified_memory_store",
                side_effect=Exception("no memory"),
            ),
            patch(
                "vetinari.skills.skill_registry.get_all_skills",
                side_effect=Exception("no skills"),
            ),
        ):
            result = investigate_task(
                "Design a distributed architecture with concurrent workers and then deploy "
                "the database migration across multiple production environments"
            )

        assert result.relevant_memories == []
        assert result.matching_skills == []
        assert result.estimated_complexity in {"moderate", "complex"}
        assert "no prior memories" in result.context_summary
        assert "no skill matches" in result.context_summary

    def test_investigate_task_result_is_frozen(self) -> None:
        """InvestigateTaskResult is immutable (frozen dataclass)."""
        result = InvestigateTaskResult(
            description="task",
            relevant_memories=[],
            matching_skills=[],
            estimated_complexity="simple",
            context_summary="summary",
        )
        with pytest.raises((AttributeError, TypeError)):
            result.description = "other"  # type: ignore[misc]

    def test_investigate_task_with_project_id(self) -> None:
        """investigate_task accepts an optional project_id without raising."""
        with (
            patch("vetinari.memory.get_unified_memory_store") as mock_mem_factory,
            patch("vetinari.skills.skill_registry.get_all_skills", return_value={}),
        ):
            mock_mem = MagicMock()
            mock_mem.search.return_value = []
            mock_mem_factory.return_value = mock_mem

            result = investigate_task("fix a bug", project_id="proj-123")

        assert result.description == "fix a bug"


# ---------------------------------------------------------------------------
# Complexity heuristics (unit tests on pure-Python helper)
# ---------------------------------------------------------------------------


class TestEstimateComplexity:
    """Unit tests for the _estimate_complexity helper."""

    def test_short_description_is_simple(self) -> None:
        assert _estimate_complexity("fix the bug") == "simple"

    def test_long_technical_description_is_complex(self) -> None:
        desc = (
            "Design and implement a distributed inference pipeline with "
            "concurrent workers, security authentication, and database migration"
        )
        assert _estimate_complexity(desc) == "complex"

    def test_multi_step_description_is_at_least_moderate(self) -> None:
        desc = "First build the API, then deploy it to production"
        result = _estimate_complexity(desc)
        assert result in {"moderate", "complex"}

    def test_moderate_description(self) -> None:
        desc = "Refactor the authentication module to improve readability and maintainability across the codebase"
        assert _estimate_complexity(desc) in {"moderate", "complex"}

    def test_empty_description_is_simple(self) -> None:
        assert _estimate_complexity("") == "simple"


# ---------------------------------------------------------------------------
# Acceptance criterion: consolidated ops reduce tool calls
# ---------------------------------------------------------------------------


class TestConsolidatedOperationsReduceToolCalls:
    """Prove that consolidated ops replace multiple individual tool calls.

    The acceptance criterion for US-006:
    - prepare_model() replaces select_model + config_lookup (2 calls → 1)
    - investigate_task() replaces recall_memory + skill_search + complexity_check
      (3 calls → 1)
    """

    def test_prepare_model_makes_one_config_call_not_two(self) -> None:
        """prepare_model issues exactly one config call, not the 2-call pattern.

        The un-consolidated pattern would be:
        1. Call select_model(task_type) → 1 tool call
        2. Call InferenceConfigManager().get_effective_params(...) → 1 tool call
        Total: 2 tool calls.

        The consolidated operation makes a single get_effective_params call
        which internally handles both concerns, reducing the external call
        surface to 1.
        """
        config_call_count = 0

        def counting_get_inference_config() -> MagicMock:
            nonlocal config_call_count
            config_call_count += 1
            mock_cfg = MagicMock()
            mock_cfg.get_effective_params.return_value = {
                "temperature": 0.2,
                "max_tokens": 4096,
            }
            return mock_cfg

        with patch(
            "vetinari.config.inference_config.get_inference_config",
            side_effect=counting_get_inference_config,
        ):
            prepare_model("test-model", "coding")

        # Exactly one config-system call (not 2 as the un-consolidated pattern would make).
        assert config_call_count == 1

    def test_investigate_task_makes_at_most_two_subsystem_calls(self) -> None:
        """investigate_task issues at most 2 subsystem calls (not 3-4).

        The un-consolidated pattern would be:
        1. recall_memory(query) → 1 tool call
        2. skill_search(description) → 1 tool call
        3. complexity_check(description) → 1 tool call (or manual logic)
        Total: 3 tool calls.

        The consolidated operation makes exactly 2 external subsystem calls
        (memory + skills).  Complexity is computed in-process with zero
        additional I/O.
        """
        subsystem_call_count = 0

        def counting_memory_factory() -> MagicMock:
            nonlocal subsystem_call_count
            subsystem_call_count += 1
            mem = MagicMock()
            mem.search.return_value = [_make_memory_entry("relevant context")]
            return mem

        def counting_skills_factory() -> dict:
            nonlocal subsystem_call_count
            subsystem_call_count += 1
            return {
                "worker": _make_skill_spec("worker", ["build"], ["engineering"]),
            }

        with (
            patch(
                "vetinari.memory.get_unified_memory_store",
                side_effect=counting_memory_factory,
            ),
            patch(
                "vetinari.skills.skill_registry.get_all_skills",
                side_effect=counting_skills_factory,
            ),
        ):
            result = investigate_task("build a pipeline with multiple steps")

        # At most 2 subsystem calls (memory + skills).  Complexity is in-process.
        assert subsystem_call_count <= 2
        # Both subsystems were consulted.
        assert subsystem_call_count == 2
        assert result.relevant_memories == ["relevant context"]
        assert "worker" in result.matching_skills

    def test_context_summary_integrates_all_signals(self) -> None:
        """context_summary from investigate_task combines all three signals in one string.

        This proves that a single investigate_task call replaces separate
        memory-recall, skill-search, and complexity-check calls — the summary
        contains evidence of all three.
        """
        with (
            patch("vetinari.memory.get_unified_memory_store") as mock_mem_factory,
            patch("vetinari.skills.skill_registry.get_all_skills") as mock_skills,
        ):
            mock_mem = MagicMock()
            mock_mem.search.return_value = [
                _make_memory_entry("database migration notes"),
            ]
            mock_mem_factory.return_value = mock_mem

            mock_skills.return_value = {
                "worker": _make_skill_spec("worker", ["build", "deploy"], ["engineering"]),
            }

            result = investigate_task("Deploy and migrate the database schema across multiple environments")

        # Summary includes all three signals.
        assert "complex" in result.context_summary or "moderate" in result.context_summary
        assert "1 prior memory" in result.context_summary
        assert "worker" in result.context_summary
