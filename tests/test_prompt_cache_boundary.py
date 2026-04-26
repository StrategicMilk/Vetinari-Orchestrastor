"""Tests for the SYSTEM_PROMPT_BOUNDARY cache boundary in base_agent_prompts.

Verifies that the boundary constant exists, that split_at_boundary correctly
divides prompts, and that build_system_prompt emits the boundary marker so
that stable content can be KV-cached across requests in a pipeline run.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vetinari.agents.base_agent_prompts import (
    SYSTEM_PROMPT_BOUNDARY,
    build_system_prompt,
    split_at_boundary,
)
from vetinari.types import AgentType

# -- SYSTEM_PROMPT_BOUNDARY constant --


def test_boundary_is_string() -> None:
    """SYSTEM_PROMPT_BOUNDARY must be a non-empty string."""
    assert isinstance(SYSTEM_PROMPT_BOUNDARY, str)
    assert len(SYSTEM_PROMPT_BOUNDARY) > 0


def test_boundary_contains_marker() -> None:
    """The boundary must contain the expected HTML-comment marker."""
    assert "SYSTEM_PROMPT_DYNAMIC_BOUNDARY" in SYSTEM_PROMPT_BOUNDARY


def test_boundary_has_surrounding_newlines() -> None:
    """Boundary must have leading/trailing newlines so sections stay separated."""
    assert SYSTEM_PROMPT_BOUNDARY.startswith("\n")
    assert SYSTEM_PROMPT_BOUNDARY.endswith("\n")


# -- split_at_boundary --


def test_split_with_boundary_present() -> None:
    """Prompt containing the boundary is split into stable and dynamic halves."""
    stable = "## Agent Identity\n\nYou are the Foreman."
    dynamic = "## Relevant Knowledge\n\n- fact one"
    prompt = stable + SYSTEM_PROMPT_BOUNDARY + dynamic

    result_stable, result_dynamic = split_at_boundary(prompt)

    assert result_stable == stable
    assert result_dynamic == dynamic


def test_split_without_boundary_returns_full_as_stable() -> None:
    """When no boundary is present the entire prompt is returned as stable."""
    prompt = "## Instructions\n\nDo the thing."

    result_stable, result_dynamic = split_at_boundary(prompt)

    assert result_stable == prompt
    assert result_dynamic == ""


def test_split_only_splits_on_first_occurrence() -> None:
    """split_at_boundary splits on the first occurrence only (maxsplit=1)."""
    marker = SYSTEM_PROMPT_BOUNDARY
    prompt = "before" + marker + "middle" + marker + "after"

    stable, dynamic = split_at_boundary(prompt)

    assert stable == "before"
    assert dynamic == "middle" + marker + "after"


def test_split_empty_dynamic_half() -> None:
    """Prompt that ends with the boundary produces an empty dynamic suffix."""
    prompt = "stable content" + SYSTEM_PROMPT_BOUNDARY

    stable, dynamic = split_at_boundary(prompt)

    assert stable == "stable content"
    assert dynamic == ""


# -- build_system_prompt with mocked agent --


def _make_mock_agent(agent_type: AgentType = AgentType.FOREMAN) -> MagicMock:
    """Return a minimal mock BaseAgent sufficient for build_system_prompt."""
    agent = MagicMock()
    agent.agent_type = agent_type
    agent.get_system_prompt.return_value = f"You are the {agent_type.value} agent."
    return agent


def test_build_system_prompt_contains_boundary_when_rag_available() -> None:
    """build_system_prompt inserts the boundary when RAG returns results."""
    agent = _make_mock_agent()

    mock_doc = MagicMock()
    mock_doc.content = "relevant knowledge snippet"

    mock_kb = MagicMock()
    mock_kb.query.return_value = [mock_doc]

    with (
        patch("vetinari.agents.base_agent_prompts.get_practices_for_mode", return_value="", create=True),
        patch("vetinari.rag.get_knowledge_base", return_value=mock_kb),
    ):
        # Suppress optional loaders so only RAG dynamic content is exercised
        with (
            patch("vetinari.agents.base_agent_prompts.build_system_prompt.__module__", create=False),
        ):
            pass

        result = build_system_prompt(agent, mode="planning")

    assert SYSTEM_PROMPT_BOUNDARY in result


def test_build_system_prompt_no_boundary_without_rag() -> None:
    """build_system_prompt omits the boundary when RAG returns no results."""
    agent = _make_mock_agent()

    mock_kb = MagicMock()
    mock_kb.query.return_value = []

    with patch("vetinari.rag.get_knowledge_base", return_value=mock_kb):
        result = build_system_prompt(agent, mode="planning")

    assert SYSTEM_PROMPT_BOUNDARY not in result


def test_build_system_prompt_no_boundary_when_rag_unavailable() -> None:
    """build_system_prompt omits the boundary when RAG raises an exception."""
    agent = _make_mock_agent()

    with patch("vetinari.rag.get_knowledge_base", side_effect=ImportError("no rag")):
        result = build_system_prompt(agent, mode="planning")

    assert SYSTEM_PROMPT_BOUNDARY not in result


def test_stable_prefix_identical_for_same_agent_and_mode() -> None:
    """Stable prefix is byte-for-byte identical across two calls for the same agent/mode."""
    agent1 = _make_mock_agent(AgentType.FOREMAN)
    agent2 = _make_mock_agent(AgentType.FOREMAN)

    with patch("vetinari.rag.get_knowledge_base", return_value=None):
        prompt1 = build_system_prompt(agent1, mode="review")
        prompt2 = build_system_prompt(agent2, mode="review")

    stable1, _ = split_at_boundary(prompt1)
    stable2, _ = split_at_boundary(prompt2)

    assert stable1 == stable2
