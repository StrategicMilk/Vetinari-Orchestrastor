"""Tests for prompt assembly contracts — verifies structure and content requirements.

Every agent type must produce a prompt that includes: role identity, task instructions,
and output format. Missing any of these degrades model performance.

This is the canonical contract test suite for PromptAssembler. It validates that:
- Each assembled prompt meets minimum length requirements (catch loading failures)
- Each agent type's role identity appears in its prompt
- Task-type-specific instructions are injected correctly
- Output format specs are present for relevant task types
- The result dict always contains all required metadata keys
- Cache control metadata is valid for KV-cache optimization
- The user portion always contains the original task description
- Unknown agent types degrade gracefully instead of crashing
"""

from __future__ import annotations

import pytest

from vetinari.prompts.assembler import PromptAssembler, get_prompt_assembler
from vetinari.types import AgentType

# Minimum chars for an assembled system prompt (any shorter indicates missing components)
MIN_PROMPT_CHARS = 100


class TestPromptAssemblyContract:
    """Verify that assembled prompts meet minimum structure requirements."""

    def setup_method(self):
        """Create a fresh assembler for each test to avoid state bleed."""
        self.assembler = PromptAssembler()

    @pytest.mark.parametrize(
        "agent_type",
        [
            AgentType.FOREMAN.value,
            AgentType.WORKER.value,
            AgentType.INSPECTOR.value,
        ],
    )
    def test_assembled_prompt_has_minimum_length(self, agent_type: str) -> None:
        """Each agent's assembled prompt must exceed the minimum char threshold.

        Prompts shorter than MIN_PROMPT_CHARS indicate a component loading failure
        (e.g. role definition silently empty, instructions dict miss).
        """
        result = self.assembler.build(
            agent_type=agent_type,
            task_type="general",
            task_description="Test task",
            include_examples=False,
            include_rules=False,
        )
        assert "system" in result
        assert len(result["system"]) >= MIN_PROMPT_CHARS, (
            f"{agent_type} prompt is {len(result['system'])} chars, expected >= {MIN_PROMPT_CHARS}"
        )

    @pytest.mark.parametrize(
        "agent_type",
        [
            AgentType.FOREMAN.value,
            AgentType.WORKER.value,
            AgentType.INSPECTOR.value,
        ],
    )
    def test_assembled_prompt_contains_role_identity(self, agent_type: str) -> None:
        """Each agent's prompt must contain its role name.

        The role name being absent means the agent has no identity context, which
        causes it to behave like an unspecialized generic assistant.
        """
        result = self.assembler.build(
            agent_type=agent_type,
            task_type="general",
            task_description="Test task",
            include_examples=False,
            include_rules=False,
        )
        system = result["system"].lower()
        role_keywords = {
            AgentType.FOREMAN.value: "foreman",
            AgentType.WORKER.value: "worker",
            AgentType.INSPECTOR.value: "inspector",
        }
        assert role_keywords[agent_type] in system, f"Prompt for {agent_type} does not mention its role"

    @pytest.mark.parametrize(
        "task_type,expected_keyword",
        [
            ("coding", "code"),
            ("research", "source"),
            ("planning", "plan"),
            ("review", "constructive"),
            ("security", "vulnerabilit"),
        ],
    )
    def test_task_instructions_injected(self, task_type: str, expected_keyword: str) -> None:
        """Task-type-specific instructions must appear in the assembled prompt.

        If the keyword is absent, the task-type lookup fell through to the generic
        "general" instructions, which means the model receives no domain-specific
        guidance for the task.
        """
        result = self.assembler.build(
            agent_type=AgentType.WORKER.value,
            task_type=task_type,
            task_description="Test task",
            include_examples=False,
            include_rules=False,
        )
        assert expected_keyword in result["system"].lower(), f"Expected '{expected_keyword}' in {task_type} prompt"

    def test_output_format_included_for_coding(self) -> None:
        """Coding tasks must have an output format specification containing JSON.

        Without a format spec the model produces free-form text instead of the
        structured JSON expected by the downstream parser.
        """
        result = self.assembler.build(
            agent_type=AgentType.WORKER.value,
            task_type="coding",
            task_description="Write a function",
            include_examples=False,
            include_rules=False,
        )
        assert "json" in result["system"].lower()

    def test_build_returns_required_keys(self) -> None:
        """The build result must contain all required metadata keys.

        Callers depend on these keys; missing any would cause a KeyError in the
        inference path when accessing cache_control, total_chars, etc.
        """
        result = self.assembler.build(
            agent_type=AgentType.WORKER.value,
            task_type="general",
            task_description="Test",
        )
        for key in ("system", "user", "total_chars", "agent_type", "task_type", "cache_control"):
            assert key in result, f"Missing key: {key}"

    def test_cache_control_has_prefix_chars(self) -> None:
        """Cache control metadata must include prefix_chars for KV cache optimization.

        The static prefix length tells the inference adapter where to split the prompt
        for prefix caching. A zero or absent value disables caching silently.
        """
        result = self.assembler.build(
            agent_type=AgentType.WORKER.value,
            task_type="general",
            task_description="Test",
            include_examples=False,
            include_rules=False,
        )
        cc = result["cache_control"]
        assert "prefix_chars" in cc
        assert isinstance(cc["prefix_chars"], int)
        assert cc["prefix_chars"] > 0

    def test_user_prompt_contains_task_description(self) -> None:
        """The user portion must contain the original task description verbatim.

        If the task description is absent, the model has no task to work on — it
        only receives role and instruction context with no actual request.
        """
        desc = "Implement a Redis cache wrapper for session storage"
        result = self.assembler.build(
            agent_type=AgentType.WORKER.value,
            task_type="coding",
            task_description=desc,
            include_examples=False,
            include_rules=False,
        )
        assert desc in result["user"]

    def test_unknown_agent_type_produces_valid_prompt(self) -> None:
        """Unknown agent types should still produce a valid prompt, not crash.

        The assembler has a fallback in _get_role() for unrecognised types.
        This test ensures that fallback is exercised and returns something usable
        rather than raising an exception that would abort the inference request.
        """
        result = self.assembler.build(
            agent_type="UNKNOWN_AGENT",
            task_type="general",
            task_description="Test",
            include_examples=False,
            include_rules=False,
        )
        assert "system" in result
        assert len(result["system"]) > 0

    def test_mode_parameter_accepted(self) -> None:
        """The build method must accept a mode parameter without error.

        Some callers pass a mode (e.g. "build", "research") to select agent
        sub-behaviour. If build() rejects the parameter, those callers fail.
        """
        result = self.assembler.build(
            agent_type=AgentType.WORKER.value,
            task_type="coding",
            task_description="Test",
            mode="build",
            include_examples=False,
            include_rules=False,
        )
        assert "system" in result


class TestPromptAssemblerSingleton:
    """Verify the singleton accessor works correctly."""

    def test_get_prompt_assembler_returns_instance(self) -> None:
        """get_prompt_assembler() must return a PromptAssembler, not None."""
        asm = get_prompt_assembler()
        assert isinstance(asm, PromptAssembler)

    def test_singleton_returns_same_instance(self) -> None:
        """Consecutive calls to get_prompt_assembler() must return the same object.

        A new instance per call would defeat the thread-safe singleton pattern and
        waste memory on repeated _examples_cache and _rules_cache initialisation.
        """
        a = get_prompt_assembler()
        b = get_prompt_assembler()
        assert a is b


class TestComposeVsReplace:
    """Verify the assembler's REPLACE behaviour is consistent and intentional.

    The assembler's output is a complete standalone system prompt that REPLACES
    any prior prompt from base_agent_prompts.build_system_prompt().  These tests
    confirm the output is self-contained and not a fragment expecting composition.
    """

    def test_assembler_output_is_complete_prompt(self) -> None:
        """The assembler's system output should be a complete standalone prompt,
        not a fragment to be composed with other sources.

        Verified by checking that role identity, task instructions, and minimum
        length are all present without any prior prompt context.
        """
        asm = PromptAssembler()
        result = asm.build(
            agent_type=AgentType.FOREMAN.value,
            task_type="planning",
            task_description="Create a deployment plan",
            include_examples=False,
            include_rules=False,
        )
        system = result["system"]
        # Must have role identity
        assert "foreman" in system.lower()
        # Must have task instructions
        assert "plan" in system.lower()
        # Must have structure (not just a bare sentence)
        assert len(system) > 200
