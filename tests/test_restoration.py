"""Tests for vetinari.context.restoration — post-compaction context re-injection."""

from __future__ import annotations

import pytest

from vetinari.context.restoration import (
    ContextRestorer,
    RestorationBudget,
    RestorationContext,
    RestorationResult,
    get_restorer,
)
from vetinari.context.session_state import SessionState
from vetinari.context.window_manager import WindowConversationMessage, estimate_tokens

# ── Helpers ────────────────────────────────────────────────────────────


def _make_session_state(task_id: str = "task-1") -> SessionState:
    return SessionState(
        task_id=task_id,
        stage="execution",
        key_decisions=["decided to use qwen2.5"],
        outputs_produced=["vetinari/context/restoration.py"],
        quality_scores={"score": 0.9},
        model_used="qwen2.5-coder-7b",
        token_count=100,
        timestamp=1000.0,
    )


def _make_long_text(word: str = "word", count: int = 2000) -> str:
    """Generate a string long enough to exceed typical token budgets."""
    return (word + " ") * count


# ── RestorationBudget ──────────────────────────────────────────────────


class TestRestorationBudget:
    def test_default_values(self):
        budget = RestorationBudget()
        assert budget.agent_instructions == 1000
        assert budget.task_and_plan == 2000
        assert budget.recent_output == 2000
        assert budget.memories == 1500

    def test_total_sums_all_fields(self):
        budget = RestorationBudget(
            agent_instructions=500,
            task_and_plan=1000,
            recent_output=800,
            memories=700,
        )
        assert budget.total == 500 + 1000 + 800 + 700

    def test_default_total(self):
        budget = RestorationBudget()
        assert budget.total == 1000 + 2000 + 2000 + 1500

    def test_frozen_raises_on_mutation(self):
        budget = RestorationBudget()
        with pytest.raises((AttributeError, TypeError)):
            budget.agent_instructions = 9999  # type: ignore[misc]


# ── RestorationContext ─────────────────────────────────────────────────


class TestRestorationContext:
    def test_defaults_are_empty(self):
        ctx = RestorationContext()
        assert ctx.agent_instructions == ""
        assert ctx.task_description == ""
        assert ctx.plan_summary == ""
        assert ctx.recent_output == ""
        assert ctx.memories == []
        assert ctx.session_state is None

    def test_accepts_session_state(self):
        state = _make_session_state()
        ctx = RestorationContext(session_state=state)
        assert ctx.session_state is state

    def test_memories_default_factory_is_independent(self):
        ctx_a = RestorationContext()
        ctx_b = RestorationContext()
        ctx_a.memories.append("note")
        assert ctx_b.memories == []


# ── RestorationResult ──────────────────────────────────────────────────


class TestRestorationResult:
    def test_frozen_raises_on_mutation(self):
        result = RestorationResult(
            messages_injected=1,
            tokens_injected=50,
            budget_used={"agent_instructions": 50},
            truncated=[],
        )
        with pytest.raises((AttributeError, TypeError)):
            result.messages_injected = 99  # type: ignore[misc]

    def test_fields_accessible(self):
        result = RestorationResult(
            messages_injected=3,
            tokens_injected=400,
            budget_used={"agent_instructions": 100, "task_and_plan": 200, "recent_output": 100},
            truncated=["recent_output"],
        )
        assert result.messages_injected == 3
        assert result.tokens_injected == 400
        assert result.budget_used["task_and_plan"] == 200
        assert "recent_output" in result.truncated


# ── ContextRestorer._truncate_to_budget ────────────────────────────────


class TestTruncateToBudget:
    @pytest.fixture
    def restorer(self) -> ContextRestorer:
        return ContextRestorer()

    def test_short_text_not_truncated(self, restorer):
        text = "short text"
        result, was_truncated = restorer._truncate_to_budget(text, max_tokens=500)
        assert result == text
        assert was_truncated is False

    def test_empty_text_not_truncated(self, restorer):
        result, was_truncated = restorer._truncate_to_budget("", max_tokens=100)
        assert result == ""
        assert was_truncated is False

    def test_long_text_is_truncated(self, restorer):
        long_text = _make_long_text(count=1000)
        result, was_truncated = restorer._truncate_to_budget(long_text, max_tokens=50)
        assert was_truncated is True
        assert "[...truncated...]" in result
        assert estimate_tokens(result) <= 55  # small slack for heuristic rounding

    def test_truncated_text_ends_with_marker(self, restorer):
        long_text = _make_long_text(count=500)
        result, was_truncated = restorer._truncate_to_budget(long_text, max_tokens=30)
        assert was_truncated is True
        assert result.endswith("[...truncated...]")

    def test_exact_budget_not_truncated(self, restorer):
        text = "hello world"
        tokens = estimate_tokens(text)
        result, was_truncated = restorer._truncate_to_budget(text, max_tokens=tokens)
        assert was_truncated is False
        assert result == text


# ── ContextRestorer._build_restoration_messages ────────────────────────


class TestBuildRestorationMessages:
    @pytest.fixture
    def restorer(self) -> ContextRestorer:
        return ContextRestorer()

    def test_empty_context_produces_no_messages(self, restorer):
        ctx = RestorationContext()
        msgs = restorer._build_restoration_messages(ctx)
        assert msgs == []

    def test_agent_instructions_message_appears_first(self, restorer):
        ctx = RestorationContext(
            agent_instructions="You are a Foreman agent.",
            task_description="Build module X.",
        )
        msgs = restorer._build_restoration_messages(ctx)
        assert msgs[0].metadata["restoration_category"] == "agent_instructions"

    def test_categories_in_priority_order(self, restorer):
        ctx = RestorationContext(
            agent_instructions="Role: Foreman",
            task_description="Task: write tests",
            plan_summary="1. do A\n2. do B",
            recent_output="Worker output: done",
            memories=["memory snippet"],
        )
        msgs = restorer._build_restoration_messages(ctx)
        categories = [m.metadata["restoration_category"] for m in msgs]
        assert categories == ["agent_instructions", "task_and_plan", "recent_output", "memories"]

    def test_task_and_plan_combined_in_one_message(self, restorer):
        ctx = RestorationContext(
            task_description="Write the context module.",
            plan_summary="Step 1: design\nStep 2: code",
        )
        msgs = restorer._build_restoration_messages(ctx)
        assert len(msgs) == 1
        assert msgs[0].metadata["restoration_category"] == "task_and_plan"
        assert "Task:" in msgs[0].content
        assert "Plan:" in msgs[0].content

    def test_only_task_no_plan(self, restorer):
        ctx = RestorationContext(task_description="Just the task.")
        msgs = restorer._build_restoration_messages(ctx)
        assert len(msgs) == 1
        assert "Task:" in msgs[0].content
        assert "Plan:" not in msgs[0].content

    def test_whitespace_only_fields_skipped(self, restorer):
        ctx = RestorationContext(
            agent_instructions="   ",
            task_description="\n\t",
            recent_output="  ",
        )
        msgs = restorer._build_restoration_messages(ctx)
        assert msgs == []

    def test_memories_joined_with_separator(self, restorer):
        ctx = RestorationContext(memories=["mem1", "mem2", "mem3"])
        msgs = restorer._build_restoration_messages(ctx)
        assert len(msgs) == 1
        assert "---" in msgs[0].content
        assert "mem1" in msgs[0].content
        assert "mem3" in msgs[0].content

    def test_empty_memory_strings_skipped(self, restorer):
        ctx = RestorationContext(memories=["  ", "", "\n"])
        msgs = restorer._build_restoration_messages(ctx)
        assert msgs == []

    def test_long_agent_instructions_get_truncated(self, restorer):
        small_budget = RestorationBudget(agent_instructions=20, task_and_plan=2000, recent_output=2000, memories=1500)
        restorer_small = ContextRestorer(budget=small_budget)
        ctx = RestorationContext(agent_instructions=_make_long_text(count=200))
        msgs = restorer_small._build_restoration_messages(ctx)
        assert msgs[0].metadata["was_truncated"] is True
        assert "[...truncated...]" in msgs[0].content

    def test_system_role_assigned_to_all_messages(self, restorer):
        ctx = RestorationContext(
            agent_instructions="Role",
            task_description="Task",
            recent_output="Output",
            memories=["mem"],
        )
        msgs = restorer._build_restoration_messages(ctx)
        for msg in msgs:
            assert msg.role == "system"

    def test_restoration_prefix_in_content(self, restorer):
        ctx = RestorationContext(agent_instructions="You are a Foreman.")
        msgs = restorer._build_restoration_messages(ctx)
        assert msgs[0].content.startswith("[Restored:")


# ── ContextRestorer.restore ────────────────────────────────────────────


class TestRestore:
    @pytest.fixture
    def restorer(self) -> ContextRestorer:
        return ContextRestorer()

    def test_prepends_restoration_messages(self, restorer):
        existing = [WindowConversationMessage(role="user", content="Hello")]
        ctx = RestorationContext(agent_instructions="You are Foreman.")
        updated, result = restorer.restore(existing, ctx)
        assert updated[-1].role == "user"
        assert updated[0].metadata["restoration_category"] == "agent_instructions"

    def test_original_messages_not_mutated(self, restorer):
        original = [WindowConversationMessage(role="user", content="original")]
        original_len = len(original)
        ctx = RestorationContext(agent_instructions="Role.")
        restorer.restore(original, ctx)
        assert len(original) == original_len

    def test_result_counts_messages(self, restorer):
        ctx = RestorationContext(
            agent_instructions="Role",
            task_description="Task",
            recent_output="Output",
            memories=["mem"],
        )
        _, result = restorer.restore([], ctx)
        assert result.messages_injected == 4

    def test_result_tokens_positive(self, restorer):
        ctx = RestorationContext(agent_instructions="You are a Foreman agent responsible for planning.")
        _, result = restorer.restore([], ctx)
        assert result.tokens_injected > 0

    def test_result_budget_used_has_correct_categories(self, restorer):
        ctx = RestorationContext(
            agent_instructions="Role",
            task_description="Task",
        )
        _, result = restorer.restore([], ctx)
        assert "agent_instructions" in result.budget_used
        assert "task_and_plan" in result.budget_used

    def test_truncated_list_populated_when_over_budget(self, restorer):
        small_budget = RestorationBudget(agent_instructions=10, task_and_plan=2000, recent_output=2000, memories=1500)
        restorer_small = ContextRestorer(budget=small_budget)
        ctx = RestorationContext(agent_instructions=_make_long_text(count=300))
        _, result = restorer_small.restore([], ctx)
        assert "agent_instructions" in result.truncated

    def test_empty_context_injects_nothing(self, restorer):
        existing = [WindowConversationMessage(role="user", content="Hi")]
        updated, result = restorer.restore(existing, RestorationContext())
        assert len(updated) == 1
        assert result.messages_injected == 0
        assert result.tokens_injected == 0

    def test_restore_with_session_state_in_context(self, restorer):
        state = _make_session_state()
        ctx = RestorationContext(
            agent_instructions="Role",
            task_description="Task",
            session_state=state,
        )
        updated, result = restorer.restore([], ctx)
        assert result.messages_injected == 2
        assert ctx.session_state.task_id == "task-1"


# ── Module singleton ───────────────────────────────────────────────────


class TestGetRestorer:
    def test_returns_context_restorer(self):
        restorer = get_restorer()
        assert isinstance(restorer, ContextRestorer)

    def test_returns_same_instance(self):
        r1 = get_restorer()
        r2 = get_restorer()
        assert r1 is r2

    def test_singleton_has_default_budget(self):
        restorer = get_restorer()
        assert restorer._budget == RestorationBudget()
