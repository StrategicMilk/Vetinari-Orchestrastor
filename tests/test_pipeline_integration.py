"""Tests for vetinari.context.pipeline_integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vetinari.context.budget import BudgetStatus
from vetinari.context.pipeline_integration import (
    ContextBudgetExceeded,
    PipelineContextManager,
    create_pipeline_context_manager,
)
from vetinari.context.restoration import RestorationContext
from vetinari.context.window_manager import WindowConversationMessage

# ── Helpers ───────────────────────────────────────────────────────────


def _make_messages(total_tokens: int, count: int = 3) -> list[WindowConversationMessage]:
    """Build a list of ConversationMessage objects whose token counts sum to total_tokens."""
    per_msg = total_tokens // count
    remainder = total_tokens - per_msg * (count - 1)
    msgs = [WindowConversationMessage(role="user", content="x" * (per_msg * 4)) for _ in range(count - 1)]
    msgs.append(WindowConversationMessage(role="assistant", content="x" * (remainder * 4)))
    # Override token_count via the dataclass field by rebuilding with explicit token hints.
    # ConversationMessage estimates tokens from content length, so we accept the estimate.
    return msgs


def _make_exact_messages(token_counts: list[int]) -> list[WindowConversationMessage]:
    """Build messages whose individual token_counts match the given list.

    Uses content length as a proxy: estimate_tokens approximates ~1.3 tokens/word,
    so we set content to a word-repeated string long enough to hit the target.
    """
    msgs = []
    for tc in token_counts:
        # Rough inverse: tokens ~ len(content.split()) * 1.3, so words ~ tc / 1.3
        word_count = max(1, int(tc / 1.3))
        content = " ".join(["word"] * word_count)
        msgs.append(WindowConversationMessage(role="user", content=content))
    return msgs


# ── TestPipelineContextManager ─────────────────────────────────────


class TestPipelineContextManager:
    """Tests for PipelineContextManager."""

    def test_creation_with_known_model(self):
        mgr = PipelineContextManager(model_id="default")
        assert mgr.budget is not None
        assert mgr._model_id == "default"

    def test_creation_with_unknown_model_falls_back_to_default(self):
        # Unknown model IDs should fall back to the default window size without error.
        mgr = PipelineContextManager(model_id="some-unknown-model-xyz")
        assert mgr.budget is not None

    def test_check_budget_ok_when_under_threshold(self):
        mgr = PipelineContextManager(model_id="default")
        # default context window is 32768; use a tiny message list well under 70%
        msgs = _make_exact_messages([100, 100, 100])
        result_msgs, check = mgr.check_budget("foreman", msgs)
        assert check.status == BudgetStatus.OK
        assert result_msgs is not None
        assert len(result_msgs) == len(msgs)

    def test_check_budget_returns_warning_at_warning_threshold(self):
        """Usage between 70% and 85% should return WARNING without compacting."""
        mgr = PipelineContextManager(model_id="default")
        context_length = mgr.budget._context_length  # 32768 for "default"
        # Force the budget into WARNING by recording usage directly (72%)
        mgr.budget.record_usage("prior_stage", int(context_length * 0.72))
        msgs = _make_exact_messages([50])
        result_msgs, check = mgr.check_budget("current_stage", msgs)
        assert check.status == BudgetStatus.WARNING
        # Messages must be unchanged — no compaction at WARNING level
        assert len(result_msgs) == 1

    def test_check_budget_raises_on_exceeded(self):
        """Usage above 95% must raise ContextBudgetExceeded."""
        mgr = PipelineContextManager(model_id="default")
        context_length = mgr.budget._context_length
        # Drive usage over 95% before calling check_budget
        mgr.budget.record_usage("prior", int(context_length * 0.96))
        msgs = _make_exact_messages([10])
        with pytest.raises(ContextBudgetExceeded, match="execution"):
            mgr.check_budget("execution", msgs)

    def test_check_budget_compacts_when_needed(self):
        """Usage between 85% and 95% must trigger compaction and return fewer tokens."""
        mgr = PipelineContextManager(model_id="default")
        context_length = mgr.budget._context_length

        # Build a realistic message list that puts total usage at ~88%
        target_tokens = int(context_length * 0.88)
        msgs = _make_exact_messages([target_tokens])

        result_msgs, check = mgr.check_budget("execution", msgs)
        # After compaction the budget check must not be EXCEEDED
        assert check.status != BudgetStatus.EXCEEDED
        # The messages list must still be valid (non-empty)
        assert len(result_msgs) >= 1

    def test_restoration_context_applied_after_compaction(self):
        """After compaction, a RestorationContext must be injected into messages."""
        mgr = PipelineContextManager(model_id="default")
        context_length = mgr.budget._context_length

        # Force COMPACTION_NEEDED
        mgr.budget.record_usage("prior", int(context_length * 0.86))
        msgs = _make_exact_messages([100])

        restoration = RestorationContext(
            agent_instructions="You are a test agent.",
            task_description="Write a unit test.",
        )
        result_msgs, _ = mgr.check_budget("execution", msgs, restoration_context=restoration)

        # Restoration injects system messages — at least one should contain the agent instructions
        combined_content = " ".join(m.content for m in result_msgs)
        assert "test agent" in combined_content

    def test_budget_property_returns_context_budget(self):
        mgr = PipelineContextManager(model_id="default")
        from vetinari.context.budget import ContextBudget

        assert isinstance(mgr.budget, ContextBudget)

    def test_check_budget_ok_no_restoration_context(self):
        """check_budget with restoration_context=None must not raise when OK."""
        mgr = PipelineContextManager(model_id="default")
        msgs = _make_exact_messages([50])
        result_msgs, check = mgr.check_budget("foreman", msgs, restoration_context=None)
        assert check.status == BudgetStatus.OK
        assert len(result_msgs) == 1


# ── TestContextBudgetExceeded ─────────────────────────────────────────


class TestContextBudgetExceeded:
    """Tests for the ContextBudgetExceeded exception."""

    def test_is_exception(self):
        exc = ContextBudgetExceeded("test message")
        assert isinstance(exc, Exception)
        assert "test message" in str(exc)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(ContextBudgetExceeded):
            raise ContextBudgetExceeded("budget exceeded at stage 'review': 96% used")


# ── TestCreatePipelineContextManager ──────────────────────────────────


class TestCreatePipelineContextManager:
    """Tests for the create_pipeline_context_manager factory."""

    def test_returns_pipeline_context_manager(self):
        mgr = create_pipeline_context_manager("default")
        assert isinstance(mgr, PipelineContextManager)
        assert mgr._model_id == "default"

    def test_each_call_returns_fresh_instance(self):
        mgr1 = create_pipeline_context_manager("default")
        mgr2 = create_pipeline_context_manager("default")
        assert mgr1 is not mgr2

    def test_factory_accepts_any_model_id(self):
        mgr = create_pipeline_context_manager("qwen2.5-coder-7b")
        assert mgr._model_id == "qwen2.5-coder-7b"
        # Budget should reflect the known window for this model
        assert mgr.budget._context_length == 32768
