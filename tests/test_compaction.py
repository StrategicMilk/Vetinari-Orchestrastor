"""Tests for vetinari.context.compaction — three-tier compaction engine."""

from __future__ import annotations

import pytest

from vetinari.context.compaction import (
    CompactionResult,
    CompactionTier,
    ContextCompactor,
    get_compactor,
)
from vetinari.context.window_manager import WindowConversationMessage

# ── Helpers ────────────────────────────────────────────────────────────


def _msg(role: str, content: str, token_count: int | None = None) -> WindowConversationMessage:
    """Create a ConversationMessage with an explicit token count to keep tests deterministic."""
    m = WindowConversationMessage(role=role, content=content)
    if token_count is not None:
        object.__setattr__(m, "token_count", token_count)
    return m


def _msgs_with_tokens(count: int, tokens_each: int = 100) -> list[WindowConversationMessage]:
    """Create a list of *count* user messages each carrying *tokens_each* tokens."""
    return [_msg("user", f"message {i}", token_count=tokens_each) for i in range(count)]


# ── CompactionResult ───────────────────────────────────────────────────


class TestCompactionResult:
    def test_frozen_slots(self) -> None:
        r = CompactionResult(
            tier=CompactionTier.SUMMARY,
            messages_before=10,
            messages_after=3,
            tokens_before=1000,
            tokens_after=300,
            tokens_saved=700,
        )
        with pytest.raises(AttributeError):
            r.tokens_saved = 999  # type: ignore[misc]

    def test_repr_contains_tier_and_counts(self) -> None:
        r = CompactionResult(
            tier=CompactionTier.TRUNCATION,
            messages_before=20,
            messages_after=5,
            tokens_before=2000,
            tokens_after=500,
            tokens_saved=1500,
        )
        text = repr(r)
        assert "truncation" in text
        assert "2000" in text
        assert "500" in text

    def test_state_extracted_defaults_none(self) -> None:
        r = CompactionResult(
            tier=CompactionTier.STATE_EXTRACTION,
            messages_before=5,
            messages_after=2,
            tokens_before=500,
            tokens_after=200,
            tokens_saved=300,
        )
        assert r.state_extracted is None


# ── ContextCompactor initialisation ───────────────────────────────────


class TestContextCompactorInit:
    def test_default_parameters(self) -> None:
        c = ContextCompactor()
        assert c._preserve_recent == 4
        assert c._head_tokens == 500
        assert c._tail_tokens == 1500

    def test_custom_parameters(self) -> None:
        c = ContextCompactor(preserve_recent=2, head_tokens=100, tail_tokens=400)
        assert c._preserve_recent == 2
        assert c._head_tokens == 100
        assert c._tail_tokens == 400

    def test_invalid_preserve_recent_raises(self) -> None:
        with pytest.raises(ValueError, match="preserve_recent"):
            ContextCompactor(preserve_recent=0)

    def test_invalid_head_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="head_tokens"):
            ContextCompactor(head_tokens=0)

    def test_invalid_tail_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="tail_tokens"):
            ContextCompactor(tail_tokens=0)


# ── compact() — already within budget ─────────────────────────────────


class TestCompactAlreadyWithinBudget:
    def test_returns_same_messages_when_under_target(self) -> None:
        compactor = ContextCompactor()
        messages = _msgs_with_tokens(3, tokens_each=50)  # 150 tokens total
        result_msgs, result = compactor.compact(messages, target_tokens=200)
        assert len(result_msgs) == 3
        assert result.tokens_saved == 0
        assert result.messages_before == 3
        assert result.messages_after == 3

    def test_result_tier_is_state_extraction_when_no_compaction_needed(self) -> None:
        compactor = ContextCompactor()
        messages = _msgs_with_tokens(2, tokens_each=10)
        _, result = compactor.compact(messages, target_tokens=1000)
        assert result.tier == CompactionTier.STATE_EXTRACTION

    def test_original_list_not_mutated(self) -> None:
        compactor = ContextCompactor()
        messages = _msgs_with_tokens(3, tokens_each=50)
        original_ids = [id(m) for m in messages]
        compactor.compact(messages, target_tokens=500)
        assert [id(m) for m in messages] == original_ids


# ── Tier 1: state extraction ───────────────────────────────────────────


class TestTier1StateExtraction:
    def test_tier1_reduces_token_count(self) -> None:
        # 10 messages × 100 tokens = 1000 tokens; target 400 forces compaction.
        compactor = ContextCompactor(preserve_recent=3)
        messages = _msgs_with_tokens(10, tokens_each=100)
        result_msgs, result = compactor.compact(messages, target_tokens=400, task_id="t1", stage="test")
        # Should produce fewer tokens than original
        assert result.tokens_after < result.tokens_before

    def test_tier1_result_tier_is_state_extraction(self) -> None:
        compactor = ContextCompactor(preserve_recent=3)
        messages = _msgs_with_tokens(10, tokens_each=100)
        # Tier 1 summary message is small, so 400 tokens should be achievable
        _, result = compactor.compact(messages, target_tokens=400, task_id="t1", stage="plan")
        assert result.tier == CompactionTier.STATE_EXTRACTION

    def test_tier1_preserves_recent_messages(self) -> None:
        compactor = ContextCompactor(preserve_recent=3)
        messages = [_msg("user", f"old message {i}", token_count=100) for i in range(7)] + [
            _msg("user", "recent A", token_count=100),
            _msg("user", "recent B", token_count=100),
            _msg("user", "recent C", token_count=100),
        ]
        result_msgs, _ = compactor.compact(messages, target_tokens=400)
        # The last 3 recent messages must survive verbatim
        contents = [m.content for m in result_msgs]
        assert "recent A" in contents
        assert "recent B" in contents
        assert "recent C" in contents

    def test_tier1_inserts_state_summary_message(self) -> None:
        compactor = ContextCompactor(preserve_recent=2)
        messages = _msgs_with_tokens(8, tokens_each=100)
        result_msgs, _ = compactor.compact(messages, target_tokens=400)
        compressed = [m for m in result_msgs if m.is_compressed]
        assert len(compressed) >= 1

    def test_tier1_result_state_extracted_not_none(self) -> None:
        compactor = ContextCompactor(preserve_recent=3)
        messages = _msgs_with_tokens(10, tokens_each=100)
        _, result = compactor.compact(messages, target_tokens=400, task_id="t-42", stage="exec")
        if result.tier == CompactionTier.STATE_EXTRACTION:
            assert result.state_extracted is not None

    def test_tier1_not_enough_messages_returns_unchanged(self) -> None:
        # With preserve_recent=4 and only 3 messages, Tier 1 cannot split.
        compactor = ContextCompactor(preserve_recent=4)
        messages = _msgs_with_tokens(3, tokens_each=10)
        result_msgs, _ = compactor.compact(messages, target_tokens=1000)
        assert len(result_msgs) == 3


# ── Tier 2: summarization ──────────────────────────────────────────────


class TestTier2Summarize:
    def _make_compactor_that_needs_tier2(self) -> ContextCompactor:
        # head+tail large enough that Tier 3 won't reduce much, but Tier 1
        # must fail to force Tier 2. We do this by making the state summary
        # itself large. Easier: call _tier2_summarize directly.
        return ContextCompactor(preserve_recent=2, head_tokens=50000, tail_tokens=50000)

    def test_tier2_direct_summarize_reduces_count(self) -> None:
        compactor = ContextCompactor(preserve_recent=2)
        messages = _msgs_with_tokens(8, tokens_each=100)
        result = compactor._tier2_summarize(messages)
        # Summary + resume + 2 recent = 4 messages (fewer than 8)
        assert len(result) < len(messages)

    def test_tier2_preserves_recent_messages_verbatim(self) -> None:
        compactor = ContextCompactor(preserve_recent=2)
        recent_a = _msg("user", "very recent A", token_count=50)
        recent_b = _msg("assistant", "very recent B", token_count=50)
        older = _msgs_with_tokens(6, tokens_each=100)
        messages = older + [recent_a, recent_b]
        result = compactor._tier2_summarize(messages)
        result_contents = [m.content for m in result]
        assert "very recent A" in result_contents
        assert "very recent B" in result_contents

    def test_tier2_inserts_summary_system_message(self) -> None:
        compactor = ContextCompactor(preserve_recent=2)
        messages = _msgs_with_tokens(6, tokens_each=100)
        result = compactor._tier2_summarize(messages)
        compressed = [m for m in result if m.is_compressed]
        assert len(compressed) >= 1
        assert any("[Conversation history summary]" in m.content for m in compressed)

    def test_tier2_inserts_resume_instruction(self) -> None:
        compactor = ContextCompactor(preserve_recent=2)
        messages = _msgs_with_tokens(6, tokens_each=100)
        result = compactor._tier2_summarize(messages)
        assert any("resuming a conversation" in m.content for m in result)

    def test_tier2_with_few_messages_returns_unchanged(self) -> None:
        compactor = ContextCompactor(preserve_recent=4)
        messages = _msgs_with_tokens(3, tokens_each=50)
        result = compactor._tier2_summarize(messages)
        assert len(result) == 3


# ── Tier 3: truncation ─────────────────────────────────────────────────


class TestTier3Truncate:
    def test_tier3_inserts_gap_marker(self) -> None:
        compactor = ContextCompactor(head_tokens=200, tail_tokens=200)
        messages = _msgs_with_tokens(20, tokens_each=50)  # 1000 tokens total
        result = compactor._tier3_truncate(messages, target_tokens=500)
        assert any("[Context truncated" in m.content for m in result)

    def test_tier3_respects_head_window(self) -> None:
        compactor = ContextCompactor(head_tokens=150, tail_tokens=150)
        # Each message is 50 tokens — head window fits 3 messages
        messages = [_msg("user", f"msg-{i}", token_count=50) for i in range(10)]
        result = compactor._tier3_truncate(messages, target_tokens=400)
        head_msgs = [m for m in result if not m.is_compressed]
        # The first message content must appear in the result
        assert any("msg-0" in m.content for m in head_msgs)

    def test_tier3_respects_tail_window(self) -> None:
        compactor = ContextCompactor(head_tokens=100, tail_tokens=150)
        messages = [_msg("user", f"msg-{i}", token_count=50) for i in range(10)]
        result = compactor._tier3_truncate(messages, target_tokens=400)
        non_compressed = [m for m in result if not m.is_compressed]
        # The last message must survive
        assert any("msg-9" in m.content for m in non_compressed)

    def test_tier3_no_duplicate_messages(self) -> None:
        # With very large windows, head and tail would overlap — dedup must fire.
        compactor = ContextCompactor(head_tokens=100000, tail_tokens=100000)
        messages = _msgs_with_tokens(5, tokens_each=50)
        result = compactor._tier3_truncate(messages, target_tokens=1000)
        contents = [m.content for m in result if not m.is_compressed]
        # No content should appear twice
        assert len(contents) == len(set(contents))

    def test_tier3_result_tier_in_compact(self) -> None:
        # Force all three tiers to fail by using a tiny target that even Tier 3 can't fully meet
        compactor = ContextCompactor(preserve_recent=1, head_tokens=10, tail_tokens=10)
        messages = _msgs_with_tokens(20, tokens_each=100)
        # Tier 3 is always applied as last resort regardless of whether it hits target
        _, result = compactor.compact(messages, target_tokens=1)
        assert result.tier == CompactionTier.TRUNCATION


# ── _simple_summarize ──────────────────────────────────────────────────


class TestSimpleSummarize:
    def test_empty_list_returns_placeholder(self) -> None:
        compactor = ContextCompactor()
        assert "no messages" in compactor._simple_summarize([])

    def test_includes_role_and_truncated_content(self) -> None:
        compactor = ContextCompactor()
        messages = [_msg("user", "hello world")]
        summary = compactor._simple_summarize(messages)
        assert "[user]" in summary
        assert "hello world" in summary

    def test_long_content_truncated(self) -> None:
        compactor = ContextCompactor()
        long_content = "x" * 500
        messages = [_msg("assistant", long_content)]
        summary = compactor._simple_summarize(messages)
        assert "..." in summary
        # Summary should be much shorter than original
        assert len(summary) < len(long_content)

    def test_compressed_messages_labelled(self) -> None:
        compactor = ContextCompactor()
        msg = WindowConversationMessage(role="system", content="already compressed", is_compressed=True)
        summary = compactor._simple_summarize([msg])
        assert "[summary]" in summary


# ── _build_resume_instruction ──────────────────────────────────────────


class TestBuildResumeInstruction:
    def test_contains_resume_header(self) -> None:
        compactor = ContextCompactor()
        instruction = compactor._build_resume_instruction(state=None, summary="some context")
        assert "resuming a conversation" in instruction

    def test_summary_included_in_output(self) -> None:
        compactor = ContextCompactor()
        instruction = compactor._build_resume_instruction(state=None, summary="test summary text")
        assert "test summary text" in instruction

    def test_state_decisions_included_when_present(self) -> None:
        from vetinari.context.session_state import SessionState

        state = SessionState(
            task_id="t1",
            stage="planning",
            key_decisions=["chose approach A", "rejected approach B"],
            outputs_produced=[],
            quality_scores={},
            model_used="local",
        )
        compactor = ContextCompactor()
        instruction = compactor._build_resume_instruction(state=state)
        assert "chose approach A" in instruction

    def test_state_outputs_included_when_present(self) -> None:
        from vetinari.context.session_state import SessionState

        state = SessionState(
            task_id="t2",
            stage="execution",
            key_decisions=[],
            outputs_produced=["vetinari/context/compaction.py"],
            quality_scores={"score": 0.9},
            model_used="local",
        )
        compactor = ContextCompactor()
        instruction = compactor._build_resume_instruction(state=state)
        assert "vetinari/context/compaction.py" in instruction
        assert "score" in instruction

    def test_fallback_when_no_summary(self) -> None:
        compactor = ContextCompactor()
        instruction = compactor._build_resume_instruction(state=None, summary="")
        assert "context was compacted" in instruction


# ── get_compactor singleton ────────────────────────────────────────────


class TestGetCompactor:
    def test_returns_context_compactor_instance(self) -> None:
        compactor = get_compactor()
        assert isinstance(compactor, ContextCompactor)

    def test_same_instance_returned_each_call(self) -> None:
        a = get_compactor()
        b = get_compactor()
        assert a is b

    def test_singleton_has_default_parameters(self) -> None:
        compactor = get_compactor()
        assert compactor._preserve_recent == 4
        assert compactor._head_tokens == 500
        assert compactor._tail_tokens == 1500


# ── Integration: tier selection order ─────────────────────────────────


class TestTierSelectionOrder:
    def test_tier1_preferred_over_tier2_and_tier3(self) -> None:
        """When Tier 1 is sufficient, tier=STATE_EXTRACTION is reported."""
        compactor = ContextCompactor(preserve_recent=2)
        # Build a large enough message list that compaction is needed,
        # but the state summary will easily fit under target.
        messages = _msgs_with_tokens(10, tokens_each=100)  # 1000 tokens
        _, result = compactor.compact(messages, target_tokens=400)
        # Tier 1 should succeed (state summary << 400 tokens)
        assert result.tier == CompactionTier.STATE_EXTRACTION

    def test_tier2_used_when_tier1_insufficient(self) -> None:
        """Tier 2 is selected when Tier 1 cannot reach the target."""
        # Monkeypatch both _tier1_state_extraction (returns too-heavy output) and
        # _tier2_summarize (returns a light result) so the tier selection is predictable.
        compactor = ContextCompactor(preserve_recent=2, head_tokens=50000, tail_tokens=50000)
        messages = _msgs_with_tokens(10, tokens_each=100)

        original_tier1 = compactor._tier1_state_extraction
        original_tier2 = compactor._tier2_summarize

        def heavy_tier1(msgs, task_id, stage, model_id):  # type: ignore[override]
            # Exceeds the 300-token target so compact() must move on to Tier 2
            heavy = [_msg("system", "x" * 10, token_count=400)]
            return heavy, None

        def light_tier2(msgs):  # type: ignore[override]
            # Under the 300-token target so Tier 2 succeeds
            return [_msg("system", "summary", token_count=50)]

        compactor._tier1_state_extraction = heavy_tier1  # type: ignore[method-assign]
        compactor._tier2_summarize = light_tier2  # type: ignore[method-assign]
        try:
            _, result = compactor.compact(messages, target_tokens=300)
        finally:
            compactor._tier1_state_extraction = original_tier1  # type: ignore[method-assign]
            compactor._tier2_summarize = original_tier2  # type: ignore[method-assign]

        assert result.tier == CompactionTier.SUMMARY

    def test_tokens_saved_always_non_negative(self) -> None:
        compactor = ContextCompactor()
        messages = _msgs_with_tokens(5, tokens_each=200)
        _, result = compactor.compact(messages, target_tokens=50)
        assert result.tokens_saved >= 0

    def test_compact_result_tokens_after_matches_returned_messages(self) -> None:
        compactor = ContextCompactor(preserve_recent=2)
        messages = _msgs_with_tokens(10, tokens_each=100)
        result_msgs, result = compactor.compact(messages, target_tokens=400)
        actual_tokens = sum(m.token_count for m in result_msgs)
        assert result.tokens_after == actual_tokens
