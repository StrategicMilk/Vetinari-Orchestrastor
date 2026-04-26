"""Tests for vetinari.context.budget — threshold transitions and budget API.

Covers all BudgetStatus transitions, record/reset mechanics, factory function,
BudgetThresholds validation, and to_dict serialisation.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from vetinari.context.budget import (
    BudgetCheck,
    BudgetStatus,
    BudgetThresholds,
    ContextBudget,
    StageUsage,
    create_budget_for_model,
)

# ── Helpers ────────────────────────────────────────────────────────────


def _budget(context_length: int = 10_000, **threshold_kwargs: float) -> ContextBudget:
    """Create a ContextBudget with a convenient 10k-token window by default."""
    thresholds = BudgetThresholds(**threshold_kwargs) if threshold_kwargs else None
    return ContextBudget(context_length=context_length, thresholds=thresholds)


# ── BudgetThresholds ───────────────────────────────────────────────────


class TestBudgetThresholds:
    def test_defaults_are_valid(self) -> None:
        t = BudgetThresholds()
        assert t.warn_ratio == 0.70
        assert t.compact_ratio == 0.85
        assert t.hard_stop_ratio == 0.95

    def test_custom_values_accepted(self) -> None:
        t = BudgetThresholds(warn_ratio=0.5, compact_ratio=0.7, hard_stop_ratio=0.9)
        assert t.warn_ratio == 0.5

    def test_wrong_order_raises(self) -> None:
        with pytest.raises(ValueError, match="must satisfy"):
            BudgetThresholds(warn_ratio=0.9, compact_ratio=0.5, hard_stop_ratio=0.95)

    def test_equal_ratios_raise(self) -> None:
        with pytest.raises(ValueError, match="must satisfy"):
            BudgetThresholds(warn_ratio=0.7, compact_ratio=0.7, hard_stop_ratio=0.95)

    def test_hard_stop_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="must satisfy"):
            BudgetThresholds(warn_ratio=0.7, compact_ratio=0.85, hard_stop_ratio=1.1)

    def test_frozen(self) -> None:
        t = BudgetThresholds()
        with pytest.raises((AttributeError, TypeError)):
            t.warn_ratio = 0.5  # type: ignore[misc]


# ── ContextBudget construction ─────────────────────────────────────────


class TestContextBudgetConstruction:
    def test_zero_context_length_raises(self) -> None:
        with pytest.raises(ValueError, match="context_length must be >= 1"):
            ContextBudget(context_length=0)

    def test_negative_context_length_raises(self) -> None:
        with pytest.raises(ValueError, match="context_length must be >= 1"):
            ContextBudget(context_length=-1)

    def test_default_thresholds_applied(self) -> None:
        b = ContextBudget(context_length=1000)
        check = b.check()
        assert check.status == BudgetStatus.OK

    def test_custom_thresholds_stored(self) -> None:
        custom = BudgetThresholds(warn_ratio=0.5, compact_ratio=0.6, hard_stop_ratio=0.7)
        b = ContextBudget(context_length=1000, thresholds=custom)
        # Record 55% usage — should trigger WARNING (above 0.5) but not COMPACTION (0.6)
        b.record_usage("stage", 550)
        assert b.check().status == BudgetStatus.WARNING


# ── record_usage ───────────────────────────────────────────────────────


class TestRecordUsage:
    def test_single_stage_accumulates(self) -> None:
        b = _budget()
        b.record_usage("foreman", 100)
        b.record_usage("foreman", 200)
        assert b.check().total_tokens == 300

    def test_multiple_stages_sum(self) -> None:
        b = _budget()
        b.record_usage("foreman", 1000)
        b.record_usage("worker", 2000)
        b.record_usage("inspector", 500)
        assert b.check().total_tokens == 3500

    def test_empty_stage_name_raises(self) -> None:
        b = _budget()
        with pytest.raises(ValueError, match="stage name must be non-empty"):
            b.record_usage("", 100)

    def test_negative_tokens_raises(self) -> None:
        b = _budget()
        with pytest.raises(ValueError, match="tokens must be >= 0"):
            b.record_usage("foreman", -1)

    def test_zero_tokens_accepted(self) -> None:
        b = _budget()
        b.record_usage("foreman", 0)
        assert b.check().total_tokens == 0


# ── BudgetStatus transitions ───────────────────────────────────────────


class TestBudgetStatusTransitions:
    """Exercise every threshold boundary with a 10_000 token window.

    Default thresholds: warn=0.70, compact=0.85, hard_stop=0.95.
    In a 10k window: warn=7000, compact=8500, hard_stop=9500 tokens.
    """

    def test_ok_below_warn(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 6_999)
        assert b.check().status == BudgetStatus.OK

    def test_ok_at_zero(self) -> None:
        b = _budget(10_000)
        assert b.check().status == BudgetStatus.OK

    def test_warning_at_exact_warn_boundary(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 7_000)  # exactly 70%
        assert b.check().status == BudgetStatus.WARNING

    def test_warning_between_warn_and_compact(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 7_500)  # 75%
        assert b.check().status == BudgetStatus.WARNING

    def test_compaction_needed_at_compact_boundary(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 8_500)  # exactly 85%
        assert b.check().status == BudgetStatus.COMPACTION_NEEDED

    def test_compaction_needed_between_compact_and_hard_stop(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 9_000)  # 90%
        assert b.check().status == BudgetStatus.COMPACTION_NEEDED

    def test_exceeded_at_hard_stop_boundary(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 9_500)  # exactly 95%
        assert b.check().status == BudgetStatus.EXCEEDED

    def test_exceeded_above_hard_stop(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 10_000)  # 100%
        assert b.check().status == BudgetStatus.EXCEEDED

    def test_just_below_warn_is_ok(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 6_999)
        assert b.check().status == BudgetStatus.OK

    def test_just_below_compact_is_warning(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 8_499)
        assert b.check().status == BudgetStatus.WARNING

    def test_just_below_hard_stop_is_compaction_needed(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 9_499)
        assert b.check().status == BudgetStatus.COMPACTION_NEEDED


# ── should_compact ─────────────────────────────────────────────────────


class TestShouldCompact:
    def test_false_when_ok(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 5_000)
        assert b.should_compact() is False

    def test_false_when_warning(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 7_500)
        assert b.should_compact() is False

    def test_true_when_compaction_needed(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 8_500)
        assert b.should_compact() is True

    def test_true_when_exceeded(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 9_600)
        assert b.should_compact() is True


# ── remaining ─────────────────────────────────────────────────────────


class TestRemaining:
    def test_remaining_full_budget(self) -> None:
        b = _budget(10_000)
        # hard stop at 95% = 9500; remaining = 9500 - 0 = 9500
        assert b.remaining() == 9_500

    def test_remaining_after_partial_usage(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 4_000)
        assert b.remaining() == 5_500

    def test_remaining_negative_when_exceeded(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 10_000)
        assert b.remaining() < 0


# ── reset ──────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_all(self) -> None:
        b = _budget(10_000)
        b.record_usage("foreman", 5_000)
        b.record_usage("worker", 3_000)
        b.reset()
        check = b.check()
        assert check.total_tokens == 0
        assert check.status == BudgetStatus.OK
        assert check.stage_breakdown == []

    def test_reset_stage_removes_only_target(self) -> None:
        b = _budget(10_000)
        b.record_usage("foreman", 5_000)
        b.record_usage("worker", 2_000)
        b.reset_stage("foreman")
        check = b.check()
        assert check.total_tokens == 2_000
        # worker still present
        assert any(s.stage == "worker" for s in check.stage_breakdown)

    def test_reset_stage_unknown_is_noop(self) -> None:
        b = _budget(10_000)
        b.record_usage("foreman", 1_000)
        b.reset_stage("nonexistent")
        assert b.check().total_tokens == 1_000

    def test_total_does_not_go_negative_after_double_reset(self) -> None:
        b = _budget(10_000)
        b.record_usage("foreman", 500)
        b.reset_stage("foreman")
        b.reset_stage("foreman")  # second call — stage already gone
        assert b.check().total_tokens == 0


# ── BudgetCheck structure ──────────────────────────────────────────────


class TestBudgetCheck:
    def test_check_returns_budget_check(self) -> None:
        b = _budget(10_000)
        result = b.check()
        assert isinstance(result, BudgetCheck)

    def test_check_usage_ratio_accurate(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 5_000)
        check = b.check()
        assert check.usage_ratio == pytest.approx(0.5, abs=1e-4)

    def test_check_message_non_empty(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 1_000)
        assert len(b.check().message) > 10

    def test_stage_breakdown_sorted_descending(self) -> None:
        b = _budget(10_000)
        b.record_usage("inspector", 100)
        b.record_usage("worker", 3_000)
        b.record_usage("foreman", 1_500)
        breakdown = b.check().stage_breakdown
        tokens = [s.tokens_used for s in breakdown]
        assert tokens == sorted(tokens, reverse=True)

    def test_stage_breakdown_contains_all_stages(self) -> None:
        b = _budget(10_000)
        b.record_usage("foreman", 500)
        b.record_usage("worker", 200)
        stages = {s.stage for s in b.check().stage_breakdown}
        assert stages == {"foreman", "worker"}

    def test_stage_usage_ratio_sums_correctly(self) -> None:
        b = _budget(10_000)
        b.record_usage("foreman", 2_000)
        b.record_usage("worker", 3_000)
        total_stage_ratio = sum(s.ratio for s in b.check().stage_breakdown)
        # 5000/10000 = 0.5 total across stages
        assert total_stage_ratio == pytest.approx(0.5, abs=1e-4)

    def test_context_limit_matches_constructor(self) -> None:
        b = ContextBudget(context_length=32_768)
        assert b.check().context_limit == 32_768


# ── to_dict ────────────────────────────────────────────────────────────


class TestToDict:
    def test_required_keys_present(self) -> None:
        b = _budget(10_000)
        d = b.to_dict()
        for key in (
            "context_length",
            "total_tokens",
            "usage_ratio",
            "remaining_tokens",
            "status",
            "thresholds",
            "stages",
        ):
            assert key in d, f"missing key: {key}"

    def test_status_is_string_value(self) -> None:
        b = _budget(10_000)
        b.record_usage("stage", 8_500)
        d = b.to_dict()
        assert d["status"] == BudgetStatus.COMPACTION_NEEDED.value

    def test_stages_reflect_recorded_usage(self) -> None:
        b = _budget(10_000)
        b.record_usage("foreman", 1_234)
        d = b.to_dict()
        assert "foreman" in d["stages"]
        assert d["stages"]["foreman"]["tokens"] == 1_234

    def test_thresholds_block_present(self) -> None:
        b = _budget(10_000)
        t = b.to_dict()["thresholds"]
        assert t["warn_ratio"] == 0.70
        assert t["compact_ratio"] == 0.85
        assert t["hard_stop_ratio"] == 0.95


# ── create_budget_for_model ────────────────────────────────────────────


class TestCreateBudgetForModel:
    def test_known_model_gets_correct_window(self) -> None:
        b = create_budget_for_model("qwen2.5-coder-7b")
        assert b.check().context_limit == 32_768

    def test_cloud_model_gets_correct_window(self) -> None:
        b = create_budget_for_model("claude-sonnet-4")
        assert b.check().context_limit == 200_000

    def test_unknown_model_falls_back_to_default(self) -> None:
        b = create_budget_for_model("totally-unknown-model-xyz")
        assert b.check().context_limit == 32_768

    def test_custom_thresholds_forwarded(self) -> None:
        custom = BudgetThresholds(warn_ratio=0.5, compact_ratio=0.6, hard_stop_ratio=0.7)
        b = create_budget_for_model("qwen2.5-coder-7b", thresholds=custom)
        # At 55% of 32768 = ~18022 tokens, status should be WARNING (above 0.5)
        b.record_usage("stage", 18_022)
        assert b.check().status == BudgetStatus.WARNING

    def test_returns_context_budget_instance(self) -> None:
        b = create_budget_for_model("default")
        assert isinstance(b, ContextBudget)

    def test_prefers_effective_window_over_static(self) -> None:
        """When get_effective_window returns a measurement, use it instead of the static table."""
        with patch(
            "vetinari.testing.context_window.get_effective_window",
            return_value=24_000,
        ):
            b = create_budget_for_model("qwen2.5-coder-7b")
        # Should use measured 24000, not the static 32768
        assert b.check().context_limit == 24_000

    def test_falls_back_to_static_when_no_measurement(self) -> None:
        """When get_effective_window returns None, fall back to the static lookup."""
        with patch(
            "vetinari.testing.context_window.get_effective_window",
            return_value=None,
        ):
            b = create_budget_for_model("qwen2.5-coder-7b")
        assert b.check().context_limit == 32_768


# ── Message content spot-checks ────────────────────────────────────────


class TestMessageContent:
    def test_ok_message_contains_ok(self) -> None:
        b = _budget(10_000)
        assert "OK" in b.check().message

    def test_warning_message_contains_warning(self) -> None:
        b = _budget(10_000)
        b.record_usage("s", 7_000)
        assert "WARNING" in b.check().message

    def test_compaction_message_contains_compaction(self) -> None:
        b = _budget(10_000)
        b.record_usage("s", 8_500)
        assert "COMPACTION" in b.check().message

    def test_exceeded_message_contains_exceeded(self) -> None:
        b = _budget(10_000)
        b.record_usage("s", 9_500)
        assert "EXCEEDED" in b.check().message
