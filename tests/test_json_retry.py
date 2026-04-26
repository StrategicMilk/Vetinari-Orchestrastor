"""Tests for JSON validation-retry with cost tracking in InferenceMixin."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vetinari.agents import inference as inference_module


@pytest.fixture(autouse=True)
def _clear_retry_stats():
    """Clear retry tracking between tests to prevent cross-test pollution."""
    inference_module._json_retry_counts.clear()
    yield
    inference_module._json_retry_counts.clear()


class TestJsonRetry:
    """Tests for _infer_json retry behavior."""

    @pytest.fixture
    def mixin(self):
        """Create a minimal InferenceMixin instance for testing.

        Binds the real _infer_json and _record_json_retries implementations
        onto a MagicMock so we can control _infer() return values while
        exercising the real retry logic.
        """
        from vetinari.types import AgentType

        m = MagicMock(spec=inference_module.InferenceMixin)
        m.agent_type = AgentType.WORKER
        m._log = MagicMock()
        m._last_model_id = "test-model-7b"
        # Bind real implementations so retry logic executes against them
        m._infer_json = inference_module.InferenceMixin._infer_json.__get__(m)
        m._record_json_retries = inference_module.InferenceMixin._record_json_retries
        return m

    def test_valid_json_first_attempt_no_retry(self, mixin):
        """Valid JSON on first attempt returns parsed result without retrying."""
        mixin._infer = MagicMock(return_value='{"result": "success"}')
        result = mixin._infer_json("test prompt")
        assert result == {"result": "success"}
        assert mixin._infer.call_count == 1

    def test_malformed_json_triggers_retry(self, mixin):
        """Malformed JSON triggers retry with error feedback on second attempt."""
        mixin._infer = MagicMock(
            side_effect=[
                "not json at all",
                '{"result": "fixed"}',
            ]
        )
        result = mixin._infer_json("test prompt")
        assert result == {"result": "fixed"}
        assert mixin._infer.call_count == 2

    def test_retry_prompt_includes_error_feedback(self, mixin):
        """Retry prompt includes the parse error message for model self-correction."""
        mixin._infer = MagicMock(
            side_effect=[
                "{invalid json}",
                '{"fixed": true}',
            ]
        )
        mixin._infer_json("original prompt")
        # Second call prompt must contain error context
        second_call_prompt = mixin._infer.call_args_list[1][0][0]
        assert "not valid JSON" in second_call_prompt
        assert "MUST respond with ONLY valid JSON" in second_call_prompt

    def test_max_retries_then_fallback(self, mixin):
        """After _MAX_JSON_RETRIES attempts all fail, returns the fallback value."""
        mixin._infer = MagicMock(return_value="never valid json!!!")
        result = mixin._infer_json("test", fallback={"default": True})
        assert result == {"default": True}
        assert mixin._infer.call_count == inference_module._MAX_JSON_RETRIES

    def test_retry_tracking_records_counts(self, mixin):
        """Retry counts are recorded per model when at least one retry was needed."""
        mixin._infer = MagicMock(
            side_effect=[
                "bad json",
                '{"ok": true}',
            ]
        )
        mixin._infer_json("test")
        stats = inference_module.get_json_retry_stats()
        assert "test-model-7b" in stats
        assert stats["test-model-7b"]["retry_events"] == 1
        assert stats["test-model-7b"]["total_retries"] == 1

    def test_no_retry_tracking_on_first_attempt_success(self, mixin):
        """No retry tracking entry is created when first attempt succeeds."""
        mixin._infer = MagicMock(return_value='{"ok": true}')
        mixin._infer_json("test")
        stats = inference_module.get_json_retry_stats()
        # Nothing recorded — no retries were needed
        assert stats == {}

    def test_regex_extraction_still_works_without_retry(self, mixin):
        """JSON embedded in surrounding text is extracted via regex without retry."""
        mixin._infer = MagicMock(return_value='Here is the result: {"data": 42} hope that helps')
        result = mixin._infer_json("test")
        assert result == {"data": 42}
        assert mixin._infer.call_count == 1

    def test_inference_error_returns_fallback_immediately(self, mixin):
        """InferenceError skips all retries and returns fallback immediately."""
        from vetinari.exceptions import InferenceError

        mixin._infer = MagicMock(side_effect=InferenceError("model down"))
        result = mixin._infer_json("test", fallback={"error": True})
        assert result == {"error": True}
        assert mixin._infer.call_count == 1

    def test_model_unavailable_error_returns_fallback_immediately(self, mixin):
        """ModelUnavailableError skips all retries and returns fallback immediately."""
        from vetinari.exceptions import ModelUnavailableError

        mixin._infer = MagicMock(side_effect=ModelUnavailableError("no model"))
        result = mixin._infer_json("test", fallback={"unavailable": True})
        assert result == {"unavailable": True}
        assert mixin._infer.call_count == 1

    def test_empty_response_triggers_retry(self, mixin):
        """Empty string response triggers a retry with an empty-response feedback prompt."""
        mixin._infer = MagicMock(
            side_effect=[
                "",
                '{"recovered": true}',
            ]
        )
        result = mixin._infer_json("test")
        assert result == {"recovered": True}
        assert mixin._infer.call_count == 2
        second_prompt = mixin._infer.call_args_list[1][0][0]
        assert "previous response was empty" in second_prompt

    def test_empty_response_all_attempts_returns_fallback(self, mixin):
        """Empty response on every attempt ultimately returns fallback."""
        mixin._infer = MagicMock(return_value="")
        result = mixin._infer_json("test", fallback={"empty": True})
        assert result == {"empty": True}
        # Should stop after _MAX_JSON_RETRIES attempts
        assert mixin._infer.call_count == inference_module._MAX_JSON_RETRIES

    def test_fallback_none_returned_when_no_fallback_given(self, mixin):
        """When no fallback is provided and all retries fail, None is returned."""
        mixin._infer = MagicMock(return_value="not json")
        result = mixin._infer_json("test")
        assert result is None

    def test_expect_json_kwarg_not_duplicated(self, mixin):
        """Passing expect_json in kwargs does not cause a duplicate keyword error."""
        mixin._infer = MagicMock(return_value='{"ok": true}')
        # Should not raise TypeError: got multiple values for keyword argument
        result = mixin._infer_json("test", expect_json=True)
        assert result == {"ok": True}

    @pytest.mark.parametrize(
        "bad,good",
        [
            ("plain text no braces", '{"x": 1}'),
            ("{broken: json", '{"x": 2}'),
            ("", '{"x": 3}'),
        ],
    )
    def test_various_bad_outputs_recover_on_retry(self, mixin, bad, good):
        """Various malformed first outputs all recover on the second attempt."""
        mixin._infer = MagicMock(side_effect=[bad, good])
        result = mixin._infer_json("test")
        assert isinstance(result, dict)
        assert "x" in result


class TestJsonRetryStats:
    """Tests for get_json_retry_stats() module-level utility."""

    def test_empty_stats_when_no_retries_recorded(self):
        """Returns empty dict when no retry events have been tracked."""
        stats = inference_module.get_json_retry_stats()
        assert stats == {}

    def test_stats_structure_for_recorded_model(self):
        """Stats dict has the expected keys and types for a recorded model."""
        inference_module._json_retry_counts["my-model"] = [1, 2, 1]
        stats = inference_module.get_json_retry_stats()
        assert "my-model" in stats
        entry = stats["my-model"]
        assert entry["total_retries"] == 4
        assert entry["retry_events"] == 3
        assert isinstance(entry["average_retries"], float)
        assert isinstance(entry["flagged_poor_structured_output"], bool)

    def test_flagged_poor_structured_output_when_high_average(self):
        """Models with consistently high retries across 5+ events are flagged."""
        # _MAX_JSON_RETRIES is 3, so threshold is >= 2 avg with 5+ events
        inference_module._json_retry_counts["bad-model"] = [3, 3, 3, 3, 3]
        stats = inference_module.get_json_retry_stats()
        assert stats["bad-model"]["flagged_poor_structured_output"] is True

    def test_not_flagged_below_threshold(self):
        """Models with low retry averages are not flagged."""
        inference_module._json_retry_counts["good-model"] = [1, 1, 1, 1, 1]
        stats = inference_module.get_json_retry_stats()
        assert stats["good-model"]["flagged_poor_structured_output"] is False

    def test_not_flagged_with_fewer_than_5_events(self):
        """Models are not flagged until at least 5 retry events are recorded."""
        # High average but only 4 events — not enough data
        inference_module._json_retry_counts["new-model"] = [3, 3, 3, 3]
        stats = inference_module.get_json_retry_stats()
        assert stats["new-model"]["flagged_poor_structured_output"] is False

    def test_multiple_models_tracked_independently(self):
        """Stats for multiple models are tracked and returned independently."""
        inference_module._json_retry_counts["model-a"] = [1, 1]
        inference_module._json_retry_counts["model-b"] = [2, 2, 2]
        stats = inference_module.get_json_retry_stats()
        assert "model-a" in stats
        assert "model-b" in stats
        assert stats["model-a"]["retry_events"] == 2
        assert stats["model-b"]["retry_events"] == 3

    def test_average_retries_rounded_to_two_decimal_places(self):
        """Average retries are rounded to 2 decimal places in the output."""
        inference_module._json_retry_counts["precise-model"] = [1, 2, 1]
        stats = inference_module.get_json_retry_stats()
        # 4/3 = 1.3333... should be 1.33
        assert stats["precise-model"]["average_retries"] == round(4 / 3, 2)
