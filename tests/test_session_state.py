"""Tests for vetinari.context.session_state — pattern-based state extraction."""

from __future__ import annotations

import pytest

from vetinari.context.session_state import (
    SessionState,
    SessionStateExtractor,
    get_session_state_extractor,
)

# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def extractor() -> SessionStateExtractor:
    """Fresh extractor for each test (patterns compiled in __init__)."""
    return SessionStateExtractor()


# ── SessionState dataclass ─────────────────────────────────────────────


class TestSessionState:
    """Tests for the SessionState frozen dataclass."""

    def test_repr_shows_task_id_and_stage(self):
        state = SessionState(
            task_id="task-1",
            stage="planning",
            key_decisions=["decided to use qwen2.5"],
            outputs_produced=["vetinari/agents/foo.py"],
            quality_scores={"score": 0.9},
            model_used="qwen2.5-coder-7b",
            token_count=120,
            timestamp=1000.0,
        )
        r = repr(state)
        assert "task-1" in r
        assert "planning" in r
        assert "decisions=1" in r
        assert "outputs=1" in r

    def test_frozen_raises_on_mutation(self):
        state = SessionState(
            task_id="t",
            stage="review",
            key_decisions=[],
            outputs_produced=[],
            quality_scores={},
            model_used="",
        )
        with pytest.raises((AttributeError, TypeError)):
            state.stage = "execution"  # type: ignore[misc]

    def test_default_token_count_is_zero(self):
        state = SessionState(
            task_id="t",
            stage="assembly",
            key_decisions=[],
            outputs_produced=[],
            quality_scores={},
            model_used="",
        )
        assert state.token_count == 0

    def test_default_timestamp_is_zero(self):
        state = SessionState(
            task_id="t",
            stage="assembly",
            key_decisions=[],
            outputs_produced=[],
            quality_scores={},
            model_used="",
        )
        assert state.timestamp == 0.0


# ── SessionStateExtractor.extract ─────────────────────────────────────


class TestExtractEmptyText:
    """extract() handles empty / blank input gracefully."""

    def test_empty_string_returns_empty_state(self, extractor: SessionStateExtractor):
        state = extractor.extract("", task_id="t1", stage="planning")
        assert state.task_id == "t1"
        assert state.stage == "planning"
        assert state.key_decisions == []
        assert state.outputs_produced == []
        assert state.quality_scores == {}
        assert state.token_count == 0

    def test_timestamp_is_set_on_empty(self, extractor: SessionStateExtractor):
        import time

        before = time.time()
        state = extractor.extract("", task_id="t", stage="planning")
        after = time.time()
        assert before <= state.timestamp <= after

    def test_model_id_propagated_on_empty(self, extractor: SessionStateExtractor):
        state = extractor.extract("", task_id="t", stage="planning", model_id="qwen3-30b-a3b")
        assert state.model_used == "qwen3-30b-a3b"


class TestExtractFull:
    """extract() populates all fields from realistic stage output."""

    _SAMPLE = """
    After reviewing options, we decided to use the Qwen2.5 model for this task.
    We also chose the structured output format over plain text.
    The implementation approved the use of vetinari/context/session_state.py.
    Created vetinari/agents/planner.py with the new planning logic.
    See https://huggingface.co/Qwen for model details.

    Quality metrics:
      score: 0.87
      confidence: 0.92
      coverage: 85%
    """

    def test_returns_session_state(self, extractor: SessionStateExtractor):
        state = extractor.extract(self._SAMPLE, task_id="t2", stage="execution")
        assert isinstance(state, SessionState)

    def test_task_id_and_stage_preserved(self, extractor: SessionStateExtractor):
        state = extractor.extract(self._SAMPLE, task_id="t2", stage="execution")
        assert state.task_id == "t2"
        assert state.stage == "execution"

    def test_model_id_preserved(self, extractor: SessionStateExtractor):
        state = extractor.extract(self._SAMPLE, task_id="t2", stage="execution", model_id="qwen2.5-coder-7b")
        assert state.model_used == "qwen2.5-coder-7b"

    def test_token_count_positive(self, extractor: SessionStateExtractor):
        state = extractor.extract(self._SAMPLE, task_id="t2", stage="execution")
        assert state.token_count > 0

    def test_timestamp_recent(self, extractor: SessionStateExtractor):
        import time

        before = time.time()
        state = extractor.extract(self._SAMPLE, task_id="t2", stage="execution")
        after = time.time()
        assert before <= state.timestamp <= after


# ── _extract_decisions ─────────────────────────────────────────────────


class TestExtractDecisions:
    """_extract_decisions picks up decision-keyword sentences."""

    @pytest.mark.parametrize(
        "text,expected_fragment",
        [
            ("We decided to use Python 3.12.", "decided"),
            ("The team chose the smaller model.", "chose"),
            ("We selected qwen2.5 for speed.", "selected"),
            ("The plan approved the new schema.", "approved"),
            ("The committee rejected the old approach.", "rejected"),
            ("We picked the fast path.", "picked"),
            ("We are using async everywhere.", "using"),
            ("The agent will use structured output.", "will use"),
            ("The system switched to the backup model.", "switched"),
        ],
    )
    def test_keyword_detected(self, extractor: SessionStateExtractor, text: str, expected_fragment: str):
        decisions = extractor._extract_decisions(text)
        assert len(decisions) >= 1
        combined = " ".join(decisions).lower()
        assert expected_fragment in combined

    def test_no_decision_keywords_returns_empty(self, extractor: SessionStateExtractor):
        text = "The sky is blue. The grass is green."
        assert extractor._extract_decisions(text) == []

    def test_deduplication(self, extractor: SessionStateExtractor):
        text = "We decided to use qwen2.5. We decided to use qwen2.5."
        decisions = extractor._extract_decisions(text)
        assert len(decisions) == len(set(decisions))

    def test_multiple_decision_sentences(self, extractor: SessionStateExtractor):
        text = "We decided to use the fast model. We chose the JSON format."
        decisions = extractor._extract_decisions(text)
        assert len(decisions) >= 2


# ── _extract_outputs ───────────────────────────────────────────────────


class TestExtractOutputs:
    """_extract_outputs finds file paths, URLs, code blocks, and artifact verbs."""

    def test_detects_file_path(self, extractor: SessionStateExtractor):
        outputs = extractor._extract_outputs("Saved to vetinari/context/session_state.py.")
        assert any("session_state.py" in o for o in outputs)

    def test_detects_url(self, extractor: SessionStateExtractor):
        outputs = extractor._extract_outputs("See https://example.com/docs for details.")
        assert any("https://example.com/docs" in o for o in outputs)

    def test_detects_code_block(self, extractor: SessionStateExtractor):
        text = "Here is the output:\n```python\nprint('hello')\n```\n"
        outputs = extractor._extract_outputs(text)
        assert any("code-block" in o for o in outputs)

    def test_detects_artifact_verb(self, extractor: SessionStateExtractor):
        outputs = extractor._extract_outputs("Generated the planning summary for review.")
        assert len(outputs) >= 1

    def test_empty_text_returns_empty(self, extractor: SessionStateExtractor):
        assert extractor._extract_outputs("") == []

    def test_deduplication(self, extractor: SessionStateExtractor):
        text = "vetinari/foo/bar.py and vetinari/foo/bar.py again."
        outputs = extractor._extract_outputs(text)
        assert len(outputs) == len(set(outputs))


# ── _extract_quality_scores ────────────────────────────────────────────


class TestExtractQualityScores:
    """_extract_quality_scores parses numeric metrics from text."""

    @pytest.mark.parametrize(
        "text,metric,expected",
        [
            ("score: 0.85", "score", 0.85),
            ("quality: 0.7", "quality", 0.7),
            ("confidence: 0.9", "confidence", 0.9),
            ("coverage: 80%", "coverage", 0.8),
            ("accuracy: 95%", "accuracy", 0.95),
        ],
    )
    def test_metric_parsed(
        self,
        extractor: SessionStateExtractor,
        text: str,
        metric: str,
        expected: float,
    ):
        scores = extractor._extract_quality_scores(text)
        assert metric in scores
        assert abs(scores[metric] - expected) < 0.001

    def test_unknown_metric_ignored(self, extractor: SessionStateExtractor):
        scores = extractor._extract_quality_scores("bananas: 0.99")
        assert "bananas" not in scores

    def test_percentage_normalised(self, extractor: SessionStateExtractor):
        scores = extractor._extract_quality_scores("score: 75%")
        assert abs(scores["score"] - 0.75) < 0.001

    def test_no_metrics_returns_empty(self, extractor: SessionStateExtractor):
        assert extractor._extract_quality_scores("no numbers here") == {}

    def test_last_occurrence_wins(self, extractor: SessionStateExtractor):
        scores = extractor._extract_quality_scores("score: 0.5\nscore: 0.9")
        assert abs(scores["score"] - 0.9) < 0.001


# ── Singleton ──────────────────────────────────────────────────────────


class TestGetSessionStateExtractor:
    """get_session_state_extractor() returns the same instance on repeat calls."""

    def test_returns_extractor_instance(self):
        result = get_session_state_extractor()
        assert isinstance(result, SessionStateExtractor)

    def test_singleton_same_object(self):
        a = get_session_state_extractor()
        b = get_session_state_extractor()
        assert a is b
