"""Tests for intent_parser.py and the query() method on UnifiedMemoryStore."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from vetinari.memory.intent_parser import (
    IntentParser,
    ParsedQuery,
    QueryIntent,
    get_intent_parser,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parser() -> IntentParser:
    return IntentParser()


# ---------------------------------------------------------------------------
# TestQueryIntentDetection — happy path for each intent
# ---------------------------------------------------------------------------


class TestQueryIntentDetection:
    @pytest.mark.parametrize(
        "query",
        [
            "what went wrong with caching last time?",
            "why did the inference step fail?",
            "what errors did we hit with the embedding pipeline?",
            "the deployment crashed — what happened?",
            "show me bugs related to search",
        ],
    )
    def test_failure_queries_classify_as_episode_recall(self, query: str) -> None:
        parsed = _parser().parse(query)
        assert parsed.intent == QueryIntent.EPISODE_RECALL
        assert parsed.success_filter is False

    @pytest.mark.parametrize(
        "query",
        [
            "what worked well with the inference pipeline?",
            "which tasks succeeded recently?",
            "show me good results from routing",
        ],
    )
    def test_success_queries_classify_as_episode_recall(self, query: str) -> None:
        parsed = _parser().parse(query)
        assert parsed.intent == QueryIntent.EPISODE_RECALL
        assert parsed.success_filter is True

    @pytest.mark.parametrize(
        "query",
        [
            "what happened last week?",
            "show me yesterday's activity",
            "what did we do recently?",
            "history of changes last month",
            "when did the update happen?",
            "what ran today?",
        ],
    )
    def test_temporal_queries_classify_as_timeline(self, query: str) -> None:
        parsed = _parser().parse(query)
        assert parsed.intent == QueryIntent.TIMELINE

    @pytest.mark.parametrize(
        "query",
        [
            "what are the rules about memory consolidation?",
            "how to configure the embedding model?",
            "what is the standard for episode scoring?",
            "documentation for the agent interface",
            "best practice for error handling",
            "guidelines on logging",
        ],
    )
    def test_knowledge_queries_classify_as_knowledge_base(self, query: str) -> None:
        parsed = _parser().parse(query)
        assert parsed.intent == QueryIntent.KNOWLEDGE_BASE

    @pytest.mark.parametrize(
        "query",
        [
            "tell me about the project architecture",
            "agent interaction patterns",
            "memory consolidation overview",
        ],
    )
    def test_generic_queries_fall_back_to_semantic_search(self, query: str) -> None:
        parsed = _parser().parse(query)
        assert parsed.intent == QueryIntent.SEMANTIC_SEARCH


# ---------------------------------------------------------------------------
# TestTimeRangeParsing
# ---------------------------------------------------------------------------


class TestTimeRangeParsing:
    def _now_ms(self) -> int:
        return int(datetime.now(timezone.utc).timestamp() * 1000)

    def test_last_week_produces_seven_day_range(self) -> None:
        parsed = _parser().parse("what happened last week?")
        start, end = parsed.time_range
        assert start is not None
        assert end is not None
        # Range should be ~7 days (allow 2 s tolerance for test execution time)
        delta_days = (end - start) / 1000 / 86400
        assert 6.9 < delta_days < 7.1

    def test_yesterday_produces_one_day_range(self) -> None:
        parsed = _parser().parse("show me yesterday's work")
        start, end = parsed.time_range
        assert start is not None
        assert end is not None
        delta_days = (end - start) / 1000 / 86400
        assert 0.9 < delta_days < 1.1

    def test_last_month_produces_thirty_day_range(self) -> None:
        parsed = _parser().parse("history of changes last month")
        start, end = parsed.time_range
        assert start is not None
        assert end is not None
        delta_days = (end - start) / 1000 / 86400
        assert 29.9 < delta_days < 30.1

    def test_recently_produces_three_day_range(self) -> None:
        parsed = _parser().parse("what happened recently?")
        start, end = parsed.time_range
        assert start is not None
        assert end is not None
        delta_days = (end - start) / 1000 / 86400
        assert 2.9 < delta_days < 3.1

    def test_today_produces_range_from_midnight(self) -> None:
        parsed = _parser().parse("what ran today?")
        start, end = parsed.time_range
        assert start is not None
        assert end is not None
        # start should be today at midnight UTC — at most 1 day ago
        delta_days = (end - start) / 1000 / 86400
        assert 0.0 <= delta_days <= 1.0

    def test_no_time_reference_returns_none_tuple(self) -> None:
        parsed = _parser().parse("tell me about the caching module")
        assert parsed.time_range == (None, None)

    def test_end_is_approximately_now(self) -> None:
        before = self._now_ms()
        parsed = _parser().parse("what happened last week?")
        after = self._now_ms()
        _, end = parsed.time_range
        assert end is not None
        # end must be within the window [before, after + 1s]
        assert before <= end <= after + 1000


# ---------------------------------------------------------------------------
# TestKeywordExtraction
# ---------------------------------------------------------------------------


class TestKeywordExtraction:
    def test_stop_words_excluded(self) -> None:
        parsed = _parser().parse("what is the best way to do this?")
        assert "the" not in parsed.keywords
        assert "is" not in parsed.keywords
        assert "to" not in parsed.keywords
        assert "do" not in parsed.keywords

    def test_meaningful_terms_included(self) -> None:
        parsed = _parser().parse("memory consolidation rules for agents")
        assert "memory" in parsed.keywords
        assert "consolidation" in parsed.keywords
        assert "rules" in parsed.keywords
        assert "agents" in parsed.keywords

    def test_short_tokens_excluded(self) -> None:
        # "a", "an", "in" are all < 3 chars or stop words
        parsed = _parser().parse("a bug in an old module")
        for kw in parsed.keywords:
            assert len(kw) >= 3, f"Short keyword leaked: {kw!r}"

    def test_punctuation_stripped(self) -> None:
        parsed = _parser().parse("what went wrong? (error in caching!)")
        for kw in parsed.keywords:
            assert "?" not in kw
            assert "!" not in kw
            assert "(" not in kw

    def test_keywords_are_lowercase(self) -> None:
        parsed = _parser().parse("Error in Caching Module")
        assert all(kw == kw.lower() for kw in parsed.keywords)


# ---------------------------------------------------------------------------
# TestTaskTypeExtraction
# ---------------------------------------------------------------------------


class TestTaskTypeExtraction:
    @pytest.mark.parametrize(
        ("query", "expected_task_type"),
        [
            ("what went wrong with caching?", "caching"),
            ("why did inference fail?", "inference"),
            ("errors in the embedding pipeline", "embedding"),
            ("search broke yesterday", "search"),
            ("deployment crashed", "deployment"),
            ("problem with routing logic", "routing"),
        ],
    )
    def test_task_type_extracted_from_episode_query(self, query: str, expected_task_type: str) -> None:
        parsed = _parser().parse(query)
        assert parsed.task_type == expected_task_type

    def test_task_type_none_when_not_detected(self) -> None:
        parsed = _parser().parse("something went wrong with the pipeline")
        # "pipeline" is not in the known task-type list
        assert parsed.task_type is None

    def test_task_type_none_for_timeline_query(self) -> None:
        parsed = _parser().parse("what happened last week?")
        assert parsed.task_type is None


# ---------------------------------------------------------------------------
# TestTopicExtraction
# ---------------------------------------------------------------------------


class TestTopicExtraction:
    def test_topic_extracted_from_rules_query(self) -> None:
        parsed = _parser().parse("what are the rules about memory consolidation?")
        assert parsed.topic is not None
        assert "memory" in parsed.topic or "consolidation" in parsed.topic

    def test_topic_extracted_from_how_to_query(self) -> None:
        parsed = _parser().parse("how to configure the embedding model?")
        assert parsed.topic is not None
        assert "configure" in parsed.topic or "embedding" in parsed.topic

    def test_topic_none_for_episode_query(self) -> None:
        parsed = _parser().parse("what went wrong with caching?")
        assert parsed.topic is None

    def test_topic_none_for_semantic_fallback(self) -> None:
        parsed = _parser().parse("general project overview")
        assert parsed.topic is None


# ---------------------------------------------------------------------------
# TestParsedQueryImmutability
# ---------------------------------------------------------------------------


class TestParsedQueryImmutability:
    def test_parsed_query_is_frozen(self) -> None:
        parsed = _parser().parse("what went wrong?")
        with pytest.raises((AttributeError, TypeError)):
            parsed.intent = QueryIntent.TIMELINE  # type: ignore[misc]

    def test_parsed_query_preserves_original(self) -> None:
        original = "  What Went Wrong With Caching?  "
        parsed = _parser().parse(original)
        assert parsed.original_query == original


# ---------------------------------------------------------------------------
# TestSingleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_intent_parser_returns_same_instance(self) -> None:
        p1 = get_intent_parser()
        p2 = get_intent_parser()
        assert p1 is p2

    def test_singleton_produces_correct_results(self) -> None:
        parsed = get_intent_parser().parse("what went wrong with search?")
        assert parsed.intent == QueryIntent.EPISODE_RECALL


# ---------------------------------------------------------------------------
# TestUnifiedMemoryStoreQuery — routing behaviour
# ---------------------------------------------------------------------------


class TestUnifiedMemoryStoreQuery:
    """Verify that UnifiedMemoryStore.query() dispatches to the right backend."""

    @pytest.fixture
    def store(self, tmp_path):
        """An in-memory store backed by a temp SQLite file."""
        from vetinari.memory.unified import UnifiedMemoryStore

        db = str(tmp_path / "test.db")
        s = UnifiedMemoryStore(db_path=db)
        yield s
        s.close()

    def test_episode_recall_intent_calls_recall_episodes(self, store) -> None:
        with patch.object(store, "recall_episodes", return_value=[]) as mock_recall:
            result = store.query("what went wrong with caching?")
        mock_recall.assert_called_once()
        call_kwargs = mock_recall.call_args[1]
        assert call_kwargs["task_type"] == "caching"
        assert call_kwargs["successful_only"] is False
        assert result == []

    def test_success_filter_true_for_success_queries(self, store) -> None:
        with patch.object(store, "recall_episodes", return_value=[]) as mock_recall:
            store.query("what worked well with inference?")
        call_kwargs = mock_recall.call_args[1]
        assert call_kwargs["successful_only"] is True

    def test_timeline_intent_calls_timeline(self, store) -> None:
        from vetinari.memory.interfaces import MemoryEntry

        fake_entries = [MemoryEntry(content="old task")]
        with patch.object(store, "timeline", return_value=fake_entries) as mock_tl:
            result = store.query("what happened last week?")
        mock_tl.assert_called_once()
        assert result == fake_entries

    def test_timeline_passes_agent_filter(self, store) -> None:
        with patch.object(store, "timeline", return_value=[]) as mock_tl:
            store.query("what happened last week?", agent="worker")
        assert mock_tl.call_args[1]["agent"] == "worker"

    def test_knowledge_base_intent_calls_search_with_entry_types(self, store) -> None:
        from vetinari.memory.interfaces import MemoryEntry

        fake = [MemoryEntry(content="rule text")]
        with patch.object(store, "search", return_value=fake) as mock_search:
            result = store.query("what are the rules about memory consolidation?")
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert "discovery" in call_kwargs["entry_types"]
        assert call_kwargs["use_semantic"] is True
        assert result == fake

    def test_semantic_fallback_calls_search(self, store) -> None:
        from vetinari.memory.interfaces import MemoryEntry

        fake = [MemoryEntry(content="general result")]
        with patch.object(store, "search", return_value=fake) as mock_search:
            result = store.query("tell me about the architecture")
        mock_search.assert_called_once()
        assert mock_search.call_args[1]["use_semantic"] is True
        assert result == fake

    def test_episode_to_entry_produces_memory_entry(self, store) -> None:
        from vetinari.memory.episode_recorder import RecordedEpisode
        from vetinari.memory.interfaces import MemoryEntry
        from vetinari.types import MemoryType

        ep = RecordedEpisode(
            episode_id="ep_test01",
            task_summary="run caching pipeline",
            agent_type="worker",
            task_type="caching",
            output_summary="cache warmed successfully",
            quality_score=0.9,
            success=True,
            model_id="model-1",
            scope="global",
        )
        entry = store._episode_to_entry(ep)
        assert isinstance(entry, MemoryEntry)
        assert entry.id == "ep_test01"
        assert entry.agent == "worker"
        assert entry.entry_type == MemoryType.SUCCESS
        assert entry.content == "run caching pipeline"
        assert entry.summary == "cache warmed successfully"
        assert entry.metadata["task_type"] == "caching"

    def test_episode_to_entry_failure_uses_problem_type(self, store) -> None:
        from vetinari.memory.episode_recorder import RecordedEpisode
        from vetinari.types import MemoryType

        ep = RecordedEpisode(
            episode_id="ep_fail01",
            task_summary="broken inference",
            success=False,
        )
        entry = store._episode_to_entry(ep)
        assert entry.entry_type == MemoryType.PROBLEM


# ---------------------------------------------------------------------------
# TestAskDeprecationWarning
# ---------------------------------------------------------------------------


class TestAskDeprecationWarning:
    def test_ask_delegates_to_query(self, tmp_path) -> None:
        from vetinari.memory.unified import UnifiedMemoryStore

        db = str(tmp_path / "dep.db")
        store = UnifiedMemoryStore(db_path=db)
        # Reset class-level flag so the warning fires in this test
        UnifiedMemoryStore._ask_deprecation_warned = False
        try:
            with patch.object(store, "query", return_value=[]) as mock_query:
                store.ask("test question", agent="foreman")
            mock_query.assert_called_once_with("test question", agent="foreman")
        finally:
            store.close()

    def test_ask_logs_deprecation_warning(self, tmp_path, caplog) -> None:
        import logging

        from vetinari.memory.unified import UnifiedMemoryStore

        db = str(tmp_path / "dep2.db")
        store = UnifiedMemoryStore(db_path=db)
        UnifiedMemoryStore._ask_deprecation_warned = False
        try:
            with patch.object(store, "query", return_value=[]):
                with caplog.at_level(logging.WARNING, logger="vetinari.memory.unified"):
                    store.ask("any question")
            assert any("deprecated" in r.message.lower() for r in caplog.records)
        finally:
            store.close()
