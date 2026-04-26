"""
Tests for vetinari/dashboard/log_aggregator.py  (Phase 4 Step 4)

Coverage:
    - LogRecord dataclass creation and serialisation
    - FileBackend: configure, send, missing config guard
    - DatadogBackend: configure, send (mocked), missing api_key guard
    - Unknown backend raises ValueError
    - LogAggregator singleton behaviour
    - configure_backend / remove_backend / list_backends
    - ingest and buffer growth
    - ingest_many
    - Auto-flush at batch_size
    - flush() forces dispatch
    - search: by trace_id, level, logger_name, message_contains, since, limit
    - get_trace_records ordering
    - correlate_span
    - get_stats
    - clear_buffer
    - AggregatorHandler bridge
    - reset_log_aggregator creates fresh instance
"""

import logging
import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from vetinari.dashboard.log_aggregator import (
    AggregatorHandler,
    BackendBase,
    LogAggregator,
    LogRecord,
    get_log_aggregator,
    reset_log_aggregator,
)
from vetinari.dashboard.log_backends import (
    DatadogBackend,
    FileBackend,
    SSEBackend,
    WebhookBackend,
    get_sse_backend,
    reset_sse_backend,
)
from vetinari.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rec(
    message="test",
    level="INFO",
    trace_id=None,
    span_id=None,
    logger_name=None,
    extra=None,
    ts=None,
) -> LogRecord:
    return LogRecord(
        message=message,
        level=level,
        trace_id=trace_id,
        span_id=span_id,
        logger_name=logger_name,
        extra=extra or {},
        timestamp=ts if ts is not None else time.time(),
    )


# ---------------------------------------------------------------------------
# LogRecord
# ---------------------------------------------------------------------------


class TestLogRecord:
    def test_defaults(self):
        r = _rec()
        assert r.message == "test"
        assert r.level == "INFO"
        assert isinstance(r.timestamp, float)

    def test_to_dict_contains_message(self):
        r = _rec(message="hello", trace_id="t1")
        d = r.to_dict()
        assert d["message"] == "hello"
        assert d["trace_id"] == "t1"

    def test_extra_fields_merged_into_dict(self):
        r = _rec(extra={"plan_id": "p42", "risk": 0.3})
        d = r.to_dict()
        assert d["plan_id"] == "p42"
        assert abs(d["risk"] - 0.3) < 1e-7

    def test_to_json_is_valid_json(self):
        import json

        r = _rec(message="json test", trace_id="xyz")
        parsed = json.loads(r.to_json())
        assert parsed["message"] == "json test"


# ---------------------------------------------------------------------------
# FileBackend
# ---------------------------------------------------------------------------


class TestFileBackend:
    def test_send_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.jsonl")
            fb = FileBackend()
            fb.configure(path=path)
            ok = fb.send([_rec("line1"), _rec("line2")])
            assert ok
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2

    def test_send_not_configured_returns_false(self):
        fb = FileBackend()
        result = fb.send([_rec()])
        assert not result

    def test_send_appends_on_multiple_calls(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "append.jsonl")
            fb = FileBackend()
            fb.configure(path=path)
            fb.send([_rec("first")])
            fb.send([_rec("second")])
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2


# ---------------------------------------------------------------------------
# DatadogBackend
# ---------------------------------------------------------------------------


class TestDatadogBackend:
    def test_configure(self):
        b = DatadogBackend()
        b.configure(api_key="ddkey123", service="vet")
        assert b._api_key == "ddkey123"
        assert b._service == "vet"

    def test_send_no_api_key_returns_false(self):
        b = DatadogBackend()
        assert not b.send([_rec()])

    def test_send_uses_requests(self):
        b = DatadogBackend()
        b.configure(api_key="ddkey")
        mock_resp = MagicMock(status_code=202, text="accepted")
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        with patch("vetinari.dashboard.log_backends.create_session", return_value=mock_session):
            ok = b.send([_rec("dd msg")])
            assert ok
            mock_session.post.assert_called_once()


# ---------------------------------------------------------------------------
# LogAggregator singleton
# ---------------------------------------------------------------------------


class TestLogAggregatorSingleton:
    def setup_method(self):
        reset_log_aggregator()

    def teardown_method(self):
        reset_log_aggregator()

    def test_get_returns_same_instance(self):
        a1 = get_log_aggregator()
        a2 = get_log_aggregator()
        assert a1 is a2

    def test_reset_creates_fresh_instance(self):
        a1 = get_log_aggregator()
        reset_log_aggregator()
        a2 = get_log_aggregator()
        assert a1 is not a2


# ---------------------------------------------------------------------------
# Backend management
# ---------------------------------------------------------------------------


class TestBackendManagement:
    def setup_method(self):
        reset_log_aggregator()
        self.agg = get_log_aggregator()

    def teardown_method(self):
        reset_log_aggregator()

    def test_configure_file_backend(self):
        with tempfile.TemporaryDirectory() as d:
            self.agg.configure_backend("file", path=os.path.join(d, "test.jsonl"))
            assert "file" in self.agg.list_backends()

    def test_unknown_backend_raises(self):
        with pytest.raises(ConfigurationError):
            self.agg.configure_backend("nonexistent")

    def test_remove_backend(self):
        with tempfile.TemporaryDirectory() as d:
            self.agg.configure_backend("file", path=os.path.join(d, "r.jsonl"))
            result = self.agg.remove_backend("file")
            assert result is True
            assert "file" not in self.agg.list_backends()

    def test_remove_nonexistent_returns_false(self):
        assert not self.agg.remove_backend("ghost")


# ---------------------------------------------------------------------------
# Ingestion and buffer
# ---------------------------------------------------------------------------


class TestIngestion:
    def setup_method(self):
        reset_log_aggregator()
        self.agg = get_log_aggregator()

    def teardown_method(self):
        reset_log_aggregator()

    def test_ingest_adds_to_buffer(self):
        self.agg.ingest(_rec("msg1"))
        assert self.agg.get_stats()["buffer_size"] == 1

    def test_ingest_many(self):
        self.agg.ingest_many([_rec(f"m{i}") for i in range(10)])
        assert self.agg.get_stats()["buffer_size"] == 10

    def test_buffer_cap(self):
        """Buffer should not grow beyond MAX_BUFFER."""
        cap = LogAggregator.MAX_BUFFER
        self.agg.ingest_many([_rec(f"m{i}") for i in range(cap + 50)])
        assert self.agg.get_stats()["buffer_size"] <= cap

    def test_flush_dispatches_to_backend(self):
        mock_backend = MagicMock()
        mock_backend.send.return_value = True
        self.agg._backends["mock"] = mock_backend
        self.agg.ingest(_rec("flush me"))
        self.agg.flush()
        mock_backend.send.assert_called_once()
        sent_records = mock_backend.send.call_args[0][0]
        assert len(sent_records) >= 1

    def test_auto_flush_at_batch_size(self):
        self.agg._batch_size = 3
        mock_backend = MagicMock()
        mock_backend.send.return_value = True
        self.agg._backends["mock"] = mock_backend
        for i in range(3):
            self.agg.ingest(_rec(f"r{i}"))
        assert mock_backend.send.call_count >= 1
        sent_records = mock_backend.send.call_args[0][0]
        assert len(sent_records) >= 1

    def test_clear_buffer(self):
        self.agg.ingest_many([_rec() for _ in range(5)])
        self.agg.clear_buffer()
        assert self.agg.get_stats()["buffer_size"] == 0


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    def setup_method(self):
        reset_log_aggregator()
        self.agg = get_log_aggregator()
        # Populate with varied records
        now = time.time()
        self.agg.ingest(_rec("alpha", level="INFO", trace_id="t1", span_id="s1", logger_name="svc.a", ts=now - 100))
        self.agg.ingest(_rec("beta", level="ERROR", trace_id="t1", span_id="s2", logger_name="svc.b", ts=now - 50))
        self.agg.ingest(_rec("gamma", level="INFO", trace_id="t2", span_id="s3", logger_name="svc.a", ts=now - 10))

    def teardown_method(self):
        reset_log_aggregator()

    def test_search_by_trace_id(self):
        results = self.agg.search(trace_id="t1")
        assert len(results) == 2
        assert all(r.trace_id == "t1" for r in results)

    def test_search_by_level(self):
        results = self.agg.search(level="ERROR")
        assert len(results) == 1
        assert results[0].message == "beta"

    def test_search_by_logger_name(self):
        results = self.agg.search(logger_name="svc.a")
        assert len(results) == 2

    def test_search_message_contains(self):
        results = self.agg.search(message_contains="gamm")
        assert len(results) == 1
        assert results[0].message == "gamma"

    def test_search_since(self):
        since = time.time() - 60  # last 60 s
        results = self.agg.search(since=since)
        # Only beta and gamma are within 60s
        assert len(results) == 2

    def test_search_limit(self):
        results = self.agg.search(limit=2)
        assert len(results) == 2

    def test_get_trace_records_ordered(self):
        records = self.agg.get_trace_records("t1")
        assert len(records) == 2
        assert records[0].timestamp <= records[1].timestamp

    def test_correlate_span(self):
        results = self.agg.correlate_span("t1", "s2")
        assert len(results) == 1
        assert results[0].message == "beta"

    def test_search_no_match_returns_empty(self):
        results = self.agg.search(trace_id="nonexistent")
        assert results == []


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def setup_method(self):
        reset_log_aggregator()
        self.agg = get_log_aggregator()

    def teardown_method(self):
        reset_log_aggregator()

    def test_stats_keys(self):
        stats = self.agg.get_stats()
        for k in ("buffer_size", "pending", "backends", "max_buffer", "batch_size"):
            assert k in stats

    def test_stats_after_ingest(self):
        self.agg.ingest(_rec())
        assert self.agg.get_stats()["buffer_size"] == 1


# ---------------------------------------------------------------------------
# AggregatorHandler
# ---------------------------------------------------------------------------


class TestAggregatorHandler:
    def setup_method(self):
        reset_log_aggregator()

    def teardown_method(self):
        reset_log_aggregator()

    def test_handler_feeds_aggregator(self):
        agg = get_log_aggregator()
        handler = AggregatorHandler()
        test_logger = logging.getLogger("test.agg_handler")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)
        test_logger.info("handler test message")
        test_logger.removeHandler(handler)

        results = agg.search(message_contains="handler test message")
        assert len(results) > 0

    def test_handler_captures_level(self):
        agg = get_log_aggregator()
        handler = AggregatorHandler()
        test_logger = logging.getLogger("test.level_check")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)
        test_logger.warning("warn check")
        test_logger.removeHandler(handler)

        results = agg.search(level="WARNING", message_contains="warn check")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# BackendBase
# ---------------------------------------------------------------------------


class TestBackendBase:
    def test_cannot_instantiate_abstract_base(self):
        """BackendBase is abstract — subclasses must implement send()."""
        with pytest.raises(TypeError, match="abstract method"):
            BackendBase()


# ---------------------------------------------------------------------------
# FileBackend — error path
# ---------------------------------------------------------------------------


class TestFileBackendErrors:
    def test_send_os_error_returns_false(self):
        from unittest.mock import patch

        fb = FileBackend()
        # Simulate an OSError during file open (platform-independent)
        with patch("pathlib.Path.open", side_effect=OSError("disk full")):
            result = fb.send([_rec()])
        assert not result


# ---------------------------------------------------------------------------
# DatadogBackend — additional paths
# ---------------------------------------------------------------------------


class TestDatadogBackendExtra:
    def test_send_missing_requests_returns_false(self):
        b = DatadogBackend()
        b.configure(api_key="key123")
        with patch("vetinari.dashboard.log_backends.create_session", side_effect=ImportError("no requests")):
            result = b.send([_rec()])
            assert not result

    def test_send_non_success_returns_false(self):
        b = DatadogBackend()
        b.configure(api_key="key123")
        mock_resp = MagicMock(status_code=400, text="bad request")
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        with patch("vetinari.dashboard.log_backends.create_session", return_value=mock_session):
            assert not b.send([_rec()])

    def test_send_exception_returns_false(self):
        b = DatadogBackend()
        b.configure(api_key="key123")
        mock_session = MagicMock()
        mock_session.post.side_effect = ConnectionError("err")
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        with patch("vetinari.dashboard.log_backends.create_session", return_value=mock_session):
            assert not b.send([_rec()])


# ---------------------------------------------------------------------------
# WebhookBackend
# ---------------------------------------------------------------------------


class TestWebhookBackend:
    def test_not_configured_returns_false(self):
        b = WebhookBackend()
        assert not b.send([_rec()])

    def test_configure_sets_url_and_headers(self):
        b = WebhookBackend()
        b.configure(url="http://hook.example.com", headers={"X-Key": "val"}, timeout=5)
        assert b._url == "http://hook.example.com"
        assert b._headers["X-Key"] == "val"
        assert b._timeout == 5

    def test_send_success(self):
        b = WebhookBackend()
        b.configure(url="http://hook.example.com")
        mock_resp = MagicMock(ok=True, status_code=200)
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        with patch("vetinari.dashboard.log_backends.create_session", return_value=mock_session):
            assert b.send([_rec("webhook msg")])

    def test_send_non_ok_returns_false(self):
        b = WebhookBackend()
        b.configure(url="http://hook.example.com")
        mock_resp = MagicMock(ok=False, status_code=500, text="error")
        mock_session = MagicMock()
        mock_session.post.return_value = mock_resp
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        with patch("vetinari.dashboard.log_backends.create_session", return_value=mock_session):
            assert not b.send([_rec()])

    def test_send_exception_returns_false(self):
        b = WebhookBackend()
        b.configure(url="http://hook.example.com")
        mock_session = MagicMock()
        mock_session.post.side_effect = ConnectionError("down")
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        with patch("vetinari.dashboard.log_backends.create_session", return_value=mock_session):
            assert not b.send([_rec()])

    def test_send_missing_requests_returns_false(self):
        b = WebhookBackend()
        b.configure(url="http://hook.example.com")
        with patch("vetinari.dashboard.log_backends.create_session", side_effect=ImportError("no requests")):
            assert not b.send([_rec()])


# ---------------------------------------------------------------------------
# SSEBackend
# ---------------------------------------------------------------------------


class TestSSEBackend:
    def test_send_buffers_records(self):
        b = SSEBackend()
        assert b.send([_rec("a"), _rec("b")])
        assert len(b.get_recent()) == 2

    def test_configure_changes_buffer_size(self):
        b = SSEBackend()
        b.configure(max_buffer=5)
        for i in range(10):
            b.send([_rec(f"m{i}")])
        assert len(b.get_recent(limit=100)) == 5

    def test_get_recent_respects_limit(self):
        b = SSEBackend()
        b.send([_rec(f"m{i}") for i in range(10)])
        assert len(b.get_recent(limit=3)) == 3

    def test_close_clears_buffer(self):
        b = SSEBackend()
        b.send([_rec("x")])
        b.close()
        assert len(b.get_recent()) == 0


# ---------------------------------------------------------------------------
# SSE singleton helpers
# ---------------------------------------------------------------------------


class TestSSESingleton:
    def setup_method(self):
        reset_sse_backend()

    def teardown_method(self):
        reset_sse_backend()

    def test_get_returns_same_instance(self):
        s1 = get_sse_backend()
        s2 = get_sse_backend()
        assert s1 is s2

    def test_reset_creates_fresh_instance(self):
        s1 = get_sse_backend()
        reset_sse_backend()
        s2 = get_sse_backend()
        assert s1 is not s2


# ---------------------------------------------------------------------------
# Flush error handling
# ---------------------------------------------------------------------------


class TestFlushErrorHandling:
    def setup_method(self):
        reset_log_aggregator()
        self.agg = get_log_aggregator()

    def teardown_method(self):
        reset_log_aggregator()

    def test_backend_exception_during_flush_is_caught(self, caplog):
        mock_backend = MagicMock()
        mock_backend.send.side_effect = RuntimeError("boom")
        self.agg._backends["broken"] = mock_backend
        self.agg.ingest(_rec("test"))
        # Should not raise — backend exception must be caught internally
        with caplog.at_level(logging.ERROR, logger="vetinari.dashboard.log_aggregator"):
            self.agg.flush()
        mock_backend.send.assert_called_once()  # send was attempted despite raising
        assert any("Backend 'broken' raised during send: boom" in record.message for record in caplog.records)

    def test_backend_returns_false_logs_warning(self):
        mock_backend = MagicMock()
        mock_backend.send.return_value = False
        self.agg._backends["failing"] = mock_backend
        self.agg.ingest(_rec("test"))
        self.agg.flush()
        mock_backend.send.assert_called_once()
        # Backend received the records even though it returned failure
        sent_records = mock_backend.send.call_args[0][0]
        assert len(sent_records) >= 1


# ---------------------------------------------------------------------------
# SSEBackend.close() clears _clients (Defect 3 regression)
# ---------------------------------------------------------------------------


class TestSSEBackendClose:
    """close() must clear _clients so no stale queues remain after shutdown."""

    def test_close_clears_client_list(self):
        """After close(), _clients is empty."""
        import queue as _queue

        b = SSEBackend()
        q = _queue.Queue()
        b.add_client(q)
        assert len(b._clients) == 1
        b.close()
        assert len(b._clients) == 0

    def test_send_after_close_reaches_no_clients(self):
        """send() after close() does not attempt delivery to any client queue."""
        import queue as _queue

        b = SSEBackend()
        q = _queue.Queue()
        b.add_client(q)
        b.close()
        # send() must not raise and no records land in the closed client queue
        b.send([_rec("after close")])
        assert q.empty()

    def test_close_clears_buffer_and_clients_together(self):
        """close() resets both the record buffer and the client list atomically."""
        import queue as _queue

        b = SSEBackend()
        b.send([_rec("x")])
        q = _queue.Queue()
        b.add_client(q)
        b.close()
        assert len(b.get_recent()) == 0
        assert len(b._clients) == 0


# ---------------------------------------------------------------------------
# SSEBackend.send() serialises via rec.to_json() (Defect 7 regression)
# ---------------------------------------------------------------------------


class TestSSEBackendSendSerialization:
    """send() must put valid JSON strings (from rec.to_json()) into client queues."""

    def test_client_receives_valid_json(self):
        """Each item put into a client queue is valid JSON."""
        import json
        import queue as _queue

        b = SSEBackend()
        q = _queue.Queue()
        b.add_client(q)
        b.send([_rec("hello")])
        raw = q.get_nowait()
        # Must parse as JSON without error
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_client_json_contains_message_field(self):
        """Serialised record includes the 'message' field from the LogRecord."""
        import json
        import queue as _queue

        b = SSEBackend()
        q = _queue.Queue()
        b.add_client(q)
        b.send([_rec("check message")])
        parsed = json.loads(q.get_nowait())
        assert parsed.get("message") == "check message"

    def test_client_does_not_receive_repr_blob(self):
        """The queue item must NOT be a Python repr string (e.g. 'LogRecord(...)' )."""
        import queue as _queue

        b = SSEBackend()
        q = _queue.Queue()
        b.add_client(q)
        b.send([_rec("repr test")])
        raw = q.get_nowait()
        # A repr blob starts with the class name; a proper JSON object starts with '{'
        assert raw.startswith("{"), f"Expected JSON object, got: {raw[:80]!r}"
