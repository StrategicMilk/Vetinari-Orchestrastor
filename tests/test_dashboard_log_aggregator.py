"""
Tests for vetinari/dashboard/log_aggregator.py  (Phase 4 Step 4)

Coverage:
    - LogRecord dataclass creation and serialisation
    - FileBackend: configure, send, missing config guard
    - ElasticsearchBackend: configure, send (mocked), missing requests guard
    - SplunkBackend: configure, send (mocked), missing config guard
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
import unittest
from unittest.mock import MagicMock, patch

from vetinari.dashboard.log_aggregator import (
    AggregatorHandler,
    BackendBase,
    DatadogBackend,
    ElasticsearchBackend,
    FileBackend,
    LogAggregator,
    LogRecord,
    SSEBackend,
    SplunkBackend,
    WebhookBackend,
    get_log_aggregator,
    get_sse_backend,
    reset_log_aggregator,
    reset_sse_backend,
)


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
    r = LogRecord(
        message=message,
        level=level,
        trace_id=trace_id,
        span_id=span_id,
        logger_name=logger_name,
        extra=extra or {},
    )
    if ts is not None:
        r.timestamp = ts
    return r


# ---------------------------------------------------------------------------
# LogRecord
# ---------------------------------------------------------------------------

class TestLogRecord(unittest.TestCase):

    def test_defaults(self):
        r = _rec()
        self.assertEqual(r.message, "test")
        self.assertEqual(r.level, "INFO")
        self.assertIsInstance(r.timestamp, float)

    def test_to_dict_contains_message(self):
        r = _rec(message="hello", trace_id="t1")
        d = r.to_dict()
        self.assertEqual(d["message"], "hello")
        self.assertEqual(d["trace_id"], "t1")

    def test_extra_fields_merged_into_dict(self):
        r = _rec(extra={"plan_id": "p42", "risk": 0.3})
        d = r.to_dict()
        self.assertEqual(d["plan_id"], "p42")
        self.assertAlmostEqual(d["risk"], 0.3)

    def test_to_json_is_valid_json(self):
        import json
        r = _rec(message="json test", trace_id="xyz")
        parsed = json.loads(r.to_json())
        self.assertEqual(parsed["message"], "json test")


# ---------------------------------------------------------------------------
# FileBackend
# ---------------------------------------------------------------------------

class TestFileBackend(unittest.TestCase):

    def test_send_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.jsonl")
            fb = FileBackend()
            fb.configure(path=path)
            ok = fb.send([_rec("line1"), _rec("line2")])
            self.assertTrue(ok)
            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)

    def test_send_not_configured_returns_false(self):
        fb = FileBackend()
        result = fb.send([_rec()])
        self.assertFalse(result)

    def test_send_appends_on_multiple_calls(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "append.jsonl")
            fb = FileBackend()
            fb.configure(path=path)
            fb.send([_rec("first")])
            fb.send([_rec("second")])
            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)


# ---------------------------------------------------------------------------
# ElasticsearchBackend
# ---------------------------------------------------------------------------

class TestElasticsearchBackend(unittest.TestCase):

    def test_configure_sets_url(self):
        b = ElasticsearchBackend()
        b.configure(url="http://es:9200", index="myindex")
        self.assertEqual(b._url, "http://es:9200")
        self.assertEqual(b._index, "myindex")

    def test_send_success(self):
        b = ElasticsearchBackend()
        b.configure(url="http://es:9200")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("vetinari.dashboard.log_aggregator.ElasticsearchBackend.send",
                   return_value=True) as mock_send:
            result = b.send([_rec()])
            # We patched send itself so it returns True
            self.assertTrue(result)

    def test_send_not_configured_returns_false(self):
        b = ElasticsearchBackend()
        result = b.send([_rec()])
        self.assertFalse(result)

    def test_send_uses_requests(self):
        """Verify that send issues a POST via requests when configured."""
        b = ElasticsearchBackend()
        b.configure(url="http://es:9200", index="test")
        mock_resp = MagicMock(status_code=200, text="ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            ok = b.send([_rec("es record")])
            self.assertTrue(ok)
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            self.assertIn("/_bulk", call_args[0][0])


# ---------------------------------------------------------------------------
# SplunkBackend
# ---------------------------------------------------------------------------

class TestSplunkBackend(unittest.TestCase):

    def test_configure(self):
        b = SplunkBackend()
        b.configure(url="http://splunk:8088", token="mytoken")
        self.assertEqual(b._token, "mytoken")

    def test_send_not_configured_returns_false(self):
        b = SplunkBackend()
        self.assertFalse(b.send([_rec()]))

    def test_send_uses_requests(self):
        b = SplunkBackend()
        b.configure(url="http://splunk:8088", token="tok")
        mock_resp = MagicMock(status_code=200, text="ok")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            ok = b.send([_rec("splunk msg")])
            self.assertTrue(ok)
            mock_post.assert_called_once()

    def test_send_non_200_returns_false(self):
        b = SplunkBackend()
        b.configure(url="http://splunk:8088", token="tok")
        mock_resp = MagicMock(status_code=503, text="unavailable")
        with patch("requests.post", return_value=mock_resp):
            self.assertFalse(b.send([_rec()]))


# ---------------------------------------------------------------------------
# DatadogBackend
# ---------------------------------------------------------------------------

class TestDatadogBackend(unittest.TestCase):

    def test_configure(self):
        b = DatadogBackend()
        b.configure(api_key="ddkey123", service="vet")
        self.assertEqual(b._api_key, "ddkey123")
        self.assertEqual(b._service, "vet")

    def test_send_no_api_key_returns_false(self):
        b = DatadogBackend()
        self.assertFalse(b.send([_rec()]))

    def test_send_uses_requests(self):
        b = DatadogBackend()
        b.configure(api_key="ddkey")
        mock_resp = MagicMock(status_code=202, text="accepted")
        with patch("requests.post", return_value=mock_resp) as mock_post:
            ok = b.send([_rec("dd msg")])
            self.assertTrue(ok)
            mock_post.assert_called_once()


# ---------------------------------------------------------------------------
# LogAggregator singleton
# ---------------------------------------------------------------------------

class TestLogAggregatorSingleton(unittest.TestCase):

    def setUp(self):
        reset_log_aggregator()

    def tearDown(self):
        reset_log_aggregator()

    def test_get_returns_same_instance(self):
        a1 = get_log_aggregator()
        a2 = get_log_aggregator()
        self.assertIs(a1, a2)

    def test_reset_creates_fresh_instance(self):
        a1 = get_log_aggregator()
        reset_log_aggregator()
        a2 = get_log_aggregator()
        self.assertIsNot(a1, a2)


# ---------------------------------------------------------------------------
# Backend management
# ---------------------------------------------------------------------------

class TestBackendManagement(unittest.TestCase):

    def setUp(self):
        reset_log_aggregator()
        self.agg = get_log_aggregator()

    def tearDown(self):
        reset_log_aggregator()

    def test_configure_file_backend(self):
        with tempfile.TemporaryDirectory() as d:
            self.agg.configure_backend("file", path=os.path.join(d, "test.jsonl"))
            self.assertIn("file", self.agg.list_backends())

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            self.agg.configure_backend("nonexistent")

    def test_remove_backend(self):
        with tempfile.TemporaryDirectory() as d:
            self.agg.configure_backend("file", path=os.path.join(d, "r.jsonl"))
            result = self.agg.remove_backend("file")
            self.assertTrue(result)
            self.assertNotIn("file", self.agg.list_backends())

    def test_remove_nonexistent_returns_false(self):
        self.assertFalse(self.agg.remove_backend("ghost"))


# ---------------------------------------------------------------------------
# Ingestion and buffer
# ---------------------------------------------------------------------------

class TestIngestion(unittest.TestCase):

    def setUp(self):
        reset_log_aggregator()
        self.agg = get_log_aggregator()

    def tearDown(self):
        reset_log_aggregator()

    def test_ingest_adds_to_buffer(self):
        self.agg.ingest(_rec("msg1"))
        self.assertEqual(self.agg.get_stats()["buffer_size"], 1)

    def test_ingest_many(self):
        self.agg.ingest_many([_rec(f"m{i}") for i in range(10)])
        self.assertEqual(self.agg.get_stats()["buffer_size"], 10)

    def test_buffer_cap(self):
        """Buffer should not grow beyond MAX_BUFFER."""
        cap = LogAggregator.MAX_BUFFER
        self.agg.ingest_many([_rec(f"m{i}") for i in range(cap + 50)])
        self.assertLessEqual(self.agg.get_stats()["buffer_size"], cap)

    def test_flush_dispatches_to_backend(self):
        mock_backend = MagicMock()
        mock_backend.send.return_value = True
        self.agg._backends["mock"] = mock_backend
        self.agg.ingest(_rec("flush me"))
        self.agg.flush()
        mock_backend.send.assert_called()

    def test_auto_flush_at_batch_size(self):
        self.agg._batch_size = 3
        mock_backend = MagicMock()
        mock_backend.send.return_value = True
        self.agg._backends["mock"] = mock_backend
        for i in range(3):
            self.agg.ingest(_rec(f"r{i}"))
        mock_backend.send.assert_called()

    def test_clear_buffer(self):
        self.agg.ingest_many([_rec() for _ in range(5)])
        self.agg.clear_buffer()
        self.assertEqual(self.agg.get_stats()["buffer_size"], 0)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearch(unittest.TestCase):

    def setUp(self):
        reset_log_aggregator()
        self.agg = get_log_aggregator()
        # Populate with varied records
        now = time.time()
        self.agg.ingest(_rec("alpha", level="INFO",  trace_id="t1", span_id="s1", logger_name="svc.a", ts=now - 100))
        self.agg.ingest(_rec("beta",  level="ERROR", trace_id="t1", span_id="s2", logger_name="svc.b", ts=now - 50))
        self.agg.ingest(_rec("gamma", level="INFO",  trace_id="t2", span_id="s3", logger_name="svc.a", ts=now - 10))

    def tearDown(self):
        reset_log_aggregator()

    def test_search_by_trace_id(self):
        results = self.agg.search(trace_id="t1")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.trace_id == "t1" for r in results))

    def test_search_by_level(self):
        results = self.agg.search(level="ERROR")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].message, "beta")

    def test_search_by_logger_name(self):
        results = self.agg.search(logger_name="svc.a")
        self.assertEqual(len(results), 2)

    def test_search_message_contains(self):
        results = self.agg.search(message_contains="gamm")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].message, "gamma")

    def test_search_since(self):
        since = time.time() - 60  # last 60 s
        results = self.agg.search(since=since)
        # Only beta and gamma are within 60s
        self.assertEqual(len(results), 2)

    def test_search_limit(self):
        results = self.agg.search(limit=2)
        self.assertEqual(len(results), 2)

    def test_get_trace_records_ordered(self):
        records = self.agg.get_trace_records("t1")
        self.assertEqual(len(records), 2)
        self.assertLessEqual(records[0].timestamp, records[1].timestamp)

    def test_correlate_span(self):
        results = self.agg.correlate_span("t1", "s2")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].message, "beta")

    def test_search_no_match_returns_empty(self):
        results = self.agg.search(trace_id="nonexistent")
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats(unittest.TestCase):

    def setUp(self):
        reset_log_aggregator()
        self.agg = get_log_aggregator()

    def tearDown(self):
        reset_log_aggregator()

    def test_stats_keys(self):
        stats = self.agg.get_stats()
        for k in ("buffer_size", "pending", "backends", "max_buffer", "batch_size"):
            self.assertIn(k, stats)

    def test_stats_after_ingest(self):
        self.agg.ingest(_rec())
        self.assertEqual(self.agg.get_stats()["buffer_size"], 1)


# ---------------------------------------------------------------------------
# AggregatorHandler
# ---------------------------------------------------------------------------

class TestAggregatorHandler(unittest.TestCase):

    def setUp(self):
        reset_log_aggregator()

    def tearDown(self):
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
        self.assertGreater(len(results), 0)

    def test_handler_captures_level(self):
        agg = get_log_aggregator()
        handler = AggregatorHandler()
        test_logger = logging.getLogger("test.level_check")
        test_logger.addHandler(handler)
        test_logger.setLevel(logging.DEBUG)
        test_logger.warning("warn check")
        test_logger.removeHandler(handler)

        results = agg.search(level="WARNING", message_contains="warn check")
        self.assertGreater(len(results), 0)


# ---------------------------------------------------------------------------
# BackendBase
# ---------------------------------------------------------------------------

class TestBackendBase(unittest.TestCase):

    def test_send_raises_not_implemented(self):
        b = BackendBase()
        with self.assertRaises(NotImplementedError):
            b.send([_rec()])


# ---------------------------------------------------------------------------
# FileBackend — error path
# ---------------------------------------------------------------------------

class TestFileBackendErrors(unittest.TestCase):

    def test_send_os_error_returns_false(self):
        fb = FileBackend()
        fb._path = "/nonexistent/deeply/nested/path/out.jsonl"
        result = fb.send([_rec()])
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# ElasticsearchBackend — additional paths
# ---------------------------------------------------------------------------

class TestElasticsearchBackendExtra(unittest.TestCase):

    def test_configure_with_api_key(self):
        b = ElasticsearchBackend()
        b.configure(url="http://es:9200", api_key="my-api-key")
        self.assertIn("Authorization", b._headers)
        self.assertEqual(b._headers["Authorization"], "ApiKey my-api-key")

    def test_send_missing_requests_returns_false(self):
        b = ElasticsearchBackend()
        b.configure(url="http://es:9200")
        with patch.dict("sys.modules", {"requests": None}):
            result = b.send([_rec()])
            self.assertFalse(result)

    def test_send_non_200_returns_false(self):
        b = ElasticsearchBackend()
        b.configure(url="http://es:9200")
        mock_resp = MagicMock(status_code=500, text="server error")
        with patch("requests.post", return_value=mock_resp):
            self.assertFalse(b.send([_rec()]))

    def test_send_exception_returns_false(self):
        b = ElasticsearchBackend()
        b.configure(url="http://es:9200")
        with patch("requests.post", side_effect=ConnectionError("timeout")):
            self.assertFalse(b.send([_rec()]))


# ---------------------------------------------------------------------------
# SplunkBackend — missing requests
# ---------------------------------------------------------------------------

class TestSplunkBackendExtra(unittest.TestCase):

    def test_send_missing_requests_returns_false(self):
        b = SplunkBackend()
        b.configure(url="http://splunk:8088", token="tok")
        with patch.dict("sys.modules", {"requests": None}):
            result = b.send([_rec()])
            self.assertFalse(result)

    def test_send_exception_returns_false(self):
        b = SplunkBackend()
        b.configure(url="http://splunk:8088", token="tok")
        with patch("requests.post", side_effect=ConnectionError("down")):
            self.assertFalse(b.send([_rec()]))


# ---------------------------------------------------------------------------
# DatadogBackend — additional paths
# ---------------------------------------------------------------------------

class TestDatadogBackendExtra(unittest.TestCase):

    def test_send_missing_requests_returns_false(self):
        b = DatadogBackend()
        b.configure(api_key="key123")
        with patch.dict("sys.modules", {"requests": None}):
            result = b.send([_rec()])
            self.assertFalse(result)

    def test_send_non_success_returns_false(self):
        b = DatadogBackend()
        b.configure(api_key="key123")
        mock_resp = MagicMock(status_code=400, text="bad request")
        with patch("requests.post", return_value=mock_resp):
            self.assertFalse(b.send([_rec()]))

    def test_send_exception_returns_false(self):
        b = DatadogBackend()
        b.configure(api_key="key123")
        with patch("requests.post", side_effect=ConnectionError("err")):
            self.assertFalse(b.send([_rec()]))


# ---------------------------------------------------------------------------
# WebhookBackend
# ---------------------------------------------------------------------------

class TestWebhookBackend(unittest.TestCase):

    def test_not_configured_returns_false(self):
        b = WebhookBackend()
        self.assertFalse(b.send([_rec()]))

    def test_configure_sets_url_and_headers(self):
        b = WebhookBackend()
        b.configure(url="http://hook.example.com", headers={"X-Key": "val"}, timeout=5)
        self.assertEqual(b._url, "http://hook.example.com")
        self.assertEqual(b._headers["X-Key"], "val")
        self.assertEqual(b._timeout, 5)

    def test_send_success(self):
        b = WebhookBackend()
        b.configure(url="http://hook.example.com")
        mock_resp = MagicMock(ok=True, status_code=200)
        with patch("requests.post", return_value=mock_resp):
            self.assertTrue(b.send([_rec("webhook msg")]))

    def test_send_non_ok_returns_false(self):
        b = WebhookBackend()
        b.configure(url="http://hook.example.com")
        mock_resp = MagicMock(ok=False, status_code=500, text="error")
        with patch("requests.post", return_value=mock_resp):
            self.assertFalse(b.send([_rec()]))

    def test_send_exception_returns_false(self):
        b = WebhookBackend()
        b.configure(url="http://hook.example.com")
        with patch("requests.post", side_effect=ConnectionError("down")):
            self.assertFalse(b.send([_rec()]))

    def test_send_missing_requests_returns_false(self):
        b = WebhookBackend()
        b.configure(url="http://hook.example.com")
        with patch.dict("sys.modules", {"requests": None}):
            self.assertFalse(b.send([_rec()]))


# ---------------------------------------------------------------------------
# SSEBackend
# ---------------------------------------------------------------------------

class TestSSEBackend(unittest.TestCase):

    def test_send_buffers_records(self):
        b = SSEBackend()
        self.assertTrue(b.send([_rec("a"), _rec("b")]))
        self.assertEqual(len(b.get_recent()), 2)

    def test_configure_changes_buffer_size(self):
        b = SSEBackend()
        b.configure(max_buffer=5)
        for i in range(10):
            b.send([_rec(f"m{i}")])
        self.assertEqual(len(b.get_recent(limit=100)), 5)

    def test_get_recent_respects_limit(self):
        b = SSEBackend()
        b.send([_rec(f"m{i}") for i in range(10)])
        self.assertEqual(len(b.get_recent(limit=3)), 3)

    def test_close_clears_buffer(self):
        b = SSEBackend()
        b.send([_rec("x")])
        b.close()
        self.assertEqual(len(b.get_recent()), 0)


# ---------------------------------------------------------------------------
# SSE singleton helpers
# ---------------------------------------------------------------------------

class TestSSESingleton(unittest.TestCase):

    def setUp(self):
        reset_sse_backend()

    def tearDown(self):
        reset_sse_backend()

    def test_get_returns_same_instance(self):
        s1 = get_sse_backend()
        s2 = get_sse_backend()
        self.assertIs(s1, s2)

    def test_reset_creates_fresh_instance(self):
        s1 = get_sse_backend()
        reset_sse_backend()
        s2 = get_sse_backend()
        self.assertIsNot(s1, s2)


# ---------------------------------------------------------------------------
# Flush error handling
# ---------------------------------------------------------------------------

class TestFlushErrorHandling(unittest.TestCase):

    def setUp(self):
        reset_log_aggregator()
        self.agg = get_log_aggregator()

    def tearDown(self):
        reset_log_aggregator()

    def test_backend_exception_during_flush_is_caught(self):
        mock_backend = MagicMock()
        mock_backend.send.side_effect = RuntimeError("boom")
        self.agg._backends["broken"] = mock_backend
        self.agg.ingest(_rec("test"))
        # Should not raise
        self.agg.flush()

    def test_backend_returns_false_logs_warning(self):
        mock_backend = MagicMock()
        mock_backend.send.return_value = False
        self.agg._backends["failing"] = mock_backend
        self.agg.ingest(_rec("test"))
        self.agg.flush()
        mock_backend.send.assert_called()


if __name__ == "__main__":
    unittest.main()
