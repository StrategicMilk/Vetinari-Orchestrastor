"""Tests for SSE (Server-Sent Events) log streaming.

Covers:
    - SSEBackend creation and configuration
    - Client add / remove lifecycle
    - Fan-out of LogRecords to connected clients
    - Ring-buffer management (size cap, replay on connect)
    - Max-clients limit enforcement
    - Dead-client cleanup (full queues)
    - LogRecord JSON serialisation
    - Singleton helpers (get_sse_backend / reset_sse_backend)
    - Flask SSE blueprint endpoints (/api/logs/stream, /api/logs/recent)
"""

import json
import queue
import threading
import time

import pytest

from vetinari.dashboard.log_aggregator import (
    LogRecord,
    SSEBackend,
    get_sse_backend,
    reset_sse_backend,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def backend():
    """Return a fresh SSEBackend for each test."""
    return SSEBackend(max_clients=5, buffer_size=10)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure the global SSEBackend singleton is clean for every test."""
    reset_sse_backend()
    yield
    reset_sse_backend()


def _make_record(msg: str = "hello", level: str = "INFO", source: str = "test") -> LogRecord:
    """Helper to build a minimal LogRecord."""
    return LogRecord(
        message=msg,
        level=level,
        timestamp=time.time(),
        logger_name=source,
    )


# ---------------------------------------------------------------------------
# SSEBackend — basic creation
# ---------------------------------------------------------------------------

class TestSSEBackendCreation:
    def test_default_construction(self):
        b = SSEBackend()
        assert b.client_count == 0
        assert b.get_buffer() == []
        assert b.name == "sse"

    def test_custom_limits(self):
        b = SSEBackend(max_clients=3, buffer_size=5)
        assert b._max_clients == 3
        assert b._buffer_size == 5


# ---------------------------------------------------------------------------
# Client lifecycle
# ---------------------------------------------------------------------------

class TestClientLifecycle:
    def test_add_and_remove_client(self, backend):
        q = queue.Queue()
        backend.add_client(q)
        assert backend.client_count == 1
        backend.remove_client(q)
        assert backend.client_count == 0

    def test_remove_unknown_client_is_noop(self, backend):
        q = queue.Queue()
        backend.remove_client(q)  # should not raise
        assert backend.client_count == 0

    def test_add_multiple_clients(self, backend):
        queues = [queue.Queue() for _ in range(5)]
        for q in queues:
            backend.add_client(q)
        assert backend.client_count == 5

    def test_max_clients_enforced(self, backend):
        # backend fixture has max_clients=5
        for _ in range(5):
            backend.add_client(queue.Queue())
        with pytest.raises(RuntimeError, match="Max SSE clients"):
            backend.add_client(queue.Queue())


# ---------------------------------------------------------------------------
# Sending records / fan-out
# ---------------------------------------------------------------------------

class TestSendRecords:
    def test_single_record_received_by_client(self, backend):
        q = queue.Queue()
        backend.add_client(q)

        rec = _make_record("test message")
        backend.send([rec])

        data = q.get_nowait()
        parsed = json.loads(data)
        assert parsed["message"] == "test message"
        assert parsed["level"] == "INFO"

    def test_multiple_records_fan_out(self, backend):
        q1 = queue.Queue()
        q2 = queue.Queue()
        backend.add_client(q1)
        backend.add_client(q2)

        records = [_make_record(f"msg-{i}") for i in range(3)]
        backend.send(records)

        for q in (q1, q2):
            for i in range(3):
                data = json.loads(q.get_nowait())
                assert data["message"] == f"msg-{i}"

    def test_send_returns_true(self, backend):
        assert backend.send([_make_record()]) is True

    def test_send_empty_list(self, backend):
        q = queue.Queue()
        backend.add_client(q)
        assert backend.send([]) is True
        assert q.empty()


# ---------------------------------------------------------------------------
# Buffer management
# ---------------------------------------------------------------------------

class TestBufferManagement:
    def test_buffer_stores_records(self, backend):
        backend.send([_make_record(f"m-{i}") for i in range(3)])
        buf = backend.get_buffer()
        assert len(buf) == 3
        assert json.loads(buf[0])["message"] == "m-0"

    def test_buffer_capped_at_buffer_size(self, backend):
        # buffer_size=10 in fixture
        backend.send([_make_record(f"m-{i}") for i in range(15)])
        buf = backend.get_buffer()
        assert len(buf) == 10
        # Should keep the most recent 10
        assert json.loads(buf[0])["message"] == "m-5"
        assert json.loads(buf[-1])["message"] == "m-14"

    def test_new_client_receives_buffered_records(self, backend):
        backend.send([_make_record(f"old-{i}") for i in range(3)])

        q = queue.Queue()
        backend.add_client(q)

        # The queue should already have the 3 buffered records
        received = []
        while not q.empty():
            received.append(json.loads(q.get_nowait()))
        assert len(received) == 3
        assert received[0]["message"] == "old-0"

    def test_get_buffer_returns_copy(self, backend):
        backend.send([_make_record("x")])
        buf = backend.get_buffer()
        buf.clear()
        assert len(backend.get_buffer()) == 1  # original unchanged


# ---------------------------------------------------------------------------
# Dead-client cleanup
# ---------------------------------------------------------------------------

class TestDeadClientCleanup:
    def test_full_queue_client_removed(self, backend):
        small_q = queue.Queue(maxsize=1)
        backend.add_client(small_q)

        # Fill the queue first
        backend.send([_make_record("fill")])
        assert small_q.full()

        # Next send should detect the full queue and remove the client
        backend.send([_make_record("overflow")])
        assert backend.client_count == 0

    def test_healthy_clients_survive_cleanup(self, backend):
        healthy_q = queue.Queue(maxsize=100)
        tiny_q = queue.Queue(maxsize=1)
        backend.add_client(healthy_q)
        backend.add_client(tiny_q)

        # Fill tiny_q
        backend.send([_make_record("a")])
        # Now tiny_q is full; next send drops it
        backend.send([_make_record("b")])

        assert backend.client_count == 1
        # Healthy client got both records
        items = []
        while not healthy_q.empty():
            items.append(healthy_q.get_nowait())
        assert len(items) == 2


# ---------------------------------------------------------------------------
# LogRecord serialisation
# ---------------------------------------------------------------------------

class TestLogRecordSerialisation:
    def test_to_json_round_trip(self):
        rec = LogRecord(
            message="something happened",
            level="WARNING",
            timestamp=1234567890.0,
            trace_id="trace-1",
            span_id="span-2",
            request_id="req-3",
            logger_name="mylogger",
            extra={"key": "value"},
        )
        data = json.loads(rec.to_json())
        assert data["message"] == "something happened"
        assert data["level"] == "WARNING"
        assert data["timestamp"] == 1234567890.0
        assert data["trace_id"] == "trace-1"
        assert data["span_id"] == "span-2"
        assert data["request_id"] == "req-3"
        assert data["key"] == "value"

    def test_to_dict_includes_extras(self):
        rec = LogRecord(message="hi", extra={"foo": 42})
        d = rec.to_dict()
        assert d["foo"] == 42
        assert d["message"] == "hi"


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_sse_backend_returns_same_instance(self):
        a = get_sse_backend()
        b = get_sse_backend()
        assert a is b

    def test_reset_creates_new_instance(self):
        a = get_sse_backend()
        reset_sse_backend()
        b = get_sse_backend()
        assert a is not b

    def test_singleton_is_an_sse_backend(self):
        assert isinstance(get_sse_backend(), SSEBackend)


# ---------------------------------------------------------------------------
# Thread safety (smoke test)
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_send_and_subscribe(self, backend):
        """Multiple threads sending and subscribing should not crash."""
        errors = []
        q = queue.Queue(maxsize=5000)
        backend.add_client(q)

        def sender():
            try:
                for i in range(100):
                    backend.send([_make_record(f"t-{i}")])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=sender) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        # All 400 records should have been queued (no drops for a big queue)
        count = 0
        while not q.empty():
            q.get_nowait()
            count += 1
        assert count == 400


# ---------------------------------------------------------------------------
# Flask blueprint integration
# ---------------------------------------------------------------------------

class TestFlaskBlueprint:
    """Integration tests using Flask test client."""

    @pytest.fixture()
    def app(self):
        from flask import Flask
        from vetinari.web.log_stream import log_stream_bp

        app = Flask(__name__)
        app.register_blueprint(log_stream_bp)
        app.config["TESTING"] = True
        return app

    @pytest.fixture()
    def client(self, app):
        return app.test_client()

    def test_recent_logs_empty(self, client):
        resp = client.get("/api/logs/recent")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "logs" in data
        assert isinstance(data["logs"], list)

    def test_recent_logs_with_data(self, client):
        backend = get_sse_backend()
        backend.send([_make_record("test-recent")])

        resp = client.get("/api/logs/recent")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["logs"]) >= 1
        # The log entry is a JSON string in the array
        parsed = json.loads(data["logs"][-1])
        assert parsed["message"] == "test-recent"

    def test_stream_endpoint_content_type(self, client):
        """The stream endpoint should return text/event-stream."""
        backend = get_sse_backend()
        # Pre-load a record so the generator yields immediately
        backend.send([_make_record("stream-test")])

        resp = client.get("/api/logs/stream")
        assert resp.content_type.startswith("text/event-stream")

    def test_stream_endpoint_headers(self, client):
        backend = get_sse_backend()
        # Pre-load a record so the generator yields immediately
        backend.send([_make_record("header-test")])

        resp = client.get("/api/logs/stream")
        assert resp.headers.get("Cache-Control") == "no-cache"
        assert resp.headers.get("X-Accel-Buffering") == "no"
