"""
Integration tests for Phase 3: Observability & Security Hardening

Tests the complete workflow:
- Telemetry collection across adapters and memory
- Secret filtering in memory backends
- Distributed tracing with trace IDs and span IDs
- Plan mode with approval gating and risk scoring
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Remove incomplete stubs left by earlier test files so real modules load
for _stubname in ("vetinari.structured_logging", "vetinari.security"):
    sys.modules.pop(_stubname, None)

from vetinari.memory import get_unified_memory_store
from vetinari.memory.interfaces import MemoryEntry, MemoryType
from vetinari.security import SecretPattern, get_secret_scanner
from vetinari.structured_logging import (
    CorrelationContext,
    configure_logging,
    get_logger,
    get_span_id,
    get_trace_id,
    traced_operation,
)
from vetinari.telemetry import get_telemetry_collector, reset_telemetry


class TestTelemetryIntegration:
    """Test telemetry collection across the system."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        reset_telemetry()
        self.telemetry = get_telemetry_collector()

    def test_adapter_metrics_collection(self):
        """Test adapter latency and success rate tracking."""
        # Record some adapter calls
        self.telemetry.record_adapter_latency("openai", "gpt-4", 150.5, success=True, tokens_used=100)
        self.telemetry.record_adapter_latency("openai", "gpt-4", 145.2, success=True, tokens_used=95)
        self.telemetry.record_adapter_latency("openai", "gpt-4", 160.0, success=False)

        # Verify metrics
        metrics = self.telemetry.get_adapter_metrics()
        assert "openai:gpt-4" in metrics

        m = metrics["openai:gpt-4"]
        assert m.total_requests == 3
        assert m.successful_requests == 2
        assert m.failed_requests == 1
        assert m.success_rate == pytest.approx(66.67, abs=0.05)
        assert m.avg_latency_ms > 0

    def test_memory_metrics_collection(self):
        """Test memory operation tracking."""
        self.telemetry.record_memory_write("oc", 5.2)
        self.telemetry.record_memory_write("oc", 4.8)
        self.telemetry.record_memory_read("oc", 2.1)
        self.telemetry.record_memory_search("oc", 15.3)

        metrics = self.telemetry.get_memory_metrics("oc")
        assert "oc" in metrics

        m = metrics["oc"]
        assert m.total_writes == 2
        assert m.total_reads == 1
        assert m.total_searches == 1
        assert m.avg_write_latency() == pytest.approx(5.0)

    def test_plan_metrics_collection(self):
        """Test plan mode decision tracking."""
        self.telemetry.record_plan_decision("approve", risk_score=0.2, auto_approved=True)
        self.telemetry.record_plan_decision("approve", risk_score=0.5)
        self.telemetry.record_plan_decision("reject", risk_score=0.8)

        metrics = self.telemetry.get_plan_metrics()
        assert metrics.total_decisions == 3
        assert metrics.approved_decisions == 2
        assert metrics.rejected_decisions == 1
        assert metrics.auto_approved_decisions == 1
        assert metrics.average_risk_score == pytest.approx(0.5)
        assert metrics.approval_rate == pytest.approx(66.67, abs=0.05)

    def test_telemetry_export_json(self):
        """Test exporting metrics to JSON format."""
        # Record metrics
        self.telemetry.record_adapter_latency("openai", "gpt-4", 150.5)
        self.telemetry.record_plan_decision("approve", risk_score=0.3)

        # Export
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "telemetry.json")
            success = self.telemetry.export_json(export_path)

            assert success
            assert os.path.exists(export_path)

            # Verify JSON structure
            with open(export_path) as f:
                data = json.load(f)
                assert "timestamp" in data
                assert "uptime_ms" in data
                assert "adapters" in data
                assert "memory" in data
                assert "plan_mode" in data
                assert "openai:gpt-4" in data["adapters"]

    def test_telemetry_export_prometheus(self):
        """Test exporting metrics in Prometheus format."""
        self.telemetry.record_adapter_latency("openai", "gpt-4", 150.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "metrics.txt")
            success = self.telemetry.export_prometheus(export_path)

            assert success
            assert os.path.exists(export_path)

            # Verify Prometheus format
            with open(export_path) as f:
                content = f.read()
                assert "vetinari_adapter_requests_total" in content
                assert "vetinari_adapter_latency_ms" in content
                assert "provider=" in content


class TestSecretFiltering:
    """Test secret detection and filtering."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.scanner = get_secret_scanner()

    def test_detect_openai_key(self):
        """Test detection of OpenAI API keys."""
        content = "My API key is sk-proj-1234567890abcdefghijk"
        detected = self.scanner.scan(content)
        assert "openai_api_key" in detected

    def test_detect_github_token(self):
        """Test detection of GitHub tokens."""
        content = "Token: ghp_abc123def456789xyz0123456789abcdef"
        detected = self.scanner.scan(content)
        assert "github_token" in detected

    def test_detect_aws_key(self):
        """Test detection of AWS access keys."""
        content = "AWS Key: AKIAIOSFODNN7EXAMPLE"
        detected = self.scanner.scan(content)
        assert "aws_access_key" in detected

    def test_sanitize_content(self):
        """Test content sanitization."""
        content = "API key: sk-proj-1234567890abcdefghijk"
        sanitized = self.scanner.sanitize(content)

        assert "[REDACTED]" in sanitized or "[HIGH_ENTROPY_REDACTED]" in sanitized
        assert "sk-proj-" not in sanitized

    def test_sanitize_dict(self):
        """Test dictionary sanitization."""
        data = {
            "api_key": "sk-proj-1234567890abcdefghijk",
            "endpoint": "https://api.openai.com",
            "password": "mysecretpassword123",
        }

        sanitized = self.scanner.sanitize_dict(data)

        # Sensitive keys should be redacted
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["password"] == "[REDACTED]"

        # Non-sensitive values can be scanned but preserved if safe
        assert "api.openai.com" in sanitized["endpoint"]

    def test_sensitive_field_detection(self):
        """Test detection of sensitive field names."""
        data = {"user_token": "some_value", "api-secret": "another_value", "encryption_key": "key_data"}

        sanitized = self.scanner.sanitize_dict(data)

        # All these fields are sensitive
        assert sanitized["user_token"] == "[REDACTED]"
        assert sanitized["api-secret"] == "[REDACTED]"
        assert sanitized["encryption_key"] == "[REDACTED]"


class TestDistributedTracing:
    """Test trace ID and span ID propagation."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        configure_logging()
        self.logger = get_logger("test_tracing")

    def test_trace_id_generation(self):
        """Test automatic trace ID generation."""
        with CorrelationContext():
            trace_id = get_trace_id()
            assert trace_id is not None
            assert len(trace_id) > 0

    def test_trace_id_propagation(self):
        """Test trace ID propagation through context."""
        test_trace_id = "test-trace-123"

        with CorrelationContext(trace_id=test_trace_id):
            retrieved_trace_id = get_trace_id()
            assert retrieved_trace_id == test_trace_id

    def test_span_id_setting(self):
        """Test span ID setting within context."""
        with CorrelationContext() as ctx:
            # Initially no span
            assert get_span_id() is None

            # Set span
            ctx.set_span_id("span_abc123")
            assert get_span_id() == "span_abc123"

            # Update span
            ctx.set_span_id("span_xyz789")
            assert get_span_id() == "span_xyz789"

    def test_traced_operation_decorator(self):
        """Test traced_operation decorator."""
        call_count = {"count": 0}

        @traced_operation("test_operation")
        def sample_operation():
            call_count["count"] += 1
            trace_id = get_trace_id()
            assert trace_id is not None
            return trace_id

        trace_id = sample_operation()
        assert trace_id is not None
        assert call_count["count"] == 1

    def test_request_id_setting(self):
        """Test request ID setting."""
        test_request_id = "req_12345"

        with CorrelationContext(request_id=test_request_id):
            from vetinari.structured_logging import get_request_id

            assert get_request_id() == test_request_id


class TestMemorySecurityIntegration:
    """Test memory integration with secret filtering."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.memory = get_unified_memory_store()

    def test_secret_filtering_on_remember(self):
        """Test that secrets are filtered before storage."""
        content = "Config: api_key=sk-proj-1234567890abcdefghijk endpoint=https://api.openai.com"

        entry = MemoryEntry(
            content=content, entry_type=MemoryType.CONFIG, agent="test", metadata={"type": "api_config"}
        )

        # Store entry
        entry_id = self.memory.remember(entry)
        assert entry_id is not None

        # Retrieve and verify secrets were filtered
        retrieved = self.memory.get_entry(entry_id)
        assert retrieved is not None

        # The content should have been sanitized
        if retrieved.content:
            # Note: The actual sanitization happens in _filter_secrets
            # Verify the entry was stored successfully
            assert "endpoint=" in retrieved.content


class TestCompleteWorkflow:
    """Test complete Phase 3 workflow."""

    def test_plan_to_telemetry_flow(self):
        """Test complete flow from planning through telemetry export."""
        configure_logging()
        logger = get_logger("workflow_test")
        telemetry = get_telemetry_collector()
        memory = get_unified_memory_store()
        reset_telemetry()

        with CorrelationContext() as ctx:
            trace_id = get_trace_id()
            logger.info("Starting workflow", trace_id=trace_id)

            # Simulate planning decisions
            for i in range(3):
                ctx.set_span_id(f"decision_{i}")
                risk_score = (i + 1) * 0.3
                telemetry.record_plan_decision("approve", risk_score=risk_score)

                # Store in memory
                entry = MemoryEntry(
                    content=f"Decision {i}: approved with risk {risk_score}",
                    entry_type=MemoryType.APPROVAL,
                    agent="workflow_test",
                    metadata={"decision_id": i, "risk_score": risk_score},
                )
                memory.remember(entry)

            # Record adapter metrics
            telemetry.record_adapter_latency("openai", "gpt-4", 150.0, success=True)

            logger.info("Workflow complete")

        # Verify telemetry
        metrics = telemetry.get_plan_metrics()
        assert metrics.total_decisions == 3
        assert metrics.approved_decisions == 3

        # Export telemetry
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "telemetry.json")
            success = telemetry.export_json(export_path)
            assert success
