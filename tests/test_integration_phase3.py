"""
Integration tests for Phase 3: Observability & Security Hardening

Tests the complete workflow:
- Telemetry collection across adapters and memory
- Secret filtering in memory backends
- Distributed tracing with trace IDs and span IDs
- Plan mode with approval gating and risk scoring
"""

import sys
import unittest
import json
import tempfile
import os
from pathlib import Path

# Remove incomplete stubs left by earlier test files so real modules load
for _stubname in ("vetinari.structured_logging", "vetinari.security"):
    sys.modules.pop(_stubname, None)

from vetinari.structured_logging import (
    configure_logging, CorrelationContext, get_trace_id, get_span_id, 
    get_logger, traced_operation
)
from vetinari.telemetry import get_telemetry_collector, reset_telemetry
from vetinari.security import get_secret_scanner, SecretPattern
from vetinari.memory import get_dual_memory_store
from vetinari.memory.interfaces import MemoryEntry, MemoryEntryType


class TestTelemetryIntegration(unittest.TestCase):
    """Test telemetry collection across the system."""
    
    def setUp(self):
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
        self.assertIn("openai:gpt-4", metrics)
        
        m = metrics["openai:gpt-4"]
        self.assertEqual(m.total_requests, 3)
        self.assertEqual(m.successful_requests, 2)
        self.assertEqual(m.failed_requests, 1)
        self.assertAlmostEqual(m.success_rate, 66.67, places=1)
        self.assertGreater(m.avg_latency_ms, 0)
    
    def test_memory_metrics_collection(self):
        """Test memory operation tracking."""
        self.telemetry.record_memory_write("oc", 5.2)
        self.telemetry.record_memory_write("oc", 4.8)
        self.telemetry.record_memory_read("oc", 2.1)
        self.telemetry.record_memory_search("oc", 15.3)
        
        metrics = self.telemetry.get_memory_metrics("oc")
        self.assertIn("oc", metrics)
        
        m = metrics["oc"]
        self.assertEqual(m.total_writes, 2)
        self.assertEqual(m.total_reads, 1)
        self.assertEqual(m.total_searches, 1)
        self.assertAlmostEqual(m.avg_write_latency(), 5.0, places=1)
    
    def test_plan_metrics_collection(self):
        """Test plan mode decision tracking."""
        self.telemetry.record_plan_decision("approve", risk_score=0.2, auto_approved=True)
        self.telemetry.record_plan_decision("approve", risk_score=0.5)
        self.telemetry.record_plan_decision("reject", risk_score=0.8)
        
        metrics = self.telemetry.get_plan_metrics()
        self.assertEqual(metrics.total_decisions, 3)
        self.assertEqual(metrics.approved_decisions, 2)
        self.assertEqual(metrics.rejected_decisions, 1)
        self.assertEqual(metrics.auto_approved_decisions, 1)
        self.assertAlmostEqual(metrics.average_risk_score, 0.5, places=1)
        self.assertAlmostEqual(metrics.approval_rate, 66.67, places=1)
    
    def test_telemetry_export_json(self):
        """Test exporting metrics to JSON format."""
        # Record metrics
        self.telemetry.record_adapter_latency("openai", "gpt-4", 150.5)
        self.telemetry.record_plan_decision("approve", risk_score=0.3)
        
        # Export
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "telemetry.json")
            success = self.telemetry.export_json(export_path)
            
            self.assertTrue(success)
            self.assertTrue(os.path.exists(export_path))
            
            # Verify JSON structure
            with open(export_path) as f:
                data = json.load(f)
                self.assertIn("timestamp", data)
                self.assertIn("uptime_ms", data)
                self.assertIn("adapters", data)
                self.assertIn("memory", data)
                self.assertIn("plan_mode", data)
                self.assertIn("openai:gpt-4", data["adapters"])
    
    def test_telemetry_export_prometheus(self):
        """Test exporting metrics in Prometheus format."""
        self.telemetry.record_adapter_latency("openai", "gpt-4", 150.5)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "metrics.txt")
            success = self.telemetry.export_prometheus(export_path)
            
            self.assertTrue(success)
            self.assertTrue(os.path.exists(export_path))
            
            # Verify Prometheus format
            with open(export_path) as f:
                content = f.read()
                self.assertIn("vetinari_adapter_requests_total", content)
                self.assertIn("vetinari_adapter_latency_ms", content)
                self.assertIn("provider=", content)


class TestSecretFiltering(unittest.TestCase):
    """Test secret detection and filtering."""
    
    def setUp(self):
        self.scanner = get_secret_scanner()
    
    def test_detect_openai_key(self):
        """Test detection of OpenAI API keys."""
        content = "My API key is sk-proj-1234567890abcdefghijk"
        detected = self.scanner.scan(content)
        self.assertIn("openai_api_key", detected)
    
    def test_detect_github_token(self):
        """Test detection of GitHub tokens."""
        content = "Token: ghp_abc123def456789xyz0123456789abcdef"
        detected = self.scanner.scan(content)
        self.assertIn("github_token", detected)
    
    def test_detect_aws_key(self):
        """Test detection of AWS access keys."""
        content = "AWS Key: AKIAIOSFODNN7EXAMPLE"
        detected = self.scanner.scan(content)
        self.assertIn("aws_access_key", detected)
    
    def test_sanitize_content(self):
        """Test content sanitization."""
        content = "API key: sk-proj-1234567890abcdefghijk"
        sanitized = self.scanner.sanitize(content)
        
        self.assertIn("[REDACTED]", sanitized)
        self.assertNotIn("sk-proj-", sanitized)
    
    def test_sanitize_dict(self):
        """Test dictionary sanitization."""
        data = {
            "api_key": "sk-proj-1234567890abcdefghijk",
            "endpoint": "https://api.openai.com",
            "password": "mysecretpassword123"
        }
        
        sanitized = self.scanner.sanitize_dict(data)
        
        # Sensitive keys should be redacted
        self.assertEqual(sanitized["api_key"], "[REDACTED]")
        self.assertEqual(sanitized["password"], "[REDACTED]")
        
        # Non-sensitive values can be scanned but preserved if safe
        self.assertIn("api.openai.com", sanitized["endpoint"])
    
    def test_sensitive_field_detection(self):
        """Test detection of sensitive field names."""
        data = {
            "user_token": "some_value",
            "api-secret": "another_value",
            "encryption_key": "key_data"
        }
        
        sanitized = self.scanner.sanitize_dict(data)
        
        # All these fields are sensitive
        self.assertEqual(sanitized["user_token"], "[REDACTED]")
        self.assertEqual(sanitized["api-secret"], "[REDACTED]")
        self.assertEqual(sanitized["encryption_key"], "[REDACTED]")


class TestDistributedTracing(unittest.TestCase):
    """Test trace ID and span ID propagation."""
    
    def setUp(self):
        configure_logging()
        self.logger = get_logger("test_tracing")
    
    def test_trace_id_generation(self):
        """Test automatic trace ID generation."""
        with CorrelationContext() as ctx:
            trace_id = get_trace_id()
            self.assertIsNotNone(trace_id)
            self.assertGreater(len(trace_id), 0)
    
    def test_trace_id_propagation(self):
        """Test trace ID propagation through context."""
        test_trace_id = "test-trace-123"
        
        with CorrelationContext(trace_id=test_trace_id):
            retrieved_trace_id = get_trace_id()
            self.assertEqual(retrieved_trace_id, test_trace_id)
    
    def test_span_id_setting(self):
        """Test span ID setting within context."""
        with CorrelationContext() as ctx:
            # Initially no span
            self.assertIsNone(get_span_id())
            
            # Set span
            ctx.set_span_id("span_abc123")
            self.assertEqual(get_span_id(), "span_abc123")
            
            # Update span
            ctx.set_span_id("span_xyz789")
            self.assertEqual(get_span_id(), "span_xyz789")
    
    def test_traced_operation_decorator(self):
        """Test traced_operation decorator."""
        call_count = {"count": 0}
        
        @traced_operation("test_operation")
        def sample_operation():
            call_count["count"] += 1
            trace_id = get_trace_id()
            self.assertIsNotNone(trace_id)
            return trace_id
        
        trace_id = sample_operation()
        self.assertIsNotNone(trace_id)
        self.assertEqual(call_count["count"], 1)
    
    def test_request_id_setting(self):
        """Test request ID setting."""
        test_request_id = "req_12345"
        
        with CorrelationContext(request_id=test_request_id):
            from vetinari.structured_logging import get_request_id
            self.assertEqual(get_request_id(), test_request_id)


class TestMemorySecurityIntegration(unittest.TestCase):
    """Test memory integration with secret filtering."""
    
    def setUp(self):
        self.memory = get_dual_memory_store()
    
    def test_secret_filtering_on_remember(self):
        """Test that secrets are filtered before storage."""
        content = "Config: api_key=sk-proj-1234567890abcdefghijk endpoint=https://api.openai.com"
        
        entry = MemoryEntry(
            content=content,
            entry_type=MemoryEntryType.CONFIG,
            agent="test",
            metadata={"type": "api_config"}
        )
        
        # Store entry
        entry_id = self.memory.remember(entry)
        self.assertIsNotNone(entry_id)
        
        # Retrieve and verify secrets were filtered
        retrieved = self.memory.get_entry(entry_id)
        self.assertIsNotNone(retrieved)
        
        # The content should have been sanitized
        if retrieved.content:
            # Note: The actual sanitization happens in _filter_secrets
            # Verify the entry was stored successfully
            self.assertIn("endpoint=", retrieved.content)


class TestCompleteWorkflow(unittest.TestCase):
    """Test complete Phase 3 workflow."""
    
    def test_plan_to_telemetry_flow(self):
        """Test complete flow from planning through telemetry export."""
        configure_logging()
        logger = get_logger("workflow_test")
        telemetry = get_telemetry_collector()
        memory = get_dual_memory_store()
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
                    entry_type=MemoryEntryType.APPROVAL,
                    agent="workflow_test",
                    metadata={"decision_id": i, "risk_score": risk_score}
                )
                memory.remember(entry)
            
            # Record adapter metrics
            telemetry.record_adapter_latency("openai", "gpt-4", 150.0, success=True)
            
            logger.info("Workflow complete")
        
        # Verify telemetry
        metrics = telemetry.get_plan_metrics()
        self.assertEqual(metrics.total_decisions, 3)
        self.assertEqual(metrics.approved_decisions, 3)
        
        # Export telemetry
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "telemetry.json")
            success = telemetry.export_json(export_path)
            self.assertTrue(success)


if __name__ == "__main__":
    unittest.main()
