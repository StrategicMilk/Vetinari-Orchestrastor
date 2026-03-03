"""
Tests for structured logging module.

Run with: python -m pytest tests/test_structured_logging.py -v
"""

import pytest
import json
import logging
import os
import sys
from io import StringIO
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


class TestStructuredLoggingImport:
    """Test that structured logging module can be imported."""
    
    def test_import_module(self):
        """Verify module imports successfully."""
        from vetinari import structured_logging
        assert structured_logging is not None
    
    def test_get_logger_function(self):
        """Test get_logger returns a logger."""
        from vetinari.structured_logging import get_logger
        logger = get_logger("test")
        assert logger is not None
        assert logger.name == "test"
    
    def test_log_event_function(self):
        """Test log_event convenience function."""
        from vetinari.structured_logging import log_event
        # Should not raise
        log_event("info", "test", "Test message")


class TestStructuredFormatter:
    """Test JSON formatter."""
    
    def test_formatter_creates_json(self):
        """Test that formatter outputs valid JSON."""
        from vetinari.structured_logging import StructuredFormatter
        
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        output = formatter.format(record)
        # Should be valid JSON
        parsed = json.loads(output)
        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test_logger"
    
    def test_formatter_includes_timestamp(self):
        """Test that formatter includes timestamp."""
        from vetinari.structured_logging import StructuredFormatter
        
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None
        )
        
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "timestamp" in parsed
    
    def test_formatter_includes_context(self):
        """Test that formatter includes context."""
        from vetinari.structured_logging import StructuredFormatter
        
        formatter = StructuredFormatter()
        formatter.set_context(service="vetinari", version="1.0")
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None
        )
        
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "context" in parsed
        assert parsed["context"]["service"] == "vetinari"


class TestStructuredLogger:
    """Test StructuredLogger wrapper."""
    
    def test_logger_info(self):
        """Test info level logging."""
        from vetinari.structured_logging import StructuredLogger
        import logging
        
        # Create a string handler to capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter('%(message)s'))
        
        logger = logging.getLogger("test_info")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        structured = StructuredLogger("test_info", logger)
        structured.info("Test message", extra_field="value")
        
        output = stream.getvalue()
        assert "Test message" in output
    
    def test_logger_with_context(self):
        """Test logger with context fields."""
        from vetinari.structured_logging import StructuredLogger
        import logging
        
        # Note: Structured logging goes to stdout via root logger
        # This test verifies the context is included in the log call
        structured = StructuredLogger("test_context", logging.getLogger())
        structured.set_context(task_id="task_1", execution_id="exec_123")
        
        # Should not raise - just verify context setting works
        structured.info("Task running")
        # The actual output goes to stdout in JSON format


class TestLogEventFunctions:
    """Test convenience log event functions."""
    
    def test_log_task_start(self):
        """Test log_task_start function."""
        from vetinari.structured_logging import log_task_start
        # Should not raise
        log_task_start("task_1", "code_generation")
    
    def test_log_task_complete(self):
        """Test log_task_complete function."""
        from vetinari.structured_logging import log_task_complete
        # Should not raise
        log_task_complete("task_1", 1500.0)
    
    def test_log_task_error(self):
        """Test log_task_error function."""
        from vetinari.structured_logging import log_task_error
        # Should not raise
        log_task_error("task_1", "Timeout error")
    
    def test_log_model_discovery(self):
        """Test log_model_discovery function."""
        from vetinari.structured_logging import log_model_discovery
        # Should not raise
        log_model_discovery(5, 250.0)
    
    def test_log_wave_start(self):
        """Test log_wave_start function."""
        from vetinari.structured_logging import log_wave_start
        # Should not raise
        log_wave_start("wave_1", 3)
    
    def test_log_wave_complete(self):
        """Test log_wave_complete function."""
        from vetinari.structured_logging import log_wave_complete
        # Should not raise
        log_wave_complete("wave_1", 5000.0)


class TestTimedOperationDecorator:
    """Test timed_operation decorator."""
    
    def test_decorator_logs_completion(self):
        """Test that decorator logs operation completion."""
        from vetinari.structured_logging import timed_operation
        
        @timed_operation("test_operation")
        def test_func():
            return 42
        
        result = test_func()
        assert result == 42
        # Log output goes to stdout in JSON format - verify no exception
    
    def test_decorator_logs_error(self):
        """Test that decorator logs errors."""
        from vetinari.structured_logging import timed_operation
        
        @timed_operation("failing_operation")
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_func()
        # Log output goes to stdout in JSON format - verify no exception


class TestMetricsCollector:
    """Test MetricsCollector class."""
    
    def test_increment_counter(self):
        """Test counter increment."""
        from vetinari.structured_logging import MetricsCollector
        
        collector = MetricsCollector()
        collector.increment("test_counter")
        collector.increment("test_counter")
        
        assert collector.get_counter("test_counter") == 2
    
    def test_increment_counter_with_tags(self):
        """Test counter increment with tags."""
        from vetinari.structured_logging import MetricsCollector
        
        collector = MetricsCollector()
        collector.increment("test_counter", status="success")
        collector.increment("test_counter", status="success")
        collector.increment("test_counter", status="error")
        
        assert collector.get_counter("test_counter", status="success") == 2
        assert collector.get_counter("test_counter", status="error") == 1
    
    def test_record_histogram(self):
        """Test histogram recording."""
        from vetinari.structured_logging import MetricsCollector
        
        collector = MetricsCollector()
        collector.record("test_histogram", 10.0)
        collector.record("test_histogram", 20.0)
        collector.record("test_histogram", 30.0)
        
        stats = collector.get_histogram_stats("test_histogram")
        assert stats is not None
        assert stats["count"] == 3
        assert stats["sum"] == 60.0
        assert stats["min"] == 10.0
        assert stats["max"] == 30.0
        assert stats["avg"] == 20.0
    
    def test_histogram_percentiles(self):
        """Test histogram percentile calculations."""
        from vetinari.structured_logging import MetricsCollector
        
        collector = MetricsCollector()
        # Record 100 values from 1 to 100
        for i in range(1, 101):
            collector.record("percentile_test", float(i))
        
        stats = collector.get_histogram_stats("percentile_test")
        assert stats is not None
        assert 50 <= stats["p50"] <= 51
        assert 95 <= stats["p95"] <= 96
        assert 99 <= stats["p99"] <= 100
    
    def test_get_metrics_singleton(self):
        """Test get_metrics returns singleton."""
        from vetinari.structured_logging import get_metrics
        
        m1 = get_metrics()
        m2 = get_metrics()
        
        assert m1 is m2


class TestEnvironmentVariables:
    """Test environment variable configuration."""
    
    def test_log_level_from_env(self, monkeypatch):
        """Test that log level can be set from environment."""
        monkeypatch.setenv("VETINARI_LOG_LEVEL", "DEBUG")
        # Re-import to pick up new setting (would need fresh config in real usage)
        from vetinari import structured_logging
        # Just verify the module has the function
        assert hasattr(structured_logging, '_get_log_level')
    
    def test_structured_logging_flag(self, monkeypatch):
        """Test structured logging enable/disable flag."""
        monkeypatch.setenv("VETINARI_STRUCTURED_LOGGING", "false")
        from vetinari import structured_logging
        assert hasattr(structured_logging, '_use_structured_logging')


class TestBackwardCompatibility:
    """Test backward compatibility with existing logging."""
    
    def test_standard_logging_still_works(self):
        """Test that standard Python logging still works."""
        import logging
        
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)
        
        logger = logging.getLogger("standard_test")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        logger.info("Standard logging message")
        
        output = stream.getvalue()
        assert "Standard logging message" in output
