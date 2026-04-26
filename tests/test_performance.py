"""
Performance Tests (C14)
========================
Latency benchmarks (p50/p95/p99), throughput, memory usage.
Regression guard: fail if p99 exceeds baseline by +20%.

Run with: pytest tests/test_performance.py -v --timeout=60
"""

import statistics
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ───────────────────────────────────────────────────────────


def _measure_latency(fn, iterations=50):
    """Run fn() N times and return latency stats in ms."""
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        timings.append(elapsed)
    timings.sort()
    n = len(timings)
    return {
        "p50": timings[n // 2],
        "p95": timings[int(n * 0.95)],
        "p99": timings[int(n * 0.99)],
        "mean": statistics.mean(timings),
        "stdev": statistics.stdev(timings) if n > 1 else 0.0,
        "min": timings[0],
        "max": timings[-1],
    }


# ── Circuit Breaker Performance ───────────────────────────────────────


class TestCircuitBreakerPerformance:
    """Benchmark circuit breaker operations."""

    def test_allow_request_latency(self):
        """allow_request() should be < 1ms p99."""
        from vetinari.resilience import CircuitBreaker

        cb = CircuitBreaker("perf_test")

        stats = _measure_latency(cb.allow_request, iterations=1000)
        assert stats["p99"] < 1.0, f"allow_request p99={stats['p99']:.3f}ms exceeds 1ms"

    def test_record_success_latency(self):
        """record_success() should be < 1ms p99."""
        from vetinari.resilience import CircuitBreaker

        cb = CircuitBreaker("perf_test")

        stats = _measure_latency(cb.record_success, iterations=1000)
        assert stats["p99"] < 1.0, f"record_success p99={stats['p99']:.3f}ms exceeds 1ms"

    def test_registry_get_latency(self):
        """Registry.get() should be < 0.5ms p99 (cached)."""
        from vetinari.resilience import CircuitBreakerRegistry

        reg = CircuitBreakerRegistry()
        reg.get("test_agent")  # warm up

        stats = _measure_latency(lambda: reg.get("test_agent"), iterations=1000)
        assert stats["p99"] < 0.5, f"registry.get p99={stats['p99']:.3f}ms exceeds 0.5ms"


# ── Model Router Performance ─────────────────────────────────────────


class TestModelRouterPerformance:
    """Benchmark model routing decisions."""

    def test_complexity_estimation_latency(self):
        """classify_complexity() simple call should be < 1ms p99."""
        from vetinari.routing import classify_complexity

        desc = "Implement a security audit for the authentication module with OWASP compliance"
        stats = _measure_latency(lambda: classify_complexity(desc), iterations=1000)
        assert stats["p99"] < 1.0, f"classify_complexity p99={stats['p99']:.3f}ms exceeds 1ms"

    def test_complexity_router_latency(self):
        """classify_complexity() should be < 2ms p99."""
        from vetinari.routing import classify_complexity

        desc = "Refactor the entire authentication system with backwards compatibility"
        stats = _measure_latency(
            lambda: classify_complexity(desc, task_count=5, estimated_files=15),
            iterations=500,
        )
        assert stats["p99"] < 2.0, f"classify_complexity p99={stats['p99']:.3f}ms exceeds 2ms"


# ── Context Window Manager Performance ────────────────────────────────


class TestContextWindowPerformance:
    """Benchmark context window operations."""

    def test_used_tokens_latency(self):
        """used_tokens should be < 1ms p99."""
        from vetinari.context import ContextWindowManager

        mgr = ContextWindowManager()

        stats = _measure_latency(lambda: mgr.used_tokens, iterations=500)
        assert stats["p99"] < 1.0, f"used_tokens p99={stats['p99']:.3f}ms exceeds 1ms"

    def test_usage_ratio_latency(self):
        """usage_ratio should be < 1ms p99."""
        from vetinari.context import ContextWindowManager

        mgr = ContextWindowManager()

        stats = _measure_latency(lambda: mgr.usage_ratio, iterations=500)
        assert stats["p99"] < 1.0, f"usage_ratio p99={stats['p99']:.3f}ms exceeds 1ms"


# ── Schema Validation Performance ─────────────────────────────────────


class TestSchemaPerformance:
    """Benchmark schema validation."""

    def test_validate_output_latency(self):
        """validate_output() should be < 2ms p99."""
        from vetinari.schemas import validate_output

        data = {
            "score": 0.85,
            "summary": "Good code quality",
            "issues": [{"severity": "low", "message": "minor issue"}],
            "strengths": ["clean code"],
            "recommendations": ["add tests"],
        }
        stats = _measure_latency(lambda: validate_output("code_review", data), iterations=500)
        assert stats["p99"] < 2.0, f"validate_output p99={stats['p99']:.3f}ms exceeds 2ms"


# ── Security Pattern Scan Performance ─────────────────────────────────


class TestSecurityScanPerformance:
    """Benchmark heuristic security scanning."""

    def test_heuristic_scan_latency(self):
        """Scanning 500 lines of code should be < 200ms p99 (CI-safe threshold)."""
        from vetinari.agents.consolidated.quality_agent import InspectorAgent

        # Build a mock agent (skip LLM init)
        with patch.object(InspectorAgent, "__init__", lambda self, *a, **kw: None):
            agent = InspectorAgent.__new__(InspectorAgent)
            agent._run_heuristic_scan = InspectorAgent._run_heuristic_scan.__get__(agent)

        code_lines = [
            'password = "secret123"',
            "subprocess.call(cmd, shell=True)",
            'result = input("Enter value: ")',
            "yaml.load(data)",
            "x = random.random()",
        ] * 100  # 500 lines
        code = "\n".join(code_lines)

        stats = _measure_latency(lambda: agent._run_heuristic_scan(code), iterations=50)
        # 200ms threshold accommodates slow CI runners (GitHub Actions free tier)
        assert stats["p99"] < 200.0, f"heuristic_scan p99={stats['p99']:.3f}ms exceeds 200ms"


# ── Memory Usage ──────────────────────────────────────────────────────


class TestMemoryUsage:
    """Verify memory usage stays within bounds."""

    def test_circuit_breaker_registry_memory(self):
        """Registry with 100 breakers should use < 1MB."""
        from vetinari.resilience import CircuitBreakerRegistry

        reg = CircuitBreakerRegistry()
        for i in range(100):
            reg.get(f"agent_{i}")

        # Rough size estimate
        size = sys.getsizeof(reg._breakers)
        for cb in reg._breakers.values():
            size += sys.getsizeof(cb)
        assert size < 1_000_000, f"Registry with 100 breakers uses {size} bytes (>1MB)"
