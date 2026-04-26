"""Tests for backend auto-detection in the system health monitor."""

from __future__ import annotations

from unittest.mock import Mock, patch

from vetinari.system import health_monitor


class TestBackendHealthMonitor:
    """Test endpoint-driven backend discovery and registration."""

    def setup_method(self) -> None:
        """Reset backend detection cache between tests."""
        health_monitor._cached_backends = []
        health_monitor._last_backend_check = 0.0

    def test_check_backends_uses_configured_vllm_endpoint(self) -> None:
        """Configured vLLM endpoints should be treated as backend availability."""
        runtime_cfg = {
            "inference_backend": {
                "primary": "vllm",
                "fallback": "llama_cpp",
                "vllm": {"enabled": True, "endpoint": "http://localhost:8000"},
                "nim": {"enabled": False, "endpoint": "http://localhost:8001"},
            }
        }

        manager = Mock()
        manager.list_providers.return_value = {"local": Mock()}
        manager.get_provider.return_value = None

        response = Mock()
        response.status_code = 200

        with (
            patch("vetinari.backend_config.load_backend_runtime_config", return_value=runtime_cfg),
            patch("vetinari.backend_config.resolve_provider_fallback_order", return_value=["vllm", "local"]),
            patch("vetinari.adapter_manager.get_adapter_manager", return_value=manager),
            patch("httpx.get", return_value=response),
            patch("time.monotonic", return_value=1000.0),
        ):
            component = health_monitor._check_backends()

        assert component.healthy is True
        assert "vllm" in component.detail
        manager.register_provider.assert_called()
        manager.set_fallback_order.assert_called_once_with(["vllm", "local"])
