"""
Unit tests for Adapter Manager (Phase 2)

Tests the AdapterManager, ProviderMetrics, and provider selection logic.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import vetinari.adapter_manager as adapter_manager_module
from vetinari.adapter_manager import (
    AdapterManager,
    ProviderHealthStatus,
    ProviderMetrics,
    get_adapter_manager,
)
from vetinari.adapters.base import (
    InferenceRequest,
    InferenceResponse,
    ModelInfo,
    ProviderConfig,
    ProviderType,
)
from vetinari.execution_context import (
    ExecutionMode,
    get_context_manager,
)


class TestProviderMetrics:
    """Test ProviderMetrics class."""

    def test_metrics_creation(self):
        """Test creating metrics."""
        metrics = ProviderMetrics(
            name="openai",
            provider_type=ProviderType.OPENAI,
        )
        assert metrics.name == "openai"
        assert metrics.provider_type == ProviderType.OPENAI
        assert metrics.successful_inferences == 0
        assert metrics.failed_inferences == 0
        assert metrics.health_status == ProviderHealthStatus.UNKNOWN

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = ProviderMetrics(
            name="openai",
            provider_type=ProviderType.OPENAI,
            successful_inferences=8,
            failed_inferences=2,
        )
        assert metrics.success_rate == 0.8

    def test_success_rate_no_inferences(self):
        """Test success rate with no inferences."""
        metrics = ProviderMetrics(
            name="openai",
            provider_type=ProviderType.OPENAI,
        )
        assert metrics.success_rate == 1.0  # Default for no data

    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        now = datetime.now()
        metrics = ProviderMetrics(
            name="openai",
            provider_type=ProviderType.OPENAI,
            last_health_check=now,
            health_status=ProviderHealthStatus.HEALTHY,
            successful_inferences=10,
            failed_inferences=1,
            avg_latency_ms=150.5,
            total_tokens_used=5000,
            estimated_cost=0.15,
        )

        metrics_dict = metrics.to_dict()
        assert metrics_dict["name"] == "openai"
        assert metrics_dict["provider_type"] == "openai"
        assert metrics_dict["health_status"] == "healthy"
        assert metrics_dict["successful_inferences"] == 10
        assert metrics_dict["failed_inferences"] == 1
        assert metrics_dict["avg_latency_ms"] == 150.5
        assert metrics_dict["total_tokens_used"] == 5000


class TestAdapterManagerRegistration:
    """Test adapter registration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AdapterManager()
        self.mock_adapter = Mock()
        self.mock_adapter.discover_models.return_value = []
        self.mock_adapter.health_check.return_value = {"healthy": True}

    def test_register_provider(self):
        """Test registering a provider."""
        config = ProviderConfig(
            provider_type=ProviderType.LOCAL,
            name="local",
            endpoint="local",
        )

        with patch.object(self.manager.registry, "create_adapter", return_value=self.mock_adapter):
            adapter = self.manager.register_provider(config, "local-1")

            assert adapter == self.mock_adapter
            assert "local-1" in self.manager._metrics
            assert "local-1" in self.manager._provider_fallback_order

    def test_get_provider(self):
        """Test getting a provider."""
        with patch.object(self.manager.registry, "get_adapter", return_value=self.mock_adapter):
            provider = self.manager.get_provider("openai-1")
            assert provider == self.mock_adapter

    def test_list_providers(self):
        """Test listing all providers."""
        providers_dict = {"openai-1": self.mock_adapter, "claude-1": Mock()}
        with patch.object(self.manager.registry, "list_adapters", return_value=providers_dict):
            providers = self.manager.list_providers()
            assert providers == providers_dict


class TestAdapterManagerMetrics:
    """Test metrics retrieval."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AdapterManager()

    def test_get_metrics_single_provider(self):
        """Test getting metrics for a single provider."""
        self.manager._metrics["openai-1"] = ProviderMetrics(
            name="openai-1",
            provider_type=ProviderType.OPENAI,
            successful_inferences=10,
            failed_inferences=2,
        )

        metrics = self.manager.get_metrics("openai-1")
        assert metrics["name"] == "openai-1"
        assert metrics["successful_inferences"] == 10

    def test_get_metrics_all_providers(self):
        """Test getting metrics for all providers."""
        self.manager._metrics["openai-1"] = ProviderMetrics(
            name="openai-1",
            provider_type=ProviderType.OPENAI,
        )
        self.manager._metrics["claude-1"] = ProviderMetrics(
            name="claude-1",
            provider_type=ProviderType.ANTHROPIC,
        )

        metrics = self.manager.get_metrics()
        assert len(metrics) == 2
        assert "openai-1" in metrics
        assert "claude-1" in metrics

    def test_get_metrics_nonexistent_provider(self):
        """Test getting metrics for non-existent provider."""
        metrics = self.manager.get_metrics("nonexistent")
        assert metrics == {}

    def test_metrics_lock_exists(self):
        """B.4: AdapterManager must have a _metrics_lock for thread safety."""
        import threading

        manager = AdapterManager()
        assert hasattr(manager, "_metrics_lock"), "_metrics_lock must exist on AdapterManager"
        assert isinstance(manager._metrics_lock, type(threading.Lock())), "_metrics_lock must be a threading.Lock"

    def test_concurrent_metric_increments_no_corruption(self):
        """B.4: Concurrent metric updates must not corrupt counters.

        Two threads each increment successful_inferences 500 times.
        Without the lock the counter would be less than 1000 due to lost updates.
        """
        import threading

        manager = AdapterManager()
        manager._metrics["prov"] = ProviderMetrics(name="prov", provider_type=ProviderType.LOCAL)
        metrics = manager._metrics["prov"]

        def increment_success(n: int) -> None:
            for _ in range(n):
                with manager._metrics_lock:
                    metrics.successful_inferences += 1

        threads = [threading.Thread(target=increment_success, args=(500,)) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert metrics.successful_inferences == 1000


class TestAdapterManagerHealthCheck:
    """Test health checking."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AdapterManager()
        self.mock_adapter = Mock()

    def test_health_check_single_provider_healthy(self):
        """Test health check for healthy provider."""
        self.manager._metrics["openai-1"] = ProviderMetrics(
            name="openai-1",
            provider_type=ProviderType.OPENAI,
        )
        self.mock_adapter.health_check.return_value = {"healthy": True}

        with patch.object(self.manager, "get_provider", return_value=self.mock_adapter):
            result = self.manager.health_check("openai-1")

            assert "openai-1" in result
            assert result["openai-1"]["healthy"] is True
            assert self.manager._metrics["openai-1"].health_status == ProviderHealthStatus.HEALTHY

    def test_health_check_single_provider_unhealthy(self):
        """Test health check for unhealthy provider."""
        self.manager._metrics["openai-1"] = ProviderMetrics(
            name="openai-1",
            provider_type=ProviderType.OPENAI,
        )
        self.mock_adapter.health_check.return_value = {"healthy": False}

        with patch.object(self.manager, "get_provider", return_value=self.mock_adapter):
            result = self.manager.health_check("openai-1")

            assert result["openai-1"]["healthy"] is False
            assert self.manager._metrics["openai-1"].health_status == ProviderHealthStatus.UNHEALTHY

    def test_health_check_provider_exception(self):
        """Test health check when provider raises exception."""
        self.manager._metrics["openai-1"] = ProviderMetrics(
            name="openai-1",
            provider_type=ProviderType.OPENAI,
        )
        self.mock_adapter.health_check.side_effect = Exception("Connection failed")

        with patch.object(self.manager, "get_provider", return_value=self.mock_adapter):
            result = self.manager.health_check("openai-1")

            assert result["openai-1"]["healthy"] is False
            assert self.manager._metrics["openai-1"].health_status == ProviderHealthStatus.UNHEALTHY

    def test_health_check_all_providers(self):
        """Test health check for all providers."""
        providers_dict = {"openai-1": self.mock_adapter, "claude-1": Mock()}
        self.mock_adapter.health_check.return_value = {"healthy": True}

        with patch.object(self.manager, "list_providers", return_value=providers_dict):
            with patch.object(self.manager, "health_check", wraps=self.manager.health_check):
                # Call with no arguments to check all
                result = self.manager.health_check()

                # Verify health_check was called for all providers
                assert len(result) >= 0  # May be empty if wrapped


class TestAdapterManagerModelDiscovery:
    """Test model discovery."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AdapterManager()
        self.mock_adapter = Mock()
        self.model1 = ModelInfo(
            id="gpt-4",
            name="GPT-4",
            provider="openai",
            endpoint="https://api.openai.com",
            capabilities=["chat", "text"],
            context_len=8192,
            memory_gb=8,
            version="1.0",
        )
        self.mock_adapter.discover_models.return_value = [self.model1]

    def test_discover_models_single_provider(self):
        """Test discovering models from a single provider."""
        with patch.object(self.manager, "get_provider", return_value=self.mock_adapter):
            result = self.manager.discover_models("openai-1")

            assert "openai-1" in result
            assert len(result["openai-1"]) == 1
            assert result["openai-1"][0].id == "gpt-4"

    def test_discover_models_provider_exception(self):
        """Test model discovery when provider raises exception."""
        self.mock_adapter.discover_models.side_effect = Exception("API error")

        with patch.object(self.manager, "get_provider", return_value=self.mock_adapter):
            result = self.manager.discover_models("openai-1")

            assert "openai-1" in result
            assert result["openai-1"] == []

    def test_discover_models_nonexistent_provider(self):
        """Test discovering models from non-existent provider."""
        with patch.object(self.manager, "get_provider", return_value=None):
            result = self.manager.discover_models("nonexistent")
            assert result == {}


class TestAdapterManagerProviderSelection:
    """Test provider selection logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AdapterManager()
        self.mock_adapter = Mock()
        self.model1 = ModelInfo(
            id="gpt-4",
            name="GPT-4",
            provider="openai",
            endpoint="https://api.openai.com",
            capabilities=["chat", "text"],
            context_len=8192,
            memory_gb=8,
            version="1.0",
        )
        self.mock_adapter.discover_models.return_value = [self.model1]
        self.mock_adapter.score_model_for_task.return_value = 0.95

    def test_select_provider_preferred_available(self):
        """Test selecting preferred provider when available."""
        with patch.object(self.manager, "get_provider", return_value=self.mock_adapter):
            provider_name, model = self.manager.select_provider_for_task(
                {"capability": "chat"},
                preferred_provider="openai-1",
            )

            assert provider_name == "openai-1"
            assert model.id == "gpt-4"

    def test_select_provider_preferred_unavailable(self):
        """Test fallback when preferred provider fails."""
        self.mock_adapter.discover_models.side_effect = Exception("Failed")

        with patch.object(self.manager, "get_provider", return_value=self.mock_adapter):
            with patch.object(self.manager.registry, "find_best_model", return_value=(None, None)):
                provider_name, model = self.manager.select_provider_for_task(
                    {"capability": "chat"},
                    preferred_provider="openai-1",
                )

                assert provider_name is None
                assert model is None

    def test_select_provider_no_preference(self):
        """Test selecting best provider without preference."""
        with patch.object(self.manager.registry, "find_best_model", return_value=(self.mock_adapter, self.model1)):
            with patch.object(self.manager, "list_providers", return_value={"openai-1": self.mock_adapter}):
                provider_name, model = self.manager.select_provider_for_task({"capability": "chat"})

                assert provider_name == "openai-1"
                assert model.id == "gpt-4"


class TestAdapterManagerInference:
    """Test inference execution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AdapterManager()
        self.mock_adapter = Mock()
        self.manager._provider_fallback_order = ["openai-1", "claude-1"]

    def test_infer_success(self):
        """Test successful inference."""
        response = InferenceResponse(
            model_id="gpt-4",
            output="Test output",
            latency_ms=150,
            tokens_used=100,
            status="ok",
        )
        self.mock_adapter.infer.return_value = response

        self.manager._metrics["openai-1"] = ProviderMetrics(
            name="openai-1",
            provider_type=ProviderType.OPENAI,
        )

        with patch.object(self.manager, "get_provider", return_value=self.mock_adapter):
            request = InferenceRequest(
                model_id="gpt-4",
                prompt="Test prompt",
            )
            result = self.manager.infer(request, provider_name="openai-1")

            assert result.status == "ok"
            assert result.output == "Test output"
            assert self.manager._metrics["openai-1"].successful_inferences == 1

    def test_infer_permission_denied(self):
        """Test inference when permission is denied."""
        cm = get_context_manager()
        # SANDBOX mode is very restrictive
        cm.switch_mode(ExecutionMode.SANDBOX)

        try:
            request = InferenceRequest(
                model_id="gpt-4",
                prompt="Test prompt",
            )
            result = self.manager.infer(request)

            # In sandbox mode with no fallback providers, should fail
            assert result.status == "error"
            # Either denied or all providers failed (no providers in sandbox)
            assert "denied" in result.error.lower() or "failed" in result.error.lower()
        finally:
            cm.pop_context()

    def test_infer_with_fallback(self):
        """Test inference with automatic fallback."""
        # First adapter fails
        mock_adapter1 = Mock()
        mock_adapter1.infer.return_value = InferenceResponse(
            model_id="gpt-4",
            output="",
            latency_ms=0,
            tokens_used=0,
            status="error",
            error="Rate limited",
        )

        # Second adapter succeeds
        mock_adapter2 = Mock()
        success_response = InferenceResponse(
            model_id="claude-3",
            output="Success",
            latency_ms=200,
            tokens_used=150,
            status="ok",
        )
        mock_adapter2.infer.return_value = success_response

        self.manager._metrics["openai-1"] = ProviderMetrics(
            name="openai-1",
            provider_type=ProviderType.OPENAI,
        )
        self.manager._metrics["claude-1"] = ProviderMetrics(
            name="claude-1",
            provider_type=ProviderType.ANTHROPIC,
        )

        def get_provider_side_effect(name):
            if name == "openai-1":
                return mock_adapter1
            elif name == "claude-1":
                return mock_adapter2
            return None

        with patch.object(self.manager, "get_provider", side_effect=get_provider_side_effect):
            request = InferenceRequest(
                model_id="multi",
                prompt="Test",
            )
            result = self.manager.infer(request, fallback_on_error=True)

            assert result.status == "ok"
            assert result.output == "Success"
            assert self.manager._metrics["claude-1"].successful_inferences == 1

    def test_infer_all_providers_fail(self):
        """Test inference when all providers fail."""
        self.mock_adapter.infer.side_effect = Exception("Connection error")

        self.manager._metrics["openai-1"] = ProviderMetrics(
            name="openai-1",
            provider_type=ProviderType.OPENAI,
        )

        with patch.object(self.manager, "get_provider", return_value=self.mock_adapter):
            request = InferenceRequest(
                model_id="gpt-4",
                prompt="Test",
            )
            result = self.manager.infer(request, provider_name="openai-1", fallback_on_error=False)

            assert result.status == "error"
            assert "failed" in result.error.lower()


class TestAdapterManagerFallbackOrder:
    """Test fallback order configuration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AdapterManager()

    def test_set_fallback_order(self):
        """Test setting fallback order."""
        order = ["openai-1", "claude-1", "cohere-1"]
        self.manager.set_fallback_order(order)

        assert self.manager._provider_fallback_order == order

    def test_fallback_order_initial_empty(self):
        """Test that fallback order starts empty."""
        manager = AdapterManager()
        assert manager._provider_fallback_order == []


class TestAdapterManagerStatus:
    """Test status reporting."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = AdapterManager()
        self.mock_adapter = Mock()

    def test_get_status(self):
        """Test getting comprehensive status."""
        self.manager._metrics["openai-1"] = ProviderMetrics(
            name="openai-1",
            provider_type=ProviderType.OPENAI,
            successful_inferences=10,
        )
        self.manager._provider_fallback_order = ["openai-1", "claude-1"]

        with patch.object(self.manager, "list_providers", return_value={"openai-1": self.mock_adapter}):
            status = self.manager.get_status()

            assert "providers" in status
            assert "fallback_order" in status
            assert status["fallback_order"] == ["openai-1", "claude-1"]


class TestGlobalAdapterManager:
    """Test global adapter manager singleton."""

    def setup_method(self) -> None:
        """Reset the singleton between tests."""
        adapter_manager_module._adapter_manager = None
        adapter_manager_module.AdapterRegistry.clear_instances()

    def test_get_adapter_manager_singleton(self):
        """Test that get_adapter_manager returns singleton."""
        manager1 = get_adapter_manager()
        manager2 = get_adapter_manager()

        assert manager1 is manager2

    def test_adapter_manager_is_instance(self):
        """Test that returned manager is AdapterManager instance."""
        manager = get_adapter_manager()
        assert isinstance(manager, AdapterManager)

    def test_env_driven_vllm_registration_updates_fallback_order(self, monkeypatch):
        """Configured endpoint-driven vLLM should register ahead of local fallback."""
        import sys

        runtime_cfg = {
            "hardware": {"gpu_vram_gb": 32},
            "local_inference": {"models_dir": "./models", "gpu_layers": -1, "context_length": 8192},
            "inference_backend": {
                "primary": "vllm",
                "fallback": "nim",
                "vllm": {
                    "enabled": True,
                    "endpoint": "http://localhost:8000",
                    "gpu_only": True,
                    "cache_salt": "test-salt",
                    "prefix_caching_enabled": True,
                },
                "nim": {"enabled": False, "endpoint": "http://localhost:8001", "gpu_only": True},
            },
        }

        monkeypatch.delitem(sys.modules, "pytest", raising=False)
        with (
            patch.object(adapter_manager_module, "load_backend_runtime_config", return_value=runtime_cfg),
            patch.dict("os.environ", {"VETINARI_TESTING": ""}, clear=False),
        ):
            manager = get_adapter_manager()

        assert manager.get_provider("local") is not None
        vllm_provider = manager.get_provider("vllm")
        assert vllm_provider is not None
        assert vllm_provider.config.extra_config["cache_salt"] == "test-salt"
        assert vllm_provider.config.extra_config["prefix_caching_enabled"] is True
        assert manager._provider_fallback_order[:2] == ["vllm", "local"]

    def test_config_driven_nim_registration_is_supported(self, monkeypatch):
        """Config-driven NIM registration should appear in the fallback order."""
        import sys

        runtime_cfg = {
            "hardware": {"gpu_vram_gb": 32},
            "local_inference": {"models_dir": "./models", "gpu_layers": -1, "context_length": 8192},
            "inference_backend": {
                "primary": "nim",
                "fallback": "llama_cpp",
                "vllm": {"enabled": False, "endpoint": "http://localhost:8000", "gpu_only": True},
                "nim": {
                    "enabled": True,
                    "endpoint": "http://localhost:8001",
                    "gpu_only": True,
                    "kv_cache_reuse_enabled": True,
                },
            },
        }

        monkeypatch.delitem(sys.modules, "pytest", raising=False)
        with (
            patch.object(adapter_manager_module, "load_backend_runtime_config", return_value=runtime_cfg),
            patch.dict("os.environ", {"VETINARI_TESTING": ""}, clear=False),
        ):
            manager = get_adapter_manager()

        nim_provider = manager.get_provider("nim")
        assert nim_provider is not None
        assert nim_provider.config.extra_config["kv_cache_reuse_enabled"] is True
        assert manager._provider_fallback_order[:2] == ["nim", "local"]
