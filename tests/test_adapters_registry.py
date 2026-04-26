"""Tests for vetinari/adapters/registry.py — Phase 7B"""

from unittest.mock import MagicMock

import pytest

from tests.factories import make_model_info, make_provider_config
from vetinari.adapters.base import (
    ProviderAdapter,
    ProviderConfig,
    ProviderType,
)
from vetinari.adapters.registry import AdapterRegistry
from vetinari.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def _isolate_adapter_registry():
    """Save and restore AdapterRegistry class state to prevent cross-test pollution."""
    saved_classes = dict(AdapterRegistry._adapter_classes)
    saved_instances = dict(AdapterRegistry._instances)
    yield
    AdapterRegistry._adapter_classes = saved_classes
    AdapterRegistry._instances = saved_instances


@pytest.fixture
def mock_adapter():
    """Return a factory that creates a MagicMock ProviderAdapter."""

    def _factory(models=None):
        a = MagicMock(spec=ProviderAdapter)
        a.models = models or []
        a.discover_models.return_value = a.models
        a.health_check.return_value = {"healthy": True, "reason": "ok", "timestamp": "now"}
        return a

    return _factory


class TestAdapterRegistry:
    @pytest.fixture(autouse=True)
    def _setup(self):
        # Save and restore both instances AND adapter classes to prevent
        # cross-test pollution from other files that modify the class registry.
        saved_classes = dict(AdapterRegistry._adapter_classes)
        AdapterRegistry.clear_instances()
        yield
        AdapterRegistry.clear_instances()
        AdapterRegistry._adapter_classes = saved_classes

    # ── list_supported_providers ──────────────────────────────────────────────

    def test_list_supported_providers(self):
        providers = AdapterRegistry.list_supported_providers()
        assert ProviderType.LOCAL in providers
        assert ProviderType.OPENAI in providers
        assert ProviderType.ANTHROPIC in providers

    # ── create_adapter ────────────────────────────────────────────────────────

    def test_create_local_adapter(self):
        cfg = make_provider_config(provider_type=ProviderType.LOCAL)
        adapter = AdapterRegistry.create_adapter(cfg)
        assert adapter is not None
        assert hasattr(adapter, "infer")

    def test_create_huggingface_provider_succeeds(self):
        """HUGGINGFACE is a supported provider type — creation must NOT raise."""
        cfg = ProviderConfig(provider_type=ProviderType.HUGGINGFACE, name="hf", endpoint="http://hf")
        adapter = AdapterRegistry.create_adapter(cfg)
        assert adapter is not None
        assert hasattr(adapter, "infer")

    def test_create_with_instance_name_caches(self):
        cfg = make_provider_config(provider_type=ProviderType.LOCAL)
        adapter = AdapterRegistry.create_adapter(cfg, instance_name="my_lms")
        cached = AdapterRegistry.get_adapter("my_lms")
        assert cached is not None
        assert cached is adapter

    def test_create_without_name_not_cached(self):
        cfg = make_provider_config(provider_type=ProviderType.LOCAL)
        AdapterRegistry.create_adapter(cfg)
        # No instance_name → nothing cached under "None"
        assert AdapterRegistry.get_adapter("") is None

    # ── get_adapter ───────────────────────────────────────────────────────────

    def test_get_nonexistent_returns_none(self):
        assert AdapterRegistry.get_adapter("ghost") is None

    # ── list_adapters ─────────────────────────────────────────────────────────

    def test_list_adapters_empty(self):
        assert AdapterRegistry.list_adapters() == {}

    def test_list_adapters_after_register(self):
        cfg = make_provider_config(provider_type=ProviderType.LOCAL)
        AdapterRegistry.create_adapter(cfg, instance_name="lms1")
        adapters = AdapterRegistry.list_adapters()
        assert "lms1" in adapters

    # ── discover_all_models ───────────────────────────────────────────────────

    def test_discover_all_models_empty(self):
        results = AdapterRegistry.discover_all_models()
        assert results == {}

    def test_discover_all_models_populated(self, mock_adapter):
        a = mock_adapter([
            make_model_info(model_id="m1", endpoint="http://ep"),
            make_model_info(model_id="m2", endpoint="http://ep"),
        ])
        AdapterRegistry._instances["test"] = a
        results = AdapterRegistry.discover_all_models()
        assert len(results["test"]) == 2

    def test_discover_all_models_handles_exception(self, mock_adapter):
        a = mock_adapter()
        a.discover_models.side_effect = RuntimeError("fail")
        AdapterRegistry._instances["bad"] = a
        results = AdapterRegistry.discover_all_models()
        assert results["bad"] == []

    # ── health_check_all ──────────────────────────────────────────────────────

    def test_health_check_all_empty(self):
        assert AdapterRegistry.health_check_all() == {}

    def test_health_check_all_healthy(self, mock_adapter):
        AdapterRegistry._instances["healthy"] = mock_adapter()
        results = AdapterRegistry.health_check_all()
        assert results["healthy"]["healthy"]

    def test_health_check_all_handles_exception(self, mock_adapter):
        a = mock_adapter()
        a.health_check.side_effect = RuntimeError("boom")
        AdapterRegistry._instances["broken"] = a
        results = AdapterRegistry.health_check_all()
        assert not results["broken"]["healthy"]

    # ── find_best_model ───────────────────────────────────────────────────────

    def test_find_best_model_empty(self):
        adapter, model = AdapterRegistry.find_best_model({"required_capabilities": ["code_gen"]})
        assert adapter is None
        assert model is None

    def test_find_best_model_picks_highest_score(self):
        from vetinari.adapters.base import ProviderAdapter

        mock_a = MagicMock(spec=ProviderAdapter)
        m_good = make_model_info(model_id="good", endpoint="http://ep")
        m_good.capabilities = ["code_gen", "chat"]
        m_good.context_len = 8192
        m_good.latency_estimate_ms = 500
        m_good.cost_per_1k_tokens = 0.0
        m_good.free_tier = True
        m_bad = make_model_info(model_id="bad", endpoint="http://ep")
        m_bad.capabilities = []
        m_bad.context_len = 512
        m_bad.latency_estimate_ms = 60000
        m_bad.cost_per_1k_tokens = 0.5
        m_bad.free_tier = False
        mock_a.models = [m_good, m_bad]
        from vetinari.adapters.base import ProviderAdapter as PA

        mock_a.score_model_for_task = lambda m, reqs: PA.score_model_for_task(mock_a, m, reqs)
        AdapterRegistry._instances["a1"] = mock_a
        _, best = AdapterRegistry.find_best_model({"required_capabilities": ["code_gen"]})
        assert best.id == "good"

    # ── register_adapter ─────────────────────────────────────────────────────

    def test_register_custom_adapter(self):
        class DummyAdapter(ProviderAdapter):
            def discover_models(self):
                return []

            def health_check(self):
                return {"healthy": True}

            def infer(self, r):
                return MagicMock()

            def get_capabilities(self):
                return {}

        AdapterRegistry.register_adapter(ProviderType.LOCAL, DummyAdapter)
        assert ProviderType.LOCAL in AdapterRegistry.list_supported_providers()
        # Clean up
        del AdapterRegistry._adapter_classes[ProviderType.LOCAL]

    # ── clear_instances ───────────────────────────────────────────────────────

    def test_clear_instances(self, mock_adapter):
        AdapterRegistry._instances["x"] = mock_adapter()
        AdapterRegistry.clear_instances()
        assert AdapterRegistry.list_adapters() == {}
