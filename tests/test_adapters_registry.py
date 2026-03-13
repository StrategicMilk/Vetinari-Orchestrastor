"""Tests for vetinari/adapters/registry.py — Phase 7B"""
import unittest
from unittest.mock import MagicMock

import pytest

from vetinari.adapters.base import (
    ModelInfo,
    ProviderAdapter,
    ProviderConfig,
    ProviderType,
)
from vetinari.adapters.registry import AdapterRegistry


def _make_config(pt=ProviderType.LM_STUDIO, endpoint="http://localhost:1234", api_key=None):
    return ProviderConfig(provider_type=pt, name="test", endpoint=endpoint, api_key=api_key or "")


def _mock_adapter(models=None):
    a = MagicMock(spec=ProviderAdapter)
    a.models = models or []
    a.discover_models.return_value = a.models
    a.health_check.return_value = {"healthy": True, "reason": "ok", "timestamp": "now"}
    return a


def _make_model(mid="m1"):
    return ModelInfo(id=mid, name=mid, provider="test", endpoint="http://ep",
                     capabilities=["code_gen"], context_len=4096, memory_gb=4, version="1.0")


class TestAdapterRegistry(unittest.TestCase):

    def setUp(self):
        AdapterRegistry.clear_instances()

    def tearDown(self):
        AdapterRegistry.clear_instances()

    # ── list_supported_providers ──────────────────────────────────────────────

    def test_list_supported_providers(self):
        providers = AdapterRegistry.list_supported_providers()
        assert ProviderType.LM_STUDIO in providers
        assert ProviderType.OPENAI in providers
        assert ProviderType.ANTHROPIC in providers

    # ── create_adapter ────────────────────────────────────────────────────────

    def test_create_lmstudio_adapter(self):
        cfg = _make_config(ProviderType.LM_STUDIO)
        adapter = AdapterRegistry.create_adapter(cfg)
        assert adapter is not None

    def test_create_unknown_provider_raises(self):
        cfg = ProviderConfig(provider_type=ProviderType.HUGGINGFACE,
                             name="hf", endpoint="http://hf")
        with pytest.raises((ValueError, KeyError)):
            AdapterRegistry.create_adapter(cfg)

    def test_create_with_instance_name_caches(self):
        cfg = _make_config(ProviderType.LM_STUDIO)
        AdapterRegistry.create_adapter(cfg, instance_name="my_lms")
        assert AdapterRegistry.get_adapter("my_lms") is not None

    def test_create_without_name_not_cached(self):
        cfg = _make_config(ProviderType.LM_STUDIO)
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
        cfg = _make_config(ProviderType.LM_STUDIO)
        AdapterRegistry.create_adapter(cfg, instance_name="lms1")
        adapters = AdapterRegistry.list_adapters()
        assert "lms1" in adapters

    # ── discover_all_models ───────────────────────────────────────────────────

    def test_discover_all_models_empty(self):
        results = AdapterRegistry.discover_all_models()
        assert results == {}

    def test_discover_all_models_populated(self):
        mock_a = _mock_adapter([_make_model("m1"), _make_model("m2")])
        AdapterRegistry._instances["test"] = mock_a
        results = AdapterRegistry.discover_all_models()
        assert len(results["test"]) == 2

    def test_discover_all_models_handles_exception(self):
        mock_a = _mock_adapter()
        mock_a.discover_models.side_effect = RuntimeError("fail")
        AdapterRegistry._instances["bad"] = mock_a
        results = AdapterRegistry.discover_all_models()
        assert results["bad"] == []

    # ── health_check_all ──────────────────────────────────────────────────────

    def test_health_check_all_empty(self):
        assert AdapterRegistry.health_check_all() == {}

    def test_health_check_all_healthy(self):
        mock_a = _mock_adapter()
        AdapterRegistry._instances["healthy"] = mock_a
        results = AdapterRegistry.health_check_all()
        assert results["healthy"]["healthy"]

    def test_health_check_all_handles_exception(self):
        mock_a = _mock_adapter()
        mock_a.health_check.side_effect = RuntimeError("boom")
        AdapterRegistry._instances["broken"] = mock_a
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
        m_good = _make_model("good")
        m_good.capabilities = ["code_gen", "chat"]
        m_good.context_len = 8192
        m_good.latency_estimate_ms = 500
        m_good.cost_per_1k_tokens = 0.0
        m_good.free_tier = True
        m_bad = _make_model("bad")
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
            def discover_models(self): return []
            def health_check(self): return {"healthy": True}
            def infer(self, r): return MagicMock()
            def get_capabilities(self): return {}

        AdapterRegistry.register_adapter(ProviderType.LOCAL, DummyAdapter)
        assert ProviderType.LOCAL in AdapterRegistry.list_supported_providers()
        # Clean up
        del AdapterRegistry._adapter_classes[ProviderType.LOCAL]

    # ── clear_instances ───────────────────────────────────────────────────────

    def test_clear_instances(self):
        AdapterRegistry._instances["x"] = _mock_adapter()
        AdapterRegistry.clear_instances()
        assert AdapterRegistry.list_adapters() == {}


if __name__ == "__main__":
    unittest.main()
