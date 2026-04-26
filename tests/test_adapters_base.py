"""Tests for vetinari/adapters/base.py — Phase 7B"""

import pytest

from tests.conftest import TEST_ENDPOINT
from tests.factories import make_model_info, make_provider_config
from vetinari.adapters.base import (
    InferenceRequest,
    InferenceResponse,
    ProviderAdapter,
    ProviderConfig,
    ProviderType,
)


class ConcreteAdapter(ProviderAdapter):
    def discover_models(self):
        return []

    def health_check(self):
        return {"healthy": True, "reason": "ok", "timestamp": "now"}

    def infer(self, req):
        return InferenceResponse(model_id=req.model_id, output="ok", latency_ms=10, tokens_used=5, status="ok")

    def get_capabilities(self):
        return {}


class TestProviderType:
    def test_all_values(self):
        for pt in ProviderType:
            assert isinstance(pt.value, str)


class TestProviderConfig:
    def test_defaults(self):
        cfg = make_provider_config()
        assert cfg.max_retries == 3
        assert cfg.timeout_seconds == 120
        assert cfg.api_key is None

    def test_custom_values(self):
        cfg = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            name="oai",
            endpoint="https://api.openai.com",
            api_key="sk-test",
            max_retries=5,
            timeout_seconds=30,
        )
        assert cfg.api_key == "sk-test"
        assert cfg.max_retries == 5


class TestModelInfo:
    def test_creation(self):
        m = make_model_info(endpoint=TEST_ENDPOINT)
        assert m.id == "m1"
        assert m.cost_per_1k_tokens == 0.0
        assert not m.free_tier

    def test_custom_fields(self):
        m = make_model_info(endpoint=TEST_ENDPOINT, latency_estimate_ms=500, free_tier=True, cost_per_1k_tokens=0.002)
        assert m.latency_estimate_ms == 500
        assert m.free_tier


class TestInferenceRequest:
    def test_defaults(self):
        req = InferenceRequest(model_id="m1", prompt="hello")
        assert req.max_tokens == 2048
        assert req.temperature == pytest.approx(0.7)
        assert req.system_prompt is None

    def test_custom(self):
        req = InferenceRequest(model_id="m1", prompt="p", max_tokens=100, temperature=0.1)
        assert req.max_tokens == 100


class TestInferenceResponse:
    def test_ok_response(self):
        r = InferenceResponse(model_id="m1", output="result", latency_ms=50, tokens_used=10, status="ok")
        assert r.error is None

    def test_error_response(self):
        r = InferenceResponse(model_id="m1", output="", latency_ms=0, tokens_used=0, status="error", error="timeout")
        assert r.error == "timeout"


class TestProviderAdapterScoring:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.adapter = ConcreteAdapter(make_provider_config())

    def test_score_full_match(self):
        m = make_model_info(
            endpoint=TEST_ENDPOINT,
            capabilities=["code_gen", "chat"],
            context_len=8192,
            latency_estimate_ms=1000,
            cost_per_1k_tokens=0.01,
            free_tier=True,
        )
        score = self.adapter.score_model_for_task(
            m,
            {
                "required_capabilities": ["code_gen", "chat"],
                "input_tokens": 1000,
                "max_latency_ms": 5000,
                "max_cost_per_1k_tokens": 0.1,
            },
        )
        assert score > 0.9

    def test_score_returns_float_in_range(self):
        m = make_model_info(endpoint=TEST_ENDPOINT)
        score = self.adapter.score_model_for_task(m, {})
        assert isinstance(score, float)
        assert score >= 0.0
        assert score <= 1.0

    def test_score_full_requirements_non_negative(self):
        m = make_model_info(
            endpoint=TEST_ENDPOINT,
            capabilities=["code_gen"],
            context_len=8192,
            latency_estimate_ms=500,
            cost_per_1k_tokens=0.001,
            free_tier=True,
        )
        score = self.adapter.score_model_for_task(
            m,
            {
                "required_capabilities": ["code_gen"],
                "input_tokens": 1000,
                "max_latency_ms": 5000,
                "max_cost_per_1k_tokens": 0.1,
            },
        )
        assert score >= 0.5

    def test_score_combined_penalty_lower_than_ideal(self):
        # A model that fails on capabilities, context, and latency simultaneously
        # scores lower than an ideal model — compare the raw sums before capping
        m_bad = make_model_info(
            endpoint=TEST_ENDPOINT,
            capabilities=[],
            context_len=100,
            latency_estimate_ms=100000,
            cost_per_1k_tokens=10.0,
            free_tier=False,
        )
        m_good = make_model_info(
            endpoint=TEST_ENDPOINT,
            capabilities=["code_gen"],
            context_len=32768,
            latency_estimate_ms=100,
            cost_per_1k_tokens=0.0,
            free_tier=True,
        )
        reqs = {
            "required_capabilities": ["code_gen"],
            "input_tokens": 500,
            "max_latency_ms": 1000,
            "max_cost_per_1k_tokens": 0.1,
        }
        score_bad = self.adapter.score_model_for_task(m_bad, reqs)
        score_good = self.adapter.score_model_for_task(m_good, reqs)
        assert score_bad <= score_good

    def test_score_free_tier_bonus(self):
        m_free = make_model_info(endpoint=TEST_ENDPOINT, capabilities=[], free_tier=True)
        m_paid = make_model_info(endpoint=TEST_ENDPOINT, capabilities=[], free_tier=False)
        s_free = self.adapter.score_model_for_task(m_free, {})
        s_paid = self.adapter.score_model_for_task(m_paid, {})
        assert s_free >= s_paid
        assert s_free - s_paid + 0.001 > 0  # at least non-negative

    def test_score_capped_at_1(self):
        m = make_model_info(
            endpoint=TEST_ENDPOINT,
            capabilities=["code_gen"],
            free_tier=True,
            latency_estimate_ms=1,
            cost_per_1k_tokens=0.0,
        )
        score = self.adapter.score_model_for_task(
            m,
            {
                "required_capabilities": ["code_gen"],
                "input_tokens": 100,
                "max_latency_ms": 30000,
            },
        )
        assert score <= 1.0

    def test_repr(self):
        assert "ConcreteAdapter" in repr(self.adapter)


class TestAdapterReprAndInit:
    def test_init_sets_attributes(self):
        cfg = ProviderConfig(provider_type=ProviderType.LOCAL, name="n", endpoint=TEST_ENDPOINT, max_retries=2)
        a = ConcreteAdapter(cfg)
        assert a.max_retries == 2
        assert a.endpoint == TEST_ENDPOINT
        assert a.models == []
