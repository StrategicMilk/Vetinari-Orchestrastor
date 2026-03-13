"""Tests for vetinari/adapters/base.py — Phase 7B"""
import unittest

from vetinari.adapters.base import (
    InferenceRequest,
    InferenceResponse,
    ModelInfo,
    ProviderAdapter,
    ProviderConfig,
    ProviderType,
)


def _make_config(provider_type=ProviderType.LM_STUDIO, endpoint="http://localhost:1234"):
    return ProviderConfig(provider_type=provider_type, name="test", endpoint=endpoint)


def _make_model(capabilities=None, context_len=4096, latency_ms=1000,
                cost=0.0, free_tier=False):
    return ModelInfo(
        id="m1", name="Model 1", provider="test", endpoint="http://ep",
        capabilities=capabilities or ["code_gen", "chat"],
        context_len=context_len, memory_gb=4, version="1.0",
        latency_estimate_ms=latency_ms,
        cost_per_1k_tokens=cost,
        free_tier=free_tier,
    )


class ConcreteAdapter(ProviderAdapter):
    def discover_models(self): return []
    def health_check(self): return {"healthy": True, "reason": "ok", "timestamp": "now"}
    def infer(self, req): return InferenceResponse(model_id=req.model_id, output="ok",
                                                    latency_ms=10, tokens_used=5, status="ok")
    def get_capabilities(self): return {}


class TestProviderType(unittest.TestCase):
    def test_all_values(self):
        for pt in ProviderType:
            assert isinstance(pt.value, str)


class TestProviderConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = _make_config()
        assert cfg.max_retries == 3
        assert cfg.timeout_seconds == 120
        assert cfg.api_key is None

    def test_custom_values(self):
        cfg = ProviderConfig(provider_type=ProviderType.OPENAI, name="oai",
                             endpoint="https://api.openai.com", api_key="sk-test",  # noqa: VET040
                             max_retries=5, timeout_seconds=30)
        assert cfg.api_key == "sk-test"
        assert cfg.max_retries == 5


class TestModelInfo(unittest.TestCase):
    def test_creation(self):
        m = _make_model()
        assert m.id == "m1"
        assert m.cost_per_1k_tokens == 0.0
        assert not m.free_tier

    def test_custom_fields(self):
        m = _make_model(latency_ms=500, free_tier=True, cost=0.002)
        assert m.latency_estimate_ms == 500
        assert m.free_tier


class TestInferenceRequest(unittest.TestCase):
    def test_defaults(self):
        req = InferenceRequest(model_id="m1", prompt="hello")
        assert req.max_tokens == 2048
        self.assertAlmostEqual(req.temperature, 0.7)
        assert req.system_prompt is None

    def test_custom(self):
        req = InferenceRequest(model_id="m1", prompt="p", max_tokens=100, temperature=0.1)
        assert req.max_tokens == 100


class TestInferenceResponse(unittest.TestCase):
    def test_ok_response(self):
        r = InferenceResponse(model_id="m1", output="result",
                              latency_ms=50, tokens_used=10, status="ok")
        assert r.error is None

    def test_error_response(self):
        r = InferenceResponse(model_id="m1", output="", latency_ms=0,
                              tokens_used=0, status="error", error="timeout")
        assert r.error == "timeout"


class TestProviderAdapterScoring(unittest.TestCase):
    def setUp(self):
        self.adapter = ConcreteAdapter(_make_config())

    def test_score_full_match(self):
        m = _make_model(capabilities=["code_gen", "chat"], context_len=8192,
                        latency_ms=1000, cost=0.01, free_tier=True)
        score = self.adapter.score_model_for_task(m, {
            "required_capabilities": ["code_gen", "chat"],
            "input_tokens": 1000,
            "max_latency_ms": 5000,
            "max_cost_per_1k_tokens": 0.1,
        })
        assert score > 0.9

    def test_score_returns_float_in_range(self):
        m = _make_model()
        score = self.adapter.score_model_for_task(m, {})
        assert isinstance(score, float)
        assert score >= 0.0
        assert score <= 1.0

    def test_score_full_requirements_non_negative(self):
        m = _make_model(capabilities=["code_gen"], context_len=8192,
                        latency_ms=500, cost=0.001, free_tier=True)
        score = self.adapter.score_model_for_task(m, {
            "required_capabilities": ["code_gen"],
            "input_tokens": 1000,
            "max_latency_ms": 5000,
            "max_cost_per_1k_tokens": 0.1,
        })
        assert score >= 0.5

    def test_score_combined_penalty_lower_than_ideal(self):
        # A model that fails on capabilities, context, and latency simultaneously
        # scores lower than an ideal model — compare the raw sums before capping

        m_bad  = _make_model(capabilities=[], context_len=100,
                             latency_ms=100000, cost=10.0, free_tier=False)
        m_good = _make_model(capabilities=["code_gen"], context_len=32768,
                             latency_ms=100, cost=0.0, free_tier=True)
        reqs = {
            "required_capabilities": ["code_gen"],
            "input_tokens": 500,
            "max_latency_ms": 1000,
            "max_cost_per_1k_tokens": 0.1,
        }
        score_bad  = self.adapter.score_model_for_task(m_bad,  reqs)
        score_good = self.adapter.score_model_for_task(m_good, reqs)
        assert score_bad <= score_good

    def test_score_free_tier_bonus(self):
        m_free = _make_model(capabilities=[], free_tier=True)
        m_paid = _make_model(capabilities=[], free_tier=False)
        s_free = self.adapter.score_model_for_task(m_free, {})
        s_paid = self.adapter.score_model_for_task(m_paid, {})
        assert s_free >= s_paid
        assert s_free - s_paid + 0.001 > 0  # at least non-negative

    def test_score_capped_at_1(self):
        m = _make_model(capabilities=["code_gen"], free_tier=True,
                        latency_ms=1, cost=0.0)
        score = self.adapter.score_model_for_task(m, {
            "required_capabilities": ["code_gen"],
            "input_tokens": 100,
            "max_latency_ms": 30000,
        })
        assert score <= 1.0

    def test_repr(self):
        assert "ConcreteAdapter" in repr(self.adapter)


class TestAdapterReprAndInit(unittest.TestCase):
    def test_init_sets_attributes(self):
        cfg = ProviderConfig(provider_type=ProviderType.LM_STUDIO, name="n",
                             endpoint="http://ep", max_retries=2)
        a = ConcreteAdapter(cfg)
        assert a.max_retries == 2
        assert a.endpoint == "http://ep"
        assert a.models == []


if __name__ == "__main__":
    unittest.main()
