"""Tests for vetinari/analytics/cost.py (Phase 5)"""

import time

import pytest

from vetinari.analytics.cost import (
    CostEntry,
    ModelPricing,
    get_cost_tracker,
    reset_cost_tracker,
)


def _tracker():
    reset_cost_tracker()
    return get_cost_tracker()


class TestModelPricing:
    def test_compute_cost(self):
        p = ModelPricing(input_per_1k=0.03, output_per_1k=0.06)
        cost = p.compute(1000, 500)
        assert cost == pytest.approx(0.03 + 0.03)

    def test_flat_fee(self):
        p = ModelPricing(per_request=0.005)
        assert p.compute(0, 0) == pytest.approx(0.005)

    def test_zero_tokens_zero_cost(self):
        p = ModelPricing(input_per_1k=0.05, output_per_1k=0.10)
        assert p.compute(0, 0) == pytest.approx(0.0)


class TestCostEntry:
    def test_to_dict(self):
        e = CostEntry(
            provider="openai", model="gpt-4", input_tokens=100, output_tokens=50, agent="builder", task_id="t1"
        )
        d = e.to_dict()
        assert d["provider"] == "openai"
        assert d["model"] == "gpt-4"
        assert d["agent"] == "builder"


class TestCostTrackerSingleton:
    @pytest.fixture(autouse=True)
    def _setup(self):
        return

    def test_same_instance(self):
        assert get_cost_tracker() is get_cost_tracker()

    def test_reset_new_instance(self):
        a = get_cost_tracker()
        reset_cost_tracker()
        assert a is not get_cost_tracker()


class TestPricing:
    @pytest.fixture(autouse=True)
    def _setup(self):
        return

    def test_set_and_get_pricing(self):
        t = _tracker()
        t.set_pricing("openai", "gpt-4", ModelPricing(input_per_1k=0.05))
        p = t.get_pricing("openai", "gpt-4")
        assert p.input_per_1k == pytest.approx(0.05)

    def test_wildcard_fallback(self):
        t = _tracker()
        t.set_pricing("local", "*", ModelPricing(input_per_1k=0.0))
        p = t.get_pricing("local", "llama-3")
        assert p.input_per_1k == pytest.approx(0.0)

    def test_unknown_model_returns_zero_pricing(self):
        t = _tracker()
        p = t.get_pricing("unknown_provider", "unknown_model")
        assert p.compute(1000, 1000) == pytest.approx(0.0)


class TestRecording:
    @pytest.fixture(autouse=True)
    def _setup(self):
        return

    def test_record_computes_cost(self):
        t = _tracker()
        t.set_pricing("openai", "gpt-4", ModelPricing(input_per_1k=0.03, output_per_1k=0.06))
        entry = CostEntry(provider="openai", model="gpt-4", input_tokens=1000, output_tokens=1000)
        returned = t.record(entry)
        assert returned.cost_usd == pytest.approx(0.09)
        assert returned.cost_usd >= 0.0

    def test_record_preserves_explicit_cost(self):
        t = _tracker()
        entry = CostEntry(provider="openai", model="gpt-4", input_tokens=1000, output_tokens=1000, cost_usd=0.5)
        t.record(entry)
        report = t.get_report()
        assert report.total_cost_usd == pytest.approx(0.5)

    def test_stats_entry_count(self):
        t = _tracker()
        for _ in range(5):
            t.record(CostEntry(provider="p", model="m", cost_usd=0.01))
        stats = t.get_stats()
        assert stats["total_entries"] == 5
        assert "configured_models" in stats
        assert stats["configured_models"] > 0  # at least the built-in defaults


class TestReport:
    @pytest.fixture(autouse=True)
    def _setup(self):
        return

    def _seed(self):
        t = _tracker()
        t.record(
            CostEntry(
                provider="openai",
                model="gpt-4",
                input_tokens=500,
                output_tokens=200,
                agent="builder",
                task_id="t1",
                cost_usd=0.02,
            )
        )
        t.record(
            CostEntry(
                provider="openai",
                model="gpt-4",
                input_tokens=300,
                output_tokens=100,
                agent="explorer",
                task_id="t1",
                cost_usd=0.01,
            )
        )
        t.record(
            CostEntry(
                provider="local",
                model="llama-3",
                input_tokens=800,
                output_tokens=400,
                agent="builder",
                task_id="t2",
                cost_usd=0.0,
            )
        )
        return t

    def test_total_cost(self):
        t = self._seed()
        r = t.get_report()
        assert r.total_cost_usd == pytest.approx(0.03)

    def test_total_tokens(self):
        t = self._seed()
        r = t.get_report()
        assert r.total_tokens == 500 + 200 + 300 + 100 + 800 + 400

    def test_by_agent(self):
        t = self._seed()
        r = t.get_report()
        assert "builder" in r.by_agent
        assert "explorer" in r.by_agent
        assert r.by_agent["builder"] == pytest.approx(0.02)
        assert r.by_agent["explorer"] == pytest.approx(0.01)
        # costs must be non-negative
        for _agent, cost in r.by_agent.items():
            assert cost >= 0.0

    def test_by_provider(self):
        t = self._seed()
        r = t.get_report()
        assert "openai" in r.by_provider
        assert r.by_provider["openai"] == pytest.approx(0.03)

    def test_filter_by_agent(self):
        t = self._seed()
        r = t.get_report(agent="explorer")
        assert r.total_requests == 1
        assert r.total_cost_usd == pytest.approx(0.01)

    def test_filter_by_task(self):
        t = self._seed()
        r = t.get_report(task_id="t2")
        assert r.total_requests == 1
        assert r.total_cost_usd == pytest.approx(0.0)  # local entry has cost_usd=0.0
        assert r.total_tokens == 800 + 400  # 800 input + 400 output
        assert "t2" in r.by_task
        assert r.by_task["t2"] == pytest.approx(0.0)

    def test_filter_since(self):
        t = self._seed()
        future = time.time() + 3600
        r = t.get_report(since=future)
        assert r.total_requests == 0

    def test_get_top_agents(self):
        t = self._seed()
        top = t.get_top_agents(n=2)
        assert len(top) <= 2
        assert len(top) == 2  # seed has exactly 2 agents: builder and explorer
        assert "agent" in top[0]
        assert "cost_usd" in top[0]
        # builder has 0.02, explorer has 0.01 — builder should be first
        assert top[0]["agent"] == "builder"
        assert top[0]["cost_usd"] == pytest.approx(0.02)
        assert top[1]["agent"] == "explorer"
        assert top[1]["cost_usd"] == pytest.approx(0.01)
        # costs must be non-negative
        for item in top:
            assert item["cost_usd"] >= 0.0

    def test_get_top_models(self):
        t = self._seed()
        top = t.get_top_models(n=3)
        assert isinstance(top, list)
        # seed has 2 distinct provider:model combos: openai:gpt-4 and local:llama-3
        assert len(top) == 2
        assert "model" in top[0]
        assert "cost_usd" in top[0]
        # openai:gpt-4 has 0.03 total cost, local:llama-3 has 0.0
        assert top[0]["model"] == "openai:gpt-4"
        assert top[0]["cost_usd"] == pytest.approx(0.03)
        assert top[1]["model"] == "local:llama-3"
        assert top[1]["cost_usd"] == pytest.approx(0.0)
        # all costs non-negative
        for item in top:
            assert item["cost_usd"] >= 0.0

    def test_clear(self):
        t = self._seed()
        t.clear()
        assert t.get_report().total_requests == 0

    def test_report_to_dict(self):
        t = self._seed()
        d = t.get_report().to_dict()
        for k in (
            "total_cost_usd",
            "total_tokens",
            "total_requests",
            "by_agent",
            "by_provider",
            "by_model",
            "by_task",
            "entries",
        ):
            assert k in d
        # Verify types and values are sensible
        assert isinstance(d["total_cost_usd"], float)
        assert d["total_cost_usd"] >= 0.0
        assert isinstance(d["total_tokens"], int)
        assert d["total_tokens"] > 0
        assert isinstance(d["total_requests"], int)
        assert d["total_requests"] == 3
        assert isinstance(d["by_agent"], dict)
        assert isinstance(d["by_provider"], dict)
        assert isinstance(d["by_model"], dict)
        assert isinstance(d["by_task"], dict)
        assert isinstance(d["entries"], int)
        assert d["entries"] == 3
