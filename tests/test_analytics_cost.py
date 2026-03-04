"""Tests for vetinari/analytics/cost.py (Phase 5)"""
import time
import unittest

from vetinari.analytics.cost import (
    CostEntry, CostReport, CostTracker, ModelPricing,
    get_cost_tracker, reset_cost_tracker,
)


def _tracker():
    reset_cost_tracker()
    return get_cost_tracker()


class TestModelPricing(unittest.TestCase):
    def test_compute_cost(self):
        p = ModelPricing(input_per_1k=0.03, output_per_1k=0.06)
        cost = p.compute(1000, 500)
        self.assertAlmostEqual(cost, 0.03 + 0.03)

    def test_flat_fee(self):
        p = ModelPricing(per_request=0.005)
        self.assertAlmostEqual(p.compute(0, 0), 0.005)

    def test_zero_tokens_zero_cost(self):
        p = ModelPricing(input_per_1k=0.05, output_per_1k=0.10)
        self.assertAlmostEqual(p.compute(0, 0), 0.0)


class TestCostEntry(unittest.TestCase):
    def test_to_dict(self):
        e = CostEntry(provider="openai", model="gpt-4",
                      input_tokens=100, output_tokens=50,
                      agent="builder", task_id="t1")
        d = e.to_dict()
        self.assertEqual(d["provider"], "openai")
        self.assertEqual(d["model"], "gpt-4")
        self.assertEqual(d["agent"], "builder")


class TestCostTrackerSingleton(unittest.TestCase):
    def setUp(self):    reset_cost_tracker()
    def tearDown(self): reset_cost_tracker()

    def test_same_instance(self):
        self.assertIs(get_cost_tracker(), get_cost_tracker())

    def test_reset_new_instance(self):
        a = get_cost_tracker()
        reset_cost_tracker()
        self.assertIsNot(a, get_cost_tracker())


class TestPricing(unittest.TestCase):
    def setUp(self):    reset_cost_tracker()
    def tearDown(self): reset_cost_tracker()

    def test_set_and_get_pricing(self):
        t = _tracker()
        t.set_pricing("openai", "gpt-4", ModelPricing(input_per_1k=0.05))
        p = t.get_pricing("openai", "gpt-4")
        self.assertAlmostEqual(p.input_per_1k, 0.05)

    def test_wildcard_fallback(self):
        t = _tracker()
        t.set_pricing("lmstudio", "*", ModelPricing(input_per_1k=0.0))
        p = t.get_pricing("lmstudio", "llama-3")
        self.assertAlmostEqual(p.input_per_1k, 0.0)

    def test_unknown_model_returns_zero_pricing(self):
        t = _tracker()
        p = t.get_pricing("unknown_provider", "unknown_model")
        self.assertAlmostEqual(p.compute(1000, 1000), 0.0)


class TestRecording(unittest.TestCase):
    def setUp(self):    reset_cost_tracker()
    def tearDown(self): reset_cost_tracker()

    def test_record_computes_cost(self):
        t = _tracker()
        t.set_pricing("openai", "gpt-4", ModelPricing(input_per_1k=0.03, output_per_1k=0.06))
        entry = CostEntry(provider="openai", model="gpt-4",
                          input_tokens=1000, output_tokens=1000)
        returned = t.record(entry)
        self.assertAlmostEqual(returned.cost_usd, 0.09)

    def test_record_preserves_explicit_cost(self):
        t = _tracker()
        entry = CostEntry(provider="openai", model="gpt-4",
                          input_tokens=1000, output_tokens=1000, cost_usd=0.5)
        t.record(entry)
        report = t.get_report()
        self.assertAlmostEqual(report.total_cost_usd, 0.5)

    def test_stats_entry_count(self):
        t = _tracker()
        for _ in range(5):
            t.record(CostEntry(provider="p", model="m", cost_usd=0.01))
        self.assertEqual(t.get_stats()["total_entries"], 5)


class TestReport(unittest.TestCase):
    def setUp(self):    reset_cost_tracker()
    def tearDown(self): reset_cost_tracker()

    def _seed(self):
        t = _tracker()
        t.record(CostEntry(provider="openai", model="gpt-4",
                           input_tokens=500, output_tokens=200,
                           agent="builder", task_id="t1", cost_usd=0.02))
        t.record(CostEntry(provider="openai", model="gpt-4",
                           input_tokens=300, output_tokens=100,
                           agent="explorer", task_id="t1", cost_usd=0.01))
        t.record(CostEntry(provider="lmstudio", model="llama-3",
                           input_tokens=800, output_tokens=400,
                           agent="builder", task_id="t2", cost_usd=0.0))
        return t

    def test_total_cost(self):
        t = self._seed()
        r = t.get_report()
        self.assertAlmostEqual(r.total_cost_usd, 0.03)

    def test_total_tokens(self):
        t = self._seed()
        r = t.get_report()
        self.assertEqual(r.total_tokens, 500+200+300+100+800+400)

    def test_by_agent(self):
        t = self._seed()
        r = t.get_report()
        self.assertIn("builder", r.by_agent)
        self.assertIn("explorer", r.by_agent)
        self.assertAlmostEqual(r.by_agent["builder"], 0.02)

    def test_by_provider(self):
        t = self._seed()
        r = t.get_report()
        self.assertIn("openai", r.by_provider)
        self.assertAlmostEqual(r.by_provider["openai"], 0.03)

    def test_filter_by_agent(self):
        t = self._seed()
        r = t.get_report(agent="explorer")
        self.assertEqual(r.total_requests, 1)
        self.assertAlmostEqual(r.total_cost_usd, 0.01)

    def test_filter_by_task(self):
        t = self._seed()
        r = t.get_report(task_id="t2")
        self.assertEqual(r.total_requests, 1)

    def test_filter_since(self):
        t = self._seed()
        future = time.time() + 3600
        r = t.get_report(since=future)
        self.assertEqual(r.total_requests, 0)

    def test_get_top_agents(self):
        t = self._seed()
        top = t.get_top_agents(n=2)
        self.assertLessEqual(len(top), 2)
        self.assertIn("agent", top[0])
        self.assertIn("cost_usd", top[0])

    def test_get_top_models(self):
        t = self._seed()
        top = t.get_top_models(n=3)
        self.assertIsInstance(top, list)

    def test_clear(self):
        t = self._seed()
        t.clear()
        self.assertEqual(t.get_report().total_requests, 0)

    def test_report_to_dict(self):
        t = self._seed()
        d = t.get_report().to_dict()
        for k in ("total_cost_usd","total_tokens","total_requests",
                  "by_agent","by_provider","by_model","by_task","entries"):
            self.assertIn(k, d)


if __name__ == "__main__":
    unittest.main()
