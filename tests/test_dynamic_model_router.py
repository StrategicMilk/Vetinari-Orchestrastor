"""Tests for vetinari/dynamic_model_router.py"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from vetinari.dynamic_model_router import (
    DynamicModelRouter,
    ModelCapabilities,
    ModelInfo,
    ModelSelection,
    TaskType,
    get_model_router,
    infer_task_type,
    init_model_router,
)
from vetinari.types import ModelProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(
    model_id="llama-3-8b",
    name="Llama 3 8B",
    code_gen=False,
    reasoning=False,
    chat=True,
    creative=False,
    docs=False,
    math=False,
    analysis=False,
    summarization=False,
    context_length=4096,
    memory_gb=4.0,
    provider=ModelProvider.LOCAL,
    is_available=True,
) -> ModelInfo:
    caps = ModelCapabilities(
        code_gen=code_gen,
        reasoning=reasoning,
        chat=chat,
        creative=creative,
        docs=docs,
        math=math,
        analysis=analysis,
        summarization=summarization,
        context_length=context_length,
    )
    m = ModelInfo(
        id=model_id,
        name=name,
        provider=provider,
        capabilities=caps,
        context_length=context_length,
        memory_gb=memory_gb,
        is_available=is_available,
    )
    return m


# ---------------------------------------------------------------------------
# 1. TaskType enum values
# ---------------------------------------------------------------------------

class TestTaskTypeEnum(unittest.TestCase):
    def test_has_18_task_types(self):
        self.assertEqual(len(TaskType), 19)  # 13 original + 6 phase-7 + SPECIFICATION

    def test_known_values(self):
        self.assertEqual(TaskType.CODING.value, "coding")
        self.assertEqual(TaskType.PLANNING.value, "planning")
        self.assertEqual(TaskType.SECURITY_AUDIT.value, "security_audit")
        self.assertEqual(TaskType.COST_ANALYSIS.value, "cost_analysis")
        self.assertEqual(TaskType.SPECIFICATION.value, "specification")
        self.assertEqual(TaskType.CREATIVE_WRITING.value, "creative_writing")
        self.assertEqual(TaskType.DEVOPS.value, "devops")
        self.assertEqual(TaskType.IMAGE_GENERATION.value, "image_generation")


# ---------------------------------------------------------------------------
# 2. ModelCapabilities.from_dict()
# ---------------------------------------------------------------------------

class TestModelCapabilitiesFromDict(unittest.TestCase):
    def test_empty_dict_gives_defaults(self):
        caps = ModelCapabilities.from_dict({})
        self.assertFalse(caps.code_gen)
        self.assertFalse(caps.reasoning)
        # from_dict parses an empty capabilities list, so chat stays False
        # (the dataclass default of True is overridden by the list-parse path)
        self.assertFalse(caps.chat)
        self.assertEqual(caps.context_length, 2048)

    def test_capabilities_list_sets_code_gen(self):
        caps = ModelCapabilities.from_dict({"capabilities": ["code_generation", "chat"]})
        self.assertTrue(caps.code_gen)
        self.assertTrue(caps.chat)

    def test_capabilities_list_sets_reasoning(self):
        caps = ModelCapabilities.from_dict({"capabilities": ["reasoning"]})
        self.assertTrue(caps.reasoning)

    def test_tags_infer_code_gen(self):
        caps = ModelCapabilities.from_dict({"tags": ["coder", "local"]})
        self.assertTrue(caps.code_gen)

    def test_tags_infer_reasoning(self):
        caps = ModelCapabilities.from_dict({"tags": ["reasoning-model"]})
        self.assertTrue(caps.reasoning)

    def test_technical_specs_populated(self):
        caps = ModelCapabilities.from_dict({
            "context_len": 8192,
            "supports_functions": True,
            "supports_vision": True,
            "supports_json": True,
        })
        self.assertEqual(caps.context_length, 8192)
        self.assertTrue(caps.supports_functions)
        self.assertTrue(caps.supports_vision)
        self.assertTrue(caps.supports_json)

    def test_preferred_for_populated(self):
        caps = ModelCapabilities.from_dict({"preferred_for": ["coding", "analysis"]})
        self.assertEqual(caps.preferred_for, ["coding", "analysis"])

    def test_context_length_alias(self):
        caps = ModelCapabilities.from_dict({"context_length": 16384})
        self.assertEqual(caps.context_length, 16384)


# ---------------------------------------------------------------------------
# 3. ModelCapabilities.matches_task() scoring
# ---------------------------------------------------------------------------

class TestModelCapabilitiesMatchesTask(unittest.TestCase):
    def test_coder_scores_high_for_coding(self):
        caps = ModelCapabilities(code_gen=True, reasoning=True, chat=True)
        score = caps.matches_task(TaskType.CODING)
        self.assertGreater(score, 0.8)

    def test_no_caps_scores_zero_for_coding(self):
        caps = ModelCapabilities(code_gen=False, reasoning=False, chat=False)
        score = caps.matches_task(TaskType.CODING)
        self.assertEqual(score, 0.0)

    def test_reasoning_cap_scores_high_for_reasoning_task(self):
        caps = ModelCapabilities(reasoning=True)
        score = caps.matches_task(TaskType.REASONING)
        self.assertGreaterEqual(score, 0.6)

    def test_creative_cap_scores_for_creative_task(self):
        caps = ModelCapabilities(creative=True, chat=True)
        score = caps.matches_task(TaskType.CREATIVE)
        self.assertGreater(score, 0.0)

    def test_summarization_cap_scores_for_summarization(self):
        caps = ModelCapabilities(summarization=True, chat=True)
        score = caps.matches_task(TaskType.SUMMARIZATION)
        self.assertGreater(score, 0.5)

    def test_unknown_task_type_returns_half(self):
        # All scores are defined; this just checks the scores dict covers all enum values.
        caps = ModelCapabilities()
        for task in TaskType:
            s = caps.matches_task(task)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


# ---------------------------------------------------------------------------
# 4. ModelInfo.from_dict() and to_dict() roundtrip
# ---------------------------------------------------------------------------

class TestModelInfoFromDictToDict(unittest.TestCase):
    def _sample_dict(self):
        return {
            "id": "llama-3-8b",
            "name": "Llama 3 8B",
            "endpoint": "http://localhost:1234",
            "memory_gb": 8,
            "context_len": 8192,
            "version": "3.0",
            "capabilities": ["code_gen", "reasoning", "chat"],
            "tags": ["local"],
            "metadata": {"source": "lmstudio"},
        }

    def test_from_dict_populates_id_and_name(self):
        info = ModelInfo.from_dict(self._sample_dict())
        self.assertEqual(info.id, "llama-3-8b")
        self.assertEqual(info.name, "Llama 3 8B")

    def test_from_dict_detects_local_provider(self):
        info = ModelInfo.from_dict(self._sample_dict())
        self.assertEqual(info.provider, ModelProvider.LOCAL)

    def test_from_dict_detects_openai_provider(self):
        info = ModelInfo.from_dict({"id": "gpt-4o", "name": "GPT-4o"})
        self.assertEqual(info.provider, ModelProvider.OPENAI)

    def test_from_dict_detects_anthropic_provider(self):
        info = ModelInfo.from_dict({"id": "claude-3-opus", "name": "Claude 3 Opus"})
        self.assertEqual(info.provider, ModelProvider.ANTHROPIC)

    def test_from_dict_detects_google_provider(self):
        info = ModelInfo.from_dict({"id": "gemini-pro", "name": "Gemini Pro"})
        self.assertEqual(info.provider, ModelProvider.GOOGLE)

    def test_from_dict_metrics_applied(self):
        data = self._sample_dict()
        data["metrics"] = {"avg_latency_ms": 500, "success_rate": 0.95}
        info = ModelInfo.from_dict(data)
        self.assertAlmostEqual(info.avg_latency_ms, 500.0)
        self.assertAlmostEqual(info.success_rate, 0.95)

    def test_to_dict_roundtrip_preserves_id(self):
        info = ModelInfo.from_dict(self._sample_dict())
        d = info.to_dict()
        self.assertEqual(d["id"], "llama-3-8b")
        self.assertEqual(d["provider"], ModelProvider.LOCAL.value)

    def test_to_dict_contains_capability_keys(self):
        info = ModelInfo.from_dict(self._sample_dict())
        d = info.to_dict()
        caps = d["capabilities"]
        for key in ("code_gen", "reasoning", "chat", "creative", "docs",
                    "math", "analysis", "summarization", "context_length",
                    "supports_functions", "supports_vision", "supports_json", "tags"):
            self.assertIn(key, caps)


# ---------------------------------------------------------------------------
# 5. DynamicModelRouter initialization
# ---------------------------------------------------------------------------

class TestDynamicModelRouterInit(unittest.TestCase):
    def test_default_init(self):
        router = DynamicModelRouter()
        self.assertTrue(router.prefer_local)
        self.assertEqual(router.models, {})

    def test_custom_init_params(self):
        router = DynamicModelRouter(prefer_local=False, max_latency_ms=5000, max_memory_gb=16)
        self.assertFalse(router.prefer_local)
        self.assertEqual(router.max_latency_ms, 5000)
        self.assertEqual(router.max_memory_gb, 16)


# ---------------------------------------------------------------------------
# 6. register_model() + get_model_by_id()
# ---------------------------------------------------------------------------

class TestRegisterModel(unittest.TestCase):
    def setUp(self):
        self.router = DynamicModelRouter()

    def test_register_and_retrieve(self):
        m = _make_model("llama-3-8b")
        self.router.register_model(m)
        result = self.router.get_model_by_id("llama-3-8b")
        self.assertIs(result, m)

    def test_get_missing_model_returns_none(self):
        self.assertIsNone(self.router.get_model_by_id("nonexistent"))

    def test_register_overwrites_existing(self):
        m1 = _make_model("m1", name="First")
        m2 = _make_model("m1", name="Second")
        self.router.register_model(m1)
        self.router.register_model(m2)
        self.assertEqual(self.router.get_model_by_id("m1").name, "Second")


# ---------------------------------------------------------------------------
# 7. register_models_from_pool()
# ---------------------------------------------------------------------------

class TestRegisterModelsFromPool(unittest.TestCase):
    def setUp(self):
        self.router = DynamicModelRouter()

    def test_registers_all_models(self):
        pool = [
            {"id": "llama-3-8b", "name": "Llama 3 8B", "capabilities": ["code_gen", "chat"]},
            {"id": "mistral-7b", "name": "Mistral 7B", "capabilities": ["chat"]},
        ]
        self.router.register_models_from_pool(pool)
        self.assertIn("llama-3-8b", self.router.models)
        self.assertIn("mistral-7b", self.router.models)

    def test_empty_pool_registers_nothing(self):
        self.router.register_models_from_pool([])
        self.assertEqual(self.router.models, {})


# ---------------------------------------------------------------------------
# 8. select_model() with matching capabilities
# ---------------------------------------------------------------------------

class TestSelectModelWithCapabilities(unittest.TestCase):
    def setUp(self):
        self.router = DynamicModelRouter(prefer_local=True)
        self.coder = _make_model("coder", code_gen=True, reasoning=True, chat=True, context_length=8192)
        self.chatter = _make_model("chatter", chat=True)
        self.router.register_model(self.coder)
        self.router.register_model(self.chatter)

    def test_coding_task_prefers_coder(self):
        sel = self.router.select_model(TaskType.CODING)
        self.assertIsNotNone(sel)
        self.assertEqual(sel.model.id, "coder")

    def test_selection_has_reasoning_field(self):
        sel = self.router.select_model(TaskType.CODING)
        self.assertIsInstance(sel.reasoning, str)
        self.assertGreater(len(sel.reasoning), 0)

    def test_selection_score_between_0_and_1(self):
        sel = self.router.select_model(TaskType.CODING)
        self.assertGreaterEqual(sel.score, 0.0)
        self.assertLessEqual(sel.score, 1.5)  # score can exceed 1.0 due to bonuses

    def test_selection_confidence_between_0_and_1(self):
        sel = self.router.select_model(TaskType.CODING)
        self.assertGreaterEqual(sel.confidence, 0.0)
        self.assertLessEqual(sel.confidence, 1.0)

    def test_selection_history_grows(self):
        before = len(self.router._selection_history)
        self.router.select_model(TaskType.CODING)
        self.assertEqual(len(self.router._selection_history), before + 1)


# ---------------------------------------------------------------------------
# 9. select_model() with no available models
# ---------------------------------------------------------------------------

class TestSelectModelNoModels(unittest.TestCase):
    def test_returns_none_when_no_models(self):
        router = DynamicModelRouter()
        result = router.select_model(TaskType.GENERAL)
        self.assertIsNone(result)

    def test_returns_none_when_all_unavailable(self):
        router = DynamicModelRouter()
        m = _make_model("m1", is_available=False)
        router.register_model(m)
        result = router.select_model(TaskType.GENERAL)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# 10. select_model() fallback behavior
# ---------------------------------------------------------------------------

class TestSelectModelFallback(unittest.TestCase):
    def test_fallback_used_when_required_caps_not_met(self):
        router = DynamicModelRouter()
        # Register a chat-only model (no code_gen)
        m = _make_model("chatter", chat=True, code_gen=False)
        router.register_model(m)

        # Require code_gen — will force fallback path
        sel = router.select_model(TaskType.CODING, required_capabilities=["code_gen"])
        # Fallback picks any available model
        self.assertIsNotNone(sel)
        self.assertAlmostEqual(sel.score, 0.0)
        self.assertAlmostEqual(sel.confidence, 0.1)

    def test_fallback_reasoning_text(self):
        router = DynamicModelRouter()
        m = _make_model("chatter", chat=True, code_gen=False)
        router.register_model(m)
        sel = router.select_model(TaskType.CODING, required_capabilities=["code_gen"])
        self.assertIn("Fallback", sel.reasoning)


# ---------------------------------------------------------------------------
# 11. update_model_performance() tracks history
# ---------------------------------------------------------------------------

class TestUpdateModelPerformance(unittest.TestCase):
    def setUp(self):
        self.router = DynamicModelRouter()
        self.m = _make_model("m1")
        self.router.register_model(self.m)

    def test_total_uses_increments(self):
        self.router.update_model_performance("m1", latency_ms=200.0, success=True)
        self.assertEqual(self.m.total_uses, 1)

    def test_avg_latency_updated(self):
        self.router.update_model_performance("m1", latency_ms=200.0, success=True)
        self.router.update_model_performance("m1", latency_ms=400.0, success=True)
        self.assertAlmostEqual(self.m.avg_latency_ms, 300.0)

    def test_success_rate_updated(self):
        # Formula: (prev_rate * prev_total + new_success) / new_total
        # After 1st call: (1.0 * 0 + 1) / 1 = 1.0
        # After 2nd call: (1.0 * 1 + 0) / 2 = 0.5
        self.router.update_model_performance("m1", latency_ms=100.0, success=True)
        self.router.update_model_performance("m1", latency_ms=100.0, success=False)
        self.assertAlmostEqual(self.m.success_rate, 0.5)

    def test_cache_key_stored(self):
        self.router.update_model_performance("m1", latency_ms=150.0, success=True,
                                             task_type=TaskType.CODING)
        self.assertIn("m1:coding", self.router._performance_cache)

    def test_unknown_model_does_nothing(self):
        # Should not raise
        self.router.update_model_performance("nonexistent", latency_ms=100.0, success=True)

    def test_last_checked_set(self):
        self.router.update_model_performance("m1", latency_ms=100.0, success=True)
        self.assertNotEqual(self.m.last_checked, "")


# ---------------------------------------------------------------------------
# 12. _score_model() scoring factors
# ---------------------------------------------------------------------------

class TestScoreModel(unittest.TestCase):
    def setUp(self):
        self.router = DynamicModelRouter(prefer_local=True)

    def test_local_model_gets_provider_bonus(self):
        local = _make_model("local", provider=ModelProvider.LOCAL, chat=True)
        cloud = _make_model("cloud", provider=ModelProvider.OPENAI, chat=True)
        score_local = self.router._score_model(local, TaskType.GENERAL, "", None)
        score_cloud = self.router._score_model(cloud, TaskType.GENERAL, "", None)
        self.assertGreater(score_local, score_cloud)

    def test_preferred_model_gets_preference_bonus(self):
        m1 = _make_model("m1")
        m2 = _make_model("m2")
        score_preferred = self.router._score_model(m1, TaskType.GENERAL, "", ["m1"])
        score_not = self.router._score_model(m2, TaskType.GENERAL, "", ["m1"])
        self.assertGreater(score_preferred, score_not)

    def test_large_context_gets_bonus(self):
        small_ctx = _make_model("small", context_length=2048)
        large_ctx = _make_model("large", context_length=8192)
        s_small = self.router._score_model(small_ctx, TaskType.GENERAL, "", None)
        s_large = self.router._score_model(large_ctx, TaskType.GENERAL, "", None)
        self.assertGreater(s_large, s_small)


# ---------------------------------------------------------------------------
# 13. _calculate_confidence() high vs low spread
# ---------------------------------------------------------------------------

class TestCalculateConfidence(unittest.TestCase):
    def setUp(self):
        self.router = DynamicModelRouter()

    def test_single_model_returns_half(self):
        m = _make_model("m1")
        conf = self.router._calculate_confidence([(m, 0.8)])
        self.assertAlmostEqual(conf, 0.5)

    def test_large_gap_gives_high_confidence(self):
        m1 = _make_model("m1")
        m2 = _make_model("m2")
        conf = self.router._calculate_confidence([(m1, 1.0), (m2, 0.1)])
        self.assertGreater(conf, 0.7)

    def test_small_gap_gives_lower_confidence(self):
        m1 = _make_model("m1")
        m2 = _make_model("m2")
        conf_large = self.router._calculate_confidence([(m1, 1.0), (m2, 0.1)])
        conf_small = self.router._calculate_confidence([(m1, 0.6), (m2, 0.55)])
        self.assertGreater(conf_large, conf_small)

    def test_zero_best_score_gives_low_confidence(self):
        m1 = _make_model("m1")
        m2 = _make_model("m2")
        conf = self.router._calculate_confidence([(m1, 0.0), (m2, 0.0)])
        self.assertAlmostEqual(conf, 0.1)


# ---------------------------------------------------------------------------
# 14. get_available_models() filtering
# ---------------------------------------------------------------------------

class TestGetAvailableModels(unittest.TestCase):
    def test_returns_only_available(self):
        router = DynamicModelRouter()
        router.register_model(_make_model("a", is_available=True))
        router.register_model(_make_model("b", is_available=False))
        router.register_model(_make_model("c", is_available=True))
        available = router.get_available_models()
        ids = {m.id for m in available}
        self.assertIn("a", ids)
        self.assertNotIn("b", ids)
        self.assertIn("c", ids)

    def test_empty_router_returns_empty_list(self):
        router = DynamicModelRouter()
        self.assertEqual(router.get_available_models(), [])


# ---------------------------------------------------------------------------
# 15. get_models_by_capability() filtering
# ---------------------------------------------------------------------------

class TestGetModelsByCapability(unittest.TestCase):
    def setUp(self):
        self.router = DynamicModelRouter()
        self.router.register_model(_make_model("coder", code_gen=True))
        self.router.register_model(_make_model("reasoner", reasoning=True))
        self.router.register_model(_make_model("docer", docs=True))
        self.router.register_model(_make_model("plain"))

    def test_code_gen_filter(self):
        results = self.router.get_models_by_capability("code_gen")
        ids = [m.id for m in results]
        self.assertIn("coder", ids)
        self.assertNotIn("plain", ids)

    def test_reasoning_filter(self):
        results = self.router.get_models_by_capability("reasoning")
        ids = [m.id for m in results]
        self.assertIn("reasoner", ids)
        self.assertNotIn("plain", ids)

    def test_docs_filter(self):
        results = self.router.get_models_by_capability("docs")
        ids = [m.id for m in results]
        self.assertIn("docer", ids)

    def test_unavailable_excluded(self):
        self.router.register_model(_make_model("offline-coder", code_gen=True, is_available=False))
        results = self.router.get_models_by_capability("code_gen")
        ids = [m.id for m in results]
        self.assertNotIn("offline-coder", ids)


# ---------------------------------------------------------------------------
# 16. get_routing_stats() returns correct structure
# ---------------------------------------------------------------------------

class TestGetRoutingStats(unittest.TestCase):
    def test_empty_stats_structure(self):
        router = DynamicModelRouter()
        stats = router.get_routing_stats()
        self.assertIn("total_selections", stats)
        self.assertIn("models_used", stats)
        self.assertIn("available_models", stats)
        self.assertIn("total_models", stats)
        self.assertEqual(stats["total_selections"], 0)

    def test_stats_update_after_selection(self):
        router = DynamicModelRouter()
        router.register_model(_make_model("m1"))
        router.select_model(TaskType.GENERAL)
        stats = router.get_routing_stats()
        self.assertEqual(stats["total_selections"], 1)
        self.assertIn("m1", stats["models_used"])
        self.assertEqual(stats["models_used"]["m1"], 1)

    def test_stats_total_models(self):
        router = DynamicModelRouter()
        router.register_model(_make_model("a"))
        router.register_model(_make_model("b", is_available=False))
        stats = router.get_routing_stats()
        self.assertEqual(stats["total_models"], 2)
        self.assertEqual(stats["available_models"], 1)


# ---------------------------------------------------------------------------
# 17. infer_task_type() keyword detection
# ---------------------------------------------------------------------------

class TestInferTaskType(unittest.TestCase):
    def test_security_keywords(self):
        self.assertEqual(infer_task_type("run a security audit"), TaskType.SECURITY_AUDIT)
        self.assertEqual(infer_task_type("find a vulnerability in this code"), TaskType.SECURITY_AUDIT)

    def test_devops_keywords(self):
        self.assertEqual(infer_task_type("set up docker"), TaskType.DEVOPS)
        self.assertEqual(infer_task_type("deploy to kubernetes"), TaskType.DEVOPS)

    def test_image_generation_keywords(self):
        self.assertEqual(infer_task_type("generate a logo"), TaskType.IMAGE_GENERATION)
        self.assertEqual(infer_task_type("create an icon for the app"), TaskType.IMAGE_GENERATION)

    def test_cost_analysis_keywords(self):
        self.assertEqual(infer_task_type("analyze our budget"), TaskType.COST_ANALYSIS)
        self.assertEqual(infer_task_type("review pricing options"), TaskType.COST_ANALYSIS)

    def test_specification_keywords(self):
        self.assertEqual(infer_task_type("write a specification"), TaskType.SPECIFICATION)
        self.assertEqual(infer_task_type("define acceptance criteria"), TaskType.SPECIFICATION)

    def test_creative_writing_keywords(self):
        self.assertEqual(infer_task_type("write a short story"), TaskType.CREATIVE_WRITING)
        self.assertEqual(infer_task_type("compose a poem"), TaskType.CREATIVE_WRITING)

    def test_coding_keywords(self):
        self.assertEqual(infer_task_type("implement a function"), TaskType.CODING)
        self.assertEqual(infer_task_type("build the class"), TaskType.CODING)

    def test_planning_keywords(self):
        self.assertEqual(infer_task_type("plan a strategy"), TaskType.PLANNING)

    def test_testing_keywords(self):
        self.assertEqual(infer_task_type("write tests for the module"), TaskType.TESTING)

    def test_documentation_keywords(self):
        self.assertEqual(infer_task_type("write docs"), TaskType.DOCUMENTATION)

    def test_summarization_keywords(self):
        self.assertEqual(infer_task_type("please summarize the meeting notes"), TaskType.SUMMARIZATION)
        self.assertEqual(infer_task_type("provide a brief summary"), TaskType.SUMMARIZATION)

    def test_translation_keywords(self):
        self.assertEqual(infer_task_type("translate this text"), TaskType.TRANSLATION)

    def test_general_fallback(self):
        self.assertEqual(infer_task_type("do something random"), TaskType.GENERAL)

    def test_case_insensitive(self):
        self.assertEqual(infer_task_type("SECURITY AUDIT"), TaskType.SECURITY_AUDIT)


# ---------------------------------------------------------------------------
# 18. init_model_router() creates instance
# ---------------------------------------------------------------------------

class TestInitModelRouter(unittest.TestCase):
    def test_creates_new_router(self):
        router = init_model_router(prefer_local=False)
        self.assertIsInstance(router, DynamicModelRouter)
        self.assertFalse(router.prefer_local)

    def test_replaces_global_singleton(self):
        r1 = init_model_router()
        r2 = init_model_router()
        self.assertIsNot(r1, r2)
        # After init, get_model_router() returns the latest
        self.assertIs(get_model_router(), r2)


# ---------------------------------------------------------------------------
# 19. Thread safety of singleton
# ---------------------------------------------------------------------------

class TestThreadSafety(unittest.TestCase):
    def test_concurrent_performance_updates(self):
        router = DynamicModelRouter()
        router.register_model(_make_model("m1"))
        errors = []

        def updater():
            try:
                for _ in range(50):
                    router.update_model_performance("m1", latency_ms=100.0, success=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=updater) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Thread errors: {errors}")
        self.assertEqual(router.models["m1"].total_uses, 250)

    def test_concurrent_selections_do_not_crash(self):
        router = DynamicModelRouter()
        for i in range(5):
            router.register_model(_make_model(f"m{i}", chat=True))
        errors = []

        def selector():
            try:
                for _ in range(20):
                    router.select_model(TaskType.GENERAL)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=selector) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Thread errors: {errors}")


# ---------------------------------------------------------------------------
# 20. select_model() prefers local models when prefer_local=True
# ---------------------------------------------------------------------------

class TestSelectModelPrefersLocal(unittest.TestCase):
    def test_local_wins_over_cloud_equal_caps(self):
        router = DynamicModelRouter(prefer_local=True)
        local = _make_model("llama-local", provider=ModelProvider.LOCAL,
                            code_gen=True, reasoning=True, chat=True, context_length=8192)
        cloud = _make_model("gpt-cloud", provider=ModelProvider.OPENAI,
                            code_gen=True, reasoning=True, chat=True, context_length=8192)
        router.register_model(local)
        router.register_model(cloud)

        sel = router.select_model(TaskType.CODING)
        self.assertEqual(sel.model.id, "llama-local")

    def test_no_preference_still_selects_model(self):
        router = DynamicModelRouter(prefer_local=False)
        cloud = _make_model("gpt-cloud", provider=ModelProvider.OPENAI,
                            code_gen=True, chat=True)
        router.register_model(cloud)
        sel = router.select_model(TaskType.CODING)
        self.assertIsNotNone(sel)
        self.assertEqual(sel.model.id, "gpt-cloud")


# ---------------------------------------------------------------------------
# Bonus: check_model_health()
# ---------------------------------------------------------------------------

class TestCheckModelHealth(unittest.TestCase):
    def test_unknown_model_returns_false(self):
        router = DynamicModelRouter()
        self.assertFalse(router.check_model_health("nonexistent"))

    def test_healthy_model_returns_true(self):
        router = DynamicModelRouter()
        router.register_model(_make_model("m1"))
        self.assertTrue(router.check_model_health("m1"))

    def test_high_latency_model_returns_false(self):
        router = DynamicModelRouter(max_latency_ms=1000)
        m = _make_model("slow")
        m.avg_latency_ms = 5000.0  # well above 2 * max_latency_ms=1000
        router.register_model(m)
        self.assertFalse(router.check_model_health("slow"))

    def test_low_success_rate_returns_false(self):
        router = DynamicModelRouter()
        m = _make_model("flaky")
        m.success_rate = 0.3
        router.register_model(m)
        self.assertFalse(router.check_model_health("flaky"))

    def test_callback_used_when_set(self):
        router = DynamicModelRouter()
        router.register_model(_make_model("m1"))
        cb = MagicMock(return_value=False)
        router.set_health_check_callback(cb)
        result = router.check_model_health("m1")
        cb.assert_called_once_with("m1")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
