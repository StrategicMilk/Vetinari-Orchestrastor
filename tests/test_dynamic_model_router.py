"""Tests for vetinari/dynamic_model_router.py"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from tests.factories import make_router_model_info as _make_model
from vetinari.models.dynamic_model_router import (
    DynamicModelRouter,
    ModelCapabilities,
    ModelInfo,
    TaskType,
    get_model_router,
    infer_task_type,
    init_model_router,
)
from vetinari.types import ModelProvider

# ---------------------------------------------------------------------------
# 1. TaskType enum values
# ---------------------------------------------------------------------------


def test_task_type_has_19_values() -> None:
    assert len(TaskType) == 19  # 13 original + 6 phase-7 + SPECIFICATION


def test_task_type_known_values() -> None:
    assert TaskType.CODE.value == "code"
    assert TaskType.PLANNING.value == "planning"
    assert TaskType.SECURITY.value == "security"
    assert TaskType.COST_ANALYSIS.value == "cost_analysis"
    assert TaskType.SPECIFICATION.value == "specification"
    assert TaskType.CREATIVE.value == "creative"
    assert TaskType.DEVOPS.value == "devops"
    assert TaskType.IMAGE.value == "image"


# ---------------------------------------------------------------------------
# 2. ModelCapabilities.from_dict()
# ---------------------------------------------------------------------------


def test_model_capabilities_empty_dict_gives_defaults() -> None:
    caps = ModelCapabilities.from_dict({})
    assert not caps.code_gen
    assert not caps.reasoning
    # from_dict parses an empty capabilities list, so chat stays False
    # (the dataclass default of True is overridden by the list-parse path)
    assert not caps.chat
    assert caps.context_length == 8192


def test_model_capabilities_list_sets_code_gen() -> None:
    caps = ModelCapabilities.from_dict({"capabilities": ["code_generation", "chat"]})
    assert caps.code_gen
    assert caps.chat


def test_model_capabilities_list_sets_reasoning() -> None:
    caps = ModelCapabilities.from_dict({"capabilities": ["reasoning"]})
    assert caps.reasoning


def test_model_capabilities_tags_infer_code_gen() -> None:
    caps = ModelCapabilities.from_dict({"tags": ["coder", "local"]})
    assert caps.code_gen


def test_model_capabilities_tags_infer_reasoning() -> None:
    caps = ModelCapabilities.from_dict({"tags": ["reasoning-model"]})
    assert caps.reasoning


def test_model_capabilities_technical_specs_populated() -> None:
    caps = ModelCapabilities.from_dict({
        "context_len": 8192,
        "supports_functions": True,
        "supports_vision": True,
        "supports_json": True,
    })
    assert caps.context_length == 8192
    assert caps.supports_functions
    assert caps.supports_vision
    assert caps.supports_json


def test_model_capabilities_preferred_for_populated() -> None:
    caps = ModelCapabilities.from_dict({"preferred_for": ["coding", "analysis"]})
    assert caps.preferred_for == ["coding", "analysis"]


def test_model_capabilities_context_length_alias() -> None:
    caps = ModelCapabilities.from_dict({"context_length": 16384})
    assert caps.context_length == 16384


# ---------------------------------------------------------------------------
# 3. ModelCapabilities.matches_task() scoring
# ---------------------------------------------------------------------------


def test_coder_scores_high_for_coding() -> None:
    caps = ModelCapabilities(code_gen=True, reasoning=True, chat=True)
    assert caps.matches_task(TaskType.CODE) > 0.8


def test_no_caps_scores_zero_for_coding() -> None:
    caps = ModelCapabilities(code_gen=False, reasoning=False, chat=False)
    assert caps.matches_task(TaskType.CODE) == 0.0


def test_reasoning_cap_scores_high_for_reasoning_task() -> None:
    caps = ModelCapabilities(reasoning=True)
    assert caps.matches_task(TaskType.REASONING) >= 0.6


def test_creative_cap_scores_for_creative_task() -> None:
    caps = ModelCapabilities(creative=True, chat=True)
    assert caps.matches_task(TaskType.CREATIVE) > 0.0


def test_summarization_cap_scores_for_summarization() -> None:
    caps = ModelCapabilities(summarization=True, chat=True)
    assert caps.matches_task(TaskType.SUMMARIZATION) > 0.5


def test_all_task_types_return_valid_score_range() -> None:
    caps = ModelCapabilities()
    for task in TaskType:
        s = caps.matches_task(task)
        assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# 4. ModelInfo.from_dict() and to_dict() roundtrip
# ---------------------------------------------------------------------------


def _sample_model_dict() -> dict:
    return {
        "id": "llama-3-8b",
        "name": "Llama 3 8B",
        "endpoint": "local",
        "memory_gb": 8,
        "context_len": 8192,
        "version": "3.0",
        "capabilities": ["code_gen", "reasoning", "chat"],
        "tags": ["local"],
        "metadata": {"source": "llama_cpp"},
    }


def test_model_info_from_dict_populates_id_and_name() -> None:
    info = ModelInfo.from_dict(_sample_model_dict())
    assert info.id == "llama-3-8b"
    assert info.name == "Llama 3 8B"


def test_model_info_from_dict_detects_local_provider() -> None:
    info = ModelInfo.from_dict(_sample_model_dict())
    assert info.provider == ModelProvider.LOCAL


def test_model_info_from_dict_detects_openai_provider() -> None:
    info = ModelInfo.from_dict({"id": "gpt-4o", "name": "GPT-4o"})
    assert info.provider == ModelProvider.OPENAI


def test_model_info_from_dict_detects_anthropic_provider() -> None:
    info = ModelInfo.from_dict({"id": "claude-3-opus", "name": "Claude 3 Opus"})
    assert info.provider == ModelProvider.ANTHROPIC


def test_model_info_from_dict_detects_google_provider() -> None:
    info = ModelInfo.from_dict({"id": "gemini-pro", "name": "Gemini Pro"})
    assert info.provider == ModelProvider.GOOGLE


def test_model_info_from_dict_metrics_applied() -> None:
    data = _sample_model_dict()
    data["metrics"] = {"avg_latency_ms": 500, "success_rate": 0.95}
    info = ModelInfo.from_dict(data)
    assert abs(info.avg_latency_ms - 500.0) < 1e-6
    assert abs(info.success_rate - 0.95) < 1e-6


def test_model_info_to_dict_roundtrip_preserves_id() -> None:
    info = ModelInfo.from_dict(_sample_model_dict())
    d = info.to_dict()
    assert d["id"] == "llama-3-8b"
    assert d["provider"] == ModelProvider.LOCAL.value


def test_model_info_to_dict_contains_capability_keys() -> None:
    info = ModelInfo.from_dict(_sample_model_dict())
    d = info.to_dict()
    caps = d["capabilities"]
    for key in (
        "code_gen",
        "reasoning",
        "chat",
        "creative",
        "docs",
        "math",
        "analysis",
        "summarization",
        "context_length",
        "supports_functions",
        "supports_vision",
        "supports_json",
        "tags",
    ):
        assert key in caps


# ---------------------------------------------------------------------------
# 5. DynamicModelRouter initialization
# ---------------------------------------------------------------------------


def test_router_default_init() -> None:
    router = DynamicModelRouter()
    assert router.prefer_local
    assert router.models == {}


def test_router_custom_init_params() -> None:
    router = DynamicModelRouter(prefer_local=False, max_latency_ms=5000, max_memory_gb=16)
    assert not router.prefer_local
    assert router.max_latency_ms == 5000
    assert router.max_memory_gb == 16


# ---------------------------------------------------------------------------
# 6. register_model() + get_model_by_id()
# ---------------------------------------------------------------------------


def test_register_and_retrieve() -> None:
    router = DynamicModelRouter()
    m = _make_model("llama-3-8b")
    router.register_model(m)
    assert router.get_model_by_id("llama-3-8b") is m


def test_get_missing_model_returns_none() -> None:
    router = DynamicModelRouter()
    assert router.get_model_by_id("nonexistent") is None


def test_register_overwrites_existing() -> None:
    router = DynamicModelRouter()
    m1 = _make_model("m1", name="First")
    m2 = _make_model("m1", name="Second")
    router.register_model(m1)
    router.register_model(m2)
    assert router.get_model_by_id("m1").name == "Second"


# ---------------------------------------------------------------------------
# 7. register_models_from_pool()
# ---------------------------------------------------------------------------


def test_register_models_from_pool_registers_all() -> None:
    router = DynamicModelRouter()
    pool = [
        {"id": "llama-3-8b", "name": "Llama 3 8B", "capabilities": ["code_gen", "chat"]},
        {"id": "mistral-7b", "name": "Mistral 7B", "capabilities": ["chat"]},
    ]
    router.register_models_from_pool(pool)
    assert "llama-3-8b" in router.models
    assert "mistral-7b" in router.models


def test_register_models_from_pool_empty_registers_nothing() -> None:
    router = DynamicModelRouter()
    router.register_models_from_pool([])
    assert router.models == {}


# ---------------------------------------------------------------------------
# 8. select_model() with matching capabilities
# ---------------------------------------------------------------------------


def test_select_model_coding_task_prefers_coder() -> None:
    router = DynamicModelRouter(prefer_local=True)
    coder = _make_model("coder", code_gen=True, reasoning=True, chat=True, context_length=8192)
    chatter = _make_model("chatter", chat=True)
    router.register_model(coder)
    router.register_model(chatter)
    sel = router.select_model(TaskType.CODE)
    assert sel is not None
    assert sel.model.id == "coder"


def test_select_model_has_reasoning_field() -> None:
    router = DynamicModelRouter(prefer_local=True)
    router.register_model(_make_model("coder", code_gen=True, reasoning=True, chat=True))
    sel = router.select_model(TaskType.CODE)
    assert isinstance(sel.reasoning, str)
    assert len(sel.reasoning) > 0


def test_select_model_score_between_0_and_1_5() -> None:
    router = DynamicModelRouter(prefer_local=True)
    router.register_model(_make_model("coder", code_gen=True, reasoning=True, chat=True))
    sel = router.select_model(TaskType.CODE)
    assert sel.score >= 0.0
    assert sel.score <= 1.5  # score can exceed 1.0 due to bonuses


def test_select_model_confidence_between_0_and_1() -> None:
    router = DynamicModelRouter(prefer_local=True)
    router.register_model(_make_model("coder", code_gen=True, reasoning=True, chat=True))
    sel = router.select_model(TaskType.CODE)
    assert 0.0 <= sel.confidence <= 1.0


def test_select_model_selection_history_grows() -> None:
    router = DynamicModelRouter(prefer_local=True)
    router.register_model(_make_model("coder", code_gen=True, chat=True))
    before = len(router._selection_history)
    router.select_model(TaskType.CODE)
    assert len(router._selection_history) == before + 1


# ---------------------------------------------------------------------------
# 9. select_model() with no available models
# ---------------------------------------------------------------------------


def test_select_model_returns_none_when_no_models() -> None:
    router = DynamicModelRouter()
    assert router.select_model(TaskType.GENERAL) is None


def test_select_model_returns_none_when_all_unavailable() -> None:
    router = DynamicModelRouter()
    router.register_model(_make_model("m1", is_available=False))
    assert router.select_model(TaskType.GENERAL) is None


# ---------------------------------------------------------------------------
# 10. select_model() fallback behavior
# ---------------------------------------------------------------------------


def test_select_model_fallback_used_when_required_caps_not_met() -> None:
    router = DynamicModelRouter()
    router.register_model(_make_model("chatter", chat=True, code_gen=False))
    sel = router.select_model(TaskType.CODE, required_capabilities=["code_gen"])
    assert sel is not None
    assert abs(sel.score - 0.0) < 1e-6
    assert abs(sel.confidence - 0.1) < 1e-6


def test_select_model_fallback_reasoning_text() -> None:
    router = DynamicModelRouter()
    router.register_model(_make_model("chatter", chat=True, code_gen=False))
    sel = router.select_model(TaskType.CODE, required_capabilities=["code_gen"])
    assert "Fallback" in sel.reasoning


# ---------------------------------------------------------------------------
# 11. update_model_performance() tracks history
# ---------------------------------------------------------------------------


def test_update_model_performance_total_uses_increments() -> None:
    router = DynamicModelRouter()
    m = _make_model("m1")
    router.register_model(m)
    router.update_model_performance("m1", latency_ms=200.0, success=True)
    assert m.total_uses == 1


def test_update_model_performance_avg_latency_updated() -> None:
    router = DynamicModelRouter()
    m = _make_model("m1")
    router.register_model(m)
    router.update_model_performance("m1", latency_ms=200.0, success=True)
    router.update_model_performance("m1", latency_ms=400.0, success=True)
    assert abs(m.avg_latency_ms - 300.0) < 1e-6


def test_update_model_performance_success_rate_updated() -> None:
    router = DynamicModelRouter()
    m = _make_model("m1")
    router.register_model(m)
    # Formula: (prev_rate * prev_total + new_success) / new_total
    # After 1st call: (1.0 * 0 + 1) / 1 = 1.0
    # After 2nd call: (1.0 * 1 + 0) / 2 = 0.5
    router.update_model_performance("m1", latency_ms=100.0, success=True)
    router.update_model_performance("m1", latency_ms=100.0, success=False)
    assert abs(m.success_rate - 0.5) < 1e-6


def test_update_model_performance_cache_key_stored() -> None:
    router = DynamicModelRouter()
    router.register_model(_make_model("m1"))
    router.update_model_performance("m1", latency_ms=150.0, success=True, task_type=TaskType.CODE)
    assert "m1:code" in router._performance_cache


def test_update_model_performance_unknown_model_does_nothing() -> None:
    router = DynamicModelRouter()
    # Should not raise — no-op for unknown model
    router.update_model_performance("nonexistent", latency_ms=100.0, success=True)
    # Performance cache must remain empty — nothing was registered
    assert "nonexistent" not in router._performance_cache


def test_update_model_performance_last_checked_set() -> None:
    router = DynamicModelRouter()
    m = _make_model("m1")
    router.register_model(m)
    router.update_model_performance("m1", latency_ms=100.0, success=True)
    assert m.last_checked != ""


# ---------------------------------------------------------------------------
# 12. _score_model() scoring factors
# ---------------------------------------------------------------------------


def test_score_model_local_gets_provider_bonus() -> None:
    router = DynamicModelRouter(prefer_local=True)
    local = _make_model("local", provider=ModelProvider.LOCAL, chat=True)
    cloud = _make_model("cloud", provider=ModelProvider.OPENAI, chat=True)
    score_local = router._score_model(local, TaskType.GENERAL, "", None)
    score_cloud = router._score_model(cloud, TaskType.GENERAL, "", None)
    assert score_local > score_cloud


def test_score_model_preferred_model_gets_preference_bonus() -> None:
    router = DynamicModelRouter()
    m1 = _make_model("m1")
    m2 = _make_model("m2")
    score_preferred = router._score_model(m1, TaskType.GENERAL, "", ["m1"])
    score_not = router._score_model(m2, TaskType.GENERAL, "", ["m1"])
    assert score_preferred > score_not


def test_score_model_large_context_gets_bonus() -> None:
    router = DynamicModelRouter()
    small_ctx = _make_model("small", context_length=2048)
    large_ctx = _make_model("large", context_length=8192)
    s_small = router._score_model(small_ctx, TaskType.GENERAL, "", None)
    s_large = router._score_model(large_ctx, TaskType.GENERAL, "", None)
    assert s_large > s_small


# ---------------------------------------------------------------------------
# 13. _calculate_confidence() high vs low spread
# ---------------------------------------------------------------------------


def test_calculate_confidence_single_model_returns_half() -> None:
    router = DynamicModelRouter()
    m = _make_model("m1")
    conf = router._calculate_confidence([(m, 0.8)])
    assert abs(conf - 0.5) < 1e-6


def test_calculate_confidence_large_gap_gives_high_confidence() -> None:
    router = DynamicModelRouter()
    m1 = _make_model("m1")
    m2 = _make_model("m2")
    conf = router._calculate_confidence([(m1, 1.0), (m2, 0.1)])
    assert conf > 0.7


def test_calculate_confidence_small_gap_gives_lower_confidence() -> None:
    router = DynamicModelRouter()
    m1 = _make_model("m1")
    m2 = _make_model("m2")
    conf_large = router._calculate_confidence([(m1, 1.0), (m2, 0.1)])
    conf_small = router._calculate_confidence([(m1, 0.6), (m2, 0.55)])
    assert conf_large > conf_small


def test_calculate_confidence_zero_best_score_gives_low_confidence() -> None:
    router = DynamicModelRouter()
    m1 = _make_model("m1")
    m2 = _make_model("m2")
    conf = router._calculate_confidence([(m1, 0.0), (m2, 0.0)])
    assert abs(conf - 0.1) < 1e-6


# ---------------------------------------------------------------------------
# 14. get_available_models() filtering
# ---------------------------------------------------------------------------


def test_get_available_models_returns_only_available() -> None:
    router = DynamicModelRouter()
    router.register_model(_make_model("a", is_available=True))
    router.register_model(_make_model("b", is_available=False))
    router.register_model(_make_model("c", is_available=True))
    available = router.get_available_models()
    ids = {m.id for m in available}
    assert "a" in ids
    assert "b" not in ids
    assert "c" in ids


def test_get_available_models_empty_router_returns_empty_list() -> None:
    router = DynamicModelRouter()
    assert router.get_available_models() == []


# ---------------------------------------------------------------------------
# 15. get_models_by_capability() filtering
# ---------------------------------------------------------------------------


def test_get_models_by_capability_code_gen_filter() -> None:
    router = DynamicModelRouter()
    router.register_model(_make_model("coder", code_gen=True))
    router.register_model(_make_model("plain"))
    results = router.get_models_by_capability("code_gen")
    ids = [m.id for m in results]
    assert "coder" in ids
    assert "plain" not in ids


def test_get_models_by_capability_reasoning_filter() -> None:
    router = DynamicModelRouter()
    router.register_model(_make_model("reasoner", reasoning=True))
    router.register_model(_make_model("plain"))
    results = router.get_models_by_capability("reasoning")
    ids = [m.id for m in results]
    assert "reasoner" in ids
    assert "plain" not in ids


def test_get_models_by_capability_docs_filter() -> None:
    router = DynamicModelRouter()
    router.register_model(_make_model("docer", docs=True))
    results = router.get_models_by_capability("docs")
    ids = [m.id for m in results]
    assert "docer" in ids


def test_get_models_by_capability_unavailable_excluded() -> None:
    router = DynamicModelRouter()
    router.register_model(_make_model("coder", code_gen=True))
    router.register_model(_make_model("offline-coder", code_gen=True, is_available=False))
    results = router.get_models_by_capability("code_gen")
    ids = [m.id for m in results]
    assert "offline-coder" not in ids


# ---------------------------------------------------------------------------
# 16. get_routing_stats() returns correct structure
# ---------------------------------------------------------------------------


def test_get_routing_stats_empty_stats_structure() -> None:
    router = DynamicModelRouter()
    stats = router.get_routing_stats()
    assert "total_selections" in stats
    assert "models_used" in stats
    assert "available_models" in stats
    assert "total_models" in stats
    assert stats["total_selections"] == 0


def test_get_routing_stats_update_after_selection() -> None:
    router = DynamicModelRouter()
    router.register_model(_make_model("m1"))
    router.select_model(TaskType.GENERAL)
    stats = router.get_routing_stats()
    assert stats["total_selections"] == 1
    assert "m1" in stats["models_used"]
    assert stats["models_used"]["m1"] == 1


def test_get_routing_stats_total_models() -> None:
    router = DynamicModelRouter()
    router.register_model(_make_model("a"))
    router.register_model(_make_model("b", is_available=False))
    stats = router.get_routing_stats()
    assert stats["total_models"] == 2
    assert stats["available_models"] == 1


# ---------------------------------------------------------------------------
# 17. infer_task_type() keyword detection
# ---------------------------------------------------------------------------


def test_infer_task_type_security_keywords() -> None:
    assert infer_task_type("run a security audit") == TaskType.SECURITY
    assert infer_task_type("find a vulnerability in this code") == TaskType.SECURITY


def test_infer_task_type_devops_keywords() -> None:
    assert infer_task_type("set up docker") == TaskType.DEVOPS
    assert infer_task_type("deploy to kubernetes") == TaskType.DEVOPS


def test_infer_task_type_image_generation_keywords() -> None:
    assert infer_task_type("generate a logo") == TaskType.IMAGE
    assert infer_task_type("create an icon for the app") == TaskType.IMAGE


def test_infer_task_type_cost_analysis_keywords() -> None:
    assert infer_task_type("analyze our budget") == TaskType.COST_ANALYSIS
    assert infer_task_type("review pricing options") == TaskType.COST_ANALYSIS


def test_infer_task_type_specification_keywords() -> None:
    assert infer_task_type("write a specification") == TaskType.SPECIFICATION
    assert infer_task_type("define acceptance criteria") == TaskType.SPECIFICATION


def test_infer_task_type_creative_writing_keywords() -> None:
    assert infer_task_type("write a short story") == TaskType.CREATIVE
    assert infer_task_type("compose a poem") == TaskType.CREATIVE


def test_infer_task_type_coding_keywords() -> None:
    assert infer_task_type("implement a function") == TaskType.CODE
    assert infer_task_type("build the class") == TaskType.CODE


def test_infer_task_type_planning_keywords() -> None:
    assert infer_task_type("plan a strategy") == TaskType.PLANNING


def test_infer_task_type_testing_keywords() -> None:
    assert infer_task_type("write tests for the module") == TaskType.TESTING


def test_infer_task_type_documentation_keywords() -> None:
    assert infer_task_type("write docs") == TaskType.DOCS


def test_infer_task_type_summarization_keywords() -> None:
    assert infer_task_type("please summarize the meeting notes") == TaskType.SUMMARIZATION
    assert infer_task_type("provide a brief summary") == TaskType.SUMMARIZATION


def test_infer_task_type_translation_keywords() -> None:
    assert infer_task_type("translate this text") == TaskType.TRANSLATION


def test_infer_task_type_general_fallback() -> None:
    assert infer_task_type("do something random") == TaskType.GENERAL


def test_infer_task_type_case_insensitive() -> None:
    assert infer_task_type("SECURITY AUDIT") == TaskType.SECURITY


# ---------------------------------------------------------------------------
# 18. init_model_router() creates instance
# ---------------------------------------------------------------------------


def test_init_model_router_creates_new_router() -> None:
    router = init_model_router(prefer_local=False)
    assert isinstance(router, DynamicModelRouter)
    assert not router.prefer_local


def test_init_model_router_replaces_global_singleton() -> None:
    r1 = init_model_router()
    r2 = init_model_router()
    assert r1 is not r2
    # After init, get_model_router() returns the latest
    assert get_model_router() is r2


# ---------------------------------------------------------------------------
# 19. Thread safety of singleton
# ---------------------------------------------------------------------------


def test_thread_safety_concurrent_performance_updates() -> None:
    router = DynamicModelRouter()
    router.register_model(_make_model("m1"))
    errors: list[Exception] = []

    def updater() -> None:
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

    assert errors == [], f"Thread errors: {errors}"
    assert router.models["m1"].total_uses == 250


def test_thread_safety_concurrent_selections_do_not_crash() -> None:
    router = DynamicModelRouter()
    for i in range(5):
        router.register_model(_make_model(f"m{i}", chat=True))
    errors: list[Exception] = []

    def selector() -> None:
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

    assert errors == [], f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# 20. select_model() prefers local models when prefer_local=True
# ---------------------------------------------------------------------------


def test_select_model_local_wins_over_cloud_equal_caps() -> None:
    router = DynamicModelRouter(prefer_local=True)
    local = _make_model(
        "llama-local", provider=ModelProvider.LOCAL, code_gen=True, reasoning=True, chat=True, context_length=8192
    )
    cloud = _make_model(
        "gpt-cloud", provider=ModelProvider.OPENAI, code_gen=True, reasoning=True, chat=True, context_length=8192
    )
    router.register_model(local)
    router.register_model(cloud)
    sel = router.select_model(TaskType.CODE)
    assert sel.model.id == "llama-local"


def test_select_model_no_preference_still_selects_model() -> None:
    router = DynamicModelRouter(prefer_local=False)
    cloud = _make_model("gpt-cloud", provider=ModelProvider.OPENAI, code_gen=True, chat=True)
    router.register_model(cloud)
    sel = router.select_model(TaskType.CODE)
    assert sel is not None
    assert sel.model.id == "gpt-cloud"


# ---------------------------------------------------------------------------
# Bonus: check_model_health()
# ---------------------------------------------------------------------------


def test_check_model_health_unknown_model_returns_false() -> None:
    router = DynamicModelRouter()
    assert not router.check_model_health("nonexistent")


def test_check_model_health_healthy_model_returns_true() -> None:
    router = DynamicModelRouter()
    router.register_model(_make_model("m1"))
    assert router.check_model_health("m1")


def test_check_model_health_high_latency_model_returns_false() -> None:
    router = DynamicModelRouter(max_latency_ms=1000)
    m = _make_model("slow")
    m.avg_latency_ms = 5000.0  # well above 2 * max_latency_ms=1000
    router.register_model(m)
    assert not router.check_model_health("slow")


def test_check_model_health_low_success_rate_returns_false() -> None:
    router = DynamicModelRouter()
    m = _make_model("flaky")
    m.success_rate = 0.3
    router.register_model(m)
    assert not router.check_model_health("flaky")


def test_check_model_health_callback_used_when_set() -> None:
    router = DynamicModelRouter()
    router.register_model(_make_model("m1"))
    cb = MagicMock(return_value=False)
    router.set_health_check_callback(cb)
    result = router.check_model_health("m1")
    cb.assert_called_once_with("m1")
    assert not result
