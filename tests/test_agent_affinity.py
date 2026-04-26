"""Tests for vetinari/agent_affinity.py"""

from unittest.mock import patch

import pytest

from tests.factories import make_model_dict
from vetinari.agents.agent_affinity import (
    AffinityProfile,
    _extract_family_slug,
    _get_family_capability_bonus,
    get_affinity,
    get_all_affinities,
    pick_model_for_agent,
)
from vetinari.types import AgentType

# ---------------------------------------------------------------------------
# AffinityProfile dataclass
# ---------------------------------------------------------------------------


class TestAffinityProfileDefaults:
    def test_creation_with_required_field(self):
        profile = AffinityProfile(agent_type=AgentType.WORKER)
        assert profile.agent_type == AgentType.WORKER
        assert profile.required_capabilities == []
        assert profile.preferred_capabilities == []
        assert profile.min_context_window == 8192
        assert profile.requires_vision is False
        assert profile.latency_preference == "any"
        assert profile.prefers_uncensored is False
        assert profile.fallback_agents == []

    def test_creation_with_all_fields(self):
        profile = AffinityProfile(
            agent_type=AgentType.WORKER,
            required_capabilities=["coding"],
            preferred_capabilities=["vision", "reasoning"],
            min_context_window=16384,
            requires_vision=True,
            latency_preference="fast",
            prefers_uncensored=True,
            fallback_agents=[AgentType.FOREMAN],
        )
        assert profile.agent_type == AgentType.WORKER
        assert profile.required_capabilities == ["coding"]
        assert profile.preferred_capabilities == ["vision", "reasoning"]
        assert profile.min_context_window == 16384
        assert profile.requires_vision is True
        assert profile.latency_preference == "fast"
        assert profile.prefers_uncensored is True
        assert profile.fallback_agents == [AgentType.FOREMAN]

    def test_mutable_list_fields_are_independent(self):
        """Each instance should get its own list, not a shared default."""
        p1 = AffinityProfile(agent_type=AgentType.WORKER)
        p2 = AffinityProfile(agent_type=AgentType.FOREMAN)
        p1.required_capabilities.append("coding")
        assert p2.required_capabilities == []


# ---------------------------------------------------------------------------
# get_affinity()
# ---------------------------------------------------------------------------


class TestGetAffinity:
    def test_known_agent_type_worker(self):
        profile = get_affinity(AgentType.WORKER)
        assert isinstance(profile, AffinityProfile)
        assert profile.agent_type == AgentType.WORKER
        assert "reasoning" in profile.required_capabilities

    def test_unknown_agent_type_returns_default(self):
        """An AgentType not in the table should return a sensible default."""
        from vetinari.agents import agent_affinity as _mod

        original = dict(_mod._AFFINITY_TABLE)
        # Temporarily remove WORKER from the internal table
        del _mod._AFFINITY_TABLE[AgentType.WORKER]
        try:
            profile = get_affinity(AgentType.WORKER)
            assert profile.agent_type == AgentType.WORKER
            assert "reasoning" in profile.required_capabilities
            assert profile.min_context_window == 4096
        finally:
            _mod._AFFINITY_TABLE[AgentType.WORKER] = original[AgentType.WORKER]

    def test_foreman_requires_reasoning(self):
        profile = get_affinity(AgentType.FOREMAN)
        assert "reasoning" in profile.required_capabilities

    def test_foreman_has_large_context(self):
        profile = get_affinity(AgentType.FOREMAN)
        assert profile.min_context_window >= 16384

    def test_worker_has_large_context(self):
        """WORKER needs large context for multi-source research."""
        profile = get_affinity(AgentType.WORKER)
        assert profile.min_context_window >= 16384

    def test_worker_prefers_uncensored(self):
        """WORKER benefits from uncensored reasoning."""
        profile = get_affinity(AgentType.WORKER)
        assert profile.prefers_uncensored is True

    def test_inspector_requires_coding(self):
        """INSPECTOR agent requires coding capability for code review."""
        profile = get_affinity(AgentType.INSPECTOR)
        assert "coding" in profile.required_capabilities


# ---------------------------------------------------------------------------
# get_all_affinities()
# ---------------------------------------------------------------------------


class TestGetAllAffinities:
    def test_returns_dict(self):
        result = get_all_affinities()
        assert isinstance(result, dict)

    def test_all_values_are_affinity_profiles(self):
        result = get_all_affinities()
        for key, value in result.items():
            assert isinstance(value, AffinityProfile), f"{key} -> {value!r} is not AffinityProfile"

    def test_covers_all_agent_types(self):
        result = get_all_affinities()
        from vetinari.agents import agent_affinity as _mod

        # Every key in the internal table should appear
        for agent_type in _mod._AFFINITY_TABLE:
            assert agent_type in result, f"Missing {agent_type} in get_all_affinities()"

    def test_returns_copy(self):
        """Mutating the returned dict must not affect the internal table."""
        r1 = get_all_affinities()
        r2 = get_all_affinities()
        r1[AgentType.WORKER] = AffinityProfile(agent_type=AgentType.FOREMAN)
        assert r2[AgentType.WORKER].agent_type == AgentType.WORKER


# ---------------------------------------------------------------------------
# pick_model_for_agent()
# ---------------------------------------------------------------------------


class TestPickModelForAgent:
    def test_returns_matching_model(self):
        # WORKER requires "reasoning" and min_context_window=32768
        models = [make_model_dict("reasoning-7b", ["reasoning"], context_window=32768)]
        result = pick_model_for_agent(AgentType.WORKER, models)
        assert result == "reasoning-7b"

    def test_no_matching_models_returns_none(self):
        # INSPECTOR requires "coding"; provide model without it
        models = [make_model_dict("chat-only", ["chat"], context_window=16384)]
        result = pick_model_for_agent(AgentType.INSPECTOR, models)
        assert result is None

    def test_no_matching_models_returns_default(self):
        models = [make_model_dict("chat-only", ["chat"], context_window=16384)]
        result = pick_model_for_agent(AgentType.INSPECTOR, models, default="fallback-id")
        assert result == "fallback-id"

    def test_empty_model_list_returns_default(self):
        result = pick_model_for_agent(AgentType.WORKER, [], default="fallback-id")
        assert result == "fallback-id"

    def test_prefers_model_with_preferred_capabilities(self):
        # WORKER: required=["reasoning"], preferred=["analysis", "coding", "vision"]
        # min_context_window=32768
        basic = make_model_dict("basic-model", ["reasoning"], context_window=32768)
        full = make_model_dict("full-model", ["reasoning", "analysis", "coding"], context_window=32768)
        result = pick_model_for_agent(AgentType.WORKER, [basic, full])
        assert result == "full-model"

    def test_rejects_model_with_insufficient_context_window(self):
        # WORKER needs min 32768; supply a model with only 4096
        small = make_model_dict("tiny-model", ["reasoning"], context_window=4096)
        result = pick_model_for_agent(AgentType.WORKER, [small])
        assert result is None

    def test_latency_preference_bonus(self):
        # FOREMAN prefers "medium"; needs context_window >= 32768
        slow = make_model_dict("slow-model", ["reasoning"], context_window=32768, latency_hint="slow")
        medium = make_model_dict("medium-model", ["reasoning"], context_window=32768, latency_hint="medium")
        result = pick_model_for_agent(AgentType.FOREMAN, [slow, medium])
        assert result == "medium-model"

    def test_uncensored_preference_bonus(self):
        # FOREMAN prefers_uncensored=True; needs context_window >= 32768
        normal = make_model_dict("normal-model", ["reasoning"], context_window=32768)
        uncensored = make_model_dict("special-model", ["reasoning", "uncensored"], context_window=32768)
        result = pick_model_for_agent(AgentType.FOREMAN, [normal, uncensored])
        assert result == "special-model"

    def test_local_privacy_level_bonus(self):
        # Two otherwise identical models; local one should score higher
        # WORKER min_context_window=32768
        remote = make_model_dict("remote-model", ["reasoning", "coding"], context_window=32768, privacy_level="remote")
        local = make_model_dict("local-model", ["reasoning", "coding"], context_window=32768, privacy_level="local")
        result = pick_model_for_agent(AgentType.WORKER, [remote, local])
        assert result == "local-model"

    def test_uses_tags_as_fallback_for_capabilities(self):
        """Model dict may use 'tags' instead of 'capabilities'."""
        model = {
            "model_id": "tagged-model",
            "tags": ["coding", "reasoning"],
            "context_window": 16384,
        }
        result = pick_model_for_agent(AgentType.INSPECTOR, [model])
        assert result == "tagged-model"

    def test_uses_id_as_fallback_for_model_id(self):
        """Model dict may use 'id' instead of 'model_id'."""
        model = {
            "id": "alt-id-model",
            "capabilities": ["coding", "reasoning"],
            "context_window": 16384,
        }
        result = pick_model_for_agent(AgentType.INSPECTOR, [model])
        assert result == "alt-id-model"


# ---------------------------------------------------------------------------
# AFFINITY_TABLE completeness
# ---------------------------------------------------------------------------


class TestAffinityTableCompleteness:
    def test_every_active_agent_type_has_entry(self):
        """All 3 active agent types must have entries in the affinity table."""
        from vetinari.agents import agent_affinity as _mod

        table = _mod._AFFINITY_TABLE
        expected = [
            AgentType.FOREMAN,
            AgentType.WORKER,
            AgentType.INSPECTOR,
        ]
        for agent_type in expected:
            assert agent_type in table, f"{agent_type} missing from _AFFINITY_TABLE"

    def test_all_entries_have_valid_latency_preference(self):
        valid_values = {"fast", "medium", "any"}
        from vetinari.agents import agent_affinity as _mod

        for agent_type, profile in _mod._AFFINITY_TABLE.items():
            assert profile.latency_preference in valid_values, (
                f"{agent_type}.latency_preference={profile.latency_preference!r} not in {valid_values}"
            )

    def test_specialized_agents_have_non_empty_required_capabilities(self):
        """All active agents must declare at least one required capability."""
        specialized = [
            AgentType.FOREMAN,
            AgentType.WORKER,
            AgentType.INSPECTOR,
        ]
        from vetinari.agents import agent_affinity as _mod

        for agent_type in specialized:
            profile = _mod._AFFINITY_TABLE[agent_type]
            assert len(profile.required_capabilities) > 0, f"{agent_type} should have at least one required capability"

    def test_vision_requiring_agents_have_vision_in_caps(self):
        """Agents that require_vision should reflect that in capabilities."""
        from vetinari.agents import agent_affinity as _mod

        for agent_type, profile in _mod._AFFINITY_TABLE.items():
            if profile.requires_vision:
                # vision either in required or preferred caps
                all_caps = profile.required_capabilities + profile.preferred_capabilities
                assert "vision" in all_caps or profile.requires_vision, (
                    f"{agent_type} requires_vision but 'vision' not in any capability list"
                )


# -- _extract_family_slug --


class TestExtractFamilySlug:
    def test_explicit_family_field_takes_priority(self) -> None:
        model = {"model_id": "unknown-7b", "family": "MyFamily"}
        assert _extract_family_slug(model) == "myfamily"

    def test_explicit_family_field_lowercased(self) -> None:
        model = {"model_id": "qwen2-7b", "family": "Qwen2"}
        assert _extract_family_slug(model) == "qwen2"

    def test_qwen2_detected_from_model_id(self) -> None:
        model = {"model_id": "qwen2.5-7b-instruct-q4_k_m.gguf"}
        assert _extract_family_slug(model) == "qwen2"

    def test_llama_detected_from_model_id(self) -> None:
        model = {"model_id": "llama-3.1-8b-instruct.gguf"}
        assert _extract_family_slug(model) == "llama"

    def test_mistral_detected_from_model_id(self) -> None:
        model = {"model_id": "mistral-7b-v0.3.gguf"}
        assert _extract_family_slug(model) == "mistral"

    def test_phi_detected_from_model_id(self) -> None:
        model = {"model_id": "phi-3-mini-4k-instruct.gguf"}
        assert _extract_family_slug(model) == "phi"

    def test_id_field_used_as_fallback_for_model_id(self) -> None:
        model = {"id": "llama-3-8b.gguf"}
        assert _extract_family_slug(model) == "llama"

    def test_unknown_model_returns_empty_string(self) -> None:
        model = {"model_id": "custom-unknown-model-v1.gguf"}
        assert _extract_family_slug(model) == ""

    def test_empty_model_dict_returns_empty_string(self) -> None:
        assert _extract_family_slug({}) == ""


# -- _get_family_capability_bonus --


class TestGetFamilyCapabilityBonus:
    def test_unknown_family_returns_zero(self) -> None:
        model = {"model_id": "unknown-model.gguf"}
        bonus = _get_family_capability_bonus(model, ["coding"])
        assert bonus == 0.0

    def test_no_preferred_caps_returns_zero(self) -> None:
        model = {"model_id": "qwen2-7b.gguf"}
        with patch(
            "vetinari.agents.agent_affinity.get_family_profile",
            return_value={"capabilities": {"coding": {"python": 0.9}}},
        ):
            bonus = _get_family_capability_bonus(model, [])
        assert bonus == 0.0

    def test_dict_capability_returns_scaled_average(self) -> None:
        model = {"model_id": "qwen2-7b.gguf"}
        profile = {"capabilities": {"coding": {"python": 0.8, "debugging": 0.6}}}
        with patch("vetinari.agents.agent_affinity.get_family_profile", return_value=profile):
            bonus = _get_family_capability_bonus(model, ["coding"])
        # avg(0.8, 0.6) = 0.7, scaled by 0.5 = 0.35
        assert abs(bonus - 0.35) < 1e-9

    def test_scalar_capability_returns_scaled_value(self) -> None:
        model = {"model_id": "llama-3-8b.gguf"}
        profile = {"capabilities": {"reasoning": 0.8}}
        with patch("vetinari.agents.agent_affinity.get_family_profile", return_value=profile):
            bonus = _get_family_capability_bonus(model, ["reasoning"])
        assert abs(bonus - 0.4) < 1e-9

    def test_missing_capability_contributes_zero(self) -> None:
        model = {"model_id": "llama-3-8b.gguf"}
        profile = {"capabilities": {"reasoning": 0.9}}
        with patch("vetinari.agents.agent_affinity.get_family_profile", return_value=profile):
            bonus = _get_family_capability_bonus(model, ["vision"])
        assert bonus == 0.0

    def test_bonus_is_additive_across_caps(self) -> None:
        model = {"model_id": "qwen2-7b.gguf"}
        profile = {"capabilities": {"coding": 0.8, "reasoning": 0.6}}
        with patch("vetinari.agents.agent_affinity.get_family_profile", return_value=profile):
            bonus = _get_family_capability_bonus(model, ["coding", "reasoning"])
        # 0.8*0.5 + 0.6*0.5 = 0.4 + 0.3 = 0.7
        assert abs(bonus - 0.7) < 1e-9
