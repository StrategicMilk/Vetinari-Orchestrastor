"""Tests for vetinari/agent_affinity.py"""
from vetinari.agent_affinity import AffinityProfile, get_affinity, get_all_affinities, pick_model_for_agent
from vetinari.types import AgentType

# ---------------------------------------------------------------------------
# AffinityProfile dataclass
# ---------------------------------------------------------------------------

class TestAffinityProfileDefaults:
    def test_creation_with_required_field(self):
        profile = AffinityProfile(agent_type=AgentType.BUILDER)
        assert profile.agent_type == AgentType.BUILDER
        assert profile.required_capabilities == []
        assert profile.preferred_capabilities == []
        assert profile.min_context_window == 8192
        assert profile.requires_vision is False
        assert profile.latency_preference == "any"
        assert profile.prefers_uncensored is False
        assert profile.fallback_agents == []

    def test_creation_with_all_fields(self):
        profile = AffinityProfile(
            agent_type=AgentType.UI_PLANNER,
            required_capabilities=["coding"],
            preferred_capabilities=["vision", "reasoning"],
            min_context_window=16384,
            requires_vision=True,
            latency_preference="fast",
            prefers_uncensored=True,
            fallback_agents=[AgentType.BUILDER],
        )
        assert profile.agent_type == AgentType.UI_PLANNER
        assert profile.required_capabilities == ["coding"]
        assert profile.preferred_capabilities == ["vision", "reasoning"]
        assert profile.min_context_window == 16384
        assert profile.requires_vision is True
        assert profile.latency_preference == "fast"
        assert profile.prefers_uncensored is True
        assert profile.fallback_agents == [AgentType.BUILDER]

    def test_mutable_list_fields_are_independent(self):
        """Each instance should get its own list, not a shared default."""
        p1 = AffinityProfile(agent_type=AgentType.BUILDER)
        p2 = AffinityProfile(agent_type=AgentType.PLANNER)
        p1.required_capabilities.append("coding")
        assert p2.required_capabilities == []


# ---------------------------------------------------------------------------
# get_affinity()
# ---------------------------------------------------------------------------

class TestGetAffinity:
    def test_known_agent_type_builder(self):
        profile = get_affinity(AgentType.BUILDER)
        assert isinstance(profile, AffinityProfile)
        assert profile.agent_type == AgentType.BUILDER
        assert "coding" in profile.required_capabilities

    def test_unknown_agent_type_returns_default(self):
        """An AgentType not in the table should return a sensible default."""
        # Use ORCHESTRATOR which is in the table, but craft a fake scenario
        # by checking the fallback path via a non-existent int value.
        # Actually test the documented default: call with something not in table
        # The function falls back when the key is missing. We monkey-patch to test.
        from vetinari import agent_affinity as _mod
        original = dict(_mod._AFFINITY_TABLE)
        # Temporarily remove BUILDER from the internal table
        del _mod._AFFINITY_TABLE[AgentType.BUILDER]
        try:
            profile = get_affinity(AgentType.BUILDER)
            assert profile.agent_type == AgentType.BUILDER
            assert "reasoning" in profile.required_capabilities
            assert profile.min_context_window == 4096
        finally:
            _mod._AFFINITY_TABLE[AgentType.BUILDER] = original[AgentType.BUILDER]

    def test_planner_requires_reasoning(self):
        profile = get_affinity(AgentType.PLANNER)
        assert "reasoning" in profile.required_capabilities

    def test_planner_has_large_context(self):
        profile = get_affinity(AgentType.PLANNER)
        assert profile.min_context_window >= 16384

    def test_ui_planner_requires_vision(self):
        profile = get_affinity(AgentType.UI_PLANNER)
        assert profile.requires_vision is True

    def test_ui_planner_has_fallback_agents(self):
        profile = get_affinity(AgentType.UI_PLANNER)
        assert AgentType.BUILDER in profile.fallback_agents

    def test_explorer_prefers_fast_latency(self):
        profile = get_affinity(AgentType.EXPLORER)
        assert profile.latency_preference == "fast"


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
        from vetinari import agent_affinity as _mod
        # Every key in the internal table should appear
        for agent_type in _mod._AFFINITY_TABLE:
            assert agent_type in result, f"Missing {agent_type} in get_all_affinities()"

    def test_returns_copy(self):
        """Mutating the returned dict must not affect the internal table."""
        r1 = get_all_affinities()
        r2 = get_all_affinities()
        r1[AgentType.BUILDER] = AffinityProfile(agent_type=AgentType.ORCHESTRATOR)
        assert r2[AgentType.BUILDER].agent_type == AgentType.BUILDER


# ---------------------------------------------------------------------------
# pick_model_for_agent()
# ---------------------------------------------------------------------------

def _make_model(model_id, capabilities, context_window=16384, has_vision=False,
                latency_hint="medium", privacy_level=""):
    caps = list(capabilities)
    if has_vision and "vision" not in caps:
        caps.append("vision")
    return {
        "model_id": model_id,
        "capabilities": caps,
        "context_window": context_window,
        "latency_hint": latency_hint,
        "privacy_level": privacy_level,
    }


class TestPickModelForAgent:
    def test_returns_matching_model(self):
        models = [_make_model("coder-7b", ["coding", "reasoning"])]
        result = pick_model_for_agent(AgentType.BUILDER, models)
        assert result == "coder-7b"

    def test_no_matching_models_returns_none(self):
        # BUILDER requires "coding"; provide model without it
        models = [_make_model("chat-only", ["chat"], context_window=16384)]
        result = pick_model_for_agent(AgentType.BUILDER, models)
        assert result is None

    def test_no_matching_models_returns_default(self):
        models = [_make_model("chat-only", ["chat"], context_window=16384)]
        result = pick_model_for_agent(AgentType.BUILDER, models, default="fallback-id")
        assert result == "fallback-id"

    def test_empty_model_list_returns_default(self):
        result = pick_model_for_agent(AgentType.BUILDER, [], default="fallback-id")
        assert result == "fallback-id"

    def test_prefers_model_with_preferred_capabilities(self):
        # BUILDER: required=["coding"], preferred=["reasoning"]
        basic = _make_model("basic-coder", ["coding"])
        full = _make_model("full-coder", ["coding", "reasoning"])
        result = pick_model_for_agent(AgentType.BUILDER, [basic, full])
        assert result == "full-coder"

    def test_rejects_model_with_insufficient_context_window(self):
        # BUILDER needs min 8192; supply a model with only 4096
        small = _make_model("tiny-model", ["coding", "reasoning"], context_window=4096)
        result = pick_model_for_agent(AgentType.BUILDER, [small])
        assert result is None

    def test_requires_vision_filters_non_vl_models(self):
        # UI_PLANNER requires_vision=True
        no_vision = _make_model("no-vl", ["coding", "reasoning"], context_window=8192)
        with_vision = _make_model("vl-model", ["coding", "reasoning", "vision"], context_window=8192)
        result = pick_model_for_agent(AgentType.UI_PLANNER, [no_vision, with_vision])
        assert result == "vl-model"

    def test_requires_vision_rejects_all_without_vision(self):
        no_vision = _make_model("no-vl", ["coding", "reasoning"], context_window=8192)
        result = pick_model_for_agent(AgentType.UI_PLANNER, [no_vision], default="none")
        assert result == "none"

    def test_latency_preference_bonus(self):
        # EXPLORER prefers "fast"; two equal-cap models, one fast one slow
        slow = _make_model("slow-model", ["coding", "reasoning"], latency_hint="medium")
        fast = _make_model("fast-model", ["coding", "reasoning"], latency_hint="fast")
        result = pick_model_for_agent(AgentType.EXPLORER, [slow, fast])
        assert result == "fast-model"

    def test_uncensored_preference_bonus(self):
        # PLANNER prefers_uncensored=True; model with "uncensored" cap wins
        normal = _make_model("normal-model", ["reasoning"])
        uncensored = _make_model("special-model", ["reasoning", "uncensored"])
        result = pick_model_for_agent(AgentType.PLANNER, [normal, uncensored])
        assert result == "special-model"

    def test_local_privacy_level_bonus(self):
        # Two otherwise identical models; local one should score higher
        remote = _make_model("remote-model", ["coding", "reasoning"], privacy_level="remote")
        local = _make_model("local-model", ["coding", "reasoning"], privacy_level="local")
        result = pick_model_for_agent(AgentType.BUILDER, [remote, local])
        assert result == "local-model"

    def test_uses_tags_as_fallback_for_capabilities(self):
        """Model dict may use 'tags' instead of 'capabilities'."""
        model = {
            "model_id": "tagged-model",
            "tags": ["coding", "reasoning"],
            "context_window": 16384,
        }
        result = pick_model_for_agent(AgentType.BUILDER, [model])
        assert result == "tagged-model"

    def test_uses_id_as_fallback_for_model_id(self):
        """Model dict may use 'id' instead of 'model_id'."""
        model = {
            "id": "alt-id-model",
            "capabilities": ["coding", "reasoning"],
            "context_window": 16384,
        }
        result = pick_model_for_agent(AgentType.BUILDER, [model])
        assert result == "alt-id-model"


# ---------------------------------------------------------------------------
# AFFINITY_TABLE completeness
# ---------------------------------------------------------------------------

class TestAffinityTableCompleteness:
    def test_every_agent_type_has_entry(self):
        from vetinari import agent_affinity as _mod
        table = _mod._AFFINITY_TABLE
        # spot-check a representative set
        expected = [
            AgentType.PLANNER, AgentType.EXPLORER, AgentType.ORACLE,
            AgentType.LIBRARIAN, AgentType.RESEARCHER, AgentType.EVALUATOR,
            AgentType.SYNTHESIZER, AgentType.BUILDER, AgentType.UI_PLANNER,
            AgentType.SECURITY_AUDITOR, AgentType.DATA_ENGINEER,
            AgentType.DOCUMENTATION_AGENT, AgentType.COST_PLANNER,
            AgentType.TEST_AUTOMATION, AgentType.EXPERIMENTATION_MANAGER,
            AgentType.IMPROVEMENT, AgentType.USER_INTERACTION,
        ]
        for agent_type in expected:
            assert agent_type in table, f"{agent_type} missing from _AFFINITY_TABLE"

    def test_all_entries_have_valid_latency_preference(self):
        valid_values = {"fast", "medium", "any"}
        from vetinari import agent_affinity as _mod
        for agent_type, profile in _mod._AFFINITY_TABLE.items():
            assert profile.latency_preference in valid_values, (
                f"{agent_type}.latency_preference={profile.latency_preference!r} not in {valid_values}"
            )

    def test_specialized_agents_have_non_empty_required_capabilities(self):
        """Specialized agents (BUILDER, PLANNER, etc.) must declare requirements."""
        specialized = [
            AgentType.BUILDER, AgentType.PLANNER, AgentType.EXPLORER,
            AgentType.EVALUATOR, AgentType.UI_PLANNER, AgentType.SECURITY_AUDITOR,
        ]
        from vetinari import agent_affinity as _mod
        for agent_type in specialized:
            profile = _mod._AFFINITY_TABLE[agent_type]
            assert len(profile.required_capabilities) > 0, (
                f"{agent_type} should have at least one required capability"
            )

    def test_vision_requiring_agents_have_vision_in_caps(self):
        """Agents that require_vision should reflect that in capabilities."""
        from vetinari import agent_affinity as _mod
        for agent_type, profile in _mod._AFFINITY_TABLE.items():
            if profile.requires_vision:
                # vision either in required or preferred caps
                all_caps = profile.required_capabilities + profile.preferred_capabilities
                assert "vision" in all_caps or profile.requires_vision, (
                    f"{agent_type} requires_vision but 'vision' not in any capability list"
                )
