"""Tests for mid-session model switching (Task 24)."""

import pytest

from vetinari.model_switching import (
    ModelSwitch,
    ModelSwitchConfig,
    ModelSwitcher,
    get_model_switcher,
)


# ---------------------------------------------------------------------------
# ModelSwitchConfig defaults
# ---------------------------------------------------------------------------

class TestModelSwitchConfig:
    def test_defaults(self):
        cfg = ModelSwitchConfig()
        assert cfg.auto_fallback is True
        assert cfg.context_handoff is True
        assert cfg.max_switches_per_session == 10
        assert len(cfg.fallback_chain) == 3
        assert "qwen2.5-coder-32b" in cfg.fallback_chain

    def test_custom_config(self):
        cfg = ModelSwitchConfig(
            fallback_chain=["a", "b"],
            auto_fallback=False,
            context_handoff=False,
            max_switches_per_session=5,
        )
        assert cfg.fallback_chain == ["a", "b"]
        assert cfg.auto_fallback is False
        assert cfg.context_handoff is False
        assert cfg.max_switches_per_session == 5


# ---------------------------------------------------------------------------
# ModelSwitcher creation and initial model
# ---------------------------------------------------------------------------

class TestModelSwitcherCreation:
    def test_create_default(self):
        s = ModelSwitcher()
        assert s.current_model is None

    def test_set_initial_model(self):
        s = ModelSwitcher()
        s.set_initial_model("model-a")
        assert s.current_model == "model-a"

    def test_create_with_config(self):
        cfg = ModelSwitchConfig(max_switches_per_session=3)
        s = ModelSwitcher(config=cfg)
        assert s._config.max_switches_per_session == 3


# ---------------------------------------------------------------------------
# Switch model (manual, fallback)
# ---------------------------------------------------------------------------

class TestModelSwitch:
    def test_manual_switch(self):
        s = ModelSwitcher()
        s.set_initial_model("model-a")
        result = s.switch("model-b", reason="manual")
        assert isinstance(result, ModelSwitch)
        assert result.from_model == "model-a"
        assert result.to_model == "model-b"
        assert result.reason == "manual"
        assert s.current_model == "model-b"

    def test_switch_from_none(self):
        s = ModelSwitcher()
        result = s.switch("model-x")
        assert result.from_model == "none"
        assert result.to_model == "model-x"

    def test_switch_preserves_context_by_default(self):
        s = ModelSwitcher()
        result = s.switch("model-x")
        assert result.context_preserved is True

    def test_switch_no_context_preservation(self):
        cfg = ModelSwitchConfig(context_handoff=False)
        s = ModelSwitcher(config=cfg)
        result = s.switch("model-x")
        assert result.context_preserved is False

    def test_switch_has_timestamp(self):
        s = ModelSwitcher()
        result = s.switch("model-x")
        assert result.timestamp  # non-empty string


# ---------------------------------------------------------------------------
# Fallback chain traversal
# ---------------------------------------------------------------------------

class TestFallbackChain:
    def test_fallback_from_first(self):
        cfg = ModelSwitchConfig(fallback_chain=["a", "b", "c"])
        s = ModelSwitcher(config=cfg)
        s.set_initial_model("a")
        result = s.fallback()
        assert result is not None
        assert result.to_model == "b"
        assert result.reason == "fallback"

    def test_fallback_chain_traversal(self):
        cfg = ModelSwitchConfig(fallback_chain=["a", "b", "c"])
        s = ModelSwitcher(config=cfg)
        s.set_initial_model("a")
        s.fallback()  # a -> b
        assert s.current_model == "b"
        s.fallback()  # b -> c
        assert s.current_model == "c"

    def test_fallback_end_of_chain(self):
        cfg = ModelSwitchConfig(fallback_chain=["a", "b"])
        s = ModelSwitcher(config=cfg)
        s.set_initial_model("b")
        result = s.fallback()
        assert result is None  # no more fallbacks

    def test_fallback_unknown_model_goes_to_first(self):
        cfg = ModelSwitchConfig(fallback_chain=["a", "b"])
        s = ModelSwitcher(config=cfg)
        s.set_initial_model("unknown")
        result = s.fallback()
        assert result is not None
        assert result.to_model == "a"

    def test_fallback_disabled(self):
        cfg = ModelSwitchConfig(auto_fallback=False)
        s = ModelSwitcher(config=cfg)
        s.set_initial_model("model-a")
        result = s.fallback()
        assert result is None

    def test_fallback_empty_chain(self):
        cfg = ModelSwitchConfig(fallback_chain=[])
        s = ModelSwitcher(config=cfg)
        s.set_initial_model("unknown")
        result = s.fallback()
        assert result is None


# ---------------------------------------------------------------------------
# Max switches limit
# ---------------------------------------------------------------------------

class TestMaxSwitches:
    def test_max_switches_exceeded(self):
        cfg = ModelSwitchConfig(max_switches_per_session=2)
        s = ModelSwitcher(config=cfg)
        s.switch("a")
        s.switch("b")
        with pytest.raises(RuntimeError, match="Max model switches"):
            s.switch("c")


# ---------------------------------------------------------------------------
# Switch history tracking
# ---------------------------------------------------------------------------

class TestSwitchHistory:
    def test_empty_history(self):
        s = ModelSwitcher()
        assert s.get_history() == []

    def test_history_after_switches(self):
        s = ModelSwitcher()
        s.set_initial_model("a")
        s.switch("b")
        s.switch("c")
        history = s.get_history()
        assert len(history) == 2
        assert history[0]["from_model"] == "a"
        assert history[0]["to_model"] == "b"
        assert history[1]["from_model"] == "b"
        assert history[1]["to_model"] == "c"

    def test_history_includes_fallbacks(self):
        cfg = ModelSwitchConfig(fallback_chain=["x", "y"])
        s = ModelSwitcher(config=cfg)
        s.set_initial_model("x")
        s.fallback()
        history = s.get_history()
        assert len(history) == 1
        assert history[0]["reason"] == "fallback"


# ---------------------------------------------------------------------------
# Context summarization
# ---------------------------------------------------------------------------

class TestContextSummarization:
    def test_empty_messages(self):
        s = ModelSwitcher()
        assert s.summarize_context([]) == ""

    def test_short_messages(self):
        s = ModelSwitcher()
        msgs = [{"role": "user", "content": "hello"}]
        result = s.summarize_context(msgs)
        assert "[user] hello" in result

    def test_long_messages_truncated(self):
        s = ModelSwitcher()
        long_content = "x" * 300
        msgs = [{"role": "assistant", "content": long_content}]
        result = s.summarize_context(msgs)
        assert "..." in result
        assert len(result) < 300

    def test_last_10_messages_only(self):
        s = ModelSwitcher()
        msgs = [{"role": "user", "content": f"msg-{i}"} for i in range(20)]
        result = s.summarize_context(msgs)
        assert "msg-10" in result
        assert "msg-0" not in result


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

class TestGetModelSwitcher:
    def test_returns_instance(self):
        import vetinari.model_switching as mod
        mod._switcher = None  # reset
        s = get_model_switcher()
        assert isinstance(s, ModelSwitcher)

    def test_returns_same_instance(self):
        import vetinari.model_switching as mod
        mod._switcher = None
        a = get_model_switcher()
        b = get_model_switcher()
        assert a is b
