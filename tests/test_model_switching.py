"""Tests for vetinari.model_switching — mid-session model switching."""

from __future__ import annotations

import pytest

from vetinari.model_switching import (
    ModelSwitch,
    ModelSwitchConfig,
    ModelSwitcher,
    get_model_switcher,
)


class TestModelSwitch:
    """Tests for the ModelSwitch dataclass."""

    def test_fields(self):
        s = ModelSwitch(
            timestamp="2026-01-01T00:00:00",
            from_model="a",
            to_model="b",
            reason="manual",
        )
        assert s.from_model == "a"
        assert s.to_model == "b"
        assert s.context_preserved is True

    def test_context_preserved_default(self):
        s = ModelSwitch("t", "a", "b", "manual")
        assert s.context_preserved is True


class TestModelSwitchConfig:
    """Tests for ModelSwitchConfig defaults."""

    def test_defaults(self):
        cfg = ModelSwitchConfig()
        assert len(cfg.fallback_chain) == 3
        assert cfg.auto_fallback is True
        assert cfg.context_handoff is True
        assert cfg.max_switches_per_session == 10


class TestModelSwitcher:
    """Tests for the ModelSwitcher class."""

    def setup_method(self):
        self.switcher = ModelSwitcher()

    def test_initial_model_is_none(self):
        assert self.switcher.current_model is None

    def test_set_initial_model(self):
        self.switcher.set_initial_model("qwen2.5-coder-32b")
        assert self.switcher.current_model == "qwen2.5-coder-32b"

    def test_switch_updates_model(self):
        self.switcher.set_initial_model("model-a")
        result = self.switcher.switch("model-b", reason="manual")
        assert self.switcher.current_model == "model-b"
        assert result.from_model == "model-a"
        assert result.to_model == "model-b"
        assert result.reason == "manual"

    def test_switch_increments_history(self):
        self.switcher.set_initial_model("a")
        self.switcher.switch("b")
        self.switcher.switch("c")
        assert len(self.switcher.get_history()) == 2

    def test_max_switches_exceeded_raises(self):
        cfg = ModelSwitchConfig(max_switches_per_session=2)
        sw = ModelSwitcher(cfg)
        sw.set_initial_model("a")
        sw.switch("b")
        sw.switch("c")
        with pytest.raises(RuntimeError, match="Max model switches"):
            sw.switch("d")

    def test_fallback_follows_chain(self):
        cfg = ModelSwitchConfig(fallback_chain=["m1", "m2", "m3"])
        sw = ModelSwitcher(cfg)
        sw.set_initial_model("m1")
        result = sw.fallback()
        assert result is not None
        assert result.to_model == "m2"
        assert result.reason == "fallback"

    def test_fallback_end_of_chain_returns_none(self):
        cfg = ModelSwitchConfig(fallback_chain=["m1"])
        sw = ModelSwitcher(cfg)
        sw.set_initial_model("m1")
        result = sw.fallback()
        assert result is None

    def test_fallback_unknown_model_uses_first(self):
        cfg = ModelSwitchConfig(fallback_chain=["m1", "m2"])
        sw = ModelSwitcher(cfg)
        sw.set_initial_model("unknown-model")
        result = sw.fallback()
        assert result is not None
        assert result.to_model == "m1"

    def test_fallback_disabled_returns_none(self):
        cfg = ModelSwitchConfig(auto_fallback=False)
        sw = ModelSwitcher(cfg)
        sw.set_initial_model("a")
        assert sw.fallback() is None

    def test_summarize_context_empty(self):
        assert self.switcher.summarize_context([]) == ""

    def test_summarize_context_truncates_long_messages(self):
        msgs = [{"role": "user", "content": "x" * 300}]
        summary = self.switcher.summarize_context(msgs)
        assert "[user]" in summary
        assert "..." in summary


class TestGetModelSwitcher:
    """Tests for the singleton accessor."""

    def test_returns_model_switcher(self):
        sw = get_model_switcher()
        assert isinstance(sw, ModelSwitcher)
