"""Tests for Thompson Sampling mode selection (US-202, Dept 6, connection #77)."""

from __future__ import annotations

import threading

import pytest

from vetinari.learning.model_selector import ThompsonSamplingSelector
from vetinari.types import AgentType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def selector(tmp_path, monkeypatch):
    """Fresh ThompsonSamplingSelector with isolated state dir."""
    monkeypatch.setenv("VETINARI_STATE_DIR", str(tmp_path))
    return ThompsonSamplingSelector()


# ---------------------------------------------------------------------------
# select_mode
# ---------------------------------------------------------------------------


class TestSelectMode:
    def test_returns_valid_mode(self, selector):
        """select_mode always returns one of the provided candidates."""
        candidates = ["code_review", "security_audit", "test_generation"]
        result = selector.select_mode("QUALITY", "coding", candidates)
        assert result in candidates

    def test_empty_candidates_returns_default(self, selector):
        """select_mode returns 'default' when candidates is empty."""
        result = selector.select_mode("QUALITY", "coding", [])
        assert result == "default"

    def test_single_candidate_returned(self, selector):
        """select_mode returns the only candidate when list has one entry."""
        result = selector.select_mode(AgentType.WORKER.value, "general", ["build"])
        assert result == "build"

    def test_thread_safe(self, selector):
        """Concurrent select_mode calls do not raise or corrupt state."""
        candidates = ["mode_a", "mode_b", "mode_c"]
        errors: list[Exception] = []

        def run():
            try:
                for _ in range(20):
                    selector.select_mode(AgentType.FOREMAN.value, "general", candidates)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ---------------------------------------------------------------------------
# update_mode
# ---------------------------------------------------------------------------


class TestUpdateMode:
    def test_success_increases_alpha(self, selector):
        """update_mode with success=True increases the arm's alpha."""
        selector.select_mode("QUALITY", "coding", ["review"])  # create arm
        arm_key = "mode_QUALITY_review:coding"
        before = selector._arms[arm_key].alpha

        selector.update_mode("QUALITY", "coding", "review", quality_score=0.9, success=True)

        assert selector._arms[arm_key].alpha > before

    def test_failure_increases_beta(self, selector):
        """update_mode with success=False increases the arm's beta."""
        selector.select_mode("QUALITY", "coding", ["review"])
        arm_key = "mode_QUALITY_review:coding"
        before = selector._arms[arm_key].beta

        selector.update_mode("QUALITY", "coding", "review", quality_score=0.1, success=False)

        assert selector._arms[arm_key].beta > before

    def test_update_increments_total_pulls(self, selector):
        """update_mode increments total_pulls on the arm."""
        selector.select_mode(AgentType.WORKER.value, "general", ["build"])
        arm_key = "mode_WORKER_build:general"
        before = selector._arms[arm_key].total_pulls

        selector.update_mode(AgentType.WORKER.value, "general", "build", quality_score=0.8, success=True)

        assert selector._arms[arm_key].total_pulls == before + 1


# ---------------------------------------------------------------------------
# has_mode_data
# ---------------------------------------------------------------------------


class TestHasModeData:
    def test_returns_false_with_no_observations(self, selector):
        """has_mode_data returns False when no arms have been updated."""
        assert selector.has_mode_data("QUALITY", "coding") is False

    def test_returns_false_below_min_pulls(self, selector):
        """has_mode_data returns False when pulls are below TIER_MIN_PULLS."""
        pulls_needed = ThompsonSamplingSelector.TIER_MIN_PULLS
        for _ in range(pulls_needed - 1):
            selector.update_mode("QUALITY", "coding", "review", quality_score=0.8, success=True)

        assert selector.has_mode_data("QUALITY", "coding") is False

    def test_returns_true_at_min_pulls(self, selector):
        """has_mode_data returns True once an arm reaches TIER_MIN_PULLS."""
        pulls_needed = ThompsonSamplingSelector.TIER_MIN_PULLS
        for _ in range(pulls_needed):
            selector.update_mode("QUALITY", "coding", "review", quality_score=0.8, success=True)

        assert selector.has_mode_data("QUALITY", "coding") is True

    def test_different_agent_type_not_counted(self, selector):
        """has_mode_data ignores arms belonging to other agent types."""
        pulls_needed = ThompsonSamplingSelector.TIER_MIN_PULLS
        for _ in range(pulls_needed):
            selector.update_mode(AgentType.WORKER.value, "coding", "build", quality_score=0.8, success=True)

        assert selector.has_mode_data(AgentType.INSPECTOR.value, "coding") is False


# ---------------------------------------------------------------------------
# Arm key format
# ---------------------------------------------------------------------------


class TestArmKeyFormat:
    def test_mode_arm_key_format(self, selector):
        """Mode arms are stored with key 'mode_{agent}_{mode}:{task_type}'."""
        selector.select_mode(AgentType.WORKER.value, "coding", ["plan"])
        assert "mode_WORKER_plan:coding" in selector._arms

    def test_update_mode_creates_arm_with_correct_key(self, selector):
        """update_mode creates/updates arm using the canonical key format."""
        selector.update_mode("ORACLE", "reasoning", "analysis", quality_score=0.7, success=True)
        assert "mode_ORACLE_analysis:reasoning" in selector._arms


# ---------------------------------------------------------------------------
# Fallback integration: MultiModeAgent uses default when no Thompson data
# ---------------------------------------------------------------------------


class TestMultiModeAgentFallback:
    def test_no_thompson_data_uses_keyword_fallback(self, monkeypatch, tmp_path):
        """_resolve_mode falls back to keyword/default when Thompson has no data."""
        monkeypatch.setenv("VETINARI_STATE_DIR", str(tmp_path))

        from vetinari.agents.contracts import AgentTask
        from vetinari.agents.multi_mode_agent import MultiModeAgent
        from vetinari.types import AgentType

        class DummyAgent(MultiModeAgent):
            MODES = {"alpha": "_do_alpha", "beta": "_do_beta"}
            DEFAULT_MODE = "alpha"
            MODE_KEYWORDS = {"beta": ["beta", "second"]}

            def _do_alpha(self, task):
                return None  # stub — only _resolve_mode is tested

            def _do_beta(self, task):
                return None  # stub — only _resolve_mode is tested

        agent = DummyAgent(AgentType.WORKER)
        task = AgentTask(
            task_id="t1",
            agent_type=AgentType.WORKER,
            description="do something",
            prompt="do something",
            context={},
        )
        # No Thompson data -> should fall back to DEFAULT_MODE
        mode = agent._resolve_mode(task)
        assert mode == "alpha"

    def test_explicit_mode_in_context_bypasses_thompson(self, monkeypatch, tmp_path):
        """Explicit context["mode"] is always honoured regardless of Thompson data."""
        monkeypatch.setenv("VETINARI_STATE_DIR", str(tmp_path))

        from vetinari.agents.contracts import AgentTask
        from vetinari.agents.multi_mode_agent import MultiModeAgent
        from vetinari.types import AgentType

        class DummyAgent(MultiModeAgent):
            MODES = {"alpha": "_do_alpha", "beta": "_do_beta"}
            DEFAULT_MODE = "alpha"

            def _do_alpha(self, task):
                return None  # stub — only _resolve_mode is tested

            def _do_beta(self, task):
                return None  # stub — only _resolve_mode is tested

        # Seed enough data for Thompson to have an opinion
        fresh_selector = ThompsonSamplingSelector()
        pulls_needed = ThompsonSamplingSelector.TIER_MIN_PULLS
        for _ in range(pulls_needed):
            fresh_selector.update_mode(AgentType.WORKER.value, "general", "beta", quality_score=1.0, success=True)

        monkeypatch.setattr(
            "vetinari.agents.multi_mode_agent.get_thompson_selector",
            lambda: fresh_selector,
            raising=False,
        )

        agent = DummyAgent(AgentType.WORKER)
        task = AgentTask(
            task_id="t2",
            agent_type=AgentType.WORKER,
            description="do something",
            prompt="do something",
            context={"mode": "alpha"},
        )
        mode = agent._resolve_mode(task)
        # Explicit mode wins
        assert mode == "alpha"
