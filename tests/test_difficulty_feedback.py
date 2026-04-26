"""Tests for vetinari/learning/difficulty_feedback.py

Covers:
- compute_observed_difficulty: signal-to-score mapping, clamping
- record_difficulty_feedback: persists metadata into episode_memory_store
- get_calibration_bias: aggregates historical deltas, handles missing data
- assess_difficulty with calibration_bias parameter
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from vetinari.learning.difficulty_feedback import (
    DifficultySignals,
    compute_observed_difficulty,
    get_calibration_bias,
    record_difficulty_feedback,
)
from vetinari.learning.episode_memory import get_episode_memory
from vetinari.models.model_router_scoring import assess_difficulty


@pytest.fixture(autouse=True)
def _reset_episode_memory_singleton(_isolate_database: None) -> Generator[None, None, None]:
    """Reset and reinitialize the EpisodeMemory singleton before each test.

    Depends on _isolate_database (conftest) to ensure get_connection() points
    at the per-test temp DB before we call get_episode_memory() to create the
    episode_memory_store table.
    """
    import vetinari.learning.episode_memory as _ep_mod

    _ep_mod._episode_memory = None
    _ep_mod.EpisodeMemory._instance = None

    # Reset the database module to ensure a fresh connection to the current temp DB
    # before _init_db() runs — necessary when sys.modules isolation may have replaced
    # the episode_memory module with a stale snapshot pointing at an old connection.
    from vetinari.database import reset_for_testing

    reset_for_testing()

    # Trigger _init_db() on the current temp database so episode_memory_store exists.
    get_episode_memory()

    # Safety net: directly create the table via get_connection() in case sys.modules
    # isolation replaced the EpisodeMemory module after _init_db() ran, leaving the
    # active connection without the table.
    from vetinari.database import get_connection

    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episode_memory_store (
            episode_id TEXT PRIMARY KEY,
            timestamp TEXT,
            task_summary TEXT,
            agent_type TEXT,
            task_type TEXT,
            output_summary TEXT,
            quality_score REAL,
            success INTEGER,
            model_id TEXT DEFAULT '',
            embedding TEXT,
            metadata TEXT DEFAULT '{}',
            importance REAL DEFAULT 0.5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_type ON episode_memory_store(task_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ep_score ON episode_memory_store(quality_score)")
    conn.commit()

    yield
    _ep_mod._episode_memory = None
    _ep_mod.EpisodeMemory._instance = None


# ---------------------------------------------------------------------------
# TestComputeObservedDifficulty
# ---------------------------------------------------------------------------


class TestComputeObservedDifficulty:
    """Unit tests for compute_observed_difficulty signal-to-score mapping."""

    def test_baseline_signals(self) -> None:
        """Minimal/empty signals should produce the 0.3 baseline score."""
        signals: DifficultySignals = {}
        result = compute_observed_difficulty(signals)
        assert abs(result - 0.3) < 1e-9

    def test_retries_increase_difficulty(self) -> None:
        """Non-zero retries should raise the score above the baseline."""
        low: DifficultySignals = {"retries": 1}
        high: DifficultySignals = {"retries": 3}
        assert compute_observed_difficulty(low) > 0.3
        assert compute_observed_difficulty(high) > compute_observed_difficulty(low)

    def test_retries_cap_at_0_3(self) -> None:
        """Retries contribution should be capped so large values don't overflow."""
        many: DifficultySignals = {"retries": 100}
        base: DifficultySignals = {}
        diff = compute_observed_difficulty(many) - compute_observed_difficulty(base)
        # Cap is 0.3 for retries alone
        assert abs(diff - 0.3) < 1e-9

    def test_rejection_increases_difficulty(self) -> None:
        """Inspection rejections should push difficulty above baseline."""
        signals: DifficultySignals = {"rejections": 2}
        assert compute_observed_difficulty(signals) > 0.3

    def test_duration_moderate_increase(self) -> None:
        """Duration > 30s (but < 2min) should add 0.1 to the score."""
        signals: DifficultySignals = {"duration_ms": 45_000.0}
        expected = 0.3 + 0.1
        assert abs(compute_observed_difficulty(signals) - expected) < 1e-9

    def test_duration_very_slow_increase(self) -> None:
        """Duration > 2min should add 0.2 to the score."""
        signals: DifficultySignals = {"duration_ms": 150_000.0}
        expected = 0.3 + 0.2
        assert abs(compute_observed_difficulty(signals) - expected) < 1e-9

    def test_low_quality_score_adds_difficulty(self) -> None:
        """Quality score below 0.5 should contribute additional difficulty."""
        signals: DifficultySignals = {"quality_score": 0.3}
        expected = 0.3 + 0.1
        assert abs(compute_observed_difficulty(signals) - expected) < 1e-9

    def test_high_quality_score_no_addition(self) -> None:
        """Quality score >= 0.5 should not change the score beyond baseline."""
        signals: DifficultySignals = {"quality_score": 0.8}
        assert abs(compute_observed_difficulty(signals) - 0.3) < 1e-9

    def test_failure_increases_difficulty(self) -> None:
        """success=False should add 0.15 to the difficulty score."""
        signals: DifficultySignals = {"success": False}
        expected = 0.3 + 0.15
        assert abs(compute_observed_difficulty(signals) - expected) < 1e-9

    def test_success_true_no_addition(self) -> None:
        """success=True should not alter the baseline score."""
        signals: DifficultySignals = {"success": True}
        assert abs(compute_observed_difficulty(signals) - 0.3) < 1e-9

    def test_clamps_to_one(self) -> None:
        """Combined extreme signals must not exceed 1.0."""
        signals: DifficultySignals = {
            "retries": 100,
            "rejections": 100,
            "duration_ms": 999_999.0,
            "quality_score": 0.0,
            "success": False,
        }
        result = compute_observed_difficulty(signals)
        assert result == 1.0

    def test_clamps_to_zero(self) -> None:
        """Score cannot go below 0.0 (sanity guard)."""
        # There is no signal that subtracts from the base, so this tests the
        # clamp on the lower bound is structurally safe.
        result = compute_observed_difficulty({})
        assert result >= 0.0

    def test_combined_signals(self) -> None:
        """Multiple signals combine additively within their individual caps."""
        signals: DifficultySignals = {
            "retries": 1,  # +0.15
            "quality_score": 0.4,  # +0.1
            "success": False,  # +0.15
        }
        expected = min(0.3 + 0.15 + 0.1 + 0.15, 1.0)
        result = compute_observed_difficulty(signals)
        assert abs(result - expected) < 1e-9


# ---------------------------------------------------------------------------
# TestRecordAndCalibrate — requires a live (temp) DB via conftest fixtures
# ---------------------------------------------------------------------------


class TestRecordAndCalibrate:
    """Integration tests for the full record → query calibration loop.

    The autouse _isolate_database fixture in conftest.py redirects
    get_connection() to a per-test temp SQLite database, so these tests
    do not touch the project's real database.
    """

    def test_record_updates_episode_metadata(self) -> None:
        """record_difficulty_feedback must write difficulty fields to the episode row."""
        mem = get_episode_memory()
        ep_id = mem.record(
            task_description="Implement a Redis cache wrapper",
            agent_type="worker",
            task_type="coding",
            output_summary="RedisCacheWrapper with get/set/delete",
            quality_score=0.85,
            success=True,
            model_id="test-model-7b",
        )

        signals: DifficultySignals = {
            "retries": 1,
            "rejections": 0,
            "duration_ms": 5_000.0,
            "quality_score": 0.85,
            "success": True,
        }
        record_difficulty_feedback(
            task_type="coding",
            predicted=0.4,
            signals=signals,
            episode_id=ep_id,
        )

        # Read back the row directly from the DB
        import json

        from vetinari.database import get_connection

        conn = get_connection()
        row = conn.execute(
            "SELECT metadata FROM episode_memory_store WHERE episode_id = ?",
            (ep_id,),
        ).fetchone()
        assert row is not None
        meta = json.loads(row[0])

        assert "predicted_difficulty" in meta
        assert "observed_difficulty" in meta
        assert "difficulty_delta" in meta
        assert abs(meta["predicted_difficulty"] - 0.4) < 1e-3
        # observed = baseline 0.3 + retries 0.15 = 0.45
        expected_observed = compute_observed_difficulty(signals)
        assert abs(meta["observed_difficulty"] - expected_observed) < 1e-3
        expected_delta = round(expected_observed - 0.4, 3)
        assert abs(meta["difficulty_delta"] - expected_delta) < 1e-3

    def test_record_with_no_episode_id_is_noop(self) -> None:
        """record_difficulty_feedback with episode_id=None must not raise."""
        signals: DifficultySignals = {"success": True, "retries": 0}
        assert record_difficulty_feedback(
            task_type="coding",
            predicted=0.5,
            signals=signals,
            episode_id=None,
        ) is None

    def test_calibration_bias_reflects_errors(self) -> None:
        """get_calibration_bias should return the mean delta across episodes (clamped)."""
        mem = get_episode_memory()

        for i in range(5):
            ep_id = mem.record(
                task_description=f"Security audit task {i}",
                agent_type="inspector",
                task_type="security",
                output_summary="Audit complete",
                quality_score=0.6,
                success=True,
                model_id="test-model-7b",
            )
            # Inject a known delta of +0.4 directly to bypass compute_observed_difficulty
            signals: DifficultySignals = {
                "retries": 1,  # +0.15
                "rejections": 1,  # +0.1
                "success": False,  # +0.15
            }
            # predicted=0.3 baseline, observed= 0.3+0.15+0.1+0.15 = 0.7, delta=0.4
            record_difficulty_feedback(
                task_type="security",
                predicted=0.3,
                signals=signals,
                episode_id=ep_id,
            )

        bias = get_calibration_bias("security")
        # Expected mean delta is 0.4, clamped to 0.3 max
        assert abs(bias - 0.3) < 1e-6

    def test_calibration_bias_no_data_returns_zero(self) -> None:
        """get_calibration_bias for an unknown task type must return 0.0."""
        bias = get_calibration_bias("nonexistent_task_type_xyz")
        assert bias == 0.0

    def test_calibration_bias_window_limits_lookback(self) -> None:
        """window parameter restricts how many episodes are considered."""
        mem = get_episode_memory()

        # Add 5 episodes with large positive delta
        for i in range(5):
            ep_id = mem.record(
                task_description=f"Hard task {i}",
                agent_type="worker",
                task_type="window_test_type",
                output_summary="Done",
                quality_score=0.5,
                success=False,
            )
            record_difficulty_feedback(
                task_type="window_test_type",
                predicted=0.3,
                signals={"success": False, "retries": 2},  # delta > 0
                episode_id=ep_id,
            )

        # window=3 should still return a non-zero bias (subset of episodes)
        bias_3 = get_calibration_bias("window_test_type", window=3)
        bias_10 = get_calibration_bias("window_test_type", window=10)
        assert bias_3 != 0.0
        assert bias_10 != 0.0

    def test_calibration_skips_episodes_without_delta(self) -> None:
        """Episodes without difficulty_delta in metadata are ignored gracefully."""
        mem = get_episode_memory()

        # Record an episode WITHOUT calling record_difficulty_feedback
        mem.record(
            task_description="Plain task without difficulty feedback",
            agent_type="worker",
            task_type="plain_task_type",
            output_summary="Done",
            quality_score=0.9,
            success=True,
        )

        # Should return 0.0 because no episode has difficulty_delta
        bias = get_calibration_bias("plain_task_type")
        assert bias == 0.0


# ---------------------------------------------------------------------------
# TestAssessDifficultyWithCalibration
# ---------------------------------------------------------------------------


class TestAssessDifficultyWithCalibration:
    """Tests for the calibration_bias parameter on assess_difficulty."""

    def test_positive_bias_increases_score(self) -> None:
        """calibration_bias > 0 should produce a higher difficulty score."""
        base = assess_difficulty("complex security audit", "security")
        biased = assess_difficulty("complex security audit", "security", calibration_bias=0.1)
        assert biased > base

    def test_negative_bias_decreases_score(self) -> None:
        """calibration_bias < 0 should produce a lower difficulty score."""
        base = assess_difficulty("write a README", "docs")
        biased = assess_difficulty("write a README", "docs", calibration_bias=-0.1)
        assert biased < base

    def test_zero_bias_matches_original(self) -> None:
        """calibration_bias=0.0 should produce the same score as the unbiased call."""
        desc = "refactor the authentication module"
        assert assess_difficulty(desc, "coding", calibration_bias=0.0) == assess_difficulty(desc, "coding")

    def test_score_still_clamped_to_one(self) -> None:
        """Even with a large positive bias the score must not exceed 1.0."""
        result = assess_difficulty("security audit vulnerability", "security", calibration_bias=1.0)
        assert result <= 1.0

    def test_score_clamped_to_zero_with_large_negative_bias(self) -> None:
        """Even with a large negative bias the score must not go below 0.0."""
        result = assess_difficulty("hi", "general", calibration_bias=-5.0)
        assert result >= 0.0
