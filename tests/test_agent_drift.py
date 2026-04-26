"""Tests for agent drift detector — sliding window, drift detection, anchoring."""

from __future__ import annotations

import vetinari.agents.drift_detector as dd_mod
from vetinari.agents.drift_detector import (
    DEFAULT_CONSECUTIVE_THRESHOLD,
    DEFAULT_DRIFT_WINDOW,
    AgentDriftDetector,
    DriftReport,
    get_drift_detector,
)
from vetinari.types import AgentType

_SESSION = "test-session-001"
_AGENT = AgentType.WORKER


def _fresh_detector(**kwargs) -> AgentDriftDetector:
    """Create a fresh detector with optional overrides."""
    return AgentDriftDetector(**kwargs)


# -- test_no_drift_stable_scores ----------------------------------------------


def test_no_drift_stable_scores() -> None:
    """Recording 10 stable scores at 0.8 produces no drift report."""
    detector = _fresh_detector()

    for _ in range(10):
        detector.record_score(_AGENT, _SESSION, 0.8)

    report = detector.check_drift(_AGENT, _SESSION)
    assert report is None


# -- test_drift_detected_declining_scores -------------------------------------


def test_drift_detected_declining_scores() -> None:
    """Consecutive scores well below the session mean return a DriftReport."""
    detector = _fresh_detector(
        window_size=10,
        consecutive_threshold=3,
        drift_margin=0.1,
    )

    # Establish a high mean with 7 good scores
    for _ in range(7):
        detector.record_score(_AGENT, _SESSION, 0.9)

    # Add 3 scores that are far below mean (0.9) minus margin (0.1) = 0.8 threshold
    for _ in range(3):
        detector.record_score(_AGENT, _SESSION, 0.5)

    report = detector.check_drift(_AGENT, _SESSION)

    assert report is not None
    assert isinstance(report, DriftReport)
    assert report.agent_type == _AGENT
    assert report.session_id == _SESSION
    assert report.drift_magnitude > 0


# -- test_consecutive_threshold -----------------------------------------------


def test_consecutive_threshold() -> None:
    """Drift is not triggered until exactly N consecutive below-mean scores appear."""
    threshold = 3
    detector = _fresh_detector(
        window_size=10,
        consecutive_threshold=threshold,
        drift_margin=0.1,
    )
    session = "thresh-session"

    # High baseline
    for _ in range(7):
        detector.record_score(_AGENT, session, 0.9)

    # Only N-1 low scores — should not trigger yet
    for _ in range(threshold - 1):
        detector.record_score(_AGENT, session, 0.4)

    assert detector.check_drift(_AGENT, session) is None

    # The Nth low score tips it over
    detector.record_score(_AGENT, session, 0.4)
    assert detector.check_drift(_AGENT, session) is not None


# -- test_reset_session_clears_scores -----------------------------------------


def test_reset_session_clears_scores() -> None:
    """reset_session() wipes all tracked scores for that agent/session pair."""
    detector = _fresh_detector()
    session = "reset-session"

    for _ in range(5):
        detector.record_score(_AGENT, session, 0.8)

    assert len(detector.get_session_scores(_AGENT, session)) == 5

    detector.reset_session(_AGENT, session)

    assert detector.get_session_scores(_AGENT, session) == []
    assert detector.check_drift(_AGENT, session) is None


# -- test_get_session_scores_returns_window -----------------------------------


def test_get_session_scores_returns_window() -> None:
    """get_session_scores() returns the current window of recorded scores."""
    detector = _fresh_detector(window_size=5)
    session = "scores-session"
    scores = [0.7, 0.8, 0.75, 0.6, 0.9]

    for s in scores:
        detector.record_score(_AGENT, session, s)

    retrieved = detector.get_session_scores(_AGENT, session)
    assert retrieved == scores


def test_get_session_scores_window_trimmed() -> None:
    """Scores beyond window_size are dropped from the oldest end."""
    window = 4
    detector = _fresh_detector(window_size=window)
    session = "trim-session"
    all_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    for s in all_scores:
        detector.record_score(_AGENT, session, s)

    retrieved = detector.get_session_scores(_AGENT, session)
    assert len(retrieved) == window
    assert retrieved == all_scores[-window:]


# -- test_drift_margin_configurable -------------------------------------------


def test_drift_margin_configurable() -> None:
    """A tighter drift_margin triggers drift on smaller score drops."""
    tight_margin = 0.01
    loose_margin = 0.5
    session_tight = "tight-margin"
    session_loose = "loose-margin"

    tight = _fresh_detector(consecutive_threshold=3, drift_margin=tight_margin)
    loose = _fresh_detector(consecutive_threshold=3, drift_margin=loose_margin)

    # Establish identical baseline for both
    for _ in range(6):
        tight.record_score(_AGENT, session_tight, 0.8)
        loose.record_score(_AGENT, session_loose, 0.8)

    # A small drop: 0.77 — below 0.8 - 0.01 = 0.79, but above 0.8 - 0.5 = 0.3
    for _ in range(3):
        tight.record_score(_AGENT, session_tight, 0.77)
        loose.record_score(_AGENT, session_loose, 0.77)

    # Tight margin detects drift; loose margin does not
    assert tight.check_drift(_AGENT, session_tight) is not None
    assert loose.check_drift(_AGENT, session_loose) is None


# -- test_singleton_pattern ---------------------------------------------------


def test_singleton_pattern() -> None:
    """get_drift_detector() always returns the same AgentDriftDetector instance."""
    # Reset singleton to ensure clean test isolation
    dd_mod._instance = None

    first = get_drift_detector()
    second = get_drift_detector()

    assert first is second
    assert isinstance(first, AgentDriftDetector)

    # Cleanup: reset after test so other tests are not affected
    dd_mod._instance = None


# -- test_drift_callback_not_refired_on_same_window ---------------------------


def test_drift_callback_not_refired_on_same_window() -> None:
    """check_drift() does NOT re-fire the callback when the window content is unchanged.

    Repeated calls to check_drift() with no new scores recorded must fire the
    callback exactly once.  Re-firing on the same stale window causes duplicate
    anchoring cycles and inflates drift metrics.
    """
    fired: list[int] = []
    detector = _fresh_detector(
        window_size=10,
        consecutive_threshold=3,
        drift_margin=0.1,
        on_drift_detected=lambda _agent, _report: fired.append(1),
    )
    session = "idem-drift-session"

    # Establish high baseline then trigger drift
    for _ in range(7):
        detector.record_score(_AGENT, session, 0.9)
    for _ in range(3):
        detector.record_score(_AGENT, session, 0.4)

    # First check fires the callback
    r1 = detector.check_drift(_AGENT, session)
    assert r1 is not None
    assert len(fired) == 1

    # Repeated checks with the same window must NOT re-fire
    r2 = detector.check_drift(_AGENT, session)
    assert r2 is not None
    assert len(fired) == 1, (
        f"Callback fired {len(fired)} times on identical window — expected exactly 1"
    )

    r3 = detector.check_drift(_AGENT, session)
    assert r3 is not None
    assert len(fired) == 1

    # After a new score is added the window changes, so the callback fires again
    detector.record_score(_AGENT, session, 0.3)
    r4 = detector.check_drift(_AGENT, session)
    assert r4 is not None
    assert len(fired) == 2, (
        f"Expected callback re-fire after new score; got {len(fired)} total fires"
    )


def test_reset_session_clears_drift_dedup_state() -> None:
    """reset_session() clears the dedup window so drift is re-detectable after reset."""
    fired: list[int] = []
    detector = _fresh_detector(
        window_size=10,
        consecutive_threshold=3,
        drift_margin=0.1,
        on_drift_detected=lambda _agent, _report: fired.append(1),
    )
    session = "reset-dedup-session"

    for _ in range(7):
        detector.record_score(_AGENT, session, 0.9)
    for _ in range(3):
        detector.record_score(_AGENT, session, 0.4)

    detector.check_drift(_AGENT, session)
    assert len(fired) == 1

    # After reset, same scores re-establish drift and the callback fires again
    detector.reset_session(_AGENT, session)
    for _ in range(7):
        detector.record_score(_AGENT, session, 0.9)
    for _ in range(3):
        detector.record_score(_AGENT, session, 0.4)

    detector.check_drift(_AGENT, session)
    assert len(fired) == 2, (
        "reset_session() must clear dedup state so drift is re-detectable"
    )
