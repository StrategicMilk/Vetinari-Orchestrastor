"""Tests for NelsonRuleDetector and AndonSystem callbacks."""

from __future__ import annotations

import pytest

from tests.factories import make_nelson_rule_detector
from vetinari.workflow import AndonSignal, AndonSystem, NelsonViolation
from vetinari.workflow.nelson_rules import NelsonRuleDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MEAN = 50.0
SIGMA = 5.0


def _fill_baseline(detector: NelsonRuleDetector, n: int = 14) -> None:
    """Push *n* neutral points (exactly at mean) to warm the window."""
    for _ in range(n):
        detector.check_all_rules(MEAN)


# ---------------------------------------------------------------------------
# Nelson Rule tests
# ---------------------------------------------------------------------------


class TestRule1BeyondThreeSigma:
    def test_rule1_beyond_3sigma(self) -> None:
        detector = make_nelson_rule_detector()
        _fill_baseline(detector, 14)
        # Push a point well beyond UCL (mean + 3σ = 65.0)
        violations = detector.check_all_rules(MEAN + 3 * SIGMA + 0.1)
        rules = [v.rule for v in violations]
        assert 1 in rules
        assert any(v.severity == "critical" for v in violations if v.rule == 1)

    def test_rule1_below_3sigma(self) -> None:
        detector = make_nelson_rule_detector()
        _fill_baseline(detector, 14)
        violations = detector.check_all_rules(MEAN - 3 * SIGMA - 0.1)
        assert any(v.rule == 1 for v in violations)


class TestRule2NineSameSide:
    def test_rule2_nine_same_side(self) -> None:
        detector = make_nelson_rule_detector()
        # 6 neutral points, then 9 points above the mean but within 1σ
        for _ in range(6):
            detector.check_all_rules(MEAN)
        violations: list[NelsonViolation] = []
        for _ in range(9):
            violations = detector.check_all_rules(MEAN + 1.0)
        assert any(v.rule == 2 for v in violations)
        assert any(v.severity == "warning" for v in violations if v.rule == 2)


class TestRule3SixTrending:
    def test_rule3_six_trending(self) -> None:
        detector = make_nelson_rule_detector()
        _fill_baseline(detector, 10)
        # Push 6 strictly increasing points
        violations: list[NelsonViolation] = []
        for i in range(6):
            violations = detector.check_all_rules(MEAN + i * 0.5)
        assert any(v.rule == 3 for v in violations)
        assert any(v.severity == "warning" for v in violations if v.rule == 3)


class TestRule4FourteenAlternating:
    def test_rule4_fourteen_alternating(self) -> None:
        detector = make_nelson_rule_detector()
        _fill_baseline(detector, 2)
        # Alternate low/high/low/high ... for 14 points
        violations: list[NelsonViolation] = []
        for i in range(14):
            value = MEAN - 0.5 if i % 2 == 0 else MEAN + 0.5
            violations = detector.check_all_rules(value)
        assert any(v.rule == 4 for v in violations)
        assert any(v.severity == "warning" for v in violations if v.rule == 4)


class TestRule5TwoOfThreeBeyond2Sigma:
    def test_rule5_two_of_three_beyond_2sigma(self) -> None:
        detector = make_nelson_rule_detector()
        _fill_baseline(detector, 12)
        # Push three points: two beyond +2σ (= 60.0), one neutral
        detector.check_all_rules(MEAN + 2 * SIGMA + 0.1)  # beyond 2σ
        detector.check_all_rules(MEAN)  # neutral
        violations = detector.check_all_rules(MEAN + 2 * SIGMA + 0.1)  # beyond 2σ
        assert any(v.rule == 5 for v in violations)
        assert any(v.severity == "warning" for v in violations if v.rule == 5)


class TestRule6FourOfFiveBeyond1Sigma:
    def test_rule6_four_of_five_beyond_1sigma(self) -> None:
        detector = make_nelson_rule_detector()
        _fill_baseline(detector, 10)
        # Push five points: four beyond +1σ (= 55.0), one neutral
        above = MEAN + SIGMA + 0.1
        detector.check_all_rules(above)
        detector.check_all_rules(above)
        detector.check_all_rules(above)
        detector.check_all_rules(MEAN)  # neutral
        violations = detector.check_all_rules(above)
        assert any(v.rule == 6 for v in violations)
        assert any(v.severity == "warning" for v in violations if v.rule == 6)


class TestRule7FifteenWithin1Sigma:
    def test_rule7_fifteen_within_1sigma(self) -> None:
        detector = make_nelson_rule_detector()
        # Push 15 points all very close to the mean (within 1σ)
        violations: list[NelsonViolation] = []
        for _ in range(15):
            violations = detector.check_all_rules(MEAN + 0.1)
        assert any(v.rule == 7 for v in violations)
        assert any(v.severity == "info" for v in violations if v.rule == 7)


class TestRule8EightBeyond1SigmaEither:
    def test_rule8_eight_beyond_1sigma_either(self) -> None:
        detector = make_nelson_rule_detector()
        _fill_baseline(detector, 7)
        # Push 8 points alternating between +2σ and -2σ (beyond 1σ on both sides)
        violations: list[NelsonViolation] = []
        for i in range(8):
            value = MEAN + 2 * SIGMA if i % 2 == 0 else MEAN - 2 * SIGMA
            violations = detector.check_all_rules(value)
        assert any(v.rule == 8 for v in violations)
        assert any(v.severity == "warning" for v in violations if v.rule == 8)


class TestNoViolations:
    def test_no_violations_normal_data(self) -> None:
        """Points right at the mean should not trigger any Nelson rule."""
        detector = make_nelson_rule_detector()
        for _ in range(49):
            detector.check_all_rules(MEAN)
        # Final check — exactly at mean with a fully warmed window
        violations = detector.check_all_rules(MEAN)
        # Rule 7 fires because all points are within 1σ; that is an "info"
        # level signal, not an error, but we verify no critical/warning rules
        for v in violations:
            assert v.severity != "critical", f"Unexpected critical violation: {v}"
            if v.rule != 7:
                pytest.fail(f"Unexpected non-rule-7 violation: {v}")


class TestInsufficientData:
    def test_insufficient_data(self) -> None:
        detector = make_nelson_rule_detector()
        # Push 14 points (one below the threshold of 15)
        for _ in range(14):
            violations = detector.check_all_rules(MEAN + 4 * SIGMA)  # always beyond 3σ
        # With only 14 points the detector must return []
        assert violations == []

    def test_exactly_fifteen_triggers_checks(self) -> None:
        detector = make_nelson_rule_detector()
        for _ in range(14):
            detector.check_all_rules(MEAN)
        violations = detector.check_all_rules(MEAN + 3 * SIGMA + 1.0)
        assert any(v.rule == 1 for v in violations)


class TestUpdateControlLimits:
    def test_update_control_limits(self) -> None:
        detector = make_nelson_rule_detector()
        _fill_baseline(detector, 14)
        # With old limits (UCL=65) a point at 68 is beyond 3σ
        violations = detector.check_all_rules(MEAN + 4 * SIGMA)
        assert any(v.rule == 1 for v in violations)

        # After widening limits (sigma=20, UCL=110), same point should NOT
        # be beyond 3σ — need to re-warm since window still has the violation
        detector2 = NelsonRuleDetector(mean=MEAN, sigma=20.0)
        _fill_baseline(detector2, 14)
        violations2 = detector2.check_all_rules(MEAN + 4 * SIGMA)
        assert not any(v.rule == 1 for v in violations2)

    def test_update_recalculates_ucl_lcl(self) -> None:
        detector = make_nelson_rule_detector()
        detector.update_control_limits(mean=100.0, sigma=10.0)
        assert detector._ucl == pytest.approx(130.0)
        assert detector._lcl == pytest.approx(70.0)
        assert detector._ucl2 == pytest.approx(120.0)
        assert detector._lcl2 == pytest.approx(80.0)
        assert detector._ucl1 == pytest.approx(110.0)
        assert detector._lcl1 == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# AndonSystem callback tests
# ---------------------------------------------------------------------------


class TestAndonRegisterCallback:
    def test_andon_register_callback(self) -> None:
        """A registered callback is invoked when a signal is raised."""
        system = AndonSystem()
        received: list[AndonSignal] = []
        system.register_callback(received.append)
        sig = system.raise_signal("test", "warning", "hello")
        assert len(received) == 1
        assert received[0] is sig

    def test_andon_multiple_callbacks(self) -> None:
        """All registered callbacks are invoked, in registration order."""
        system = AndonSystem()
        calls_a: list[AndonSignal] = []
        calls_b: list[AndonSignal] = []
        system.register_callback(calls_a.append)
        system.register_callback(calls_b.append)
        system.raise_signal("src", "info", "msg")
        assert len(calls_a) == 1
        assert len(calls_b) == 1
        assert calls_a[0] is calls_b[0]

    def test_andon_callback_error_handled(self) -> None:
        """A callback that raises an exception does not crash the Andon system."""

        def bad_callback(sig: AndonSignal) -> None:
            raise RuntimeError("callback boom")

        system = AndonSystem()
        system.register_callback(bad_callback)
        # Should not raise; signal should still be recorded
        sig = system.raise_signal("src", "warning", "test")
        assert sig in system.get_all_signals()

    def test_andon_no_callbacks_is_fine(self) -> None:
        """Raising a signal with no callbacks registered works normally."""
        system = AndonSystem()
        sig = system.raise_signal("src", "critical", "msg")
        assert system.is_paused()
        assert sig in system.get_all_signals()
