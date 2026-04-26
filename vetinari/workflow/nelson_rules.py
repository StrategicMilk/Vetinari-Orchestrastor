"""Nelson Rule Detector — eight standard SPC rules for process control.

Detects violations of the eight Nelson SPC rules using a rolling window
of observations.  Each rule maps to a :class:`~vetinari.workflow.andon.NelsonViolation`.
"""

from __future__ import annotations

import logging
from collections import deque

from vetinari.workflow.andon import NelsonViolation

logger = logging.getLogger(__name__)


class NelsonRuleDetector:
    """Detect violations of the eight Nelson SPC rules.

    The detector maintains a rolling window of observations and checks all
    eight Nelson rules on each new data point.  At least 15 observations must
    be present before any violation is reported, preventing false positives
    during the warm-up phase.

    Usage::

        detector = NelsonRuleDetector(mean=50.0, sigma=5.0)
        violations = detector.check_all_rules(observation=65.0)
        for v in violations:
            logger.warning("Nelson rule %d violated: %s", v.rule, v.description)
    """

    def __init__(self, mean: float, sigma: float, window_size: int = 50) -> None:
        """Initialise the detector with known process parameters.

        Args:
            mean: The process mean (center line).
            sigma: The process standard deviation (one sigma).
            window_size: Maximum number of observations to retain in the
                rolling window.  Defaults to 50.
        """
        self._mean = mean
        self._sigma = sigma
        self._window: deque[float] = deque(maxlen=window_size)
        self._update_limits()

    # -- limit management ---------------------------------------------------

    def _update_limits(self) -> None:
        """Recompute the sigma-banded control limits from mean and sigma."""
        self._ucl = self._mean + 3 * self._sigma
        self._lcl = self._mean - 3 * self._sigma
        self._ucl2 = self._mean + 2 * self._sigma
        self._lcl2 = self._mean - 2 * self._sigma
        self._ucl1 = self._mean + self._sigma
        self._lcl1 = self._mean - self._sigma

    def update_control_limits(self, mean: float, sigma: float) -> None:
        """Update the process mean and sigma, then recompute all limits.

        Args:
            mean: New process mean.
            sigma: New process standard deviation.
        """
        self._mean = mean
        self._sigma = sigma
        self._update_limits()
        logger.debug(
            "Nelson limits updated: mean=%.4f sigma=%.4f UCL=%.4f LCL=%.4f",
            mean,
            sigma,
            self._ucl,
            self._lcl,
        )

    # -- main interface -----------------------------------------------------

    def check_all_rules(self, observation: float) -> list[NelsonViolation]:
        """Append *observation* and return any Nelson violations detected.

        Returns an empty list when the window contains fewer than 15 points
        because the Nelson rules require sufficient history to be meaningful.

        Args:
            observation: The latest process measurement.

        Returns:
            A list of :class:`NelsonViolation` instances, one per violated
            rule.  May be empty if the process is in control.
        """
        self._window.append(observation)
        if len(self._window) < 15:
            return []

        violations: list[NelsonViolation] = []
        pts = list(self._window)

        # Rule 1: one point beyond 3 sigma
        if self._check_rule1(pts):
            violations.append(
                NelsonViolation(
                    rule=1,
                    severity="critical",
                    description="One point beyond 3 sigma control limits",
                ),
            )

        # Rule 2: nine consecutive points on same side of center
        if self._check_rule2(pts):
            violations.append(
                NelsonViolation(
                    rule=2,
                    severity="warning",
                    description="Nine consecutive points on the same side of the center line",
                ),
            )

        # Rule 3: six consecutive points steadily increasing or decreasing
        if self._check_rule3(pts):
            violations.append(
                NelsonViolation(
                    rule=3,
                    severity="warning",
                    description="Six consecutive points steadily increasing or decreasing",
                ),
            )

        # Rule 4: fourteen consecutive alternating up/down
        if self._check_rule4(pts):
            violations.append(
                NelsonViolation(
                    rule=4,
                    severity="warning",
                    description="Fourteen consecutive points alternating up and down",
                ),
            )

        # Rule 5: two of three consecutive points beyond 2 sigma on same side
        if self._check_rule5(pts):
            violations.append(
                NelsonViolation(
                    rule=5,
                    severity="warning",
                    description="Two of three consecutive points beyond 2 sigma on the same side",
                ),
            )

        # Rule 6: four of five consecutive points beyond 1 sigma on same side
        if self._check_rule6(pts):
            violations.append(
                NelsonViolation(
                    rule=6,
                    severity="warning",
                    description="Four of five consecutive points beyond 1 sigma on the same side",
                ),
            )

        # Rule 7: fifteen consecutive points within 1 sigma of center
        if self._check_rule7(pts):
            violations.append(
                NelsonViolation(
                    rule=7,
                    severity="info",
                    description="Fifteen consecutive points within 1 sigma of the center line",
                ),
            )

        # Rule 8: eight consecutive points beyond 1 sigma on either side
        if self._check_rule8(pts):
            violations.append(
                NelsonViolation(
                    rule=8,
                    severity="warning",
                    description="Eight consecutive points beyond 1 sigma on either side of center",
                ),
            )

        return violations

    # -- individual rule checks ---------------------------------------------

    def _check_rule1(self, pts: list[float]) -> bool:
        """One point beyond 3 sigma (the most recent point)."""
        last = pts[-1]
        return last > self._ucl or last < self._lcl

    def _check_rule2(self, pts: list[float]) -> bool:
        """Nine consecutive points on the same side of the mean."""
        tail = pts[-9:]
        if len(tail) < 9:
            return False
        above = all(p > self._mean for p in tail)
        below = all(p < self._mean for p in tail)
        return above or below

    def _check_rule3(self, pts: list[float]) -> bool:
        """Six consecutive points steadily increasing or decreasing."""
        tail = pts[-6:]
        if len(tail) < 6:
            return False
        increasing = all(tail[i] < tail[i + 1] for i in range(5))
        decreasing = all(tail[i] > tail[i + 1] for i in range(5))
        return increasing or decreasing

    def _check_rule4(self, pts: list[float]) -> bool:
        """Fourteen consecutive points alternating up and down."""
        tail = pts[-14:]
        if len(tail) < 14:
            return False
        for i in range(len(tail) - 1):
            if i % 2 == 0:
                # Even indices: expect next to be higher
                if tail[i] >= tail[i + 1]:
                    return False
            # Odd indices: expect next to be lower
            elif tail[i] <= tail[i + 1]:
                return False
        return True

    def _check_rule5(self, pts: list[float]) -> bool:
        """Two of three consecutive points beyond 2 sigma on the same side."""
        tail = pts[-3:]
        if len(tail) < 3:
            return False
        above = sum(1 for p in tail if p > self._ucl2)
        below = sum(1 for p in tail if p < self._lcl2)
        return above >= 2 or below >= 2

    def _check_rule6(self, pts: list[float]) -> bool:
        """Four of five consecutive points beyond 1 sigma on the same side."""
        tail = pts[-5:]
        if len(tail) < 5:
            return False
        above = sum(1 for p in tail if p > self._ucl1)
        below = sum(1 for p in tail if p < self._lcl1)
        return above >= 4 or below >= 4

    def _check_rule7(self, pts: list[float]) -> bool:
        """Fifteen consecutive points within 1 sigma of the center line."""
        tail = pts[-15:]
        if len(tail) < 15:
            return False
        return all(self._lcl1 <= p <= self._ucl1 for p in tail)

    def _check_rule8(self, pts: list[float]) -> bool:
        """Eight consecutive points beyond 1 sigma on either side (stratification)."""
        tail = pts[-8:]
        if len(tail) < 8:
            return False
        return all(p > self._ucl1 or p < self._lcl1 for p in tail)
