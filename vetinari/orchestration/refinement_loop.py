"""
Verification as Iterative Refinement
======================================

After a verification failure, invokes refinement with specific issues,
tracks convergence across cycles, and bails out to Human-in-the-Loop
if refinement diverges.

Usage:
    from vetinari.orchestration.refinement_loop import RefinementLoop

    loop = RefinementLoop()
    refined = loop.refine(result, verification_report, max_cycles=3)
"""

import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class VerificationIssue:
    """A single issue found during verification."""
    issue_id: str = field(default_factory=lambda: f"iss_{uuid.uuid4().hex[:6]}")
    category: str = ""          # "correctness", "quality", "style", "security", "performance"
    severity: str = "medium"    # "critical", "high", "medium", "low"
    description: str = ""
    location: str = ""          # File/function/line reference
    suggestion: str = ""        # Suggested fix
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationIssue":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class VerificationReport:
    """Report from a verification pass."""
    passed: bool = False
    quality_score: float = 0.0      # 0.0 to 1.0
    issues: List[VerificationIssue] = field(default_factory=list)
    summary: str = ""
    verifier_agent: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def critical_issues(self) -> List[VerificationIssue]:
        return [i for i in self.issues if i.severity in ("critical", "high")]

    @property
    def unresolved_issues(self) -> List[VerificationIssue]:
        return [i for i in self.issues if not i.resolved]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["critical_count"] = len(self.critical_issues)
        d["unresolved_count"] = len(self.unresolved_issues)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationReport":
        data = dict(data)  # Avoid mutating caller's dictionary
        issues_data = data.pop("issues", [])
        data.pop("critical_count", None)
        data.pop("unresolved_count", None)
        report = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        report.issues = [VerificationIssue.from_dict(i) if isinstance(i, dict) else i for i in issues_data]
        return report


@dataclass
class RefinementCycle:
    """Record of a single refinement cycle."""
    cycle_number: int
    input_quality: float
    output_quality: float
    issues_addressed: int
    issues_remaining: int
    duration_seconds: float
    converging: bool              # True if quality improved
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RefinementResult:
    """Final result of the iterative refinement process."""
    refinement_id: str = ""
    original_quality: float = 0.0
    final_quality: float = 0.0
    cycles_completed: int = 0
    max_cycles: int = 3
    converged: bool = False
    bailed_to_hitl: bool = False
    bail_reason: str = ""
    cycles: List[RefinementCycle] = field(default_factory=list)
    refined_result: Any = None
    final_report: Optional[Dict[str, Any]] = None
    total_duration_seconds: float = 0.0

    @property
    def quality_improvement(self) -> float:
        return self.final_quality - self.original_quality

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["quality_improvement"] = self.quality_improvement
        return d


class RefinementLoop:
    """
    Iterative refinement loop that improves results after verification failures.

    Flow:
    1. Receive result + verification report with specific issues
    2. Invoke refinement agent with targeted issue descriptions
    3. Re-verify the refined result
    4. Track convergence (does quality improve each cycle?)
    5. Bail out to HITL if quality diverges or critical issues persist

    Integrates with CheckpointManager for HITL escalation.
    """

    def __init__(
        self,
        refine_fn: Optional[Callable] = None,
        verify_fn: Optional[Callable] = None,
        min_quality_threshold: float = 0.7,
        divergence_tolerance: int = 1,
        on_bail_to_hitl: Optional[Callable] = None,
    ):
        """
        Args:
            refine_fn: Callable(result, issues) -> refined_result.
                       Invokes an agent to refine the result based on issues.
            verify_fn: Callable(result) -> VerificationReport.
                       Re-verifies a refined result.
            min_quality_threshold: Quality score at which refinement stops.
            divergence_tolerance: Number of cycles quality can drop before bail.
            on_bail_to_hitl: Callback when bailing to human review.
        """
        self._refine_fn = refine_fn or self._default_refine
        self._verify_fn = verify_fn or self._default_verify
        self._min_quality = min_quality_threshold
        self._divergence_tolerance = divergence_tolerance
        self._on_bail = on_bail_to_hitl
        logger.info(
            "RefinementLoop initialized (min_quality=%.2f, divergence_tolerance=%d)",
            min_quality_threshold, divergence_tolerance,
        )

    def refine(
        self,
        result: Any,
        verification_report: Any,
        max_cycles: int = 3,
        context: Optional[Dict[str, Any]] = None,
    ) -> RefinementResult:
        """
        Run iterative refinement on a result that failed verification.

        Args:
            result: The output to refine (code, text, plan, etc.).
            verification_report: A VerificationReport or dict with keys:
                passed, quality_score, issues (list of dicts/VerificationIssues).
            max_cycles: Maximum number of refinement cycles.
            context: Additional context for the refinement agent.

        Returns:
            RefinementResult with the refined output and convergence data.
        """
        refinement_id = f"ref_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        # Normalize the verification report
        report = self._normalize_report(verification_report)
        original_quality = report.quality_score
        current_result = result
        current_report = report
        cycles: List[RefinementCycle] = []
        divergence_count = 0

        logger.info(
            "Starting refinement %s: initial_quality=%.2f, issues=%d, max_cycles=%d",
            refinement_id, original_quality, len(report.issues), max_cycles,
        )

        for cycle_num in range(1, max_cycles + 1):
            cycle_start = time.time()

            # Check if already passing
            if current_report.passed and current_report.quality_score >= self._min_quality:
                logger.info("Refinement %s: quality sufficient at cycle %d (%.2f)",
                            refinement_id, cycle_num, current_report.quality_score)
                break

            # Get unresolved issues to address
            unresolved = current_report.unresolved_issues
            if not unresolved:
                logger.info("Refinement %s: no unresolved issues remain", refinement_id)
                break

            # Run refinement
            input_quality = current_report.quality_score
            try:
                current_result = self._refine_fn(
                    current_result,
                    unresolved,
                    context=context,
                )
            except Exception as exc:
                logger.error("Refinement failed at cycle %d: %s", cycle_num, exc)
                cycles.append(RefinementCycle(
                    cycle_number=cycle_num,
                    input_quality=input_quality,
                    output_quality=input_quality,
                    issues_addressed=0,
                    issues_remaining=len(unresolved),
                    duration_seconds=time.time() - cycle_start,
                    converging=False,
                ))
                break

            # Re-verify
            try:
                current_report = self._verify_fn(current_result)
                if isinstance(current_report, dict):
                    current_report = self._normalize_report(current_report)
            except Exception as exc:
                logger.error("Verification failed at cycle %d: %s", cycle_num, exc)
                current_report = VerificationReport(
                    passed=False,
                    quality_score=input_quality,
                    summary=f"Verification error: {exc}",
                )

            output_quality = current_report.quality_score
            converging = output_quality > input_quality
            issues_resolved = len(unresolved) - len(current_report.unresolved_issues)

            cycle = RefinementCycle(
                cycle_number=cycle_num,
                input_quality=input_quality,
                output_quality=output_quality,
                issues_addressed=max(0, issues_resolved),
                issues_remaining=len(current_report.unresolved_issues),
                duration_seconds=time.time() - cycle_start,
                converging=converging,
            )
            cycles.append(cycle)

            logger.info(
                "Refinement cycle %d: quality %.2f -> %.2f, issues %d -> %d, %s",
                cycle_num, input_quality, output_quality,
                len(unresolved), len(current_report.unresolved_issues),
                "converging" if converging else "DIVERGING",
            )

            # Divergence detection
            if not converging:
                divergence_count += 1
                if divergence_count > self._divergence_tolerance:
                    logger.warning(
                        "Refinement %s: diverged %d times, bailing to HITL",
                        refinement_id, divergence_count,
                    )
                    return self._bail_to_hitl(
                        refinement_id, current_result, current_report,
                        cycles, original_quality, max_cycles, start_time,
                        reason=f"Quality diverged {divergence_count} times",
                    )
            else:
                divergence_count = 0  # Reset on improvement

            # Check for persistent critical issues
            if cycle_num >= 2 and current_report.critical_issues:
                critical_ids = {i.issue_id for i in current_report.critical_issues}
                prev_critical = {i.issue_id for i in report.critical_issues} if report.critical_issues else set()
                if critical_ids == prev_critical:
                    logger.warning(
                        "Refinement %s: critical issues unchanged, bailing to HITL",
                        refinement_id,
                    )
                    return self._bail_to_hitl(
                        refinement_id, current_result, current_report,
                        cycles, original_quality, max_cycles, start_time,
                        reason="Critical issues persist unchanged across cycles",
                    )

        # Determine final state
        final_quality = current_report.quality_score if current_report else original_quality
        converged = final_quality >= self._min_quality

        result_obj = RefinementResult(
            refinement_id=refinement_id,
            original_quality=original_quality,
            final_quality=final_quality,
            cycles_completed=len(cycles),
            max_cycles=max_cycles,
            converged=converged,
            bailed_to_hitl=False,
            cycles=cycles,
            refined_result=current_result,
            final_report=current_report.to_dict() if current_report else None,
            total_duration_seconds=time.time() - start_time,
        )

        logger.info(
            "Refinement %s completed: quality %.2f -> %.2f (%+.2f), %d cycles, converged=%s",
            refinement_id, original_quality, final_quality,
            result_obj.quality_improvement, len(cycles), converged,
        )
        return result_obj

    def _bail_to_hitl(
        self,
        refinement_id: str,
        current_result: Any,
        current_report: VerificationReport,
        cycles: List[RefinementCycle],
        original_quality: float,
        max_cycles: int,
        start_time: float,
        reason: str,
    ) -> RefinementResult:
        """Create a bail-to-HITL result and invoke the callback."""
        result = RefinementResult(
            refinement_id=refinement_id,
            original_quality=original_quality,
            final_quality=current_report.quality_score,
            cycles_completed=len(cycles),
            max_cycles=max_cycles,
            converged=False,
            bailed_to_hitl=True,
            bail_reason=reason,
            cycles=cycles,
            refined_result=current_result,
            final_report=current_report.to_dict(),
            total_duration_seconds=time.time() - start_time,
        )

        if self._on_bail:
            try:
                self._on_bail(result)
            except Exception as exc:
                logger.error("HITL bail callback failed: %s", exc)

        return result

    def _normalize_report(self, report: Any) -> VerificationReport:
        """Convert a dict or report object to VerificationReport."""
        if isinstance(report, VerificationReport):
            return report
        if isinstance(report, dict):
            return VerificationReport.from_dict(dict(report))
        # Try to duck-type
        return VerificationReport(
            passed=getattr(report, "passed", False),
            quality_score=getattr(report, "quality_score", 0.0),
            issues=[
                VerificationIssue.from_dict(i) if isinstance(i, dict) else i
                for i in getattr(report, "issues", [])
            ],
            summary=getattr(report, "summary", ""),
        )

    @staticmethod
    def _default_refine(result: Any, issues: List[VerificationIssue], **kwargs) -> Any:
        """Default no-op refinement (pass-through). Replace with real agent call."""
        logger.warning("Default refinement pass-through: %d issues", len(issues))
        return result

    @staticmethod
    def _default_verify(result: Any) -> VerificationReport:
        """Default no-op verification. Replace with real verifier."""
        logger.warning("Default verification pass-through")
        return VerificationReport(passed=False, quality_score=0.0, summary="No verifier configured — verification skipped")
