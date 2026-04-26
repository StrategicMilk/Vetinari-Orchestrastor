"""Semgrep Security Scanning Tool.

AST-aware security scanning using Semgrep. Catches aliased imports,
multi-line patterns, and complex code structures that regex misses.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vetinari.agents.contracts import OutcomeSignal, Provenance, ToolEvidence
from vetinari.constants import GREP_TIMEOUT
from vetinari.types import EvidenceBasis

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SemgrepFinding:
    """A single Semgrep finding."""

    rule_id: str
    file: str
    line: int
    message: str
    severity: str = "WARNING"

    def __repr__(self) -> str:
        return f"SemgrepFinding({self.rule_id}:{self.file}:{self.line})"


@dataclass
class SemgrepResult:
    """Aggregated Semgrep scan result."""

    findings: list[SemgrepFinding] = field(default_factory=list)
    error: str = ""
    is_available: bool = True

    @property
    def has_findings(self) -> bool:
        """Whether any issues were detected."""
        return len(self.findings) > 0


def run_semgrep(
    target_dir: Path | str,
    config: str = "auto",
    extra_rules: list[dict[str, Any]] | None = None,
    timeout: int = GREP_TIMEOUT * 3,
) -> SemgrepResult:
    """Run Semgrep scan on a directory.

    Args:
        target_dir: Directory or file to scan.
        config: Semgrep config.
        extra_rules: Reserved for future inline rules.
        timeout: Max execution time in seconds.

    Returns:
        SemgrepResult with findings or error information.
    """
    target = Path(target_dir)
    if not target.exists():
        return SemgrepResult(error=f"Target does not exist: {target}", is_available=False)
    cmd = ["semgrep", "--json", "--config", config, str(target)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)  # noqa: S603 - argv is controlled and shell interpolation is not used
    except FileNotFoundError:
        logger.warning(
            "Semgrep not on PATH — security scanning unavailable. Install Vetinari's dev extras: pip install -e .[dev]"
        )
        return SemgrepResult(error="semgrep not installed", is_available=False)
    except subprocess.TimeoutExpired:
        logger.warning("Semgrep timed out after %ds — security scan incomplete", timeout)
        return SemgrepResult(error=f"Semgrep timed out after {timeout}s")
    if proc.returncode not in (0, 1):
        return SemgrepResult(error=f"Semgrep failed: {proc.stderr[:500]}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        logger.warning("Could not parse Semgrep JSON output — security scan results unavailable")
        return SemgrepResult(error="Could not parse Semgrep JSON output")
    findings = [
        SemgrepFinding(
            rule_id=result.get("check_id", "unknown"),
            file=result.get("path", ""),
            line=result.get("start", {}).get("line", 0),
            message=result.get("extra", {}).get("message", ""),
            severity=result.get("extra", {}).get("severity", "WARNING"),
        )
        for result in data.get("results", [])
    ]
    return SemgrepResult(findings=findings)


# -- OutcomeSignal wrapper ---------------------------------------------------


def _semgrep_utc_now() -> str:
    """Return current UTC time as ISO-8601 string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _semgrep_sha256(text: str) -> tuple[str, str]:
    """Return (first-2KB snippet, SHA-256 hex) for stdout text.

    Args:
        text: Raw stdout text from semgrep.

    Returns:
        Tuple of (snippet, sha256_hex).
    """
    import hashlib

    snippet = text[:2048]
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return snippet, digest


def run_semgrep_signal(
    target_dir: Path | str,
    config: str = "auto",
    extra_rules: list[dict[str, Any]] | None = None,
    timeout: int = GREP_TIMEOUT * 3,
) -> OutcomeSignal:
    """Run Semgrep and return an evidence-backed OutcomeSignal.

    Wraps ``run_semgrep()`` with provenance metadata.  When semgrep is not
    installed the signal is ``passed=False, basis=UNSUPPORTED`` with a
    descriptive issue naming the tool and the install remediation.

    No default-pass fallback: any unavailability, timeout, or parse error
    yields ``passed=False`` (Rule 2).

    Args:
        target_dir: Directory or file to scan.
        config: Semgrep config string (default ``"auto"``).
        extra_rules: Reserved for future inline rules.
        timeout: Max execution time in seconds.

    Returns:
        OutcomeSignal with basis=TOOL_EVIDENCE and passed=True only when
        semgrep ran successfully and reported zero findings.
        passed=False with basis=UNSUPPORTED when semgrep is unavailable.
    """
    result = run_semgrep(target_dir, config=config, extra_rules=extra_rules, timeout=timeout)

    if not result.is_available:
        return OutcomeSignal(
            passed=False,
            score=0.0,
            basis=EvidenceBasis.UNSUPPORTED,
            issues=(
                "semgrep not on PATH — security scanning unavailable. "
                "Install Vetinari's dev extras: pip install -e .[dev]",
            ),
            provenance=Provenance(
                source="vetinari.tools.semgrep_tool",
                timestamp_utc=_semgrep_utc_now(),
                tool_name="semgrep",
            ),
        )

    if result.error:
        snippet, sha = _semgrep_sha256(result.error)
        return OutcomeSignal(
            passed=False,
            score=0.0,
            basis=EvidenceBasis.TOOL_EVIDENCE,
            tool_evidence=(
                ToolEvidence(
                    tool_name="semgrep",
                    command=f"semgrep --json --config {config} {target_dir}",
                    exit_code=2,
                    stdout_snippet=snippet,
                    stdout_hash=sha,
                    passed=False,
                ),
            ),
            issues=(f"semgrep error: {result.error}",),
            provenance=Provenance(
                source="vetinari.tools.semgrep_tool",
                timestamp_utc=_semgrep_utc_now(),
                tool_name="semgrep",
            ),
        )

    findings_text = (
        "; ".join(f"{f.file}:{f.line} [{f.severity}] {f.rule_id}: {f.message}" for f in result.findings)
        or "no findings"
    )
    snippet, sha = _semgrep_sha256(findings_text)

    passed = not result.has_findings
    score = 1.0 if passed else round(1.0 / (1.0 + len(result.findings)), 3)
    issues: tuple[str, ...] = tuple(
        f"{f.file}:{f.line} [{f.severity}] {f.rule_id}: {f.message}" for f in result.findings
    )

    return OutcomeSignal(
        passed=passed,
        score=score,
        basis=EvidenceBasis.TOOL_EVIDENCE,
        tool_evidence=(
            ToolEvidence(
                tool_name="semgrep",
                command=f"semgrep --json --config {config} {target_dir}",
                exit_code=0,
                stdout_snippet=snippet,
                stdout_hash=sha,
                passed=passed,
            ),
        ),
        issues=issues,
        provenance=Provenance(
            source="vetinari.tools.semgrep_tool",
            timestamp_utc=_semgrep_utc_now(),
            tool_name="semgrep",
        ),
    )
