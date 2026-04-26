"""Static Analysis Pipeline Tool.

Four-stage gate: ast.parse -> pyright -> ruff -> vulture.

The module-level ``run_static_analysis_signal()`` wraps the pipeline result
into an ``OutcomeSignal`` so callers in the Inspector pipeline get a
provenance-bearing, evidence-backed verdict instead of raw gate lists.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from vetinari.agents.contracts import OutcomeSignal, Provenance, ToolEvidence
from vetinari.constants import GREP_TIMEOUT
from vetinari.types import EvidenceBasis

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AnalysisFinding:
    """A single static analysis finding."""

    tool: str
    file: str
    line: int
    severity: str
    message: str
    code: str = ""

    def __repr__(self) -> str:
        return f"{self.tool}:{self.file}:{self.line} [{self.severity}] {self.message[:60]}"


@dataclass
class AnalysisResult:
    """Aggregated 4-stage pipeline result."""

    findings: list[AnalysisFinding] = field(default_factory=list)
    gates_passed: list[str] = field(default_factory=list)
    gates_failed: list[str] = field(default_factory=list)
    gates_skipped: list[str] = field(default_factory=list)
    skip_reasons: dict[str, str] = field(default_factory=dict)
    error: str = ""

    def __repr__(self) -> str:
        return "AnalysisResult(...)"

    @property
    def is_clean(self) -> bool:
        """Whether all gates passed with no errors."""
        return (
            not self.gates_failed and not self.gates_skipped and not any(f.severity == "error" for f in self.findings)
        )


def _mark_gate_skipped(result: AnalysisResult, gate: str, reason: str) -> None:
    """Record that an analysis gate was skipped instead of passed."""
    result.gates_skipped.append(gate)
    result.skip_reasons[gate] = reason


def _tool_command(module_name: str, executable_name: str) -> list[str]:
    """Return the preferred invocation command for an external analysis tool."""
    if importlib.util.find_spec(module_name) is not None:
        return [sys.executable, "-m", module_name]
    return [executable_name]


def run_static_analysis(target: Path | str, timeout: int = GREP_TIMEOUT * 3) -> AnalysisResult:
    """Run the full 4-stage static analysis pipeline.

    Args:
        target: File or directory to analyze.
        timeout: Per-tool timeout in seconds.

    Returns:
        AnalysisResult with findings and gate status.
    """
    target = Path(target)
    if not target.exists():
        return AnalysisResult(error=f"Target does not exist: {target}")
    result = AnalysisResult()

    # Gate 1: ast.parse
    py_files = list(target.rglob("*.py")) if target.is_dir() else [target]
    for f in py_files[:200]:
        try:
            ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except SyntaxError as e:
            result.findings.append(
                AnalysisFinding(
                    tool="ast", file=str(f), line=e.lineno or 0, severity="error", message=f"SyntaxError: {e.msg}"
                )
            )
    if any(f.severity == "error" for f in result.findings):
        result.gates_failed.append("ast")
        return result
    result.gates_passed.append("ast")

    # Gate 2: pyright
    try:
        pyright_cmd = _tool_command("pyright", "pyright")
        proc = subprocess.run(
            [*pyright_cmd, "--outputjson", str(target)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        diagnostics = json.loads(proc.stdout).get("generalDiagnostics", [])
        pyright_has_error = False
        for diag in diagnostics:
            severity = "error" if diag.get("severity") == "error" else "warning"
            if severity == "error":
                pyright_has_error = True
            result.findings.append(
                AnalysisFinding(
                    tool="pyright",
                    file=diag.get("file", ""),
                    line=diag.get("range", {}).get("start", {}).get("line", 0),
                    severity=severity,
                    message=diag.get("message", ""),
                    code=diag.get("rule", ""),
                )
            )
        if pyright_has_error:
            result.gates_failed.append("pyright")
        else:
            result.gates_passed.append("pyright")
    except FileNotFoundError:
        logger.warning("pyright unavailable — skipping pyright gate")
        _mark_gate_skipped(result, "pyright", "unavailable")
    except subprocess.TimeoutExpired:
        logger.warning("pyright timed out — skipping pyright gate")
        _mark_gate_skipped(result, "pyright", "timeout")
    except json.JSONDecodeError:
        logger.warning("pyright produced invalid JSON — skipping pyright gate")
        _mark_gate_skipped(result, "pyright", "invalid-json")

    # Gate 3: ruff
    try:
        proc = subprocess.run(
            ["ruff", "check", "--output-format", "json", str(target)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        ruff_items = json.loads(proc.stdout)
        for item in ruff_items:
            result.findings.append(
                AnalysisFinding(
                    tool="ruff",
                    file=item.get("filename", ""),
                    line=item.get("location", {}).get("row", 0),
                    severity="warning",
                    message=item.get("message", ""),
                    code=item.get("code", ""),
                )
            )
        if proc.returncode != 0 or ruff_items:
            result.gates_failed.append("ruff")
        else:
            result.gates_passed.append("ruff")
    except FileNotFoundError:
        logger.warning("ruff unavailable — skipping ruff gate")
        _mark_gate_skipped(result, "ruff", "unavailable")
    except subprocess.TimeoutExpired:
        logger.warning("ruff timed out — skipping ruff gate")
        _mark_gate_skipped(result, "ruff", "timeout")
    except json.JSONDecodeError:
        logger.warning("ruff produced invalid JSON — skipping ruff gate")
        _mark_gate_skipped(result, "ruff", "invalid-json")

    # Gate 4: vulture
    try:
        vulture_cmd = _tool_command("vulture", "vulture")
        proc = subprocess.run(
            [*vulture_cmd, str(target), "--min-confidence", "80"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        vulture_findings = 0
        for line in proc.stdout.strip().splitlines():
            parts = line.split(":", 2)
            if len(parts) >= 3:
                vulture_findings += 1
                result.findings.append(
                    AnalysisFinding(
                        tool="vulture",
                        file=parts[0],
                        line=int(parts[1]) if parts[1].isdigit() else 0,
                        severity="info",
                        message=parts[2].strip(),
                    )
                )
        if proc.returncode != 0 or vulture_findings:
            result.gates_failed.append("vulture")
        else:
            result.gates_passed.append("vulture")
    except FileNotFoundError:
        logger.warning("vulture unavailable — skipping vulture gate")
        _mark_gate_skipped(result, "vulture", "unavailable")
    except subprocess.TimeoutExpired:
        logger.warning("vulture timed out — skipping vulture gate")
        _mark_gate_skipped(result, "vulture", "timeout")

    return result


# -- OutcomeSignal wrapper ---------------------------------------------------


def _sha256_snippet(text: str) -> tuple[str, str]:
    """Return (first-2KB snippet, full-text SHA-256 hex) for a stdout string.

    Args:
        text: The raw stdout text from a tool invocation.

    Returns:
        Tuple of (snippet, sha256_hex).
    """
    import hashlib

    snippet = text[:2048]
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return snippet, digest


def _utc_now() -> str:
    """Return the current UTC time as ISO-8601 string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def run_static_analysis_signal(target: Path | str, timeout: int = GREP_TIMEOUT * 3) -> OutcomeSignal:
    """Run the 4-stage static analysis pipeline and return an evidence-backed OutcomeSignal.

    Wraps ``run_static_analysis()`` with provenance metadata so Inspector
    pipeline callers receive a fail-closed ``OutcomeSignal`` instead of raw
    gate lists.  One ``ToolEvidence`` entry is emitted per gate that ran.
    Skipped gates lower the evidence count but do NOT cause ``passed=True``
    — any skip or failure keeps the signal at ``passed=False``.

    On target-not-found or all-gates-skipped the signal uses
    ``basis=EvidenceBasis.UNSUPPORTED`` with ``passed=False``.

    Args:
        target: File or directory to analyse.
        timeout: Per-tool timeout in seconds.

    Returns:
        OutcomeSignal with basis=TOOL_EVIDENCE when at least one gate ran
        and all executed gates passed; basis=UNSUPPORTED when no gates ran;
        passed=False whenever any gate failed or was skipped.
    """
    analysis = run_static_analysis(target, timeout=timeout)

    if analysis.error:
        return OutcomeSignal(
            passed=False,
            score=0.0,
            basis=EvidenceBasis.UNSUPPORTED,
            issues=(f"Static analysis error: {analysis.error}",),
            provenance=Provenance(
                source="vetinari.tools.static_analysis",
                timestamp_utc=_utc_now(),
                tool_name="static_analysis_pipeline",
            ),
        )

    tool_evidences: list[ToolEvidence] = []

    for gate in analysis.gates_passed:
        gate_findings = [f for f in analysis.findings if f.tool == gate]
        stdout_text = "; ".join(f"{f.file}:{f.line} {f.message}" for f in gate_findings) or "no findings"
        snippet, sha = _sha256_snippet(stdout_text)
        tool_evidences.append(
            ToolEvidence(
                tool_name=gate,
                command=f"{gate} {target}",
                exit_code=0,
                stdout_snippet=snippet,
                stdout_hash=sha,
                passed=True,
            )
        )

    for gate in analysis.gates_failed:
        gate_findings = [f for f in analysis.findings if f.tool == gate]
        stdout_text = "; ".join(f"{f.file}:{f.line} [{f.severity}] {f.message}" for f in gate_findings) or "gate failed"
        snippet, sha = _sha256_snippet(stdout_text)
        tool_evidences.append(
            ToolEvidence(
                tool_name=gate,
                command=f"{gate} {target}",
                exit_code=1,
                stdout_snippet=snippet,
                stdout_hash=sha,
                passed=False,
            )
        )

    if not tool_evidences:
        skip_summary = "; ".join(f"{g}={r}" for g, r in analysis.skip_reasons.items()) or "all gates skipped"
        return OutcomeSignal(
            passed=False,
            score=0.0,
            basis=EvidenceBasis.UNSUPPORTED,
            issues=(f"No gates executed — {skip_summary}",),
            provenance=Provenance(
                source="vetinari.tools.static_analysis",
                timestamp_utc=_utc_now(),
                tool_name="static_analysis_pipeline",
            ),
        )

    all_passed = len(analysis.gates_failed) == 0 and len(analysis.gates_skipped) == 0
    issues: tuple[str, ...] = tuple(f"{f.tool}:{f.file}:{f.line} [{f.severity}] {f.message}" for f in analysis.findings)
    skipped_issues: tuple[str, ...] = tuple(f"gate '{g}' skipped: {r}" for g, r in analysis.skip_reasons.items())
    score = round(
        len(analysis.gates_passed) / max(len(analysis.gates_passed) + len(analysis.gates_failed), 1),
        3,
    )

    return OutcomeSignal(
        passed=all_passed,
        score=score,
        basis=EvidenceBasis.TOOL_EVIDENCE,
        tool_evidence=tuple(tool_evidences),
        issues=issues + skipped_issues,
        provenance=Provenance(
            source="vetinari.tools.static_analysis",
            timestamp_utc=_utc_now(),
            tool_name="static_analysis_pipeline",
        ),
    )
