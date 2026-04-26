"""Execution logic for Inspector agent code-review and security-audit modes.

These are module-level functions that accept the InspectorAgent instance so
they can call agent methods (_infer_json, _run_antipattern_scan, etc.) without
inheriting from the class. Keeping them here holds quality_agent.py under the
550-line file limit.

This is step 4 of the pipeline: Intake → Planning → Execution → **Quality Gate** → Assembly.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from vetinari.agents.contracts import AgentResult, AgentTask
from vetinari.config.inference_config import get_inference_config
from vetinari.constants import TRUNCATE_CODE_ANALYSIS, TRUNCATE_CONTEXT
from vetinari.safety.prompt_sanitizer import sanitize_worker_output

if TYPE_CHECKING:
    from vetinari.agents.consolidated.quality_agent import InspectorAgent

logger = logging.getLogger(__name__)


def _publish_quality_gate_event(task_id: str, passed: bool, score: float, issues: list[str]) -> None:
    """Publish a QualityGateResult event to the event bus.

    Logs at WARNING if the event bus is unavailable so the caller never has to
    catch this internally.

    Args:
        task_id: Identifier of the task that was quality-checked.
        passed: Whether the quality gate passed.
        score: Numeric quality score between 0.0 and 1.0.
        issues: Human-readable list of issue messages (truncated to 20 items by caller).
    """
    try:
        from vetinari.events import QualityGateResult, get_event_bus

        get_event_bus().publish(
            QualityGateResult(
                event_type="QualityGateResult",
                timestamp=time.time(),
                task_id=task_id,
                passed=passed,
                score=float(score),
                issues=issues,
            )
        )
    except Exception:
        logger.warning(
            "Could not publish QualityGateResult event for task %s — event bus unavailable",
            task_id,
        )


def execute_code_review(agent: InspectorAgent, task: AgentTask) -> AgentResult:
    """Run the code-review execution pipeline for one task.

    Performs four sequential phases: static analysis, anti-pattern scan,
    LLM semantic review, and (optionally) reflexion + meta-rewarding passes.
    Falls back gracefully when optional tools (static analysis, repo map) are
    unavailable.

    Args:
        agent: The InspectorAgent instance, used for _infer_json and scan methods.
        task: The AgentTask containing code, file_path, and review_type context keys.

    Returns:
        AgentResult with review findings and quality score.  Score < 0.5 triggers
        root-cause analysis and QualityGateResult event with passed=False.
    """
    # Sanitize Worker output before it reaches the Inspector's LLM prompt.
    # Worker output may contain adversarial instructions from user content or
    # from a compromised Worker. Wrapping in untrusted-content delimiters
    # prevents the Inspector from treating it as additional instructions.
    # Decision: sanitize Worker output at Inspector boundary (ADR-0097)
    _raw_code = task.context.get("code", task.description)
    code = sanitize_worker_output(_raw_code) if _raw_code else ""
    review_type = task.context.get("review_type", "general")

    # Static analysis (ast + pyright + ruff + vulture) if file path available
    target_file = task.context.get("file_path")
    if target_file:
        try:
            from pathlib import Path

            from vetinari.tools.static_analysis import run_static_analysis

            sa_result = run_static_analysis(Path(target_file))
            if sa_result.findings:
                "\n\nStatic analysis findings:\n" + "\n".join(
                    f"- [{f.severity}] {f.tool}:{f.line}: {f.message}" for f in sa_result.findings[:8]
                )
        except Exception:
            logger.warning("Static analysis unavailable for code review of %s — skipping static findings", target_file)

    # Deterministic anti-pattern scan
    antipattern_findings = agent._run_antipattern_scan(code)
    antipattern_summary = ""
    if antipattern_findings:
        antipattern_summary = "\n\nDeterministic anti-pattern scan found these issues:\n" + "\n".join(
            f"- [{f['severity']}] Line {f['line']}: {f['finding']}" for f in antipattern_findings[:10]
        )

    # Assemble structured context (callers, tests)
    structured_context = ""
    if target_file:
        try:
            from pathlib import Path

            from vetinari.repo_map import get_repo_map

            repo_map = get_repo_map()
            task_context = repo_map.generate_for_task(
                str(Path(target_file).parent), target_file
            )
            if task_context:
                structured_context = f"\n\nCaller context:\n{task_context[:1500]}"
        except Exception:
            logger.warning(
                "Repo map context unavailable for code review of %s — proceeding without caller context", target_file
            )

        try:
            from vetinari.grep_context import GrepContext

            grep_ctx = GrepContext()
            test_refs = grep_ctx.find_references(target_file, pattern="test_")
            if test_refs:
                structured_context += f"\n\nRelated tests:\n{test_refs[:800]}"
        except Exception:
            logger.warning(
                "Grep context unavailable for code review of %s — proceeding without test references", target_file
            )

    # LLM-based semantic review
    prompt = (
        f"Review the following code for quality and maintainability:\n\n"
        f"```\n{code[:TRUNCATE_CONTEXT]}\n```\n\n"
        f"Review focus: {review_type}{antipattern_summary}{structured_context}\n\n"
        "Respond as JSON:\n"
        '{"score": 0.75, "summary": "...", '
        '"issues": [{"severity": "high|medium|low", "category": "...", '
        '"message": "...", "line": 0, "suggestion": "..."}], '
        '"strengths": [...], "recommendations": [...]}'
    )
    result = agent._infer_json(prompt, fallback={"score": 0.5, "issues": [], "summary": "Review unavailable"})

    # Merge anti-pattern findings into result issues (avoid duplicating LLM findings)
    if isinstance(result, dict) and antipattern_findings:
        existing_issues = result.setdefault("issues", [])
        existing_msgs = {i.get("message", "").lower() for i in existing_issues if isinstance(i, dict)}
        for af in antipattern_findings:
            if af["finding"].lower() not in existing_msgs:
                existing_issues.append({
                    "severity": af["severity"].lower(),
                    "category": "maintainability",
                    "message": af["finding"],
                    "line": af["line"],
                    "suggestion": f"Detected at line {af['line']}: {af['evidence']}",
                })

    # Reflexion — Inspector verifies its own findings with citations
    if isinstance(result, dict) and result.get("issues"):
        try:
            from vetinari.llm_helpers import quick_llm_call

            issues_text = "\n".join(
                f"- [{i.get('severity', '?')}] {i.get('message', '')}" for i in result["issues"][:5]
            )
            cfg = get_inference_config().get_profile("code_review")
            reflexion = quick_llm_call(
                prompt=(
                    f"You just reviewed this code and found these issues:\n{issues_text}\n\n"
                    f"For each issue, cite the specific line or pattern that supports your finding. "
                    f"If any finding is uncertain, say so. Format: ISSUE: citation"
                ),
                system_prompt="You are a code reviewer verifying your own findings with evidence.",
                max_tokens=cfg.max_tokens,
            )
            if reflexion:
                result["reflexion"] = reflexion
        except Exception:
            logger.warning("Reflexion pass unavailable for %s — review returned without self-critique", review_type)

    # Meta-rewarding — score the quality of this review
    if isinstance(result, dict):
        try:
            from vetinari.llm_helpers import quick_llm_call

            review_text = result.get("summary", "")
            meta_score = quick_llm_call(
                prompt=(
                    f"Rate this code review on a scale of 0.0-1.0 for thoroughness, "
                    f"actionability, and accuracy:\n\nReview summary: {review_text}\n"
                    f"Issues found: {len(result.get('issues', []))}\n"
                    f"Respond with only a decimal number."
                ),
                system_prompt="You evaluate code review quality.",
                max_tokens=10,  # noqa: VET129 — single decimal output, deterministic
            )
            if meta_score and meta_score.strip().replace(".", "", 1).isdigit():
                result["meta_review_score"] = max(0.0, min(1.0, float(meta_score.strip())))
        except Exception:
            logger.warning("Meta-rewarding pass unavailable — review quality score not recorded for %s", review_type)

    # Compute a rubric-weighted overall score before building the final result.
    # For INSPECTOR there is no dedicated rubric, so compute_overall_score falls
    # back to a simple average of available signal dimensions.
    _raw_score = result.get("score", 1.0) if isinstance(result, dict) else 1.0
    _review_scores: dict[str, float] = {"review_score": float(_raw_score)}
    if isinstance(result, dict) and "meta_review_score" in result:
        _review_scores["meta_review_score"] = float(result["meta_review_score"])
    try:
        from vetinari.agents.skill_contract import compute_overall_score
        from vetinari.types import AgentType

        _overall = compute_overall_score(_review_scores, AgentType.INSPECTOR.value)
    except Exception:
        logger.warning(
            "compute_overall_score unavailable for %s review — using raw LLM score",
            review_type,
        )
        _overall = _raw_score

    # Compute a rubric-weighted overall score when the LLM returned named
    # dimension sub-scores (e.g. {"accuracy": 0.8, "completeness": 0.7}).
    # Falls back to the single "score" field when no dimension map is present.
    _dimension_scores: dict[str, float] = {}
    if isinstance(result, dict):
        _dimension_scores = {
            k: float(v)
            for k, v in result.items()
            if k not in {"score", "summary", "issues", "strengths", "recommendations", "reflexion", "meta_review_score"}
            and isinstance(v, (int, float))
        }
    _rubric_score: float | None = None
    if _dimension_scores:
        _rubric_score = compute_overall_score(
            _dimension_scores, agent.agent_type.value if hasattr(agent, "agent_type") else ""
        )

    # Determine whether the LLM result is real or a fallback/malformed response.
    # The inference fallback dict {"score": 0.5, "issues": [], "summary": "Review unavailable"}
    # must be treated as failure — not success — because it carries no review content.
    # A result is considered a fallback when:
    #   - it is not a dict at all, OR
    #   - it has the sentinel summary "Review unavailable" with no issues
    _is_fallback_result = not isinstance(result, dict) or (
        result.get("summary") == "Review unavailable" and not result.get("issues")
    )

    if _is_fallback_result and antipattern_findings:
        # LLM was unavailable but we have deterministic antipattern evidence — promote
        # antipatterns into a synthetic result so the evidence is not discarded.
        result = {
            "score": 0.4,  # below passing threshold to trigger RCA
            "summary": f"LLM review unavailable — {len(antipattern_findings)} antipattern(s) detected by static analysis",
            "issues": [
                {
                    "severity": af["severity"].lower(),
                    "category": "maintainability",
                    "message": af["finding"],
                    "line": af["line"],
                    "suggestion": f"Detected at line {af['line']}: {af['evidence']}",
                }
                for af in antipattern_findings
            ],
        }
        _is_fallback_result = False  # synthetic result from real evidence is not a fallback
        logger.warning(
            "LLM review unavailable for %s — synthetic result built from %d antipattern finding(s)",
            review_type,
            len(antipattern_findings),
        )

    _is_success = not _is_fallback_result
    if _is_fallback_result:
        logger.warning(
            "Code review for %s produced no content — LLM returned fallback and no antipatterns were found",
            review_type,
        )

    # Recompute scores from the (possibly synthetic) result
    _raw_score = result.get("score", 1.0) if isinstance(result, dict) else 0.0
    _review_scores = {"review_score": float(_raw_score)}
    if isinstance(result, dict) and "meta_review_score" in result:
        _review_scores["meta_review_score"] = float(result["meta_review_score"])
    try:
        from vetinari.agents.skill_contract import compute_overall_score
        from vetinari.types import AgentType

        _overall = compute_overall_score(_review_scores, AgentType.INSPECTOR.value)
    except Exception:
        logger.warning(
            "compute_overall_score unavailable for %s review — using raw score",
            review_type,
        )
        _overall = _raw_score

    agent_result = AgentResult(
        success=_is_success,
        output=result,
        metadata={
            "mode": "code_review",
            "review_type": review_type,
            "antipattern_count": len(antipattern_findings),
            "overall_score": _overall,
            "rubric_score": _rubric_score,
            "is_fallback": _is_fallback_result,
        },
        errors=(
            [f"Code review unavailable for {review_type} — no LLM output and no antipatterns detected"]
            if _is_fallback_result
            else []
        ),
    )
    score = result.get("score", 0.0) if isinstance(result, dict) else 0.0

    # Factor in upstream self_check results
    _self_check_passed = task.context.get("self_check_passed", True)
    if not _self_check_passed:
        _self_check_issues = task.context.get("self_check_issues", [])
        score = min(score, 0.4)  # force RCA when self-check failed
        agent_result.metadata["self_check_override"] = True
        agent_result.metadata["self_check_issues"] = _self_check_issues
        logger.info("Self-check failed upstream — forcing RCA (issues=%d)", len(_self_check_issues))

    if score < 0.5:
        agent_result = agent._perform_root_cause_analysis(task, agent_result)

    _issues_out = result.get("issues", []) if isinstance(result, dict) else []
    _issue_msgs = [i.get("message", str(i)) if isinstance(i, dict) else str(i) for i in _issues_out[:20]]
    _publish_quality_gate_event(str(task.task_id), score >= 0.5, float(score), _issue_msgs)

    return agent_result


def execute_security_audit(agent: InspectorAgent, task: AgentTask) -> AgentResult:
    """Run the security-audit execution pipeline for one task.

    Runs deterministic tools first (Semgrep, heuristic patterns), then sends
    findings to the LLM for deep semantic analysis that regex cannot perform
    (business-logic flaws, insecure design, trust-boundary violations).

    Args:
        agent: The InspectorAgent instance, used for _infer_json and _run_heuristic_scan.
        task: The AgentTask containing code, file_path, and project_path context keys.

    Returns:
        AgentResult with security findings, overall_risk, and score.  Score < 0.5
        triggers root-cause analysis.
    """
    # Sanitize Worker output before it reaches the Inspector's LLM prompt.
    # Worker output entering the security audit may contain adversarial instructions
    # embedded in the code under review. Wrap in untrusted-content delimiters.
    # Decision: sanitize Worker output at Inspector boundary (ADR-0097)
    _raw_audit_code = task.context.get("code", task.description)
    code = sanitize_worker_output(_raw_audit_code) if _raw_audit_code else ""

    # Semgrep AST-aware scan (if available and file path provided)
    target_path = task.context.get("file_path") or task.context.get("project_path")
    if target_path:
        try:
            from pathlib import Path

            from vetinari.tools.semgrep_tool import run_semgrep

            sg_result = run_semgrep(Path(target_path))
            if sg_result.has_findings:
                [
                    {"severity": f.severity.lower(), "finding": f.message, "line": f.line, "tool": "semgrep"}
                    for f in sg_result.findings
                ]
        except Exception:
            logger.warning("Semgrep unavailable for security audit of %s — skipping semgrep findings", target_path)

    # Heuristic pattern scan
    heuristic_findings = agent._run_heuristic_scan(code)

    # LLM-based deep analysis
    heuristic_summary = ""
    if heuristic_findings:
        heuristic_summary = "\n\nHeuristic scan found these preliminary issues:\n" + "\n".join(
            f"- [{f['severity']}] {f['finding']}" for f in heuristic_findings[:10]
        )

    prompt = (
        f"Perform a comprehensive security audit of this code:\n\n"
        f"```\n{code[:TRUNCATE_CODE_ANALYSIS]}\n```\n"
        f"{heuristic_summary}\n\n"
        "Analyze for: injection, broken auth, sensitive data exposure, "
        "XXE, broken access control, misconfig, XSS, insecure deserialization, "
        "vulnerable components, insufficient logging.\n\n"
        "Respond as JSON:\n"
        '{"findings": [{"severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO", '
        '"finding": "...", "cwe": "CWE-79 (use real CWE IDs: 79=XSS, 89=SQLi, 22=PathTraversal, 78=OSCmd, 502=Deserialization, 798=HardcodedCreds, 327=BrokenCrypto, 306=MissingAuth)", "owasp": "A01-A10", '
        '"line": 0, "remediation": "...", "code_example": "..."}], '
        '"summary": "...", "overall_risk": "high|medium|low", '
        '"score": 0.75}'
    )

    llm_result = agent._infer_json(prompt, fallback=None)

    # Merge heuristic + LLM findings
    if llm_result and isinstance(llm_result, dict):
        llm_findings = llm_result.get("findings", [])
        llm_finding_names = {f.get("finding", "").lower() for f in llm_findings}
        for hf in heuristic_findings:
            if hf["finding"].lower() not in llm_finding_names:
                llm_findings.append(hf)
        llm_result["findings"] = llm_findings
        llm_result.setdefault("heuristic_count", len(heuristic_findings))
        llm_agent_result = AgentResult(
            success=True,
            output=llm_result,
            metadata={"mode": "security_audit", "heuristic_findings": len(heuristic_findings)},
        )
        sec_score = llm_result.get("score", 1.0) if isinstance(llm_result, dict) else 1.0
        if sec_score < 0.5:
            llm_agent_result = agent._perform_root_cause_analysis(task, llm_agent_result)
        _findings = llm_result.get("findings", [])
        _issue_msgs = [f.get("finding", str(f)) if isinstance(f, dict) else str(f) for f in _findings[:20]]
        _publish_quality_gate_event(str(task.task_id), sec_score >= 0.5, float(sec_score), _issue_msgs)
        return llm_agent_result

    # Heuristic-only fallback (LLM unavailable)
    heuristic_output = {
        "findings": heuristic_findings,
        "summary": f"Heuristic scan found {len(heuristic_findings)} issues (LLM unavailable)",
        "overall_risk": "high" if any(f["severity"] in ("CRITICAL", "HIGH") for f in heuristic_findings) else "medium",
        "score": max(0.0, 1.0 - len(heuristic_findings) * 0.1),
    }
    heuristic_agent_result = AgentResult(
        success=True,
        output=heuristic_output,
        metadata={"mode": "security_audit", "heuristic_only": True},
    )
    heuristic_score = heuristic_output["score"]
    if heuristic_score < 0.5:
        heuristic_agent_result = agent._perform_root_cause_analysis(task, heuristic_agent_result)
    _heuristic_issue_msgs = [
        f.get("finding", str(f)) if isinstance(f, dict) else str(f) for f in heuristic_findings[:20]
    ]
    _publish_quality_gate_event(
        str(task.task_id), heuristic_score >= 0.5, float(heuristic_score), _heuristic_issue_msgs
    )
    return heuristic_agent_result
