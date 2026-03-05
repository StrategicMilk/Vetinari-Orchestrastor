"""
Reflection Agent — Stage 6.5 of the Vetinari pipeline.

Sits between Output Reviewer (Stage 6) and Final Assembler (Stage 7).
Implements a 3-phase evaluation cycle with loop detection and self-healing.

Phases
------
1. Context Building  — extract task summary, catalogue actions taken.
2. Self-Assessment   — isolated LLM call evaluating work against requirements.
3. Validation        — secondary evaluation enforcing objective workflow gates.

Verdicts
--------
PASS       — work is complete and correct.
INCOMPLETE — work is partially complete; feedback provided for re-execution.
NEEDS_USER — work cannot proceed without human clarification.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import AgentResult, AgentTask, VerificationResult
from vetinari.types import AgentType

_MAX_REFLECTION_CYCLES = 3

# Minimum write-operation ratio before we flag a planning loop
_MIN_WRITE_RATIO = 0.10

# Regex for detecting identical sequential tool calls (loop detection)
_IDENTICAL_CALL_WINDOW = 4


class ReflectionAgent(BaseAgent):
    """
    Stage 6.5 agent: 3-phase evaluation + loop detection + self-healing.

    Input context:
        task_description, task_type, outputs (list of prior agent outputs),
        tool_calls (list of action names taken), cycle (int — which retry)

    Output JSON schema:
    {
        "task_summary": "...",
        "status": "PASS|INCOMPLETE|NEEDS_USER",
        "confidence": 0.0-1.0,
        "evidence": ["..."],
        "remaining_work": ["..."],
        "next_steps": ["..."],
        "needs_user_action": false,
        "stuck": false,
        "verdict": "PASS|INCOMPLETE|NEEDS_USER",
        "loop_detected": false,
        "loop_type": null
    }
    """

    _SYSTEM = (
        "You are Vetinari's Reflection Agent — a rigorous evaluator of AI work. "
        "Your role is to assess whether a task has been FULLY and CORRECTLY completed.\n\n"
        "You MUST output ONLY valid JSON with these fields:\n"
        '{"task_summary": "...", "status": "PASS|INCOMPLETE|NEEDS_USER", '
        '"confidence": 0.95, "evidence": ["evidence items"], '
        '"remaining_work": ["unfinished items"], "next_steps": ["action items"], '
        '"needs_user_action": false, "stuck": false, '
        '"verdict": "PASS|INCOMPLETE|NEEDS_USER"}\n\n'
        "Objective gates (MUST be PASS before verdict is PASS):\n"
        "- No placeholder, stub, or TODO code in outputs.\n"
        "- All files listed in the plan have been generated.\n"
        "- If task_type is 'coding', at least one test exists.\n"
        "- If task_type is 'research', at least 3 findings are cited.\n"
        "Be strict. Any doubt → INCOMPLETE."
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.EVALUATOR, config)

    def get_system_prompt(self) -> str:
        return self._SYSTEM

    def execute(self, task: AgentTask) -> AgentResult:
        ctx = task.context or {}
        task_description = task.prompt or task.description or ""
        task_type = ctx.get("task_type", "general")
        outputs: List[Any] = ctx.get("outputs", [])
        tool_calls: List[str] = ctx.get("tool_calls", [])
        cycle: int = ctx.get("cycle", 1)

        # Phase 1 — Context Building
        context_summary = self._build_context(
            task_description, task_type, outputs, tool_calls
        )

        # Phase 2 — Loop detection (heuristic, no LLM needed)
        loop_type = self._detect_loop(tool_calls)

        # Phase 3 — Self-Assessment via LLM
        assessment = self._self_assess(
            task_description, task_type, outputs, context_summary, cycle, loop_type
        )

        # Phase 4 — Objective gate validation
        assessment = self._apply_objective_gates(
            assessment, task_type, outputs
        )

        assessment["loop_detected"] = loop_type is not None
        assessment["loop_type"] = loop_type
        assessment["cycle"] = cycle

        verdict = assessment.get("verdict", "INCOMPLETE")
        success = verdict == "PASS"

        return AgentResult(
            task_id=task.task_id,
            success=success,
            output=assessment,
            agent_type=self._agent_type.value,
            errors=[] if success else [f"Reflection verdict: {verdict}"],
        )

    def verify(self, result: AgentResult) -> VerificationResult:
        output = result.output
        if not isinstance(output, dict):
            return VerificationResult(
                passed=False, score=0.0, feedback="Output is not a dict"
            )
        if "verdict" not in output:
            return VerificationResult(
                passed=False, score=0.5, feedback="Missing 'verdict' field"
            )
        return VerificationResult(
            passed=output.get("verdict") == "PASS",
            score=float(output.get("confidence", 0.5)),
            feedback=output.get("verdict", "INCOMPLETE"),
        )

    # ------------------------------------------------------------------
    # Phase 1 — Context building
    # ------------------------------------------------------------------

    def _build_context(
        self,
        task_description: str,
        task_type: str,
        outputs: List[Any],
        tool_calls: List[str],
    ) -> str:
        write_ops = sum(
            1 for t in tool_calls
            if any(w in t.lower() for w in ("write", "create", "save", "generate"))
        )
        total_ops = len(tool_calls) or 1
        write_ratio = write_ops / total_ops

        parts = [
            f"Task: {task_description[:200]}",
            f"Type: {task_type}",
            f"Actions taken: {len(tool_calls)} ({write_ops} write ops, "
            f"{write_ratio:.0%} write ratio)",
            f"Outputs produced: {len(outputs)}",
        ]
        if outputs:
            sample = str(outputs[0])[:300]
            parts.append(f"Sample output: {sample}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Phase 2 — Loop detection
    # ------------------------------------------------------------------

    def _detect_loop(self, tool_calls: List[str]) -> Optional[str]:
        """Return loop type string or None."""
        if not tool_calls:
            return None

        # Planning loop: many calls with very low write ratio
        write_count = sum(
            1 for t in tool_calls
            if any(w in t.lower() for w in ("write", "create", "save", "generate"))
        )
        if len(tool_calls) >= 8 and write_count / len(tool_calls) < _MIN_WRITE_RATIO:
            return "planning_loop"

        # Action loop: identical consecutive calls
        window = tool_calls[-_IDENTICAL_CALL_WINDOW:]
        if len(window) >= _IDENTICAL_CALL_WINDOW and len(set(window)) == 1:
            return "action_loop"

        return None

    # ------------------------------------------------------------------
    # Phase 3 — Self-assessment via LLM
    # ------------------------------------------------------------------

    def _self_assess(
        self,
        task_description: str,
        task_type: str,
        outputs: List[Any],
        context_summary: str,
        cycle: int,
        loop_type: Optional[str],
    ) -> Dict[str, Any]:
        loop_note = ""
        if loop_type == "planning_loop":
            loop_note = (
                "\n\nWARNING: A planning loop was detected. "
                "Stop researching and start implementing."
            )
        elif loop_type == "action_loop":
            loop_note = (
                "\n\nWARNING: An action loop was detected. "
                "You are repeating the same action. Try a different approach."
            )

        user_msg = (
            f"Evaluate this work:\n\n"
            f"{context_summary}\n\n"
            f"Task description: {task_description[:500]}\n"
            f"Task type: {task_type}\n"
            f"Reflection cycle: {cycle} of {_MAX_REFLECTION_CYCLES}\n"
            f"{loop_note}\n\n"
            f"Outputs (first 2):\n"
            + "\n---\n".join(str(o)[:400] for o in outputs[:2])
        )

        try:
            raw = self._call_llm(self._SYSTEM, user_msg, max_tokens=1024)
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                # Normalise verdict
                data.setdefault("verdict", data.get("status", "INCOMPLETE"))
                return data
        except Exception as e:
            self._log("warning", f"Self-assessment LLM failed: {e}")

        return self._heuristic_assess(task_description, outputs, cycle, loop_type)

    # ------------------------------------------------------------------
    # Phase 4 — Objective gate validation
    # ------------------------------------------------------------------

    _PLACEHOLDER_RE = re.compile(
        r'\b(TODO|FIXME|raise\s+NotImplementedError|\.\.\.)\b'
        r'|["\']placeholder["\']'
        r'|\bpass\b',
        re.IGNORECASE,
    )

    def _apply_objective_gates(
        self,
        assessment: Dict[str, Any],
        task_type: str,
        outputs: List[Any],
    ) -> Dict[str, Any]:
        """Force INCOMPLETE if objective criteria are not met."""
        issues: List[str] = []

        all_text = " ".join(str(o) for o in outputs)

        # Gate: no placeholder code
        if self._PLACEHOLDER_RE.search(all_text):
            issues.append("Output contains placeholder or stub code")

        # Gate: coding tasks must have a test
        if task_type in ("coding", "implementation") and "def test_" not in all_text:
            issues.append("Coding task output has no test functions")

        # Gate: research tasks need findings
        if task_type == "research":
            finding_count = all_text.lower().count("finding") + all_text.count("•")
            if finding_count < 3:
                issues.append("Research output has fewer than 3 findings")

        if issues:
            assessment["verdict"] = "INCOMPLETE"
            assessment["status"] = "INCOMPLETE"
            assessment.setdefault("remaining_work", []).extend(issues)
            assessment["confidence"] = min(
                float(assessment.get("confidence", 0.5)), 0.4
            )

        return assessment

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _heuristic_assess(
        self,
        task_description: str,
        outputs: List[Any],
        cycle: int,
        loop_type: Optional[str],
    ) -> Dict[str, Any]:
        has_output = bool(outputs and any(outputs))
        verdict = "PASS" if (has_output and cycle == 1 and not loop_type) else "INCOMPLETE"
        return {
            "task_summary": task_description[:120],
            "status": verdict,
            "confidence": 0.5 if verdict == "INCOMPLETE" else 0.7,
            "evidence": ["Heuristic assessment (LLM unavailable)"],
            "remaining_work": [] if verdict == "PASS" else ["Verify output completeness"],
            "next_steps": ["Review output manually"],
            "needs_user_action": loop_type is not None,
            "stuck": cycle >= _MAX_REFLECTION_CYCLES,
            "verdict": verdict,
        }


# Module-level singleton
_reflection_instance: Optional[ReflectionAgent] = None


def get_reflection_agent() -> ReflectionAgent:
    """Return the module-level ReflectionAgent singleton."""
    global _reflection_instance
    if _reflection_instance is None:
        _reflection_instance = ReflectionAgent()
    return _reflection_instance
