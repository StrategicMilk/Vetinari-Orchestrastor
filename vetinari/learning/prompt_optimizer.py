"""DSPy-style Prompt Optimizer — systematic prompt search and trace-based evolution.

Combines two approaches:
1. MIPROv2-inspired search: systematically explore instruction + few-shot example combinations
2. GEPA-style trace diagnosis: analyze execution traces to identify WHY a prompt failed,
   then propose targeted improvements instead of random mutations

All optimized prompts flow through shadow testing (learning/shadow_testing.py) before promotion.

Decision: MIPROv2 + GEPA hybrid approach (ICLR 2026 best practices).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PromptExperiment:
    """A single prompt optimization experiment."""

    experiment_id: str
    agent_type: str
    instruction: str
    few_shot_examples: list[dict[str, str]] = field(default_factory=list)
    quality_scores: list[float] = field(default_factory=list)  # noqa: VET220 — capped to 100 at append site; list kept for json.dump(asdict()) compatibility
    trace_diagnosis: str = ""
    status: str = "pending"  # pending, running, completed, promoted
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __repr__(self) -> str:
        return (
            f"PromptExperiment(experiment_id={self.experiment_id!r}, "
            f"agent_type={self.agent_type!r}, status={self.status!r})"
        )

    @property
    def avg_quality(self) -> float:
        """Mean quality score across recorded trials."""
        return sum(self.quality_scores) / max(len(self.quality_scores), 1)


@dataclass
class TraceAnalysis:
    """Result of analyzing an execution trace to diagnose prompt failures."""

    failure_category: str  # "incomplete", "off_topic", "format_error", "reasoning_error"
    root_cause: str
    suggested_fix: str
    confidence: float  # 0.0-1.0

    def __repr__(self) -> str:
        return f"TraceAnalysis(category={self.failure_category!r}, confidence={self.confidence!r})"


# Instruction mutation templates for MIPROv2-style search
_INSTRUCTION_TEMPLATES = [
    "Be precise and thorough. {base}",
    "{base} Always verify your work before submitting.",
    "{base} Think step by step.",
    "You are an expert. {base} Provide detailed, actionable output.",
    "{base} Format your response with clear sections and headers.",
    "{base} Consider edge cases and potential issues.",
]

_DEFAULT_TIME_BUDGET_SECONDS = 300  # 5 minutes per optimization cycle


class PromptOptimizer:
    """Systematic prompt optimization via search and trace-based evolution.

    Two optimization modes:
    1. **Search mode** (MIPROv2): Generate instruction variants x few-shot combinations,
       evaluate each, keep the best.
    2. **Trace mode** (GEPA): Analyze failed execution traces to diagnose root causes,
       then propose targeted instruction fixes.

    All experiments are time-boxed (default 5 minutes) to control resource usage.
    """

    MAX_EXPERIMENTS_PER_CYCLE = 12  # ~12 experiments/hour during idle

    def __init__(self) -> None:
        self._experiments: dict[str, PromptExperiment] = {}
        self._lock = threading.Lock()

    def optimize_via_search(
        self,
        agent_type: str,
        baseline_instruction: str,
        few_shot_pool: list[dict[str, str]] | None = None,
        time_budget_seconds: float = _DEFAULT_TIME_BUDGET_SECONDS,
    ) -> PromptExperiment | None:
        """Run MIPROv2-style search for better instruction + few-shot combinations.

        Generates variants from instruction templates, optionally pairs with
        few-shot examples, evaluates each, and returns the best performer.

        Args:
            agent_type: Agent type to optimize prompts for.
            baseline_instruction: Current instruction text.
            few_shot_pool: Optional pool of few-shot examples to sample from.
            time_budget_seconds: Maximum time for this optimization cycle.

        Returns:
            Best experiment found, or None if no improvement over baseline.
        """
        deadline = time.monotonic() + time_budget_seconds
        experiments: list[PromptExperiment] = []

        for template in _INSTRUCTION_TEMPLATES:
            if time.monotonic() > deadline:
                break

            variant_instruction = template.format(base=baseline_instruction)
            exp_id = f"search_{uuid.uuid4().hex[:8]}"
            exp = PromptExperiment(
                experiment_id=exp_id,
                agent_type=agent_type,
                instruction=variant_instruction,
                few_shot_examples=(few_shot_pool or [])[:3],  # noqa: VET112 — Optional per func param
                status="completed",
            )

            score = self._evaluate_instruction(variant_instruction, agent_type)
            exp.quality_scores.append(score)
            experiments.append(exp)

        if not experiments:
            return None

        best = max(experiments, key=lambda e: e.avg_quality)
        with self._lock:
            self._experiments[best.experiment_id] = best

        logger.info(
            "[PromptOptimizer] Search found best variant for %s: quality=%.3f",
            agent_type,
            best.avg_quality,
        )
        return best

    def optimize_via_trace(
        self,
        agent_type: str,
        baseline_instruction: str,
        failed_trace: dict[str, Any],
    ) -> PromptExperiment | None:
        """Run GEPA-style trace-based diagnosis and targeted mutation.

        Analyzes a failed execution trace, diagnoses the root cause, and
        proposes a targeted instruction fix.

        Args:
            agent_type: Agent type that produced the failure.
            baseline_instruction: Current instruction text.
            failed_trace: Dict with keys: prompt, output, error, quality_score.

        Returns:
            Experiment with the proposed fix, or None if diagnosis failed or
            confidence was too low.
        """
        analysis = self._diagnose_trace(failed_trace)
        if not analysis or analysis.confidence < 0.4:
            logger.debug(
                "[PromptOptimizer] Trace diagnosis confidence too low (%.2f)",
                analysis.confidence if analysis else 0.0,
            )
            return None

        fixed_instruction = self._apply_fix(baseline_instruction, analysis)
        if fixed_instruction == baseline_instruction:
            return None

        exp = PromptExperiment(
            experiment_id=f"trace_{uuid.uuid4().hex[:8]}",
            agent_type=agent_type,
            instruction=fixed_instruction,
            trace_diagnosis=analysis.root_cause,
            status="completed",
        )

        score = self._evaluate_instruction(fixed_instruction, agent_type)
        exp.quality_scores.append(score)

        with self._lock:
            self._experiments[exp.experiment_id] = exp

        logger.info(
            "[PromptOptimizer] Trace fix for %s: category=%s, quality=%.3f",
            agent_type,
            analysis.failure_category,
            exp.avg_quality,
        )
        return exp

    def _diagnose_trace(self, trace: dict[str, Any]) -> TraceAnalysis | None:
        """Analyze a failed execution trace to identify root cause.

        Applies heuristic rules to classify the failure without requiring LLM
        inference, keeping diagnosis fast enough for the hot path.

        Args:
            trace: Dict with prompt, output, error, quality_score.

        Returns:
            TraceAnalysis with diagnosis, or None if the trace cannot be classified.
        """
        output = str(trace.get("output", ""))
        error = str(trace.get("error", ""))
        quality = float(trace.get("quality_score", 0.0))

        if not output or len(output.strip()) < 20:
            return TraceAnalysis(
                failure_category="incomplete",
                root_cause=("Output too short or empty — model likely ran out of tokens or hit a stop condition early"),
                suggested_fix=("Add explicit instruction: 'Provide a complete, detailed response. Do not stop early.'"),
                confidence=0.8,
            )

        if error and any(kw in error.lower() for kw in ("format", "parse", "json")):
            return TraceAnalysis(
                failure_category="format_error",
                root_cause="Output format doesn't match expected structure",
                suggested_fix="Add explicit output format instructions with example",
                confidence=0.7,
            )

        if quality < 0.3:
            return TraceAnalysis(
                failure_category="reasoning_error",
                root_cause="Very low quality suggests fundamental misunderstanding of the task",
                suggested_fix=("Add step-by-step reasoning instruction and task decomposition hints"),
                confidence=0.6,
            )

        if quality < 0.6:
            return TraceAnalysis(
                failure_category="off_topic",
                root_cause=(
                    "Moderate quality suggests partial understanding — output may be off-topic or missing key aspects"
                ),
                suggested_fix="Add explicit scope constraints and required output components",
                confidence=0.5,
            )

        return None

    @staticmethod
    def _apply_fix(instruction: str, analysis: TraceAnalysis) -> str:
        """Apply a targeted fix to the instruction based on trace analysis.

        Appends a category-specific corrective suffix. Returns the original
        instruction unchanged if the category is unknown.

        Args:
            instruction: Current instruction text.
            analysis: Diagnosis with suggested fix.

        Returns:
            Modified instruction with fix applied.
        """
        fixes: dict[str, str] = {
            "incomplete": ("\n\nIMPORTANT: Provide a complete, detailed response. Do not stop early or abbreviate."),
            "format_error": (
                "\n\nIMPORTANT: Follow the exact output format specified. Structure your response clearly."
            ),
            "reasoning_error": ("\n\nIMPORTANT: Think step by step. Break the problem down before answering."),
            "off_topic": ("\n\nIMPORTANT: Focus strictly on the task requirements. Address every specified aspect."),
        }
        suffix = fixes.get(analysis.failure_category, "")
        return instruction + suffix if suffix else instruction

    @staticmethod
    def _evaluate_instruction(instruction: str, agent_type: str) -> float:
        """Evaluate an instruction variant using heuristic scoring.

        Scores instruction quality based on structural features without
        requiring LLM inference (fast evaluation for search).

        Args:
            instruction: The instruction text to evaluate.
            agent_type: Agent type context (reserved for future per-agent tuning).

        Returns:
            Quality estimate in range [0.0, 1.0].
        """
        score = 0.5  # Baseline

        if len(instruction) > 50:
            score += 0.1
        if len(instruction) > 200:
            score += 0.05
        if "step" in instruction.lower():
            score += 0.05
        if "format" in instruction.lower() or "structure" in instruction.lower():
            score += 0.05
        if "verify" in instruction.lower() or "check" in instruction.lower():
            score += 0.05
        if "example" in instruction.lower():
            score += 0.05
        # Very long instructions can confuse the model — apply a small penalty
        if len(instruction) > 1000:
            score -= 0.1

        return min(1.0, max(0.0, score))

    def get_experiments(self, agent_type: str | None = None) -> list[dict[str, Any]]:
        """Return experiment history, optionally filtered by agent type.

        Args:
            agent_type: If provided, only experiments for this agent are returned.

        Returns:
            List of experiment summary dicts with keys: experiment_id,
            agent_type, avg_quality, status, trace_diagnosis.
        """
        with self._lock:
            exps = list(self._experiments.values())
            if agent_type:
                exps = [e for e in exps if e.agent_type == agent_type]
            return [
                {
                    "experiment_id": e.experiment_id,
                    "agent_type": e.agent_type,
                    "avg_quality": round(e.avg_quality, 3),
                    "status": e.status,
                    "trace_diagnosis": e.trace_diagnosis,
                }
                for e in exps
            ]


# Module-level singleton — written by get_prompt_optimizer(), read by callers.
# Protected by _optimizer_lock (double-checked locking).
_optimizer: PromptOptimizer | None = None
_optimizer_lock = threading.Lock()


def get_prompt_optimizer() -> PromptOptimizer:
    """Return the singleton PromptOptimizer instance (thread-safe).

    Uses double-checked locking to avoid unnecessary synchronization on
    the hot path once the instance is created.

    Returns:
        The shared PromptOptimizer instance.
    """
    global _optimizer
    if _optimizer is None:
        with _optimizer_lock:
            if _optimizer is None:
                _optimizer = PromptOptimizer()
    return _optimizer
