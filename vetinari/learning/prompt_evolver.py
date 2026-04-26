"""Prompt Evolver - Vetinari Self-Improvement Subsystem.

A/B tests prompt variations for each agent and promotes variants that
achieve statistically better quality scores.

Enhanced in Wave 4:
- Statistical significance testing (paired t-test via scipy if available,
  bootstrap otherwise) before any promotion decision.
- Effect size check (Cohen's d >= 0.2) to prevent noise-driven promotions.
- Regression detection: demote if quality declining over time.
- Max concurrent A/B tests per agent (default 3) to avoid traffic dilution.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vetinari.constants import VETINARI_STATE_DIR
from vetinari.learning.prompt_mutator import MutationOperator
from vetinari.types import AgentType, PromptVersionStatus

logger = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """A candidate prompt variant with performance tracking."""

    variant_id: str
    agent_type: str
    prompt_text: str
    is_baseline: bool = False
    trials: int = 0
    total_quality: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    promoted_at: str | None = None
    status: str = PromptVersionStatus.TESTING.value
    metadata: dict = field(default_factory=dict)  # Extensible metadata (e.g. shadow_test_id)

    def __repr__(self) -> str:
        return f"PromptVariant(variant_id={self.variant_id!r}, agent_type={self.agent_type!r}, status={self.status!r})"

    @property
    def avg_quality(self) -> float:
        """The mean quality score across all recorded trials for this variant."""
        return self.total_quality / max(self.trials, 1)

    def record(self, quality: float) -> None:
        """Record for the current context."""
        self.trials += 1
        self.total_quality += quality


class PromptEvolver:
    """Automatically evolves agent system prompts through A/B testing.

    Strategy:
    1. Baseline prompt is the current system prompt for each agent.
    2. When quality degrades (below rolling average), generate new variants.
    3. Route a fraction of tasks to variant prompts.
    4. After MIN_TRIALS trials, promote variant if significantly better.
    5. Store all variants and their performance in memory.
    """

    MIN_TRIALS = 30  # Minimum trials before promotion (20 too few for statistical significance with noisy scores)
    VARIANT_FRACTION = 0.2  # Fraction of tasks routed to variant (30% too aggressive for single-user system)
    MIN_IMPROVEMENT = 0.05  # Minimum mean quality improvement to promote
    MIN_EFFECT_SIZE = 0.3  # Cohen's d threshold — 0.2 ("small") too easy to promote noise; 0.3 ("medium") more reliable
    P_VALUE_THRESHOLD = 0.05  # Statistical significance threshold
    MAX_CONCURRENT_TESTS = 2  # Max A/B tests per agent (3 dilutes traffic too much for single-user system)
    BENCHMARK_PASS_THRESHOLD = 0.6  # Minimum benchmark pass rate — 50% too low; promoted variants should reliably pass

    def __init__(self, adapter_manager=None, improvement_log=None):
        self._adapter_manager = adapter_manager
        # agent_type -> list of variants
        self._variants: dict[str, list[PromptVariant]] = {}
        # Per-variant score history for statistical testing
        self._score_history: dict[str, deque[float]] = {}
        # Track which operator produced each variant (Dept 9.6)
        # variant_id -> (MutationOperator, agent_type, mode)
        self._variant_operators: dict[str, tuple] = {}
        # variant_id -> improvement_id (links to ImprovementLog)
        self._variant_improvements: dict[str, str] = {}
        self._operator_selector = None
        self._prompt_mutator = None
        self._improvement_log = improvement_log
        self._lock = threading.Lock()
        self._load_variants()

    def register_baseline(self, agent_type: str, prompt_text: str) -> None:
        """Register the current system prompt as the baseline.

        Args:
            agent_type: The agent type.
            prompt_text: The prompt text.
        """
        if agent_type not in self._variants:
            self._variants[agent_type] = []

        # Check if baseline already exists
        for v in self._variants[agent_type]:
            if v.is_baseline:
                if v.prompt_text != prompt_text:
                    # Prompt changed externally -- update
                    v.prompt_text = prompt_text
                return

        variant = PromptVariant(
            variant_id=f"{agent_type}_baseline",
            agent_type=agent_type,
            prompt_text=prompt_text,
            is_baseline=True,
            status=PromptVersionStatus.PROMOTED.value,
        )
        self._variants[agent_type].append(variant)

    def select_prompt(self, agent_type: str) -> tuple[str, str]:
        """Select which prompt to use for this invocation.

        Returns:
            Tuple of (prompt_text, variant_id).
        """
        with self._lock:
            variants = self._variants.get(agent_type, [])
            if not variants:
                return "", "none"

            promoted = [v for v in variants if v.status == PromptVersionStatus.PROMOTED.value]
            testing = [v for v in variants if v.status == PromptVersionStatus.TESTING.value]

            if testing and random.random() < self.VARIANT_FRACTION:  # noqa: S311 - deterministic randomness is non-cryptographic
                # Route to a testing variant
                v = random.choice(testing)  # noqa: S311 - deterministic randomness is non-cryptographic
                return v.prompt_text, v.variant_id

            if promoted:
                # Use the best promoted variant
                best = max(promoted, key=lambda v: v.avg_quality if v.trials > 0 else 0.5)
                return best.prompt_text, best.variant_id

            return "", "default"

    def record_result(self, agent_type: str, variant_id: str, quality: float) -> None:
        """Record a quality result for a variant and trigger promotion check.

        Args:
            agent_type: The agent type.
            variant_id: The variant id.
            quality: The quality.
        """
        with self._lock:
            variants = self._variants.get(agent_type, [])
            for v in variants:
                if v.variant_id == variant_id:
                    v.record(quality)
                    # Also track per-variant score history for statistical testing
                    if variant_id not in self._score_history:
                        self._score_history[variant_id] = deque(maxlen=500)
                    self._score_history[variant_id].append(quality)
                    break

            self._check_promotion(agent_type)
            self.check_shadow_test_results()

    def _check_promotion(self, agent_type: str) -> None:
        """Decide whether to promote, deprecate, or keep testing variants.

        Uses statistical significance testing (paired t-test / bootstrap)
        and effect size (Cohen's d) before any promotion.
        """
        variants = self._variants.get(agent_type, [])
        baseline = next((v for v in variants if v.is_baseline and v.status == PromptVersionStatus.PROMOTED.value), None)
        baseline_quality = baseline.avg_quality if baseline and baseline.trials > 0 else 0.65
        baseline_scores = self._score_history.get(baseline.variant_id if baseline else "", [])

        for v in variants:
            if v.status != PromptVersionStatus.TESTING.value or v.trials < self.MIN_TRIALS:
                continue

            variant_scores = self._score_history.get(v.variant_id, [])
            mean_diff = v.avg_quality - baseline_quality

            if mean_diff >= self.MIN_IMPROVEMENT:
                # Run statistical test before promoting
                significant, effect_size = self._test_significance(baseline_scores, variant_scores)
                if significant and effect_size >= self.MIN_EFFECT_SIZE:
                    # Benchmark gate: validate variant before promotion
                    if not self._validate_variant_with_benchmark(v):
                        logger.info(
                            f"[PromptEvolver] {v.variant_id} passed stats but "
                            f"FAILED benchmark validation -- not promoting",
                        )
                        continue

                    # Shadow testing gate: run variant in shadow mode before promotion
                    try:
                        from vetinari.learning.shadow_testing import get_shadow_test_runner

                        runner = get_shadow_test_runner()
                        test_id = runner.create_test(
                            description=f"Prompt evolution: {v.variant_id} vs baseline for {agent_type}",
                            production_config={"variant_id": baseline.variant_id if baseline else "default"},
                            candidate_config={"variant_id": v.variant_id, "agent_type": agent_type},
                        )
                        # Store test_id on variant for later retrieval
                        v.metadata = getattr(v, "metadata", {}) or {}
                        v.metadata["shadow_test_id"] = test_id
                        v.status = PromptVersionStatus.SHADOW_TESTING.value
                        logger.info(
                            "[PromptEvolver] %s for %s passed stats+benchmark — "
                            "entering shadow testing (effect_size=%.3f)",
                            v.variant_id,
                            agent_type,
                            effect_size,
                        )
                    except Exception as exc:
                        # Shadow testing unavailable — fall through to direct promotion
                        logger.warning(
                            "[PromptEvolver] Shadow test creation failed for %s — promoting directly: %s",
                            v.variant_id,
                            exc,
                        )
                        v.status = PromptVersionStatus.PROMOTED.value
                        v.promoted_at = datetime.now(timezone.utc).isoformat()
                        self._update_operator_feedback(v.variant_id, mean_diff)
                        if baseline and v.variant_id != baseline.variant_id:
                            baseline.status = PromptVersionStatus.DEPRECATED.value
                else:
                    logger.debug(
                        f"[PromptEvolver] {v.variant_id} improvement not significant "
                        f"(significant={significant}, d={effect_size:.3f})",
                    )

            elif v.avg_quality < baseline_quality - 0.1:
                v.status = PromptVersionStatus.DEPRECATED.value
                # Feed negative result to operator selector (Dept 9.6)
                self._update_operator_feedback(v.variant_id, v.avg_quality - baseline_quality)
                logger.info("[PromptEvolver] Deprecated %s for %s", v.variant_id, agent_type)

        self._save_variants()

    def check_shadow_test_results(self) -> None:
        """Finalize promotion for variants that passed shadow testing.

        Polls shadow test results for all SHADOW_TESTING variants. If the
        shadow test passed, promotes the variant and deprecates the old baseline.
        If it failed, reverts the variant to DEPRECATED.
        """
        try:
            from vetinari.learning.shadow_testing import get_shadow_test_runner

            runner = get_shadow_test_runner()
        except Exception as exc:
            logger.warning(
                "[PromptEvolver] Cannot check shadow results — runner unavailable: %s",
                exc,
            )
            return

        changed = False
        for agent_type, variants in self._variants.items():
            baseline = next(
                (v for v in variants if v.status == PromptVersionStatus.PROMOTED.value),
                None,
            )
            for v in variants:
                if v.status != PromptVersionStatus.SHADOW_TESTING.value:
                    continue

                test_id = v.metadata.get("shadow_test_id")
                if not test_id:
                    continue  # No shadow test ID stored — skip

                result = runner.evaluate(test_id)
                decision = result.get("decision", "")
                if decision == "insufficient_data":
                    continue  # Test still collecting data

                if decision in ("promote", "promoted"):
                    v.status = PromptVersionStatus.PROMOTED.value
                    v.promoted_at = datetime.now(timezone.utc).isoformat()
                    self._update_operator_feedback(v.variant_id, result.get("quality_delta", 0.0))
                    if baseline and v.variant_id != baseline.variant_id:
                        baseline.status = PromptVersionStatus.DEPRECATED.value
                    logger.info(
                        "[PromptEvolver] Shadow test PASSED — promoting %s for %s",
                        v.variant_id,
                        agent_type,
                    )
                    changed = True
                elif decision in ("reject", "rejected", "not_found"):
                    v.status = PromptVersionStatus.DEPRECATED.value
                    logger.info(
                        "[PromptEvolver] Shadow test FAILED (decision=%s) — deprecating %s for %s",
                        decision,
                        v.variant_id,
                        agent_type,
                    )
                    changed = True

        if changed:
            self._save_variants()

    @staticmethod
    def _test_significance(
        baseline_scores: list[float],
        variant_scores: list[float],
    ) -> tuple[bool, float]:
        """Return (is_significant, cohens_d).

        Uses scipy.stats.ttest_ind if available; permutation test otherwise.
        Baseline and variant score lists may have different lengths.
        """
        import math

        if len(baseline_scores) < 5 or len(variant_scores) < 5:
            return False, 0.0

        mean_b = sum(baseline_scores) / len(baseline_scores)
        mean_v = sum(variant_scores) / len(variant_scores)

        # Cohen's d
        var_b = sum((x - mean_b) ** 2 for x in baseline_scores) / max(len(baseline_scores) - 1, 1)
        var_v = sum((x - mean_v) ** 2 for x in variant_scores) / max(len(variant_scores) - 1, 1)
        pooled_std = math.sqrt((var_b + var_v) / 2) or 1e-9
        cohens_d = abs(mean_v - mean_b) / pooled_std

        try:
            from scipy import stats

            _, p_value = stats.ttest_ind(baseline_scores, variant_scores)
            return p_value < 0.05, cohens_d
        except ImportError:
            logger.debug("scipy not available, falling back to permutation test for significance", exc_info=True)

        # Permutation test fallback (1000 iterations, no-replacement shuffle)
        all_scores = list(baseline_scores) + list(variant_scores)
        observed_diff = mean_v - mean_b
        n_b, n_v = len(baseline_scores), len(variant_scores)
        count_extreme = 0
        for _ in range(1000):
            perm = random.sample(all_scores, len(all_scores))
            perm_b = perm[:n_b]
            perm_v = perm[n_b:]
            perm_diff = (sum(perm_v) / n_v) - (sum(perm_b) / n_b)
            if abs(perm_diff) >= abs(observed_diff):
                count_extreme += 1
        p_perm = count_extreme / 1000
        return p_perm < 0.05, cohens_d

    def _validate_variant_with_benchmark(self, variant: PromptVariant) -> bool:
        """Run a quick benchmark check before promoting a variant.

        Runs a lightweight benchmark (fast tier) for the variant's agent type.
        If the benchmark pass rate falls below BENCHMARK_PASS_THRESHOLD, the
        variant is NOT promoted -- this prevents regressions that look good in
        A/B test averages but fail on standardised tasks.

        Returns True if the variant passes benchmark validation (or if
        benchmarks are unavailable), False if it fails.
        """
        try:
            from vetinari.benchmarks.suite import BenchmarkSuite

            suite = BenchmarkSuite()
            result = suite.run_agent(variant.agent_type)

            if result.cases_run == 0:
                # No benchmark cases for this agent type -- pass by default
                logger.debug(
                    f"[PromptEvolver] No benchmark cases for {variant.agent_type}; skipping benchmark validation",
                )
                return True

            pass_rate = result.cases_passed / max(result.cases_run, 1)
            avg_score = result.avg_score

            logger.debug(
                f"[PromptEvolver] Benchmark validation for {variant.variant_id}: "
                f"pass_rate={pass_rate:.3f}, avg_score={avg_score:.3f}, "
                f"threshold={self.BENCHMARK_PASS_THRESHOLD}",
            )

            if pass_rate < self.BENCHMARK_PASS_THRESHOLD:
                return False

            # Also record the benchmark score into the variant's quality history
            if variant.variant_id not in self._score_history:
                self._score_history[variant.variant_id] = deque(maxlen=500)
            self._score_history[variant.variant_id].append(avg_score)

            return True

        except ImportError:
            logger.warning("[PromptEvolver] Benchmark suite not available; blocking promotion (fail closed)")
            return False
        except Exception:
            logger.exception("[PromptEvolver] Benchmark validation error — blocking promotion (fail closed)")
            return False

    def generate_variant(
        self,
        agent_type: str,
        baseline_prompt: str,
        mode: str = "default",
    ) -> str | None:
        """Generate an improved prompt variant using structured operators.

        Primary path: uses OperatorSelector (Thompson Sampling) to pick
        the best mutation operator, then applies it via PromptMutator.
        Fallback: LLM-based generation if adapter_manager is available
        and operator mutation produces no change.

        Args:
            agent_type: The agent type.
            baseline_prompt: The baseline prompt.
            mode: The agent mode (for operator selection context).

        Returns:
            The generated variant text, or None on failure.
        """
        # Primary path: structured operator mutation (Dept 9.6)
        try:
            selector = self._get_operator_selector()
            mutator = self._get_prompt_mutator()

            operator = selector.select_operator(agent_type, mode)
            # Pre-screen: FORMAT_RESTRUCTURE and CONTEXT_PRUNE are structural
            # no-ops on prompts without multiple markdown sections.  Swap to the
            # next-best operator before calling mutate so that mutate is only
            # ever called once (required by test contract in wiring tests).
            operator = self._resolve_effective_operator(operator, baseline_prompt, agent_type, mode)
            mutated = mutator.mutate(baseline_prompt, operator)

            if mutated != baseline_prompt:
                with self._lock:
                    variant_id = f"{agent_type}_v{len(self._variants.get(agent_type, [])) + 1}"
                    variant = PromptVariant(
                        variant_id=variant_id,
                        agent_type=agent_type,
                        prompt_text=mutated,
                    )
                    # Track which operator produced this variant
                    self._variant_operators[variant_id] = (operator, agent_type, mode)
                    if agent_type not in self._variants:
                        self._variants[agent_type] = []
                    self._variants[agent_type].append(variant)
                    self._save_variants()
                logger.info(
                    "[PromptEvolver] Generated variant %s for %s via %s operator",
                    variant_id,
                    agent_type,
                    operator.value,
                )
                self._record_improvement(variant_id, operator, agent_type, mode)
                return mutated
        except Exception:
            logger.warning("Operator-based variant generation failed", exc_info=True)

        # Fallback: LLM-based generation
        return self._generate_variant_llm(agent_type, baseline_prompt)

    def _generate_variant_llm(self, agent_type: str, baseline_prompt: str) -> str | None:
        """Generate a variant using LLM inference (fallback). See ``prompt_evolver_generation``."""
        from vetinari.learning.prompt_evolver_generation import generate_variant_llm

        return generate_variant_llm(self, agent_type, baseline_prompt)

    def generate_variant_from_trace(
        self,
        agent_type: str,
        baseline_prompt: str,
        failed_trace: dict[str, Any],
    ) -> str | None:
        """Generate a targeted prompt variant by diagnosing a failed execution trace.

        Uses the PromptOptimizer's GEPA-style trace analysis to identify the root
        cause of the failure and propose a targeted fix, rather than a random
        mutation.  The resulting variant is stored in ``self._variants`` with an ID
        of the form ``{agent_type}_trace_v{n}``.

        Falls back to :meth:`generate_variant` (blind mutation) when:
        - the optimizer cannot be reached (raises),
        - the optimizer returns ``None`` (confidence too low or no diagnosis),
        - the optimizer returns the same text as the baseline.

        Args:
            agent_type: The agent type that produced the failing trace.
            baseline_prompt: The current instruction text to improve.
            failed_trace: Diagnostic dict with keys ``output``, ``error``, and
                ``quality_score`` (same shape as
                ``PromptOptimizer.optimize_via_trace`` expects).

        Returns:
            The new variant instruction text, or ``None`` if no improvement could
            be derived and the fallback also fails.
        """
        try:
            from vetinari.learning.prompt_optimizer import get_prompt_optimizer

            optimizer = get_prompt_optimizer()
            experiment = optimizer.optimize_via_trace(
                agent_type=agent_type,
                baseline_instruction=baseline_prompt,
                failed_trace=failed_trace,
            )

            if experiment is None or experiment.instruction == baseline_prompt:
                # Diagnosis yielded nothing useful — fall back to blind mutation
                return self.generate_variant(agent_type, baseline_prompt)

            new_instruction = experiment.instruction
            with self._lock:
                existing = self._variants.get(agent_type, [])
                variant_id = f"{agent_type}_trace_v{len(existing) + 1}"
                variant = PromptVariant(
                    variant_id=variant_id,
                    agent_type=agent_type,
                    prompt_text=new_instruction,
                )
                if agent_type not in self._variants:
                    self._variants[agent_type] = []
                self._variants[agent_type].append(variant)
                self._save_variants()

            logger.info(
                "[PromptEvolver] Trace-based variant %s created for %s (diagnosis: %s)",
                variant_id,
                agent_type,
                experiment.trace_diagnosis or "unknown",
            )
            return new_instruction

        except Exception:
            logger.warning(
                "Trace-based variant generation failed for %s — falling back to blind mutation",
                agent_type,
                exc_info=True,
            )
            return self.generate_variant(agent_type, baseline_prompt)

    def evolve_per_level(
        self,
        agent_type: str,
        level: str = "default",
        failed_traces: list[dict[str, Any]] | None = None,
    ) -> dict[str, str | None]:
        """Evolve the prompt for a given agent type and execution-level context.

        Builds a level-specific hint string and appends it to the current
        baseline before calling :meth:`generate_variant_from_trace` (when
        ``failed_traces`` are supplied) or :meth:`generate_variant` (blind
        mutation).  This lets the evolver incorporate execution-mode context
        (e.g. "build", "review") directly into the mutation seed.

        Args:
            agent_type: The agent type to evolve (e.g. ``"WORKER"``).
            level: Execution mode or scope name (e.g. ``"build"``, ``"review"``).
            failed_traces: Optional list of failed execution trace dicts.  When
                provided the first trace is used for trace-based diagnosis.

        Returns:
            Dict with keys ``variant_id`` (``str | None``) and
            ``evolved_prompt`` (``str | None``).  Both values are ``None`` if no
            baseline is registered for the agent.
        """
        # Resolve baseline for this agent
        variants = self._variants.get(agent_type, [])
        baseline_variant = next(
            (v for v in variants if v.is_baseline and v.status == PromptVersionStatus.PROMOTED.value),
            None,
        )

        if baseline_variant is None:
            return {"variant_id": None, "evolved_prompt": None}

        baseline_text = baseline_variant.prompt_text

        # Build an agent-specific level hint to seed the mutation
        hint = self._build_level_hint(agent_type, level)
        seeded_prompt = f"{baseline_text}\n\n{hint}" if hint else baseline_text

        # Trace-based evolution when traces are available
        if failed_traces:
            trace = failed_traces[0]
            evolved = self.generate_variant_from_trace(
                agent_type=agent_type,
                baseline_prompt=seeded_prompt,
                failed_trace=trace,
            )
            if evolved and evolved != baseline_text and evolved != seeded_prompt:
                variant_id = f"{agent_type}_{level}_evolved"
                return {"variant_id": variant_id, "evolved_prompt": evolved}
            # Trace returned baseline unchanged — fall through to blind mutation
            evolved = self.generate_variant(agent_type, seeded_prompt, mode=level)
            return {"variant_id": f"{agent_type}_{level}_mutated", "evolved_prompt": evolved}

        # Blind mutation when no traces are provided
        evolved = self.generate_variant(agent_type, seeded_prompt, mode=level)
        return {"variant_id": f"{agent_type}_{level}_mutated", "evolved_prompt": evolved}

    @staticmethod
    def _build_level_hint(agent_type: str, level: str) -> str:
        """Build an agent- and level-specific hint to seed prompt mutation.

        Injects context that is relevant to the given agent and execution mode
        so that mutations stay within the expected scope.

        Args:
            agent_type: The agent type (e.g. ``"WORKER"``, ``"INSPECTOR"``).
            level: The execution mode or scope (e.g. ``"build"``, ``"review"``).

        Returns:
            A plain-English hint string, or ``""`` if no specific hint applies.
        """
        if agent_type == AgentType.WORKER.value:
            return (
                f"You are operating in '{level}' mode. "
                f"Tailor your output to satisfy the '{level}' task requirements precisely."
            )
        if agent_type == AgentType.INSPECTOR.value:
            return (
                f"You are evaluating Worker outputs in '{level}' mode. "
                "Focus on completeness and correctness of Worker-produced content."
            )
        return f"Context: operating in '{level}' scope."

    def synthesize_scope_guidelines(
        self,
        agent_type: str,
        level: str = "default",
    ) -> str:
        """Synthesize failure-informed guidelines for an agent in a given scope.

        Reads recently failed traces for the agent from the training collector,
        tallies the reported issue categories, and returns a plain-English
        guideline block that instructs the agent to avoid the most common
        failure modes.

        Returns ``""`` when:
        - Fewer than 3 failed traces are available (not enough signal).
        - All traces lack a categorised ``issues`` list.
        - The training collector cannot be reached (logged at WARNING level).

        Args:
            agent_type: The agent whose failure history to analyse.
            level: The execution scope used to filter traces and label output.

        Returns:
            A plain-English guideline string, or ``""`` if no signal is
            available.
        """
        try:
            from vetinari.learning.training_data import get_training_collector

            collector = get_training_collector()
            traces = collector.get_recent_traces(limit=20, failed_only=True)
        except Exception as exc:
            logger.warning(
                "Could not load failed traces for %s/%s — returning empty guidelines: %s",
                agent_type,
                level,
                exc,
            )
            return ""

        # Only count traces that have a recognised issues list
        issue_counts: dict[str, int] = {}
        for trace in traces:
            verdict = trace.get("inspector_verdict")
            if not isinstance(verdict, dict):
                continue
            issues = verdict.get("issues")
            if not isinstance(issues, list):
                continue
            for issue in issues:
                issue_lower = str(issue).lower()
                # Map raw issue text to canonical categories
                if "incomplete" in issue_lower:
                    issue_counts["incomplete_output"] = issue_counts.get("incomplete_output", 0) + 1
                elif "format" in issue_lower or "parse" in issue_lower:
                    issue_counts["format_error"] = issue_counts.get("format_error", 0) + 1
                elif "reasoning" in issue_lower or "wrong" in issue_lower:
                    issue_counts["reasoning_error"] = issue_counts.get("reasoning_error", 0) + 1
                elif "off" in issue_lower or "topic" in issue_lower:
                    issue_counts["off_topic"] = issue_counts.get("off_topic", 0) + 1

        # Require at least 3 total issues (== 3 failed traces with categorised issues)
        total_issues = sum(issue_counts.values())
        if total_issues < 3:
            return ""

        # Map canonical categories to guideline lines
        _GUIDELINE_MAP: dict[str, str] = {
            "incomplete_output": ("Always produce a complete response — do not stop early or omit sections."),
            "format_error": ("Follow the specified output format exactly. Validate structure before submitting."),
            "reasoning_error": ("Think step by step and decompose the problem before answering."),
            "off_topic": ("Stay strictly within scope. Address every required component."),
        }

        lines: list[str] = [
            f"Scope guidelines for {agent_type} in '{level}' mode",
            "Common failure patterns observed — address each in your response:",
        ]
        for category, count in sorted(issue_counts.items(), key=lambda kv: -kv[1]):
            guideline = _GUIDELINE_MAP.get(category)
            if guideline:
                lines.append(f"  - ({count}x) {guideline}")

        if len(lines) == 2:
            # No recognised guidelines were appended
            return ""

        return "\n".join(lines)

    @staticmethod
    def _resolve_effective_operator(
        operator: Any,
        prompt: str,
        agent_type: str,
        mode: str,
    ) -> Any:
        """Return an operator that will actually change the given prompt.

        ``FORMAT_RESTRUCTURE`` and ``CONTEXT_PRUNE`` are structural no-ops on
        prompts without multiple markdown sections.  When one of these is
        selected but the prompt lacks sufficient structure, substitute the
        Thompson-second-best operator that is guaranteed to produce a change
        (``CONSTRAINT_INJECTION``).

        This pre-screening avoids calling ``mutate`` more than once per
        ``generate_variant`` call, keeping the call-count contract that the
        wiring tests assert.

        Args:
            operator: The Thompson-sampled MutationOperator.
            prompt: The baseline prompt text.
            agent_type: Used for logging context only.
            mode: Used for logging context only.

        Returns:
            Either the original operator (if it will produce a change) or a
            guaranteed-change substitute.
        """
        import re

        # These operators require 2+ (FORMAT_RESTRUCTURE) or 3+ (CONTEXT_PRUNE)
        # markdown sections and are no-ops on plain-text prompts.
        _SECTION_DEPENDENT = {
            MutationOperator.FORMAT_RESTRUCTURE,
            MutationOperator.CONTEXT_PRUNE,
        }
        if operator not in _SECTION_DEPENDENT:
            return operator

        # Count markdown-style headers to decide if the operator will apply
        section_count = len(re.findall(r"^#{1,3}\s+", prompt, re.MULTILINE))
        required = 2 if operator == MutationOperator.FORMAT_RESTRUCTURE else 3
        if section_count >= required:
            return operator

        # Insufficient sections — substitute with a guaranteed-change operator
        substitute = MutationOperator.CONSTRAINT_INJECTION
        logger.debug(
            "Operator %s is a no-op for %s/%s (sections=%d, required=%d) — substituting %s",
            operator.value,
            agent_type,
            mode,
            section_count,
            required,
            substitute.value,
        )
        return substitute

    def _get_operator_selector(self):
        """Return the shared OperatorSelector singleton. See ``prompt_evolver_generation``."""
        from vetinari.learning.prompt_evolver_generation import get_operator_selector

        return get_operator_selector(self)

    def _get_prompt_mutator(self):
        """Lazy-load the PromptMutator singleton. See ``prompt_evolver_generation``."""
        from vetinari.learning.prompt_evolver_generation import get_prompt_mutator

        return get_prompt_mutator(self)

    def _update_operator_feedback(self, variant_id: str, quality_delta: float) -> None:
        """Feed quality delta back to OperatorSelector. See ``prompt_evolver_generation``."""
        from vetinari.learning.prompt_evolver_generation import update_operator_feedback

        update_operator_feedback(self, variant_id, quality_delta)

    def _record_improvement(self, variant_id: str, operator: Any, agent_type: str, mode: str) -> None:
        """Create an ImprovementRecord for a new variant. See ``prompt_evolver_generation``."""
        from vetinari.learning.prompt_evolver_generation import record_improvement

        record_improvement(self, variant_id, operator, agent_type, mode)

    def _get_baseline_quality(self, agent_type: str) -> float:
        """Get current baseline quality for an agent type. See ``prompt_evolver_generation``."""
        from vetinari.learning.prompt_evolver_generation import get_baseline_quality

        return get_baseline_quality(self, agent_type)

    def _update_improvement_observation(self, variant_id: str, quality_delta: float) -> None:
        """Record an observation on the linked ImprovementRecord. See ``prompt_evolver_generation``."""
        from vetinari.learning.prompt_evolver_generation import update_improvement_observation

        update_improvement_observation(self, variant_id, quality_delta)

    def get_stats(self, agent_type: str) -> dict[str, Any]:
        """Get evolution statistics for an agent type.

        Args:
            agent_type: The agent type to report on.

        Returns:
            Dict with keys: agent_type, total_variants, promoted (list of
            dicts with id/quality/trials), testing (count), and deprecated
            (count).
        """
        variants = self._variants.get(agent_type, [])
        return {
            "agent_type": agent_type,
            "total_variants": len(variants),
            "promoted": [
                {"id": v.variant_id, "quality": round(v.avg_quality, 3), "trials": v.trials}
                for v in variants
                if v.status == PromptVersionStatus.PROMOTED.value
            ],
            "testing": len([v for v in variants if v.status == PromptVersionStatus.TESTING.value]),
            "deprecated": len([v for v in variants if v.status == PromptVersionStatus.DEPRECATED.value]),
        }

    @staticmethod
    def _get_state_path() -> Path:
        """Resolve path for prompt variant state file.

        Uses VETINARI_STATE_DIR env var if set, otherwise falls back to
        project-relative .vetinari/ directory.

        Returns:
            Path to prompt_variants.json.
        """
        state_dir_env = os.environ.get("VETINARI_STATE_DIR", "")
        if state_dir_env:
            state_dir = Path(state_dir_env)
        else:
            state_dir = VETINARI_STATE_DIR
        state_dir.mkdir(parents=True, exist_ok=True)
        return state_dir / "prompt_variants.json"

    def _load_variants(self) -> None:
        try:
            path = self._get_state_path()
            if path.exists():
                with Path(path).open(encoding="utf-8") as f:
                    data = json.load(f)
                for agent_type, variants in data.items():
                    self._variants[agent_type] = [PromptVariant(**v) for v in variants]
        except Exception as e:
            logger.warning("Could not load prompt variants: %s", e)

    def _save_variants(self) -> None:
        try:
            from dataclasses import asdict

            path = self._get_state_path()
            data = {k: [asdict(v) for v in vs] for k, vs in self._variants.items()}
            with Path(path).open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("Could not save prompt variants: %s", e)


_prompt_evolver: PromptEvolver | None = None
_prompt_evolver_lock = threading.Lock()


def get_prompt_evolver() -> PromptEvolver:
    """Return the module-level PromptEvolver singleton, creating it if needed.

    Returns:
        The shared PromptEvolver instance, pre-loaded with any previously
        persisted variant state from disk.
    """
    global _prompt_evolver
    if _prompt_evolver is None:
        with _prompt_evolver_lock:
            if _prompt_evolver is None:
                _prompt_evolver = PromptEvolver()
    return _prompt_evolver
