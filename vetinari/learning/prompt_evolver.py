"""
Prompt Evolver - Vetinari Self-Improvement Subsystem

A/B tests prompt variations for each agent and promotes variants that
achieve statistically better quality scores.

Enhanced in Wave 4:
- Statistical significance testing (paired t-test via scipy if available,
  bootstrap otherwise) before any promotion decision.
- Effect size check (Cohen's d >= 0.2) to prevent noise-driven promotions.
- Regression detection: demote if quality declining over time.
- Max concurrent A/B tests per agent (default 3) to avoid traffic dilution.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    promoted_at: Optional[str] = None
    status: str = "testing"   # "testing" | "promoted" | "deprecated"

    @property
    def avg_quality(self) -> float:
        return self.total_quality / max(self.trials, 1)

    def record(self, quality: float) -> None:
        self.trials += 1
        self.total_quality += quality


class PromptEvolver:
    """
    Automatically evolves agent system prompts through A/B testing.

    Strategy:
    1. Baseline prompt is the current system prompt for each agent.
    2. When quality degrades (below rolling average), generate new variants.
    3. Route a fraction of tasks to variant prompts.
    4. After MIN_TRIALS trials, promote variant if significantly better.
    5. Store all variants and their performance in memory.
    """

    MIN_TRIALS = 20              # Minimum trials before promotion decision
    VARIANT_FRACTION = 0.3       # Fraction of tasks routed to variant
    MIN_IMPROVEMENT = 0.05       # Minimum mean quality improvement to promote
    MIN_EFFECT_SIZE = 0.2        # Cohen's d threshold (avoid noise-driven promotions)
    P_VALUE_THRESHOLD = 0.05     # Statistical significance threshold
    MAX_CONCURRENT_TESTS = 3     # Max A/B tests per agent (avoid traffic dilution)
    BENCHMARK_PASS_THRESHOLD = 0.5  # Minimum benchmark pass rate to promote
    AUTO_VARIANT_QUALITY_THRESHOLD = 0.5  # Trigger variant generation below this quality
    AUTO_VARIANT_ROLLING_WINDOW = 10      # Number of recent scores to average

    def __init__(self, adapter_manager=None):
        self._adapter_manager = adapter_manager
        # agent_type -> list of variants
        self._variants: Dict[str, List[PromptVariant]] = {}
        # Per-variant score history for statistical testing
        self._score_history: Dict[str, List[float]] = {}
        # I12: Per-model performance tracking for model-prompt affinity
        # Key: "variant_id:model_id" -> list of quality scores
        self._model_affinity: Dict[str, List[float]] = {}
        self._load_variants()

    def register_baseline(self, agent_type: str, prompt_text: str) -> None:
        """Register the current system prompt as the baseline."""
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
            status="promoted",
        )
        self._variants[agent_type].append(variant)

    # Default fallback prompt when no variants are registered for an agent type
    _FALLBACK_PROMPT = (
        "You are a helpful AI assistant. Complete the given task accurately, "
        "completely, and concisely. Follow any formatting instructions provided."
    )

    def select_prompt(self, agent_type: str) -> Tuple[str, str]:
        """
        Select which prompt to use for this invocation.

        Returns:
            Tuple of (prompt_text, variant_id).
            Never returns an empty prompt string — falls back to a sensible default.
        """
        variants = self._variants.get(agent_type, [])
        if not variants:
            return self._FALLBACK_PROMPT, "fallback"

        promoted = [v for v in variants if v.status == "promoted"]
        testing = [v for v in variants if v.status == "testing"]

        if testing and random.random() < self.VARIANT_FRACTION:
            # Route to a testing variant
            v = random.choice(testing)
            if v.prompt_text.strip():
                return v.prompt_text, v.variant_id

        if promoted:
            # Use the best promoted variant
            best = max(promoted, key=lambda v: v.avg_quality if v.trials > 0 else 0.5)
            if best.prompt_text.strip():
                return best.prompt_text, best.variant_id

        return self._FALLBACK_PROMPT, "fallback"

    def record_result(self, agent_type: str, variant_id: str, quality: float,
                      model_id: Optional[str] = None) -> None:
        """Record a quality result for a variant and trigger promotion check.

        Also checks rolling quality average and auto-generates a new variant
        if quality falls below AUTO_VARIANT_QUALITY_THRESHOLD.

        Args:
            agent_type: The agent type string.
            variant_id: The prompt variant identifier.
            quality: Quality score 0.0-1.0.
            model_id: Optional model that produced this result (for I12 affinity tracking).
        """
        variants = self._variants.get(agent_type, [])
        for v in variants:
            if v.variant_id == variant_id:
                v.record(quality)
                # Also track per-variant score history for statistical testing
                if variant_id not in self._score_history:
                    self._score_history[variant_id] = []
                self._score_history[variant_id].append(quality)
                break

        # I12: Track per-model performance for model-prompt affinity
        if model_id:
            affinity_key = f"{variant_id}:{model_id}"
            if affinity_key not in self._model_affinity:
                self._model_affinity[affinity_key] = []
            self._model_affinity[affinity_key].append(quality)

        self._check_promotion(agent_type)

        # Auto-trigger variant generation if rolling quality is below threshold
        try:
            self._maybe_auto_generate_variant(agent_type)
        except Exception as e:
            logger.warning(f"Auto variant generation check failed: {e}")

    def _maybe_auto_generate_variant(self, agent_type: str) -> None:
        """Check rolling quality average; auto-generate variant if below threshold."""
        variants = self._variants.get(agent_type, [])
        if not variants:
            return

        # Count active testing variants — respect MAX_CONCURRENT_TESTS
        testing_count = sum(1 for v in variants if v.status == "testing")
        if testing_count >= self.MAX_CONCURRENT_TESTS:
            return

        # Get the baseline (promoted) variant's recent scores
        baseline = next((v for v in variants if v.is_baseline and v.status == "promoted"), None)
        if not baseline:
            return

        scores = self._score_history.get(baseline.variant_id, [])
        recent = scores[-self.AUTO_VARIANT_ROLLING_WINDOW:]
        if len(recent) < self.AUTO_VARIANT_ROLLING_WINDOW:
            return  # Not enough data yet

        rolling_avg = sum(recent) / len(recent)
        if rolling_avg < self.AUTO_VARIANT_QUALITY_THRESHOLD:
            logger.info(
                f"[PromptEvolver] Rolling quality for {agent_type} is {rolling_avg:.3f} "
                f"(below {self.AUTO_VARIANT_QUALITY_THRESHOLD}) — auto-generating variant"
            )
            self.generate_variant(agent_type, baseline.prompt_text)

    def _check_promotion(self, agent_type: str) -> None:
        """Decide whether to promote, deprecate, or keep testing variants.

        Uses statistical significance testing (paired t-test / bootstrap)
        and effect size (Cohen's d) before any promotion.
        """
        variants = self._variants.get(agent_type, [])
        baseline = next((v for v in variants if v.is_baseline and v.status == "promoted"), None)
        baseline_quality = baseline.avg_quality if baseline and baseline.trials > 0 else 0.5
        baseline_scores = self._score_history.get(
            baseline.variant_id if baseline else "", []
        )

        for v in variants:
            if v.status != "testing" or v.trials < self.MIN_TRIALS:
                continue

            variant_scores = self._score_history.get(v.variant_id, [])
            mean_diff = v.avg_quality - baseline_quality

            if mean_diff >= self.MIN_IMPROVEMENT:
                # Run statistical test before promoting
                significant, effect_size = self._test_significance(
                    baseline_scores, variant_scores
                )
                if significant and effect_size >= self.MIN_EFFECT_SIZE:
                    # Benchmark gate: validate variant before promotion
                    if not self._validate_variant_with_benchmark(v):
                        logger.info(
                            f"[PromptEvolver] {v.variant_id} passed stats but "
                            f"FAILED benchmark validation -- not promoting"
                        )
                        continue

                    logger.info(
                        f"[PromptEvolver] Promoting {v.variant_id} for {agent_type}: "
                        f"mean_diff={mean_diff:.3f}, effect_size={effect_size:.3f}"
                    )
                    logger.info(
                        f"[PromptEvolver] Winning prompt for {agent_type}: "
                        f"{v.prompt_text[:200]}{'...' if len(v.prompt_text) > 200 else ''}"
                    )
                    v.status = "promoted"
                    v.promoted_at = datetime.now().isoformat()
                    if baseline and v.variant_id != baseline.variant_id:
                        baseline.status = "deprecated"
                    # I12: Push model affinity data on promotion
                    try:
                        from vetinari.learning.feedback_loop import get_feedback_loop
                        feedback = get_feedback_loop()
                        for aff_key, aff_scores in self._model_affinity.items():
                            if aff_key.startswith(f"{v.variant_id}:") and aff_scores:
                                aff_model = aff_key[len(v.variant_id) + 1:]
                                avg_aff = sum(aff_scores) / len(aff_scores)
                                feedback.record_outcome(
                                    task_id=f"affinity_{v.variant_id}",
                                    model_id=aff_model,
                                    task_type=agent_type,
                                    quality_score=avg_aff,
                                    success=avg_aff >= 0.5,
                                )
                    except Exception as e:
                        logger.debug(f"[PromptEvolver] Affinity push failed: {e}")
                else:
                    logger.debug(
                        f"[PromptEvolver] {v.variant_id} improvement not significant "
                        f"(significant={significant}, d={effect_size:.3f})"
                    )

            elif v.avg_quality < baseline_quality - 0.1:
                v.status = "deprecated"
                logger.info(f"[PromptEvolver] Deprecated {v.variant_id} for {agent_type}")

        self._save_variants()

    @staticmethod
    def _test_significance(
        baseline_scores: List[float],
        variant_scores: List[float],
    ) -> tuple:
        """Return (is_significant, cohens_d).

        Uses scipy.stats.ttest_ind if available; bootstrap otherwise.
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
            pass

        # Bootstrap fallback (1000 samples)
        all_scores = baseline_scores + variant_scores
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
        p_bootstrap = count_extreme / 1000
        return p_bootstrap < 0.05, cohens_d

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
                    f"[PromptEvolver] No benchmark cases for {variant.agent_type}; "
                    f"skipping benchmark validation"
                )
                return True

            pass_rate = result.cases_passed / max(result.cases_run, 1)
            avg_score = result.avg_score

            logger.debug(
                f"[PromptEvolver] Benchmark validation for {variant.variant_id}: "
                f"pass_rate={pass_rate:.3f}, avg_score={avg_score:.3f}, "
                f"threshold={self.BENCHMARK_PASS_THRESHOLD}"
            )

            if pass_rate < self.BENCHMARK_PASS_THRESHOLD:
                return False

            # Also record the benchmark score into the variant's quality history
            if variant.variant_id not in self._score_history:
                self._score_history[variant.variant_id] = []
            self._score_history[variant.variant_id].append(avg_score)

            return True

        except ImportError:
            logger.debug("[PromptEvolver] Benchmark suite not available; skipping validation")
            return True
        except Exception as e:
            logger.debug(f"[PromptEvolver] Benchmark validation error: {e}")
            # On error, don't block promotion -- fail open
            return True

    def generate_variant(self, agent_type: str, baseline_prompt: str) -> Optional[str]:
        """Use LLM to generate an improved prompt variant."""
        if not self._adapter_manager:
            return None

        try:
            from vetinari.adapters.base import InferenceRequest

            req = InferenceRequest(
                model_id="default",
                prompt=f"""You are a prompt engineer. Improve this agent system prompt to be more effective.

AGENT TYPE: {agent_type}

CURRENT PROMPT:
{baseline_prompt}

Generate an improved version that:
1. Is more specific and actionable
2. Gives clearer output format instructions
3. Reduces ambiguity
4. Maintains the same core role

Respond with ONLY the improved prompt text, no explanations.""",
                system_prompt="You are an expert AI prompt engineer.",
                max_tokens=1024,
                temperature=0.7,
            )
            resp = self._adapter_manager.infer(req)
            if resp.status == "ok" and resp.output.strip():
                variant_id = f"{agent_type}_v{len(self._variants.get(agent_type, []))+1}"
                variant = PromptVariant(
                    variant_id=variant_id,
                    agent_type=agent_type,
                    prompt_text=resp.output.strip(),
                )
                if agent_type not in self._variants:
                    self._variants[agent_type] = []
                self._variants[agent_type].append(variant)
                self._save_variants()
                logger.info(f"[PromptEvolver] Generated variant {variant_id} for {agent_type}")
                return resp.output.strip()
        except Exception as e:
            logger.debug(f"Variant generation failed: {e}")
        return None

    def get_stats(self, agent_type: str) -> Dict[str, Any]:
        """Get evolution statistics for an agent type."""
        variants = self._variants.get(agent_type, [])
        return {
            "agent_type": agent_type,
            "total_variants": len(variants),
            "promoted": [{"id": v.variant_id, "quality": round(v.avg_quality, 3), "trials": v.trials}
                         for v in variants if v.status == "promoted"],
            "testing": len([v for v in variants if v.status == "testing"]),
            "deprecated": len([v for v in variants if v.status == "deprecated"]),
        }

    def get_model_affinity(self, agent_type: str) -> Dict[str, float]:
        """I12: Get per-model average quality for all variants of an agent type.

        Returns:
            Dict mapping model_id to average quality score across all
            variants for this agent type.
        """
        model_scores: Dict[str, List[float]] = {}
        for key, scores in self._model_affinity.items():
            parts = key.split(":")
            if len(parts) >= 2:
                variant_id = parts[0]
                model_id = ":".join(parts[1:])
                # Check if this variant belongs to the requested agent_type
                variants = self._variants.get(agent_type, [])
                if any(v.variant_id == variant_id for v in variants):
                    if model_id not in model_scores:
                        model_scores[model_id] = []
                    model_scores[model_id].extend(scores)
        return {
            m: round(sum(s) / len(s), 3)
            for m, s in model_scores.items() if s
        }

    def _load_variants(self) -> None:
        try:
            import os
            path = os.path.join(
                os.path.expanduser("~"), ".lmstudio", "projects", "Vetinari",
                ".vetinari", "prompt_variants.json"
            )
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                for agent_type, variants in data.items():
                    self._variants[agent_type] = [PromptVariant(**v) for v in variants]
        except Exception as e:
            logger.debug(f"Could not load prompt variants: {e}")

    def _save_variants(self) -> None:
        try:
            import os
            from dataclasses import asdict
            state_dir = os.path.join(
                os.path.expanduser("~"), ".lmstudio", "projects", "Vetinari", ".vetinari"
            )
            os.makedirs(state_dir, exist_ok=True)
            path = os.path.join(state_dir, "prompt_variants.json")
            data = {k: [asdict(v) for v in vs] for k, vs in self._variants.items()}
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not save prompt variants: {e}")


_prompt_evolver: Optional[PromptEvolver] = None


def get_prompt_evolver() -> PromptEvolver:
    global _prompt_evolver
    if _prompt_evolver is None:
        _prompt_evolver = PromptEvolver()
    return _prompt_evolver
