"""Training Quality Gate — evaluates trained models against baseline before deployment.

After training completes, this module runs the trained model against a held-out
evaluation set and compares quality, latency, and token efficiency against the
baseline. Decision logic: worse -> reject, better -> deploy, marginal -> flag.

This is step 4 of the training pipeline:
Data Curation -> Training -> **Quality Gate** -> Deployment.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

# Optional heavy dependencies — imported at module level so they can be patched
# in tests and so local-import overhead is not paid per evaluation call.
# If the adapter or scorer is unavailable at import time, the attribute is set
# to None and _evaluate_model falls back gracefully.
try:
    from vetinari.adapters.llama_cpp_local_adapter import LocalInferenceAdapter
except Exception:
    LocalInferenceAdapter = None  # type: ignore[assignment,misc]

try:
    from vetinari.learning.quality_scorer import get_quality_scorer
except Exception:
    get_quality_scorer = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# -- Module-level state --
# _quality_gate: singleton instance, created on first call to get_training_quality_gate().
# Protected by: _quality_gate_lock (double-checked locking).
_quality_gate: TrainingQualityGate | None = None
_quality_gate_lock = threading.Lock()

# Thresholds that drive the three-way gate decision.
# Reject if quality drops more than 3 percentage points.
_QUALITY_REJECT_THRESHOLD: float = -0.03
# Deploy if quality improves at least 2 percentage points.
_QUALITY_DEPLOY_THRESHOLD: float = 0.02
# Reject if candidate latency exceeds 2x the baseline.
_LATENCY_REJECT_RATIO: float = 2.0
# Reject if candidate uses 50% more tokens than the baseline.
_TOKEN_REJECT_RATIO: float = 1.5


@dataclass
class TrainingGateDecision:
    """Result of a training quality gate evaluation.

    Attributes:
        decision: One of "deploy", "reject", or "flag_for_review".
        baseline_quality: Average quality score for the baseline model (0-1).
        candidate_quality: Average quality score for the candidate model (0-1).
        quality_delta: candidate_quality minus baseline_quality.
        baseline_latency_ms: Average inference latency for baseline (ms).
        candidate_latency_ms: Average inference latency for candidate (ms).
        latency_ratio: candidate_latency_ms / baseline_latency_ms. Values <1 mean faster.
        token_efficiency: candidate tokens / baseline tokens. Values <1 mean more efficient.
        reasoning: Human-readable explanation of the decision.
        timestamp: ISO-8601 UTC timestamp of when the gate ran.
        eval_tasks_run: Number of evaluation tasks executed.
    """

    decision: str  # "deploy", "reject", "flag_for_review"
    baseline_quality: float
    candidate_quality: float
    quality_delta: float
    baseline_latency_ms: float
    candidate_latency_ms: float
    latency_ratio: float  # candidate/baseline; <1.0 means faster
    token_efficiency: float  # candidate tokens / baseline tokens; <1.0 means more efficient
    reasoning: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    eval_tasks_run: int = 0

    def __repr__(self) -> str:
        return (
            f"GateDecision(decision={self.decision!r}, "
            f"quality_delta={self.quality_delta:+.3f}, "
            f"latency_ratio={self.latency_ratio:.2f})"
        )


class TrainingQualityGate:
    """Evaluates trained models against baseline before deployment.

    Runs a held-out evaluation set through both the baseline and candidate
    models, then compares quality scores, latency, and token efficiency.
    The result is a three-way gate decision: deploy, reject, or flag for review.

    Side effects:
      - Logs gate decisions at INFO level.
      - Stores all decisions in memory for history inspection.
    """

    def __init__(self) -> None:
        # _decisions: audit log of every gate evaluation run this session.
        # Protected by _lock to allow concurrent evaluate() calls.
        self._decisions: list[TrainingGateDecision] = []
        self._lock = threading.Lock()

    def evaluate(
        self,
        candidate_model_id: str,
        baseline_model_id: str,
        eval_tasks: list[dict[str, str]] | None = None,
    ) -> TrainingGateDecision:
        """Run quality gate evaluation comparing candidate against baseline.

        Executes each eval task against both models, aggregates quality,
        latency, and token metrics, then applies the decision thresholds.

        Args:
            candidate_model_id: The newly trained model to evaluate.
            baseline_model_id: The current production model to compare against.
            eval_tasks: Optional list of eval task dicts. Each dict must have
                a "prompt" key and may have "task_type" and "expected" keys.
                If None, the default evaluation set is used.

        Returns:
            GateDecision with a deploy/reject/flag_for_review verdict and
            supporting metrics.
        """
        if eval_tasks is None:
            eval_tasks = self._get_default_eval_set()

        if not eval_tasks:
            logger.warning(
                "[TrainingQualityGate] No eval tasks available for %s vs %s — flagging for review",
                candidate_model_id,
                baseline_model_id,
            )
            return TrainingGateDecision(
                decision="flag_for_review",
                baseline_quality=0.0,
                candidate_quality=0.0,
                quality_delta=0.0,
                baseline_latency_ms=0.0,
                candidate_latency_ms=0.0,
                latency_ratio=1.0,
                token_efficiency=1.0,
                reasoning="No evaluation tasks available — cannot assess quality",
                eval_tasks_run=0,
            )

        baseline_scores = self._evaluate_model(baseline_model_id, eval_tasks)
        candidate_scores = self._evaluate_model(candidate_model_id, eval_tasks)

        b_quality = sum(s["quality"] for s in baseline_scores) / max(len(baseline_scores), 1)
        c_quality = sum(s["quality"] for s in candidate_scores) / max(len(candidate_scores), 1)
        b_latency = sum(s["latency_ms"] for s in baseline_scores) / max(len(baseline_scores), 1)
        c_latency = sum(s["latency_ms"] for s in candidate_scores) / max(len(candidate_scores), 1)
        b_tokens = sum(s["tokens"] for s in baseline_scores) / max(len(baseline_scores), 1)
        c_tokens = sum(s["tokens"] for s in candidate_scores) / max(len(candidate_scores), 1)

        quality_delta = c_quality - b_quality
        # Avoid division by zero if baseline latency is 0 (e.g. all tasks failed).
        latency_ratio = c_latency / max(b_latency, 1.0)
        token_ratio = c_tokens / max(b_tokens, 1.0)

        decision, reasoning = self._make_decision(quality_delta, latency_ratio, token_ratio)

        gate_decision = TrainingGateDecision(
            decision=decision,
            baseline_quality=round(b_quality, 4),
            candidate_quality=round(c_quality, 4),
            quality_delta=round(quality_delta, 4),
            baseline_latency_ms=round(b_latency, 1),
            candidate_latency_ms=round(c_latency, 1),
            latency_ratio=round(latency_ratio, 3),
            token_efficiency=round(token_ratio, 3),
            reasoning=reasoning,
            eval_tasks_run=len(eval_tasks),
        )

        with self._lock:
            self._decisions.append(gate_decision)

        logger.info(
            "[TrainingQualityGate] %s | %s vs %s | quality_delta=%+.3f latency_ratio=%.2f token_ratio=%.2f | %s",
            decision.upper(),
            candidate_model_id,
            baseline_model_id,
            quality_delta,
            latency_ratio,
            token_ratio,
            reasoning,
        )
        return gate_decision

    def _make_decision(
        self,
        quality_delta: float,
        latency_ratio: float,
        token_ratio: float,
    ) -> tuple[str, str]:
        """Apply threshold logic to evaluation metrics and return a verdict.

        Hard-reject conditions are checked first. Then the deploy condition
        (meaningful quality gain with acceptable overhead). Anything else
        is marginal and sent for human review.

        Args:
            quality_delta: Candidate quality minus baseline quality.
            latency_ratio: Candidate latency divided by baseline latency.
            token_ratio: Candidate token usage divided by baseline token usage.

        Returns:
            Tuple of (decision, reasoning) where decision is one of
            "deploy", "reject", or "flag_for_review".
        """
        if quality_delta < _QUALITY_REJECT_THRESHOLD:
            return (
                "reject",
                f"Quality regression: {quality_delta:+.3f} is below reject threshold {_QUALITY_REJECT_THRESHOLD:+.3f}",
            )
        if latency_ratio > _LATENCY_REJECT_RATIO:
            return (
                "reject",
                f"Latency regression: candidate is {latency_ratio:.2f}x baseline "
                f"(max allowed {_LATENCY_REJECT_RATIO:.1f}x)",
            )
        if token_ratio > _TOKEN_REJECT_RATIO:
            return (
                "reject",
                f"Token efficiency regression: candidate uses {token_ratio:.2f}x baseline tokens "
                f"(max allowed {_TOKEN_REJECT_RATIO:.1f}x)",
            )

        # Deploy: meaningful quality gain AND latency/token overhead is small.
        if quality_delta >= _QUALITY_DEPLOY_THRESHOLD and latency_ratio <= 1.2 and token_ratio <= 1.2:
            return (
                "deploy",
                f"Quality improved by {quality_delta:+.3f} with acceptable overhead "
                f"(latency {latency_ratio:.2f}x, tokens {token_ratio:.2f}x)",
            )

        return (
            "flag_for_review",
            f"Marginal result: quality_delta={quality_delta:+.3f}, "
            f"latency_ratio={latency_ratio:.2f}, token_ratio={token_ratio:.2f}",
        )

    def _evaluate_model(
        self,
        model_id: str,
        eval_tasks: list[dict[str, str]],
    ) -> list[dict[str, float]]:
        """Run all eval tasks through a single model and return per-task scores.

        Each task is run independently. If a task fails (adapter error or
        scorer error), it is recorded with zero scores so it doesn't silently
        inflate the aggregate.

        Args:
            model_id: Model to evaluate.
            eval_tasks: List of eval task dicts with at minimum a "prompt" key.

        Returns:
            List of score dicts with "quality", "latency_ms", and "tokens" keys.
            One entry per task in eval_tasks.
        """
        scores: list[dict[str, float]] = []
        if get_quality_scorer is None:
            logger.warning(
                "[TrainingQualityGate] Quality scorer unavailable for model %s — all tasks will score 0.0",
                model_id,
            )
            return [{"quality": 0.0, "latency_ms": 0.0, "tokens": 0.0}] * len(eval_tasks)

        try:
            scorer = get_quality_scorer()
        except Exception as exc:
            logger.warning(
                "[TrainingQualityGate] Could not load quality scorer for model %s — all tasks will score 0.0: %s",
                model_id,
                exc,
            )
            return [{"quality": 0.0, "latency_ms": 0.0, "tokens": 0.0}] * len(eval_tasks)

        if LocalInferenceAdapter is None:
            logger.warning(
                "[TrainingQualityGate] LocalInferenceAdapter unavailable for model %s — all tasks will score 0.0",
                model_id,
            )
            return [{"quality": 0.0, "latency_ms": 0.0, "tokens": 0.0}] * len(eval_tasks)

        for i, task in enumerate(eval_tasks):
            prompt = task.get("prompt", "")
            try:
                adapter = LocalInferenceAdapter()
                start = time.monotonic()
                result = adapter.chat(model_id, "You are a helpful assistant.", prompt)
                elapsed_ms = (time.monotonic() - start) * 1000

                output = result.get("output", "")
                tokens = float(result.get("tokens_used", max(len(output) // 4, 1)))

                quality_score = scorer.score(
                    task_id=f"gate_eval_{i}",
                    model_id=model_id,
                    task_type=task.get("task_type", "general"),
                    task_description=prompt,
                    output=output,
                    use_llm=False,  # Heuristic scoring only — LLM judge is too slow for gate eval
                )
                scores.append({
                    "quality": quality_score.overall_score,
                    "latency_ms": elapsed_ms,
                    "tokens": tokens,
                })
            except Exception as exc:
                logger.warning(
                    "[TrainingQualityGate] Eval task %d failed for model %s — recording zero scores and continuing: %s",
                    i,
                    model_id,
                    exc,
                )
                scores.append({"quality": 0.0, "latency_ms": 0.0, "tokens": 0.0})

        return scores

    @staticmethod
    def _get_default_eval_set() -> list[dict[str, str]]:
        """Return a minimal default evaluation set covering common task types.

        Returns:
            List of five eval task dicts, one per major task category
            (coding, documentation, analysis, planning, review).
        """
        return [
            {
                "prompt": "Write a Python function to sort a list of integers.",
                "task_type": "coding",
            },
            {
                "prompt": "Explain the difference between TCP and UDP.",
                "task_type": "documentation",
            },
            {
                "prompt": "Analyze the time complexity of merge sort.",
                "task_type": "analysis",
            },
            {
                "prompt": "Create a plan for migrating a production database with zero downtime.",
                "task_type": "planning",
            },
            {
                "prompt": (
                    "Review this code for security issues: "
                    "def login(user, pw): return db.query(f'SELECT * FROM users WHERE name={user}')"
                ),
                "task_type": "review",
            },
        ]

    def get_history(self) -> list[dict[str, Any]]:
        """Return all gate decisions made this session, most recent first.

        Returns:
            List of GateDecision dicts ordered most-recent-first.
        """
        with self._lock:
            return [asdict(d) for d in reversed(self._decisions)]


def get_training_quality_gate() -> TrainingQualityGate:
    """Return the singleton TrainingQualityGate instance (thread-safe).

    Uses double-checked locking so the instance is created at most once
    even under concurrent first-call pressure.

    Returns:
        The shared TrainingQualityGate instance.
    """
    global _quality_gate
    if _quality_gate is None:
        with _quality_gate_lock:
            if _quality_gate is None:
                _quality_gate = TrainingQualityGate()
    return _quality_gate
