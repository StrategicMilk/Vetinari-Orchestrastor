"""Quality Scorer - Vetinari Self-Improvement Subsystem.

Evaluates the quality of task outputs using LLM-as-judge and heuristics.
Produces structured quality scores that feed the feedback loop.

Enhanced in Wave 4:
- SQLite persistence: scores survive restarts
- Improved LLM-as-judge: uses a DIFFERENT model from the one being evaluated
- Self-rationalization: judge generates reasoning before scoring
- Per-task-type rubrics with calibrated dimensions
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vetinari.agents.contracts import LLMJudgment, OutcomeSignal, Provenance
from vetinari.constants import _PROJECT_ROOT, TRUNCATE_OUTPUT_PREVIEW
from vetinari.database import get_connection
from vetinari.learning.quality_scorer_heuristics import _score_heuristic_output
from vetinari.types import EvidenceBasis
from vetinari.utils.serialization import dataclass_to_dict

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Structured quality assessment for a task output.

    A score of 0.0 with method="unmeasured" means "we have no data for this
    dimension" — distinct from a measured 0.0 which means "genuinely terrible".
    Check ``measured_dimensions`` to know which scores are backed by evidence.
    """

    task_id: str
    model_id: str
    task_type: str
    overall_score: float  # 0.0 - 1.0
    correctness: float = 0.0  # 0.0 = unmeasured by default, not "bad"
    completeness: float = 0.0
    efficiency: float = 0.0
    style: float = 0.0
    dimensions: dict[str, float] = field(default_factory=dict)
    measured_dimensions: list[str] = field(default_factory=list)  # Which dimensions have real data
    issues: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    method: str = "unmeasured"  # "llm" | "heuristic" | "hybrid" | "unmeasured" | "rejected"

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return (
            f"QualityScore(task_id={self.task_id!r}, model_id={self.model_id!r},"
            f" task_type={self.task_type!r}, overall_score={self.overall_score!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return dataclass_to_dict(self)


class QualityScorer:
    """Evaluates output quality using LLM-as-judge + heuristics.

    Provides structured quality signals that feed back into model
    selection, prompt evolution, and workflow strategy learning.
    """

    # Per-task-type quality dimensions
    DIMENSIONS = {
        "coding": ["correctness", "completeness", "efficiency", "style", "test_coverage"],
        "research": ["accuracy", "completeness", "source_quality", "actionability"],
        "analysis": ["depth", "accuracy", "actionability", "clarity"],
        "documentation": ["clarity", "completeness", "accuracy", "examples"],
        "testing": ["coverage", "correctness", "clarity", "edge_cases"],
        "default": ["correctness", "completeness", "quality"],
    }

    # Flat-score detection: if last N scores within this range, force calibration
    _FLAT_SCORE_WINDOW = 5  # Number of recent scores to check
    _FLAT_SCORE_THRESHOLD = 0.05  # Max range to consider "flat"

    # Variance monitoring: warn if variance below threshold over N+ scores
    _VARIANCE_WARN_MIN_SCORES = 20  # Minimum scores before checking variance
    _VARIANCE_WARN_THRESHOLD = 0.01  # Variance below this triggers warning

    def __init__(self, adapter_manager=None):
        self._adapter_manager = adapter_manager
        self._scores: deque[QualityScore] = deque(maxlen=1000)
        self._score_count: int = 0

        # Per model+task_type score history for flat-detection and variance monitoring
        self._score_history: dict[tuple[str, str], deque[float]] = {}

        self._calibration_interval: int = 10
        self._baselines: dict[str, dict[str, float]] = {}
        try:
            import yaml

            config_path = _PROJECT_ROOT / "config" / "ml_config.yaml"
            with Path(config_path).open(encoding="utf-8") as f:
                ml_config = yaml.safe_load(f)
            self._calibration_interval = ml_config.get("quality_scoring", {}).get("calibration_interval", 10)
        except Exception:
            logger.warning("Could not load ml_config.yaml — using default calibration interval of 10")

        try:
            import yaml

            baselines_path = _PROJECT_ROOT / "config" / "quality_baselines.yaml"
            with Path(baselines_path).open(encoding="utf-8") as f:
                self._baselines = yaml.safe_load(f) or {}
        except Exception:
            logger.warning("Could not load quality_baselines.yaml — using conservative 0.45 defaults")

    def score(
        self,
        task_id: str,
        model_id: str,
        task_type: str,
        task_description: str,
        output: str,
        use_llm: bool = True,
        inference_confidence: float | None = None,
        temperature_used: float | None = None,
    ) -> QualityScore:
        """Score a task output.

        Args:
            task_id: Unique task identifier.
            model_id: Model that produced the output.
            task_type: Type of task (coding, research, etc.).
            task_description: What the task asked for.
            output: The output to evaluate.
            use_llm: Whether to attempt LLM-as-judge evaluation.
            inference_confidence: Optional 0.0-1.0 confidence from logprob
                variance analysis. Low values penalize heuristic scores.
            temperature_used: The actual temperature used during inference.
                Passed to Thompson strategy feedback so the bandit learns
                which temperatures produce better outputs. None if unknown.

        Returns:
            QualityScore with all dimensions populated.
        """
        # Data quality gates — reject known fallback/mock outputs before scoring
        _FALLBACK_PATTERNS = frozenset({
            "",
            "{}",
            '{"content":"","sections":[]}',
            '{"content": "", "sections": []}',
        })
        if not output or output.strip() in _FALLBACK_PATTERNS:
            logger.warning(
                "[QualityScorer] Rejected fallback/empty output for task %s — not scoring",
                task_id,
            )
            return QualityScore(
                task_id=task_id,
                model_id=model_id,
                task_type=task_type,
                overall_score=0.0,
                correctness=0.0,
                completeness=0.0,
                efficiency=0.0,
                style=0.0,
                measured_dimensions=["correctness", "completeness", "efficiency", "style"],
                issues=["Rejected: fallback or empty output"],
                method="rejected",
            )

        dims = self.DIMENSIONS.get(task_type.lower(), self.DIMENSIONS["default"])

        self._score_count += 1

        # Determine if calibration is needed: periodic OR flat-score forced
        is_periodic_calibration = self._score_count % self._calibration_interval == 0
        is_flat_forced = self._is_score_distribution_flat(model_id, task_type)
        is_calibration = use_llm and self._adapter_manager and (is_periodic_calibration or is_flat_forced)

        if is_flat_forced and is_calibration:
            logger.info(
                "[QualityScorer] Forcing LLM calibration for %s/%s — last %d heuristic scores within %.2f range",
                model_id,
                task_type,
                self._FLAT_SCORE_WINDOW,
                self._FLAT_SCORE_THRESHOLD,
            )

        if is_calibration:
            llm_score = self._score_with_llm(task_id, model_id, task_type, task_description, output, dims)
            if llm_score:
                # Compare LLM score with heuristic for drift monitoring
                heuristic_score = self._score_heuristic(
                    task_id, model_id, task_type, output, dims, inference_confidence
                )
                delta = round(llm_score.overall_score - heuristic_score.overall_score, 3)
                logger.debug(
                    "[QualityScorer] Calibration run: LLM=%.2f, heuristic=%.2f, delta=%.2f",
                    llm_score.overall_score,
                    heuristic_score.overall_score,
                    delta,
                )
                self._scores.append(llm_score)
                self._persist(llm_score)
                return llm_score

        # Non-calibration run (or LLM unavailable): use heuristic scoring
        score = self._score_heuristic(task_id, model_id, task_type, output, dims, inference_confidence)
        self._scores.append(score)
        self._persist(score)

        # Track score in per-model+task history for flat-detection and variance monitoring
        self._record_score_history(model_id, task_type, score.overall_score)
        self._check_score_distribution(model_id, task_type)

        # Feed quality score back into Thompson strategy arms so the bandit
        # learns which temperatures work best for each agent+task combination.
        self._update_thompson_temperature(task_type, score.overall_score, temperature_used)

        return score

    def _score_with_llm(
        self,
        task_id: str,
        model_id: str,
        task_type: str,
        task_description: str,
        output: str,
        dims: list[str],
    ) -> QualityScore | None:
        """Use LLM-as-judge with self-rationalization to score the output.

        The judge model is deliberately chosen to be DIFFERENT from model_id
        to avoid self-evaluation bias. Uses LocalInferenceAdapter to call
        the local model rather than the adapter manager to ensure independence.

        Args:
            task_id: Unique task identifier.
            model_id: Model that produced the output (will be avoided for judging).
            task_type: Type of task being scored.
            task_description: What the task asked for.
            output: The output to evaluate.
            dims: List of dimension names to score.

        Returns:
            QualityScore with LLM-assigned scores, or None if scoring fails.
        """
        try:
            dims_list = ", ".join(dims)
            # Self-rationalization: judge explains reasoning BEFORE scoring
            dims_json_template = ", ".join(f'"{d}": 0.0' for d in dims)
            prompt = (
                f"You are an objective quality evaluator assessing a {task_type} output.\n\n"
                f"TASK: {task_description[:400]}\n\n"
                f"OUTPUT:\n{output[:TRUNCATE_OUTPUT_PREVIEW]}\n\n"
                f"Step 1 - REASONING: Briefly analyse the output strengths and weaknesses "
                f"for each dimension: {dims_list}\n\n"
                f"Step 2 - SCORES: Now score each dimension 0.0-1.0 based on your reasoning.\n\n"
                "Respond ONLY with valid JSON:\n"
                '{\n  "reasoning": "your analysis here",\n'
                '  "overall": 0.0,\n'
                f'  "dimensions": {{{dims_json_template}}},\n'
                '  "issues": ["..."],\n'
                '  "confidence": 0.0\n}'
            )

            # Prefer a different, fast local model for judging.
            # If only one model is loaded, self-evaluation is unreliable —
            # fall back to heuristic scoring to avoid bias.
            judge_model = self._pick_judge_model(model_id)
            if judge_model == model_id:
                logger.warning(
                    "Only one model loaded (%s) — using heuristic scoring instead of self-evaluation to avoid bias",
                    model_id,
                )
                return None  # Caller falls back to heuristic scoring

            from vetinari.adapters.llama_cpp_local_adapter import LocalInferenceAdapter

            adapter = LocalInferenceAdapter()
            result = adapter.chat(
                judge_model,
                "You are an objective quality evaluator. Score honestly and precisely.",
                prompt,
            )
            text = result.get("output", "").strip()
            if not text:
                return None
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group(0))

            dim_scores = data.get("dimensions", {})
            raw_overall = data.get("overall")
            if raw_overall is not None:
                overall = float(raw_overall)
            elif dim_scores:
                overall = sum(dim_scores.values()) / len(dim_scores)
            else:
                overall = 0.0  # No dimensions returned — unmeasured

            return QualityScore(
                task_id=task_id,
                model_id=model_id,
                task_type=task_type,
                overall_score=round(overall, 3),
                correctness=float(dim_scores.get("correctness", 0.0)),
                completeness=float(dim_scores.get("completeness", 0.0)),
                efficiency=float(dim_scores.get("efficiency", 0.0)),
                style=float(dim_scores.get("style", 0.0)),
                dimensions=dim_scores,
                measured_dimensions=list(dim_scores.keys()),
                issues=data.get("issues", []),
                method="llm",
            )
        except Exception as e:
            logger.warning("LLM quality scoring failed — quality tracking degraded: %s", e)
            return None

    def _pick_judge_model(self, evaluated_model_id: str) -> str:
        """Pick a judge model that is DIFFERENT from the model being evaluated."""
        try:
            from vetinari.models.model_registry import get_model_registry

            loaded = get_model_registry().get_loaded_local_models()
            for m in loaded:
                if m.model_id != evaluated_model_id:
                    return m.model_id
        except Exception:
            logger.warning("Failed to pick judge model different from %s", evaluated_model_id, exc_info=True)
        # Fallback: just use whatever is loaded (slight bias, but better than nothing)
        return evaluated_model_id

    def _persist(self, score: QualityScore) -> None:
        """Persist a quality score to the unified SQLite database."""
        try:
            conn = get_connection()
            conn.execute(
                """INSERT INTO quality_scores
                   (task_id, model_id, task_type, overall_score, completeness_score,
                    correctness_score, style_score, llm_calibrated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    score.task_id,
                    score.model_id,
                    score.task_type,
                    score.overall_score,
                    score.completeness,
                    score.correctness,
                    score.style,
                    1 if score.method == "llm" else 0,
                ),
            )
            conn.commit()
        except Exception as e:
            logger.warning("[QualityScorer] persist failed: %s", e)

    def _score_heuristic(
        self,
        task_id: str,
        model_id: str,
        task_type: str,
        output: str,
        dims: list[str],
        inference_confidence: float | None = None,
    ) -> QualityScore:
        """Heuristic quality scoring with structural checks per task type.

        Args:
            task_id: Unique task identifier.
            model_id: Model that produced the output.
            task_type: Type of task (coding, research, etc.).
            output: The output to evaluate.
            dims: List of dimension names to score.
            inference_confidence: Optional confidence from logprob variance (0.0-1.0).

        Returns:
            QualityScore with heuristic dimensions populated.
        """
        return _score_heuristic_output(
            task_id=task_id,
            model_id=model_id,
            task_type=task_type,
            output=output,
            dims=dims,
            inference_confidence=inference_confidence,
            baseline_config=self._baselines,
            score_factory=QualityScore,
        )

    def _update_thompson_temperature(
        self, task_type: str, quality_score: float, temperature_used: float | None = None
    ) -> None:
        """Update Thompson strategy arms with quality feedback for temperature learning.

        Called after quality scoring to teach the bandit which temperature
        values produce better outputs per task type. Skipped when the actual
        temperature used during inference is not known — recording the wrong
        arm would corrupt the bandit's temperature-quality mapping.

        Args:
            task_type: The task type that was scored.
            quality_score: The overall quality score (0.0-1.0).
            temperature_used: The actual temperature used during inference.
                If None, the update is skipped to avoid recording wrong data.
        """
        if temperature_used is None:
            return  # Cannot record without knowing which temperature was used
        try:
            from vetinari.learning.model_selector import get_thompson_selector

            ts = get_thompson_selector()
            ts.update_strategy(
                "WORKER",
                task_type.lower() if task_type else "default",
                "temperature",
                temperature_used,
                quality_score,
            )
        except Exception:
            logger.warning(
                "Thompson temperature feedback skipped — selector unavailable",
                exc_info=True,
            )

    def _record_score_history(self, model_id: str, task_type: str, score: float) -> None:
        """Record a score in per-model+task history for trend analysis."""
        key = (model_id, task_type.lower())
        if key not in self._score_history:
            self._score_history[key] = deque(maxlen=50)
        self._score_history[key].append(score)

    def _is_score_distribution_flat(self, model_id: str, task_type: str) -> bool:
        """Check if last N scores are suspiciously flat (within threshold range).

        Returns True if the last _FLAT_SCORE_WINDOW scores for this model+task
        are all within _FLAT_SCORE_THRESHOLD of each other, indicating the
        heuristic scorer is not producing meaningful variance.
        """
        key = (model_id, task_type.lower())
        history = self._score_history.get(key)
        if not history or len(history) < self._FLAT_SCORE_WINDOW:
            return False
        recent = list(history)[-self._FLAT_SCORE_WINDOW :]
        score_range = max(recent) - min(recent)
        return score_range < self._FLAT_SCORE_THRESHOLD

    def _check_score_distribution(self, model_id: str, task_type: str) -> None:
        """Log WARNING if score variance is suspiciously low over many scores.

        Monitors per-model+task score distributions to catch broken scorers
        that produce identical scores regardless of output quality.
        """
        key = (model_id, task_type.lower())
        history = self._score_history.get(key)
        if not history or len(history) < self._VARIANCE_WARN_MIN_SCORES:
            return
        scores_list = list(history)
        mean = sum(scores_list) / len(scores_list)
        variance = sum((s - mean) ** 2 for s in scores_list) / len(scores_list)
        if variance < self._VARIANCE_WARN_THRESHOLD:
            logger.warning(
                "[QualityScorer] Score variance too low for %s/%s: variance=%.4f over %d scores "
                "(mean=%.3f) — scores may not reflect actual quality differences",
                model_id,
                task_type,
                variance,
                len(scores_list),
                mean,
            )

    def score_with_signal(
        self,
        task_id: str,
        model_id: str,
        task_type: str,
        task_description: str,
        output: str,
        use_llm: bool = True,
        inference_confidence: float | None = None,
        temperature_used: float | None = None,
    ) -> OutcomeSignal:
        """Score a task output and return an evidence-backed OutcomeSignal.

        Wraps ``score()`` with ``LLMJudgment`` and ``Provenance`` metadata so
        Inspector pipeline callers receive a fail-closed signal with basis
        ``LLM_JUDGMENT`` instead of a raw ``QualityScore``.

        Rejected/fallback outputs yield ``passed=False, score=0.0,
        basis=UNSUPPORTED`` (Rule 2 — no default-pass).

        Args:
            task_id: Unique task identifier.
            model_id: Model that produced the output.
            task_type: Type of task (coding, research, etc.).
            task_description: What the task asked for.
            output: The output to evaluate.
            use_llm: Whether to attempt LLM-as-judge evaluation.
            inference_confidence: Optional 0.0-1.0 confidence from logprob
                variance analysis.
            temperature_used: The actual temperature used during inference.

        Returns:
            OutcomeSignal with basis=LLM_JUDGMENT and a populated LLMJudgment,
            or basis=UNSUPPORTED when the output was rejected as fallback/empty.
        """
        qs = self.score(
            task_id=task_id,
            model_id=model_id,
            task_type=task_type,
            task_description=task_description,
            output=output,
            use_llm=use_llm,
            inference_confidence=inference_confidence,
            temperature_used=temperature_used,
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        if qs.method == "rejected":
            return OutcomeSignal(
                passed=False,
                score=0.0,
                basis=EvidenceBasis.UNSUPPORTED,
                issues=tuple(qs.issues),
                provenance=Provenance(
                    source="vetinari.learning.quality_scorer",
                    timestamp_utc=timestamp,
                    model_id=model_id,
                ),
            )

        judgment = LLMJudgment(
            model_id=qs.model_id,
            summary=f"Quality score {qs.overall_score:.3f} via {qs.method} for task_type={qs.task_type}",
            score=qs.overall_score,
            reasoning="; ".join(qs.issues) if qs.issues else "",
        )

        passed = qs.overall_score >= 0.5 and not qs.issues
        return OutcomeSignal(
            passed=passed,
            score=qs.overall_score,
            basis=EvidenceBasis.LLM_JUDGMENT,
            llm_judgment=judgment,
            issues=tuple(qs.issues),
            provenance=Provenance(
                source="vetinari.learning.quality_scorer",
                timestamp_utc=timestamp,
                model_id=qs.model_id,
            ),
        )

    def get_history(self, model_id: str | None = None, task_type: str | None = None) -> list[QualityScore]:
        """Get scoring history from SQLite + in-memory cache, optionally filtered.

        Args:
            model_id: The model id.
            task_type: The task type.

        Returns:
            List of results.
        """
        try:
            query = (
                "SELECT task_id, model_id, task_type, overall_score, completeness_score,"
                " correctness_score, style_score, llm_calibrated FROM quality_scores WHERE 1=1"
            )
            params: list = []
            if model_id:
                query += " AND model_id = ?"
                params.append(model_id)
            if task_type:
                query += " AND task_type = ?"
                params.append(task_type)
            query += " ORDER BY created_at DESC LIMIT 1000"

            conn = get_connection()
            rows = conn.execute(query, params).fetchall()

            scores = [
                QualityScore(
                    task_id=row[0],
                    model_id=row[1],
                    task_type=row[2],
                    overall_score=row[3],
                    completeness=row[4] or row[3],
                    correctness=row[5] or row[3],
                    style=row[6] or row[3],
                    method="llm" if row[7] else "heuristic",
                )
                for row in rows
            ]
            return scores
        except Exception:
            # Fall back to in-memory
            result = self._scores
            if model_id:
                result = [s for s in result if s.model_id == model_id]
            if task_type:
                result = [s for s in result if s.task_type == task_type]
            logger.warning(
                "Quality score DB query failed for model_id=%r task_type=%r — falling back to in-memory scores (%d records)",
                model_id,
                task_type,
                len(result),
            )
            return result

    def get_model_average(self, model_id: str, task_type: str | None = None) -> float:
        """Get average quality score for a model (optionally filtered by task type).

        Args:
            model_id: The model id.
            task_type: The task type.

        Returns:
            The computed value.
        """
        scores = self.get_history(model_id=model_id, task_type=task_type)
        if not scores:
            return 0.0  # No data — unmeasured, not "good"
        return sum(s.overall_score for s in scores) / len(scores)


# Singleton
_quality_scorer: QualityScorer | None = None
_quality_scorer_lock = threading.Lock()


def get_quality_scorer() -> QualityScorer:
    """Return the singleton QualityScorer instance (thread-safe).

    Returns:
        The shared QualityScorer instance.
    """
    global _quality_scorer
    if _quality_scorer is None:
        with _quality_scorer_lock:
            if _quality_scorer is None:
                _quality_scorer = QualityScorer()
    return _quality_scorer
