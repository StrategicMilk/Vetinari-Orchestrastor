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
import os
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_DB_PATH = os.environ.get("VETINARI_QUALITY_DB", "./vetinari_quality_scores.db")


@dataclass
class QualityScore:
    """Structured quality assessment for a task output."""

    task_id: str
    model_id: str
    task_type: str
    overall_score: float  # 0.0 - 1.0
    correctness: float = 0.7  # Is the output correct?
    completeness: float = 0.7  # Does it address the full task?
    efficiency: float = 0.7  # Is it efficient/concise?
    style: float = 0.7  # Follows conventions?
    dimensions: dict[str, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    method: str = "heuristic"  # "llm" | "heuristic" | "hybrid"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

    def __init__(self, adapter_manager=None, db_path: str = _DB_PATH):
        self._adapter_manager = adapter_manager
        self._scores: list[QualityScore] = []
        self._db_path = db_path
        self._score_count: int = 0
        try:
            import yaml

            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "config", "ml_config.yaml"
            )
            with open(config_path) as f:
                ml_config = yaml.safe_load(f)
            self._calibration_interval: int = ml_config.get("quality_scoring", {}).get("calibration_interval", 10)
        except Exception:
            self._calibration_interval = 10
        self._init_db()

    def _init_db(self) -> None:
        """Create quality_scores table if it doesn't exist."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS quality_scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL,
                        model_id TEXT NOT NULL,
                        task_type TEXT NOT NULL,
                        overall_score REAL NOT NULL,
                        correctness REAL,
                        completeness REAL,
                        efficiency REAL,
                        style REAL,
                        dimensions TEXT,
                        issues TEXT,
                        method TEXT,
                        timestamp TEXT,
                        created_at REAL DEFAULT (unixepoch())
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_qs_model ON quality_scores(model_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_qs_task_type ON quality_scores(task_type)")
        except Exception as e:
            logger.debug("[QualityScorer] DB init failed (scores will be in-memory only): %s", e)

    def score(
        self,
        task_id: str,
        model_id: str,
        task_type: str,
        task_description: str,
        output: str,
        use_llm: bool = True,
    ) -> QualityScore:
        """Score a task output.

        Args:
            task_id: Unique task identifier.
            model_id: Model that produced the output.
            task_type: Type of task (coding, research, etc.).
            task_description: What the task asked for.
            output: The output to evaluate.
            use_llm: Whether to attempt LLM-as-judge evaluation.

        Returns:
            QualityScore with all dimensions populated.
        """
        dims = self.DIMENSIONS.get(task_type.lower(), self.DIMENSIONS["default"])

        self._score_count += 1
        is_calibration = use_llm and self._adapter_manager and (self._score_count % self._calibration_interval == 0)

        if is_calibration:
            llm_score = self._score_with_llm(task_id, model_id, task_type, task_description, output, dims)
            if llm_score:
                # Compare LLM score with heuristic for drift monitoring
                heuristic_score = self._score_heuristic(task_id, model_id, task_type, output, dims)
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
        score = self._score_heuristic(task_id, model_id, task_type, output, dims)
        self._scores.append(score)
        self._persist(score)
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
        to avoid self-evaluation bias.  We use a direct LM Studio call here
        rather than the adapter manager to ensure independence.
        """
        try:
            dims_list = ", ".join(dims)
            # Self-rationalization: judge explains reasoning BEFORE scoring
            dims_json_template = ", ".join(f'"{d}": 0.0' for d in dims)
            prompt = (
                f"You are an objective quality evaluator assessing a {task_type} output.\n\n"
                f"TASK: {task_description[:400]}\n\n"
                f"OUTPUT:\n{output[:2000]}\n\n"
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

            # Prefer a different, fast local model for judging
            judge_model = self._pick_judge_model(model_id)

            host = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")
            import requests as _req

            resp = _req.post(
                f"{host}/v1/chat/completions",
                json={
                    "model": judge_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an objective quality evaluator. Score honestly and precisely.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 600,
                    "temperature": 0.1,
                },
                timeout=60,
            )
            if resp.status_code != 200:
                return None

            text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
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
                overall = 0.65  # conservative default when no dimensions returned

            return QualityScore(
                task_id=task_id,
                model_id=model_id,
                task_type=task_type,
                overall_score=round(overall, 3),
                correctness=float(dim_scores.get("correctness", overall)),
                completeness=float(dim_scores.get("completeness", overall)),
                efficiency=float(dim_scores.get("efficiency", overall)),
                style=float(dim_scores.get("style", overall)),
                dimensions=dim_scores,
                issues=data.get("issues", []),
                method="llm",
            )
        except Exception as e:
            logger.debug("LLM scoring failed: %s", e)
            return None

    def _pick_judge_model(self, evaluated_model_id: str) -> str:
        """Pick a judge model that is DIFFERENT from the model being evaluated."""
        try:
            from vetinari.model_registry import get_model_registry

            loaded = get_model_registry().get_loaded_local_models()
            for m in loaded:
                if m.model_id != evaluated_model_id:
                    return m.model_id
        except Exception:
            logger.debug("Failed to pick judge model different from %s", evaluated_model_id, exc_info=True)
        # Fallback: just use whatever is loaded (slight bias, but better than nothing)
        return evaluated_model_id

    def _persist(self, score: QualityScore) -> None:
        """Persist a quality score to SQLite."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT INTO quality_scores
                       (task_id, model_id, task_type, overall_score, correctness,
                        completeness, efficiency, style, dimensions, issues, method, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        score.task_id,
                        score.model_id,
                        score.task_type,
                        score.overall_score,
                        score.correctness,
                        score.completeness,
                        score.efficiency,
                        score.style,
                        json.dumps(score.dimensions),
                        json.dumps(score.issues),
                        score.method,
                        score.timestamp,
                    ),
                )
        except Exception as e:
            logger.debug("[QualityScorer] persist failed: %s", e)

    def _score_heuristic(
        self,
        task_id: str,
        model_id: str,
        task_type: str,
        output: str,
        dims: list[str],
    ) -> QualityScore:
        """Heuristic quality scoring based on output characteristics."""
        issues: list[str] = []
        scores: dict[str, float] = {}

        if not output or not output.strip():
            return QualityScore(
                task_id=task_id,
                model_id=model_id,
                task_type=task_type,
                overall_score=0.0,
                issues=["Empty output"],
                dimensions=dict.fromkeys(dims, 0.0),
                method="heuristic",
            )

        words = len(output.split())

        # Length heuristic
        if words < 10:
            scores["completeness"] = 0.3
            issues.append("Very short output")
        elif words > 2000:
            scores["efficiency"] = 0.5
        else:
            scores["completeness"] = min(1.0, words / 200)

        # Code-specific heuristics
        if task_type == "coding":
            has_def = "def " in output or "class " in output
            has_docstring = '"""' in output or "'''" in output
            has_test = "assert" in output or "test" in output.lower()
            scores["correctness"] = 0.7 if has_def else 0.4
            scores["style"] = 0.8 if has_docstring else 0.5
            scores["test_coverage"] = 0.8 if has_test else 0.3
            if not has_def:
                issues.append("No function/class definitions found")

        # Research-specific heuristics
        elif task_type == "research":
            has_sources = "http" in output or "source" in output.lower()
            has_sections = output.count("\n#") >= 2 or output.count("\n\n") >= 3
            scores["source_quality"] = 0.8 if has_sources else 0.4
            scores["actionability"] = 0.7 if has_sections else 0.5
            if not has_sources:
                issues.append("No source citations found")

        # Fill missing dimensions with default
        for d in dims:
            if d not in scores:
                scores[d] = 0.65

        overall = sum(scores.values()) / max(len(scores), 1) if scores else 0.65

        return QualityScore(
            task_id=task_id,
            model_id=model_id,
            task_type=task_type,
            overall_score=round(overall, 3),
            correctness=scores.get("correctness", overall),
            completeness=scores.get("completeness", overall),
            efficiency=scores.get("efficiency", overall),
            style=scores.get("style", overall),
            dimensions=scores,
            issues=issues,
            method="heuristic",
        )

    def get_history(self, model_id: str | None = None, task_type: str | None = None) -> list[QualityScore]:
        """Get scoring history from SQLite + in-memory cache, optionally filtered."""
        try:
            query = "SELECT task_id, model_id, task_type, overall_score, correctness, completeness, efficiency, style, dimensions, issues, method, timestamp FROM quality_scores WHERE 1=1"
            params: list = []
            if model_id:
                query += " AND model_id = ?"
                params.append(model_id)
            if task_type:
                query += " AND task_type = ?"
                params.append(task_type)
            query += " ORDER BY created_at DESC LIMIT 1000"

            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(query, params).fetchall()

            scores = []
            for row in rows:
                scores.append(
                    QualityScore(
                        task_id=row[0],
                        model_id=row[1],
                        task_type=row[2],
                        overall_score=row[3],
                        correctness=row[4] or row[3],
                        completeness=row[5] or row[3],
                        efficiency=row[6] or row[3],
                        style=row[7] or row[3],
                        dimensions=json.loads(row[8] or "{}"),
                        issues=json.loads(row[9] or "[]"),
                        method=row[10] or "heuristic",
                        timestamp=row[11] or "",
                    )
                )
            return scores
        except Exception:
            # Fall back to in-memory
            result = self._scores
            if model_id:
                result = [s for s in result if s.model_id == model_id]
            if task_type:
                result = [s for s in result if s.task_type == task_type]
            return result

    def get_model_average(self, model_id: str, task_type: str | None = None) -> float:
        """Get average quality score for a model (optionally filtered by task type)."""
        scores = self.get_history(model_id=model_id, task_type=task_type)
        if not scores:
            return 0.7  # Default prior
        return sum(s.overall_score for s in scores) / len(scores)


# Singleton
_quality_scorer: QualityScorer | None = None


def get_quality_scorer() -> QualityScorer:
    global _quality_scorer
    if _quality_scorer is None:
        _quality_scorer = QualityScorer()
    return _quality_scorer
