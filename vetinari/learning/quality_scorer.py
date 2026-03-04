"""
Quality Scorer - Vetinari Self-Improvement Subsystem

Evaluates the quality of task outputs using LLM-as-judge and heuristics.
Produces structured quality scores that feed the feedback loop.
"""

import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Structured quality assessment for a task output."""
    task_id: str
    model_id: str
    task_type: str
    overall_score: float          # 0.0 - 1.0
    correctness: float = 0.7      # Is the output correct?
    completeness: float = 0.7     # Does it address the full task?
    efficiency: float = 0.7       # Is it efficient/concise?
    style: float = 0.7            # Follows conventions?
    dimensions: Dict[str, float] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    method: str = "heuristic"     # "llm" | "heuristic" | "hybrid"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QualityScorer:
    """
    Evaluates output quality using LLM-as-judge + heuristics.

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

    def __init__(self, adapter_manager=None):
        self._adapter_manager = adapter_manager
        self._scores: List[QualityScore] = []

    def score(
        self,
        task_id: str,
        model_id: str,
        task_type: str,
        task_description: str,
        output: str,
        use_llm: bool = True,
    ) -> QualityScore:
        """
        Score a task output.

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

        if use_llm and self._adapter_manager:
            score = self._score_with_llm(task_id, model_id, task_type, task_description, output, dims)
            if score:
                self._scores.append(score)
                return score

        # Fallback: heuristic scoring
        score = self._score_heuristic(task_id, model_id, task_type, output, dims)
        self._scores.append(score)
        return score

    def _score_with_llm(
        self,
        task_id: str,
        model_id: str,
        task_type: str,
        task_description: str,
        output: str,
        dims: List[str],
    ) -> Optional[QualityScore]:
        """Use LLM-as-judge to score the output."""
        try:
            from vetinari.adapters.base import InferenceRequest
            dims_list = ", ".join(dims)
            prompt = f"""You are a quality evaluator. Score this {task_type} output.

TASK: {task_description[:300]}

OUTPUT:
{output[:1500]}

Score each dimension from 0.0 to 1.0:
Dimensions: {dims_list}

Respond as JSON:
{{
  "overall": 0.0-1.0,
  "dimensions": {{{", ".join(f'"{d}": 0.0-1.0' for d in dims)}}},
  "issues": ["issue1", ...],
  "rationale": "brief explanation"
}}"""

            req = InferenceRequest(
                model_id="default",
                prompt=prompt,
                system_prompt="You are an objective quality evaluator. Score honestly.",
                max_tokens=512,
                temperature=0.1,
            )
            resp = self._adapter_manager.infer(req)
            if resp.status != "ok":
                return None

            import json
            import re
            text = resp.output.strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group(0))

            dim_scores = data.get("dimensions", {})
            overall = float(data.get("overall", sum(dim_scores.values()) / max(len(dim_scores), 1)))

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
            logger.debug(f"LLM scoring failed: {e}")
            return None

    def _score_heuristic(
        self,
        task_id: str,
        model_id: str,
        task_type: str,
        output: str,
        dims: List[str],
    ) -> QualityScore:
        """Heuristic quality scoring based on output characteristics."""
        issues: List[str] = []
        scores: Dict[str, float] = {}

        if not output or not output.strip():
            return QualityScore(
                task_id=task_id, model_id=model_id, task_type=task_type,
                overall_score=0.0, issues=["Empty output"],
                dimensions={d: 0.0 for d in dims}, method="heuristic"
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

        overall = sum(scores.values()) / len(scores) if scores else 0.65

        return QualityScore(
            task_id=task_id, model_id=model_id, task_type=task_type,
            overall_score=round(overall, 3),
            correctness=scores.get("correctness", overall),
            completeness=scores.get("completeness", overall),
            efficiency=scores.get("efficiency", overall),
            style=scores.get("style", overall),
            dimensions=scores, issues=issues, method="heuristic"
        )

    def get_history(self, model_id: Optional[str] = None, task_type: Optional[str] = None) -> List[QualityScore]:
        """Get scoring history, optionally filtered."""
        result = self._scores
        if model_id:
            result = [s for s in result if s.model_id == model_id]
        if task_type:
            result = [s for s in result if s.task_type == task_type]
        return result

    def get_model_average(self, model_id: str, task_type: str = None) -> float:
        """Get average quality score for a model (optionally filtered by task type)."""
        scores = self.get_history(model_id=model_id, task_type=task_type)
        if not scores:
            return 0.7  # Default prior
        return sum(s.overall_score for s in scores) / len(scores)


# Singleton
_quality_scorer: Optional[QualityScorer] = None


def get_quality_scorer() -> QualityScorer:
    global _quality_scorer
    if _quality_scorer is None:
        _quality_scorer = QualityScorer()
    return _quality_scorer
