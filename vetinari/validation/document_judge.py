"""LLM-as-judge evaluator for document quality.

Provides a higher-fidelity evaluation path that delegates to an LLM for
dimensions that heuristics struggle with (accuracy, relevance, technical
depth).  Falls back gracefully to the heuristic rubric when no LLM is
available.

Usage::

    from vetinari.validation.document_judge import DocumentJudge

    judge = DocumentJudge()
    report = judge.evaluate("# My README\\n...", doc_type="readme")
    print(report.overall_score, report.passed)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from vetinari.validation.document_quality import (
    QualityReport,
    evaluate_document,
)
from vetinari.validation.document_types import (
    DocumentProfile,
    DocumentType,
    get_profile_for_type,
)

logger = logging.getLogger(__name__)


@dataclass
class JudgeConfig:
    """Configuration for the LLM-as-judge evaluator.

    Args:
        model_id: LLM model to use for evaluation. Empty string means
            use heuristics only (no LLM call).
        temperature: Sampling temperature for the judge LLM.
        max_tokens: Maximum response tokens for the judge.
        dimensions_for_llm: Which dimensions to delegate to the LLM
            instead of using heuristics.  Dimensions not listed here
            always use heuristic scoring.
        fallback_to_heuristic: If True, fall back to heuristic scoring
            when the LLM is unavailable or returns an error.
    """

    model_id: str = ""
    temperature: float = 0.1
    max_tokens: int = 1024
    dimensions_for_llm: list[str] = field(
        default_factory=lambda: [
            "accuracy",
            "relevance",
            "technical_depth",
            "completeness",
        ]
    )
    fallback_to_heuristic: bool = True


class DocumentJudge:
    """LLM-as-judge evaluator with heuristic fallback.

    When ``config.model_id`` is empty or the LLM is unreachable, all
    dimensions are scored by the heuristic rubric in
    ``document_quality.evaluate_document``.

    When an LLM is available, the dimensions listed in
    ``config.dimensions_for_llm`` are evaluated by the LLM, and the
    remaining dimensions are evaluated by heuristics.  The two sets of
    scores are merged into a single QualityReport.
    """

    def __init__(self, config: JudgeConfig | None = None):
        self._config = config or JudgeConfig()

    @property
    def uses_llm(self) -> bool:
        """Whether this judge is configured to use an LLM."""
        return bool(self._config.model_id)

    def evaluate(
        self,
        text: str,
        doc_type: str | DocumentType = "default",
        profile: DocumentProfile | None = None,
        context: str = "",
    ) -> QualityReport:
        """Evaluate a document, optionally using LLM-as-judge.

        Args:
            text: Document text to evaluate.
            doc_type: Document type for profile lookup.
            profile: Explicit profile override.
            context: Additional context for the LLM judge (e.g. the task
                that produced this document).

        Returns:
            QualityReport with per-dimension scores.
        """
        type_key = doc_type.value if isinstance(doc_type, DocumentType) else doc_type
        if profile is None:
            profile = get_profile_for_type(type_key)

        # Always start with heuristic evaluation
        report = evaluate_document(text, doc_type=type_key, profile=profile)

        # If LLM is configured, try to enhance specific dimensions
        if self.uses_llm:
            try:
                llm_scores = self._llm_evaluate(text, type_key, context)
                report = self._merge_scores(report, llm_scores, profile)
            except Exception as exc:
                if self._config.fallback_to_heuristic:
                    logger.warning("LLM judge failed, using heuristic fallback: %s", exc)
                else:
                    raise

        return report

    def _llm_evaluate(self, text: str, doc_type: str, context: str) -> dict[str, float]:
        """Call the LLM to score selected dimensions.

        Args:
            text: The document text.
            doc_type: Document type name.
            context: Additional context.

        Returns:
            Mapping of dimension name to LLM-assigned score (0.0-1.0).

        Raises:
            RuntimeError: If the LLM call fails and fallback is disabled.
        """
        prompt = self._build_judge_prompt(text, doc_type, context)

        try:
            from vetinari.adapters import get_adapter_manager

            manager = get_adapter_manager()
            response = manager.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self._config.model_id,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
            return self._parse_llm_response(response)
        except ImportError:
            raise RuntimeError("AdapterManager not available for LLM judge")

    def _build_judge_prompt(self, text: str, doc_type: str, context: str) -> str:
        """Build the evaluation prompt for the LLM judge.

        Args:
            text: Document text to evaluate.
            doc_type: Document type name.
            context: Additional evaluation context.

        Returns:
            Formatted prompt string.
        """
        dims = ", ".join(self._config.dimensions_for_llm)
        ctx_section = f"\nAdditional context: {context}" if context else ""

        return (
            f"You are a document quality evaluator. Score the following "
            f"{doc_type} document on these dimensions: {dims}.\n\n"
            f"For each dimension, provide a score from 0.0 to 1.0 and a "
            f"brief explanation.\n\n"
            f"Respond in this exact format for each dimension:\n"
            f"DIMENSION: score | explanation\n\n"
            f"Document to evaluate:\n---\n{text[:4000]}\n---"
            f"{ctx_section}"
        )

    def _parse_llm_response(self, response: Any) -> dict[str, float]:
        """Parse dimension scores from the LLM response text.

        Args:
            response: Raw LLM response (string or object with .content).

        Returns:
            Mapping of dimension name to score.
        """
        text = str(getattr(response, "content", response))
        scores: dict[str, float] = {}

        for dim in self._config.dimensions_for_llm:
            import re

            pattern = rf"{dim}:\s*([\d.]+)"
            match = re.search(pattern, text, re.I)
            if match:
                try:
                    score = float(match.group(1))
                    scores[dim] = max(0.0, min(1.0, score))
                except ValueError:
                    pass

        return scores

    def _merge_scores(
        self,
        heuristic_report: QualityReport,
        llm_scores: dict[str, float],
        profile: DocumentProfile,
    ) -> QualityReport:
        """Merge LLM scores into the heuristic report.

        LLM scores replace heuristic scores for the dimensions they cover.
        The overall score is recalculated with the merged dimension scores.

        Args:
            heuristic_report: Base report from heuristic evaluation.
            llm_scores: LLM-provided scores for selected dimensions.
            profile: Document profile for weight lookup.

        Returns:
            Updated QualityReport with merged scores.
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for ds in heuristic_report.dimension_scores:
            if ds.dimension in llm_scores:
                ds.score = llm_scores[ds.dimension]
                ds.findings.append("(scored by LLM judge)")
            weighted_sum += ds.score * ds.weight
            total_weight += ds.weight

        overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        heuristic_report.overall_score = overall
        heuristic_report.passed = overall >= profile.min_score

        return heuristic_report

    def get_config(self) -> dict[str, Any]:
        """Return current judge configuration as a dictionary."""
        return {
            "model_id": self._config.model_id,
            "uses_llm": self.uses_llm,
            "temperature": self._config.temperature,
            "dimensions_for_llm": self._config.dimensions_for_llm,
            "fallback_to_heuristic": self._config.fallback_to_heuristic,
        }
