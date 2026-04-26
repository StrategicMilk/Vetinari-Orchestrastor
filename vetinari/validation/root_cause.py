"""Root Cause Analysis module for classifying quality rejections.

When the Quality agent rejects output, this module classifies WHY the
defect occurred — enabling intelligent corrective routing rather than
a blind redo. Classification is rule-based using heuristics derived
from the rejection reasons and quality score.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DefectCategory(Enum):
    """Classification of why a quality rejection occurred."""

    BAD_SPEC = "bad_spec"
    WRONG_MODEL = "wrong_model"
    INSUFFICIENT_CONTEXT = "context"
    PROMPT_WEAKNESS = "prompt"
    COMPLEXITY_UNDERESTIMATE = "complexity"
    INTEGRATION_ERROR = "integration"
    HALLUCINATION = "hallucination"


@dataclass
class RootCauseAnalysis:
    """Result of a root cause analysis on a rejected task output.

    Attributes:
        category: The classified defect category explaining why rejection occurred.
        confidence: Classifier confidence in the range [0.0, 1.0].
        evidence: List of excerpts or signals that support the classification.
        corrective_action: Immediate action to fix the current defect.
        preventive_action: Systemic action to prevent the defect recurring.
    """

    category: DefectCategory
    confidence: float
    evidence: list[str]
    corrective_action: str
    preventive_action: str

    def __repr__(self) -> str:
        """Show key identifying fields for debugging."""
        return f"RootCauseAnalysis(category={self.category!r}, confidence={self.confidence!r})"


class RootCauseAnalyzer:
    """Classifies the root cause of a quality rejection.

    Uses rule-based heuristics applied to the rejection reasons and
    quality score to determine which defect category best explains the
    failure, then prescribes corrective and preventive actions.
    """

    def analyze(
        self,
        task_description: str,
        rejection_reasons: list[str],
        quality_score: float,
        task_mode: str = "",
    ) -> RootCauseAnalysis:
        """Classify the root cause of a quality rejection.

        Args:
            task_description: The original task description that was rejected.
            rejection_reasons: List of reason strings provided by the Quality agent.
            quality_score: Numeric quality score between 0.0 and 1.0.
            task_mode: Optional task mode string for additional context.

        Returns:
            A RootCauseAnalysis with category, confidence, evidence, and actions.
        """
        # ── Try LLM diagnosis for context-specific actions ────────────────
        try:
            from vetinari.llm_helpers import diagnose_defect_via_llm

            combined_reason = "; ".join(rejection_reasons[:3])
            llm_result = diagnose_defect_via_llm(
                task_description=task_description,
                rejection_reason=combined_reason,
                agent_type=task_mode,
            )
            if llm_result:
                llm_category, llm_explanation = llm_result
                category_map = {
                    "hallucinated_import": DefectCategory.HALLUCINATION,
                    "ambiguous_spec": DefectCategory.BAD_SPEC,
                    "model_limitation": DefectCategory.WRONG_MODEL,
                    "insufficient_context": DefectCategory.INSUFFICIENT_CONTEXT,
                    "output_format": DefectCategory.PROMPT_WEAKNESS,
                    "runtime_error": DefectCategory.INTEGRATION_ERROR,
                    "quality_below_threshold": DefectCategory.PROMPT_WEAKNESS,
                }
                mapped = category_map.get(llm_category)
                if mapped:
                    logger.info("LLM diagnosed root cause as %s: %s", mapped.value, llm_explanation)
                    return RootCauseAnalysis(
                        category=mapped,
                        confidence=0.85,
                        evidence=[llm_explanation, *rejection_reasons[:2]],
                        corrective_action=llm_explanation,
                        preventive_action=f"Address underlying {mapped.value} pattern: {llm_explanation}",
                    )
        except Exception:
            logger.warning("LLM root cause analysis unavailable — falling back to heuristic root cause detection")

        # ── Heuristic fallback ────────────────────────────────────────────
        if self._is_hallucination(rejection_reasons):
            evidence = [
                r
                for r in rejection_reasons
                if any(
                    kw in r.lower() for kw in ("import", "not found", "does not exist", "hallucinated", "fabricated")
                )
            ]
            logger.info("Root cause classified as HALLUCINATION (evidence count=%d)", len(evidence))
            return RootCauseAnalysis(
                category=DefectCategory.HALLUCINATION,
                confidence=0.90,
                evidence=evidence or rejection_reasons[:1],
                corrective_action=(
                    "Regenerate the output grounding all references in verified sources. "
                    "Remove any imports, functions, or facts that cannot be confirmed."
                ),
                preventive_action=(
                    "Add a verification pass that checks all referenced symbols and imports exist "
                    "before accepting output from the agent."
                ),
            )

        if self._is_bad_spec(rejection_reasons):
            evidence = [
                r
                for r in rejection_reasons
                if any(kw in r.lower() for kw in ("ambiguous", "unclear", "acceptance criteria", "spec", "incomplete"))
            ]
            logger.info("Root cause classified as BAD_SPEC (evidence count=%d)", len(evidence))
            return RootCauseAnalysis(
                category=DefectCategory.BAD_SPEC,
                confidence=0.85,
                evidence=evidence or rejection_reasons[:1],
                corrective_action=(
                    "Clarify the task specification with the requester before retrying. "
                    "Resolve all ambiguous requirements and add explicit acceptance criteria."
                ),
                preventive_action=(
                    "Introduce a spec review step in the planning phase that enforces "
                    "acceptance criteria and unambiguous requirements before agent dispatch."
                ),
            )

        # Check evidence-based classifications BEFORE low-score fallbacks
        if self._is_insufficient_context(rejection_reasons):
            evidence = [
                r
                for r in rejection_reasons
                if any(kw in r.lower() for kw in ("context", "missing information", "not provided"))
            ]
            logger.info("Root cause classified as INSUFFICIENT_CONTEXT (evidence count=%d)", len(evidence))
            return RootCauseAnalysis(
                category=DefectCategory.INSUFFICIENT_CONTEXT,
                confidence=0.82,
                evidence=evidence or rejection_reasons[:1],
                corrective_action=(
                    "Gather the missing context or documentation and retry with an enriched prompt. "
                    "Include relevant background, prior decisions, and data sources."
                ),
                preventive_action=(
                    "Expand the context-retrieval step in the pipeline to proactively fetch "
                    "related artifacts, prior outputs, and domain references before agent execution."
                ),
            )

        if self._is_integration_error(rejection_reasons):
            evidence = [
                r
                for r in rejection_reasons
                if any(kw in r.lower() for kw in ("integration", "breaks", "conflict", "incompatible"))
            ]
            logger.info("Root cause classified as INTEGRATION_ERROR (evidence count=%d)", len(evidence))
            return RootCauseAnalysis(
                category=DefectCategory.INTEGRATION_ERROR,
                confidence=0.85,
                evidence=evidence or rejection_reasons[:1],
                corrective_action=(
                    "Run integration tests to identify the failing interface. "
                    "Align the output with the contract expected by downstream components."
                ),
                preventive_action=(
                    "Add integration smoke tests to the quality gate so integration "
                    "failures are caught before the output reaches review."
                ),
            )

        if self._is_complexity_issue(rejection_reasons):
            evidence = [
                r
                for r in rejection_reasons
                if any(kw in r.lower() for kw in ("complex", "too large", "split", "decompose"))
            ]
            logger.info("Root cause classified as COMPLEXITY_UNDERESTIMATE (evidence count=%d)", len(evidence))
            return RootCauseAnalysis(
                category=DefectCategory.COMPLEXITY_UNDERESTIMATE,
                confidence=0.78,
                evidence=evidence or rejection_reasons[:1],
                corrective_action=(
                    "Decompose the task into smaller, independently verifiable subtasks. "
                    "Re-plan with granular steps before retrying."
                ),
                preventive_action=(
                    "Improve complexity estimation in the planning phase. "
                    "Set a maximum task size threshold and enforce decomposition above it."
                ),
            )

        # Only check WRONG_MODEL if no other evidence-based category matched
        if self._is_wrong_model(rejection_reasons, quality_score):
            evidence = [r for r in rejection_reasons if any(kw in r.lower() for kw in ("capability", "model"))]
            if quality_score < 0.3:
                evidence = evidence or [f"Quality score {quality_score:.2f} is below capability threshold 0.30"]
            logger.info(
                "Root cause classified as WRONG_MODEL (score=%.2f, evidence count=%d)", quality_score, len(evidence)
            )
            return RootCauseAnalysis(
                category=DefectCategory.WRONG_MODEL,
                confidence=0.80,
                evidence=evidence or rejection_reasons[:1],
                corrective_action=(
                    "Re-route the task to a more capable model or a specialist agent. "
                    "Consider decomposing the task into subtasks within model capabilities."
                ),
                preventive_action=(
                    "Update the model routing rules to match task capability requirements "
                    "against model profiles before dispatch."
                ),
            )

        # Fallback: prompt weakness
        logger.info(
            "Root cause classified as PROMPT_WEAKNESS (fallback, no strong signal in %d reasons)",
            len(rejection_reasons),
        )
        return RootCauseAnalysis(
            category=DefectCategory.PROMPT_WEAKNESS,
            confidence=0.50,
            evidence=rejection_reasons[:2] if rejection_reasons else ["No specific signal detected"],
            corrective_action=(
                "Revise the prompt to be more explicit about output format, constraints, "
                "and success criteria, then retry."
            ),
            preventive_action=(
                "Review and refine prompt templates for this task type. "
                "Add few-shot examples demonstrating the expected output."
            ),
        )

    # ── Private heuristic helpers ──────────────────────────────────────────

    def _is_hallucination(self, reasons: list[str]) -> bool:
        """Return True if any reason signals a hallucination defect.

        Args:
            reasons: List of rejection reason strings.

        Returns:
            True when hallucination keywords are present in any reason.
        """
        keywords = ("import", "not found", "does not exist", "hallucinated", "fabricated")
        combined = " ".join(reasons).lower()
        return any(kw in combined for kw in keywords)

    def _is_bad_spec(self, reasons: list[str]) -> bool:
        """Return True if any reason signals an ambiguous or incomplete specification.

        Args:
            reasons: List of rejection reason strings.

        Returns:
            True when spec-quality keywords are present in any reason.
        """
        keywords = ("ambiguous", "unclear", "acceptance criteria", "spec", "incomplete", "bad_spec")
        combined = " ".join(reasons).lower()
        # Use word boundaries to avoid matching "spec" inside "specific", "speculation", etc.
        return any(re.search(r"\b" + re.escape(kw) + r"\b", combined) for kw in keywords)

    def _is_wrong_model(self, reasons: list[str], score: float) -> bool:
        """Return True if the model lacked capability for this task.

        Args:
            reasons: List of rejection reason strings.
            score: Numeric quality score between 0.0 and 1.0.

        Returns:
            True when capability keywords are present or the score is critically low.
        """
        keywords = ("capability", "model")
        combined = " ".join(reasons).lower()
        keyword_hit = any(kw in combined for kw in keywords)
        # Require BOTH a critically low score AND keyword evidence, OR allow
        # keyword evidence alone.  Score-only triggering (score < 0.3 with no
        # capability keyword) is too broad — many non-model failures also score low.
        return keyword_hit or (score < 0.3 and keyword_hit)

    def _is_insufficient_context(self, reasons: list[str]) -> bool:
        """Return True if the agent lacked sufficient context to complete the task.

        Args:
            reasons: List of rejection reason strings.

        Returns:
            True when context-gap keywords are present in any reason.
        """
        keywords = ("context", "missing information", "not provided")
        combined = " ".join(reasons).lower()
        return any(kw in combined for kw in keywords)

    def _is_integration_error(self, reasons: list[str]) -> bool:
        """Return True if the output works in isolation but breaks integration.

        Args:
            reasons: List of rejection reason strings.

        Returns:
            True when integration-failure keywords are present in any reason.
        """
        keywords = ("integration", "breaks", "conflict", "incompatible")
        combined = " ".join(reasons).lower()
        return any(kw in combined for kw in keywords)

    def _is_complexity_issue(self, reasons: list[str]) -> bool:
        """Return True if the task was harder or larger than originally estimated.

        Args:
            reasons: List of rejection reason strings.

        Returns:
            True when complexity keywords are present in any reason.
        """
        keywords = ("complex", "too large", "split", "decompose")
        combined = " ".join(reasons).lower()
        return any(kw in combined for kw in keywords)


# ── Causal Root Cause Analysis ────────────────────────────────────────────


@dataclass
class CausalEdge:
    """A directed cause-effect relationship in the causal graph.

    Attributes:
        cause: The upstream cause node identifier.
        effect: The downstream effect node identifier.
        strength: Confidence in this causal link (0.0-1.0).
        evidence: What supports this causal link.
    """

    cause: str
    effect: str
    strength: float = 0.8
    evidence: str = ""

    def __repr__(self) -> str:
        return "CausalEdge(...)"


class CausalGraph:
    """Directed acyclic graph of cause-effect relationships for failure analysis.

    Builds a graph where nodes are failure events or conditions and edges
    represent causal links. Walking the graph from a symptom backwards
    through causes finds the deepest root cause.
    """

    def __init__(self) -> None:
        # All edges in the graph.
        # Written by: add_edge(), build_from_failures().
        # Read by: get_root_causes(), get_all_paths().
        self._edges: list[CausalEdge] = []

        # Inverted adjacency: effect -> list of CausalEdge where that node is the effect.
        # This lets us walk backwards from symptom to cause efficiently.
        self._adjacency: dict[str, list[CausalEdge]] = {}

        # Isolated node IDs — failures with no caused_by and no edges.
        # These are genuine standalone root causes that have no predecessor.
        # Without tracking them, get_root_causes() would return an empty list
        # for a single failure that has no causal chain.
        # Written by: build_from_failures(). Read by: get_root_causes().
        self._isolated_nodes: set[str] = set()

    def add_edge(self, cause: str, effect: str, strength: float = 0.8, evidence: str = "") -> None:
        """Add a directed causal link from *cause* to *effect*.

        Args:
            cause: Upstream node identifier.
            effect: Downstream node identifier.
            strength: Confidence in the causal link (0.0-1.0).
            evidence: Human-readable description supporting this link.

        Raises:
            ValueError: If cause and effect are the same node (self-loop).
        """
        if cause == effect:
            raise ValueError(f"Self-loop not allowed: {cause!r}")
        edge = CausalEdge(cause=cause, effect=effect, strength=strength, evidence=evidence)
        self._edges.append(edge)
        self._adjacency.setdefault(effect, []).append(edge)
        logger.debug("Causal edge added: %s -> %s (strength=%.2f)", cause, effect, strength)

    def build_from_failures(self, failures: list[dict[str, Any]]) -> None:
        """Construct the causal graph from a list of failure records.

        Each failure dict should have:
        - ``"id"`` (str): unique failure identifier
        - ``"caused_by"`` (str, optional): id of the failure that caused this one
        - ``"evidence"`` (str, optional): description of the causal link
        - ``"category"`` (str, optional): defect category for context

        Args:
            failures: List of failure dictionaries.
        """
        for failure in failures:
            failure_id = failure.get("id", "")
            if not failure_id:
                continue
            caused_by = failure.get("caused_by")
            if caused_by and caused_by != failure_id:
                self.add_edge(
                    cause=caused_by,
                    effect=failure_id,
                    evidence=failure.get("evidence", ""),
                )
            else:
                # No caused_by — this failure has no known predecessor and is
                # itself a root cause. Track it so get_root_causes() includes it.
                self._isolated_nodes.add(failure_id)

    def walk_to_root_cause(self, symptom: str) -> list[str]:
        """Walk backwards from *symptom* through causes to the deepest root cause.

        At each step, follows the highest-strength incoming edge. Uses
        cycle detection to prevent infinite loops in malformed graphs.

        Args:
            symptom: The node to start walking from.

        Returns:
            Path from symptom to root cause (symptom first, root cause last).
            Returns ``[symptom]`` if the node has no known causes.
        """
        path = [symptom]
        visited: set[str] = {symptom}
        current = symptom

        while True:
            incoming = self._adjacency.get(current, [])
            if not incoming:
                break
            # Follow highest-strength cause
            best = max(incoming, key=lambda e: e.strength)
            if best.cause in visited:
                logger.warning("Cycle detected in causal graph at %s — stopping walk", best.cause)
                break
            visited.add(best.cause)
            path.append(best.cause)
            current = best.cause

        return path

    def get_root_causes(self) -> list[str]:
        """Return all nodes that are causes but never effects (source nodes).

        Includes isolated failures — those with no caused_by and no edges —
        because a standalone failure is its own root cause.

        Returns:
            Sorted list of root cause node identifiers.
        """
        all_causes = {e.cause for e in self._edges}
        all_effects = {e.effect for e in self._edges}
        # Chain root causes (causes with no incoming edge) plus isolated nodes
        # (standalone failures that never appear in any edge).
        return sorted((all_causes - all_effects) | self._isolated_nodes)

    def get_all_paths(self, symptom: str) -> list[list[str]]:
        """Return all paths from *symptom* to root causes.

        Useful when a symptom has multiple independent cause chains.

        Args:
            symptom: The node to start from.

        Returns:
            List of paths, each a list of node IDs from symptom to root cause.
        """
        results: list[list[str]] = []
        self._dfs_all_paths(symptom, [symptom], set(), results)
        return results or [[symptom]]

    def _dfs_all_paths(
        self,
        node: str,
        current_path: list[str],
        visited: set[str],
        results: list[list[str]],
    ) -> None:
        """Recursive DFS to enumerate all paths to root causes."""
        incoming = self._adjacency.get(node, [])
        if not incoming:
            results.append(list(current_path))
            return

        for edge in incoming:
            if edge.cause not in visited:
                visited.add(edge.cause)
                current_path.append(edge.cause)
                self._dfs_all_paths(edge.cause, current_path, visited, results)
                current_path.pop()
                visited.discard(edge.cause)


def build_causal_graph(failures: list[dict[str, Any]]) -> CausalGraph:
    """Build a CausalGraph from a list of failure records.

    Args:
        failures: List of failure dicts with ``"id"``, ``"caused_by"``,
            and optional ``"evidence"`` keys.

    Returns:
        A populated CausalGraph ready for root cause analysis.
    """
    graph = CausalGraph()
    graph.build_from_failures(failures)
    return graph


def walk_graph_for_root_cause(graph: CausalGraph, symptom: str) -> str | None:
    """Walk the causal graph to find the deepest root cause of a symptom.

    Args:
        graph: A populated CausalGraph.
        symptom: The failure node to trace backwards from.

    Returns:
        The root cause node ID, or None if symptom is not in the graph.
    """
    # Check if symptom appears anywhere in the graph (as cause or effect)
    all_nodes = {e.cause for e in graph._edges} | {e.effect for e in graph._edges}
    if symptom not in all_nodes:
        return None
    path = graph.walk_to_root_cause(symptom)
    if not path:
        return None
    return path[-1]
