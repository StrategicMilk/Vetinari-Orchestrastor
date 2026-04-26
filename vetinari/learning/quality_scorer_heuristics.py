"""Heuristic scoring helpers for Vetinari quality assessment."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vetinari.learning.quality_scorer import QualityScore

_QualityScoreFactory = Callable[..., "QualityScore"]


def _score_heuristic_output(
    task_id: str,
    model_id: str,
    task_type: str,
    output: str,
    dims: list[str],
    inference_confidence: float | None = None,
    baseline_config: Mapping[str, Mapping[str, float]] | None = None,
    score_factory: _QualityScoreFactory | None = None,
) -> QualityScore:
    """Heuristic quality scoring with structural checks per task type.

    Uses task-specific baselines from quality_baselines.yaml instead of
    flat defaults.  Each check adjusts scores up or down from baseline
    based on concrete structural evidence found in the output.

    Args:
        task_id: Unique task identifier.
        model_id: Model that produced the output.
        task_type: Type of task (coding, research, etc.).
        output: The output to evaluate.
        dims: List of dimension names to score.
        inference_confidence: Optional confidence from logprob variance (0.0-1.0).
        baseline_config: Task-type baseline scores used by the heuristic scorer.
        score_factory: Factory used to construct returned quality score values.
    """
    if score_factory is None:
        raise ValueError("score_factory is required for heuristic quality scoring")
    if baseline_config is None:
        baseline_config = {}

    issues: list[str] = []
    scores: dict[str, float] = {}
    measured: list[str] = []

    if not output or not output.strip():
        return score_factory(
            task_id=task_id,
            model_id=model_id,
            task_type=task_type,
            overall_score=0.0,
            issues=["Empty output"],
            dimensions=dict.fromkeys(dims, 0.0),
            measured_dimensions=["completeness"],
            method="heuristic",
        )

    baselines = baseline_config.get(task_type.lower(), baseline_config.get("default", {}))
    words = len(output.split())
    lines = output.split("\n")

    # -- Completeness: length relative to expected output size --
    if words < 10:
        scores["completeness"] = 0.15
        issues.append("Very short output (< 10 words)")
        measured.append("completeness")
    elif words > 2000:
        scores["efficiency"] = 0.4
        scores["completeness"] = 0.85
        issues.append("Very long output — may lack focus")
        measured.extend(["efficiency", "completeness"])
    else:
        scores["completeness"] = min(0.9, 0.2 + (words / 300))
        measured.append("completeness")

    # -- Task-specific structural checks --
    if task_type.lower() == "coding":
        _score_coding_structural(output, lines, scores, measured, issues)
    elif task_type.lower() == "research":
        _score_research_structural(output, lines, scores, measured, issues)
    elif task_type.lower() == "documentation":
        _score_documentation_structural(output, lines, scores, measured, issues)
    elif task_type.lower() == "analysis":
        _score_analysis_structural(output, lines, scores, measured, issues)
    elif task_type.lower() == "testing":
        _score_testing_structural(output, lines, scores, measured, issues)

    # -- General format compliance (all task types) --
    if not any(c in output for c in (".", "!", "?", ":", "```")):
        scores.setdefault("quality", max(0.0, baselines.get("quality", 0.45) - 0.15))
        issues.append("No sentence-ending punctuation or code blocks")
        if "quality" not in measured:
            measured.append("quality")

    # -- Inference confidence adjustment (Item 3.5) --
    if inference_confidence is not None and inference_confidence < 0.5:
        confidence_penalty = (0.5 - inference_confidence) * 0.2  # Max 0.1 penalty
        for dim in list(scores):
            scores[dim] = max(0.0, scores[dim] - confidence_penalty)
        issues.append(f"Low inference confidence ({inference_confidence:.2f})")

    # Fill unmeasured dimensions with 0.0 (not baselines — unmeasured means unknown)
    for d in dims:
        if d not in scores:
            scores[d] = 0.0

    measured_set = set(measured)
    overall_measured = [scores[d] for d in dims if d in measured_set]
    if overall_measured:
        overall = sum(overall_measured) / len(overall_measured)
    else:
        overall = 0.0

    return score_factory(
        task_id=task_id,
        model_id=model_id,
        task_type=task_type,
        overall_score=round(overall, 3),
        correctness=scores.get("correctness", 0.0),
        completeness=scores.get("completeness", 0.0),
        efficiency=scores.get("efficiency", 0.0),
        style=scores.get("style", 0.0),
        dimensions=scores,
        measured_dimensions=measured,
        issues=issues,
        method="heuristic",
    )


def _score_coding_structural(
    output: str,
    lines: list[str],
    scores: dict[str, float],
    measured: list[str],
    issues: list[str],
) -> None:
    """Structural checks for coding output: defs, error handling, types, arity."""
    has_def = "def " in output or "class " in output
    has_docstring = '"""' in output or "'''" in output
    has_error_handling = "try:" in output or "except " in output
    has_type_hints = ": str" in output or ": int" in output or "-> " in output or ": list" in output
    has_imports = any(line.strip().startswith(("import ", "from ")) for line in lines)
    has_test = "assert " in output or "def test_" in output

    # Correctness: function structure + error handling
    correctness = 0.3
    if has_def:
        correctness += 0.25
    else:
        issues.append("No function/class definitions found")
    if has_error_handling:
        correctness += 0.15
    if has_imports:
        correctness += 0.1
    scores["correctness"] = min(1.0, correctness)
    measured.append("correctness")

    # Style: docstrings + type hints
    style = 0.3
    if has_docstring:
        style += 0.25
    if has_type_hints:
        style += 0.25
    # Check function arity — penalize functions with > 6 params
    def_lines = [line for line in lines if "def " in line]
    if def_lines:
        avg_params = sum(line.count(",") + 1 for line in def_lines) / len(def_lines)
        if avg_params <= 4:
            style += 0.1
        elif avg_params > 6:
            style -= 0.1
            issues.append("Functions have high arity (> 6 params)")
    scores["style"] = min(1.0, max(0.0, style))
    measured.append("style")

    # Test coverage
    scores["test_coverage"] = 0.7 if has_test else 0.15
    measured.append("test_coverage")

    # Efficiency: check for common anti-patterns
    efficiency = 0.55
    if "time.sleep" in output:
        efficiency -= 0.15
        issues.append("Contains time.sleep — potential performance issue")
    if output.count("for ") > 5 and output.count("for ") > len(def_lines) * 2:
        efficiency -= 0.1
        issues.append("High loop density relative to function count")
    scores["efficiency"] = max(0.0, efficiency)
    measured.append("efficiency")


def _score_research_structural(
    output: str,
    lines: list[str],
    scores: dict[str, float],
    measured: list[str],
    issues: list[str],
) -> None:
    """Structural checks for research output: citations, evidence, actionability."""
    has_urls = "http" in output
    has_source_refs = (
        "source" in output.lower() or "reference" in output.lower() or "according to" in output.lower()
    )
    section_count = output.count("\n#") + output.count("\n## ")
    has_conclusion = "conclusion" in output.lower() or "summary" in output.lower() or "recommend" in output.lower()
    has_evidence = "evidence" in output.lower() or "data shows" in output.lower() or "study" in output.lower()

    # Source quality
    source_score = 0.2
    if has_urls:
        source_score += 0.35
    if has_source_refs:
        source_score += 0.2
    scores["source_quality"] = min(1.0, source_score)
    measured.append("source_quality")
    if not has_urls and not has_source_refs:
        issues.append("No source citations or references found")

    # Accuracy (evidence-based claims)
    accuracy = 0.35
    if has_evidence:
        accuracy += 0.3
    if has_source_refs:
        accuracy += 0.15
    scores["accuracy"] = min(1.0, accuracy)
    measured.append("accuracy")

    # Actionability
    actionability = 0.3
    if has_conclusion:
        actionability += 0.25
    if section_count >= 3:
        actionability += 0.2
    scores["actionability"] = min(1.0, actionability)
    measured.append("actionability")

    # Completeness (section depth)
    completeness_bonus = min(0.3, section_count * 0.08)
    scores["completeness"] = min(1.0, scores.get("completeness", 0.4) + completeness_bonus)
    measured.append("completeness")


def _score_documentation_structural(
    output: str,
    lines: list[str],
    scores: dict[str, float],
    measured: list[str],
    issues: list[str],
) -> None:
    """Structural checks for documentation: examples, cross-refs, hierarchy."""
    has_examples = "example" in output.lower() or "```" in output
    has_cross_refs = "see " in output.lower() or "[" in output or "refer to" in output.lower()
    heading_count = sum(1 for line in lines if line.strip().startswith("#"))
    has_code_blocks = output.count("```") >= 2

    # Clarity: headings + structure
    clarity = 0.35
    if heading_count >= 2:
        clarity += 0.25
    if len(lines) > 10:
        clarity += 0.15
    scores["clarity"] = min(1.0, clarity)
    measured.append("clarity")

    # Examples
    examples_score = 0.15
    if has_examples:
        examples_score += 0.35
    if has_code_blocks:
        examples_score += 0.25
    scores["examples"] = min(1.0, examples_score)
    measured.append("examples")
    if not has_examples:
        issues.append("No examples found in documentation")

    # Accuracy (cross-references suggest verified content)
    accuracy = 0.4
    if has_cross_refs:
        accuracy += 0.25
    scores["accuracy"] = min(1.0, accuracy)
    measured.append("accuracy")


def _score_analysis_structural(
    output: str,
    lines: list[str],
    scores: dict[str, float],
    measured: list[str],
    issues: list[str],
) -> None:
    """Structural checks for analysis output: depth, accuracy, actionability."""
    section_count = sum(1 for line in lines if line.strip().startswith("#"))
    has_data_refs = any(kw in output.lower() for kw in ("data", "metric", "percent", "ratio", "trend"))
    has_recommendations = any(kw in output.lower() for kw in ("recommend", "suggest", "should", "action"))
    word_count = len(output.split())

    # Depth: section structure + data references + length
    depth = 0.25
    if section_count >= 3:
        depth += 0.2
    if has_data_refs:
        depth += 0.2
    if word_count > 300:
        depth += 0.15
    scores["depth"] = min(1.0, depth)
    measured.append("depth")

    # Actionability
    actionability = 0.25
    if has_recommendations:
        actionability += 0.35
    scores["actionability"] = min(1.0, actionability)
    measured.append("actionability")

    # Clarity
    clarity = 0.4
    if section_count >= 2:
        clarity += 0.2
    scores["clarity"] = min(1.0, clarity)
    measured.append("clarity")


def _score_testing_structural(
    output: str,
    lines: list[str],
    scores: dict[str, float],
    measured: list[str],
    issues: list[str],
) -> None:
    """Structural checks for testing output: coverage, assertions, edge cases."""
    assert_count = output.count("assert ")
    test_func_count = sum(1 for line in lines if "def test_" in line)
    has_edge_cases = any(kw in output.lower() for kw in ("edge", "boundary", "empty", "none", "zero", "negative"))
    has_fixtures = "fixture" in output.lower() or "@pytest" in output

    # Coverage (test function density)
    coverage = 0.2
    if test_func_count >= 3:
        coverage += 0.3
    elif test_func_count >= 1:
        coverage += 0.15
    if has_fixtures:
        coverage += 0.15
    scores["coverage"] = min(1.0, coverage)
    measured.append("coverage")

    # Correctness (assertion density)
    correctness = 0.25
    if assert_count >= 3:
        correctness += 0.35
    elif assert_count >= 1:
        correctness += 0.2
    scores["correctness"] = min(1.0, correctness)
    measured.append("correctness")

    # Edge cases
    edge_score = 0.15
    if has_edge_cases:
        edge_score += 0.45
    scores["edge_cases"] = min(1.0, edge_score)
    measured.append("edge_cases")
    if not has_edge_cases:
        issues.append("No edge case testing detected")
