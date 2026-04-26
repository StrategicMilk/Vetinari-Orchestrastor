"""Tests for vetinari.orchestration.request_spec.

Covers RequestSpec dataclass, QA dataclass, RequestSpecBuilder methods,
file reference extraction, acceptance criteria generation, complexity
estimation, confidence computation, and singleton management.
"""

from __future__ import annotations

import pytest

from vetinari.orchestration.intake import Tier
from vetinari.orchestration.request_spec import (
    QA,
    RequestSpec,
    RequestSpecBuilder,
    get_spec_builder,
    reset_spec_builder,
)
from vetinari.types import AgentType

# ── QA Dataclass ───────────────────────────────────────────────────────


class TestQA:
    """Test QA dataclass."""

    def test_defaults(self) -> None:
        qa = QA()
        assert qa.question == ""
        assert qa.answer == ""

    def test_to_dict(self) -> None:
        qa = QA(question="What?", answer="This.")
        d = qa.to_dict()
        assert d == {"question": "What?", "answer": "This."}

    def test_from_dict(self) -> None:
        qa = QA.from_dict({"question": "Q", "answer": "A"})
        assert qa.question == "Q"
        assert qa.answer == "A"

    def test_from_dict_missing_keys(self) -> None:
        qa = QA.from_dict({})
        assert qa.question == ""
        assert qa.answer == ""


# ── RequestSpec Dataclass ──────────────────────────────────────────────


class TestRequestSpec:
    """Test RequestSpec dataclass."""

    def test_defaults(self) -> None:
        spec = RequestSpec()
        assert spec.goal == ""
        assert spec.tier == "standard"
        assert spec.category == "general"
        assert spec.acceptance_criteria == []
        assert spec.scope == []
        assert spec.confidence == 1.0
        assert spec.estimated_complexity == 5

    def test_to_dict(self) -> None:
        spec = RequestSpec(
            goal="Build X",
            tier="express",
            category="code",
            acceptance_criteria=["Tests pass"],
            scope=["main.py"],
            clarifications=[QA("Q?", "A")],
        )
        d = spec.to_dict()
        assert d["goal"] == "Build X"
        assert d["tier"] == "express"
        assert d["scope"] == ["main.py"]
        assert d["clarifications"] == [{"question": "Q?", "answer": "A"}]

    def test_from_dict(self) -> None:
        data = {
            "goal": "Fix bug",
            "tier": "custom",
            "category": "debug",
            "acceptance_criteria": ["Bug fixed"],
            "scope": ["auth.py"],
            "confidence": 0.9,
            "estimated_complexity": 3,
            "clarifications": [{"question": "Where?", "answer": "auth.py"}],
        }
        spec = RequestSpec.from_dict(data)
        assert spec.goal == "Fix bug"
        assert spec.tier == "custom"
        assert spec.confidence == 0.9
        assert len(spec.clarifications) == 1
        assert spec.clarifications[0].question == "Where?"

    def test_from_dict_empty(self) -> None:
        spec = RequestSpec.from_dict({})
        assert spec.goal == ""
        assert spec.tier == "standard"

    def test_roundtrip(self) -> None:
        original = RequestSpec(
            goal="Test",
            tier="standard",
            category="code",
            acceptance_criteria=["A", "B"],
            scope=["x.py"],
            clarifications=[QA("Q", "A")],
            confidence=0.85,
            estimated_complexity=7,
        )
        restored = RequestSpec.from_dict(original.to_dict())
        assert restored.goal == original.goal
        assert restored.tier == original.tier
        assert restored.acceptance_criteria == original.acceptance_criteria
        assert restored.confidence == original.confidence

    def test_from_dict_string_where_list_expected_coerces_to_empty(self) -> None:
        """from_dict() coerces wrong-type list fields to [] rather than passing through.

        Regression for Bug #8: a string payload for acceptance_criteria/scope/
        out_of_scope/suggested_pipeline was silently stored as a string, causing
        downstream code that iterates the list to crash.
        """
        spec = RequestSpec.from_dict(
            {
                "goal": "test",
                "acceptance_criteria": "not-a-list",
                "scope": 42,
                "out_of_scope": True,
                "suggested_pipeline": "wrong",
            }
        )
        assert spec.acceptance_criteria == [], "String acceptance_criteria must become []"
        assert spec.scope == [], "Non-list scope must become []"
        assert spec.out_of_scope == [], "Non-list out_of_scope must become []"
        assert spec.suggested_pipeline == [], "String suggested_pipeline must become []"

    def test_from_dict_string_where_dict_expected_coerces_to_empty(self) -> None:
        """from_dict() coerces wrong-type dict fields to {} rather than passing through.

        Regression for Bug #8: a string payload for constraints was silently stored,
        causing downstream code that calls .items() on constraints to crash.
        """
        spec = RequestSpec.from_dict({"goal": "test", "constraints": "not-a-dict"})
        assert spec.constraints == {}, "String constraints must become {}"

    def test_from_dict_none_fields_coerce_to_defaults(self) -> None:
        """Explicit None for list/dict fields yields empty collections, not None."""
        spec = RequestSpec.from_dict(
            {
                "acceptance_criteria": None,
                "scope": None,
                "constraints": None,
                "suggested_pipeline": None,
            }
        )
        assert spec.acceptance_criteria == []
        assert spec.scope == []
        assert spec.constraints == {}
        assert spec.suggested_pipeline == []


# ── File Reference Extraction ──────────────────────────────────────────


class TestFileReferenceExtraction:
    """Test _extract_file_references."""

    def test_python_files(self) -> None:
        builder = RequestSpecBuilder()
        refs = builder._extract_file_references("Update auth.py and config.yaml")
        assert "auth.py" in refs
        assert "config.yaml" in refs

    def test_no_files(self) -> None:
        builder = RequestSpecBuilder()
        refs = builder._extract_file_references("Add error handling")
        assert refs == []

    def test_deduplication(self) -> None:
        builder = RequestSpecBuilder()
        refs = builder._extract_file_references("Fix main.py, then check main.py again")
        assert refs.count("main.py") == 1

    def test_various_extensions(self) -> None:
        builder = RequestSpecBuilder()
        refs = builder._extract_file_references("Update index.html, style.css, app.js, schema.sql")
        assert len(refs) == 4


# ── Acceptance Criteria Generation ─────────────────────────────────────


class TestAcceptanceCriteriaGeneration:
    """Test _generate_acceptance_criteria."""

    def test_code_category_defaults(self) -> None:
        builder = RequestSpecBuilder()
        criteria = builder._generate_acceptance_criteria("Build feature", "code", [])
        assert "All tests pass" in criteria
        assert "No lint errors" in criteria

    def test_security_category(self) -> None:
        builder = RequestSpecBuilder()
        criteria = builder._generate_acceptance_criteria("Fix vuln", "security", [])
        assert "No new vulnerabilities introduced" in criteria

    def test_general_fallback(self) -> None:
        builder = RequestSpecBuilder()
        criteria = builder._generate_acceptance_criteria("Do thing", "unknown_category", [])
        assert "Task requirements met" in criteria

    def test_clarification_adds_criteria(self) -> None:
        builder = RequestSpecBuilder()
        clarifications = [QA("What format?", "JSON with pagination")]
        criteria = builder._generate_acceptance_criteria("Build API", "code", clarifications)
        assert any("JSON with pagination" in c for c in criteria)

    def test_short_answer_skipped(self) -> None:
        builder = RequestSpecBuilder()
        clarifications = [QA("Ok?", "yes")]
        criteria = builder._generate_acceptance_criteria("Build API", "code", clarifications)
        assert not any("yes" in c for c in criteria)


# ── Complexity Estimation ──────────────────────────────────────────────


class TestComplexityEstimation:
    """Test _estimate_complexity."""

    def test_short_goal_low_complexity(self) -> None:
        builder = RequestSpecBuilder()
        assert builder._estimate_complexity("Fix typo", [], "code") <= 4

    def test_long_goal_high_complexity(self) -> None:
        builder = RequestSpecBuilder()
        goal = " ".join(["word"] * 70)
        assert builder._estimate_complexity(goal, [], "code") >= 7

    def test_many_files_increases_complexity(self) -> None:
        builder = RequestSpecBuilder()
        low = builder._estimate_complexity("Build feature", [], "code")
        high = builder._estimate_complexity("Build feature", ["a.py", "b.py", "c.py"], "code")
        assert high > low

    def test_security_adds_complexity(self) -> None:
        builder = RequestSpecBuilder()
        code = builder._estimate_complexity("Fix thing", [], "code")
        security = builder._estimate_complexity("Fix thing", [], "security")
        assert security > code

    def test_clamped_to_range(self) -> None:
        builder = RequestSpecBuilder()
        assert 1 <= builder._estimate_complexity("x", [], "code") <= 10
        goal = " ".join(["word"] * 100)
        assert 1 <= builder._estimate_complexity(goal, ["a.py"] * 10, "security") <= 10


# ── Confidence Computation ─────────────────────────────────────────────


class TestConfidenceComputation:
    """Test _compute_confidence."""

    def test_base_confidence(self) -> None:
        builder = RequestSpecBuilder()
        conf = builder._compute_confidence("goal", [], [])
        assert 0.5 <= conf <= 1.0

    def test_more_criteria_increases_confidence(self) -> None:
        builder = RequestSpecBuilder()
        low = builder._compute_confidence("goal", [], ["a"])
        high = builder._compute_confidence("goal", [], ["a", "b", "c"])
        assert high >= low

    def test_clarifications_increase_confidence(self) -> None:
        builder = RequestSpecBuilder()
        low = builder._compute_confidence("goal", [], ["a"])
        high = builder._compute_confidence("goal", [QA("Q", "A")], ["a"])
        assert high >= low

    def test_longer_goal_increases_confidence(self) -> None:
        builder = RequestSpecBuilder()
        short = builder._compute_confidence("fix it", [], ["a"])
        long_ = builder._compute_confidence(
            "Add comprehensive error handling to the authentication module with proper logging and unit tests for each branch",
            [],
            ["a"],
        )
        assert long_ >= short

    def test_clamped_to_unit_interval(self) -> None:
        builder = RequestSpecBuilder()
        conf = builder._compute_confidence("x", [], [])
        assert 0.0 <= conf <= 1.0


# ── Full Build ─────────────────────────────────────────────────────────


class TestBuild:
    """Test RequestSpecBuilder.build() end-to-end."""

    def test_basic_build(self) -> None:
        builder = RequestSpecBuilder()
        spec = builder.build("Fix auth.py bug", Tier.EXPRESS, "debug")
        assert spec.goal == "Fix auth.py bug"
        assert spec.tier == "express"
        assert spec.category == "debug"
        assert "auth.py" in spec.scope
        assert len(spec.acceptance_criteria) > 0
        assert spec.suggested_pipeline == [AgentType.WORKER.value, AgentType.INSPECTOR.value]

    def test_custom_tier_pipeline(self) -> None:
        builder = RequestSpecBuilder()
        spec = builder.build("Complex task", Tier.CUSTOM, "code")
        assert AgentType.WORKER.value in spec.suggested_pipeline
        assert AgentType.FOREMAN.value in spec.suggested_pipeline

    def test_with_clarifications(self) -> None:
        builder = RequestSpecBuilder()
        clarifications = [QA("What format?", "JSON with nested objects")]
        spec = builder.build("Build API endpoint", Tier.STANDARD, "code", clarifications)
        assert len(spec.clarifications) == 1
        assert any("JSON" in c for c in spec.acceptance_criteria)

    def test_all_fields_populated(self) -> None:
        builder = RequestSpecBuilder()
        spec = builder.build("Add feature to main.py", Tier.STANDARD, "code")
        assert spec.goal != ""
        assert spec.tier in ("express", "standard", "custom")
        assert spec.category != ""
        assert len(spec.acceptance_criteria) > 0
        assert isinstance(spec.estimated_complexity, int)
        assert 0.0 <= spec.confidence <= 1.0
        assert len(spec.suggested_pipeline) > 0


# ── Singleton ──────────────────────────────────────────────────────────


class TestSingleton:
    """Test singleton management."""

    def setup_method(self) -> None:
        reset_spec_builder()

    def teardown_method(self) -> None:
        reset_spec_builder()

    def test_returns_same_instance(self) -> None:
        b1 = get_spec_builder()
        b2 = get_spec_builder()
        assert b1 is b2

    def test_is_spec_builder(self) -> None:
        assert isinstance(get_spec_builder(), RequestSpecBuilder)

    def test_reset(self) -> None:
        b1 = get_spec_builder()
        reset_spec_builder()
        b2 = get_spec_builder()
        assert b1 is not b2
