"""Tests for vetinari.validation.document_quality."""

from __future__ import annotations

from vetinari.validation.document_quality import (
    DIMENSIONS,
    EXCELLENCE,
    FOUNDATIONAL,
    PROFESSIONAL,
    DimensionScore,
    QualityReport,
    evaluate_document,
)


class TestDimensions:
    """Tests for dimension constants."""

    def test_twelve_dimensions(self):
        assert len(DIMENSIONS) == 12

    def test_tier_groupings(self):
        assert len(FOUNDATIONAL) == 4
        assert len(PROFESSIONAL) == 4
        assert len(EXCELLENCE) == 4
        assert FOUNDATIONAL + PROFESSIONAL + EXCELLENCE == DIMENSIONS

    def test_known_dimensions(self):
        assert "accuracy" in DIMENSIONS
        assert "clarity" in DIMENSIONS
        assert "technical_depth" in DIMENSIONS
        assert "style" in DIMENSIONS


class TestDimensionScore:
    """Tests for the DimensionScore dataclass."""

    def test_defaults(self):
        ds = DimensionScore(dimension="accuracy", score=0.9)
        assert ds.weight == 1.0
        assert ds.findings == []

    def test_custom_fields(self):
        ds = DimensionScore(
            dimension="clarity",
            score=0.7,
            weight=0.8,
            findings=["Long sentences detected"],
        )
        assert ds.score == 0.7
        assert ds.weight == 0.8
        assert len(ds.findings) == 1


class TestQualityReport:
    """Tests for the QualityReport dataclass."""

    def test_defaults(self):
        report = QualityReport(doc_type="readme", text_length=100)
        assert report.overall_score == 0.0
        assert report.passed is False
        assert report.dimension_scores == []
        assert report.anti_ai_findings == []

    def test_to_dict(self):
        ds = DimensionScore(dimension="accuracy", score=0.85, weight=1.0)
        report = QualityReport(
            doc_type="adr",
            text_length=500,
            dimension_scores=[ds],
            overall_score=0.85,
            passed=True,
        )
        d = report.to_dict()
        assert d["doc_type"] == "adr"
        assert d["text_length"] == 500
        assert d["overall_score"] == 0.85
        assert d["passed"] is True
        assert "accuracy" in d["dimensions"]
        assert d["dimensions"]["accuracy"]["score"] == 0.85

    def test_to_dict_includes_rules(self):
        report = QualityReport(
            doc_type="plan",
            text_length=200,
            profile_rules_passed=["rule1"],
            profile_rules_failed=["rule2"],
        )
        d = report.to_dict()
        assert d["rules_passed"] == ["rule1"]
        assert d["rules_failed"] == ["rule2"]


class TestEvaluateDocument:
    """Tests for the evaluate_document() function."""

    def test_basic_evaluation(self):
        text = "# My README\n\nThis project provides a tool for processing data efficiently."
        report = evaluate_document(text, doc_type="readme")
        assert isinstance(report, QualityReport)
        assert report.doc_type == "readme"
        assert report.text_length == len(text)
        assert 0.0 <= report.overall_score <= 1.0
        assert len(report.dimension_scores) == 12

    def test_empty_text_penalizes_completeness(self):
        report = evaluate_document("")
        # Empty text should trigger completeness penalty
        completeness = next(
            (ds for ds in report.dimension_scores if ds.dimension == "completeness"),
            None,
        )
        assert completeness is not None
        assert completeness.score < 1.0

    def test_hedging_lowers_accuracy(self):
        text = "This probably maybe works. Perhaps it might be correct."
        report = evaluate_document(text)
        accuracy = next(ds for ds in report.dimension_scores if ds.dimension == "accuracy")
        assert accuracy.score < 1.0
        assert any("Hedging" in f for f in accuracy.findings)

    def test_placeholders_lower_correctness(self):
        text = "TODO implement this. FIXME broken logic here."
        report = evaluate_document(text)
        correctness = next(ds for ds in report.dimension_scores if ds.dimension == "correctness")
        assert correctness.score < 1.0

    def test_filler_phrases_lower_conciseness(self):
        text = "It is important to note that in order to achieve this, we basically need to act."
        report = evaluate_document(text)
        conciseness = next(ds for ds in report.dimension_scores if ds.dimension == "conciseness")
        assert conciseness.score < 1.0

    def test_code_blocks_boost_technical_depth(self):
        text_with_code = "# Guide\n\n```python\ndef hello():\n    print('hi')\n```\n"
        text_without = "# Guide\n\nCall the hello function to print a greeting.\n"
        report_with = evaluate_document(text_with_code)
        report_without = evaluate_document(text_without)
        depth_with = next(ds for ds in report_with.dimension_scores if ds.dimension == "technical_depth")
        depth_without = next(ds for ds in report_without.dimension_scores if ds.dimension == "technical_depth")
        assert depth_with.score >= depth_without.score

    def test_mixed_list_markers_lower_consistency(self):
        text = "- item one\n- item two\n* item three\n* item four\n"
        report = evaluate_document(text)
        consistency = next(ds for ds in report.dimension_scores if ds.dimension == "consistency")
        assert consistency.score < 1.0

    def test_long_document_without_headings_lowers_organization(self):
        # 25 lines of plain text, no headings
        text = "\n".join([f"Line {i} of content here." for i in range(25)])
        report = evaluate_document(text)
        org = next(ds for ds in report.dimension_scores if ds.dimension == "organization")
        assert org.score < 1.0

    def test_pass_threshold(self):
        # Well-structured document should pass default threshold
        text = (
            "# Project Overview\n\n"
            "This project implements a data processing pipeline.\n\n"
            "## Installation\n\n"
            "Run `pip install mypackage` to install.\n\n"  # noqa: VET301 — user guidance string
            "## Usage\n\n"
            "```python\nimport mypackage\nmypackage.run()\n```\n"
        )
        report = evaluate_document(text, doc_type="default")
        assert report.passed is True

    def test_custom_profile_override(self):
        from vetinari.validation import DocumentProfile

        strict_profile = DocumentProfile(
            doc_type="strict",
            min_score=0.99,
            dimension_weights={"accuracy": 1.0},
        )
        text = "This is probably fine, maybe."
        report = evaluate_document(text, profile=strict_profile)
        # With 0.99 threshold and hedging, should fail
        assert report.passed is False
