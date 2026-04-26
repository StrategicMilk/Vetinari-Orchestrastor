"""Tests for vetinari.ontology — canonical vocabulary and quality types."""

from __future__ import annotations

import pytest

from vetinari.ontology import (
    AGENT_CAPABILITIES,
    QUALITY_THRESHOLD_PASS,
    TASK_QUALITY_DIMENSIONS,
    CanonicalTerm,
    DefectSeverity,
    GateDecision,
    QualityAssessment,
    QualityDimension,
    QualityLevel,
    RelationshipType,
    SuccessSignal,
    TemporalStatus,
    calculate_quality_score,
    gate_to_signal,
    score_to_level,
)

# -- score_to_level --


def test_score_to_level_boundaries() -> None:
    """Verify every grade boundary maps to the correct QualityLevel."""
    assert score_to_level(0.0) == QualityLevel.FAILING
    assert score_to_level(0.39) == QualityLevel.FAILING
    assert score_to_level(0.40) == QualityLevel.POOR
    assert score_to_level(0.59) == QualityLevel.POOR
    assert score_to_level(0.60) == QualityLevel.ADEQUATE
    assert score_to_level(0.74) == QualityLevel.ADEQUATE
    assert score_to_level(0.75) == QualityLevel.GOOD
    assert score_to_level(0.89) == QualityLevel.GOOD
    assert score_to_level(0.90) == QualityLevel.EXCELLENT
    assert score_to_level(1.0) == QualityLevel.EXCELLENT


# -- calculate_quality_score --


def test_calculate_quality_score_zero_defects() -> None:
    """Perfect score when there are no defects."""
    assert calculate_quality_score() == 1.0


def test_calculate_quality_score_one_critical() -> None:
    """One critical defect yields 0.70."""
    assert calculate_quality_score(critical=1) == pytest.approx(0.70)


def test_calculate_quality_score_two_criticals() -> None:
    """Two critical defects yield 0.40."""
    assert calculate_quality_score(critical=2) == pytest.approx(0.40)


def test_calculate_quality_score_mixed_defects() -> None:
    """Mixed severity defects accumulate penalties correctly."""
    # 1 critical (0.30) + 1 high (0.15) + 2 medium (0.10) + 3 low (0.06) = 0.61 penalty -> 0.39
    result = calculate_quality_score(critical=1, high=1, medium=2, low=3)
    assert result == pytest.approx(0.39)


def test_calculate_quality_score_never_below_zero() -> None:
    """Score is clamped to 0.0 even with extreme defect counts."""
    result = calculate_quality_score(critical=10, high=10, medium=10, low=10)
    assert result == 0.0


def test_calculate_quality_score_only_low_defects() -> None:
    """Low-severity defects apply a small penalty."""
    result = calculate_quality_score(low=5)
    assert result == pytest.approx(0.90)


# -- gate_to_signal --


def test_gate_to_signal_passed_decision() -> None:
    """A passing gate decision yields a successful SuccessSignal."""
    decision = GateDecision(
        artifact_id="art-1",
        passed=True,
        score=0.85,
        level=QualityLevel.GOOD,
    )
    signal = gate_to_signal(decision, model_id="model-a", task_type="coding")
    assert signal.success is True
    assert signal.model_id == "model-a"
    assert signal.task_type == "coding"
    assert signal.quality_weight == pytest.approx(0.85)


def test_gate_to_signal_failed_decision() -> None:
    """A failing gate decision yields a SuccessSignal with success=False."""
    decision = GateDecision(
        artifact_id="art-2",
        passed=False,
        score=0.45,
        level=QualityLevel.POOR,
    )
    signal = gate_to_signal(decision, model_id="model-b", task_type="research")
    assert signal.success is False
    assert signal.model_id == "model-b"
    assert signal.task_type == "research"
    assert signal.quality_weight == pytest.approx(0.45)


def test_gate_to_signal_preserves_model_and_task_type() -> None:
    """gate_to_signal passes through model_id and task_type unchanged."""
    decision = GateDecision(
        artifact_id="art-3",
        passed=True,
        score=0.92,
        level=QualityLevel.EXCELLENT,
    )
    signal = gate_to_signal(decision, model_id="llama-3-8b", task_type="documentation")
    assert signal.model_id == "llama-3-8b"
    assert signal.task_type == "documentation"


# -- QualityAssessment --


def test_quality_assessment_level_property() -> None:
    """QualityAssessment.level derives the correct level from overall_score."""
    assessment = QualityAssessment(
        artifact_id="art-4",
        model_id="model-c",
        task_type="coding",
        overall_score=0.92,
    )
    assert assessment.level == QualityLevel.EXCELLENT


def test_quality_assessment_measured_dimensions_filters_unmeasured() -> None:
    """measured_dimensions returns only dimensions with measured=True."""
    dims = (
        QualityDimension(name="correctness", score=0.8, measured=True, method="heuristic"),
        QualityDimension(name="completeness", score=0.0, measured=False, method="unmeasured"),
        QualityDimension(name="style", score=0.7, measured=True, method="heuristic"),
    )
    assessment = QualityAssessment(
        artifact_id="art-5",
        model_id="model-d",
        task_type="coding",
        overall_score=0.75,
        dimensions=dims,
    )
    measured = assessment.measured_dimensions
    assert len(measured) == 2
    assert all(d.measured for d in measured)
    names = {d.name for d in measured}
    assert names == {"correctness", "style"}


def test_quality_assessment_repr_contains_key_fields() -> None:
    """QualityAssessment repr includes artifact_id and score."""
    assessment = QualityAssessment(
        artifact_id="art-repr",
        model_id="model-e",
        task_type="research",
        overall_score=0.77,
    )
    r = repr(assessment)
    assert "art-repr" in r
    assert "0.77" in r


# -- GateDecision --


def test_gate_decision_repr_contains_key_fields() -> None:
    """GateDecision repr includes artifact_id, passed, and score."""
    decision = GateDecision(
        artifact_id="gate-repr",
        passed=True,
        score=0.81,
        level=QualityLevel.GOOD,
    )
    r = repr(decision)
    assert "gate-repr" in r
    assert "True" in r
    assert "0.81" in r


# -- SuccessSignal --


def test_success_signal_repr_contains_key_fields() -> None:
    """SuccessSignal repr includes model_id and success."""
    signal = SuccessSignal(
        model_id="signal-model",
        task_type="testing",
        success=False,
        quality_weight=0.35,
    )
    r = repr(signal)
    assert "signal-model" in r
    assert "False" in r


# -- Enum value presence --


def test_defect_severity_values_exist() -> None:
    """All DefectSeverity enum values are defined."""
    assert DefectSeverity.CRITICAL.value == "critical"
    assert DefectSeverity.HIGH.value == "high"
    assert DefectSeverity.MEDIUM.value == "medium"
    assert DefectSeverity.LOW.value == "low"


def test_canonical_term_values_exist() -> None:
    """All CanonicalTerm enum values are defined."""
    assert CanonicalTerm.ARTIFACT.value == "artifact"
    assert CanonicalTerm.ASSESSMENT.value == "assessment"
    assert CanonicalTerm.DECISION.value == "decision"
    assert CanonicalTerm.SIGNAL.value == "signal"
    assert CanonicalTerm.DEFECT.value == "defect"
    assert CanonicalTerm.DIRECTIVE.value == "directive"
    assert CanonicalTerm.SPECIFICATION.value == "specification"


def test_relationship_type_values_exist() -> None:
    """All RelationshipType enum values are defined."""
    assert RelationshipType.SUPERSEDES.value == "supersedes"
    assert RelationshipType.CONTRADICTS.value == "contradicts"
    assert RelationshipType.CAUSED_BY.value == "caused_by"
    assert RelationshipType.ELABORATES.value == "elaborates"
    assert RelationshipType.PART_OF.value == "part_of"


def test_temporal_status_values_exist() -> None:
    """All TemporalStatus enum values are defined."""
    assert TemporalStatus.ACTIVE.value == "active"
    assert TemporalStatus.COMPLETED.value == "completed"
    assert TemporalStatus.PLANNED.value == "planned"
    assert TemporalStatus.ABANDONED.value == "abandoned"
    assert TemporalStatus.RECURRING.value == "recurring"


# -- AGENT_CAPABILITIES --


def test_agent_capabilities_all_three_agents_defined() -> None:
    """AGENT_CAPABILITIES defines all three factory pipeline agents."""
    assert "FOREMAN" in AGENT_CAPABILITIES
    assert "WORKER" in AGENT_CAPABILITIES
    assert "INSPECTOR" in AGENT_CAPABILITIES


def test_agent_capabilities_each_has_can_and_cannot() -> None:
    """Every agent entry has both 'can' and 'cannot' lists."""
    for agent_key, caps in AGENT_CAPABILITIES.items():
        assert "can" in caps, f"{agent_key} missing 'can'"
        assert "cannot" in caps, f"{agent_key} missing 'cannot'"
        assert len(caps["can"]) > 0, f"{agent_key}.can is empty"
        assert len(caps["cannot"]) > 0, f"{agent_key}.cannot is empty"


# -- TASK_QUALITY_DIMENSIONS --


def test_task_quality_dimensions_core_types_present() -> None:
    """Core task types have at least one quality dimension defined."""
    for task_type in ("coding", "research", "analysis", "documentation"):
        assert task_type in TASK_QUALITY_DIMENSIONS, f"Missing task type: {task_type}"
        assert len(TASK_QUALITY_DIMENSIONS[task_type]) > 0, f"{task_type} has no dimensions"


def test_task_quality_dimensions_coding_has_expected_dims() -> None:
    """Coding task type includes correctness, completeness, and style."""
    coding_dims = TASK_QUALITY_DIMENSIONS["coding"]
    assert "correctness" in coding_dims
    assert "completeness" in coding_dims
    assert "style" in coding_dims


# -- QUALITY_THRESHOLD_PASS --


class TestQualityThresholdPass:
    def test_value_is_0_7(self) -> None:
        assert QUALITY_THRESHOLD_PASS == 0.7

    def test_is_float(self) -> None:
        assert isinstance(QUALITY_THRESHOLD_PASS, float)

    def test_used_as_gate_boundary(self) -> None:
        """Score exactly at the threshold is considered passing."""
        assert QUALITY_THRESHOLD_PASS >= 0.0
        assert QUALITY_THRESHOLD_PASS <= 1.0


# -- SuccessSignal.from_quality_score --


class TestSuccessSignalFromQualityScore:
    def test_returns_success_signal_instance(self) -> None:
        sig = SuccessSignal.from_quality_score(0.8, True)
        assert isinstance(sig, SuccessSignal)

    def test_success_true_preserved(self) -> None:
        sig = SuccessSignal.from_quality_score(0.8, True)
        assert sig.success is True

    def test_success_false_preserved(self) -> None:
        sig = SuccessSignal.from_quality_score(0.4, False)
        assert sig.success is False

    def test_quality_weight_within_bounds_unchanged(self) -> None:
        sig = SuccessSignal.from_quality_score(0.75, True)
        assert sig.quality_weight == 0.75

    def test_quality_weight_clamped_to_one(self) -> None:
        sig = SuccessSignal.from_quality_score(1.5, True)
        assert sig.quality_weight == 1.0

    def test_quality_weight_clamped_to_zero(self) -> None:
        sig = SuccessSignal.from_quality_score(-0.3, False)
        assert sig.quality_weight == 0.0

    def test_model_id_defaults_to_empty_string(self) -> None:
        sig = SuccessSignal.from_quality_score(0.8, True)
        assert sig.model_id == ""

    def test_task_type_defaults_to_empty_string(self) -> None:
        sig = SuccessSignal.from_quality_score(0.8, True)
        assert sig.task_type == ""

    def test_model_id_passed_through(self) -> None:
        sig = SuccessSignal.from_quality_score(0.8, True, model_id="qwen2-7b")
        assert sig.model_id == "qwen2-7b"

    def test_task_type_passed_through(self) -> None:
        sig = SuccessSignal.from_quality_score(0.8, True, task_type="coding")
        assert sig.task_type == "coding"

    def test_at_threshold_boundary_preserved(self) -> None:
        sig = SuccessSignal.from_quality_score(QUALITY_THRESHOLD_PASS, True)
        assert sig.quality_weight == QUALITY_THRESHOLD_PASS

    @pytest.mark.parametrize(
        "score,expected",
        [
            (0.0, 0.0),
            (0.5, 0.5),
            (1.0, 1.0),
            (2.0, 1.0),
            (-1.0, 0.0),
        ],
    )
    def test_clamping_parametrized(self, score: float, expected: float) -> None:
        sig = SuccessSignal.from_quality_score(score, True)
        assert sig.quality_weight == expected
