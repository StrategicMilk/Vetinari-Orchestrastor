"""
Tests for vetinari/agents/skill_contract.py

Covers:
1.  SkillOutput creation and to_dict() serialization
2.  SkillOutput.from_dict() round-trip deserialization
3.  Finding.to_dict() with all severity levels
4.  Artifact.to_dict() with content truncation
5.  compute_overall_score() with known rubric weights
6.  compute_overall_score() with unknown agent type (equal weights)
7.  self_check() — passes clean output
8.  self_check() — catches missing evidence
9.  self_check() — catches vague recommendations
10. self_check() — catches extreme scores
11. SCORING_RUBRICS has all 8 agent types
12. All rubric weights sum to ~1.0
"""

import pytest

from vetinari.agents.skill_contract import (
    ArtifactType,
    DataProvenance,
    Finding,
    Artifact,
    SCORING_RUBRICS,
    Severity,
    SkillOutput,
    Verdict,
    compute_overall_score,
    self_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clean_output() -> SkillOutput:
    return SkillOutput(
        agent_type="BUILDER",
        task_summary="Build a widget",
        verdict=Verdict.PASS,
        confidence=0.85,
        scores={"syntax_validity": 0.9, "completeness": 0.8},
        overall_score=0.85,
        data_provenance=DataProvenance.MEASURED,
    )


def _make_finding(
    fid="F-001",
    severity=Severity.HIGH,
    evidence="def foo(): pass",
    location="main.py:42",
    recommendation="Refactor foo to use proper error handling",
) -> Finding:
    return Finding(
        id=fid,
        severity=severity,
        category="code_quality",
        title="Missing error handling",
        location=location,
        evidence=evidence,
        recommendation=recommendation,
    )


# ---------------------------------------------------------------------------
# 1. SkillOutput creation and to_dict() serialization
# ---------------------------------------------------------------------------

class TestSkillOutputCreation:
    def test_basic_creation(self):
        out = _make_clean_output()
        assert out.agent_type == "BUILDER"
        assert out.verdict == Verdict.PASS
        assert out.confidence == 0.85

    def test_to_dict_keys(self):
        d = _make_clean_output().to_dict()
        for key in (
            "agent_type", "task_summary", "verdict", "confidence",
            "findings", "scores", "overall_score", "artifacts",
            "sources", "data_provenance", "self_check_passed", "self_check_issues",
        ):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_enum_values(self):
        d = _make_clean_output().to_dict()
        assert d["verdict"] == "pass"
        assert d["data_provenance"] == "measured"

    def test_to_dict_findings_list(self):
        out = _make_clean_output()
        out.findings.append(_make_finding())
        d = out.to_dict()
        assert len(d["findings"]) == 1
        assert d["findings"][0]["id"] == "F-001"


# ---------------------------------------------------------------------------
# 2. SkillOutput.from_dict() round-trip deserialization
# ---------------------------------------------------------------------------

class TestSkillOutputRoundTrip:
    def test_round_trip_empty(self):
        original = _make_clean_output()
        restored = SkillOutput.from_dict(original.to_dict())
        assert restored.agent_type == original.agent_type
        assert restored.verdict == original.verdict
        assert restored.confidence == original.confidence
        assert restored.data_provenance == original.data_provenance

    def test_round_trip_with_finding(self):
        original = _make_clean_output()
        original.findings.append(_make_finding())
        d = original.to_dict()
        restored = SkillOutput.from_dict(d)
        assert len(restored.findings) == 1
        assert restored.findings[0].id == "F-001"
        assert restored.findings[0].severity == Severity.HIGH

    def test_round_trip_with_artifact(self):
        original = _make_clean_output()
        original.artifacts.append(Artifact(
            filename="out.py",
            content="x = 1",
            artifact_type=ArtifactType.CODE,
            language="python",
            validated=True,
        ))
        restored = SkillOutput.from_dict(original.to_dict())
        assert len(restored.artifacts) == 1
        assert restored.artifacts[0].filename == "out.py"
        assert restored.artifacts[0].validated is True

    def test_from_dict_defaults(self):
        restored = SkillOutput.from_dict({})
        assert restored.agent_type == "unknown"
        assert restored.verdict == Verdict.NEEDS_REVIEW
        assert restored.confidence == 0.0
        assert restored.data_provenance == DataProvenance.UNKNOWN


# ---------------------------------------------------------------------------
# 3. Finding.to_dict() with all severity levels
# ---------------------------------------------------------------------------

class TestFindingToDict:
    @pytest.mark.parametrize("severity", list(Severity))
    def test_all_severities_serialise(self, severity):
        f = _make_finding(severity=severity)
        d = f.to_dict()
        assert d["severity"] == severity.value

    def test_finding_to_dict_keys(self):
        d = _make_finding().to_dict()
        for key in ("id", "severity", "category", "title", "location", "evidence", "recommendation", "confidence"):
            assert key in d

    def test_finding_confidence_default(self):
        f = _make_finding()
        assert f.confidence == 0.8
        assert f.to_dict()["confidence"] == 0.8


# ---------------------------------------------------------------------------
# 4. Artifact.to_dict() with content truncation
# ---------------------------------------------------------------------------

class TestArtifactToDict:
    def test_content_truncated_at_500(self):
        long_content = "x" * 1000
        a = Artifact(
            filename="big.py",
            content=long_content,
            artifact_type=ArtifactType.CODE,
        )
        d = a.to_dict()
        assert len(d["content"]) == 500

    def test_short_content_not_truncated(self):
        a = Artifact(filename="small.py", content="abc", artifact_type=ArtifactType.TEST)
        assert a.to_dict()["content"] == "abc"

    def test_artifact_type_serialised(self):
        for at in ArtifactType:
            a = Artifact(filename="f", content="", artifact_type=at)
            assert a.to_dict()["artifact_type"] == at.value


# ---------------------------------------------------------------------------
# 5. compute_overall_score() with known rubric weights
# ---------------------------------------------------------------------------

class TestComputeOverallScore:
    def test_builder_known_weights(self):
        # BUILDER rubric: syntax_validity=0.25, completeness=0.25
        # With only those two dimensions covered:
        # weighted_sum = 0.8*0.25 + 0.6*0.25 = 0.2 + 0.15 = 0.35
        # total_weight = 0.5
        # result = 0.35 / 0.5 = 0.7
        scores = {"syntax_validity": 0.8, "completeness": 0.6}
        result = compute_overall_score(scores, "BUILDER")
        assert abs(result - 0.7) < 0.001

    def test_all_rubric_dimensions_full_score(self):
        # All RESEARCHER dims at 1.0 should give 1.0
        rubric = SCORING_RUBRICS["RESEARCHER"]
        scores = {dim: 1.0 for dim in rubric}
        result = compute_overall_score(scores, "RESEARCHER")
        assert abs(result - 1.0) < 0.001

    def test_all_rubric_dimensions_zero_score(self):
        rubric = SCORING_RUBRICS["BUILDER"]
        scores = {dim: 0.0 for dim in rubric}
        result = compute_overall_score(scores, "BUILDER")
        assert result == 0.0


# ---------------------------------------------------------------------------
# 6. compute_overall_score() with unknown agent type (equal weights)
# ---------------------------------------------------------------------------

class TestComputeOverallScoreUnknownType:
    def test_unknown_agent_type_equal_weights(self):
        scores = {"a": 0.6, "b": 0.8, "c": 1.0}
        result = compute_overall_score(scores, "UNKNOWN_AGENT")
        expected = (0.6 + 0.8 + 1.0) / 3
        assert abs(result - expected) < 0.001

    def test_empty_scores_unknown_type(self):
        result = compute_overall_score({}, "UNKNOWN_AGENT")
        assert result == 0.0


# ---------------------------------------------------------------------------
# 7. self_check() — passes clean output
# ---------------------------------------------------------------------------

class TestSelfCheckPass:
    def test_clean_output_passes(self):
        out = _make_clean_output()
        result = self_check(out)
        assert result.self_check_passed is True
        assert result.self_check_issues == []

    def test_clean_with_finding_passes(self):
        out = _make_clean_output()
        out.findings.append(_make_finding())
        result = self_check(out)
        assert result.self_check_passed is True


# ---------------------------------------------------------------------------
# 8. self_check() — catches missing evidence
# ---------------------------------------------------------------------------

class TestSelfCheckMissingEvidence:
    def test_missing_evidence_flagged(self):
        out = _make_clean_output()
        out.findings.append(_make_finding(evidence=""))
        result = self_check(out)
        assert result.self_check_passed is False
        assert any("no evidence" in issue for issue in result.self_check_issues)

    def test_whitespace_only_evidence_flagged(self):
        out = _make_clean_output()
        out.findings.append(_make_finding(evidence="   "))
        result = self_check(out)
        assert result.self_check_passed is False

    def test_missing_location_flagged(self):
        out = _make_clean_output()
        out.findings.append(_make_finding(location=""))
        result = self_check(out)
        assert result.self_check_passed is False
        assert any("no location" in issue for issue in result.self_check_issues)

    def test_na_location_flagged(self):
        out = _make_clean_output()
        out.findings.append(_make_finding(location="N/A"))
        result = self_check(out)
        assert result.self_check_passed is False


# ---------------------------------------------------------------------------
# 9. self_check() — catches vague recommendations
# ---------------------------------------------------------------------------

class TestSelfCheckVagueRecommendations:
    @pytest.mark.parametrize("vague_phrase", [
        "consider",
        "might want to",
        "could potentially",
        "it may be helpful",
        "you should think about",
    ])
    def test_vague_phrase_flagged(self, vague_phrase):
        out = _make_clean_output()
        out.findings.append(_make_finding(recommendation=f"You should {vague_phrase} rewriting this"))
        result = self_check(out)
        assert result.self_check_passed is False
        assert any(vague_phrase in issue for issue in result.self_check_issues)

    def test_specific_recommendation_passes(self):
        out = _make_clean_output()
        out.findings.append(_make_finding(recommendation="Replace bare except with except ValueError as e"))
        result = self_check(out)
        assert result.self_check_passed is True


# ---------------------------------------------------------------------------
# 10. self_check() — catches extreme scores
# ---------------------------------------------------------------------------

class TestSelfCheckExtremeScores:
    def test_zero_score_flagged(self):
        out = _make_clean_output()
        out.scores["completeness"] = 0.0
        result = self_check(out)
        assert result.self_check_passed is False
        assert any("completeness" in issue for issue in result.self_check_issues)

    def test_one_score_flagged(self):
        out = _make_clean_output()
        out.scores["syntax_validity"] = 1.0
        result = self_check(out)
        assert result.self_check_passed is False

    def test_near_extreme_passes(self):
        out = SkillOutput(
            agent_type="BUILDER",
            task_summary="Build widget",
            verdict=Verdict.PASS,
            confidence=0.85,
            scores={"syntax_validity": 0.99, "completeness": 0.01},
        )
        result = self_check(out)
        assert result.self_check_passed is True

    def test_zero_confidence_flagged(self):
        out = SkillOutput(
            agent_type="BUILDER",
            task_summary="Build widget",
            verdict=Verdict.PASS,
            confidence=0.0,
        )
        result = self_check(out)
        assert result.self_check_passed is False
        assert any("Confidence" in issue for issue in result.self_check_issues)


# ---------------------------------------------------------------------------
# 11. SCORING_RUBRICS has all 8 agent types
# ---------------------------------------------------------------------------

class TestScoringRubrics:
    EXPECTED_AGENTS = {"RESEARCHER", "BUILDER", "TESTER", "ARCHITECT", "DOCUMENTER", "RESILIENCE", "META", "PLANNER"}

    def test_all_8_agent_types_present(self):
        assert set(SCORING_RUBRICS.keys()) == self.EXPECTED_AGENTS

    def test_each_rubric_has_dimensions(self):
        for agent_type, rubric in SCORING_RUBRICS.items():
            assert len(rubric) >= 3, f"{agent_type} rubric has fewer than 3 dimensions"

    def test_each_dimension_has_weight_and_description(self):
        for agent_type, rubric in SCORING_RUBRICS.items():
            for dim, config in rubric.items():
                assert "weight" in config, f"{agent_type}.{dim} missing 'weight'"
                assert "description" in config, f"{agent_type}.{dim} missing 'description'"
                assert isinstance(config["weight"], float), f"{agent_type}.{dim} weight not float"


# ---------------------------------------------------------------------------
# 12. All rubric weights sum to ~1.0
# ---------------------------------------------------------------------------

class TestRubricWeightSums:
    @pytest.mark.parametrize("agent_type", list(SCORING_RUBRICS.keys()))
    def test_weights_sum_to_one(self, agent_type):
        rubric = SCORING_RUBRICS[agent_type]
        total = sum(config["weight"] for config in rubric.values())
        assert abs(total - 1.0) < 0.001, (
            f"{agent_type} weights sum to {total:.4f}, expected 1.0"
        )
