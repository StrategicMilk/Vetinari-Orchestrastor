"""\nComprehensive tests for vetinari/goal_verifier.py\n\nCovers:\n- FeatureVerification dataclass\n- GoalVerificationReport dataclass: get_corrective_tasks(), to_dict()\n- GoalVerifier: __init__, verify, _verify_features, _check_avoid_list,\n_check_tests_present, _check_output_type, _llm_evaluation, _security_check\n- Singleton: get_goal_verifier()\n"""

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

import vetinari.goal_verifier as gv_module
from vetinari.goal_verifier import (
    FeatureVerification,
    GoalVerificationReport,
    GoalVerifier,
    get_goal_verifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evaluator_result(
    verdict="pass",
    quality_score=0.8,
    feature_checks=None,
    improvements=None,
    model_used="test-model",
):
    result = MagicMock()
    result.success = True
    result.output = {
        "verdict": verdict,
        "quality_score": quality_score,
        "feature_checks": feature_checks or [],
        "improvements": improvements or [],
        "model_used": model_used,
    }
    return result


def _make_security_result(findings=None, score=100):
    result = MagicMock()
    result.success = True
    result.output = {
        "findings": findings or [],
        "score": score,
    }
    return result


# ---------------------------------------------------------------------------
# FeatureVerification
# ---------------------------------------------------------------------------

class TestFeatureVerification:
    def test_required_fields(self):
        fv = FeatureVerification(
            feature="Authentication",
            implemented=True,
            confidence=0.9,
        )
        assert fv.feature == "Authentication"
        assert fv.implemented is True
        assert fv.confidence == pytest.approx(0.9)

    def test_optional_fields_defaults(self):
        fv = FeatureVerification(feature="Login", implemented=False, confidence=0.0)
        assert fv.evidence == ""
        assert fv.location == ""
        assert fv.severity == "major"

    def test_custom_optional_fields(self):
        fv = FeatureVerification(
            feature="API",
            implemented=True,
            confidence=0.75,
            evidence="Found in api.py",
            location="api.py:42",
            severity="minor",
        )
        assert fv.evidence == "Found in api.py"
        assert fv.location == "api.py:42"
        assert fv.severity == "minor"

    def test_confidence_can_be_zero(self):
        fv = FeatureVerification(feature="X", implemented=False, confidence=0.0)
        assert fv.confidence == 0.0

    def test_confidence_can_be_one(self):
        fv = FeatureVerification(feature="X", implemented=True, confidence=1.0)
        assert fv.confidence == 1.0


# ---------------------------------------------------------------------------
# GoalVerificationReport fields
# ---------------------------------------------------------------------------

class TestGoalVerificationReportFields:
    def test_required_fields(self):
        report = GoalVerificationReport(project_id="proj1", goal="Build an API")
        assert report.project_id == "proj1"
        assert report.goal == "Build an API"

    def test_created_at_is_iso(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        datetime.fromisoformat(report.created_at)

    def test_default_features_empty(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        assert report.features == []

    def test_default_security_passed(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        assert report.security_passed is True
        assert report.security_score == 1.0
        assert report.security_findings == []

    def test_default_quality_fields(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        assert report.quality_score == 0.0
        assert report.quality_issues == []

    def test_default_tests_not_present(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        assert report.tests_present is False
        assert report.test_coverage_estimate == 0.0

    def test_default_compliance(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        assert report.compliance_score == 0.0
        assert report.fully_compliant is False
        assert report.missing_features == []
        assert report.corrective_suggestions == []

    def test_default_metadata(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        assert report.evaluator_verdict == "inconclusive"
        assert report.model_used == ""

    def test_lists_are_independent_per_instance(self):
        r1 = GoalVerificationReport(project_id="p", goal="g")
        r2 = GoalVerificationReport(project_id="q", goal="h")
        r1.missing_features.append("Feature X")
        assert r2.missing_features == []


# ---------------------------------------------------------------------------
# GoalVerificationReport.get_corrective_tasks()
# ---------------------------------------------------------------------------

class TestGetCorrectiveTasks:
    def test_no_gaps_returns_empty_list(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        report.tests_present = True
        report.security_passed = True
        tasks = report.get_corrective_tasks()
        assert tasks == []

    def test_missing_feature_generates_implement_task(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        report.missing_features = ["User authentication"]
        tasks = report.get_corrective_tasks()
        types = [t["type"] for t in tasks]
        assert "implement_missing_feature" in types

    def test_missing_feature_task_content(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        report.missing_features = ["Dark mode"]
        tasks = report.get_corrective_tasks()
        task = next(t for t in tasks if t["type"] == "implement_missing_feature")
        assert "Dark mode" in task["description"]
        assert task["priority"] == "high"
        assert task["assigned_agent"] == "BUILDER"

    def test_multiple_missing_features_generate_multiple_tasks(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        report.missing_features = ["Feature A", "Feature B", "Feature C"]
        tasks = report.get_corrective_tasks()
        impl_tasks = [t for t in tasks if t["type"] == "implement_missing_feature"]
        assert len(impl_tasks) == 3

    def test_security_failure_generates_fix_task(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        report.security_passed = False
        report.security_findings = [{"id": "CVE-123", "severity": "critical"}]
        tasks = report.get_corrective_tasks()
        types = [t["type"] for t in tasks]
        assert "fix_security_issues" in types

    def test_security_task_has_critical_priority(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        report.security_passed = False
        report.security_findings = [{"id": "f1"}]
        tasks = report.get_corrective_tasks()
        sec_task = next(t for t in tasks if t["type"] == "fix_security_issues")
        assert sec_task["priority"] == "critical"
        assert sec_task["assigned_agent"] == "SECURITY_AUDITOR"

    def test_security_task_includes_findings(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        report.security_passed = False
        findings = [{"id": "f1", "severity": "critical"}]
        report.security_findings = findings
        tasks = report.get_corrective_tasks()
        sec_task = next(t for t in tasks if t["type"] == "fix_security_issues")
        assert sec_task["details"] == findings

    def test_security_passed_no_task_generated(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        report.security_passed = True
        report.security_findings = []
        tasks = report.get_corrective_tasks()
        types = [t["type"] for t in tasks]
        assert "fix_security_issues" not in types

    def test_security_failed_no_findings_no_task(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        report.security_passed = False
        report.security_findings = []
        tasks = report.get_corrective_tasks()
        types = [t["type"] for t in tasks]
        assert "fix_security_issues" not in types

    def test_missing_tests_generates_add_tests_task(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        report.tests_present = False
        tasks = report.get_corrective_tasks()
        types = [t["type"] for t in tasks]
        assert "add_tests" in types

    def test_add_tests_task_fields(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        report.tests_present = False
        tasks = report.get_corrective_tasks()
        task = next(t for t in tasks if t["type"] == "add_tests")
        assert task["priority"] == "high"
        assert task["assigned_agent"] == "TEST_AUTOMATION"

    def test_tests_present_no_add_tests_task(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        report.tests_present = True
        tasks = report.get_corrective_tasks()
        types = [t["type"] for t in tasks]
        assert "add_tests" not in types

    def test_corrective_suggestions_capped_at_3(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        report.corrective_suggestions = [
            "Fix linting",
            "Add docstrings",
            "Improve error handling",
            "Increase test coverage",
        ]
        tasks = report.get_corrective_tasks()
        quality_tasks = [t for t in tasks if t["type"] == "quality_improvement"]
        assert len(quality_tasks) == 3

    def test_quality_improvement_task_fields(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        report.corrective_suggestions = ["Add type hints"]
        tasks = report.get_corrective_tasks()
        task = next(t for t in tasks if t["type"] == "quality_improvement")
        assert "Add type hints" in task["description"]
        assert task["priority"] == "medium"
        assert task["assigned_agent"] == "EVALUATOR"

    def test_combined_gaps_generates_all_task_types(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        report.missing_features = ["Feature X"]
        report.security_passed = False
        report.security_findings = [{"id": "f1"}]
        report.tests_present = False
        report.corrective_suggestions = ["Improve naming"]
        tasks = report.get_corrective_tasks()
        types = {t["type"] for t in tasks}
        assert "implement_missing_feature" in types
        assert "fix_security_issues" in types
        assert "add_tests" in types
        assert "quality_improvement" in types


# ---------------------------------------------------------------------------
# GoalVerificationReport.to_dict()
# ---------------------------------------------------------------------------

class TestGoalVerificationReportToDict:
    def test_returns_dict(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        d = report.to_dict()
        assert isinstance(d, dict)

    def test_expected_keys_present(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        d = report.to_dict()
        expected = {
            "project_id", "goal", "created_at", "features",
            "security_passed", "security_score", "security_findings",
            "quality_score", "quality_issues", "tests_present",
            "test_coverage_estimate", "compliance_score", "fully_compliant",
            "missing_features", "corrective_suggestions",
            "evaluator_verdict", "model_used",
        }
        assert expected.issubset(set(d.keys()))

    def test_features_serialized_as_list_of_dicts(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        report.features = [
            FeatureVerification("Auth", True, 0.9, "Found in auth.py", "auth.py", "minor")
        ]
        d = report.to_dict()
        assert len(d["features"]) == 1
        feat = d["features"][0]
        assert feat["feature"] == "Auth"
        assert feat["implemented"] is True
        assert feat["confidence"] == pytest.approx(0.9)
        assert feat["evidence"] == "Found in auth.py"
        assert feat["location"] == "auth.py"
        assert feat["severity"] == "minor"

    def test_scalar_fields_match(self):
        report = GoalVerificationReport(project_id="proj", goal="Build API")
        report.quality_score = 0.85
        report.compliance_score = 0.9
        report.fully_compliant = True
        report.tests_present = True
        report.evaluator_verdict = "pass"
        d = report.to_dict()
        assert d["project_id"] == "proj"
        assert d["goal"] == "Build API"
        assert d["quality_score"] == pytest.approx(0.85)
        assert d["compliance_score"] == pytest.approx(0.9)
        assert d["fully_compliant"] is True
        assert d["tests_present"] is True
        assert d["evaluator_verdict"] == "pass"

    def test_empty_features_serialized(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        d = report.to_dict()
        assert d["features"] == []

    def test_missing_features_list_serialized(self):
        report = GoalVerificationReport(project_id="p", goal="g")
        report.missing_features = ["Auth", "API"]
        d = report.to_dict()
        assert d["missing_features"] == ["Auth", "API"]


# ---------------------------------------------------------------------------
# GoalVerifier.__init__
# ---------------------------------------------------------------------------

class TestGoalVerifierInit:
    def test_default_threshold(self):
        verifier = GoalVerifier()
        assert verifier._threshold == pytest.approx(0.75)

    def test_custom_threshold(self):
        verifier = GoalVerifier(quality_threshold=0.9)
        assert verifier._threshold == pytest.approx(0.9)

    def test_low_threshold(self):
        verifier = GoalVerifier(quality_threshold=0.1)
        assert verifier._threshold == pytest.approx(0.1)

    def test_zero_threshold(self):
        verifier = GoalVerifier(quality_threshold=0.0)
        assert verifier._threshold == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# GoalVerifier._verify_features
# ---------------------------------------------------------------------------

class TestVerifyFeatures:
    @pytest.fixture
    def verifier(self):
        return GoalVerifier()

    def test_empty_features_returns_empty(self, verifier):
        result = verifier._verify_features("goal", [], "some output", [])
        assert result == []

    def test_feature_found_in_output(self, verifier):
        output = "The authentication system is implemented with JWT tokens."
        results = verifier._verify_features(
            "goal", ["authentication system"], output, []
        )
        assert len(results) == 1
        assert results[0].implemented is True

    def test_feature_not_in_output(self, verifier):
        output = "This is about database migration only."
        results = verifier._verify_features(
            "goal", ["authentication token validation"], output, []
        )
        assert len(results) == 1
        assert results[0].implemented is False

    def test_returns_list_of_feature_verifications(self, verifier):
        results = verifier._verify_features("goal", ["login"], "login page here", [])
        for r in results:
            assert isinstance(r, FeatureVerification)

    def test_confidence_between_zero_and_one(self, verifier):
        results = verifier._verify_features(
            "goal", ["some feature"], "some feature output", []
        )
        for r in results:
            assert 0.0 <= r.confidence <= 1.0

    def test_multiple_features_processed(self, verifier):
        output = "authentication and authorization are implemented"
        features = ["authentication", "authorization", "caching"]
        results = verifier._verify_features("goal", features, output, [])
        assert len(results) == 3

    def test_task_outputs_searched_too(self, verifier):
        final_output = "main deliverable"
        task_outputs = [{"output": "authentication implemented here"}]
        results = verifier._verify_features(
            "goal", ["authentication"], final_output, task_outputs
        )
        assert results[0].implemented is True

    def test_blank_feature_skipped(self, verifier):
        results = verifier._verify_features(
            "goal", ["", "  ", "real feature"], "real feature is here", []
        )
        assert len(results) == 1

    def test_evidence_message_populated(self, verifier):
        output = "login authentication system present"
        results = verifier._verify_features("goal", ["login authentication"], output, [])
        assert results[0].evidence != ""

    def test_severity_defaults_to_major(self, verifier):
        output = "authentication present"
        results = verifier._verify_features("goal", ["authentication"], output, [])
        assert results[0].severity == "major"

    def test_short_words_filtered_from_keywords(self, verifier):
        output = "no matches at all"
        results = verifier._verify_features("goal", ["is it so"], output, [])
        assert len(results) == 1
        assert results[0].implemented is False

    def test_high_confidence_when_all_keywords_present(self, verifier):
        output = "pagination sorting filtering search results"
        results = verifier._verify_features(
            "goal", ["pagination sorting filtering search results"], output, []
        )
        assert results[0].confidence >= 0.5


# ---------------------------------------------------------------------------
# GoalVerifier._check_avoid_list
# ---------------------------------------------------------------------------

class TestCheckAvoidList:
    @pytest.fixture
    def verifier(self):
        return GoalVerifier()

    def test_empty_avoid_list_returns_empty(self, verifier):
        violations = verifier._check_avoid_list([], "any output")
        assert violations == []

    def test_item_present_in_output_is_violation(self, verifier):
        violations = verifier._check_avoid_list(
            ["jQuery"], "We used jQuery for the frontend."
        )
        assert "jQuery" in violations

    def test_item_absent_not_violation(self, verifier):
        violations = verifier._check_avoid_list(
            ["jQuery"], "We used vanilla JS only."
        )
        assert violations == []

    def test_case_insensitive_matching(self, verifier):
        violations = verifier._check_avoid_list(
            ["jquery"], "We used JQUERY for the frontend."
        )
        assert "jquery" in violations

    def test_multiple_violations(self, verifier):
        violations = verifier._check_avoid_list(
            ["deprecated_api", "legacy_code"],
            "uses deprecated_api and legacy_code throughout"
        )
        assert len(violations) == 2

    def test_blank_item_skipped(self, verifier):
        violations = verifier._check_avoid_list(["", "  "], "some output")
        assert violations == []

    def test_partial_substring_match(self, verifier):
        violations = verifier._check_avoid_list(["eval"], "we evaluate the output")
        assert "eval" in violations

    def test_returns_list_type(self, verifier):
        violations = verifier._check_avoid_list(["bad"], "no bad words here... wait")
        assert isinstance(violations, list)

    def test_only_present_items_returned(self, verifier):
        violations = verifier._check_avoid_list(
            ["present", "absent"], "only present is here"
        )
        assert "present" in violations
        assert "absent" not in violations


# ---------------------------------------------------------------------------
# GoalVerifier._check_tests_present
# ---------------------------------------------------------------------------

class TestCheckTestsPresent:
    @pytest.fixture
    def verifier(self):
        return GoalVerifier()

    def test_def_test_present(self, verifier):
        assert verifier._check_tests_present("def test_login():", []) is True

    def test_import_pytest_present(self, verifier):
        assert verifier._check_tests_present("import pytest\n", []) is True

    def test_import_unittest_present(self, verifier):
        assert verifier._check_tests_present("import unittest", []) is True

    def test_class_test_present(self, verifier):
        assert verifier._check_tests_present("class TestMyFeature:", []) is True

    def test_assert_statement_present(self, verifier):
        assert verifier._check_tests_present("assert x == 1", []) is True

    def test_pytest_decorator_present(self, verifier):
        assert verifier._check_tests_present("@pytest.mark.parametrize", []) is True

    def test_js_describe_present(self, verifier):
        assert verifier._check_tests_present("describe('suite', () => {", []) is True

    def test_js_it_present(self, verifier):
        assert verifier._check_tests_present("it('should work', () => {", []) is True

    def test_go_func_test_present(self, verifier):
        assert verifier._check_tests_present("func TestMyFunc(t *testing.T)", []) is True

    def test_go_testing_t_present(self, verifier):
        assert verifier._check_tests_present("testing.T", []) is True

    def test_no_tests_in_output(self, verifier):
        assert verifier._check_tests_present("just some code x = 1", []) is False

    def test_test_in_task_outputs(self, verifier):
        task_outputs = [{"output": "import pytest\ndef test_foo(): pass"}]
        assert verifier._check_tests_present("main output", task_outputs) is True

    def test_empty_output_no_tests(self, verifier):
        assert verifier._check_tests_present("", []) is False

    def test_returns_bool(self, verifier):
        result = verifier._check_tests_present("def test_x(): pass", [])
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# GoalVerifier._check_output_type
# ---------------------------------------------------------------------------

class TestCheckOutputType:
    @pytest.fixture
    def verifier(self):
        return GoalVerifier()

    def test_code_type_def_detected(self, verifier):
        assert verifier._check_output_type("code", "def main():\npass\n", []) is True

    def test_code_type_class_detected(self, verifier):
        assert verifier._check_output_type("code", "class MyClass:\npass", []) is True

    def test_code_type_import_detected(self, verifier):
        assert verifier._check_output_type("code", "import os\nimport sys", []) is True

    def test_tests_type_detected(self, verifier):
        assert verifier._check_output_type("tests", "import pytest\ndef test_foo(): pass", []) is True

    def test_docs_hash_detected(self, verifier):
        assert verifier._check_output_type("docs", "# Header\n## Subheader", []) is True

    def test_docs_readme_detected(self, verifier):
        assert verifier._check_output_type("docs", "See README for details", []) is True

    def test_ci_yml_detected(self, verifier):
        assert verifier._check_output_type("ci", "ci-pipeline.yml configuration", []) is True

    def test_docker_from_detected(self, verifier):
        assert verifier._check_output_type("docker", "FROM python:3.11-slim", []) is True

    def test_docker_dockerfile_detected(self, verifier):
        assert verifier._check_output_type("docker", "See Dockerfile for build", []) is True

    def test_assets_svg_detected(self, verifier):
        assert verifier._check_output_type("assets", "<svg width='100'>", []) is True

    def test_unknown_type_falls_back_to_substring(self, verifier):
        assert verifier._check_output_type("custom_marker", "contains custom_marker text", []) is True

    def test_type_not_found(self, verifier):
        assert verifier._check_output_type("docker", "just plain text", []) is False

    def test_case_insensitive_type_key(self, verifier):
        assert verifier._check_output_type("CODE", "def my_function(): pass", []) is True

    def test_task_outputs_searched(self, verifier):
        task_outputs = [{"output": "FROM ubuntu:22.04"}]
        assert verifier._check_output_type("docker", "", task_outputs) is True

    def test_returns_bool(self, verifier):
        result = verifier._check_output_type("code", "def foo(): pass", [])
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# GoalVerifier._llm_evaluation
# ---------------------------------------------------------------------------

class TestLlmEvaluation:
    @pytest.fixture
    def verifier(self):
        return GoalVerifier()

    def test_returns_dict_on_success(self, verifier):
        mock_evaluator = MagicMock()
        mock_evaluator.execute.return_value = _make_evaluator_result()
        with patch("vetinari.agents.evaluator_agent.get_evaluator_agent",
                   return_value=mock_evaluator):
            with patch("vetinari.agents.contracts.AgentTask", return_value=MagicMock()):
                with patch("vetinari.agents.contracts.AgentType"):
                    result = verifier._llm_evaluation(
                        "Build an API", "output text", ["feature1"], []
                    )
        assert isinstance(result, dict)

    def test_returns_none_on_import_error(self, verifier):
        with patch("vetinari.agents.evaluator_agent.get_evaluator_agent",
                   side_effect=ImportError("no module")):
            result = verifier._llm_evaluation("goal", "output", [], [])
        assert result is None

    def test_returns_none_when_result_not_success(self, verifier):
        mock_evaluator = MagicMock()
        mock_evaluator.execute.return_value = MagicMock(success=False, output=None)
        with patch("vetinari.agents.evaluator_agent.get_evaluator_agent",
                   return_value=mock_evaluator):
            with patch("vetinari.agents.contracts.AgentTask", return_value=MagicMock()):
                with patch("vetinari.agents.contracts.AgentType"):
                    result = verifier._llm_evaluation("goal", "output", [], [])
        assert result is None

    def test_returns_none_when_output_not_dict(self, verifier):
        mock_evaluator = MagicMock()
        mock_evaluator.execute.return_value = MagicMock(success=True, output="just a string")
        with patch("vetinari.agents.evaluator_agent.get_evaluator_agent",
                   return_value=mock_evaluator):
            with patch("vetinari.agents.contracts.AgentTask", return_value=MagicMock()):
                with patch("vetinari.agents.contracts.AgentType"):
                    result = verifier._llm_evaluation("goal", "output", [], [])
        assert result is None

    def test_returns_none_on_general_exception(self, verifier):
        with patch("vetinari.agents.evaluator_agent.get_evaluator_agent",
                   side_effect=RuntimeError("unexpected")):
            result = verifier._llm_evaluation("goal", "output", [], [])
        assert result is None


# ---------------------------------------------------------------------------
# GoalVerifier._security_check
# ---------------------------------------------------------------------------

class TestSecurityCheck:
    @pytest.fixture
    def verifier(self):
        return GoalVerifier()

    def test_returns_three_tuple(self, verifier):
        mock_auditor = MagicMock()
        mock_auditor.execute.return_value = _make_security_result(findings=[], score=100)
        with patch("vetinari.agents.security_auditor_agent.get_security_auditor_agent",
                   return_value=mock_auditor):
            with patch("vetinari.agents.contracts.AgentTask", return_value=MagicMock()):
                with patch("vetinari.agents.contracts.AgentType"):
                    result = verifier._security_check("output", [])
        assert len(result) == 3

    def test_clean_output_passes(self, verifier):
        mock_auditor = MagicMock()
        mock_auditor.execute.return_value = _make_security_result(findings=[], score=100)
        with patch("vetinari.agents.security_auditor_agent.get_security_auditor_agent",
                   return_value=mock_auditor):
            with patch("vetinari.agents.contracts.AgentTask", return_value=MagicMock()):
                with patch("vetinari.agents.contracts.AgentType"):
                    passed, findings, score = verifier._security_check("safe code", [])
        assert passed is True
        assert findings == []

    def test_critical_finding_fails(self, verifier):
        mock_auditor = MagicMock()
        mock_auditor.execute.return_value = _make_security_result(
            findings=[{"severity": "critical", "id": "SQL_INJECTION"}], score=40,
        )
        with patch("vetinari.agents.security_auditor_agent.get_security_auditor_agent",
                   return_value=mock_auditor):
            with patch("vetinari.agents.contracts.AgentTask", return_value=MagicMock()):
                with patch("vetinari.agents.contracts.AgentType"):
                    passed, findings, score = verifier._security_check("bad code", [])
        assert passed is False

    def test_high_finding_fails(self, verifier):
        mock_auditor = MagicMock()
        mock_auditor.execute.return_value = _make_security_result(
            findings=[{"severity": "high", "id": "XSS"}], score=60,
        )
        with patch("vetinari.agents.security_auditor_agent.get_security_auditor_agent",
                   return_value=mock_auditor):
            with patch("vetinari.agents.contracts.AgentTask", return_value=MagicMock()):
                with patch("vetinari.agents.contracts.AgentType"):
                    passed, findings, score = verifier._security_check("code", [])
        assert passed is False

    def test_low_severity_does_not_fail(self, verifier):
        mock_auditor = MagicMock()
        mock_auditor.execute.return_value = _make_security_result(
            findings=[{"severity": "low", "id": "INFO_LEAK"}], score=90,
        )
        with patch("vetinari.agents.security_auditor_agent.get_security_auditor_agent",
                   return_value=mock_auditor):
            with patch("vetinari.agents.contracts.AgentTask", return_value=MagicMock()):
                with patch("vetinari.agents.contracts.AgentType"):
                    passed, findings, score = verifier._security_check("code", [])
        assert passed is True

    def test_import_error_returns_safe_defaults(self, verifier):
        with patch("vetinari.agents.security_auditor_agent.get_security_auditor_agent",
                   side_effect=ImportError("no module")):
            passed, findings, score = verifier._security_check("output", [])
        assert passed is True
        assert findings == []
        assert score == 1.0

    def test_score_normalized_from_100(self, verifier):
        mock_auditor = MagicMock()
        mock_auditor.execute.return_value = _make_security_result(findings=[], score=75)
        with patch("vetinari.agents.security_auditor_agent.get_security_auditor_agent",
                   return_value=mock_auditor):
            with patch("vetinari.agents.contracts.AgentTask", return_value=MagicMock()):
                with patch("vetinari.agents.contracts.AgentType"):
                    passed, findings, score = verifier._security_check("code", [])
        assert score == pytest.approx(0.75)

    def test_medium_severity_does_not_fail(self, verifier):
        mock_auditor = MagicMock()
        mock_auditor.execute.return_value = _make_security_result(
            findings=[{"severity": "medium", "id": "CSRF"}], score=80,
        )
        with patch("vetinari.agents.security_auditor_agent.get_security_auditor_agent",
                   return_value=mock_auditor):
            with patch("vetinari.agents.contracts.AgentTask", return_value=MagicMock()):
                with patch("vetinari.agents.contracts.AgentType"):
                    passed, findings, score = verifier._security_check("code", [])
        assert passed is True


# ---------------------------------------------------------------------------
# GoalVerifier.verify -- integration-style (agents mocked at method level)
# ---------------------------------------------------------------------------

class TestGoalVerifierVerify:
    @pytest.fixture
    def verifier(self):
        return GoalVerifier(quality_threshold=0.75)

    def _run_verify(self, verifier, **kwargs):
        defaults = {
            "project_id": "test_proj",
            "goal": "Build a REST API",
            "final_output": "def get_users(): pass\nimport pytest\ndef test_users(): assert True",
            "required_features": [],
            "things_to_avoid": [],
            "task_outputs": [],
            "expected_outputs": [],
        }
        defaults.update(kwargs)
        with patch.object(
            verifier, "_llm_evaluation",
            return_value={
                "verdict": "pass",
                "quality_score": 0.85,
                "feature_checks": [],
                "improvements": [],
                "model_used": "mock-model",
            }
        ):
            with patch.object(verifier, "_security_check", return_value=(True, [], 1.0)):
                return verifier.verify(**defaults)

    def test_returns_report_instance(self, verifier):
        report = self._run_verify(verifier)
        assert isinstance(report, GoalVerificationReport)

    def test_project_id_in_report(self, verifier):
        report = self._run_verify(verifier, project_id="my_project")
        assert report.project_id == "my_project"

    def test_goal_in_report(self, verifier):
        report = self._run_verify(verifier, goal="Create a CLI tool")
        assert report.goal == "Create a CLI tool"

    def test_tests_present_detected(self, verifier):
        report = self._run_verify(
            verifier, final_output="import pytest\ndef test_foo(): pass"
        )
        assert report.tests_present is True

    def test_tests_not_present(self, verifier):
        report = self._run_verify(verifier, final_output="x = 1\ny = 2\n")
        assert report.tests_present is False

    def test_feature_verification_runs(self, verifier):
        report = self._run_verify(
            verifier,
            final_output="authentication and authorization implemented",
            required_features=["authentication", "authorization"],
        )
        assert len(report.features) == 2

    def test_avoid_violation_adds_feature(self, verifier):
        report = self._run_verify(
            verifier,
            final_output="we use deprecated_stuff here",
            things_to_avoid=["deprecated_stuff"],
        )
        avoid_features = [f for f in report.features if "AVOID" in f.feature]
        assert len(avoid_features) == 1

    def test_avoid_list_no_violation(self, verifier):
        report = self._run_verify(
            verifier,
            final_output="clean code only",
            things_to_avoid=["deprecated_stuff"],
        )
        avoid_features = [f for f in report.features if "AVOID" in f.feature]
        assert len(avoid_features) == 0

    def test_quality_score_from_llm(self, verifier):
        report = self._run_verify(verifier)
        assert report.quality_score == pytest.approx(0.85)

    def test_evaluator_verdict_set(self, verifier):
        report = self._run_verify(verifier)
        assert report.evaluator_verdict == "pass"

    def test_security_passed(self, verifier):
        report = self._run_verify(verifier)
        assert report.security_passed is True

    def test_compliance_score_is_float(self, verifier):
        report = self._run_verify(verifier)
        assert isinstance(report.compliance_score, float)

    def test_compliance_score_between_0_and_1(self, verifier):
        report = self._run_verify(verifier)
        assert 0.0 <= report.compliance_score <= 1.0

    def test_missing_features_is_list(self, verifier):
        report = self._run_verify(
            verifier,
            final_output="no relevant content at all whatsoever",
            required_features=["authentication token validation system xyz"],
        )
        assert isinstance(report.missing_features, list)

    def test_fully_compliant_is_bool(self, verifier):
        report = self._run_verify(verifier)
        assert isinstance(report.fully_compliant, bool)

    def test_not_compliant_when_security_fails(self, verifier):
        with patch.object(
            verifier, "_llm_evaluation",
            return_value={"verdict": "pass", "quality_score": 0.9,
                          "feature_checks": [], "improvements": [], "model_used": "m"}
        ):
            with patch.object(
                verifier, "_security_check",
                return_value=(False, [{"severity": "critical"}], 0.0)
            ):
                report = verifier.verify(
                    project_id="p", goal="g", final_output="code with issues"
                )
        assert report.fully_compliant is False
        assert report.security_passed is False

    def test_llm_failure_uses_default_quality_score(self, verifier):
        # P2.4: LLM failure must default to 0.3 (failing), not 0.7 (passing).
        with patch.object(verifier, "_llm_evaluation", side_effect=Exception("fail")):
            with patch.object(verifier, "_security_check", return_value=(True, [], 1.0)):
                report = verifier.verify(project_id="p", goal="g", final_output="code")
        assert report.quality_score == pytest.approx(0.3)

    def test_expected_outputs_missing_adds_quality_issue(self, verifier):
        report = self._run_verify(
            verifier,
            final_output="some text without docker patterns",
            expected_outputs=["docker"],
        )
        assert any("docker" in issue.lower() for issue in report.quality_issues)

    def test_expected_outputs_found_no_quality_issue(self, verifier):
        report = self._run_verify(
            verifier,
            final_output="FROM python:3.11-slim\n",
            expected_outputs=["docker"],
        )
        docker_issues = [i for i in report.quality_issues if "docker" in i.lower()]
        assert docker_issues == []

    def test_null_task_outputs_handled_gracefully(self, verifier):
        with patch.object(
            verifier, "_llm_evaluation",
            return_value={"verdict": "pass", "quality_score": 0.8,
                          "feature_checks": [], "improvements": [], "model_used": "m"}
        ):
            with patch.object(verifier, "_security_check", return_value=(True, [], 1.0)):
                report = verifier.verify(
                    project_id="p", goal="g", final_output="code", task_outputs=None
                )
        assert isinstance(report, GoalVerificationReport)

    def test_corrective_suggestions_from_llm(self, verifier):
        with patch.object(
            verifier, "_llm_evaluation",
            return_value={
                "verdict": "partial",
                "quality_score": 0.6,
                "feature_checks": [],
                "improvements": ["Add error handling", "Write tests"],
                "model_used": "m",
            }
        ):
            with patch.object(verifier, "_security_check", return_value=(True, [], 1.0)):
                report = verifier.verify(project_id="p", goal="g", final_output="code")
        assert "Add error handling" in report.corrective_suggestions

    def test_model_used_set_from_llm(self, verifier):
        with patch.object(
            verifier, "_llm_evaluation",
            return_value={"verdict": "pass", "quality_score": 0.8,
                          "feature_checks": [], "improvements": [], "model_used": "gpt-4"}
        ):
            with patch.object(verifier, "_security_check", return_value=(True, [], 1.0)):
                report = verifier.verify(project_id="p", goal="g", final_output="x")
        assert report.model_used == "gpt-4"

    def test_llm_feature_checks_merge_with_heuristic(self, verifier):
        with patch.object(
            verifier, "_llm_evaluation",
            return_value={
                "verdict": "pass",
                "quality_score": 0.9,
                "feature_checks": [
                    {"feature": "authentication", "implemented": True,
                     "confidence": 0.95, "evidence": "LLM found it"},
                ],
                "improvements": [],
                "model_used": "m",
            }
        ):
            with patch.object(verifier, "_security_check", return_value=(True, [], 1.0)):
                report = verifier.verify(
                    project_id="p", goal="g",
                    final_output="authentication is here",
                    required_features=["authentication"],
                )
        feat = next((f for f in report.features if f.feature == "authentication"), None)
        assert feat is not None
        assert feat.confidence >= 0.5


# ---------------------------------------------------------------------------
# Compliance score formula
# ---------------------------------------------------------------------------

class TestComplianceScoreFormula:
    def test_perfect_score_when_everything_passes(self):
        # P2.4: feature_score defaults to 0.5 when no features are provided.
        # Supply a matching feature so feature_score reaches 1.0 for a perfect result.
        verifier = GoalVerifier(quality_threshold=0.5)
        with patch.object(
            verifier, "_llm_evaluation",
            return_value={"verdict": "pass", "quality_score": 1.0,
                          "feature_checks": [], "improvements": [], "model_used": "m"}
        ):
            with patch.object(verifier, "_security_check", return_value=(True, [], 1.0)):
                report = verifier.verify(
                    project_id="p", goal="g",
                    final_output="import pytest\ndef test_foo(): pass",
                    required_features=["pytest test"],
                )
        assert report.compliance_score == pytest.approx(1.0, abs=0.01)

    def test_security_failure_deducts_from_score(self):
        verifier = GoalVerifier()
        with patch.object(
            verifier, "_llm_evaluation",
            return_value={"verdict": "pass", "quality_score": 1.0,
                          "feature_checks": [], "improvements": [], "model_used": "m"}
        ):
            with patch.object(
                verifier, "_security_check",
                return_value=(False, [{"severity": "critical"}], 0.5)
            ):
                report = verifier.verify(
                    project_id="p", goal="g",
                    final_output="import pytest\ndef test_a(): pass",
                )
        assert report.compliance_score < 1.0

    def test_fully_compliant_requires_security_pass(self):
        verifier = GoalVerifier(quality_threshold=0.1)
        with patch.object(
            verifier, "_llm_evaluation",
            return_value={"verdict": "pass", "quality_score": 1.0,
                          "feature_checks": [], "improvements": [], "model_used": "m"}
        ):
            with patch.object(
                verifier, "_security_check",
                return_value=(False, [{"severity": "high"}], 0.0)
            ):
                report = verifier.verify(project_id="p", goal="g", final_output="code")
        assert report.fully_compliant is False

    def test_missing_features_prevent_full_compliance(self):
        verifier = GoalVerifier(quality_threshold=0.0)
        with patch.object(
            verifier, "_llm_evaluation",
            return_value={"verdict": "pass", "quality_score": 1.0,
                          "feature_checks": [], "improvements": [], "model_used": "m"}
        ):
            with patch.object(verifier, "_security_check", return_value=(True, [], 1.0)):
                report = verifier.verify(
                    project_id="p", goal="g",
                    final_output="zzz no matching keywords anywhere",
                    required_features=["authentication token validation system xyz abc"],
                )
        if report.missing_features:
            assert report.fully_compliant is False

    def test_no_tests_reduces_score_vs_tests_present(self):
        verifier = GoalVerifier()
        with patch.object(
            verifier, "_llm_evaluation",
            return_value={"verdict": "pass", "quality_score": 1.0,
                          "feature_checks": [], "improvements": [], "model_used": "m"}
        ):
            with patch.object(verifier, "_security_check", return_value=(True, [], 1.0)):
                r_no_tests = verifier.verify(
                    project_id="p", goal="g", final_output="pure code no tests"
                )
        with patch.object(
            verifier, "_llm_evaluation",
            return_value={"verdict": "pass", "quality_score": 1.0,
                          "feature_checks": [], "improvements": [], "model_used": "m"}
        ):
            with patch.object(verifier, "_security_check", return_value=(True, [], 1.0)):
                r_with_tests = verifier.verify(
                    project_id="p", goal="g",
                    final_output="import pytest\ndef test_x(): pass"
                )
        assert r_with_tests.compliance_score >= r_no_tests.compliance_score


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestGetGoalVerifier:
    def setup_method(self):
        gv_module._goal_verifier = None

    def teardown_method(self):
        gv_module._goal_verifier = None

    def test_returns_goal_verifier_instance(self):
        verifier = get_goal_verifier()
        assert isinstance(verifier, GoalVerifier)

    def test_returns_same_instance_on_repeated_calls(self):
        v1 = get_goal_verifier()
        v2 = get_goal_verifier()
        assert v1 is v2

    def test_creates_if_none(self):
        gv_module._goal_verifier = None
        verifier = get_goal_verifier()
        assert verifier is not None
        assert gv_module._goal_verifier is verifier

    def test_reset_creates_new_instance(self):
        v1 = get_goal_verifier()
        gv_module._goal_verifier = None
        v2 = get_goal_verifier()
        assert isinstance(v2, GoalVerifier)

    def test_default_threshold_on_singleton(self):
        verifier = get_goal_verifier()
        assert verifier._threshold == pytest.approx(0.75)
