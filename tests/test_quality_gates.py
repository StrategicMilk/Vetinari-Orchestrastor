"""Tests for the agentic verification / quality gates system (Task 26)."""

import pytest

from vetinari.validation.quality_gates import (
    GateCheckResult,
    GateResult,
    QualityGateConfig,
    QualityGateRunner,
    VerificationMode,
)


# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------

class TestVerificationMode:
    def test_mode_values(self):
        assert VerificationMode.VERIFY_QUALITY.value == "verify_quality"
        assert VerificationMode.SECURITY.value == "security"
        assert VerificationMode.VERIFY_COVERAGE.value == "verify_coverage"
        assert VerificationMode.VERIFY_ARCHITECTURE.value == "verify_architecture"

    def test_gate_result_values(self):
        assert GateResult.PASSED.value == "passed"
        assert GateResult.FAILED.value == "failed"
        assert GateResult.WARNING.value == "warning"


# ---------------------------------------------------------------------------
# QualityGateConfig
# ---------------------------------------------------------------------------

class TestQualityGateConfig:
    def test_defaults(self):
        config = QualityGateConfig("test_gate", VerificationMode.VERIFY_QUALITY)
        assert config.name == "test_gate"
        assert config.required is True
        assert config.min_score == 0.6
        assert config.timeout_seconds == 60
        assert config.auto_fix is False

    def test_custom_config(self):
        config = QualityGateConfig(
            "strict_gate",
            VerificationMode.SECURITY,
            required=True,
            min_score=0.9,
            timeout_seconds=120,
            auto_fix=True,
        )
        assert config.min_score == 0.9
        assert config.auto_fix is True


# ---------------------------------------------------------------------------
# GateCheckResult
# ---------------------------------------------------------------------------

class TestGateCheckResult:
    def test_creation(self):
        result = GateCheckResult(
            gate_name="test",
            mode=VerificationMode.SECURITY,
            result=GateResult.PASSED,
            score=0.95,
        )
        assert result.gate_name == "test"
        assert result.score == 0.95
        assert result.issues == []

    def test_to_dict(self):
        result = GateCheckResult(
            gate_name="g1",
            mode=VerificationMode.VERIFY_QUALITY,
            result=GateResult.PASSED,
            score=0.85,
            issues=[{"severity": "info", "message": "ok"}],
            suggestions=["keep going"],
        )
        d = result.to_dict()
        assert d["gate_name"] == "g1"
        assert d["mode"] == "verify_quality"
        assert d["result"] == "passed"
        assert d["score"] == 0.85
        assert len(d["issues"]) == 1


# ---------------------------------------------------------------------------
# QualityGateRunner — pipeline gates
# ---------------------------------------------------------------------------

class TestQualityGateRunner:
    def test_default_gates_exist(self):
        runner = QualityGateRunner()
        for stage in ["post_planning", "post_execution", "post_testing", "pre_assembly"]:
            gates = runner.get_gates_for_stage(stage)
            assert len(gates) > 0, f"No gates for {stage}"

    def test_unknown_stage_returns_empty(self):
        runner = QualityGateRunner()
        results = runner.run_gate("nonexistent_stage", {})
        assert results == []

    def test_custom_gates(self):
        custom = {
            "custom_stage": [
                QualityGateConfig("my_gate", VerificationMode.VERIFY_QUALITY, min_score=0.5),
            ]
        }
        runner = QualityGateRunner(custom_gates=custom)
        gates = runner.get_gates_for_stage("custom_stage")
        assert len(gates) == 1
        assert gates[0].name == "my_gate"


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

class TestQualityCheck:
    def test_quality_clean_code(self):
        runner = QualityGateRunner()
        config = QualityGateConfig("q", VerificationMode.VERIFY_QUALITY, min_score=0.6)
        result = runner.check_quality(
            {"code": 'def hello():\n    """Say hello."""\n    return "hello"\n'},
            config,
        )
        assert result.result == GateResult.PASSED
        assert result.score >= 0.6

    def test_quality_no_code(self):
        runner = QualityGateRunner()
        config = QualityGateConfig("q", VerificationMode.VERIFY_QUALITY)
        result = runner.check_quality({}, config)
        assert result.result == GateResult.WARNING

    def test_quality_bare_except_penalty(self):
        runner = QualityGateRunner()
        config = QualityGateConfig("q", VerificationMode.VERIFY_QUALITY, min_score=0.9)
        code = 'def f():\n    """Doc."""\n    try:\n        pass\n    except:\n        pass\n'
        result = runner.check_quality({"code": code}, config)
        assert result.score < 1.0
        assert any("bare except" in str(i.get("message", "")) for i in result.issues)


# ---------------------------------------------------------------------------
# Security checks
# ---------------------------------------------------------------------------

class TestSecurityCheck:
    def test_security_clean_code(self):
        runner = QualityGateRunner()
        config = QualityGateConfig("s", VerificationMode.SECURITY, min_score=0.8)
        result = runner.check_security(
            {"code": 'def safe():\n    return 42\n'},
            config,
        )
        assert result.result == GateResult.PASSED
        assert result.score >= 0.8

    def test_security_eval_detected(self):
        runner = QualityGateRunner()
        config = QualityGateConfig("s", VerificationMode.SECURITY, min_score=0.8)
        result = runner.check_security(
            {"code": 'x = eval("1+1")\n'},
            config,
        )
        assert result.score < 1.0
        assert any("eval" in str(i.get("message", "")) for i in result.issues)

    def test_security_no_code(self):
        runner = QualityGateRunner()
        config = QualityGateConfig("s", VerificationMode.SECURITY)
        result = runner.check_security({}, config)
        assert result.result == GateResult.WARNING


# ---------------------------------------------------------------------------
# Coverage checks
# ---------------------------------------------------------------------------

class TestCoverageCheck:
    def test_coverage_with_tests(self):
        runner = QualityGateRunner()
        config = QualityGateConfig("c", VerificationMode.VERIFY_COVERAGE, min_score=0.5)
        result = runner.check_coverage(
            {
                "code": "def add(a, b):\n    return a + b\n",
                "tests": "def test_add():\n    assert add(1, 2) == 3\n",
                "coverage_percent": 85,
            },
            config,
        )
        assert result.result == GateResult.PASSED

    def test_coverage_no_tests(self):
        runner = QualityGateRunner()
        config = QualityGateConfig("c", VerificationMode.VERIFY_COVERAGE, min_score=0.5)
        result = runner.check_coverage({"code": "def f(): pass"}, config)
        assert result.score < 1.0
        assert any("No test" in str(i.get("message", "")) for i in result.issues)

    def test_coverage_low_percentage(self):
        runner = QualityGateRunner()
        config = QualityGateConfig("c", VerificationMode.VERIFY_COVERAGE, min_score=0.5)
        result = runner.check_coverage(
            {"tests": "def test_x(): pass", "coverage_percent": 30},
            config,
        )
        assert result.score < 1.0


# ---------------------------------------------------------------------------
# Architecture checks
# ---------------------------------------------------------------------------

class TestArchitectureCheck:
    def test_arch_clean_code(self):
        runner = QualityGateRunner()
        config = QualityGateConfig("a", VerificationMode.VERIFY_ARCHITECTURE, min_score=0.7)
        result = runner.check_architecture(
            {"code": "import os\n\nclass Foo:\n    pass\n"},
            config,
        )
        assert result.result == GateResult.PASSED

    def test_arch_wildcard_import_penalized(self):
        runner = QualityGateRunner()
        config = QualityGateConfig("a", VerificationMode.VERIFY_ARCHITECTURE, min_score=0.9)
        result = runner.check_architecture(
            {"code": "from os import *\nfrom sys import *\n"},
            config,
        )
        assert result.score < 1.0
        assert any("wildcard" in str(i.get("message", "")) for i in result.issues)

    def test_arch_no_code(self):
        runner = QualityGateRunner()
        config = QualityGateConfig("a", VerificationMode.VERIFY_ARCHITECTURE)
        result = runner.check_architecture({}, config)
        assert result.result == GateResult.WARNING


# ---------------------------------------------------------------------------
# Full gate run and history
# ---------------------------------------------------------------------------

class TestGateRunAndHistory:
    def test_run_post_execution_gates(self):
        runner = QualityGateRunner()
        results = runner.run_gate("post_execution", {"code": "def f():\n    return 1\n"})
        assert len(results) == 2  # quality_check + security_check
        for r in results:
            assert isinstance(r, GateCheckResult)
            assert "execution_time_ms" in r.metadata

    def test_history_tracking(self):
        runner = QualityGateRunner()
        runner.run_gate("post_execution", {"code": "x = 1"})
        runner.run_gate("post_testing", {"tests": "def test_x(): pass"})
        history = runner.get_history()
        assert len(history) >= 3  # 2 from post_execution + 1 from post_testing

    def test_stage_passed_all_pass(self):
        runner = QualityGateRunner()
        results = runner.run_gate("post_execution", {"code": "def f():\n    return 1\n"})
        assert runner.stage_passed(results) is True

    def test_stage_passed_security_fail(self):
        runner = QualityGateRunner()
        # Code with multiple security issues to ensure failure
        bad_code = 'x = eval("bad")\nos.system("rm -rf /")\npassword = "secret123"\n'
        results = runner.run_gate("pre_assembly", {"code": bad_code})
        # pre_assembly has final_security with min_score=0.9
        assert runner.stage_passed(results) is False


# ---------------------------------------------------------------------------
# EvaluatorAgent mode routing
# ---------------------------------------------------------------------------

class TestEvaluatorAgentModes:
    def test_evaluator_imports(self):
        from vetinari.agents.evaluator_agent import EvaluatorAgent
        agent = EvaluatorAgent()
        assert "code_quality_analysis" in agent.get_capabilities()
