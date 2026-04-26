"""Tests for PreventionGate wiring into QualityGateRunner and TwoLayerOrchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestPreventionModeInEnum:
    def test_prevention_mode_in_enum(self):
        """VerificationMode.PRE_EXECUTION must exist with the correct value."""
        from vetinari.validation.quality_gates import VerificationMode

        assert VerificationMode.PRE_EXECUTION.value == "pre_execution"


class TestPreExecutionGateExists:
    def test_pre_execution_gate_exists(self):
        """QualityGateRunner must have a 'pre_execution' stage in PIPELINE_GATES."""
        from vetinari.validation.quality_gates import QualityGateRunner, VerificationMode

        assert "pre_execution" in QualityGateRunner.PIPELINE_GATES
        configs = QualityGateRunner.PIPELINE_GATES["pre_execution"]
        assert len(configs) == 1
        assert configs[0].mode == VerificationMode.PRE_EXECUTION
        assert configs[0].required is True

    def test_pre_execution_gate_accessible_via_runner(self):
        """get_gates_for_stage('pre_execution') returns the configured gates."""
        from vetinari.validation.quality_gates import QualityGateRunner

        runner = QualityGateRunner()
        gates = runner.get_gates_for_stage("pre_execution")
        assert len(gates) == 1
        assert gates[0].name == "prevention_check"


class TestCheckPreventionPass:
    def test_check_prevention_pass(self):
        """Valid artifacts with acceptance criteria pass the prevention check."""
        from vetinari.validation.quality_gates import GateResult, QualityGateRunner, VerificationMode

        runner = QualityGateRunner()
        config = runner.get_gates_for_stage("pre_execution")[0]
        artifacts = {
            "task_description": "Implement a feature that does something meaningful here",
            "acceptance_criteria": ["Feature works correctly", "Tests pass"],
            "referenced_files": [],
            "model_capabilities": set(),
            "required_capabilities": set(),
            "estimated_tokens": 1000,
            "token_budget": 100_000,
            "active_file_scopes": set(),
        }

        result = runner.check_prevention(artifacts, config)

        assert result.result == GateResult.PASSED
        assert result.score == 1.0
        assert result.issues == []

    def test_check_prevention_returns_gate_check_result(self):
        """check_prevention returns a GateCheckResult with correct mode."""
        from vetinari.validation.quality_gates import GateCheckResult, QualityGateRunner, VerificationMode

        runner = QualityGateRunner()
        config = runner.get_gates_for_stage("pre_execution")[0]
        artifacts = {
            "task_description": "A sufficiently long description for the task at hand",
            "acceptance_criteria": ["Criterion one"],
        }

        result = runner.check_prevention(artifacts, config)

        assert isinstance(result, GateCheckResult)
        assert result.mode == VerificationMode.PRE_EXECUTION
        assert result.gate_name == "prevention_check"


class TestCheckPreventionFail:
    def test_check_prevention_fail_missing_criteria(self):
        """Empty acceptance_criteria causes prevention check to fail."""
        from vetinari.validation.quality_gates import GateResult, QualityGateRunner

        runner = QualityGateRunner()
        config = runner.get_gates_for_stage("pre_execution")[0]
        artifacts = {
            "task_description": "A sufficiently long task description here",
            "acceptance_criteria": [],  # Missing — should fail
        }

        result = runner.check_prevention(artifacts, config)

        assert result.result != GateResult.PASSED
        assert result.score < 1.0
        assert len(result.issues) > 0

    def test_check_prevention_fail_populates_issues(self):
        """Failed prevention checks are surfaced as issues in the result."""
        from vetinari.validation.quality_gates import QualityGateRunner

        runner = QualityGateRunner()
        config = runner.get_gates_for_stage("pre_execution")[0]
        artifacts = {
            "task_description": "",  # Too short
            "acceptance_criteria": [],
        }

        result = runner.check_prevention(artifacts, config)

        messages = [issue["message"] for issue in result.issues]
        assert any(messages), "Expected at least one issue message"
        assert all(issue["category"] == "prevention" for issue in result.issues)

    def test_check_prevention_fail_includes_suggestion(self):
        """Failed prevention gate includes a recommendation suggestion."""
        from vetinari.validation.quality_gates import QualityGateRunner

        runner = QualityGateRunner()
        config = runner.get_gates_for_stage("pre_execution")[0]
        artifacts = {
            "task_description": "Short",
            "acceptance_criteria": [],
        }

        result = runner.check_prevention(artifacts, config)

        assert len(result.suggestions) > 0


class TestPreventionDispatch:
    def test_prevention_dispatch(self):
        """_run_single_gate dispatches PRE_EXECUTION mode to check_prevention."""
        from vetinari.validation.quality_gates import (
            QualityGateConfig,
            QualityGateRunner,
            VerificationMode,
        )

        runner = QualityGateRunner()
        config = QualityGateConfig(
            name="test_prevention",
            mode=VerificationMode.PRE_EXECUTION,
            min_score=0.5,
        )
        artifacts = {
            "task_description": "A meaningful task description that is long enough",
            "acceptance_criteria": ["At least one criterion"],
        }

        # _run_single_gate should call check_prevention without raising
        result = runner._run_single_gate(config, artifacts)

        # Should not fall through to the "No handler" warning path
        assert result.gate_name == "test_prevention"
        assert "No handler for verification mode" not in str(result.issues)

    def test_run_gate_pre_execution_stage(self):
        """run_gate('pre_execution', ...) returns results for the configured gate."""
        from vetinari.validation.quality_gates import QualityGateRunner

        runner = QualityGateRunner()
        artifacts = {
            "task_description": "Implement something meaningful and complete",
            "acceptance_criteria": ["Works correctly"],
        }

        results = runner.run_gate("pre_execution", artifacts)

        assert len(results) == 1
        assert results[0].gate_name == "prevention_check"


class TestOrchestratorHasPreventionGate:
    def test_orchestrator_has_prevention_gate(self):
        """TwoLayerOrchestrator._run_prevention_gate must exist and be callable."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        assert hasattr(TwoLayerOrchestrator, "_run_prevention_gate")
        assert callable(TwoLayerOrchestrator._run_prevention_gate)

    def test_run_prevention_gate_returns_bool(self):
        """_run_prevention_gate returns a bool for valid inputs."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        # Minimal stub — we only need the method to exist and return bool
        orchestrator = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)

        goal = "Implement a meaningful feature with enough description"
        context = {
            "acceptance_criteria": ["Feature completes correctly"],
            "referenced_files": [],
        }

        result = orchestrator._run_prevention_gate(goal, context)

        assert isinstance(result, bool)

    def test_run_prevention_gate_passes_for_good_input(self):
        """_run_prevention_gate returns True when all prevention checks pass."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orchestrator = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)

        goal = "Implement the full authentication feature with token refresh support"
        context = {
            "acceptance_criteria": ["Login works", "Token refresh works"],
            "referenced_files": [],
            "model_capabilities": set(),
            "required_capabilities": set(),
            "estimated_tokens": 500,
            "token_budget": 100_000,
            "active_file_scopes": set(),
        }

        result = orchestrator._run_prevention_gate(goal, context)

        assert result is True

    def test_run_prevention_gate_fails_for_bad_input(self):
        """_run_prevention_gate returns False when enough prevention checks fail.

        Triggers 3+ failures (criteria absent, description too short, model
        missing required capabilities) to push the score below the FAILED
        threshold (min_score * 0.7 = 0.56), causing stage_passed() to return False.
        """
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orchestrator = TwoLayerOrchestrator.__new__(TwoLayerOrchestrator)

        goal = "Short"  # Too short — fails context completeness
        context = {
            "acceptance_criteria": [],  # Fails acceptance-criteria check + context check
            "model_capabilities": set(),
            "required_capabilities": {"vision", "code-interpreter", "tool-use"},  # 3 missing caps
        }

        result = orchestrator._run_prevention_gate(goal, context)

        assert result is False
