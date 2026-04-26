"""Tests for vetinari.validation.prevention — Poka-Yoke prevention gate."""

from __future__ import annotations

import pytest

from vetinari.validation import CheckResult, PreventionGate, PreventionResult

# ── Shared fixtures ────────────────────────────────────────────────────────────

VALID_DESCRIPTION = "Implement the token-budget check in the prevention gate."
VALID_CRITERIA = ["Check returns False when estimated > 90% of budget", "All tests pass"]
VALID_MODEL_CAPS: set[str] = {"code_generation", "reasoning"}
VALID_REQUIRED_CAPS: set[str] = {"code_generation"}


@pytest.fixture
def gate() -> PreventionGate:
    """Return a fresh PreventionGate for each test."""
    return PreventionGate()


@pytest.fixture
def existing_file(tmp_path):
    """Create a real temporary file and return its string path."""
    p = tmp_path / "source.py"
    p.write_text("# placeholder", encoding="utf-8")
    return str(p)


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestPreventionGate:
    def test_all_checks_pass(self, gate: PreventionGate, existing_file: str) -> None:
        """All valid inputs should produce a passing PreventionResult."""
        result = gate.validate(
            task_description=VALID_DESCRIPTION,
            acceptance_criteria=VALID_CRITERIA,
            referenced_files=[existing_file],
            model_capabilities=VALID_MODEL_CAPS,
            required_capabilities=VALID_REQUIRED_CAPS,
            estimated_tokens=500,
            token_budget=1000,
            active_file_scopes=set(),
        )
        assert result.passed is True
        assert result.failures == []
        assert result.recommendation == "proceed"

    def test_no_acceptance_criteria(self, gate: PreventionGate, existing_file: str) -> None:
        """Empty acceptance criteria should cause at least one failure."""
        result = gate.validate(
            task_description=VALID_DESCRIPTION,
            acceptance_criteria=[],
            referenced_files=[existing_file],
            model_capabilities=VALID_MODEL_CAPS,
            required_capabilities=VALID_REQUIRED_CAPS,
            estimated_tokens=500,
            token_budget=1000,
            active_file_scopes=set(),
        )
        assert result.passed is False
        assert any(not c.passed for c in result.failures)

    def test_missing_files(self, gate: PreventionGate, tmp_path) -> None:
        """Referencing a nonexistent file should fail the files-exist check."""
        nonexistent = str(tmp_path / "does_not_exist.py")
        result = gate.validate(
            task_description=VALID_DESCRIPTION,
            acceptance_criteria=VALID_CRITERIA,
            referenced_files=[nonexistent],
            model_capabilities=VALID_MODEL_CAPS,
            required_capabilities=VALID_REQUIRED_CAPS,
            estimated_tokens=500,
            token_budget=1000,
            active_file_scopes=set(),
        )
        assert result.passed is False
        assert any("does_not_exist.py" in f.reason for f in result.failures)

    def test_short_description(self, gate: PreventionGate, existing_file: str) -> None:
        """A description shorter than 20 characters should fail context check."""
        result = gate.validate(
            task_description="Too short",
            acceptance_criteria=VALID_CRITERIA,
            referenced_files=[existing_file],
            model_capabilities=VALID_MODEL_CAPS,
            required_capabilities=VALID_REQUIRED_CAPS,
            estimated_tokens=500,
            token_budget=1000,
            active_file_scopes=set(),
        )
        assert result.passed is False
        assert any("too short" in f.reason.lower() for f in result.failures)

    def test_missing_model_capability(self, gate: PreventionGate, existing_file: str) -> None:
        """Required capability absent from model caps should fail."""
        result = gate.validate(
            task_description=VALID_DESCRIPTION,
            acceptance_criteria=VALID_CRITERIA,
            referenced_files=[existing_file],
            model_capabilities={"reasoning"},
            required_capabilities={"code_generation", "tool_use"},
            estimated_tokens=500,
            token_budget=1000,
            active_file_scopes=set(),
        )
        assert result.passed is False
        assert any("capabilities" in f.reason.lower() for f in result.failures)

    def test_token_budget_exceeded(self, gate: PreventionGate, existing_file: str) -> None:
        """Estimated tokens over 90% of budget should fail."""
        result = gate.validate(
            task_description=VALID_DESCRIPTION,
            acceptance_criteria=VALID_CRITERIA,
            referenced_files=[existing_file],
            model_capabilities=VALID_MODEL_CAPS,
            required_capabilities=VALID_REQUIRED_CAPS,
            estimated_tokens=950,
            token_budget=1000,
            active_file_scopes=set(),
        )
        assert result.passed is False
        assert any("budget" in f.reason.lower() or "tokens" in f.reason.lower() for f in result.failures)

    def test_token_budget_at_exactly_90_percent(self, gate: PreventionGate, existing_file: str) -> None:
        """Estimated tokens at exactly 90% of budget should pass (boundary condition)."""
        result = gate.validate(
            task_description=VALID_DESCRIPTION,
            acceptance_criteria=VALID_CRITERIA,
            referenced_files=[existing_file],
            model_capabilities=VALID_MODEL_CAPS,
            required_capabilities=VALID_REQUIRED_CAPS,
            estimated_tokens=900,
            token_budget=1000,
            active_file_scopes=set(),
        )
        assert result.passed is True

    def test_concurrent_conflicts(self, gate: PreventionGate, existing_file: str) -> None:
        """A file already in active_file_scopes should cause a conflict failure."""
        result = gate.validate(
            task_description=VALID_DESCRIPTION,
            acceptance_criteria=VALID_CRITERIA,
            referenced_files=[existing_file],
            model_capabilities=VALID_MODEL_CAPS,
            required_capabilities=VALID_REQUIRED_CAPS,
            estimated_tokens=500,
            token_budget=1000,
            active_file_scopes={existing_file},
        )
        assert result.passed is False
        assert any("already being modified" in f.reason for f in result.failures)

    def test_recommend_add_context(self, gate: PreventionGate, existing_file: str) -> None:
        """A short description with no model/token issues should recommend add_context."""
        result = gate.validate(
            task_description="Too short",
            acceptance_criteria=VALID_CRITERIA,
            referenced_files=[existing_file],
            model_capabilities=VALID_MODEL_CAPS,
            required_capabilities=VALID_REQUIRED_CAPS,
            estimated_tokens=100,
            token_budget=1000,
            active_file_scopes=set(),
        )
        assert result.recommendation == "add_context"

    def test_recommend_change_model(self, gate: PreventionGate, existing_file: str) -> None:
        """Missing model capability should recommend change_model."""
        result = gate.validate(
            task_description=VALID_DESCRIPTION,
            acceptance_criteria=VALID_CRITERIA,
            referenced_files=[existing_file],
            model_capabilities=set(),
            required_capabilities={"code_generation"},
            estimated_tokens=100,
            token_budget=1000,
            active_file_scopes=set(),
        )
        assert result.recommendation == "change_model"

    def test_recommend_split_task(self, gate: PreventionGate, existing_file: str) -> None:
        """Token budget exceeded (with no model issue) should recommend split_task."""
        result = gate.validate(
            task_description=VALID_DESCRIPTION,
            acceptance_criteria=VALID_CRITERIA,
            referenced_files=[existing_file],
            model_capabilities=VALID_MODEL_CAPS,
            required_capabilities=VALID_REQUIRED_CAPS,
            estimated_tokens=950,
            token_budget=1000,
            active_file_scopes=set(),
        )
        assert result.recommendation == "split_task"

    def test_recommend_clarify_with_user(self, gate: PreventionGate, existing_file: str) -> None:
        """Missing acceptance criteria with adequate description should recommend clarify_with_user."""
        result = gate.validate(
            task_description=VALID_DESCRIPTION,
            acceptance_criteria=[],
            referenced_files=[existing_file],
            model_capabilities=VALID_MODEL_CAPS,
            required_capabilities=VALID_REQUIRED_CAPS,
            estimated_tokens=100,
            token_budget=1000,
            active_file_scopes=set(),
        )
        assert result.recommendation == "clarify_with_user"

    def test_multiple_failures(self, gate: PreventionGate, tmp_path) -> None:
        """Multiple simultaneous problems should all appear in failures list."""
        nonexistent = str(tmp_path / "ghost.py")
        result = gate.validate(
            task_description="Too short",
            acceptance_criteria=[],
            referenced_files=[nonexistent],
            model_capabilities=set(),
            required_capabilities={"code_generation"},
            estimated_tokens=999,
            token_budget=1000,
            active_file_scopes=set(),
        )
        assert result.passed is False
        # Expect failures from criteria, file-existence, context, model cap, and token budget
        assert len(result.failures) >= 4

    def test_check_result_defaults(self) -> None:
        """CheckResult should default reason to empty string."""
        cr = CheckResult(passed=True)
        assert cr.reason == ""

    def test_prevention_result_defaults(self) -> None:
        """PreventionResult should default failures to empty list and recommendation to proceed."""
        pr = PreventionResult(passed=True)
        assert pr.failures == []
        assert pr.recommendation == "proceed"

    def test_empty_referenced_files_passes(self, gate: PreventionGate) -> None:
        """No referenced files should not trigger the file-existence check."""
        result = gate.validate(
            task_description=VALID_DESCRIPTION,
            acceptance_criteria=VALID_CRITERIA,
            referenced_files=[],
            model_capabilities=VALID_MODEL_CAPS,
            required_capabilities=VALID_REQUIRED_CAPS,
            estimated_tokens=100,
            token_budget=1000,
            active_file_scopes=set(),
        )
        assert result.passed is True

    def test_no_required_capabilities_passes(self, gate: PreventionGate) -> None:
        """Empty required capabilities should always pass the capability check."""
        result = gate.validate(
            task_description=VALID_DESCRIPTION,
            acceptance_criteria=VALID_CRITERIA,
            referenced_files=[],
            model_capabilities=set(),
            required_capabilities=set(),
            estimated_tokens=100,
            token_budget=1000,
            active_file_scopes=set(),
        )
        assert result.passed is True
