"""Tests for vetinari.context.condenser.

Covers ContextCondenser with all known handoff pairs, generic fallback,
truncation, cap length, and singleton management.
"""

from __future__ import annotations

import pytest

from vetinari.context import ContextCondenser, get_context_condenser
from vetinari.types import AgentType

# ── Foreman → Worker ───────────────────────────────────────────────────


class TestForemanToWorker:
    """Test Foreman-to-Worker condensation."""

    def test_dict_output_with_all_fields(self) -> None:
        condenser = ContextCondenser()
        output = {
            "description": "Build the login page",
            "acceptance_criteria": ["Tests pass", "No lint errors"],
            "files": ["auth.py", "login.html"],
            "dependencies": ["flask", "jinja2"],
        }
        result = condenser.condense_for_handoff(AgentType.FOREMAN.value, AgentType.WORKER.value, output)
        assert "Task Assignment from Planner" in result
        assert "Build the login page" in result
        assert "Tests pass" in result
        assert "auth.py" in result
        assert "flask" in result

    def test_string_output(self) -> None:
        condenser = ContextCondenser()
        result = condenser.condense_for_handoff(AgentType.FOREMAN.value, AgentType.WORKER.value, "Build feature X")
        assert "Task Assignment from Planner" in result
        assert "Build feature X" in result

    def test_consolidated_builder_variant(self) -> None:
        condenser = ContextCondenser()
        result = condenser.condense_for_handoff(
            AgentType.FOREMAN.value, AgentType.WORKER.value, {"description": "task"}
        )
        assert "Task Assignment from Planner" in result


# ── Worker → Inspector ─────────────────────────────────────────────────


class TestWorkerToInspector:
    """Test Worker-to-Inspector condensation."""

    def test_dict_output_with_changes(self) -> None:
        condenser = ContextCondenser()
        output = {
            "changes": ["Added auth module", "Updated config"],
            "files_modified": ["auth.py", "config.yaml"],
            "test_results": "12 passed, 0 failed",
            "summary": "Auth module complete",
        }
        result = condenser.condense_for_handoff(AgentType.WORKER.value, AgentType.INSPECTOR.value, output)
        assert "Build Output for Review" in result
        assert "Added auth module" in result
        assert "auth.py" in result
        assert "12 passed" in result

    def test_string_output(self) -> None:
        condenser = ContextCondenser()
        result = condenser.condense_for_handoff(
            AgentType.WORKER.value, AgentType.INSPECTOR.value, "Code changes applied"
        )
        assert "Build Output for Review" in result

    def test_consolidated_builder_variant(self) -> None:
        condenser = ContextCondenser()
        result = condenser.condense_for_handoff(AgentType.WORKER.value, AgentType.INSPECTOR.value, {"summary": "done"})
        assert "Build Output for Review" in result


# ── Inspector → Worker ─────────────────────────────────────────────────


class TestInspectorToWorker:
    """Test Inspector-to-Worker rework condensation."""

    def test_dict_output_with_issues(self) -> None:
        condenser = ContextCondenser()
        output = {
            "issues": ["Minor: unused import"],
            "suggestions": ["Remove unused import at top of file"],
        }
        result = condenser.condense_for_handoff(AgentType.INSPECTOR.value, AgentType.WORKER.value, output)
        assert "Rework Instructions from Quality" in result
        assert "unused import" in result
        assert "Remove unused import" in result

    def test_dict_output_security_issue(self) -> None:
        condenser = ContextCondenser()
        output = {"issues": ["Security: SQL injection"]}
        result = condenser.condense_for_handoff(AgentType.INSPECTOR.value, AgentType.WORKER.value, output)
        assert "Rework Instructions from Quality" in result
        assert "SQL injection" in result

    def test_string_output(self) -> None:
        condenser = ContextCondenser()
        result = condenser.condense_for_handoff(
            AgentType.INSPECTOR.value, AgentType.WORKER.value, "Fix the auth module"
        )
        assert "Rework Instructions from Quality" in result


# ── Inspector → Worker (Rework) ────────────────────────────────────────


class TestInspectorToWorkerRework:
    """Test Inspector-to-Worker rework condensation."""

    def test_dict_issues_with_file_refs(self) -> None:
        condenser = ContextCondenser()
        output = {
            "issues": [
                {"file": "auth.py", "line": "42", "description": "Missing error handling"},
                {"file": "config.py", "description": "No type hints"},
            ],
            "suggestions": ["Add try/except around DB call", "Use TypedDict"],
        }
        result = condenser.condense_for_handoff(AgentType.INSPECTOR.value, AgentType.WORKER.value, output)
        assert "Rework Instructions from Quality" in result
        assert "auth.py:42" in result
        assert "Missing error handling" in result
        assert "Add try/except" in result

    def test_dict_issues_as_strings(self) -> None:
        condenser = ContextCondenser()
        output = {"issues": ["Fix the bug", "Add tests"]}
        result = condenser.condense_for_handoff(AgentType.INSPECTOR.value, AgentType.WORKER.value, output)
        assert "Fix the bug" in result

    def test_string_output(self) -> None:
        condenser = ContextCondenser()
        result = condenser.condense_for_handoff(AgentType.INSPECTOR.value, AgentType.WORKER.value, "Fix auth module")
        assert "Rework Instructions from Quality" in result


# ── Worker → Worker (Research) ─────────────────────────────────────────


class TestWorkerResearchOutput:
    """Test Worker research output condensation (falls through to generic handler)."""

    def test_dict_with_findings(self) -> None:
        condenser = ContextCondenser()
        output = {
            "findings": [
                {"title": "API Rate Limits", "summary": "Max 100 req/min"},
                "Simple text finding",
            ],
            "sources": ["docs.example.com", "api.example.com"],
            "confidence": "high",
        }
        result = condenser.condense_for_handoff(AgentType.WORKER.value, AgentType.WORKER.value, output)
        assert "Agent Output Summary" in result
        assert "confidence" in result
        assert "high" in result

    def test_consolidated_researcher_variant(self) -> None:
        condenser = ContextCondenser()
        result = condenser.condense_for_handoff(AgentType.WORKER.value, AgentType.WORKER.value, "findings")
        assert "Agent Output Summary" in result


# ── Worker → Foreman (Architecture) ────────────────────────────────────


class TestWorkerArchitectureOutput:
    """Test Worker architecture output condensation (falls through to generic handler)."""

    def test_dict_with_decision(self) -> None:
        condenser = ContextCondenser()
        output = {
            "decision": "Use PostgreSQL",
            "rationale": "Better JSON support",
            "alternatives": ["MySQL", "SQLite"],
            "adr_id": "ADR-0042",
        }
        result = condenser.condense_for_handoff(AgentType.WORKER.value, AgentType.FOREMAN.value, output)
        assert "Agent Output Summary" in result
        assert "Use PostgreSQL" in result
        assert "ADR-0042" in result

    def test_consolidated_oracle_variant(self) -> None:
        condenser = ContextCondenser()
        result = condenser.condense_for_handoff(AgentType.WORKER.value, AgentType.FOREMAN.value, {"decision": "yes"})
        assert "Agent Output Summary" in result


# ── Generic Fallback ───────────────────────────────────────────────────


class TestGenericFallback:
    """Test generic/unknown agent pair condensation."""

    @pytest.mark.parametrize(
        "output",
        [
            {"key1": "value1", "reasoning": "long verbose text", "result": "ok"},
            "some output",
            [1, 2, 3],
        ],
    )
    def test_unknown_pair_always_has_summary_header(self, output) -> None:
        """Fallback condenser always includes 'Agent Output Summary' regardless of output type."""
        condenser = ContextCondenser()
        result = condenser.condense_for_handoff("UNKNOWN", "OTHER", output)
        assert "Agent Output Summary" in result

    def test_unknown_pair_dict_filters_verbose_keys(self) -> None:
        condenser = ContextCondenser()
        output = {"key1": "value1", "reasoning": "long verbose text", "result": "ok"}
        result = condenser.condense_for_handoff("UNKNOWN", "OTHER", output)
        assert "key1" in result
        assert "reasoning" not in result  # Filtered out as verbose

    def test_long_dict_values_truncated(self) -> None:
        condenser = ContextCondenser()
        output = {"big_field": "x" * 1000}
        result = condenser.condense_for_handoff("X", "Y", output)
        assert "[...output truncated...]" in result


# ── Truncation and Cap ─────────────────────────────────────────────────


class TestTruncationAndCap:
    """Test truncation and length capping."""

    def test_truncate_short_text_unchanged(self) -> None:
        condenser = ContextCondenser()
        assert condenser._truncate("hello") == "hello"

    def test_truncate_long_text(self) -> None:
        condenser = ContextCondenser()
        text = "x" * 5000
        result = condenser._truncate(text, max_len=100)
        assert len(result) == 100
        assert result.endswith("[...output truncated...]")

    def test_cap_length_short_text_unchanged(self) -> None:
        condenser = ContextCondenser()
        assert condenser._cap_length("short") == "short"

    def test_cap_length_long_text(self) -> None:
        condenser = ContextCondenser()
        text = "x" * 20000
        result = condenser._cap_length(text)
        assert len(result) == 16000
        assert result.endswith("[...output truncated...]")


# ── Case Insensitivity ─────────────────────────────────────────────────


class TestCaseInsensitivity:
    """Test that agent name matching is case-insensitive."""

    def test_lowercase_agents(self) -> None:
        condenser = ContextCondenser()
        result = condenser.condense_for_handoff("foreman", "worker", "task")
        assert "Task Assignment from Planner" in result

    def test_mixed_case_agents(self) -> None:
        condenser = ContextCondenser()
        result = condenser.condense_for_handoff("Worker", "Inspector", "output")
        assert "Build Output for Review" in result


# ── Singleton ──────────────────────────────────────────────────────────


class TestCondenserSingleton:
    """Test singleton management."""

    def test_returns_same_instance(self) -> None:
        c1 = get_context_condenser()
        c2 = get_context_condenser()
        assert c1 is c2

    def test_is_context_condenser(self) -> None:
        c = get_context_condenser()
        assert isinstance(c, ContextCondenser)
