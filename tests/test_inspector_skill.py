"""Tests for InspectorSkillTool — independent quality gate.

Covers all 4 modes (code_review, security_audit, test_generation, simplification),
self_check integration, score/grade calculation, and ToolResult contract.
"""

from __future__ import annotations

import pytest

from vetinari.skills.inspector_skill import (
    InspectorMode,
    InspectorResult,
    InspectorSkillTool,
    ReviewIssue,
)
from vetinari.tool_interface import ToolResult
from vetinari.types import AgentType

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def inspector():
    """Create a fresh InspectorSkillTool instance."""
    return InspectorSkillTool()


@pytest.fixture
def clean_code():
    """A clean Python code sample with no issues."""
    return '''def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def multiply(x: int, y: int) -> int:
    """Multiply two integers."""
    return x * y
'''


@pytest.fixture
def problematic_code():
    """A Python code sample with multiple issues."""
    return """import os

def process():
    print("starting")
    try:
        data = open("file.txt").read()
    except:
        pass
    # TODO: fix this later
    return data
"""


# ═══════════════════════════════════════════════════════════════════════════
# Initialization and Metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestInspectorInitialization:
    """Tests for InspectorSkillTool initialization and metadata."""

    def test_initialization(self, inspector):
        """InspectorSkillTool initializes with correct metadata."""
        assert inspector.metadata.name == "inspector"
        assert inspector.metadata.version == "2.0.0"

    def test_description_is_meaningful(self, inspector):
        """Description is meaningful."""
        assert len(inspector.metadata.description) > 20
        assert "quality" in inspector.metadata.description.lower()

    def test_code_parameter_required(self, inspector):
        """The 'code' parameter is required."""
        param_names = [p.name for p in inspector.metadata.parameters]
        assert "code" in param_names
        code_param = next(p for p in inspector.metadata.parameters if p.name == "code")
        assert code_param.required is True

    def test_mode_parameter_lists_all_modes(self, inspector):
        """The 'mode' parameter lists all valid modes."""
        mode_param = next(p for p in inspector.metadata.parameters if p.name == "mode")
        assert set(mode_param.allowed_values) == {m.value for m in InspectorMode}

    def test_tags_include_quality(self, inspector):
        """Tags include quality-related keywords."""
        assert "quality" in inspector.metadata.tags
        assert "security" in inspector.metadata.tags
        assert "gate" in inspector.metadata.tags


# ═══════════════════════════════════════════════════════════════════════════
# Code Review Mode
# ═══════════════════════════════════════════════════════════════════════════


class TestCodeReview:
    """Tests for code_review mode."""

    def test_clean_code_passes(self, inspector, clean_code):
        """Clean code passes review with high score."""
        result = inspector.execute(code=clean_code, mode="code_review")
        assert result.success is True
        assert result.output["passed"] is True
        assert result.output["score"] >= 0.9

    def test_finds_print_statements(self, inspector, problematic_code):
        """Detects print() in production code."""
        result = inspector.execute(code=problematic_code, mode="code_review")
        issues = result.output["issues"]
        print_issues = [i for i in issues if "print" in i["description"].lower()]
        assert len(print_issues) >= 1

    def test_finds_bare_except(self, inspector, problematic_code):
        """Detects bare except clause."""
        result = inspector.execute(code=problematic_code, mode="code_review")
        issues = result.output["issues"]
        bare_except = [i for i in issues if "except" in i["description"].lower()]
        assert len(bare_except) >= 1

    def test_finds_todo_comments(self, inspector, problematic_code):
        """Detects TODO comments."""
        result = inspector.execute(code=problematic_code, mode="code_review")
        issues = result.output["issues"]
        todos = [i for i in issues if "TODO" in i["description"]]
        assert len(todos) >= 1

    def test_empty_code_passes(self, inspector):
        """Empty code string passes with no issues."""
        result = inspector.execute(code="", mode="code_review")
        assert result.success is True
        assert result.output["passed"] is True

    def test_metrics_include_line_count(self, inspector, clean_code):
        """Metrics include lines_reviewed count."""
        result = inspector.execute(code=clean_code, mode="code_review")
        assert "lines_reviewed" in result.output["metrics"]
        assert result.output["metrics"]["lines_reviewed"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# Security Audit Mode
# ═══════════════════════════════════════════════════════════════════════════


class TestSecurityAudit:
    """Tests for security_audit mode."""

    def test_clean_code_passes(self, inspector, clean_code):
        """Clean code passes security audit."""
        result = inspector.execute(code=clean_code, mode="security_audit")
        assert result.success is True
        assert result.output["passed"] is True

    def test_detects_eval(self, inspector):
        """Detects eval() usage."""
        code = "result = eval(user_input)"
        result = inspector.execute(code=code, mode="security_audit")
        issues = result.output["issues"]
        eval_issues = [i for i in issues if "eval" in i["description"].lower()]
        assert len(eval_issues) >= 1

    def test_detects_yaml_load(self, inspector):
        """Detects unsafe yaml.load() usage."""
        code = "data = yaml.load(content)"
        result = inspector.execute(code=code, mode="security_audit")
        issues = result.output["issues"]
        yaml_issues = [i for i in issues if "yaml" in i["description"].lower()]
        assert len(yaml_issues) >= 1

    def test_detects_pickle(self, inspector):
        """Detects pickle.loads() usage."""
        code = "obj = pickle.loads(data)"
        result = inspector.execute(code=code, mode="security_audit")
        issues = result.output["issues"]
        pickle_issues = [
            i for i in issues if "pickle" in i["description"].lower() or "deserialization" in i["description"].lower()
        ]
        assert len(pickle_issues) >= 1

    def test_detects_ssl_bypass(self, inspector):
        """Detects verify=False SSL bypass."""
        code = "response = requests.get(url, verify=False)"
        result = inspector.execute(code=code, mode="security_audit")
        issues = result.output["issues"]
        ssl_issues = [i for i in issues if "ssl" in i["description"].lower() or "verify" in i["description"].lower()]
        assert len(ssl_issues) >= 1

    def test_metrics_include_patterns_checked(self, inspector, clean_code):
        """Metrics include patterns_checked count."""
        result = inspector.execute(code=clean_code, mode="security_audit")
        assert "patterns_checked" in result.output["metrics"]

    def test_issues_have_cwe_and_owasp(self, inspector):
        """Security issues include CWE and OWASP references."""
        code = "result = eval(user_input)"
        result = inspector.execute(code=code, mode="security_audit")
        for issue in result.output["issues"]:
            if issue.get("cwe"):
                assert issue["cwe"].startswith("CWE-")
            if issue.get("owasp"):
                assert ":" in issue["owasp"]


# ═══════════════════════════════════════════════════════════════════════════
# Test Generation Mode
# ═══════════════════════════════════════════════════════════════════════════


class TestTestGeneration:
    """Tests for test_generation mode."""

    def test_generates_suggestions(self, inspector, clean_code):
        """Generates test suggestions for public functions."""
        result = inspector.execute(code=clean_code, mode="test_generation")
        assert result.success is True
        suggestions = result.output["suggestions"]
        assert len(suggestions) > 0

    def test_counts_public_functions(self, inspector, clean_code):
        """Metrics include public function count."""
        result = inspector.execute(code=clean_code, mode="test_generation")
        assert result.output["metrics"]["public_functions"] >= 2

    def test_fails_when_gaps_found(self, inspector, problematic_code):
        """Test generation mode fails when untested public functions are found."""
        result = inspector.execute(code=problematic_code, mode="test_generation")
        # Bug fix: _test_generation now returns passed=False when gaps exist
        assert result.output["passed"] is False
        assert len(result.output["suggestions"]) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Simplification Mode
# ═══════════════════════════════════════════════════════════════════════════


class TestSimplification:
    """Tests for simplification mode."""

    def test_simple_code_passes(self, inspector, clean_code):
        """Simple code passes simplification check."""
        result = inspector.execute(code=clean_code, mode="simplification")
        assert result.success is True

    def test_detects_long_functions(self, inspector):
        """Detects functions over 50 lines."""
        long_func = (
            "def long_function():\n"
            + "\n".join([f"    x_{i} = {i}" for i in range(55)])
            + "\ndef next_function():\n    pass\n"
        )
        result = inspector.execute(code=long_func, mode="simplification")
        issues = result.output["issues"]
        long_issues = [i for i in issues if "lines" in i.get("description", "")]
        assert len(long_issues) >= 1

    def test_metrics_include_nesting(self, inspector, clean_code):
        """Metrics include max_nesting_depth."""
        result = inspector.execute(code=clean_code, mode="simplification")
        assert "max_nesting_depth" in result.output["metrics"]


# ═══════════════════════════════════════════════════════════════════════════
# Self-Check Integration (Phase 5.24)
# ═══════════════════════════════════════════════════════════════════════════


class TestSelfCheckIntegration:
    """Tests for self_check_passed integration from context."""

    def test_self_check_passed_included(self, inspector, clean_code):
        """self_check_passed from context is included in output."""
        result = inspector.execute(
            code=clean_code,
            mode="code_review",
            context={"self_check_passed": True},
        )
        assert result.output["self_check_passed"] is True

    def test_self_check_failed_adds_advisory(self, inspector, clean_code):
        """Failed self_check adds advisory suggestion."""
        result = inspector.execute(
            code=clean_code,
            mode="code_review",
            context={
                "self_check_passed": False,
                "self_check_issues": ["Missing type hints", "No docstring"],
            },
        )
        assert result.output["self_check_passed"] is False
        # Advisory suggestion should mention self-check
        suggestions = result.output["suggestions"]
        assert any("self-check" in s.lower() for s in suggestions)

    def test_no_self_check_omits_field(self, inspector, clean_code):
        """When no self_check context, field is omitted from output."""
        result = inspector.execute(code=clean_code, mode="code_review")
        assert "self_check_passed" not in result.output


# ═══════════════════════════════════════════════════════════════════════════
# Score and Grade Calculation
# ═══════════════════════════════════════════════════════════════════════════


class TestScoreGrade:
    """Tests for score-to-grade calculation."""

    def test_perfect_score_gets_a(self, inspector):
        """Score >= 0.9 gets grade A."""
        assert inspector._score_to_grade(1.0) == "A"
        assert inspector._score_to_grade(0.95) == "A"
        assert inspector._score_to_grade(0.9) == "A"

    def test_good_score_gets_b(self, inspector):
        """Score 0.8-0.89 gets grade B."""
        assert inspector._score_to_grade(0.85) == "B"
        assert inspector._score_to_grade(0.8) == "B"

    def test_average_score_gets_c(self, inspector):
        """Score 0.7-0.79 gets grade C."""
        assert inspector._score_to_grade(0.75) == "C"
        assert inspector._score_to_grade(0.7) == "C"

    def test_poor_score_gets_d(self, inspector):
        """Score 0.6-0.69 gets grade D."""
        assert inspector._score_to_grade(0.65) == "D"
        assert inspector._score_to_grade(0.6) == "D"

    def test_failing_score_gets_f(self, inspector):
        """Score < 0.6 gets grade F."""
        assert inspector._score_to_grade(0.5) == "F"
        assert inspector._score_to_grade(0.0) == "F"


# ═══════════════════════════════════════════════════════════════════════════
# Error Handling
# ═══════════════════════════════════════════════════════════════════════════


class TestInspectorErrors:
    """Tests for error handling in InspectorSkillTool."""

    def test_invalid_mode(self, inspector):
        """Invalid mode returns error ToolResult."""
        result = inspector.execute(code="test", mode="nonexistent")
        assert result.success is False
        assert "Unknown mode" in result.error

    def test_empty_code_accepted(self, inspector):
        """Empty code is accepted (might be intentional)."""
        result = inspector.execute(code="", mode="code_review")
        assert result.success is True

    def test_default_mode_is_code_review(self, inspector):
        """Default mode is code_review when omitted."""
        result = inspector.execute(code="x = 1")
        assert result.metadata["mode"] == "code_review"


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════


class TestReviewIssue:
    """Tests for ReviewIssue dataclass."""

    def test_minimal_issue(self):
        """ReviewIssue with only required fields uses empty defaults for optional."""
        from dataclasses import asdict

        issue = ReviewIssue(severity="high", description="Bad thing")
        d = asdict(issue)
        assert d["severity"] == "high"
        assert d["description"] == "Bad thing"
        assert d.get("file") == ""  # default empty string, not absent
        assert d.get("cwe") == ""  # default empty string, not absent

    def test_full_issue(self):
        """ReviewIssue with all fields populated."""
        from dataclasses import asdict

        issue = ReviewIssue(
            severity="critical",
            description="SQL injection",
            file="app.py",
            line=42,
            category="security",
            cwe="CWE-89",
            owasp="A03:2021",
            suggestion="Use parameterized queries",
        )
        d = asdict(issue)
        assert d["file"] == "app.py"
        assert d["line"] == 42
        assert d["cwe"] == "CWE-89"
        assert d["owasp"] == "A03:2021"
        assert d["suggestion"] == "Use parameterized queries"


class TestInspectorResult:
    """Tests for InspectorResult dataclass."""

    def test_defaults(self):
        """InspectorResult has passing defaults."""
        r = InspectorResult()
        assert r.passed is True
        assert r.grade == "A"
        assert r.score == 1.0
        assert r.issues == []
        assert r.suggestions == []
        assert r.self_check_passed is None

    def test_to_dict_with_self_check(self):
        """to_dict includes self_check_passed when set."""
        r = InspectorResult(self_check_passed=True)
        d = r.to_dict()
        assert d["self_check_passed"] is True

    def test_to_dict_without_self_check(self):
        """to_dict omits self_check_passed when None."""
        r = InspectorResult()
        d = r.to_dict()
        assert "self_check_passed" not in d


# ═══════════════════════════════════════════════════════════════════════════
# Merge/Dedup Logic
# ═══════════════════════════════════════════════════════════════════════════


class TestMergeQualityIssues:
    """Tests for _merge_quality_issues() deduplication logic."""

    def test_empty_quality_result_returns_existing(self, inspector):
        """None or failed quality result returns existing issues unchanged."""
        existing = [ReviewIssue(severity="high", description="Existing issue")]
        result = inspector._merge_quality_issues(existing, None)
        assert len(result) == 1
        assert result[0].description == "Existing issue"

    def test_dedup_by_description(self, inspector):
        """Duplicate descriptions are skipped (case-insensitive)."""
        from unittest.mock import MagicMock

        existing = [ReviewIssue(severity="high", description="Bare except clause")]
        quality_result = MagicMock()
        quality_result.success = True
        quality_result.output = {
            "issues": [
                {"title": "bare except clause", "severity": "medium"},  # duplicate
                {"title": "Missing type hints", "severity": "low"},  # new
            ]
        }
        merged = inspector._merge_quality_issues(existing, quality_result, dedup_field="description")
        assert len(merged) == 2  # 1 existing + 1 new (duplicate skipped)
        descriptions = {i.description for i in merged}
        assert "Bare except clause" in descriptions
        assert "Missing type hints" in descriptions

    def test_dedup_by_cwe(self, inspector):
        """Duplicate CWE IDs are skipped in security audit merge."""
        from unittest.mock import MagicMock

        existing = [ReviewIssue(severity="high", description="Eval issue", cwe="CWE-95")]
        quality_result = MagicMock()
        quality_result.success = True
        quality_result.output = {
            "issues": [
                {"title": "Eval injection", "severity": "critical", "cwe_id": "CWE-95"},  # dup
                {"title": "Debug mode", "severity": "high", "cwe_id": "CWE-489"},  # new
            ]
        }
        merged = inspector._merge_quality_issues(existing, quality_result, dedup_field="cwe")
        assert len(merged) == 2  # 1 existing + 1 new (CWE-95 dup skipped)

    def test_failed_quality_result_skipped(self, inspector):
        """Failed quality result returns existing issues unchanged."""
        from unittest.mock import MagicMock

        existing = [ReviewIssue(severity="low", description="Minor")]
        quality_result = MagicMock()
        quality_result.success = False
        quality_result.output = None
        merged = inspector._merge_quality_issues(existing, quality_result)
        assert len(merged) == 1

    def test_empty_issues_in_quality_result(self, inspector):
        """Quality result with empty issues list returns existing unchanged."""
        from unittest.mock import MagicMock

        existing = [ReviewIssue(severity="low", description="Minor")]
        quality_result = MagicMock()
        quality_result.success = True
        quality_result.output = {"issues": []}
        merged = inspector._merge_quality_issues(existing, quality_result)
        assert len(merged) == 1


# ═══════════════════════════════════════════════════════════════════════════
# InspectorMode Enum
# ═══════════════════════════════════════════════════════════════════════════


class TestInspectorModeEnum:
    """Tests for InspectorMode enum."""

    def test_has_four_modes(self):
        """InspectorMode has exactly 4 values."""
        assert len(InspectorMode) == 4

    def test_expected_modes(self):
        """All expected modes exist."""
        expected = {"code_review", "security_audit", "test_generation", "simplification"}
        actual = {m.value for m in InspectorMode}
        assert actual == expected


# ═══════════════════════════════════════════════════════════════════════════
# All Modes Execute Successfully
# ═══════════════════════════════════════════════════════════════════════════


class TestInspectorAllModes:
    """Verify every mode executes without error."""

    @pytest.mark.parametrize("mode", [m.value for m in InspectorMode])
    def test_all_modes_succeed(self, inspector, mode):
        """Every mode returns success=True."""
        result = inspector.execute(code="def foo(): pass", mode=mode)
        assert result.success is True
        assert result.metadata["mode"] == mode
        assert result.metadata["agent"] == AgentType.INSPECTOR.value


# ═══════════════════════════════════════════════════════════════════════════
# OutcomeSignal integration
# ═══════════════════════════════════════════════════════════════════════════


class TestInspectorOutcomeSignal:
    """Tests for _inspector_result_to_signal(), aggregate_outcome_signals(), and execute() wiring."""

    def test_execute_metadata_contains_outcome_signal(self, inspector, clean_code):
        """execute() always injects outcome_signal into metadata."""
        result = inspector.execute(code=clean_code, mode="code_review")
        assert "outcome_signal" in result.metadata
        sig = result.metadata["outcome_signal"]
        assert "passed" in sig
        assert "score" in sig
        assert "basis" in sig
        assert "issues" in sig

    def test_execute_outcome_signal_basis_is_llm_judgment(self, inspector, clean_code):
        """Inspector uses heuristic scoring — outcome_signal basis must be llm_judgment."""
        result = inspector.execute(code=clean_code, mode="code_review")
        assert result.metadata["outcome_signal"]["basis"] == "llm_judgment"

    def test_execute_outcome_signal_passed_matches_inspector_result(self, inspector, problematic_code):
        """outcome_signal.passed must agree with the top-level passed field."""
        result = inspector.execute(code=problematic_code, mode="code_review")
        assert result.metadata["outcome_signal"]["passed"] == result.output["passed"]

    def test_inspector_result_to_signal_returns_llm_judgment(self):
        """_inspector_result_to_signal produces LLM_JUDGMENT basis."""
        from vetinari.skills.inspector_skill import _inspector_result_to_signal
        from vetinari.types import EvidenceBasis

        ir = InspectorResult(passed=True, grade="A", score=1.0, issues=[], suggestions=[])
        sig = _inspector_result_to_signal(ir, "code_review")

        assert sig.basis is EvidenceBasis.LLM_JUDGMENT
        assert sig.passed is True
        assert sig.score == 1.0

    def test_inspector_result_to_signal_populates_llm_judgment(self):
        """_inspector_result_to_signal populates llm_judgment with model_id and summary."""
        from vetinari.skills.inspector_skill import _inspector_result_to_signal

        ir = InspectorResult(passed=False, grade="F", score=0.4, issues=[], suggestions=[])
        sig = _inspector_result_to_signal(ir, "security_audit")

        assert sig.llm_judgment is not None
        assert sig.llm_judgment.model_id == "inspector_heuristic"
        assert "security_audit" in sig.llm_judgment.summary
        assert sig.llm_judgment.score == 0.4

    def test_inspector_result_to_signal_issues_encoded(self):
        """Issues from InspectorResult appear in OutcomeSignal.issues."""
        from vetinari.skills.inspector_skill import _inspector_result_to_signal

        issues = [
            ReviewIssue(severity="high", description="eval() usage", line=5),
            ReviewIssue(severity="low", description="missing docstring"),
        ]
        ir = InspectorResult(passed=False, grade="D", score=0.6, issues=issues)
        sig = _inspector_result_to_signal(ir, "security_audit")

        assert len(sig.issues) == 2
        assert any("eval" in i for i in sig.issues)
        assert any("line 5" in i for i in sig.issues)

    def test_inspector_result_to_signal_provenance_populated(self):
        """Provenance carries source and timestamp_utc."""
        from vetinari.skills.inspector_skill import _inspector_result_to_signal

        ir = InspectorResult()
        sig = _inspector_result_to_signal(ir, "code_review")

        assert sig.provenance is not None
        assert "inspector_skill" in sig.provenance.source
        assert sig.provenance.timestamp_utc
        assert "T" in sig.provenance.timestamp_utc

    # -- aggregate_outcome_signals -------------------------------------------

    def test_aggregate_empty_returns_unsupported(self):
        """Empty signal list yields UNSUPPORTED basis, passed=False (Rule 2 — fail-closed)."""
        from vetinari.skills.inspector_outcome import aggregate_outcome_signals
        from vetinari.types import EvidenceBasis

        sig = aggregate_outcome_signals([])

        assert sig.passed is False
        assert sig.basis is EvidenceBasis.UNSUPPORTED
        assert len(sig.issues) > 0

    def test_aggregate_tool_only_gives_tool_evidence(self):
        """All TOOL_EVIDENCE signals aggregate to TOOL_EVIDENCE."""
        from vetinari.agents.contracts import OutcomeSignal, Provenance, ToolEvidence
        from vetinari.skills.inspector_outcome import aggregate_outcome_signals
        from vetinari.types import EvidenceBasis

        te = ToolEvidence(
            tool_name="ruff",
            command="ruff check .",
            exit_code=0,
            stdout_snippet="no findings",
            stdout_hash="a" * 64,
            passed=True,
        )
        prov = Provenance(source="test", timestamp_utc="2026-01-01T00:00:00+00:00", tool_name="ruff")
        s1 = OutcomeSignal(
            passed=True, score=1.0, basis=EvidenceBasis.TOOL_EVIDENCE, tool_evidence=(te,), provenance=prov
        )
        s2 = OutcomeSignal(
            passed=True, score=0.9, basis=EvidenceBasis.TOOL_EVIDENCE, tool_evidence=(te,), provenance=prov
        )

        agg = aggregate_outcome_signals([s1, s2])

        assert agg.basis is EvidenceBasis.TOOL_EVIDENCE
        assert agg.passed is True
        assert agg.score == pytest.approx(0.95)

    def test_aggregate_llm_only_gives_advisory(self):
        """All LLM_JUDGMENT signals aggregate to LLM_JUDGMENT with advisory issue."""
        from vetinari.agents.contracts import LLMJudgment, OutcomeSignal, Provenance
        from vetinari.skills.inspector_outcome import aggregate_outcome_signals
        from vetinari.types import EvidenceBasis

        j = LLMJudgment(model_id="inspector_heuristic", summary="ok", score=0.8, reasoning="")
        prov = Provenance(source="test", timestamp_utc="2026-01-01T00:00:00+00:00")
        s = OutcomeSignal(passed=True, score=0.8, basis=EvidenceBasis.LLM_JUDGMENT, llm_judgment=j, provenance=prov)

        agg = aggregate_outcome_signals([s])

        assert agg.basis is EvidenceBasis.LLM_JUDGMENT
        assert any("no tool evidence" in i for i in agg.issues)

    def test_aggregate_hybrid_when_tool_and_llm_mixed(self):
        """Mix of TOOL_EVIDENCE and LLM_JUDGMENT aggregates to HYBRID."""
        from vetinari.agents.contracts import LLMJudgment, OutcomeSignal, Provenance, ToolEvidence
        from vetinari.skills.inspector_outcome import aggregate_outcome_signals
        from vetinari.types import EvidenceBasis

        te = ToolEvidence(
            tool_name="ruff",
            command="ruff check .",
            exit_code=0,
            stdout_snippet="no findings",
            stdout_hash="a" * 64,
            passed=True,
        )
        j = LLMJudgment(model_id="m1", summary="ok", score=0.7, reasoning="")
        prov = Provenance(source="test", timestamp_utc="2026-01-01T00:00:00+00:00")

        tool_sig = OutcomeSignal(
            passed=True, score=1.0, basis=EvidenceBasis.TOOL_EVIDENCE, tool_evidence=(te,), provenance=prov
        )
        llm_sig = OutcomeSignal(
            passed=True, score=0.7, basis=EvidenceBasis.LLM_JUDGMENT, llm_judgment=j, provenance=prov
        )

        agg = aggregate_outcome_signals([tool_sig, llm_sig])

        assert agg.basis is EvidenceBasis.HYBRID

    def test_aggregate_unsupported_propagates(self):
        """Any UNSUPPORTED signal in the list makes the whole aggregate UNSUPPORTED."""
        from vetinari.agents.contracts import OutcomeSignal, Provenance, ToolEvidence
        from vetinari.skills.inspector_outcome import aggregate_outcome_signals
        from vetinari.types import EvidenceBasis

        te = ToolEvidence(
            tool_name="ruff",
            command="ruff check .",
            exit_code=0,
            stdout_snippet="no findings",
            stdout_hash="a" * 64,
            passed=True,
        )
        prov = Provenance(source="test", timestamp_utc="2026-01-01T00:00:00+00:00")
        good = OutcomeSignal(
            passed=True, score=1.0, basis=EvidenceBasis.TOOL_EVIDENCE, tool_evidence=(te,), provenance=prov
        )
        bad = OutcomeSignal(
            passed=False, score=0.0, basis=EvidenceBasis.UNSUPPORTED, issues=("tool unavailable",), provenance=prov
        )

        agg = aggregate_outcome_signals([good, bad])

        assert agg.passed is False
        assert agg.basis is EvidenceBasis.UNSUPPORTED
        assert any("tool unavailable" in i for i in agg.issues)

    def test_aggregate_all_fail_gives_false(self):
        """Aggregate passed=False when any constituent signal is False."""
        from vetinari.agents.contracts import LLMJudgment, OutcomeSignal, Provenance
        from vetinari.skills.inspector_outcome import aggregate_outcome_signals
        from vetinari.types import EvidenceBasis

        j = LLMJudgment(model_id="m1", summary="bad", score=0.3, reasoning="issues found")
        prov = Provenance(source="test", timestamp_utc="2026-01-01T00:00:00+00:00")
        passing = OutcomeSignal(
            passed=True, score=0.9, basis=EvidenceBasis.LLM_JUDGMENT, llm_judgment=j, provenance=prov
        )
        failing = OutcomeSignal(
            passed=False,
            score=0.3,
            basis=EvidenceBasis.LLM_JUDGMENT,
            llm_judgment=j,
            issues=("bad output",),
            provenance=prov,
        )

        agg = aggregate_outcome_signals([passing, failing])

        assert agg.passed is False
