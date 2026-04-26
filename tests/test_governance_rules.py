"""Tests for governance and anti-pattern prevention rules.

Proves that rule checks fail when anti-pattern examples are reintroduced.
These are fixture-based tests — they verify the rules themselves work,
not specific production code.

This is step 0 of the pipeline: meta-verification of the rule enforcement layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vetinari.agents.contracts import AgentResult, AgentTask, VerificationResult
from vetinari.agents.multi_mode_agent import MultiModeAgent
from vetinari.types import AgentType

# -- Constants --

_RULES_ROOT = Path(__file__).parent.parent / ".claude" / "rules"
_ANTI_PATTERNS_PATH = _RULES_ROOT / "anti-patterns.md"
_GOVERNANCE_RULES_PATH = _RULES_ROOT / "governance-rules.md"


# -- Minimal concrete MultiModeAgent for verify() testing --


class _MinimalAgent(MultiModeAgent):
    """Minimal MultiModeAgent subclass used to exercise the verify() contract.

    Uses a single no-op mode so __init__ validation passes without needing
    any real inference infrastructure.
    """

    MODES = {"run": "_execute_run"}
    DEFAULT_MODE = "run"
    MODE_KEYWORDS = {"run": ["run", "execute"]}

    def __init__(self) -> None:
        super().__init__(AgentType.WORKER)

    def _execute_run(self, task: AgentTask) -> AgentResult:
        """No-op execution handler required by MODES registry."""
        return AgentResult(success=True, output="ok")


# ===================================================================
# Rule 2: No verifier passes by default without positive evidence
# ===================================================================


class TestVerifyDefaultFallback:
    """Rule 2: No verifier, gate, or safety check passes without explicit positive evidence."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Create a fresh agent instance for each test."""
        self._agent = _MinimalAgent()

    def test_none_fails_verification(self) -> None:
        """None output must not pass verification — no evidence of substance."""
        result = self._agent.verify(None)

        assert not result.passed, (
            "Rule 2 violation: verify(None) returned passed=True. A default-pass verifier certifies garbage."
        )
        assert result.score == 0.0, (
            f"Rule 2 violation: verify(None) returned score={result.score}. Score must be 0.0 when output is None."
        )

    def test_empty_dict_fails_verification(self) -> None:
        """Empty dicts must not pass verification — no evidence of content."""
        result = self._agent.verify({})

        assert not result.passed, (
            "Rule 2 violation: verify({}) returned passed=True. "
            "An empty dict contains no positive evidence of correctness."
        )

    def test_non_dict_string_fails_verification(self) -> None:
        """Arbitrary strings must not pass — verify() requires structured output."""
        result = self._agent.verify("garbage output that means nothing")

        assert not result.passed, (
            "Rule 2 violation: verify('garbage...') returned passed=True. "
            "Unstructured string output must not pass agent verification."
        )
        assert result.score == 0.0, f"Rule 2 violation: non-dict output returned score={result.score}, expected 0.0."

    def test_verify_returns_verification_result_type(self) -> None:
        """verify() must always return a VerificationResult, never None or a raw bool."""
        result = self._agent.verify(None)

        assert isinstance(result, VerificationResult), (
            f"verify() returned {type(result).__name__}, expected VerificationResult. "
            "Callers depend on .passed and .score attributes."
        )

    def test_populated_dict_can_pass_verification(self) -> None:
        """A non-empty dict with content may pass — proves the gate is not always-false."""
        result = self._agent.verify({"content": "substantive output", "sections": ["intro"]})

        # The base implementation passes non-empty dicts at score 0.7.
        # This test ensures the gate is a real discriminator, not always-false.
        assert isinstance(result, VerificationResult)
        assert result.score > 0.0, (
            "verify() returned score=0.0 for a populated dict. "
            "The gate must accept valid outputs, not reject everything."
        )


# ===================================================================
# Rule 6: No governance checker seeds its baseline from live code
# ===================================================================


class TestNoSelfSeedingGovernance:
    """Rule 6: No governance checker may seed its baseline from the live code it audits."""

    def test_anti_patterns_documents_self_seeding_rule(self) -> None:
        """The anti-patterns rules file must contain the self-seeding prohibition."""
        content = _ANTI_PATTERNS_PATH.read_text(encoding="utf-8")

        assert "self-seeding" in content.lower() or "seed" in content.lower(), (
            "Rule 6 violation: anti-patterns.md does not document the self-seeding "
            "governance rule. Add 'Self-seeding governance baseline' to the table."
        )

    def test_governance_rules_documents_checker_requirements(self) -> None:
        """governance-rules.md must specify that checkers use external baselines."""
        content = _GOVERNANCE_RULES_PATH.read_text(encoding="utf-8")

        assert "baseline" in content.lower(), (
            "governance-rules.md must document the external-baseline requirement for governance checkers (Rule 6)."
        )


# ===================================================================
# Rule 2 extension: module-level verifiers must also fail closed
# ===================================================================


class TestEntailmentCheckerFailsClosed:
    """Rule 2: vetinari.verification.entailment_checker MUST fail closed without a model.

    The module exposes verify_claim() and verify_claims() as module-level functions
    (not a class with .verify()), so it is not covered by the TestVerifyDefaultFallback
    fixture above. This class is the entailment checker's fixture entry in the
    verifier-contract suite.
    """

    def test_factual_claim_without_model_returns_trust_zero(self) -> None:
        """Without a loaded entailment model, factual claims fail closed at trust_score=0.0."""
        from vetinari.verification.claim_extractor import Claim, ClaimType
        from vetinari.verification.entailment_checker import verify_claim

        claim = Claim(
            text="The algorithm runs in O(n log n).",
            claim_type=ClaimType.FACTUAL,
            source_span=(0, 32),
            confidence=0.9,
        )

        verdict = verify_claim(claim)

        assert verdict.trust_score == 0.0, (
            f"Rule 2 violation: entailment_checker returned trust_score={verdict.trust_score} "
            "on a factual claim with no loaded model. It must fail closed at 0.0."
        )
        assert verdict.method == "stub"

    def test_empty_claims_aggregate_returns_trust_zero(self) -> None:
        """verify_claims([]) MUST report overall_trust=0.0, never 1.0 or 0.5."""
        from vetinari.verification.entailment_checker import verify_claims

        report = verify_claims([])

        assert report.overall_trust == 0.0, (
            f"Rule 2 violation: verify_claims([]) returned overall_trust={report.overall_trust}. "
            "An empty claim list must fail closed at 0.0."
        )


# ===================================================================
# Rule 9: Bounded tasks use deterministic methods, not LLM output
# ===================================================================


class TestDeterministicBoundary:
    """Rule 9: Validation, parsing, auth, and config tasks must be deterministic."""

    def test_json_parsing_is_deterministic(self) -> None:
        """JSON parsing must produce consistent results without model involvement."""
        valid_payload = '{"key": "value", "count": 42}'

        result = json.loads(valid_payload)

        assert result == {"key": "value", "count": 42}, (
            "json.loads must produce the exact same output every call — "
            "this is the deterministic baseline. Never replace with LLM parsing."
        )

    def test_json_parsing_rejects_invalid_input(self) -> None:
        """Invalid JSON must raise JSONDecodeError, not silently succeed or produce None."""
        invalid_payload = "not json at all"

        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_payload)

    def test_enum_lookup_is_deterministic(self) -> None:
        """Enum member lookup must be exact-match — no fuzzy/LLM resolution needed."""
        result = AgentType["WORKER"]

        assert result is AgentType.WORKER, (
            "Enum lookup by name must be deterministic. Never delegate enum resolution to model output."
        )

    def test_enum_lookup_raises_on_unknown_key(self) -> None:
        """Unknown enum keys must raise KeyError, not silently fall back."""
        with pytest.raises(KeyError):
            _ = AgentType["NONEXISTENT_AGENT_TYPE"]


# ===================================================================
# Meta-tests: rule files exist and are substantive
# ===================================================================


class TestGovernanceRulesExist:
    """Meta-test: governance rule files exist, are non-trivial, and contain expected sections."""

    def test_anti_patterns_file_exists_and_is_substantive(self) -> None:
        """anti-patterns.md must exist and contain enough content to be useful."""
        assert _ANTI_PATTERNS_PATH.exists(), (
            f"anti-patterns.md not found at {_ANTI_PATTERNS_PATH}. This file is required by the rules system."
        )
        content = _ANTI_PATTERNS_PATH.read_text(encoding="utf-8")
        assert len(content) > 500, (
            f"anti-patterns.md has only {len(content)} chars — too short to be useful. "
            "The file must document substantive anti-patterns."
        )

    def test_anti_patterns_has_governance_section(self) -> None:
        """anti-patterns.md must contain the Governance and Verification section."""
        content = _ANTI_PATTERNS_PATH.read_text(encoding="utf-8")

        assert "Governance and Verification Anti-Patterns" in content, (
            "anti-patterns.md is missing the 'Governance and Verification Anti-Patterns' section. "
            "This section documents the 10 governance theater rules."
        )

    def test_governance_rules_file_exists(self) -> None:
        """governance-rules.md must exist as the enforcement map."""
        assert _GOVERNANCE_RULES_PATH.exists(), (
            f"governance-rules.md not found at {_GOVERNANCE_RULES_PATH}. "
            "Create this file with the Rule Enforcement Map."
        )

    def test_governance_rules_has_enforcement_map(self) -> None:
        """governance-rules.md must contain the Rule Enforcement Map table."""
        content = _GOVERNANCE_RULES_PATH.read_text(encoding="utf-8")

        assert "Rule Enforcement Map" in content, (
            "governance-rules.md is missing the 'Rule Enforcement Map' table. "
            "This table maps each rule to its enforcement mechanism."
        )

    def test_governance_rules_has_deterministic_boundary_section(self) -> None:
        """governance-rules.md must document the deterministic vs model boundary."""
        content = _GOVERNANCE_RULES_PATH.read_text(encoding="utf-8")

        assert "Deterministic vs Model" in content, (
            "governance-rules.md is missing the 'Deterministic vs Model Boundaries' section. "
            "This section defines which task types must use deterministic methods."
        )

    def test_anti_patterns_contains_all_ten_governance_rules(self) -> None:
        """The governance section must cover all 10 rules from SESSION-27 item 27.8."""
        content = _ANTI_PATTERNS_PATH.read_text(encoding="utf-8")

        required_keywords = [
            "live code path",  # Rule 1: no claim without live path
            "default-pass",  # Rule 2: no default-pass verifiers
            "bad-behavior",  # Rule 3: no tests asserting bad behavior
            "migration",  # Rule 4: no false migration-complete
            "collapsed",  # Rule 5: no collapsed internal delegates
            "self-seeding",  # Rule 6: no self-seeding baselines
            "import-only",  # Rule 7: no import-only certification
            "pass-through",  # Rule 8: no unavailable-dependency pass-through
            "deterministic",  # Rule 9: deterministic tasks use deterministic methods
            "keyword heuristics",  # Rule 10: semantic tasks use model reasoning
        ]
        missing = [kw for kw in required_keywords if kw.lower() not in content.lower()]

        assert not missing, (
            f"anti-patterns.md governance section is missing rules covering: {missing}. "
            "All 10 SESSION-27.8 governance rules must be documented."
        )


# ===================================================================
# Rule 2 extension: score_with_signal must fail closed
# ===================================================================


class TestScoreWithSignalFailsClosed:
    """Rule 2: QualityScorer.score_with_signal must never pass without positive evidence.

    This class is the score_with_signal verifier-contract fixture entry: it
    proves the function satisfies the three fail-closed conditions that apply
    to all verifiers (empty → UNSUPPORTED, reject → UNSUPPORTED, passing path
    uses the correct basis).
    """

    def test_empty_output_returns_unsupported(self) -> None:
        """Empty output is rejected: basis=UNSUPPORTED, passed=False, score=0.0."""
        from vetinari.learning.quality_scorer import QualityScorer
        from vetinari.types import EvidenceBasis

        scorer = QualityScorer(adapter_manager=None)
        sig = scorer.score_with_signal(
            task_id="t-empty",
            model_id="m1",
            task_type="coding",
            task_description="Write code",
            output="",
            use_llm=False,
        )

        assert sig.basis is EvidenceBasis.UNSUPPORTED, (
            f"Rule 2 violation: empty output returned basis={sig.basis.value}, expected UNSUPPORTED."
        )
        assert sig.passed is False, "Rule 2 violation: empty output returned passed=True."
        assert sig.score == 0.0, f"Rule 2 violation: empty output returned score={sig.score}, expected 0.0."

    def test_none_output_returns_unsupported(self) -> None:
        """None-equivalent output (whitespace only) is rejected with UNSUPPORTED basis."""
        from vetinari.learning.quality_scorer import QualityScorer
        from vetinari.types import EvidenceBasis

        scorer = QualityScorer(adapter_manager=None)
        sig = scorer.score_with_signal(
            task_id="t-ws",
            model_id="m1",
            task_type="coding",
            task_description="Write code",
            output="   ",
            use_llm=False,
        )

        # Whitespace-only output strips to "" — treated as empty/rejected.
        assert sig.basis is EvidenceBasis.UNSUPPORTED, (
            f"Rule 2 violation: whitespace-only output returned basis={sig.basis.value}, expected UNSUPPORTED."
        )
        assert sig.passed is False, "Rule 2 violation: whitespace-only output returned passed=True."

    def test_passing_path_carries_llm_judgment_basis(self) -> None:
        """A populated output returns basis=LLM_JUDGMENT, not TOOL_EVIDENCE.

        score_with_signal is an LLM-as-judge heuristic, so a passing result
        must carry LLM_JUDGMENT basis — never TOOL_EVIDENCE (which would be a
        false claim about evidence type).
        """
        from vetinari.learning.quality_scorer import QualityScorer
        from vetinari.types import EvidenceBasis

        scorer = QualityScorer(adapter_manager=None)
        output = "def add(a: int, b: int) -> int:\n    '''Return a + b.'''\n    return a + b\n"
        sig = scorer.score_with_signal(
            task_id="t-pass",
            model_id="m1",
            task_type="coding",
            task_description="Write an add function",
            output=output,
            use_llm=False,
        )

        assert sig.basis is EvidenceBasis.LLM_JUDGMENT, (
            f"Passing path must use LLM_JUDGMENT basis, got {sig.basis.value}. "
            "score_with_signal is heuristic/LLM-as-judge — never TOOL_EVIDENCE."
        )
        assert sig.provenance is not None, "Passing path must include provenance metadata."


# ===================================================================
# Rule 2 extension: aggregate_outcome_signals must fail closed
# ===================================================================


class TestAggregateOutcomeSignalsFailsClosed:
    """Rule 2: aggregate_outcome_signals must never inflate trust.

    Proves three invariants:
    1. Empty input -> UNSUPPORTED, passed=False (fail-closed baseline).
    2. Any UNSUPPORTED constituent -> aggregate is UNSUPPORTED, passed=False
       (no silent trust inflation).
    3. All-passing tool-evidence signals -> TOOL_EVIDENCE, passed=True
       (the gate is a real discriminator, not always-false).
    """

    def test_empty_input_returns_unsupported(self) -> None:
        """aggregate_outcome_signals([]) must return UNSUPPORTED, passed=False."""
        from vetinari.skills.inspector_outcome import aggregate_outcome_signals
        from vetinari.types import EvidenceBasis

        result = aggregate_outcome_signals([])

        assert result.basis is EvidenceBasis.UNSUPPORTED, (
            f"Rule 2 violation: aggregate_outcome_signals([]) returned basis={result.basis.value}. "
            "Empty input must fail closed at UNSUPPORTED."
        )
        assert result.passed is False, (
            "Rule 2 violation: aggregate_outcome_signals([]) returned passed=True. Empty input must fail closed."
        )
        assert result.score == 0.0, (
            f"Rule 2 violation: aggregate_outcome_signals([]) returned score={result.score}, expected 0.0."
        )

    def test_any_unsupported_constituent_propagates(self) -> None:
        """A mix of passing and UNSUPPORTED signals must aggregate to UNSUPPORTED.

        This is the core no-trust-inflation invariant: one bad apple spoils the
        aggregate.  Downstream consumers must not see a passing aggregate when
        any constituent claim is unsupported.
        """
        from vetinari.agents.contracts import OutcomeSignal, Provenance
        from vetinari.skills.inspector_outcome import aggregate_outcome_signals
        from vetinari.types import EvidenceBasis

        ts = "2026-01-01T00:00:00+00:00"
        passing_tool = OutcomeSignal(
            passed=True,
            score=1.0,
            basis=EvidenceBasis.TOOL_EVIDENCE,
            provenance=Provenance(source="test", timestamp_utc=ts, tool_name="pytest"),
        )
        unsupported = OutcomeSignal(
            passed=False,
            score=0.0,
            basis=EvidenceBasis.UNSUPPORTED,
            issues=("no citation",),
            provenance=Provenance(source="test", timestamp_utc=ts),
        )

        result = aggregate_outcome_signals([passing_tool, unsupported])

        assert result.basis is EvidenceBasis.UNSUPPORTED, (
            f"Rule 2 violation: UNSUPPORTED constituent did not propagate to aggregate "
            f"(got basis={result.basis.value}). Trust must not be inflated silently."
        )
        assert result.passed is False, (
            "Rule 2 violation: aggregate returned passed=True despite UNSUPPORTED constituent."
        )

    def test_all_passing_tool_evidence_aggregates_pass(self) -> None:
        """All-passing tool-evidence signals must produce passed=True, basis=TOOL_EVIDENCE.

        Verifies the gate is a real discriminator, not always-false.
        """
        from vetinari.agents.contracts import OutcomeSignal, Provenance, ToolEvidence
        from vetinari.skills.inspector_outcome import aggregate_outcome_signals
        from vetinari.types import EvidenceBasis

        ts = "2026-01-01T00:00:00+00:00"
        tool_ev = ToolEvidence(
            tool_name="ruff",
            command="ruff check .",
            exit_code=0,
            stdout_snippet="All checks passed.",
            stdout_hash="abc123",
            passed=True,
        )
        sig1 = OutcomeSignal(
            passed=True,
            score=1.0,
            basis=EvidenceBasis.TOOL_EVIDENCE,
            tool_evidence=(tool_ev,),
            provenance=Provenance(source="test", timestamp_utc=ts, tool_name="ruff"),
        )
        sig2 = OutcomeSignal(
            passed=True,
            score=0.9,
            basis=EvidenceBasis.TOOL_EVIDENCE,
            tool_evidence=(tool_ev,),
            provenance=Provenance(source="test", timestamp_utc=ts, tool_name="pytest"),
        )

        result = aggregate_outcome_signals([sig1, sig2])

        assert result.passed is True, (
            f"Rule 2 violation: all-passing tool-evidence signals returned passed=False "
            f"(score={result.score}, issues={result.issues}). The gate must accept valid outputs."
        )
        assert result.basis is EvidenceBasis.TOOL_EVIDENCE, (
            f"All TOOL_EVIDENCE constituents must aggregate to TOOL_EVIDENCE, got {result.basis.value}."
        )
