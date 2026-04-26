"""Verification hook and claim-evidence gate tests.

Covers two distinct concerns:
1. SESSION-34D2 script-level hook gates (hook_verification_gate.py,
   hook_claim_evidence_gate.py) — the existing tests below.
2. SESSION-05 SHARD-03 hook-level fail-closed behavior for the
   ``vetinari.verification.claim_verifier`` module — added in the
   ``TestClaimVerifierHookFailClosed`` class at the bottom of this file.

The second set of tests exercises the verifier at the hook level (not just as
an isolated unit), confirming that each of the six failure modes produces
``passed=False`` when wired through the real ``verify_claim_fail_closed``
composite gate.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vetinari.agents.contracts import AttestedArtifact, OutcomeSignal, Provenance
from vetinari.types import ArtifactKind, EvidenceBasis
from vetinari.verification.cascade_report import build_cascade_report
from vetinari.verification.claim_verifier import verify_claim_fail_closed

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

verification_gate = importlib.import_module("hook_verification_gate")
claim_gate = importlib.import_module("hook_claim_evidence_gate")
test_quality = importlib.import_module("check_test_quality")


def _write_session(tmp_path: Path, records: list[dict], session_id: str = "session-34d2") -> str:
    path = tmp_path / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return session_id


def _run_verification_gate(tmp_path: Path, records: list[dict]) -> int:
    session_id = _write_session(tmp_path, records)
    with (
        patch.object(verification_gate, "SESSION_DIR", tmp_path),
        patch("sys.stdin", io.StringIO(json.dumps({"session_id": session_id}))),
    ):
        return verification_gate.main()


def _run_claim_gate(tmp_path: Path, records: list[dict], monkeypatch: pytest.MonkeyPatch) -> int:
    session_id = _write_session(tmp_path, records)
    monkeypatch.setattr(claim_gate, "SESSION_DIR", tmp_path)
    monkeypatch.setattr(claim_gate, "MIN_WORDS", 1)
    with patch("sys.stdin", io.StringIO(json.dumps({"session_id": session_id}))):
        return claim_gate.main()


def test_verification_hook_rejects_command_text_without_result() -> None:
    content_items = [
        {
            "type": "tool_use",
            "name": "Bash",
            "input": {"command": "python -m pytest tests -q"},
        }
    ]
    assert verification_gate.has_verifying_tool_use(content_items) is True
    assert verification_gate.has_verifying_signal(content_items, []) is False


@pytest.mark.parametrize("phrase", ["pytest passed", "tests passed", "ruff passed", "verification passed"])
def test_verification_hook_rejects_pass_phrases_without_result(tmp_path: Path, phrase: str) -> None:
    rc = _run_verification_gate(
        tmp_path,
        [{"type": "assistant", "content": [{"type": "text", "text": phrase}]}],
    )
    assert rc == 2


def test_verification_hook_accepts_structured_tool_result_content(tmp_path: Path) -> None:
    rc = _run_verification_gate(
        tmp_path,
        [
            {
                "type": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [{"type": "text", "text": "5 passed in 0.12s"}],
                    }
                ],
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "tests passed"}]},
            },
        ],
    )
    assert rc == 0


def test_verification_hook_rejects_failed_output_with_pass_words(tmp_path: Path) -> None:
    rc = _run_verification_gate(
        tmp_path,
        [
            {
                "type": "user",
                "content": [{"type": "tool_result", "content": "1 failed, 5 passed in 0.12s"}],
            },
            {"type": "assistant", "content": [{"type": "text", "text": "tests passed"}]},
        ],
    )
    assert rc == 2


def test_verification_hook_rejects_stale_archive_output(tmp_path: Path) -> None:
    rc = _run_verification_gate(
        tmp_path,
        [
            {
                "type": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": "stale convergence archive header\n5 passed in 0.12s",
                    }
                ],
            },
            {"type": "assistant", "content": [{"type": "text", "text": "verification passed"}]},
        ],
    )
    assert rc == 2


def test_claim_evidence_requires_full_path_not_extension_capture() -> None:
    assert claim_gate.PATH_RX.findall("The file scripts/hook_claim_evidence_gate.py exists") == [
        "scripts/hook_claim_evidence_gate.py"
    ]


def test_claim_evidence_rejects_self_matching_claim_path() -> None:
    claim = "The file scripts/hook_claim_evidence_gate.py is implemented in the hook."
    assert claim_gate.is_supported(claim, {("Grep", "hook_claim_evidence_gate")}) is False


def test_claim_evidence_accepts_root_content_when_tool_result_matches_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc = _run_claim_gate(
        tmp_path,
        [
            {
                "type": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "Read",
                        "input": {"file_path": "scripts/hook_claim_evidence_gate.py"},
                    }
                ],
            },
            {
                "type": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": [{"type": "text", "text": "def is_supported(...): ..."}],
                    }
                ],
            },
            {
                "type": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The file scripts/hook_claim_evidence_gate.py is implemented in the hook.",
                    }
                ],
            },
        ],
        monkeypatch,
    )
    assert rc == 0


def test_claim_evidence_rejects_message_content_when_evidence_path_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc = _run_claim_gate(
        tmp_path,
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Read",
                            "input": {"file_path": "scripts/other_hook.py"},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": [{"type": "text", "text": "other hook output"}],
                    }
                ],
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "The file scripts/hook_claim_evidence_gate.py is implemented in the hook.",
                        }
                    ]
                },
            },
        ],
        monkeypatch,
    )
    assert rc == 2


def test_coverage_gate_rejects_bare_pytest_raises_reference() -> None:
    lines = [
        "def test_bad():",
        '    "assert result == 1"',
        "    # assert result == 1",
        "    pytest.raises",
    ]
    violations = test_quality.check_zero_assert(Path("tests/test_bad.py"), lines)
    assert any(code == "VET241" for _, code, _, _ in violations)


def test_coverage_gate_rejects_uncalled_pytest_raises_context_factory() -> None:
    lines = [
        "def test_bad():",
        "    pytest.raises(ValueError)",
    ]
    violations = test_quality.check_zero_assert(Path("tests/test_bad.py"), lines)
    assert any(code == "VET241" for _, code, _, _ in violations)


@pytest.mark.parametrize(
    "line",
    [
        "    with pytest.raises(ValueError):",
        '    pytest.raises(ValueError, int, "not-an-int")',
    ],
)
def test_coverage_gate_accepts_executable_pytest_raises_forms(line: str) -> None:
    lines = [
        "def test_good():",
        line,
        "        raise ValueError('bad')",
    ]
    violations = test_quality.check_zero_assert(Path("tests/test_good.py"), lines)
    assert not any(code == "VET241" for _, code, _, _ in violations)


# ---------------------------------------------------------------------------
# SESSION-05 SHARD-03: Hook-level fail-closed tests (Task 3.3)
# These tests exercise ``verify_claim_fail_closed`` as a composed hook, not
# the individual sub-checks in isolation.  They confirm that each of the six
# failure modes produces passed=False when wired through the real gate.
# ---------------------------------------------------------------------------


def _fresh_signal() -> OutcomeSignal:
    """Build a minimal passing TOOL_EVIDENCE signal with current provenance."""
    return OutcomeSignal(
        passed=True,
        score=1.0,
        basis=EvidenceBasis.TOOL_EVIDENCE,
        provenance=Provenance(
            source="test-hook",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )


class TestClaimVerifierHookFailClosed:
    """Hook-level fail-closed behavior via verify_claim_fail_closed composite gate.

    Each test exercises a distinct failure mode through the REAL composite hook
    (not a mock), confirming that passed=False is produced and not hidden at any
    aggregation step.
    """

    def test_hook_rejects_missing_citation(self) -> None:
        """Hook failure mode 1: signal with no Provenance -> passed=False."""
        signal = OutcomeSignal(
            passed=True,
            score=1.0,
            basis=EvidenceBasis.TOOL_EVIDENCE,
            provenance=None,
        )
        result = verify_claim_fail_closed(signal)
        assert result.passed is False
        assert result.basis is EvidenceBasis.UNSUPPORTED
        assert result.score == 0.0  # Rule 2 invariant

    def test_hook_rejects_stale_evidence(self) -> None:
        """Hook failure mode 2: evidence older than freshness window -> passed=False."""
        old_ts = (datetime.now(timezone.utc) - timedelta(seconds=7200)).isoformat()
        signal = OutcomeSignal(
            passed=True,
            score=1.0,
            basis=EvidenceBasis.TOOL_EVIDENCE,
            provenance=Provenance(source="test", timestamp_utc=old_ts),
        )
        result = verify_claim_fail_closed(signal, freshness_window_seconds=3600)
        assert result.passed is False
        assert result.basis is EvidenceBasis.UNSUPPORTED
        assert "stale evidence" in result.issues[0]

    def test_hook_rejects_nonexistent_file_claim(self, tmp_path: Path) -> None:
        """Hook failure mode 3: claimed file does not exist -> passed=False."""
        signal = _fresh_signal()
        missing = str(tmp_path / "ghost.py")
        result = verify_claim_fail_closed(signal, claimed_path=missing)
        assert result.passed is False
        assert result.basis is EvidenceBasis.UNSUPPORTED
        assert "file claim unverified" in result.issues[0]

    def test_hook_rejects_hash_mismatch(self, tmp_path: Path) -> None:
        """Hook failure mode 3b: file exists but hash mismatches -> passed=False."""
        f = tmp_path / "code.py"
        f.write_text("real content", encoding="utf-8")
        signal = _fresh_signal()
        result = verify_claim_fail_closed(signal, claimed_path=str(f), expected_sha256="deadbeef" * 8)
        assert result.passed is False
        assert "hash mismatch" in result.issues[0]

    def test_hook_rejects_entailment_contradiction(self) -> None:
        """Hook failure mode 4: entailment contradiction -> passed=False."""
        signal = _fresh_signal()
        result = verify_claim_fail_closed(signal, entailment_passed=False, claim_text="no errors found")
        assert result.passed is False
        assert result.basis is EvidenceBasis.UNSUPPORTED
        assert "entailment contradiction" in result.issues[0]

    def test_hook_rejects_bare_human_attested(self) -> None:
        """Hook failure mode 5: bare HUMAN_ATTESTED on factual claim -> passed=False.

        We construct the signal with INTENT_CONFIRMATION to satisfy the
        OutcomeSignal constructor invariant, then call the hook with
        is_intent_confirmation=False to exercise the factual-claim path.
        """
        signal = OutcomeSignal(
            passed=True,
            score=1.0,
            basis=EvidenceBasis.HUMAN_ATTESTED,
            use_case="INTENT_CONFIRMATION",
            provenance=Provenance(source="test", timestamp_utc=datetime.now(timezone.utc).isoformat()),
        )
        result = verify_claim_fail_closed(signal, is_intent_confirmation=False)
        assert result.passed is False
        assert result.basis is EvidenceBasis.UNSUPPORTED
        assert "human attestation requires a concrete attested artifact" in result.issues[0]

    def test_hook_flags_llm_only_advisory_visible(self) -> None:
        """Hook failure mode 6: LLM-only on high-accuracy path adds visible advisory.

        The signal is NOT forced to passed=False (spec: advisory only), but the
        advisory MUST appear in issues so it survives aggregation.
        """
        signal = OutcomeSignal(
            passed=True,
            score=0.85,
            basis=EvidenceBasis.LLM_JUDGMENT,
            provenance=Provenance(source="test", timestamp_utc=datetime.now(timezone.utc).isoformat()),
        )
        result = verify_claim_fail_closed(signal, high_accuracy=True)
        assert result.basis is EvidenceBasis.LLM_JUDGMENT
        assert any("no tool evidence" in issue for issue in result.issues)

    def test_hook_cascade_does_not_convert_false_to_true(self) -> None:
        """Cascade aggregation must NOT silently convert passed=False to passed=True.

        Anti-branch-accepting test: we assert the aggregate is False and will
        FAIL if the aggregation step starts returning True for mixed batches.
        """
        good = _fresh_signal()
        bad = OutcomeSignal(
            passed=False,
            score=0.0,
            basis=EvidenceBasis.UNSUPPORTED,
            issues=("no citation attached",),
            provenance=Provenance(source="test", timestamp_utc=datetime.now(timezone.utc).isoformat()),
        )
        report = build_cascade_report([("good", "good claim", good), ("bad", "bad claim", bad)])
        # Aggregation must not inflate the failed signal to True
        assert report.aggregate_signal.passed is False
        assert report.aggregate_signal.basis is EvidenceBasis.UNSUPPORTED
        # unsupported_claims must surface the bad record — not hidden in aggregate
        bad_ids = [r.claim_id for r in report.unsupported_claims]
        assert "bad" in bad_ids
        assert "good" not in bad_ids
