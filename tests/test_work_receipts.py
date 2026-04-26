"""Regression tests for the WorkReceipt subsystem.

Covers the SHARD-01 acceptance criteria: invariant enforcement on
``awaiting_user``/``awaiting_reason``; round-trip serialisation;
atomic-write behaviour under simulated mid-write failure;
``find_awaiting`` filtering; and event emission on append.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from vetinari.agents.contracts import OutcomeSignal, Provenance, ToolEvidence
from vetinari.events import EventBus
from vetinari.receipts import (
    ReceiptAppended,
    WorkReceipt,
    WorkReceiptKind,
    WorkReceiptStore,
)
from vetinari.types import AgentType, EvidenceBasis


def _make_outcome(*, passed: bool = True) -> OutcomeSignal:
    """Construct a representative tool-backed OutcomeSignal for receipts."""
    return OutcomeSignal(
        passed=passed,
        score=0.92 if passed else 0.0,
        basis=EvidenceBasis.TOOL_EVIDENCE,
        tool_evidence=(
            ToolEvidence(
                tool_name="pytest",
                command="pytest tests/test_work_receipts.py -q",
                exit_code=0 if passed else 1,
                stdout_snippet="1 passed in 0.05s" if passed else "1 failed",
                stdout_hash="cafebabe" * 8,
                passed=passed,
            ),
        ),
        provenance=Provenance(
            source="vetinari.receipts.tests",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            tool_name="pytest",
        ),
    )


def _make_receipt(
    *,
    project_id: str = "proj-test",
    awaiting_user: bool = False,
    awaiting_reason: str | None = None,
    kind: WorkReceiptKind = WorkReceiptKind.WORKER_TASK,
    outcome: OutcomeSignal | None = None,
    linked_claim_ids: tuple[str, ...] = (),
    inputs_summary: str = "task=lint vetinari/receipts/",
    outputs_summary: str = "ruff check passed; 0 errors",
) -> WorkReceipt:
    """Construct a representative WorkReceipt with sensible defaults."""
    return WorkReceipt(
        project_id=project_id,
        agent_id="worker-instance-001",
        agent_type=AgentType.WORKER,
        kind=kind,
        outcome=outcome if outcome is not None else _make_outcome(),
        inputs_summary=inputs_summary,
        outputs_summary=outputs_summary,
        awaiting_user=awaiting_user,
        awaiting_reason=awaiting_reason,
        linked_claim_ids=linked_claim_ids,
    )


# -- WorkReceipt construction invariants --------------------------------------


class TestWorkReceiptConstruction:
    """Construction-time invariants for the WorkReceipt record."""

    def test_awaiting_user_true_with_empty_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="awaiting_user=True requires a non-empty awaiting_reason"):
            _make_receipt(awaiting_user=True, awaiting_reason="")

    def test_awaiting_user_true_with_whitespace_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="awaiting_user=True requires a non-empty awaiting_reason"):
            _make_receipt(awaiting_user=True, awaiting_reason="   \t\n")

    def test_awaiting_user_true_with_none_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="awaiting_user=True requires a non-empty awaiting_reason"):
            _make_receipt(awaiting_user=True, awaiting_reason=None)

    def test_awaiting_user_true_with_real_reason_succeeds(self) -> None:
        receipt = _make_receipt(
            awaiting_user=True,
            awaiting_reason="plan reviewer refused -- TS-14 non-goals matched",
        )
        assert receipt.awaiting_user is True
        assert receipt.awaiting_reason == "plan reviewer refused -- TS-14 non-goals matched"

    def test_inputs_summary_over_200_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="inputs_summary exceeds 200 chars"):
            _make_receipt(inputs_summary="x" * 201)

    def test_outputs_summary_over_200_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="outputs_summary exceeds 200 chars"):
            _make_receipt(outputs_summary="x" * 201)

    def test_repr_shows_only_identifying_fields(self) -> None:
        receipt = _make_receipt()
        rendered = repr(receipt)
        assert "receipt_id=" in rendered
        assert "kind='worker_task'" in rendered
        assert "passed=True" in rendered
        assert "awaiting_user=False" in rendered
        # Internal large fields must not bleed into the repr.
        assert "tool_evidence" not in rendered
        assert "provenance" not in rendered

    def test_started_finished_default_to_iso8601_utc(self) -> None:
        receipt = _make_receipt()
        # ISO-8601 UTC includes the "+00:00" suffix when produced via datetime.now(timezone.utc).isoformat()
        assert receipt.started_at_utc.endswith("+00:00")
        assert receipt.finished_at_utc.endswith("+00:00")


# -- WorkReceiptStore append + read round trip --------------------------------


@pytest.fixture
def isolated_bus() -> EventBus:
    """An isolated EventBus that does not poison the global singleton."""
    return EventBus()


@pytest.fixture
def store(tmp_path: Path, isolated_bus: EventBus) -> WorkReceiptStore:
    """A receipts store rooted at a per-test tmp_path with its own bus."""
    return WorkReceiptStore(repo_root=tmp_path, event_bus=isolated_bus)


class TestWorkReceiptStoreRoundTrip:
    """Append-then-read parity for the JSONL store."""

    def test_append_creates_per_project_file(
        self,
        store: WorkReceiptStore,
        tmp_path: Path,
    ) -> None:
        receipt = _make_receipt(project_id="proj-A")
        store.append(receipt)
        expected_path = tmp_path / "outputs" / "receipts" / "proj-A" / "receipts.jsonl"
        assert expected_path.exists()
        assert expected_path.read_text(encoding="utf-8").strip().count("\n") == 0  # one line

    def test_iter_receipts_returns_in_append_order(
        self,
        store: WorkReceiptStore,
    ) -> None:
        first = _make_receipt(project_id="proj-B", inputs_summary="first")
        second = _make_receipt(
            project_id="proj-B",
            inputs_summary="second",
            kind=WorkReceiptKind.PLAN_ROUND,
        )
        third = _make_receipt(
            project_id="proj-B",
            inputs_summary="third",
            kind=WorkReceiptKind.INSPECTOR_PASS,
        )
        for r in (first, second, third):
            store.append(r)

        retrieved = list(store.iter_receipts("proj-B"))
        assert [r.inputs_summary for r in retrieved] == ["first", "second", "third"]
        assert [r.kind for r in retrieved] == [
            WorkReceiptKind.WORKER_TASK,
            WorkReceiptKind.PLAN_ROUND,
            WorkReceiptKind.INSPECTOR_PASS,
        ]

    def test_iter_receipts_round_trips_auxiliary_agent_types(
        self,
        store: WorkReceiptStore,
    ) -> None:
        """Auxiliary actor labels (TRAINING/RELEASE) round-trip through JSONL.

        Per ADR-0103, training and release runners get their own
        ``AgentType`` values rather than masquerading as ``WORKER``.
        The serialiser must preserve those values so downstream consumers
        (Control Center filters, audit tools) see the same actor label
        on read that the producer wrote.
        """
        release_receipt = WorkReceipt(
            project_id="proj-aux",
            agent_id="release-doctor:0.0.0-test",
            agent_type=AgentType.RELEASE,
            kind=WorkReceiptKind.RELEASE_STEP,
            outcome=_make_outcome(),
            inputs_summary="release pipeline step: build",
            outputs_summary="version=0.0.0-test | step=build | success=True",
        )
        training_receipt = WorkReceipt(
            project_id="proj-aux",
            agent_id="training-runner:run_xyz",
            agent_type=AgentType.TRAINING,
            kind=WorkReceiptKind.TRAINING_STEP,
            outcome=_make_outcome(),
            inputs_summary="base_model=stub | algo=qlora",
            outputs_summary="run_id=run_xyz | success=True",
        )
        store.append(release_receipt)
        store.append(training_receipt)

        loaded = list(store.iter_receipts("proj-aux"))
        assert [r.agent_type for r in loaded] == [AgentType.RELEASE, AgentType.TRAINING]
        assert [r.kind for r in loaded] == [
            WorkReceiptKind.RELEASE_STEP,
            WorkReceiptKind.TRAINING_STEP,
        ]

    def test_iter_receipts_round_trips_outcome_evidence(
        self,
        store: WorkReceiptStore,
    ) -> None:
        receipt = _make_receipt(
            project_id="proj-C",
            outcome=_make_outcome(passed=False),
            linked_claim_ids=("claim-001", "claim-002"),
        )
        store.append(receipt)
        loaded = next(store.iter_receipts("proj-C"))

        assert loaded.outcome.passed is False
        assert loaded.outcome.basis is EvidenceBasis.TOOL_EVIDENCE
        assert len(loaded.outcome.tool_evidence) == 1
        assert loaded.outcome.tool_evidence[0].tool_name == "pytest"
        assert loaded.outcome.tool_evidence[0].exit_code == 1
        assert loaded.linked_claim_ids == ("claim-001", "claim-002")
        assert loaded.outcome.provenance is not None
        assert loaded.outcome.provenance.source == "vetinari.receipts.tests"

    def test_iter_receipts_on_missing_project_yields_empty(
        self,
        store: WorkReceiptStore,
    ) -> None:
        retrieved = list(store.iter_receipts("nonexistent"))
        assert retrieved == []

    def test_receipts_path_rejects_empty_project_id(
        self,
        store: WorkReceiptStore,
    ) -> None:
        with pytest.raises(ValueError, match="project_id must be a non-empty string"):
            store.receipts_path("")

    def test_append_rejects_non_receipt(self, store: WorkReceiptStore) -> None:
        with pytest.raises(TypeError, match="WorkReceipt"):
            store.append("not a receipt")  # type: ignore[arg-type]

    def test_receipts_root_property_returns_outputs_receipts_dir(
        self,
        store: WorkReceiptStore,
        tmp_path: Path,
    ) -> None:
        """``receipts_root`` is the cross-project parent directory."""
        assert store.receipts_root == tmp_path / "outputs" / "receipts"

    def test_receipts_path_is_under_receipts_root(
        self,
        store: WorkReceiptStore,
    ) -> None:
        """A project's path must live inside the receipts_root tree."""
        path = store.receipts_path("proj-root-test")
        assert store.receipts_root in path.parents


# -- find_awaiting ------------------------------------------------------------


class TestFindAwaiting:
    """Filtering for the Attention track."""

    def test_returns_only_awaiting_receipts(self, store: WorkReceiptStore) -> None:
        store.append(_make_receipt(project_id="proj-D"))
        store.append(
            _make_receipt(
                project_id="proj-D",
                awaiting_user=True,
                awaiting_reason="inspector surfaced unsupported claims: 2",
            )
        )
        store.append(_make_receipt(project_id="proj-D"))
        store.append(
            _make_receipt(
                project_id="proj-D",
                awaiting_user=True,
                awaiting_reason="plan reviewer refused -- destructive scope",
            )
        )

        awaiting = store.find_awaiting("proj-D")

        assert len(awaiting) == 2
        assert all(r.awaiting_user for r in awaiting)
        assert awaiting[0].awaiting_reason == "inspector surfaced unsupported claims: 2"
        assert awaiting[1].awaiting_reason == "plan reviewer refused -- destructive scope"

    def test_returns_empty_when_no_awaiting(self, store: WorkReceiptStore) -> None:
        store.append(_make_receipt(project_id="proj-E"))
        store.append(_make_receipt(project_id="proj-E"))
        assert store.find_awaiting("proj-E") == []


# -- write durability + corruption-resistant read -----------------------------


class TestAppendDurability:
    """Durability + corruption-resistance contract for the JSONL append.

    The store uses O(1) append-with-fsync. A process crash after a
    partial write may leave a torn last line; ``iter_receipts`` skips
    malformed lines so the surviving records remain readable. These
    tests exercise both halves of that contract.
    """

    def test_open_failure_leaves_existing_file_intact(
        self,
        store: WorkReceiptStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Write one good receipt first so a baseline file exists.
        store.append(_make_receipt(project_id="proj-F", inputs_summary="ok"))
        path = store.receipts_path("proj-F")
        original = path.read_text(encoding="utf-8")
        assert original.count("\n") == 1

        # Force ``Path.open`` on the receipts path to fail BEFORE any
        # bytes are written, proving the existing file is untouched
        # when an append cannot proceed.
        real_open = Path.open

        def _boom_open(self_: Path, *args: object, **kwargs: object) -> object:
            if self_ == path:
                raise OSError("simulated open failure")
            return real_open(self_, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "open", _boom_open)

        with pytest.raises(OSError, match="simulated open failure"):
            store.append(_make_receipt(project_id="proj-F", inputs_summary="boom"))

        # Restore Path.open before reading so the assertion uses real I/O.
        monkeypatch.setattr(Path, "open", real_open)
        assert path.read_text(encoding="utf-8") == original
        # No leftover tempfiles from the legacy rename strategy either.
        leftovers = [p for p in path.parent.iterdir() if p.name.startswith(".receipts_tmp_")]
        assert leftovers == []

    def test_iter_receipts_skips_malformed_last_line(
        self,
        store: WorkReceiptStore,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A torn final line must not poison reads of surviving records."""
        import logging

        store.append(_make_receipt(project_id="proj-torn", inputs_summary="first"))
        store.append(_make_receipt(project_id="proj-torn", inputs_summary="second"))

        # Simulate a torn last line by appending invalid JSON.
        path = store.receipts_path("proj-torn")
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"this is not": "valid", broken-json\n')

        with caplog.at_level(logging.WARNING, logger="vetinari.receipts.store"):
            survivors = list(store.iter_receipts("proj-torn"))

        assert [r.inputs_summary for r in survivors] == ["first", "second"]
        assert any("malformed receipts line" in r.message for r in caplog.records)

    def test_append_is_constant_time_no_full_rewrite(
        self,
        store: WorkReceiptStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Append must NOT re-read the existing file (regression guard)."""
        from vetinari.receipts import store as store_module

        # Seed a non-trivial file so a regression to full-file rewrite
        # would have observable cost. Then forbid Path.read_text on the
        # receipts path during the next append.
        for i in range(5):
            store.append(_make_receipt(project_id="proj-O1", inputs_summary=f"r{i}"))

        path = store.receipts_path("proj-O1")
        real_read_text = Path.read_text

        def _no_read(self_: Path, *args: object, **kwargs: object) -> str:
            if self_ == path:
                raise AssertionError("append() must not read the existing receipts file")
            return real_read_text(self_, *args, **kwargs)  # type: ignore[arg-type,return-value]

        monkeypatch.setattr(Path, "read_text", _no_read)
        # The append must succeed without re-reading the existing JSONL.
        store.append(_make_receipt(project_id="proj-O1", inputs_summary="r5"))

        monkeypatch.setattr(Path, "read_text", real_read_text)
        retrieved = list(store.iter_receipts("proj-O1"))
        assert [r.inputs_summary for r in retrieved] == ["r0", "r1", "r2", "r3", "r4", "r5"]
        # Reference store_module so the import linter does not flag it.
        assert store_module.WorkReceiptStore is WorkReceiptStore


# -- event publication --------------------------------------------------------


class TestReceiptEvents:
    """The append path must publish ``receipt.appended``."""

    def test_append_publishes_receipt_appended(
        self,
        store: WorkReceiptStore,
        isolated_bus: EventBus,
    ) -> None:
        seen: list[ReceiptAppended] = []
        isolated_bus.subscribe(ReceiptAppended, seen.append)

        receipt = _make_receipt(
            project_id="proj-G",
            awaiting_user=True,
            awaiting_reason="awaiting human approval for destructive op",
        )
        store.append(receipt)

        assert len(seen) == 1
        event = seen[0]
        assert event.event_type == "receipt.appended"
        assert event.project_id == "proj-G"
        assert event.receipt_id == receipt.receipt_id
        assert event.kind == WorkReceiptKind.WORKER_TASK.value
        assert event.passed is True
        assert event.awaiting_user is True

    def test_event_is_published_after_durable_write(
        self,
        store: WorkReceiptStore,
        isolated_bus: EventBus,
        tmp_path: Path,
    ) -> None:
        # When a subscriber observes the event, the receipt must already be
        # readable from the store — this proves the publish happens AFTER
        # os.replace, not before.
        observed_round_trip: list[bool] = []

        def _verify_durable(event: ReceiptAppended) -> None:
            path = tmp_path / "outputs" / "receipts" / event.project_id / "receipts.jsonl"
            content = path.read_text(encoding="utf-8") if path.exists() else ""
            observed_round_trip.append(event.receipt_id in content)

        isolated_bus.subscribe(ReceiptAppended, _verify_durable)

        store.append(_make_receipt(project_id="proj-H"))

        assert observed_round_trip == [True]
