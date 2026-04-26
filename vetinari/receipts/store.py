"""Durable JSONL store for WorkReceipts.

Each receipt is appended to ``outputs/receipts/<project_id>/receipts.jsonl``
in O(1) append mode (open-for-append + flush + fsync). The previous
implementation read the whole file and rewrote it for every append,
which scaled as O(n) and was wasteful for projects accumulating
hundreds or thousands of receipts.

Durability story: the JSON line + trailing newline is written in a
single ``write()`` call followed by an explicit ``flush()`` and
``os.fsync()``. A process crash mid-write may leave a partially
written final line on disk; ``iter_receipts`` is corruption-resistant
and skips malformed lines with a WARNING log so a torn last line
never breaks reads of the surviving records.

On every successful append the store publishes a ``receipt.appended``
event via the global event bus so the Control Center SSE channel can
push the new receipt to attached clients without polling.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vetinari.agents.contracts import (
    AttestedArtifact,
    LLMJudgment,
    OutcomeSignal,
    Provenance,
    ToolEvidence,
)
from vetinari.events import EventBus, get_event_bus
from vetinari.receipts.events import receipt_appended
from vetinari.receipts.record import WorkReceipt, WorkReceiptKind
from vetinari.types import AgentType, ArtifactKind, EvidenceBasis

logger = logging.getLogger(__name__)


def _default_repo_root() -> Path:
    """Resolve the repository root from this file's location."""
    return Path(__file__).resolve().parent.parent.parent


class WorkReceiptStore:
    """Per-project append-only JSONL store for WorkReceipts.

    The store writes to ``<repo_root>/outputs/receipts/<project_id>/receipts.jsonl``
    and publishes a ``receipt.appended`` event on every successful append.

    Args:
        repo_root: Repository root path. Defaults to three parents above
            this source file (i.e. ``vetinari/receipts/store.py`` -> repo
            root). Receipt files are resolved relative to this root.
        event_bus: Event bus instance for publishing append events.
            Defaults to the global singleton from ``get_event_bus()``;
            callers may inject a fresh instance for test isolation.
    """

    def __init__(
        self,
        repo_root: Path | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialise the store.

        Args:
            repo_root: Repository root path. ``None`` resolves to the
                location three parents above this file.
            event_bus: Optional EventBus override; ``None`` uses the
                process-wide singleton.
        """
        self._repo_root: Path = repo_root if repo_root is not None else _default_repo_root()
        self._event_bus: EventBus = event_bus if event_bus is not None else get_event_bus()

    @property
    def receipts_root(self) -> Path:
        """Directory holding every project's receipts subdirectory.

        Returns ``<repo_root>/outputs/receipts``. Useful for cross-project
        scans (e.g. the Attention API walks every project under this
        root) without resolving a synthetic project id first.
        """
        return self._repo_root / "outputs" / "receipts"

    def receipts_path(self, project_id: str) -> Path:
        """Return the absolute path to a project's receipts file.

        Args:
            project_id: Project identifier; must be non-empty.

        Returns:
            Absolute ``Path`` to ``outputs/receipts/<project_id>/receipts.jsonl``.

        Raises:
            ValueError: If project_id is empty or whitespace-only.
        """
        if not project_id or not project_id.strip():
            raise ValueError("project_id must be a non-empty string")
        return self.receipts_root / project_id / "receipts.jsonl"

    def append(self, receipt: WorkReceipt) -> None:
        """Append one receipt and publish ``receipt.appended``.

        Strategy: open the per-project JSONL in append mode, write the
        new line + newline in a single ``write()``, then ``flush()`` and
        ``os.fsync()`` so the bytes are durably on disk before the event
        fires. This is O(1) regardless of how many receipts already
        live in the file.

        A process crash mid-write may leave a torn final line; reads
        survive that case because ``iter_receipts`` skips malformed
        lines with a WARNING log.

        After the line is durably persisted, publish a
        ``receipt.appended`` event with payload
        ``{project_id, receipt_id, kind, passed, awaiting_user}``.

        Args:
            receipt: The WorkReceipt to persist.

        Raises:
            TypeError: If *receipt* is not a WorkReceipt instance.
            OSError: If the receipts directory cannot be created or
                written.
        """
        if not isinstance(receipt, WorkReceipt):
            raise TypeError(f"append() expects a WorkReceipt, got {type(receipt).__name__!r}")

        path = self.receipts_path(receipt.project_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        line = _receipt_to_jsonl(receipt) + "\n"

        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

        logger.debug(
            "Appended receipt %s (kind=%s, project=%s) to %s",
            receipt.receipt_id,
            receipt.kind.value,
            receipt.project_id,
            path,
        )

        self._event_bus.publish(
            receipt_appended(
                project_id=receipt.project_id,
                receipt_id=receipt.receipt_id,
                kind=receipt.kind.value,
                passed=receipt.outcome.passed,
                awaiting_user=receipt.awaiting_user,
            )
        )

    def iter_receipts(self, project_id: str) -> Iterator[WorkReceipt]:
        """Stream a project's receipts in append order, line-by-line.

        Lines that fail to deserialise are skipped with a WARNING log so
        a single corrupted record does not block the rest of the stream.

        Args:
            project_id: Project identifier whose receipts should be
                streamed.

        Yields:
            One WorkReceipt per line in append order.
        """
        path = self.receipts_path(project_id)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    yield _receipt_from_jsonl(stripped)
                except (ValueError, KeyError, TypeError) as exc:
                    logger.warning(
                        "Skipping malformed receipts line %d in %s — %s",
                        lineno,
                        path,
                        exc,
                    )
                    continue

    def find_awaiting(self, project_id: str) -> list[WorkReceipt]:
        """Return only this project's receipts that block on user input.

        Args:
            project_id: Project identifier to filter on.

        Returns:
            Receipts where ``awaiting_user`` is True, in append order.
            Empty list when the project has no receipts file or no
            awaiting receipts.
        """
        return [r for r in self.iter_receipts(project_id) if r.awaiting_user]


# --- serialization helpers ---------------------------------------------------


def _receipt_to_jsonl(receipt: WorkReceipt) -> str:
    """Serialise a WorkReceipt to a single-line JSON string.

    Enum values are written as their string ``.value`` so the JSONL is
    self-contained and does not require the enum imports at read time.

    Args:
        receipt: The receipt to serialise.

    Returns:
        Single-line JSON string (no trailing newline).
    """
    raw: dict[str, Any] = {
        "receipt_id": receipt.receipt_id,
        "project_id": receipt.project_id,
        "agent_id": receipt.agent_id,
        "agent_type": receipt.agent_type.value,
        "kind": receipt.kind.value,
        "started_at_utc": receipt.started_at_utc,
        "finished_at_utc": receipt.finished_at_utc,
        "inputs_summary": receipt.inputs_summary,
        "outputs_summary": receipt.outputs_summary,
        "outcome": _outcome_to_dict(receipt.outcome),
        "awaiting_user": receipt.awaiting_user,
        "awaiting_reason": receipt.awaiting_reason,
        "linked_claim_ids": list(receipt.linked_claim_ids),
    }
    return json.dumps(raw, separators=(",", ":"))


def _outcome_to_dict(outcome: OutcomeSignal) -> dict[str, Any]:
    """Convert an OutcomeSignal (with nested frozen dataclasses) to a dict."""
    return {
        "passed": outcome.passed,
        "score": outcome.score,
        "basis": outcome.basis.value,
        "tool_evidence": [asdict(te) for te in outcome.tool_evidence],
        "llm_judgment": asdict(outcome.llm_judgment) if outcome.llm_judgment is not None else None,
        "attested_artifacts": [_attested_to_dict(a) for a in outcome.attested_artifacts],
        "provenance": asdict(outcome.provenance) if outcome.provenance is not None else None,
        "issues": list(outcome.issues),
        "suggestions": list(outcome.suggestions),
        "use_case": outcome.use_case,
    }


def _attested_to_dict(artifact: AttestedArtifact) -> dict[str, Any]:
    """Convert an AttestedArtifact to a JSON-friendly dict."""
    return {
        "kind": artifact.kind.value,
        "attested_by": artifact.attested_by,
        "attested_at_utc": artifact.attested_at_utc,
        "payload": dict(artifact.payload),
    }


def _receipt_from_jsonl(line: str) -> WorkReceipt:
    """Deserialise one JSONL line into a WorkReceipt.

    Args:
        line: The raw JSON line (no trailing newline).

    Returns:
        The reconstructed WorkReceipt.

    Raises:
        ValueError: If the line is not valid JSON or required fields
            are missing.
        KeyError: If a required field is absent from the payload.
    """
    raw: dict[str, Any] = json.loads(line)
    return WorkReceipt(
        receipt_id=raw["receipt_id"],
        project_id=raw["project_id"],
        agent_id=raw["agent_id"],
        agent_type=AgentType(raw["agent_type"]),
        kind=WorkReceiptKind(raw["kind"]),
        started_at_utc=raw["started_at_utc"],
        finished_at_utc=raw["finished_at_utc"],
        inputs_summary=raw.get("inputs_summary", ""),
        outputs_summary=raw.get("outputs_summary", ""),
        outcome=_outcome_from_dict(raw["outcome"]),
        awaiting_user=bool(raw.get("awaiting_user")),
        awaiting_reason=raw.get("awaiting_reason"),
        linked_claim_ids=tuple(raw.get("linked_claim_ids", ())),
    )


def _outcome_from_dict(raw: dict[str, Any]) -> OutcomeSignal:
    """Reconstruct an OutcomeSignal from a JSON dict."""
    tool_evidence = tuple(ToolEvidence(**te) for te in raw.get("tool_evidence", ()))
    llm_raw = raw.get("llm_judgment")
    llm_judgment = LLMJudgment(**llm_raw) if llm_raw is not None else None
    attested = tuple(_attested_from_dict(a) for a in raw.get("attested_artifacts", ()))
    prov_raw = raw.get("provenance")
    provenance = Provenance(**prov_raw) if prov_raw is not None else None
    return OutcomeSignal(
        passed=bool(raw.get("passed")),
        score=float(raw.get("score", 0.0)),
        basis=EvidenceBasis(raw.get("basis", EvidenceBasis.UNSUPPORTED.value)),
        tool_evidence=tool_evidence,
        llm_judgment=llm_judgment,
        attested_artifacts=attested,
        provenance=provenance,
        issues=tuple(raw.get("issues", ())),
        suggestions=tuple(raw.get("suggestions", ())),
        use_case=raw.get("use_case"),
    )


def _attested_from_dict(raw: dict[str, Any]) -> AttestedArtifact:
    """Reconstruct an AttestedArtifact from a JSON dict."""
    return AttestedArtifact(
        kind=ArtifactKind(raw["kind"]),
        attested_by=raw["attested_by"],
        attested_at_utc=raw["attested_at_utc"],
        payload=dict(raw.get("payload", {})),
    )


__all__ = ["WorkReceiptStore"]
