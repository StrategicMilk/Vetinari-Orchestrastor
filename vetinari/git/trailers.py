"""Git commit trailer generation for decision lineage.

Appends ``Decision-Ref: @ADR-XXXX`` and ``Decision-Ref: @DJ-XXXX`` trailers
to git commits, linking code changes to the ADRs and decision journal entries
that motivated them. Lightweight complement to ADRs: ADRs document the "why"
at design time, trailers link runtime decisions to commits.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Pattern matching ADR references in decision metadata
_ADR_PATTERN = re.compile(r"ADR-(\d+)")


def generate_trailers(
    decision_ids: list[str] | None = None,
    trace_id: str | None = None,
    adr_ids: list[str] | None = None,
) -> list[str]:
    """Generate Decision-Ref git trailers for commit messages.

    Produces trailer lines linking a commit to decision journal entries
    and/or ADRs. When a trace_id is provided, looks up all decisions
    associated with that pipeline trace.

    Args:
        decision_ids: Explicit list of decision journal IDs to reference.
        trace_id: Pipeline trace ID — look up associated decisions automatically.
        adr_ids: Explicit list of ADR numbers to reference (e.g. ["0061", "0076"]).

    Returns:
        List of trailer strings like "Decision-Ref: @DJ-abc123" and
        "Decision-Ref: @ADR-0061", ready to append to a commit message.
    """
    trailers: list[str] = []
    seen: set[str] = set()  # Deduplicate trailers

    # Collect decision IDs from explicit list
    all_decision_ids = list(decision_ids) if decision_ids is not None else []

    # Collect decision IDs from trace
    if trace_id is not None:
        trace_decisions = _get_decisions_for_trace(trace_id)
        all_decision_ids.extend(trace_decisions)

    # Generate trailers for decision journal entries
    for did in all_decision_ids:
        trailer = f"Decision-Ref: @DJ-{did}"
        if trailer not in seen:
            trailers.append(trailer)
            seen.add(trailer)

    # Extract ADR references from decision metadata
    adr_refs = _extract_adr_refs_from_decisions(all_decision_ids)

    # Add explicit ADR references
    if adr_ids:
        adr_refs.extend(adr_ids)

    # Generate trailers for ADR references
    for adr_id in adr_refs:
        trailer = f"Decision-Ref: @ADR-{adr_id}"
        if trailer not in seen:
            trailers.append(trailer)
            seen.add(trailer)

    if trailers:
        logger.info("Generated %d decision trailers for commit", len(trailers))
    return trailers


def format_trailers_for_commit(trailers: list[str]) -> str:
    """Format trailer lines for appending to a git commit message.

    Adds a blank line separator before the trailers block, as required
    by the git trailer convention.

    Args:
        trailers: List of trailer strings from ``generate_trailers()``.

    Returns:
        Formatted string ready to append to a commit message body.
        Empty string if no trailers.
    """
    if not trailers:
        return ""
    return "\n" + "\n".join(trailers)


def _get_decisions_for_trace(trace_id: str) -> list[str]:
    """Look up decision journal IDs associated with a pipeline trace.

    Args:
        trace_id: Pipeline trace ID.

    Returns:
        List of decision_id strings. Empty list if journal unavailable.
    """
    try:
        from vetinari.observability.decision_journal import get_decision_journal

        journal = get_decision_journal()
        return journal.get_decision_ids_for_trace(trace_id)
    except Exception:
        logger.warning(
            "Could not look up decisions for trace %s — proceeding without decision trailers",
            trace_id,
        )
        return []


def _extract_adr_refs_from_decisions(decision_ids: list[str]) -> list[str]:
    """Extract ADR references from decision journal entry metadata.

    Scans decision metadata and reasoning text for ADR-NNNN patterns.

    Args:
        decision_ids: List of decision journal IDs to scan.

    Returns:
        List of ADR number strings (e.g. ["0061", "0076"]).
    """
    if not decision_ids:
        return []

    adr_refs: list[str] = []
    try:
        from vetinari.observability.decision_journal import get_decision_journal

        journal = get_decision_journal()
        # Fetch all decisions once and build an O(1) lookup by ID.
        # Calling get_decisions(limit=100) inside a per-ID loop silently misses
        # any decisions beyond position 100 and causes O(n*m) DB round-trips.
        all_records = journal.get_decisions(limit=10_000)
        record_index = {r.decision_id: r for r in all_records}

        for did in decision_ids:
            record = record_index.get(did)
            if record is None:
                continue
            # Scan all text fields that may contain ADR references
            text = " ".join([
                record.description,
                record.action_taken,
                record.outcome,
                str(record.context),
                str(record.confidence_factors),
            ])
            for match in _ADR_PATTERN.finditer(text):
                adr_num = match.group(1).zfill(4)
                if adr_num not in adr_refs:
                    adr_refs.append(adr_num)
    except Exception:
        logger.warning("Could not extract ADR refs from decisions — proceeding without ADR trailers")

    return adr_refs
