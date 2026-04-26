"""Awareness subsystem — confidence computation and uncertainty handling.

Provides the canonical ConfidenceResult contract used across the pipeline:
Intake → Planning → Execution → **Confidence Gate** → Quality Gate → Assembly.
"""

from __future__ import annotations

from vetinari.awareness.confidence import (
    ConfidenceComputer,
    ConfidenceResult,
    UnknownSituation,
    UnknownSituationProtocol,
    classify_confidence_score,
)

__all__ = [
    "ConfidenceComputer",
    "ConfidenceResult",
    "UnknownSituation",
    "UnknownSituationProtocol",
    "classify_confidence_score",
]
