"""Validation, verification, and goal checking subsystem."""

from __future__ import annotations

from vetinari.validation.document_judge import DocumentJudge, JudgeConfig
from vetinari.validation.document_quality import (
    DIMENSIONS,
    DimensionScore,
    QualityReport,
    evaluate_document,
)
from vetinari.validation.document_types import (
    DocumentProfile,
    DocumentType,
    get_profile_for_type,
    load_document_profiles,
)
from vetinari.validation.goal_verifier import *  # noqa: F403, VET006
from vetinari.validation.validator import *  # noqa: F403, VET006
from vetinari.validation.verification import *  # noqa: F403, VET006

__all__ = [
    "DIMENSIONS",
    "DimensionScore",
    "DocumentJudge",
    "DocumentProfile",
    "DocumentType",
    "JudgeConfig",
    "QualityReport",
    "evaluate_document",
    "get_profile_for_type",
    "load_document_profiles",
]
