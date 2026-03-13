"""Vetinari Skills Package.

========================
Skill manifests, specifications, implementations, and the programmatic skill registry.
"""

from __future__ import annotations

# Unified consolidated skill tools (Phase 3)
from vetinari.skills.architect_skill import ArchitectSkillTool

# Legacy skill tools (individual)
from vetinari.skills.builder import BuilderSkillTool
from vetinari.skills.evaluator import EvaluatorSkillTool
from vetinari.skills.explorer import ExplorerSkillTool
from vetinari.skills.librarian import LibrarianSkillTool
from vetinari.skills.operations_skill import OperationsSkillTool
from vetinari.skills.oracle import OracleSkillTool
from vetinari.skills.quality_skill import QualitySkillTool
from vetinari.skills.researcher import ResearcherSkillTool
from vetinari.skills.skill_registry import (
    SKILL_REGISTRY,
    get_all_skills,
    get_skill,
    get_skill_for_agent_type,
    get_skills_by_capability,
    get_skills_by_standard_category,
    get_skills_by_tag,
    validate_all,
)
from vetinari.skills.skill_spec import (
    SkillConstraint,
    SkillGuideline,
    SkillSpec,
    SkillStandard,
)
from vetinari.skills.synthesizer import SynthesizerSkillTool
from vetinari.skills.ui_planner import UIPlannerSkillTool

__all__ = [
    # Registry API
    "SKILL_REGISTRY",
    # Unified consolidated skill tools
    "ArchitectSkillTool",
    # Legacy skill tools
    "BuilderSkillTool",
    "EvaluatorSkillTool",
    "ExplorerSkillTool",
    "LibrarianSkillTool",
    "OperationsSkillTool",
    "OracleSkillTool",
    "QualitySkillTool",
    "ResearcherSkillTool",
    "SkillConstraint",
    "SkillGuideline",
    # Spec types
    "SkillSpec",
    "SkillStandard",
    "SynthesizerSkillTool",
    "UIPlannerSkillTool",
    "get_all_skills",
    "get_skill",
    "get_skill_for_agent_type",
    "get_skills_by_capability",
    "get_skills_by_standard_category",
    "get_skills_by_tag",
    "validate_all",
]
