"""
Vetinari Skills Package
========================
Skill manifests, specifications, implementations, and the programmatic skill registry.
"""

from vetinari.skills.skill_spec import (
    SkillSpec,
    SkillStandard,
    SkillGuideline,
    SkillConstraint,
)
from vetinari.skills.skill_registry import (
    SKILL_REGISTRY,
    get_skill,
    get_skill_for_agent_type,
    get_all_skills,
    get_skills_by_capability,
    get_skills_by_tag,
    get_skills_by_standard_category,
    validate_all,
)

# Legacy skill tools (individual)
from vetinari.skills.builder import BuilderSkillTool
from vetinari.skills.explorer import ExplorerSkillTool
from vetinari.skills.evaluator import EvaluatorSkillTool
from vetinari.skills.librarian import LibrarianSkillTool
from vetinari.skills.oracle import OracleSkillTool
from vetinari.skills.researcher import ResearcherSkillTool
from vetinari.skills.synthesizer import SynthesizerSkillTool
from vetinari.skills.ui_planner import UIPlannerSkillTool

# Unified consolidated skill tools (Phase 3)
from vetinari.skills.architect_skill import ArchitectSkillTool
from vetinari.skills.quality_skill import QualitySkillTool
from vetinari.skills.operations_skill import OperationsSkillTool

__all__ = [
    # Spec types
    "SkillSpec",
    "SkillStandard",
    "SkillGuideline",
    "SkillConstraint",
    # Registry API
    "SKILL_REGISTRY",
    "get_skill",
    "get_skill_for_agent_type",
    "get_all_skills",
    "get_skills_by_capability",
    "get_skills_by_tag",
    "get_skills_by_standard_category",
    "validate_all",
    # Legacy skill tools
    "BuilderSkillTool",
    "ExplorerSkillTool",
    "EvaluatorSkillTool",
    "LibrarianSkillTool",
    "OracleSkillTool",
    "ResearcherSkillTool",
    "SynthesizerSkillTool",
    "UIPlannerSkillTool",
    # Unified consolidated skill tools
    "ArchitectSkillTool",
    "QualitySkillTool",
    "OperationsSkillTool",
]
