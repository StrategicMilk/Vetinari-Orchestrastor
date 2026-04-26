"""Vetinari Skills Package.

Three-agent skill hierarchy aligned to the factory pipeline (ADR-0061):

Primary skill tools (one per factory agent):
  - ``ForemanSkillTool`` — planning, clarification, orchestration (6 modes)
  - ``WorkerSkillTool``  — research, architecture, build, operations (25 modes)
  - ``InspectorSkillTool`` — independent quality gate (4 modes)

Internal component tools (used by WorkerSkillTool and InspectorSkillTool):
  - ``ArchitectSkillTool``  — architecture mode group component of Worker
  - ``OperationsSkillTool`` — operations mode group component of Worker
  - ``QualitySkillTool``    — supplementary analysis component of Inspector

Skill specifications and registry:
  - ``SkillSpec``     — canonical contract for all skills
  - ``SkillStandard`` — mandatory quality standard
  - ``SkillGuideline`` — advisory best-practice recommendation
  - ``SkillConstraint`` — hard resource/scope/safety limit
  - ``SKILL_REGISTRY`` — programmatic registry with 3 entries (foreman, worker, inspector)
"""

from __future__ import annotations

from vetinari.skills.architect_skill import ArchitectSkillTool
from vetinari.skills.foreman_skill import ForemanSkillTool
from vetinari.skills.inspector_skill import InspectorSkillTool
from vetinari.skills.operations_skill import OperationsSkillTool
from vetinari.skills.quality_skill import QualitySkillTool
from vetinari.skills.skill_registry import (  # noqa: VET123 — get_skills_by_standard_category has no external callers but removing causes VET120
    get_all_skills,
    get_skill,
    get_skill_for_agent_type,
    get_skill_validation_detail,
    get_skills_by_capability,
    get_skills_by_standard_category,
    get_skills_by_tag,
    validate_all,
)
from vetinari.skills.skill_spec import (  # noqa: VET123 - barrel export preserves public import compatibility
    SkillSpec,
)
from vetinari.skills.worker_skill import WorkerSkillTool

__all__ = [
    "ArchitectSkillTool",  # Internal component of WorkerSkillTool
    "ForemanSkillTool",  # Primary: planning, clarification, orchestration
    "InspectorSkillTool",  # Primary: independent quality gate
    "OperationsSkillTool",  # Internal component of WorkerSkillTool
    "QualitySkillTool",  # Internal component of InspectorSkillTool
    "SkillSpec",
    "WorkerSkillTool",  # Primary: research, architecture, build, operations
    "get_all_skills",
    "get_skill",
    "get_skill_for_agent_type",
    "get_skill_validation_detail",
    "get_skills_by_capability",
    "get_skills_by_standard_category",
    "get_skills_by_tag",
    "validate_all",
]
