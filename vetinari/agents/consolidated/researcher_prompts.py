"""Consolidated Researcher Agent mode prompts."""

from __future__ import annotations

from vetinari.agents.consolidated.researcher_prompts_core import (
    _API_LOOKUP_PROMPT,
    _CODE_DISCOVERY_PROMPT,
    _DOMAIN_RESEARCH_PROMPT,
    _LATERAL_THINKING_PROMPT,
)
from vetinari.agents.consolidated.researcher_prompts_design import (
    _DATABASE_PROMPT,
    _DEVOPS_PROMPT,
    _GIT_WORKFLOW_PROMPT,
    _UI_DESIGN_PROMPT,
)

RESEARCHER_MODE_PROMPTS: dict[str, str] = {
    "code_discovery": _CODE_DISCOVERY_PROMPT,
    "domain_research": _DOMAIN_RESEARCH_PROMPT,
    "api_lookup": _API_LOOKUP_PROMPT,
    "lateral_thinking": _LATERAL_THINKING_PROMPT,
    "ui_design": _UI_DESIGN_PROMPT,
    "database": _DATABASE_PROMPT,
    "devops": _DEVOPS_PROMPT,
    "git_workflow": _GIT_WORKFLOW_PROMPT,
}
