"""Consolidated Agents Package (v0.4.0).

======================================
4 multi-mode agents in the consolidated sub-package.
Combined with PlannerAgent and BuilderAgent in the parent package,
this gives 6 total agents.

The consolidation mapping (22 -> 6):
- PLANNER = PLANNER + ORCHESTRATOR (USER_INTERACTION + CONTEXT_MANAGER)
- BUILDER = BUILDER + IMAGE_GENERATOR
- CONSOLIDATED_RESEARCHER = EXPLORER + RESEARCHER + LIBRARIAN + ARCHITECT
    (UI_PLANNER + DATA_ENGINEER + DEVOPS + VERSION_CONTROL)
- CONSOLIDATED_ORACLE = ORACLE + PONDER
- QUALITY = EVALUATOR + SECURITY_AUDITOR + TEST_AUTOMATION
- OPERATIONS = SYNTHESIZER + DOCUMENTATION_AGENT + COST_PLANNER +
               EXPERIMENTATION_MANAGER + IMPROVEMENT + ERROR_RECOVERY
"""

from __future__ import annotations

from vetinari.agents.consolidated.operations_agent import (
    OperationsAgent,
    get_operations_agent,
)
from vetinari.agents.consolidated.oracle_agent import (
    ConsolidatedOracleAgent,
    get_consolidated_oracle_agent,
)
from vetinari.agents.consolidated.quality_agent import (
    QualityAgent,
    get_quality_agent,
)
from vetinari.agents.consolidated.researcher_agent import (
    ConsolidatedResearcherAgent,
    get_consolidated_researcher_agent,
)

__all__ = [
    "ConsolidatedOracleAgent",
    "ConsolidatedResearcherAgent",
    "OperationsAgent",
    "QualityAgent",
    "get_consolidated_oracle_agent",
    "get_consolidated_researcher_agent",
    "get_operations_agent",
    "get_quality_agent",
]
