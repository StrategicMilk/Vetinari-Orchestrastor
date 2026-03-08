"""
Consolidated Agents Package (Phase 3)
======================================
8 multi-mode agents replacing the original 22 single-purpose agents.

The consolidation mapping:
- PLANNER (1:1, unchanged)
- ORCHESTRATOR = USER_INTERACTION + CONTEXT_MANAGER
- CONSOLIDATED_RESEARCHER = EXPLORER + RESEARCHER + LIBRARIAN
- CONSOLIDATED_ORACLE = ORACLE + PONDER
- BUILDER (1:1, unchanged)
- ARCHITECT = UI_PLANNER + DATA_ENGINEER + DEVOPS + VERSION_CONTROL
- QUALITY = EVALUATOR + SECURITY_AUDITOR + TEST_AUTOMATION
- OPERATIONS = SYNTHESIZER + DOCUMENTATION_AGENT + COST_PLANNER +
               EXPERIMENTATION_MANAGER + IMPROVEMENT + ERROR_RECOVERY +
               IMAGE_GENERATOR
"""

from vetinari.agents.consolidated.orchestrator_agent import (
    OrchestratorAgent,
    get_orchestrator_agent,
)
from vetinari.agents.consolidated.researcher_agent import (
    ConsolidatedResearcherAgent,
    get_consolidated_researcher_agent,
)
from vetinari.agents.consolidated.oracle_agent import (
    ConsolidatedOracleAgent,
    get_consolidated_oracle_agent,
)
from vetinari.agents.consolidated.architect_agent import (
    ArchitectAgent,
    get_architect_agent,
)
from vetinari.agents.consolidated.quality_agent import (
    QualityAgent,
    get_quality_agent,
)
from vetinari.agents.consolidated.operations_agent import (
    OperationsAgent,
    get_operations_agent,
)

__all__ = [
    "OrchestratorAgent", "get_orchestrator_agent",
    "ConsolidatedResearcherAgent", "get_consolidated_researcher_agent",
    "ConsolidatedOracleAgent", "get_consolidated_oracle_agent",
    "ArchitectAgent", "get_architect_agent",
    "QualityAgent", "get_quality_agent",
    "OperationsAgent", "get_operations_agent",
]
