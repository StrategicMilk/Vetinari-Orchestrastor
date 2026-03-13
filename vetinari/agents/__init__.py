"""Vetinari Agents Module (v0.4.0).

6 consolidated multi-mode agents replacing the original 22 single-purpose agents.
Legacy imports are available via vetinari.agents.compat.
"""

from __future__ import annotations

from vetinari.types import AgentType, ExecutionMode, TaskStatus

from .base_agent import BaseAgent
from .builder_agent import BuilderAgent, get_builder_agent

# Legacy coding bridge support
from .coding_bridge import (
    CodingBridge,
    CodingResult,
    CodingTask,
    CodingTaskStatus,
    CodingTaskType,
    get_coding_bridge,
    init_coding_bridge,
)

# ── Legacy compatibility (re-exports with deprecation warnings) ──────
from .compat import (
    ArchitectAgent,
    ContextManagerAgent,
    CostPlannerAgent,
    DataEngineerAgent,
    DevOpsAgent,
    DocumentationAgent,
    ErrorRecoveryAgent,
    EvaluatorAgent,
    ExperimentationManagerAgent,
    ExplorerAgent,
    ImageGeneratorAgent,
    ImprovementAgent,
    LibrarianAgent,
    OracleAgent,
    OrchestratorAgent,
    ResearcherAgent,
    SecurityAuditorAgent,
    SynthesizerAgent,
    TestAutomationAgent,
    UIPlannerAgent,
    UserInteractionAgent,
    VersionControlAgent,
    get_architect_agent,
    get_context_manager_agent,
    get_cost_planner_agent,
    get_data_engineer_agent,
    get_devops_agent,
    get_documentation_agent,
    get_error_recovery_agent,
    get_evaluator_agent,
    get_experimentation_manager_agent,
    get_explorer_agent,
    get_image_generator_agent,
    get_improvement_agent,
    get_librarian_agent,
    get_oracle_agent,
    get_orchestrator_agent,
    get_researcher_agent,
    get_security_auditor_agent,
    get_synthesizer_agent,
    get_test_automation_agent,
    get_ui_planner_agent,
    get_user_interaction_agent,
    get_version_control_agent,
)
from .consolidated.operations_agent import OperationsAgent, get_operations_agent
from .consolidated.oracle_agent import (
    ConsolidatedOracleAgent,
    get_consolidated_oracle_agent,
)
from .consolidated.quality_agent import QualityAgent, get_quality_agent
from .consolidated.researcher_agent import (
    ConsolidatedResearcherAgent,
    get_consolidated_researcher_agent,
)
from .contracts import (
    AGENT_REGISTRY,
    AgentResult,
    AgentSpec,
    AgentTask,
    Plan,
    Task,
    VerificationResult,
    get_agent_spec,
)

# ── The 6 consolidated agents ────────────────────────────────────────
from .multi_mode_agent import MultiModeAgent
from .planner_agent import PlannerAgent, get_planner_agent

__all__ = [
    "AGENT_REGISTRY",
    "AgentResult",
    "AgentSpec",
    "AgentTask",
    # Contracts and types
    "AgentType",
    "ArchitectAgent",
    # Base classes
    "BaseAgent",
    "BuilderAgent",
    # Coding bridge
    "CodingBridge",
    "CodingResult",
    "CodingTask",
    "CodingTaskStatus",
    "CodingTaskType",
    "ConsolidatedOracleAgent",
    "ConsolidatedResearcherAgent",
    "ContextManagerAgent",
    "CostPlannerAgent",
    "DataEngineerAgent",
    "DevOpsAgent",
    "DocumentationAgent",
    "ErrorRecoveryAgent",
    "EvaluatorAgent",
    "ExecutionMode",
    "ExperimentationManagerAgent",
    # ── Legacy compat (deprecated) ──
    "ExplorerAgent",
    "ImageGeneratorAgent",
    "ImprovementAgent",
    "LibrarianAgent",
    "MultiModeAgent",
    "OperationsAgent",
    "OracleAgent",
    "OrchestratorAgent",
    "Plan",
    # ── 6 consolidated agents ──
    "PlannerAgent",
    "QualityAgent",
    "ResearcherAgent",
    "SecurityAuditorAgent",
    "SynthesizerAgent",
    "Task",
    "TaskStatus",
    "TestAutomationAgent",
    "UIPlannerAgent",
    "UserInteractionAgent",
    "VerificationResult",
    "VersionControlAgent",
    "get_agent_spec",
    "get_architect_agent",
    "get_builder_agent",
    "get_coding_bridge",
    "get_consolidated_oracle_agent",
    "get_consolidated_researcher_agent",
    "get_context_manager_agent",
    "get_cost_planner_agent",
    "get_data_engineer_agent",
    "get_devops_agent",
    "get_documentation_agent",
    "get_error_recovery_agent",
    "get_evaluator_agent",
    "get_experimentation_manager_agent",
    "get_explorer_agent",
    "get_image_generator_agent",
    "get_improvement_agent",
    "get_librarian_agent",
    "get_operations_agent",
    "get_oracle_agent",
    "get_orchestrator_agent",
    "get_planner_agent",
    "get_quality_agent",
    "get_researcher_agent",
    "get_security_auditor_agent",
    "get_synthesizer_agent",
    "get_test_automation_agent",
    "get_ui_planner_agent",
    "get_user_interaction_agent",
    "get_version_control_agent",
    "init_coding_bridge",
]
