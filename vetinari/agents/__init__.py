"""
Vetinari Agents Module (v0.4.0)

6 consolidated multi-mode agents replacing the original 22 single-purpose agents.
Legacy imports are available via vetinari.agents.compat.
"""

from .base_agent import BaseAgent
from .contracts import (
    AgentType,
    AgentSpec,
    Task,
    AgentTask,
    Plan,
    AgentResult,
    VerificationResult,
    TaskStatus,
    ExecutionMode,
    get_agent_spec,
    AGENT_REGISTRY,
)

# ── The 6 consolidated agents ────────────────────────────────────────
from .multi_mode_agent import MultiModeAgent
from .planner_agent import PlannerAgent, get_planner_agent
from .builder_agent import BuilderAgent, get_builder_agent
from .consolidated.researcher_agent import (
    ConsolidatedResearcherAgent,
    get_consolidated_researcher_agent,
)
from .consolidated.oracle_agent import (
    ConsolidatedOracleAgent,
    get_consolidated_oracle_agent,
)
from .consolidated.quality_agent import QualityAgent, get_quality_agent
from .consolidated.operations_agent import OperationsAgent, get_operations_agent

# ── Legacy compatibility (re-exports with deprecation warnings) ──────
from .compat import (
    ExplorerAgent, get_explorer_agent,
    LibrarianAgent, get_librarian_agent,
    ResearcherAgent, get_researcher_agent,
    OracleAgent, get_oracle_agent,
    EvaluatorAgent, get_evaluator_agent,
    SynthesizerAgent, get_synthesizer_agent,
    UIPlannerAgent, get_ui_planner_agent,
    SecurityAuditorAgent, get_security_auditor_agent,
    DataEngineerAgent, get_data_engineer_agent,
    DocumentationAgent, get_documentation_agent,
    CostPlannerAgent, get_cost_planner_agent,
    TestAutomationAgent, get_test_automation_agent,
    ExperimentationManagerAgent, get_experimentation_manager_agent,
    ImprovementAgent, get_improvement_agent,
    UserInteractionAgent, get_user_interaction_agent,
    DevOpsAgent, get_devops_agent,
    VersionControlAgent, get_version_control_agent,
    ErrorRecoveryAgent, get_error_recovery_agent,
    ContextManagerAgent, get_context_manager_agent,
    ImageGeneratorAgent, get_image_generator_agent,
    OrchestratorAgent, get_orchestrator_agent,
    ArchitectAgent, get_architect_agent,
)

# Legacy coding bridge support
from .coding_bridge import (
    CodingBridge,
    CodingTask,
    CodingResult,
    CodingTaskType,
    CodingTaskStatus,
    get_coding_bridge,
    init_coding_bridge,
)

__all__ = [
    # Base classes
    "BaseAgent",
    "MultiModeAgent",

    # Contracts and types
    "AgentType", "AgentSpec", "Task", "AgentTask", "Plan",
    "AgentResult", "VerificationResult", "TaskStatus", "ExecutionMode",
    "get_agent_spec", "AGENT_REGISTRY",

    # ── 6 consolidated agents ──
    "PlannerAgent", "get_planner_agent",
    "BuilderAgent", "get_builder_agent",
    "ConsolidatedResearcherAgent", "get_consolidated_researcher_agent",
    "ConsolidatedOracleAgent", "get_consolidated_oracle_agent",
    "QualityAgent", "get_quality_agent",
    "OperationsAgent", "get_operations_agent",

    # ── Legacy compat (deprecated) ──
    "ExplorerAgent", "get_explorer_agent",
    "LibrarianAgent", "get_librarian_agent",
    "ResearcherAgent", "get_researcher_agent",
    "OracleAgent", "get_oracle_agent",
    "EvaluatorAgent", "get_evaluator_agent",
    "SynthesizerAgent", "get_synthesizer_agent",
    "UIPlannerAgent", "get_ui_planner_agent",
    "SecurityAuditorAgent", "get_security_auditor_agent",
    "DataEngineerAgent", "get_data_engineer_agent",
    "DocumentationAgent", "get_documentation_agent",
    "CostPlannerAgent", "get_cost_planner_agent",
    "TestAutomationAgent", "get_test_automation_agent",
    "ExperimentationManagerAgent", "get_experimentation_manager_agent",
    "ImprovementAgent", "get_improvement_agent",
    "UserInteractionAgent", "get_user_interaction_agent",
    "DevOpsAgent", "get_devops_agent",
    "VersionControlAgent", "get_version_control_agent",
    "ErrorRecoveryAgent", "get_error_recovery_agent",
    "ContextManagerAgent", "get_context_manager_agent",
    "ImageGeneratorAgent", "get_image_generator_agent",
    "OrchestratorAgent", "get_orchestrator_agent",
    "ArchitectAgent", "get_architect_agent",

    # Coding bridge
    "CodingBridge", "CodingTask", "CodingResult",
    "CodingTaskType", "CodingTaskStatus",
    "get_coding_bridge", "init_coding_bridge",
]
