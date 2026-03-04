"""
Vetinari Agents Module.

This module provides specialized agents for Vetinari's orchestration system.
"""

from .base_agent import BaseAgent, AsyncBaseAgent
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
    AGENT_REGISTRY
)

# Core agents
from .planner_agent import PlannerAgent, get_planner_agent
from .explorer_agent import ExplorerAgent, get_explorer_agent
from .oracle_agent import OracleAgent, get_oracle_agent

# Core expansion agents
from .librarian_agent import LibrarianAgent, get_librarian_agent
from .researcher_agent import ResearcherAgent, get_researcher_agent
from .evaluator_agent import EvaluatorAgent, get_evaluator_agent
from .synthesizer_agent import SynthesizerAgent, get_synthesizer_agent
from .builder_agent import BuilderAgent, get_builder_agent
from .ui_planner_agent import UIPlannerAgent, get_ui_planner_agent

# Extended agents
from .security_auditor_agent import SecurityAuditorAgent, get_security_auditor_agent
from .data_engineer_agent import DataEngineerAgent, get_data_engineer_agent
from .documentation_agent import DocumentationAgent, get_documentation_agent
from .cost_planner_agent import CostPlannerAgent, get_cost_planner_agent
from .test_automation_agent import TestAutomationAgent, get_test_automation_agent
from .experimentation_manager_agent import ExperimentationManagerAgent, get_experimentation_manager_agent

# Legacy coding bridge support
from .coding_bridge import (
    CodingBridge,
    CodingTask,
    CodingResult,
    CodingTaskType,
    CodingTaskStatus,
    get_coding_bridge,
    init_coding_bridge
)

__all__ = [
    # Base classes
    "BaseAgent",
    "AsyncBaseAgent",
    
    # Contracts and types
    "AgentType",
    "AgentSpec",
    "Task",
    "AgentTask",
    "Plan",
    "AgentResult",
    "VerificationResult",
    "TaskStatus",
    "ExecutionMode",
    "get_agent_spec",
    "AGENT_REGISTRY",
    
    # Core agents
    "PlannerAgent",
    "get_planner_agent",
    "ExplorerAgent",
    "get_explorer_agent",
    "OracleAgent",
    "get_oracle_agent",
    
    # Core expansion agents
    "LibrarianAgent",
    "get_librarian_agent",
    "ResearcherAgent",
    "get_researcher_agent",
    "EvaluatorAgent",
    "get_evaluator_agent",
    "SynthesizerAgent",
    "get_synthesizer_agent",
    "BuilderAgent",
    "get_builder_agent",
    "UIPlannerAgent",
    "get_ui_planner_agent",
    
    # Extended agents
    "SecurityAuditorAgent",
    "get_security_auditor_agent",
    "DataEngineerAgent",
    "get_data_engineer_agent",
    "DocumentationAgent",
    "get_documentation_agent",
    "CostPlannerAgent",
    "get_cost_planner_agent",
    "TestAutomationAgent",
    "get_test_automation_agent",
    "ExperimentationManagerAgent",
    "get_experimentation_manager_agent",
    
    # Legacy coding bridge support
    "CodingBridge",
    "CodingTask",
    "CodingResult",
    "CodingTaskType",
    "CodingTaskStatus",
    "get_coding_bridge",
    "init_coding_bridge"
]
