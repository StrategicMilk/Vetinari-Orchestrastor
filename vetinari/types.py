"""Vetinari Canonical Type Definitions.

====================================
Single source of truth for all shared enums and base types.
All modules should import from here rather than defining their own.
"""

from __future__ import annotations

from enum import Enum


class TaskStatus(Enum):
    """Canonical task execution status."""

    PENDING = "pending"
    BLOCKED = "blocked"  # Waiting for dependencies
    READY = "ready"  # Dependencies met, awaiting execution
    ASSIGNED = "assigned"  # Assigned to a model/agent
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING = "waiting"  # Waiting for human input


class PlanStatus(Enum):
    """Canonical plan lifecycle status."""

    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    ACTIVE = "active"
    EXECUTING = "executing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class AgentType(Enum):
    """All recognized agent types in the system.

    The six active agents are: PLANNER, CONSOLIDATED_RESEARCHER,
    CONSOLIDATED_ORACLE, BUILDER, QUALITY, OPERATIONS.
    Legacy values are retained for backward-compatible deserialization
    of stored plans and memory entries.
    """

    # --- Active consolidated agents (Phase 3) ---
    PLANNER = "PLANNER"
    CONSOLIDATED_RESEARCHER = "CONSOLIDATED_RESEARCHER"
    CONSOLIDATED_ORACLE = "CONSOLIDATED_ORACLE"
    BUILDER = "BUILDER"
    QUALITY = "QUALITY"
    OPERATIONS = "OPERATIONS"

    # --- Aliases used in specific subsystems ---
    ORCHESTRATOR = "ORCHESTRATOR"
    ARCHITECT = "ARCHITECT"
    RESEARCHER = "RESEARCHER"
    PONDER = "PONDER"

    # --- Legacy (deprecated) — kept for deserialization only ---
    EXPLORER = "EXPLORER"
    ORACLE = "ORACLE"
    LIBRARIAN = "LIBRARIAN"
    EVALUATOR = "EVALUATOR"
    SYNTHESIZER = "SYNTHESIZER"
    UI_PLANNER = "UI_PLANNER"
    SECURITY_AUDITOR = "SECURITY_AUDITOR"
    DATA_ENGINEER = "DATA_ENGINEER"
    DOCUMENTATION_AGENT = "DOCUMENTATION_AGENT"
    COST_PLANNER = "COST_PLANNER"
    TEST_AUTOMATION = "TEST_AUTOMATION"
    EXPERIMENTATION_MANAGER = "EXPERIMENTATION_MANAGER"
    IMPROVEMENT = "IMPROVEMENT"
    USER_INTERACTION = "USER_INTERACTION"
    DEVOPS = "DEVOPS"
    VERSION_CONTROL = "VERSION_CONTROL"
    ERROR_RECOVERY = "ERROR_RECOVERY"
    CONTEXT_MANAGER = "CONTEXT_MANAGER"
    IMAGE_GENERATOR = "IMAGE_GENERATOR"


# Mapping from legacy agent types to their consolidated replacements.
LEGACY_TO_CONSOLIDATED: dict[AgentType, AgentType] = {
    AgentType.EXPLORER: AgentType.CONSOLIDATED_RESEARCHER,
    AgentType.LIBRARIAN: AgentType.CONSOLIDATED_RESEARCHER,
    AgentType.RESEARCHER: AgentType.CONSOLIDATED_RESEARCHER,
    AgentType.UI_PLANNER: AgentType.CONSOLIDATED_RESEARCHER,
    AgentType.DATA_ENGINEER: AgentType.CONSOLIDATED_RESEARCHER,
    AgentType.EVALUATOR: AgentType.QUALITY,
    AgentType.SECURITY_AUDITOR: AgentType.QUALITY,
    AgentType.TEST_AUTOMATION: AgentType.QUALITY,
    AgentType.SYNTHESIZER: AgentType.OPERATIONS,
    AgentType.DOCUMENTATION_AGENT: AgentType.OPERATIONS,
    AgentType.COST_PLANNER: AgentType.OPERATIONS,
    AgentType.ERROR_RECOVERY: AgentType.OPERATIONS,
    AgentType.EXPERIMENTATION_MANAGER: AgentType.OPERATIONS,
    AgentType.IMPROVEMENT: AgentType.OPERATIONS,
    AgentType.USER_INTERACTION: AgentType.OPERATIONS,
    AgentType.DEVOPS: AgentType.OPERATIONS,
    AgentType.VERSION_CONTROL: AgentType.OPERATIONS,
    AgentType.CONTEXT_MANAGER: AgentType.PLANNER,
    AgentType.IMAGE_GENERATOR: AgentType.BUILDER,
    AgentType.ORACLE: AgentType.CONSOLIDATED_ORACLE,
    AgentType.ARCHITECT: AgentType.CONSOLIDATED_ORACLE,
}


def resolve_agent_type(agent_type: AgentType) -> AgentType:
    """Return the consolidated agent type, resolving legacy aliases."""
    return LEGACY_TO_CONSOLIDATED.get(agent_type, agent_type)


class ExecutionMode(Enum):
    """Execution modes available in Vetinari."""

    PLANNING = "planning"  # Read-only mode for analysis and planning
    EXECUTION = "execution"  # Full read/write mode for implementation
    SANDBOX = "sandbox"  # Restricted mode for untrusted code


class ModelProvider(Enum):
    """All recognized model provider types."""

    LOCAL = "local"
    LMSTUDIO = "lmstudio"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    REPLICATE = "replicate"
    COHERE = "cohere"
    OTHER = "other"


class GoalCategory(Enum):
    """9-category goal classification for agent routing.

    Inspired by Oh-My-OpenCode's task category system (visual-engineering,
    deep, quick, ultrabrain).  Each category maps to an agent + mode + model
    tier combination in the TwoLayerOrchestrator.
    """

    CODE = "code"  # implement, build, develop, fix, refactor
    RESEARCH = "research"  # research, analyze, investigate, study
    DOCS = "docs"  # document, readme, api docs, manual
    CREATIVE = "creative"  # write, story, campaign, fiction, narrative
    SECURITY = "security"  # security, audit, vulnerability, pentest
    DATA = "data"  # database, schema, migration, ETL, SQL
    DEVOPS = "devops"  # deploy, CI/CD, docker, kubernetes, pipeline
    UI = "ui"  # UI, UX, frontend, design, wireframe
    IMAGE = "image"  # logo, icon, mockup, diagram, image
    GENERAL = "general"  # fallback — routes to PLANNER for decomposition


class FailureType(Enum):
    """Failure taxonomy for intelligent error handling in the execution engine.

    Classifying failures enables the orchestrator to choose the correct
    recovery strategy rather than blindly retrying or giving up.
    """

    TRANSIENT = "transient"  # Timeout, temp error -> retry same agent
    DECOMPOSITION = "decomposition"  # Too complex -> post to PLANNER for subtask split
    DELEGATION = "delegation"  # Wrong agent -> post to Blackboard for reassignment
    UNSOLVABLE = "unsolvable"  # Genuine failure -> ErrorRecoveryAgent -> user escalation
    POLICY_VIOLATION = "policy"  # Security/constraint violation -> block + report


class SubtaskStatus(str, Enum):
    """Canonical subtask lifecycle status.

    Superset of plan_types.py and subtask_tree.py variants.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class ThinkingMode(str, Enum):
    """Canonical thinking depth levels for skill tools.

    Controls the level of detail and depth in skill tool outputs.
    All skill tools should import this from types.py rather than
    defining their own copy.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class MemoryType(Enum):
    """Canonical memory entry types.

    Superset of all memory types used across the system:
    shared_memory.py, enhanced_memory.py, and core pipeline.
    """

    # Core types
    INTENT = "intent"
    DISCOVERY = "discovery"
    DECISION = "decision"
    PROBLEM = "problem"
    SOLUTION = "solution"
    PATTERN = "pattern"
    WARNING = "warning"
    SUCCESS = "success"
    REFACTOR = "refactor"
    BUGFIX = "bugfix"
    FEATURE = "feature"
    APPROVAL = "approval"
    CONFIG = "config"
    ERROR = "error"
    CONTEXT = "context"
    # Plan execution types (from shared_memory)
    PLAN = "plan"
    WAVE = "wave"
    TASK = "task"
    PLAN_RESULT = "plan_result"
    WAVE_RESULT = "wave_result"
    TASK_RESULT = "task_result"
    MODEL_SELECTION = "model_selection"
    SANDBOX_EVENT = "sandbox_event"
    GOVERNANCE = "governance"
    # Enhanced memory types (from enhanced_memory)
    KNOWLEDGE = "knowledge"
    CODE = "code"
    CONVERSATION = "conversation"
    RESULT = "result"


class CodingTaskType(str, Enum):
    """Types of coding tasks.

    Canonical definition — replaces duplicates in:
    - vetinari.agents.coding_bridge
    - vetinari.coding_agent.engine
    - vetinari.coding_agent.bridge (as BridgeTaskType)
    """

    SCAFFOLD = "scaffold"
    IMPLEMENT = "implement"
    TEST = "test"
    REFACTOR = "refactor"
    REVIEW = "review"
    FIX = "fix"
    DOCUMENT = "document"


class CodingTaskStatus(str, Enum):
    """Status of coding tasks.

    Canonical definition — replaces duplicates in:
    - vetinari.agents.coding_bridge
    - vetinari.coding_agent.engine
    - vetinari.coding_agent.bridge (as BridgeTaskStatus)
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SeverityLevel(str, Enum):
    """Issue severity levels for quality reviews.

    Canonical definition — replaces duplicates in:
    - vetinari.skills.quality_skill
    - vetinari.skills.evaluator
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class QualityGrade(str, Enum):
    """Overall quality grades.

    Canonical definition — replaces duplicates in:
    - vetinari.skills.quality_skill (as QualityGrade)
    - vetinari.skills.evaluator (as QualityScore)
    """

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"
