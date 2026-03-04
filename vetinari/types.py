"""
Vetinari Canonical Type Definitions
====================================
Single source of truth for all shared enums and base types.
All modules should import from here rather than defining their own.
"""

from enum import Enum


class TaskStatus(Enum):
    """Canonical task execution status."""
    PENDING = "pending"
    BLOCKED = "blocked"       # Waiting for dependencies
    READY = "ready"           # Dependencies met, awaiting execution
    ASSIGNED = "assigned"     # Assigned to a model/agent
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING = "waiting"       # Waiting for human input


class PlanStatus(Enum):
    """Canonical plan lifecycle status."""
    DRAFT = "draft"
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentType(Enum):
    """All recognized agent types in the system."""
    PLANNER = "planner"
    EXPLORER = "explorer"
    ORACLE = "oracle"
    LIBRARIAN = "librarian"
    RESEARCHER = "researcher"
    EVALUATOR = "evaluator"
    SYNTHESIZER = "synthesizer"
    BUILDER = "builder"
    UI_PLANNER = "ui_planner"
    SECURITY_AUDITOR = "security_auditor"
    DATA_ENGINEER = "data_engineer"
    DOCUMENTATION_AGENT = "documentation_agent"
    COST_PLANNER = "cost_planner"
    TEST_AUTOMATION = "test_automation"
    EXPERIMENTATION_MANAGER = "experimentation_manager"
    IMPROVEMENT = "improvement"
    USER_INTERACTION = "user_interaction"
    DEVOPS = "devops"
    VERSION_CONTROL = "version_control"
    ERROR_RECOVERY = "error_recovery"
    CONTEXT_MANAGER = "context_manager"
    IMAGE_GENERATOR = "image_generator"


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


class MemoryType(Enum):
    """Canonical memory entry types."""
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
