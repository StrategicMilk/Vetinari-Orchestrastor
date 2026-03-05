"""
HTN-Style Decomposition Template Library
==========================================
Encodes proven decomposition patterns for common task types.

Templates can be used by the DecompositionEngine as a starting point when the
PlannerAgent LLM is unavailable or when the task type matches a known pattern.
Each template maps a task type to an ordered list of standard subtask
descriptions with agent-type hints and dependency arcs.

Custom templates captured from successful executions by WorkflowLearner are
merged in at runtime via ``register_template()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TemplateStep:
    """A single step within a decomposition template."""
    description: str
    agent_type: str = "BUILDER"
    depends_on: List[str] = field(default_factory=list)  # step names


@dataclass
class DecompositionTemplate:
    """
    A reusable task decomposition pattern.

    Attributes:
        name: Human-readable template name.
        keywords: Keywords that trigger this template (lowercased).
        steps: Ordered list of template steps.
        dod_level: Default Definition-of-Done level (Light / Standard / Hard).
    """
    name: str
    keywords: List[str]
    steps: List[TemplateStep]
    dod_level: str = "Standard"

    def to_subtask_list(
        self, parent_task_id: str, depth: int, id_prefix: str = "tmpl"
    ) -> List[Dict]:
        """
        Expand the template into a subtask list compatible with
        DecompositionEngine's output format.
        """
        import uuid

        step_id_map: Dict[str, str] = {}
        subtasks: List[Dict] = []
        for i, step in enumerate(self.steps):
            sid = f"{id_prefix}_{uuid.uuid4().hex[:6]}"
            step_id_map[step.description] = sid
            dep_ids = [
                step_id_map[dep]
                for dep in step.depends_on
                if dep in step_id_map
            ]
            subtasks.append({
                "subtask_id": sid,
                "parent_task_id": parent_task_id,
                "description": step.description,
                "agent_type": step.agent_type,
                "depth": depth + 1,
                "inputs": [],
                "outputs": [],
                "dependencies": dep_ids,
                "acceptance_criteria": f"{step.description} is complete",
            })
        return subtasks


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

_BUILTIN_TEMPLATES: List[DecompositionTemplate] = [
    DecompositionTemplate(
        name="Web Application",
        keywords=["web", "app", "frontend", "react", "vue", "html", "website"],
        dod_level="Standard",
        steps=[
            TemplateStep("Define requirements and wireframes", "EXPLORER"),
            TemplateStep("Set up project structure and dependencies", "BUILDER",
                         ["Define requirements and wireframes"]),
            TemplateStep("Implement backend API", "BUILDER",
                         ["Set up project structure and dependencies"]),
            TemplateStep("Implement frontend components", "BUILDER",
                         ["Implement backend API"]),
            TemplateStep("Write tests", "TEST_AUTOMATION",
                         ["Implement frontend components"]),
            TemplateStep("Deploy and configure CI/CD", "DEVOPS",
                         ["Write tests"]),
        ],
    ),
    DecompositionTemplate(
        name="REST API",
        keywords=["api", "rest", "endpoint", "fastapi", "flask", "django", "service"],
        dod_level="Standard",
        steps=[
            TemplateStep("Define API schema (OpenAPI / Pydantic)", "EXPLORER"),
            TemplateStep("Implement data models", "DATA_ENGINEER",
                         ["Define API schema (OpenAPI / Pydantic)"]),
            TemplateStep("Implement endpoint handlers", "BUILDER",
                         ["Implement data models"]),
            TemplateStep("Add authentication and authorisation", "SECURITY_AUDITOR",
                         ["Implement endpoint handlers"]),
            TemplateStep("Write API tests", "TEST_AUTOMATION",
                         ["Add authentication and authorisation"]),
            TemplateStep("Generate API documentation", "DOCUMENTATION_AGENT",
                         ["Write API tests"]),
        ],
    ),
    DecompositionTemplate(
        name="Data Pipeline",
        keywords=["data", "pipeline", "etl", "database", "sql", "ingest", "transform"],
        dod_level="Standard",
        steps=[
            TemplateStep("Define data schema and models", "DATA_ENGINEER"),
            TemplateStep("Implement data ingestion", "DATA_ENGINEER",
                         ["Define data schema and models"]),
            TemplateStep("Implement transformation logic", "DATA_ENGINEER",
                         ["Implement data ingestion"]),
            TemplateStep("Add validation and error handling", "BUILDER",
                         ["Implement transformation logic"]),
            TemplateStep("Write pipeline tests", "TEST_AUTOMATION",
                         ["Add validation and error handling"]),
            TemplateStep("Document data flow", "DOCUMENTATION_AGENT",
                         ["Write pipeline tests"]),
        ],
    ),
    DecompositionTemplate(
        name="CLI Tool",
        keywords=["cli", "command line", "terminal", "script", "tool", "argparse"],
        dod_level="Standard",
        steps=[
            TemplateStep("Design CLI interface and argument schema", "EXPLORER"),
            TemplateStep("Implement core logic", "BUILDER",
                         ["Design CLI interface and argument schema"]),
            TemplateStep("Implement input validation and error messages", "BUILDER",
                         ["Implement core logic"]),
            TemplateStep("Write tests", "TEST_AUTOMATION",
                         ["Implement input validation and error messages"]),
            TemplateStep("Write README and usage docs", "DOCUMENTATION_AGENT",
                         ["Write tests"]),
        ],
    ),
    DecompositionTemplate(
        name="ML / AI Feature",
        keywords=["ml", "machine learning", "model", "training", "inference",
                  "neural", "transformer", "llm", "ai", "deep learning"],
        dod_level="Hard",
        steps=[
            TemplateStep("Research and select approach", "RESEARCHER"),
            TemplateStep("Prepare and validate dataset", "DATA_ENGINEER",
                         ["Research and select approach"]),
            TemplateStep("Implement model architecture", "BUILDER",
                         ["Prepare and validate dataset"]),
            TemplateStep("Train and tune model", "BUILDER",
                         ["Implement model architecture"]),
            TemplateStep("Evaluate model performance", "EVALUATOR",
                         ["Train and tune model"]),
            TemplateStep("Write inference pipeline", "BUILDER",
                         ["Evaluate model performance"]),
            TemplateStep("Document model card and usage", "DOCUMENTATION_AGENT",
                         ["Write inference pipeline"]),
        ],
    ),
    DecompositionTemplate(
        name="Library / Package",
        keywords=["library", "package", "module", "sdk", "plugin", "extension"],
        dod_level="Hard",
        steps=[
            TemplateStep("Define public API and interface contracts", "EXPLORER"),
            TemplateStep("Implement core functionality", "BUILDER",
                         ["Define public API and interface contracts"]),
            TemplateStep("Write unit tests", "TEST_AUTOMATION",
                         ["Implement core functionality"]),
            TemplateStep("Write integration tests", "TEST_AUTOMATION",
                         ["Write unit tests"]),
            TemplateStep("Generate API reference docs", "DOCUMENTATION_AGENT",
                         ["Write integration tests"]),
            TemplateStep("Create examples and quickstart", "DOCUMENTATION_AGENT",
                         ["Generate API reference docs"]),
        ],
    ),
    DecompositionTemplate(
        name="Bug Fix",
        keywords=["fix", "bug", "error", "crash", "broken", "issue", "regression"],
        dod_level="Standard",
        steps=[
            TemplateStep("Reproduce and document the bug", "EXPLORER"),
            TemplateStep("Identify root cause", "EXPLORER",
                         ["Reproduce and document the bug"]),
            TemplateStep("Implement fix", "BUILDER",
                         ["Identify root cause"]),
            TemplateStep("Write regression test", "TEST_AUTOMATION",
                         ["Implement fix"]),
            TemplateStep("Verify fix in context", "EVALUATOR",
                         ["Write regression test"]),
        ],
    ),
    DecompositionTemplate(
        name="Refactor",
        keywords=["refactor", "clean", "reorganize", "restructure", "rewrite",
                  "extract", "simplify"],
        dod_level="Standard",
        steps=[
            TemplateStep("Analyse existing code and identify smells", "EXPLORER"),
            TemplateStep("Create refactoring plan", "EXPLORER",
                         ["Analyse existing code and identify smells"]),
            TemplateStep("Apply refactoring changes", "BUILDER",
                         ["Create refactoring plan"]),
            TemplateStep("Run existing test suite", "TEST_AUTOMATION",
                         ["Apply refactoring changes"]),
            TemplateStep("Review and validate behavioural equivalence", "EVALUATOR",
                         ["Run existing test suite"]),
        ],
    ),
    DecompositionTemplate(
        name="Documentation",
        keywords=["document", "readme", "docs", "explain", "guide", "tutorial",
                  "wiki", "docstring"],
        dod_level="Light",
        steps=[
            TemplateStep("Analyse existing code and interfaces", "EXPLORER"),
            TemplateStep("Draft documentation outline", "DOCUMENTATION_AGENT",
                         ["Analyse existing code and interfaces"]),
            TemplateStep("Write documentation content", "DOCUMENTATION_AGENT",
                         ["Draft documentation outline"]),
            TemplateStep("Review for accuracy and completeness", "EVALUATOR",
                         ["Write documentation content"]),
        ],
    ),
    DecompositionTemplate(
        name="Research Task",
        keywords=["research", "analyze", "investigate", "study", "survey",
                  "compare", "evaluate"],
        dod_level="Light",
        steps=[
            TemplateStep("Define research scope and questions", "RESEARCHER"),
            TemplateStep("Gather sources and references", "LIBRARIAN",
                         ["Define research scope and questions"]),
            TemplateStep("Analyze and synthesize findings", "SYNTHESIZER",
                         ["Gather sources and references"]),
            TemplateStep("Write research report", "DOCUMENTATION_AGENT",
                         ["Analyze and synthesize findings"]),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

class TemplateRegistry:
    """Manages the library of decomposition templates."""

    def __init__(self) -> None:
        self._templates: List[DecompositionTemplate] = list(_BUILTIN_TEMPLATES)

    def register(self, template: DecompositionTemplate) -> None:
        """Register a custom template (replaces any with the same name)."""
        self._templates = [t for t in self._templates if t.name != template.name]
        self._templates.append(template)
        logger.info(f"TemplateRegistry: registered '{template.name}'")

    def find(
        self,
        *,
        keywords: Optional[List[str]] = None,
        name: Optional[str] = None,
    ) -> Optional[DecompositionTemplate]:
        """
        Find the best-matching template.

        Priority:
        1. Exact name match.
        2. Highest keyword overlap.
        """
        if name:
            for t in self._templates:
                if t.name.lower() == name.lower():
                    return t

        if keywords:
            kw_lower = [k.lower() for k in keywords]
            scored = [
                (sum(1 for kw in kw_lower if kw in t.keywords), t)
                for t in self._templates
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored and scored[0][0] > 0:
                return scored[0][1]

        return None

    def all_templates(self) -> List[DecompositionTemplate]:
        return list(self._templates)


# Module-level singleton
_registry: Optional[TemplateRegistry] = None


def get_template_registry() -> TemplateRegistry:
    """Return the module-level TemplateRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = TemplateRegistry()
    return _registry


def register_template(template: DecompositionTemplate) -> None:
    """Convenience wrapper for WorkflowLearner and other callers."""
    get_template_registry().register(template)
