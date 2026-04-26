"""Decomposition Engine.

====================
Provides task decomposition services used by the Decomposition Lab UI.
Wraps the ForemanAgent and planning infrastructure.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from vetinari.types import AgentType

logger = logging.getLogger(__name__)

# Decomposition configuration knobs
DEFAULT_MAX_DEPTH = 14
MIN_MAX_DEPTH = 12
MAX_MAX_DEPTH = 16
SEED_RATE = 0.3  # 30% of tasks seeded with refined subtasks
SEED_MIX = 0.5  # Balance between breadth and depth seeding

# Recursive decomposition limit (ADR-0081) — prevents unbounded Foreman chains
MAX_RECURSIVE_DEPTH = 3

# Definition of Done criteria per level
_DOD_CRITERIA = {
    "Light": [
        "Code compiles without errors",
        "Basic functionality works",
        "No blocking security issues",
    ],
    "Standard": [
        "Code compiles and lints cleanly",
        "Unit tests pass (>70% coverage)",
        "Security scan passes",
        "Documentation updated",
        "Code reviewed",
    ],
    "Hard": [
        "Code compiles, lints, and type-checks",
        "Unit + integration tests pass (>85% coverage)",
        "Security scan passes with no high/critical findings",
        "Full API documentation generated",
        "Performance benchmarks met",
        "Accessibility audit passed",
        "Peer reviewed and approved",
    ],
}

# Definition of Ready criteria
_DOR_CRITERIA = {
    "Light": [
        "Task description is clear",
        "Inputs are defined",
    ],
    "Standard": [
        "Task description is unambiguous",
        "Inputs and expected outputs defined",
        "Dependencies identified",
        "Estimated effort provided",
    ],
    "Hard": [
        "Task description is unambiguous and reviewed",
        "All inputs, outputs, and side-effects documented",
        "Dependencies fully resolved",
        "Effort estimate reviewed by at least one peer",
        "Acceptance criteria agreed upon",
        "Risk assessment completed",
    ],
}


@dataclass(frozen=True)
class SubtaskSpec:
    """A single decomposed subtask."""

    subtask_id: str
    parent_task_id: str
    description: str
    agent_type: str
    depth: int
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    dod_criteria: list[str] = field(default_factory=list)
    dor_criteria: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __repr__(self) -> str:
        return f"SubtaskSpec(subtask_id={self.subtask_id!r}, agent_type={self.agent_type!r}, depth={self.depth!r})"


@dataclass(frozen=True)
class DecompositionEvent:
    """A historical decomposition event."""

    event_id: str
    plan_id: str
    task_id: str
    depth: int
    seeds_used: list[str]
    subtasks_created: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __repr__(self) -> str:
        return (
            f"DecompositionEvent(event_id={self.event_id!r}, plan_id={self.plan_id!r}, "
            f"task_id={self.task_id!r}, depth={self.depth!r}, "
            f"subtasks_created={self.subtasks_created!r})"
        )


class DecompositionEngine:
    """Orchestrates task decomposition using the ForemanAgent.

    Used by the Decomposition Lab in the web UI.
    """

    SEED_MIX = SEED_MIX
    SEED_RATE = SEED_RATE
    DEFAULT_MAX_DEPTH = DEFAULT_MAX_DEPTH
    MIN_MAX_DEPTH = MIN_MAX_DEPTH
    MAX_MAX_DEPTH = MAX_MAX_DEPTH

    def __init__(self):
        self._history: list[DecompositionEvent] = []
        self._templates: list[dict[str, Any]] = self._build_default_templates()

    def _build_default_templates(self) -> list[dict[str, Any]]:
        """Build built-in decomposition templates."""
        return [
            {
                "template_id": "web_app",
                "name": "Web Application",
                "keywords": ["web", "app", "frontend", "react", "vue", "html"],
                "agent_type": AgentType.WORKER.value,
                "dod_level": "Standard",
                "subtasks": [
                    "Define requirements and wireframes",
                    "Set up project structure and dependencies",
                    "Implement backend API",
                    "Implement frontend components",
                    "Write tests",
                    "Deploy and configure CI/CD",
                ],
            },
            {
                "template_id": "data_pipeline",
                "name": "Data Pipeline",
                "keywords": ["data", "pipeline", "etl", "database", "sql"],
                "agent_type": AgentType.WORKER.value,
                "dod_level": "Standard",
                "subtasks": [
                    "Define data schema and models",
                    "Implement data ingestion",
                    "Implement transformation logic",
                    "Add validation and error handling",
                    "Write pipeline tests",
                    "Document data flow",
                ],
            },
            {
                "template_id": "research",
                "name": "Research Task",
                "keywords": ["research", "analyze", "investigate", "study"],
                "agent_type": AgentType.WORKER.value,
                "dod_level": "Light",
                "subtasks": [
                    "Define research scope and questions",
                    "Gather sources and references",
                    "Analyze and synthesize findings",
                    "Write research report",
                ],
            },
            {
                "template_id": "cli_tool",
                "name": "CLI Tool",
                "keywords": ["cli", "command", "terminal", "argparse", "click", "typer", "shell", "script"],
                "agent_type": AgentType.WORKER.value,
                "dod_level": "Standard",
                "subtasks": [
                    "Define commands, flags, and argument schema",
                    "Implement argument parsing and validation",
                    "Implement core command logic",
                    "Add help text, usage examples, and error messages",
                    "Write unit and integration tests",
                    "Package and document installation instructions",
                ],
            },
            {
                "template_id": "api_service",
                "name": "REST API Service",
                "keywords": ["api", "rest", "endpoint", "service", "fastapi", "flask", "django", "http"],
                "agent_type": AgentType.WORKER.value,
                "dod_level": "Hard",
                "subtasks": [
                    "Define API contract, endpoints, and data models",
                    "Implement authentication and authorization",
                    "Implement endpoint handlers and business logic",
                    "Add request validation and error handling",
                    "Write unit and integration tests",
                    "Generate OpenAPI documentation",
                    "Configure deployment and environment settings",
                ],
            },
            {
                "template_id": "library",
                "name": "Reusable Library",
                "keywords": ["library", "package", "module", "sdk", "framework", "reusable", "pypi", "npm"],
                "agent_type": AgentType.WORKER.value,
                "dod_level": "Hard",
                "subtasks": [
                    "Design public API and interface contracts",
                    "Implement core functionality",
                    "Write comprehensive unit and integration tests",
                    "Write API reference documentation and usage examples",
                    "Configure packaging, versioning, and build tooling",
                    "Publish to package registry",
                ],
            },
            {
                "template_id": "document_generation",
                "name": "Document Generation",
                "keywords": ["document", "report", "generate", "pdf", "markdown", "template", "export"],
                "agent_type": AgentType.WORKER.value,
                "dod_level": "Standard",
                "subtasks": [
                    "Define document structure and outline",
                    "Gather and validate source data or content",
                    "Draft document sections",
                    "Review and revise for accuracy and clarity",
                    "Apply formatting, styling, and branding",
                    "Export and validate final output",
                ],
            },
            {
                "template_id": "creative_writing",
                "name": "Creative Writing",
                "keywords": ["creative", "writing", "story", "content", "blog", "article", "fiction", "copy"],
                "agent_type": AgentType.WORKER.value,
                "dod_level": "Light",
                "subtasks": [
                    "Brainstorm concepts, themes, and angle",
                    "Create detailed outline and structure",
                    "Write first draft",
                    "Revise for voice, pacing, and coherence",
                    "Polish grammar, style, and final presentation",
                ],
            },
            {
                "template_id": "testing",
                "name": "Test Suite Development",
                "keywords": ["test", "testing", "qa", "coverage", "pytest", "jest", "unittest", "tdd"],
                "agent_type": AgentType.INSPECTOR.value,
                "dod_level": "Standard",
                "subtasks": [
                    "Define test plan and coverage goals",
                    "Write unit tests for core components",
                    "Write integration tests for system boundaries",
                    "Add edge case and negative path tests",
                    "Measure coverage and close gaps",
                    "Integrate tests into CI pipeline",
                ],
            },
            {
                "template_id": "refactoring",
                "name": "Code Refactoring",
                "keywords": ["refactor", "refactoring", "cleanup", "technical debt", "restructure", "reorganize"],
                "agent_type": AgentType.WORKER.value,
                "dod_level": "Standard",
                "subtasks": [
                    "Analyze codebase and identify problem areas",
                    "Define refactoring plan and risk assessment",
                    "Apply incremental structural changes",
                    "Verify existing tests still pass after each change",
                    "Update documentation and comments",
                    "Perform final review and cleanup",
                ],
            },
            {
                "template_id": "debugging",
                "name": "Bug Investigation and Fix",
                "keywords": ["bug", "debug", "fix", "error", "crash", "issue", "defect", "regression"],
                "agent_type": AgentType.INSPECTOR.value,
                "dod_level": "Standard",
                "subtasks": [
                    "Reproduce the bug reliably with a minimal test case",
                    "Diagnose root cause through logs and code inspection",
                    "Implement targeted fix",
                    "Write regression test to prevent recurrence",
                    "Verify fix across affected scenarios",
                    "Document root cause and resolution",
                ],
            },
            {
                "template_id": "migration",
                "name": "System or Data Migration",
                "keywords": ["migration", "migrate", "upgrade", "port", "convert", "transfer", "move"],
                "agent_type": AgentType.WORKER.value,
                "dod_level": "Hard",
                "subtasks": [
                    "Inventory current state and map to target state",
                    "Create migration plan with rollback strategy",
                    "Implement migration scripts or procedures",
                    "Run migration in staging and validate data integrity",
                    "Execute production migration with monitoring",
                    "Decommission legacy resources and update documentation",
                ],
            },
            {
                "template_id": "security_audit",
                "name": "Security Audit",
                "keywords": ["security", "audit", "vulnerability", "penetration", "pentest", "cve", "owasp"],
                "agent_type": AgentType.INSPECTOR.value,
                "dod_level": "Hard",
                "subtasks": [
                    "Define scope and threat model",
                    "Run automated vulnerability scans",
                    "Manually analyze authentication, authorization, and data handling",
                    "Prioritize findings by severity and exploitability",
                    "Remediate critical and high-severity issues",
                    "Verify remediations and produce final report",
                ],
            },
            {
                "template_id": "data_analysis",
                "name": "Data Analysis Project",
                "keywords": ["analysis", "analytics", "dataset", "statistics", "visualization", "insight", "notebook"],
                "agent_type": AgentType.WORKER.value,
                "dod_level": "Standard",
                "subtasks": [
                    "Define analysis objectives and success metrics",
                    "Collect and load raw data",
                    "Clean, normalize, and validate data quality",
                    "Perform exploratory analysis and statistical tests",
                    "Create visualizations and charts",
                    "Write findings report with recommendations",
                ],
            },
            {
                "template_id": "infrastructure",
                "name": "Infrastructure and DevOps",
                "keywords": [
                    "infrastructure",
                    "devops",
                    "deployment",
                    "kubernetes",
                    "terraform",
                    "ansible",
                    "ci",
                    "cd",
                    "cloud",
                ],
                "agent_type": AgentType.WORKER.value,
                "dod_level": "Hard",
                "subtasks": [
                    "Define infrastructure requirements and architecture",
                    "Write provisioning scripts or IaC configuration",
                    "Configure networking, security groups, and IAM",
                    "Deploy to staging and run smoke tests",
                    "Set up monitoring, alerting, and logging",
                    "Document runbooks and maintenance procedures",
                ],
            },
        ]

    def get_templates(
        self,
        keywords: list[str] | None = None,
        agent_type: str | None = None,
        dod_level: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return matching decomposition templates.

        Args:
            keywords: The keywords.
            agent_type: The agent type.
            dod_level: The dod level.

        Returns:
            List of template dicts that satisfy all provided filters.
            Filters are applied conjunctively; passing no filters returns
            all registered templates.
        """
        results = self._templates[:]
        if keywords:
            kw_lower = [k.lower() for k in keywords]
            results = [t for t in results if any(kw in t.get("keywords", []) for kw in kw_lower)]
        if agent_type:
            results = [t for t in results if t.get("agent_type") == agent_type.upper()]
        if dod_level:
            results = [t for t in results if t.get("dod_level") == dod_level]
        return results

    def get_dod_criteria(self, level: str = "Standard") -> list[str]:
        """Return Definition of Done criteria for the given quality level.

        Args:
            level: Quality tier name (e.g. "Standard", "Strict"). Falls back
                to "Standard" when the requested level is not defined.

        Returns:
            List of criterion strings that must be satisfied for task completion.
        """
        return _DOD_CRITERIA.get(level, _DOD_CRITERIA["Standard"])

    def get_dor_criteria(self, level: str = "Standard") -> list[str]:
        """Return Definition of Ready criteria for the given quality level.

        Args:
            level: Quality tier name (e.g. "Standard", "Strict"). Falls back
                to "Standard" when the requested level is not defined.

        Returns:
            List of criterion strings that must be met before a task can start.
        """
        return _DOR_CRITERIA.get(level, _DOR_CRITERIA["Standard"])

    def decompose_task(
        self,
        task_prompt: str,
        parent_task_id: str = "root",
        depth: int = 0,
        max_depth: int = DEFAULT_MAX_DEPTH,
        plan_id: str = "default",
    ) -> list[dict[str, Any]]:
        """Decompose a task into subtasks using the ForemanAgent.

        Falls back to keyword-based decomposition.

        Args:
            task_prompt: The task prompt.
            parent_task_id: The parent task id.
            depth: The depth.
            max_depth: The max depth.
            plan_id: The plan id.

        Returns:
            List of subtask dicts, each containing ``subtask_id``,
            ``parent_task_id``, ``description``, ``agent_type``, ``depth``,
            ``inputs``, ``outputs``, ``dependencies``, and
            ``acceptance_criteria``.  Returns an empty list when
            ``max_depth`` is reached.
        """
        max_depth = max(MIN_MAX_DEPTH, min(max_depth, MAX_MAX_DEPTH))

        if depth >= max_depth:
            logger.warning("Max decomposition depth %s reached for task: %s", max_depth, task_prompt[:50])
            return []

        try:
            from vetinari.agents import get_foreman_agent
            from vetinari.agents.contracts import AgentTask

            planner = get_foreman_agent()
            agent_task = AgentTask(
                task_id=f"decomp_{uuid.uuid4().hex[:8]}",
                agent_type=AgentType.FOREMAN,
                description=f"Decompose: {task_prompt}",
                prompt=task_prompt,
                context={"depth": depth, "max_depth": max_depth, "plan_id": plan_id},
            )
            result = planner.execute(agent_task)
            if result.success and isinstance(result.output, dict):
                tasks = result.output.get("tasks", [])
                subtasks = []
                for t in tasks:
                    subtask = {
                        "subtask_id": t.get("id", str(uuid.uuid4())[:8]),
                        "parent_task_id": parent_task_id,
                        "description": t.get("description", ""),
                        "agent_type": t.get("assigned_agent", AgentType.WORKER.value),
                        "depth": depth + 1,
                        "inputs": t.get("inputs", []),
                        "outputs": t.get("outputs", []),
                        "dependencies": t.get("dependencies", []),
                        "acceptance_criteria": t.get("acceptance_criteria", ""),
                    }
                    subtasks.append(subtask)

                # Record history
                self._history.append(
                    DecompositionEvent(
                        event_id=str(uuid.uuid4()),
                        plan_id=plan_id,
                        task_id=parent_task_id,
                        depth=depth,
                        seeds_used=[],
                        subtasks_created=len(subtasks),
                    ),
                )
                return subtasks
        except Exception as e:
            logger.warning("LLM decomposition failed, using keyword fallback: %s", e)

        # Keyword fallback
        return self._keyword_decompose(task_prompt, parent_task_id, depth)

    def _keyword_decompose(self, task_prompt: str, parent_task_id: str, depth: int) -> list[dict[str, Any]]:
        """Simple keyword-based decomposition fallback."""
        task_lower = task_prompt.lower()
        subtasks = []

        def make_subtask(desc: str, agent: str, deps: list[str] | None = None) -> dict[str, Any]:
            """Build a subtask dict with a fresh UUID-based ID.

            Args:
                desc: The desc.
                agent: The agent.
                deps: The deps.

            Returns:
                Subtask dict with a unique ``subtask_id``, the given
                ``description`` and ``agent_type``, and the current
                recursion ``depth + 1``.
            """
            sid = f"st_{uuid.uuid4().hex[:6]}"
            return {
                "subtask_id": sid,
                "parent_task_id": parent_task_id,
                "description": desc,
                "agent_type": agent,
                "depth": depth + 1,
                "inputs": [],
                "outputs": [],
                "dependencies": deps or [],  # noqa: VET112 - empty fallback preserves optional request metadata contract
                "acceptance_criteria": f"{desc} is complete",
            }

        s1 = make_subtask("Analyze requirements and define scope", AgentType.WORKER.value)
        subtasks.append(s1)

        if any(kw in task_lower for kw in ["code", "implement", "build", "develop"]):
            s2 = make_subtask("Implement core functionality", AgentType.WORKER.value, [s1["subtask_id"]])
            subtasks.append(s2)
            s3 = make_subtask("Write tests", AgentType.INSPECTOR.value, [s2["subtask_id"]])
            subtasks.append(s3)

        if any(kw in task_lower for kw in ["ui", "frontend", "web", "interface"]):
            prev = subtasks[-1]["subtask_id"] if subtasks else s1["subtask_id"]
            subtasks.append(make_subtask("Design and implement UI", AgentType.WORKER.value, [prev]))

        last = subtasks[-1]["subtask_id"] if subtasks else s1["subtask_id"]
        subtasks.append(make_subtask("Review and document", AgentType.INSPECTOR.value, [last]))

        return subtasks

    def decompose_recursive(
        self,
        task_prompt: str,
        parent_task_id: str = "root",
        plan_id: str = "default",
        recursive_depth: int = 0,
        complexity_threshold: int = 4,
    ) -> list[dict[str, Any]]:
        """Recursively decompose a task up to MAX_RECURSIVE_DEPTH levels.

        Performs an initial decomposition then recurses into any subtask
        whose description word count exceeds *complexity_threshold*, treating
        it as a candidate for further breakdown.  Each recursion decrements
        the available depth, and a hard stop at MAX_RECURSIVE_DEPTH prevents
        unbounded Foreman chains (ADR-0081).

        Args:
            task_prompt: The task or goal description to decompose.
            parent_task_id: The ID of the parent task.
            plan_id: Plan identifier for history tracking.
            recursive_depth: Current recursion depth (0 = top-level call).
            complexity_threshold: Minimum word count for a subtask to be
                recursively decomposed.  Lower values produce deeper trees.

        Returns:
            Flat list of subtask dicts, including all recursively generated
            subtasks with updated ``parent_task_id`` linkages.
        """
        if recursive_depth >= MAX_RECURSIVE_DEPTH:
            logger.debug(
                "[DecompositionEngine] MAX_RECURSIVE_DEPTH=%d reached for task: %s",
                MAX_RECURSIVE_DEPTH,
                task_prompt[:60],
            )
            return self.decompose_task(
                task_prompt=task_prompt,
                parent_task_id=parent_task_id,
                depth=0,
                plan_id=plan_id,
            )

        first_level = self.decompose_task(
            task_prompt=task_prompt,
            parent_task_id=parent_task_id,
            depth=0,
            plan_id=plan_id,
        )

        all_subtasks: list[dict[str, Any]] = []
        for subtask in first_level:
            all_subtasks.append(subtask)
            desc = subtask.get("description", "")
            # Recurse into subtasks that look complex (many words suggest compound task)
            if len(desc.split()) >= complexity_threshold:
                child_tasks = self.decompose_recursive(
                    task_prompt=desc,
                    parent_task_id=subtask.get("subtask_id", parent_task_id),
                    plan_id=plan_id,
                    recursive_depth=recursive_depth + 1,
                    complexity_threshold=complexity_threshold,
                )
                all_subtasks.extend(child_tasks)

        logger.info(
            "[DecompositionEngine] Recursive decomp depth=%d produced %d total subtasks for: %s",
            recursive_depth,
            len(all_subtasks),
            task_prompt[:60],
        )
        return all_subtasks

    def get_decomposition_history(self, plan_id: str | None = None) -> list[DecompositionEvent]:
        """Return decomposition history, optionally filtered by plan_id.

        Returns:
            List of results.
        """
        if plan_id:
            return [e for e in self._history if e.plan_id == plan_id]
        return list(self._history)


# Module-level singleton
_decomposition_engine: DecompositionEngine | None = None


def _get_engine() -> DecompositionEngine:
    global _decomposition_engine
    if _decomposition_engine is None:
        _decomposition_engine = DecompositionEngine()
    return _decomposition_engine


# Exported instance used by web_ui.py
decomposition_engine = _get_engine()
