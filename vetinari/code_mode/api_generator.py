"""Code Mode API Generator (C18).

===============================
Introspects all 6 agents and their modes to produce a typed Python API
that LLM-generated code can call.

Generates a ``VetinariAPI`` class with methods like:
  - ``research(query, mode="domain_research")``
  - ``build(spec, mode="build")``
  - ``review(code, mode="code_review")``

Type hints come from C6 Pydantic schemas. Auto-regenerates when modes change.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Agent → method mapping ────────────────────────────────────────────

_AGENT_API_MAP = {
    "PLANNER": {
        "method": "plan",
        "modes": ["plan", "clarify", "summarise", "prune", "extract", "consolidate"],
        "description": "Planning, clarification, and summarisation",
    },
    "CONSOLIDATED_RESEARCHER": {
        "method": "research",
        "modes": [
            "code_discovery",
            "domain_research",
            "api_lookup",
            "lateral_thinking",
            "ui_design",
            "database",
            "devops",
            "git_workflow",
        ],
        "description": "Research, code discovery, and domain analysis",
    },
    "CONSOLIDATED_ORACLE": {
        "method": "advise",
        "modes": [
            "architecture",
            "risk_assessment",
            "ontological_analysis",
            "contrarian_review",
        ],
        "description": "Architecture advice, risk assessment, contrarian review",
    },
    "BUILDER": {
        "method": "build",
        "modes": ["build", "image_generation"],
        "description": "Code scaffolding and image generation",
    },
    "QUALITY": {
        "method": "review",
        "modes": ["code_review", "security_audit", "test_generation", "simplification"],
        "description": "Code review, security audit, test generation",
    },
    "OPERATIONS": {
        "method": "operate",
        "modes": [
            "documentation",
            "creative_writing",
            "cost_analysis",
            "experiment",
            "error_recovery",
            "synthesis",
            "improvement",
            "monitor",
            "devops_ops",
        ],
        "description": "Documentation, cost analysis, error recovery, synthesis",
    },
}


def generate_api_docstring() -> str:
    """Generate comprehensive API documentation for LLM consumption.

    This docstring is injected into the LLM prompt so it knows what
    methods are available on the ``VetinariAPI`` object.

    Returns:
        The result string.
    """
    lines = [
        "# VetinariAPI Reference",
        "",
        "The `api` object provides methods to invoke Vetinari agents.",
        "Each method takes a description/query string and an optional mode.",
        "",
    ]

    for agent_type, info in _AGENT_API_MAP.items():
        method = info["method"]
        modes = info["modes"]
        desc = info["description"]
        default_mode = modes[0] if modes else "default"

        lines.append(f'## api.{method}(description, mode="{default_mode}")')
        lines.append(f"  Agent: {agent_type}")
        lines.append(f"  {desc}")
        lines.append(f"  Available modes: {', '.join(modes)}")
        lines.append("  Returns: dict with agent output")
        lines.append("")

    lines.extend(
        [
            "## Usage Example",
            "```python",
            'research_result = api.research("Find authentication patterns in codebase")',
            'plan = api.plan("Create implementation plan for OAuth2")',
            'code = api.build("Implement OAuth2 login endpoint", mode="build")',
            'review = api.review(code["scaffold_code"], mode="security_audit")',
            'docs = api.operate("Generate API docs for OAuth2 module", mode="documentation")',
            "```",
        ]
    )

    return "\n".join(lines)


class VetinariAPI:
    """Runtime API class that wraps agent execution.

    Injected into the sandbox so LLM-generated code can call agents.
    All methods are read-only wrappers — no filesystem/network access.
    """

    def __init__(self, agent_context: dict[str, Any] | None = None):
        self._context = agent_context or {}
        self._execution_log: list[dict[str, Any]] = []

    def _execute_agent(
        self,
        agent_type: str,
        description: str,
        mode: str = "",
    ) -> dict[str, Any]:
        """Internal: execute an agent task and return the result dict."""
        try:
            from vetinari.agents.contracts import AgentTask
            from vetinari.types import AgentType

            at = AgentType[agent_type]
            task = AgentTask(
                task_id=f"code_mode_{len(self._execution_log)}",
                agent_type=at,
                description=description,
                prompt=description,
                context={"mode": mode} if mode else {},
            )

            # Get the agent singleton
            from vetinari.orchestration.two_layer import get_two_layer_orchestrator

            orch = get_two_layer_orchestrator()
            agent = orch._get_agent(agent_type)

            if agent is None:
                return {"error": f"Agent {agent_type} not available"}

            result = agent.execute(task)
            output = result.output if result.success else {"error": result.errors}

            self._execution_log.append(
                {
                    "agent_type": agent_type,
                    "mode": mode,
                    "description": description[:200],
                    "success": result.success,
                }
            )

            return output if isinstance(output, dict) else {"result": output}
        except Exception as e:
            logger.warning("Code mode agent call failed: %s", e)
            return {"error": str(e)}

    def plan(self, description: str, mode: str = "plan") -> dict[str, Any]:
        """Invoke PlannerAgent."""
        return self._execute_agent("PLANNER", description, mode)

    def research(self, query: str, mode: str = "domain_research") -> dict[str, Any]:
        """Invoke ConsolidatedResearcherAgent."""
        return self._execute_agent("CONSOLIDATED_RESEARCHER", query, mode)

    def advise(self, description: str, mode: str = "architecture") -> dict[str, Any]:
        """Invoke ConsolidatedOracleAgent."""
        return self._execute_agent("CONSOLIDATED_ORACLE", description, mode)

    def build(self, spec: str, mode: str = "build") -> dict[str, Any]:
        """Invoke BuilderAgent."""
        return self._execute_agent("BUILDER", spec, mode)

    def review(self, code: str, mode: str = "code_review") -> dict[str, Any]:
        """Invoke QualityAgent."""
        return self._execute_agent("QUALITY", code, mode)

    def operate(self, description: str, mode: str = "documentation") -> dict[str, Any]:
        """Invoke OperationsAgent."""
        return self._execute_agent("OPERATIONS", description, mode)

    @property
    def execution_log(self) -> list[dict[str, Any]]:
        """Return the log of all agent calls made."""
        return list(self._execution_log)
