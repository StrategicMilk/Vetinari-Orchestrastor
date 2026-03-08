"""
Consolidated Architect Agent (Phase 3)
=======================================
Replaces: UI_PLANNER + DATA_ENGINEER + DEVOPS + VERSION_CONTROL

Modes:
- ui_design: UI/UX design, wireframes, components, accessibility (from UI_PLANNER)
- database: Schema design, migrations, ETL pipelines (from DATA_ENGINEER)
- devops: CI/CD, containers, IaC, deployment strategies (from DEVOPS)
- git_workflow: Branch strategy, commit conventions, PR templates (from VERSION_CONTROL)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from vetinari.agents.multi_mode_agent import MultiModeAgent
from vetinari.agents.contracts import AgentResult, AgentTask, AgentType, VerificationResult

logger = logging.getLogger(__name__)


class ArchitectAgent(MultiModeAgent):
    """Unified architecture agent for UI, data, DevOps, and version control design."""

    MODES = {
        "ui_design": "_execute_ui_design",
        "database": "_execute_database",
        "devops": "_execute_devops",
        "git_workflow": "_execute_git_workflow",
    }
    DEFAULT_MODE = "ui_design"
    MODE_KEYWORDS = {
        "ui_design": ["ui", "ux", "frontend", "component", "wireframe", "layout", "design token",
                       "accessibility", "wcag", "responsive", "css", "react", "interface"],
        "database": ["database", "schema", "table", "migration", "etl", "pipeline", "sql",
                      "data model", "foreign key", "index", "query", "orm"],
        "devops": ["ci/cd", "docker", "kubernetes", "terraform", "ansible", "deploy",
                    "container", "pipeline", "helm", "monitoring", "infrastructure"],
        "git_workflow": ["git", "branch", "commit", "merge", "pull request", "pr", "release",
                         "changelog", "tag", "rebase", "version"],
    }
    LEGACY_TYPE_TO_MODE = {
        "UI_PLANNER": "ui_design",
        "DATA_ENGINEER": "database",
        "DEVOPS": "devops",
        "VERSION_CONTROL": "git_workflow",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.ARCHITECT, config)

    def _get_base_system_prompt(self) -> str:
        return (
            "You are Vetinari's Architect Agent. You design UI/UX, database schemas, "
            "DevOps pipelines, and git workflows. You produce specifications, templates, "
            "and design documents — not implementation code."
        )

    def _get_mode_system_prompt(self, mode: str) -> str:
        prompts = {
            "ui_design": (
                "You are Vetinari's UI/UX Design Specialist. Your role is to:\n"
                "- Design user interfaces with wireframes and component hierarchies\n"
                "- Define design tokens (colors, typography, spacing, shadows)\n"
                "- Create UX flows and interaction patterns\n"
                "- Ensure accessibility compliance (WCAG AA)\n"
                "- Generate component specifications with props and states\n\n"
                "Default framework: React. Include responsive breakpoints."
            ),
            "database": (
                "You are Vetinari's Data Architecture Specialist. Your role is to:\n"
                "- Design database schemas (tables, relationships, constraints)\n"
                "- Create migration scripts with version tracking\n"
                "- Design ETL/data pipelines (batch, streaming, CDC)\n"
                "- Optimize query patterns and indexing strategies\n\n"
                "Include data validation rules and integrity constraints."
            ),
            "devops": (
                "You are Vetinari's DevOps Specialist. Your role is to:\n"
                "- Design CI/CD pipelines with stages and gates\n"
                "- Create containerization configs (Dockerfile, docker-compose)\n"
                "- Generate IaC templates (Terraform, Ansible)\n"
                "- Plan deployment strategies (blue-green, canary, rolling)\n"
                "- Design monitoring, alerting, and health checks\n\n"
                "Include security hardening and runbook references."
            ),
            "git_workflow": (
                "You are Vetinari's Version Control Specialist. Your role is to:\n"
                "- Design branch naming conventions and strategies\n"
                "- Generate commit messages following Conventional Commits\n"
                "- Create PR/MR templates with review checklists\n"
                "- Analyze merge conflicts and suggest resolutions\n"
                "- Generate changelogs and release notes\n\n"
                "Follow GitFlow or trunk-based development as appropriate."
            ),
        }
        return prompts.get(mode, "")

    def verify(self, output: Any) -> VerificationResult:
        if output is None:
            return VerificationResult(passed=False, issues=[{"message": "No output"}], score=0.0)
        if isinstance(output, dict):
            has_design = bool(
                output.get("design") or output.get("schema") or output.get("pipeline")
                or output.get("components") or output.get("workflow")
            )
            return VerificationResult(passed=has_design, score=0.8 if has_design else 0.4)
        return VerificationResult(passed=True, score=0.6)

    # ------------------------------------------------------------------
    # UI Design (from UIPlannerAgent)
    # ------------------------------------------------------------------

    def _execute_ui_design(self, task: AgentTask) -> AgentResult:
        request = task.context.get("design_request", task.description)
        framework = task.context.get("framework", "React")

        prompt = (
            f"Design a UI/UX solution for:\n{request}\n\n"
            f"Framework: {framework}\n\n"
            "Respond as JSON:\n"
            '{"design": {"summary": "...", "layout": "..."}, '
            '"components": [{"name": "...", "props": [...], "children": [...], "accessibility": "..."}], '
            '"design_tokens": {"colors": {...}, "typography": {...}, "spacing": {...}}, '
            '"ux_flows": [{"flow": "...", "steps": [...]}], '
            '"responsive_breakpoints": {"mobile": 375, "tablet": 768, "desktop": 1280}}'
        )
        result = self._infer_json(prompt, fallback={"design": {"summary": request}, "components": []})
        return AgentResult(success=True, output=result, metadata={"mode": "ui_design", "framework": framework})

    # ------------------------------------------------------------------
    # Database (from DataEngineerAgent)
    # ------------------------------------------------------------------

    def _execute_database(self, task: AgentTask) -> AgentResult:
        request = task.context.get("design_request", task.description)
        db_type = task.context.get("database", "PostgreSQL")

        prompt = (
            f"Design a database schema for:\n{request}\n\n"
            f"Database: {db_type}\n\n"
            "Respond as JSON:\n"
            '{"schema": {"tables": [{"name": "...", "columns": [{"name": "...", "type": "...", '
            '"constraints": [...]}], "indexes": [...], "foreign_keys": [...]}]}, '
            '"migrations": [{"version": "001", "description": "...", "sql": "..."}], '
            '"pipeline": {"type": "batch|streaming", "stages": [...]}, '
            '"validation_rules": [...]}'
        )
        result = self._infer_json(prompt, fallback={"schema": {"tables": []}, "migrations": []})
        return AgentResult(success=True, output=result, metadata={"mode": "database", "db_type": db_type})

    # ------------------------------------------------------------------
    # DevOps (from DevOpsAgent)
    # ------------------------------------------------------------------

    def _execute_devops(self, task: AgentTask) -> AgentResult:
        request = task.context.get("design_request", task.description)
        platform = task.context.get("platform", "generic")

        prompt = (
            f"Design DevOps infrastructure for:\n{request}\n\n"
            f"Platform: {platform}\n\n"
            "Respond as JSON:\n"
            '{"pipeline": {"stages": [{"name": "...", "steps": [...], "gates": [...]}]}, '
            '"containerization": {"dockerfile": "...", "compose": "..."}, '
            '"infrastructure": {"provider": "...", "resources": [...]}, '
            '"deployment_strategy": {"type": "blue-green|canary|rolling", "config": {...}}, '
            '"monitoring": {"health_checks": [...], "alerts": [...]}}'
        )
        result = self._infer_json(prompt, fallback={"pipeline": {"stages": []}, "containerization": {}})
        return AgentResult(success=True, output=result, metadata={"mode": "devops", "platform": platform})

    # ------------------------------------------------------------------
    # Git Workflow (from VersionControlAgent)
    # ------------------------------------------------------------------

    def _execute_git_workflow(self, task: AgentTask) -> AgentResult:
        request = task.context.get("design_request", task.description)
        strategy = task.context.get("strategy", "gitflow")

        prompt = (
            f"Design git workflow for:\n{request}\n\n"
            f"Strategy: {strategy}\n\n"
            "Respond as JSON:\n"
            '{"workflow": {"strategy": "...", "branches": [{"name": "...", "purpose": "...", "naming": "..."}]}, '
            '"commit_convention": {"type": "conventional", "scopes": [...], "examples": [...]}, '
            '"pr_template": "...", "changelog_format": "...", '
            '"release_process": {"steps": [...], "versioning": "semver"}}'
        )
        result = self._infer_json(prompt, fallback={"workflow": {"strategy": strategy}, "commit_convention": {}})
        return AgentResult(success=True, output=result, metadata={"mode": "git_workflow", "strategy": strategy})

    def get_capabilities(self) -> List[str]:
        return [
            "ui_design", "wireframing", "design_tokens", "accessibility",
            "database_schema", "migration_design", "etl_pipeline",
            "cicd_pipeline", "containerization", "infrastructure_as_code",
            "branch_strategy", "commit_conventions", "release_management",
        ]


# Singleton
_architect_agent: Optional[ArchitectAgent] = None


def get_architect_agent(config: Optional[Dict[str, Any]] = None) -> ArchitectAgent:
    global _architect_agent
    if _architect_agent is None:
        _architect_agent = ArchitectAgent(config)
    return _architect_agent
