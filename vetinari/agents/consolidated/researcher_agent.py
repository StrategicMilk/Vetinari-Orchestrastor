"""
Consolidated Researcher Agent (v0.4.0)
========================================
Replaces: EXPLORER + RESEARCHER + LIBRARIAN + UI_PLANNER + DATA_ENGINEER +
          DEVOPS + VERSION_CONTROL

Absorbs: ARCHITECT (ui_design, database, devops, git_workflow)

Modes:
- code_discovery: Fast code/document/pattern extraction (from Explorer)
- domain_research: Feasibility analysis, competitive analysis (from Researcher)
- api_lookup: API/docs lookup, library discovery (from Librarian)
- lateral_thinking: Creative problem-solving and alternative approaches
- ui_design: UI/UX design, wireframes, accessibility (from Architect/UI_PLANNER)
- database: Schema design, migrations, ETL pipelines (from Architect/DATA_ENGINEER)
- devops: CI/CD, containers, IaC, deployment (from Architect/DEVOPS)
- git_workflow: Branch strategy, commit conventions, PRs (from Architect/VERSION_CONTROL)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from vetinari.agents.multi_mode_agent import MultiModeAgent
from vetinari.agents.contracts import AgentResult, AgentTask, AgentType, VerificationResult

logger = logging.getLogger(__name__)


class ConsolidatedResearcherAgent(MultiModeAgent):
    """Unified research agent for code discovery, domain research, API lookup,
    and architecture design (UI, database, DevOps, git workflow)."""

    MODES = {
        "code_discovery": "_execute_code_discovery",
        "domain_research": "_execute_domain_research",
        "api_lookup": "_execute_api_lookup",
        "lateral_thinking": "_execute_lateral_thinking",
        "ui_design": "_execute_ui_design",
        "database": "_execute_database",
        "devops": "_execute_devops",
        "git_workflow": "_execute_git_workflow",
    }
    DEFAULT_MODE = "code_discovery"
    MODE_KEYWORDS = {
        "code_discovery": ["code", "file", "class", "function", "pattern", "codebase",
                           "discover", "explore", "search code"],
        "domain_research": ["research", "feasib", "competit", "market", "domain", "analys"],
        "api_lookup": ["api", "library", "framework", "package", "documentation", "docs",
                       "license", "dependency"],
        "lateral_thinking": ["lateral", "creative", "alternative", "novel", "brainstorm",
                             "unconventional"],
        "ui_design": ["ui", "ux", "frontend", "component", "wireframe", "layout",
                      "design token", "accessibility", "wcag", "responsive", "css",
                      "react", "interface"],
        "database": ["database", "schema", "table", "migration", "etl", "pipeline", "sql",
                     "data model", "foreign key", "index", "query", "orm"],
        "devops": ["ci/cd", "docker", "kubernetes", "terraform", "ansible", "deploy",
                   "container", "pipeline", "helm", "monitoring", "infrastructure"],
        "git_workflow": ["git", "branch", "commit", "merge", "pull request", "pr",
                         "release", "changelog", "tag", "rebase", "version"],
    }
    LEGACY_TYPE_TO_MODE = {
        "EXPLORER": "code_discovery",
        "RESEARCHER": "domain_research",
        "LIBRARIAN": "api_lookup",
        "UI_PLANNER": "ui_design",
        "DATA_ENGINEER": "database",
        "DEVOPS": "devops",
        "VERSION_CONTROL": "git_workflow",
        "ARCHITECT": "ui_design",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.CONSOLIDATED_RESEARCHER, config)

    def _get_base_system_prompt(self) -> str:
        return (
            "You are Vetinari's Research & Architecture Agent. You handle code discovery, "
            "domain research, API/library lookup, creative problem-solving, and architecture "
            "design (UI/UX, database schemas, DevOps pipelines, git workflows)."
        )

    def _get_mode_system_prompt(self, mode: str) -> str:
        prompts = {
            "code_discovery": (
                "You are Vetinari's Code Explorer. Your role is to:\n"
                "- Search codebases for relevant files, classes, functions, and patterns\n"
                "- Map project structure and dependencies\n"
                "- Extract reusable patterns and identify code smells\n"
                "- Discover entry points, test files, and configuration\n\n"
                "Be thorough but focused. Return structured findings."
            ),
            "domain_research": (
                "You are Vetinari's Domain Researcher. Your role is to:\n"
                "- Conduct feasibility analysis for technical approaches\n"
                "- Research domain-specific best practices\n"
                "- Analyze competing solutions and their tradeoffs\n"
                "- Assess technical risks and opportunities\n\n"
                "Provide evidence-based recommendations with confidence scores."
            ),
            "api_lookup": (
                "You are Vetinari's API & Library Specialist. Your role is to:\n"
                "- Research APIs, libraries, and frameworks\n"
                "- Evaluate library fitness for specific use cases\n"
                "- Check compatibility, licensing, and maintenance status\n"
                "- Provide integration examples and best practices\n\n"
                "Include version info, license type, and fit score (0-1)."
            ),
            "lateral_thinking": (
                "You are Vetinari's Lateral Thinking Specialist. Your role is to:\n"
                "- Generate unconventional approaches to problems\n"
                "- Challenge assumptions and propose alternatives\n"
                "- Draw inspiration from other domains\n"
                "- Evaluate creative solutions for feasibility\n\n"
                "Think outside the box while remaining practical."
            ),
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
            has_findings = bool(
                output.get("findings") or output.get("results")
                or output.get("recommendations") or output.get("design")
                or output.get("schema") or output.get("pipeline")
                or output.get("components") or output.get("workflow")
            )
            return VerificationResult(passed=has_findings, score=0.8 if has_findings else 0.3)
        return VerificationResult(passed=True, score=0.6)

    # ------------------------------------------------------------------
    # Code Discovery (from ExplorerAgent)
    # ------------------------------------------------------------------

    def _execute_code_discovery(self, task: AgentTask) -> AgentResult:
        query = task.context.get("query", task.description)
        scope = task.context.get("scope", "code")

        prompt = (
            f"Discover and analyze code/patterns related to:\n{query}\n\n"
            f"Scope: {scope}\n\n"
            "Respond as JSON:\n"
            '{"findings": [{"file": "...", "type": "class|function|pattern", '
            '"name": "...", "description": "...", "relevance": 0.9}], '
            '"patterns": [...], "recommendations": [...]}'
        )
        result = self._infer_json(prompt, fallback={"findings": [], "patterns": []})
        return AgentResult(
            success=True, output=result or {"findings": []},
            metadata={"mode": "code_discovery", "scope": scope},
        )

    # ------------------------------------------------------------------
    # Domain Research (from ResearcherAgent)
    # ------------------------------------------------------------------

    def _execute_domain_research(self, task: AgentTask) -> AgentResult:
        query = task.context.get("query", task.description)
        scope = task.context.get("scope", "general")

        search_results = self._search(query, max_results=5)
        search_context = ""
        if search_results:
            search_context = "\n\nWeb search results:\n" + "\n".join(
                f"- {r['title']}: {r['snippet']}" for r in search_results[:3]
            )

        prompt = (
            f"Research the following topic:\n{query}\n\n"
            f"Scope: {scope}\n{search_context}\n\n"
            "Respond as JSON:\n"
            '{"findings": [{"topic": "...", "summary": "...", "confidence": 0.9}], '
            '"recommendations": [...], "sources": [...], '
            '"feasibility_score": 0.8}'
        )
        result = self._infer_json(prompt, fallback={"findings": [], "recommendations": []})
        return AgentResult(
            success=True, output=result or {"findings": []},
            metadata={"mode": "domain_research", "search_results": len(search_results)},
        )

    # ------------------------------------------------------------------
    # API Lookup (from LibrarianAgent)
    # ------------------------------------------------------------------

    def _execute_api_lookup(self, task: AgentTask) -> AgentResult:
        query = task.context.get("query", task.description)

        search_results = self._search(f"{query} API documentation library", max_results=5)
        search_context = ""
        if search_results:
            search_context = "\n\nSearch results:\n" + "\n".join(
                f"- {r['title']}: {r['snippet']}" for r in search_results[:3]
            )

        prompt = (
            f"Research APIs, libraries, and frameworks for:\n{query}\n"
            f"{search_context}\n\n"
            "Respond as JSON:\n"
            '{"findings": [{"name": "...", "type": "library|api|framework", '
            '"version": "...", "license": "...", "fit_score": 0.8, '
            '"pros": [...], "cons": [...], "integration_notes": "..."}], '
            '"recommendations": [...]}'
        )
        result = self._infer_json(prompt, fallback={"findings": [], "recommendations": []})
        return AgentResult(
            success=True, output=result or {"findings": []},
            metadata={"mode": "api_lookup"},
        )

    # ------------------------------------------------------------------
    # Lateral Thinking
    # ------------------------------------------------------------------

    def _execute_lateral_thinking(self, task: AgentTask) -> AgentResult:
        problem = task.context.get("problem", task.description)

        prompt = (
            f"Apply lateral thinking to this problem:\n{problem}\n\n"
            "Generate at least 3 unconventional approaches. For each:\n"
            "- Describe the approach\n"
            "- Explain why it might work\n"
            "- Rate feasibility (0-1)\n\n"
            "Respond as JSON:\n"
            '{"approaches": [{"description": "...", "rationale": "...", '
            '"feasibility": 0.7, "inspiration": "..."}], '
            '"recommendations": [...]}'
        )
        result = self._infer_json(prompt, fallback={"approaches": [], "recommendations": []})
        return AgentResult(
            success=True, output=result or {"approaches": []},
            metadata={"mode": "lateral_thinking"},
        )

    # ------------------------------------------------------------------
    # UI Design (absorbed from ArchitectAgent)
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
        return AgentResult(
            success=True, output=result,
            metadata={"mode": "ui_design", "framework": framework},
        )

    # ------------------------------------------------------------------
    # Database (absorbed from ArchitectAgent)
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
        return AgentResult(
            success=True, output=result,
            metadata={"mode": "database", "db_type": db_type},
        )

    # ------------------------------------------------------------------
    # DevOps (absorbed from ArchitectAgent)
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
        return AgentResult(
            success=True, output=result,
            metadata={"mode": "devops", "platform": platform},
        )

    # ------------------------------------------------------------------
    # Git Workflow (absorbed from ArchitectAgent)
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
        return AgentResult(
            success=True, output=result,
            metadata={"mode": "git_workflow", "strategy": strategy},
        )

    def get_capabilities(self) -> List[str]:
        return [
            "code_discovery", "pattern_extraction", "project_mapping",
            "domain_research", "feasibility_analysis", "competitive_analysis",
            "api_lookup", "library_evaluation", "license_assessment",
            "lateral_thinking", "creative_problem_solving",
            "ui_design", "wireframing", "design_tokens", "accessibility",
            "database_schema", "migration_design", "etl_pipeline",
            "cicd_pipeline", "containerization", "infrastructure_as_code",
            "branch_strategy", "commit_conventions", "release_management",
        ]


# Singleton
_consolidated_researcher_agent: Optional[ConsolidatedResearcherAgent] = None


def get_consolidated_researcher_agent(config: Optional[Dict[str, Any]] = None) -> ConsolidatedResearcherAgent:
    global _consolidated_researcher_agent
    if _consolidated_researcher_agent is None:
        _consolidated_researcher_agent = ConsolidatedResearcherAgent(config)
    return _consolidated_researcher_agent
