"""
VersionControlAgent - Git operations, branch strategy, PR creation, code review coordination.

Provides intelligent version control guidance including:
- Branch strategy recommendations
- Commit message generation and changelog writing
- PR/MR description creation
- Merge conflict analysis
- Code review coordination
- Git workflow best practices
"""

from __future__ import annotations

import logging
import subprocess
import os
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentTask, AgentResult, AgentType, VerificationResult,
)

logger = logging.getLogger(__name__)


class VersionControlAgent(BaseAgent):
    """Agent for version control operations, branch strategy, and code review coordination."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.VERSION_CONTROL, config)
        self._safe_git_timeout = int(
            (config or {}).get("git_timeout", os.environ.get("VETINARI_GIT_TIMEOUT", "10"))
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        return (
            "You are a version control expert specialising in Git workflows, branch strategies, "
            "and collaborative development practices. You provide:\n"
            "- Branch naming conventions and strategies (GitFlow, trunk-based, etc.)\n"
            "- Commit message guidelines following Conventional Commits\n"
            "- Pull/Merge request templates and descriptions\n"
            "- Merge conflict analysis and resolution strategies\n"
            "- Code review checklists and coordination\n"
            "- Changelog generation from commit history\n"
            "- Release tagging and versioning guidance\n\n"
            "Always respond with structured JSON as specified in the task context."
        )

    def get_capabilities(self) -> List[str]:
        return [
            "branch_strategy",
            "commit_message_generation",
            "pr_description_creation",
            "merge_conflict_analysis",
            "code_review_coordination",
            "changelog_generation",
            "release_tagging",
            "git_workflow_guidance",
            "versioning_strategy",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        if not self.validate_task(task):
            return AgentResult(success=False, output=None, errors=[f"Invalid task for {self._agent_type.value}"])
        self.prepare_task(task)
        try:
            result = self._perform_vc_operation(task)
            agent_result = AgentResult(
                success=True,
                output=result,
                metadata={
                    "task_id": task.task_id,
                    "agent_type": self._agent_type.value,
                    "operation": task.context.get("operation", "general"),
                },
            )
            self.complete_task(task, agent_result)
            return agent_result
        except Exception as exc:
            logger.error(f"[VersionControlAgent] execute() failed: {exc}")
            return AgentResult(
                success=False,
                output={},
                metadata={"task_id": task.task_id, "agent_type": self._agent_type.value},
                errors=[str(exc)],
            )

    def verify(self, output: Any) -> VerificationResult:
        issues: list = []
        score = 1.0
        if not isinstance(output, dict):
            return VerificationResult(
                passed=False, score=0.0,
                issues=[{"severity": "error", "message": "Output must be a dict"}],
            )
        if not output.get("recommendations") and not output.get("branch_strategy") \
                and not output.get("commit_messages") and not output.get("pr_description") \
                and not output.get("changelog"):
            issues.append({"severity": "warning", "message": "No actionable VC output produced"})
            score -= 0.5
        passed = score >= 0.5 and not issues
        return VerificationResult(passed=passed, score=score, issues=issues)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _perform_vc_operation(self, task: AgentTask) -> Dict[str, Any]:
        """Determine the operation type and execute it."""
        ctx = task.context or {}
        operation = ctx.get("operation", "").lower()
        description = task.description or ""

        # Gather git context from the repository if available
        git_context = self._gather_git_context(ctx.get("repo_path", "."))

        # Determine search topics
        search_context = ""
        try:
            results = self._search(f"git {operation or 'workflow'} best practices 2026", max_results=3)
            if results:
                search_context = "\n".join(
                    f"- [{r['title']}]: {r['snippet']}" for r in results[:3]
                )
        except Exception:
            pass

        prompt = f"""You are a version control expert. Analyse the following request and provide structured guidance.

## Task
{description}

## Operation
{operation or 'General version control guidance'}

## Context
{ctx}

## Git Repository State
{git_context}

## Current Best Practices
{search_context or 'Use standard Git conventions.'}

## Required Output (JSON)
{{
  "operation": "{operation or 'general'}",
  "branch_strategy": {{
    "recommended_branches": [...],
    "naming_convention": "...",
    "workflow": "gitflow|trunk-based|github-flow"
  }},
  "commit_messages": ["feat: ...", "fix: ...", ...],
  "pr_description": {{
    "title": "...",
    "summary": "...",
    "changes": [...],
    "testing": "...",
    "checklist": [...]
  }},
  "changelog": "...",
  "merge_strategy": "...",
  "recommendations": ["...", ...],
  "conflicts": [],
  "risks": ["...", ...]
}}

Return ONLY valid JSON."""

        result = self._infer_json(prompt, fallback=self._fallback_vc_guidance(task))
        if result and isinstance(result, dict):
            return result
        return self._fallback_vc_guidance(task)

    def _gather_git_context(self, repo_path: str) -> str:
        """Safely gather git repository state."""
        context_parts = []
        try:
            safe_path = os.path.abspath(repo_path) if repo_path else "."
            if not os.path.exists(os.path.join(safe_path, ".git")):
                return "No git repository detected."

            def _run(args: List[str]) -> str:
                try:
                    r = subprocess.run(
                        ["git"] + args,
                        cwd=safe_path,
                        capture_output=True,
                        text=True,
                        timeout=self._safe_git_timeout,
                    )
                    return r.stdout.strip() if r.returncode == 0 else ""
                except Exception:
                    return ""

            branch = _run(["branch", "--show-current"])
            if branch:
                context_parts.append(f"Current branch: {branch}")

            recent_log = _run(["log", "--oneline", "-10"])
            if recent_log:
                context_parts.append(f"Recent commits:\n{recent_log}")

            status = _run(["status", "--short"])
            if status:
                context_parts.append(f"Working tree status:\n{status}")

            remotes = _run(["remote", "-v"])
            if remotes:
                context_parts.append(f"Remotes:\n{remotes}")

        except Exception as exc:
            context_parts.append(f"Git context unavailable: {exc}")

        return "\n\n".join(context_parts) if context_parts else "Git context not available."

    def _fallback_vc_guidance(self, task: AgentTask) -> Dict[str, Any]:
        """Rich fallback when LLM is unavailable."""
        ctx = task.context or {}
        operation = ctx.get("operation", "general")
        return {
            "operation": operation,
            "branch_strategy": {
                "recommended_branches": ["main", "develop", "feature/*", "bugfix/*", "release/*"],
                "naming_convention": "kebab-case: feature/TICKET-description",
                "workflow": "gitflow",
            },
            "commit_messages": [
                "feat(scope): add new feature",
                "fix(scope): resolve bug",
                "docs: update documentation",
                "test: add unit tests",
                "refactor: improve code structure",
            ],
            "pr_description": {
                "title": f"feat: {task.description[:60] if task.description else 'changes'}",
                "summary": "Summary of changes made in this PR.",
                "changes": ["Change 1", "Change 2"],
                "testing": "Unit tests added/updated. Manual testing performed.",
                "checklist": [
                    "Code reviewed",
                    "Tests pass",
                    "Documentation updated",
                    "No breaking changes",
                ],
            },
            "changelog": "## Unreleased\n### Added\n- New functionality\n### Fixed\n- Bug fixes",
            "merge_strategy": "squash-and-merge for feature branches, merge commit for releases",
            "recommendations": [
                "Use Conventional Commits for consistent commit messages",
                "Require PR reviews before merging to main",
                "Enable branch protection rules on main/develop",
                "Run CI/CD pipeline on all PRs",
                "Tag releases with semantic versioning (v1.2.3)",
            ],
            "conflicts": [],
            "risks": ["Merge conflicts may arise if long-lived branches diverge significantly"],
        }


# Singleton
_version_control_agent: Optional[VersionControlAgent] = None


def get_version_control_agent(
    config: Optional[Dict[str, Any]] = None
) -> VersionControlAgent:
    """Get the singleton VersionControlAgent instance."""
    global _version_control_agent
    if _version_control_agent is None:
        _version_control_agent = VersionControlAgent(config)
    return _version_control_agent
