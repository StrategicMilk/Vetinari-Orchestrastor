"""Vetinari Documenter Agent — consolidated from Documentation Agent + Version Control."""
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import AgentResult, AgentTask, AgentType, VerificationResult

import logging

logger = logging.getLogger(__name__)


class DocumenterAgent(BaseAgent):
    """Documenter agent — documentation generation and version control operations.

    Absorbs:
        - DocumentationAgent: auto-generated docs, API docs, user guides
        - VersionControlAgent: git operations, branch strategy, PR creation, code review coordination
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.DOCUMENTER, config)

    def get_system_prompt(self) -> str:
        return """You are Vetinari's Documenter. You handle both documentation generation and
version control operations to keep projects well-documented and properly managed.

Your responsibilities:
1. Generate API docs, README files, user guides, and inline code documentation
2. Execute git operations: branching, committing, merging, PR creation
3. Coordinate code review workflows and branch strategies
4. Maintain changelogs and release notes
5. Ensure documentation stays in sync with code changes

Output must include structured documentation or git operation results depending on task type."""

    def get_capabilities(self) -> List[str]:
        return [
            "api_documentation",
            "readme_generation",
            "user_guide_authoring",
            "git_operations",
            "branch_strategy",
            "pr_creation",
            "changelog_maintenance",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute task, delegating to documentation or version control sub-agent."""
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"],
            )

        task = self.prepare_task(task)
        desc = (task.description or "").lower()

        try:
            if any(kw in desc for kw in ("git", "commit", "branch", "merge", "pr", "pull request", "version control", "tag", "release", "changelog")):
                result = self._delegate_to_version_control(task)
            elif any(kw in desc for kw in ("doc", "readme", "api doc", "user guide", "docstring", "comment", "annotate", "document")):
                result = self._delegate_to_documentation(task)
            else:
                result = self._execute_default(task)

            self.complete_task(task, result)
            return result

        except Exception as e:
            self._log("error", f"DocumenterAgent execution failed: {e}")
            return AgentResult(success=False, output=None, errors=[str(e)])

    def _delegate_to_documentation(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.documentation_agent import DocumentationAgent
        agent = DocumentationAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _delegate_to_version_control(self, task: AgentTask) -> AgentResult:
        from vetinari.agents.version_control_agent import VersionControlAgent
        agent = VersionControlAgent(self._config)
        agent._adapter_manager = self._adapter_manager
        agent._web_search = self._web_search
        agent._initialized = self._initialized
        return agent.execute(task)

    def _execute_default(self, task: AgentTask) -> AgentResult:
        goal = task.prompt or task.description
        prompt = f"""Generate documentation and version control recommendations for:\n\n{goal}\n\nReturn JSON with documentation and git_operations fields."""
        output = self._infer_json(prompt)
        if output is None:
            output = {
                "documentation": f"# Documentation\n\nGenerated for: {goal}",
                "git_operations": {"recommended_branch": "feature/new-work", "commit_message": f"feat: {goal[:60]}"},
            }
        return AgentResult(success=True, output=output, metadata={"mode": "default"})

    def verify(self, output: Any) -> VerificationResult:
        issues = []
        score = 1.0
        if not isinstance(output, dict):
            return VerificationResult(passed=False, issues=[{"type": "invalid_type", "message": "Output must be a dict"}], score=0.5)
        has_content = any(output.get(k) for k in ("documentation", "git_operations", "content", "sections", "commands"))
        if not has_content:
            issues.append({"type": "no_content", "message": "No documentation or git operations found"})
            score -= 0.5
        return VerificationResult(passed=score >= 0.5, issues=issues, score=max(0.0, score))


def get_documenter_agent(config: Optional[Dict[str, Any]] = None) -> DocumenterAgent:
    return DocumenterAgent(config)
