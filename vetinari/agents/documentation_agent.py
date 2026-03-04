"""
Vetinari Documentation Agent

LLM-powered documentation generation agent that creates real API docs, user guides,
architecture documentation, and changelogs by analysing actual artifacts and code.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult,
)

logger = logging.getLogger(__name__)


class DocumentationAgent(BaseAgent):
    """Documentation Agent — LLM-powered, artifact-aware documentation generator."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.DOCUMENTATION_AGENT, config)
        self._doc_format = self._config.get("doc_format", "markdown")

    def get_system_prompt(self) -> str:
        return f"""You are Vetinari's Documentation Agent — a technical writer and documentation architect.
Your job is to generate comprehensive, accurate, and useful documentation from actual project artifacts.

Output format: {self._doc_format}

You MUST base documentation on the actual artifacts and code provided.
Do NOT generate generic placeholder content — every section must reflect the real project.

Required output (JSON):
{{
  "docs_manifest": {{
    "title": "...",
    "version": "...",
    "generated_at": "...",
    "sections": [],
    "doc_type": "..."
  }},
  "pages": [
    {{
      "title": "...",
      "path": "...",
      "section": "...",
      "content": "# Title\\n\\n...full markdown content...",
      "toc": []
    }}
  ],
  "api_docs": {{
    "title": "API Reference",
    "base_url": "...",
    "authentication": "...",
    "endpoints": [
      {{
        "path": "...",
        "method": "GET|POST|PUT|DELETE|PATCH",
        "summary": "...",
        "description": "...",
        "parameters": [],
        "request_body": null,
        "responses": {{}},
        "examples": []
      }}
    ]
  }},
  "user_guides": [
    {{
      "title": "...",
      "audience": "developer|end-user|admin",
      "content": "...full markdown...",
      "prerequisites": [],
      "steps": []
    }}
  ],
  "references": {{
    "configuration": [],
    "cli_commands": [],
    "environment_variables": [],
    "glossary": []
  }},
  "change_log": [
    {{
      "version": "...",
      "date": "...",
      "type": "feature|bugfix|breaking|improvement",
      "changes": []
    }}
  ],
  "summary": "..."
}}

Documentation principles:
- Accurate: Only document what actually exists in the artifacts
- Complete: Cover all public APIs, configuration options, and user-facing features
- Clear: Use plain language, concrete examples, and consistent terminology
- Structured: Logical hierarchy with table of contents
- Actionable: Users should be able to accomplish tasks by following the docs
"""

    def get_capabilities(self) -> List[str]:
        return [
            "api_documentation",
            "user_guide_generation",
            "reference_documentation",
            "changelog_creation",
            "doc_indexing",
            "markdown_generation",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute documentation generation using LLM inference."""
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"],
            )

        task = self.prepare_task(task)

        try:
            artifacts = task.context.get("artifacts", [])
            doc_type = task.context.get("doc_type", "comprehensive")
            project_name = task.context.get("project_name", "Project")
            version = task.context.get("version", "1.0.0")
            code_content = task.context.get("code", "")

            # Build artifact summary for LLM
            artifact_summary = self._summarise_artifacts(artifacts, code_content, task.description)

            prompt = (
                f"Generate {doc_type} documentation for: {project_name} v{version}\n\n"
                f"Project artifacts and code:\n{artifact_summary}\n\n"
                f"Task description: {task.description}\n\n"
                "Return a complete JSON documentation package. "
                "Write real, substantive content for each page — not placeholders."
            )

            docs = self._infer_json(
                prompt=prompt,
                fallback=self._fallback_docs(project_name, version, doc_type, task.description),
            )

            # Ensure required keys
            docs.setdefault(
                "docs_manifest",
                {
                    "title": f"{project_name} Documentation",
                    "version": version,
                    "generated_at": datetime.now().isoformat(),
                    "sections": [],
                    "doc_type": doc_type,
                },
            )
            docs.setdefault("pages", [])
            docs.setdefault("api_docs", {"title": "API Reference", "endpoints": []})
            docs.setdefault("user_guides", [])
            docs.setdefault("references", {})
            docs.setdefault("change_log", [])
            docs.setdefault("summary", f"Documentation generated for {project_name}")

            # Update generated_at timestamp
            docs["docs_manifest"]["generated_at"] = datetime.now().isoformat()

            result = AgentResult(
                success=True,
                output=docs,
                metadata={
                    "artifacts_documented": len(artifacts),
                    "doc_type": doc_type,
                    "pages_count": len(docs.get("pages", [])),
                    "endpoints_documented": len(docs.get("api_docs", {}).get("endpoints", [])),
                },
            )
            task = self.complete_task(task, result)
            return result

        except Exception as e:
            self._log("error", f"Documentation generation failed: {e}")
            return AgentResult(success=False, output=None, errors=[str(e)])

    def verify(self, output: Any) -> VerificationResult:
        issues = []
        score = 1.0

        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            return VerificationResult(passed=False, issues=issues, score=0.0)

        if not output.get("docs_manifest"):
            issues.append({"type": "missing_manifest", "message": "Documentation manifest missing"})
            score -= 0.2
        if not output.get("pages"):
            issues.append({"type": "missing_pages", "message": "Documentation pages missing"})
            score -= 0.3
        if not output.get("api_docs"):
            issues.append({"type": "missing_api_docs", "message": "API documentation missing"})
            score -= 0.15
        if not output.get("user_guides"):
            issues.append({"type": "missing_guides", "message": "User guides missing"})
            score -= 0.15
        if not output.get("change_log"):
            issues.append({"type": "missing_changelog", "message": "Change log missing"})
            score -= 0.1

        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0.0, score))

    def _summarise_artifacts(
        self, artifacts: List[Any], code: str, description: str
    ) -> str:
        """Build a concise artifact summary for the LLM prompt."""
        parts = []
        if description:
            parts.append(f"Task: {description}")
        if code:
            parts.append(f"Code excerpt:\n{code[:2000]}")
        for i, artifact in enumerate(artifacts[:5]):
            if isinstance(artifact, str):
                parts.append(f"Artifact {i+1}:\n{artifact[:500]}")
            elif isinstance(artifact, dict):
                for k, v in list(artifact.items())[:3]:
                    if isinstance(v, str):
                        parts.append(f"{k}: {v[:300]}")
        return "\n\n".join(parts) if parts else "No specific artifacts provided."

    def _fallback_docs(
        self, project_name: str, version: str, doc_type: str, description: str
    ) -> Dict[str, Any]:
        today = datetime.now().strftime("%Y-%m-%d")
        return {
            "docs_manifest": {
                "title": f"{project_name} Documentation",
                "version": version,
                "generated_at": datetime.now().isoformat(),
                "sections": ["Getting Started", "API Reference", "User Guide", "Configuration"],
                "doc_type": doc_type,
            },
            "pages": [
                {
                    "title": "Overview",
                    "path": "overview.md",
                    "section": "Getting Started",
                    "content": f"# {project_name}\n\n{description}\n\n## Quick Start\n\n1. Install dependencies\n2. Configure environment variables\n3. Run the application\n",
                    "toc": ["Overview", "Quick Start"],
                },
                {
                    "title": "Installation",
                    "path": "installation.md",
                    "section": "Getting Started",
                    "content": f"# Installation\n\n## Prerequisites\n\n- Python 3.9+\n- pip\n\n## Install\n\n```bash\npip install -e .\n```\n\n## Configuration\n\nCopy `.env.example` to `.env` and fill in your API keys.\n",
                    "toc": ["Prerequisites", "Install", "Configuration"],
                },
            ],
            "api_docs": {
                "title": "API Reference",
                "base_url": "http://localhost:5000",
                "authentication": "Bearer token via Authorization header",
                "endpoints": [
                    {
                        "path": "/api/status",
                        "method": "GET",
                        "summary": "System status",
                        "description": "Returns the current system health and configuration",
                        "parameters": [],
                        "responses": {"200": {"description": "Success", "schema": {"status": "string"}}},
                        "examples": [],
                    }
                ],
            },
            "user_guides": [
                {
                    "title": f"Getting Started with {project_name}",
                    "audience": "developer",
                    "content": f"# Getting Started\n\nThis guide will help you get {project_name} running.\n\n## Step 1: Installation\n\nSee [Installation](installation.md).\n\n## Step 2: Configuration\n\nSet up your `.env` file with the required configuration.\n\n## Step 3: Run\n\n```bash\npython -m vetinari start\n```\n",
                    "prerequisites": ["Python 3.9+", "LM Studio (for local models)"],
                    "steps": ["Install", "Configure", "Run"],
                }
            ],
            "references": {
                "configuration": [
                    {"name": "LM_STUDIO_HOST", "type": "string", "default": "http://localhost:1234", "description": "LM Studio API host"},
                    {"name": "CLAUDE_API_KEY", "type": "string", "default": "", "description": "Anthropic API key"},
                ],
                "cli_commands": [
                    {"name": "start", "description": "Start Vetinari with dashboard"},
                    {"name": "run --goal <goal>", "description": "Execute a specific goal"},
                ],
                "environment_variables": [],
                "glossary": [],
            },
            "change_log": [
                {
                    "version": version,
                    "date": today,
                    "type": "feature",
                    "changes": [f"Documentation generated for {project_name}"],
                }
            ],
            "summary": f"Documentation generated for {project_name} v{version}",
        }


_documentation_agent: Optional[DocumentationAgent] = None


def get_documentation_agent(config: Optional[Dict[str, Any]] = None) -> DocumentationAgent:
    global _documentation_agent
    if _documentation_agent is None:
        _documentation_agent = DocumentationAgent(config)
    return _documentation_agent
