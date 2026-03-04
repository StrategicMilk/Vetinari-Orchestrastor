"""
Vetinari Documentation Agent

The Documentation Agent is responsible for automatic documentation generation
and maintenance.
"""

from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult
)


class DocumentationAgent(BaseAgent):
    """Documentation Agent - auto-generated docs, API docs, user guides."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.DOCUMENTATION_AGENT, config)
        
    def get_system_prompt(self) -> str:
        return """You are Vetinari's Documentation Agent. Generate, update, and maintain API docs, 
user docs, and internal references. Produce a doc skeleton, auto-generated references, and change logs.

You must:
1. Generate API documentation
2. Create user guides and tutorials
3. Write configuration references
4. Document architecture decisions
5. Maintain change logs
6. Generate index and navigation

Output format must include docs_manifest, api_docs, user_guides, references, and change_log."""
    
    def get_capabilities(self) -> List[str]:
        return [
            "api_documentation",
            "user_guide_generation",
            "reference_documentation",
            "changelog_creation",
            "doc_indexing",
            "markdown_generation"
        ]
    
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the documentation generation task.
        
        Args:
            task: The task containing artifacts to document
            
        Returns:
            AgentResult containing the documentation
        """
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )
        
        task = self.prepare_task(task)
        
        try:
            artifacts = task.context.get("artifacts", [])
            doc_type = task.context.get("doc_type", "comprehensive")
            
            # Generate documentation (simulated - in production would use actual doc generation)
            docs = self._generate_documentation(artifacts, doc_type)
            
            return AgentResult(
                success=True,
                output=docs,
                metadata={
                    "artifacts_documented": len(artifacts),
                    "doc_type": doc_type,
                    "pages_count": len(docs.get("pages", []))
                }
            )
            
        except Exception as e:
            self._log("error", f"Documentation generation failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)]
            )
    
    def verify(self, output: Any) -> VerificationResult:
        """Verify the documentation output meets quality standards.
        
        Args:
            output: The documentation to verify
            
        Returns:
            VerificationResult with pass/fail status
        """
        issues = []
        score = 1.0
        
        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            score -= 0.5
            return VerificationResult(passed=False, issues=issues, score=score)
        
        # Check for docs manifest
        if not output.get("docs_manifest"):
            issues.append({"type": "missing_manifest", "message": "Documentation manifest missing"})
            score -= 0.2
        
        # Check for pages
        if not output.get("pages"):
            issues.append({"type": "missing_pages", "message": "Documentation pages missing"})
            score -= 0.3
        
        # Check for API docs
        if not output.get("api_docs"):
            issues.append({"type": "missing_api_docs", "message": "API documentation missing"})
            score -= 0.15
        
        # Check for user guides
        if not output.get("user_guides"):
            issues.append({"type": "missing_guides", "message": "User guides missing"})
            score -= 0.15
        
        # Check for changelog
        if not output.get("change_log"):
            issues.append({"type": "missing_changelog", "message": "Change log missing"})
            score -= 0.1
        
        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0, score))
    
    def _generate_documentation(self, artifacts: List[str], doc_type: str) -> Dict[str, Any]:
        """Generate documentation from artifacts.
        
        Args:
            artifacts: List of artifacts to document
            doc_type: Type of documentation (comprehensive, api, user, internal)
            
        Returns:
            Dictionary containing documentation
        """
        # This is a simplified implementation
        # In production, this would use actual doc generation tools
        
        docs_manifest = {
            "title": "Vetinari Documentation",
            "version": "1.0.0",
            "sections": ["Getting Started", "API Reference", "User Guide", "Architecture"]
        }
        
        pages = [
            {
                "title": "Getting Started",
                "path": "getting_started.md",
                "content": "## Getting Started\n\nWelcome to Vetinari!"
            },
            {
                "title": "Installation",
                "path": "installation.md",
                "content": "## Installation\n\npip install vetinari"
            },
            {
                "title": "Configuration",
                "path": "configuration.md",
                "content": "## Configuration\n\nSee config.yaml for options"
            }
        ]
        
        api_docs = {
            "title": "API Reference",
            "endpoints": [
                {
                    "path": "/plans",
                    "method": "GET",
                    "description": "List all plans",
                    "parameters": [{"name": "page", "type": "int"}],
                    "response": {"type": "object"}
                },
                {
                    "path": "/plans",
                    "method": "POST",
                    "description": "Create a new plan",
                    "body": {"type": "object"},
                    "response": {"type": "object"}
                }
            ]
        }
        
        user_guides = [
            {
                "title": "Creating Your First Plan",
                "sections": [
                    "Prerequisites",
                    "Step-by-step guide",
                    "Troubleshooting",
                    "Next steps"
                ]
            },
            {
                "title": "Using the Dashboard",
                "sections": [
                    "Overview",
                    "Navigation",
                    "Features",
                    "Tips and tricks"
                ]
            }
        ]
        
        references = {
            "configuration_ref": {
                "title": "Configuration Reference",
                "options": [
                    {"name": "debug", "type": "boolean", "default": "false"},
                    {"name": "log_level", "type": "string", "default": "INFO"}
                ]
            },
            "cli_ref": {
                "title": "CLI Reference",
                "commands": [
                    {"name": "plan", "description": "Manage plans"},
                    {"name": "execute", "description": "Execute a plan"}
                ]
            }
        }
        
        change_log = [
            {
                "version": "1.0.0",
                "date": "2026-03-03",
                "changes": [
                    "Initial release",
                    "Core agent system",
                    "Basic orchestration"
                ]
            }
        ]
        
        return {
            "docs_manifest": docs_manifest,
            "pages": pages,
            "api_docs": api_docs,
            "user_guides": user_guides,
            "references": references,
            "change_log": change_log,
            "summary": f"Generated {doc_type} documentation with {len(pages)} pages"
        }


# Singleton instance
_documentation_agent: Optional[DocumentationAgent] = None


def get_documentation_agent(config: Optional[Dict[str, Any]] = None) -> DocumentationAgent:
    """Get the singleton Documentation Agent instance."""
    global _documentation_agent
    if _documentation_agent is None:
        _documentation_agent = DocumentationAgent(config)
    return _documentation_agent
