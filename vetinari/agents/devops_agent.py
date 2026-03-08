"""
Vetinari DevOps Agent

LLM-powered DevOps and deployment specialist that handles CI/CD pipeline design,
containerisation, infrastructure-as-code, deployment strategies, and operational
runbooks.
"""

import logging
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult,
)

logger = logging.getLogger(__name__)


class DevOpsAgent(BaseAgent):
    """DevOps Agent — CI/CD, containerisation, IaC, deployment, and operational excellence."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.DEVOPS, config)
        self._platform = self._config.get("platform", "docker")

    def get_system_prompt(self) -> str:
        return f"""You are Vetinari's DevOps Engineer — a senior platform engineer and SRE.
Your role is to design and implement reliable, automated deployment pipelines and infrastructure.

Preferred platform: {self._platform}

You MUST analyse the actual project requirements and produce tailored DevOps artefacts.

Required output (JSON):
{{
  "ci_pipeline": {{
    "platform": "github-actions" | "gitlab-ci" | "jenkins" | "...",
    "stages": [{{"name": "...", "steps": [], "conditions": []}}],
    "triggers": [],
    "environment_variables": [],
    "secrets_required": [],
    "estimated_duration": "..."
  }},
  "containerisation": {{
    "dockerfile": "...",
    "docker_compose": "...",
    "base_image": "...",
    "exposed_ports": [],
    "volumes": [],
    "env_vars": []
  }},
  "deployment_strategy": {{
    "type": "blue-green" | "canary" | "rolling" | "recreate",
    "rollback_plan": "...",
    "health_checks": [],
    "readiness_probe": "...",
    "liveness_probe": "..."
  }},
  "infrastructure": {{
    "iac_tool": "terraform" | "pulumi" | "ansible" | "...",
    "resources": [],
    "iac_snippet": "...",
    "estimated_cost": "..."
  }},
  "monitoring": {{
    "metrics_to_track": [],
    "alerting_rules": [],
    "dashboards": [],
    "logging_strategy": "..."
  }},
  "runbooks": [
    {{"title": "...", "trigger": "...", "steps": [], "escalation": "..."}}
  ],
  "security_hardening": [],
  "summary": "..."
}}
"""

    def get_capabilities(self) -> List[str]:
        return [
            "ci_cd_pipeline_design",
            "containerisation",
            "infrastructure_as_code",
            "deployment_strategy",
            "monitoring_setup",
            "runbook_creation",
            "security_hardening",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute DevOps planning task using LLM inference."""
        if not self.validate_task(task):
            return AgentResult(
                success=False, output=None,
                errors=[f"Invalid task for {self._agent_type.value}"],
            )
        task = self.prepare_task(task)
        try:
            project_desc = task.context.get("project_description", task.description)
            tech_stack = task.context.get("tech_stack", "Python")

            # Search for current DevOps best practices
            devops_context = ""
            try:
                results = self._search(f"{self._platform} deployment best practices CI/CD 2025 {tech_stack}")
                if results:
                    devops_context = "\n".join([r.get("snippet", "") for r in results[:2]])
            except Exception:
                logger.debug("Failed to search for DevOps best practices", exc_info=True)

            prompt = (
                f"Design a complete DevOps setup for this project:\n{project_desc}\n\n"
                f"Tech stack: {tech_stack}\n"
                f"Platform preference: {self._platform}\n\n"
                f"Current best practices:\n{devops_context[:600]}\n\n"
                "Return a complete JSON DevOps specification."
            )

            result_data = self._infer_json(
                prompt=prompt,
                fallback=self._fallback_devops(project_desc, tech_stack),
            )

            result_data.setdefault("ci_pipeline", {})
            result_data.setdefault("containerisation", {})
            result_data.setdefault("deployment_strategy", {})
            result_data.setdefault("infrastructure", {})
            result_data.setdefault("monitoring", {})
            result_data.setdefault("runbooks", [])
            result_data.setdefault("summary", "DevOps specification generated")

            result = AgentResult(success=True, output=result_data, metadata={"platform": self._platform})
            task = self.complete_task(task, result)
            return result
        except Exception as e:
            self._log("error", f"DevOps agent failed: {e}")
            return AgentResult(success=False, output=None, errors=[str(e)])

    def verify(self, output: Any) -> VerificationResult:
        issues = []
        score = 1.0
        if not isinstance(output, dict):
            return VerificationResult(passed=False, issues=[{"type": "invalid_type", "message": "Output must be dict"}], score=0.0)
        if not output.get("ci_pipeline"):
            issues.append({"type": "missing_pipeline", "message": "CI pipeline missing"})
            score -= 0.3
        if not output.get("containerisation"):
            issues.append({"type": "missing_container", "message": "Containerisation missing"})
            score -= 0.2
        if not output.get("deployment_strategy"):
            issues.append({"type": "missing_deploy", "message": "Deployment strategy missing"})
            score -= 0.2
        return VerificationResult(passed=score >= 0.5, issues=issues, score=max(0.0, score))

    def _fallback_devops(self, description: str, tech_stack: str) -> Dict[str, Any]:
        return {
            "ci_pipeline": {
                "platform": "github-actions",
                "stages": [
                    {"name": "lint", "steps": ["pip install flake8", "flake8 ."], "conditions": []},
                    {"name": "test", "steps": ["pip install pytest pytest-cov", "pytest tests/ --cov"], "conditions": []},
                    {"name": "build", "steps": ["docker build -t app:${{ github.sha }} ."], "conditions": ["on: push"]},
                    {"name": "deploy", "steps": ["docker push", "deploy to staging"], "conditions": ["on: main branch"]},
                ],
                "triggers": ["push", "pull_request"],
                "environment_variables": ["APP_ENV", "DATABASE_URL"],
                "secrets_required": ["DOCKER_USERNAME", "DOCKER_PASSWORD"],
                "estimated_duration": "~5 minutes",
            },
            "containerisation": {
                "dockerfile": "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .\nEXPOSE 5000\nCMD [\"python\", \"-m\", \"vetinari\", \"serve\"]",
                "docker_compose": "version: '3.8'\nservices:\n  app:\n    build: .\n    ports:\n      - '5000:5000'\n    environment:\n      - LM_STUDIO_HOST=${LM_STUDIO_HOST:-http://host.docker.internal:1234}",
                "base_image": "python:3.11-slim",
                "exposed_ports": [5000],
                "volumes": ["./logs:/app/logs", "./projects:/app/projects"],
                "env_vars": ["LM_STUDIO_HOST", "CLAUDE_API_KEY"],
            },
            "deployment_strategy": {
                "type": "blue-green",
                "rollback_plan": "Switch load balancer back to previous version",
                "health_checks": ["/api/status returns 200", "LM Studio connection OK"],
                "readiness_probe": "GET /api/status",
                "liveness_probe": "GET /api/status",
            },
            "infrastructure": {
                "iac_tool": "docker-compose",
                "resources": ["app container", "optional: nginx reverse proxy"],
                "iac_snippet": "# Use docker-compose for local/small deployments\n# For cloud: consider Kubernetes or Fly.io",
                "estimated_cost": "$0 (local) / ~$10-30/month (cloud VPS)",
            },
            "monitoring": {
                "metrics_to_track": ["request latency", "error rate", "token usage", "model availability"],
                "alerting_rules": ["error rate > 5%", "latency > 10s", "model unavailable"],
                "dashboards": ["Vetinari built-in dashboard at /dashboard"],
                "logging_strategy": "Structured JSON logs to ./logs/, rotate daily",
            },
            "runbooks": [
                {"title": "LM Studio Unreachable", "trigger": "Health check fails", "steps": ["Check LM_STUDIO_HOST env var", "Verify LM Studio is running", "Check network connectivity"], "escalation": "Restart LM Studio service"},
                {"title": "High Memory Usage", "trigger": "Memory > 90%", "steps": ["Identify large model", "Offload unused models", "Reduce concurrency"], "escalation": "Restart with smaller model"},
            ],
            "security_hardening": [
                "Run container as non-root user",
                "Use secrets management (not env vars) for API keys",
                "Enable rate limiting on all endpoints",
                "Add Content-Security-Policy headers",
            ],
            "summary": f"DevOps setup designed for {tech_stack} project. Using {self._platform}.",
        }


_devops_agent: Optional[DevOpsAgent] = None


def get_devops_agent(config: Optional[Dict[str, Any]] = None) -> DevOpsAgent:
    global _devops_agent
    if _devops_agent is None:
        _devops_agent = DevOpsAgent(config)
    return _devops_agent
