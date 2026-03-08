"""
ErrorRecoveryAgent - Failure analysis, retry strategies, circuit breaking, fallback planning.

Provides intelligent error recovery guidance including:
- Root cause analysis from error logs and stack traces
- Retry strategy generation (exponential backoff, circuit breaking)
- Fallback plan creation
- Error pattern recognition and classification
- Recovery action prioritisation
- Post-mortem analysis
"""

from __future__ import annotations

import json
import logging
import re
import traceback
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentTask, AgentResult, AgentType, VerificationResult,
)

logger = logging.getLogger(__name__)

# Known error patterns for quick classification
_ERROR_PATTERNS: Dict[str, Dict[str, str]] = {
    "connection_refused": {
        "pattern": r"ConnectionRefusedError|connection refused|ECONNREFUSED",
        "category": "network",
        "severity": "high",
        "quick_fix": "Check that the target service is running and the port is correct",
    },
    "timeout": {
        "pattern": r"TimeoutError|ReadTimeout|ConnectTimeout|timed out",
        "category": "network",
        "severity": "medium",
        "quick_fix": "Increase timeout threshold or check network latency",
    },
    "rate_limit": {
        "pattern": r"429|RateLimitError|rate.?limit|too many requests",
        "category": "api",
        "severity": "medium",
        "quick_fix": "Implement exponential backoff with jitter",
    },
    "out_of_memory": {
        "pattern": r"MemoryError|OOMKilled|out of memory|ENOMEM",
        "category": "resource",
        "severity": "critical",
        "quick_fix": "Reduce batch size, add streaming, or scale memory",
    },
    "import_error": {
        "pattern": r"ImportError|ModuleNotFoundError|No module named",
        "category": "dependency",
        "severity": "high",
        "quick_fix": "Install missing package or check virtual environment",
    },
    "attribute_error": {
        "pattern": r"AttributeError",
        "category": "code",
        "severity": "high",
        "quick_fix": "Check object type and attribute name; likely an API change",
    },
    "key_error": {
        "pattern": r"KeyError",
        "category": "code",
        "severity": "medium",
        "quick_fix": "Use .get() with a default value or validate dict keys before access",
    },
    "type_error": {
        "pattern": r"TypeError",
        "category": "code",
        "severity": "high",
        "quick_fix": "Check type annotations and function call signatures",
    },
    "permission_denied": {
        "pattern": r"PermissionError|EACCES|Permission denied|403",
        "category": "auth",
        "severity": "high",
        "quick_fix": "Check file/API permissions and credentials",
    },
    "not_found": {
        "pattern": r"FileNotFoundError|404|not found",
        "category": "resource",
        "severity": "medium",
        "quick_fix": "Verify path/URL exists and is accessible",
    },
    "json_decode": {
        "pattern": r"JSONDecodeError|json.decoder|Invalid JSON",
        "category": "data",
        "severity": "medium",
        "quick_fix": "Validate JSON input before parsing; add error handling around json.loads()",
    },
}


class ErrorRecoveryAgent(BaseAgent):
    """Agent for error analysis, recovery strategy generation, and resilience improvement."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.ERROR_RECOVERY, config)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def get_system_prompt(self) -> str:
        return (
            "You are an expert in software resilience and error recovery. Your role is to:\n"
            "- Analyse error messages, stack traces, and logs to identify root causes\n"
            "- Classify errors by category (network, code, resource, auth, data, etc.)\n"
            "- Generate prioritised recovery strategies (immediate fixes, preventive measures)\n"
            "- Design retry policies (exponential backoff, circuit breaking, fallback chains)\n"
            "- Identify error patterns and systemic issues\n"
            "- Produce actionable post-mortem summaries\n\n"
            "Always respond with structured JSON as specified in the task context."
        )

    def get_capabilities(self) -> List[str]:
        return [
            "root_cause_analysis",
            "error_classification",
            "retry_strategy_generation",
            "circuit_breaker_design",
            "fallback_planning",
            "error_pattern_recognition",
            "post_mortem_analysis",
            "resilience_improvement",
            "recovery_prioritisation",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        self.validate_task(task)
        self.prepare_task(task)
        try:
            result = self._analyse_and_recover(task)
            agent_result = AgentResult(
                success=True,
                output=result,
                metadata={
                    "task_id": task.task_id,
                    "agent_type": self._agent_type.value,
                    "error_count": len(result.get("errors_identified", [])),
                },
            )
            self.complete_task(task, agent_result)
            return agent_result
        except Exception as exc:
            logger.error("[ErrorRecoveryAgent] execute() failed: %s", exc)
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
        if not output.get("root_cause") and not output.get("errors_identified"):
            issues.append({"severity": "warning", "message": "No root cause or errors identified"})
            score -= 0.3
        if not output.get("recovery_strategies"):
            issues.append({"severity": "warning", "message": "No recovery strategies provided"})
            score -= 0.3
        if not output.get("immediate_actions"):
            issues.append({"severity": "info", "message": "No immediate actions specified"})
            score -= 0.2
        passed = score >= 0.5
        return VerificationResult(passed=passed, score=round(score, 2), issues=issues)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyse_and_recover(self, task: AgentTask) -> Dict[str, Any]:
        """Perform error analysis and generate recovery plan."""
        ctx = task.context or {}
        error_text = ctx.get("error", "") or ctx.get("error_message", "") or task.description or ""
        stack_trace = ctx.get("stack_trace", "") or ctx.get("traceback", "")
        log_snippet = ctx.get("logs", "") or ctx.get("log_snippet", "")
        component = ctx.get("component", "unknown")

        # Quick heuristic classification
        heuristic_findings = self._classify_errors_heuristically(
            f"{error_text}\n{stack_trace}"
        )

        # Web search for known solutions
        search_context = ""
        if heuristic_findings:
            primary_error = heuristic_findings[0].get("category", "")
            try:
                results = self._search(
                    f"{primary_error} error recovery best practices retry strategy 2026",
                    max_results=3,
                )
                if results:
                    search_context = "\n".join(
                        f"- {r['title']}: {r['snippet']}" for r in results[:3]
                    )
            except Exception:
                logger.debug("Failed to search for error recovery best practices", exc_info=True)

        prompt = f"""You are an error recovery expert. Analyse the following error context and produce a recovery plan.

## Component
{component}

## Error / Description
{error_text[:2000]}

## Stack Trace
{stack_trace[:2000] if stack_trace else 'Not provided'}

## Log Snippet
{log_snippet[:1000] if log_snippet else 'Not provided'}

## Heuristic Classification
{json.dumps(heuristic_findings, indent=2)}

## Reference Best Practices
{search_context or 'Apply standard resilience patterns.'}

## Required Output (JSON)
{{
  "root_cause": "Concise root cause description",
  "errors_identified": [
    {{"type": "...", "category": "network|code|resource|auth|data|config", "severity": "low|medium|high|critical", "location": "..."}}
  ],
  "immediate_actions": [
    {{"priority": 1, "action": "...", "command": "...", "expected_result": "..."}}
  ],
  "recovery_strategies": [
    {{
      "name": "...",
      "type": "retry|circuit_breaker|fallback|bulkhead|timeout",
      "description": "...",
      "implementation": "...",
      "code_snippet": "..."
    }}
  ],
  "retry_policy": {{
    "max_attempts": 3,
    "initial_delay_ms": 1000,
    "backoff_multiplier": 2.0,
    "max_delay_ms": 30000,
    "jitter": true
  }},
  "circuit_breaker": {{
    "enabled": true,
    "failure_threshold": 5,
    "timeout_seconds": 60,
    "half_open_requests": 1
  }},
  "fallback_plan": "...",
  "preventive_measures": ["...", ...],
  "post_mortem_summary": "...",
  "monitoring_recommendations": ["...", ...]
}}

Return ONLY valid JSON."""

        result = self._infer_json(
            prompt, fallback=self._fallback_recovery(task, heuristic_findings)
        )
        if result and isinstance(result, dict):
            return result
        return self._fallback_recovery(task, heuristic_findings)

    def _classify_errors_heuristically(self, text: str) -> List[Dict[str, str]]:
        """Quick regex-based error classification before LLM analysis."""
        findings = []
        for name, spec in _ERROR_PATTERNS.items():
            if re.search(spec["pattern"], text, re.IGNORECASE):
                findings.append({
                    "type": name,
                    "category": spec["category"],
                    "severity": spec["severity"],
                    "quick_fix": spec["quick_fix"],
                })
        return findings

    def _fallback_recovery(
        self, task: AgentTask, heuristic_findings: List[Dict]
    ) -> Dict[str, Any]:
        """Structured fallback when LLM is unavailable."""
        immediate_actions = []
        for i, finding in enumerate(heuristic_findings[:3], 1):
            immediate_actions.append({
                "priority": i,
                "action": finding["quick_fix"],
                "command": "",
                "expected_result": f"Resolve {finding['type']} error",
            })
        if not immediate_actions:
            immediate_actions = [
                {"priority": 1, "action": "Check logs for detailed error context",
                 "command": "tail -100 logs/server.log", "expected_result": "Error details visible"},
                {"priority": 2, "action": "Verify service health",
                 "command": "python -m vetinari health", "expected_result": "All services healthy"},
            ]
        return {
            "root_cause": "Unable to determine root cause without LLM - review error details manually",
            "errors_identified": heuristic_findings or [
                {"type": "unknown", "category": "unknown", "severity": "medium",
                 "location": "unknown"}
            ],
            "immediate_actions": immediate_actions,
            "recovery_strategies": [
                {
                    "name": "Exponential Backoff Retry",
                    "type": "retry",
                    "description": "Retry failed operations with exponential backoff",
                    "implementation": "Use tenacity or custom retry decorator",
                    "code_snippet": (
                        "from tenacity import retry, stop_after_attempt, wait_exponential\n\n"
                        "@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))\n"
                        "def your_function():\n    ..."
                    ),
                },
                {
                    "name": "Circuit Breaker",
                    "type": "circuit_breaker",
                    "description": "Prevent cascade failures by opening circuit after repeated failures",
                    "implementation": "Use pybreaker or custom circuit breaker",
                    "code_snippet": (
                        "import pybreaker\ncb = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60)\n"
                        "@cb\ndef your_function():\n    ..."
                    ),
                },
            ],
            "retry_policy": {
                "max_attempts": 3,
                "initial_delay_ms": 1000,
                "backoff_multiplier": 2.0,
                "max_delay_ms": 30000,
                "jitter": True,
            },
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 5,
                "timeout_seconds": 60,
                "half_open_requests": 1,
            },
            "fallback_plan": "Log error, notify operator, continue with degraded functionality",
            "preventive_measures": [
                "Add comprehensive error handling with specific exception types",
                "Implement health checks for all external dependencies",
                "Add distributed tracing to identify failure points",
                "Set up alerting for error rate thresholds",
            ],
            "post_mortem_summary": f"Error in {task.context.get('component', 'unknown') if task.context else 'unknown'}: {task.description[:100] if task.description else 'unknown error'}",
            "monitoring_recommendations": [
                "Alert on error rate > 1% per minute",
                "Track P99 latency for all external calls",
                "Monitor circuit breaker state changes",
            ],
        }


# Singleton
_error_recovery_agent: Optional[ErrorRecoveryAgent] = None


def get_error_recovery_agent(
    config: Optional[Dict[str, Any]] = None
) -> ErrorRecoveryAgent:
    """Get the singleton ErrorRecoveryAgent instance."""
    global _error_recovery_agent
    if _error_recovery_agent is None:
        _error_recovery_agent = ErrorRecoveryAgent(config)
    return _error_recovery_agent
