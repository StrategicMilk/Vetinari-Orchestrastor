"""
Consolidated Operations Agent (Phase 3)
========================================
Replaces: SYNTHESIZER + DOCUMENTATION_AGENT + COST_PLANNER +
          EXPERIMENTATION_MANAGER + IMPROVEMENT + ERROR_RECOVERY +
          IMAGE_GENERATOR

Modes:
- documentation: API docs, user guides, changelogs (from DOCUMENTATION_AGENT)
- creative_writing: Creative content generation (from SYNTHESIZER)
- cost_analysis: Model selection, cost optimization (from COST_PLANNER)
- experiment: A/B testing, experiment tracking (from EXPERIMENTATION_MANAGER)
- error_recovery: Failure analysis, retry strategies (from ERROR_RECOVERY)
- synthesis: Multi-source artifact fusion (from SYNTHESIZER)
- image_generation: Logo, icon, diagram generation (from IMAGE_GENERATOR)
- improvement: System performance analysis (from IMPROVEMENT)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from vetinari.agents.multi_mode_agent import MultiModeAgent
from vetinari.agents.contracts import AgentResult, AgentTask, AgentType, VerificationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error pattern registry (preserved from ErrorRecoveryAgent)
# ---------------------------------------------------------------------------

_ERROR_PATTERNS: Dict[str, Dict[str, Any]] = {
    "connection_refused": {
        "patterns": [r"ConnectionRefusedError", r"Connection refused", r"ECONNREFUSED"],
        "category": "network", "severity": "high",
        "quick_fix": "Check if the target service is running and accessible",
    },
    "timeout": {
        "patterns": [r"TimeoutError", r"timed? ?out", r"deadline exceeded"],
        "category": "network", "severity": "medium",
        "quick_fix": "Increase timeout or check network connectivity",
    },
    "rate_limit": {
        "patterns": [r"429", r"rate.?limit", r"too many requests", r"quota exceeded"],
        "category": "api", "severity": "medium",
        "quick_fix": "Implement exponential backoff or request queuing",
    },
    "out_of_memory": {
        "patterns": [r"MemoryError", r"OOM", r"out of memory", r"ENOMEM"],
        "category": "resource", "severity": "critical",
        "quick_fix": "Reduce batch size, use streaming, or increase memory allocation",
    },
    "import_error": {
        "patterns": [r"ImportError", r"ModuleNotFoundError", r"No module named"],
        "category": "dependency", "severity": "high",
        "quick_fix": "Install missing package or check virtual environment",
    },
    "permission_denied": {
        "patterns": [r"PermissionError", r"Permission denied", r"EACCES"],
        "category": "filesystem", "severity": "high",
        "quick_fix": "Check file/directory permissions and ownership",
    },
    "file_not_found": {
        "patterns": [r"FileNotFoundError", r"No such file", r"ENOENT"],
        "category": "filesystem", "severity": "medium",
        "quick_fix": "Verify file path and ensure parent directories exist",
    },
    "json_decode": {
        "patterns": [r"JSONDecodeError", r"json\.decoder", r"Expecting value"],
        "category": "parsing", "severity": "medium",
        "quick_fix": "Validate JSON input, check for empty responses or HTML errors",
    },
    "key_error": {
        "patterns": [r"KeyError", r"key not found"],
        "category": "data", "severity": "medium",
        "quick_fix": "Use .get() with default values or validate dict keys before access",
    },
    "type_error": {
        "patterns": [r"TypeError", r"not callable", r"expected .* got"],
        "category": "logic", "severity": "medium",
        "quick_fix": "Check argument types and function signatures",
    },
}

# Model pricing (preserved from CostPlannerAgent)
_MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "qwen2.5-coder-7b": {"input_per_1k": 0.0001, "output_per_1k": 0.0002, "tier": "small"},
    "qwen2.5-72b": {"input_per_1k": 0.001, "output_per_1k": 0.002, "tier": "large"},
    "qwen3-30b-a3b": {"input_per_1k": 0.0005, "output_per_1k": 0.001, "tier": "medium"},
    "qwen2.5-vl-32b": {"input_per_1k": 0.0005, "output_per_1k": 0.001, "tier": "medium"},
    "claude-3.5-sonnet": {"input_per_1k": 0.003, "output_per_1k": 0.015, "tier": "premium"},
    "gpt-4o": {"input_per_1k": 0.005, "output_per_1k": 0.015, "tier": "premium"},
    "gemini-1.5-pro": {"input_per_1k": 0.00125, "output_per_1k": 0.005, "tier": "large"},
}


class OperationsAgent(MultiModeAgent):
    """Unified operations agent for docs, cost analysis, experiments,
    error recovery, synthesis, image generation, and system improvement."""

    MODES = {
        "documentation": "_execute_documentation",
        "creative_writing": "_execute_creative_writing",
        "cost_analysis": "_execute_cost_analysis",
        "experiment": "_execute_experiment",
        "error_recovery": "_execute_error_recovery",
        "synthesis": "_execute_synthesis",
        "image_generation": "_execute_image_generation",
        "improvement": "_execute_improvement",
    }
    DEFAULT_MODE = "documentation"
    MODE_KEYWORDS = {
        "documentation": ["document", "api doc", "user guide", "changelog", "readme", "reference"],
        "creative_writing": ["creative", "story", "narrative", "prose", "write"],
        "cost_analysis": ["cost", "pricing", "budget", "roi", "expense", "token cost", "model selection"],
        "experiment": ["experiment", "a/b test", "hypothesis", "metric", "telemetry", "variant"],
        "error_recovery": ["error", "failure", "crash", "exception", "retry", "recover", "circuit break"],
        "synthesis": ["synthesiz", "synthesise", "merge", "fusion", "consolidat", "combin"],
        "image_generation": ["image", "logo", "icon", "diagram", "mockup", "visual", "generate image"],
        "improvement": ["improv", "optimiz", "performance", "bottleneck", "tune", "enhance"],
    }
    LEGACY_TYPE_TO_MODE = {
        "DOCUMENTATION_AGENT": "documentation",
        "SYNTHESIZER": "synthesis",
        "COST_PLANNER": "cost_analysis",
        "EXPERIMENTATION_MANAGER": "experiment",
        "IMPROVEMENT": "improvement",
        "ERROR_RECOVERY": "error_recovery",
        "IMAGE_GENERATOR": "image_generation",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.OPERATIONS, config)

    def _get_base_system_prompt(self) -> str:
        return (
            "You are Vetinari's Operations Agent. You handle documentation, "
            "cost analysis, experiments, error recovery, synthesis, image generation, "
            "and system improvement."
        )

    def _get_mode_system_prompt(self, mode: str) -> str:
        prompts = {
            "documentation": (
                "You are Vetinari's Documentation Specialist. Generate:\n"
                "- API reference docs (OpenAPI-style)\n"
                "- User guides (audience-aware)\n"
                "- Changelogs (semantic versioning)\n"
                "- README files\n\n"
                "Use clear, professional Markdown formatting."
            ),
            "error_recovery": (
                "You are Vetinari's Error Recovery Specialist. Your role is to:\n"
                "- Analyze failures and identify root causes\n"
                "- Design retry strategies (exponential backoff, circuit breaking)\n"
                "- Create fallback plans for degraded operation\n"
                "- Generate post-mortem analysis\n\n"
                "Prioritize system stability and data integrity."
            ),
            "cost_analysis": (
                "You are Vetinari's Cost Analyst. Your role is to:\n"
                "- Calculate token costs per model and task type\n"
                "- Recommend cost-efficient model selections\n"
                "- Analyze ROI of different approaches\n"
                "- Track and forecast spending\n\n"
                "Include concrete numbers and comparisons."
            ),
        }
        return prompts.get(mode, "")

    def verify(self, output: Any) -> VerificationResult:
        if output is None:
            return VerificationResult(passed=False, issues=[{"message": "No output"}], score=0.0)
        return VerificationResult(passed=True, score=0.7)

    # ------------------------------------------------------------------
    # Documentation (from DocumentationAgent)
    # ------------------------------------------------------------------

    def _execute_documentation(self, task: AgentTask) -> AgentResult:
        content = task.context.get("content", task.description)
        doc_type = task.context.get("doc_type", "api_reference")
        audience = task.context.get("audience", "developers")

        prompt = (
            f"Generate {doc_type} documentation for:\n{content[:4000]}\n\n"
            f"Audience: {audience}\n\n"
            "Respond as JSON:\n"
            '{"content": "...markdown content...", "type": "' + doc_type + '", '
            '"sections": [{"title": "...", "content": "..."}], '
            '"metadata": {"audience": "' + audience + '", "word_count": 0}}'
        )
        result = self._infer_json(prompt, fallback={"content": "", "type": doc_type, "sections": []})
        return AgentResult(success=True, output=result, metadata={"mode": "documentation", "doc_type": doc_type})

    # ------------------------------------------------------------------
    # Creative Writing (from SynthesizerAgent)
    # ------------------------------------------------------------------

    def _execute_creative_writing(self, task: AgentTask) -> AgentResult:
        content = task.context.get("content", task.description)
        style = task.context.get("style", "professional")

        prompt = (
            f"Create creative content:\n{content[:4000]}\n\nStyle: {style}\n\n"
            "Respond as JSON:\n"
            '{"content": "...", "type": "creative", "word_count": 0, "tone": "' + style + '"}'
        )
        result = self._infer_json(prompt, fallback={"content": "", "type": "creative"})
        return AgentResult(success=True, output=result, metadata={"mode": "creative_writing"})

    # ------------------------------------------------------------------
    # Cost Analysis (from CostPlannerAgent — preserves MODEL_PRICING)
    # ------------------------------------------------------------------

    def _execute_cost_analysis(self, task: AgentTask) -> AgentResult:
        context = task.context or {}
        analysis_type = context.get("analysis_type", "model_comparison")

        if analysis_type == "model_comparison":
            models = context.get("models", list(_MODEL_PRICING.keys()))
            estimated_tokens = context.get("estimated_tokens", 10000)

            comparisons = []
            for model_id in models:
                pricing = _MODEL_PRICING.get(model_id, {"input_per_1k": 0.001, "output_per_1k": 0.002, "tier": "unknown"})
                input_cost = (estimated_tokens / 1000) * pricing["input_per_1k"]
                output_cost = (estimated_tokens / 1000) * pricing["output_per_1k"]
                comparisons.append({
                    "model": model_id, "tier": pricing["tier"],
                    "input_cost": round(input_cost, 4),
                    "output_cost": round(output_cost, 4),
                    "total_cost": round(input_cost + output_cost, 4),
                })
            comparisons.sort(key=lambda c: c["total_cost"])

            return AgentResult(
                success=True,
                output={
                    "comparisons": comparisons,
                    "recommendation": comparisons[0]["model"] if comparisons else "unknown",
                    "estimated_tokens": estimated_tokens,
                    "cheapest": comparisons[0] if comparisons else None,
                    "most_expensive": comparisons[-1] if comparisons else None,
                },
                metadata={"mode": "cost_analysis", "analysis_type": analysis_type},
            )

        # General cost analysis via LLM
        prompt = (
            f"Perform cost analysis for:\n{task.description[:4000]}\n\n"
            "Respond as JSON:\n"
            '{"analysis": "...", "recommendations": [...], "estimated_savings": "..."}'
        )
        result = self._infer_json(prompt, fallback={"analysis": "", "recommendations": []})
        return AgentResult(success=True, output=result, metadata={"mode": "cost_analysis"})

    # ------------------------------------------------------------------
    # Experiment (from ExperimentationManagerAgent)
    # ------------------------------------------------------------------

    def _execute_experiment(self, task: AgentTask) -> AgentResult:
        context = task.context or {}
        experiment_type = context.get("experiment_type", "design")

        prompt = (
            f"Design/analyze experiment:\n{task.description[:4000]}\n\n"
            f"Type: {experiment_type}\n\n"
            "Respond as JSON:\n"
            '{"experiment": {"name": "...", "hypothesis": "...", "type": "' + experiment_type + '"}, '
            '"metrics": [{"name": "...", "type": "primary|secondary", "target": 0.0}], '
            '"variants": [{"name": "control", "description": "..."}, {"name": "treatment", "description": "..."}], '
            '"sample_size": 0, "duration_days": 7, '
            '"analysis_plan": "..."}'
        )
        result = self._infer_json(prompt, fallback={"experiment": {"name": "", "hypothesis": ""}, "metrics": []})
        return AgentResult(success=True, output=result, metadata={"mode": "experiment"})

    # ------------------------------------------------------------------
    # Error Recovery (from ErrorRecoveryAgent — preserves pattern registry)
    # ------------------------------------------------------------------

    def _execute_error_recovery(self, task: AgentTask) -> AgentResult:
        error_text = task.context.get("error", task.description)

        # Pattern matching
        matched_patterns = []
        for pattern_name, info in _ERROR_PATTERNS.items():
            for regex in info["patterns"]:
                try:
                    if re.search(regex, error_text, re.IGNORECASE):
                        matched_patterns.append({
                            "pattern": pattern_name,
                            "category": info["category"],
                            "severity": info["severity"],
                            "quick_fix": info["quick_fix"],
                        })
                        break
                except re.error:
                    pass

        # LLM analysis
        pattern_context = ""
        if matched_patterns:
            pattern_context = "\n\nPattern matches:\n" + "\n".join(
                f"- [{p['severity']}] {p['pattern']}: {p['quick_fix']}" for p in matched_patterns
            )

        prompt = (
            f"Analyze this error and provide recovery strategy:\n{error_text[:4000]}\n"
            f"{pattern_context}\n\n"
            "Respond as JSON:\n"
            '{"root_cause": "...", "category": "...", "severity": "critical|high|medium|low", '
            '"recovery_strategy": {"immediate": "...", "retry_policy": {"type": "exponential_backoff", '
            '"max_retries": 3, "base_delay_ms": 1000}, "fallback": "..."}, '
            '"prevention": [...], "post_mortem": "..."}'
        )
        result = self._infer_json(prompt, fallback={
            "root_cause": "Analysis unavailable",
            "matched_patterns": matched_patterns,
            "recovery_strategy": {"immediate": matched_patterns[0]["quick_fix"] if matched_patterns else "Manual investigation required"},
        })
        if result and isinstance(result, dict):
            result["matched_patterns"] = matched_patterns
        return AgentResult(success=True, output=result,
                           metadata={"mode": "error_recovery", "patterns_matched": len(matched_patterns)})

    # ------------------------------------------------------------------
    # Synthesis (from SynthesizerAgent)
    # ------------------------------------------------------------------

    def _execute_synthesis(self, task: AgentTask) -> AgentResult:
        sources = task.context.get("sources", [])
        content = task.context.get("content", task.description)

        prompt = (
            f"Synthesize the following sources into a unified artifact:\n\n"
            f"Content: {content[:3000]}\n"
            f"Sources ({len(sources)}): {str(sources)[:2000]}\n\n"
            "Respond as JSON:\n"
            '{"synthesis": "...", "sources_used": [...], '
            '"conflicts_resolved": [...], "confidence": 0.8}'
        )
        result = self._infer_json(prompt, fallback={"synthesis": "", "sources_used": []})
        return AgentResult(success=True, output=result, metadata={"mode": "synthesis"})

    # ------------------------------------------------------------------
    # Image Generation (from ImageGeneratorAgent)
    # ------------------------------------------------------------------

    def _execute_image_generation(self, task: AgentTask) -> AgentResult:
        description = task.context.get("description", task.description)
        style = task.context.get("style", "logo")
        width = task.context.get("width", 512)
        height = task.context.get("height", 512)

        # Try Stable Diffusion WebUI API
        try:
            import requests
            sd_host = task.context.get("sd_host", "http://localhost:7860")
            payload = {
                "prompt": description,
                "negative_prompt": "blurry, low quality, distorted",
                "width": width, "height": height,
                "steps": 30, "cfg_scale": 7.5,
            }
            resp = requests.post(f"{sd_host}/sdapi/v1/txt2img", json=payload, timeout=60)
            if resp.status_code == 200:
                images = resp.json().get("images", [])
                return AgentResult(
                    success=True,
                    output={"images": images, "prompt": description, "style": style,
                            "dimensions": f"{width}x{height}", "generator": "stable_diffusion"},
                    metadata={"mode": "image_generation", "generator": "stable_diffusion"},
                )
        except Exception as e:
            logger.debug("Stable Diffusion unavailable: %s", e)

        # SVG fallback
        svg = self._generate_svg_fallback(description, style, width, height)
        return AgentResult(
            success=True,
            output={"svg": svg, "prompt": description, "style": style,
                    "dimensions": f"{width}x{height}", "generator": "svg_fallback"},
            metadata={"mode": "image_generation", "generator": "svg_fallback"},
        )

    def _generate_svg_fallback(self, desc: str, style: str, w: int, h: int) -> str:
        colors = {"logo": "#2563EB", "icon": "#10B981", "diagram": "#6366F1", "ui_mockup": "#8B5CF6"}
        color = colors.get(style, "#3B82F6")
        label = desc[:20] if desc else style
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<rect width="{w}" height="{h}" fill="#f8fafc" rx="8"/>'
            f'<rect x="{w//4}" y="{h//4}" width="{w//2}" height="{h//2}" fill="{color}" rx="12" opacity="0.8"/>'
            f'<text x="{w//2}" y="{h//2}" text-anchor="middle" dy=".3em" fill="white" '
            f'font-family="system-ui" font-size="16" font-weight="600">{label}</text>'
            f'</svg>'
        )

    # ------------------------------------------------------------------
    # Improvement (from ImprovementAgent)
    # ------------------------------------------------------------------

    def _execute_improvement(self, task: AgentTask) -> AgentResult:
        context = task.context or {}
        focus = context.get("focus", "general")

        prompt = (
            f"Analyze system performance and recommend improvements:\n{task.description[:4000]}\n\n"
            f"Focus: {focus}\n\n"
            "Respond as JSON:\n"
            '{"analysis": {"summary": "...", "bottlenecks": [...]}, '
            '"recommendations": [{"title": "...", "impact": "high|medium|low", '
            '"effort": "high|medium|low", "description": "..."}], '
            '"metrics": {"current": {...}, "projected": {...}}, '
            '"priority_actions": [...]}'
        )
        result = self._infer_json(prompt, fallback={"analysis": {"summary": ""}, "recommendations": []})
        return AgentResult(success=True, output=result, metadata={"mode": "improvement", "focus": focus})

    def get_capabilities(self) -> List[str]:
        return [
            "documentation", "api_docs", "user_guides", "changelog",
            "creative_writing", "narrative_generation",
            "cost_analysis", "model_comparison", "roi_analysis",
            "experiment_design", "ab_testing", "metrics_tracking",
            "error_recovery", "failure_analysis", "retry_strategy",
            "multi_source_synthesis", "conflict_resolution",
            "image_generation", "svg_generation",
            "performance_analysis", "system_improvement",
        ]


# Singleton
_operations_agent: Optional[OperationsAgent] = None


def get_operations_agent(config: Optional[Dict[str, Any]] = None) -> OperationsAgent:
    global _operations_agent
    if _operations_agent is None:
        _operations_agent = OperationsAgent(config)
    return _operations_agent
