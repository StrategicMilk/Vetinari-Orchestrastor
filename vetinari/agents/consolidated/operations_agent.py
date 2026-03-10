"""
Consolidated Operations Agent (v0.4.0)
========================================
Replaces: SYNTHESIZER + DOCUMENTATION_AGENT + COST_PLANNER +
          EXPERIMENTATION_MANAGER + IMPROVEMENT + ERROR_RECOVERY

Modes:
- documentation: API docs, user guides, changelogs (from DOCUMENTATION_AGENT)
- creative_writing: Creative content generation (from SYNTHESIZER)
- cost_analysis: Model selection, cost optimization (from COST_PLANNER)
- experiment: A/B testing, experiment tracking (from EXPERIMENTATION_MANAGER)
- error_recovery: Failure analysis, retry strategies (from ERROR_RECOVERY)
- synthesis: Multi-source artifact fusion (from SYNTHESIZER)
- improvement: System performance analysis (from IMPROVEMENT)
- monitor: System health and performance tracking (from ORCHESTRATOR)
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
    error recovery, synthesis, system improvement, and monitoring."""

    MODES = {
        "documentation": "_execute_documentation",
        "creative_writing": "_execute_creative_writing",
        "cost_analysis": "_execute_cost_analysis",
        "experiment": "_execute_experiment",
        "error_recovery": "_execute_error_recovery",
        "synthesis": "_execute_synthesis",
        "improvement": "_execute_improvement",
        "monitor": "_execute_monitor",
        "devops_ops": "_execute_devops_ops",
    }
    DEFAULT_MODE = "documentation"
    MODE_KEYWORDS = {
        "documentation": ["document", "api doc", "user guide", "changelog", "readme", "reference"],
        "creative_writing": ["creative", "story", "narrative", "prose", "write"],
        "cost_analysis": ["cost", "pricing", "budget", "roi", "expense", "token cost", "model selection"],
        "experiment": ["experiment", "a/b test", "hypothesis", "metric", "telemetry", "variant"],
        "error_recovery": ["error", "failure", "crash", "exception", "retry", "recover", "circuit break"],
        "synthesis": ["synthesiz", "synthesise", "merge", "fusion", "consolidat", "combin"],
        "improvement": ["improv", "optimiz", "performance", "bottleneck", "tune", "enhance"],
        "monitor": ["monitor", "status", "health", "performance check", "system check"],
        "devops_ops": ["deploy", "ci/cd", "pipeline", "docker", "infrastructure", "devops"],
    }
    LEGACY_TYPE_TO_MODE = {
        "DOCUMENTATION_AGENT": "documentation",
        "SYNTHESIZER": "synthesis",
        "COST_PLANNER": "cost_analysis",
        "EXPERIMENTATION_MANAGER": "experiment",
        "IMPROVEMENT": "improvement",
        "ERROR_RECOVERY": "error_recovery",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.OPERATIONS, config)

    def _get_base_system_prompt(self) -> str:
        return (
            "You are Vetinari's Operations Agent — the system's operational backbone\n"
            "responsible for documentation, cost optimisation, experimentation, error recovery,\n"
            "multi-source synthesis, continuous improvement, system monitoring, and DevOps.\n\n"
            "You operate across 9 specialised modes, each with domain-specific expertise.\n"
            "Always produce structured JSON output. Prioritise actionable recommendations\n"
            "over theoretical analysis. Include concrete numbers, code snippets, and\n"
            "evidence-based reasoning in all outputs."
        )

    def _get_mode_system_prompt(self, mode: str) -> str:
        prompts = {
            "documentation": (
                "You are Vetinari's Documentation Specialist — a technical writer with deep\n"
                "expertise in API documentation, developer experience, and information architecture.\n"
                "You produce documentation that is accurate, scannable, and immediately useful.\n\n"
                "## Core Responsibilities\n"
                "- Generate API reference documentation following OpenAPI/Swagger conventions\n"
                "- Write audience-aware user guides (beginner, intermediate, advanced tiers)\n"
                "- Produce changelogs following Keep a Changelog + Semantic Versioning\n"
                "- Create README files with Quick Start sections that get users running in <5 min\n"
                "- Write inline code comments and docstrings (Google/NumPy style)\n\n"
                "## Documentation Standards\n"
                "- Use imperative mood for instructions ('Install the package', not 'You should install')\n"
                "- Every public function/class documents: purpose, parameters, return value, exceptions, example\n"
                "- Include runnable code examples — never pseudo-code in API docs\n"
                "- Structure with progressive disclosure: overview > quick start > detailed reference\n"
                "- Cross-reference related sections with relative links\n"
                "- Keep README files under 500 lines; link to /docs/ for deep content\n\n"
                "## Quality Checks\n"
                "- Verify all code examples are syntactically valid\n"
                "- Ensure parameter names match actual function signatures\n"
                "- Check that version numbers and paths are current\n"
                "- Flag undocumented public APIs as gaps\n"
                "- Validate Markdown renders correctly (no broken tables or links)\n\n"
                "## Output Format\n"
                "Return structured JSON with 'content' (full Markdown), 'type' (api_reference|guide|changelog),\n"
                "'sections' (array of {title, content}), and 'metadata' (audience, word_count, reading_time_min).\n"
                "Use clear, professional Markdown formatting with consistent heading hierarchy."
            ),
            "creative_writing": (
                "You are Vetinari's Creative Content Specialist — a versatile writer capable of\n"
                "producing engaging, well-structured content across multiple formats and styles.\n"
                "You balance creativity with clarity, and voice with purpose.\n\n"
                "## Core Responsibilities\n"
                "- Generate release announcements, blog posts, and marketing copy for technical products\n"
                "- Write project narratives and case studies that explain complex systems accessibly\n"
                "- Produce onboarding content, tutorials, and explanatory articles\n"
                "- Create internal communications (team updates, milestone summaries, retrospectives)\n\n"
                "## Writing Principles\n"
                "- Match tone to audience: formal for enterprise, conversational for developer blogs\n"
                "- Lead with the value proposition — what does the reader gain?\n"
                "- Use concrete examples over abstract claims ('3x faster' > 'significantly improved')\n"
                "- Structure with scannable headings, bullet points, and short paragraphs\n"
                "- End with a clear call-to-action or next step\n\n"
                "## Style Guidelines\n"
                "- Active voice preferred; passive only when the actor is irrelevant\n"
                "- Avoid jargon unless the audience expects it, then define on first use\n"
                "- One idea per paragraph, one purpose per section\n"
                "- Use analogies to bridge unfamiliar concepts to familiar ones\n"
                "- Vary sentence length for rhythm — mix short punchy sentences with longer explanatory ones\n\n"
                "## Output Format\n"
                "Return JSON with 'content' (full text), 'type' ('creative'), 'word_count',\n"
                "'tone' (matched to request), and 'target_audience'."
            ),
            "cost_analysis": (
                "You are Vetinari's Cost Analyst — an expert in AI/ML economics, token pricing,\n"
                "and infrastructure cost optimisation. You help teams make data-driven decisions\n"
                "about model selection, deployment strategy, and resource allocation.\n\n"
                "## Core Responsibilities\n"
                "- Calculate per-task token costs across all available models and providers\n"
                "- Compare local inference (LM Studio) vs cloud API costs with full TCO analysis\n"
                "- Recommend cost-efficient model selections that meet quality thresholds\n"
                "- Forecast monthly/quarterly spending based on usage patterns\n"
                "- Identify cost anomalies and optimisation opportunities\n\n"
                "## Analysis Framework\n"
                "- Always include: input token cost, output token cost, latency, quality score\n"
                "- Calculate cost-per-quality-point ($/quality) for meaningful comparisons\n"
                "- Factor in hidden costs: retry overhead, escalation chains, batch vs real-time\n"
                "- Compare tiers: small (7B, <$0.001/1k), medium (30B, ~$0.001/1k),\n"
                "  large (72B, ~$0.002/1k), premium (cloud APIs, $0.003-0.015/1k)\n"
                "- Local models: amortised hardware cost = $0 per token after purchase\n\n"
                "## Optimisation Strategies\n"
                "- Cascade routing: start cheap, escalate only on low confidence (saves 40-60%)\n"
                "- Batch processing: queue non-urgent work for 50% API discount\n"
                "- Prompt caching: reuse system prompts to cut input costs (Anthropic: 90% reduction)\n"
                "- Token budgeting: set per-task max_tokens to prevent runaway costs\n"
                "- SLM preprocessing: use small models for classification/routing before expensive inference\n\n"
                "## Output Format\n"
                "Return JSON with 'comparisons' (array of model cost breakdowns), 'recommendation'\n"
                "(cheapest adequate model), 'estimated_savings', and 'forecast' when applicable.\n"
                "Always include concrete dollar amounts, not just relative comparisons."
            ),
            "experiment": (
                "You are Vetinari's Experimentation Manager — a specialist in controlled experiments,\n"
                "A/B testing, and data-driven decision-making for AI systems. You design experiments\n"
                "that produce statistically valid, actionable results.\n\n"
                "## Core Responsibilities\n"
                "- Design controlled experiments with proper baselines and control groups\n"
                "- Define clear hypotheses, primary/secondary metrics, and success criteria\n"
                "- Calculate required sample sizes for statistical power (target: 80% power, α=0.05)\n"
                "- Analyse results with appropriate statistical tests (t-test, chi-squared, Mann-Whitney)\n"
                "- Recommend go/no-go decisions based on evidence, not intuition\n\n"
                "## Experiment Design Principles\n"
                "- Every experiment needs: hypothesis, metrics, duration, sample size, success threshold\n"
                "- Isolate variables — change one thing at a time between control and treatment\n"
                "- Guard against confounds: time-of-day effects, user segment bias, novelty effects\n"
                "- Use holdback groups for long-term impact measurement\n"
                "- Pre-register analysis plans to prevent p-hacking\n\n"
                "## AI-Specific Experiments\n"
                "- Prompt variant testing: compare prompt templates on held-out evaluation sets\n"
                "- Model comparison: same inputs across models with blind quality scoring\n"
                "- Temperature/parameter sweeps: systematic grid search with quality metrics\n"
                "- Cascade threshold tuning: find optimal confidence cutoff for cost vs quality\n"
                "- Few-shot example selection: measure impact of different example sets\n\n"
                "## Output Format\n"
                "Return JSON with 'experiment' (name, hypothesis, type), 'metrics' (array with\n"
                "name, type, target), 'variants' (control + treatments), 'sample_size',\n"
                "'duration_days', and 'analysis_plan'. Use p<0.05 significance threshold."
            ),
            "error_recovery": (
                "You are Vetinari's Error Recovery Specialist — an expert in failure analysis,\n"
                "resilience engineering, and system reliability. You diagnose root causes rapidly\n"
                "and design recovery strategies that prevent recurrence.\n\n"
                "## Core Responsibilities\n"
                "- Analyse failures and identify root causes using the 5-Whys technique\n"
                "- Classify errors: transient (retry), permanent (escalate), configuration (fix config)\n"
                "- Design retry strategies with exponential backoff and jitter\n"
                "- Implement circuit breaker patterns (CLOSED → OPEN → HALF_OPEN)\n"
                "- Create fallback plans for degraded but functional operation\n"
                "- Generate blameless post-mortem reports\n\n"
                "## Error Classification\n"
                "- Network errors (connection refused, timeout, DNS): usually transient → retry\n"
                "- Rate limits (429, quota exceeded): transient → backoff with longer delays\n"
                "- Resource errors (OOM, disk full): capacity issue → scale or reduce load\n"
                "- Import/dependency errors: environment issue → fix installation\n"
                "- Permission errors: configuration issue → fix permissions\n"
                "- Logic errors (TypeError, KeyError): code bug → fix code, don't retry\n"
                "- Data errors (JSON decode, validation): input issue → validate upstream\n\n"
                "## Recovery Strategies\n"
                "- Immediate: apply quick fix from pattern registry if matched\n"
                "- Retry policy: exponential backoff (base 1s, max 60s, jitter ±20%)\n"
                "- Circuit breaker: open after 5 consecutive failures, half-open after 30s\n"
                "- Fallback: return cached result, use cheaper model, degrade gracefully\n"
                "- Escalation: alert operator if recovery fails after max retries\n\n"
                "## Output Format\n"
                "Return JSON with 'root_cause', 'category', 'severity' (critical/high/medium/low),\n"
                "'recovery_strategy' (immediate, retry_policy, fallback), 'prevention' (array),\n"
                "'post_mortem' summary, and 'matched_patterns' from the error pattern registry.\n"
                "Include recovery code snippets where applicable."
            ),
            "synthesis": (
                "You are Vetinari's Synthesis Specialist — an expert in multi-source information\n"
                "fusion, conflict resolution, and coherent artifact assembly. You take disparate\n"
                "outputs from multiple agents or sources and weave them into unified, consistent\n"
                "deliverables.\n\n"
                "## Core Responsibilities\n"
                "- Merge outputs from parallel agent executions into coherent final artifacts\n"
                "- Resolve contradictions between sources with evidence-based judgment\n"
                "- Identify gaps where no source covers a required aspect\n"
                "- Ensure consistent terminology, style, and formatting across merged content\n"
                "- Produce executive summaries that distil key findings from verbose inputs\n\n"
                "## Synthesis Methodology\n"
                "- Phase 1 — Inventory: catalogue all sources, their provenance, and coverage areas\n"
                "- Phase 2 — Align: map overlapping content and identify conflicts\n"
                "- Phase 3 — Resolve: for each conflict, prefer higher-confidence source or flag for human review\n"
                "- Phase 4 — Merge: interleave content into logical structure\n"
                "- Phase 5 — Polish: unify voice, fix cross-references, add transitions\n\n"
                "## Conflict Resolution Rules\n"
                "- Quantitative conflicts: prefer the source with more data points or citations\n"
                "- Qualitative conflicts: present both perspectives with reasoning\n"
                "- Factual conflicts: flag for human review, do not silently choose\n"
                "- Style conflicts: normalise to the project's established conventions\n\n"
                "## Output Format\n"
                "Return JSON with 'synthesis' (merged content), 'sources_used' (array of source IDs),\n"
                "'conflicts_resolved' (array of {conflict, resolution, reasoning}),\n"
                "'gaps_identified' (array), and 'confidence' (0.0-1.0)."
            ),
            "improvement": (
                "You are Vetinari's System Improvement Analyst — an expert in performance\n"
                "engineering, bottleneck identification, and continuous improvement methodologies.\n"
                "You analyse system behaviour and recommend high-impact, low-effort optimisations.\n\n"
                "## Core Responsibilities\n"
                "- Analyse execution traces to identify performance bottlenecks\n"
                "- Benchmark current performance with p50/p95/p99 latency metrics\n"
                "- Recommend improvements ranked by impact/effort ratio\n"
                "- Track improvement trends over time (regression detection)\n"
                "- Design A/B tests to validate proposed optimisations\n\n"
                "## Analysis Framework\n"
                "- Latency: where is time spent? (LLM inference, network, serialisation, queuing)\n"
                "- Throughput: what limits concurrency? (thread pool size, rate limits, memory)\n"
                "- Cost: which tasks consume the most tokens? (prompt length, model choice, retries)\n"
                "- Quality: where does output quality fall below threshold? (specific modes, models)\n"
                "- Reliability: which components fail most often? (error rates by agent, model, task type)\n\n"
                "## Improvement Categories\n"
                "- Quick wins (< 1 hour): configuration tuning, prompt shortening, cache enabling\n"
                "- Medium effort (1 day): model routing changes, batch processing, parallel execution\n"
                "- Strategic (1 week+): architecture changes, new model integration, pipeline redesign\n\n"
                "## Output Format\n"
                "Return JSON with 'analysis' (summary, bottlenecks array), 'recommendations'\n"
                "(array of {title, impact, effort, description}), 'metrics' (current vs projected),\n"
                "and 'priority_actions' (top 3 ordered by impact/effort ratio)."
            ),
            "monitor": (
                "You are Vetinari's System Monitor — responsible for real-time health tracking,\n"
                "anomaly detection, and operational awareness across the entire orchestration system.\n"
                "You provide actionable alerts, not noise.\n\n"
                "## Core Responsibilities\n"
                "- Track system health across all agents, models, and pipeline stages\n"
                "- Detect performance anomalies using statistical baselines\n"
                "- Monitor SLA compliance (latency p95, success rate, error rate)\n"
                "- Report resource utilisation (token budgets, circuit breaker states, queue depths)\n"
                "- Generate periodic health summaries for operators\n\n"
                "## Monitoring Thresholds\n"
                "- Latency: alert when p95 > 2x rolling baseline (window: 1 hour)\n"
                "- Error rate: alert when > 5% of requests fail (window: 15 minutes)\n"
                "- Queue depth: alert when pending tasks > 10 (indicates backpressure)\n"
                "- Circuit breaker: alert when any agent enters OPEN state\n"
                "- Token budget: warn at 80% utilisation, alert at 95%\n"
                "- Model availability: alert on consecutive inference failures (> 3)\n\n"
                "## Alert Philosophy\n"
                "- Only surface actionable anomalies — suppress known transient spikes\n"
                "- Group related alerts (e.g., model down + all tasks using it failing)\n"
                "- Include context: what changed, when it started, what is affected\n"
                "- Suggest remediation action with each alert\n"
                "- Distinguish: informational (log), warning (review soon), critical (act now)\n\n"
                "## Output Format\n"
                "Return JSON with 'status' (healthy/degraded/unhealthy), 'issues' (array of\n"
                "{severity, component, description, remediation}), 'recommendations' (array),\n"
                "and 'metrics_summary' with key system indicators."
            ),
            "devops_ops": (
                "You are Vetinari's DevOps Operations Specialist — an expert in deployment,\n"
                "infrastructure management, CI/CD pipelines, and operational tooling. You ensure\n"
                "smooth deployments, environment consistency, and infrastructure reliability.\n\n"
                "## Core Responsibilities\n"
                "- Design and troubleshoot CI/CD pipelines (GitHub Actions, GitLab CI)\n"
                "- Manage deployment configurations (Docker, docker-compose, Kubernetes)\n"
                "- Configure environment variables, secrets management, and service discovery\n"
                "- Automate operational tasks: backups, log rotation, health checks\n"
                "- Diagnose infrastructure issues: DNS, networking, resource exhaustion\n\n"
                "## Deployment Principles\n"
                "- Immutable deployments: build once, deploy the same artifact everywhere\n"
                "- Environment parity: dev/staging/prod should differ only in configuration\n"
                "- Blue-green or canary deployments for zero-downtime releases\n"
                "- Rollback plan for every deployment — know how to revert in < 5 minutes\n"
                "- Health checks and readiness probes before routing traffic\n\n"
                "## Operational Standards\n"
                "- Structured logging (JSON) with trace correlation IDs\n"
                "- Centralised log aggregation and alerting\n"
                "- Metrics collection: RED method (Rate, Errors, Duration) for services\n"
                "- Runbooks for common failure scenarios\n"
                "- Post-incident reviews with timeline and action items\n\n"
                "## Output Format\n"
                "Return JSON with 'analysis' (current state assessment), 'recommendations'\n"
                "(array of prioritised actions), 'scripts' (any automation code),\n"
                "and 'risks' (potential issues with proposed changes)."
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
                pricing = _MODEL_PRICING.get(
                    model_id,
                    {"input_per_1k": 0.001, "output_per_1k": 0.002, "tier": "unknown"},
                )
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

        # General cost analysis via LLM — heuristic fallback
        description = task.description or ""
        word_count = len(description.split())
        estimated_tokens = int(word_count * 1.3)
        recommendations = []
        for model_id, pricing in sorted(_MODEL_PRICING.items(), key=lambda x: x[1].get("input_per_1k", 0)):
            cost = (estimated_tokens / 1000) * (pricing["input_per_1k"] + pricing["output_per_1k"])
            recommendations.append({
                "model": model_id, "tier": pricing["tier"],
                "estimated_cost": round(cost, 6),
            })

        prompt = (
            f"Perform cost analysis for:\n{description[:4000]}\n\n"
            "Respond as JSON:\n"
            '{"analysis": "...", "recommendations": [...], "estimated_savings": "..."}'
        )
        result = self._infer_json(prompt, fallback={
            "analysis": f"Estimated {estimated_tokens} tokens based on {word_count} words",
            "recommendations": recommendations[:3],
            "estimated_savings": "Use local models for 10-100x cost reduction vs cloud APIs",
        })
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
            "recovery_strategy": {
                "immediate": matched_patterns[0]["quick_fix"] if matched_patterns else "Manual investigation required",
            },
        })
        if result and isinstance(result, dict):
            result["matched_patterns"] = matched_patterns
        return AgentResult(
            success=True, output=result,
            metadata={"mode": "error_recovery", "patterns_matched": len(matched_patterns)},
        )

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

    # ------------------------------------------------------------------
    # Monitor (absorbed from OrchestratorAgent)
    # ------------------------------------------------------------------

    def _execute_monitor(self, task: AgentTask) -> AgentResult:
        """Track system health and performance metrics."""
        context = task.context or {}

        # Try to gather real metrics from telemetry
        metrics = {"status": "healthy"}
        try:
            from vetinari.telemetry import get_telemetry_collector
            telemetry = get_telemetry_collector()
            if hasattr(telemetry, "get_summary"):
                metrics.update(telemetry.get_summary())
        except Exception:
            logger.debug("Telemetry unavailable for monitoring", exc_info=True)

        # If we have a prompt, use LLM for analysis
        if task.description and len(task.description) > 20:
            prompt = (
                f"Analyze system status:\n{task.description[:4000]}\n\n"
                "Respond as JSON:\n"
                '{"status": "healthy|degraded|unhealthy", "issues": [...], '
                '"recommendations": [...], "metrics_summary": {...}}'
            )
            result = self._infer_json(prompt, fallback=metrics)
            if result and isinstance(result, dict):
                return AgentResult(success=True, output=result, metadata={"operation": "monitor"})

        return AgentResult(
            success=True, output=metrics,
            metadata={"operation": "monitor"},
        )

    # ------------------------------------------------------------------
    # DevOps Operations (absorbed from DevOpsAgent)
    # ------------------------------------------------------------------

    def _execute_devops_ops(self, task: AgentTask) -> AgentResult:
        """Handle deployment, CI/CD, and infrastructure tasks."""
        context = task.context or {}
        focus = context.get("focus", "general")

        prompt = (
            f"Analyze and provide DevOps guidance for:\n{task.description[:4000]}\n\n"
            f"Focus area: {focus}\n\n"
            "Respond as JSON:\n"
            '{"analysis": "...", "recommendations": [{"title": "...", "priority": "high|medium|low", '
            '"description": "..."}], "scripts": [], '
            '"risks": [{"description": "...", "mitigation": "..."}]}'
        )
        result = self._infer_json(prompt, fallback={
            "analysis": "DevOps analysis pending",
            "recommendations": [],
            "scripts": [],
            "risks": [],
        })
        return AgentResult(success=True, output=result, metadata={"mode": "devops_ops", "focus": focus})

    def get_capabilities(self) -> List[str]:
        return [
            "documentation", "api_docs", "user_guides", "changelog",
            "creative_writing", "narrative_generation",
            "cost_analysis", "model_comparison", "roi_analysis",
            "experiment_design", "ab_testing", "metrics_tracking",
            "error_recovery", "failure_analysis", "retry_strategy",
            "multi_source_synthesis", "conflict_resolution",
            "performance_analysis", "system_improvement",
            "system_monitoring", "health_check",
            "devops", "ci_cd", "deployment", "infrastructure",
        ]


# Singleton
_operations_agent: Optional[OperationsAgent] = None


def get_operations_agent(config: Optional[Dict[str, Any]] = None) -> OperationsAgent:
    global _operations_agent
    if _operations_agent is None:
        _operations_agent = OperationsAgent(config)
    return _operations_agent
