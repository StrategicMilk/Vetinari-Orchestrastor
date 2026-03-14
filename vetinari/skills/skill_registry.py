"""Programmatic Skill Registry.

==============================
Typed, validated registry of all Vetinari skills.

Maps the 8 consolidated agent roles to their skill specifications.
Each entry uses ``SkillSpec`` to declare capabilities, schemas, constraints,
and quality standards.

Currently references the *future* consolidated agent types (PLANNER,
ORCHESTRATOR, RESEARCHER, ORACLE, BUILDER, ARCHITECT, QUALITY, OPERATIONS).
During the transition period (before Phase 3 agent consolidation), the
existing 22 agents can look up skills via ``get_skill_for_agent_type()``
which maps legacy types to consolidated skill specs.

Usage::

    from vetinari.skills.skill_registry import SKILL_REGISTRY, get_skill, validate_all

    spec = get_skill("builder")
    errors = validate_all()     # [] if all specs are valid
"""

from __future__ import annotations

import logging

from vetinari.skills.skill_spec import (
    SkillConstraint,
    SkillGuideline,
    SkillSpec,
    SkillStandard,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Shared / Cross-Cutting Standards
# ═══════════════════════════════════════════════════════════════════════════

_UNIVERSAL_STANDARDS = [
    SkillStandard(
        id="STD-UNI-001",
        category="output_format",
        rule="All skill outputs MUST conform to the declared output_schema",
        severity="error",
        check_hint="jsonschema.validate(output, spec.output_schema)",
    ),
    SkillStandard(
        id="STD-UNI-002",
        category="error_handling",
        rule="All skill executions MUST return a ToolResult; exceptions MUST be caught and reported via error field",
        severity="error",
        check_hint="assert isinstance(result, ToolResult)",
    ),
    SkillStandard(
        id="STD-UNI-003",
        category="logging",
        rule="All skill executions MUST log entry and exit at INFO level with timing",
        severity="warning",
        check_hint="logger.info() at start and end",
    ),
    SkillStandard(
        id="STD-UNI-004",
        category="idempotency",
        rule="Skill executions SHOULD be idempotent — running twice with same input produces same output",
        severity="warning",
    ),
]

_UNIVERSAL_CONSTRAINTS = [
    SkillConstraint(
        id="CON-UNI-001",
        category="safety",
        description="No skill may execute arbitrary code from untrusted input without sandbox",
        limit="sandbox_required_for_untrusted=true",
        enforcement="hard",
    ),
    SkillConstraint(
        id="CON-UNI-002",
        category="resource",
        description="All skills must respect their max_tokens and timeout_seconds limits",
        limit="enforced_at_runtime",
        enforcement="hard",
    ),
    SkillConstraint(
        id="CON-UNI-003",
        category="resource",
        description="All skills must respect their max_cost_usd budget per invocation",
        limit="enforced_at_runtime",
        enforcement="hard",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

SKILL_REGISTRY: dict[str, SkillSpec] = {
    # ───────────────────────────────────────────────────────────────────
    # PLANNER
    # ───────────────────────────────────────────────────────────────────
    "planner": SkillSpec(
        skill_id="planner",
        name="Planning",
        version="1.1.0",
        description="Goal decomposition, task scheduling, dependency mapping, specification crystallization",
        agent_type="PLANNER",
        modes=["decompose", "schedule", "specification"],
        capabilities=["goal_decomposition", "task_scheduling", "dependency_mapping", "specification"],
        input_schema={
            "type": "object",
            "required": ["goal"],
            "properties": {
                "goal": {"type": "string", "minLength": 1, "description": "The high-level goal to decompose"},
                "constraints": {"type": "array", "items": {"type": "string"}, "description": "Constraints to respect"},
                "context": {"type": "object", "description": "Additional context (codebase state, prior plans)"},
                "max_depth": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
            },
        },
        output_schema={
            "type": "object",
            "required": ["plan", "tasks"],
            "properties": {
                "plan": {
                    "type": "object",
                    "required": ["goal", "strategy", "phases"],
                    "properties": {
                        "goal": {"type": "string"},
                        "strategy": {"type": "string"},
                        "phases": {"type": "array", "items": {"type": "object"}},
                        "estimated_effort": {"type": "string"},
                        "risks": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "description", "agent_type"],
                        "properties": {
                            "id": {"type": "string"},
                            "description": {"type": "string"},
                            "agent_type": {"type": "string"},
                            "dependencies": {"type": "array", "items": {"type": "string"}},
                            "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                        },
                    },
                },
                "dependency_graph": {"type": "object", "description": "Adjacency list of task dependencies"},
            },
        },
        max_tokens=8192,
        timeout_seconds=180,
        max_cost_usd=1.00,
        requires_tools=["GeneratePlan"],
        min_verification_score=0.6,
        tags=["planning", "decomposition", "scheduling"],
        standards=[
            *_UNIVERSAL_STANDARDS,
            SkillStandard(
                id="STD-PLN-001",
                category="planning",
                rule="Every generated task MUST have a unique ID and at least one assigned agent_type",
                severity="error",
                check_hint="all(t.get('id') and t.get('agent_type') for t in tasks)",
            ),
            SkillStandard(
                id="STD-PLN-002",
                category="planning",
                rule="Dependency graphs MUST be acyclic (DAG); circular dependencies are forbidden",
                severity="error",
                check_hint="topological_sort(dependency_graph) succeeds",
            ),
            SkillStandard(
                id="STD-PLN-003",
                category="planning",
                rule="Plans MUST include at least one verification/testing task per implementation task",
                severity="warning",
                check_hint="count(verify_tasks) >= count(impl_tasks)",
            ),
            SkillStandard(
                id="STD-PLN-004",
                category="completeness",
                rule="Plans MUST include risk assessment and rollback strategy for destructive operations",
                severity="warning",
            ),
        ],
        guidelines=[
            SkillGuideline(
                id="GDL-PLN-001",
                category="usage",
                recommendation="Prefer depth-first decomposition for complex goals; limit to 3 levels for simple goals",
                rationale="Deep decomposition increases precision but adds overhead",
            ),
            SkillGuideline(
                id="GDL-PLN-002",
                category="integration",
                recommendation="Schedule I/O-bound tasks in parallel; sequence CPU-bound tasks to avoid resource contention",
                rationale="Maximizes throughput without overloading the system",
            ),
            SkillGuideline(
                id="GDL-PLN-003",
                category="output_format",
                recommendation="Include effort estimates (S/M/L/XL) on each task for capacity planning",
                rationale="Helps orchestrator allocate resources and set expectations",
            ),
        ],
        constraints=[
            *_UNIVERSAL_CONSTRAINTS,
            SkillConstraint(
                id="CON-PLN-001",
                category="scope",
                description="Maximum number of tasks in a single plan",
                limit="50 tasks",
                enforcement="hard",
            ),
            SkillConstraint(
                id="CON-PLN-002",
                category="scope",
                description="Maximum decomposition depth",
                limit="5 levels",
                enforcement="hard",
            ),
            SkillConstraint(
                id="CON-PLN-003",
                category="safety",
                description="Plans involving file deletion or destructive ops require explicit confirmation",
                limit="confirmation_required=true",
                enforcement="hard",
            ),
        ],
    ),
    # ───────────────────────────────────────────────────────────────────
    # ORCHESTRATOR
    # ───────────────────────────────────────────────────────────────────
    "orchestrator": SkillSpec(
        skill_id="orchestrator",
        name="Orchestration",
        version="1.1.0",
        description="User interaction, Socratic clarification, system monitoring, memory consolidation",
        agent_type="ORCHESTRATOR",
        modes=["clarify", "monitor", "consolidate"],
        capabilities=["clarification", "ambiguity_detection", "performance_monitoring", "memory_consolidation"],
        input_schema={
            "type": "object",
            "required": ["context"],
            "properties": {
                "context": {
                    "type": "object",
                    "description": "Current conversation/execution context",
                    "properties": {
                        "user_message": {"type": "string"},
                        "history": {"type": "array", "items": {"type": "object"}},
                        "active_tasks": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "mode": {"type": "string", "enum": ["clarify", "monitor", "consolidate"]},
            },
        },
        output_schema={
            "type": "object",
            "required": ["action", "result"],
            "properties": {
                "action": {"type": "string", "enum": ["clarify", "delegate", "monitor", "consolidate", "respond"]},
                "result": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "questions": {"type": "array", "items": {"type": "string"}},
                        "delegations": {"type": "array", "items": {"type": "object"}},
                        "metrics": {"type": "object"},
                    },
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        max_tokens=4096,
        timeout_seconds=60,
        max_cost_usd=0.30,
        requires_tools=["MemoryRecall", "MemoryRemember"],
        min_verification_score=0.5,
        tags=["orchestration", "clarification", "monitoring"],
        standards=[
            *_UNIVERSAL_STANDARDS,
            SkillStandard(
                id="STD-ORC-001",
                category="interaction",
                rule="Ambiguity score > 0.7 MUST trigger a clarification question before proceeding",
                severity="error",
                check_hint="if ambiguity > 0.7: assert action == 'clarify'",
            ),
            SkillStandard(
                id="STD-ORC-002",
                category="interaction",
                rule="Clarification questions MUST be specific, closed-ended, and actionable",
                severity="warning",
            ),
            SkillStandard(
                id="STD-ORC-003",
                category="monitoring",
                rule="Monitor mode MUST report task completion percentage and blocked tasks",
                severity="warning",
            ),
        ],
        guidelines=[
            SkillGuideline(
                id="GDL-ORC-001",
                category="usage",
                recommendation="Limit clarification to 3 questions maximum per interaction to avoid user fatigue",
                rationale="Too many questions reduce user engagement and trust",
            ),
            SkillGuideline(
                id="GDL-ORC-002",
                category="integration",
                recommendation="Consolidate memory after every 5 completed tasks to prevent context bloat",
                rationale="Keeps working memory focused and reduces token usage",
            ),
        ],
        constraints=[
            *_UNIVERSAL_CONSTRAINTS,
            SkillConstraint(
                id="CON-ORC-001",
                category="scope",
                description="Maximum clarification rounds before requiring human escalation",
                limit="5 rounds",
                enforcement="hard",
            ),
            SkillConstraint(
                id="CON-ORC-002",
                category="safety",
                description="Never execute destructive actions without explicit user confirmation",
                limit="confirmation_required=true",
                enforcement="hard",
            ),
        ],
    ),
    # ───────────────────────────────────────────────────────────────────
    # RESEARCHER
    # ───────────────────────────────────────────────────────────────────
    "researcher": SkillSpec(
        skill_id="researcher",
        name="Research",
        version="1.1.0",
        description="Code discovery, API/library research, domain research, lateral thinking",
        agent_type="RESEARCHER",
        modes=["code_discovery", "api_lookup", "domain_research", "lateral_thinking"],
        capabilities=["code_discovery", "api_lookup", "domain_research", "lateral_thinking"],
        input_schema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 1, "description": "Research question or search query"},
                "scope": {"type": "string", "enum": ["codebase", "web", "documentation", "all"], "default": "all"},
                "depth": {"type": "string", "enum": ["shallow", "medium", "deep"], "default": "medium"},
                "max_sources": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            },
        },
        output_schema={
            "type": "object",
            "required": ["findings"],
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["title", "content", "source"],
                        "properties": {
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "source": {"type": "string"},
                            "relevance_score": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                    },
                },
                "sources": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "knowledge_gaps": {"type": "array", "items": {"type": "string"}},
            },
        },
        max_tokens=8192,
        timeout_seconds=180,
        max_cost_usd=0.75,
        requires_tools=["WebSearch", "FileOperations"],
        min_verification_score=0.5,
        tags=["research", "discovery", "analysis"],
        standards=[
            *_UNIVERSAL_STANDARDS,
            SkillStandard(
                id="STD-RES-001",
                category="accuracy",
                rule="Every finding MUST cite its source (file path, URL, or document reference)",
                severity="error",
                check_hint="all(f.get('source') for f in findings)",
            ),
            SkillStandard(
                id="STD-RES-002",
                category="accuracy",
                rule="Findings from web sources MUST include a confidence score and recency indicator",
                severity="warning",
            ),
            SkillStandard(
                id="STD-RES-003",
                category="completeness",
                rule="Research MUST identify and report knowledge gaps — areas where information was insufficient",
                severity="warning",
                check_hint="'knowledge_gaps' in output",
            ),
            SkillStandard(
                id="STD-RES-004",
                category="accuracy",
                rule="Conflicting sources MUST be flagged with both perspectives presented",
                severity="error",
            ),
        ],
        guidelines=[
            SkillGuideline(
                id="GDL-RES-001",
                category="usage",
                recommendation="Start with codebase search before web search to prefer local knowledge",
                rationale="Local knowledge is more relevant and costs less to retrieve",
            ),
            SkillGuideline(
                id="GDL-RES-002",
                category="performance",
                recommendation="Use shallow depth for quick lookups, deep depth only for novel/complex topics",
                rationale="Deep research is expensive; shallow search resolves 80% of queries",
            ),
            SkillGuideline(
                id="GDL-RES-003",
                category="output_format",
                recommendation="Order findings by relevance_score descending for consumer convenience",
                rationale="Consumers can truncate at their confidence threshold",
            ),
        ],
        constraints=[
            *_UNIVERSAL_CONSTRAINTS,
            SkillConstraint(
                id="CON-RES-001",
                category="resource",
                description="Maximum number of web search queries per invocation",
                limit="10 queries",
                enforcement="hard",
            ),
            SkillConstraint(
                id="CON-RES-002",
                category="scope",
                description="Codebase search must not traverse outside the project root",
                limit="project_root_only=true",
                enforcement="hard",
            ),
            SkillConstraint(
                id="CON-RES-003",
                category="safety",
                description="Never include credentials, tokens, or secrets found during research in output",
                limit="filter_secrets=true",
                enforcement="hard",
            ),
        ],
    ),
    # ───────────────────────────────────────────────────────────────────
    # ORACLE
    # ───────────────────────────────────────────────────────────────────
    "oracle": SkillSpec(
        skill_id="oracle",
        name="Oracle",
        version="1.1.0",
        description="Architecture decisions, risk assessment, ontological analysis, assumption challenging",
        agent_type="ORACLE",
        modes=["architecture", "risk_assessment", "ontological_analysis", "contrarian_review"],
        capabilities=["architecture_decision", "risk_assessment", "ontological_analysis", "contrarian_review"],
        input_schema={
            "type": "object",
            "required": ["subject"],
            "properties": {
                "subject": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Subject to analyze or decision to evaluate",
                },
                "context": {
                    "type": "object",
                    "properties": {
                        "codebase_state": {"type": "string"},
                        "constraints": {"type": "array", "items": {"type": "string"}},
                        "prior_decisions": {"type": "array", "items": {"type": "object"}},
                    },
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Decision options to evaluate",
                },
            },
        },
        output_schema={
            "type": "object",
            "required": ["analysis", "recommendations"],
            "properties": {
                "analysis": {
                    "type": "object",
                    "required": ["summary", "trade_offs"],
                    "properties": {
                        "summary": {"type": "string"},
                        "trade_offs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "option": {"type": "string"},
                                    "pros": {"type": "array", "items": {"type": "string"}},
                                    "cons": {"type": "array", "items": {"type": "string"}},
                                    "risk_level": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                                },
                            },
                        },
                        "assumptions": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["action", "rationale"],
                        "properties": {
                            "action": {"type": "string"},
                            "rationale": {"type": "string"},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                    },
                },
                "risk_score": {"type": "number", "minimum": 0, "maximum": 1},
                "contrarian_view": {"type": "string", "description": "Devil's advocate perspective"},
            },
        },
        max_tokens=8192,
        timeout_seconds=180,
        max_cost_usd=1.00,
        min_verification_score=0.6,
        tags=["architecture", "risk", "analysis", "decision"],
        standards=[
            *_UNIVERSAL_STANDARDS,
            SkillStandard(
                id="STD-ORA-001",
                category="analysis",
                rule="Every recommendation MUST include a rationale and confidence score",
                severity="error",
                check_hint="all(r.get('rationale') and r.get('confidence') for r in recommendations)",
            ),
            SkillStandard(
                id="STD-ORA-002",
                category="analysis",
                rule="Trade-off analysis MUST present at least 2 options with pros/cons for each",
                severity="error",
                check_hint="len(trade_offs) >= 2",
            ),
            SkillStandard(
                id="STD-ORA-003",
                category="analysis",
                rule="High-risk recommendations (risk_score > 0.7) MUST include mitigation strategies",
                severity="error",
            ),
            SkillStandard(
                id="STD-ORA-004",
                category="completeness",
                rule="Analysis MUST explicitly list assumptions that could invalidate the recommendation",
                severity="warning",
                check_hint="len(analysis.get('assumptions', [])) > 0",
            ),
            SkillStandard(
                id="STD-ORA-005",
                category="objectivity",
                rule="Contrarian review mode MUST provide genuine counter-arguments, not strawmen",
                severity="warning",
            ),
        ],
        guidelines=[
            SkillGuideline(
                id="GDL-ORA-001",
                category="usage",
                recommendation="Use ontological_analysis mode for fundamental 'what should this be?' questions; architecture mode for 'how should we build it?' questions",
                rationale="Ontological analysis explores the problem space; architecture explores the solution space",
            ),
            SkillGuideline(
                id="GDL-ORA-002",
                category="integration",
                recommendation="Run contrarian_review on any recommendation with confidence < 0.8 before acting",
                rationale="Low-confidence decisions benefit from adversarial examination",
            ),
            SkillGuideline(
                id="GDL-ORA-003",
                category="output_format",
                recommendation="Include reversibility assessment for each option: easy/hard/impossible to reverse",
                rationale="Irreversible decisions warrant more scrutiny",
            ),
        ],
        constraints=[
            *_UNIVERSAL_CONSTRAINTS,
            SkillConstraint(
                id="CON-ORA-001",
                category="scope",
                description="Oracle provides advisory analysis only; it must not execute implementation",
                limit="read_only=true",
                enforcement="hard",
            ),
            SkillConstraint(
                id="CON-ORA-002",
                category="safety",
                description="Risk scores must use calibrated scale; never report 0.0 risk on non-trivial decisions",
                limit="min_risk=0.05 for non-trivial decisions",
                enforcement="soft",
            ),
        ],
    ),
    # ───────────────────────────────────────────────────────────────────
    # BUILDER
    # ───────────────────────────────────────────────────────────────────
    "builder": SkillSpec(
        skill_id="builder",
        name="Builder",
        version="1.1.0",
        description="Code implementation, scaffolding, refactoring, test writing, debugging",
        agent_type="BUILDER",
        modes=[
            "feature_implementation",
            "refactoring",
            "test_writing",
            "error_handling",
            "code_generation",
            "debugging",
        ],
        capabilities=[
            "feature_implementation",
            "refactoring",
            "test_writing",
            "error_handling",
            "code_generation",
            "debugging",
        ],
        input_schema={
            "type": "object",
            "required": ["task"],
            "properties": {
                "task": {"type": "string", "minLength": 1, "description": "What to build, refactor, or fix"},
                "language": {"type": "string", "description": "Target programming language"},
                "files": {"type": "array", "items": {"type": "string"}, "description": "Files to create or modify"},
                "context": {"type": "string", "description": "Existing code or codebase context"},
                "requirements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific requirements or acceptance criteria",
                },
                "thinking_mode": {"type": "string", "enum": ["low", "medium", "high", "xhigh"], "default": "medium"},
            },
        },
        output_schema={
            "type": "object",
            "required": ["code", "language"],
            "properties": {
                "code": {"type": "string", "description": "Generated or modified code"},
                "language": {"type": "string"},
                "file_path": {"type": "string"},
                "files_affected": {"type": "array", "items": {"type": "string"}},
                "tests": {"type": "string", "description": "Associated test code"},
                "tests_added": {"type": "integer", "minimum": 0},
                "explanation": {"type": "string", "description": "What was done and why"},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
        },
        max_tokens=8192,
        timeout_seconds=180,
        max_cost_usd=1.50,
        requires_tools=["FileOperations", "GitTool"],
        min_verification_score=0.6,
        forbidden_patterns=[
            "eval(",
            "exec(",
            "__import__(",
            "os.system(",
            "subprocess.call(",
            "# TODO(#pending)",  # noqa: VET030  (blocklist pattern string, not a live TODO)
            "# HACK(#pending)",  # noqa: VET030
            "# FIXME(#pending)",  # noqa: VET030
        ],
        tags=["code", "implementation", "building", "refactoring", "testing"],
        standards=[
            *_UNIVERSAL_STANDARDS,
            SkillStandard(
                id="STD-BLD-001",
                category="code_quality",
                rule="All generated code MUST include type hints for function signatures (Python) or explicit types (TypeScript)",
                severity="error",
                check_hint="check_type_annotations(code)",
            ),
            SkillStandard(
                id="STD-BLD-002",
                category="code_quality",
                rule="All public functions and classes MUST have docstrings/JSDoc with parameter descriptions",
                severity="error",
                check_hint="check_docstrings(code)",
            ),
            SkillStandard(
                id="STD-BLD-003",
                category="testing",
                rule="Feature implementations (thinking_mode >= medium) MUST include at least one unit test",
                severity="error",
                check_hint="tests_added >= 1 when thinking_mode != 'low'",
            ),
            SkillStandard(
                id="STD-BLD-004",
                category="security",
                rule="Generated code MUST NOT contain hardcoded secrets, credentials, or API keys",
                severity="error",
                check_hint="no_secrets_in_code(code)",
            ),
            SkillStandard(
                id="STD-BLD-005",
                category="code_quality",
                rule="Functions MUST NOT exceed 50 lines; classes MUST NOT exceed 300 lines",
                severity="warning",
                check_hint="max_function_length(code) <= 50",
            ),
            SkillStandard(
                id="STD-BLD-006",
                category="code_quality",
                rule="No wildcard imports (from x import *); all imports must be explicit",
                severity="error",
                check_hint="'import *' not in code",
            ),
            SkillStandard(
                id="STD-BLD-007",
                category="error_handling",
                rule="All I/O operations MUST be wrapped in try/except with specific exception types",
                severity="warning",
                check_hint="no_bare_except(code)",
            ),
            SkillStandard(
                id="STD-BLD-008",
                category="code_quality",
                rule="Generated code MUST follow the project's existing naming conventions and style",
                severity="warning",
            ),
        ],
        guidelines=[
            SkillGuideline(
                id="GDL-BLD-001",
                category="usage",
                recommendation="Use thinking_mode='low' for boilerplate/scaffolding, 'high'/'xhigh' for business logic",
                rationale="Right-sizing thinking mode balances quality against cost and latency",
            ),
            SkillGuideline(
                id="GDL-BLD-002",
                category="integration",
                recommendation="Always provide existing code as context when refactoring to preserve conventions",
                rationale="Without context, generated code may conflict with existing patterns",
            ),
            SkillGuideline(
                id="GDL-BLD-003",
                category="output_format",
                recommendation="Include an 'explanation' field describing the approach and design decisions",
                rationale="Helps reviewers understand intent, not just the code",
            ),
            SkillGuideline(
                id="GDL-BLD-004",
                category="performance",
                recommendation="Prefer composition over inheritance; use dependency injection for testability",
                rationale="Leads to more modular, testable, and maintainable code",
            ),
        ],
        constraints=[
            *_UNIVERSAL_CONSTRAINTS,
            SkillConstraint(
                id="CON-BLD-001",
                category="scope",
                description="Maximum files modified in a single invocation",
                limit="10 files",
                enforcement="hard",
            ),
            SkillConstraint(
                id="CON-BLD-002",
                category="scope",
                description="Maximum lines of code generated in a single invocation",
                limit="500 lines",
                enforcement="soft",
            ),
            SkillConstraint(
                id="CON-BLD-003",
                category="safety",
                description="Code generation must not include system-level commands or shell execution",
                limit="no_shell_exec=true",
                enforcement="hard",
            ),
            SkillConstraint(
                id="CON-BLD-004",
                category="safety",
                description="Refactoring must preserve all existing public API signatures unless explicitly requested",
                limit="preserve_public_api=true",
                enforcement="hard",
            ),
        ],
    ),
    # ───────────────────────────────────────────────────────────────────
    # ARCHITECT
    # ───────────────────────────────────────────────────────────────────
    "architect": SkillSpec(
        skill_id="architect",
        name="Architect",
        version="1.1.0",
        description="UI/UX design, database schema, DevOps pipelines, git workflow, system design",
        agent_type="ARCHITECT",
        modes=["ui_design", "database", "devops", "git_workflow", "system_design", "api_design"],
        capabilities=[
            "ui_design",
            "database",
            "devops",
            "git_workflow",
            "system_design",
            "api_design",
        ],
        input_schema={
            "type": "object",
            "required": ["design_request"],
            "properties": {
                "design_request": {"type": "string", "minLength": 1, "description": "What to design or architect"},
                "domain": {"type": "string", "description": "Domain context (web, mobile, backend, infra)"},
                "constraints": {"type": "array", "items": {"type": "string"}, "description": "Technical constraints"},
                "existing_architecture": {"type": "object", "description": "Current system state if evolving"},
            },
        },
        output_schema={
            "type": "object",
            "required": ["design", "components"],
            "properties": {
                "design": {
                    "type": "object",
                    "required": ["summary", "architecture_pattern"],
                    "properties": {
                        "summary": {"type": "string"},
                        "architecture_pattern": {"type": "string"},
                        "rationale": {"type": "string"},
                        "alternatives_considered": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "responsibility"],
                        "properties": {
                            "name": {"type": "string"},
                            "responsibility": {"type": "string"},
                            "interfaces": {"type": "array", "items": {"type": "string"}},
                            "dependencies": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "diagrams": {"type": "array", "items": {"type": "object"}},
                "migration_plan": {
                    "type": "object",
                    "description": "Steps to migrate from current to target architecture",
                },
            },
        },
        max_tokens=8192,
        timeout_seconds=240,
        max_cost_usd=1.00,
        requires_tools=["FileOperations", "GitTool"],
        min_verification_score=0.6,
        tags=["architecture", "design", "infrastructure", "schema", "ui"],
        standards=[
            *_UNIVERSAL_STANDARDS,
            SkillStandard(
                id="STD-ARC-001",
                category="design",
                rule="Every design MUST state the architecture pattern used and why it was chosen",
                severity="error",
                check_hint="design.get('architecture_pattern') and design.get('rationale')",
            ),
            SkillStandard(
                id="STD-ARC-002",
                category="design",
                rule="Component designs MUST define clear boundaries: responsibility, interfaces, and dependencies",
                severity="error",
            ),
            SkillStandard(
                id="STD-ARC-003",
                category="design",
                rule="Database schemas MUST include index strategy, constraint definitions, and migration path",
                severity="error",
            ),
            SkillStandard(
                id="STD-ARC-004",
                category="security",
                rule="API designs MUST include authentication, authorization, and rate limiting considerations",
                severity="error",
            ),
            SkillStandard(
                id="STD-ARC-005",
                category="design",
                rule="DevOps pipelines MUST include rollback procedures and health check definitions",
                severity="warning",
            ),
            SkillStandard(
                id="STD-ARC-006",
                category="completeness",
                rule="Designs MUST list alternatives_considered with reasons for rejection",
                severity="warning",
                check_hint="len(design.get('alternatives_considered', [])) > 0",
            ),
        ],
        guidelines=[
            SkillGuideline(
                id="GDL-ARC-001",
                category="usage",
                recommendation="Use system_design mode for greenfield projects; use domain-specific modes for targeted changes",
                rationale="System design provides holistic view; domain modes are more focused and efficient",
            ),
            SkillGuideline(
                id="GDL-ARC-002",
                category="integration",
                recommendation="Include migration_plan when modifying existing architecture to ensure safe transitions",
                rationale="Architecture changes without migration plans risk production outages",
            ),
            SkillGuideline(
                id="GDL-ARC-003",
                category="output_format",
                recommendation="Use C4 model notation (Context, Container, Component, Code) for diagrams when possible",
                rationale="C4 is widely understood and provides consistent abstraction levels",
            ),
        ],
        constraints=[
            *_UNIVERSAL_CONSTRAINTS,
            SkillConstraint(
                id="CON-ARC-001",
                category="scope",
                description="Architect skill is advisory only; it must not modify code or infrastructure directly",
                limit="read_only=true",
                enforcement="hard",
            ),
            SkillConstraint(
                id="CON-ARC-002",
                category="scope",
                description="Maximum components in a single design",
                limit="20 components",
                enforcement="soft",
            ),
            SkillConstraint(
                id="CON-ARC-003",
                category="safety",
                description="Database schema changes must flag potential data loss operations",
                limit="flag_data_loss=true",
                enforcement="hard",
            ),
        ],
    ),
    # ───────────────────────────────────────────────────────────────────
    # QUALITY
    # ───────────────────────────────────────────────────────────────────
    "quality": SkillSpec(
        skill_id="quality",
        name="Quality",
        version="1.1.0",
        description="Code review, test generation, security audit, simplification, performance review",
        agent_type="QUALITY",
        modes=[
            "code_review",
            "test_generation",
            "security_audit",
            "simplification",
            "performance_review",
            "best_practices",
        ],
        capabilities=[
            "code_review",
            "test_generation",
            "security_audit",
            "simplification",
            "performance_review",
            "best_practices",
        ],
        input_schema={
            "type": "object",
            "required": ["code"],
            "properties": {
                "code": {"type": "string", "minLength": 1, "description": "Code to review, test, or audit"},
                "review_type": {
                    "type": "string",
                    "enum": [
                        "code_review",
                        "security_audit",
                        "performance_review",
                        "test_generation",
                        "simplification",
                        "best_practices",
                    ],
                },
                "context": {"type": "string", "description": "File path, PR description, or surrounding context"},
                "focus_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific areas to prioritize",
                },
                "severity_threshold": {
                    "type": "string",
                    "enum": ["critical", "high", "medium", "low", "info"],
                    "default": "low",
                },
            },
        },
        output_schema={
            "type": "object",
            "required": ["issues", "score", "summary"],
            "properties": {
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["title", "severity", "description"],
                        "properties": {
                            "title": {"type": "string"},
                            "severity": {"type": "string", "enum": ["critical", "high", "medium", "low", "info"]},
                            "location": {"type": "string"},
                            "description": {"type": "string"},
                            "suggestion": {"type": "string"},
                            "cwe_id": {"type": "string", "description": "CWE identifier for security issues"},
                        },
                    },
                },
                "score": {"type": "number", "minimum": 0, "maximum": 100},
                "quality_grade": {"type": "string", "enum": ["A", "B", "C", "D", "F"]},
                "summary": {"type": "string"},
                "tests": {"type": "string", "description": "Generated test code"},
                "recommendations": {"type": "array", "items": {"type": "string"}},
                "metrics": {
                    "type": "object",
                    "properties": {
                        "cyclomatic_complexity": {"type": "number"},
                        "lines_of_code": {"type": "integer"},
                        "test_coverage_estimate": {"type": "number"},
                        "security_issues_count": {"type": "integer"},
                    },
                },
            },
        },
        max_tokens=8192,
        timeout_seconds=180,
        max_cost_usd=1.00,
        requires_tools=["FileOperations"],
        min_verification_score=0.7,
        forbidden_patterns=[
            "eval(",
            "exec(",
            "__import__(",
        ],
        tags=["quality", "review", "testing", "security", "audit", "performance"],
        standards=[
            *_UNIVERSAL_STANDARDS,
            SkillStandard(
                id="STD-QUA-001",
                category="security",
                rule="Security audits MUST check for OWASP Top 10 vulnerability categories",
                severity="error",
                check_hint="check_owasp_coverage(audit_result)",
            ),
            SkillStandard(
                id="STD-QUA-002",
                category="security",
                rule="Security issues MUST include CWE identifiers where applicable",
                severity="warning",
                check_hint="security_issues have cwe_id",
            ),
            SkillStandard(
                id="STD-QUA-003",
                category="testing",
                rule="Generated tests MUST follow Arrange-Act-Assert pattern with descriptive names",
                severity="error",
            ),
            SkillStandard(
                id="STD-QUA-004",
                category="testing",
                rule="Test generation MUST cover happy path, edge cases, and error cases",
                severity="error",
                check_hint="test_categories include happy_path, edge_case, error_case",
            ),
            SkillStandard(
                id="STD-QUA-005",
                category="code_quality",
                rule="Code review MUST assess: correctness, readability, maintainability, and performance",
                severity="error",
            ),
            SkillStandard(
                id="STD-QUA-006",
                category="code_quality",
                rule="All issues MUST have actionable suggestions, not just problem descriptions",
                severity="warning",
                check_hint="all(issue.get('suggestion') for issue in issues)",
            ),
            SkillStandard(
                id="STD-QUA-007",
                category="security",
                rule="Hardcoded credentials, secrets, and tokens MUST be flagged as CRITICAL severity",
                severity="error",
            ),
            SkillStandard(
                id="STD-QUA-008",
                category="performance",
                rule="Performance reviews MUST identify O(n^2+) algorithms and suggest improvements",
                severity="warning",
            ),
        ],
        guidelines=[
            SkillGuideline(
                id="GDL-QUA-001",
                category="usage",
                recommendation="Run security_audit mode on any code that handles user input, authentication, or file I/O",
                rationale="These are the highest-risk areas for security vulnerabilities",
            ),
            SkillGuideline(
                id="GDL-QUA-002",
                category="integration",
                recommendation="Use severity_threshold='high' for PR reviews; 'low' for comprehensive audits",
                rationale="PR reviews need focus on blocking issues; audits need full visibility",
            ),
            SkillGuideline(
                id="GDL-QUA-003",
                category="output_format",
                recommendation="Group issues by file and severity for easy triage by developers",
                rationale="Grouped output reduces cognitive load for reviewers",
            ),
            SkillGuideline(
                id="GDL-QUA-004",
                category="performance",
                recommendation="Include cyclomatic_complexity and lines_of_code metrics in all reviews",
                rationale="Quantitative metrics enable trend tracking across reviews",
            ),
        ],
        constraints=[
            *_UNIVERSAL_CONSTRAINTS,
            SkillConstraint(
                id="CON-QUA-001",
                category="scope",
                description="Quality skill is read-only; it must not modify the code under review",
                limit="read_only=true",
                enforcement="hard",
            ),
            SkillConstraint(
                id="CON-QUA-002",
                category="scope",
                description="Maximum lines of code that can be reviewed in a single invocation",
                limit="2000 lines",
                enforcement="soft",
            ),
            SkillConstraint(
                id="CON-QUA-003",
                category="safety",
                description="Security audit results must never include the actual secret values found",
                limit="redact_secrets=true",
                enforcement="hard",
            ),
        ],
    ),
    # ───────────────────────────────────────────────────────────────────
    # OPERATIONS
    # ───────────────────────────────────────────────────────────────────
    "operations": SkillSpec(
        skill_id="operations",
        name="Operations",
        version="1.1.0",
        description="Documentation, creative writing, cost analysis, experiments, error recovery, synthesis, image generation",
        agent_type="OPERATIONS",
        modes=[
            "documentation",
            "creative_writing",
            "cost_analysis",
            "experiment",
            "error_recovery",
            "synthesis",
            "image_generation",
            "improvement",
        ],
        capabilities=[
            "documentation",
            "creative_writing",
            "cost_analysis",
            "experiment",
            "error_recovery",
            "synthesis",
            "image_generation",
            "improvement",
        ],
        input_schema={
            "type": "object",
            "required": ["task_type"],
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": [
                        "documentation",
                        "creative_writing",
                        "cost_analysis",
                        "experiment",
                        "error_recovery",
                        "synthesis",
                        "image_generation",
                        "improvement",
                    ],
                },
                "content": {"type": "string", "description": "Input content to process"},
                "context": {"type": "object", "description": "Additional context for the operation"},
                "format": {"type": "string", "enum": ["markdown", "html", "plain", "json"], "default": "markdown"},
            },
        },
        output_schema={
            "type": "object",
            "required": ["content", "type"],
            "properties": {
                "content": {"type": "string", "description": "Generated content"},
                "type": {"type": "string"},
                "format": {"type": "string", "enum": ["markdown", "html", "plain", "json"]},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "order": {"type": "integer"},
                        },
                    },
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "word_count": {"type": "integer"},
                        "estimated_cost": {"type": "number"},
                        "experiment_results": {"type": "object"},
                    },
                },
            },
        },
        max_tokens=16384,
        timeout_seconds=300,
        max_cost_usd=1.50,
        requires_tools=["WebSearch", "FileOperations"],
        min_verification_score=0.5,
        tags=["documentation", "creative", "operations", "cost", "recovery"],
        standards=[
            *_UNIVERSAL_STANDARDS,
            SkillStandard(
                id="STD-OPS-001",
                category="documentation",
                rule="Generated documentation MUST include: purpose, usage examples, and API reference sections",
                severity="error",
            ),
            SkillStandard(
                id="STD-OPS-002",
                category="documentation",
                rule="Documentation MUST be written in the specified format (default: markdown)",
                severity="error",
                check_hint="output.format == input.format",
            ),
            SkillStandard(
                id="STD-OPS-003",
                category="cost_analysis",
                rule="Cost analyses MUST include: current cost, projected cost, and savings opportunities",
                severity="error",
            ),
            SkillStandard(
                id="STD-OPS-004",
                category="error_recovery",
                rule="Error recovery plans MUST include: root cause, fix steps, verification, and prevention measures",
                severity="error",
            ),
            SkillStandard(
                id="STD-OPS-005",
                category="experiment",
                rule="Experiments MUST define: hypothesis, methodology, success criteria, and rollback plan",
                severity="error",
            ),
            SkillStandard(
                id="STD-OPS-006",
                category="synthesis",
                rule="Synthesis outputs MUST cite all source inputs and note any conflicts between them",
                severity="warning",
            ),
        ],
        guidelines=[
            SkillGuideline(
                id="GDL-OPS-001",
                category="usage",
                recommendation="Use synthesis mode to combine outputs from multiple skills into a coherent report",
                rationale="Synthesis reduces information overload for the consumer",
            ),
            SkillGuideline(
                id="GDL-OPS-002",
                category="output_format",
                recommendation="Include table of contents for documentation exceeding 500 words",
                rationale="Long documents need navigation aids for usability",
            ),
            SkillGuideline(
                id="GDL-OPS-003",
                category="integration",
                recommendation="Run improvement mode after every major feature completion to identify tech debt",
                rationale="Continuous improvement prevents debt accumulation",
            ),
        ],
        constraints=[
            *_UNIVERSAL_CONSTRAINTS,
            SkillConstraint(
                id="CON-OPS-001",
                category="resource",
                description="Maximum output length for documentation generation",
                limit="10000 words",
                enforcement="soft",
            ),
            SkillConstraint(
                id="CON-OPS-002",
                category="safety",
                description="Error recovery must not execute destructive operations without confirmation",
                limit="confirmation_required=true",
                enforcement="hard",
            ),
            SkillConstraint(
                id="CON-OPS-003",
                category="scope",
                description="Cost analysis must not access external billing APIs without explicit permission",
                limit="external_api_permission_required=true",
                enforcement="hard",
            ),
        ],
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Legacy agent type -> consolidated skill mapping
# ═══════════════════════════════════════════════════════════════════════════

_LEGACY_AGENT_TO_SKILL: dict[str, str] = {
    # Planning / Orchestration
    "PLANNER": "planner",
    "USER_INTERACTION": "orchestrator",
    "CONTEXT_MANAGER": "orchestrator",
    "ORCHESTRATOR": "orchestrator",
    # Research
    "EXPLORER": "researcher",
    "RESEARCHER": "researcher",
    "LIBRARIAN": "researcher",
    "CONSOLIDATED_RESEARCHER": "researcher",
    # Oracle
    "ORACLE": "oracle",
    "PONDER": "oracle",
    "CONSOLIDATED_ORACLE": "oracle",
    # Building
    "BUILDER": "builder",
    # Architecture / Infrastructure
    "UI_PLANNER": "architect",
    "DATA_ENGINEER": "architect",
    "DEVOPS": "architect",
    "VERSION_CONTROL": "architect",
    "ARCHITECT": "architect",
    # Quality
    "EVALUATOR": "quality",
    "SECURITY_AUDITOR": "quality",
    "TEST_AUTOMATION": "quality",
    "QUALITY": "quality",
    # Operations
    "SYNTHESIZER": "operations",
    "DOCUMENTATION_AGENT": "operations",
    "COST_PLANNER": "operations",
    "EXPERIMENTATION_MANAGER": "operations",
    "IMPROVEMENT": "operations",
    "ERROR_RECOVERY": "operations",
    "IMAGE_GENERATOR": "operations",
    "OPERATIONS": "operations",
}


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def get_skill(skill_id: str) -> SkillSpec | None:
    """Get a skill spec by ID."""
    return SKILL_REGISTRY.get(skill_id)


def get_skill_for_agent_type(agent_type: str) -> SkillSpec | None:
    """Map a legacy agent type string to its consolidated skill spec.

    Returns:
        The SkillSpec | None result.
    """
    skill_id = _LEGACY_AGENT_TO_SKILL.get(agent_type.upper())
    if skill_id:
        return SKILL_REGISTRY.get(skill_id)
    return None


def get_all_skills() -> dict[str, SkillSpec]:
    """Return the full registry."""
    return dict(SKILL_REGISTRY)


def get_skills_by_capability(capability: str) -> list[SkillSpec]:
    """Find all skills that declare a given capability."""
    return [spec for spec in SKILL_REGISTRY.values() if capability in spec.capabilities]


def get_skills_by_tag(tag: str) -> list[SkillSpec]:
    """Find all skills with a given tag."""
    return [spec for spec in SKILL_REGISTRY.values() if tag in spec.tags]


def get_skills_by_standard_category(category: str) -> list[SkillSpec]:
    """Find all skills that have standards in a given category."""
    return [spec for spec in SKILL_REGISTRY.values() if any(s.category == category for s in spec.standards)]


def validate_all() -> list[str]:
    """Validate every skill spec in the registry. Returns list of errors.

    Returns:
        The result string.
    """
    errors = []
    for skill_id, spec in SKILL_REGISTRY.items():
        if spec.skill_id != skill_id:
            errors.append(f"Mismatched key '{skill_id}' vs spec.skill_id '{spec.skill_id}'")
        errors.extend(spec.validate())
    return errors


def auto_populate_from_agents() -> dict[str, SkillSpec]:
    """Auto-derive SkillSpecs from all MultiModeAgent subclasses.

    For each agent class, calls ``to_skill_spec()`` to generate a baseline
    spec. If a hand-written spec already exists in SKILL_REGISTRY, the
    auto-derived fields (modes, capabilities) are merged *under* the
    hand-written spec — i.e., hand-written standards, constraints, and
    schemas always take precedence.

    Returns:
        Dict of skill_id -> merged SkillSpec for all discovered agents.
    """
    try:
        from vetinari.agents.multi_mode_agent import MultiModeAgent
    except ImportError:
        logger.debug("MultiModeAgent not available for auto-population")
        return {}

    # Import agent classes to ensure subclasses are registered
    _agent_modules = [
        "vetinari.agents.builder_agent",
        "vetinari.agents.planner_agent",
        "vetinari.agents.consolidated.researcher_agent",
        "vetinari.agents.consolidated.oracle_agent",
        "vetinari.agents.consolidated.quality_agent",
        "vetinari.agents.consolidated.operations_agent",
    ]
    import importlib

    for mod_name in _agent_modules:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            logger.debug("Could not import %s for auto-population", mod_name)

    result: dict[str, SkillSpec] = {}

    for subclass in MultiModeAgent.__subclasses__():
        if not subclass.MODES:
            continue
        try:
            auto_spec = subclass.to_skill_spec()
        except Exception as exc:
            logger.debug("Failed to derive SkillSpec from %s: %s", subclass.__name__, exc)
            continue

        existing = SKILL_REGISTRY.get(auto_spec.skill_id)
        if existing is not None:
            # Merge: auto-derived modes/capabilities fill gaps,
            # hand-written standards/constraints/schemas always win
            merged_modes = existing.modes or auto_spec.modes
            merged_caps = existing.capabilities or auto_spec.capabilities
            merged = SkillSpec(
                skill_id=existing.skill_id,
                name=existing.name,
                description=existing.description,
                version=existing.version,
                agent_type=existing.agent_type or auto_spec.agent_type,
                modes=merged_modes,
                capabilities=merged_caps,
                input_schema=existing.input_schema or auto_spec.input_schema,
                output_schema=existing.output_schema or auto_spec.output_schema,
                max_tokens=existing.max_tokens,
                max_retries=existing.max_retries,
                timeout_seconds=existing.timeout_seconds,
                max_cost_usd=existing.max_cost_usd,
                requires_tools=existing.requires_tools,
                min_verification_score=existing.min_verification_score,
                require_schema_validation=existing.require_schema_validation,
                forbidden_patterns=existing.forbidden_patterns,
                standards=existing.standards,
                guidelines=existing.guidelines,
                constraints=existing.constraints,
                author=existing.author,
                tags=list(set(existing.tags + auto_spec.tags)),
                enabled=existing.enabled,
                deprecated=existing.deprecated,
                deprecated_by=existing.deprecated_by,
            )
            result[merged.skill_id] = merged
        else:
            result[auto_spec.skill_id] = auto_spec

    return result
