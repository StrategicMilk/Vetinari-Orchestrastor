"""Create all 35 Architecture Decision Records for the Vetinari project.

Run once to populate the ADR storage directory with research-backed decisions
covering every major subsystem.  Safe to re-run — checks for existing ADRs
before creating.

Usage::

    python scripts/create_adrs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vetinari.adr import ADRSystem  # noqa: E402


# fmt: off
ADRS: list[dict] = [
    # ── Agent Architecture (1-6) ──
    {
        "adr_id": "ADR-0001",
        "title": "Six-Agent Consolidated Pipeline",
        "category": "architecture",
        "context": (
            "Vetinari needs a structured agent hierarchy to coordinate complex "
            "software engineering tasks.  Alternatives considered: LangGraph, "
            "CrewAI, AutoGen, and a custom pipeline.  The codebase already has "
            "20+ legacy agent classes that need consolidation."
        ),
        "decision": (
            "Keep the custom six-agent consolidated pipeline: Planner → Researcher → "
            "Oracle → Builder → Quality → Operations.  Each agent is a multi-mode "
            "class supporting multiple specialist behaviors via a 'mode' parameter.  "
            "LangGraph/CrewAI were rejected because they add external dependencies "
            "without improving the specific orchestration patterns Vetinari requires."
        ),
        "consequences": (
            "Pros: Full control over agent behavior, no external framework lock-in, "
            "simpler debugging, lower dependency surface.  "
            "Cons: Must maintain orchestration logic ourselves, no community plugins."
        ),
        "related_adrs": ["ADR-0002", "ADR-0003", "ADR-0005"],
        "notes": "Reviewed LangGraph v0.2, CrewAI v0.28, AutoGen v0.2. None justified the dependency."
    },
    {
        "adr_id": "ADR-0002",
        "title": "Agent Contracts via Dataclass Specs",
        "category": "agent_design",
        "context": (
            "Agents need a shared contract system to define tasks, results, and "
            "capabilities.  Options: Pydantic models, dataclasses, Protocol classes, "
            "or plain dicts."
        ),
        "decision": (
            "Use Python dataclasses in vetinari/agents/contracts.py for AgentSpec, "
            "AgentTask, AgentResult, Plan, and Task.  Pydantic is used elsewhere for "
            "validation but dataclasses are lighter for internal contracts."
        ),
        "consequences": (
            "Pros: Zero-dependency contracts, fast construction, easy serialization.  "
            "Cons: No built-in validation (must validate manually at boundaries)."
        ),
        "related_adrs": ["ADR-0001", "ADR-0004"],
    },
    {
        "adr_id": "ADR-0003",
        "title": "Multi-Mode Agent Pattern",
        "category": "agent_design",
        "context": (
            "Each consolidated agent (e.g. Researcher) supports multiple specialist "
            "behaviors (domain_research, database, devops, ui_design, lateral_thinking).  "
            "Options: one class per mode, mode parameter on a single class, or "
            "strategy pattern with injected behaviors."
        ),
        "decision": (
            "Use a single class per consolidated agent with a 'mode' parameter that "
            "selects the behavior.  Mode dispatch is handled via a dict mapping mode "
            "names to handler methods.  This keeps the agent count at 6 while supporting "
            "30+ specialist behaviors."
        ),
        "consequences": (
            "Pros: Fewer classes, centralized agent lifecycle, easy to add modes.  "
            "Cons: Individual agent classes are large (500+ lines), mode coupling."
        ),
        "related_adrs": ["ADR-0001"],
    },
    {
        "adr_id": "ADR-0004",
        "title": "Canonical Type Definitions in vetinari.types",
        "category": "architecture",
        "context": (
            "Enums like AgentType, TaskStatus, ExecutionMode, and PlanStatus are "
            "used across all agents and modules.  Defining them in multiple places "
            "caused import cycles and inconsistencies."
        ),
        "decision": (
            "All shared enums live in vetinari/types.py as the single source of truth.  "
            "Domain dataclasses (Plan, Task, AgentResult) live in vetinari/agents/contracts.py.  "
            "All imports MUST use canonical sources — enforced by VET001 linting rule."
        ),
        "consequences": (
            "Pros: No duplicate enum definitions, clear import hierarchy, enforceable.  "
            "Cons: types.py becomes a high-impact shared module requiring careful changes."
        ),
        "related_adrs": ["ADR-0002"],
    },
    {
        "adr_id": "ADR-0005",
        "title": "Agent Graph Execution Backend",
        "category": "architecture",
        "context": (
            "Task execution needs a backend that supports dependency ordering, "
            "parallel execution, and error recovery.  Options: simple sequential, "
            "ThreadPoolExecutor, agent graph with DAG scheduling, or external "
            "workflow engine (Temporal, Prefect)."
        ),
        "decision": (
            "Use a custom AgentGraph (vetinari/orchestration/agent_graph.py) with "
            "DAG-based task scheduling.  The TwoLayerOrchestrator provides the "
            "assembly-line pipeline on top.  External workflow engines rejected "
            "due to operational complexity for a local-first tool."
        ),
        "consequences": (
            "Pros: No external services, DAG enables parallelism, integrated with "
            "agent contracts.  Cons: Must implement our own checkpointing and recovery."
        ),
        "related_adrs": ["ADR-0001", "ADR-0006"],
    },
    {
        "adr_id": "ADR-0006",
        "title": "Durable Execution with Checkpointing",
        "category": "architecture",
        "context": (
            "Long-running agent tasks can fail midway.  Need a mechanism to "
            "checkpoint progress and resume.  Options: Temporal SDK, custom "
            "checkpoint files, SQLite WAL, or in-memory snapshots."
        ),
        "decision": (
            "Use a custom DurableExecutionEngine (vetinari/orchestration/durable_execution.py) "
            "with JSON checkpoint files.  Each task's state is persisted after completion.  "
            "Temporal rejected as too heavy for a local tool."
        ),
        "consequences": (
            "Pros: Simple file-based checkpoints, no external service.  "
            "Cons: JSON serialization limits (no arbitrary objects), manual cleanup needed."
        ),
        "related_adrs": ["ADR-0005"],
        "notes": "Consider consolidating with execution_graph checkpoint logic in future."
    },
    # ── Memory System (7-8) ──
    {
        "adr_id": "ADR-0007",
        "title": "Dual-Store Memory Architecture",
        "category": "data_flow",
        "context": (
            "Agents need both short-term working memory (current task context) and "
            "long-term memory (learned patterns, past decisions).  Options: single "
            "store, dual store, vector DB, knowledge graph (Graphiti)."
        ),
        "decision": (
            "Keep the dual-store architecture: working memory (dict-based, per-session) "
            "and long-term memory (file-backed, persistent).  Evaluate Graphiti v0.6 "
            "for knowledge graph features in a future phase."
        ),
        "consequences": (
            "Pros: Simple, fast, no external DB dependency.  "
            "Cons: No semantic search, no relationship queries across memories."
        ),
        "related_adrs": ["ADR-0008"],
        "notes": "Graphiti v0.6 evaluation deferred to post-MVP."
    },
    {
        "adr_id": "ADR-0008",
        "title": "Blackboard Pattern for Agent Communication",
        "category": "data_flow",
        "context": (
            "Agents need to share intermediate results during orchestration.  "
            "Options: direct message passing, shared blackboard, event bus, "
            "or centralized state store."
        ),
        "decision": (
            "Use a Blackboard pattern (vetinari/blackboard.py) as a shared scratchpad "
            "that any agent can read/write.  The orchestrator manages blackboard lifecycle.  "
            "This is simpler than message passing for a single-process system."
        ),
        "consequences": (
            "Pros: Simple shared state, easy debugging, agents are decoupled.  "
            "Cons: No ordering guarantees, potential for stale reads in concurrent scenarios."
        ),
        "related_adrs": ["ADR-0007"],
    },
    # ── Safety and Security (9-11) ──
    {
        "adr_id": "ADR-0009",
        "title": "Guardrails-Based Safety System",
        "category": "security",
        "context": (
            "Agent outputs need safety checks before execution (code execution, "
            "file writes, API calls).  Options: custom guardrails, NeMo Guardrails, "
            "LLM Guard, or manual review gates."
        ),
        "decision": (
            "Use custom guardrails (vetinari/safety/guardrails.py) with pluggable "
            "validators for sensitive data, code safety, and prompt injection.  "
            "Amend to add LLM Guard as an additional layer for input/output scanning."
        ),
        "consequences": (
            "Pros: Fine-grained control, no external dependency for core checks.  "
            "Cons: Must maintain detection patterns, LLM Guard adds a dependency."
        ),
        "related_adrs": ["ADR-0010", "ADR-0011"],
        "notes": "LLM Guard integration is Phase 4 work."
    },
    {
        "adr_id": "ADR-0010",
        "title": "Sandboxed Code Execution",
        "category": "security",
        "context": (
            "The Builder agent generates and executes code.  Need isolation to "
            "prevent damage.  Options: subprocess with restricted permissions, "
            "Docker containers, WebAssembly, or OS-level sandboxing."
        ),
        "decision": (
            "Use subprocess-based code sandbox (vetinari/code_sandbox.py) with "
            "restricted imports, timeout enforcement, and output capture.  Docker "
            "isolation is available as an optional upgrade."
        ),
        "consequences": (
            "Pros: Works without Docker, fast startup, captures stdout/stderr.  "
            "Cons: Subprocess isolation is weaker than container isolation."
        ),
        "related_adrs": ["ADR-0009"],
    },
    {
        "adr_id": "ADR-0011",
        "title": "Maker-Checker Pattern for Quality Gates",
        "category": "security",
        "context": (
            "Builder agent output must be reviewed before being accepted.  Need a "
            "pattern that separates creation from approval.  Options: manual review, "
            "automated quality gates, maker-checker with Quality agent."
        ),
        "decision": (
            "Implement maker-checker pattern: Builder (maker) produces artifacts, "
            "Quality agent (checker) reviews them.  Quality gate decisions are mandatory "
            "and cannot be overridden by other agents — only humans can bypass."
        ),
        "consequences": (
            "Pros: Separation of concerns, automated review, auditable decisions.  "
            "Cons: Adds latency to the pipeline, Quality agent must be reliable."
        ),
        "related_adrs": ["ADR-0009", "ADR-0001"],
    },
    # ── Observability (12-14) ──
    {
        "adr_id": "ADR-0012",
        "title": "OpenTelemetry for Distributed Tracing",
        "category": "integration",
        "context": (
            "Need observability into agent execution: spans, metrics, and logs.  "
            "Options: custom logging, OpenTelemetry, Datadog SDK, or Prometheus."
        ),
        "decision": (
            "Use OpenTelemetry (OTel) with the GenAI semantic conventions for "
            "tracing agent execution.  Amend to complete the OTel GenAI integration "
            "that is currently partial (vetinari/observability/otel_genai.py)."
        ),
        "consequences": (
            "Pros: Industry standard, vendor-neutral, rich semantic conventions.  "
            "Cons: OTel SDK is heavy, GenAI conventions are still evolving."
        ),
        "related_adrs": ["ADR-0013", "ADR-0014"],
        "notes": "OTel GenAI completion is Phase 4 work."
    },
    {
        "adr_id": "ADR-0013",
        "title": "Structured Logging with JSON Format",
        "category": "integration",
        "context": (
            "Agents produce logs that need to be searchable and parseable.  "
            "Options: plain text, JSON structured, or logfmt."
        ),
        "decision": (
            "Use Python logging with JSON structured output via "
            "vetinari/structured_logging.py.  All modules use "
            "logger = logging.getLogger(__name__) with %-style formatting.  "
            "No print() in production code."
        ),
        "consequences": (
            "Pros: Machine-parseable, compatible with log aggregators, standard library.  "
            "Cons: JSON logs are less human-readable in console."
        ),
        "related_adrs": ["ADR-0012"],
    },
    {
        "adr_id": "ADR-0014",
        "title": "Dashboard with Flask Web UI",
        "category": "integration",
        "context": (
            "Need a web interface for monitoring agent execution, viewing plans, "
            "and managing ADRs.  Options: Flask, FastAPI, Streamlit, or React SPA."
        ),
        "decision": (
            "Use Flask with server-side templates (vetinari/web/) for the dashboard.  "
            "Blueprint-based organization: admin, plans, analytics, ADR routes.  "
            "FastAPI rejected because the dashboard is read-heavy with minimal API needs."
        ),
        "consequences": (
            "Pros: Simple, mature, good template support, easy to deploy.  "
            "Cons: No async support (threads used instead), no built-in API docs."
        ),
        "related_adrs": ["ADR-0012"],
    },
    # ── Planning System (15-17) ──
    {
        "adr_id": "ADR-0015",
        "title": "Two-Layer Orchestration Pattern",
        "category": "decomposition",
        "context": (
            "Need to combine high-level planning with low-level task execution.  "
            "Options: flat task list, hierarchical planner, two-layer (plan + execute), "
            "or reactive agent loop."
        ),
        "decision": (
            "Use two-layer orchestration: Layer 1 generates a DAG-based execution plan "
            "(PlanGenerator), Layer 2 executes it (DurableExecutionEngine or AgentGraph).  "
            "The assembly-line pipeline has 7 stages: analyze → plan → decompose → "
            "assign → execute → review → assemble."
        ),
        "consequences": (
            "Pros: Clean separation of planning and execution, supports replanning.  "
            "Cons: More complex than a simple loop, plan generation adds latency."
        ),
        "related_adrs": ["ADR-0005", "ADR-0006", "ADR-0016"],
    },
    {
        "adr_id": "ADR-0016",
        "title": "Subtask Tree for Hierarchical Decomposition",
        "category": "decomposition",
        "context": (
            "Complex goals need recursive decomposition into subtasks.  Options: "
            "flat list, tree structure, or DAG with cross-dependencies."
        ),
        "decision": (
            "Use a subtask tree (vetinari/planning/subtask_tree.py) for hierarchical "
            "decomposition.  The planner recursively breaks goals into subtrees.  "
            "Cross-dependencies are tracked at the execution graph level."
        ),
        "consequences": (
            "Pros: Natural hierarchical representation, easy visualization.  "
            "Cons: Tree doesn't capture all dependency patterns (DAG augments it)."
        ),
        "related_adrs": ["ADR-0015"],
    },
    {
        "adr_id": "ADR-0017",
        "title": "Plan Approval Gate for Human Oversight",
        "category": "security",
        "context": (
            "Plans generated by the Planner agent may need human review before "
            "execution, especially for high-stakes operations.  Options: always "
            "auto-approve, always require approval, or risk-based approval."
        ),
        "decision": (
            "Implement a configurable plan approval gate (vetinari/plan_mode.py).  "
            "High-stakes categories (architecture, security, data_flow) require "
            "explicit approval.  Other categories can be auto-approved."
        ),
        "consequences": (
            "Pros: Human oversight for risky operations, configurable per category.  "
            "Cons: Adds friction for non-risky tasks if misconfigured."
        ),
        "related_adrs": ["ADR-0015", "ADR-0011"],
    },
    # ── Model Management (18-21) ──
    {
        "adr_id": "ADR-0018",
        "title": "LM Studio as Primary Inference Backend",
        "category": "integration",
        "context": (
            "Vetinari needs an LLM inference backend.  Options: direct API calls "
            "(OpenAI, Anthropic), LM Studio local server, Ollama, vLLM, or "
            "multi-provider with failover."
        ),
        "decision": (
            "Use LM Studio as the primary local inference backend via its OpenAI-compatible "
            "API.  Support additional providers (OpenAI, Anthropic, Gemini, Cohere) via "
            "the adapter system.  Local-first design prioritizes LM Studio."
        ),
        "consequences": (
            "Pros: Free local inference, privacy, no API costs for development.  "
            "Cons: Hardware-dependent, model quality varies, no cloud scaling."
        ),
        "related_adrs": ["ADR-0019", "ADR-0020"],
    },
    {
        "adr_id": "ADR-0019",
        "title": "Multi-Provider Adapter System",
        "category": "api_design",
        "context": (
            "Need to support multiple LLM providers with a unified interface.  "
            "Options: direct SDK calls per provider, adapter pattern, or "
            "LiteLLM proxy."
        ),
        "decision": (
            "Use the adapter pattern (vetinari/adapters/) with a base class "
            "(BaseModelAdapter) and provider-specific implementations.  The "
            "AdapterManager (vetinari/adapter_manager.py) handles provider "
            "selection and failover.  LiteLLM rejected to avoid the dependency."
        ),
        "consequences": (
            "Pros: Clean abstraction, easy to add providers, no proxy overhead.  "
            "Cons: Must maintain each adapter, API differences require workarounds."
        ),
        "related_adrs": ["ADR-0018", "ADR-0020"],
    },
    {
        "adr_id": "ADR-0020",
        "title": "Dynamic Model Router for Task-Based Selection",
        "category": "performance",
        "context": (
            "Different tasks benefit from different models (fast model for simple "
            "queries, large model for complex reasoning).  Need automated model "
            "selection.  Options: fixed mapping, dynamic router, or user selection."
        ),
        "decision": (
            "Use a DynamicModelRouter (vetinari/dynamic_model_router.py) that maps "
            "task types to model tiers (fast, general, coder, vision).  The router "
            "considers VRAM availability and task complexity."
        ),
        "consequences": (
            "Pros: Optimized resource usage, automatic model selection.  "
            "Cons: Router heuristics may not always pick the best model."
        ),
        "related_adrs": ["ADR-0018", "ADR-0021"],
    },
    {
        "adr_id": "ADR-0021",
        "title": "VRAM Manager for GPU Resource Tracking",
        "category": "performance",
        "context": (
            "Local inference requires GPU memory management.  Loading multiple "
            "models can exceed VRAM.  Options: manual management, automatic "
            "VRAM tracking, or defer to LM Studio."
        ),
        "decision": (
            "Implement a VRAMManager (vetinari/vram_manager.py) that tracks GPU "
            "memory usage and coordinates model loading.  Works alongside LM Studio's "
            "own model management."
        ),
        "consequences": (
            "Pros: Prevents OOM errors, enables informed model selection.  "
            "Cons: GPU detection can be unreliable across platforms."
        ),
        "related_adrs": ["ADR-0020"],
    },
    # ── Learning and Optimization (22-24) ──
    {
        "adr_id": "ADR-0022",
        "title": "Feedback Loop for Agent Self-Improvement",
        "category": "agent_design",
        "context": (
            "Agents should improve over time based on task outcomes.  Options: "
            "static prompts, feedback loop with episode memory, reinforcement "
            "learning, or DSPy optimization."
        ),
        "decision": (
            "Use a feedback loop (vetinari/learning/feedback_loop.py) with episode "
            "memory that records task outcomes.  The auto-tuner adjusts parameters "
            "based on historical performance.  Amend to integrate DSPy for prompt "
            "optimization in a future phase."
        ),
        "consequences": (
            "Pros: Continuous improvement, data-driven tuning.  "
            "Cons: Requires sufficient history, cold-start problem."
        ),
        "related_adrs": ["ADR-0023", "ADR-0024"],
        "notes": "DSPy integration skeleton is Phase 4 work."
    },
    {
        "adr_id": "ADR-0023",
        "title": "Cost Optimizer for Inference Budget Management",
        "category": "performance",
        "context": (
            "Cloud API calls have costs.  Need to track and optimize spending.  "
            "Options: no tracking, simple counter, cost optimizer with learning, "
            "or hard budget limits."
        ),
        "decision": (
            "Use a CostOptimizer (vetinari/learning/cost_optimizer.py) that tracks "
            "per-request costs, learns cost patterns, and suggests cheaper alternatives "
            "when quality impact is minimal.  Hard budget limits available as a safety net."
        ),
        "consequences": (
            "Pros: Cost awareness, automatic optimization, budget protection.  "
            "Cons: Cost estimation requires model pricing data maintenance."
        ),
        "related_adrs": ["ADR-0022", "ADR-0020"],
    },
    {
        "adr_id": "ADR-0024",
        "title": "Training Pipeline for Custom Model Fine-Tuning",
        "category": "agent_design",
        "context": (
            "Collected interaction data could be used to fine-tune models for "
            "Vetinari-specific tasks.  Options: no fine-tuning, LoRA adapters, "
            "full fine-tuning, or distillation."
        ),
        "decision": (
            "Provide a training pipeline framework (vetinari/training/pipeline.py) "
            "that collects and formats training data.  Actual fine-tuning is deferred "
            "to when sufficient data quality is achieved.  LoRA is the preferred "
            "fine-tuning approach for efficiency."
        ),
        "consequences": (
            "Pros: Data collection starts early, pipeline ready when needed.  "
            "Cons: Fine-tuning not yet active, data quality uncertain."
        ),
        "related_adrs": ["ADR-0022"],
    },
    # ── Configuration and Infrastructure (25-27) ──
    {
        "adr_id": "ADR-0025",
        "title": "YAML Configuration with Pydantic Validation",
        "category": "api_design",
        "context": (
            "Runtime configuration needs validation and documentation.  Current "
            "approach uses raw YAML loading.  Options: raw YAML, Pydantic Settings, "
            "dynaconf, or Python dataclasses."
        ),
        "decision": (
            "Amend to migrate from raw YAML loading to Pydantic Settings "
            "(pydantic-settings package) for configuration validation.  This provides "
            "type checking, environment variable support, and auto-documentation."
        ),
        "consequences": (
            "Pros: Type-safe config, env var support, validation on load.  "
            "Cons: Adds pydantic-settings dependency, migration effort."
        ),
        "related_adrs": ["ADR-0026"],
        "notes": "Pydantic Settings migration is Phase 4 work."
    },
    {
        "adr_id": "ADR-0026",
        "title": "Circuit Breaker for Resilient API Calls",
        "category": "performance",
        "context": (
            "LLM API calls can fail or timeout.  Need a resilience pattern.  "
            "Options: simple retry, exponential backoff, circuit breaker, or "
            "bulkhead isolation."
        ),
        "decision": (
            "Use a circuit breaker pattern (vetinari/resilience/circuit_breaker.py "
            "and vetinari/orchestration/circuit_breaker.py).  Note: there are currently "
            "two implementations that should be consolidated."
        ),
        "consequences": (
            "Pros: Prevents cascade failures, automatic recovery, fast fail.  "
            "Cons: Two implementations create confusion — consolidation needed."
        ),
        "related_adrs": ["ADR-0019"],
        "notes": "Consolidation of circuit breaker implementations is Phase 4 work."
    },
    {
        "adr_id": "ADR-0027",
        "title": "Credential Management via Environment Variables",
        "category": "security",
        "context": (
            "API keys and secrets must be managed securely.  Options: .env files, "
            "OS keyring, HashiCorp Vault, or AWS Secrets Manager."
        ),
        "decision": (
            "Use environment variables loaded from .env files (gitignored) via "
            "vetinari/credentials.py.  Secrets are never stored in source code "
            "or committed to git.  VET040/041 linting rules enforce this."
        ),
        "consequences": (
            "Pros: Simple, standard, no external service.  "
            "Cons: .env files can be accidentally committed, no rotation support."
        ),
        "related_adrs": ["ADR-0009"],
    },
    # ── Tools and Integrations (28-30) ──
    {
        "adr_id": "ADR-0028",
        "title": "Tool Interface with Registry Pattern",
        "category": "api_design",
        "context": (
            "Agents need access to external tools (file operations, git, web search).  "
            "Options: hardcoded tool calls, tool registry, MCP protocol, or "
            "function calling."
        ),
        "decision": (
            "Use a tool registry (vetinari/tools/) with a common ToolInterface "
            "base class.  Tools are registered and discovered at runtime.  MCP "
            "(Model Context Protocol) support is provided via vetinari/mcp/ for "
            "interoperability with external tool servers."
        ),
        "consequences": (
            "Pros: Extensible, discoverable, MCP-compatible.  "
            "Cons: Registry adds indirection, tool discovery can be slow."
        ),
        "related_adrs": ["ADR-0029"],
    },
    {
        "adr_id": "ADR-0029",
        "title": "MCP Server for External Tool Exposure",
        "category": "integration",
        "context": (
            "Vetinari's tools should be accessible to external systems via a "
            "standard protocol.  Options: REST API, gRPC, MCP, or custom protocol."
        ),
        "decision": (
            "Implement an MCP server (vetinari/mcp/server.py) that exposes "
            "Vetinari's tools and resources to MCP-compatible clients.  This "
            "enables integration with Claude Code, VS Code, and other MCP hosts."
        ),
        "consequences": (
            "Pros: Standard protocol, broad compatibility, bidirectional.  "
            "Cons: MCP spec is still evolving, limited tooling."
        ),
        "related_adrs": ["ADR-0028"],
    },
    {
        "adr_id": "ADR-0030",
        "title": "Git Workflow Integration",
        "category": "integration",
        "context": (
            "Builder agent needs to interact with git for commits, branches, "
            "and PRs.  Options: subprocess git calls, GitPython, dulwich, or "
            "direct libgit2 bindings."
        ),
        "decision": (
            "Use subprocess-based git commands (vetinari/git_workflow.py) wrapped "
            "in a GitTool class.  GitPython rejected due to memory leaks and "
            "complexity.  Subprocess approach is simpler and more predictable."
        ),
        "consequences": (
            "Pros: Simple, reliable, uses system git (user's config).  "
            "Cons: Subprocess spawning overhead, shell injection risk (mitigated "
            "by argument list passing)."
        ),
        "related_adrs": ["ADR-0028"],
    },
    # ── Analytics and Monitoring (31-32) ──
    {
        "adr_id": "ADR-0031",
        "title": "Analytics Subsystem for Operational Metrics",
        "category": "data_flow",
        "context": (
            "Need to track agent performance, costs, anomalies, and SLA compliance.  "
            "Options: custom analytics, Prometheus + Grafana, or embedded dashboards."
        ),
        "decision": (
            "Use a custom analytics subsystem (vetinari/analytics/) with modules "
            "for cost tracking, anomaly detection, forecasting, and SLA monitoring.  "
            "Data is stored locally and exposed via the Flask dashboard."
        ),
        "consequences": (
            "Pros: No external infrastructure, integrated with dashboard.  "
            "Cons: Limited scalability, no long-term time-series storage."
        ),
        "related_adrs": ["ADR-0014", "ADR-0023"],
    },
    {
        "adr_id": "ADR-0032",
        "title": "Drift Monitor for Agent Behavior Tracking",
        "category": "agent_design",
        "context": (
            "Agent behavior can drift over time as models change or prompts evolve.  "
            "Need to detect and alert on behavioral drift.  Options: statistical "
            "testing, schema validation, or manual review."
        ),
        "decision": (
            "Use a drift monitoring system (vetinari/drift/) with contract validation, "
            "schema checking, capability auditing, and goal tracking.  Alerts are "
            "raised when agent outputs deviate from expected patterns."
        ),
        "consequences": (
            "Pros: Early drift detection, contract enforcement, auditable.  "
            "Cons: Requires baseline data, may produce false positives."
        ),
        "related_adrs": ["ADR-0022", "ADR-0002"],
    },
    # ── Code Intelligence (33-34) ──
    {
        "adr_id": "ADR-0033",
        "title": "Repository Map for Codebase Understanding",
        "category": "agent_design",
        "context": (
            "Agents need to understand codebase structure to make informed "
            "decisions.  Options: grep-based search, AST parsing, LSP integration, "
            "or embedding-based code search."
        ),
        "decision": (
            "Use a repo_map module (vetinari/repo_map.py) combined with code_search "
            "(vetinari/code_search.py) and grep_context (vetinari/grep_context.py) "
            "for codebase understanding.  AST-based for structure, text-based for "
            "content search."
        ),
        "consequences": (
            "Pros: No external service, language-aware, fast for local repos.  "
            "Cons: Limited cross-file semantic understanding."
        ),
        "related_adrs": ["ADR-0028"],
    },
    {
        "adr_id": "ADR-0034",
        "title": "RAG Knowledge Base for Domain Context",
        "category": "data_flow",
        "context": (
            "Agents need domain-specific knowledge beyond what's in the codebase.  "
            "Options: static prompts, RAG with vector store, fine-tuned models, "
            "or external knowledge graph."
        ),
        "decision": (
            "Use a RAG knowledge base (vetinari/rag/knowledge_base.py) for "
            "domain context injection.  Documents are chunked, embedded, and "
            "retrieved based on query similarity."
        ),
        "consequences": (
            "Pros: Dynamic knowledge, updatable, no fine-tuning needed.  "
            "Cons: Retrieval quality depends on embedding model, chunk strategy."
        ),
        "related_adrs": ["ADR-0007", "ADR-0033"],
    },
    # ── Project Quality (35) ──
    {
        "adr_id": "ADR-0035",
        "title": "Automated Quality Enforcement via Custom Linting",
        "category": "architecture",
        "context": (
            "Code quality rules specific to Vetinari (canonical imports, no print(), "
            "encoding=utf-8, no hardcoded secrets) need automated enforcement.  "
            "Options: ruff-only, custom linter, pre-commit hooks, or CI checks."
        ),
        "decision": (
            "Use a layered enforcement system: ruff for standard Python linting, "
            "plus scripts/check_vetinari_rules.py with 31 custom VET rules for "
            "project-specific conventions.  Enforced at save (PostToolUse hook), "
            "commit (pre-commit), and session end (Stop hook)."
        ),
        "consequences": (
            "Pros: Comprehensive coverage, automated, catches AI-specific mistakes.  "
            "Cons: Custom linter requires maintenance, false positives possible."
        ),
        "related_adrs": ["ADR-0004", "ADR-0011"],
    },
]
# fmt: on


def main() -> None:
    """Create all 35 ADRs."""
    # Use a clean storage path to avoid conflicts with any existing singleton
    storage_path = Path.home() / ".lmstudio" / "projects" / "Vetinari" / "adr"
    system = ADRSystem(str(storage_path))

    created = 0
    skipped = 0

    for adr_data in ADRS:
        adr_id = adr_data["adr_id"]

        # Skip if already exists
        if system.get_adr(adr_id):
            print(f"  SKIP {adr_id}: already exists")  # noqa: T201
            skipped += 1
            continue

        system.create_adr(
            adr_id=adr_id,
            title=adr_data["title"],
            category=adr_data["category"],
            context=adr_data["context"],
            decision=adr_data["decision"],
            consequences=adr_data.get("consequences", ""),
            created_by="audit-phase3",
            status="accepted",
            related_adrs=adr_data.get("related_adrs", []),
            notes=adr_data.get("notes", ""),
        )
        print(f"  OK   {adr_id}: {adr_data['title']}")  # noqa: T201
        created += 1

    print(f"\nDone: {created} created, {skipped} skipped, {len(ADRS)} total")  # noqa: T201

    # Verify
    stats = system.get_statistics()
    print(f"Statistics: {stats['total']} ADRs, {stats['by_status']}, {stats['by_category']}")  # noqa: T201


if __name__ == "__main__":
    main()
