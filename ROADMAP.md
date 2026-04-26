# Vetinari Roadmap

**Authoritative planning document for Vetinari AI Orchestration System**
**Current Version: v0.6.0** | **Last Updated: 2026-04-22**

---

## Table of Contents

1. [Project Vision](#project-vision)
2. [Current Architecture](#current-architecture)
3. [Completed Work](#completed-work)
4. [Current State Assessment](#current-state-assessment)
5. [v0.6.0 — Structural Integrity](#v060--structural-integrity)
6. [v0.7.0 — Production Reliability](#v070--production-reliability)
7. [v0.8.0 — Intelligence Layer](#v080--intelligence-layer)
8. [v0.9.0 — Interoperability & Scale](#v090--interoperability--scale)
9. [v1.0.0 — Production Release](#v100--production-release)
10. [Architecture Decision Records](#architecture-decision-records)
11. [Design Principles](#design-principles)
12. [Risk Register](#risk-register)
13. [Verification Strategy](#verification-strategy)

---

## Project Vision

Vetinari is a **local-first multi-agent AI orchestration system** that decomposes complex goals into structured task graphs and executes them through specialist agents, primarily using local GGUF models via llama-cpp-python. The system aims to be the most capable open-source orchestrator for developers who want full control over their AI pipeline — running locally, routing intelligently across model tiers, and learning from every execution.

### North Star

A developer types a goal in natural language. Vetinari decomposes it, routes each subtask to the optimal local or cloud model, executes through specialist agents with quality gates, learns from the outcome, and delivers a verified result — all with full observability, deterministic replay, and sub-dollar cost for most workflows.

### Core Differentiators

1. **Local-first**: Runs entirely on local GGUF models (llama-cpp-python) with optional cloud escalation — no mandatory API keys
2. **Three-agent factory pipeline**: Orchestrate > Execute > Review (Foreman > Worker > Inspector)
3. **Cascade cost routing**: Start cheap (7B), escalate only on low confidence — 14% improvement measured
4. **Self-improving**: Thompson Sampling model selection, prompt evolution, workflow learning
5. **Full observability**: OpenTelemetry tracing, SQLite cost tracking, real-time dashboard

---

## Current Architecture

### Three-Agent Factory Architecture (ADR-0061)

| Agent | Modes | Cognitive Role |
|-------|-------|----------------|
| **Foreman** | plan, clarify, summarise, prune, extract, consolidate (6) | Orchestrate, decompose, sequence |
| **Worker** | 24 modes across 4 groups: Research (8), Architecture (5), Build (2), Operations (9) | All production work |
| **Inspector** | code_review, security_audit, test_generation, simplification (4) | Mandatory pass/fail gate |

**Total: 34 modes across 3 agents**

### Assembly-Line Pipeline

```
User Input > [Input Analyzer] > [Plan Generator] > [Task Decomposer]
    > [Model Assigner] > [Parallel Executor] > [Output Reviewer] > [Final Assembler]
```

Coordinated by `TwoLayerOrchestrator` with wave-by-wave DAG execution via `ThreadPoolExecutor`.

### Key Architecture Files

```
vetinari/
  types.py                       # Canonical enum source (AgentType, TaskStatus, etc.)
  exceptions.py                  # Custom exception hierarchy
  adr.py                         # Architecture Decision Records system
  agents/
    base_agent.py                # BaseAgent with circuit breakers, token budgets
    multi_mode_agent.py          # MultiModeAgent with MODES routing
    prompt_loader.py             # Runtime prompt loading from .claude/agents/*.md
    planner_agent.py             # PlannerAgent (6 modes)
    builder_agent.py             # BuilderAgent (2 modes)
    contracts.py                 # AgentSpec, AGENT_REGISTRY
    interfaces.py                # AgentInterface ABC
    consolidated/
      researcher_agent.py        # ResearcherAgent (8 modes)
      oracle_agent.py            # OracleAgent (4 modes)
      quality_agent.py           # QualityAgent (4 modes)
      operations_agent.py        # OperationsAgent (9 modes)
  orchestration/two_layer.py     # TwoLayerOrchestrator pipeline
  memory/unified.py              # UnifiedMemoryStore (SQLite + FTS5 + embeddings)
  learning/                      # Self-improvement (Thompson Sampling, feedback, etc.)
  adapters/                      # Multi-provider adapters (llama-cpp-python, OpenAI, etc.)
  schemas/agent_outputs.py       # Pydantic output schemas (34 modes)
  observability/otel_genai.py    # OpenTelemetry GenAI semantic conventions
  structured_logging.py          # Correlation context, JSON logging
  analytics/cost_tracker.py      # SQLite cost tracking
  safety/guardrails.py           # Guardrails pipeline
  security.py                    # Per-agent permissions, enforcement
```

---

## Completed Work

### Phase 1: Foundation (v0.1.0)

- ExecutionContext system with mode enforcement (PLANNING, EXECUTION, SANDBOX)
- Tool interface with ToolRegistry, ToolMetadata, ToolParameter
- AdapterManager for 7 providers (LMStudio, OpenAI, Anthropic, Gemini, Cohere, Ollama, HuggingFace)
- Verification pipeline (CodeSyntax, Security, Import, JSONStructure verifiers)
- Enhanced CLI with 16 subcommands and rich visual feedback

### Phase 2: Agent System (v0.2.0-v0.3.0)

- 22 specialized agents built across all cognitive domains
- Assembly-line pipeline with 7-stage execution and DAG scheduler
- Self-improvement system: QualityScorer, FeedbackLoop, Thompson Sampling, PromptEvolver, WorkflowLearner, CostOptimizer, AutoTuner
- Flask web dashboard with Chart.js, SSE streaming, dark mode
- DualMemoryStore with blackboard system (request_help, claim_entry, publish_finding, consensus voting)
- Structured JSON logging, OpenTelemetry tracing, alert system

### Phase 3: Agent Consolidation (v0.4.0)

- **ADR-001**: Consolidated 22 agents into 6 multi-mode agents (research: 5-7 optimal; further refined to 3 agents in v0.6.0 per ADR-0061)
- **ADR-002**: Flat ensemble over hierarchy — no sub-agent spawning
- Typed Pydantic output schemas for all 33 modes (34 modes in current architecture)
- Circuit breakers per agent (CLOSED/OPEN/HALF_OPEN)
- Per-agent token budgeting with warn at 80%, truncate at 100%
- DynamicModelRouter with Thompson Sampling + capability matching
- SLM/LLM hybrid routing, speculative decoding, continuous batching
- SQLite-backed cost tracking, benchmark framework
- 8 stub remediations (sandbox hooks, log aggregator, verification, cost estimation, MODES validation, CWE patterns, SVG metadata, upgrader)

### Phase 4: Security Hardening (v0.5.0)

- `hmac.compare_digest` for constant-time token comparison
- `require_admin` decorator on all mutating web endpoints
- `validate_json_fields` for API input validation
- Trusted proxy configuration for X-Forwarded-For
- Rate limiting: 10 req/60s per client on sandbox execution
- Fail-closed Fernet credential vault
- Auth decorators on ADR, decomposition, ponder, rules, training routes

### Phase 5: Analytics Pipeline (v0.5.0)

- 7 REST endpoints: `/api/analytics/{cost,sla,anomalies,forecast,models,agents,summary}`

### Phase 6: Cost-Optimised Routing (v0.5.0)

- CascadeRouter with tiered escalation (7B > 30B > 72B)
- Heuristic confidence estimation, configurable threshold (default 0.7)
- BatchProcessor with Anthropic/OpenAI batch API backends (50% cost discount)

### Phase 7: Agent Governance (v0.5.0)

- File-based agent definitions in `.claude/agents/` (6 governance files)
- Root `AGENTS.md` with architecture, jurisdiction, delegation rules, quality gates
- `CLAUDE.md` with build commands, conventions, architecture overview

### Phase 8: Registry & Configuration (v0.5.0)

- `agent_skill_map.json` with 6 consolidated + 20 legacy entries, 12 workflow pipelines
- `skills_registry.json` with 8 skills, cascade_routing, batch_processing config
- 28 AGENT_REGISTRY entries in contracts.py

### Phase 9: Operations Enrichment (v0.5.0)

- Enriched error_recovery mode with pattern-based diagnostics and remediation strategies

### Phase 10: Full Remediation (v0.6.0)

A comprehensive 10-phase remediation spanning code quality, security, observability, memory, documentation, and legacy cleanup.

**Security hardening:**
- Per-agent permission enforcement system (`vetinari/security.py`)
- Sandbox hardening with security policies
- Guardrails pipeline for input/output validation (`vetinari/safety/guardrails.py`)
- Audit logging for all agent actions (`vetinari/audit.py`)
- Cross-model validation for critical decisions (ADR-0043)

**Memory consolidation:**
- Replaced 3 legacy memory backends (DualMemoryStore, OcMemoryStore, MnemosyneMemoryStore) with UnifiedMemoryStore
- SQLite + FTS5 full-text search with BM25 ranking
- Optional embedding-based semantic search via local inference (llama-cpp-python)
- Session management, dedup, and CLI (`scripts/memory_cli.py`)

**Observability:**
- OpenTelemetry GenAI semantic conventions (`vetinari/observability/otel_genai.py`)
- Structured JSON logging with correlation context (`vetinari/structured_logging.py`)
- Evaluation framework (`vetinari/evaluation/`)

**Agent system:**
- Runtime prompt loading from `.claude/agents/*.md` with mtime caching (ADR-0041)
- CLAUDE.md decomposed into 9 modular rule files (ADR-0042)
- Pydantic-based agent contracts
- Config validation schemas (`vetinari/config/schemas.py`)

**Infrastructure:**
- HTTP session factory with connection pooling (`vetinari/http.py`)
- Graceful shutdown with in-flight task draining (`vetinari/shutdown.py`)
- MCP client for external tool integration (`vetinari/mcp/client.py`)
- Web UI refactored into modular Flask blueprints
- Enforcement package for policy enforcement (`vetinari/enforcement/`)

**Legacy cleanup:**
- Deleted 22 legacy agent redirect files from `vetinari/agents/`
- Deleted 14 deprecated skill wrapper directories from `vetinari/skills/`
- Deleted 3 legacy memory backends
- Deleted `vetinari/agents/compat.py` backward-compatibility shim
- Deleted 8 legacy skill test files and outdated examples
- Cleaned `vetinari/agents/__init__.py` and `vetinari/skills/__init__.py` to remove all legacy imports
- Removed 30+ temporary test output files

**Documentation:**
- Comprehensive CHANGELOG.md covering all phases
- 3 new ADRs (ADR-0041, ADR-0042, ADR-0043)
- Updated README, ROADMAP, AGENTS.md to reflect post-remediation state

---

## Current State Assessment

### Codebase Metrics (as of 2026-03-15)

| Metric | Value |
|--------|-------|
| Agent modes | 34 across 3 agents |
| Active skills | 3 (Architect, Operations, Quality) |
| ADRs | 10+ (JSON in `adr/`) |
| Custom lint rules | 31 across 9 categories |
| Python version | 3.10+ required |

### Resolved Gaps (Phase 10 Remediation)

| Gap | Status | Resolution |
|-----|--------|------------|
| 22 legacy agent files | **RESOLVED** | All deleted; imports updated |
| `compat.py` backward-compat shim | **RESOLVED** | Deleted; `__init__.py` rewritten |
| 14 deprecated skill wrappers | **RESOLVED** | All deleted; skill registry cleaned |
| 3 legacy memory backends | **RESOLVED** | Replaced by UnifiedMemoryStore |
| Memory has no semantic search | **RESOLVED** | FTS5 + optional embeddings via local inference |
| Documentation drift (version numbers) | **RESOLVED** | All release-bearing docs aligned to v0.6.0 |
| No structured observability conventions | **RESOLVED** | OpenTelemetry GenAI + structured logging |
| No per-agent permissions | **RESOLVED** | Permission enforcement in `security.py` |
| No guardrails pipeline | **RESOLVED** | Guardrails with configurable rules |

### Remaining Gaps

#### 1. Async Gap (HIGH - performance ceiling)

The core pipeline (`TwoLayerOrchestrator`, all 3 agents, all adapters) is entirely synchronous. This means:
- Pipeline stages block on each LLM call
- No streaming between stages
- ThreadPoolExecutor is the only parallelism mechanism
- Cannot leverage modern async LLM client libraries

#### 2. MCP Integration (MEDIUM - interoperability)

MCP subsystem has working stdio server/client scaffolding, external tool
registration, and JSON-RPC-over-HTTP through the Litestar `/mcp/message`
mount. It is not yet the canonical Worker execution pipeline, and resource
providers plus HTTP+SSE streaming remain future work.

#### 3. Training Pipeline (LOW - future capability)

Local QLoRA/DoRA training paths exist and are gated by the optional training
stack. `learning/training_manager.py` now tracks local job outcomes, including
failed/degraded runs, but production-ready automatic retraining and cloud
fine-tuning are still future work.

#### 4. Remaining Stubs

| Location | Issue |
|----------|-------|
| `adapters/batch_processor.py` | Provider batch backends exist, but live inference fallback behavior still needs end-to-end proof |
| `learning/training_manager.py` | Local job tracking is wired; cloud training and fully automated retraining remain unsupported |
| `benchmarks/*` adapters | Live-path failures now fail closed; CI benchmark labels are smoke gates, not canonical runtime latency claims |

---

## v0.6.0 — Structural Integrity

**Theme**: Clean the house. Remove all dead code, resolve duplicates, standardize naming, fix documentation. This is the foundation everything else builds on.

### 6.1 Duplicate Module Resolution (CRITICAL) -- DONE

~~Delete all 23 flat duplicate files in `vetinari/` root.~~ Completed in earlier remediation phases. Subdirectory versions are canonical.

### 6.2 Legacy Agent File Deletion (CRITICAL) -- DONE

All 22 pre-consolidation agent files deleted from `vetinari/agents/`. All 14 deprecated skill wrapper directories deleted from `vetinari/skills/`. `compat.py` deleted. All imports updated, `__init__.py` files rewritten.

### 6.3 Naming Convention Standardization (HIGH) -- PARTIALLY DONE

Remaining items:
- Ambiguous file renames (`code_mode/engine.py`, `coding_agent/engine.py`, `constraints/registry.py`, `adapters/registry.py`)
- AgentType enum still contains legacy values for backward compatibility -- evaluate removal

### 6.4 Stub Resolution (HIGH) -- PARTIALLY DONE

| Stub | Status |
|------|--------|
| `adapters/batch_processor.py` fallback proof | Remaining |
| `training_manager.py` cloud training | Remaining |
| `mcp/tools.py` canonical pipeline integration | Partially wired via MCP client/server |
| `benchmarks/*` live execution coverage | Partially fixed: failure paths fail closed; canonical runtime coverage remains |

### 6.5 Documentation Alignment (HIGH) -- DONE

- Python version aligned to "3.10+" everywhere
- README, ROADMAP, AGENTS.md updated to current architecture
- CHANGELOG.md created with entries for all completed phases
- Project structure tree updated

### 6.6 Quality of Life -- PARTIALLY DONE

Remaining:
- Resolve remaining TODO/FIXME/HACK comments
- AgentType enum cleanup (legacy values)
- Apply Python 3.10+ idioms consistently

**Exit criteria for v0.6.0**:
- `python -m pytest tests/ -x -q` passes with zero failures
- Zero legacy agent files remain -- **DONE**
- Zero legacy skill wrappers remain -- **DONE**
- Zero legacy memory backends remain -- **DONE**
- All docs reference current architecture only -- **DONE**
- Remaining: stub resolution, naming standardization, enum cleanup

---

## v0.7.0 — Production Reliability

**Theme**: Make the system reliable enough for real daily use. Plan checkpointing, hard budget enforcement, end-to-end pipeline testing, and observability that actually catches problems.

### 7.1 Plan Checkpointing & Resume (CRITICAL)

**Problem**: If execution crashes mid-plan, all progress is lost. Long-running plans (10+ tasks) are unrecoverable.

**Solution**: Save plan state to disk after each wave completion. On restart, detect incomplete plans and offer resume.

**Scope**: `TwoLayerOrchestrator` + new `PlanCheckpoint` dataclass serialized to `plans/{plan_id}/checkpoint.json`.

**Implementation**:
- Serialize plan state (completed tasks, pending tasks, shared memory snapshot) after each wave
- On startup, scan for incomplete checkpoints
- Resume from last completed wave
- Add `--resume` flag to CLI

### 7.2 Hard Budget Enforcement (CRITICAL)

**Problem**: Runaway plans can consume unlimited tokens/time with no kill switch.

**Solution**: Add hard token and time budget limits that abort execution with a clear error.

**Scope**: `constraints/` + `TwoLayerOrchestrator`

**Implementation**:
- `max_total_tokens` per plan (default: 500K tokens)
- `max_wall_time_seconds` per plan (default: 3600s)
- `max_iterations` per agent (default: 10 retries)
- Budget tracking in CostTracker with abort on breach
- Configurable via manifest YAML

### 7.3 End-to-End Integration Tests (HIGH)

**Problem**: 5,119 tests exist, but most mock the LLM layer. Zero tests exercise the full pipeline from goal input to assembled output.

**Solution**: Create integration test suite that runs with a local GGUF model loaded via llama-cpp-python (skipped in CI when unavailable).

**Implementation**:
- `tests/integration/` directory with pytest markers
- Test cases: simple goal to plan to execute to assemble
- Verify plan structure, task routing, quality gate pass
- Measure and assert p95 latency for single-task plans
- Use `@pytest.mark.integration` for conditional execution

### 7.4 MCP Server Implementation (HIGH)

**Problem**: MCP subsystem has stub handlers. With MCP at 97M+ monthly downloads and Linux Foundation governance, this is a critical interoperability gap.

**Solution**: Wire MCP tools to real Vetinari subsystems. Expose agent capabilities as MCP tools.

**Scope**: `vetinari/mcp/`

**Implementation**:
- Expose `plan`, `execute`, `review`, `search` as MCP tools
- Implement MCP resource providers for plan state, agent metrics, memory
- Support MCP transport over stdio and Litestar JSON-RPC HTTP; add HTTP+SSE/resource streaming later
- Publish server in MCP registry

### 7.5 Observability Enrichment (MEDIUM)

**Problem**: OpenTelemetry tracing exists but doesn't follow GenAI semantic conventions. No session replay. No cost attribution per span.

**Solution**: Adopt OpenTelemetry GenAI semantic conventions for all LLM calls.

**Implementation**:
- Add `gen_ai.*` span attributes: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
- Correlate cost tracking with trace spans
- Add session replay capability (record/replay full execution traces)
- Export to Jaeger/Zipkin for visualization

### 7.6 Web UI Stabilization (MEDIUM)

**Problem**: Dashboard has 20+ known UI bugs. Task execution bypasses agent pipeline. Plan approval not enforced in web flow.

**Solution**: Fix critical web flow bugs, ensure web and CLI share the same execution path.

**Implementation**:
- Route web task execution through TwoLayerOrchestrator (not direct agent calls)
- Enforce plan approval in web flow
- Fix SSE streaming reliability
- Add WebSocket support for real-time updates
- Fix ModelSearchEngine to return real data (currently hardcoded)

### 7.7 Error Recovery Hardening (MEDIUM)

**Problem**: Circuit breakers exist per-agent but recovery is manual. No automatic retry with exponential backoff. No dead letter queue for failed tasks.

**Solution**: Implement production-grade error recovery patterns.

**Implementation**:
- Exponential backoff with jitter for transient failures
- Dead letter queue for tasks that exceed retry limit
- Automatic fallback to alternative model on adapter failure
- Structured error classification (transient vs permanent vs resource)

### 7.8 Dynamic Complexity Routing (MEDIUM)

**Problem**: Every task goes through the full pipeline (plan → research → advise → build → verify → operate), even trivial ones like "rename a variable." This wastes time and tokens.

**Solution**: Add a `ComplexityEstimator` that classifies incoming tasks as trivial/simple/moderate/complex and an `AdaptivePipeline` that skips unnecessary stages for simpler tasks.

**Implementation**:
- Keyword analysis, token-count heuristics, and task-type mapping to estimate complexity
- Trivial tasks skip planning and research stages entirely
- Simple tasks skip oracle/advise stage
- 40-50% of tasks could skip the full pipeline, reducing latency significantly
- Complexity classification logged for observability

**Source**: Extracted from archived technical research.

### 7.9 Budget Alert System (MEDIUM)

**Problem**: Hard budget enforcement (7.2) aborts execution, but there is no early warning system. Users discover cost overruns only after they happen.

**Solution**: Add a `BudgetAlertSystem` with configurable monthly limits, threshold alerts, and predicted monthly cost based on daily rate trajectory.

**Implementation**:
- Configurable monthly token and cost budgets
- Alert thresholds at 50%, 80%, 90% of budget
- Predicted monthly cost extrapolated from rolling daily rate
- Alert channels: log warnings, web dashboard banner, optional webhook

**Source**: Extracted from archived technical research.

### 7.10 Context Window Management with Auto-Summarization (LOW)

**Problem**: Long-running agent conversations exhaust context windows, leading to truncation and lost context. Token budgets (7.2) abort, but don't manage context intelligently.

**Solution**: Implement `ConversationSummaryBufferMemory` pattern — when token count approaches the context limit, automatically summarize older messages to reclaim space.

**Implementation**:
- Token counting via `tiktoken` for accurate budget tracking
- When context exceeds configurable threshold (e.g., 80% of window), summarize oldest messages
- Preserve recent messages verbatim, compress older context into summaries
- 30-50% token savings on long conversations

**Source**: Extracted from archived technical research.

**Exit criteria for v0.7.0**:
- Plans survive process restart via checkpointing
- Hard budget limits prevent runaway execution
- At least 5 end-to-end integration tests pass with a local GGUF model loaded
- MCP server responds to tool discovery and basic invocations
- Zero critical web UI bugs
- Budget alerts fire before hard limits are reached

---

## v0.8.0 — Intelligence Layer

**Theme**: Make the system smarter. Semantic memory, dense retrieval, structured reflection, and progressive autonomy. This is where Vetinari transitions from "orchestrator that routes tasks" to "system that learns and adapts."

### 8.1 Semantic Memory with Vector Retrieval (HIGH)

**Problem**: Current memory is key-value only. No semantic search. Agents cannot find relevant past experiences by meaning, only by exact key.

**Solution**: Add embedding-based semantic memory tier using sentence-transformers (local) or OpenAI embeddings (cloud fallback).

**Scope**: `vetinari/memory/` + `vetinari/learning/episode_memory.py`

**Implementation**:
- Three-tier memory architecture:
  - **Working memory**: Current session context (existing DualMemoryStore)
  - **Episodic memory**: Past execution traces with embeddings (extend episode_memory.py)
  - **Semantic memory**: Learned concepts and patterns (new vector store)
- Local embedding via `sentence-transformers/all-MiniLM-L6-v2` (22M params, runs on CPU)
- SQLite + numpy for vector storage (no external DB dependency, preserves local-first principle)
- Graceful fallback to n-gram matching when embeddings unavailable
- Configurable memory depth per task complexity

### 8.2 Structured Reflection & Self-Correction (HIGH)

**Problem**: When an agent produces poor output, the system retries with the same prompt. No structured reflection on what went wrong.

**Solution**: Implement Reflexion-style self-correction: after quality gate failure, generate a structured critique, then re-prompt with the critique as context.

**Implementation**:
- QualityAgent generates structured feedback on failure (not just pass/fail)
- Feedback includes: what specifically failed, why, and suggested correction
- Builder receives feedback as additional context on retry
- Track reflection effectiveness (does retry succeed more often with reflection?)
- Cap reflection loops at 3 iterations to prevent infinite self-critique

### 8.3 Progressive Autonomy Spectrum (MEDIUM)

**Problem**: System is binary — either fully autonomous or requires human approval for every step. No middle ground.

**Solution**: Implement three autonomy levels configurable per task type:
- **Human-in-the-loop**: Approval required before every execution step
- **Human-on-the-loop**: System executes, human reviews output and can intervene
- **Human-out-of-the-loop**: Full autonomy for trusted task types

**Implementation**:
- Autonomy level configurable in manifest YAML per task type
- Trust scores accumulate over successful executions
- Automatic promotion: after N successful autonomous runs, suggest higher autonomy
- Emergency brake: any quality gate failure drops back to human-in-the-loop

### 8.4 Enhanced RAG Pipeline (MEDIUM)

**Problem**: `vetinari/rag/knowledge_base.py` is a single file with no indexing, chunking, or retrieval pipeline. Cannot ingest project documentation or codebases.

**Solution**: Build a proper RAG pipeline with tree-sitter for code, recursive chunking for docs, and hybrid retrieval (BM25 + dense).

**Implementation**:
- Tree-sitter AST parsing for Python (function/class-level chunks with signatures)
- Recursive text chunking with overlap for documentation
- Hybrid retrieval: BM25 keyword matching + dense embedding similarity
- RepoMap enhancement: send function signatures + docstrings, not raw files
- Configurable chunk size and overlap
- Index stored as SQLite FTS5 + numpy vectors

### 8.5 Uncertainty-Aware Model Routing (MEDIUM)

**Problem**: CascadeRouter uses heuristic confidence estimation. No principled uncertainty quantification.

**Solution**: Implement ACAR (Adaptive Confidence-Aware Routing) with calibrated confidence from multiple signals.

**Implementation**:
- Combine: LLM self-reported confidence, output length heuristic, semantic similarity to known-good outputs, historical success rate for task type
- Calibration: track predicted vs actual success, adjust routing thresholds
- Per-model confidence profiles learned from execution history

### 8.6 Workflow Pattern Mining (LOW)

**Problem**: WorkflowLearner stores patterns but doesn't actively improve decomposition strategies based on success rates.

**Solution**: Mine successful execution traces to discover optimal task decomposition patterns per domain.

**Implementation**:
- Cluster successful plans by domain/complexity
- Extract common subtask patterns (e.g., "API endpoint" always decomposes as: schema, handler, tests, docs)
- Suggest learned decomposition patterns to PlannerAgent
- A/B test learned patterns vs default decomposition

### 8.7 Cost Optimization Recommendations Engine (MEDIUM)

**Problem**: Cost tracking exists but is passive — it records what was spent without suggesting how to spend less.

**Solution**: Add a `CostOptimizer` that analyzes per-agent and per-model cost patterns and generates actionable optimization suggestions.

**Implementation**:
- Analyze cost distribution across agents, models, and task types
- Identify expensive patterns: overuse of large models for simple tasks, unnecessary retries, redundant research queries
- Generate ranked recommendations (e.g., "Switch ResearcherAgent to 7B for code_discovery — 85% success rate at 3x lower cost")
- Track recommendation acceptance and measured savings

**Source**: Extracted from archived technical research.

### 8.8 Multi-Model Lifecycle Management (MEDIUM)

**Problem**: The llama-cpp-python adapter loads one model at a time. Switching between models requires unloading the current one, limiting multi-model orchestration.

**Solution**: Implement programmatic model lifecycle management within llama-cpp-python for on-demand load, unload, and VRAM monitoring.

**Implementation**:
- Model pool with configurable max loaded models per VRAM budget
- Auto-evict idle models with configurable TTL per model size
- VRAM usage tracking via `vetinari/models/vram_manager.py`
- Preload models based on upcoming plan requirements
- Graceful degradation when VRAM is insufficient (fall back to smaller models)

**Source**: Extracted from archived technical research.

### 8.9 Orchestration-Level Continuous Batching (LOW)

**Problem**: When multiple agents in the same wave need LLM calls, each call is dispatched independently. No request coalescing.

**Solution**: Add an async request coalescing queue that batches concurrent LLM calls to the same model, leveraging llama-cpp-python's native continuous batching for 3-5x throughput improvement.

**Implementation**:
- `BatchManager` with configurable batch size and flush timeout
- Group requests by target model
- Submit batched requests and demultiplex responses
- Requires async pipeline (9.1) as a prerequisite
- 3-5x throughput improvement on multi-task waves

**Source**: Extracted from archived technical research.

**Exit criteria for v0.8.0**:
- Semantic search returns relevant past episodes by meaning
- Quality gate failures trigger structured reflection before retry
- At least one task type operates in human-out-of-the-loop mode
- RAG pipeline indexes a sample Python project and retrieves relevant chunks
- Model routing decisions correlate with actual success rates (above 70% calibration)
- Cost optimizer generates at least 3 actionable recommendations per 100 tasks

---

## v0.9.0 — Interoperability & Scale

**Theme**: Open the system up. A2A protocol support, async pipeline, multi-language coding, and the foundations for multi-user deployment.

### 9.1 Async Pipeline (HIGH)

**Problem**: Core pipeline is synchronous. Blocks on every LLM call. Cannot stream results between stages. ThreadPoolExecutor is the only parallelism.

**Solution**: Convert core pipeline to async/await. This is a large refactor touching the orchestrator, all adapters, and agent dispatch.

**Implementation**:
- Phase 1: Async adapters (async llama-cpp-python calls, async client libs for cloud providers)
- Phase 2: Async agent dispatch in TwoLayerOrchestrator
- Phase 3: Streaming between pipeline stages (async generators)
- Phase 4: Replace ThreadPoolExecutor with `asyncio.TaskGroup`
- Maintain synchronous fallback for testing and simple use cases
- Migration: start with adapters (lowest risk), work up to orchestrator

### 9.2 Google A2A Protocol Support (HIGH)

**Problem**: Vetinari agents are internal only. Cannot interoperate with external agent systems.

**Solution**: Implement A2A protocol server, exposing Vetinari agents as discoverable A2A agents with Agent Cards.

**Scope**: New `vetinari/a2a/` package

**Implementation**:
- Agent Card generation for each of the 3 agents (capabilities, modes, input/output schemas)
- A2A task lifecycle: create, execute, stream updates, complete
- JSON-RPC 2.0 over HTTPS transport
- Optional gRPC transport for low-latency internal communication
- Client capability: discover and invoke external A2A agents as additional tools

### 9.3 Multi-Language Coding Support (MEDIUM)

**Problem**: BuilderAgent generates Python only. No support for JavaScript/TypeScript, Go, Rust, or other languages.

**Solution**: Extend Builder with language-aware code generation and verification.

**Implementation**:
- Language detection from task context
- Language-specific prompt templates and style guides
- Language-specific verification (ESLint for JS/TS, `go vet` for Go, `cargo clippy` for Rust)
- Tree-sitter grammar loading per language for RepoMap
- Start with Python + TypeScript (highest demand), add others incrementally

### 9.4 Docker/Container Deployment (MEDIUM)

**Problem**: Runs only as a local Python process. No containerization, no horizontal scaling, no multi-user support.

**Solution**: Dockerize the system with proper configuration management.

**Implementation**:
- `Dockerfile` with multi-stage build (slim Python 3.10+ image)
- `docker-compose.yml` with Vetinari + model volume mounts as services
- Environment-based configuration (12-factor app)
- Health check endpoints for container orchestration
- Volume mounts for persistent state (plans, memory, cost DB)

### 9.5 Langfuse Evaluation Integration (LOW)

**Problem**: No automated evaluation of agent output quality beyond the built-in QualityAgent.

**Solution**: Integrate with Langfuse for LLM-as-judge evaluation, tracing, and continuous monitoring.

**Implementation**:
- Langfuse callback handler for all LLM calls
- Custom evaluation criteria per agent mode
- Dashboard for quality trends over time
- Alert on quality regression

### 9.6 Tool Gating (LOW)

**Problem**: All agents have access to all tools. No runtime per-agent tool permissions.

**Solution**: Implement tool permission matrix configurable per agent type.

**Implementation**:
- Tool permission enum: ALLOW, DENY, REQUIRE_APPROVAL
- Per-agent tool ACL in agent governance files
- Runtime enforcement in BaseAgent.execute_tool()
- Audit log for tool usage per agent

### 9.7 Distributed Multi-Instance Execution (LOW)

**Problem**: Vetinari runs with a single local model process. Users with multiple GPU hosts cannot distribute load.

**Solution**: Fan out agent execution across multiple llama-cpp-python worker processes or remote inference instances with load balancing.

**Implementation**:
- `InstanceRegistry` with health-checked worker endpoints
- Round-robin or least-loaded routing across instances
- Per-instance model availability tracking (different instances may have different models loaded)
- Graceful failover when an instance goes offline
- Prerequisite: async pipeline (9.1)

**Source**: Extracted from archived planning files.

### 9.8 External Log Shipping (LOW)

**Problem**: Structured JSON logging exists but logs are only written locally. No integration with centralized log aggregation systems.

**Solution**: Add configurable log shipping to external backends for production deployments.

**Implementation**:
- Pluggable log shipping backends: Elasticsearch, Splunk HEC, Datadog
- Configurable via YAML with endpoint, auth, and batch size
- Async buffered shipping to avoid blocking the pipeline
- Structured fields preserved through the shipping pipeline

**Source**: Extracted from archived planning files.

### 9.9 YAML-Configurable Permission Policies (LOW)

**Problem**: Per-agent permissions exist in `security.py` but are code-defined. Users cannot customize permissions without modifying source code.

**Solution**: Allow per-agent or per-project permission policies defined in YAML config files.

**Implementation**:
- Permission policy YAML schema with agent-level and project-level scopes
- Override hierarchy: project config > agent defaults > global defaults
- Runtime reload on config file change
- Validation on load with clear error messages for invalid policies

**Source**: Extracted from archived planning files.

**Exit criteria for v0.9.0**:
- Core pipeline operates in async mode with measurable latency improvement
- A2A Agent Cards discoverable by external systems
- Builder generates and verifies TypeScript code
- System runs in Docker container with docker-compose
- At least one evaluation metric tracked in Langfuse (or equivalent)

---

## v1.0.0 — Production Release

**Theme**: Polish, stability, and production-readiness. This is the "it just works" release.

### 10.1 Stability & Performance

- All known bugs resolved
- p95 latency for single-task plans under 30 seconds (local 72B model)
- p95 latency for 5-task plans under 120 seconds
- Memory usage stable under sustained load (no leaks)
- 100+ integration tests passing

### 10.2 Security Audit

- Third-party security review of sandbox execution
- Credential vault penetration testing
- Input validation coverage: 100% of web endpoints
- Rate limiting on all public-facing endpoints
- OWASP Top 10 compliance verification

### 10.3 Documentation Complete

- API reference generated from docstrings
- User guide with tutorials for common workflows
- Architecture deep-dive document
- Deployment guide (local, Docker, cloud)
- Contributing guide for external developers

### 10.4 Packaging & Distribution

- PyPI package (`pip install vetinari`)
- Versioned releases with semantic versioning
- Migration guide for each minor version upgrade
- Backward compatibility policy documented

### 10.5 Plugin System

- Plugin API for community agent modes
- Skill marketplace with discovery and installation
- Plugin sandboxing for security isolation
- Template generator for new plugins

### 10.6 Web UI Polish

- Built-in trace explorer UI with timeline visualization, span details, and log correlation
- Light theme completion (CSS variables partially defined, needs visual testing)
- Font size accessibility picker for WCAG compliance
- llama-cpp-python configuration guide (optimal GPU layer settings per model size)

---

## Architecture Decision Records

| ADR | Decision | Rationale | Date |
|-----|----------|-----------|------|
| ADR-001 | Agent Consolidation | 22 to 3 agents (Foreman/Worker/Inspector per ADR-0061); eliminates handoff losses | 2026-02 |
| ADR-002 | Flat Ensemble over Hierarchy | No sub-agent spawning; TwoLayerOrchestrator coordinates directly | 2026-02 |
| ADR-003 | Code Mode Orchestration | LLM generates Python code chaining agent APIs in sandbox | 2026-02 |
| ADR-004 | Circuit Breaker Pattern | Per-agent CLOSED/OPEN/HALF_OPEN prevents cascade failures | 2026-02 |
| ADR-005 | MultiModeAgent Pattern | Internal mode routing within agents; single class per cognitive role | 2026-02 |
| ADR-006 | File-Based Agent Jurisdiction | `.claude/agents/` + root AGENTS.md; clear ownership boundaries | 2026-03 |
| ADR-007 | Context Engineering | Just-in-time context injection, few-shot examples per mode | 2026-03 |
| ADR-0041 | Runtime Prompt Loading | Agent prompts from `.claude/agents/*.md` with mtime caching | 2026-03 |
| ADR-0042 | CLAUDE.md Decomposition | 668-line monolith split into 9 focused rule files | 2026-03 |
| ADR-0043 | Cross-Model Validation | Opt-in secondary model verification for critical decisions | 2026-03 |
| ADR-008 | Local-First Memory | SQLite + numpy vectors over external vector DB; no infra dependency | Proposed |
| ADR-009 | Async Migration Strategy | Adapters first, then agents, then orchestrator; sync fallback preserved | Proposed |
| ADR-010 | A2A over Custom Protocol | Google A2A for external interop; internal agents use direct dispatch | Proposed |

---

## Design Principles

These principles guide all architectural decisions:

1. **Local-first**: The system must work entirely offline with local GGUF models via llama-cpp-python. Cloud providers are optional escalation paths, never requirements.

2. **Start cheap, escalate on evidence**: Always begin with the smallest model that might work. Escalate only when measured confidence is low. Never default to the most expensive option.

3. **Observability is not optional**: Every LLM call, every agent decision, every cost must be tracked. If it is not measured, it does not exist.

4. **No stubs in production**: Code either works completely or does not exist. Placeholder implementations, TODO comments, and "stub mode" are technical debt that compounds.

5. **Fail-closed security**: When in doubt, deny. Credential vault refuses plaintext. Sandbox refuses unsafe operations. Auth decorators default to rejection.

6. **Three agents, flat ensemble**: Resist the urge to add agents. New capabilities become modes on existing agents. The coordination cost of N agents grows quadratically.

7. **Deterministic replay**: Given the same inputs and model weights, the system should produce the same outputs. Randomness must be seeded and controllable.

8. **Progressive trust**: Start with human-in-the-loop. Earn autonomy through demonstrated reliability. Never assume trust.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Async migration breaks existing tests | HIGH | HIGH | Phased migration (adapters then agents then orchestrator); sync fallback preserved |
| Local model quality insufficient for complex tasks | MEDIUM | HIGH | Cascade routing to cloud; quality gates catch failures; track success rates |
| MCP/A2A protocol specifications change | MEDIUM | MEDIUM | Abstract protocol layer; version pinning; adapter pattern |
| Vector memory adds latency to every agent call | MEDIUM | MEDIUM | Lazy loading; cache warm embeddings; skip for simple tasks |
| 40%+ agentic AI project cancellation rate (Gartner) | LOW | HIGH | Focus on reliability over features; extensive testing; real user feedback |
| Training pipeline requires GPU hardware | HIGH | LOW | Document as optional; training_manager clearly gates on GPU availability |

---

## Verification Strategy

After each version:

1. **Unit tests**: `python -m pytest tests/ -x -q` — all pass, zero new failures
2. **Import check**: `python -c "import vetinari; print('OK')"` — clean import
3. **Type verification**: `python -c "from vetinari.types import AgentType; print(list(AgentType))"` — only current values
4. **Integration tests** (v0.7.0+): `python -m pytest tests/integration/ -x -q --timeout=120`
5. **Stub audit**: Zero `NotImplementedError` or `stub` in production code paths
6. **Security audit**: Zero unreviewed bare except, eval, or exec instances
7. **Documentation audit**: All version numbers consistent across README, ROADMAP, CLAUDE.md, AGENTS.md
8. **Performance benchmark**: canonical runtime p50/p95/p99 latency within named thresholds, with hardware, backend, model, command, and sample size recorded. CI smoke gates do not satisfy this requirement.

---

## Appendix: Research References

Architecture patterns and best practices informing this roadmap:

- Multi-Agent Orchestration Patterns 2026 (ai-agentsplus.com) — Six core patterns: hierarchical, peer-to-peer, event-driven, sequential, consensus, dynamic routing
- Multi-Agent AI Systems: Enterprise Guide 2026 (neomanex.com) — Coordinator/specialist model, context preservation across handoffs, three-tier memory
- Google A2A Protocol (a2a-protocol.org) — Agent-to-agent interoperability standard, now under Linux Foundation governance
- MCP Architecture Patterns (developer.ibm.com) — Model Context Protocol for tool/resource integration in multi-agent systems
- AI Agent Orchestration Predictions 2026 (deloitte.com) — Progressive autonomy spectrum, enterprise adoption patterns
- Choosing Orchestration Patterns (kore.ai) — Supervisor, adaptive network, and custom patterns for multi-agent systems

---

*This roadmap is a living document. Update it when architecture decisions are made, features are completed, or priorities shift. The plan is not the territory — adjust based on what you learn.*
