# Vetinari Roadmap

**Authoritative planning document for Vetinari AI Orchestration System**
**Current Version: v0.5.0** | **Last Updated: 2026-03-11**

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

Vetinari is a **local-first multi-agent AI orchestration system** that decomposes complex goals into structured task graphs and executes them through specialist agents, primarily using local LLM models via LM Studio. The system aims to be the most capable open-source orchestrator for developers who want full control over their AI pipeline — running locally, routing intelligently across model tiers, and learning from every execution.

### North Star

A developer types a goal in natural language. Vetinari decomposes it, routes each subtask to the optimal local or cloud model, executes through specialist agents with quality gates, learns from the outcome, and delivers a verified result — all with full observability, deterministic replay, and sub-dollar cost for most workflows.

### Core Differentiators

1. **Local-first**: Runs entirely on LM Studio with optional cloud escalation — no mandatory API keys
2. **Six-agent cognitive pipeline**: Plan > Research > Advise > Build > Verify > Operate
3. **Cascade cost routing**: Start cheap (7B), escalate only on low confidence — 14% improvement measured
4. **Self-improving**: Thompson Sampling model selection, prompt evolution, workflow learning
5. **Full observability**: OpenTelemetry tracing, SQLite cost tracking, real-time dashboard

---

## Current Architecture

### Six-Agent Consolidated Architecture (ADR-001)

| Agent | Modes | Cognitive Role |
|-------|-------|----------------|
| **PlannerAgent** | plan, clarify, summarise, prune, extract, consolidate (6) | Decompose goals into task DAGs |
| **ResearcherAgent** | code_discovery, domain_research, api_lookup, lateral_thinking, ui_design, database, devops, git_workflow (8) | Gather evidence before decisions |
| **OracleAgent** | architecture, risk_assessment, ontological_analysis, contrarian_review (4) | Deliberate and produce ADRs |
| **BuilderAgent** | build, image_generation (2) | Sole production code writer |
| **QualityAgent** | code_review, security_audit, test_generation, simplification (4) | Mandatory pass/fail gate |
| **OperationsAgent** | documentation, creative_writing, cost_analysis, experiment, error_recovery, synthesis, improvement, monitor, devops_ops (9) | Docs, synthesis, sustainability |

**Total: 33 modes across 6 agents**

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
  two_layer_orchestration.py     # Main execution engine
  cascade_router.py              # Cost-optimised model escalation
  agents/
    base_agent.py                # BaseAgent with circuit breakers, token budgets
    multi_mode_agent.py          # MultiModeAgent with MODES routing
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
  memory/dual_memory.py          # DualMemoryStore (primary memory)
  learning/                      # Self-improvement (Thompson Sampling, feedback, etc.)
  adapters/                      # Multi-provider adapters (LM Studio, OpenAI, etc.)
  schemas/agent_outputs.py       # Pydantic output schemas (33 modes)
  observability/tracing.py       # OpenTelemetry distributed tracing
  analytics/cost_tracker.py      # SQLite cost tracking
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

- **ADR-001**: Consolidated 22 agents into 6 multi-mode agents (research: 5-7 optimal)
- **ADR-002**: Flat ensemble over hierarchy — no sub-agent spawning
- Typed Pydantic output schemas for all 33 modes
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

### Phase 10: Compat Shim Elimination (v0.5.1 — PR #10)

- Removed all 280+ stale legacy agent references across 31 files
- Updated all agent name mappings: EXPLORER to RESEARCHER, EVALUATOR to QUALITY, etc.
- Updated "22 agents" references to "6 consolidated agents"
- Updated test fixtures and benchmark cases
- Net reduction: -288 lines across 31 files

---

## Current State Assessment

### Codebase Metrics (as of 2026-03-11)

| Metric | Value |
|--------|-------|
| Source files (`vetinari/`) | 297 Python files |
| Source lines | 83,676 |
| Test files (`tests/`) | 105 Python files |
| Test lines | 54,742 |
| Subdirectories | 35 packages |
| Agent modes | 33 across 6 agents |
| Test count | 5,119+ (0 failures) |

### Critical Gaps Identified

#### 1. Structural Debt (HIGH - blocks everything)

**23 flat duplicate files** remain in `vetinari/` root alongside their proper subdirectory counterparts. These are the same modules that exist in `vetinari/memory/`, `vetinari/planning/`, `vetinari/models/`, etc. — creating import confusion and dead weight. Estimated 8,000-12,000 lines of pure duplication.

**20+ legacy agent files** still exist in `vetinari/agents/` (e.g., `architect_agent.py`, `explorer_agent.py`, `librarian_agent.py`, `evaluator_agent.py`) alongside the consolidated agents they were absorbed into. These are dead code from Phase 2 that survived consolidation.

**`compat.py`** backward-compatibility shim still exists despite all references being updated in PR #10.

#### 2. Async Gap (HIGH - performance ceiling)

Only **44 async/await occurrences across 8 files**, all in `async_support/`. The core pipeline (`TwoLayerOrchestrator`, all 6 agents, all adapters) is entirely synchronous. This means:
- Pipeline stages block on each LLM call
- No streaming between stages
- ThreadPoolExecutor is the only parallelism mechanism
- Cannot leverage modern async LLM client libraries

#### 3. Memory Architecture (MEDIUM - quality ceiling)

Current memory is a simple key-value `DualMemoryStore` with no semantic search capability. The RAG subsystem (`vetinari/rag/`) contains only a single `knowledge_base.py` file. No vector embeddings, no similarity search, no temporal knowledge graphs. This limits the system's ability to learn from past executions and retrieve relevant context.

#### 4. MCP Integration (MEDIUM - interoperability)

MCP subsystem (`vetinari/mcp/`) has 3 files with stub handlers. Not connected to the main execution pipeline. Given MCP's adoption (97M+ monthly SDK downloads, Linux Foundation governance), this is a significant interoperability gap.

#### 5. Training Pipeline (LOW - future capability)

`vetinari/training/pipeline.py` is a single-file stub. The `learning/training_manager.py` explicitly notes it operates in "stub mode." No actual QLoRA or fine-tuning capability exists.

#### 6. Remaining Stubs

| Location | Issue |
|----------|-------|
| `optimization/batch_processor.py:215` | Returns placeholder response |
| `learning/training_manager.py:343` | Job created in stub mode |
| `mcp/tools.py:111` | Default handlers are stubs |
| `benchmarks/toolbench.py:278` | Agent-based tool selection not wired |

#### 7. Documentation Drift

- ROADMAP.md said v0.4.0, README.md said v0.5.0 — now aligned
- `docs/IMPLEMENTATION_ROADMAP.md` and `docs/ARCHITECTURE.md` — removed (content consolidated into ROADMAP.md and `.claude/docs/architecture.md`)
- README says "Python 3.9+", CLAUDE.md says "Python 3.10+"

---

## v0.6.0 — Structural Integrity

**Theme**: Clean the house. Remove all dead code, resolve duplicates, standardize naming, fix documentation. This is the foundation everything else builds on.

**Target**: 15-25% line count reduction from 83,676. Zero duplicate modules. Zero legacy agent files. Zero stubs in production paths.

### 6.1 Duplicate Module Resolution (CRITICAL)

Delete all 23 flat duplicate files in `vetinari/` root. The subdirectory versions are canonical. For each pair:

1. Diff flat vs subdirectory version
2. Merge any unique code from flat into subdirectory
3. Delete flat file
4. Update all imports to use subdirectory path
5. Run tests after each batch

**Files to delete** (after merge):
`shared_memory.py`, `blackboard.py`, `planning_engine.py`, `model_pool.py`, `goal_verifier.py`, `decomposition.py`, `verification.py`, `agent_affinity.py`, `dynamic_model_router.py`, `lmstudio_adapter.py`, `vram_manager.py`, `plan_api.py`, `plan_mode.py`, `plan_types.py`, `assignment_pass.py`, `subtask_tree.py`, `validator.py`, `ponder.py`, `planning.py`, `multi_agent_orchestrator.py`, `explain_agent.py`, `decomposition_agent.py`, `builder.py`

**Estimated savings**: 8,000-12,000 lines

### 6.2 Legacy Agent File Deletion (CRITICAL)

Delete all pre-consolidation agent files from `vetinari/agents/`:

| Legacy File | Absorbed By |
|-------------|-------------|
| `architect_agent.py` | Oracle |
| `context_manager_agent.py` | Planner |
| `cost_planner_agent.py` | Operations |
| `data_engineer_agent.py` | Researcher |
| `decomposition_agent.py` | Planner |
| `devops_agent.py` | Researcher + Operations |
| `documentation_agent.py` | Operations |
| `error_recovery_agent.py` | Operations |
| `evaluator_agent.py` | Quality |
| `experimentation_manager_agent.py` | Operations |
| `explain_agent.py` | Operations |
| `explorer_agent.py` | Researcher |
| `image_generator_agent.py` | Builder |
| `improvement_agent.py` | Operations |
| `librarian_agent.py` | Researcher |
| `oracle_agent.py` | `consolidated/oracle_agent.py` |
| `researcher_agent.py` | `consolidated/researcher_agent.py` |
| `security_auditor_agent.py` | Quality |
| `synthesizer_agent.py` | Operations |
| `test_automation_agent.py` | Quality |
| `ui_planner_agent.py` | Researcher |
| `user_interaction_agent.py` | Planner |
| `version_control_agent.py` | Researcher |
| `compat.py` | No longer needed |

Also delete legacy skill wrappers in `vetinari/skills/` that wrap absorbed agents:
`cost_planner/`, `data_engineer/`, `documentation/`, `evaluator/`, `experimentation_manager/`, `explorer/`, `librarian/`, `security_auditor/`, `synthesizer/`, `test_automation/`, `ui_planner/`

### 6.3 Naming Convention Standardization (HIGH)

Codify and enforce naming standards:

**Ambiguous file renames**:
- `code_mode/engine.py` to `code_mode/code_mode_engine.py`
- `coding_agent/engine.py` to `coding_agent/coding_engine.py`
- `constraints/registry.py` to `constraints/constraint_registry.py`
- `adapters/registry.py` to `adapters/adapter_registry.py`

**Class naming standard**: `{Role}Agent`, `{Backend}MemoryStore`, `{Strategy}Router`

**AgentType enum cleanup**: Remove all legacy values (EXPLORER, LIBRARIAN, EVALUATOR, SYNTHESIZER, UI_PLANNER, SECURITY_AUDITOR, DATA_ENGINEER, DOCUMENTATION_AGENT). Keep only the 6 current agents.

### 6.4 Stub Resolution (HIGH)

For every remaining stub: implement fully OR delete entirely. No stub survives v0.6.0.

| Stub | Resolution |
|------|------------|
| `batch_processor.py:215` placeholder response | Implement real batch dispatch or remove |
| `training_manager.py:343` stub mode | Document as "requires GPU backend" with clear setup guide, remove pretend-success |
| `mcp/tools.py:111` stub handlers | Wire to real Vetinari subsystems (see 7.4) |
| `toolbench.py:278` agent tool selection | Wire to ResearcherAgent or remove benchmark |

### 6.5 Documentation Alignment (HIGH)

- Fix Python version everywhere to "3.10+"
- ~~Update `docs/ARCHITECTURE.md` from v0.3.0 to current~~ — removed; architecture lives in `.claude/docs/architecture.md`
- ~~Consolidate `docs/IMPLEMENTATION_ROADMAP.md` into this ROADMAP.md~~ — already removed
- Update README project structure tree to match post-cleanup reality
- Update file/test counts in all docs
- Create `CHANGELOG.md` with entries for all completed phases

### 6.6 Quality of Life

- Resolve all remaining TODO/FIXME/HACK comments (implement or remove)
- Clean imports: remove unused, fix order, use canonical sources
- Ensure all loggers use `logging.getLogger(__name__)`
- Apply Python 3.10+ idioms: `X | Y`, `list[str]`, match/case

**Exit criteria for v0.6.0**:
- `python -m pytest tests/ -x -q` passes with zero failures
- `grep -r "NotImplementedError" vetinari/ --include="*.py"` returns zero hits in production code paths
- Zero flat/subdirectory duplicate pairs remain
- Zero legacy agent files remain
- All docs reference current architecture only

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

**Solution**: Create integration test suite that runs against a local LM Studio instance (skipped in CI when unavailable).

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
- Support MCP transport over stdio and HTTP+SSE
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

**Exit criteria for v0.7.0**:
- Plans survive process restart via checkpointing
- Hard budget limits prevent runaway execution
- At least 5 end-to-end integration tests pass against local LM Studio
- MCP server responds to tool discovery and basic invocations
- Zero critical web UI bugs

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

**Exit criteria for v0.8.0**:
- Semantic search returns relevant past episodes by meaning
- Quality gate failures trigger structured reflection before retry
- At least one task type operates in human-out-of-the-loop mode
- RAG pipeline indexes a sample Python project and retrieves relevant chunks
- Model routing decisions correlate with actual success rates (above 70% calibration)

---

## v0.9.0 — Interoperability & Scale

**Theme**: Open the system up. A2A protocol support, async pipeline, multi-language coding, and the foundations for multi-user deployment.

### 9.1 Async Pipeline (HIGH)

**Problem**: Core pipeline is synchronous. Blocks on every LLM call. Cannot stream results between stages. ThreadPoolExecutor is the only parallelism.

**Solution**: Convert core pipeline to async/await. This is a large refactor touching the orchestrator, all adapters, and agent dispatch.

**Implementation**:
- Phase 1: Async adapters (`aiohttp` for LM Studio, async client libs for cloud providers)
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
- Agent Card generation for each of the 6 agents (capabilities, modes, input/output schemas)
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
- `docker-compose.yml` with Vetinari + LM Studio services
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

---

## Architecture Decision Records

| ADR | Decision | Rationale | Date |
|-----|----------|-----------|------|
| ADR-001 | 6-Agent Consolidation | 22 to 6 agents (5-7 optimal per research); reduces coordination overhead | 2026-02 |
| ADR-002 | Flat Ensemble over Hierarchy | No sub-agent spawning; TwoLayerOrchestrator coordinates directly | 2026-02 |
| ADR-003 | Code Mode Orchestration | LLM generates Python code chaining agent APIs in sandbox; eliminates intermediate round-trips | 2026-02 |
| ADR-004 | Circuit Breaker Pattern | Per-agent CLOSED/OPEN/HALF_OPEN prevents cascade failures | 2026-02 |
| ADR-005 | MultiModeAgent Pattern | Internal mode routing within agents; single class per cognitive role | 2026-02 |
| ADR-006 | File-Based Agent Jurisdiction | `.claude/agents/` + root AGENTS.md; clear ownership boundaries | 2026-03 |
| ADR-007 | Context Engineering | Just-in-time context injection, few-shot examples per mode | 2026-03 |
| ADR-008 | Local-First Memory | SQLite + numpy vectors over external vector DB; no infra dependency | Proposed |
| ADR-009 | Async Migration Strategy | Adapters first, then agents, then orchestrator; sync fallback preserved | Proposed |
| ADR-010 | A2A over Custom Protocol | Google A2A for external interop; internal agents use direct dispatch | Proposed |

---

## Design Principles

These principles guide all architectural decisions:

1. **Local-first**: The system must work entirely offline with LM Studio. Cloud providers are optional escalation paths, never requirements.

2. **Start cheap, escalate on evidence**: Always begin with the smallest model that might work. Escalate only when measured confidence is low. Never default to the most expensive option.

3. **Observability is not optional**: Every LLM call, every agent decision, every cost must be tracked. If it is not measured, it does not exist.

4. **No stubs in production**: Code either works completely or does not exist. Placeholder implementations, TODO comments, and "stub mode" are technical debt that compounds.

5. **Fail-closed security**: When in doubt, deny. Credential vault refuses plaintext. Sandbox refuses unsafe operations. Auth decorators default to rejection.

6. **Six agents, flat ensemble**: Resist the urge to add agents. New capabilities become modes on existing agents. The coordination cost of N agents grows quadratically.

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
8. **Performance benchmark**: p50/p95/p99 latency within acceptable thresholds

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
