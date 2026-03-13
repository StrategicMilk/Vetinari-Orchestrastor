# Changelog

All notable changes to Vetinari are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Security

- Removed `.env`, `vault/.key`, and generated `outputs/`/`projects/`/`logs/` from git tracking
- Extracted `api_key` literal in DSPy optimizer to env-var-backed constant

### Added

- LLM Guard safety scanner integration (`vetinari/safety/llm_guard_scanner.py`) with ML-based input/output scanning; graceful degradation when llm-guard not installed
- LLM Guard configuration (`config/llm_guard.yaml`) with context-based scanning rules
- Pydantic Settings configuration layer (`vetinari/config/settings.py`) with env-var support (`VETINARI_` prefix) and validated inference profile loading
- `pydantic-settings>=2.0` dependency for typed configuration management
- 35 Architecture Decision Records (ADR-0001 through ADR-0035)
- OpenTelemetry GenAI semantic convention attributes in `otel_genai.py`
- Cross-reference docstrings between orchestration and resilience circuit breakers

### Changed

- Modernized typing across codebase: `Dict`/`List`/`Optional` to `dict`/`list`/`X | None`
- Added `from __future__ import annotations` to all `vetinari/` files
- Fixed 60+ f-string logger calls to %-style formatting
- Added `encoding="utf-8"` to all `open()` calls missing it
- Migrated `os.path` usage to `pathlib.Path` in `inference_config.py` and `dspy_optimizer.py`
- Canonical imports enforced: enums from `vetinari.types`, not re-exported from `contracts`
- Removed outdated `docs/ARCHITECTURE.md` and `docs/DEVELOPER_GUIDE.md`
- Consolidated 3 memory docs into single `docs/memory.md`

### Fixed

- Bare `except` clauses replaced with specific exception types
- Inline `import re` moved to module level in `inference_config.py`
- Ruff compliance: removed unused imports, fixed docstring formats

---

## [0.5.0] — 2026-03-11

### Security

- **Thread safety**: `SharedMemory` now uses `RLock` for reentrant locking; `web_ui.py` global state guarded with `threading.Lock`
- **Fail-closed credential vault**: `CredentialVault` raises `VaultError` on decryption failure instead of returning partial/plaintext data
- **Traceback leak prevention**: unhandled exceptions return generic `500` responses in production; full tracebacks logged server-side only
- **Constant-time token comparison**: `hmac.compare_digest` enforced on all token verification paths
- **Unauthenticated endpoint fix**: `/api/sandbox/execute`, `/api/models`, and `/api/plans` POST now require valid session token
- **Trusted proxy configuration**: `X-Forwarded-For` header validated against a configurable trusted proxy list
- **Rate limiting**: 10 req/60s per client on sandbox execution endpoints
- **Input validation**: `validate_json_fields` applied at all API boundaries; malformed requests rejected before business logic
- **Auth decorators**: `require_admin` applied to ADR, decomposition, ponder, rules, and training routes
- **MCP protocol update**: MCP subsystem updated to protocol version `2025-11-25`; transport negotiation validates `protocolVersion` on handshake
- **Circuit breaker at graph level**: `AgentGraph` now has its own circuit breaker — a failing agent no longer cascades to healthy pipeline stages
- **Stagnation detection**: `TwoLayerOrchestrator` detects plans with no task progress after a configurable timeout and triggers error recovery
- **Inter-agent guardrails**: all agent-to-agent messages validated through `RailContext.INTERNAL_AGENT` guardrail context
- **Configurable quality gates**: Quality gate thresholds moved to `config/quality_gates.yaml`; defaults preserved; overridable per deployment

### Added

- Analytics REST API: 7 endpoints (`/api/analytics/{cost,sla,anomalies,forecast,models,agents,summary}`)
- `CascadeRouter` with tiered model escalation (7B → 30B → 72B) and heuristic confidence estimation
- `BatchProcessor` with Anthropic and OpenAI batch API backends (50% cost discount on eligible requests)
- File-based agent governance: `.claude/agents/` directory with 6 agent definition files
- Root `AGENTS.md` with full architecture specification, file jurisdiction map, quality gates, and delegation rules
- `agent_skill_map.json`: 6 consolidated + 20 legacy entries, 12 workflow pipeline definitions
- `skills_registry.json`: 8 skills, cascade routing config, batch processing config
- Enriched `error_recovery` mode in `ConsolidatedOperationsAgent` with pattern-based diagnostics and remediation strategies
- 28 `AGENT_REGISTRY` entries in `vetinari/agents/contracts.py`

### Changed

- All 280+ stale legacy agent references updated across 31 files (PR #10)
- Agent name mappings updated: `EXPLORER` → `RESEARCHER`, `EVALUATOR` → `QUALITY`, etc.
- "22 agents" references updated to "6 consolidated agents" throughout docs and code
- Test fixtures and benchmark cases updated to use consolidated agent types
- Net line reduction: −288 lines across 31 files from compat shim elimination

---

## [0.4.0] — 2026-02

### Added — Agent Consolidation (Phase 3)

- **ADR-001**: Consolidated 22 legacy agents into 6 multi-mode agents following research showing 5–7 agents optimal for full-system orchestration
- **ADR-002**: Flat ensemble over hierarchy — `TwoLayerOrchestrator` coordinates agents directly; no sub-agent spawning
- Six consolidated agents with 33 total modes:
  - `PlannerAgent` (6 modes): plan, clarify, summarise, prune, extract, consolidate
  - `ConsolidatedResearcherAgent` (8 modes): code_discovery, domain_research, api_lookup, lateral_thinking, ui_design, database, devops, git_workflow
  - `ConsolidatedOracleAgent` (4 modes): architecture, risk_assessment, ontological_analysis, contrarian_review
  - `BuilderAgent` (2 modes): build, image_generation
  - `QualityAgent` (4 modes): code_review, security_audit, test_generation, simplification
  - `ConsolidatedOperationsAgent` (9 modes): documentation, creative_writing, cost_analysis, experiment, error_recovery, synthesis, improvement, monitor, devops_ops
- Typed Pydantic output schemas for all 33 modes (`vetinari/schemas/agent_outputs.py`)
- Circuit breakers per agent: CLOSED / OPEN / HALF_OPEN state machine
- Per-agent token budgeting: warn at 80%, hard truncate at 100%
- `DynamicModelRouter` with Thompson Sampling and capability matching
- SLM/LLM hybrid routing, speculative decoding, and continuous batching support
- SQLite-backed cost tracking and benchmark framework
- `TwoLayerOrchestrator` as the single execution engine replacing the previous assembly-line orchestrator
- 8 stub remediations: sandbox hooks, log aggregator, verification, cost estimation, MODES validation, CWE patterns, SVG metadata, upgrader

### Changed

- `AgentType` enum updated: legacy values deprecated; 6 canonical values active
- All agent dispatch tables updated to use consolidated agent classes

---

## [0.3.0] — 2026-01

### Added — Multi-Agent System & Web UI (Phase 2)

- 22 specialised agents covering all cognitive domains (later consolidated in v0.4.0)
- Assembly-line 7-stage execution pipeline with DAG scheduler
- `DualMemoryStore` with blackboard system: `request_help`, `claim_entry`, `publish_finding`, consensus voting
- Self-improvement system: `QualityScorer`, `FeedbackLoop`, Thompson Sampling, `PromptEvolver`, `WorkflowLearner`, `CostOptimizer`, `AutoTuner`
- Flask web dashboard with Chart.js visualisations, SSE streaming, dark mode, real-time task progress
- Structured JSON logging, OpenTelemetry distributed tracing, alert system
- Multi-source web search: DuckDuckGo, Wikipedia, arXiv with two-source anti-hallucination verification
- `vetinari/adapters/` supporting 7 providers: LM Studio, OpenAI, Anthropic (with prompt caching), Google Gemini, Cohere, HuggingFace, Replicate
- RepoMap: tree-sitter-inspired structural codebase mapping — sends function signatures instead of raw file contents
- Token optimiser: per-task budgets, dynamic `max_tokens`, context deduplication, local LLM preprocessing (30–60% cloud token reduction)

### Changed

- Flask web server (`vetinari/web_ui.py`) replaces CLI-only interface as primary entry point
- `python -m vetinari` now starts the web server by default

---

## [0.2.0] — 2025-12

### Added — Planning Engine & Memory System (Phase 1 continued)

- `PlanningEngine` with goal decomposition, wave sequencing, and dependency graph validation
- `DualMemoryStore`: episodic (execution traces) and semantic (learned patterns) memory tiers
- `SharedMemory` blackboard for inter-agent communication within a plan execution
- `vetinari/constraints/` package with constraint definitions and runtime enforcement
- `vetinari/safety/` package with content filters and guardrail policies
- Checkpoint-based recovery: plan state serialised after each wave; `--resume` flag for crash recovery
- `vetinari/analytics/cost_tracker.py`: SQLite-backed per-call cost attribution
- `vetinari/observability/tracing.py`: OpenTelemetry span generation for all LLM calls
- `vetinari/learning/` package: feedback loop, quality scoring, auto-tuner stubs

### Changed

- `AdapterManager` extended to support all 7 providers
- Configuration moved from hardcoded values to `config/` YAML files

---

## [0.1.0] — 2025-11

### Added — Initial Release (Phase 1 foundation)

- `LMStudioAdapter`: HTTP client for LM Studio REST API; model enumeration, chat completions, streaming
- `ExecutionContext` system with mode enforcement: PLANNING, EXECUTION, SANDBOX
- `ToolRegistry` with `ToolMetadata` and `ToolParameter` for structured tool definitions
- `AdapterManager` for provider-agnostic model access
- Verification pipeline: `CodeSyntaxVerifier`, `SecurityVerifier`, `ImportVerifier`, `JSONStructureVerifier`
- Enhanced CLI with 16 subcommands and rich visual feedback (progress bars, coloured output)
- `vetinari/exceptions.py`: custom exception hierarchy (`VetinariError`, `AgentError`, `PlanError`, `VaultError`)
- `vetinari/types.py`: canonical enum source for `AgentType`, `TaskStatus`, `ExecutionMode`, `PlanStatus`
- `vetinari/agents/contracts.py`: `AgentSpec`, `Task`, `Plan`, `AgentResult` dataclasses and `AGENT_REGISTRY`
- `vetinari/agents/interfaces.py`: `AgentInterface` ABC
- `vetinari/agents/base_agent.py`: `BaseAgent` base class with token budgeting and retry logic
- Basic `pyproject.toml` and `requirements.txt`
- MIT licence

---

[Unreleased]: https://github.com/StrategicMilk/Vetinari/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/StrategicMilk/Vetinari/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/StrategicMilk/Vetinari/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/StrategicMilk/Vetinari/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/StrategicMilk/Vetinari/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/StrategicMilk/Vetinari/releases/tag/v0.1.0
