# Vetinari Implementation Roadmap

**Version:** v0.5.0 | **Updated:** 2026-03-10

---

## Completed Phases

### Phase 1: Foundation (v0.1.0)

1. **ExecutionContext System** (`vetinari/execution_context.py`)
   - ExecutionMode enum (PLANNING, EXECUTION, SANDBOX)
   - ToolPermission enum with 12 permission types
   - ContextManager with context stacking and enforcement
   - Pre/post-execution hooks and audit trails

2. **Tool Interface** (`vetinari/tool_interface.py`)
   - Abstract Tool base class, ToolMetadata, ToolParameter dataclasses
   - ToolCategory enum, ToolRegistry for management
   - VerificationResult and ToolResult dataclasses

3. **AdapterManager** (`vetinari/adapter_manager.py`)
   - Multi-provider adapter management (LMStudio, OpenAI, Anthropic, Gemini, Cohere, Ollama)
   - ProviderMetrics tracking, intelligent model selection, fallback support

4. **Verification Pipeline** (`vetinari/verification.py`)
   - CodeSyntaxVerifier, SecurityVerifier, ImportVerifier, JSONStructureVerifier

5. **Enhanced CLI** (`cli.py`) -- 16 subcommands with rich visual feedback

### Phase 2: Agent System (v0.2.0-v0.3.0)

1. **22 Specialized Agents** -- Full agent ecosystem built
   - Planner, Explorer, Oracle, Librarian, Researcher, Evaluator, Synthesizer, Builder
   - UI Planner, Security Auditor, Data Engineer, Documentation, Cost Planner
   - Test Automation, Experimentation Manager, Improvement Agent
   - DevOps, Version Control, Error Recovery, Context Manager, Image Generator, Ponder

2. **Assembly-Line Pipeline** (`vetinari/two_layer_orchestration.py`)
   - 7-stage pipeline: Input Analysis > Plan > Decompose > Assign > Execute > Review > Assemble
   - DAG scheduler with ThreadPoolExecutor for parallel execution
   - Recursive task decomposition with depth cap (16)

3. **Self-Improvement System** (`vetinari/learning/`)
   - QualityScorer (LLM-as-judge + heuristics)
   - FeedbackLoop with EMA updates to ModelPerformance
   - Thompson Sampling model selection with Beta distributions
   - PromptEvolver for A/B testing prompt variants
   - WorkflowLearner for domain-specific decomposition strategies
   - CostOptimizer for cost-aware routing
   - AutoTuner for SLA-driven adjustment

4. **Dashboard & Observability**
   - Flask web dashboard with Chart.js, dark mode
   - SSE streaming for real-time task progress
   - Structured JSON logging with trace correlation
   - OpenTelemetry distributed tracing
   - Alert system with deduplication and webhook channels
   - Performance test suite with p50/p95/p99 benchmarks

5. **Memory & Communication**
   - DualMemoryStore (SharedMemory consolidation)
   - Blackboard system with request_help(), claim_entry(), publish_finding()
   - Consensus voting via request_consensus()

### Phase 3: Agent Consolidation (v0.4.0)

1. **22 Agents Consolidated to 6** (ADR-001)
   - PlannerAgent (6 modes): plan, clarify, summarise, prune, extract, consolidate
   - ResearcherAgent (8 modes): code_discovery, domain_research, api_lookup, lateral_thinking, ui_design, database, devops, git_workflow
   - OracleAgent (4 modes): architecture, risk_assessment, ontological_analysis, contrarian_review
   - BuilderAgent (2 modes): build, image_generation
   - QualityAgent (4 modes): code_review, security_audit, test_generation, simplification
   - OperationsAgent (9 modes): documentation, creative_writing, cost_analysis, experiment, error_recovery, synthesis, improvement, monitor, devops_ops

2. **Agent Infrastructure**
   - Typed Pydantic output schemas for all 33 modes
   - Circuit breakers per-agent (CLOSED/OPEN/HALF_OPEN states)
   - Per-agent token budgeting with warn/truncate
   - Context window management with token tracking
   - Quality gates per agent type with thresholds

3. **Model Layer Enhancements**
   - DynamicModelRouter with Thompson Sampling + capability matching
   - SLM/LLM hybrid routing, dynamic complexity routing
   - Speculative decoding (draft-verify pattern)
   - Continuous batching (thread-safe inference queue)
   - SQLite-backed cost tracking
   - Benchmark framework (BenchmarkRunner/Case/Report)

4. **Stub Remediations** (B1-B8)
   - Sandbox hooks, log aggregator send(), verification base, cost estimation
   - MODES validation, CWE patterns, SVG metadata, upgrader install

### Phase 4: Security Hardening (v0.5.0)

1. **Critical Security Fixes** (P1.C1-C4, P1.H1-H10)
   - `hmac.compare_digest` for constant-time token comparison (`vetinari/web/__init__.py`)
   - `require_admin` decorator on mutating web endpoints
   - `validate_json_fields` helper for API input validation
   - Trusted proxy configuration for X-Forwarded-For validation
   - Auth decorators on ADR, decomposition, ponder, rules, training routes

2. **Rate Limiting** (`vetinari/sandbox.py`)
   - 10 requests per 60 seconds per client on sandbox execution paths
   - Per-client tracking with automatic window expiry

3. **Credential Vault** (`vetinari/credentials.py`)
   - Fail-closed Fernet encryption (refuses plaintext fallback)
   - Secure key derivation and storage

4. **Dashboard REST API Auth** (`vetinari/dashboard/rest_api.py`)
   - Auth token validation on admin endpoints
   - Consistent error responses for unauthorized access

### Phase 5: Analytics Pipeline (v0.5.0)

1. **7 Analytics REST Endpoints** (`vetinari/web/analytics_routes.py`)
   - `/api/analytics/cost` -- Cost breakdown by model, agent, time period
   - `/api/analytics/sla` -- SLA compliance metrics and violations
   - `/api/analytics/anomalies` -- Detected anomalies in system behavior
   - `/api/analytics/forecast` -- Cost and usage forecasting
   - `/api/analytics/models` -- Per-model performance and cost stats
   - `/api/analytics/agents` -- Per-agent utilization and quality metrics
   - `/api/analytics/summary` -- System-wide analytics summary

### Phase 6: Cost-Optimised Routing (v0.5.0)

1. **Cascade Router** (`vetinari/cascade_router.py`)
   - CascadeRouter with tiered model escalation (small > medium > large)
   - CascadeTier and CascadeResult dataclasses
   - Heuristic confidence estimation per response
   - Configurable confidence threshold (default: 0.7)
   - Automatic escalation on low confidence

2. **Batch Processor** (`vetinari/adapters/batch_processor.py`)
   - BatchProcessor with Anthropic and OpenAI batch API backends
   - Queue non-urgent inference for 50% cost discount
   - Configurable batch window and max batch size

### Phase 7: Agent Governance (v0.5.0)

1. **File-Based Agent Definitions** (`.claude/agents/`)
   - 6 governance files: planner.md, researcher.md, oracle.md, builder.md, quality.md, operations.md
   - YAML frontmatter with name, description, tools, model, permissions
   - Mode definitions, file jurisdiction, collaboration rules

2. **Root AGENTS.md**
   - Architecture overview, agent roster, file jurisdiction map
   - Three-role pattern, delegation rules, quality gates
   - Resource constraints, collaboration matrix, legacy deprecation

3. **Project CLAUDE.md**
   - Build/test commands, project conventions, architecture overview
   - Key file locations, development workflow

### Phase 8: Registry & Configuration (v0.5.0)

1. **Agent Skill Map** (`vetinari/config/agent_skill_map.json`)
   - 6 consolidated agent entries with modes, absorbs, skills
   - 20 legacy agent entries preserved for backward compatibility
   - 12 workflow pipelines (8 legacy + 4 consolidated)
   - Environment overrides (dev, staging, prod)

2. **Skills Registry** (`vetinari/skills_registry.json`)
   - 8 skills with capabilities, triggers, contexts
   - agent_skill_mapping section mapping 6 agents to skills/modes
   - cascade_routing and batch_processing configuration
   - Workflow templates, orchestration features, skill dependencies

3. **AgentSpec Expansion** (`vetinari/agents/contracts.py`)
   - 28 AGENT_REGISTRY entries (22 legacy + 6 consolidated)
   - AgentInterface contracts for all 6 target agents

### Phase 9: Operations Agent Enrichment (v0.5.0)

1. **Error Pattern Registry** (`vetinari/agents/consolidated/operations_agent.py`)
   - Enriched error_recovery mode with pattern-based diagnostics
   - Error classification and remediation strategies

---

## Remaining Work

### Priority 2: Architecture Fixes
- P2.1: Web task execution bypasses agent pipeline
- P2.2: Duplicate enum definitions (TaskStatus, AgentType)
- P2.3: Plan approval not enforced in web flow
- P2.7: Web UI frontend bugs (20 issues)
- P2.8: Five coexisting planning systems consolidation
- P2.9: ModelSearchEngine returns hardcoded data

### Priority 4: Model Configuration
- P4.1-P4.4: Per-task inference profiles with hot-reload
- P4.5: Sampling parameter gaps across adapters

### Priority 5: Code Quality (Remaining)
- P5.1: Bare except clause audit
- P5.2: Replace 5 LLM calls with algorithms
- P5.7: Agent prompt quality improvement (40+ lines/mode target)
- P5.8: 16 unaddressed Python files
- P5.9: 6 missing package dependencies

### Priority 6: Safety & Guardrails
- P6.1: Input/output guardrails at trust boundaries
- P6.2: Sensitive data detection in output (PII, API keys)

### Priority 7: Structural Cleanup
- P7.1: Merge duplicate planning modules
- P7.5: Consolidate 5 planning systems into single path
- P7.6: Project directory reorganization

---

## Future Roadmap

### v0.6.0 -- Observability & Evaluation
| Item | Description |
|------|-------------|
| P10.1 | OpenTelemetry GenAI semantic conventions |
| P10.10 | Intermediate step evaluation (PlanQualityMetric, PlanAdherenceMetric) |
| P10.12 | CI/CD continuous evaluation pipeline |
| P10.11 | Safety-specific benchmark suite (ToolEmu-style) |
| P10.17 | Session replay for debugging |

### v0.7.0 -- Advanced Architecture
| Item | Description |
|------|-------------|
| P9.1 | Async/await pipeline stages |
| P9.2 | Multi-turn conversation memory |
| P9.3 | Streaming pipeline with backpressure |
| P10.2 | Google A2A protocol support |
| P10.9 | Temporal knowledge graph memory |
| P10.13 | Structured reflection/self-correction pattern |
| P10.14 | Cyclical state machine graphs |
| P10.16 | Progressive autonomy spectrum |

---

## Architecture Decision Records

| ADR | Decision | Rationale |
|-----|----------|-----------|
| ADR-001 | 6-Agent Consolidation | 22 merged to 6 (5-7 optimal per research) |
| ADR-002 | Flat Ensemble over Hierarchy | TwoLayerOrchestrator coordinates directly |
| ADR-003 | Code Mode Orchestration | LLM generates agent API chains in sandbox |
| ADR-004 | Circuit Breaker Pattern | Per-agent CLOSED/OPEN/HALF_OPEN states |
| ADR-005 | MultiModeAgent Pattern | Internal mode routing within agents |
| ADR-006 | File-Based Agent Jurisdiction | .claude/agents/ + root AGENTS.md |
| ADR-007 | Context Engineering | Just-in-time context, few-shot examples |

---

## Verification Strategy

After each phase:
1. `python -m pytest tests/ -x -q` -- all tests pass
2. `python -c "import vetinari; print('Import OK')"` -- clean import
3. For security: targeted security tests
4. For UI: manual dashboard smoke test
5. For architecture: integration test with sample project
