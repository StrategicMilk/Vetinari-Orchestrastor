# Vetinari Roadmap

**Authoritative planning document for Vetinari AI Orchestration System**

---

## Current Version: v0.4.0

### Architecture: 6 Consolidated Agents

Vetinari uses a flat ensemble of 6 multi-mode agents coordinated by a
TwoLayerOrchestrator. Each agent internally routes tasks to the appropriate
mode handler via MultiModeAgent base class.

| Agent | Modes | Purpose |
|-------|-------|---------|
| **PlannerAgent** | plan, clarify, summarise, prune, extract, consolidate | Planning, clarification, context management |
| **ResearcherAgent** | code_discovery, domain_research, api_lookup, lateral_thinking, ui_design, database, devops, git_workflow | Research, exploration, domain analysis |
| **OracleAgent** | architecture, risk_assessment, ontological_analysis, contrarian_review | Architecture advice, risk assessment |
| **BuilderAgent** | build, image_generation | Code scaffolding, image generation |
| **QualityAgent** | code_review, security_audit, test_generation, simplification | Code review, security, testing |
| **OperationsAgent** | documentation, creative_writing, cost_analysis, experiment, error_recovery, synthesis, improvement, monitor, devops_ops | Docs, cost analysis, monitoring, ops |

**Total: 33 modes across 6 agents**

---

## v0.4.0 Features (Completed)

### Part A: Agent Consolidation (22 agents -> 6)
- Merged Orchestrator modes into PlannerAgent
- Merged Architect modes into ResearcherAgent
- Moved image_generation from Operations to BuilderAgent
- Deleted 20 legacy agent files
- Created `agents/compat.py` for backward-compatible imports
- Updated contracts.py, two_layer.py, and all imports

### Part B: Stub Remediation
- **B1**: Sandbox hook execution — real plugin loading via importlib
- **B2**: Log aggregator backend send() — Elasticsearch, Splunk, Datadog, File
- **B3**: Verification base class — default verify() with passing result
- **B4**: Cost analysis fallback — heuristic token estimation
- **B5**: MODES validation in MultiModeAgent at init time
- **B6**: Security patterns — 15 new CWE patterns (434, 611, 918, 502, etc.)
- **B7**: SVG placeholder metadata — data-placeholder attribute
- **B8**: Upgrader install_upgrade() — LM Studio API integration

### Part C: Feature Implementations (18 features)

| ID | Feature | Status |
|----|---------|--------|
| C1 | Circuit Breakers | Done — CLOSED/OPEN/HALF_OPEN states, per-agent registry |
| C2 | Stage-Boundary Validation | Done — validates between pipeline stages |
| C3 | SLM/LLM Hybrid Routing | Done — ModelTier classification, LM Studio discovery |
| C4 | Context Window Management | Done — token estimation, truncation strategies |
| C5 | Token Budgeting per Agent | Done — per-agent budgets, warn at 80%, truncate at 100% |
| C6 | Typed Output Schemas | Done — Pydantic schemas for all 33 modes |
| C7 | Distributed Tracing | Done — OpenTelemetry with no-op fallback |
| C8 | Cost Tracking | Done — SQLite-backed CostTracker |
| C9 | Memory Consolidation | Done — SharedMemory delegates to DualMemoryStore |
| C10 | Agent Performance Dashboard | Done — AgentMetrics, SystemHealth aggregation |
| C11 | Dynamic Complexity Routing | Done — simple/moderate/complex classification |
| C12 | Dashboard UI | Done — Flask routes, metrics, timeline |
| C13 | Alert System | Done — AlertManager with spike/threshold alerts |
| C14 | Performance Testing | Done — p50/p95/p99 benchmarks, memory tests |
| C15 | OpenCode Phase 2 | Done — batch_edit, diff_preview, rollback, checkpoints |
| C16 | Speculative Decoding | Done — draft-verify pattern with fallback |
| C17 | Continuous Batching | Done — thread-safe queue, batch dispatch |
| C18 | Code Mode Orchestration | Done — VetinariAPI, CodeModeEngine, sandbox execution |

### Part D: Document Consolidation
- Created this ROADMAP.md as single authoritative planning document
- Archived old PHASE_*.md, audit reports, and research docs to docs/archive/
- Cleaned up root directory (only README.md and ROADMAP.md remain)

---

## Key Architecture Files

```
vetinari/
  agents/
    base_agent.py              # BaseAgent with circuit breakers, token budgets
    multi_mode_agent.py        # MultiModeAgent with MODES routing
    planner_agent.py           # PlannerAgent (6 modes)
    consolidated/
      researcher_agent.py      # ResearcherAgent (8 modes)
      oracle_agent.py          # OracleAgent (4 modes)
      quality_agent.py         # QualityAgent (4 modes)
      operations_agent.py      # OperationsAgent (9 modes)
    builder_agent.py           # BuilderAgent (2 modes)
    contracts.py               # AgentType enum, AGENT_REGISTRY
    compat.py                  # Legacy import compatibility
  orchestration/
    two_layer.py               # TwoLayerOrchestrator pipeline
  resilience/
    circuit_breaker.py         # CircuitBreaker, CircuitBreakerRegistry
  routing/
    model_router.py            # SLM/LLM hybrid routing
    complexity_router.py       # Task complexity classification
  context/
    window_manager.py          # Context window management
  schemas/
    agent_outputs.py           # Pydantic output schemas (33 modes)
  observability/
    tracing.py                 # OpenTelemetry distributed tracing
  analytics/
    cost_tracker.py            # SQLite cost tracking
  dashboard/
    agent_dashboard.py         # Agent metrics aggregation
  inference/
    speculative.py             # Speculative decoding
    batcher.py                 # Continuous batching
  code_mode/
    engine.py                  # CodeModeEngine
    api_generator.py           # VetinariAPI generation
  coding_agent/
    bridge.py                  # CodeBridge with batch_edit, rollback
    engine.py                  # CodeAgentEngine
  memory/
    dual_memory.py             # DualMemoryStore (primary)
    oc_memory.py               # OcMemoryStore backend
    mnemosyne_memory.py        # MnemosyneMemoryStore backend
```

---

## v0.5.0 Vision (Future)

### Planned Features
1. **Multi-turn Conversation Memory** — persistent conversation context across sessions
2. **Agent Collaboration Protocols** — structured inter-agent message passing
3. **Plugin Marketplace** — community-contributed agent modes and skills
4. **Cloud Deployment** — Docker/K8s packaging with horizontal scaling
5. **Fine-tuning Pipeline** — automated model fine-tuning from execution feedback
6. **Visual Pipeline Editor** — drag-and-drop pipeline composition in dashboard
7. **Multi-language Support** — extend coding agent beyond Python
8. **Streaming Pipeline** — real-time streaming between pipeline stages

### Architecture Considerations
- Consider moving to async/await for pipeline stages
- Evaluate gRPC for inter-agent communication
- Investigate vector DB integration for semantic memory search
- Profile and optimize hot paths identified by C14 benchmarks

---

## Architecture Decision Records

### ADR-001: 6-Agent Consolidation
**Context:** Original 22 individual agents had overlapping concerns and high coordination overhead.
**Decision:** Consolidate to 6 flat peer agents with internal mode routing.
**Rationale:** Research shows 5-7 agents is optimal. Below 5 = overloaded; above 7 = quadratic coordination overhead.

### ADR-002: Flat Ensemble over Hierarchy
**Context:** Sub-agent spawning adds message-passing overhead and debugging complexity.
**Decision:** Flat peer agents, no sub-agent hierarchy. TwoLayerOrchestrator coordinates.
**Rationale:** The orchestrator already acts as coordinator — internal hierarchy would be redundant.

### ADR-003: Code Mode Orchestration
**Context:** Sequential agent calls require N-1 intermediate LLM round-trips for N-step tasks.
**Decision:** LLM generates Python code that chains agent API calls, executed in sandbox.
**Rationale:** Eliminates intermediate round-trips, improves coherence, leverages existing sandbox.

### ADR-004: Circuit Breaker Pattern
**Context:** LLM failures can cascade through the pipeline.
**Decision:** Per-agent circuit breakers with CLOSED/OPEN/HALF_OPEN states.
**Rationale:** Prevents cascade failures, enables graceful degradation.

---

## Testing

```bash
# Full test suite
python -m pytest tests/ -x -q

# Performance benchmarks
python -m pytest tests/test_performance.py -v --timeout=60

# Specific agent tests
python -m pytest tests/test_consolidated_*.py -v
```

**Test coverage:** 96+ test files, 6000+ tests covering all agents, modes, pipeline stages, and subsystems.
