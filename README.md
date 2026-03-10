# Vetinari AI Orchestration System

**v0.5.0** -- Multi-Agent AI Orchestration with 6-Agent Consolidated Architecture, Assembly-Line Execution, Self-Improvement, and Cost-Optimised Routing

## Overview

Vetinari is a comprehensive AI orchestration system that automatically plans, decomposes, routes, executes, reviews, and assembles complex tasks using an assembly-line multi-agent pipeline. It supports local LLM models (via LM Studio) and cloud providers, with minimal human intervention.

### Key Features

- **6-Agent Consolidated Architecture**: 22 legacy agents consolidated into 6 multi-mode agents with 33 total modes (ADR-001)
- **Assembly-Line Pipeline**: Input Analysis > Plan Generation > Recursive Task Decomposition > Model Assignment > Parallel Execution > Output Review > Final Assembly
- **Cascade Model Routing**: Start with cheapest model, escalate to expensive only on low confidence (14% improvement on complex benchmarks)
- **Batch API Processing**: Queue non-urgent inference for Anthropic/OpenAI batch endpoints (50% cost discount)
- **Token Optimizer**: Per-task token budgets, dynamic max_tokens by task type, context deduplication, local LLM preprocessing (30-60% cloud token reduction)
- **RepoMap**: Tree-sitter-inspired structural codebase mapping -- sends function signatures instead of raw files
- **Security Hardened**: hmac.compare_digest token comparison, require_admin decorators, rate limiting, fail-closed credential vault, trusted proxy configuration, input validation
- **Analytics Pipeline**: 7 REST API endpoints for cost tracking, SLA monitoring, anomaly detection, and forecasting
- **Self-Improvement System**: Thompson Sampling model selection, prompt A/B testing, workflow learning, cost optimization, auto-tuning
- **Real Web Search**: Multi-source search with DuckDuckGo, Wikipedia, arXiv -- with anti-hallucination verification (two-source rule)
- **Multi-Provider Support**: LM Studio (local), OpenAI, Anthropic (with prompt caching), Google Gemini, Cohere, HuggingFace, Replicate
- **Durable Execution**: Checkpoint-based recovery for long-running projects
- **Real-Time UI**: SSE streaming of task progress, cancel button, token counter, global search, analytics dashboard
- **Full Observability**: Structured JSON logging, distributed tracing, telemetry, analytics -- all wired into the execution pipeline
- **Agent Governance**: File-based agent definitions (.claude/agents/), root AGENTS.md, file jurisdiction map, quality gates

---

## 6-Agent Architecture (ADR-001)

Based on research showing 5-7 agents optimal for full-system orchestration, Vetinari consolidates 22 legacy agents into 6 multi-mode agents following the **Plan > Research > Advise > Build > Verify > Operate** cognitive pipeline.

| # | Agent | Modes | Absorbs |
|---|-------|-------|---------|
| 1 | **PlannerAgent** | plan, clarify, summarise, prune, extract, consolidate (6) | Orchestrator, UserInteraction, ContextManager |
| 2 | **ResearcherAgent** | code_discovery, domain_research, api_lookup, lateral_thinking, ui_design, database, devops, git_workflow (8) | Architect, Explorer, Librarian, UIPlanner, DataEngineer, VersionControl |
| 3 | **OracleAgent** | architecture, risk_assessment, ontological_analysis, contrarian_review (4) | Ponder |
| 4 | **BuilderAgent** | build, image_generation (2) | ImageGenerator |
| 5 | **QualityAgent** | code_review, security_audit, test_generation, simplification (4) | Evaluator, SecurityAuditor, TestAutomation |
| 6 | **OperationsAgent** | documentation, creative_writing, cost_analysis, experiment, error_recovery, synthesis, improvement, monitor, devops_ops (9) | Synthesizer, Documentation, CostPlanner, Experimentation, Improvement, ErrorRecovery |

**Total: 33 modes across 6 agents.**

---

## Quick Start

### 1. Setup

```bash
# Clone or navigate to project
cd Vetinari

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your LM Studio host and any API keys
```

### 2. Start Vetinari

**Recommended (CLI + Dashboard):**
```bash
# Windows
start.bat

# Linux/macOS
./start.sh

# Or directly with Python
python -m vetinari start
```

**Goal-based execution:**
```bash
python -m vetinari start --goal "Build a Python REST API with user authentication"
```

**Dashboard only:**
```bash
python -m vetinari serve --port 5000
```

**CLI only (no dashboard):**
```bash
python -m vetinari start --no-dashboard --goal "Research best practices for microservices"
```

### 3. All CLI Commands

```
vetinari run       --goal "..."    Execute a goal through the full pipeline
vetinari run       --task t1       Execute a specific manifest task
vetinari serve     --port 5000     Start web dashboard
vetinari start                     Start with dashboard (recommended)
vetinari status                    Show system status
vetinari health                    Health check all providers
vetinari upgrade                   Check for model upgrades
vetinari review                    Run self-improvement analysis
vetinari interactive               Enter REPL mode

Global flags:
  --config PATH   Manifest file path
  --host URL      LM Studio host (default: http://100.78.30.7:1234)
  --mode          planning | execution | sandbox
  --verbose       Debug logging
```

---

## Architecture

### Assembly-Line Pipeline

```
User Input
    |
    v
[1. INPUT ANALYZER]    Classifies request type, domain, complexity
    |
    v
[2. PLAN GENERATOR]    PlannerAgent decomposes into task DAG
    |
    v
[3. TASK DECOMPOSER]   Recursive breakdown to atomic tasks (depth cap: 16)
    |
    v
[4. MODEL ASSIGNER]    CascadeRouter + Thompson Sampling selects model per task
    |
    v
[5. PARALLEL EXECUTOR] DAG scheduler + ThreadPoolExecutor runs tasks in parallel
    |
    v
[6. OUTPUT REVIEWER]   QualityAgent checks quality and consistency
    |
    v
[7. FINAL ASSEMBLER]   OperationsAgent creates unified final output
```

### Cost-Optimised Routing

```
Task arrives
    |
    v
[CascadeRouter]  Start with cheapest model (7B)
    |
    v
[Confidence Check]  Heuristic confidence estimation
    |               |
    |  >= 0.7       |  < 0.7
    v               v
[Return result]  [Escalate to medium (30B)]
                    |
                    v
                 [Confidence Check]
                    |               |
                    |  >= 0.7       |  < 0.7
                    v               v
                 [Return result]  [Escalate to large (72B)]
```

### Self-Improvement Feedback Loop

```
Execution Results
    |
    v
[Quality Scorer]      LLM-as-judge + heuristics
    |
    v
[Feedback Loop]       Updates ModelPerformance table (EMA)
    |
    v
[Thompson Sampling]   Beta distribution updates per model+task_type
    |
    v
[Cost Optimizer]      Routes to cheapest adequate model
    |
    v
[Auto-Tuner]          Adjusts concurrency, thresholds from SLA data
```

### Analytics API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analytics/cost` | GET | Cost breakdown by model, agent, time period |
| `/api/analytics/sla` | GET | SLA compliance metrics and violations |
| `/api/analytics/anomalies` | GET | Detected anomalies in system behavior |
| `/api/analytics/forecast` | GET | Cost and usage forecasting |
| `/api/analytics/models` | GET | Per-model performance and cost stats |
| `/api/analytics/agents` | GET | Per-agent utilization and quality metrics |
| `/api/analytics/summary` | GET | System-wide analytics summary |

---

## Security

v0.5.0 includes hardening across 17 security findings:

- **Authentication**: `require_admin` decorator on all mutating endpoints, `hmac.compare_digest` for constant-time token comparison
- **Rate Limiting**: 10 requests per 60 seconds per client on sandbox execution paths
- **Credential Vault**: Fail-closed Fernet encryption (refuses plaintext fallback)
- **Trusted Proxy**: Configurable trusted proxy for `X-Forwarded-For` header validation
- **Input Validation**: `validate_json_fields` helper for API endpoint input checking
- **Web Security**: Auth decorators on ADR, decomposition, ponder, rules, and training routes

---

## Agent Governance

Each of the 6 agents has a file-based governance definition in `.claude/agents/`:

```
.claude/agents/
  planner.md       # PlannerAgent: plan, clarify, summarise, prune, extract, consolidate
  researcher.md    # ResearcherAgent: code_discovery, domain_research, api_lookup, ...
  oracle.md        # OracleAgent: architecture, risk_assessment, ontological_analysis, ...
  builder.md       # BuilderAgent: build, image_generation
  quality.md       # QualityAgent: code_review, security_audit, test_generation, ...
  operations.md    # OperationsAgent: documentation, cost_analysis, synthesis, ...
```

See `AGENTS.md` at the repository root for the complete governance document covering architecture, file jurisdiction, delegation rules, quality gates, and collaboration matrix.

---

## Configuration

### Environment Variables (`.env`)

```bash
# LM Studio
LM_STUDIO_HOST=http://100.78.30.7:1234

# Cloud providers (optional)
CLAUDE_API_KEY=your_key
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key

# Plan mode
PLAN_MODE_ENABLE=true
PLAN_DEPTH_CAP=16

# Web search
VETINARI_SEARCH_BACKEND=duckduckgo  # duckduckgo | tavily | serpapi
TAVILY_API_KEY=your_key             # Optional
SERPAPI_KEY=your_key                # Optional

# Dashboard
VETINARI_WEB_PORT=5000
```

### Manifest (`manifest/vetinari.yaml`)

The manifest defines the project structure, model registry, and tasks. See the existing file for the full schema.

---

## Project Structure

```
Vetinari/
+-- cli.py                         # Root CLI shim
+-- start.bat / start.sh           # One-click startup scripts
+-- requirements.txt               # Dependencies
+-- setup.py                       # Package setup
+-- AGENTS.md                      # Agent governance document
+-- CLAUDE.md                      # Project conventions
+-- manifest/
|   +-- vetinari.yaml             # Primary manifest
+-- config/
|   +-- models.yaml               # Multi-provider model catalog
|   +-- sandbox_policy.yaml       # Security policies
+-- .claude/agents/                # 6 agent governance files
+-- vetinari/                      # Main package
|   +-- __main__.py               # python -m vetinari entry
|   +-- cli.py                    # Unified CLI (16 subcommands)
|   +-- orchestrator.py           # Main orchestrator
|   +-- two_layer_orchestration.py # Assembly-line pipeline
|   +-- cascade_router.py         # Cost-optimised cascade routing
|   +-- agents/                   # Agent system
|   |   +-- base_agent.py         # Base class with LLM + web search
|   |   +-- contracts.py          # AgentSpec, AGENT_REGISTRY (28 entries)
|   |   +-- interfaces.py         # AgentInterface contracts
|   |   +-- planner_agent.py      # PlannerAgent (6 modes)
|   |   +-- builder_agent.py      # BuilderAgent (2 modes)
|   |   +-- consolidated/         # Consolidated multi-mode agents
|   |       +-- researcher_agent.py  # ResearcherAgent (8 modes)
|   |       +-- oracle_agent.py      # OracleAgent (4 modes)
|   |       +-- quality_agent.py     # QualityAgent (4 modes)
|   |       +-- operations_agent.py  # OperationsAgent (9 modes)
|   +-- adapters/                 # Multi-provider adapter system
|   |   +-- batch_processor.py    # Batch API processing (50% discount)
|   +-- analytics/                # Anomaly, cost, SLA, forecasting
|   +-- learning/                 # Self-improvement system
|   +-- web/                      # Flask web routes
|   |   +-- analytics_routes.py   # 7 analytics REST endpoints
|   +-- dashboard/                # Flask web dashboard + REST API
|   +-- skills/                   # Skill registry and definitions
|   +-- config/                   # Agent skill map, context catalog
|   +-- tools/                    # 14 skill tool wrappers
|   +-- memory/                   # Dual memory store
|   +-- mcp/                      # MCP server integration
|   +-- safety/                   # Guardrails and safety
|   +-- constraints/              # Architecture, quality gates, resources
|   +-- drift/                    # Contract drift prevention
|   +-- sandbox.py                # Code execution sandbox (rate-limited)
|   +-- credentials.py            # Fail-closed credential vault
+-- docs/                          # 30+ documentation files
+-- tests/                         # 60+ test files, 6100+ tests
+-- ui/                            # Web UI (HTML/CSS/JS)
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -x -q

# Run with coverage
python -m pytest tests/ --cov=vetinari --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_registry.py -v

# Run registry + orchestration tests
python -m pytest tests/test_registry.py tests/test_registry_orchestration.py -v
```

---

## Requirements

- Python 3.9+
- LM Studio running at configured host (default: http://100.78.30.7:1234)
- Windows 10/11 recommended (64-bit)
- 16GB+ RAM (64GB+ recommended for large models)

---

## Architecture Decision Records

| ADR | Decision | Rationale |
|-----|----------|-----------|
| ADR-001 | 6-Agent Consolidation | 22 merged to 6 (5-7 optimal per research) |
| ADR-002 | Flat Ensemble over Hierarchy | TwoLayerOrchestrator coordinates directly |
| ADR-003 | Code Mode Orchestration | LLM generates agent API chains in sandbox |
| ADR-004 | Circuit Breaker Pattern | Per-agent CLOSED/OPEN/HALF_OPEN states |
| ADR-005 | MultiModeAgent Pattern | Internal mode routing within consolidated agents |
| ADR-006 | File-Based Agent Jurisdiction | .claude/agents/ + root AGENTS.md governance |
| ADR-007 | Context Engineering | Just-in-time context, few-shot examples (Anthropic Sep 2025) |

---

## License

Proprietary. All rights reserved.
