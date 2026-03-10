# Vetinari AI Orchestration System

**v0.4.0** — Multi-Agent AI Orchestration with 6 Consolidated Agents, Circuit Breakers, Code Mode, and Full Observability

## Overview

Vetinari is a comprehensive AI orchestration system that automatically plans, decomposes, routes, executes, reviews, and assembles complex tasks using an assembly-line multi-agent pipeline. It supports local LLM models (via LM Studio) and cloud providers, with minimal human intervention.

### Key Features

- **Assembly-Line Pipeline**: Input Analysis > Plan Generation > Task Decomposition > Model Assignment > Parallel Execution > Output Review > Final Assembly
- **6 Consolidated Multi-Mode Agents**: Planner, Researcher, Oracle, Builder, Quality, Operations — covering 33 modes
- **Circuit Breakers**: Per-agent fault isolation with CLOSED/OPEN/HALF_OPEN states
- **Code Mode Orchestration**: LLM generates Python code chaining agent API calls, executed in sandbox
- **SLM/LLM Hybrid Routing**: Automatic model tier selection based on task complexity
- **Context Window Management**: Token estimation, smart truncation, per-agent budgets
- **Typed Output Schemas**: Pydantic validation for all 33 agent modes
- **Distributed Tracing**: OpenTelemetry integration with no-op fallback
- **Plan Mode**: LLM-generated multi-candidate plans with risk evaluation and domain-specific templates
- **Self-Improvement System**: Thompson Sampling model selection, prompt A/B testing, workflow learning
- **Analytics Suite**: Cost tracking, SLA monitoring, anomaly detection, forecasting
- **Real Web Search**: Multi-source search with DuckDuckGo, Wikipedia, arXiv
- **Multi-Provider Support**: LM Studio (local), OpenAI, Anthropic, Google Gemini, Cohere, HuggingFace, Replicate
- **Performance Benchmarks**: p50/p95/p99 latency tests, memory bounds, regression guards
- **Full Observability**: Structured JSON logging, distributed tracing, alert system, agent dashboard

---

## Quick Start

### 1. Setup

```bash
cd Vetinari

pip install -r requirements.txt

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
vetinari run          --goal "..."    Execute a goal through the full pipeline
vetinari run          --task t1       Execute a specific manifest task
vetinari serve        --port 5000     Start web dashboard
vetinari start                        Start with dashboard (recommended)
vetinari status                       Show system status
vetinari health                       Health check all providers
vetinari upgrade                      Check for model upgrades
vetinari review                       Run self-improvement agent review
vetinari interactive                  Enter interactive REPL mode
vetinari benchmark    [--agents ...]  Run agent benchmarks
vetinari drift-check                  Check for contract drift across agents

Global flags:
  --config PATH   Manifest file path
  --host URL      LM Studio host (default: http://localhost:1234)
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
[2. PLAN GENERATOR]    LLM-powered multi-candidate plans with risk evaluation
    |
    v
[3. TASK DECOMPOSER]   Recursive breakdown to atomic tasks (depth cap: 16)
    |
    v
[4. MODEL ASSIGNER]    Thompson Sampling + DynamicModelRouter + SLA/cost awareness
    |
    v
[5. PARALLEL EXECUTOR] DAG scheduler + ThreadPoolExecutor + Blackboard coordination
    |
    v
[6. OUTPUT REVIEWER]   EvaluatorAgent checks quality and consistency
    |
    v
[7. FINAL ASSEMBLER]   SynthesizerAgent creates unified final output
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
[Prompt Evolver]      A/B tests prompt variants for underperforming agents
    |
    v
[Workflow Learner]    Updates domain-specific decomposition strategies
    |
    v
[Cost Optimizer]      Routes to cheapest adequate model
    |
    v
[Auto-Tuner]          Adjusts concurrency, thresholds (config persisted)
    |
    v
[SLA Tracker]         Monitors latency/error SLOs per model
    |
    v
[Anomaly Detector]    Flags model performance anomalies
    |
    v
[Improvement Agent]   Periodic meta-review with recommendations
```

### 6 Consolidated Multi-Mode Agents

| Agent | Modes | Capabilities |
|-------|-------|-------------|
| **PlannerAgent** | plan, clarify, summarise, prune, extract, consolidate | Task decomposition, ambiguity detection, context management |
| **ResearcherAgent** | code_discovery, domain_research, api_lookup, lateral_thinking, ui_design, database, devops, git_workflow | Web search, code exploration, domain analysis, UI/DB/DevOps design |
| **OracleAgent** | architecture, risk_assessment, ontological_analysis, contrarian_review | Architecture guidance, risk quantification, contrarian analysis |
| **BuilderAgent** | build, image_generation | Code scaffolding, syntax validation, image generation |
| **QualityAgent** | code_review, security_audit, test_generation, simplification | Code review, 45+ security patterns, test generation |
| **OperationsAgent** | documentation, creative_writing, cost_analysis, experiment, error_recovery, synthesis, improvement, monitor, devops_ops | Docs, cost analysis, monitoring, error recovery, synthesis |

---

## Configuration

### Environment Variables (`.env`)

```bash
# LM Studio
LM_STUDIO_HOST=http://localhost:1234

# Cloud providers (optional)
CLAUDE_API_KEY=your_key
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key

# Plan mode
PLAN_MODE_ENABLE=true
PLAN_DEPTH_CAP=16

# Model discovery
ENABLE_PONDER_MODEL_DISCOVERY=true

# Web search
VETINARI_SEARCH_BACKEND=duckduckgo  # duckduckgo | tavily | serpapi
TAVILY_API_KEY=your_key             # Optional
SERPAPI_KEY=your_key                 # Optional

# Dashboard
VETINARI_WEB_PORT=5000
```

### Manifest (`manifest/vetinari.yaml`)

The manifest defines the project structure, model registry, and tasks. See the existing file for the full schema.

---

## Project Structure

```
Vetinari/
├── cli.py                         # Root CLI shim
├── start.bat / start.sh           # One-click startup scripts
├── requirements.txt               # Dependencies
├── pyproject.toml                 # Package config (v0.4.0)
├── manifest/
│   └── vetinari.yaml             # Primary manifest
├── config/
│   ├── models.yaml               # Multi-provider model catalog
│   └── sandbox_policy.yaml       # Security policies
├── vetinari/                      # Main package
│   ├── __main__.py               # python -m vetinari entry
│   ├── cli.py                    # Unified CLI (10 subcommands)
│   ├── orchestrator.py           # Legacy orchestrator (deprecated)
│   ├── orchestration/            # Modern orchestration subsystem
│   │   ├── two_layer.py          # Assembly-line pipeline
│   │   ├── execution_graph.py    # DAG task execution
│   │   ├── durable_execution.py  # Checkpoint/recovery
│   │   └── plan_generator.py     # Plan generation
│   ├── agents/                   # 6 consolidated multi-mode agents (33 modes)
│   │   ├── base_agent.py         # Base class with LLM + web search + prompt framework
│   │   ├── contracts.py          # Agent type contracts
│   │   ├── planner_agent.py      # LLM-powered planning
│   │   ├── researcher_agent.py   # Real web search + synthesis
│   │   ├── coding_bridge.py      # Routes to CodingEngine
│   │   ├── consolidated/         # Multi-mode agents
│   │   │   ├── orchestrator_agent.py
│   │   │   ├── quality_agent.py
│   │   │   └── operations_agent.py
│   │   └── ...
│   ├── skills/                   # Skill tool wrappers
│   │   ├── quality_skill.py      # Code review, security, testing
│   │   ├── architect_skill.py    # System/UI/DB/API design
│   │   ├── operations_skill.py   # Docs, cost, experiments
│   │   ├── librarian/            # API/library lookup
│   │   └── researcher/           # Multi-source research
│   ├── learning/                 # Self-improvement system
│   │   ├── quality_scorer.py     # LLM-as-judge quality scoring
│   │   ├── feedback_loop.py      # Execution outcome tracking
│   │   ├── model_selector.py     # Thompson Sampling selection
│   │   ├── prompt_evolver.py     # Prompt A/B testing
│   │   ├── workflow_learner.py   # Workflow strategy learning
│   │   ├── cost_optimizer.py     # Cost-aware routing
│   │   └── auto_tuner.py         # SLA-driven auto-adjustment (persistent)
│   ├── analytics/                # Analytics subsystem
│   │   ├── cost.py               # Cost tracking per model/provider
│   │   ├── sla.py                # SLA compliance monitoring
│   │   ├── anomaly.py            # Anomaly detection (Z-score + IQR)
│   │   └── forecasting.py        # Time-series forecasting (SMA, ES, OLS, seasonal)
│   ├── adapters/                 # Multi-provider adapter system
│   │   ├── lmstudio_adapter.py   # LM Studio (local, with streaming)
│   │   ├── openai_adapter.py     # OpenAI
│   │   ├── anthropic_adapter.py  # Anthropic (prompt caching)
│   │   ├── gemini_adapter.py     # Google Gemini
│   │   └── ...
│   ├── coding_agent/             # Code generation engine
│   ├── constraints/              # Architecture + quality gate constraints
│   ├── dashboard/                # Flask web dashboard + log aggregation
│   ├── drift/                    # Contract drift detection
│   ├── memory/                   # Dual-layer memory (short + long term)
│   ├── safety/                   # Safety guardrails
│   ├── tools/                    # Tool registry integration
│   ├── web/                      # Web route modules + preferences API
│   ├── blackboard.py             # Inter-agent communication
│   ├── dynamic_model_router.py   # SLA/anomaly/cost-aware model routing
│   ├── plan_mode.py              # LLM-powered plan generation engine
│   ├── benchmarks/               # Agent benchmark suite
│   └── ...
├── tests/                        # 96 test files, 6000+ tests
├── docs/                         # Documentation
└── ui/                           # Web UI (HTML/CSS/JS)
```

---

## Testing

```bash
# Full test suite
python -m pytest tests/ -x -q

# Skip slow tests
python -m pytest tests/ -x -q -m "not slow"

# Verbose with traceback
python -m pytest tests/ -v --tb=short
```

96 test files with 6000+ tests covering all modules, agents, skills, analytics, and web routes.

---

## Requirements

- Python 3.10+
- LM Studio running at configured host (default: http://localhost:1234)
- Windows 10/11 recommended (64-bit)
- 16GB+ RAM (64GB+ recommended for large models)

---

## License

Proprietary. All rights reserved.
