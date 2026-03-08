# Vetinari AI Orchestration System

**v3.5.0** -- Production-Ready Multi-Agent AI Orchestration with Self-Improvement, Benchmarking, and Manufacturing-Grade Quality Control

## Overview

Vetinari is a production-ready AI orchestration system that automatically plans, decomposes, routes, executes, reviews, and assembles complex tasks using an assembly-line multi-agent pipeline. It supports local LLM models (via LM Studio) and cloud providers, with minimal human intervention.

**63,000+ lines of Python | 2,227 tests | 8 primary agents | 5 benchmark suites**

### Key Features

- **Assembly-Line Pipeline**: Input Analysis -> Plan Generation -> Recursive Task Decomposition -> Model Assignment -> Parallel Execution -> Quality Gates -> Output Review -> Final Assembly
- **8 Primary Agents**: Planner, Researcher, Architect, Builder, Tester, Documenter, Resilience, Meta (with backward-compatible legacy agent types)
- **2-Stage LLM Pipeline**: Architect model plans, Executor model implements -- optimizes for quality vs. cost
- **Self-Improvement System**: Thompson Sampling model selection, prompt A/B testing, workflow learning, cost optimization, auto-tuning, training data export
- **5 Benchmark Suites**: SWE-bench Lite, Tau-bench, ToolBench, TaskBench, API-Bank -- with automatic learning integration
- **Manufacturing Workflow**: Statistical Process Control (SPC), Andon alert system, WIP limits -- industrial quality management
- **Quality Gates**: Automated verification at each pipeline stage (quality, security, coverage, architecture checks)
- **Milestone Checkpoints**: Configurable approval gates at feature boundaries with approve/revise/skip controls
- **Anti-Drift System**: Goal tracking with scope creep detection and automatic realignment
- **MCP Server**: JSON-RPC 2.0 Model Context Protocol server exposing Vetinari as tools
- **Watch Mode**: File monitoring with `@vetinari` directive scanning for inline task dispatch
- **Multi-Provider Support**: LM Studio (local), OpenAI, Anthropic, Google Gemini, Cohere, HuggingFace, Replicate
- **Token Optimization**: Per-task budgets, context compression, grep-based context extraction, RepoMap structural mapping
- **Real-Time UI**: SSE streaming, variant system (LOW/MEDIUM/HIGH depth), Discworld-themed agent nicknames, analytics dashboards
- **Full Observability**: Structured JSON logging, distributed tracing, telemetry, cost/SLA analytics
- **Durable Execution**: Checkpoint-based recovery for long-running projects
- **Git Workflow**: Conventional commits, branch management, PR generation, conflict detection

---

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/StrategicMilk/Vetinari-Orchestrastor.git
cd Vetinari-Orchestrastor

# Install in development mode
pip install -e .

# Copy and configure environment
cp .env.example .env
# Edit .env with your LM Studio host and any API keys
```

### 2. Start Vetinari

**Recommended (CLI + Dashboard):**
```bash
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

### 3. CLI Commands

```
vetinari run         --goal "..."    Execute a goal through the full pipeline
vetinari serve       --port 5000     Start web dashboard
vetinari start                       Start with dashboard (recommended)
vetinari status                      Show system status
vetinari health                      Health check all providers
vetinari benchmark   run|list|report Run benchmark suites
vetinari switch-model MODEL_ID       Switch active model mid-session
vetinari watch       [directory]     Watch files for @vetinari directives
vetinari upgrade                     Check for model upgrades
vetinari review                      Run self-improvement analysis
vetinari interactive                 Enter REPL mode

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
[1. INPUT ANALYZER]      Classify request type, domain, complexity
    |
    v
[2. PLANNER AGENT]       LLM-powered task decomposition with DAG validation
    |
    v
[3. QUALITY GATE]        Post-planning verification (architecture check)
    |
    v
[4. MODEL ASSIGNER]      Thompson Sampling + benchmark-weighted selection
    |
    v
[5. PARALLEL EXECUTOR]   DAG scheduler + ThreadPoolExecutor
    |                     Architect model plans -> Executor model implements
    v
[6. QUALITY GATE]        Post-execution verification (quality + security)
    |
    v
[7. TESTER AGENT]        Automated test generation and coverage analysis
    |
    v
[8. QUALITY GATE]        Post-testing verification (coverage check)
    |
    v
[9. ASSEMBLER]           Final output assembly with conflict resolution
    |
    v
[10. QUALITY GATE]       Pre-assembly verification (final security scan)
    |
    v
Final Output + Learning Feedback
```

### Self-Improvement Feedback Loop

```
Execution Results
    |
    v
[Quality Scorer]        LLM-as-judge + heuristic scoring
    |
    v
[Feedback Loop]         Updates model performance (EMA)
    |
    v
[Thompson Sampling]     Beta distribution updates per model+task_type
    |                   (3x weight for benchmark results)
    v
[Prompt Evolver]        A/B tests prompt variants with benchmark validation
    |
    v
[Workflow Learner]      Domain-specific decomposition strategies
    |
    v
[Cost Optimizer]        Routes to cheapest adequate model
    |
    v
[Episode Memory]        Stores and retrieves past execution patterns
    |
    v
[Training Pipeline]     Export HuggingFace/Alpaca format for fine-tuning
```

### 8 Primary Agents

| Agent | Role | Capabilities |
|-------|------|-------------|
| **Planner** | Orchestration | Task decomposition, dependency mapping, DAG validation, episode memory |
| **Researcher** | Information | Multi-source web search, synthesis, library lookup, code exploration |
| **Architect** | Design | Architecture guidance, risk assessment, cost planning, creative suggestions |
| **Builder** | Implementation | Code generation, scaffolding, UI design, data engineering, DevOps |
| **Tester** | Verification | Test generation, coverage analysis, security auditing, quality evaluation |
| **Documenter** | Documentation | API docs, README generation, version control documentation |
| **Resilience** | Recovery | Error recovery, retry strategies, checkpoint management |
| **Meta** | Self-Improvement | Performance meta-analysis, experimentation management |

### Benchmark Suites

| Suite | Layer | What It Measures |
|-------|-------|-----------------|
| **SWE-bench Lite** | Pipeline | End-to-end patch generation for real GitHub issues |
| **Tau-bench** | Pipeline | Pass@k reliability across repeated runs |
| **ToolBench** | Agent | Tool selection accuracy for single-step tasks |
| **TaskBench** | Orchestration | Task decomposition quality and DAG correctness |
| **API-Bank** | Orchestration | Multi-step tool calling chains |

---

## Configuration

### Environment Variables (`.env`)

```bash
# LM Studio (local models)
LM_STUDIO_HOST=http://localhost:1234

# Cloud providers (optional)
CLAUDE_API_KEY=your_key
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key

# Dashboard
VETINARI_WEB_PORT=5000

# Web search
VETINARI_SEARCH_BACKEND=duckduckgo

# Milestone checkpoints
VETINARI_MILESTONE_POLICY=features  # all | features | phases | critical | none
```

---

## Project Structure

```
Vetinari/
├── vetinari/                      # Main package (239 modules, 63K+ lines)
│   ├── __main__.py                # python -m vetinari entry point
│   ├── cli.py                     # Unified CLI
│   ├── agents/                    # 8 primary + legacy-compat agents
│   │   ├── base_agent.py          # Base class with LLM + web search
│   │   ├── planner_agent.py       # DAG-validated task decomposition
│   │   ├── researcher_agent.py    # Multi-source research + synthesis
│   │   ├── architect_agent.py     # Architecture + creative suggestions
│   │   ├── builder_agent.py       # Code generation + scaffolding
│   │   ├── tester_agent.py        # Test generation + security audit
│   │   ├── documenter_agent.py    # Documentation generation
│   │   ├── resilience_agent.py    # Error recovery + checkpoints
│   │   └── meta_agent.py          # Self-improvement meta-analysis
│   ├── learning/                  # Self-improvement pipeline
│   │   ├── model_selector.py      # Thompson Sampling selection
│   │   ├── prompt_evolver.py      # Prompt A/B testing
│   │   ├── quality_scorer.py      # LLM-as-judge scoring
│   │   ├── feedback_loop.py       # Execution outcome tracking
│   │   ├── episode_memory.py      # Past execution recall
│   │   ├── training_data.py       # HF/Alpaca export
│   │   └── ...
│   ├── benchmarks/                # 5 benchmark suite adapters
│   ├── orchestration/             # 2-stage pipeline + durable engine
│   ├── validation/                # Quality gates system
│   ├── workflow/                   # SPC, Andon, WIP manufacturing
│   ├── tools/                     # 21 skill tool wrappers
│   ├── adapters/                  # Multi-provider adapter system
│   ├── analytics/                 # Cost, SLA, anomaly, forecasting
│   ├── dashboard/                 # Flask web dashboard + SSE
│   ├── web/                       # API blueprints (learning, analytics, preferences)
│   ├── mcp/                       # Model Context Protocol server
│   ├── drift/                     # Anti-drift goal tracking
│   ├── memory/                    # Shared state + enhanced memory
│   ├── planning/                  # Plan types, engine, decomposition
│   ├── models/                    # Model routing, registry, VRAM
│   ├── training/                  # Training pipeline manager
│   └── ...
├── tests/                         # 94 test files, 2,227 tests
├── docs/                          # Documentation
├── ui/                            # Web UI (HTML/CSS/JS)
├── manifest/                      # Project manifests
└── config/                        # Model catalog + policies
```

---

## Requirements

- Python 3.9+
- LM Studio running locally (or cloud API keys configured)
- 16GB+ RAM (64GB+ recommended for large local models)

---

## License

Proprietary. All rights reserved.
