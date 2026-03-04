# Vetinari AI Orchestration System

**v0.2.0** -- Comprehensive Multi-Agent AI Orchestration with Assembly-Line Execution and Self-Improvement

## Overview

Vetinari is a comprehensive AI orchestration system that automatically plans, decomposes, routes, executes, reviews, and assembles complex tasks using an assembly-line multi-agent pipeline. It supports local LLM models (via LM Studio) and cloud providers, with minimal human intervention.

### Key Features

- **Assembly-Line Pipeline**: Input Analysis → Plan Generation → Recursive Task Decomposition → Model Assignment → Parallel Execution → Output Review → Final Assembly
- **16 Specialized Agents**: Planner, Explorer, Oracle, Librarian, Researcher, Evaluator, Synthesizer, Builder, UI Planner, Security Auditor, Data Engineer, Documentation, Cost Planner, Test Automation, Experimentation Manager, Improvement Agent
- **Self-Improvement System**: Thompson Sampling model selection, prompt A/B testing, workflow learning, cost optimization, auto-tuning
- **Real Web Search**: Multi-source search with DuckDuckGo, Wikipedia, arXiv -- with anti-hallucination verification (two-source rule)
- **User Interaction**: Detects ambiguous goals and asks targeted clarifying questions
- **Multi-Provider Support**: LM Studio (local), OpenAI, Anthropic, Google Gemini, Cohere, HuggingFace, Replicate
- **Durable Execution**: Checkpoint-based recovery for long-running projects
- **Full Observability**: Structured JSON logging, distributed tracing, telemetry, analytics

---

## Quick Start

### 1. Setup

```bash
# Clone or navigate to project
cd C:\Users\darst\.lmstudio\projects\Vetinari

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
    │
    ▼
[1. INPUT ANALYZER]    Classifies request type, domain, complexity
    │
    ▼
[2. PLAN GENERATOR]    LLM-powered task decomposition (PlannerAgent)
    │
    ▼
[3. TASK DECOMPOSER]   Recursive breakdown to atomic tasks (depth cap: 16)
    │
    ▼
[4. MODEL ASSIGNER]    Thompson Sampling + DynamicModelRouter selects best model per task
    │
    ▼
[5. PARALLEL EXECUTOR] DAG scheduler + ThreadPoolExecutor runs tasks in parallel
    │
    ▼
[6. OUTPUT REVIEWER]   EvaluatorAgent checks quality and consistency
    │
    ▼
[7. FINAL ASSEMBLER]   SynthesizerAgent creates unified final output
```

### Self-Improvement Feedback Loop

```
Execution Results
    │
    ▼
[Quality Scorer]      LLM-as-judge + heuristics
    │
    ▼
[Feedback Loop]       Updates ModelPerformance table (EMA)
    │
    ▼
[Thompson Sampling]   Beta distribution updates per model+task_type
    │
    ▼
[Prompt Evolver]      A/B tests prompt variants for underperforming agents
    │
    ▼
[Workflow Learner]    Updates domain-specific decomposition strategies
    │
    ▼
[Cost Optimizer]      Routes to cheapest adequate model
    │
    ▼
[Auto-Tuner]          Adjusts concurrency, thresholds from SLA data
    │
    ▼
[Improvement Agent]   Periodic meta-review with recommendations
```

### Agents

| Agent | Capabilities |
|-------|-------------|
| Planner | LLM-powered task decomposition, dependency mapping |
| Explorer | Web search + code pattern discovery |
| Oracle | LLM architectural guidance, risk assessment |
| Librarian | Real API/library lookup with web search |
| Researcher | Multi-source web research + LLM synthesis |
| Evaluator | LLM code quality review, security analysis |
| Synthesizer | LLM artifact fusion and conflict resolution |
| Builder | LLM code scaffolding and generation |
| UI Planner | UI/UX design, wireframe planning |
| Security Auditor | Vulnerability scanning, compliance |
| Data Engineer | Pipeline design, schema creation |
| Documentation | API docs, README generation |
| Cost Planner | Budget planning, model cost optimization |
| Test Automation | Test generation, coverage analysis |
| Experimentation Manager | A/B test management |
| **Improvement Agent** | Self-improvement meta-analysis |

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
TAVILY_API_KEY=your_key             # Optional, for Tavily backend
SERPAPI_KEY=your_key                # Optional, for SerpAPI backend

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
├── setup.py                       # Package setup (v0.2.0)
├── manifest/
│   └── vetinari.yaml             # Primary manifest
├── config/
│   ├── models.yaml               # Multi-provider model catalog
│   └── sandbox_policy.yaml       # Security policies
├── vetinari/                      # Main package
│   ├── __main__.py               # python -m vetinari entry
│   ├── cli.py                    # Unified CLI (16 subcommands)
│   ├── orchestrator.py           # Main orchestrator
│   ├── two_layer_orchestration.py # Assembly-line pipeline
│   ├── agents/                   # 16 specialized agents
│   │   ├── base_agent.py         # Base class with LLM + web search
│   │   ├── planner_agent.py      # LLM-powered planning
│   │   ├── researcher_agent.py   # Real web search + synthesis
│   │   ├── improvement_agent.py  # Self-improvement meta-agent
│   │   ├── user_interaction_agent.py # Clarification gathering
│   │   └── ... (12 more agents)
│   ├── learning/                  # Self-improvement system
│   │   ├── quality_scorer.py     # LLM-as-judge quality scoring
│   │   ├── feedback_loop.py      # Execution outcome tracking
│   │   ├── model_selector.py     # Thompson Sampling selection
│   │   ├── prompt_evolver.py     # Prompt A/B testing
│   │   ├── workflow_learner.py   # Workflow strategy learning
│   │   ├── cost_optimizer.py     # Cost-aware routing
│   │   └── auto_tuner.py        # SLA-driven auto-adjustment
│   ├── tools/                    # 14 skill tool wrappers
│   ├── adapters/                 # Multi-provider adapter system
│   ├── analytics/                # Anomaly, cost, SLA, forecasting
│   ├── dashboard/                # Flask web dashboard
│   └── ... (many more modules)
├── docs/                          # 30+ documentation files
├── tests/                         # 60+ test files
└── ui/                            # Web UI (HTML/CSS/JS)
```

---

## Requirements

- Python 3.9+
- LM Studio running at configured host (default: http://100.78.30.7:1234)
- Windows 10/11 recommended (64-bit)
- 16GB+ RAM (64GB+ recommended for large models)

---

## License

Proprietary. All rights reserved.
