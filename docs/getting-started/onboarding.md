# Vetinari — New Developer Onboarding Guide

**Audience**: Engineers joining the project for the first time

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Tour](#2-repository-tour)
3. [Development Environment Setup](#3-development-environment-setup)
4. [Architecture Primer](#4-architecture-primer)
5. [Running Tests](#5-running-tests)
6. [Making Your First Change](#6-making-your-first-change)
7. [Key Concepts](#7-key-concepts)
8. [Debugging Tips](#8-debugging-tips)
9. [Getting Help](#9-getting-help)

---

## 1. Project Overview

Vetinari (v0.6.0) is a **local-first multi-agent LLM orchestration system with self-improving routing**. It prefers local native-model backends such as `vllm` or NIM when available, and falls back to GGUF models loaded via llama-cpp-python. There are no required cloud API calls for inference - the system is designed to stay local-first.

The core idea: decompose a user's goal into a plan, route each task to a specialized agent and the cheapest model that can handle it, enforce quality gates, then continuously learn which model/agent pairings work best. Over time, routing improves automatically via Thompson Sampling (Bayesian bandits).

**What Vetinari is NOT:**
- Not a chatbot wrapper around a cloud API
- Not a RAG pipeline bolted onto a vector store
- Not a Python script that calls OpenAI

**What Vetinari IS:**
- A three-agent factory pipeline (Foreman → Worker → Inspector) executing structured plans
- A self-improving system: idle time is used for prompt evolution, synthetic data generation, and model fine-tuning
- A production-grade Litestar ASGI application with a large API surface and retained Svelte UI source assets that are not part of the current shipped package boundary
- A hybrid local inference stack: `vllm`/NIM for native models, `llama.cpp` for GGUF fallback

---

## 2. Repository Tour

### Top-level layout

```
vetinari/          Python package — all production code lives here
tests/             Pytest test suite mirroring vetinari/ structure
config/            YAML/JSON runtime configuration (models, inference profiles, standards)
adr/               Architecture Decision Records (JSON) — dev reference, not runtime
docs/              Documentation (you are here)
scripts/           Dev utility scripts (test_summary.py, check_vetinari_rules.py, memory_cli.py)
ui/                Svelte front-end (compiled output in ui/static/)
.claude/           AI assistant workflow support; not runtime authority
```

### Inside `vetinari/`

```
vetinari/
├── adapters/             Model adapters: llama-cpp-python, LiteLLM, OpenAI-compatible server
├── agents/               Agent infrastructure
│   ├── consolidated/     Worker and Inspector internals
│   ├── base_agent.py     BaseAgent: circuit breakers, token budgets, retry logic
│   ├── planner_agent.py  ForemanAgent (6 modes: plan, clarify, consolidate,
│   │                       summarise, prune, extract)
│   ├── builder_agent.py  Internal Worker build delegate (build, image_generation)
│   ├── multi_mode_agent.py  Mode-dispatch base class
│   ├── contracts.py      AgentSpec, Task, Plan, ExecutionPlan, AGENT_REGISTRY
│   ├── interfaces.py     AgentInterface ABC
│   └── inference.py      Model inference orchestration (route → infer → record)
├── analytics/            Cost tracking, anomaly detection, SLA monitoring, forecasting
├── coding_agent/         In-process coding agent engine
├── config/               Config loading: inference profiles, agent specs, standards
├── constraints/          Constraint definitions and enforcement
├── context/              Context compression for cross-agent handoffs
├── drift/                Contract and API drift detection
├── enforcement/          Delegation depth, quality gate, and jurisdiction enforcers
├── evaluation/           Evaluation framework for agent outputs
├── events.py             Event bus for inter-component communication
├── exceptions.py         All project exception classes (never rename these)
├── kaizen/               Continuous improvement — PDCA lifecycle tracking
├── learning/             Self-improvement: Thompson Sampling, prompt evolution, feedback
├── memory/               UnifiedMemoryStore — SQLite + FTS5 + BM25 + embeddings
├── models/               Model pool, dynamic router, VRAM manager, ponder, best-of-N
├── mcp/                  MCP client/server integration
├── orchestration/        TwoLayerOrchestrator, DAG executor, agent graph, durable execution
├── planning/             Plan generation, decomposition, wave scheduling
├── prompts/              Prompt versioning and assembler
├── rag/                  Knowledge base with semantic search
├── resilience/           Circuit breakers, retry logic with backoff
├── safety/               Content filters, guardrail policies
├── schemas/              Pydantic output schemas for all 34 agent modes
├── security/             Per-agent permissions and enforcement
├── setup/                Init wizard, model recommender
├── skills/               Runtime skill catalog with SKILL.md definitions per skill
├── structured_logging.py Correlation context and JSON structured logging
├── tools/                Web search, git, file, and static analysis tools
├── training/             Idle-time training: curriculum, synthetic data, LoRA
├── types.py              CANONICAL enum source — AgentType, StatusEnum, ExecutionMode, etc.
├── validation/           Root cause analysis and prevention gates
├── verification/         Verification framework
├── web/                  Litestar ASGI app and REST API
│   ├── litestar_app.py   App factory — entry point for the web server
│   ├── lifespan.py       Startup/shutdown lifecycle handlers
│   ├── chat_api.py       Streaming chat endpoint
│   ├── projects_api.py   Project lifecycle (create, run, status, cancel)
│   ├── models_api.py     Model management (list, load, unload, benchmark)
│   ├── training_api.py   Training control (start, stop, status)
│   └── ...               20+ additional API modules
└── workflow/             Statistical process control (SPC)
```

### Configuration files

| File | Purpose |
|------|---------|
| `config/models.yaml` | All known models with hardware requirements and capabilities |
| `config/task_inference_profiles.json` | Sampling parameters (temperature, max_tokens, etc.) per task type |
| `config/rules.yaml` | Agent behavior rules and quality thresholds |
| `config/standards/constraints.yaml` | Constraint definitions by agent jurisdiction |
| `pyproject.toml` | Build system, dependencies, and tool configuration |

---

## 3. Development Environment Setup

### Prerequisites

- Python 3.10 or later
- A C++ compiler (for llama-cpp-python): MSVC on Windows, gcc/clang on Linux/macOS
- Git

### Install

```bash
# Clone and enter the repo
git clone <repo-url>
cd vetinari

# Create a virtual environment
python -m venv .venv312
source .venv312/bin/activate          # Windows: .venv312\Scripts\activate

# Install the contributor baseline
pip install -e ".[dev,local,ml,search,notifications]"
# or, if you have uv:
uv pip install -e ".[dev,local,ml,search,notifications]"

# Add heavy optional stacks only when needed
pip install -e ".[training]"   # QLoRA / fine-tuning
pip install -e ".[vllm]"       # vLLM backend
```

### Verify the install

```bash
python -c "import vetinari; print('OK')"
```

If that fails, check that your virtual environment is activated and `pip install -e ".[dev,local,ml,search,notifications]"` completed without errors.

### Add models

Vetinari supports two local model paths:

- native Hugging Face-format models for `vllm` or NIM
- GGUF models for the `llama.cpp` fallback

For GGUF fallback models, place your `.gguf` files in a models directory and point Vetinari at them:

```bash
# In your .env file or shell:
export VETINARI_MODELS_DIR=/path/to/your/models

# Or run the interactive setup wizard:
python -m vetinari init
```

The init wizard (`python -m vetinari init`) will scan your hardware, recommend models for your VRAM budget, and write `~/.vetinari/config.yaml`.

### Windows + WSL native backend setup

If you are running on Windows and want `vllm` as the primary backend, the supported operator path is:

1. keep Vetinari in the Windows `.venv312`
2. install and run `vllm` inside WSL Ubuntu
3. export `VETINARI_VLLM_ENDPOINT` in the Windows shell before starting Vetinari
4. keep `VETINARI_MODELS_DIR` pointed at your GGUF fallback directory

Official setup links:

- [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
- [Get Ubuntu on WSL](https://ubuntu.com/download/wsl)
- [Ubuntu on the Microsoft Store](https://apps.microsoft.com/detail/9pdxgncfsczv?gl=US&hl=en-US)
- [Ubuntu on WSL docs](https://documentation.ubuntu.com/wsl/stable/)
- [vLLM GPU installation docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [NVIDIA Windows driver download](https://www.nvidia.com/Download/index.aspx)

WSL setup:

```bash
# From Windows PowerShell:
wsl -d Ubuntu

# In Ubuntu:
sudo apt update
sudo apt install -y python3-pip python3.12-venv

python3 -m venv ~/.venvs/vllm
source ~/.venvs/vllm/bin/activate

# From a Vetinari checkout in WSL, install the pyproject-managed optional stack.
python -m pip install -e ".[vllm]"

python -c "import vllm, torch; print('vllm', vllm.__version__); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
vllm serve Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8000
```

Windows shell setup:

```powershell
. C:\dev\Vetinari\enter-vetinari-shell.ps1
$env:VETINARI_VLLM_ENDPOINT = "http://localhost:8000"
$env:VETINARI_NATIVE_MODELS_DIR = "C:\dev\Vetinari\models\native"
$env:VETINARI_MODELS_DIR = "C:\dev\Vetinari\models"
python -m vetinari doctor --json
```

Repeatable repo helper:

```powershell
. C:\dev\Vetinari\enter-vetinari-shell.ps1
C:\dev\Vetinari\start-vllm-wsl.ps1
python -m vetinari doctor --json
```

Useful variants:

```powershell
C:\dev\Vetinari\start-vllm-wsl.ps1 -Model C:\dev\Vetinari\models\native\YOUR_MODEL_DIR
C:\dev\Vetinari\start-vllm-wsl.ps1 -ForceRestart
C:\dev\Vetinari\stop-vllm-wsl.ps1
```

If you want new PowerShell sessions to pick up the project environment automatically:

```powershell
if (!(Test-Path $PROFILE)) { New-Item -ItemType File -Path $PROFILE -Force | Out-Null }
Add-Content $PROFILE "`n. C:\dev\Vetinari\enter-vetinari-shell.ps1"
. $PROFILE
```

Expected operator-owned prerequisites:

- `WSL2` installed and an Ubuntu distro available
- `nvidia-smi` working inside WSL
- a running `vllm` server or a reachable NIM endpoint
- native-model files under the configured native models directory
- GGUF fallback files under the configured GGUF models directory

If any of those are missing, use the official links above before proceeding with the shell commands.

Safe cleanup note:

- Git now ignores the temporary pytest/pip/probe directories created during local environment debugging.
- If you need to clean them from disk manually, the safe patterns are `.pytest-tmp-root*`, `.pytest-temp*`, `.pip-temp/`, `.pip-work/`, `.pip-cache/`, `.probe-tmp/`, `.tmp-perm-check*`, `probe700/`, `probe755/`, and `unsloth_compiled_cache/`.

### Start the server

```bash
python -m vetinari serve --port 5000
# Server starts on http://localhost:5000 by default
```

The web UI is served at `http://localhost:5000`. The REST API is at `http://localhost:5000/api/`.

---

## 4. Architecture Primer

### The three-agent factory pipeline (ADR-0061)

Every user request flows through exactly three agents in sequence:

```
User Input
    │
    ▼
[1] Foreman (PlannerAgent)
    Decomposes the goal into a structured plan (DAG of tasks).
    6 modes: plan · clarify · consolidate · summarise · prune · extract
    │
    ▼
[2] Worker (WorkerAgent — single class, 24 modes across 4 groups)
    Executes each task in the plan. Tasks run in parallel where the DAG allows.
    Research group (8 modes): code_discovery, domain_research, api_lookup, lateral_thinking,
                              ui_design, database, devops, git_workflow
    Architecture group (5 modes): architecture, risk_assessment, ontological_analysis,
                                  contrarian_review, suggest
    Build group (2 modes): build, image_generation
    Operations group (9 modes): documentation, creative_writing, cost_analysis, experiment,
                                error_recovery, synthesis, improvement, monitor, devops_ops
    │
    ▼
[3] Inspector (quality gate)
    Reviews Worker outputs. Issues pass/fail verdict per task.
    4 modes: code_review · security_audit · test_generation · simplification
    │
    ▼
Output Assembler → Final response
```

The Worker is **one public runtime agent** (`vetinari/agents/consolidated/worker_agent.py`) that routes internally to the appropriate mode. Legacy delegate classes such as `BuilderAgent`, `ConsolidatedResearcherAgent`, `ConsolidatedOracleAgent`, and `OperationsAgent` can still exist as implementation details, but they are not active public agent identities.

### Full execution flow

```
User Input
  → Input Analyzer
  → Plan Generator (Foreman — creates task DAG)
  → Model Assigner (CascadeRouter + Thompson Sampling picks cheapest viable model per task)
  → DAG Executor (runs tasks in parallel waves respecting dependencies)
  → Quality Gate (Inspector reviews each output; failures re-queue with feedback)
  → Output Assembler
  → Response
```

### Model cascade

Tasks start on the smallest/cheapest model that fits the task profile (typically a 7B). If the model's confidence score falls below the task's threshold, the cascade router escalates to a larger model (30B, then 72B). This happens automatically — callers never hardcode a model name.

Sampling parameters (temperature, max_tokens, top_p, etc.) come exclusively from `config/task_inference_profiles.json` via `InferenceConfigManager`. They are never hardcoded in source.

### Thompson Sampling

The learning system uses Bayesian bandits to track which model performs best for each task type. After each successful inference, the system updates the Beta distribution for that (model, task_type) pair. Over time, the router naturally concentrates on the best performers. This requires no explicit retraining trigger — it updates on every inference result.

### Memory

`UnifiedMemoryStore` (`vetinari/memory/unified.py`) is the single memory backend. It uses:
- SQLite as the storage layer
- FTS5 for fast keyword search
- BM25 scoring for relevance ranking
- Optional embedding vectors for semantic similarity

There is no separate short-term/long-term memory split. All memory goes through `UnifiedMemoryStore`.

### Web application

The web server is a **Litestar ASGI application** defined in `vetinari/web/litestar_app.py`. It is not Flask. Key characteristics:
- Async-first: route handlers are `async def`
- Type-safe: Litestar validates request/response types via Pydantic schemas
- Lifecycle managed by `vetinari/web/lifespan.py` (startup model loading, graceful shutdown)
- 259 routes across 20+ API modules

---

## 5. Running Tests

### Standard commands

```bash
# Run the full test suite (stop on first failure)
python -m pytest tests/ -x -q

# Read the results summary (always use this — it formats the XML output readably)
python scripts/test_summary.py

# Run with coverage
python -m pytest tests/ -x -q --cov=vetinari

# Run a specific file
python -m pytest tests/test_inference_config.py -v

# Run tests matching a keyword
python -m pytest tests/ -k "thompson" -v
```

### Lint and rules checks

```bash
# Auto-fix lint issues then format
python -m ruff check vetinari/ --fix
python -m ruff format vetinari/

# Run the custom 52-rule project validator (11 categories)
python scripts/check_vetinari_rules.py

# Run only the rules that block commits
python scripts/check_vetinari_rules.py --errors-only
```

### What the hooks run automatically

| Trigger | What runs |
|---------|-----------|
| Every `.py` file save | `ruff check --fix` + `ruff format` |
| Before `git commit` | `pytest` + `ruff check` + `check_vetinari_rules.py --errors-only` |
| Session end | Full `ruff check` + `ruff format --check` + `check_vetinari_rules.py` + `pytest` |

### Writing tests

- Test files live in `tests/` and mirror source structure: `vetinari/models/ponder.py` → `tests/test_ponder.py`
- Use `pytest` fixtures, not `unittest.TestCase` (unless extending existing unittest-based tests)
- Mock inference with `unittest.mock.patch` — never make real inference calls in tests
- Never delete or weaken a test to make code pass. Fix the code.

---

## 6. Making Your First Change

### Workflow

```bash
# 1. Create a feature branch
git checkout -b feat/your-feature-name

# 2. Read the relevant rules before writing code
#    Runtime style, typing, imports, and testing rules: vetinari/config/standards/ and vetinari/config/rules.yaml

# 3. Write the call site FIRST, then implement the function
#    (Unwired code is the #1 AI/developer mistake — see CLAUDE.md)

# 4. Run tests
python -m pytest tests/ -x -q

# 5. Lint and format
python -m ruff check vetinari/ --fix && python -m ruff format vetinari/

# 6. Run the project rules checker
python scripts/check_vetinari_rules.py

# 7. Verify the package still imports
python -c "import vetinari; print('OK')"

# 8. Commit (pre-commit hook runs tests automatically)
git add vetinari/your_module.py tests/test_your_module.py
git commit -m "feat(your-area): describe what and why"
```

### Definition of Done checklist

A change is not complete until ALL of these pass:

- [ ] No stubs, `TODO`, `pass` bodies, or `NotImplementedError` in production code
- [ ] All new public functions have full type annotations and Google-style docstrings
- [ ] Enums imported from `vetinari.types`, specs from `vetinari.agents.contracts`
- [ ] Every new function is called from at least one place (grep to verify)
- [ ] `python -m pytest tests/ -x -q` — zero failures
- [ ] `python -m ruff check vetinari/` — zero errors
- [ ] `python scripts/check_vetinari_rules.py` — zero errors
- [ ] `python -c "import vetinari; print('OK')"` — succeeds
- [ ] Significant decisions documented as an ADR (see `docs/internal/ai-workflows/vetinari-agent-system-reference.md`)

### Adding a new Worker mode

The Worker agent routes via `WorkerAgent` in `vetinari/agents/consolidated/worker_agent.py`. To add a mode:

1. Add the mode name to `AgentType` or the relevant mode enum in `vetinari/types.py` (only add — never rename/remove existing values)
2. Add the Pydantic output schema in `vetinari/schemas/`
3. Add the prompt template in `vetinari/prompts/`
4. Wire the mode in `WorkerAgent`'s dispatch logic
5. Add the inference profile entry in `config/task_inference_profiles.json`
6. Write tests in `tests/`

### Adding a new API endpoint

1. Find or create the appropriate API module in `vetinari/web/`
2. Define the Litestar route handler (`async def` with full type annotations)
3. Register it in `vetinari/web/litestar_app.py`
4. Write tests

---

## 7. Key Concepts

### Canonical enums (vetinari/types.py)

`vetinari/types.py` is the **single source of truth** for all enums. Always import from there:

```python
from vetinari.types import AgentType, StatusEnum, ExecutionMode, TaskStatus
```

Never use string literals for agent types or statuses:

```python
# Correct
if agent_type == AgentType.FOREMAN:
    ...

# Wrong — causes silent failures
if agent_type == "FOREMAN":
    ...
```

The `types.py` file follows strict safe-modification rules: only ADD new enum values, never rename or remove existing ones. Renaming breaks every comparison in the codebase.

### Agent specs and contracts (vetinari/agents/contracts.py)

`AgentSpec`, `Task`, `Plan`, `ExecutionPlan`, and `AgentResult` are defined here. When adding fields to these dataclasses, always provide defaults so existing callers do not break.

### Config-driven inference

Inference parameters are never hardcoded. The pipeline for getting them:

1. Task is tagged with a task type (e.g., `code_generation`, `research`)
2. `InferenceConfigManager` looks up the profile in `config/task_inference_profiles.json`
3. The profile specifies `temperature`, `max_tokens`, `top_p`, etc.
4. `CascadeRouter` selects the model; `InferenceConfigManager` provides the params

If you need a new parameter set, add a new profile entry in JSON — do not hardcode values in Python.

### Logging (no print())

Every module uses `logging`, never `print()`:

```python
import logging
logger = logging.getLogger(__name__)

# Use %-style inside logging calls
logger.info("Processing task %s for agent %s", task_id, agent_type)

# In except blocks, use exception() for automatic traceback
try:
    result = run_inference(task)
except RuntimeError as exc:
    logger.exception("Inference failed for task %s — task will be retried", task_id)
    raise
```

Ruff enforces the no-`print()` rule (T20) on every file save.

### Structured logging and correlation

Wrap operations in a `CorrelationContext` to propagate a trace ID through all log lines in a request:

```python
from vetinari.structured_logging import CorrelationContext

with CorrelationContext() as ctx:
    ctx.set_span_id("plan-execute")
    logger.info("Starting plan execution")  # log line carries trace_id and span_id
```

### Singleton pattern

Services that must be shared (memory store, model pool, event bus) use thread-safe singletons with double-checked locking:

```python
_instance: SomeService | None = None
_lock = threading.Lock()

def get_some_service() -> SomeService:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SomeService()
    return _instance
```

In tests, call the matching `reset_*()` or `clear_*()` function in both `setUp` and `tearDown` to avoid state leaking between tests.

### Event bus

Components communicate via `vetinari/events.py` rather than direct method calls. This keeps the call graph decoupled. If you add an event publisher, verify there is at least one subscriber — events with no consumer are write-only dead code.

### Wiring discipline

The number-one failure mode in AI-assisted development is writing a function that nothing calls. Vetinari's custom rules checker (VET120-124) blocks commits with unwired code. The mitigation:

1. Write the call site first — the line of code that calls your new function — before implementing the function itself.
2. After implementing, grep to confirm the function has a caller:
   ```bash
   grep -rn "def your_function" vetinari/
   grep -rn "your_function(" vetinari/ tests/
   ```
   If the second grep only shows the definition line, the function is unwired.

### Continuous improvement (Kaizen)

`vetinari/kaizen/` tracks all improvement opportunities using a PDCA (Plan-Do-Check-Act) lifecycle. The system automatically identifies regressions, logs improvement proposals, and tracks their outcomes. This is a first-class feature — do not remove or bypass it.

### Self-training

`vetinari/training/` runs during idle time to:
- Generate synthetic training data from high-quality inference outputs
- Evolve prompts using feedback signals
- Fine-tune models via LoRA adapters

Training data quality is enforced: records with `tokens_used=0` or `latency_ms=0` are rejected as fallback/mock outputs. Fallback responses must always be flagged with `_is_fallback=True` and are never recorded as training data.

---

## 8. Debugging Tips

| Problem | Solution |
|---------|---------|
| `ImportError: vetinari` | Activate your virtual environment, then run `pip install -e ".[dev,local,ml,search,notifications]"` from the project root |
| `ImportError: No module named X` | The package may not be in `pyproject.toml`. Add it there — never `pip install` without also updating `pyproject.toml` |
| Model not found at startup | Check `VETINARI_MODELS_DIR` points to a directory containing `.gguf` files; run `python -m vetinari init` to re-scan |
| Inference returns empty string | Empty inference results are forbidden — the system should raise or return a typed error. Check adapter logs for the actual failure |
| Thompson Sampling not learning | Verify `tokens_used > 0` and `latency_ms > 0` in recorded inference results. Zero-valued records are discarded |
| Test fails due to singleton state | Call the matching `reset_*()` / `clear_*()` in both `setUp` and `tearDown` |
| Ruff errors on save | Run `python -m ruff check vetinari/ --fix` manually to see which rules are triggering |
| `check_vetinari_rules.py` VET120-124 | A function has no callers. Wire it to a call site (default); only delete if the code is superseded or deprecated — do not suppress the rule |
| Circuit breaker open | The adapter hit its failure threshold. Check adapter logs for the root cause; the breaker resets after the cooldown window |
| Port 5000 already in use | Another Vetinari instance is running, or a stale process is holding the port. `kill $(lsof -ti:5000)` on Linux/macOS |
| Web route returns 500 with no log | Check `vetinari/web/lifespan.py` — startup failures may have left a service uninitialized |

### Reading the test summary

After running `python -m pytest tests/ -x -q`, always read results via:

```bash
python scripts/test_summary.py
```

This parses `.vetinari/test-results.xml` and formats failures with full context. Raw pytest output truncates tracebacks on slow terminals.

### Using the memory CLI

```bash
# Search project memory
python scripts/memory_cli.py search "thompson sampling"

# Start a session with context
python scripts/memory_cli.py session start --context "Working on model router"

# Store a feedback note
python scripts/memory_cli.py store --type feedback --title "Router edge case" --content "..."
```

---

## 9. Getting Help

### In-repo references

| Resource | Location | Contents |
|----------|----------|---------|
| Architecture | `docs/internal/ai-workflows/vetinari-agent-system-reference.md` | Current runtime agent architecture |
| Ontology truth | `docs/audit/ONTOLOGY-AND-ARCHITECTURE-TRUTH-TABLE.md` | Live mode counts, prompt-source truth, authority caveats |
| Pipeline | `docs/architecture/pipeline.md` | Current quick-start pipeline reference |
| Agent prompts | `vetinari/config/agents/` | Runtime prompt text; frontmatter model/tool fields are metadata |
| Standards | `vetinari/config/standards/` | Runtime standards and defect guidance |
| Rules | `vetinari/config/rules.yaml` | Runtime prompt/control-plane rules |
| ADR decisions | `adr/` | Every significant design decision with context and trade-offs |
| Agent pipeline | `AGENTS.md` | Foreman/Worker/Inspector spec and mode catalog |

### Architecture Decision Records

Before changing how something works, check `adr/` for an existing decision record. ADRs explain *why* the system is designed the way it is — reading them prevents re-litigating settled decisions. Key ADRs for new developers:

- **ADR-0061**: Three-agent factory pipeline (Foreman → Worker → Inspector) — the core architecture
- Search `adr/` for the subsystem you are working in: `grep -l "memory" adr/*.json`

### When you need to make a significant decision

If you are making a choice that affects architecture, data models, security, API contracts, or agent behavior, you must write an ADR. ADR creation is mandatory policy, not optional documentation. Use `vetinari/adr.py` and `adr/` as the loader-backed authority; avoid fields that the loader does not persist.

### Asking questions

- Check `adr/`, `docs/internal/ai-workflows/`, and `docs/audit/ONTOLOGY-AND-ARCHITECTURE-TRUTH-TABLE.md` first; many "why" questions are already answered
- For unclear behavior, read the relevant test file — tests are the most accurate specification
- For model/hardware questions, `config/models.yaml` has per-model hardware requirements
