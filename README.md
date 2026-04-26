# AM Workbench

**v0.6.0** - Local-first multi-agent LLM orchestration with self-improving model routing, quality gates, and cost-optimised inference.

## What It Is

AM Workbench is a local-first automation and orchestration workbench. Vetinari
is its orchestration engine, Python package, and CLI command.

Vetinari coordinates multiple LLM agents to plan, research, build, and verify
complex software tasks. It now prefers native-model backends such as vLLM and
NVIDIA NIM, with local GGUF inference via llama-cpp-python kept as the fallback
path.

Three specialised agents form a factory pipeline (ADR-0061):

```
Orchestrate → Execute → Review
  Foreman      Worker    Inspector
```

The three agents expose 34 total execution modes. A DAG-based task scheduler runs independent tasks in parallel, escalates failures through a quality gate, and feeds outcomes back into a Bayesian model selection loop.

## What It Does Differently

**Self-improving routing.** Thompson Sampling (Beta-distribution bandits) learns which model performs best for each task type, balancing exploration of new models against exploitation of proven ones. The same mechanism extends to strategy hyperparameters (temperature, context window size, prompt template variant, decomposition granularity).

**Structured prompt optimisation.** Eight deterministic mutation operators (rephrase, constrain, inject examples, restructure format, scaffold reasoning, reinforce role, tighten output schema, prune context) evolve agent prompts through A/B testing with statistical significance gates (paired t-test, Cohen's d ≥ 0.2). Thompson Sampling selects which operators work best for which agent modes.

**Cost-aware cascade routing.** Tasks start on the cheapest model and escalate only on low confidence. Combined with per-task token budgets and batch API support, this reduces cloud inference costs by 30-60% on mixed workloads.

**Quality gates with corrective routing.** The Inspector agent issues mandatory pass/fail decisions. Rejections are classified by defect category and routed to corrective actions (replan, retry with different model, research then retry, escalate to user). Confirmed improvements are monitored for regression with automatic revert.

**Test-time compute scaling.** Budget forcing caps thinking tokens per tier/complexity. Best-of-N selection generates multiple candidates for high-stakes tasks and returns the highest-scoring one via quality pre-screening.

**Predictive cost modelling.** Before a task executes, the system estimates token usage, latency, and cost using calibrated regression on historical outcomes.

## Architecture

### Agent Pipeline (ADR-0061)

| # | Agent | Modes | Role |
|---|-------|-------|------|
| 1 | **Foreman** | plan, clarify, summarise, prune, extract, consolidate (6) | Orchestrate, decompose, sequence |
| 2 | **Worker** | 24 modes across 4 groups: Research (8), Architecture (5), Build (2), Operations (9) | All production work — research, decide, build, operate |
| 3 | **Inspector** | code_review, security_audit, test_generation, simplification (4) | Mandatory pass/fail gate |

### Execution Flow

```
User Input
    │
    ▼
[Input Analyzer]     Classify request type, domain, complexity
    │
    ▼
[Plan Generator]     Foreman decomposes into task DAG
    │
    ▼
[Model Assigner]     CascadeRouter + Thompson Sampling selects model per task
    │
    ▼
[DAG Executor]       ThreadPoolExecutor runs independent tasks in parallel
    │
    ▼
[Quality Gate]       Quality agent reviews outputs (pass/fail)
    │
    ▼
[Assembler]          Operations agent produces final output
```

### Cascade Routing

```
Task arrives → Start with cheapest model (7B)
                  │
            confidence ≥ 0.7? ──yes──► Return result
                  │ no
            Escalate to 30B
                  │
            confidence ≥ 0.7? ──yes──► Return result
                  │ no
            Escalate to 72B
```

### Self-Improvement Loop

```
Execution → Quality Scorer (LLM-as-judge + heuristics)
               │
               ▼
           Feedback Loop → Thompson Sampling arm updates
               │
               ▼
           Cost Optimizer → Route to cheapest adequate model
               │
               ▼
           Auto-Tuner → Adjust concurrency, thresholds from SLA data
```

### Memory System

| Layer | Storage | Purpose |
|-------|---------|---------|
| **Session** | In-memory with file backup | Active task context |
| **Long-term** | SQLite + FTS5 | Searchable store with BM25 ranking |
| **Embeddings** | SQLite (float32 BLOBs) | Semantic similarity via local inference |

### Continuous Improvement

All system improvements are tracked as first-class entities with PDCA lifecycle:

- **ImprovementLog**: propose → activate → observe → evaluate → confirm or revert
- **Automated review**: identifies recurring failures, high rework rates, low value-add ratios
- **Regression monitoring**: 10%+ degradation triggers auto-revert of confirmed improvements
- **Defect trends**: week-over-week tracking with hotspot identification
- **Root-cause routing**: Quality rejections classified by defect category and routed to appropriate corrective action

### Analytics API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analytics/cost` | GET | Cost breakdown by model, agent, time period |
| `/api/analytics/sla` | GET | SLA compliance metrics |
| `/api/analytics/anomalies` | GET | Anomaly detection |
| `/api/analytics/forecast` | GET | Cost and usage forecasting |
| `/api/analytics/models` | GET | Per-model performance stats |
| `/api/analytics/agents` | GET | Per-agent utilization metrics |
| `/api/analytics/summary` | GET | System-wide summary |

## Quick Start

### Setup

```bash
cd am-workbench

# Create the canonical project environment
python -m venv .venv312
source .venv312/bin/activate          # Windows: .venv312\Scripts\activate

# Install the contributor baseline (CLI, local inference, search, notifications, dev tools)
uv pip install -e ".[dev,local,ml,search,notifications]"
# Or with pip:
pip install -e ".[dev,local,ml,search,notifications]"

# Add heavy optional stacks only when needed
pip install -e ".[training]"   # QLoRA / fine-tuning
pip install -e ".[vllm]"       # vLLM backend

# Run the setup wizard
python -m vetinari init
```

### Native Backend Setup On Windows + WSL

If you want `vllm` or NIM to be the primary backend on Windows, the supported path is:

1. Run Vetinari itself from the Windows `.venv312` environment.
2. Run `vllm` inside WSL Ubuntu.
3. Point the Windows shell at the WSL-hosted endpoint with `VETINARI_VLLM_ENDPOINT`.
4. Keep local GGUF files available as the `llama.cpp` fallback.

Operator download/install links:

- [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
- [Get Ubuntu on WSL](https://ubuntu.com/download/wsl)
- [Ubuntu on the Microsoft Store](https://apps.microsoft.com/detail/9pdxgncfsczv?gl=US&hl=en-US)
- [Install a specific Ubuntu distro on WSL](https://documentation.ubuntu.com/wsl/latest/howto/install-ubuntu-wsl2/)
- [vLLM GPU installation docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [NVIDIA Windows driver download](https://www.nvidia.com/Download/index.aspx)

WSL-side install and server startup:

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

# Verify the install:
python -c "import vllm, torch; print('vllm', vllm.__version__); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"

# Example server using a Hub model:
vllm serve Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8000

# Or serve a native local model directory:
vllm serve /mnt/c/dev/am-workbench/models/native/YOUR_MODEL_DIR --host 0.0.0.0 --port 8000
```

Windows-side shell setup for Vetinari:

```powershell
. .\enter-vetinari-shell.ps1
$env:VETINARI_VLLM_ENDPOINT = "http://localhost:8000"
$env:VETINARI_NATIVE_MODELS_DIR = ".\models\native"
python -m vetinari doctor --json
```

Repo helper for repeatable WSL startup:

```powershell
. .\enter-vetinari-shell.ps1
.\start-vllm-wsl.ps1
python -m vetinari doctor --json
```

Optional variants:

```powershell
# Serve a specific local native model directory
.\start-vllm-wsl.ps1 -Model .\models\native\YOUR_MODEL_DIR

# Force-restart the server on the default port
.\start-vllm-wsl.ps1 -ForceRestart

# Stop the WSL-hosted server later
.\stop-vllm-wsl.ps1
```

Optional permanent PowerShell setup:

```powershell
if (!(Test-Path $PROFILE)) { New-Item -ItemType File -Path $PROFILE -Force | Out-Null }
Add-Content $PROFILE "`n. <repo>\\enter-vetinari-shell.ps1"
Add-Content $PROFILE "`n`$env:VETINARI_VLLM_ENDPOINT = 'http://localhost:8000'"
Add-Content $PROFILE "`n`$env:VETINARI_NATIVE_MODELS_DIR = '<repo>\\models\\native'"
. $PROFILE
```

Manual requirements for operators:

- `WSL2` with an Ubuntu distro installed and running
- NVIDIA drivers working in WSL (`nvidia-smi` should work inside Ubuntu)
- a running OpenAI-compatible native backend endpoint, usually `vllm` on `http://localhost:8000`
- native Hugging Face-format models under `models/native/` for `vllm`/NIM
- GGUF files under `models/` for the local `llama.cpp` fallback
- `python -m vetinari doctor --json` should show the backend checks you expect before treating setup as complete

If you are missing a prerequisite, use the official links above rather than installing ad hoc Windows-native Linux tooling.

Full WSL-native operation can work when the repo, virtual environment, model
paths, and server all live on the Linux filesystem. It is not the documented
primary Windows path yet. The checked-in helper scripts target a Windows host
that points at a WSL-hosted native backend endpoint, and they do not maintain
the Ubuntu distro, CUDA packages, drivers, or model downloads for you.

### Run

```bash
# Start backend services and optional goal execution
python -m vetinari start

# Goal-based execution
python -m vetinari start --goal "Build a Python REST API with user authentication"

# API/backend server only
python -m vetinari serve --port 5000

# CLI only
python -m vetinari start --no-dashboard --goal "Research best practices for microservices"
```

### CLI Commands

```
vetinari run       --goal "..."    Execute a goal through the pipeline
vetinari run       --task t1       Execute a specific manifest task
vetinari serve     --port 5000     Start API/backend server
vetinari start                     Start backend services and optional goal execution
vetinari status                    Show system status
vetinari health                    Health check all providers
vetinari upgrade                   Check for model upgrades
vetinari review                    Run self-improvement analysis
vetinari interactive               Enter REPL mode

Global flags:
  --config PATH   Manifest file path
  --mode          planning | execution | sandbox
  --verbose       Debug logging
```

## Security

- **Per-agent permissions**: Read/write/execute capabilities enforced per agent type
- **Sandbox**: Rate-limited code execution with filesystem allowlists and network isolation
- **Guardrails**: Input/output validation pipeline with configurable rules
- **Audit logging**: Structured JSONL trail for all agent actions
- **Authentication**: Litestar `admin_guard` protects selected control routes with `VETINARI_ADMIN_TOKEN` via `X-Admin-Token` or `Authorization: Bearer <token>`, using `hmac.compare_digest` for token comparison. The route matrices still list public and CSRF-only surfaces; do not treat every mutating endpoint as uniformly admin-protected until Sessions 34F/34I close the proof.
- **Rate limiting**: Request limiting is configured with `VETINARI_RATE_LIMIT_RPM` and `VETINARI_RATE_LIMIT_BURST`; treat endpoint-specific enforcement as route-matrix evidence, not a blanket release claim.
- **Credential vault**: Fail-closed Fernet encryption
- **Cross-model validation**: Opt-in secondary model verification for critical decisions
- **MCP integration**: Client (`vetinari/mcp/client.py`) consumes external MCP tools via subprocess stdio transport; server (`vetinari/mcp/server.py`) exposes Vetinari capabilities to MCP-compatible hosts. CLI: `vetinari mcp`; web: `/mcp/*` routes.

## Configuration

### Environment Variables

```bash
VETINARI_MODELS_DIR=./models     # Directory containing GGUF fallback model files
VETINARI_NATIVE_MODELS_DIR=./models/native  # Native HF-format model root for vLLM/NIM
VETINARI_GPU_LAYERS=-1           # GPU layers to offload (-1 = auto)
VETINARI_CONTEXT_LENGTH=8192     # Context window size (default: 8192)
VETINARI_VLLM_ENDPOINT=http://localhost:8000
VETINARI_NIM_ENDPOINT=http://localhost:8001
CLAUDE_API_KEY=your_key          # Optional
GEMINI_API_KEY=your_key          # Optional
OPENAI_API_KEY=your_key          # Optional
VETINARI_ADMIN_TOKEN=your_token  # Required before exposing admin routes beyond local trusted use
PLAN_MODE_ENABLE=true
PLAN_DEPTH_CAP=16
VETINARI_SEARCH_BACKEND=duckduckgo
VETINARI_WEB_PORT=5000
```

## Testing

```bash
.\python.cmd scripts/dev/run_tests.py                 # Run tests + readable summary on Windows
.\python.cmd -m pytest tests/ --cov=vetinari      # With coverage
python -c "import vetinari; print('OK')"          # Verify import
python scripts/quality/check_vetinari_rules.py            # Custom VET rule gate
```

## Known Limitations

Detailed public status lives in `docs/status/known-limitations.md`.

- **Single-user/local-only**: The dashboard and API assume a trusted local operator. They do not provide multi-user tenancy, per-user data isolation, or internet-facing auth hardening. Put an external proxy/auth layer in front before exposing anything beyond localhost.
- **Cloud fallback**: Cloud API adapters (Anthropic, OpenAI, Gemini) are integrated but minimally tested in production workflows. Native vLLM/NIM endpoints are the intended primary path; local GGUF inference remains the backup.
- **Native backend on Windows**: Upstream `vllm` is still a Linux/WSL path. The supported Windows setup is a Windows `.venv312` running Vetinari against a WSL-hosted vLLM/NIM endpoint. Full WSL-native operation remains a manual advanced path.
- **WSL maintenance burden**: The repo helpers can start/stop the WSL-hosted server, but distro updates, CUDA/driver repair, Python package repair, and model placement remain operator-managed.
- **Async pipeline**: The canonical Foreman/Worker/Inspector pipeline is still mostly synchronous and uses thread pools for parallelism. Async support modules exist, but they are not the main execution path.
- **MCP integration**: Stdio server/client paths and external tool registration exist. MCP is still not the canonical Worker execution pipeline, HTTP is the Litestar JSON-RPC endpoint at `/mcp/message`, and HTTP+SSE resource streaming is future work.
- **Training pipeline**: Local QLoRA/DoRA training is optional and gated by the `training` extra plus usable ML dependencies. Training jobs now record failed/degraded outcomes, but cloud training and fully automated production retraining remain unsupported.
- **Benchmarks**: `python scripts/inspect/run_benchmarks.py` and `run_ci_benchmarks()` are smoke gates, not canonical runtime p50/p95/p99 claims. Benchmark adapters now fail closed when the live planner/orchestrator/tool path is unavailable instead of scoring expected-output fallbacks.
- **Model discovery**: GGUF and native Hugging Face-format assets are discovered under configured roots, but new files still require `vetinari models scan` or a running native endpoint to be visible.
- **Dashboard metrics**: Telemetry counters restore from the latest SQLite snapshot on startup. Counters after the last snapshot and before a crash are still lost.

Resolved limitations retained for traceability:

- **SSE replay**: Completed events are persisted to SQLite (`sse_event_log`) and can be replayed after reconnect. Mid-stream live queue delivery is still ephemeral.
- **ConversationStore**: Chat history is backed by SQLite (`conversation_messages`), with an in-memory LRU cache that reloads evicted sessions on demand.

## Requirements

- Python 3.10+
- Either a reachable vLLM/NIM endpoint for native-model serving or GGUF model files in the configured fallback directory (default: `./models`)
- llama-cpp-python for GGUF fallback (installed with `pip install -e ".[dev,local,ml,search,notifications]"`)
- Windows 10/11 or Linux/macOS
- 16GB+ RAM (64GB+ recommended for large models)

## License

MIT License. See `pyproject.toml` for details.
