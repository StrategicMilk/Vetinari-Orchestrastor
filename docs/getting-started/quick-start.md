# AM Workbench Quick Start Guide

Get from zero to a running Vetinari engine in under 10 minutes.

## Prerequisites

| Requirement | Minimum | Check |
|---|---|---|
| Python | 3.10+ | `python --version` |
| Git | any | `git --version` |
| Models | GGUF fallback or native-model backend | Provision via `vetinari init` or a `vllm` endpoint |

No external LLM server is required for the GGUF fallback path. If you want the
preferred native-model path on Windows, run `vllm` in WSL and point Vetinari at
`VETINARI_VLLM_ENDPOINT`.

## 1. Clone And Install

```bash
git clone https://github.com/your-org/am-workbench.git
cd am-workbench

python -m venv .venv312
source .venv312/bin/activate          # Windows: .venv312\Scripts\activate

pip install -e ".[dev,local,ml,search,notifications]"
```

`uv` users can substitute:

```bash
uv pip install -e ".[dev,local,ml,search,notifications]"
```

Add `.[training]` only when you need fine-tuning, and `.[vllm]` only when you
intend to run the vLLM backend.

## 2. First-Run Setup

```bash
python -m vetinari init
```

The wizard detects available CPU/GPU resources, recommends model paths, and
writes initial configuration to `~/.vetinari/config.yaml`.

If you already have GGUF files, place them in the directory you specify during
`init` or set `VETINARI_MODELS_DIR`. If you use native models with `vllm` or
NIM, keep those assets under `./models/native` or set
`VETINARI_NATIVE_MODELS_DIR`.

## 3. Verify The Installation

```bash
python -c "import vetinari; print('OK')"
python -m pytest tests/ -x -q
python -m ruff check vetinari/
python scripts/quality/check_vetinari_rules.py
```

## 4. Start The System

```bash
python -m vetinari start
```

To start the API server without an active goal:

```bash
python -m vetinari serve --port 5000
```

## 5. Run Your First Goal

```bash
python -m vetinari start --goal "Summarise the key points of the README and write them to summary.md"
```

Vetinari decomposes the goal through its three-agent factory pipeline:

1. Foreman breaks the goal into a structured plan.
2. Worker executes each task using the most appropriate configured model.
3. Inspector reviews outputs for quality and completeness.

See `docs/architecture/pipeline.md` for details.

## 6. Check Status

```bash
python -m vetinari status
python -m vetinari health
python -m vetinari doctor
```

Use `doctor` when troubleshooting. It checks model loading, memory, config
validity, and backend readiness.

## 7. Manage Models

```bash
python -m vetinari models list
python -m vetinari models scan
python -m vetinari models download <hf-repo-id>
```

Do not commit downloaded model files. Model artifacts belong in local operator
model directories, not in Git history.

## UI

The real UI source is `ui/svelte`.

```bash
cd ui/svelte
npm install
npm run dev
```

The Svelte dev server proxies API calls to the backend. Generated Svelte build
assets go to `ui/static/svelte` and are excluded from Python package artifacts.

## Key Entry Points

| What | Command / URL |
|---|---|
| CLI help | `python -m vetinari --help` |
| Start backend services | `python -m vetinari start` |
| Goal-based execution | `python -m vetinari start --goal "..."` |
| API server | `python -m vetinari serve --port 5000` |
| System status | `python -m vetinari status` |
| Health check | `python -m vetinari health` |
| Diagnostics | `python -m vetinari doctor` |
| First-run wizard | `python -m vetinari init` |
| Svelte dev UI | `cd ui/svelte && npm run dev` |

## Next Steps

- `docs/getting-started/onboarding.md` - full developer walkthrough
- `docs/architecture/pipeline.md` - pipeline internals and agent conventions
- `docs/architecture/decisions.md` - public architecture decisions
- `docs/reference/production.md` - production deployment checklist
- `docs/status/known-limitations.md` - current known limitations
