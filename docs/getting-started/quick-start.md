# Vetinari — Quick Start Guide

Get from zero to a running system in under 10 minutes.

---

## Prerequisites

| Requirement | Minimum | Check |
|---|---|---|
| Python | 3.10+ | `python --version` |
| Git | any | `git --version` |
| Models | GGUF fallback or native-model backend | Or provision via `vetinari init` / WSL `vllm` |

No external LLM server is required for the GGUF fallback path. If you want the preferred native-model path on Windows, run `vllm` in WSL and point Vetinari at `VETINARI_VLLM_ENDPOINT`.

---

## 1. Clone and install

```bash
git clone https://github.com/your-org/vetinari.git
cd vetinari

python -m venv .venv312
source .venv312/bin/activate          # Windows: .venv312\Scripts\activate

pip install -e ".[dev,local,ml,search,notifications]"
```

`uv` users can substitute `uv pip install -e ".[dev,local,ml,search,notifications]"`. Add `.[training]` only when you need fine-tuning, and `.[vllm]` only when you intend to run the vLLM backend. The `pyproject.toml` is the authoritative source of dependencies.

---

## 2. First-run setup

Run the interactive setup wizard. It detects your hardware, recommends models, and writes `~/.vetinari/config.yaml`:

```bash
python -m vetinari init
```

The wizard will:

- Detect available CPU/GPU resources
- Recommend a model path appropriate for your hardware
- Offer to download a model from HuggingFace if you do not already have one
- Write initial configuration to `~/.vetinari/config.yaml`

If you already have GGUF files, place them in the directory you specify during `init` (default: `./models`). If you are using native models with `vllm` or NIM, keep those assets under `./models/native` or set `VETINARI_NATIVE_MODELS_DIR`.

### Manual config (optional)

If you prefer to skip the wizard, set the runtime environment variables directly:

```bash
export VETINARI_MODELS_DIR=./models
export VETINARI_NATIVE_MODELS_DIR=./models/native
export VETINARI_WEB_PORT=5000
```

Key variables to set:

```
VETINARI_MODELS_DIR=./models      # path to your GGUF fallback model files
VETINARI_NATIVE_MODELS_DIR=./models/native
VETINARI_GPU_LAYERS=-1            # GPU layers to offload (-1 = auto-detect)
VETINARI_CONTEXT_LENGTH=8192      # context window size
VETINARI_VLLM_ENDPOINT=http://localhost:8000
VETINARI_VLLM_SETUP_MODE=guided   # manual, guided, or auto
VETINARI_VLLM_MODEL=              # Hugging Face ID or container-visible model path
VETINARI_NIM_ENDPOINT=http://localhost:8001
VETINARI_NIM_SETUP_MODE=guided    # manual, guided, or auto
VETINARI_NIM_IMAGE=               # NGC NIM image, required for guided/auto container setup
```

On Windows + WSL, see the operator setup section in [README.md](C:/dev/Vetinari/README.md) for the exact `vllm` install and endpoint-export commands.

---

## 3. Verify the installation

```bash
python -c "import vetinari; print('OK')"
python -m pytest tests/ -x -q
python -m ruff check vetinari/
python scripts/check_vetinari_rules.py
```

All four checks should pass with zero errors before you proceed.

---

## 4. Start the system

Start Vetinari with the web dashboard:

```bash
python -m vetinari start
```

Open `http://localhost:5000` in your browser. The dashboard shows live agent activity, model routing decisions, and system health.

To start on a different port (dashboard only, no active goal):

```bash
python -m vetinari serve --port 5000
```

---

## 5. Run your first goal

Pass a goal directly on the command line:

```bash
python -m vetinari start --goal "Summarise the key points of the README and write them to summary.md"
```

Vetinari decomposes the goal through its three-agent factory pipeline:

1. **Foreman** — breaks the goal into a structured plan
2. **Worker** — executes each task in the plan using the most appropriate local model
3. **Inspector** — reviews outputs for quality and completeness

The pipeline supports 34 execution modes. The mode is selected automatically based on the task type and the models available. See `docs/architecture/pipeline.md` for details.

---

## 6. Check system status

```bash
python -m vetinari status      # Summary of agents, models, and active work
python -m vetinari health      # Pass/fail health check (useful for CI and scripts)
python -m vetinari doctor      # Full diagnostic report, including backend checks
```

Use `doctor` when troubleshooting. It checks model loading, memory, config validity, and more.

---

## 7. Manage models

```bash
python -m vetinari models list                        # Show loaded models and their status
python -m vetinari models scan                        # Discover GGUF files on disk
python -m vetinari models download <hf-repo-id>       # Download a model from HuggingFace
```

---

## Key entry points

| What | Command / URL |
|---|---|
| CLI help | `python -m vetinari --help` |
| Full start with dashboard | `python -m vetinari start` |
| Goal-based execution | `python -m vetinari start --goal "..."` |
| Dashboard only | `python -m vetinari serve --port 5000` |
| System status | `python -m vetinari status` |
| Health check | `python -m vetinari health` |
| Diagnostics | `python -m vetinari doctor` |
| First-run wizard | `python -m vetinari init` |
| Web dashboard | `http://localhost:5000` |
| REST API | `http://localhost:5000/api` |

---

## Next steps

- `docs/getting-started/onboarding.md` — full onboarding walkthrough
- `docs/architecture/pipeline.md` — pipeline internals and agent conventions
- `docs/reference/production.md` — production deployment checklist
- `adr/` — architecture decision records explaining every major design choice
