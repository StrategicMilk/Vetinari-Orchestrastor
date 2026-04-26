# Troubleshooting Guide

Common errors and how to fix them. Run `vetinari doctor` for automated diagnostics.

## Model & Inference

### "Local inference: UNREACHABLE"

**Cause:** No GGUF model file found or llama-cpp-python not installed.

**Fix:**
1. Check `VETINARI_MODELS_DIR` points to a directory with `.gguf` files
2. Run `vetinari models scan` to discover models on disk
3. Run `vetinari init` to download a recommended model
4. Verify llama-cpp-python is installed: `pip install llama-cpp-python`

### "Model not found: {id}"

**Cause:** The requested model ID doesn't match any loaded model.

**Fix:**
1. Run `vetinari models scan` to list available models
2. Download a model: `vetinari models download <repo_id>`
3. Check model ID spelling in your config

### "Circuit breaker OPEN for {model_id}"

**Cause:** The model failed repeatedly and the circuit breaker tripped.

**Fix:**
1. Check model health: `vetinari health`
2. Review recent errors: check logs for the specific model
3. Wait for the circuit breaker cooldown (shown in the error message)
4. If persistent, the model file may be corrupted — redownload it

### "CUDA out of memory"

**Cause:** Model requires more VRAM than available on your GPU.

**Fix:**
1. Use a smaller quantization: Q4_K_M instead of Q6_K
2. Reduce `n_ctx` in config (e.g. 2048 instead of 4096)
3. Set `n_gpu_layers` to a lower number (partial GPU offload)
4. Run `vetinari init` to get hardware-appropriate recommendations

## Configuration

### "Inference config not found at config/task_inference_profiles.json"

**Cause:** Missing inference config file (non-fatal, defaults are used).

**Fix:**
1. This is safe to ignore — sensible defaults are applied
2. The file `config/task_inference_profiles.json` ships with the project and contains ready-to-use defaults — no copy step is needed. If the file is missing, restore it from git: `git checkout HEAD -- config/task_inference_profiles.json`

### "Config validation error"

**Cause:** Invalid value in YAML or JSON config file.

**Fix:**
1. Run `vetinari doctor` to identify the specific config issue
2. Check YAML syntax (indentation must use spaces, not tabs)
3. Run `vetinari config reload` after fixing

## Web Dashboard

### "Dashboard port 5000 already in use"

**Cause:** Another process is using port 5000.

**Fix:**
1. Use a different port: `vetinari serve --port 5001`
2. Set `VETINARI_WEB_PORT=5001` in your environment
3. Find and stop the conflicting process: `lsof -i :5000` (macOS/Linux) or `netstat -ano | findstr :5000` (Windows)

### "Web UI not loading" or import errors on startup

**Cause:** Missing web dependencies or incomplete installation.

**Fix:**
```bash
pip install -e ".[dev]"
# Or with uv:
uv pip install -e ".[dev]"
```

## Database & Memory

### "SQLite database locked"

**Cause:** Multiple processes accessing the database simultaneously.

**Fix:**
1. Ensure only one Vetinari instance is running
2. Check for stale lock files: `vetinari doctor`
3. Delete stale `.lock` files in `~/.vetinari/`

### "Migration failed"

**Cause:** Database schema is outdated.

**Fix:**
```bash
vetinari migrate --db-path vetinari_memory.db
```

## Agent Pipeline

### "Task failed after 3 retries"

**Cause:** The agent couldn't produce acceptable output after max retries.

**Fix:**
1. Check the task description — is it clear and specific?
2. Review Inspector rejection reasons in the logs
3. Try a more capable model for complex tasks
4. Decompose the task into smaller subtasks

### "Orchestrator not available"

**Cause:** The two-layer orchestrator couldn't be initialized.

**Fix:**
1. Run `vetinari health` to check all subsystems
2. Verify local inference is working: `vetinari status`
3. Check logs for initialization errors

## Installation

### "ModuleNotFoundError: No module named 'vetinari'"

**Cause:** Vetinari package not installed in the current Python environment.

**Fix:**
```bash
pip install -e "."
# Verify:
python -c "import vetinari; print('OK')"
```

### "No GPU detected (CPU inference only)"

**Cause:** GPU driver not installed or pynvml package missing.

**Fix:**
1. Install pynvml: `pip install nvidia-ml-py` (NVIDIA)
2. Verify GPU driver: `nvidia-smi` (should show your GPU)
3. For AMD: install ROCm and ensure `rocm-smi` is on PATH
4. For Apple Silicon: Metal support is automatic on macOS arm64

## Getting Help

- Run `vetinari doctor` for automated health checks
- Run `vetinari status` to see system state
- Check `VETINARI_LOG_LEVEL=DEBUG vetinari start` for verbose logs
- File issues at the project repository
