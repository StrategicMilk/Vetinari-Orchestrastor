"""
Vetinari Constants
==================
All magic numbers and default values in one place.
Import from here rather than scattering literals throughout the codebase.
"""

import os

# ── LM Studio / API ──────────────────────────────────────────────────────────
DEFAULT_LM_STUDIO_HOST: str = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")
DEFAULT_API_TOKEN: str = os.environ.get("LM_STUDIO_API_TOKEN", "")

# ── Web UI ────────────────────────────────────────────────────────────────────
DEFAULT_WEB_PORT: int = int(os.environ.get("VETINARI_WEB_PORT", 5000))
DEFAULT_WEB_HOST: str = os.environ.get("VETINARI_WEB_HOST", "0.0.0.0")
FLASK_DEBUG: bool = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")

# ── Timeouts (seconds) ────────────────────────────────────────────────────────
TIMEOUT_SHORT: int = 5       # Quick health checks, model discovery pings
TIMEOUT_MEDIUM: int = 30     # Web searches, API calls
TIMEOUT_LONG: int = 60       # Model inference calls
TIMEOUT_VERY_LONG: int = 180 # Long-running LLM completions
TIMEOUT_TRAINING: int = 3600 # Training runs

# ── Retry policies ────────────────────────────────────────────────────────────
MAX_RETRIES: int = 3
RETRY_BASE_DELAY: float = 2.0   # Seconds; used in 2^attempt backoff

# ── Execution ─────────────────────────────────────────────────────────────────
DEFAULT_MAX_CONCURRENT: int = 4
DEFAULT_PLAN_DEPTH_CAP: int = int(os.environ.get("PLAN_DEPTH_CAP", 16))
MIN_PLAN_DEPTH: int = 12
MAX_PLAN_DEPTH: int = 16
MIN_TASKS_PER_PLAN: int = 3   # Minimum valid tasks (was 5, now 3 to allow simple goals)
MAX_TASKS_PER_PLAN: int = 20

# ── Context / Tokens ─────────────────────────────────────────────────────────
DEFAULT_CONTEXT_BUDGET: int = 28000   # Characters (~7K tokens)
DEFAULT_MAX_TOKENS: int = 4096
TOKEN_ESTIMATION_RATIO: int = 4       # ~4 chars per token (rough estimate)

# ── Truncation limits ────────────────────────────────────────────────────────
TRUNCATE_PROMPT: int = 4000
TRUNCATE_RESPONSE: int = 8000
TRUNCATE_CONTEXT: int = 6000
TRUNCATE_TASK_DESC: int = 500
TRUNCATE_SEARCH_RESULT: int = 1000

# ── Quality thresholds ───────────────────────────────────────────────────────
DEFAULT_QUALITY_THRESHOLD: float = 0.70
HIGH_QUALITY_THRESHOLD: float = 0.85
CRITICAL_QUALITY_THRESHOLD: float = 0.50

# ── Model cache ──────────────────────────────────────────────────────────────
MODELS_CACHE_TTL: float = 60.0        # Seconds
MODEL_SEARCH_CACHE_DAYS: int = 7      # Days to cache external search results

# ── Training ─────────────────────────────────────────────────────────────────
MIN_TRAINING_EXAMPLES: int = 50       # Minimum records before training
TRAINING_DATA_FILE: str = "training_data.jsonl"
TRAINING_ROTATION_LIMIT: int = 100_000  # Rotate JSONL after this many records

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR: str = "./vetinari_checkpoints"
OUTPUTS_DIR: str = "./outputs"
PROJECTS_DIR: str = "./projects"
LOGS_DIR: str = "./logs"
MODEL_CACHE_DIR: str = "./model_cache"

# ── Coding bridge ─────────────────────────────────────────────────────────────
CODING_BRIDGE_HOST: str = os.environ.get("CODING_BRIDGE_HOST", "http://localhost:4096")
CODING_BRIDGE_ENABLED: bool = os.environ.get("CODING_BRIDGE_ENABLED", "false").lower() in ("1", "true", "yes")

# ── Image generation (Stable Diffusion) ──────────────────────────────────────
SD_WEBUI_HOST: str = os.environ.get("SD_WEBUI_HOST", "http://localhost:7860")
SD_WEBUI_ENABLED: bool = os.environ.get("SD_WEBUI_ENABLED", "false").lower() in ("1", "true", "yes")
SD_DEFAULT_WIDTH: int = 512
SD_DEFAULT_HEIGHT: int = 512
SD_DEFAULT_STEPS: int = 20
SD_DEFAULT_CFG: float = 7.0

# ── Admin / Security ─────────────────────────────────────────────────────────
ADMIN_TOKEN: str = os.environ.get("VETINARI_ADMIN_TOKEN", "")
ENABLE_EXTERNAL_DISCOVERY: bool = os.environ.get(
    "ENABLE_EXTERNAL_DISCOVERY", "true"
).lower() in ("1", "true", "yes")

# ── Plan Mode ────────────────────────────────────────────────────────────────
PLAN_MODE_ENABLE: bool = os.environ.get("PLAN_MODE_ENABLE", "true").lower() in ("1", "true", "yes")
PLAN_MODE_DEFAULT: bool = os.environ.get("PLAN_MODE_DEFAULT", "true").lower() in ("1", "true", "yes")
PLAN_MAX_CANDIDATES: int = int(os.environ.get("PLAN_MAX_CANDIDATES", 3))
DRY_RUN_ENABLED: bool = os.environ.get("DRY_RUN_ENABLED", "false").lower() in ("1", "true", "yes")

# ── Execution ────────────────────────────────────────────────────────────────
EXECUTION_MODE: str = os.environ.get("EXECUTION_MODE", "execution").lower()
VERIFICATION_LEVEL: str = os.environ.get("VERIFICATION_LEVEL", "standard").lower()
UPGRADE_AUTO_APPROVE: bool = os.environ.get(
    "VETINARI_UPGRADE_AUTO_APPROVE", "false"
).lower() in ("1", "true", "yes")

# ── External services ─────────────────────────────────────────────────────────
ELASTICSEARCH_HOST: str = os.environ.get("ELASTICSEARCH_HOST", "http://localhost:9200")
SPLUNK_HOST: str = os.environ.get("SPLUNK_HOST", "http://localhost:8088")
