"""OpenAI-compatible server adapter for vLLM, NVIDIA NIMs, and similar backends.

Provides a unified adapter for any inference server exposing the OpenAI-compatible
``/v1/chat/completions`` and ``/v1/models`` API endpoints.  This covers:

- **vLLM**: GPU-only, high-throughput local inference server
- **NVIDIA NIMs**: NVIDIA's inference microservice containers
- **Any OpenAI-compatible server**: LM Studio, text-generation-inference, etc.

Key design choice: vLLM and NIMs are GPU-only — they cannot offload model layers
to system RAM.  The ``gpu_only`` flag in ``extra_config`` communicates this
constraint to the DynamicModelRouter so it can exclude oversized models from
these backends while still allowing llama-cpp to run them via CPU offload.

Decision: single adapter for all OpenAI-compatible servers (ADR-0084).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from vetinari.constants import (
    INFERENCE_STATUS_ERROR,
    INFERENCE_STATUS_OK,
)

from .base import InferenceRequest, InferenceResponse, ModelInfo, ProviderAdapter, ProviderConfig
from .llama_cpp_model_info import _infer_capabilities, _infer_context_window

logger = logging.getLogger(__name__)

# Timeout for health checks and model discovery (seconds)
_DISCOVERY_TIMEOUT_S = 10
# Timeout for inference requests (seconds) — overridden by config.timeout_seconds
_DEFAULT_INFERENCE_TIMEOUT_S = 120
# How long (seconds) before the model cache is considered stale
_CACHE_TTL_S = 300


def _httpx() -> Any:
    """Lazy-import httpx to avoid import-time cost when not used."""
    try:
        import httpx

        return httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required for OpenAIServerAdapter.  Install with: pip install httpx",  # noqa: VET301 — user guidance string
        ) from exc


# URL patterns that identify NVIDIA NIM endpoints.
# NIM can be self-hosted (using /nim/ path) or cloud-hosted on NVIDIA's API gateway.
_NIM_URL_PATTERNS = ("api.nvcf.nvidia.com", "ngc.nvidia.com", "/nim/")
SERVER_ADAPTER_CACHE_VERSION = "openai_server_adapter:v2"
_CACHE_SENSITIVE_PAYLOAD_KEYS = {"cache_salt"}
_PROVENANCE_EXTRA_KEYS = (
    "artifact_sha256",
    "cache_namespace",
    "container_digest",
    "container_image",
    "deployment_id",
    "engine_version",
    "gpu_only",
    "kv_cache_host_offload_enabled",
    "kv_cache_reuse_enabled",
    "model_digest",
    "model_revision",
    "nim_profile",
    "prefix_caching_enabled",
    "prefix_caching_hash_algo",
    "served_model_name",
    "served_model_revision",
    "server_build",
    "tensor_parallel_size",
    "tokenizer_revision",
    "vllm_engine_args_hash",
)


def _stable_json(value: Any) -> str:
    """Return stable JSON for cache/provenance identity material."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _stable_hash(value: Any) -> str:
    """Return SHA-256 over stable JSON identity material."""
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _hash_secret(value: object) -> str:
    """Hash sensitive identity material without storing the raw value."""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _cache_safe_payload(value: Any) -> Any:
    """Return payload material safe to store in semantic-cache context."""
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            if key in _CACHE_SENSITIVE_PAYLOAD_KEYS and item is not None:
                safe[key] = {"sha256": _hash_secret(item)}
            else:
                safe[key] = _cache_safe_payload(item)
        return safe
    if isinstance(value, list):
        return [_cache_safe_payload(item) for item in value]
    return value


def _cache_identity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return output-affecting payload identity without embedding the query text."""
    safe = _cache_safe_payload(payload)
    if not isinstance(safe, dict):
        return {}

    messages = safe.get("messages")
    if isinstance(messages, list):
        identity_messages: list[Any] = []
        for message in messages:
            if isinstance(message, dict):
                identity_message = dict(message)
                if identity_message.get("role") == "user":
                    identity_message["content"] = "<query>"
                identity_messages.append(identity_message)
            else:
                identity_messages.append(message)
        safe["messages"] = identity_messages
    return safe


def _detect_provider_label(api_base: str, configured_type_value: str) -> str:
    """Return the correct provider label for a server endpoint.

    Detects NVIDIA NIM endpoints by URL pattern so that ModelInfo objects
    are correctly labelled even when the adapter was registered under a
    generic ``vllm`` provider type.

    Args:
        api_base: The base URL of the inference server (already rstrip'd of ``/``).
        configured_type_value: The ``ProviderType.value`` string from the config.

    Returns:
        ``"nim"`` when the URL matches a known NIM pattern, otherwise
        ``configured_type_value`` unchanged.
    """
    for pattern in _NIM_URL_PATTERNS:
        if pattern in api_base:
            return "nim"
    return configured_type_value


def _coerce_bool(value: object, default: bool) -> bool:
    """Coerce a config value to bool, handling string representations from YAML.

    YAML parses unquoted booleans correctly, but quoted values like ``"false"``
    arrive as strings.  This guard prevents ``"false"`` from being truthy.

    Args:
        value: Raw config value (bool, str, int, or None).
        default: Returned when value is None.

    Returns:
        Coerced boolean.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("false", "0", "no", "off")


def _semantic_cache_identity(
    *,
    request: InferenceRequest,
    provider_type: str,
    provider_name: str,
    provider_label: str,
    api_base: str,
    gpu_only: bool,
    payload: dict[str, Any],
    raw_model_entry: dict[str, Any] | None,
    extra_config: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Build strict cache identity for OpenAI-compatible server backends."""
    model_entry = raw_model_entry or {}
    model_entry_hash = _stable_hash(model_entry)
    provenance = {
        key: extra_config[key]
        for key in _PROVENANCE_EXTRA_KEYS
        if key in extra_config and extra_config[key] not in (None, "")
    }
    provenance_hash = _stable_hash(provenance)
    endpoint_hash = hashlib.sha256(api_base.encode("utf-8")).hexdigest()
    cache_model_id = (
        f"{provider_type}:{provider_name}:{endpoint_hash}:{request.model_id}:{model_entry_hash}:{provenance_hash}"
    )
    identity: dict[str, Any] = {
        "adapter_version": SERVER_ADAPTER_CACHE_VERSION,
        "endpoint_sha256": endpoint_hash,
        "gpu_only": gpu_only,
        "model_entry": model_entry,
        "model_entry_sha256": model_entry_hash,
        "payload": _cache_identity_payload(payload),
        "provider_label": provider_label,
        "provider_name": provider_name,
        "provider_type": provider_type,
        "requested_model_id": request.model_id,
        "server_provenance": provenance,
        "server_provenance_sha256": provenance_hash,
    }
    cache_context = _stable_json(identity)
    return cache_model_id, cache_context, identity


def _get_semantic_cache() -> Any:
    """Return the global semantic cache lazily for patchable server-adapter use."""
    from vetinari.optimization.semantic_cache import get_semantic_cache

    return get_semantic_cache()


class OpenAIServerAdapter(ProviderAdapter):
    """Adapter for OpenAI-compatible inference servers (vLLM, NVIDIA NIMs, etc.).

    Communicates via the standard ``/v1/models`` and ``/v1/chat/completions``
    endpoints.  Does not depend on litellm — uses httpx directly for minimal
    footprint.

    Configuration via ``extra_config``:

    - ``gpu_only`` (bool, default True): Whether the backend requires models
      to fit entirely in VRAM.  True for vLLM, usually True for NIMs.  Accepts
      YAML string values (``"false"`` is correctly coerced to ``False``).
    - ``api_key`` (str, optional): Bearer token for authenticated endpoints.
    - ``semantic_cache_enabled`` (bool, default True): Enables Vetinari's
      local semantic response cache with server/artifact/sampler isolation.
    - ``cache_salt`` (str, vLLM only): Optional request-level vLLM prefix-cache
      salt. The raw value is sent to vLLM but only its hash is stored locally.
    - ``kv_cache_reuse_enabled`` (bool, NIM): Records server-side NIM KV cache
      reuse state in cache provenance. NIM exposes this as deployment config,
      not as an OpenAI request parameter.

    Cache staleness: ``_last_discovery_ts`` records the epoch-second of the last
    successful discovery.  When rediscovery fails and the cache is older than
    ``_CACHE_TTL_S`` seconds, ``_cache_is_stale`` is set to ``True`` and every
    subsequent request logs a WARNING so operators know they are reading stale
    data.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Store endpoint URL and parse backend-specific flags from extra_config.

        Args:
            config: Provider configuration with endpoint URL and optional api_key.
        """
        super().__init__(config)
        self._api_base = config.endpoint.rstrip("/") if config.endpoint else ""
        # Coerce gpu_only — YAML may deliver "false" as a truthy string
        self._gpu_only = _coerce_bool(config.extra_config.get("gpu_only", True), default=True)
        self._semantic_cache_enabled = _coerce_bool(
            config.extra_config.get("semantic_cache_enabled", True),
            default=True,
        )
        self._discovered_models: list[ModelInfo] = []
        self._raw_model_entries: dict[str, dict[str, Any]] = {}
        # Cache staleness tracking — set after failed rediscovery past TTL
        self._last_discovery_ts: float = 0.0  # epoch seconds of last successful discovery
        self._cache_is_stale: bool = False  # True when cache is past TTL and rediscovery failed
        # Cached supported-matrix verdict for the vLLM engine version.
        # _engine_version_checked flips to True after the first successful
        # /version probe so we do not re-fetch on every health check.
        # _engine_version_block_reason is set when the supported matrix flags
        # the running version as a blocker (e.g. vLLM 0.18.1 on SM120). When
        # populated, health_check returns unhealthy with this reason so the
        # router skips this adapter.
        self._engine_version_checked: bool = False
        self._engine_version_block_reason: str | None = None

    # ── Discovery ──────────────────────────────────────────────────────

    def discover_models(self) -> list[ModelInfo]:
        """Query the ``/v1/models`` endpoint and map results to ModelInfo.

        On success, updates the internal cache and resets the stale flag.  On
        failure, marks the cache stale when it is older than ``_CACHE_TTL_S``
        so callers know they are reading potentially outdated data.

        Returns:
            List of ModelInfo objects for models served by this backend.
        """
        if not self._api_base:
            logger.warning("No endpoint configured for %s — cannot discover models", self.name)
            return []

        httpx = _httpx()
        url = f"{self._api_base}/v1/models"
        headers = self._auth_headers()

        try:
            resp = httpx.get(url, headers=headers, timeout=_DISCOVERY_TIMEOUT_S)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning(
                "Model discovery failed for %s at %s — server may not be running: %s",
                self.name,
                url,
                exc,
            )
            # Mark cache stale when past TTL so every subsequent caller sees a warning
            now = time.time()
            if self._last_discovery_ts > 0 and (now - self._last_discovery_ts) > _CACHE_TTL_S:
                self._cache_is_stale = True
                age_s = int(now - self._last_discovery_ts)
                logger.warning(
                    "Serving stale model list for %s — last successful discovery was %ds ago",
                    self.name,
                    age_s,
                )
            return self._discovered_models

        raw = resp.json()
        # Always read from "data"; servers that omit it return an empty list
        data = raw.get("data", [])
        discovered: list[ModelInfo] = []
        raw_model_entries: dict[str, dict[str, Any]] = {}

        for entry in data:
            model_id = entry.get("id", "")
            if not model_id:
                continue
            raw_model_entries[model_id] = dict(entry)

            capabilities = _infer_capabilities(model_id)
            context_len = _infer_context_window(model_id)

            provider_label = _detect_provider_label(self._api_base, self.provider_type.value)
            discovered.append(
                ModelInfo(
                    id=model_id,
                    name=model_id,
                    provider=provider_label,
                    endpoint=self._api_base,
                    capabilities=capabilities,
                    context_len=context_len,
                    memory_gb=0,  # Server manages its own VRAM
                    version=entry.get("owned_by", "unknown"),
                    latency_estimate_ms=500,  # Server-based inference is typically fast
                    throughput_tokens_per_sec=100.0,
                    cost_per_1k_tokens=0.0,  # Local server — no API cost
                    free_tier=True,
                    tags=capabilities,
                )
            )

        self._discovered_models = discovered
        self._raw_model_entries = raw_model_entries
        self.models = discovered
        self._last_discovery_ts = time.time()
        self._cache_is_stale = False
        logger.info("Discovered %d models from %s at %s", len(discovered), self.name, self._api_base)
        return discovered

    # ── Health Check ───────────────────────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        """Verify the server is reachable by hitting ``/v1/models``.

        For vLLM endpoints, also probe ``/version`` once and cross-check the
        engine version against ``config/runtime/supported_matrix.yaml``. If the
        running version is in a known-bad range on this hardware (e.g. vLLM
        0.18.1 on SM120), report unhealthy so the router skips this adapter
        rather than dispatching inference to a regressed engine.

        Returns:
            Dict with ``healthy``, ``reason``, and ``timestamp`` keys.
        """
        if not self._api_base:
            return {
                "healthy": False,
                "reason": f"No endpoint configured for {self.name}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        httpx = _httpx()
        url = f"{self._api_base}/v1/models"
        try:
            resp = httpx.get(url, headers=self._auth_headers(), timeout=_DISCOVERY_TIMEOUT_S)
            resp.raise_for_status()
            model_count = len(resp.json().get("data", []))
        except Exception as exc:
            logger.warning("Health check for %s at %s failed: %s — reporting unhealthy", self.name, url, exc)
            return {
                "healthy": False,
                "reason": f"Cannot reach {self.name} at {url}: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        if self.provider_type.value == "vllm":
            self._check_vllm_engine_version_once(httpx)
            if self._engine_version_block_reason:
                return {
                    "healthy": False,
                    "reason": self._engine_version_block_reason,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        return {
            "healthy": True,
            "reason": f"{model_count} model(s) available",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _check_vllm_engine_version_once(self, httpx_module: Any) -> None:
        """Probe vLLM ``/version`` once and validate against the supported matrix.

        Best-effort: if the endpoint is missing, returns silently and treats the
        engine version as unknown (no block, no false positive). When the matrix
        flags the running version as a blocker on the local hardware, sets
        ``self._engine_version_block_reason`` so subsequent health checks return
        unhealthy without re-probing.

        Args:
            httpx_module: The lazily-imported httpx module already in use by
                the caller; passed to avoid a second import.
        """
        if self._engine_version_checked:
            return
        self._engine_version_checked = True

        version_url = f"{self._api_base}/version"
        try:
            resp = httpx_module.get(version_url, headers=self._auth_headers(), timeout=_DISCOVERY_TIMEOUT_S)
            if resp.status_code != 200:
                logger.debug(
                    "vLLM /version probe at %s returned HTTP %s — skipping known-bad-range gate",
                    version_url,
                    resp.status_code,
                )
                return
            payload = resp.json()
        except Exception as exc:
            # Network or parse failure on /version — record at INFO so operators
            # can see the gate did not run, but do not flip the adapter unhealthy
            # since this is best-effort verification, not a hard precondition.
            logger.info(
                "vLLM /version probe at %s failed (%s) — engine version treated as unknown; "
                "supported-matrix gate skipped for this session",
                version_url,
                exc,
            )
            return

        engine_version = payload.get("version") if isinstance(payload, dict) else None
        if not isinstance(engine_version, str) or not engine_version.strip():
            logger.debug(
                "vLLM /version at %s returned no version string — skipping known-bad-range gate",
                version_url,
            )
            return

        try:
            from vetinari.runtime.runtime_doctor import validate_runtime_version

            result = validate_runtime_version("vllm", engine_version.strip())
        except Exception as exc:
            logger.warning(
                "Could not validate vLLM engine version %r against supported matrix (%s) — "
                "leaving adapter healthy; re-run vetinari --doctor to see matrix diagnostics",
                engine_version,
                exc,
            )
            return

        if not result.passed and result.is_blocker:
            sources = ", ".join(result.matrix_sources) if result.matrix_sources else "supported_matrix.yaml"
            self._engine_version_block_reason = (
                f"vLLM engine version {engine_version} is blocked by the supported matrix: "
                f"{result.reason} Remediation: pin to a version outside the known-bad range "
                f"(see {sources}). Until then, this adapter is held unhealthy so the router skips it."
            )
            logger.error(self._engine_version_block_reason)

    # ── Inference ──────────────────────────────────────────────────────

    def _provider_label(self) -> str:
        """Return the endpoint-aware provider label used for model metadata."""
        return _detect_provider_label(self._api_base, self.provider_type.value)

    def _raw_model_entry_for(self, model_id: str) -> dict[str, Any]:
        """Return provider-reported model metadata captured during discovery."""
        if model_id not in self._raw_model_entries and _coerce_bool(
            self.config.extra_config.get("refresh_model_identity_on_infer"), default=False
        ):
            try:
                self.discover_models()
            except Exception:
                logger.warning("Could not refresh model identity for %s", model_id, exc_info=True)
        return dict(self._raw_model_entries.get(model_id, {}))

    def _server_cache_payload_controls(self, request: InferenceRequest) -> dict[str, Any]:
        """Return cache-related request fields supported by this server backend."""
        controls: dict[str, Any] = {}
        metadata = request.metadata or {}
        extra = self.config.extra_config
        supports_cache_salt = self.provider_type.value == "vllm" or _coerce_bool(
            extra.get("supports_cache_salt", False),
            default=False,
        )

        if supports_cache_salt:
            cache_salt = metadata.get("cache_salt", extra.get("cache_salt"))
            if cache_salt not in (None, ""):
                controls["cache_salt"] = str(cache_salt)

        if self.provider_type.value == "vllm":
            for key in ("kv_transfer_params", "vllm_xargs"):
                value = metadata.get(key, extra.get(key))
                if value not in (None, ""):
                    controls[key] = value

        return controls

    def _cache_identity_for_request(
        self,
        request: InferenceRequest,
        payload: dict[str, Any],
    ) -> tuple[str, str, dict[str, Any]]:
        """Build this request's strict semantic-cache identity."""
        return _semantic_cache_identity(
            request=request,
            provider_type=self.provider_type.value,
            provider_name=self.name,
            provider_label=self._provider_label(),
            api_base=self._api_base,
            gpu_only=self._gpu_only,
            payload=payload,
            raw_model_entry=self._raw_model_entry_for(request.model_id),
            extra_config=self.config.extra_config,
        )

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Send a chat completion request to the OpenAI-compatible endpoint.

        Translates Vetinari's InferenceRequest to the ``/v1/chat/completions``
        payload format, executes the request, and maps the response back.

        Args:
            request: InferenceRequest with model_id, prompt, and sampling params.

        Returns:
            InferenceResponse with the model's output text, or an error response
            when the server is unreachable or returns zero choices.
        """
        self._emit_inference_started(request)
        # Warn on every request when the model list is known stale
        if self._cache_is_stale:
            age_s = int(time.time() - self._last_discovery_ts) if self._last_discovery_ts > 0 else 0
            logger.warning(
                "Serving stale model list for %s — last successful discovery was %ds ago",
                self.name,
                age_s,
            )

        if not self._api_base:
            return InferenceResponse(
                model_id=request.model_id,
                output="",
                latency_ms=0,
                tokens_used=0,
                status=INFERENCE_STATUS_ERROR,
                error=f"No endpoint configured for {self.name}",
            )

        url = f"{self._api_base}/v1/chat/completions"

        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": request.model_id,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": False,
        }

        # Optional sampling parameters — only include non-default values
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences
        if request.frequency_penalty != 0.0:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty != 0.0:
            payload["presence_penalty"] = request.presence_penalty
        if request.seed >= 0:
            payload["seed"] = request.seed
        if request.response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        if request.logit_bias:
            payload["logit_bias"] = {str(k): v for k, v in request.logit_bias.items()}

        # Extended sampler fields — pass through when non-default so backends
        # that support them (e.g. vLLM with custom sampling) receive full params
        if request.top_k > 0:
            payload["top_k"] = request.top_k
        if request.repeat_penalty != 1.0:
            payload["repetition_penalty"] = request.repeat_penalty
        if request.min_p > 0.0:
            payload["min_p"] = request.min_p
        if request.typical_p not in (0.0, 1.0):
            payload["typical_p"] = request.typical_p
        if request.tfs_z not in (0.0, 1.0):
            payload["tfs_z"] = request.tfs_z
        if request.grammar:
            payload["grammar"] = request.grammar

        server_cache_controls = self._server_cache_payload_controls(request)
        payload.update(server_cache_controls)

        cache_model_id = ""
        cache_context = ""
        cache_identity: dict[str, Any] = {}
        try:
            cache_model_id, cache_context, cache_identity = self._cache_identity_for_request(request, payload)
        except Exception:
            logger.warning("Could not build server cache identity for %s", request.model_id, exc_info=True)

        if self._semantic_cache_enabled and cache_model_id and cache_context:
            try:
                cache = _get_semantic_cache()
                cached = cache.get(
                    request.prompt,
                    task_type=request.task_type or "",
                    model_id=cache_model_id,
                    system_prompt=cache_context,
                )
                if cached is not None:
                    logger.debug("Semantic cache hit for %s via %s", request.model_id, self.name)
                    response = InferenceResponse(
                        model_id=request.model_id,
                        output=cached,
                        latency_ms=0,
                        tokens_used=0,
                        status=INFERENCE_STATUS_OK,
                        metadata={
                            "cache_hit": True,
                            "cache_tier": "semantic",
                            "endpoint_sha256": cache_identity.get("endpoint_sha256", ""),
                            "gpu_only": self._gpu_only,
                            "model_entry_sha256": cache_identity.get("model_entry_sha256", ""),
                            "provider": self.name,
                            "provider_label": self._provider_label(),
                            "provider_type": self.provider_type.value,
                            "semantic_cache_enabled": True,
                            "server_cache_controls": _cache_safe_payload(server_cache_controls),
                            "server_identity_sha256": _stable_hash(cache_identity),
                            "server_provenance_sha256": cache_identity.get("server_provenance_sha256", ""),
                        },
                    )
                    self._record_telemetry(request, response)
                    return response
            except Exception:
                logger.warning("Semantic cache unavailable for %s; proceeding without cache", self.name, exc_info=True)

        httpx = _httpx()
        start_ms = time.perf_counter_ns() // 1_000_000

        try:
            resp = httpx.post(
                url,
                json=payload,
                headers=self._auth_headers(),
                timeout=self.timeout_seconds or _DEFAULT_INFERENCE_TIMEOUT_S,
            )
            resp.raise_for_status()
        except Exception as exc:
            latency_ms = (time.perf_counter_ns() // 1_000_000) - start_ms
            logger.warning(
                "Inference failed on %s via %s — %s",
                request.model_id,
                self.name,
                exc,
            )
            response = InferenceResponse(
                model_id=request.model_id,
                output="",
                latency_ms=int(latency_ms),
                tokens_used=0,
                status=INFERENCE_STATUS_ERROR,
                error=f"Inference request to {self.name} failed: {exc}",
            )
            self._record_telemetry(request, response)
            return response

        latency_ms = (time.perf_counter_ns() // 1_000_000) - start_ms
        result = resp.json()

        choices = result.get("choices", [])
        if not choices:
            # Server returned a valid HTTP 200 but no completion choices — this
            # is a protocol violation; treat as an error rather than returning
            # empty output which callers cannot distinguish from a real empty response.
            logger.warning(
                "Inference on %s via %s returned zero choices — treating as error",
                request.model_id,
                self.name,
            )
            response = InferenceResponse(
                model_id=result.get("model", request.model_id),
                output="",
                latency_ms=int(latency_ms),
                tokens_used=0,
                status=INFERENCE_STATUS_ERROR,
                error=f"Server returned zero choices for model {request.model_id} on {self.name}",
            )
            self._record_telemetry(request, response)
            return response

        message = choices[0].get("message", {})
        output = message.get("content", "")

        usage = result.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)

        response = InferenceResponse(
            model_id=result.get("model", request.model_id),
            output=output,
            latency_ms=int(latency_ms),
            tokens_used=tokens_used,
            status=INFERENCE_STATUS_OK,
            metadata={
                "cache_hit": False,
                "completion_tokens": usage.get("completion_tokens", 0),
                "endpoint_sha256": cache_identity.get("endpoint_sha256", ""),
                "gpu_only": self._gpu_only,
                "model_entry_sha256": cache_identity.get("model_entry_sha256", ""),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "provider": self.name,
                "provider_label": self._provider_label(),
                "provider_type": self.provider_type.value,
                "semantic_cache_enabled": self._semantic_cache_enabled,
                "server_cache_controls": _cache_safe_payload(server_cache_controls),
                "server_identity_sha256": _stable_hash(cache_identity) if cache_identity else "",
                "server_model": result.get("model", request.model_id),
                "server_provenance_sha256": cache_identity.get("server_provenance_sha256", ""),
                "system_fingerprint": result.get("system_fingerprint"),
            },
        )
        self._record_telemetry(request, response)
        if self._semantic_cache_enabled and response.output and cache_model_id and cache_context:
            try:
                _get_semantic_cache().put(
                    request.prompt,
                    response.output,
                    model_id=cache_model_id,
                    system_prompt=cache_context,
                )
            except Exception:
                logger.warning(
                    "Semantic cache store failed for %s; result will not be cached",
                    self.name,
                    exc_info=True,
                )
        return response

    # ── Capabilities ───────────────────────────────────────────────────

    def get_capabilities(self) -> dict[str, list[str]]:
        """Return inferred capabilities for all discovered models.

        Returns:
            Dict mapping model_id to list of capability tags.
        """
        if not self._discovered_models:
            self.discover_models()
        return {m.id: m.capabilities for m in self._discovered_models}

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def gpu_only(self) -> bool:
        """Whether this backend requires models to fit entirely in VRAM.

        vLLM is always GPU-only.  NIMs may support CPU offload in some
        configurations but default to GPU-only.
        """
        return bool(self._gpu_only)

    # ── Internals ──────────────────────────────────────────────────────

    def _auth_headers(self) -> dict[str, str]:
        """Build HTTP headers with optional Bearer auth.

        Returns:
            Headers dict with Content-Type and optional Authorization.
        """
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
