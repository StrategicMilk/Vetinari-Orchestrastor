"""LLM inference mixin for Vetinari agents.

Provides _infer(), _infer_json(), and _infer_with_fallback() to any agent
that inherits from BaseAgent.  These methods rely on attributes set by
BaseAgent.__init__() and BaseAgent.initialize() — they MUST NOT be used
standalone.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from vetinari.adapters.base import InferenceRequest
from vetinari.agents import inference_dependencies as _inference_dependencies
from vetinari.agents import inference_json as _inference_json
from vetinari.agents.observability import _ObservabilitySpan
from vetinari.agents.prompt_loader import check_prompt_budget
from vetinari.constants import BATCH_RESULT_TIMEOUT, INFERENCE_STATUS_OK
from vetinari.exceptions import InferenceError, ModelUnavailableError

logger = logging.getLogger(__name__)

_JSON_EXTRACT_RE = _inference_json._JSON_EXTRACT_RE
_MAX_JSON_RETRIES = _inference_json._MAX_JSON_RETRIES
_json_retry_counts = _inference_json._json_retry_counts
_json_retry_lock = _inference_json._json_retry_lock
get_json_retry_stats = _inference_json.get_json_retry_stats

_get_circuit_breaker_registry = _inference_dependencies._get_circuit_breaker_registry
_get_local_preprocessor_cls = _inference_dependencies._get_local_preprocessor_cls
_PROMPT_EVOLVER_ENABLED = _inference_dependencies._PROMPT_EVOLVER_ENABLED
_LOCAL_ONLY_MODE = _inference_dependencies._LOCAL_ONLY_MODE
_lazy_get_adapter_manager = _inference_dependencies._lazy_get_adapter_manager
_lazy_get_prompt_evolver = _inference_dependencies._lazy_get_prompt_evolver
_lazy_get_prompt_assembler = _inference_dependencies._lazy_get_prompt_assembler
_lazy_get_inference_config = _inference_dependencies._lazy_get_inference_config
_lazy_get_token_optimizer = _inference_dependencies._lazy_get_token_optimizer
_lazy_get_batch_processor = _inference_dependencies._lazy_get_batch_processor
_lazy_get_thompson_strategy = _inference_dependencies._lazy_get_thompson_strategy
_lazy_get_semantic_cache = _inference_dependencies._lazy_get_semantic_cache

# Per-request constant — allocated once (Item 1.26)
_EPHEMERAL_CACHE_CONTROL: dict[str, str] = {"type": "ephemeral"}


class InferenceMixin(_inference_json._JsonInferenceMixin):
    """LLM inference capabilities mixed into BaseAgent.

    All methods rely on ``self._adapter_manager``, ``self.agent_type``,
    ``self.default_model``, ``self.get_system_prompt()``, and
    ``self._log()`` being present (supplied by BaseAgent at runtime).
    """

    def _infer(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_id: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        expect_json: bool = False,
        use_cascade: bool = False,
    ) -> str:
        """Call an LLM via the AdapterManager and return the text output.

        Falls back gracefully if the adapter manager is unavailable.

        Args:
            prompt: The user/task prompt.
            system_prompt: Optional system prompt override. Uses agent's
                           get_system_prompt() when not provided.
            model_id: Optional model override. Uses agent's default_model
                      when not provided.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            expect_json: If True, appends a JSON-output instruction and
                         attempts to strip markdown fences from the result.
            use_cascade: If True, route this call through the CascadeRouter
                         (try cheapest model first, escalate on low
                         confidence).  Requires the AdapterManager to have
                         been configured via ``enable_cascade_routing()``.

        Returns:
            The generated text string.

        Raises:
            ModelUnavailableError: If no inference adapter is available.
            InferenceError: If the circuit breaker is open, the token budget is
                exhausted, or an unexpected exception occurs during inference.
        """
        # ── C1: Circuit breaker pre-check (skip in LOCAL_ONLY_MODE) ───
        _cb_registry = None if _LOCAL_ONLY_MODE else _get_circuit_breaker_registry()
        if _cb_registry is not None:
            try:
                _cb = _cb_registry.get(self.agent_type.value)
                if not _cb.allow_request():
                    self._log("warning", "Circuit breaker OPEN for %s", self.agent_type.value)
                    raise InferenceError(f"Circuit breaker open for {self.agent_type.value}")
            except InferenceError:
                raise
            except Exception as _cb_err:
                logger.warning("Circuit breaker check failed: %s", _cb_err)

        # ── C5: Budget pre-check (iteration cap + token budget) ────────
        _bt = getattr(self, "_budget_tracker", None)
        if _bt is not None and not _bt.check_budget():
            _snap = _bt.snapshot()
            self._log(
                "warning",
                "Budget exhausted for %s (tokens=%d, iterations=%d)",
                self.agent_type.value,
                _snap.tokens_used,
                _snap.iterations_used,
            )
            raise InferenceError(
                f"Budget exhausted for {self.agent_type.value}: "
                f"tokens_used={_snap.tokens_used}, iterations={_snap.iterations_used}"
            )
        _budget_remaining = getattr(self, "_token_budget_remaining", None)
        if _budget_remaining is not None and _budget_remaining <= 0:
            self._log("warning", "Token budget exhausted for %s", self.agent_type.value)
            raise InferenceError(f"Token budget exhausted for {self.agent_type.value}")

        if self._adapter_manager is None:
            # No adapter: try to use the singleton if available
            self._adapter_manager = _lazy_get_adapter_manager()

        if self._adapter_manager is None:
            # Last resort: call local inference directly
            try:
                from vetinari.adapters.llama_cpp_adapter import (
                    LocalInferenceAdapter,  # noqa: VET022, VET130 — rare fallback path, not hot
                )

                _adapter = LocalInferenceAdapter()
                _sys = system_prompt or self.get_system_prompt()
                _model = model_id or self.default_model or "default"
                resp = _adapter.chat(_model, _sys, prompt)
                output = resp.get("output")
                if output is None:
                    raise InferenceError(f"Inference adapter returned no 'output' key for model {_model!r}")
                return output
            except Exception as e:  # Broad: last-resort adapter is optional; any failure mode is possible
                self._log("error", "LLM inference failed (no adapter_manager): %s", e)
                raise ModelUnavailableError(f"No inference adapter available: {e}") from e

        # Prepend model-size-aware base prompt framework to agent-specific system prompt
        _agent_system = system_prompt or self.get_system_prompt()
        _resolved_model_id = model_id or self.default_model
        _active_system_prompt = self._get_prompt_tier(model_id=_resolved_model_id) + "\n\n" + _agent_system
        _variant_id = "default"
        _evolver = _lazy_get_prompt_evolver() if _PROMPT_EVOLVER_ENABLED else None
        if _evolver is not None:
            try:
                evolved_prompt, _variant_id = _evolver.select_prompt(self.agent_type.value)
                if evolved_prompt and evolved_prompt != _active_system_prompt:
                    _active_system_prompt = evolved_prompt
            except Exception:  # Broad: prompt evolver is optional; any runtime failure must not block inference
                logger.warning("Failed to select evolved prompt for agent %s", self.agent_type.value, exc_info=True)

        # Enhance prompt with PromptAssembler: few-shot examples, failure
        # patterns, and task-type instructions (ADR: PromptAssembler wiring)
        _assembler = _lazy_get_prompt_assembler()
        if _assembler is not None:
            try:
                _task_type = getattr(self, "_current_task_type", None) or "general"
                _assembled = _assembler.build(
                    agent_type=self.agent_type.value,
                    task_type=_task_type,
                    task_description=prompt,
                    mode=getattr(self, "_current_mode", None),
                    context_budget=getattr(self, "_context_budget", 28000),
                )
                if _assembled.get("system"):
                    _active_system_prompt = _assembled["system"]
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "PromptAssembler built %d-char prompt for %s/%s (cache_hit=%s)",
                            _assembled.get("total_chars", 0),
                            self.agent_type.value,
                            _task_type,
                            _assembled.get("cache_hit", False),
                        )
            except Exception:  # Broad: assembler is optional; any failure must not block inference
                logger.warning("PromptAssembler unavailable for %s", self.agent_type.value, exc_info=True)

        # Inject prior memories into the task prompt if available
        _prior_memories = getattr(self, "_current_task_memories", None)
        if _prior_memories:
            _memory_lines = []
            for _mem in _prior_memories[:5]:
                if isinstance(_mem, dict):
                    _memory_lines.append(f"- {_mem.get('summary', str(_mem))}")
                else:
                    _memory_lines.append(f"- {_mem}")
            if _memory_lines:
                prompt = prompt + "\n\n## Relevant Prior Experience\n" + "\n".join(_memory_lines)

        # Apply task-specific inference params from external config
        _model_for_config = model_id or self.default_model
        _icfg = _lazy_get_inference_config()
        if _icfg is not None:
            try:
                _task_key = getattr(self, "_current_task_type", None) or self.agent_type.value.lower()
                _effective = _icfg.get_effective_params(_task_key, _model_for_config)
                if max_tokens == 4096:
                    max_tokens = _effective.get("max_tokens", max_tokens)
                if temperature == 0.3:
                    temperature = _effective.get("temperature", temperature)
            except (KeyError, ValueError):
                # Fallback to token_optimizer when config params missing
                _opt = _lazy_get_token_optimizer()
                if _opt is not None:
                    try:
                        _profile = _opt.get_task_profile(
                            getattr(self, "_current_task_type", None) or self.agent_type.value.lower()
                        )
                        _profile_max_tokens, _profile_temp, _ = _profile
                        if max_tokens == 4096:
                            max_tokens = _profile_max_tokens
                        if temperature == 0.3:
                            temperature = _profile_temp
                    except (KeyError, ValueError):
                        logger.warning(
                            "Failed to load token_optimizer task profile for %s",
                            self.agent_type.value,
                            exc_info=True,
                        )
        else:
            # Neither config manager available — try token_optimizer directly
            _opt = _lazy_get_token_optimizer()
            if _opt is not None:
                try:
                    _profile = _opt.get_task_profile(
                        getattr(self, "_current_task_type", None) or self.agent_type.value.lower()
                    )
                    _profile_max_tokens, _profile_temp, _ = _profile
                    if max_tokens == 4096:
                        max_tokens = _profile_max_tokens
                    if temperature == 0.3:
                        temperature = _profile_temp
                except (KeyError, ValueError):
                    logger.warning(
                        "Token optimizer profile not available for %s — using default token limits",
                        self.agent_type.value,
                    )

        # Thompson Sampling temperature override — when the bandit has learned
        # optimal temperatures for this agent+task combination, prefer its
        # selection over the config-based default.
        _thompson_pair = _lazy_get_thompson_strategy()
        if _thompson_pair is not None:
            try:
                _ts, _select_strategy_fn = _thompson_pair
                _agent_key = self.agent_type.value if hasattr(self.agent_type, "value") else str(self.agent_type)
                _mode_key = getattr(self, "mode", "default")
                _thompson_temp = _ts.select_strategy(_agent_key, _mode_key, "temperature")
                if isinstance(_thompson_temp, (int, float)):
                    temperature = float(_thompson_temp)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "[Thompson] Selected temperature %.2f for %s/%s",
                            temperature,
                            _agent_key,
                            _mode_key,
                        )
            except Exception:
                logger.warning(
                    "Thompson temperature selection failed for %s/%s — keeping config-based value",
                    _agent_key,
                    _mode_key,
                    exc_info=True,
                )

        if expect_json:
            prompt = prompt + "\n\nRespond ONLY with valid JSON. Do not include markdown code fences or explanation."

        # Build metadata with cache_control for prefix caching (ADR-0062)
        _request_metadata: dict[str, Any] = {
            "agent": self.agent_type.value,
            "cache_control": _EPHEMERAL_CACHE_CONTROL,
        }

        # Wire grammar-constrained JSON output for local adapters
        _response_format: str | None = None
        if expect_json:
            _request_metadata["response_format"] = "json"
            _response_format = "json"

        # Enforce context window budget: compress first, raise if still too large.
        # This prevents silent truncation of oversized prompts — truncation produces
        # partial, misleading outputs that are worse than an explicit failure.
        try:
            _n_ctx = getattr(self, "_context_length", None) or 8192
            _budget = check_prompt_budget(_active_system_prompt, prompt, max_tokens, _n_ctx)
            if not _budget["fits"]:
                # Attempt compression before giving up
                _preprocessor_cls = _get_local_preprocessor_cls()
                if _preprocessor_cls is not None:
                    _preprocessor = _preprocessor_cls()
                    _compressed = _preprocessor.compress(prompt)
                    _budget2 = check_prompt_budget(_active_system_prompt, _compressed, max_tokens, _n_ctx)
                    if _budget2["fits"]:
                        prompt = _compressed
                        logger.warning(
                            "Prompt for %s compressed %d→%d tokens to fit n_ctx=%d",
                            self.agent_type.value,
                            _budget["task_tokens"],
                            _budget2["task_tokens"],
                            _n_ctx,
                        )
                    else:
                        raise InferenceError(
                            f"Prompt for {self.agent_type.value} exceeds context window "
                            f"({_budget['total_tokens']} tokens > {_n_ctx} n_ctx) even after compression "
                            f"({_budget2['total_tokens']} tokens) — reduce prompt size"
                        )
                else:
                    # LocalPreprocessor unavailable — raise without compression
                    raise InferenceError(
                        f"Prompt for {self.agent_type.value} exceeds context window "
                        f"({_budget['total_tokens']} tokens > {_n_ctx} n_ctx) — reduce prompt size"
                    )
        except InferenceError:
            raise
        except Exception as exc:
            # Budget check itself raised an unexpected error — fail closed rather
            # than proceeding without context-window enforcement.  An unchecked
            # prompt may silently truncate, producing partial or misleading output.
            raise InferenceError(
                f"Prompt budget check failed for {self.agent_type.value} — cannot verify context window fit, "
                "aborting inference to prevent silent truncation"
            ) from exc

        request = InferenceRequest(
            model_id=model_id or self.default_model or "default",
            prompt=prompt,
            system_prompt=_active_system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=_response_format,
            metadata=_request_metadata,
        )

        # Semantic cache check — return cached response on hit.
        # Cache failure must never break inference; all lookups are guarded.
        _cache = _lazy_get_semantic_cache()
        _cache_task_key = getattr(self, "_current_task_type", None) or self.agent_type.value.lower()
        if _cache is not None:
            try:
                _cached_response = _cache.get(prompt, task_type=_cache_task_key)
                if _cached_response is not None:
                    logger.debug("Semantic cache hit for task %s", _cache_task_key)
                    return _cached_response
            except Exception:
                logger.warning(
                    "Semantic cache lookup failed for task %s — proceeding with direct inference", _cache_task_key
                )

        # ── Batch processing path for non-urgent inference ────────────
        _batch_mode = getattr(self, "_batch_mode", False)
        if _batch_mode:
            try:
                _bp = _lazy_get_batch_processor()
                if _bp is not None and _bp.enabled:
                    _provider = request.metadata.get("provider", "anthropic")
                    _future = _bp.enqueue(request, provider=_provider)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Request enqueued for batch processing (provider=%s)", _provider)
                    _batch_result = _future.result(timeout=BATCH_RESULT_TIMEOUT)
                    if _batch_result.success:
                        return _batch_result.response.output if _batch_result.response else ""
                    logger.warning("Batch result failed: %s — falling back to direct", _batch_result.error)
            except Exception as _batch_err:  # Broad: batch processor is optional; any failure mode is possible
                logger.warning("Batch processing unavailable, falling back to direct: %s", _batch_err)

        # Guard observability metadata construction with tracing check (Item 1.26)
        _obs_metadata = {
            "agent_type": self.agent_type.value,
            "model_name": request.model_id,
        }
        with _ObservabilitySpan(
            f"agent.{self.__class__.__name__}.infer",
            metadata=_obs_metadata,
        ) as _obs_span:
            try:
                _call_start = time.monotonic()
                response = self._adapter_manager.infer(request, use_cascade=use_cascade)
                if response.status == INFERENCE_STATUS_OK:
                    result = response.output
                    if expect_json:
                        # Strip any accidental markdown fences
                        result = result.strip()
                        if result.startswith("```"):
                            lines = result.split("\n")
                            result = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

                    # ── C1: Record success via cached registry ────────
                    if _cb_registry is not None:
                        try:
                            _cb_registry.get(self.agent_type.value).record_success()
                        except Exception:  # Broad: circuit breaker is optional
                            logger.warning(
                                "Circuit breaker success recording failed for %s",
                                self.agent_type.value,
                                exc_info=True,
                            )

                    # ── C5: Track token usage (no split()) ────────────
                    _estimated_tokens = response.tokens_used or len(result) // 4
                    if hasattr(self, "_token_budget_remaining") and self._token_budget_remaining is not None:
                        self._token_budget_remaining -= _estimated_tokens
                    # Record on BudgetTracker if present (ADR-0075)
                    _bt = getattr(self, "_budget_tracker", None)
                    if _bt is not None:
                        _bt.record_usage(_estimated_tokens)

                    # ── D6.6: Record token counts on span ─────────────
                    _obs_span.set_attribute("prompt_tokens", len(prompt) // 4)
                    _obs_span.set_attribute("completion_tokens", _estimated_tokens)

                    # Store successful result in semantic cache for future hits.
                    # Failures are non-fatal — inference already succeeded.
                    if _cache is not None and result:
                        try:
                            _cache.put(prompt, result)
                        except Exception:
                            logger.warning(
                                "Semantic cache store failed for task %s — result not cached", _cache_task_key
                            )

                    # Store actual model_id and temperature from inference response
                    # for downstream quality scoring and Thompson temperature
                    # feedback (not the requested/default values).
                    self._last_inference_model_id = response.model_id
                    self._last_inference_temperature = temperature

                    # Store inference confidence from logprob variance (if available)
                    # for downstream quality scoring — high variance = low confidence
                    self._last_inference_confidence = response.metadata.get("inference_confidence")
                    self._last_inference_metadata = response.metadata

                    # Store actual telemetry for training data attribution (ADR-0091).
                    # These must be set from the real inference call, not defaults,
                    # so complete_task() can pass truthful values to TrainingDataCollector.
                    # _last_trace_id reads from _current_trace_id which the pipeline engine
                    # sets on the agent instance before execution; falls back to "" if absent.
                    self._last_tokens_used = _estimated_tokens
                    self._last_latency_ms = (time.monotonic() - _call_start) * 1000
                    self._last_variant_id = _variant_id
                    self._last_trace_id = str(getattr(self, "_current_trace_id", "") or "")

                    return result
                self._log("warning", "Inference failed: %s", response.error)
                if _cb_registry is not None:
                    try:
                        _cb_registry.get(self.agent_type.value).record_failure()
                    except Exception:  # Broad: circuit breaker is optional
                        logger.warning(
                            "Circuit breaker failure recording failed for %s",
                            self.agent_type.value,
                            exc_info=True,
                        )
                raise InferenceError(f"Inference failed for {self.agent_type.value}: {response.error}")
            except Exception as e:
                self._log("error", "Inference exception: %s", e)
                if _cb_registry is not None:
                    try:
                        _cb_registry.get(self.agent_type.value).record_failure()
                    except Exception:  # Broad: circuit breaker is optional
                        logger.warning(
                            "Circuit breaker failure recording failed for %s after inference exception",
                            self.agent_type.value,
                            exc_info=True,
                        )
                raise InferenceError(f"Inference exception for {self.agent_type.value}: {e}") from e

    def _infer_with_fallback(
        self,
        prompt: str,
        fallback_fn: Any | None = None,
        required_keys: list[str] | None = None,
    ) -> Any | None:
        """Infer from LLM with optional fallback and key validation.

        Args:
            prompt: The prompt to send to the LLM.
            fallback_fn: Optional callable to run if LLM fails.
            required_keys: Optional list of keys the JSON response must contain.

        Returns:
            Parsed response dict, or fallback result, or None.
        """
        try:
            response = self._infer_json(prompt)
            if response and required_keys:
                if all(k in response for k in required_keys):
                    return response
            elif response:
                return response
        except (InferenceError, ModelUnavailableError) as e:
            logger.warning("[%s] LLM inference failed: %s", self.agent_type, e)

        if fallback_fn:
            return fallback_fn()
        return None
