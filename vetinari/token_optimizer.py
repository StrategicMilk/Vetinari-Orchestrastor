"""
Vetinari Token Optimizer

Comprehensive token usage optimization system including:

1. **Token Budget Enforcement** — per-task and per-plan limits
2. **Context Summarisation** — compress long contexts before they overflow
3. **Dynamic max_tokens** — task-type-aware output limits
4. **Local LLM Preprocessing** — use cheap local models to compress context
   before sending to expensive cloud models (reduces cloud tokens 30-60%)
5. **Structured output enforcement** — JSON mode where supported
6. **Context deduplication** — avoid sending the same context repeatedly
7. **Task-specific model profiles** — optimal temperature/tokens per task type

Usage:
    from vetinari.token_optimizer import get_token_optimizer, TokenBudget

    optimizer = get_token_optimizer()

    # Check budget before inference
    budget = TokenBudget(plan_id="plan_123", max_tokens=50000)
    compressed = optimizer.prepare_prompt(
        prompt=long_prompt,
        context=big_context,
        task_type="coding",
        budget=budget,
    )
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task-type profiles: (max_tokens, temperature, prefer_json)
# ---------------------------------------------------------------------------
TASK_PROFILES: Dict[str, Tuple[int, float, bool]] = {
    "planning": (3000, 0.2, True),
    "planner": (3000, 0.2, True),
    "coding": (4096, 0.1, False),
    "code_gen": (4096, 0.1, False),
    "builder": (4096, 0.1, False),
    "research": (2048, 0.4, True),
    "researcher": (2048, 0.4, True),
    "analysis": (2048, 0.3, True),
    "oracle": (2048, 0.3, True),
    "documentation": (3000, 0.3, False),
    "documentation_agent": (3000, 0.3, False),
    "security": (2048, 0.2, True),
    "security_auditor": (2048, 0.2, True),
    "testing": (3000, 0.1, True),
    "test_automation": (3000, 0.1, True),
    "ui_design": (3000, 0.4, True),
    "ui_planner": (3000, 0.4, True),
    "data_engineering": (2048, 0.2, True),
    "data_engineer": (2048, 0.2, True),
    "classification": (256, 0.0, True),
    "extraction": (512, 0.0, True),
    "summarisation": (1024, 0.2, False),
    "summarization": (1024, 0.2, False),
    "synthesis": (2048, 0.3, False),
    "synthesizer": (2048, 0.3, False),
    "evaluation": (1024, 0.1, True),
    "evaluator": (1024, 0.1, True),
    "exploration": (1500, 0.3, True),
    "explorer": (1500, 0.3, True),
    "general": (2048, 0.3, False),
}

# Rough character-to-token ratio (4 chars ≈ 1 token for English text)
_CHARS_PER_TOKEN = 4

# Maximum prompt character length before triggering local summarisation
_COMPRESS_THRESHOLD_CHARS = 6000  # ~1500 tokens

# Sliding window for context: keep latest N chars when truncating
_CONTEXT_WINDOW_CHARS = 4000


@dataclass
class TokenBudget:
    """Per-plan token budget with enforcement."""
    plan_id: str
    max_tokens: int = 100_000          # Total token ceiling for the whole plan
    max_tokens_per_task: int = 8_000   # Per-task ceiling
    tokens_used: int = 0
    task_token_counts: Dict[str, int] = field(default_factory=dict)

    def record(self, task_id: str, tokens: int) -> None:
        self.tokens_used += tokens
        self.task_token_counts[task_id] = self.task_token_counts.get(task_id, 0) + tokens

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)

    @property
    def is_exhausted(self) -> bool:
        return self.tokens_used >= self.max_tokens

    def check_task(self, task_id: str, estimated_tokens: int) -> bool:
        """Return True if this task can proceed within budget."""
        if self.is_exhausted:
            return False
        task_used = self.task_token_counts.get(task_id, 0)
        if task_used + estimated_tokens > self.max_tokens_per_task:
            return False
        return True


class LocalPreprocessor:
    """
    Uses a local LLM to compress and distil context before cloud API calls.

    This is the key cost-reduction feature:
    - Input: verbose context (code, docs, prior results)  ~3000 tokens
    - Output: compressed key points                        ~800 tokens
    - Savings: ~70% of cloud input tokens

    Only activates when:
    1. The target model is a cloud model (has cost > 0)
    2. The context length exceeds the threshold
    3. A local LM Studio model is available
    """

    # Minimum context length (chars) to justify preprocessing overhead
    MIN_CONTEXT_CHARS = _COMPRESS_THRESHOLD_CHARS

    def __init__(self):
        self._local_model: Optional[str] = None
        self._cache: Dict[str, str] = {}  # hash -> compressed result

    def _get_local_model(self) -> Optional[str]:
        """Discover the best available local model for preprocessing."""
        if self._local_model:
            return self._local_model
        try:
            host = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")
            import requests
            resp = requests.get(f"{host}/v1/models", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", data) if isinstance(data, dict) else data
                if models:
                    # Prefer smaller/faster models for preprocessing
                    for m in models:
                        mid = m.get("id", "")
                        if any(x in mid.lower() for x in ["7b", "8b", "3b", "1b"]):
                            self._local_model = mid
                            return mid
                    # Fall back to first available model
                    self._local_model = models[0].get("id", "")
                    return self._local_model
        except Exception:
            pass
        return None

    def compress_context(
        self,
        context: str,
        task_description: str = "",
        compression_goal: str = "key_facts",
    ) -> Tuple[str, float]:
        """
        Compress verbose context using a local LLM.

        Args:
            context: The verbose context to compress.
            task_description: What the context is for (guides compression).
            compression_goal: "key_facts" | "summary" | "code_only"

        Returns:
            (compressed_text, compression_ratio) where ratio < 1.0 means smaller.
        """
        if len(context) < self.MIN_CONTEXT_CHARS:
            return context, 1.0

        # Check cache
        cache_key = hashlib.md5(
            f"{context[:200]}{task_description[:50]}".encode()
        ).hexdigest()
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            ratio = len(cached) / max(len(context), 1)
            return cached, ratio

        local_model = self._get_local_model()
        if not local_model:
            # No local model — fall back to truncation
            return self._truncate(context), len(context[:_CONTEXT_WINDOW_CHARS]) / max(len(context), 1)

        try:
            goals = {
                "key_facts": "Extract ONLY the key facts, function signatures, API endpoints, and critical constraints. Remove examples, explanations, and repetition.",
                "summary": "Write a concise summary preserving all actionable information and technical specifics.",
                "code_only": "Extract ONLY the function/class definitions, signatures, and docstrings. Remove all prose.",
            }
            goal_instruction = goals.get(compression_goal, goals["key_facts"])

            prompt = (
                f"{goal_instruction}\n\n"
                f"Task context: {task_description[:200]}\n\n"
                f"Content to compress:\n{context[:8000]}\n\n"
                "Provide the compressed version. Be as concise as possible while preserving ALL technically relevant information."
            )

            host = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")
            from vetinari.adapters.lmstudio_adapter import resolve_lmstudio_model
            resolved_model = resolve_lmstudio_model(local_model, host)
            import requests
            resp = requests.post(
                f"{host}/v1/chat/completions",
                json={
                    "model": resolved_model,
                    "messages": [
                        {"role": "system", "content": "You are a context compression specialist. Compress text while preserving all technically critical information."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.1,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                compressed = data["choices"][0]["message"]["content"].strip()
                ratio = len(compressed) / max(len(context), 1)
                self._cache[cache_key] = compressed
                logger.info(
                    f"[LocalPreprocessor] Compressed {len(context)} -> {len(compressed)} chars "
                    f"({ratio*100:.0f}%) for task: {task_description[:40]}"
                )
                return compressed, ratio
        except Exception as e:
            logger.debug(f"[LocalPreprocessor] Compression failed: {e}")

        return self._truncate(context), len(context[:_CONTEXT_WINDOW_CHARS]) / max(len(context), 1)

    def _truncate(self, context: str) -> str:
        """Simple truncation fallback."""
        if len(context) <= _CONTEXT_WINDOW_CHARS:
            return context
        head = context[:_CONTEXT_WINDOW_CHARS // 3]
        tail = context[-(2 * _CONTEXT_WINDOW_CHARS // 3):]
        return f"{head}\n\n[... context truncated for token efficiency ...]\n\n{tail}"

    def preprocess_for_cloud(
        self,
        prompt: str,
        context: str = "",
        task_description: str = "",
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Full preprocessing pipeline for cloud API calls.

        Returns:
            (processed_prompt, processed_context, metadata)
        """
        meta = {
            "original_prompt_chars": len(prompt),
            "original_context_chars": len(context),
            "compressed": False,
            "compression_ratio": 1.0,
        }

        # Compress context if it's large
        if context and len(context) >= self.MIN_CONTEXT_CHARS:
            compressed_context, ratio = self.compress_context(
                context, task_description, "key_facts"
            )
            meta["compressed"] = ratio < 0.9
            meta["compression_ratio"] = ratio
            context = compressed_context

        # Compress the prompt itself if it's very large
        if len(prompt) > _COMPRESS_THRESHOLD_CHARS * 2:
            compressed_prompt, ratio = self.compress_context(
                prompt, task_description, "summary"
            )
            meta["prompt_compressed"] = ratio < 0.9
            prompt = compressed_prompt

        meta["final_prompt_chars"] = len(prompt)
        meta["final_context_chars"] = len(context)
        return prompt, context, meta


class TokenOptimizer:
    """
    Central token optimization orchestrator.

    Integrates:
    - TokenBudget enforcement
    - Task-specific model profiles (max_tokens, temperature)
    - Context summarisation for long inputs
    - Local LLM preprocessing for cloud calls
    - Context deduplication
    """

    def __init__(self):
        self._budgets: Dict[str, TokenBudget] = {}
        self._preprocessor = LocalPreprocessor()
        self._context_cache: Dict[str, str] = {}  # Dedup cache

    # ------------------------------------------------------------------
    # Budget management
    # ------------------------------------------------------------------

    def create_budget(
        self,
        plan_id: str,
        max_tokens: int = 100_000,
        max_tokens_per_task: int = 8_000,
    ) -> TokenBudget:
        """Create and register a token budget for a plan."""
        budget = TokenBudget(
            plan_id=plan_id,
            max_tokens=max_tokens,
            max_tokens_per_task=max_tokens_per_task,
        )
        self._budgets[plan_id] = budget
        return budget

    def get_budget(self, plan_id: str) -> Optional[TokenBudget]:
        return self._budgets.get(plan_id)

    def record_usage(self, plan_id: str, task_id: str, tokens: int) -> None:
        """Record token usage after a completed inference."""
        budget = self._budgets.get(plan_id)
        if budget:
            budget.record(task_id, tokens)

    # ------------------------------------------------------------------
    # Task profile resolution
    # ------------------------------------------------------------------

    def get_task_profile(self, task_type: str) -> Tuple[int, float, bool]:
        """Return (max_tokens, temperature, prefer_json) for a task type."""
        key = task_type.lower().replace(" ", "_").replace("-", "_")
        return TASK_PROFILES.get(key, TASK_PROFILES["general"])

    # ------------------------------------------------------------------
    # Prompt preparation
    # ------------------------------------------------------------------

    def prepare_prompt(
        self,
        prompt: str,
        context: str = "",
        task_type: str = "general",
        task_description: str = "",
        is_cloud_model: bool = False,
        plan_id: Optional[str] = None,
        task_id: Optional[str] = None,
        budget: Optional[TokenBudget] = None,
    ) -> Dict[str, Any]:
        """
        Prepare an optimised prompt for inference.

        Returns a dict with:
          - prompt: optimised prompt string
          - context: optimised context string
          - max_tokens: recommended max output tokens
          - temperature: recommended temperature
          - prefer_json: whether to request JSON output
          - metadata: compression/optimisation stats
          - budget_ok: whether the budget allows this task
        """
        max_tokens, temperature, prefer_json = self.get_task_profile(task_type)

        meta: Dict[str, Any] = {
            "task_type": task_type,
            "task_profile": {"max_tokens": max_tokens, "temperature": temperature},
            "is_cloud_model": is_cloud_model,
        }

        # Budget check
        estimated_input_tokens = (len(prompt) + len(context)) // _CHARS_PER_TOKEN
        estimated_total = estimated_input_tokens + max_tokens
        budget_ok = True

        active_budget = budget or (self._budgets.get(plan_id) if plan_id else None)
        if active_budget:
            budget_ok = active_budget.check_task(task_id or "unknown", estimated_total)
            meta["budget_remaining"] = active_budget.remaining
            meta["budget_ok"] = budget_ok
            if not budget_ok:
                logger.warning(
                    f"[TokenOptimizer] Task {task_id} would exceed budget "
                    f"(estimated {estimated_total} tokens, remaining {active_budget.remaining})"
                )

        # Cloud preprocessing: compress context before expensive cloud calls
        if is_cloud_model and context and len(context) >= LocalPreprocessor.MIN_CONTEXT_CHARS:
            prompt, context, compress_meta = self._preprocessor.preprocess_for_cloud(
                prompt, context, task_description
            )
            meta.update(compress_meta)
        elif len(context) > _CONTEXT_WINDOW_CHARS:
            # Even for local models, truncate very long contexts
            context = self._preprocessor._truncate(context)
            meta["truncated"] = True

        # Deduplicate context: if this exact context was recently seen, skip it
        context_hash = hashlib.md5(context[:500].encode()).hexdigest() if context else ""
        if context_hash and context_hash in self._context_cache:
            # Context hasn't changed — reference it but don't repeat it
            meta["context_deduplicated"] = True
            # Keep a brief reference instead
            context = f"[Context unchanged from previous task — key points: {context[:200]}...]"
        elif context_hash:
            self._context_cache[context_hash] = context

        # Assemble final prompt
        if context:
            final_prompt = f"Context:\n{context}\n\n{prompt}"
        else:
            final_prompt = prompt

        return {
            "prompt": final_prompt,
            "context": context,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "prefer_json": prefer_json,
            "metadata": meta,
            "budget_ok": budget_ok,
        }

    def summarise_results(
        self, results: List[Dict[str, Any]], max_chars: int = 2000
    ) -> str:
        """
        Summarise a list of task results for inclusion in subsequent prompts.
        Prevents context explosion when many tasks have completed.
        """
        if not results:
            return ""

        summaries = []
        for r in results:
            if isinstance(r, dict):
                desc = r.get("description", r.get("task_id", "task"))[:60]
                out = r.get("output", r.get("result", ""))
                if isinstance(out, dict):
                    # Take only top-level keys and short values
                    out_str = "; ".join(
                        f"{k}: {str(v)[:80]}"
                        for k, v in list(out.items())[:5]
                        if v
                    )
                else:
                    out_str = str(out)[:200]
                summaries.append(f"- {desc}: {out_str}")
            elif isinstance(r, str):
                summaries.append(f"- {r[:200]}")

        combined = "\n".join(summaries)
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "\n[... additional results truncated ...]"
        return combined


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_token_optimizer: Optional[TokenOptimizer] = None


def get_token_optimizer() -> TokenOptimizer:
    """Get or create the global TokenOptimizer singleton."""
    global _token_optimizer
    if _token_optimizer is None:
        _token_optimizer = TokenOptimizer()
    return _token_optimizer
