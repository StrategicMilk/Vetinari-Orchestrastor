"""Prompt optimization wiring — cache lookup and token budgeting for agent prompts.

Connects the PromptCache and TokenOptimizer into the agent prompt assembly
path so that prompts are cached for reuse and optimized for token limits
before being sent to the LLM backend.

Pipeline role: sits between prompt construction and inference dispatch.
    Context Gathering -> **Prompt Cache** -> **Token Optimization** -> Inference
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def cached_prompt_lookup(prompt: str) -> tuple[bool, str, int]:
    """Check the prompt cache before assembling a new prompt.

    If the prompt has been seen recently, returns the cached version and
    the estimated token savings. On a miss, the prompt is stored for
    future lookups.

    Args:
        prompt: The assembled prompt string to look up or cache.

    Returns:
        Tuple of (cache_hit: bool, prompt: str, savings_tokens: int).
        On a hit, savings_tokens > 0. On a miss, the original prompt is
        returned with savings_tokens=0.
    """
    try:
        from vetinari.optimization.prompt_cache import get_prompt_cache, hash_prompt

        cache = get_prompt_cache()
        prompt_hash = hash_prompt(prompt)
        result = cache.get_or_cache(prompt_hash, prompt)

        if result.hit:
            logger.debug(
                "Prompt cache HIT — saved ~%d tokens",
                result.savings_tokens,
            )
        return (result.hit, result.prompt, result.savings_tokens)
    except Exception:
        logger.warning("Prompt cache unavailable — proceeding without cache, prompt assembly may be slower")
        return (False, prompt, 0)


def invalidate_prompt_cache(prompt: str) -> None:
    """Invalidate a cached prompt when context changes.

    Call this when the context that built the prompt has changed (e.g.,
    memory updated, new tool results) to force re-assembly on next use.

    Args:
        prompt: The original prompt string whose cache entry should be removed.
    """
    try:
        from vetinari.optimization.prompt_cache import get_prompt_cache, hash_prompt

        cache = get_prompt_cache()
        cache.invalidate(hash_prompt(prompt))
        logger.debug("Prompt cache entry invalidated")
    except Exception:
        logger.warning("Prompt cache unavailable — invalidation skipped, stale cache entry may remain")


def get_prompt_cache_stats() -> dict[str, Any]:
    """Return prompt cache statistics for dashboard display.

    Returns:
        Dictionary with hit_rate, total_hits, total_misses, cache_size.
        Returns empty dict if cache is unavailable.
    """
    try:
        from vetinari.optimization.prompt_cache import get_prompt_cache

        return get_prompt_cache().get_stats()
    except Exception:
        logger.warning("Prompt cache unavailable — get_prompt_cache_stats returning empty dict")
        return {}


def optimize_prompt_for_budget(
    prompt: str,
    context: str = "",
    task_type: str = "general",
    task_description: str = "",
    plan_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Prepare and optimize a prompt within token budget constraints.

    Calls TokenOptimizer.prepare_prompt() to compress verbose sections,
    enforce token limits, and apply task-type-specific optimizations.
    This is the final step of prompt assembly before inference.

    Args:
        prompt: The raw prompt to optimize.
        context: Additional context to prepend (may be compressed).
        task_type: Task type for profile lookup (e.g., "coding", "planning").
        task_description: Human-readable description for logging.
        plan_id: Optional plan ID for per-plan budget tracking.
        task_id: Optional task ID for per-task budget tracking.

    Returns:
        Dictionary from TokenOptimizer.prepare_prompt() containing at minimum:
        - ``"prompt"`` (str): The optimized prompt
        - ``"estimated_tokens"`` (int): Estimated token count
        - ``"compressed"`` (bool): Whether compression was applied
        Returns a fallback dict with the original prompt if the optimizer
        is unavailable.
    """
    try:
        from vetinari.token_optimizer import get_token_optimizer

        optimizer = get_token_optimizer()
        result = optimizer.prepare_prompt(
            prompt=prompt,
            context=context,
            task_type=task_type,
            task_description=task_description,
            plan_id=plan_id,
            task_id=task_id,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Prompt optimized for task_type=%s — estimated %d tokens, compressed=%s",
                task_type,
                result.get("estimated_tokens", 0),
                result.get("compressed", False),
            )
        return result
    except Exception:
        logger.warning(
            "Token optimizer unavailable for %s — returning prompt as-is without budget optimization", task_type
        )
        return {
            "prompt": prompt,
            "context": context,
            "estimated_tokens": len(prompt) // 4,
            "compressed": False,
        }


def wire_prompt_optimization() -> None:
    """Verify prompt cache and token optimizer are importable.

    Call once during application startup so import errors are surfaced
    early rather than on the first inference call.  Failures are logged
    as warnings — the system degrades gracefully without these optimizations.
    """
    cache_ok = False
    optimizer_ok = False

    try:
        from vetinari.optimization.prompt_cache import get_prompt_cache

        get_prompt_cache()
        cache_ok = True
    except Exception:
        logger.warning("Prompt cache not available — prompts will not be cached")

    try:
        from vetinari.token_optimizer import get_token_optimizer

        get_token_optimizer()
        optimizer_ok = True
    except Exception:
        logger.warning("Token optimizer not available — prompts will not be optimized")

    logger.info(
        "Prompt optimization wiring: cache=%s, optimizer=%s",
        "OK" if cache_ok else "UNAVAILABLE",
        "OK" if optimizer_ok else "UNAVAILABLE",
    )
