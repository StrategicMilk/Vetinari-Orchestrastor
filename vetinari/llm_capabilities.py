"""LLM-powered analysis capabilities beyond basic classification.

New capabilities that leverage LLM calls for tasks tedious for humans
and easy for language models. Each function is designed to be called
selectively (not on every request) to keep token costs low.

Capabilities:
  - Error diagnosis: failing function + recent changes -> LLM diagnosis
  - Changelog generation: commit history -> structured changelog
  - Code complexity explanation: generate "why" comments for complex functions
"""

from __future__ import annotations

import logging
from typing import Any

from vetinari.config.inference_config import get_inference_config

logger = logging.getLogger(__name__)


def diagnose_error(
    function_name: str,
    error_message: str,
    source_code: str = "",
    recent_changes: str = "",
) -> dict[str, Any] | None:
    """Diagnose why a function is failing using LLM analysis.

    Combines the failing function's source, the error message, and recent
    git changes to produce a targeted diagnosis. Triggered selectively
    for opaque errors or tests that passed on the last commit.

    Args:
        function_name: Name of the failing function.
        error_message: The error or exception message.
        source_code: Source code of the failing function (optional, truncated).
        recent_changes: Recent git diff or blame output for context.

    Returns:
        Dict with ``diagnosis``, ``likely_cause``, ``suggested_fix`` keys,
        or None if LLM is unavailable.
    """
    from vetinari.llm_helpers import quick_llm_call

    prompt = f"Function: {function_name}\nError: {error_message}\n"
    if source_code:
        prompt += f"\nSource code:\n```\n{source_code[:2000]}\n```\n"
    if recent_changes:
        prompt += f"\nRecent changes:\n{recent_changes[:1000]}\n"
    prompt += (
        "\nDiagnose the error. Provide:\n"
        "1. DIAGNOSIS: one-sentence explanation\n"
        "2. LIKELY_CAUSE: the specific line or pattern causing the failure\n"
        "3. SUGGESTED_FIX: concrete fix (code snippet if applicable)"
    )

    _profile = get_inference_config().get_profile("debugging")
    result = quick_llm_call(
        prompt=prompt,
        system_prompt=(
            "You are a debugging expert. Analyze the error and provide a precise diagnosis. "
            "Focus on the most likely root cause based on the error message and code context."
        ),
        max_tokens=_profile.max_tokens,
        temperature=_profile.temperature,
    )
    if not result:
        return None

    parsed: dict[str, Any] = {"raw": result}
    for line in result.split("\n"):
        line = line.strip()
        if line.upper().startswith("DIAGNOSIS:"):
            parsed["diagnosis"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("LIKELY_CAUSE:"):
            parsed["likely_cause"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("SUGGESTED_FIX:"):
            parsed["suggested_fix"] = line.split(":", 1)[1].strip()

    return parsed if "diagnosis" in parsed else {"diagnosis": result, "raw": result}


def generate_changelog(
    commits: list[dict[str, str]],
    version: str = "",
) -> str | None:
    """Generate a structured changelog from commit history.

    Categorizes commits into Added, Changed, Fixed, Removed, Security
    sections following the Keep a Changelog format.

    Args:
        commits: List of commit dicts with ``message`` and optional ``hash``, ``author`` keys.
        version: Version string for the changelog header (e.g. "1.2.0").

    Returns:
        Formatted changelog markdown string, or None if LLM is unavailable.
    """
    from vetinari.llm_helpers import quick_llm_call

    commit_text = "\n".join(f"- {c.get('hash', '')[:8]} {c['message']}" for c in commits[:50])
    prompt = (
        f"Commits:\n{commit_text}\n\n"
        "Generate a changelog in Keep a Changelog format with sections: "
        "Added, Changed, Fixed, Removed, Security. "
        "Group related commits. Write concise, user-facing descriptions. "
        "Skip internal/tooling commits."
    )
    if version:
        prompt += f"\nVersion: {version}"

    _profile = get_inference_config().get_profile("documentation")
    return quick_llm_call(
        prompt=prompt,
        system_prompt="You are a technical writer generating a changelog for a software release.",
        max_tokens=_profile.max_tokens,
        temperature=_profile.temperature,
    )


def explain_complex_function(
    function_name: str,
    source_code: str,
    cyclomatic_complexity: int = 0,
) -> str | None:
    """Generate a "why" comment explaining a complex function's purpose and approach.

    Triggered for functions with cyclomatic complexity > 5 or containing
    magic numbers. The explanation focuses on WHY the approach was chosen,
    not WHAT the code does (the code shows what).

    Args:
        function_name: Name of the function to explain.
        source_code: The function's source code.
        cyclomatic_complexity: Measured cyclomatic complexity score.

    Returns:
        A "why" comment string suitable for placing above the function,
        or None if LLM is unavailable.
    """
    from vetinari.llm_helpers import quick_llm_call

    complexity_note = ""
    if cyclomatic_complexity > 0:
        complexity_note = f"\nCyclomatic complexity: {cyclomatic_complexity}"

    prompt = (
        f"Function: {function_name}{complexity_note}\n"
        f"```python\n{source_code[:3000]}\n```\n\n"
        "Write a concise 2-3 sentence comment explaining WHY this function uses "
        "this particular approach. Focus on design decisions, trade-offs, and "
        "non-obvious constraints. Do NOT describe what the code does — the code "
        "shows that. Start with '# Why:'"
    )

    _profile = get_inference_config().get_profile("compression")
    return quick_llm_call(
        prompt=prompt,
        system_prompt="You explain code design decisions concisely. Focus on WHY, not WHAT.",
        max_tokens=_profile.max_tokens,
        temperature=_profile.temperature,
    )
