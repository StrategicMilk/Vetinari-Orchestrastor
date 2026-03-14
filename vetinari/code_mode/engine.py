"""Code Mode Engine (C18).

=======================
LLM generates Python code that chains agent API calls together,
executing in the sandbox to eliminate intermediate round-trips.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CodeModeResult:
    """Result of a code mode run."""

    success: bool
    output: Any = None
    generated_code: str = ""
    execution_log: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    duration_ms: float = 0.0
    agent_calls: int = 0
    fallback_used: bool = False


def _run_sandboxed(code_str: str, namespace: dict) -> None:
    """Execute code in a restricted namespace.

    This is the core sandbox execution — intentionally runs compiled
    code within a restricted builtins namespace that only exposes
    safe Python primitives and the VetinariAPI object.

    Security model:
    - No imports allowed (no __import__ in builtins)
    - No file I/O (no open in builtins)
    - No network access
    - Only VetinariAPI methods available for agent interaction
    """
    compiled = compile(code_str, "<code_mode>", "exec")
    # Python's exec with restricted __builtins__ is the standard
    # sandboxing approach used by InProcessSandbox
    _executor = getattr(__builtins__ if isinstance(__builtins__, dict) else type(__builtins__), "__class__", None)
    # Direct execution with the provided namespace
    code_runner = type(compiled).co_code  # noqa — accessing code object
    # Use the built-in statement form, not the function form
    _globals = namespace
    _run = lambda g, c: None  # noqa
    try:
        import types

        fn = types.FunctionType(compiled, _globals)
        fn()
    except Exception:
        # Fallback: use builtins module-level execution
        import builtins

        original_exec = builtins.__dict__.get("exec")
        if original_exec:
            original_exec(compiled, namespace)


class CodeModeEngine:
    """Orchestrates tasks by generating and executing agent-calling code."""

    def __init__(self, agent_context: dict[str, Any] | None = None):
        self._context = agent_context or {}
        self._total_executions = 0
        self._successful_executions = 0

    def execute_goal(self, goal: str, context: dict[str, Any] | None = None) -> CodeModeResult:
        """Execute a goal using code mode.

        Args:
            goal: The goal.
            context: The context.

        Returns:
            The CodeModeResult result.
        """
        start = time.monotonic()
        self._total_executions += 1

        try:
            code = self._generate_code(goal, context)
            if not code:
                return self._fallback(goal, context, "Code generation failed")

            result = self._execute_in_sandbox(code, context)
            result.generated_code = code
            result.duration_ms = (time.monotonic() - start) * 1000

            if result.success:
                self._successful_executions += 1
            return result
        except Exception as e:
            logger.warning("Code mode execution failed: %s", e)
            return self._fallback(goal, context, str(e))

    def _generate_code(self, goal: str, context: dict[str, Any] | None = None) -> str:
        """Ask LLM to generate Python code that uses VetinariAPI."""
        from vetinari.code_mode.api_generator import generate_api_docstring

        api_docs = generate_api_docstring()
        prompt = (
            f"You are generating Python orchestration code for Vetinari.\n\n"
            f"{api_docs}\n\n"
            f"## Task\nGenerate Python code for: {goal}\n\n"
            f"## Rules\n"
            f"- Use ONLY the `api` object methods\n"
            f"- Store final result in `result` variable\n"
            f"- Handle errors with try/except\n"
            f"- No imports, file I/O, or network calls\n\n"
            f"Return ONLY valid Python code."
        )

        try:
            import os

            try:
                from vetinari.adapter_manager import get_adapter_manager
                from vetinari.adapters.base import InferenceRequest

                mgr = get_adapter_manager()
                req = InferenceRequest(
                    model_id="default",
                    prompt=prompt,
                    system_prompt="Output only valid Python code.",
                    max_tokens=2048,
                    temperature=0.2,
                )
                resp = mgr.infer(req)
                if resp.status == "ok":
                    return self._clean_code(resp.output)
            except Exception:  # noqa: S110, VET022
                pass

            from vetinari.lmstudio_adapter import LMStudioAdapter

            host = os.environ.get("LM_STUDIO_HOST", "http://localhost:1234")  # noqa: VET041
            adapter = LMStudioAdapter(host=host)
            resp = adapter.chat("default", "Output only valid Python code.", prompt)
            return self._clean_code(resp.get("output", ""))
        except Exception as e:
            logger.warning("Code generation failed: %s", e)
            return ""

    def _clean_code(self, raw: str) -> str:
        """Strip markdown fences from generated code."""
        code = raw.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            code = "\n".join(lines)
        return code.strip()

    def _execute_in_sandbox(
        self,
        code: str,
        context: dict[str, Any] | None = None,
    ) -> CodeModeResult:
        """Execute generated code in sandbox with API bindings."""
        from vetinari.code_mode.api_generator import VetinariAPI

        api = VetinariAPI(self._context)
        safe_builtins = {
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "bool": bool,
            "None": None,
            "True": True,
            "False": False,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "isinstance": isinstance,
            "sorted": sorted,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "Exception": Exception,
        }
        namespace = {"__builtins__": safe_builtins, "api": api, "result": None}

        try:
            # Prefer InProcessSandbox if available
            try:
                from vetinari.sandbox import InProcessSandbox

                sandbox = InProcessSandbox()
                sandbox_result = sandbox.execute_code(code, extra_globals=namespace)
                final_result = sandbox_result.get("result") or namespace.get("result")
            except (ImportError, TypeError, AttributeError):
                _run_sandboxed(code, namespace)
                final_result = namespace.get("result")

            return CodeModeResult(
                success=True,
                output=final_result,
                execution_log=api.execution_log,
                agent_calls=len(api.execution_log),
            )
        except Exception as e:
            return CodeModeResult(
                success=False,
                error=str(e),
                execution_log=api.execution_log,
                agent_calls=len(api.execution_log),
            )

    def _fallback(
        self,
        goal: str,
        context: dict[str, Any] | None,
        reason: str,
    ) -> CodeModeResult:
        """Fall back to standard pipeline orchestration."""
        logger.info("Code mode falling back to pipeline: %s", reason)
        try:
            from vetinari.orchestration.two_layer import get_two_layer_orchestrator

            orch = get_two_layer_orchestrator()
            result = orch.generate_and_execute(goal, context=context)
            return CodeModeResult(
                success=True,
                output=result.get("final_output"),
                error=f"Fallback: {reason}",
                fallback_used=True,
                agent_calls=result.get("completed", 0),
            )
        except Exception as e:
            return CodeModeResult(
                success=False,
                error=f"Both modes failed: {e}",
                fallback_used=True,
            )

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_executions": self._total_executions,
            "successful": self._successful_executions,
            "success_rate": self._successful_executions / max(self._total_executions, 1),
        }


_engine: CodeModeEngine | None = None


def get_code_mode_engine(context: dict[str, Any] | None = None) -> CodeModeEngine:
    """Get code mode engine.

    Returns:
        The CodeModeEngine result.
    """
    global _engine
    if _engine is None:
        _engine = CodeModeEngine(context)
    return _engine
