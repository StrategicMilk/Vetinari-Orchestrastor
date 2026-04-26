"""Tests for agent mode handlers (cost_analysis, creative_writing, documentation).

Covers handler protocol conformance, HandlerRouter registration, and the
wiring into WorkerAgent's _dispatch_operations.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.factories import make_agent_task
from vetinari.agents.handlers import HandlerRouter
from vetinari.agents.handlers.cost_analysis_handler import (
    MODEL_PRICING,
    CostAnalysisHandler,
)
from vetinari.agents.handlers.creative_writing_handler import CreativeWritingHandler
from vetinari.agents.handlers.documentation_handler import DocumentationHandler
from vetinari.exceptions import AgentError
from vetinari.types import AgentType

# ── Helpers ──────────────────────────────────────────────────────────────────


def _noop_infer_json(prompt: str, fallback: Any = None) -> Any:
    """Stub infer_json that always returns the fallback."""
    return fallback


# ── HandlerRouter ─────────────────────────────────────────────────────────────


class TestHandlerRouter:
    """Tests for HandlerRouter registration and dispatch."""

    def test_register_and_available_modes(self):
        """Registered handlers appear in available_modes()."""
        router = HandlerRouter()
        router.register(DocumentationHandler())
        router.register(CostAnalysisHandler())
        modes = router.available_modes()
        assert "documentation" in modes
        assert "cost_analysis" in modes

    def test_duplicate_registration_raises(self):
        """Registering the same mode twice raises AgentError."""
        router = HandlerRouter()
        router.register(DocumentationHandler())
        with pytest.raises(AgentError, match="already registered"):
            router.register(DocumentationHandler())

    def test_get_handler_returns_none_for_unknown(self):
        """get_handler() returns None for unregistered modes."""
        router = HandlerRouter()
        assert router.get_handler("nonexistent") is None

    def test_route_returns_none_for_unknown_mode(self):
        """route() returns None when no handler is registered for the mode."""
        router = HandlerRouter()
        task = make_agent_task()
        result = router.route("unknown", task, {})
        assert result is None

    def test_route_dispatches_to_handler(self):
        """route() dispatches to the correct handler and returns its result."""
        router = HandlerRouter()
        router.register(CostAnalysisHandler())
        task = make_agent_task(context={"analysis_type": "model_comparison", "models": ["qwen2.5-coder-7b"]})
        result = router.route("cost_analysis", task, {"infer_json": _noop_infer_json})
        assert result is not None
        assert result.success is True

    def test_available_modes_sorted(self):
        """available_modes() returns a sorted list."""
        router = HandlerRouter()
        router.register(DocumentationHandler())
        router.register(CreativeWritingHandler())
        router.register(CostAnalysisHandler())
        modes = router.available_modes()
        assert modes == sorted(modes)


# ── CostAnalysisHandler ───────────────────────────────────────────────────────


class TestCostAnalysisHandler:
    """Tests for CostAnalysisHandler."""

    @pytest.fixture
    def handler(self) -> CostAnalysisHandler:
        """Fresh CostAnalysisHandler instance."""
        return CostAnalysisHandler()

    def test_mode_name(self, handler):
        """mode_name is 'cost_analysis'."""
        assert handler.mode_name == "cost_analysis"

    def test_description_is_meaningful(self, handler):
        """description is non-empty."""
        assert len(handler.description) > 10

    def test_system_prompt_non_empty(self, handler):
        """get_system_prompt() returns a non-empty string."""
        prompt = handler.get_system_prompt()
        assert len(prompt) > 50
        assert "cost" in prompt.lower()

    def test_model_comparison_returns_sorted_comparisons(self, handler):
        """model_comparison analysis returns comparisons sorted by total cost."""
        task = make_agent_task(
            context={
                "analysis_type": "model_comparison",
                "models": ["qwen2.5-coder-7b", "gpt-4o"],
                "estimated_tokens": 1000,
            },
        )
        result = handler.execute(task, {})
        assert result.success is True
        comparisons = result.output["comparisons"]
        assert len(comparisons) == 2
        costs = [c["total_cost"] for c in comparisons]
        assert costs == sorted(costs)

    def test_model_comparison_recommendation_is_cheapest(self, handler):
        """Recommendation is the cheapest model."""
        task = make_agent_task(
            context={
                "analysis_type": "model_comparison",
                "models": list(MODEL_PRICING.keys()),
                "estimated_tokens": 10000,
            },
        )
        result = handler.execute(task, {})
        assert result.success is True
        cheapest = result.output["comparisons"][0]["model"]
        assert result.output["recommendation"] == cheapest

    def test_general_analysis_uses_infer_json(self, handler):
        """General analysis delegates to infer_json when provided."""
        expected = {"analysis": "done", "recommendations": [], "estimated_savings": "50%"}
        infer_json = MagicMock(return_value=expected)
        task = make_agent_task(description="analyze cost", context={"analysis_type": "general"})
        result = handler.execute(task, {"infer_json": infer_json})
        assert result.success is True
        infer_json.assert_called_once()

    def test_general_analysis_uses_fallback_without_infer_json(self, handler):
        """General analysis returns success=False with _is_fallback flag when no infer_json provided."""
        task = make_agent_task(description="analyze cost", context={"analysis_type": "general"})
        result = handler.execute(task, {})
        assert result.success is False
        assert isinstance(result.output, dict)
        assert result.metadata.get("_is_fallback") is True

    def test_model_comparison_metadata(self, handler):
        """model_comparison result carries correct metadata."""
        task = make_agent_task(context={"analysis_type": "model_comparison"})
        result = handler.execute(task, {})
        assert result.metadata["mode"] == "cost_analysis"
        assert result.metadata["analysis_type"] == "model_comparison"


# ── CreativeWritingHandler ────────────────────────────────────────────────────


class TestCreativeWritingHandler:
    """Tests for CreativeWritingHandler."""

    @pytest.fixture
    def handler(self) -> CreativeWritingHandler:
        """Fresh CreativeWritingHandler instance."""
        return CreativeWritingHandler()

    def test_mode_name(self, handler):
        """mode_name is 'creative_writing'."""
        assert handler.mode_name == "creative_writing"

    def test_system_prompt_non_empty(self, handler):
        """get_system_prompt() returns a non-empty string."""
        prompt = handler.get_system_prompt()
        assert len(prompt) > 50
        assert "creative" in prompt.lower()

    def test_execute_uses_infer_json(self, handler):
        """execute() calls infer_json and wraps result in AgentResult."""
        expected = {"content": "Once upon a time...", "type": "creative"}
        infer_json = MagicMock(return_value=expected)
        task = make_agent_task(
            description="write a story",
            context={"content": "write a story", "style": "narrative"},
        )
        result = handler.execute(task, {"infer_json": infer_json})
        assert result.success is True
        assert result.output == expected
        infer_json.assert_called_once()

    def test_execute_fallback_without_infer_json(self, handler):
        """execute() uses fallback when no infer_json in context."""
        task = make_agent_task(description="write something", context={})
        result = handler.execute(task, {})
        assert result.success is True
        assert isinstance(result.output, dict)

    def test_metadata_mode(self, handler):
        """Result metadata contains mode key."""
        task = make_agent_task(context={})
        result = handler.execute(task, {"infer_json": _noop_infer_json})
        assert result.metadata["mode"] == "creative_writing"

    def test_description_from_task_when_no_content(self, handler):
        """Falls back to task.description when context lacks 'content'."""
        called_with: list[str] = []

        def capturing_infer(prompt: str, fallback: Any = None) -> Any:
            called_with.append(prompt)
            return fallback

        task = make_agent_task(description="My unique description XYZ", context={})
        handler.execute(task, {"infer_json": capturing_infer})
        assert any("My unique description XYZ" in p for p in called_with)


# ── DocumentationHandler ──────────────────────────────────────────────────────


class TestDocumentationHandler:
    """Tests for DocumentationHandler."""

    @pytest.fixture
    def handler(self) -> DocumentationHandler:
        """Fresh DocumentationHandler instance."""
        return DocumentationHandler()

    def test_mode_name(self, handler):
        """mode_name is 'documentation'."""
        assert handler.mode_name == "documentation"

    def test_system_prompt_non_empty(self, handler):
        """get_system_prompt() returns a non-empty string."""
        prompt = handler.get_system_prompt()
        assert len(prompt) > 50
        assert "documentation" in prompt.lower()

    def test_execute_uses_infer_json(self, handler):
        """execute() calls infer_json with the documentation prompt."""
        expected = {"content": "# API\n...", "type": "api_reference", "sections": []}
        infer_json = MagicMock(return_value=expected)
        task = make_agent_task(
            context={"content": "class Foo:", "doc_type": "api_reference", "audience": "developers"},
        )
        result = handler.execute(task, {"infer_json": infer_json})
        assert result.success is True
        assert result.output == expected
        infer_json.assert_called_once()

    def test_execute_fallback_without_infer_json(self, handler):
        """execute() returns success=False with _is_fallback flag when no infer_json in context."""
        task = make_agent_task(context={})
        result = handler.execute(task, {})
        assert result.success is False
        assert isinstance(result.output, dict)
        assert result.metadata.get("_is_fallback") is True

    def test_metadata_mode(self, handler):
        """Result metadata contains mode and doc_type keys."""
        task = make_agent_task(context={"doc_type": "guide"})
        result = handler.execute(task, {"infer_json": _noop_infer_json})
        assert result.metadata["mode"] == "documentation"
        assert result.metadata["doc_type"] == "guide"

    def test_default_doc_type_is_api_reference(self, handler):
        """Default doc_type is 'api_reference' when not specified."""
        task = make_agent_task(context={})
        result = handler.execute(task, {"infer_json": _noop_infer_json})
        assert result.metadata["doc_type"] == "api_reference"


# ── WorkerAgent integration ───────────────────────────────────────────────────


class TestWorkerHandlerIntegration:
    """Verify handlers are wired into WorkerAgent._dispatch_operations."""

    @pytest.fixture
    def worker(self):
        """Create a WorkerAgent instance."""
        from vetinari.agents.consolidated.worker_agent import WorkerAgent

        return WorkerAgent()

    def test_handler_router_initialized(self, worker):
        """WorkerAgent has a _handler_router attribute."""
        assert hasattr(worker, "_handler_router")

    def test_handlers_registered_in_router(self, worker):
        """All three handlers are registered in the router."""
        modes = worker._handler_router.available_modes()
        assert "documentation" in modes
        assert "creative_writing" in modes
        assert "cost_analysis" in modes

    def test_handlers_importable_from_worker_module(self):
        """HandlerRouter and handlers can be imported from their canonical paths."""
        from vetinari.agents.handlers import HandlerRouter
        from vetinari.agents.handlers.cost_analysis_handler import CostAnalysisHandler
        from vetinari.agents.handlers.creative_writing_handler import CreativeWritingHandler
        from vetinari.agents.handlers.documentation_handler import DocumentationHandler

        assert issubclass(CostAnalysisHandler, object)
        assert issubclass(CreativeWritingHandler, object)
        assert issubclass(DocumentationHandler, object)
        assert callable(HandlerRouter)

    def test_cost_analysis_mode_in_available_modes(self, worker):
        """cost_analysis is in the worker's available_modes."""
        assert "cost_analysis" in worker.available_modes

    def test_documentation_mode_in_available_modes(self, worker):
        """documentation is in the worker's available_modes."""
        assert "documentation" in worker.available_modes

    def test_creative_writing_mode_in_available_modes(self, worker):
        """creative_writing is in the worker's available_modes."""
        assert "creative_writing" in worker.available_modes
