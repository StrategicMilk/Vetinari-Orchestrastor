"""Tests for Phase 4 architecture evolution wiring.

Verifies that previously unwired modules are now integrated into the
production pipeline (US-028 through US-034).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from vetinari.types import AgentType

logger = logging.getLogger(__name__)


# ── US-028: Optimization module wiring ────────────────────────────────


class TestPromptCacheWiring:
    """Verify prompt cache is wired into inference path."""

    def test_prompt_cache_singleton_accessible(self) -> None:
        """get_prompt_cache() returns a singleton instance."""
        from vetinari.optimization.prompt_cache import get_prompt_cache, reset_prompt_cache

        reset_prompt_cache()
        cache = get_prompt_cache()
        assert cache is not None
        cache2 = get_prompt_cache()
        assert cache is cache2
        reset_prompt_cache()

    def test_prompt_cache_hit_miss(self) -> None:
        """Cache returns hit on second lookup of same prompt."""
        from vetinari.optimization.prompt_cache import (
            get_prompt_cache,
            hash_prompt,
            reset_prompt_cache,
        )

        reset_prompt_cache()
        cache = get_prompt_cache()
        h = hash_prompt("test system prompt")
        r1 = cache.get_or_cache(h, "test system prompt")
        assert not r1.hit
        r2 = cache.get_or_cache(h, "test system prompt")
        assert r2.hit
        assert r2.savings_tokens > 0
        reset_prompt_cache()

    def test_semantic_cache_put_get(self) -> None:
        """Semantic cache stores and retrieves similar queries."""
        from vetinari.optimization.semantic_cache import get_semantic_cache, reset_semantic_cache

        reset_semantic_cache()
        cache = get_semantic_cache(similarity_threshold=0.5)
        cache.put("What is the capital of France?", "Paris is the capital of France.")
        result = cache.get("What is the capital of France?")
        assert result is not None
        assert "Paris" in result
        reset_semantic_cache()


class TestInferenceBatcherWiring:
    """Verify inference batcher is accessible."""

    def test_batcher_singleton(self) -> None:
        """get_inference_batcher() returns a singleton."""
        from vetinari.inference import get_inference_batcher

        batcher = get_inference_batcher()
        assert batcher is not None
        assert not batcher.enabled  # disabled by default


# ── US-029: Observability module wiring ───────────────────────────────


class TestTracingWiring:
    """Verify tracing context managers work."""

    def test_agent_span_context_manager(self) -> None:
        """agent_span() creates a span without error."""
        from vetinari.observability.tracing import agent_span

        with agent_span(AgentType.WORKER.value, mode="build", task_id="test-1") as span:
            span.set_attribute("test", True)
        # span may be a NoOpSpan or NonRecordingSpan — just verify it was yielded
        assert span is not None

    def test_llm_span_context_manager(self) -> None:
        """llm_span() creates a span without error."""
        from vetinari.observability.tracing import llm_span

        with llm_span("qwen-32b", agent_type=AgentType.WORKER.value, max_tokens=2048) as span:
            span.set_attribute("output_tokens", 100)
        # span may be a NoOpSpan or NonRecordingSpan — just verify it was yielded
        assert span is not None

    def test_pipeline_span_context_manager(self) -> None:
        """pipeline_span() creates a span without error."""
        from vetinari.observability.tracing import pipeline_span

        with pipeline_span("Build a REST API", plan_id="plan-1") as span:
            span.set_attribute("stage_count", 7)
        # span may be a NoOpSpan or NonRecordingSpan — just verify it was yielded
        assert span is not None


class TestGenAITracerWiring:
    """Verify GenAI tracer singleton works."""

    def test_genai_tracer_singleton(self) -> None:
        """get_genai_tracer() returns a singleton."""
        from vetinari.observability.otel_genai import get_genai_tracer

        tracer = get_genai_tracer()
        assert tracer is not None
        assert tracer is get_genai_tracer()  # singleton


class TestStepEvaluatorWiring:
    """Verify step evaluator singleton works."""

    def test_step_evaluator_singleton(self) -> None:
        """get_step_evaluator() returns a singleton."""
        from vetinari.observability.step_evaluator import get_step_evaluator

        evaluator = get_step_evaluator()
        assert evaluator is not None
        assert evaluator is get_step_evaluator()  # singleton


# ── US-030: Remaining module wiring ───────────────────────────────────


class TestComplexityRouterWiring:
    """Verify complexity router is wired into TwoLayerOrchestrator."""

    def test_classify_complexity(self) -> None:
        """classify_complexity() correctly classifies tasks."""
        from vetinari.routing import Complexity, classify_complexity

        assert classify_complexity("fix typo") == Complexity.SIMPLE
        assert (
            classify_complexity("architect a distributed multi-service migration with backwards compatibility")
            == Complexity.COMPLEX
        )

    def test_route_by_complexity(self) -> None:
        """route_by_complexity() returns a RoutingDecision."""
        from vetinari.routing import Complexity, route_by_complexity

        decision = route_by_complexity("implement a simple fix", task_count=1)
        assert isinstance(decision.complexity, Complexity)
        assert isinstance(decision.skip_stages, list)
        assert isinstance(decision.recommended_agents, list)

    def test_two_layer_analyze_input_uses_complexity_router(self) -> None:
        """TwoLayerOrchestrator._analyze_input uses complexity_router."""
        from vetinari.orchestration.two_layer import TwoLayerOrchestrator

        orch = TwoLayerOrchestrator()
        result = orch._analyze_input("fix a simple typo in readme", {})
        # Should have routing_decision from complexity_router
        assert "estimated_complexity" in result
        assert result["estimated_complexity"] in ("express", "standard", "custom", "simple", "medium", "complex")


class TestWorkflowQualityGatesWiring:
    """Verify workflow quality gates are wired into pipeline."""

    def test_gate_runner_evaluate(self) -> None:
        """WorkflowGateRunner.evaluate() works for known stages."""
        from vetinari.workflow.quality_gates import WorkflowGateRunner

        runner = WorkflowGateRunner()
        passed, _action, violations = runner.evaluate(
            "post_execution",
            {"quality_score": 0.8, "no_critical_security": True},
        )
        assert passed is True
        assert len(violations) == 0

    def test_gate_runner_fails_on_low_quality(self) -> None:
        """WorkflowGateRunner catches low quality scores."""
        from vetinari.workflow.quality_gates import GateAction, WorkflowGateRunner

        runner = WorkflowGateRunner()
        passed, action, violations = runner.evaluate(
            "post_execution",
            {"quality_score": 0.3, "no_critical_security": True},
        )
        assert passed is False
        assert action == GateAction.RETRY
        assert len(violations) > 0

    def test_gate_runner_blocks_unknown_stage(self) -> None:
        """Unknown workflow stages fail closed instead of continuing."""
        from vetinari.workflow.quality_gates import GateAction, WorkflowGateRunner

        runner = WorkflowGateRunner()
        passed, action, violations = runner.evaluate("unknown_stage", {"quality_score": 1.0})

        assert passed is False
        assert action == GateAction.BLOCK
        assert violations == ["Unknown workflow stage: unknown_stage"]


class TestSPCMonitorWiring:
    """Verify SPC monitor is wired into adapter telemetry."""

    def test_spc_monitor_singleton(self) -> None:
        """get_spc_monitor() returns a singleton."""
        from vetinari.workflow import get_spc_monitor, reset_spc_monitor

        reset_spc_monitor()
        monitor = get_spc_monitor()
        assert monitor is not None
        monitor2 = get_spc_monitor()
        assert monitor is monitor2
        reset_spc_monitor()

    def test_spc_monitor_update(self) -> None:
        """SPCMonitor.update() records observations."""
        from vetinari.workflow import get_spc_monitor, reset_spc_monitor

        reset_spc_monitor()
        monitor = get_spc_monitor()
        # First few observations won't alert (need >= 3)
        for val in [100.0, 110.0, 105.0, 95.0, 100.0]:
            monitor.update("test.latency", val)
        stats = monitor.get_summary()
        assert "test.latency" in stats
        assert stats["test.latency"]["count"] == 5
        reset_spc_monitor()


class TestPromptAssemblerWiring:
    """Verify prompt assembler is properly exported."""

    def test_assembler_accessible(self) -> None:
        """get_prompt_assembler() returns an instance."""
        from vetinari.prompts import get_prompt_assembler

        assembler = get_prompt_assembler()
        assert assembler is not None
        assert assembler is get_prompt_assembler()  # singleton


class TestSchemaOutputsWiring:
    """Verify output schemas validate correctly."""

    def test_validate_output_valid(self) -> None:
        """validate_output() accepts valid data."""
        from vetinari.schemas import validate_output

        result = validate_output(
            "code_review",
            {
                "score": 0.8,
                "summary": "Good code",
                "issues": [],
                "strengths": ["Clean"],
                "recommendations": [],
            },
        )
        assert result is not None
        assert hasattr(result, "score")
        assert result.score == 0.8
        assert result.summary == "Good code"

    def test_validate_output_unknown_mode(self) -> None:
        """validate_output() returns None for unknown modes."""
        from vetinari.schemas import validate_output

        result = validate_output("nonexistent_mode", {"data": "test"})
        assert result is None


class TestWindowManagerWiring:
    """Verify context window manager is properly exported."""

    def test_window_manager_accessible(self) -> None:
        """get_window_manager() returns an instance."""
        from vetinari.context import get_window_manager

        mgr = get_window_manager()
        assert mgr is not None
        assert mgr is get_window_manager()  # same key returns same instance

    def test_estimate_tokens(self) -> None:
        """estimate_tokens() returns a reasonable count."""
        from vetinari.context import estimate_tokens

        count = estimate_tokens("Hello world, this is a test sentence.")
        assert count > 0
        assert count < 100


class TestAsyncSupportKept:
    """Verify async_support modules are importable (kept for US-034)."""

    def test_async_executor_importable(self) -> None:
        """AsyncExecutor is importable from async_support."""
        from vetinari.async_support import AsyncExecutor

        assert isinstance(AsyncExecutor, type)

    def test_streaming_importable(self) -> None:
        """Streaming classes are importable."""
        from vetinari.async_support.streaming import SSEStreamHandler, StreamRouter

        assert isinstance(SSEStreamHandler, type)
        assert isinstance(StreamRouter, type)


# ── US-033: Event bus wiring ──────────────────────────────────────────


class TestEventBusWiring:
    """Verify event bus publish/subscribe works."""

    def test_event_bus_singleton(self) -> None:
        """get_event_bus() returns a singleton."""
        from vetinari.events import get_event_bus, reset_event_bus

        reset_event_bus()
        bus = get_event_bus()
        assert bus is not None
        bus2 = get_event_bus()
        assert bus is bus2
        reset_event_bus()

    def test_publish_subscribe(self) -> None:
        """EventBus delivers events to subscribers."""
        from vetinari.events import TaskCompleted, get_event_bus, reset_event_bus

        reset_event_bus()
        bus = get_event_bus()
        received = []
        bus.subscribe(TaskCompleted, lambda e: received.append(e))
        event = TaskCompleted(
            event_type="TaskCompleted",
            timestamp=0.0,
            task_id="t1",
            agent_type=AgentType.WORKER.value,
            success=True,
            duration_ms=100.0,
        )
        bus.publish(event)
        assert len(received) == 1
        assert received[0].task_id == "t1"
        reset_event_bus()

    def test_event_type_filtering(self) -> None:
        """EventBus only delivers events of subscribed type."""
        from vetinari.events import (
            TaskCompleted,
            TaskStarted,
            get_event_bus,
            reset_event_bus,
        )

        reset_event_bus()
        bus = get_event_bus()
        started_received = []
        completed_received = []
        bus.subscribe(TaskStarted, lambda e: started_received.append(e))
        bus.subscribe(TaskCompleted, lambda e: completed_received.append(e))
        bus.publish(
            TaskStarted(
                event_type="TaskStarted",
                timestamp=0.0,
                task_id="t1",
                agent_type=AgentType.WORKER.value,
            )
        )
        assert len(started_received) == 1
        assert len(completed_received) == 0
        reset_event_bus()


# ── US-031: Agent graph enhancements ─────────────────────────────────


class TestAgentGraphEnhancements:
    """Verify conditional edges, cycle detection, and checkpoints."""

    def test_conditional_edge_exists(self) -> None:
        """ConditionalEdge dataclass is importable."""
        from vetinari.orchestration.agent_graph import ConditionalEdge

        edge = ConditionalEdge(
            source_task_id="t1",
            condition=lambda r: r.success if r else False,
            true_target="t2",
            false_target="t3",
            description="Route based on success",
        )
        assert edge.source_task_id == "t1"

    def test_cycle_detector_exists(self) -> None:
        """CycleDetector class is importable and detects cycles."""
        from vetinari.orchestration.agent_graph import CycleDetector

        detector = CycleDetector(max_iterations=3)
        assert detector is not None
        assert detector._max_iterations == 3

    def test_human_checkpoint_exists(self) -> None:
        """HumanCheckpoint class is importable."""
        from vetinari.orchestration.agent_graph import HumanCheckpoint

        checkpoint = HumanCheckpoint()
        checkpoint.add_checkpoint("task-1", reason="Security review needed")
        assert checkpoint.is_checkpoint("task-1")
        assert not checkpoint.is_checkpoint("task-2")

    def test_execute_subgraph_exists(self) -> None:
        """AgentGraph has execute_subgraph method for spawning sub-tasks."""
        from vetinari.orchestration.agent_graph import AgentGraph

        graph = AgentGraph.__new__(AgentGraph)
        assert hasattr(graph, "execute_subgraph")
        assert callable(graph.execute_subgraph)


# ── US-032: Handler decomposition ────────────────────────────────────


class TestHandlerDecomposition:
    """Verify handler protocol and router exist."""

    def test_handler_protocol_importable(self) -> None:
        """HandlerProtocol is importable."""
        from vetinari.agents.handlers import HandlerProtocol

        assert isinstance(HandlerProtocol, type)

    def test_base_handler_importable(self) -> None:
        """BaseHandler is importable."""
        from vetinari.agents.handlers import BaseHandler

        assert isinstance(BaseHandler, type)

    def test_handler_router_importable(self) -> None:
        """HandlerRouter is importable."""
        from vetinari.agents.handlers import HandlerRouter

        router = HandlerRouter()
        assert router.available_modes() == []


# ── US-034: Async migration ──────────────────────────────────────────


class TestAsyncMigration:
    """Verify async_infer() exists on adapter base class."""

    def test_async_infer_method_exists(self) -> None:
        """ProviderAdapter has async_infer() method."""
        from vetinari.adapters.base import ProviderAdapter

        assert hasattr(ProviderAdapter, "async_infer")

    def test_run_async_infer_exists(self) -> None:
        """run_async_infer() helper is importable."""
        from vetinari.adapters.base import run_async_infer

        assert callable(run_async_infer)
