"""Integration tests for OpenTelemetry GenAI semantic-conventions tracer (US-035).

Covers:
- GenAITracer span creation with all required GenAI attributes
- gen_ai.system attribute always populated as "vetinari"
- Child spans reference parent trace_id and parent_span_id
- export_traces produces valid JSON containing all spans
- Hierarchical nesting: pipeline > agent > llm
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from vetinari.observability.otel_genai import (
    ATTR_AGENT_NAME,
    ATTR_OPERATION,
    ATTR_REQUEST_MODEL,
    ATTR_SYSTEM,
    GenAITracer,
    SpanContext,
    get_genai_tracer,
    reset_genai_tracer,
)


@pytest.fixture(autouse=True)
def _fresh_tracer():
    """Reset the singleton before and after every test for isolation."""
    reset_genai_tracer()
    yield
    reset_genai_tracer()


@pytest.fixture
def tracer() -> GenAITracer:
    """Return a fresh GenAITracer instance (not the singleton)."""
    return GenAITracer()


class TestSpanCreation:
    """GenAITracer creates spans with all required GenAI attributes."""

    def test_start_agent_span_returns_span_context(self, tracer: GenAITracer) -> None:
        """start_agent_span returns a SpanContext with non-empty IDs."""
        span = tracer.start_agent_span("builder", "chat")
        assert isinstance(span, SpanContext)
        assert len(span.span_id) == 16
        assert len(span.trace_id) == 32

    def test_span_has_agent_name_attribute(self, tracer: GenAITracer) -> None:
        """gen_ai.agent.name attribute is set on newly created spans."""
        span = tracer.start_agent_span("planner", "chat")
        assert span.attributes[ATTR_AGENT_NAME] == "planner"

    def test_span_has_operation_attribute(self, tracer: GenAITracer) -> None:
        """gen_ai.operation.name attribute is set on newly created spans."""
        span = tracer.start_agent_span("builder", "embeddings")
        assert span.attributes[ATTR_OPERATION] == "embeddings"

    def test_span_has_model_attribute_when_provided(self, tracer: GenAITracer) -> None:
        """gen_ai.request.model attribute is set when model is supplied."""
        span = tracer.start_agent_span("builder", "chat", model="qwen-32b")
        assert span.attributes[ATTR_REQUEST_MODEL] == "qwen-32b"

    def test_span_has_no_model_attribute_when_omitted(self, tracer: GenAITracer) -> None:
        """gen_ai.request.model is absent when model is not supplied."""
        span = tracer.start_agent_span("builder", "chat")
        assert ATTR_REQUEST_MODEL not in span.attributes

    def test_span_is_active_initially(self, tracer: GenAITracer) -> None:
        """A freshly created span reports is_active=True."""
        span = tracer.start_agent_span("builder", "chat")
        assert span.is_active is True

    def test_span_inactive_after_end(self, tracer: GenAITracer) -> None:
        """Span reports is_active=False after end_agent_span is called."""
        span = tracer.start_agent_span("builder", "chat")
        tracer.end_agent_span(span)
        assert span.is_active is False


class TestGenAiSystemAttribute:
    """start_agent_span populates gen_ai.system = 'vetinari'."""

    def test_system_attribute_set_on_new_span(self, tracer: GenAITracer) -> None:
        """Every new span includes gen_ai.system = 'vetinari'."""
        span = tracer.start_agent_span("quality", "chat")
        assert span.attributes[ATTR_SYSTEM] == "vetinari"

    def test_system_attribute_value_is_vetinari(self, tracer: GenAITracer) -> None:
        """The gen_ai.system value is always the string 'vetinari'."""
        for agent in ("planner", "builder", "quality", "operations"):
            span = tracer.start_agent_span(agent, "chat")
            assert span.attributes[ATTR_SYSTEM] == "vetinari", (
                f"Expected 'vetinari' for agent={agent}, got {span.attributes[ATTR_SYSTEM]!r}"
            )


class TestChildSpans:
    """Child spans reference parent trace_id and parent_span_id."""

    def test_child_span_inherits_trace_id(self, tracer: GenAITracer) -> None:
        """Child span has the same trace_id as its parent."""
        parent = tracer.start_agent_span("pipeline", "orchestrate")
        child = tracer.start_child_span(parent, "builder", "chat")
        assert child.trace_id == parent.trace_id

    def test_child_span_records_parent_span_id(self, tracer: GenAITracer) -> None:
        """Child span's parent_span_id equals the parent's span_id."""
        parent = tracer.start_agent_span("pipeline", "orchestrate")
        child = tracer.start_child_span(parent, "builder", "chat")
        assert child.parent_span_id == parent.span_id

    def test_child_span_has_unique_span_id(self, tracer: GenAITracer) -> None:
        """Child span has a different span_id from its parent."""
        parent = tracer.start_agent_span("pipeline", "orchestrate")
        child = tracer.start_child_span(parent, "builder", "chat")
        assert child.span_id != parent.span_id

    def test_root_span_has_no_parent(self, tracer: GenAITracer) -> None:
        """Root spans (created by start_agent_span) have parent_span_id=None."""
        span = tracer.start_agent_span("builder", "chat")
        assert span.parent_span_id is None

    def test_child_span_has_system_attribute(self, tracer: GenAITracer) -> None:
        """Child spans also carry the gen_ai.system = 'vetinari' attribute."""
        parent = tracer.start_agent_span("pipeline", "orchestrate")
        child = tracer.start_child_span(parent, "builder", "chat", model="llama")
        assert child.attributes[ATTR_SYSTEM] == "vetinari"

    def test_child_span_model_attribute(self, tracer: GenAITracer) -> None:
        """Child span records model when provided."""
        parent = tracer.start_agent_span("pipeline", "orchestrate")
        child = tracer.start_child_span(parent, "builder", "chat", model="llama-3")
        assert child.attributes[ATTR_REQUEST_MODEL] == "llama-3"


class TestExportTraces:
    """export_traces produces valid JSON with all spans."""

    def test_export_traces_writes_file(self, tracer: GenAITracer) -> None:
        """export_traces creates a file at the given path."""
        span = tracer.start_agent_span("builder", "chat")
        tracer.end_agent_span(span)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            tracer.export_traces(tmp_path)
            assert Path(tmp_path).exists()
        finally:
            os.unlink(tmp_path)

    def test_export_traces_valid_json(self, tracer: GenAITracer) -> None:
        """Exported file contains valid JSON."""
        span = tracer.start_agent_span("builder", "chat")
        tracer.end_agent_span(span, tokens_used=100)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            tracer.export_traces(tmp_path)
            with open(tmp_path, encoding="utf-8") as fh:
                data = json.load(fh)
            assert isinstance(data, dict)
        finally:
            os.unlink(tmp_path)

    def test_export_traces_contains_all_spans(self, tracer: GenAITracer) -> None:
        """All completed spans appear in the exported JSON."""
        span1 = tracer.start_agent_span("builder", "chat")
        span2 = tracer.start_agent_span("quality", "review")
        tracer.end_agent_span(span1)
        tracer.end_agent_span(span2)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            count = tracer.export_traces(tmp_path)
            assert count == 2

            with open(tmp_path, encoding="utf-8") as fh:
                data = json.load(fh)
            span_ids = {s["span_id"] for s in data["spans"]}
            assert span1.span_id in span_ids
            assert span2.span_id in span_ids
        finally:
            os.unlink(tmp_path)

    def test_export_traces_span_has_parent_span_id_field(self, tracer: GenAITracer) -> None:
        """Every span dict in the export contains the parent_span_id field."""
        parent = tracer.start_agent_span("pipeline", "orchestrate")
        child = tracer.start_child_span(parent, "builder", "chat")
        tracer.end_agent_span(parent)
        tracer.end_agent_span(child)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            tracer.export_traces(tmp_path)
            with open(tmp_path, encoding="utf-8") as fh:
                data = json.load(fh)

            for span_dict in data["spans"]:
                assert "parent_span_id" in span_dict, f"span {span_dict['span_id']} missing parent_span_id"
        finally:
            os.unlink(tmp_path)

    def test_export_traces_schema_metadata(self, tracer: GenAITracer) -> None:
        """Exported JSON includes service name and schema version metadata."""
        span = tracer.start_agent_span("builder", "chat")
        tracer.end_agent_span(span)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            tracer.export_traces(tmp_path)
            with open(tmp_path, encoding="utf-8") as fh:
                data = json.load(fh)
            assert data["service"] == "vetinari"
            assert "schema_version" in data
            assert "spans" in data
        finally:
            os.unlink(tmp_path)


class TestHierarchicalNesting:
    """Hierarchical nesting: pipeline > agent > llm spans."""

    def test_three_level_hierarchy(self, tracer: GenAITracer) -> None:
        """Pipeline > agent > llm span chain shares trace_id and correct parent links."""
        pipeline = tracer.start_agent_span("pipeline", "orchestrate")
        agent = tracer.start_child_span(pipeline, "builder", "chat")
        llm = tracer.start_child_span(agent, "llm", "inference", model="qwen-32b")

        # All share the same trace_id
        assert pipeline.trace_id == agent.trace_id == llm.trace_id

        # Parent chain is correct
        assert agent.parent_span_id == pipeline.span_id
        assert llm.parent_span_id == agent.span_id

        # Root has no parent
        assert pipeline.parent_span_id is None

        tracer.end_agent_span(llm)
        tracer.end_agent_span(agent)
        tracer.end_agent_span(pipeline)

    def test_stats_counts_all_completed_spans(self, tracer: GenAITracer) -> None:
        """get_stats() reports the correct number of completed spans."""
        pipeline = tracer.start_agent_span("pipeline", "orchestrate")
        agent = tracer.start_child_span(pipeline, "builder", "chat")
        llm = tracer.start_child_span(agent, "llm", "inference")

        tracer.end_agent_span(llm)
        tracer.end_agent_span(agent)
        tracer.end_agent_span(pipeline)

        stats = tracer.get_stats()
        assert stats["total_spans"] == 3
        assert stats["active_spans"] == 0

    def test_nested_spans_in_export(self, tracer: GenAITracer) -> None:
        """Exported JSON correctly captures the three-level span hierarchy."""
        pipeline = tracer.start_agent_span("pipeline", "orchestrate")
        agent = tracer.start_child_span(pipeline, "builder", "chat")
        llm = tracer.start_child_span(agent, "llm", "inference")

        tracer.end_agent_span(llm)
        tracer.end_agent_span(agent)
        tracer.end_agent_span(pipeline)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            tracer.export_traces(tmp_path)
            with open(tmp_path, encoding="utf-8") as fh:
                data = json.load(fh)

            by_id = {s["span_id"]: s for s in data["spans"]}
            assert by_id[pipeline.span_id]["parent_span_id"] is None
            assert by_id[agent.span_id]["parent_span_id"] == pipeline.span_id
            assert by_id[llm.span_id]["parent_span_id"] == agent.span_id
        finally:
            os.unlink(tmp_path)


class TestSingleton:
    """get_genai_tracer returns the same instance each call."""

    def test_singleton_returns_same_instance(self) -> None:
        """get_genai_tracer() returns the same object on repeated calls."""
        t1 = get_genai_tracer()
        t2 = get_genai_tracer()
        assert t1 is t2

    def test_reset_clears_singleton(self) -> None:
        """reset_genai_tracer() causes the next call to return a new instance."""
        t1 = get_genai_tracer()
        reset_genai_tracer()
        t2 = get_genai_tracer()
        assert t1 is not t2
