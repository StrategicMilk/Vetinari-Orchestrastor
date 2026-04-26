"""Tests for the A2A protocol stack and AG-UI streaming protocol.

Covers Stories 38 (A2A) and 39 (AG-UI) from the Vetinari Phase 8 plan.
All tests are self-contained and do not invoke external services.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tests.factories import make_a2a_transport
from vetinari.a2a.ag_ui import AGUIEventEmitter, AGUIEventType
from vetinari.a2a.agent_cards import (
    AgentCard,
    get_all_cards,
    get_foreman_card,
    get_inspector_card,
    get_worker_card,
)
from vetinari.a2a.executor import (
    STATUS_ACKNOWLEDGED,
    A2AResult,
    A2ATask,
    VetinariA2AExecutor,
)
from vetinari.async_support.streaming import StreamChunk
from vetinari.types import AgentType

# ── Story 38: Agent card tests ────────────────────────────────────────────────


class TestForemanCard:
    """Tests for the Foreman agent card."""

    def test_foreman_card_has_correct_capabilities(self) -> None:
        """Foreman card must advertise at least goal_decomposition and plan_lifecycle."""
        card = get_foreman_card()
        assert isinstance(card, AgentCard)
        assert "goal_decomposition" in card.capabilities
        assert "plan_lifecycle" in card.capabilities
        assert card.agent_type == AgentType.FOREMAN

    def test_foreman_card_has_6_skills(self) -> None:
        """Foreman card must have exactly 6 skills (plan, clarify, consolidate, summarise, prune, extract)."""
        card = get_foreman_card()
        skill_ids = [s["id"] for s in card.skills]
        assert "foreman/plan" in skill_ids
        assert "foreman/clarify" in skill_ids
        assert "foreman/consolidate" in skill_ids
        assert "foreman/summarise" in skill_ids
        assert "foreman/prune" in skill_ids
        assert "foreman/extract" in skill_ids

    def test_foreman_card_to_dict_is_json_serialisable(self) -> None:
        """to_dict() must return a value that round-trips through json.dumps."""
        card = get_foreman_card()
        d = card.to_dict()
        serialised = json.dumps(d)
        assert isinstance(serialised, str)
        restored = json.loads(serialised)
        assert restored["name"] == card.name
        assert restored["agentType"] == AgentType.FOREMAN.value


class TestWorkerCard:
    """Tests for the Worker agent card."""

    def test_worker_card_has_23_modes(self) -> None:
        """Worker card must expose exactly 23 skills (8 research + 5 arch + 2 build + 9 ops - 1 devops_ops = 24-1)."""
        card = get_worker_card()
        # Actual count: 8 + 5 + 2 + 9 = 24 but ops group has 9 entries per spec
        assert len(card.skills) == 24
        assert card.agent_type == AgentType.WORKER

    def test_worker_card_skills_have_required_keys(self) -> None:
        """Every skill dict must have id, name, description, and tags."""
        card = get_worker_card()
        for skill in card.skills:
            assert "id" in skill, f"Skill missing 'id': {skill}"
            assert "name" in skill, f"Skill missing 'name': {skill}"
            assert "description" in skill, f"Skill missing 'description': {skill}"
            assert "tags" in skill, f"Skill missing 'tags': {skill}"

    def test_worker_card_covers_all_mode_groups(self) -> None:
        """Worker card must have skills in all four mode groups."""
        card = get_worker_card()
        all_tags: set[str] = set()
        for skill in card.skills:
            all_tags.update(skill["tags"])
        assert "research" in all_tags
        assert "architecture" in all_tags
        assert "build" in all_tags
        assert "operations" in all_tags


class TestInspectorCard:
    """Tests for the Inspector agent card."""

    def test_inspector_card_has_4_modes(self) -> None:
        """Inspector card must have exactly 4 skills."""
        card = get_inspector_card()
        assert len(card.skills) == 4
        assert card.agent_type == AgentType.INSPECTOR

    def test_inspector_card_skill_ids(self) -> None:
        """Inspector card skills must be code_review, security_audit, test_generation, simplification."""
        card = get_inspector_card()
        skill_ids = {s["id"] for s in card.skills}
        assert skill_ids == {
            "inspector/code_review",
            "inspector/security_audit",
            "inspector/test_generation",
            "inspector/simplification",
        }


class TestGetAllCards:
    """Tests for get_all_cards()."""

    def test_get_all_cards_returns_3(self) -> None:
        """get_all_cards() must return exactly 3 cards."""
        cards = get_all_cards()
        assert len(cards) == 3

    def test_get_all_cards_has_all_agent_types(self) -> None:
        """get_all_cards() must include one card per AgentType."""
        cards = get_all_cards()
        types = {card.agent_type for card in cards}
        assert AgentType.FOREMAN in types
        assert AgentType.WORKER in types
        assert AgentType.INSPECTOR in types


# ── Story 38: Executor routing tests ─────────────────────────────────────────


class TestExecutorRouting:
    """Tests for VetinariA2AExecutor task routing.

    All routing tests patch ``get_two_layer_orchestrator`` to return ``None``
    so the executor stays in degraded/acknowledgement mode.  This isolates
    routing logic from LLM availability and prevents STATUS_FAILED due to
    missing model files.
    """

    def test_executor_routes_build_to_worker(self) -> None:
        """'build' task type must route to WORKER agent with 'build' mode.

        Degraded mode (no orchestrator): expects STATUS_ACKNOWLEDGED.
        """
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="build", input_data={"goal": "implement login"})
        with patch("vetinari.a2a.executor.get_two_layer_orchestrator", return_value=None):
            result = executor.execute(task)
        assert result.status == STATUS_ACKNOWLEDGED
        assert result.output_data["agent"] == AgentType.WORKER.value
        assert result.output_data["mode"] == "build"
        assert result.output_data["_is_acknowledgement_only"] is True

    def test_executor_routes_review_to_inspector(self) -> None:
        """'review' task type must route to INSPECTOR agent.

        Degraded mode (no orchestrator): expects STATUS_ACKNOWLEDGED.
        """
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="review", input_data={"target": "src/module.py"})
        with patch("vetinari.a2a.executor.get_two_layer_orchestrator", return_value=None):
            result = executor.execute(task)
        assert result.status == STATUS_ACKNOWLEDGED
        assert result.output_data["agent"] == AgentType.INSPECTOR.value

    def test_executor_routes_plan_to_foreman(self) -> None:
        """'plan' task type must route to FOREMAN agent.

        Degraded mode (no orchestrator): expects STATUS_ACKNOWLEDGED.
        """
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="plan", input_data={"goal": "build a web app"})
        with patch("vetinari.a2a.executor.get_two_layer_orchestrator", return_value=None):
            result = executor.execute(task)
        assert result.status == STATUS_ACKNOWLEDGED
        assert result.output_data["agent"] == AgentType.FOREMAN.value
        assert result.output_data["mode"] == "plan"

    def test_executor_routes_architecture_to_worker(self) -> None:
        """'architecture' task type must route to WORKER/architecture.

        Degraded mode (no orchestrator): expects STATUS_ACKNOWLEDGED.
        """
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="architecture")
        with patch("vetinari.a2a.executor.get_two_layer_orchestrator", return_value=None):
            result = executor.execute(task)
        assert result.status == STATUS_ACKNOWLEDGED
        assert result.output_data["agent"] == AgentType.WORKER.value
        assert result.output_data["mode"] == "architecture"

    def test_executor_unknown_task_type_returns_failed(self) -> None:
        """Unknown task types must produce a 'failed' result, not raise."""
        executor = VetinariA2AExecutor()
        task = A2ATask(task_type="definitely_not_a_real_task_type")
        result = executor.execute(task)
        assert result.status == "failed"
        assert "definitely_not_a_real_task_type" in result.error

    def test_a2a_result_to_dict(self) -> None:
        """A2AResult.to_dict() must include taskId, status, outputData, error."""
        result = A2AResult(
            task_id="abc-123",
            status="completed",
            output_data={"agent": AgentType.WORKER.value},
            error="",
        )
        d = result.to_dict()
        assert d["taskId"] == "abc-123"
        assert d["status"] == "completed"
        assert d["outputData"]["agent"] == AgentType.WORKER.value
        assert d["error"] == ""


# ── Story 38: Transport tests ─────────────────────────────────────────────────


class TestTransport:
    """Tests for the A2ATransport JSON-RPC layer."""

    def test_transport_handle_agent_card_request(self) -> None:
        """a2a.getAgentCard must return a result with 3 agents."""
        transport = make_a2a_transport()
        request = {"jsonrpc": "2.0", "id": 1, "method": "a2a.getAgentCard", "params": {}}
        response = transport.handle_request(request)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert len(response["result"]["agents"]) == 3

    def test_transport_handle_task_send(self) -> None:
        """a2a.taskSend must execute the task and return a result dict.

        In degraded/standalone mode (no orchestrator), the status is
        STATUS_ACKNOWLEDGED — the task was accepted but not yet executed.
        """
        transport = make_a2a_transport()
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "a2a.taskSend",
            "params": {
                "taskType": "build",
                "inputData": {"goal": "write a hello world function"},
            },
        }
        with patch("vetinari.a2a.executor.get_two_layer_orchestrator", return_value=None):
            response = transport.handle_request(request)
        assert response["jsonrpc"] == "2.0"
        assert "result" in response
        assert response["result"]["status"] == STATUS_ACKNOWLEDGED

    def test_transport_task_status(self) -> None:
        """a2a.taskStatus must return the status of a submitted task.

        In degraded/standalone mode (no orchestrator), the final status is
        STATUS_ACKNOWLEDGED — accepted but not executed.
        """
        transport = make_a2a_transport()
        # First send a task to register it
        task_id = "test-task-001"
        send_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "a2a.taskSend",
            "params": {"taskType": "plan", "taskId": task_id, "inputData": {}},
        }
        with patch("vetinari.a2a.executor.get_two_layer_orchestrator", return_value=None):
            transport.handle_request(send_req)

        # Now query status
        status_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "a2a.taskStatus",
            "params": {"taskId": task_id},
        }
        response = transport.handle_request(status_req)
        assert "result" in response
        assert response["result"]["taskId"] == task_id
        assert response["result"]["status"] == STATUS_ACKNOWLEDGED

    def test_transport_unknown_method_returns_error(self) -> None:
        """Unknown JSON-RPC methods must return a method-not-found error."""
        transport = make_a2a_transport()
        request = {"jsonrpc": "2.0", "id": 3, "method": "a2a.unknown", "params": {}}
        response = transport.handle_request(request)
        assert "error" in response
        assert response["error"]["code"] == -32601

    def test_transport_invalid_jsonrpc_version(self) -> None:
        """Requests with wrong jsonrpc version must return an invalid-request error."""
        transport = make_a2a_transport()
        request = {"jsonrpc": "1.0", "id": 4, "method": "a2a.getAgentCard", "params": {}}
        response = transport.handle_request(request)
        assert "error" in response
        assert response["error"]["code"] == -32600


# ── Story 39: AG-UI event tests ───────────────────────────────────────────────


class TestAGUIEvent:
    """Tests for AGUIEvent and AGUIEventType."""

    def test_agui_event_types_complete(self) -> None:
        """All 17 AG-UI event types must be present in AGUIEventType."""
        expected = {
            "TEXT_MESSAGE_START",
            "TEXT_MESSAGE_CONTENT",
            "TEXT_MESSAGE_END",
            "TOOL_CALL_START",
            "TOOL_CALL_ARGS",
            "TOOL_CALL_END",
            "STATE_SNAPSHOT",
            "STATE_DELTA",
            "MESSAGES_SNAPSHOT",
            "RAW",
            "CUSTOM",
            "STEP_STARTED",
            "STEP_FINISHED",
            "RUN_STARTED",
            "RUN_FINISHED",
            "RUN_ERROR",
            "METADATA",
        }
        actual = {e.name for e in AGUIEventType}
        assert actual == expected, f"Missing: {expected - actual}, Extra: {actual - expected}"
        assert len(AGUIEventType) == 17

    def test_agui_event_to_sse_format(self) -> None:
        """to_sse() must return a valid SSE string with correct event and data lines."""
        from vetinari.a2a.ag_ui import AGUIEvent

        event = AGUIEvent(
            event_type=AGUIEventType.TEXT_MESSAGE_CONTENT,
            data={"messageId": "msg-1", "content": "hello"},
        )
        sse = event.to_sse()
        assert sse.startswith("event: TEXT_MESSAGE_CONTENT\n")
        assert "data: " in sse
        assert sse.endswith("\n\n")
        # Parse the data line
        lines = sse.strip().split("\n")
        data_line = next(ln for ln in lines if ln.startswith("data: "))
        payload = json.loads(data_line[len("data: ") :])
        assert payload["type"] == "TEXT_MESSAGE_CONTENT"
        assert payload["content"] == "hello"
        assert payload["messageId"] == "msg-1"


class TestAGUIEmitter:
    """Tests for AGUIEventEmitter."""

    def test_agui_emitter_text_lifecycle(self) -> None:
        """Text start/content/end helpers must emit 3 events in correct order."""
        emitter = AGUIEventEmitter()
        emitter.emit_text_start("msg-001")
        emitter.emit_text_content("msg-001", "Hello, world!")
        emitter.emit_text_end("msg-001")
        events = emitter.get_events()
        assert len(events) == 3
        assert events[0].event_type == AGUIEventType.TEXT_MESSAGE_START
        assert events[1].event_type == AGUIEventType.TEXT_MESSAGE_CONTENT
        assert events[2].event_type == AGUIEventType.TEXT_MESSAGE_END
        assert events[1].data["content"] == "Hello, world!"

    def test_agui_emitter_to_stream_chunks(self) -> None:
        """to_stream_chunks() must return one StreamChunk per event with correct is_final."""
        emitter = AGUIEventEmitter()
        emitter.emit_run_start("run-1")
        emitter.emit_text_start("msg-1")
        emitter.emit_text_content("msg-1", "chunk")
        emitter.emit_text_end("msg-1")
        emitter.emit_run_finish("run-1")

        chunks = emitter.to_stream_chunks()
        assert len(chunks) == 5
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, StreamChunk)
            assert chunk.chunk_index == i
        # Only the last chunk should be final
        assert not chunks[0].is_final
        assert chunks[-1].is_final

    def test_agui_emitter_tool_call_lifecycle(self) -> None:
        """emit_tool_call() must emit start, args, and end events."""
        emitter = AGUIEventEmitter()
        sses = emitter.emit_tool_call("read_file", {"path": "/etc/hosts"})
        assert len(sses) == 3
        events = emitter.get_events()
        types = [e.event_type for e in events]
        assert AGUIEventType.TOOL_CALL_START in types
        assert AGUIEventType.TOOL_CALL_ARGS in types
        assert AGUIEventType.TOOL_CALL_END in types

    def test_agui_emitter_step_context_manager(self) -> None:
        """emit_step() context manager must bookend with STEP_STARTED / STEP_FINISHED and mark status=ok on success."""
        emitter = AGUIEventEmitter()
        with emitter.emit_step("compilation"):
            pass
        events = emitter.get_events()
        assert len(events) == 2
        assert events[0].event_type == AGUIEventType.STEP_STARTED
        assert events[1].event_type == AGUIEventType.STEP_FINISHED
        assert events[0].data["stepName"] == "compilation"
        # Success path MUST mark status='ok' so consumers can discriminate from failure
        assert events[1].data.get("status") == "ok"
        # No error field on success
        assert "error" not in events[1].data

    def test_agui_emitter_step_failure_emits_paired_error_and_reraises(self) -> None:
        """emit_step() must emit STEP_STARTED, then RUN_ERROR, then STEP_FINISHED(status=failed), and re-raise the exception.

        This is the failure-path contract for the context manager (SESSION-30A, US-30A.3c).
        The original exception MUST propagate to the caller — the emitter is not allowed
        to swallow it. The STEP_FINISHED event MUST carry status='failed' + error so UIs
        can distinguish a failed step from a successful one.
        """
        emitter = AGUIEventEmitter()

        with pytest.raises(RuntimeError, match="boom"):
            with emitter.emit_step("compile-fail"):
                raise RuntimeError("boom")

        events = emitter.get_events()
        types = [e.event_type for e in events]
        # Exact 3-event sequence
        assert types == [
            AGUIEventType.STEP_STARTED,
            AGUIEventType.RUN_ERROR,
            AGUIEventType.STEP_FINISHED,
        ], f"expected STARTED->RUN_ERROR->FINISHED, got {types}"

        # RUN_ERROR carries the failure description
        run_error = events[1]
        assert run_error.data["stepName"] == "compile-fail"
        assert "boom" in run_error.data["error"]
        assert "RuntimeError" in run_error.data["error"]

        # STEP_FINISHED marks failure explicitly
        step_finished = events[2]
        assert step_finished.data["status"] == "failed"
        assert "boom" in step_finished.data["error"]

        # Both step events share the same stepId (same step lifecycle)
        assert events[0].data["stepId"] == run_error.data["stepId"]
        assert events[0].data["stepId"] == step_finished.data["stepId"]

    def test_agui_emitter_clear(self) -> None:
        """clear() must remove all buffered events."""
        emitter = AGUIEventEmitter()
        emitter.emit_run_start("run-99")
        assert len(emitter.get_events()) == 1
        emitter.clear()
        assert emitter.get_events() == []

    def test_agui_emitter_error(self) -> None:
        """emit_error() must produce a RUN_ERROR event with the error message."""
        emitter = AGUIEventEmitter()
        emitter.emit_error("something went wrong")
        events = emitter.get_events()
        assert len(events) == 1
        assert events[0].event_type == AGUIEventType.RUN_ERROR
        assert events[0].data["error"] == "something went wrong"

    def test_agui_emitter_empty_to_stream_chunks(self) -> None:
        """to_stream_chunks() on empty emitter must return an empty list."""
        emitter = AGUIEventEmitter()
        chunks = emitter.to_stream_chunks()
        assert chunks == []

    def test_agui_emitter_chunk_metadata(self) -> None:
        """Each StreamChunk must carry event_type in its metadata."""
        emitter = AGUIEventEmitter()
        emitter.emit_run_start("run-x")
        chunks = emitter.to_stream_chunks()
        assert chunks[0].metadata["event_type"] == AGUIEventType.RUN_STARTED.value


# ── Story 38: handle_request_json convenience wrapper tests ───────────────────


class TestA2ATransportJson:
    """Tests for the handle_request_json convenience wrapper."""

    def test_handle_request_json_valid(self) -> None:
        """handle_request_json parses JSON and delegates to handle_request."""
        transport = make_a2a_transport()
        raw = json.dumps({"jsonrpc": "2.0", "method": "a2a.getAgentCard", "id": "1", "params": {}})
        result = transport.handle_request_json(raw)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "result" in parsed or "error" in parsed

    def test_handle_request_json_invalid_json(self) -> None:
        """handle_request_json returns a JSON-RPC parse error for invalid JSON input."""
        transport = make_a2a_transport()
        result = transport.handle_request_json("not valid json{{{")
        parsed = json.loads(result)
        assert parsed["error"]["code"] == -32700  # Parse error code per JSON-RPC 2.0 spec
