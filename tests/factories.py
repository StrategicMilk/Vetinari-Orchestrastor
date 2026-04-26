"""Test data factories for the Vetinari test suite.

Provides factory functions that create valid, fully-populated instances
of the core domain objects (Task, Plan, AgentResult, AgentSpec,
VerificationResult).  Use these instead of constructing dataclasses
inline with ad-hoc strings — they guarantee well-formed defaults and
allow selective overrides via keyword arguments.

Example::

    from tests.factories import make_task, make_plan

    task = make_task(description="Summarise the README")
    plan = make_plan(goal="Build a feature", tasks=[task])
"""

from __future__ import annotations

import random
import subprocess
import time
import types
import uuid
from datetime import datetime, timezone
from importlib.machinery import ModuleSpec
from typing import Any
from unittest.mock import MagicMock

from vetinari.adapters.base import ModelInfo, ProviderConfig, ProviderType
from vetinari.agents.contracts import AgentResult, AgentSpec, AgentTask, ExecutionPlan, Task, VerificationResult
from vetinari.types import AgentRxFailureCategory, AgentType, StatusEnum

# ── Reusable test constants ────────────────────────────────────────────────
TEST_PROJECT_ID: str = "test-project-001"
TEST_MODEL_ID: str = "test-model-7b-q4"
TEST_PLAN_ID: str = "plan_test0001"
TEST_TASK_ID: str = "task_test_001"

# Path-traversal test payloads for identifier confinement tests.
# Every route and persistence layer that accepts an ID parameter must reject
# all of these with a bounded client error (400) or ValueError.
TRAVERSAL_IDS: list[str] = [
    "..",
    "../outside",
    "..\\outside",
    "../../etc/passwd",
    "valid/../escape",
    "/absolute/path",
    "C:\\Windows",
    "normal\x00null",
]

# Subset of TRAVERSAL_IDS that resolve() confirms escape the storage directory
# via Path.is_relative_to().  Use these when testing filesystem-path confinement
# guards that rely on Path.resolve() + is_relative_to() — the others either stay
# inside the directory after resolution (".." → "...json") or are caught by the
# OS at write time rather than by our containment check.
ESCAPING_TRAVERSAL_IDS: list[str] = [
    "../outside",
    "..\\outside",
    "../../etc/passwd",
    "/absolute/path",
    "C:\\Windows",
]


def _unique_id(prefix: str = "test") -> str:
    """Generate a short unique ID for test objects."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ── Task factories ──────────────────────────────────────────────────────────


def make_task(
    *,
    id: str | None = None,
    description: str = "Test task: process the input and produce output",
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    dependencies: list[str] | None = None,
    assigned_agent: AgentType = AgentType.WORKER,
    status: StatusEnum = StatusEnum.PENDING,
    depth: int = 0,
    parent_id: str = "",
    model_override: str = "",
    metadata: dict[str, Any] | None = None,
) -> Task:
    """Create a well-formed Task with sensible defaults.

    All parameters are optional — pass only what differs from the default.

    Args:
        id: Task identifier (auto-generated if not provided).
        description: Human-readable task description.
        inputs: Input data references.
        outputs: Expected output references.
        dependencies: IDs of tasks this depends on.
        assigned_agent: Which agent type should execute this.
        status: Current task status.
        depth: Decomposition depth level.
        parent_id: ID of the parent task (for subtasks).
        model_override: Force a specific model for this task.
        metadata: Extra key-value pairs.

    Returns:
        A fully populated Task instance.
    """
    return Task(
        id=id or _unique_id("task"),
        description=description,
        inputs=inputs if inputs is not None else [],
        outputs=outputs if outputs is not None else [],
        dependencies=dependencies if dependencies is not None else [],
        assigned_agent=assigned_agent,
        status=status,
        depth=depth,
        parent_id=parent_id,
        model_override=model_override,
        metadata=metadata if metadata is not None else {},
    )


# ── Plan factories ──────────────────────────────────────────────────────────


def make_plan(
    *,
    plan_id: str | None = None,
    goal: str = "Test goal: achieve a well-defined outcome",
    phase: int = 0,
    tasks: list[Task] | None = None,
    version: str = "v0.1.0",
    notes: str = "",
    warnings: list[str] | None = None,
    needs_context: bool = False,
) -> ExecutionPlan:
    """Create a well-formed Plan with sensible defaults.

    Args:
        plan_id: Plan identifier (auto-generated if not provided).
        goal: What the plan is trying to accomplish.
        phase: Current execution phase.
        tasks: List of Task objects in this plan.
        version: Plan schema version.
        notes: Free-text notes.
        warnings: Plan-level warnings.
        needs_context: Whether the plan requires more context from the user.

    Returns:
        A fully populated Plan instance.
    """
    return ExecutionPlan(
        plan_id=plan_id or _unique_id("plan"),
        goal=goal,
        phase=phase,
        tasks=tasks if tasks is not None else [make_task()],
        version=version,
        notes=notes,
        warnings=warnings if warnings is not None else [],
        needs_context=needs_context,
    )


# ── AgentSpec factories ─────────────────────────────────────────────────────


def make_agent_spec(
    *,
    agent_type: AgentType = AgentType.WORKER,
    name: str = "test-worker",
    description: str = "Test worker agent for unit tests",
    default_model: str = TEST_MODEL_ID,
    enabled: bool = True,
    system_prompt: str = "You are a test agent.",
) -> AgentSpec:
    """Create a well-formed AgentSpec with sensible defaults.

    Args:
        agent_type: The type of agent.
        name: Human-readable agent name.
        description: What this agent does.
        default_model: Model to use for inference.
        enabled: Whether the agent is active.
        system_prompt: The system prompt template.

    Returns:
        A fully populated AgentSpec instance.
    """
    return AgentSpec(
        agent_type=agent_type,
        name=name,
        description=description,
        default_model=default_model,
        enabled=enabled,
        system_prompt=system_prompt,
    )


# ── AgentResult factories ──────────────────────────────────────────────────


def make_agent_result(
    *,
    success: bool = True,
    output: Any = "Test output content",
    metadata: dict[str, Any] | None = None,
    errors: list[str] | None = None,
) -> AgentResult:
    """Create a well-formed AgentResult with sensible defaults.

    Args:
        success: Whether the agent execution succeeded.
        output: The agent's output.
        metadata: Extra metadata from execution.
        errors: Error messages (if any).

    Returns:
        A fully populated AgentResult instance.
    """
    return AgentResult(
        success=success,
        output=output,
        metadata=metadata if metadata is not None else {},
        errors=errors if errors is not None else [],
    )


def make_failed_result(error_message: str = "Test error occurred") -> AgentResult:
    """Create a failed AgentResult for error-path testing.

    Args:
        error_message: The error message to include.

    Returns:
        An AgentResult with success=False and the given error.
    """
    return AgentResult(
        success=False,
        output="",
        metadata={},
        errors=[error_message],
    )


# ── VerificationResult factories ───────────────────────────────────────────


def make_verification_result(
    *,
    passed: bool = True,
    score: float = 0.85,
    issues: list[dict[str, Any]] | None = None,
    suggestions: list[str] | None = None,
) -> VerificationResult:
    """Create a well-formed VerificationResult with sensible defaults.

    Args:
        passed: Whether verification passed.
        score: Quality score between 0 and 1.
        issues: List of issue dicts.
        suggestions: Improvement suggestions.

    Returns:
        A fully populated VerificationResult instance.
    """
    return VerificationResult(
        passed=passed,
        score=score,
        issues=issues if issues is not None else [],
        suggestions=suggestions if suggestions is not None else [],
    )


# ── AgentTask factories ───────────────────────────────────────────────────


def make_agent_task(
    *,
    task_id: str | None = None,
    agent_type: AgentType = AgentType.WORKER,
    description: str = "Test task: process the input",
    prompt: str = "Do something",
    mode: str = "",
    context: dict[str, Any] | None = None,
) -> AgentTask:
    """Create a well-formed AgentTask with sensible defaults.

    Use this instead of defining per-file ``_make_task()`` helpers for
    ``AgentTask`` construction.

    Args:
        task_id: Task identifier (auto-generated if not provided).
        agent_type: Which agent type should execute this.
        description: Human-readable task description.
        prompt: The prompt text for the agent.
        mode: Execution mode hint.
        context: Extra context key-value pairs.

    Returns:
        A fully populated AgentTask instance.
    """
    return AgentTask(
        task_id=task_id or _unique_id("task"),
        agent_type=agent_type,
        description=description,
        prompt=prompt,
        mode=mode,
        context=context if context is not None else {},
    )


# ── Module stub factory ──────────────────────────────────────────────────


def make_stub_module(name: str, **attrs: Any) -> types.ModuleType:
    """Create a lightweight module stub for sys.modules patching.

    Use this instead of defining per-file ``_make_stub()`` helpers.
    The stub is a real ``types.ModuleType`` with optional attributes.

    Args:
        name: Fully-qualified module name (e.g. ``"vetinari.adapters"``).
        **attrs: Arbitrary attributes to set on the module.

    Returns:
        A module stub with the given name and attributes.
    """
    mod = types.ModuleType(name)
    mod.__spec__ = ModuleSpec(name, loader=None)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# ── Validation subsystem factories ──────────────────────────────────────────


def make_gate_check_result(
    *,
    gate_name: str = "test_gate",
    result: Any = None,
    score: float = 0.9,
) -> Any:
    """Create a GateCheckResult for testing quality gate logic.

    Args:
        gate_name: Name of the gate being checked.
        result: A GateResult enum value; defaults to GateResult.PASSED.
        score: The numeric gate score (0.0–1.0).

    Returns:
        A GateCheckResult instance.
    """
    from vetinari.validation.quality_gates import GateCheckResult, GateResult, VerificationMode

    if result is None:
        result = GateResult.PASSED
    return GateCheckResult(
        gate_name=gate_name,
        mode=VerificationMode.VERIFY_QUALITY,
        result=result,
        score=score,
    )


def make_validation_verification_result(
    status: Any = None,
    errors: int = 0,
    warnings: int = 0,
) -> Any:
    """Create a vetinari.validation.verification.VerificationResult for testing.

    Distinct from ``make_verification_result`` which creates the contracts
    VerificationResult used by agents.

    Args:
        status: A VerificationStatus enum value; defaults to VerificationStatus.PASSED.
        errors: Number of error-severity issues to attach.
        warnings: Number of warning-severity issues to attach.

    Returns:
        A VerificationResult instance populated with the requested issues.
    """
    from vetinari.validation.verification import ValidationVerificationResult, VerificationIssue, VerificationStatus

    if status is None:
        status = VerificationStatus.PASSED
    vr = ValidationVerificationResult(status=status, check_name="mock_check")
    vr.issues = [VerificationIssue(severity="error", category="test", message="err")] * errors + [
        VerificationIssue(severity="warning", category="test", message="warn")
    ] * warnings
    return vr


# ── Notification factories ────────────────────────────────────────────────


def make_notification(
    *,
    notification_id: str | None = None,
    title: str = "Test Notification",
    body: str = "Test notification body",
    priority: str = "high",
    action_type: str = "test",
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Create a test Notification instance.

    Args:
        notification_id: Unique ID. Auto-generated if omitted.
        title: Notification title.
        body: Notification body text.
        priority: Priority level string.
        action_type: Action type string.
        metadata: Optional metadata dict.

    Returns:
        A Notification instance.
    """
    from vetinari.notifications.manager import Notification, NotificationPriority

    priority_map = {
        "low": NotificationPriority.LOW,
        "medium": NotificationPriority.MEDIUM,
        "high": NotificationPriority.HIGH,
        "critical": NotificationPriority.CRITICAL,
    }
    return Notification(
        notification_id=notification_id or _unique_id("ntf"),
        title=title,
        body=body,
        priority=priority_map.get(priority, NotificationPriority.HIGH),
        action_type=action_type,
        metadata=metadata or {},
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ── MemoryEntry factories ────────────────────────────────────────────────────


def make_memory_entry(
    *,
    entry_id: str | None = None,
    agent: str = "worker",
    content: str = "test content",
    summary: str | None = None,
    timestamp: float = 0.0,
    provenance: str = "test",
    source_backends: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Create a well-formed MemoryEntry with sensible defaults.

    Args:
        entry_id: Unique memory entry ID (auto-generated if not provided).
        agent: Agent type string that produced this memory.
        content: Main memory content string.
        summary: Short summary (defaults to first 50 chars of content).
        timestamp: Unix timestamp of when this was recorded.
        provenance: Where this memory came from.
        source_backends: Which backends store this entry.
        metadata: Optional extra key-value pairs.

    Returns:
        A fully populated MemoryEntry instance.
    """
    from vetinari.memory.interfaces import MemoryEntry, MemoryType

    return MemoryEntry(
        id=entry_id or f"mem_{uuid.uuid4().hex[:8]}",
        agent=agent,
        entry_type=MemoryType.DISCOVERY,
        content=content,
        summary=(summary if summary is not None else content[:50]),
        timestamp=timestamp,
        provenance=provenance,
        source_backends=source_backends if source_backends is not None else ["test"],
        metadata=metadata,
    )


# ── Episode factories ────────────────────────────────────────────────────────


def make_episode(
    *,
    episode_id: str = "ep_test",
    timestamp: str = "2026-01-01T00:00:00",
    task_summary: str = "A task summary",
    agent_type: str | None = None,
    task_type: str = "coding",
    output_summary: str = "Output summary",
    quality_score: float = 0.9,
    success: bool = True,
    model_id: str = "test-model",
    embedding: list[float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Create a well-formed Episode dataclass with sensible defaults.

    Args:
        episode_id: Unique episode identifier.
        timestamp: ISO-format timestamp string.
        task_summary: Short description of what the task was.
        agent_type: Agent type string (defaults to AgentType.FOREMAN.value).
        task_type: Category of the task (e.g. "coding", "planning").
        output_summary: Short description of the produced output.
        quality_score: Quality score in [0, 1].
        success: Whether the episode completed successfully.
        model_id: Model used for this episode.
        embedding: Feature embedding vector (256-dim 0.1 vector if omitted).
        metadata: Optional extra key-value pairs.

    Returns:
        A fully populated Episode dataclass instance.
    """
    from vetinari.learning.episode_memory import MemoryRecordedEpisode

    return MemoryRecordedEpisode(
        episode_id=episode_id,
        timestamp=timestamp,
        task_summary=task_summary,
        agent_type=agent_type if agent_type is not None else AgentType.FOREMAN.value,
        task_type=task_type,
        output_summary=output_summary,
        quality_score=quality_score,
        success=success,
        model_id=model_id,
        embedding=embedding if embedding is not None else [0.1] * 256,
        metadata=metadata if metadata is not None else {},
    )


# ── Mock Episode factories ───────────────────────────────────────────────────


def make_mock_episode(
    *,
    task_summary: str = "Build a cache",
    output_summary: str = "Created RedisCacheWrapper",
    quality_score: float = 0.9,
    agent_type: str | None = None,
    model_id: str = "qwen-7b",
    task_type: str = "coding",
) -> MagicMock:
    """Create a mock Episode-like object for training pipeline tests.

    Returns a MagicMock with Episode interface attributes populated.
    Use this when the consumer only reads attributes (e.g. in
    ContextDistillationDatasetBuilder tests) and a real Episode dataclass
    is not needed.

    Args:
        task_summary: Short description of what the task was.
        output_summary: Short description of the produced output.
        quality_score: Quality score in [0, 1].
        agent_type: Agent type string (defaults to AgentType.WORKER.value).
        model_id: Model used for this episode.
        task_type: Category of the task.

    Returns:
        A MagicMock with Episode attribute access pre-configured.
    """
    ep = MagicMock()
    ep.task_summary = task_summary
    ep.output_summary = output_summary
    ep.quality_score = quality_score
    ep.agent_type = agent_type if agent_type is not None else AgentType.WORKER.value
    ep.model_id = model_id
    ep.task_type = task_type
    return ep


def make_mock_episodes(count: int, quality: float = 0.9) -> list[MagicMock]:
    """Create a list of mock episodes with sequential task/output summaries.

    Args:
        count: Number of mock episodes to generate.
        quality: Quality score applied to all episodes.

    Returns:
        A list of MagicMock episodes with ascending numeric summaries.
    """
    return [
        make_mock_episode(
            task_summary=f"Task {i}",
            output_summary=f"Output {i}",
            quality_score=quality,
        )
        for i in range(count)
    ]


# ── FailureRecord factories ──────────────────────────────────────────────────


def make_failure_record(
    *,
    category: AgentRxFailureCategory = AgentRxFailureCategory.PLAN_ADHERENCE_FAILURE,
    task_id: str = "task-1",
    agent_type: str | None = None,
    description: str = "test failure",
    timestamp: float | None = None,
    severity: str = "error",
) -> Any:
    """Create a well-formed FailureRecord with sensible defaults.

    Args:
        category: The AgentRx failure category enum value.
        task_id: ID of the task that failed.
        agent_type: Agent type string (defaults to AgentType.WORKER.value).
        description: Human-readable failure description.
        timestamp: Unix timestamp (defaults to current time).
        severity: Severity level string (e.g. "error", "warning").

    Returns:
        A fully populated FailureRecord instance.
    """
    from vetinari.analytics.failure_taxonomy import FailureRecord

    return FailureRecord(
        category=category,
        task_id=task_id,
        agent_type=agent_type if agent_type is not None else AgentType.WORKER.value,
        description=description,
        timestamp=timestamp if timestamp is not None else time.time(),
        severity=severity,
    )


# ── Plan ORM record factories ────────────────────────────────────────────────


def make_plan_record(
    *,
    plan_id: str = "plan_abc123",
    goal: str = "Test goal",
    status: str = "draft",
    risk_score: float = 0.1,
    dry_run: bool = False,
    auto_approved: bool = False,
    subtasks: list[Any] | None = None,
    created_at: str = "2025-01-01T00:00:00",
    updated_at: str = "2025-01-01T00:00:00",
) -> Any:
    """Create a Plan ORM record with sensible defaults.

    Populates the attributes expected by plan_api route handlers.  Uses a
    deferred import so the factory can be called after sys.modules stubs have
    been installed by the calling test module.

    Args:
        plan_id: Unique plan identifier.
        goal: Human-readable goal description.
        status: PlanStatus string value (e.g. "draft", "approved").
        risk_score: Numeric risk score (0 to 1).
        dry_run: Whether this is a dry-run plan.
        auto_approved: Whether the plan was auto-approved.
        subtasks: List of subtask objects attached to the plan.
        created_at: ISO-format creation timestamp string.
        updated_at: ISO-format last-updated timestamp string.

    Returns:
        A Plan instance with all required fields populated.
    """
    from vetinari.planning.plan_types import Plan, PlanRiskLevel
    from vetinari.types import PlanStatus

    plan = Plan()
    plan.plan_id = plan_id
    plan.plan_version = 1
    plan.goal = goal
    plan.constraints = ""
    plan.status = PlanStatus(status)
    plan.risk_score = risk_score
    plan.risk_level = PlanRiskLevel.LOW
    plan.dry_run = dry_run
    plan.auto_approved = auto_approved
    plan.approved_by = None
    plan.approved_at = None
    plan.subtasks = subtasks if subtasks is not None else []
    plan.dependencies = {}
    plan.plan_candidates = []
    plan.chosen_plan_id = None
    plan.plan_justification = ""
    plan.plan_explanation_json = ""
    plan.created_at = created_at
    plan.updated_at = updated_at
    plan.completed_at = None
    return plan


def make_subtask_record(
    *,
    subtask_id: str = "subtask_001",
    plan_id: str = "plan_abc123",
    description: str = "Test subtask",
) -> Any:
    """Create a Subtask ORM record with sensible defaults.

    Args:
        subtask_id: Unique subtask identifier.
        plan_id: ID of the owning plan.
        description: Human-readable subtask description.

    Returns:
        A Subtask instance with all required fields populated.
    """
    from vetinari.planning.plan_types import Subtask, TaskDomain
    from vetinari.types import StatusEnum

    st = Subtask()
    st.subtask_id = subtask_id
    st.plan_id = plan_id
    st.description = description
    st.domain = TaskDomain.CODING
    st.status = StatusEnum.PENDING
    st.subtask_explanation_json = ""
    st.definition_of_done = MagicMock(criteria=[])
    st.definition_of_ready = MagicMock(prerequisites=[])
    st.to_dict = lambda: {
        "subtask_id": st.subtask_id,
        "plan_id": st.plan_id,
        "description": st.description,
        "domain": st.domain.value,
        "status": st.status.value,
    }
    return st


# ── subprocess.CompletedProcess mock factory ─────────────────────────────────


def make_proc_result(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> MagicMock:
    """Create a mock subprocess.CompletedProcess for sandbox and shell tests.

    Returns a MagicMock with the CompletedProcess interface so tests can
    exercise code that inspects stdout, stderr, and returncode without
    spawning a real subprocess.

    Args:
        stdout: Simulated standard output text.
        stderr: Simulated standard error text.
        returncode: Simulated process return code.

    Returns:
        A MagicMock(spec=subprocess.CompletedProcess) with attributes set.
    """
    mock = MagicMock(spec=subprocess.CompletedProcess)
    mock.stdout = stdout
    mock.stderr = stderr
    mock.returncode = returncode
    return mock


# ── ConversationMessage factory ───────────────────────────────────────────────


def make_conversation_message(
    role: str = "user",
    content: str = "Hello",
    timestamp: float | None = None,
) -> Any:
    """Create a ConversationMessage for async_support tests.

    Args:
        role: Speaker role (e.g. ``"user"``, ``"assistant"``).
        content: Message content string.
        timestamp: Unix timestamp (defaults to current time).

    Returns:
        A ConversationMessage dataclass instance.
    """
    from vetinari.async_support.conversation import ConversationMessage

    return ConversationMessage(
        role=role,
        content=content,
        timestamp=timestamp if timestamp is not None else time.time(),
    )


# ── Drift detection helpers ───────────────────────────────────────────────────


def make_drift_vector(value: float, dim: int = 10) -> list[float]:
    """Create a uniform feature vector for drift detection tests.

    Args:
        value: The scalar value to repeat across all dimensions.
        dim: Number of dimensions in the vector.

    Returns:
        A list of ``dim`` floats all equal to ``value``.
    """
    return [value] * dim


def make_point_cluster(
    n: int,
    center: list[float],
    spread: float,
    rng: random.Random | None = None,
) -> list[list[float]]:
    """Generate n data points sampled uniformly within spread of a center.

    Used to create inlier/outlier clusters for isolation-forest tests.

    Args:
        n: Number of points to generate.
        center: Center coordinates of the cluster.
        spread: Half-width of the uniform sampling range around each center.
        rng: Random number generator (creates a default one if not provided).

    Returns:
        A list of n points, each a list of floats the same length as center.
    """
    if rng is None:
        rng = random.Random()
    return [[c + rng.uniform(-spread, spread) for c in center] for _ in range(n)]


# ── InferenceRequest factory ──────────────────────────────────────────────────


def make_inference_request(
    *,
    model_id: str = "test-model-7b",
    prompt: str = "What is the answer?",
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> Any:
    """Create a minimal InferenceRequest for adapter-level tests.

    Args:
        model_id: Model identifier to embed in the request.
        prompt: The user prompt text.
        system_prompt: The system prompt text.
        max_tokens: Maximum tokens for the response.
        temperature: Sampling temperature.

    Returns:
        A populated InferenceRequest with the given parameters.
    """
    from vetinari.adapters.base import InferenceRequest

    return InferenceRequest(
        model_id=model_id,
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ── ASGI scope factory ────────────────────────────────────────────────────────


def make_asgi_scope(
    method: str = "GET",
    path: str = "/api/v1/health",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> dict[str, Any]:
    """Build a minimal ASGI HTTP scope dict for middleware testing.

    Args:
        method: HTTP method string (e.g. ``"GET"``, ``"POST"``).
        path: Request path string.
        headers: List of raw header tuples. Defaults to empty list.

    Returns:
        A minimal ASGI scope dict with type, method, path, headers, and client.
    """
    return {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers if headers is not None else [],
        "client": ("127.0.0.1", 9999),
    }


# ── Replay mock request/response factories ────────────────────────────────────


def make_mock_inference_request(
    prompt: str = "hello",
    model_id: str = "test-7b",
) -> MagicMock:
    """Create a MagicMock satisfying the InferenceRequest protocol for replay tests.

    Args:
        prompt: The prompt text to assign to the mock.
        model_id: The model identifier to assign to the mock.

    Returns:
        A MagicMock with prompt, model_id, and metadata attributes populated.
    """
    req = MagicMock()
    req.prompt = prompt
    req.model_id = model_id
    req.metadata = {"agent": AgentType.WORKER.value}
    return req


def make_mock_inference_response(
    output: str = "world",
    model_id: str = "test-7b",
    tokens_used: int = 10,
) -> MagicMock:
    """Create a MagicMock satisfying the InferenceResponse protocol for replay tests.

    Args:
        output: The output text to assign to the mock.
        model_id: The model identifier to assign to the mock.
        tokens_used: Token count to assign to the mock.

    Returns:
        A MagicMock with output, model_id, and tokens_used attributes populated.
    """
    resp = MagicMock()
    resp.output = output
    resp.model_id = model_id
    resp.tokens_used = tokens_used
    return resp


# ── Intake mock factory ───────────────────────────────────────────────────────


def make_intake_mock(tier: Any = None) -> MagicMock:
    """Build a mock RequestIntake that returns the given intake tier.

    Args:
        tier: The Tier value to return from classify_with_features. Defaults
            to ``Tier.STANDARD`` when not provided.

    Returns:
        A MagicMock whose classify_with_features returns ``(tier, features)``.
    """
    from vetinari.orchestration.intake import IntakeFeatures, Tier

    resolved_tier = tier if tier is not None else Tier.STANDARD
    features = IntakeFeatures(
        word_count=5,
        confidence=0.9,
        has_ambiguous_words=False,
        cross_cutting_keywords=0,
        domain_novelty_score=0.0,
    )
    mock = MagicMock()
    mock.classify_with_features.return_value = (resolved_tier, features)
    return mock


# ── A2A protocol factories ───────────────────────────────────────────────────


def make_a2a_transport(*, host: str = "localhost", port: int = 8000) -> Any:
    """Create an A2ATransport instance for transport-layer tests.

    Args:
        host: Hostname for the transport endpoint.
        port: Port number for the transport endpoint.

    Returns:
        A fresh A2ATransport instance.
    """
    from vetinari.a2a.transport import A2ATransport

    return A2ATransport(host=host, port=port)


# ── Analytics / anomaly factories ────────────────────────────────────────────


def make_ensemble_anomaly_detector(*, agent_type: str = "TEST_AGENT") -> Any:
    """Create an EnsembleAnomalyDetector for anomaly detection tests.

    Args:
        agent_type: Agent type string to associate with the detector.

    Returns:
        An EnsembleAnomalyDetector instance.
    """
    from vetinari.analytics.anomaly import EnsembleAnomalyDetector

    return EnsembleAnomalyDetector(agent_type=agent_type)


# ── Workflow / Nelson rules factories ─────────────────────────────────────────


def make_nelson_rule_detector(
    *,
    mean: float = 50.0,
    sigma: float = 5.0,
    window_size: int = 50,
) -> Any:
    """Create a NelsonRuleDetector pre-seeded with stable baseline statistics.

    Args:
        mean: Baseline process mean.
        sigma: Baseline standard deviation.
        window_size: Maximum history window.

    Returns:
        A NelsonRuleDetector instance.
    """
    from vetinari.workflow.nelson_rules import NelsonRuleDetector

    return NelsonRuleDetector(mean=mean, sigma=sigma, window_size=window_size)


# -- Quality score / subprocess mock factories ---------------------------


def make_mock_quality_score(overall_score=0.8):
    """Create a mock quality score object with an ``overall_score`` attribute.

    Args:
        overall_score: The numeric score to expose on the mock.

    Returns:
        A ``MagicMock`` with ``overall_score`` set to the given value.
    """
    score = MagicMock()
    score.overall_score = overall_score
    return score


def make_subprocess_result_mock(stdout="[]", returncode=0):
    """Create a mock subprocess.run result for scanner and tool tests.

    Args:
        stdout: Captured standard output string.
        returncode: Process return code (0 = success).

    Returns:
        A ``MagicMock`` with ``stdout`` and ``returncode`` set.
    """
    m = MagicMock()
    m.stdout = stdout
    m.returncode = returncode
    return m


# ── Approval queue factories ──────────────────────────────────────────────────


def make_pending_action(
    *,
    action_id: str = "act_001",
    action_type: str = "model_substitution",
    details: dict[str, Any] | None = None,
    confidence: float = 0.85,
    status: str = "pending",
    created_at: str = "2026-01-01T00:00:00Z",
) -> Any:
    """Create a PendingAction for approval queue tests.

    Args:
        action_id: Unique identifier for this approval request.
        action_type: The kind of action (e.g. ``"model_substitution"``).
        details: JSON-serializable metadata about the action. Defaults to ``{}``.
        confidence: Agent's confidence score (0.0–1.0).
        status: Current status string (e.g. ``"pending"``).
        created_at: ISO 8601 UTC timestamp string.

    Returns:
        A populated PendingAction dataclass instance.
    """
    from vetinari.autonomy.approval_queue import PendingAction

    return PendingAction(
        action_id=action_id,
        action_type=action_type,
        details=details if details is not None else {},
        confidence=confidence,
        status=status,
        created_at=created_at,
    )


# ── Agent tree spec factory ───────────────────────────────────────────────────


def make_tree_agent_spec(
    agent_type: AgentType = AgentType.WORKER,
    *,
    parent_agent_id: str = "",
    agent_instance_id: str = "",
) -> AgentSpec:
    """Create an AgentSpec for agent-tree registration tests.

    Unlike ``make_agent_spec``, this factory reads the canonical defaults
    from ``AGENT_REGISTRY`` so the spec is valid for registration calls that
    check registry membership.  It also exposes ``parent_agent_id`` and
    ``agent_instance_id``, which are required by the tree API.

    Args:
        agent_type: The agent type to look up in AGENT_REGISTRY.
        parent_agent_id: Instance ID of the parent agent (empty = root node).
        agent_instance_id: Pre-assigned instance ID (empty = assigned on registration).

    Returns:
        An AgentSpec populated from AGENT_REGISTRY with the given tree fields.
    """
    # Build spec from defaults rather than AGENT_REGISTRY lookup — the registry
    # is mutable global state that other tests may clear, causing KeyError.
    _DEFAULTS = {
        AgentType.FOREMAN: ("Foreman", "Plans and delegates tasks"),
        AgentType.WORKER: ("Worker", "Executes assigned tasks"),
        AgentType.INSPECTOR: ("Inspector", "Verifies task output quality"),
    }
    name, description = _DEFAULTS.get(agent_type, (agent_type.value.title(), f"{agent_type.value} agent"))
    return AgentSpec(
        agent_type=agent_type,
        name=name,
        description=description,
        default_model="",
        parent_agent_id=parent_agent_id,
        agent_instance_id=agent_instance_id,
    )


# ── LLM output content factories ─────────────────────────────────────────────


def make_mock_llm_output(task_description: str) -> str:
    """Return a deterministic mock LLM output based on the task description.

    Generates plausible-looking Python code so that heuristic quality scoring
    produces non-trivial scores (length, keywords, structure).  Use this in
    end-to-end pipeline tests that need non-empty LLM output without a real
    model.

    Args:
        task_description: Human-readable task description string.

    Returns:
        A multi-line Python code string containing the task description.
    """
    snippet = task_description[:60]
    return (
        f'def solve():\n    """Solve: {snippet}."""\n'
        f'    result = "implemented"\n    return result\n\n'
        f'assert solve() == "implemented"  # basic verification\n'
    )


def make_decision_log_entry(
    *,
    action_id: str = "dec_001",
    action_type: str = "model_substitution",
    autonomy_level: str = "supervised",
    decision: str = "approve",
    confidence: float = 0.9,
    outcome: str = "success",
    timestamp: str = "2026-01-01T00:00:00Z",
) -> Any:
    """Create a DecisionLogEntry for audit log tests.

    Args:
        action_id: Unique identifier for the action.
        action_type: The kind of action.
        autonomy_level: The autonomy level applied.
        decision: The permission decision (e.g. ``"approve"``, ``"deny"``).
        confidence: Agent confidence score.
        outcome: Result of the action execution.
        timestamp: ISO 8601 UTC timestamp string.

    Returns:
        A populated DecisionLogEntry dataclass instance.
    """
    from vetinari.autonomy.approval_queue import DecisionLogEntry

    return DecisionLogEntry(
        action_id=action_id,
        action_type=action_type,
        autonomy_level=autonomy_level,
        decision=decision,
        confidence=confidence,
        outcome=outcome,
        timestamp=timestamp,
    )


def make_promotion_suggestion(
    *,
    action_type: str = "model_select",
    success_rate: float = 0.95,
    total_actions: int = 20,
) -> Any:
    """Create a PromotionSuggestion for autonomy governor tests.

    Args:
        action_type: The action type eligible for promotion.
        success_rate: Historical success rate (0.0–1.0).
        total_actions: Total number of past executions used for scoring.

    Returns:
        A populated PromotionSuggestion dataclass instance.
    """
    from vetinari.autonomy.governor import PromotionSuggestion
    from vetinari.types import AutonomyLevel

    return PromotionSuggestion(
        action_type=action_type,
        current_level=AutonomyLevel.L2_ACT_REPORT,
        suggested_level=AutonomyLevel.L3_ACT_LOG,
        success_rate=success_rate,
        total_actions=total_actions,
    )


# ── Adapter / provider factories ─────────────────────────────────────────────


def make_provider_config(
    *,
    provider_type: ProviderType = ProviderType.LOCAL,
    name: str = "test",
    endpoint: str = "local",
    api_key: str | None = None,
    max_retries: int = 3,
    timeout_seconds: int = 120,
    memory_budget_gb: int = 0,
    extra_config: dict[str, str] | None = None,
) -> ProviderConfig:
    """Create a ProviderConfig with sensible defaults for tests.

    Args:
        provider_type: The provider type enum value.
        name: Human-readable name for the provider.
        endpoint: Endpoint URL or identifier.
        api_key: Optional API key; defaults to None.
        max_retries: Maximum retry attempts.
        timeout_seconds: Request timeout in seconds.
        memory_budget_gb: VRAM budget in gigabytes.
        extra_config: Additional string key/value config pairs.

    Returns:
        A ProviderConfig instance.
    """
    return ProviderConfig(
        name=name,
        provider_type=provider_type,
        endpoint=endpoint,
        api_key=api_key,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        memory_budget_gb=memory_budget_gb,
        extra_config=extra_config or {},
    )


def make_llama_cpp_config(
    *,
    models_dir: str = "./models",
    gpu_layers: int = -1,
    context_length: int = 8192,
    memory_budget_gb: int = 32,
) -> ProviderConfig:
    """Create a ProviderConfig for local llama-cpp-python inference.

    Args:
        models_dir: Path to the directory containing GGUF model files.
        gpu_layers: Number of layers to offload to GPU (-1 = all).
        context_length: Maximum context window size in tokens.
        memory_budget_gb: VRAM budget in gigabytes.

    Returns:
        A ProviderConfig with LOCAL provider type and llama-cpp extra config.
    """
    return ProviderConfig(
        name="test-local",
        provider_type=ProviderType.LOCAL,
        endpoint="local",
        memory_budget_gb=memory_budget_gb,
        extra_config={
            "models_dir": models_dir,
            "gpu_layers": str(gpu_layers),
            "context_length": str(context_length),
        },
    )


def make_model_info(
    *,
    model_id: str = "m1",
    name: str = "Model 1",
    provider: str = "test",
    endpoint: str = "http://localhost:1234",
    capabilities: list[str] | None = None,
    context_len: int = 4096,
    memory_gb: int = 4,
    version: str = "1.0",
    latency_estimate_ms: int = 1000,
    cost_per_1k_tokens: float = 0.0,
    free_tier: bool = False,
) -> ModelInfo:
    """Create a ModelInfo instance for adapter and registry tests.

    Args:
        model_id: Unique model identifier.
        name: Human-readable model name.
        provider: Provider name string.
        endpoint: Model endpoint URL.
        capabilities: List of capability strings (defaults to ["general"]).
        context_len: Context window length in tokens.
        memory_gb: Memory requirement in gigabytes (integer).
        version: Model version string.
        latency_estimate_ms: Estimated inference latency in milliseconds.
        cost_per_1k_tokens: Cost per 1000 tokens (0 for free/local models).
        free_tier: Whether the model is on a free tier.

    Returns:
        A ModelInfo instance.
    """
    return ModelInfo(
        id=model_id,
        name=name,
        provider=provider,
        endpoint=endpoint,
        capabilities=capabilities if capabilities is not None else ["general"],
        context_len=context_len,
        memory_gb=memory_gb,
        version=version,
        latency_estimate_ms=latency_estimate_ms,
        cost_per_1k_tokens=cost_per_1k_tokens,
        free_tier=free_tier,
    )


def make_model_dict(
    model_id: str,
    capabilities: list[str],
    *,
    context_window: int = 16384,
    has_vision: bool = False,
    latency_hint: str = "medium",
    privacy_level: str = "",
) -> dict[str, Any]:
    """Create a plain model dict for agent affinity and routing tests.

    This is distinct from ``make_model_info`` which creates a ``ModelInfo``
    dataclass.  Some components (e.g. agent affinity scoring) work with plain
    dicts rather than the typed dataclass.

    Args:
        model_id: Unique model identifier.
        capabilities: List of capability strings.
        context_window: Context window size in tokens.
        has_vision: Whether the model has vision capability.
        latency_hint: Qualitative latency hint (e.g. "fast", "medium", "slow").
        privacy_level: Privacy classification string.

    Returns:
        A plain dict with the standard model keys (uses ``"model_id"`` key,
        consistent with the ``pick_model_for_agent`` lookup convention).
        When ``has_vision`` is True, ``"vision"`` is appended to capabilities.
    """
    caps = list(capabilities)
    if has_vision and "vision" not in caps:
        caps.append("vision")
    return {
        "model_id": model_id,
        "capabilities": caps,
        "context_window": context_window,
        "latency_hint": latency_hint,
        "privacy_level": privacy_level,
    }


# -- Dashboard / alert factories -------------------------------------------------


def make_alert_threshold(
    *,
    name="test-alert",
    metric_key="adapters.average_latency_ms",
    condition=None,
    threshold_value=500.0,
    severity=None,
    channels=None,
    duration_seconds=0,
):
    """Create an AlertThreshold with sensible defaults for use in tests.

    Args:
        name: Human-readable alert rule name.
        metric_key: Dot-path key into the system snapshot dict.
        condition: AlertCondition enum value; defaults to GREATER_THAN.
        threshold_value: Numeric value the metric is compared against.
        severity: AlertSeverity enum value; defaults to MEDIUM.
        channels: List of delivery channel names; defaults to ["log"].
        duration_seconds: Minimum seconds the condition must hold before firing.

    Returns:
        An AlertThreshold instance ready for assertion.
    """
    from vetinari.dashboard.alerts import AlertCondition, AlertSeverity, AlertThreshold

    return AlertThreshold(
        name=name,
        metric_key=metric_key,
        condition=condition if condition is not None else AlertCondition.GREATER_THAN,
        threshold_value=threshold_value,
        severity=severity if severity is not None else AlertSeverity.MEDIUM,
        channels=channels if channels is not None else ["log"],
        duration_seconds=duration_seconds,
    )


def make_alert_record(
    *,
    threshold=None,
    current_value=999.0,
    trigger_time=None,
):
    """Create an AlertRecord with sensible defaults for use in tests.

    Args:
        threshold: AlertThreshold that fired; defaults to make_alert_threshold().
        current_value: The metric value that exceeded the threshold.
        trigger_time: Unix timestamp of the trigger; defaults to 1700000000.0.

    Returns:
        An AlertRecord instance ready for assertion.
    """
    from vetinari.dashboard.alerts import AlertRecord

    return AlertRecord(
        threshold=threshold if threshold is not None else make_alert_threshold(),
        current_value=current_value,
        trigger_time=trigger_time if trigger_time is not None else 1700000000.0,
    )


def make_system_snapshot(
    *,
    latency=100.0,
    approval_rate=95.0,
    risk=0.2,
) -> dict:
    """Create a system snapshot dict shaped like DashboardAPI.get_system_snapshot().

    Args:
        latency: Average adapter latency in milliseconds.
        approval_rate: Plan approval rate as a percentage.
        risk: Average risk score for plan decisions.

    Returns:
        A dict with the canonical snapshot structure used by the alert engine.
    """
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "uptime_ms": 1000.0,
        "adapters": {
            "total_providers": 1,
            "total_requests": 10,
            "total_successful": 9,
            "total_failed": 1,
            "average_latency_ms": latency,
            "total_tokens_used": 500,
            "providers": {},
        },
        "memory": {"backends": {}},
        "plan": {
            "total_decisions": 5,
            "approved": 4,
            "rejected": 1,
            "auto_approved": 2,
            "approval_rate": approval_rate,
            "average_risk_score": risk,
            "average_approval_time_ms": 120.0,
        },
    }


# -- Dashboard performance / trace factories ------------------------------------


def make_trace_detail(
    *,
    trace_id=None,
    start_time=None,
    end_time=None,
    duration_ms=50.0,
    status="success",
    spans=None,
    index=0,
):
    """Create a TraceDetail with sensible defaults for use in tests.

    Args:
        trace_id: Unique trace identifier; auto-generated from index if omitted.
        start_time: ISO-8601 start timestamp; defaults to current UTC time.
        end_time: ISO-8601 end timestamp; defaults to current UTC time.
        duration_ms: Total trace duration in milliseconds.
        status: Trace status string, e.g. ``"success"``.
        spans: List of span dicts; defaults to two spans derived from index.
        index: Integer used to generate deterministic IDs when not specified.

    Returns:
        A TraceDetail instance ready for assertion.
    """
    from datetime import datetime, timezone

    from vetinari.dashboard.api import TraceDetail

    now = datetime.now(timezone.utc).isoformat()
    resolved_id = trace_id if trace_id is not None else f"trace-test-{index:06d}"
    resolved_spans = (
        spans
        if spans is not None
        else [
            {"span_id": f"span-{index}-0", "operation": "root", "duration_ms": duration_ms},
            {"span_id": f"span-{index}-1", "operation": "child", "duration_ms": duration_ms / 5},
        ]
    )
    return TraceDetail(
        trace_id=resolved_id,
        start_time=start_time if start_time is not None else now,
        end_time=end_time if end_time is not None else now,
        duration_ms=duration_ms,
        status=status,
        spans=resolved_spans,
    )


def make_log_record(
    *,
    message=None,
    level="INFO",
    trace_id=None,
    span_id=None,
    logger_name="test.logger",
    extra=None,
    index=0,
):
    """Create a LogRecord with sensible defaults for use in tests.

    Args:
        message: Log message text; auto-generated from index if omitted.
        level: Log level string, e.g. ``"INFO"``.
        trace_id: Trace ID for distributed tracing correlation; auto-generated if omitted.
        span_id: Span ID within the trace; auto-generated if omitted.
        logger_name: Dotted logger name string.
        extra: Arbitrary extra metadata dict; defaults to ``{"index": index}``.
        index: Integer used to generate deterministic IDs when not specified.

    Returns:
        A LogRecord frozen dataclass instance ready for assertion.
    """
    from vetinari.dashboard.log_aggregator import LogRecord

    return LogRecord(
        message=message if message is not None else f"test-record-{index}",
        level=level,
        trace_id=trace_id if trace_id is not None else f"t-{index % 100}",
        span_id=span_id if span_id is not None else f"s-{index % 10}",
        logger_name=logger_name,
        extra=extra if extra is not None else {"index": index},
    )


# -- Notification / digest factories -------------------------------------------


def make_daily_digest(
    *,
    generated_at=None,
    overall_health="healthy",
    sections=None,
):
    """Create a DailyDigest with sensible defaults for use in tests.

    Args:
        generated_at: ISO-8601 generation timestamp; defaults to a fixed test date.
        overall_health: Health summary string, e.g. ``"healthy"``.
        sections: List of DigestSection objects; defaults to Tasks and Health sections.

    Returns:
        A DailyDigest instance ready for assertion.
    """
    from vetinari.notifications.digest import DailyDigest, DigestSection

    resolved_sections = (
        sections
        if sections is not None
        else [
            DigestSection(title="Tasks", items=["3 done", "0 failed"]),
            DigestSection(title="Health", items=["OK"], metrics={"status": "healthy"}),
        ]
    )
    return DailyDigest(
        generated_at=generated_at if generated_at is not None else "2026-04-01T00:00:00+00:00",
        sections=resolved_sections,
        overall_health=overall_health,
    )


# ── Dynamic model router factories ───────────────────────────────────────────


def make_router_model_info(
    model_id: str = "llama-3-8b",
    name: str = "Llama 3 8B",
    code_gen: bool = False,
    reasoning: bool = False,
    chat: bool = True,
    creative: bool = False,
    docs: bool = False,
    math: bool = False,
    analysis: bool = False,
    summarization: bool = False,
    context_length: int = 4096,
    memory_gb: float = 4.0,
    provider: Any = None,
    is_available: bool = True,
) -> Any:
    """Create a ModelInfo for the dynamic model router with sensible defaults.

    Uses the vetinari.models.dynamic_model_router ModelInfo, not the
    adapter-layer ModelInfo from vetinari.adapters.base.

    Args:
        model_id: Unique model identifier.
        name: Human-readable model name.
        code_gen: Whether the model supports code generation.
        reasoning: Whether the model supports reasoning tasks.
        chat: Whether the model supports chat (default True).
        creative: Whether the model supports creative writing.
        docs: Whether the model supports documentation tasks.
        math: Whether the model supports maths reasoning.
        analysis: Whether the model supports analysis tasks.
        summarization: Whether the model supports summarization.
        context_length: Maximum context length in tokens.
        memory_gb: VRAM/RAM required in gigabytes.
        provider: ModelProvider enum value (defaults to LOCAL).
        is_available: Whether the model is currently available.

    Returns:
        A fully populated ModelInfo instance from dynamic_model_router.
    """
    from vetinari.models.dynamic_model_router import ModelCapabilities, ModelInfo
    from vetinari.types import ModelProvider

    caps = ModelCapabilities(
        code_gen=code_gen,
        reasoning=reasoning,
        chat=chat,
        creative=creative,
        docs=docs,
        math=math,
        analysis=analysis,
        summarization=summarization,
        context_length=context_length,
    )
    return ModelInfo(
        id=model_id,
        name=name,
        provider=provider if provider is not None else ModelProvider.LOCAL,
        capabilities=caps,
        context_length=context_length,
        memory_gb=memory_gb,
        is_available=is_available,
    )


# ── Model discovery candidate factory ────────────────────────────────────────


def make_model_candidate(
    *,
    model_id: str | None = None,
    name: str = "TestModel-7B",
    source_type: str = "huggingface",
    memory_gb: int = 4,
    final_score: float = 0.75,
    short_rationale: str = "Test candidate",
) -> Any:
    """Create a ModelCandidate for model discovery and scout tests.

    Args:
        model_id: Unique candidate ID (defaults to a slug derived from name).
        name: Human-readable model name.
        source_type: Discovery source (e.g. huggingface, reddit).
        memory_gb: Estimated VRAM/RAM required in gigabytes.
        final_score: Pre-computed quality score in [0, 1].
        short_rationale: Brief explanation for the score.

    Returns:
        A ModelCandidate dataclass instance.
    """
    from vetinari.model_discovery import ModelCandidate

    resolved_id = model_id if model_id is not None else name.lower().replace(" ", "-")
    return ModelCandidate(
        id=resolved_id,
        name=name,
        source_type=source_type,
        memory_gb=memory_gb,
        final_score=final_score,
        short_rationale=short_rationale,
    )


# ── Adapter-layer mock model info factory ────────────────────────────────────


def make_mock_adapter_model_info(
    model_id: str = "test-model-7b",
    memory_gb: int = 4,
    context_len: int = 4096,
    capabilities: list[str] | None = None,
) -> MagicMock:
    """Create a MagicMock matching LlamaCppProviderAdapter.discover_models() output.

    Provides the same attribute interface as the real adapter ModelInfo for
    unit tests that patch the adapter layer.

    Args:
        model_id: Unique model identifier (also used as the name).
        memory_gb: Estimated memory requirement in gigabytes.
        context_len: Maximum context length in tokens.
        capabilities: List of capability strings (defaults to ["general"]).

    Returns:
        A MagicMock with .id, .name, .memory_gb, .endpoint, .capabilities,
        and .context_len attributes set.
    """
    if capabilities is None:
        capabilities = ["general"]
    m = MagicMock()
    m.id = model_id
    m.name = model_id
    m.memory_gb = memory_gb
    m.endpoint = f"/models/{model_id}.gguf"
    m.capabilities = capabilities
    m.context_len = context_len
    return m


# ── Hardware profile factory ──────────────────────────────────────────────────


def make_hardware_profile(
    *,
    vram_gb: float = 0.0,
    ram_gb: float = 16.0,
    cpu_count: int = 8,
    vendor: Any = None,
) -> Any:
    """Create a HardwareProfile for model recommender and hardware detection tests.

    Accepts a GpuVendor enum value or a plain string (nvidia, apple, amd, none).
    When vram_gb is 0 or vendor resolves to NONE, no GPU is attached.

    Args:
        vram_gb: GPU VRAM in gigabytes (0 means CPU-only).
        ram_gb: System RAM in gigabytes.
        cpu_count: Number of logical CPU cores.
        vendor: GpuVendor enum or string (defaults to NONE when vram_gb == 0,
            else NVIDIA).

    Returns:
        A HardwareProfile dataclass instance.
    """
    from vetinari.system.hardware_detect import GpuInfo, GpuVendor, HardwareProfile

    if vendor is None:
        resolved_vendor = GpuVendor.NONE if vram_gb == 0.0 else GpuVendor.NVIDIA
    elif isinstance(vendor, GpuVendor):
        resolved_vendor = vendor
    else:
        vendor_map = {
            "nvidia": GpuVendor.NVIDIA,
            "apple": GpuVendor.APPLE,
            "amd": GpuVendor.AMD,
            "none": GpuVendor.NONE,
        }
        resolved_vendor = vendor_map.get(str(vendor).lower(), GpuVendor.NONE)

    if resolved_vendor == GpuVendor.NONE:
        gpu = GpuInfo()
    else:
        gpu = GpuInfo(
            name=f"Test GPU {vram_gb}GB",
            vendor=resolved_vendor,
            vram_gb=vram_gb,
            cuda_available=resolved_vendor == GpuVendor.NVIDIA,
            metal_available=resolved_vendor == GpuVendor.APPLE,
        )
    return HardwareProfile(cpu_count=cpu_count, ram_gb=ram_gb, gpu=gpu)


# ── Blackboard entry factory ──────────────────────────────────────────────────


def make_blackboard_entry(
    *,
    entry_id: str = "bb_test0001",
    content: str = "do something",
    request_type: str = "code_search",
    requested_by: str | None = None,
    priority: int = 5,
    ttl_seconds: float = 3600.0,
    state: Any = None,
) -> Any:
    """Create a BlackboardEntry with sensible defaults for blackboard tests.

    Args:
        entry_id: Unique entry identifier.
        content: The task or request content.
        request_type: The type of request (e.g. ``"code_search"``).
        requested_by: Agent type string; defaults to ``AgentType.WORKER.value``.
        priority: Entry priority (1 = highest, 10 = lowest).
        ttl_seconds: Time-to-live in seconds.
        state: ``EntryState`` enum value; defaults to ``EntryState.PENDING``.

    Returns:
        A fully populated ``BlackboardEntry`` instance.
    """
    from vetinari.memory.blackboard import BlackboardEntry, EntryState

    return BlackboardEntry(
        entry_id=entry_id,
        content=content,
        request_type=request_type,
        requested_by=requested_by if requested_by is not None else AgentType.WORKER.value,
        priority=priority,
        ttl_seconds=ttl_seconds,
        state=state if state is not None else EntryState.PENDING,
    )


# ── DurableExecutionEngine / ExecutionGraph factories ─────────────────────────


def make_execution_graph(
    *,
    plan_id: str = "plan-1",
    goal: str = "test goal",
) -> Any:
    """Create a fresh ``ExecutionGraph`` with no tasks.

    Args:
        plan_id: Unique plan identifier for the graph.
        goal: Human-readable goal description.

    Returns:
        An empty ``ExecutionGraph`` instance.
    """
    from vetinari.orchestration.execution_graph import ExecutionGraph

    return ExecutionGraph(plan_id=plan_id, goal=goal)


def make_execution_task_node(
    *,
    task_id: str = "t1",
    description: str = "Test task",
    max_retries: int = 3,
    depends_on: list[str] | None = None,
) -> Any:
    """Create an ``ExecutionTaskNode`` for durable execution tests.

    Args:
        task_id: Unique task identifier.
        description: Human-readable task description.
        max_retries: Maximum retry attempts on failure.
        depends_on: List of task IDs this node depends on.

    Returns:
        An ``ExecutionTaskNode`` instance.
    """
    from vetinari.orchestration.execution_graph import ExecutionTaskNode

    return ExecutionTaskNode(
        id=task_id,
        description=description,
        max_retries=max_retries,
        depends_on=depends_on if depends_on is not None else [],
    )


def make_durable_engine(
    checkpoint_dir: str,
    *,
    max_concurrent: int = 2,
    default_timeout: float = 10.0,
) -> Any:
    """Create a ``DurableExecutionEngine`` for integration tests.

    The caller is responsible for providing a temporary ``checkpoint_dir``
    and cleaning it up after the test.

    Args:
        checkpoint_dir: Path to the checkpoint storage directory.
        max_concurrent: Maximum number of concurrently executing tasks.
        default_timeout: Per-task timeout in seconds.

    Returns:
        A ``DurableExecutionEngine`` instance.
    """
    from vetinari.orchestration.durable_execution import DurableExecutionEngine

    return DurableExecutionEngine(
        checkpoint_dir=checkpoint_dir,
        max_concurrent=max_concurrent,
        default_timeout=default_timeout,
    )


# ── GoalVerifier mock result factories ────────────────────────────────────────


def make_evaluator_mock_result(
    *,
    verdict: str = "pass",
    quality_score: float = 0.8,
    feature_checks: list[Any] | None = None,
    improvements: list[Any] | None = None,
    model_used: str = "test-model",
) -> MagicMock:
    """Create a mock agent result shaped like GoalVerifier's LLM evaluator output.

    The returned mock has ``success=True`` and an ``output`` dict with the
    keys that ``GoalVerifier._llm_evaluation`` expects.

    Args:
        verdict: Evaluation verdict string (e.g. ``"pass"`` or ``"fail"``).
        quality_score: Quality score between 0.0 and 1.0.
        feature_checks: List of feature check dicts; defaults to empty list.
        improvements: List of improvement suggestion strings; defaults to empty.
        model_used: Model identifier string in the result output.

    Returns:
        A ``MagicMock`` with the expected ``success`` and ``output`` attributes.
    """
    result = MagicMock()
    result.success = True
    result.output = {
        "verdict": verdict,
        "quality_score": quality_score,
        "feature_checks": feature_checks if feature_checks is not None else [],
        "improvements": improvements if improvements is not None else [],
        "model_used": model_used,
    }
    return result


def make_security_mock_result(
    *,
    findings: list[Any] | None = None,
    score: int = 100,
) -> MagicMock:
    """Create a mock agent result shaped like GoalVerifier's security check output.

    The returned mock has ``success=True`` and an ``output`` dict with the
    keys that ``GoalVerifier._security_check`` expects.

    Args:
        findings: List of security finding dicts; defaults to empty list.
        score: Security score integer (0–100).

    Returns:
        A ``MagicMock`` with the expected ``success`` and ``output`` attributes.
    """
    result = MagicMock()
    result.success = True
    result.output = {
        "findings": findings if findings is not None else [],
        "score": score,
    }
    return result
