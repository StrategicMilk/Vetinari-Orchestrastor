"""Regression tests for SHARD-02 agent / training / release receipt emission.

Covers the three emission helpers (``record_agent_completion``,
``record_training_step``, ``record_release_step``) plus the parametrized
negative cases the shard requires:
- worker awaiting_user is suppressed (only Foreman/Inspector may block users);
- fallback training runs do NOT produce TRAINING_STEP receipts;
- failed agent execution produces an ``UNSUPPORTED`` outcome.

Helpers are tested directly to avoid the cost of standing up real Foreman /
Worker / Inspector instances; the ``complete_task`` integration is verified
in ``test_complete_task_emits_receipt`` through a stub agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from vetinari.agents.contracts import AgentResult, AgentTask
from vetinari.events import EventBus
from vetinari.receipts import (
    ReceiptAppended,
    WorkReceipt,
    WorkReceiptKind,
    WorkReceiptStore,
    record_agent_completion,
    record_release_step,
    record_training_step,
)
from vetinari.types import AgentType, EvidenceBasis


@pytest.fixture
def isolated_bus() -> EventBus:
    """Per-test EventBus so subscribers don't bleed across tests."""
    return EventBus()


@pytest.fixture
def store(tmp_path: Path, isolated_bus: EventBus) -> WorkReceiptStore:
    """Per-test receipts store rooted in tmp_path."""
    return WorkReceiptStore(repo_root=tmp_path, event_bus=isolated_bus)


def _make_stub_agent(agent_type: AgentType, name: str = "stub-agent-001") -> Any:
    """Build a minimal agent stub for record_agent_completion."""
    agent = MagicMock()
    agent.agent_type = agent_type
    agent.name = name
    return agent


def _make_task(
    *,
    task_id: str = "task-001",
    description: str = "ruff check vetinari/",
    project_id: str = "proj-receipts",
) -> AgentTask:
    return AgentTask(
        task_id=task_id,
        agent_type=AgentType.WORKER,
        description=description,
        prompt="run ruff",
        context={"project_id": project_id},
    )


def _make_result(
    *,
    success: bool = True,
    output: str = "ruff check passed; 0 errors",
    metadata: dict[str, Any] | None = None,
    errors: list[str] | None = None,
) -> AgentResult:
    return AgentResult(
        success=success,
        output=output,
        metadata=metadata or {},
        errors=errors or [],
    )


# -- record_agent_completion: kind mapping ------------------------------------


class TestAgentCompletionKindMapping:
    """AgentType -> WorkReceiptKind mapping coverage."""

    @pytest.mark.parametrize(
        ("agent_type", "expected_kind"),
        [
            (AgentType.FOREMAN, WorkReceiptKind.PLAN_ROUND),
            (AgentType.WORKER, WorkReceiptKind.WORKER_TASK),
            (AgentType.INSPECTOR, WorkReceiptKind.INSPECTOR_PASS),
        ],
    )
    def test_each_agent_type_maps_to_correct_kind(
        self,
        agent_type: AgentType,
        expected_kind: WorkReceiptKind,
        store: WorkReceiptStore,
    ) -> None:
        agent = _make_stub_agent(agent_type)
        receipt = record_agent_completion(
            agent=agent,
            task=_make_task(),
            result=_make_result(),
            score=0.92,
            store=store,
        )
        assert receipt is not None
        assert receipt.kind is expected_kind
        assert receipt.agent_type is agent_type


# -- record_agent_completion: outcome basis -----------------------------------


class TestAgentCompletionOutcome:
    """Outcome basis reflects success/failure correctly."""

    def test_success_uses_llm_judgment_basis(self, store: WorkReceiptStore) -> None:
        receipt = record_agent_completion(
            agent=_make_stub_agent(AgentType.WORKER),
            task=_make_task(),
            result=_make_result(success=True),
            score=0.85,
            store=store,
        )
        assert receipt is not None
        assert receipt.outcome.passed is True
        assert receipt.outcome.basis is EvidenceBasis.LLM_JUDGMENT
        assert receipt.outcome.score == 0.85

    def test_failure_uses_unsupported_basis(self, store: WorkReceiptStore) -> None:
        receipt = record_agent_completion(
            agent=_make_stub_agent(AgentType.WORKER),
            task=_make_task(),
            result=_make_result(success=False, errors=["ruff exited 1"]),
            score=0.0,
            store=store,
        )
        assert receipt is not None
        assert receipt.outcome.passed is False
        assert receipt.outcome.basis is EvidenceBasis.UNSUPPORTED
        assert receipt.outcome.score == 0.0
        assert "ruff exited 1" in receipt.outcome.issues

    def test_success_with_scoring_unavailable_uses_unsupported_basis(
        self,
        store: WorkReceiptStore,
    ) -> None:
        """Successful work + crashed scorer must NOT look like score=0.0 verdict.

        Regression: when the quality scorer crashes during ``complete_task``
        the receipt path used to emit ``passed=True, score=0.0,
        basis=LLM_JUDGMENT`` — indistinguishable from "scored zero because
        the work was bad". The fix routes ``scoring_available=False``
        through to ``OutcomeSignal.basis=UNSUPPORTED`` so consumers can
        distinguish a missing verdict from a real one.
        """
        receipt = record_agent_completion(
            agent=_make_stub_agent(AgentType.WORKER),
            task=_make_task(),
            result=_make_result(success=True),
            score=0.0,
            scoring_available=False,
            store=store,
        )
        assert receipt is not None
        assert receipt.outcome.passed is True
        assert receipt.outcome.basis is EvidenceBasis.UNSUPPORTED
        assert receipt.outcome.score == 0.0
        assert any("quality scoring unavailable" in issue for issue in receipt.outcome.issues)

    def test_inspector_unsupported_claims_set_awaiting(
        self,
        store: WorkReceiptStore,
    ) -> None:
        # Inspector / Foreman explicitly set awaiting metadata when they
        # surface a user-blocking condition.
        receipt = record_agent_completion(
            agent=_make_stub_agent(AgentType.INSPECTOR),
            task=_make_task(),
            result=_make_result(
                metadata={
                    "awaiting_user": True,
                    "awaiting_reason": "inspector surfaced unsupported claims: 2",
                },
            ),
            score=0.5,
            store=store,
        )
        assert receipt is not None
        assert receipt.awaiting_user is True
        assert receipt.awaiting_reason == "inspector surfaced unsupported claims: 2"


class TestWorkerCannotBlockUser:
    """Worker MUST NOT set awaiting_user (shard 02 task 2.2)."""

    def test_worker_awaiting_flag_is_suppressed(
        self,
        store: WorkReceiptStore,
    ) -> None:
        receipt = record_agent_completion(
            agent=_make_stub_agent(AgentType.WORKER),
            task=_make_task(),
            result=_make_result(
                metadata={
                    "awaiting_user": True,
                    "awaiting_reason": "worker should not be allowed to set this",
                },
            ),
            score=0.7,
            store=store,
        )
        assert receipt is not None
        assert receipt.awaiting_user is False
        assert receipt.awaiting_reason is None


class TestAgentCompletionLinkedClaims:
    """linked_claim_ids round-trips from result.metadata."""

    def test_linked_claim_ids_carried_through(self, store: WorkReceiptStore) -> None:
        receipt = record_agent_completion(
            agent=_make_stub_agent(AgentType.FOREMAN),
            task=_make_task(),
            result=_make_result(
                metadata={"linked_claim_ids": ("claim-A", "claim-B")},
            ),
            score=0.9,
            store=store,
        )
        assert receipt is not None
        assert receipt.linked_claim_ids == ("claim-A", "claim-B")


class TestEmissionFailureCounter:
    """Silent receipt emission failures must increment a metric counter.

    The emission helpers swallow exceptions so receipt subsystem faults
    never crash agent execution, but a silent failure leaves the
    Control Center blind. The ``vetinari.receipts.emission_failures``
    counter makes those silent failures observable in metrics.
    """

    def test_store_io_failure_increments_counter(
        self,
        store: WorkReceiptStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from vetinari.metrics import get_metrics

        # Force the store's append to raise. record_agent_completion
        # must catch it, log a warning, and increment the counter.
        def _boom(_receipt: WorkReceipt) -> None:
            raise OSError("simulated store failure")

        monkeypatch.setattr(WorkReceiptStore, "append", _boom)

        before = get_metrics().get_counter(
            "vetinari.receipts.emission_failures",
            kind="agent_completion",
        )

        receipt = record_agent_completion(
            agent=_make_stub_agent(AgentType.WORKER),
            task=_make_task(),
            result=_make_result(),
            score=0.9,
            store=store,
        )
        assert receipt is None

        after = get_metrics().get_counter(
            "vetinari.receipts.emission_failures",
            kind="agent_completion",
        )
        assert after == before + 1

    def test_unknown_agent_type_increments_counter(
        self,
        store: WorkReceiptStore,
    ) -> None:
        """Receipts with no kind mapping increment the counter too."""
        from unittest.mock import MagicMock

        from vetinari.metrics import get_metrics

        # Build an agent stub whose agent_type is not in _AGENT_KIND_MAP.
        # Sentinel object compares by identity so the dict lookup misses
        # without monkeypatching the AgentType enum itself.
        agent = MagicMock()
        agent.agent_type = object()
        agent.name = "weird-agent"

        before = get_metrics().get_counter(
            "vetinari.receipts.emission_failures",
            kind="unknown_agent_type",
        )

        receipt = record_agent_completion(
            agent=agent,
            task=_make_task(),
            result=_make_result(),
            score=0.9,
            store=store,
        )
        assert receipt is None

        after = get_metrics().get_counter(
            "vetinari.receipts.emission_failures",
            kind="unknown_agent_type",
        )
        assert after == before + 1


class TestAgentCompletionEventEmission:
    """Subscribers receive ``receipt.appended`` with the right kind."""

    def test_event_published_on_completion(
        self,
        store: WorkReceiptStore,
        isolated_bus: EventBus,
    ) -> None:
        seen: list[ReceiptAppended] = []
        isolated_bus.subscribe(ReceiptAppended, seen.append)

        receipt = record_agent_completion(
            agent=_make_stub_agent(AgentType.WORKER),
            task=_make_task(project_id="proj-events"),
            result=_make_result(),
            score=0.91,
            store=store,
        )

        assert len(seen) == 1
        event = seen[0]
        assert receipt is not None
        assert event.project_id == "proj-events"
        assert event.kind == WorkReceiptKind.WORKER_TASK.value
        assert event.receipt_id == receipt.receipt_id


# -- record_training_step ------------------------------------------------------


class TestTrainingStepReceipt:
    """Training receipt emission, including fallback skip."""

    def test_success_run_emits_receipt(self, store: WorkReceiptStore) -> None:
        receipt = record_training_step(
            project_id="qlora-v1",
            run_id="run_abc12345",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            algorithm="qlora",
            epochs=3,
            training_examples=5000,
            success=True,
            eval_score=0.873,
            store=store,
        )
        assert receipt is not None
        assert receipt.kind is WorkReceiptKind.TRAINING_STEP
        # Training pipeline is auxiliary, not a factory-pipeline agent;
        # AgentType.TRAINING labels it honestly per ADR-0103.
        assert receipt.agent_type is AgentType.TRAINING
        assert receipt.outcome.passed is True
        assert receipt.outcome.basis is EvidenceBasis.TOOL_EVIDENCE
        assert receipt.outcome.score == 0.873
        assert "algo=qlora" in receipt.inputs_summary
        assert "examples=5000" in receipt.inputs_summary

    def test_failed_run_emits_unsupported_receipt(self, store: WorkReceiptStore) -> None:
        receipt = record_training_step(
            project_id="qlora-v1",
            run_id="run_def67890",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            algorithm="qlora",
            epochs=3,
            training_examples=12,
            success=False,
            error="Insufficient training data (12 examples)",
            store=store,
        )
        assert receipt is not None
        assert receipt.outcome.passed is False
        assert receipt.outcome.basis is EvidenceBasis.UNSUPPORTED
        assert "Insufficient training data" in receipt.outcome.issues[0]

    def test_fallback_run_does_not_emit_receipt(self, store: WorkReceiptStore) -> None:
        receipt = record_training_step(
            project_id="qlora-v1",
            run_id="run_fallback_001",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            algorithm="qlora",
            epochs=0,
            training_examples=0,
            success=False,
            is_fallback=True,
            store=store,
        )
        # Anti-pattern Fallback as success — fallbacks must never inflate
        # training receipts.
        assert receipt is None

        # And the JSONL file is empty / non-existent.
        path = store.receipts_path("qlora-v1")
        assert not path.exists() or path.read_text(encoding="utf-8") == ""


# -- record_release_step -------------------------------------------------------


class TestReleaseStepReceipt:
    """Release receipt emission with linked claim IDs."""

    def test_success_emits_release_step(self, store: WorkReceiptStore) -> None:
        receipt = record_release_step(
            project_id="release",
            version="0.9.0",
            step_name="run_release_doctor",
            success=True,
            proof_path="outputs/release/0.9.0/proof.json",
            linked_claim_ids=("claim-build", "claim-smoke", "claim-sign"),
            store=store,
        )
        assert receipt is not None
        assert receipt.kind is WorkReceiptKind.RELEASE_STEP
        # Release doctor is auxiliary, not a factory-pipeline agent;
        # AgentType.RELEASE labels it honestly per ADR-0103.
        assert receipt.agent_type is AgentType.RELEASE
        assert receipt.outcome.passed is True
        assert receipt.outcome.basis is EvidenceBasis.TOOL_EVIDENCE
        assert receipt.linked_claim_ids == ("claim-build", "claim-smoke", "claim-sign")
        assert "step=run_release_doctor" in receipt.outputs_summary
        # Provenance source must be the actual module location, not the
        # imaginary vetinari.release.doctor.
        assert receipt.outcome.provenance is not None
        assert receipt.outcome.provenance.source == "scripts.release_doctor"

    def test_failure_emits_unsupported_release_step(self, store: WorkReceiptStore) -> None:
        receipt = record_release_step(
            project_id="release",
            version="0.9.0",
            step_name="run_release_doctor",
            success=False,
            error="doctor=1 smoke=0",
            store=store,
        )
        assert receipt is not None
        assert receipt.outcome.passed is False
        assert receipt.outcome.basis is EvidenceBasis.UNSUPPORTED
        assert "doctor=1 smoke=0" in receipt.outcome.issues[0]


# -- complete_task integration -------------------------------------------------


class TestCompleteTaskWiring:
    """End-to-end proof that complete_task emits a WorkReceipt.

    Exercises the real ``base_agent_completion.complete_task`` path with a
    stub agent + failing result so the early-return branch is covered.
    Source-string searches alone are forbidden (anti-pattern: source-string
    pseudo-tests) — this test asserts the runtime effect (a receipt JSONL
    file appears in tmp_path).
    """

    def test_complete_task_emits_receipt_on_failure_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from vetinari.agents import base_agent_completion
        from vetinari.receipts import store as store_module

        # Redirect every default-constructed WorkReceiptStore to tmp_path
        # so the worktree's outputs/receipts/ tree isn't touched. This is
        # the most robust patch point — works regardless of how the helper
        # imports its dependencies.
        monkeypatch.setattr(store_module, "_default_repo_root", lambda: tmp_path)

        agent = MagicMock()
        agent.agent_type = AgentType.WORKER
        agent.name = "Worker"
        agent._last_inference_model_id = ""
        agent.default_model = "stub-model"
        agent._adapter_manager = None
        agent.get_system_prompt = MagicMock(return_value="stub system prompt")

        task = AgentTask(
            task_id="task-int-001",
            agent_type=AgentType.WORKER,
            description="failing task",
            prompt="trigger failure",
            context={"project_id": "proj-int"},
        )
        result = AgentResult(
            success=False,
            output="",
            errors=["simulated failure"],
            metadata={},
        )

        base_agent_completion.complete_task(agent, task, result)

        path = tmp_path / "outputs" / "receipts" / "proj-int" / "receipts.jsonl"
        assert path.exists(), "complete_task did not emit a receipt for the failure path"

        verifying_store = WorkReceiptStore(repo_root=tmp_path)
        receipts = list(verifying_store.iter_receipts("proj-int"))
        assert len(receipts) == 1
        appended = receipts[0]
        assert appended.kind is WorkReceiptKind.WORKER_TASK
        assert appended.outcome.passed is False
        assert appended.outcome.basis is EvidenceBasis.UNSUPPORTED
        assert "simulated failure" in appended.outcome.issues


class TestTrainingPipelineWiring:
    """End-to-end proof that the training pipeline emits a TRAINING_STEP receipt.

    Drives ``TrainingPipeline.run`` through the early-return path (insufficient
    data) so the receipt-emission helper is exercised at runtime. The test
    asserts the JSONL file appears in tmp_path — a pure runtime effect, not
    a source-text grep (anti-pattern: source-string pseudo-tests).
    """

    def test_pipeline_early_exit_emits_training_step_receipt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from vetinari.receipts import store as store_module
        from vetinari.training import pipeline as pipeline_mod

        monkeypatch.setattr(store_module, "_default_repo_root", lambda: tmp_path)

        # Build a stub pipeline whose curator writes a tiny dataset (<10 rows)
        # so run() falls into the insufficient-data early-return path. We
        # replace _resolve_base_model + _resolve_model_revision so the run
        # doesn't try to talk to HuggingFace.
        small_dataset = tmp_path / "small.jsonl"
        small_dataset.write_text('{"prompt": "x", "response": "y"}\n', encoding="utf-8")

        captured_run_dir: dict[str, Path] = {}
        original_init = pipeline_mod.TrainingPipeline.__init__

        def _stub_init(self, *args, **kwargs) -> None:
            original_init(self, *args, **kwargs)
            self._curator = MagicMock()
            self._curator.curate = MagicMock(return_value=str(small_dataset))
            self._replay_buffer = MagicMock()
            self._regularizer = MagicMock()
            self._trainer = MagicMock()
            self._converter = MagicMock()
            self._deployer = MagicMock()
            self._adapter_manager = MagicMock()

        monkeypatch.setattr(pipeline_mod.TrainingPipeline, "__init__", _stub_init)
        monkeypatch.setattr(
            pipeline_mod.TrainingPipeline,
            "_resolve_base_model",
            lambda self, m: m,
        )
        monkeypatch.setattr(
            pipeline_mod.TrainingPipeline,
            "_resolve_model_revision",
            lambda self, base, rev: rev or "",
        )

        out_dir = tmp_path / "training_runs"
        pipeline = pipeline_mod.TrainingPipeline()
        run = pipeline.run(
            base_model="stub-model",
            task_type="proj-train-int",
            output_base_dir=str(out_dir),
            backend="vllm",
        )
        captured_run_dir["dir"] = out_dir / run.run_id

        # Pipeline should have early-exited on insufficient data and still
        # produced a receipt under the project-id matching task_type.
        assert run.success is False
        assert "Insufficient training data" in run.error

        receipts_path = tmp_path / "outputs" / "receipts" / "proj-train-int" / "receipts.jsonl"
        assert receipts_path.exists(), "TrainingPipeline early-exit did not emit a TRAINING_STEP receipt"
        verifying_store = WorkReceiptStore(repo_root=tmp_path)
        receipts = list(verifying_store.iter_receipts("proj-train-int"))
        assert len(receipts) == 1
        receipt = receipts[0]
        assert receipt.kind is WorkReceiptKind.TRAINING_STEP
        assert receipt.outcome.passed is False
        assert receipt.outcome.basis is EvidenceBasis.UNSUPPORTED


class TestReleaseDoctorWiring:
    """End-to-end proof that release_doctor.main emits a RELEASE_STEP receipt.

    Drives the script's ``main`` entry point with a stubbed
    ``run_release_doctor`` so the receipt-emission block at the end of
    ``main`` is exercised at runtime. Asserts the JSONL appears in tmp_path
    rather than grepping source (anti-pattern: source-string pseudo-tests).
    """

    def test_main_emits_release_step_receipt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import importlib.util

        # Load the script as a module from disk (it lives in scripts/, not in
        # the package). Once imported it's reachable via sys.modules.
        repo_root = Path(__file__).resolve().parent.parent
        script_path = repo_root / "scripts" / "release_doctor.py"
        spec = importlib.util.spec_from_file_location("_release_doctor_under_test", script_path)
        assert spec is not None and spec.loader is not None
        rd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rd)

        from vetinari.receipts import store as store_module

        monkeypatch.setattr(store_module, "_default_repo_root", lambda: tmp_path)

        # Stub run_release_doctor to return a passing ReleaseProof without
        # actually building a wheel. ReleaseProof and ReleaseClaimRecord are
        # the real types — we just construct an empty claims tuple.
        from vetinari.release.proof_schema import PROOF_SCHEMA_VERSION, ReleaseProof

        stub_proof = ReleaseProof(
            schema_version=PROOF_SCHEMA_VERSION,
            version="0.0.0-test",
            built_at="2026-04-25T00:00:00+00:00",
            python_version="3.12.0",
            host_os="test-os",
            wheel_sha256="0" * 64,
            wheel_size_bytes=1,
            doctor_exit_code=0,
            smoke_exit_code=0,
            smoke_latency_ms=1.0,
            signed=False,
            claims=(),
        )
        monkeypatch.setattr(rd, "run_release_doctor", lambda **kwargs: stub_proof)
        # Redirect _OUTPUTS_DIR so the proof_path the helper builds points
        # under tmp_path (the helper composes _OUTPUTS_DIR / version / proof.json).
        monkeypatch.setattr(rd, "_OUTPUTS_DIR", tmp_path / "outputs" / "release")

        exit_code = rd.main(["--version", "0.0.0-test", "--quiet"])
        assert exit_code == 0

        receipts_path = tmp_path / "outputs" / "receipts" / "release" / "receipts.jsonl"
        assert receipts_path.exists(), "release_doctor.main did not emit a RELEASE_STEP receipt"
        verifying_store = WorkReceiptStore(repo_root=tmp_path)
        receipts = list(verifying_store.iter_receipts("release"))
        assert len(receipts) == 1
        receipt = receipts[0]
        assert receipt.kind is WorkReceiptKind.RELEASE_STEP
        assert receipt.outcome.passed is True
        assert receipt.outcome.basis is EvidenceBasis.TOOL_EVIDENCE
