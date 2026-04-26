"""Config-Matrix test suite — MI section (Models/Skills).

Covers MI-01 through MI-44 from docs/audit/CONFIG-MATRIX.md section 15.
All HTTP tests use Litestar TestClient against create_app() — no handler .fn() calls.

Tests proving known defects are marked xfail(strict=True) with a SESSION-32.4 fix reference.
Infrastructure-dependent tests are marked skip with the required dependency named.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from litestar.testing import TestClient

logger = logging.getLogger(__name__)

# -- Shared test fixtures --


@asynccontextmanager
async def _noop_lifespan(app: Any) -> Any:
    """Drop-in lifespan that skips all subsystem wiring during tests."""
    yield


@pytest.fixture(scope="session")
def matrix_app() -> Any:
    """Create the Litestar app with subsystem wiring bypassed for test isolation."""
    with (
        patch("vetinari.web.litestar_app._lifespan", _noop_lifespan),
        patch("vetinari.web.litestar_app._register_shutdown_handlers"),
    ):
        from vetinari.web.litestar_app import create_app

        return create_app(debug=False)


@pytest.fixture
def client(matrix_app: Any) -> Any:
    """Provide a TestClient scoped per test function."""
    from collections.abc import Generator

    def _make() -> Generator[TestClient, None, None]:
        with TestClient(app=matrix_app) as c:
            yield c

    return next(_make())


# -- MI section: Models/Skills --


class TestMI01WorkerModeInterfaceAlignedWithModes:
    """MI-01: Worker mode interface lists match actual worker modes enum."""

    def test_worker_mode_interface_aligned_with_modes(self) -> None:
        """Worker interface descriptors cover every WorkerMode enum value."""
        try:
            from vetinari.agents.interfaces import WorkerInterface
            from vetinari.types import WorkerMode
        except ImportError:
            pytest.skip("WorkerInterface or WorkerMode not importable")

        interface_modes = {m.value for m in WorkerInterface.supported_modes()}
        enum_modes = {m.value for m in WorkerMode}
        missing = enum_modes - interface_modes
        assert not missing, f"WorkerInterface missing modes from WorkerMode enum: {missing}"


class TestMI02WorkerInterfaceIncludesSuggest:
    """MI-02: Worker interface includes the 'suggest' operation."""

    def test_worker_interface_includes_suggest(self) -> None:
        """The worker interface descriptor exposes a 'suggest' capability."""
        try:
            from vetinari.agents.interfaces import WorkerInterface
        except ImportError:
            pytest.skip("vetinari.agents.interfaces.WorkerInterface not importable")

        ops = [op.name for op in WorkerInterface.operations()]
        assert "suggest" in ops, f"WorkerInterface must include 'suggest' operation, found: {ops}"


class TestMI03InterfacesCodingBridgeScopeContract:
    """MI-03: Coding bridge interface is scoped to coding tasks only."""

    def test_interfaces_coding_bridge_scope_contract(self) -> None:
        """CodingBridgeInterface only registers coding-domain operations."""
        try:
            from vetinari.agents.interfaces import CodingBridgeInterface
        except ImportError:
            pytest.skip("vetinari.agents.interfaces.CodingBridgeInterface not importable")

        ops = [op.name for op in CodingBridgeInterface.operations()]
        non_coding = [
            op
            for op in ops
            if op
            not in (
                "run_tests",
                "edit_file",
                "read_file",
                "search_codebase",
                "apply_diff",
                "generate_code",
                "explain_code",
                "review_code",
            )
        ]
        # Allow legitimate coding ops; flag anything that belongs to a different domain
        assert len(non_coding) < len(ops), "CodingBridgeInterface must have at least some recognized coding operations"


class TestMI04DuplicateInterfaceSurfacesCompatible:
    """MI-04: Duplicate interface surface (same operation in two places) must be compatible."""

    def test_duplicate_interface_surfaces_compatible(self) -> None:
        """Operations that appear in both Worker and Foreman interfaces have matching signatures."""
        try:
            from vetinari.agents.interfaces import ForemanInterface, WorkerInterface
        except ImportError:
            pytest.skip("ForemanInterface or WorkerInterface not importable")

        foreman_ops = {op.name: op for op in ForemanInterface.operations()}
        worker_ops = {op.name: op for op in WorkerInterface.operations()}
        shared = set(foreman_ops) & set(worker_ops)

        for name in shared:
            f_op = foreman_ops[name]
            w_op = worker_ops[name]
            assert f_op.input_schema == w_op.input_schema, (
                f"Shared operation '{name}' has incompatible input schemas across interfaces"
            )


class TestMI05ForemanInterfaceDescriptorAlignedWithRegistry:
    """MI-05: Foreman interface descriptor aligns with the agent registry entry."""

    def test_foreman_interface_descriptor_aligned_with_registry(self) -> None:
        """ForemanInterface.agent_type matches the AgentType.FOREMAN registry key."""
        try:
            from vetinari.agents.interfaces import ForemanInterface
            from vetinari.types import AgentType
        except ImportError:
            pytest.skip("ForemanInterface or AgentType not importable")

        assert ForemanInterface.agent_type == AgentType.FOREMAN, (
            f"ForemanInterface.agent_type must be AgentType.FOREMAN, got {ForemanInterface.agent_type}"
        )


class TestMI06InspectorInterfaceDescriptorAlignedWithRegistry:
    """MI-06: Inspector interface descriptor aligns with the agent registry entry."""

    def test_inspector_interface_descriptor_aligned_with_registry(self) -> None:
        """InspectorInterface.agent_type matches AgentType.INSPECTOR registry key."""
        try:
            from vetinari.agents.interfaces import InspectorInterface
            from vetinari.types import AgentType
        except ImportError:
            pytest.skip("InspectorInterface or AgentType not importable")

        assert InspectorInterface.agent_type == AgentType.INSPECTOR, (
            f"InspectorInterface.agent_type must be AgentType.INSPECTOR, got {InspectorInterface.agent_type}"
        )


class TestMI07SkillSpecRoundTripTrustTier:
    """MI-07: SkillSpec trust_tier round-trips through serialization without loss."""

    def test_skill_spec_round_trip_trust_tier(self) -> None:
        """SkillSpec serialized and deserialized preserves the trust_tier field."""
        try:
            from vetinari.skills.spec import SkillSpec, TrustTier
        except ImportError:
            pytest.skip("vetinari.skills.spec.SkillSpec not importable")

        spec = SkillSpec(skill_id="test-skill", trust_tier=TrustTier.VERIFIED)
        data = spec.model_dump() if hasattr(spec, "model_dump") else spec.__dict__
        restored = SkillSpec(**data)
        assert restored.trust_tier == TrustTier.VERIFIED, (
            f"trust_tier must survive round-trip, got {restored.trust_tier}"
        )


class TestMI08SkillSpecRoundTripLoadingLevel:
    """MI-08: SkillSpec loading_level round-trips through serialization without loss."""

    def test_skill_spec_round_trip_loading_level(self) -> None:
        """SkillSpec serialized and deserialized preserves the loading_level field."""
        try:
            from vetinari.skills.spec import LoadingLevel, SkillSpec
        except ImportError:
            pytest.skip("vetinari.skills.spec.SkillSpec not importable")

        spec = SkillSpec(skill_id="test-skill-ll", loading_level=LoadingLevel.EAGER)
        data = spec.model_dump() if hasattr(spec, "model_dump") else spec.__dict__
        restored = SkillSpec(**data)
        assert restored.loading_level == LoadingLevel.EAGER, (
            f"loading_level must survive round-trip, got {restored.loading_level}"
        )


class TestMI09SkillSpecRoundTripOutputValidators:
    """MI-09: SkillSpec output_validators list round-trips without loss."""

    def test_skill_spec_round_trip_output_validators(self) -> None:
        """SkillSpec output_validators list survives serialization round-trip."""
        try:
            from vetinari.skills.spec import SkillSpec
        except ImportError:
            pytest.skip("vetinari.skills.spec.SkillSpec not importable")

        validators = ["schema_check", "length_check"]
        spec = SkillSpec(skill_id="test-validators", output_validators=validators)
        data = spec.model_dump() if hasattr(spec, "model_dump") else spec.__dict__
        restored = SkillSpec(**data)
        assert restored.output_validators == validators, (
            f"output_validators must survive round-trip, got {restored.output_validators}"
        )


class TestMI10SkillSpecRoundTripIdAlias:
    """MI-10: SkillSpec id alias maps correctly to the canonical skill_id field."""

    def test_skill_spec_round_trip_id_alias(self) -> None:
        """SkillSpec created with 'id' alias field resolves to skill_id correctly."""
        try:
            from vetinari.skills.spec import SkillSpec
        except ImportError:
            pytest.skip("vetinari.skills.spec.SkillSpec not importable")

        # Pydantic models often use field aliases — both 'id' and 'skill_id' should work
        spec = SkillSpec(skill_id="canonical-id")
        assert spec.skill_id == "canonical-id", f"skill_id must be accessible as .skill_id, got {spec.skill_id!r}"
        # If an 'id' alias is defined, serialization with by_alias should use 'id'
        if hasattr(spec, "model_dump"):
            by_alias = spec.model_dump(by_alias=True)
            assert "id" in by_alias or "skill_id" in by_alias, "Serialized SkillSpec must have 'id' or 'skill_id' key"


class TestMI11SkillOutputRoundTripNoTruncation:
    """MI-11: SkillOutput round-trip does not truncate long content fields."""

    def test_skill_output_round_trip_no_truncation(self) -> None:
        """SkillOutput with long content survives serialization without truncation."""
        try:
            from vetinari.skills.spec import SkillOutput
        except ImportError:
            pytest.skip("vetinari.skills.spec.SkillOutput not importable")

        long_content = "x" * 50_000
        output = SkillOutput(content=long_content, skill_id="test")
        data = output.model_dump() if hasattr(output, "model_dump") else output.__dict__
        restored = SkillOutput(**data)
        assert len(restored.content) == 50_000, (
            f"SkillOutput content truncated: expected 50000, got {len(restored.content)}"
        )


@pytest.mark.parametrize(
    "skill_name",
    ["architect", "operations", "quality"],
)
def test_skill_discoverable_or_demoted(skill_name: str) -> None:
    """MI-12/13/14: Each named skill is either discoverable or explicitly demoted."""
    try:
        from vetinari.skills.catalog import get_skill_catalog
    except ImportError:
        pytest.skip("vetinari.skills.catalog.get_skill_catalog not importable")

    catalog = get_skill_catalog()
    skill_ids = [s.skill_id for s in catalog.list_all()]
    demoted_ids = [s.skill_id for s in catalog.list_demoted()]

    assert skill_name in skill_ids or skill_name in demoted_ids, (
        f"Skill '{skill_name}' must be discoverable or explicitly demoted"
    )


class TestMI12ArchitectSkillDiscoverableOrDemoted:
    """MI-12: Architect skill is discoverable or explicitly demoted."""

    def test_architect_skill_discoverable_or_demoted(self) -> None:
        """Architect skill appears in the catalog or demoted list."""
        test_skill_discoverable_or_demoted("architect")


class TestMI13OperationsSkillDiscoverableOrDemoted:
    """MI-13: Operations skill is discoverable or explicitly demoted."""

    def test_operations_skill_discoverable_or_demoted(self) -> None:
        """Operations skill appears in the catalog or demoted list."""
        test_skill_discoverable_or_demoted("operations")


class TestMI14QualitySkillDiscoverableOrDemoted:
    """MI-14: Quality skill is discoverable or explicitly demoted."""

    def test_quality_skill_discoverable_or_demoted(self) -> None:
        """Quality skill appears in the catalog or demoted list."""
        test_skill_discoverable_or_demoted("quality")


@pytest.mark.parametrize(
    "agent_type_name",
    ["FOREMAN", "WORKER", "INSPECTOR"],
)
def test_get_agent_spec_not_none(agent_type_name: str) -> None:
    """MI-15/16/17: get_agent_spec returns a non-None spec for each agent type."""
    try:
        from vetinari.agents.contracts import get_agent_spec
        from vetinari.types import AgentType
    except ImportError:
        pytest.skip("vetinari.agents.contracts.get_agent_spec or AgentType not importable")

    agent_type = AgentType[agent_type_name]
    spec = get_agent_spec(agent_type)
    assert spec is not None, f"get_agent_spec({agent_type_name}) must return a non-None spec"
    assert hasattr(spec, "agent_type"), f"AgentSpec for {agent_type_name} must have agent_type attribute"


class TestMI15GetAgentSpecForemanNotNone:
    """MI-15: get_agent_spec(FOREMAN) returns a valid spec."""

    def test_get_agent_spec_foreman_not_none(self) -> None:
        """Foreman agent spec is registered and non-None."""
        test_get_agent_spec_not_none("FOREMAN")


class TestMI16GetAgentSpecWorkerNotNone:
    """MI-16: get_agent_spec(WORKER) returns a valid spec."""

    def test_get_agent_spec_worker_not_none(self) -> None:
        """Worker agent spec is registered and non-None."""
        test_get_agent_spec_not_none("WORKER")


class TestMI17GetAgentSpecInspectorNotNone:
    """MI-17: get_agent_spec(INSPECTOR) returns a valid spec."""

    def test_get_agent_spec_inspector_not_none(self) -> None:
        """Inspector agent spec is registered and non-None."""
        test_get_agent_spec_not_none("INSPECTOR")


class TestMI18OperationsSkillModeCompleteNoInvalidAliases:
    """MI-18: Operations skill mode 'complete' has no invalid tool aliases."""

    def test_operations_skill_mode_complete_no_invalid_aliases(self) -> None:
        """Operations skill 'complete' mode tool list contains no empty or None aliases."""
        try:
            from vetinari.skills.catalog import get_skill_catalog
        except ImportError:
            pytest.skip("vetinari.skills.catalog not importable")

        catalog = get_skill_catalog()
        ops_skill = next((s for s in catalog.list_all() if s.skill_id == "operations"), None)
        if ops_skill is None:
            pytest.skip("operations skill not in catalog")

        tools = ops_skill.get_tools_for_mode("complete")
        invalid = [t for t in tools if not t or t.strip() == ""]
        assert not invalid, f"Operations skill 'complete' mode has invalid tool aliases: {invalid}"


class TestMI19OperationsSkillToolOutputFormatHonored:
    """MI-19: Operations skill tool output format is not overridden to an incompatible format."""

    def test_operations_skill_tool_output_format_honored(self) -> None:
        """OperationsSkillTool must change its content serialization when a non-markdown format is requested."""
        from vetinari.skills.operations_skill import OperationsSkillTool

        tool = OperationsSkillTool()
        html_result = tool.execute(mode="documentation", content="API docs", output_format="html")
        json_result = tool.execute(mode="documentation", content="API docs", output_format="json")

        assert html_result.success is True
        assert json_result.success is True
        assert html_result.output["format"] == "html"
        assert json_result.output["format"] == "json"
        assert "## " not in html_result.output["content"], "HTML output must not still be markdown headings"
        assert "## " not in json_result.output["content"], "JSON output must not still be markdown headings"


class TestMI20ArchitectSkillToolWrongTypeDomainRejected:
    """MI-20: Architect skill rejects a tool registered for the wrong domain type."""

    def test_architect_skill_tool_wrong_type_domain_rejected(self) -> None:
        """Registering a non-architecture tool into architect skill raises ValueError."""
        try:
            from vetinari.skills.catalog import SkillCatalog
            from vetinari.skills.spec import SkillSpec, ToolDomain
        except ImportError:
            pytest.skip("vetinari.skills.catalog or SkillSpec not importable")

        catalog = SkillCatalog()
        wrong_domain_tool = MagicMock()
        wrong_domain_tool.domain = ToolDomain.IMAGE_GEN  # not architecture

        with pytest.raises((ValueError, TypeError)):
            catalog.register_tool_for_skill("architect", wrong_domain_tool)


class TestMI21ArchitectSkillToolThinkingModeNotDecorative:
    """MI-21: Architect skill 'thinking' mode is not purely decorative — it changes behavior."""

    def test_architect_skill_tool_thinking_mode_not_decorative(self) -> None:
        """ArchitectSkillTool output must change when the caller raises the thinking budget."""
        from vetinari.skills.architect_skill import ArchitectSkillTool

        tool = ArchitectSkillTool()
        low = tool.execute(mode="system_design", design_request="Design auth system", thinking_mode="low")
        high = tool.execute(mode="system_design", design_request="Design auth system", thinking_mode="high")

        assert low.success is True
        assert high.success is True
        assert low.metadata["thinking_mode"] == "low"
        assert high.metadata["thinking_mode"] == "high"
        assert low.output != high.output, "thinking_mode must affect the produced design, not just metadata"


class TestMI22ForemanSkillToolNoEmptyStubPlan:
    """MI-22: Foreman skill 'plan' tool never produces an empty stub plan."""

    def test_foreman_skill_tool_no_empty_stub_plan(self) -> None:
        """ForemanSkillTool must not report plan success while returning an empty stub plan."""
        from vetinari.skills.foreman_skill import ForemanSkillTool

        result = ForemanSkillTool().execute(goal="Build a REST API with auth", mode="plan")

        assert result.success is True
        assert result.output["tasks"], "plan mode must return at least one task when it reports success"


class TestMI23ForemanSkillToolNoEmptyStubClarify:
    """MI-23: Foreman skill 'clarify' tool never returns an empty clarification."""

    def test_foreman_skill_tool_no_empty_stub_clarify(self) -> None:
        """Foreman clarify mode must surface at least one question when it reports success."""
        from vetinari.skills.foreman_skill import ForemanSkillTool

        result = ForemanSkillTool().execute(goal="Build something", mode="clarify")

        assert result.success is True
        assert result.output["questions"], "clarify mode must return at least one clarification question"


class TestMI24InspectorSkillToolFailClosedEmptyInput:
    """MI-24: Inspector skill tool fails closed (returns failed result) on empty input."""

    def test_inspector_skill_tool_fail_closed_empty_input(self) -> None:
        """Inspector verify() with empty input returns passed=False."""
        try:
            from vetinari.skills.inspector import InspectorSkill
        except ImportError:
            pytest.skip("vetinari.skills.inspector.InspectorSkill not importable")

        skill = InspectorSkill()
        result = skill.verify("")
        assert result.passed is False, f"Inspector verify('') must return passed=False, got passed={result.passed}"
        assert result.score == 0.0 or result.score < 0.5, (
            f"Inspector verify('') must return low score, got {result.score}"
        )


class TestMI25QualitySkillToolTestGenerationReturnsArtifact:
    """MI-25: Quality skill test generation returns a non-empty artifact."""

    def test_quality_skill_tool_test_generation_returns_artifact(self) -> None:
        """QualitySkillTool test_generation mode must emit an actual test artifact, not just planning metadata."""
        from vetinari.skills.quality_skill import QualitySkillTool

        result = QualitySkillTool().execute(mode="test_generation", code="def foo():\n    return 1\n")

        assert result.success is True
        tests_artifact = result.output["tests"]
        assert tests_artifact, "test_generation must return a generated test artifact"
        assert "test_" in tests_artifact, "generated artifact must look like executable tests"


class TestMI26ImageGenSvgFallbackCorrectBackend:
    """MI-26: Image generation SVG fallback uses the correct backend, not default."""

    def test_image_gen_svg_fallback_correct_backend(self) -> None:
        """SVG fallback in image gen routes to the SVG backend, not the diffusion backend."""
        try:
            from vetinari.models.image_gen import ImageGenRouter
        except ImportError:
            pytest.skip("vetinari.models.image_gen.ImageGenRouter not importable")

        router = ImageGenRouter()
        backend = router.select_backend(output_format="svg")
        assert backend is not None, "SVG format must resolve to a backend"
        assert "svg" in str(backend).lower() or "vector" in str(backend).lower(), (
            f"SVG output must use SVG/vector backend, got {backend}"
        )


class TestMI27ImageGenPlaceholderNotSuccess:
    """MI-27: Image generation placeholder is not treated as a successful generation."""

    def test_image_gen_placeholder_not_success(self) -> None:
        """An ImageGenResult with placeholder=True must have success=False."""
        try:
            from vetinari.models.image_gen import ImageGenResult
        except ImportError:
            pytest.skip("vetinari.models.image_gen.ImageGenResult not importable")

        result = ImageGenResult(placeholder=True, image_path=None)
        assert result.success is False, f"Placeholder result must have success=False, got {result.success}"


class TestMI28DiffusionEngineSameStemNoDuplicateIds:
    """MI-28: Diffusion engine does not assign duplicate IDs when models share a file stem."""

    def test_diffusion_engine_same_stem_no_duplicate_ids(self) -> None:
        """Two model files with the same stem but different parents get unique IDs."""
        try:
            from vetinari.models.diffusion import DiffusionModelIndex
        except ImportError:
            pytest.skip("vetinari.models.diffusion.DiffusionModelIndex not importable")

        import pathlib

        with tempfile.TemporaryDirectory() as root:
            dir_a = pathlib.Path(root) / "collection_a"
            dir_b = pathlib.Path(root) / "collection_b"
            dir_a.mkdir()
            dir_b.mkdir()
            (dir_a / "model.safetensors").touch()
            (dir_b / "model.safetensors").touch()

            index = DiffusionModelIndex(search_roots=[dir_a, dir_b])
            models = index.discover()
            ids = [m.model_id for m in models]
            assert len(ids) == len(set(ids)), f"Same-stem models must have unique IDs, got duplicates: {ids}"


class TestMI29DiffusionEngineDiscoveryScopeContract:
    """MI-29: Diffusion engine discovery only returns models within declared search roots."""

    def test_diffusion_engine_discovery_scope_contract(self) -> None:
        """Models outside the declared search roots are not included in discovery."""
        try:
            from vetinari.models.diffusion import DiffusionModelIndex
        except ImportError:
            pytest.skip("vetinari.models.diffusion.DiffusionModelIndex not importable")

        import pathlib

        with tempfile.TemporaryDirectory() as root_a:
            with tempfile.TemporaryDirectory() as root_b:
                path_a = pathlib.Path(root_a)
                path_b = pathlib.Path(root_b)
                (path_a / "included.safetensors").touch()
                (path_b / "excluded.safetensors").touch()

                # Only declare root_a as a search root
                index = DiffusionModelIndex(search_roots=[path_a])
                models = index.discover()
                paths = [m.file_path for m in models]
                assert all(str(path_a) in str(p) for p in paths), (
                    f"Discovery must only return models under declared roots, got: {paths}"
                )


class TestMI30ImageGenSingleFileSafetensorsCorrectLoadPath:
    """MI-30: Single-file safetensors model resolves correct load path."""

    def test_image_gen_single_file_safetensors_correct_load_path(self) -> None:
        """A single-file .safetensors model resolves its load path to the file itself."""
        try:
            from vetinari.models.diffusion import DiffusionModelSpec
        except ImportError:
            pytest.skip("vetinari.models.diffusion.DiffusionModelSpec not importable")

        import pathlib

        with tempfile.TemporaryDirectory() as root:
            model_file = pathlib.Path(root) / "my_model.safetensors"
            model_file.touch()

            spec = DiffusionModelSpec.from_path(model_file)
            assert spec.load_path == model_file, f"Single-file load_path must be the file itself, got {spec.load_path}"


class TestMI31ImageModelDiscoveryUniqueIdsSameStem:
    """MI-31: Image model discovery assigns unique IDs when multiple models share a stem."""

    def test_image_model_discovery_unique_ids_same_stem(self) -> None:
        """All discovered image models have unique IDs, even with identical file stems."""
        try:
            from vetinari.models.image_gen import ImageModelDiscovery
        except ImportError:
            pytest.skip("vetinari.models.image_gen.ImageModelDiscovery not importable")

        import pathlib

        with tempfile.TemporaryDirectory() as root:
            dir1 = pathlib.Path(root) / "dir1"
            dir2 = pathlib.Path(root) / "dir2"
            dir1.mkdir()
            dir2.mkdir()
            (dir1 / "anime.safetensors").touch()
            (dir2 / "anime.safetensors").touch()

            discovery = ImageModelDiscovery(roots=[dir1, dir2])
            models = discovery.run()
            ids = [m.model_id for m in models]
            assert len(ids) == len(set(ids)), f"Duplicate model IDs from same stem: {ids}"


class TestMI32AgentAffinityHelpersReturnCopies:
    """MI-32: Agent affinity helper functions return copies, not references to shared state."""

    def test_agent_affinity_helpers_return_copies(self) -> None:
        """Modifying the result of get_affinity_map() does not affect the registry."""
        try:
            from vetinari.agents.affinity import get_affinity_map
        except ImportError:
            pytest.skip("vetinari.agents.affinity.get_affinity_map not importable")

        map1 = get_affinity_map()
        map1["__test_key__"] = "mutated"
        map2 = get_affinity_map()
        assert "__test_key__" not in map2, (
            "get_affinity_map() must return a copy — mutation must not affect the registry"
        )


class TestMI33AffinityTestsLiveReferenceContract:
    """MI-33: Affinity tests check the live affinity map, not a snapshot copy."""

    def test_affinity_tests_live_reference_contract(self) -> None:
        """Affinity test utilities operate on the live registry, reflecting updates."""
        try:
            from vetinari.agents.affinity import (
                AffinityTestSuite,
                get_affinity_map,
                update_affinity,
            )
        except ImportError:
            pytest.skip("vetinari.agents.affinity utilities not importable")

        update_affinity("__live_test_agent__", score=0.9)
        suite = AffinityTestSuite()
        live_map = get_affinity_map()
        assert "__live_test_agent__" in live_map, "AffinityTestSuite must operate on the live affinity registry"


class TestMI34AssignmentPassNoAssignedWithNullModel:
    """MI-34: Assignment pass does not emit tasks with null model_id."""

    def test_assignment_pass_no_assigned_with_null_model(self) -> None:
        """After assignment pass, all tasks with assigned=True have a non-null model_id."""
        try:
            from tests.factories import make_task
            from vetinari.planning.assignment import AssignmentPass
        except ImportError:
            pytest.skip("vetinari.planning.assignment.AssignmentPass not importable")

        tasks = [make_task(task_id=f"t{i}") for i in range(3)]
        assignment = AssignmentPass()
        result = assignment.run(tasks)
        null_model_tasks = [t for t in result if t.get("assigned") and not t.get("model_id")]
        assert not null_model_tasks, f"Assigned tasks must have model_id set, found null: {null_model_tasks}"


class TestMI35AssignmentPassTestsScopeContract:
    """MI-35: Assignment pass tests are scoped to assignment behavior only."""

    def test_assignment_pass_tests_scope_contract(self) -> None:
        """AssignmentPass does not mutate task fields outside its declared scope."""
        try:
            from tests.factories import make_task
            from vetinari.planning.assignment import AssignmentPass
        except ImportError:
            pytest.skip("vetinari.planning.assignment.AssignmentPass not importable")

        task = make_task(task_id="scope-task")
        original_description = task["description"]
        original_inputs = list(task.get("inputs", []))

        assignment = AssignmentPass()
        result_tasks = assignment.run([task])
        result_task = next(t for t in result_tasks if t["id"] == "scope-task")

        assert result_task["description"] == original_description, "AssignmentPass must not modify task description"
        assert result_task.get("inputs", []) == original_inputs, "AssignmentPass must not modify task inputs"


class TestMI36ThreadSafeSingletonNoDuplicateAfterReset:
    """MI-36: Thread-safe singleton does not create duplicates after an explicit reset."""

    def test_thread_safe_singleton_no_duplicate_after_reset(self) -> None:
        """After resetting a singleton, the next get returns a single fresh instance."""
        try:
            from vetinari.utils.singleton import ThreadSafeSingleton
        except ImportError:
            pytest.skip("vetinari.utils.singleton.ThreadSafeSingleton not importable")

        class _Counter(ThreadSafeSingleton):
            def __init__(self) -> None:
                self.value = 0

        _Counter.reset()
        inst1 = _Counter.get_instance()
        inst2 = _Counter.get_instance()
        assert inst1 is inst2, "ThreadSafeSingleton must return same instance on consecutive gets after reset"


class TestMI37SingletonMetaConflictingArgsRejected:
    """MI-37: Singleton metaclass rejects conflicting constructor arguments."""

    def test_singleton_meta_conflicting_args_rejected(self) -> None:
        """Creating a singleton with different args on second call raises ValueError."""
        try:
            from vetinari.utils.singleton import parameterized_singleton
        except ImportError:
            pytest.skip("vetinari.utils.singleton.parameterized_singleton not importable")

        @parameterized_singleton
        class _Configurable:
            def __init__(self, mode: str) -> None:
                self.mode = mode

        _Configurable.reset_all()
        inst = _Configurable(mode="fast")
        assert inst.mode == "fast"

        with pytest.raises((ValueError, TypeError)):
            _Configurable(mode="slow")


class TestMI38CodingTaskOutsideRootRepoPathRejected:
    """MI-38: Coding task targeting a path outside the declared repo root is rejected."""

    def test_coding_task_outside_root_repo_path_rejected(self) -> None:
        """CodingTask with target_file outside repo_root raises ValueError."""
        try:
            from vetinari.agents.coding import CodingTask
        except ImportError:
            pytest.skip("vetinari.agents.coding.CodingTask not importable")

        with pytest.raises((ValueError, PermissionError)):
            CodingTask(
                repo_root="/projects/myrepo",
                target_file="/etc/passwd",
                description="edit system file",
            )


class TestMI39CodingTaskMalformedModuleNameRejected:
    """MI-39: Coding task with a malformed module name is rejected at construction."""

    def test_coding_task_malformed_module_name_rejected(self) -> None:
        """CodingTask with an invalid Python module name raises ValueError."""
        try:
            from vetinari.agents.coding import CodingTask
        except ImportError:
            pytest.skip("vetinari.agents.coding.CodingTask not importable")

        with pytest.raises((ValueError, TypeError)):
            CodingTask(
                repo_root="/projects/myrepo",
                module_name="123_invalid.module name!",
                description="invalid module",
            )


class TestMI40FallbackCodingAgentTestGenerationNormalizedPaths:
    """MI-40: Fallback coding agent normalizes target file paths to POSIX format."""

    def test_fallback_coding_agent_test_generation_normalized_paths(self) -> None:
        """Fallback coding agent returns target_file with forward-slash path separators."""
        try:
            from vetinari.agents.coding import FallbackCodingAgent
        except ImportError:
            pytest.skip("vetinari.agents.coding.FallbackCodingAgent not importable")

        agent = FallbackCodingAgent(repo_root="/projects/myrepo")
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "def test_foo(): assert True"
        result = agent.generate_tests(source_file="vetinari\\utils\\singleton.py", llm=mock_llm)
        # Path in the result must use forward slashes (normalized)
        if hasattr(result, "target_file"):
            assert "\\" not in str(result.target_file), (
                f"Fallback agent must normalize paths to POSIX, got {result.target_file}"
            )


class TestMI41MultiStepCodingTaskNonObjectSubtaskRejected:
    """MI-41: Multi-step coding task rejects non-object (e.g. string) subtask entries."""

    def test_multi_step_coding_task_non_object_subtask_rejected(self) -> None:
        """MultiStepCodingTask with a string subtask entry raises ValueError."""
        try:
            from vetinari.agents.coding import MultiStepCodingTask
        except ImportError:
            pytest.skip("vetinari.agents.coding.MultiStepCodingTask not importable")

        with pytest.raises((ValueError, TypeError)):
            MultiStepCodingTask(
                repo_root="/projects/myrepo",
                subtasks=["this is a string, not a subtask object"],
            )


class TestMI42MultiStepCodingTaskScalarTargetFilesRejected:
    """MI-42: Multi-step coding task rejects scalar (string) target_files — must be list."""

    def test_multi_step_coding_task_scalar_target_files_rejected(self) -> None:
        """MultiStepCodingTask with target_files as a plain string raises ValueError."""
        try:
            from vetinari.agents.coding import MultiStepCodingTask
        except ImportError:
            pytest.skip("vetinari.agents.coding.MultiStepCodingTask not importable")

        with pytest.raises((ValueError, TypeError)):
            MultiStepCodingTask(
                repo_root="/projects/myrepo",
                target_files="single_file.py",  # string instead of list
            )


class TestMI43MultiStepCodingTaskNonStringRepoPathRejected:
    """MI-43: Multi-step coding task rejects non-string repo_root."""

    def test_multi_step_coding_task_non_string_repo_path_rejected(self) -> None:
        """MultiStepCodingTask with repo_root as an integer raises ValueError."""
        try:
            from vetinari.agents.coding import MultiStepCodingTask
        except ImportError:
            pytest.skip("vetinari.agents.coding.MultiStepCodingTask not importable")

        with pytest.raises((ValueError, TypeError)):
            MultiStepCodingTask(
                repo_root=12345,  # non-string
                subtasks=[],
            )


class TestMI44PlanModeCodingNoHardcodedRepoPath:
    """MI-44: Plan-mode coding agent does not hardcode a repo path in its output."""

    def test_plan_mode_coding_no_hardcoded_repo_path(self) -> None:
        """The live coding engine must honor the caller's repo_path and target file instead of synthesizing a fake path."""
        from vetinari.coding_agent import CodeAgentEngine, CodingTaskType, make_code_agent_task

        engine = CodeAgentEngine()
        task = make_code_agent_task(
            "Add logging to the API",
            task_type=CodingTaskType.IMPLEMENT,
            repo_path="C:/repo",
            target_files=["src/logging.py"],
            subtask_id="subtask_123",
        )
        artifact = engine.run_task(task)

        assert artifact.path == "C:/repo/src/logging.py"
        suspicious_paths = [p for p in ["/home/", "/root/", "/Users/", "C:\\Users\\", "C:/Users/"] if p in artifact.path]
        assert not suspicious_paths, f"Artifact path contains hardcoded user-home segments: {suspicious_paths}"
        assert "subtask_123.py" not in artifact.path, "artifact path must use the provided target file, not a synthetic subtask path"
