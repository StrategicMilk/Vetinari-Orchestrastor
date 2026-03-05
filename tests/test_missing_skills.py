"""
Tests for the 7 new skill tool wrappers and get_all_skills() discovery.

Covers:
1.  DevOpsSkillTool instantiation and metadata
2.  VersionControlSkillTool instantiation and metadata
3.  ErrorRecoverySkillTool instantiation and metadata
4.  ContextManagerSkillTool instantiation and metadata
5.  ImageGeneratorSkillTool instantiation and metadata
6.  UserInteractionSkillTool instantiation and metadata
7.  ImprovementSkillTool instantiation and metadata
8.  get_all_skills() discovers all expected skills (>=21)
9.  All skills have unique names
10. All skills have required ToolMetadata fields
"""

import pytest

from vetinari.tools.devops_skill import DevOpsSkillTool
from vetinari.tools.version_control_skill import VersionControlSkillTool
from vetinari.tools.error_recovery_skill import ErrorRecoverySkillTool
from vetinari.tools.context_manager_skill import ContextManagerSkillTool
from vetinari.tools.image_generator_skill import ImageGeneratorSkillTool
from vetinari.tools.user_interaction_skill import UserInteractionSkillTool
from vetinari.tools.improvement_skill import ImprovementSkillTool
from vetinari.tools import get_all_skills
from vetinari.tool_interface import ToolCategory
from vetinari.execution_context import ExecutionMode


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_metadata(tool, expected_name: str, expected_tags: list):
    """Assert common metadata invariants."""
    assert tool.metadata is not None
    assert tool.metadata.name == expected_name
    assert isinstance(tool.metadata.description, str)
    assert len(tool.metadata.description) > 0
    assert tool.metadata.category is not None
    for tag in expected_tags:
        assert tag in tool.metadata.tags, f"Expected tag '{tag}' in {tool.metadata.tags}"
    assert ExecutionMode.EXECUTION in tool.metadata.allowed_modes


# ---------------------------------------------------------------------------
# 1. DevOpsSkillTool
# ---------------------------------------------------------------------------

class TestDevOpsSkillTool:
    def test_instantiation(self):
        tool = DevOpsSkillTool()
        assert tool is not None

    def test_metadata(self):
        tool = DevOpsSkillTool()
        _check_metadata(tool, "devops", ["devops", "ci", "deployment", "infrastructure"])

    def test_has_target_parameter(self):
        tool = DevOpsSkillTool()
        param_names = [p.name for p in tool.metadata.parameters]
        assert "target" in param_names

    def test_has_focus_parameter(self):
        tool = DevOpsSkillTool()
        param_names = [p.name for p in tool.metadata.parameters]
        assert "focus" in param_names

    def test_target_is_required(self):
        tool = DevOpsSkillTool()
        target_param = next(p for p in tool.metadata.parameters if p.name == "target")
        assert target_param.required is True


# ---------------------------------------------------------------------------
# 2. VersionControlSkillTool
# ---------------------------------------------------------------------------

class TestVersionControlSkillTool:
    def test_instantiation(self):
        tool = VersionControlSkillTool()
        assert tool is not None

    def test_metadata(self):
        tool = VersionControlSkillTool()
        _check_metadata(tool, "version_control", ["git", "version-control", "commits", "pr"])

    def test_has_target_parameter(self):
        tool = VersionControlSkillTool()
        param_names = [p.name for p in tool.metadata.parameters]
        assert "target" in param_names

    def test_has_action_parameter(self):
        tool = VersionControlSkillTool()
        param_names = [p.name for p in tool.metadata.parameters]
        assert "action" in param_names

    def test_target_is_required(self):
        tool = VersionControlSkillTool()
        target_param = next(p for p in tool.metadata.parameters if p.name == "target")
        assert target_param.required is True


# ---------------------------------------------------------------------------
# 3. ErrorRecoverySkillTool
# ---------------------------------------------------------------------------

class TestErrorRecoverySkillTool:
    def test_instantiation(self):
        tool = ErrorRecoverySkillTool()
        assert tool is not None

    def test_metadata(self):
        tool = ErrorRecoverySkillTool()
        _check_metadata(tool, "error_recovery", ["error", "recovery", "resilience", "debugging"])

    def test_has_error_info_parameter(self):
        tool = ErrorRecoverySkillTool()
        param_names = [p.name for p in tool.metadata.parameters]
        assert "error_info" in param_names

    def test_error_info_is_required(self):
        tool = ErrorRecoverySkillTool()
        param = next(p for p in tool.metadata.parameters if p.name == "error_info")
        assert param.required is True

    def test_context_is_optional(self):
        tool = ErrorRecoverySkillTool()
        param = next(p for p in tool.metadata.parameters if p.name == "context")
        assert param.required is False


# ---------------------------------------------------------------------------
# 4. ContextManagerSkillTool
# ---------------------------------------------------------------------------

class TestContextManagerSkillTool:
    def test_instantiation(self):
        tool = ContextManagerSkillTool()
        assert tool is not None

    def test_metadata(self):
        tool = ContextManagerSkillTool()
        _check_metadata(tool, "context_manager", ["context", "memory", "consolidation"])

    def test_has_scope_parameter(self):
        tool = ContextManagerSkillTool()
        param_names = [p.name for p in tool.metadata.parameters]
        assert "scope" in param_names

    def test_scope_is_required(self):
        tool = ContextManagerSkillTool()
        param = next(p for p in tool.metadata.parameters if p.name == "scope")
        assert param.required is True

    def test_has_action_parameter(self):
        tool = ContextManagerSkillTool()
        param_names = [p.name for p in tool.metadata.parameters]
        assert "action" in param_names


# ---------------------------------------------------------------------------
# 5. ImageGeneratorSkillTool
# ---------------------------------------------------------------------------

class TestImageGeneratorSkillTool:
    def test_instantiation(self):
        tool = ImageGeneratorSkillTool()
        assert tool is not None

    def test_metadata(self):
        tool = ImageGeneratorSkillTool()
        _check_metadata(tool, "image_generator", ["image", "diagram", "visualization"])

    def test_has_description_parameter(self):
        tool = ImageGeneratorSkillTool()
        param_names = [p.name for p in tool.metadata.parameters]
        assert "description" in param_names

    def test_description_is_required(self):
        tool = ImageGeneratorSkillTool()
        param = next(p for p in tool.metadata.parameters if p.name == "description")
        assert param.required is True

    def test_has_format_parameter(self):
        tool = ImageGeneratorSkillTool()
        param_names = [p.name for p in tool.metadata.parameters]
        assert "format" in param_names

    def test_execute_mermaid(self):
        tool = ImageGeneratorSkillTool()
        result = tool.execute(description="My system", format="mermaid")
        assert result.success is True
        assert "mermaid" in str(result.output.get("format", ""))
        assert result.output["content"].strip().startswith("graph")

    def test_execute_ascii(self):
        tool = ImageGeneratorSkillTool()
        result = tool.execute(description="My system", format="ascii")
        assert result.success is True

    def test_execute_svg(self):
        tool = ImageGeneratorSkillTool()
        result = tool.execute(description="My system", format="svg")
        assert result.success is True
        assert "<svg" in result.output["content"]

    def test_execute_unknown_format(self):
        tool = ImageGeneratorSkillTool()
        result = tool.execute(description="X", format="pdf")
        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# 6. UserInteractionSkillTool
# ---------------------------------------------------------------------------

class TestUserInteractionSkillTool:
    def test_instantiation(self):
        tool = UserInteractionSkillTool()
        assert tool is not None

    def test_metadata(self):
        tool = UserInteractionSkillTool()
        _check_metadata(tool, "user_interaction", ["user", "interaction", "clarification", "qa"])

    def test_has_question_parameter(self):
        tool = UserInteractionSkillTool()
        param_names = [p.name for p in tool.metadata.parameters]
        assert "question" in param_names

    def test_question_is_required(self):
        tool = UserInteractionSkillTool()
        param = next(p for p in tool.metadata.parameters if p.name == "question")
        assert param.required is True

    def test_context_is_optional(self):
        tool = UserInteractionSkillTool()
        param = next(p for p in tool.metadata.parameters if p.name == "context")
        assert param.required is False


# ---------------------------------------------------------------------------
# 7. ImprovementSkillTool
# ---------------------------------------------------------------------------

class TestImprovementSkillTool:
    def test_instantiation(self):
        tool = ImprovementSkillTool()
        assert tool is not None

    def test_metadata(self):
        tool = ImprovementSkillTool()
        _check_metadata(tool, "improvement", ["improvement", "optimization", "meta"])

    def test_has_target_parameter(self):
        tool = ImprovementSkillTool()
        param_names = [p.name for p in tool.metadata.parameters]
        assert "target" in param_names

    def test_target_is_required(self):
        tool = ImprovementSkillTool()
        param = next(p for p in tool.metadata.parameters if p.name == "target")
        assert param.required is True

    def test_has_focus_parameter(self):
        tool = ImprovementSkillTool()
        param_names = [p.name for p in tool.metadata.parameters]
        assert "focus" in param_names


# ---------------------------------------------------------------------------
# 8. get_all_skills() discovers all expected skills (>=21)
# ---------------------------------------------------------------------------

class TestGetAllSkills:
    def test_discovers_at_least_21_skills(self):
        skills = get_all_skills()
        assert len(skills) >= 21, f"Expected >=21 skills, got {len(skills)}"

    def test_new_skills_are_discovered(self):
        skills = get_all_skills()
        class_names = {s.__name__ for s in skills}
        expected = {
            "DevOpsSkillTool",
            "VersionControlSkillTool",
            "ErrorRecoverySkillTool",
            "ContextManagerSkillTool",
            "ImageGeneratorSkillTool",
            "UserInteractionSkillTool",
            "ImprovementSkillTool",
        }
        missing = expected - class_names
        assert not missing, f"Missing from get_all_skills(): {missing}"

    def test_returns_classes_not_instances(self):
        skills = get_all_skills()
        for skill_cls in skills:
            assert isinstance(skill_cls, type), f"{skill_cls} is not a class"


# ---------------------------------------------------------------------------
# 9. All skills have unique names
# ---------------------------------------------------------------------------

class TestSkillUniqueNames:
    def test_all_skill_names_are_unique(self):
        skills = get_all_skills()
        instances = []
        for skill_cls in skills:
            try:
                instances.append(skill_cls())
            except Exception:
                pass  # Skip skills that fail to instantiate (agent unavailable etc.)
        names = [inst.metadata.name for inst in instances]
        assert len(names) == len(set(names)), (
            f"Duplicate skill names found: {[n for n in names if names.count(n) > 1]}"
        )


# ---------------------------------------------------------------------------
# 10. All skills have required ToolMetadata fields
# ---------------------------------------------------------------------------

class TestSkillMetadataFields:
    NEW_SKILL_CLASSES = [
        DevOpsSkillTool,
        VersionControlSkillTool,
        ErrorRecoverySkillTool,
        ContextManagerSkillTool,
        ImageGeneratorSkillTool,
        UserInteractionSkillTool,
        ImprovementSkillTool,
    ]

    @pytest.mark.parametrize("skill_cls", NEW_SKILL_CLASSES)
    def test_has_name(self, skill_cls):
        tool = skill_cls()
        assert tool.metadata.name
        assert isinstance(tool.metadata.name, str)

    @pytest.mark.parametrize("skill_cls", NEW_SKILL_CLASSES)
    def test_has_description(self, skill_cls):
        tool = skill_cls()
        assert tool.metadata.description
        assert isinstance(tool.metadata.description, str)

    @pytest.mark.parametrize("skill_cls", NEW_SKILL_CLASSES)
    def test_has_category(self, skill_cls):
        tool = skill_cls()
        assert tool.metadata.category is not None
        assert isinstance(tool.metadata.category, ToolCategory)

    @pytest.mark.parametrize("skill_cls", NEW_SKILL_CLASSES)
    def test_has_tags(self, skill_cls):
        tool = skill_cls()
        assert isinstance(tool.metadata.tags, list)
        assert len(tool.metadata.tags) > 0

    @pytest.mark.parametrize("skill_cls", NEW_SKILL_CLASSES)
    def test_has_allowed_modes(self, skill_cls):
        tool = skill_cls()
        assert isinstance(tool.metadata.allowed_modes, list)
        assert len(tool.metadata.allowed_modes) > 0

    @pytest.mark.parametrize("skill_cls", NEW_SKILL_CLASSES)
    def test_has_parameters(self, skill_cls):
        tool = skill_cls()
        assert isinstance(tool.metadata.parameters, list)
        assert len(tool.metadata.parameters) >= 1
