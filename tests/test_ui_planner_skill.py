"""
Unit tests for UI Planner Skill Tool
"""

import pytest
from unittest.mock import Mock
from vetinari.tools.ui_planner_skill import UIPlannerSkillTool, UIPlannerCapability, ThinkingMode
from vetinari.execution_context import ExecutionMode


class TestUIPlannerMetadata:
    def test_init(self):
        t = UIPlannerSkillTool()
        assert t.metadata.name == "ui-planner"

    def test_params(self):
        t = UIPlannerSkillTool()
        names = {p.name for p in t.metadata.parameters}
        assert "capability" in names
        assert "element" in names


class TestUIPlannerExecution:
    def setup_method(self):
        self.tool = UIPlannerSkillTool()
        self.mock_ctx = Mock()
        self.mock_ctx.mode = ExecutionMode.EXECUTION
        self.tool._context_manager = Mock(current_context=self.mock_ctx)

    def test_css_design(self):
        r = self.tool.execute(capability="css_design", element="button")
        assert r.success is True
        assert r.output.get("css_code") is not None

    def test_responsive_layout(self):
        r = self.tool.execute(capability="responsive_layout", element="container")
        assert r.success is True
        assert "@media" in r.output.get("css_code", "")

    def test_animation(self):
        r = self.tool.execute(capability="animation", element="modal")
        assert r.success is True
        assert "keyframes" in r.output.get("css_code", "")

    def test_accessibility(self):
        r = self.tool.execute(capability="accessibility", element="input")
        assert r.success is True
        assert "focus-visible" in r.output.get("css_code", "")

    def test_design_systems(self):
        r = self.tool.execute(capability="design_systems", element="theme")
        assert r.success is True
        assert "--" in r.output.get("css_code", "")

    def test_visual_polish(self):
        r = self.tool.execute(capability="visual_polish", element="card")
        assert r.success is True
        assert "shadow" in r.output.get("css_code", "").lower() or "radius" in r.output.get("css_code", "").lower()

    def test_planning_mode(self):
        self.mock_ctx.mode = ExecutionMode.PLANNING
        r = self.tool.execute(capability="css_design", element="test")
        assert r.success is True
        assert "Planning" in r.output["summary"]

    def test_invalid_capability(self):
        r = self.tool.execute(capability="invalid", element="test")
        assert r.success is False

    def test_missing_element(self):
        r = self.tool.execute(capability="css_design")
        assert r.success is False

    def test_all_capabilities(self):
        for c in UIPlannerCapability:
            r = self.tool.execute(capability=c.value, element="test")
            assert r.success is True
