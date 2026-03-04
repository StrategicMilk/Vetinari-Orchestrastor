"""
Unit tests for the explorer skill Tool wrapper.

Tests cover:
- Tool initialization and metadata
- All explorer capabilities (grep, file discovery, patterns, symbols, imports, mapping)
- Parameter validation
- Permission checking
- Execution mode handling (PLANNING vs EXECUTION)
- Error cases
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from vetinari.tools.explorer_skill import (
    ExplorerSkillTool,
    ExplorerCapability,
    ThinkingMode,
    SearchStrategy,
    ExplorationRequest,
    SearchResult,
    ExplorationResult,
)
from vetinari.execution_context import (
    ToolPermission,
    ExecutionMode,
    ExecutionContext,
    ContextManager,
)
from vetinari.tool_interface import ToolResult


class TestExplorerSkillToolMetadata:
    """Tests for explorer skill metadata and initialization."""
    
    def test_initialization(self):
        """Test ExplorerSkillTool initialization."""
        tool = ExplorerSkillTool()
        
        assert tool.metadata.name == "explorer"
        assert "search" in tool.metadata.tags
        assert "discovery" in tool.metadata.tags
        assert ToolPermission.FILE_READ in tool.metadata.required_permissions
    
    def test_allowed_execution_modes(self):
        """Test allowed execution modes."""
        tool = ExplorerSkillTool()
        
        assert ExecutionMode.EXECUTION in tool.metadata.allowed_modes
        assert ExecutionMode.PLANNING in tool.metadata.allowed_modes
    
    def test_tool_parameters(self):
        """Test tool parameters are properly defined."""
        tool = ExplorerSkillTool()
        
        param_names = {p.name for p in tool.metadata.parameters}
        assert "capability" in param_names
        assert "query" in param_names
        assert "thinking_mode" in param_names
        assert "search_strategy" in param_names
        assert "file_extensions" in param_names
        assert "context_lines" in param_names
        assert "max_results" in param_names
    
    def test_capability_parameter_validation(self):
        """Test capability parameter validation."""
        tool = ExplorerSkillTool()
        
        capability_param = next(
            p for p in tool.metadata.parameters if p.name == "capability"
        )
        
        assert capability_param.required is True
        assert all(c.value in capability_param.allowed_values for c in ExplorerCapability)
    
    def test_thinking_mode_parameter_validation(self):
        """Test thinking_mode parameter validation."""
        tool = ExplorerSkillTool()
        
        thinking_param = next(
            p for p in tool.metadata.parameters if p.name == "thinking_mode"
        )
        
        assert thinking_param.required is False
        assert thinking_param.default == "medium"
        assert all(m.value in thinking_param.allowed_values for m in ThinkingMode)


class TestExplorerSkillToolExecution:
    """Tests for explorer skill execution logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = ExplorerSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(spec=ExecutionContext)
        self.mock_ctx_manager.current_context = self.mock_context
        self.mock_ctx_manager.current_mode = ExecutionMode.EXECUTION
        
        self.mock_context.mode = ExecutionMode.EXECUTION
        self.mock_context.pre_execution_hooks = []
        self.mock_context.post_execution_hooks = []
        
        self.tool._context_manager = self.mock_ctx_manager
    
    def test_grep_search_execution_mode(self):
        """Test grep search in EXECUTION mode."""
        self.mock_context.mode = ExecutionMode.EXECUTION
        
        result = self.tool.execute(
            capability="grep_search",
            query="function foo",
            search_strategy="regex",
        )
        
        assert result.success is True
        assert result.output is not None
        assert "Grep Search" in result.output.get("results", [{}])[0] if result.output.get("results") else True
    
    def test_grep_search_planning_mode(self):
        """Test grep search in PLANNING mode."""
        self.mock_context.mode = ExecutionMode.PLANNING
        
        result = self.tool.execute(
            capability="grep_search",
            query="pattern",
        )
        
        assert result.success is True
        # In planning mode, it should indicate it would search
        assert result.output is not None
    
    def test_file_discovery(self):
        """Test file discovery capability."""
        result = self.tool.execute(
            capability="file_discovery",
            query="**/*.py",
            file_extensions=[".py"],
        )
        
        assert result.success is True
        assert result.output["capability"] == "file_discovery"
    
    def test_file_discovery_planning_mode(self):
        """Test file discovery in PLANNING mode."""
        self.mock_context.mode = ExecutionMode.PLANNING
        
        result = self.tool.execute(
            capability="file_discovery",
            query="src/**/*.ts",
        )
        
        assert result.success is True
        # Should return valid result even in planning mode
        assert result.output is not None
    
    def test_pattern_matching(self):
        """Test pattern matching capability."""
        result = self.tool.execute(
            capability="pattern_matching",
            query=r"\bconst\s+\w+\s*=",
            search_strategy="regex",
        )
        
        assert result.success is True
        assert result.output["capability"] == "pattern_matching"
    
    def test_symbol_lookup(self):
        """Test symbol lookup capability."""
        result = self.tool.execute(
            capability="symbol_lookup",
            query="getUserById",
        )
        
        assert result.success is True
        assert result.output["capability"] == "symbol_lookup"
    
    def test_symbol_lookup_thinking_modes(self):
        """Test symbol lookup with different thinking modes."""
        modes = ["low", "medium", "high", "xhigh"]
        
        for mode in modes:
            result = self.tool.execute(
                capability="symbol_lookup",
                query="User",
                thinking_mode=mode,
            )
            
            assert result.success is True
    
    def test_import_analysis(self):
        """Test import analysis capability."""
        result = self.tool.execute(
            capability="import_analysis",
            query="src/utils/auth.ts",
        )
        
        assert result.success is True
        assert result.output["capability"] == "import_analysis"
    
    def test_project_mapping(self):
        """Test project mapping capability."""
        result = self.tool.execute(
            capability="project_mapping",
            query="payment feature",
            thinking_mode="high",
        )
        
        assert result.success is True
        assert result.output["capability"] == "project_mapping"
    
    def test_project_mapping_thinking_modes(self):
        """Test project mapping with different thinking modes."""
        for mode in ["low", "medium", "high", "xhigh"]:
            result = self.tool.execute(
                capability="project_mapping",
                query="codebase",
                thinking_mode=mode,
            )
            
            assert result.success is True
    
    def test_invalid_capability(self):
        """Test execution with invalid capability."""
        result = self.tool.execute(
            capability="invalid_capability",
            query="test",
        )
        
        assert result.success is False
        assert "Invalid capability" in result.error
    
    def test_invalid_thinking_mode(self):
        """Test execution with invalid thinking mode."""
        result = self.tool.execute(
            capability="grep_search",
            query="test",
            thinking_mode="invalid_mode",
        )
        
        assert result.success is False
        assert "Invalid thinking_mode" in result.error
    
    def test_invalid_search_strategy(self):
        """Test execution with invalid search strategy."""
        result = self.tool.execute(
            capability="grep_search",
            query="test",
            search_strategy="invalid_strategy",
        )
        
        assert result.success is False
        assert "Invalid search_strategy" in result.error
    
    def test_context_lines_parameter(self):
        """Test context_lines parameter."""
        result = self.tool.execute(
            capability="grep_search",
            query="foo",
            context_lines=5,
        )
        
        assert result.success is True
    
    def test_max_results_parameter(self):
        """Test max_results parameter."""
        result = self.tool.execute(
            capability="grep_search",
            query="bar",
            max_results=50,
        )
        
        assert result.success is True
    
    def test_file_extensions_parameter(self):
        """Test file_extensions parameter."""
        result = self.tool.execute(
            capability="grep_search",
            query="import",
            file_extensions=[".ts", ".tsx", ".js"],
        )
        
        assert result.success is True


class TestExplorationRequest:
    """Tests for ExplorationRequest dataclass."""
    
    def test_creation_with_defaults(self):
        """Test creating request with default values."""
        request = ExplorationRequest(
            capability=ExplorerCapability.GREP_SEARCH,
            query="test pattern",
        )
        
        assert request.capability == ExplorerCapability.GREP_SEARCH
        assert request.query == "test pattern"
        assert request.thinking_mode == ThinkingMode.MEDIUM
        assert request.search_strategy == SearchStrategy.PARTIAL
        assert request.context_lines == 2
        assert request.max_results == 20
    
    def test_creation_with_all_fields(self):
        """Test creating request with all fields."""
        request = ExplorationRequest(
            capability=ExplorerCapability.SYMBOL_LOOKUP,
            query="User",
            thinking_mode=ThinkingMode.HIGH,
            search_strategy=SearchStrategy.EXACT,
            file_extensions=[".ts", ".py"],
            context_lines=5,
            max_results=100,
        )
        
        assert request.capability == ExplorerCapability.SYMBOL_LOOKUP
        assert request.thinking_mode == ThinkingMode.HIGH
        assert request.search_strategy == SearchStrategy.EXACT
        assert request.file_extensions == [".ts", ".py"]
        assert request.context_lines == 5
        assert request.max_results == 100
    
    def test_to_dict(self):
        """Test converting request to dictionary."""
        request = ExplorationRequest(
            capability=ExplorerCapability.FILE_DISCOVERY,
            query="**/*.py",
            thinking_mode=ThinkingMode.XHIGH,
        )
        
        result_dict = request.to_dict()
        
        assert result_dict["capability"] == "file_discovery"
        assert result_dict["query"] == "**/*.py"
        assert result_dict["thinking_mode"] == "xhigh"


class TestSearchResult:
    """Tests for SearchResult dataclass."""
    
    def test_creation(self):
        """Test creating search result."""
        result = SearchResult(
            file_path="src/auth.ts",
            line_number=42,
            line_content="function login() {",
            before_context=["// Authentication", ""],
            after_context=["  // body", "}"],
        )
        
        assert result.file_path == "src/auth.ts"
        assert result.line_number == 42
        assert len(result.before_context) == 2
        assert len(result.after_context) == 2
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = SearchResult(
            file_path="test.py",
            line_number=10,
            line_content="def foo():",
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["file_path"] == "test.py"
        assert result_dict["line_number"] == 10
        assert result_dict["line_content"] == "def foo():"


class TestExplorationResult:
    """Tests for ExplorationResult dataclass."""
    
    def test_success_result(self):
        """Test creating successful result."""
        result = ExplorationResult(
            success=True,
            query="getUserById",
            capability="symbol_lookup",
            total_found=5,
            files_searched=150,
        )
        
        assert result.success is True
        assert result.total_found == 5
        assert result.files_searched == 150
    
    def test_failure_result(self):
        """Test creating failure result."""
        result = ExplorationResult(
            success=False,
            query="pattern",
            capability="grep_search",
            warnings=["Pattern is too broad"],
        )
        
        assert result.success is False
        assert "Pattern is too broad" in result.warnings
    
    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ExplorationResult(
            success=True,
            query="foo",
            capability="grep_search",
            total_found=10,
            project_type="Python",
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["success"] is True
        assert result_dict["total_found"] == 10
        assert result_dict["project_type"] == "Python"


class TestExplorerCapabilityEnum:
    """Tests for ExplorerCapability enum."""
    
    def test_all_capabilities_have_values(self):
        """Test that all capabilities have proper values."""
        capabilities = [
            ExplorerCapability.GREP_SEARCH,
            ExplorerCapability.FILE_DISCOVERY,
            ExplorerCapability.PATTERN_MATCHING,
            ExplorerCapability.SYMBOL_LOOKUP,
            ExplorerCapability.IMPORT_ANALYSIS,
            ExplorerCapability.PROJECT_MAPPING,
        ]
        
        for cap in capabilities:
            assert cap.value is not None
            assert isinstance(cap.value, str)
            assert len(cap.value) > 0


class TestThinkingModeEnum:
    """Tests for ThinkingMode enum."""
    
    def test_all_modes_have_values(self):
        """Test that all modes have proper values."""
        modes = [
            ThinkingMode.LOW,
            ThinkingMode.MEDIUM,
            ThinkingMode.HIGH,
            ThinkingMode.XHIGH,
        ]
        
        for mode in modes:
            assert mode.value is not None
            assert isinstance(mode.value, str)


class TestSearchStrategyEnum:
    """Tests for SearchStrategy enum."""
    
    def test_all_strategies_have_values(self):
        """Test that all strategies have proper values."""
        strategies = [
            SearchStrategy.EXACT,
            SearchStrategy.REGEX,
            SearchStrategy.PARTIAL,
        ]
        
        for strat in strategies:
            assert strat.value is not None
            assert isinstance(strat.value, str)


class TestExplorerSkillToolParameterValidation:
    """Tests for input parameter validation."""
    
    def test_missing_required_capability(self):
        """Test that missing capability is caught by validation."""
        tool = ExplorerSkillTool()
        
        is_valid, error = tool.validate_inputs({"query": "test"})
        
        assert is_valid is False
        assert "capability" in error.lower()
    
    def test_missing_required_query(self):
        """Test that missing query is caught by validation."""
        tool = ExplorerSkillTool()
        
        is_valid, error = tool.validate_inputs({"capability": "grep_search"})
        
        assert is_valid is False
        assert "query" in error.lower()
    
    def test_invalid_capability_value(self):
        """Test that invalid capability value is caught."""
        tool = ExplorerSkillTool()
        
        is_valid, error = tool.validate_inputs({
            "capability": "invalid_capability",
            "query": "test",
        })
        
        assert is_valid is False
        assert "capability" in error.lower()
    
    def test_valid_parameters(self):
        """Test that valid parameters pass validation."""
        tool = ExplorerSkillTool()
        
        is_valid, error = tool.validate_inputs({
            "capability": "grep_search",
            "query": "test pattern",
            "thinking_mode": "high",
            "search_strategy": "regex",
        })
        
        assert is_valid is True
        assert error is None


class TestExplorerSkillToolEdgeCases:
    """Tests for edge cases and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = ExplorerSkillTool()
        self.mock_ctx_manager = Mock()
        self.mock_context = Mock(spec=ExecutionContext)
        self.mock_ctx_manager.current_context = self.mock_context
        self.mock_ctx_manager.current_mode = ExecutionMode.EXECUTION
        
        self.mock_context.mode = ExecutionMode.EXECUTION
        self.mock_context.pre_execution_hooks = []
        self.mock_context.post_execution_hooks = []
        
        self.tool._context_manager = self.mock_ctx_manager
    
    def test_empty_query(self):
        """Test with empty query."""
        result = self.tool.execute(
            capability="grep_search",
            query="",
        )
        
        assert result.success is True
    
    def test_very_long_query(self):
        """Test with very long query."""
        long_query = "pattern" * 1000
        
        result = self.tool.execute(
            capability="grep_search",
            query=long_query,
        )
        
        assert result.success is True
    
    def test_regex_special_characters(self):
        """Test with regex special characters."""
        result = self.tool.execute(
            capability="grep_search",
            query=r"^def\s+\w+\(.*\):",
            search_strategy="regex",
        )
        
        assert result.success is True
    
    def test_unicode_in_query(self):
        """Test with unicode characters in query."""
        result = self.tool.execute(
            capability="grep_search",
            query="日本語 pattern 🚀",
        )
        
        assert result.success is True
    
    def test_multiple_file_extensions(self):
        """Test with multiple file extensions."""
        result = self.tool.execute(
            capability="grep_search",
            query="import",
            file_extensions=[".ts", ".tsx", ".js", ".jsx", ".py", ".java"],
        )
        
        assert result.success is True
    
    def test_large_context_lines(self):
        """Test with large context_lines value."""
        result = self.tool.execute(
            capability="grep_search",
            query="foo",
            context_lines=100,
        )
        
        assert result.success is True
    
    def test_large_max_results(self):
        """Test with large max_results value."""
        result = self.tool.execute(
            capability="grep_search",
            query="bar",
            max_results=10000,
        )
        
        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
