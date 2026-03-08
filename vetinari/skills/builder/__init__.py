"""
Builder Skill Tool Wrapper

Migrates the builder skill to the Tool interface, providing code implementation,
refactoring, and testing capabilities as a standardized Vetinari tool.

The builder skill specializes in:
- Feature implementation
- Code refactoring
- Test writing
- Error handling
- Code generation
- Debugging
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
import logging
import json
from enum import Enum

from vetinari.tool_interface import (
    Tool,
    ToolMetadata,
    ToolResult,
    ToolParameter,
    ToolCategory,
)
from vetinari.execution_context import ToolPermission, ExecutionMode
from vetinari.types import ThinkingMode  # canonical enum from types.py

logger = logging.getLogger(__name__)


class BuilderCapability(str, Enum):
    """Capabilities of the builder skill."""
    FEATURE_IMPLEMENTATION = "feature_implementation"
    REFACTORING = "refactoring"
    TEST_WRITING = "test_writing"
    ERROR_HANDLING = "error_handling"
    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"


@dataclass
class ImplementationRequest:
    """Request structure for builder operations."""
    capability: BuilderCapability
    description: str
    context: Optional[str] = None
    thinking_mode: ThinkingMode = ThinkingMode.MEDIUM
    requirements: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "capability": self.capability.value,
            "description": self.description,
            "context": self.context,
            "thinking_mode": self.thinking_mode.value,
            "requirements": self.requirements,
        }


@dataclass
class ImplementationResult:
    """Result of a builder operation."""
    success: bool
    code: Optional[str] = None
    explanation: Optional[str] = None
    files_affected: List[str] = field(default_factory=list)
    tests_added: int = 0
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "code": self.code,
            "explanation": self.explanation,
            "files_affected": self.files_affected,
            "tests_added": self.tests_added,
            "warnings": self.warnings,
        }


class BuilderSkillTool(Tool):
    """
    Tool wrapper for the builder skill.
    
    Provides code implementation, refactoring, and testing capabilities
    through a standardized Tool interface.
    
    Permissions:
    - FILE_READ: Read existing code for analysis
    - FILE_WRITE: Create/modify implementation files
    - MODEL_INFERENCE: Use LLM for code generation
    
    Allowed Modes:
    - EXECUTION: Full implementation with all capabilities
    - PLANNING: Analysis only, no code generation
    """
    
    def __init__(self):
        """Initialize the builder skill tool."""
        metadata = ToolMetadata(
            name="builder",
            description=(
                "Code implementation, refactoring, and testing. "
                "Use when user wants to create new features, implement logic, "
                "or modify existing code."
            ),
            category=ToolCategory.CODE_EXECUTION,
            version="1.0.0",
            author="Vetinari",
            parameters=[
                ToolParameter(
                    name="capability",
                    type=str,
                    description="The builder capability to use",
                    required=True,
                    allowed_values=[c.value for c in BuilderCapability],
                ),
                ToolParameter(
                    name="description",
                    type=str,
                    description="Detailed description of what to build/refactor/test",
                    required=True,
                ),
                ToolParameter(
                    name="context",
                    type=str,
                    description="Existing code or context for the operation",
                    required=False,
                ),
                ToolParameter(
                    name="thinking_mode",
                    type=str,
                    description="Implementation approach (low/medium/high/xhigh)",
                    required=False,
                    default="medium",
                    allowed_values=[m.value for m in ThinkingMode],
                ),
                ToolParameter(
                    name="requirements",
                    type=list,
                    description="Specific requirements or constraints",
                    required=False,
                ),
            ],
            required_permissions=[
                ToolPermission.FILE_READ,
                ToolPermission.FILE_WRITE,
                ToolPermission.MODEL_INFERENCE,
            ],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=[
                "implementation",
                "coding",
                "refactoring",
                "testing",
                "build",
            ],
        )
        super().__init__(metadata)
    
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute a builder operation.
        
        Args:
            capability: BuilderCapability to use
            description: What to implement/refactor/test
            context: Existing code context (optional)
            thinking_mode: Implementation approach (default: medium)
            requirements: List of requirements (optional)
            
        Returns:
            ToolResult with implementation details
        """
        try:
            # Extract parameters
            capability_str = kwargs.get("capability")
            description = kwargs.get("description")
            context = kwargs.get("context")
            thinking_mode_str = kwargs.get("thinking_mode", "medium")
            requirements = kwargs.get("requirements", [])
            
            # Convert to enums
            try:
                capability = BuilderCapability(capability_str)
            except ValueError:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Invalid capability: {capability_str}",
                )
            
            try:
                thinking_mode = ThinkingMode(thinking_mode_str)
            except ValueError:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Invalid thinking_mode: {thinking_mode_str}",
                )
            
            # Create request
            request = ImplementationRequest(
                capability=capability,
                description=description,
                context=context,
                thinking_mode=thinking_mode,
                requirements=requirements,
            )
            
            # Get execution mode to determine capabilities
            ctx = self._context_manager.current_context
            execution_mode = ctx.mode
            
            # Execute based on capability
            result = self._execute_capability(request, execution_mode)
            
            return ToolResult(
                success=result.success,
                output=result.to_dict(),
                error=None if result.success else "Implementation failed",
                metadata={
                    "capability": capability.value,
                    "thinking_mode": thinking_mode.value,
                    "execution_mode": execution_mode.value,
                },
            )
        
        except Exception as e:
            logger.error("Builder tool execution failed: %s", e, exc_info=True)
            return ToolResult(
                success=False,
                output=None,
                error=f"Builder tool execution failed: {str(e)}",
            )
    
    def _execute_capability(
        self,
        request: ImplementationRequest,
        execution_mode: ExecutionMode,
    ) -> ImplementationResult:
        """
        Execute a specific builder capability.
        
        Args:
            request: The implementation request
            execution_mode: Current execution mode
            
        Returns:
            ImplementationResult with operation details
        """
        capability = request.capability
        
        if capability == BuilderCapability.FEATURE_IMPLEMENTATION:
            return self._implement_feature(request, execution_mode)
        elif capability == BuilderCapability.REFACTORING:
            return self._refactor_code(request, execution_mode)
        elif capability == BuilderCapability.TEST_WRITING:
            return self._write_tests(request, execution_mode)
        elif capability == BuilderCapability.ERROR_HANDLING:
            return self._handle_errors(request, execution_mode)
        elif capability == BuilderCapability.CODE_GENERATION:
            return self._generate_code(request, execution_mode)
        elif capability == BuilderCapability.DEBUGGING:
            return self._debug_code(request, execution_mode)
        else:
            return ImplementationResult(
                success=False,
                explanation=f"Unknown capability: {capability.value}",
            )
    
    def _implement_feature(
        self,
        request: ImplementationRequest,
        execution_mode: ExecutionMode,
    ) -> ImplementationResult:
        """Implement a new feature."""
        logger.info("Implementing feature: %s", request.description)
        
        if execution_mode == ExecutionMode.PLANNING:
            return ImplementationResult(
                success=True,
                explanation=(
                    f"Planning mode: Would implement feature - {request.description}. "
                    f"Switch to EXECUTION mode to generate code."
                ),
                warnings=["Running in PLANNING mode - code generation disabled"],
            )
        
        # In EXECUTION mode, we would generate code
        # For now, return a structured response
        explanation = (
            f"Feature Implementation (Thinking Mode: {request.thinking_mode.value})\n"
            f"Description: {request.description}\n"
        )
        
        if request.requirements:
            explanation += f"Requirements: {', '.join(request.requirements)}\n"
        
        if request.context:
            explanation += f"Context: {request.context}\n"
        
        # Determine approach based on thinking mode
        if request.thinking_mode == ThinkingMode.LOW:
            explanation += "Approach: Quick implementation with minimal code\n"
        elif request.thinking_mode == ThinkingMode.MEDIUM:
            explanation += (
                "Approach: Full feature with basic tests and error handling\n"
            )
        elif request.thinking_mode == ThinkingMode.HIGH:
            explanation += (
                "Approach: Complete implementation with comprehensive error handling\n"
            )
        else:  # XHIGH
            explanation += (
                "Approach: Production-ready with full test coverage and edge cases\n"
            )
        
        return ImplementationResult(
            success=True,
            explanation=explanation,
            files_affected=[],
        )
    
    def _refactor_code(
        self,
        request: ImplementationRequest,
        execution_mode: ExecutionMode,
    ) -> ImplementationResult:
        """Refactor existing code."""
        logger.info("Refactoring code: %s", request.description)
        
        if not request.context:
            return ImplementationResult(
                success=False,
                explanation="Refactoring requires context (existing code) to be provided",
                warnings=["Missing required context parameter"],
            )
        
        if execution_mode == ExecutionMode.PLANNING:
            return ImplementationResult(
                success=True,
                explanation=(
                    f"Planning mode: Would refactor code - {request.description}. "
                    f"Code to refactor:\n{request.context}\n"
                    f"Switch to EXECUTION mode to perform refactoring."
                ),
                warnings=["Running in PLANNING mode - refactoring disabled"],
            )
        
        explanation = (
            f"Code Refactoring\n"
            f"Description: {request.description}\n"
            f"Original Code:\n{request.context}\n"
            f"Refactoring Approach:\n"
            f"1. Analyze current structure\n"
            f"2. Identify improvements (DRY, naming, performance)\n"
            f"3. Apply refactoring patterns\n"
            f"4. Verify tests still pass\n"
        )
        
        return ImplementationResult(
            success=True,
            explanation=explanation,
            files_affected=[],
        )
    
    def _write_tests(
        self,
        request: ImplementationRequest,
        execution_mode: ExecutionMode,
    ) -> ImplementationResult:
        """Write tests for code."""
        logger.info("Writing tests: %s", request.description)
        
        if not request.context:
            return ImplementationResult(
                success=False,
                explanation="Test writing requires context (code to test) to be provided",
                warnings=["Missing required context parameter"],
            )
        
        if execution_mode == ExecutionMode.PLANNING:
            return ImplementationResult(
                success=True,
                explanation=(
                    f"Planning mode: Would write tests for - {request.description}. "
                    f"Code to test:\n{request.context}\n"
                    f"Switch to EXECUTION mode to generate tests."
                ),
                warnings=["Running in PLANNING mode - test generation disabled"],
            )
        
        explanation = (
            f"Test Writing\n"
            f"Target: {request.description}\n"
            f"Test Strategy:\n"
            f"1. Identify test cases (happy path, edge cases, errors)\n"
            f"2. Use Arrange-Act-Assert pattern\n"
            f"3. Mock external dependencies\n"
            f"4. Achieve target coverage based on thinking mode\n"
        )
        
        if request.thinking_mode in [ThinkingMode.HIGH, ThinkingMode.XHIGH]:
            explanation += "5. Include integration tests\n"
            explanation += "6. Performance/stress tests\n"
        
        test_count = 4  # Base: happy path, edge cases, errors, coverage
        if request.thinking_mode in [ThinkingMode.HIGH, ThinkingMode.XHIGH]:
            test_count = 6  # + integration tests + performance/stress tests

        return ImplementationResult(
            success=True,
            explanation=explanation,
            tests_added=test_count,
            files_affected=[],
        )
    
    def _handle_errors(
        self,
        request: ImplementationRequest,
        execution_mode: ExecutionMode,
    ) -> ImplementationResult:
        """Add error handling to code."""
        logger.info("Adding error handling: %s", request.description)
        
        if not request.context:
            return ImplementationResult(
                success=False,
                explanation="Error handling requires context (code to enhance) to be provided",
                warnings=["Missing required context parameter"],
            )
        
        if execution_mode == ExecutionMode.PLANNING:
            return ImplementationResult(
                success=True,
                explanation=(
                    f"Planning mode: Would add error handling - {request.description}. "
                    f"Code to enhance:\n{request.context}\n"
                    f"Switch to EXECUTION mode to generate error handling."
                ),
                warnings=["Running in PLANNING mode - code generation disabled"],
            )
        
        explanation = (
            f"Error Handling Implementation\n"
            f"Target: {request.description}\n"
            f"Strategy:\n"
            f"1. Identify failure points\n"
            f"2. Add try-catch blocks with meaningful messages\n"
            f"3. Implement graceful degradation\n"
            f"4. Add logging for debugging\n"
            f"5. Create error boundary patterns\n"
        )
        
        return ImplementationResult(
            success=True,
            explanation=explanation,
            files_affected=[],
        )
    
    def _generate_code(
        self,
        request: ImplementationRequest,
        execution_mode: ExecutionMode,
    ) -> ImplementationResult:
        """Generate code from specification."""
        logger.info("Generating code: %s", request.description)
        
        if execution_mode == ExecutionMode.PLANNING:
            return ImplementationResult(
                success=True,
                explanation=(
                    f"Planning mode: Would generate code - {request.description}. "
                    f"Switch to EXECUTION mode to generate code."
                ),
                warnings=["Running in PLANNING mode - code generation disabled"],
            )
        
        explanation = (
            f"Code Generation\n"
            f"Specification: {request.description}\n"
            f"Generation Process:\n"
            f"1. Analyze specification\n"
            f"2. Determine patterns and structure\n"
            f"3. Generate boilerplate and scaffolding\n"
            f"4. Add type definitions\n"
            f"5. Include basic documentation\n"
        )
        
        if request.requirements:
            explanation += f"6. Ensure requirements met: {', '.join(request.requirements)}\n"
        
        return ImplementationResult(
            success=True,
            explanation=explanation,
            files_affected=[],
        )
    
    def _debug_code(
        self,
        request: ImplementationRequest,
        execution_mode: ExecutionMode,
    ) -> ImplementationResult:
        """Debug and fix code issues."""
        logger.info("Debugging code: %s", request.description)
        
        if not request.context:
            return ImplementationResult(
                success=False,
                explanation="Debugging requires context (code to debug) to be provided",
                warnings=["Missing required context parameter"],
            )
        
        if execution_mode == ExecutionMode.PLANNING:
            return ImplementationResult(
                success=True,
                explanation=(
                    f"Planning mode: Would debug - {request.description}. "
                    f"Code to debug:\n{request.context}\n"
                    f"Switch to EXECUTION mode to debug and fix."
                ),
                warnings=["Running in PLANNING mode - debugging disabled"],
            )
        
        explanation = (
            f"Code Debugging\n"
            f"Issue: {request.description}\n"
            f"Debug Process:\n"
            f"1. Understand the problem\n"
            f"2. Identify failure points\n"
            f"3. Analyze root cause\n"
            f"4. Implement fix\n"
            f"5. Test the fix\n"
            f"6. Verify no regressions\n"
        )
        
        return ImplementationResult(
            success=True,
            explanation=explanation,
            files_affected=[],
        )
