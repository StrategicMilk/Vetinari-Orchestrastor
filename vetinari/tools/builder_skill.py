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
from vetinari.tools.output_validation import validate_output

logger = logging.getLogger(__name__)


class BuilderCapability(str, Enum):
    """Capabilities of the builder skill."""
    FEATURE_IMPLEMENTATION = "feature_implementation"
    REFACTORING = "refactoring"
    TEST_WRITING = "test_writing"
    ERROR_HANDLING = "error_handling"
    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"


class ThinkingMode(str, Enum):
    """Thinking modes for implementation approach."""
    LOW = "low"          # Quick implementation
    MEDIUM = "medium"    # Full feature with tests
    HIGH = "high"        # Complete with error handling
    XHIGH = "xhigh"      # Production-ready with full test coverage


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

            # Validate output before returning
            validation = validate_output(
                result, required_fields=["success"]
            )
            if not validation["valid"]:
                logger.warning("Builder output validation failed: %s", validation["errors"])

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
            logger.error(f"Builder tool execution failed: {e}", exc_info=True)
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
    
    def _try_llm_generate(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Attempt LLM-based code generation via BaseAgent._infer_json().

        Returns parsed JSON on success, None on failure.
        """
        try:
            from vetinari.agents.base_agent import BaseAgent
            agent = BaseAgent.__new__(BaseAgent)
            # Minimal init so _infer_json works
            if hasattr(agent, '_infer_json'):
                result = agent._infer_json(prompt, fallback=None)
                if result and isinstance(result, dict):
                    return result
        except Exception as e:
            logger.debug(f"LLM inference attempt failed: {e}")
        return None

    def _implement_feature(
        self,
        request: ImplementationRequest,
        execution_mode: ExecutionMode,
    ) -> ImplementationResult:
        """Implement a new feature."""
        logger.info(f"Implementing feature: {request.description}")

        if execution_mode == ExecutionMode.PLANNING:
            return ImplementationResult(
                success=True,
                explanation=(
                    f"Planning mode: Would implement feature - {request.description}. "
                    f"Switch to EXECUTION mode to generate code."
                ),
                warnings=["Running in PLANNING mode - code generation disabled"],
            )

        # Try LLM-based implementation
        prompt = (
            f"Implement the following feature and return JSON with keys: "
            f"code, explanation, files_affected, tests_added.\n\n"
            f"Feature: {request.description}\n"
        )
        if request.requirements:
            prompt += f"Requirements: {', '.join(request.requirements)}\n"
        if request.context:
            prompt += f"Context: {request.context}\n"
        prompt += f"Thinking mode: {request.thinking_mode.value}\n"

        llm_result = self._try_llm_generate(prompt)
        if llm_result:
            return ImplementationResult(
                success=True,
                code=llm_result.get("code"),
                explanation=llm_result.get("explanation", f"Feature implemented: {request.description}"),
                files_affected=llm_result.get("files_affected", []),
                tests_added=llm_result.get("tests_added", 0),
            )

        # LLM unavailable
        return ImplementationResult(
            success=False,
            explanation="LLM inference unavailable",
            warnings=["llm_unavailable"],
        )
    
    def _refactor_code(
        self,
        request: ImplementationRequest,
        execution_mode: ExecutionMode,
    ) -> ImplementationResult:
        """Refactor existing code."""
        logger.info(f"Refactoring code: {request.description}")
        
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

        # Try LLM-based refactoring
        prompt = (
            f"Refactor the following code and return JSON with keys: "
            f"code, explanation, files_affected.\n\n"
            f"Description: {request.description}\n"
            f"Original Code:\n{request.context}\n"
        )
        llm_result = self._try_llm_generate(prompt)
        if llm_result:
            return ImplementationResult(
                success=True,
                code=llm_result.get("code"),
                explanation=llm_result.get("explanation", f"Refactored: {request.description}"),
                files_affected=llm_result.get("files_affected", []),
            )

        return ImplementationResult(
            success=False,
            explanation="LLM inference unavailable",
            warnings=["llm_unavailable"],
        )
    
    def _write_tests(
        self,
        request: ImplementationRequest,
        execution_mode: ExecutionMode,
    ) -> ImplementationResult:
        """Write tests for code."""
        logger.info(f"Writing tests: {request.description}")
        
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

        # Try LLM-based test generation
        prompt = (
            f"Write tests for the following code and return JSON with keys: "
            f"code, explanation, files_affected, tests_added (integer count).\n\n"
            f"Target: {request.description}\n"
            f"Code to test:\n{request.context}\n"
            f"Thinking mode: {request.thinking_mode.value}\n"
        )
        llm_result = self._try_llm_generate(prompt)
        if llm_result:
            return ImplementationResult(
                success=True,
                code=llm_result.get("code"),
                explanation=llm_result.get("explanation", f"Tests written for: {request.description}"),
                tests_added=llm_result.get("tests_added", 0),
                files_affected=llm_result.get("files_affected", []),
            )

        return ImplementationResult(
            success=False,
            explanation="LLM inference unavailable",
            tests_added=0,
            warnings=["llm_unavailable"],
        )
    
    def _handle_errors(
        self,
        request: ImplementationRequest,
        execution_mode: ExecutionMode,
    ) -> ImplementationResult:
        """Add error handling to code."""
        logger.info(f"Adding error handling: {request.description}")
        
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

        # Try LLM-based error handling
        prompt = (
            f"Add error handling to the following code and return JSON with keys: "
            f"code, explanation, files_affected.\n\n"
            f"Target: {request.description}\n"
            f"Code to enhance:\n{request.context}\n"
        )
        llm_result = self._try_llm_generate(prompt)
        if llm_result:
            return ImplementationResult(
                success=True,
                code=llm_result.get("code"),
                explanation=llm_result.get("explanation", f"Error handling added: {request.description}"),
                files_affected=llm_result.get("files_affected", []),
            )

        return ImplementationResult(
            success=False,
            explanation="LLM inference unavailable",
            warnings=["llm_unavailable"],
        )
    
    def _generate_code(
        self,
        request: ImplementationRequest,
        execution_mode: ExecutionMode,
    ) -> ImplementationResult:
        """Generate code from specification."""
        logger.info(f"Generating code: {request.description}")
        
        if execution_mode == ExecutionMode.PLANNING:
            return ImplementationResult(
                success=True,
                explanation=(
                    f"Planning mode: Would generate code - {request.description}. "
                    f"Switch to EXECUTION mode to generate code."
                ),
                warnings=["Running in PLANNING mode - code generation disabled"],
            )

        # Try LLM-based code generation
        prompt = (
            f"Generate code for the following specification and return JSON with keys: "
            f"code, explanation, files_affected, tests_added.\n\n"
            f"Specification: {request.description}\n"
        )
        if request.requirements:
            prompt += f"Requirements: {', '.join(request.requirements)}\n"
        if request.context:
            prompt += f"Context: {request.context}\n"

        llm_result = self._try_llm_generate(prompt)
        if llm_result:
            return ImplementationResult(
                success=True,
                code=llm_result.get("code"),
                explanation=llm_result.get("explanation", f"Code generated: {request.description}"),
                files_affected=llm_result.get("files_affected", []),
                tests_added=llm_result.get("tests_added", 0),
            )

        return ImplementationResult(
            success=False,
            explanation="LLM inference unavailable",
            warnings=["llm_unavailable"],
        )
    
    def _debug_code(
        self,
        request: ImplementationRequest,
        execution_mode: ExecutionMode,
    ) -> ImplementationResult:
        """Debug and fix code issues."""
        logger.info(f"Debugging code: {request.description}")
        
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

        # Try LLM-based debugging
        prompt = (
            f"Debug the following code and return JSON with keys: "
            f"code, explanation, files_affected.\n\n"
            f"Issue: {request.description}\n"
            f"Code to debug:\n{request.context}\n"
        )
        llm_result = self._try_llm_generate(prompt)
        if llm_result:
            return ImplementationResult(
                success=True,
                code=llm_result.get("code"),
                explanation=llm_result.get("explanation", f"Debugged: {request.description}"),
                files_affected=llm_result.get("files_affected", []),
            )

        return ImplementationResult(
            success=False,
            explanation="LLM inference unavailable",
            warnings=["llm_unavailable"],
        )
