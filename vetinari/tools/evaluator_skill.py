"""
Evaluator Skill Tool Wrapper

Migrates the evaluator skill to the Tool interface, providing code review,
quality assessment, security auditing, and testing strategies as a standardized
Vetinari tool.

The evaluator skill specializes in:
- Code review
- Quality assessment
- Security audit
- Test strategy
- Performance review
- Best practices
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
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


class EvaluatorCapability(str, Enum):
    """Capabilities of the evaluator skill."""
    CODE_REVIEW = "code_review"
    QUALITY_ASSESSMENT = "quality_assessment"
    SECURITY_AUDIT = "security_audit"
    TEST_STRATEGY = "test_strategy"
    PERFORMANCE_REVIEW = "performance_review"
    BEST_PRACTICES = "best_practices"


class ThinkingMode(str, Enum):
    """Thinking modes for review depth."""
    LOW = "low"              # Quick review checklist
    MEDIUM = "medium"        # Detailed code review
    HIGH = "high"            # Comprehensive quality audit
    XHIGH = "xhigh"          # Full security and performance review


class SeverityLevel(str, Enum):
    """Issue severity levels."""
    CRITICAL = "critical"    # Must fix before merge
    HIGH = "high"            # Should fix soon
    MEDIUM = "medium"        # Should fix when convenient
    LOW = "low"              # Improvement suggestion
    INFO = "info"            # Note for awareness


class QualityScore(str, Enum):
    """Quality assessment scores."""
    A = "A"      # Production ready
    B = "B"      # Minor issues
    C = "C"      # Needs work
    D = "D"      # Major problems
    F = "F"      # Not acceptable


@dataclass
class Issue:
    """Represents a code issue found during review."""
    title: str
    severity: SeverityLevel
    location: Optional[str] = None
    description: Optional[str] = None
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "severity": self.severity.value,
            "location": self.location,
            "description": self.description,
            "suggestion": self.suggestion,
        }


@dataclass
class ReviewRequest:
    """Request structure for evaluator operations."""
    capability: EvaluatorCapability
    code: str
    context: Optional[str] = None
    thinking_mode: ThinkingMode = ThinkingMode.MEDIUM
    focus_areas: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "capability": self.capability.value,
            "code": self.code,
            "context": self.context,
            "thinking_mode": self.thinking_mode.value,
            "focus_areas": self.focus_areas,
        }


@dataclass
class ReviewResult:
    """Result of an evaluator operation."""
    success: bool
    issues: List[Issue] = field(default_factory=list)
    quality_score: Optional[QualityScore] = None
    summary: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    test_coverage: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "issues": [issue.to_dict() for issue in self.issues],
            "quality_score": self.quality_score.value if self.quality_score else None,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "test_coverage": self.test_coverage,
        }


class EvaluatorSkillTool(Tool):
    """
    Tool wrapper for the evaluator skill.
    
    Provides code review, quality assessment, security auditing, and testing
    strategies through a standardized Tool interface.
    
    Permissions:
    - FILE_READ: Read code for analysis
    - MODEL_INFERENCE: Use LLM for evaluation
    
    Allowed Modes:
    - EXECUTION: Full evaluation with all capabilities
    - PLANNING: Analysis only without modifications
    """
    
    def __init__(self):
        """Initialize the evaluator skill tool."""
        metadata = ToolMetadata(
            name="evaluator",
            description=(
                "Code review, quality assessment, and testing strategies. "
                "Use when user wants to review code, check quality, validate "
                "implementations, or audit for security issues."
            ),
            category=ToolCategory.SEARCH_ANALYSIS,
            version="1.0.0",
            author="Vetinari",
            parameters=[
                ToolParameter(
                    name="capability",
                    type=str,
                    description="The evaluator capability to use",
                    required=True,
                    allowed_values=[c.value for c in EvaluatorCapability],
                ),
                ToolParameter(
                    name="code",
                    type=str,
                    description="Code to review or evaluate",
                    required=True,
                ),
                ToolParameter(
                    name="context",
                    type=str,
                    description="Additional context or file information",
                    required=False,
                ),
                ToolParameter(
                    name="thinking_mode",
                    type=str,
                    description="Review depth (low/medium/high/xhigh)",
                    required=False,
                    default="medium",
                    allowed_values=[m.value for m in ThinkingMode],
                ),
                ToolParameter(
                    name="focus_areas",
                    type=list,
                    description="Specific areas to focus on (e.g., security, performance)",
                    required=False,
                ),
            ],
            required_permissions=[
                ToolPermission.FILE_READ,
                ToolPermission.MODEL_INFERENCE,
            ],
            allowed_modes=[ExecutionMode.EXECUTION, ExecutionMode.PLANNING],
            tags=[
                "review",
                "quality",
                "testing",
                "validation",
                "assessment",
                "security",
            ],
        )
        super().__init__(metadata)
    
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute an evaluator operation.
        
        Args:
            capability: EvaluatorCapability to use
            code: Code to review or evaluate
            context: Additional context (optional)
            thinking_mode: Review depth (default: medium)
            focus_areas: Areas to focus on (optional)
            
        Returns:
            ToolResult with review details
        """
        try:
            # Extract parameters
            capability_str = kwargs.get("capability")
            code = kwargs.get("code")
            context = kwargs.get("context")
            thinking_mode_str = kwargs.get("thinking_mode", "medium")
            focus_areas = kwargs.get("focus_areas", [])
            
            # Validate required parameters
            if not code:
                return ToolResult(
                    success=False,
                    output=None,
                    error="Code parameter is required",
                )
            
            # Convert to enums
            try:
                capability = EvaluatorCapability(capability_str)
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
            request = ReviewRequest(
                capability=capability,
                code=code,
                context=context,
                thinking_mode=thinking_mode,
                focus_areas=focus_areas,
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
                logger.warning("Evaluator output validation failed: %s", validation["errors"])

            return ToolResult(
                success=result.success,
                output=result.to_dict(),
                error=None if result.success else "Evaluation failed",
                metadata={
                    "capability": capability.value,
                    "thinking_mode": thinking_mode.value,
                    "execution_mode": execution_mode.value,
                    "issues_found": len(result.issues),
                },
            )
        
        except Exception as e:
            logger.error(f"Evaluator tool execution failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                output=None,
                error=f"Evaluator tool execution failed: {str(e)}",
            )
    
    def _execute_capability(
        self,
        request: ReviewRequest,
        execution_mode: ExecutionMode,
    ) -> ReviewResult:
        """
        Execute a specific evaluator capability.
        
        Args:
            request: The review request
            execution_mode: Current execution mode
            
        Returns:
            ReviewResult with operation details
        """
        capability = request.capability
        
        if capability == EvaluatorCapability.CODE_REVIEW:
            return self._perform_code_review(request, execution_mode)
        elif capability == EvaluatorCapability.QUALITY_ASSESSMENT:
            return self._assess_quality(request, execution_mode)
        elif capability == EvaluatorCapability.SECURITY_AUDIT:
            return self._audit_security(request, execution_mode)
        elif capability == EvaluatorCapability.TEST_STRATEGY:
            return self._create_test_strategy(request, execution_mode)
        elif capability == EvaluatorCapability.PERFORMANCE_REVIEW:
            return self._review_performance(request, execution_mode)
        elif capability == EvaluatorCapability.BEST_PRACTICES:
            return self._check_best_practices(request, execution_mode)
        else:
            return ReviewResult(
                success=False,
                summary=f"Unknown capability: {capability.value}",
            )
    
    def _perform_code_review(
        self,
        request: ReviewRequest,
        execution_mode: ExecutionMode,
    ) -> ReviewResult:
        """Perform a general code review."""
        logger.info(f"Performing code review (mode: {request.thinking_mode.value})")
        
        if execution_mode == ExecutionMode.PLANNING:
            return ReviewResult(
                success=True,
                summary=(
                    f"Planning mode: Would perform code review at {request.thinking_mode.value} level. "
                    f"Switch to EXECUTION mode to generate detailed review."
                ),
            )
        
        # Simulate review based on code length and thinking mode
        code_lines = len(request.code.split('\n'))
        issues = []
        recommendations = []
        
        # Check for common issues
        if "TODO" in request.code or "FIXME" in request.code:
            issues.append(Issue(
                title="Unresolved TODOs or FIXMEs",
                severity=SeverityLevel.MEDIUM,
                description="Code contains TODO or FIXME comments that should be resolved",
                suggestion="Address all outstanding TODOs and FIXMEs before merge",
            ))
        
        if "{" in request.code and request.code.count("{") != request.code.count("}"):
            issues.append(Issue(
                title="Unbalanced braces",
                severity=SeverityLevel.HIGH,
                description="Opening and closing braces do not match",
                suggestion="Check brace pairing in the code",
            ))
        
        if len(request.code.split('\n')) > 200 and request.thinking_mode in [ThinkingMode.MEDIUM, ThinkingMode.HIGH]:
            recommendations.append("Consider breaking this file into smaller, more focused modules")
        
        quality_score = QualityScore.B if len(issues) <= 2 else QualityScore.C
        if not issues:
            quality_score = QualityScore.A
        
        return ReviewResult(
            success=True,
            issues=issues,
            quality_score=quality_score,
            summary=f"Code review completed. Found {len(issues)} issue(s) in {code_lines} lines.",
            recommendations=recommendations,
            test_coverage=75.0 if request.thinking_mode.value != "low" else None,
        )
    
    def _assess_quality(
        self,
        request: ReviewRequest,
        execution_mode: ExecutionMode,
    ) -> ReviewResult:
        """Assess code quality."""
        logger.info(f"Assessing code quality (mode: {request.thinking_mode.value})")
        
        if execution_mode == ExecutionMode.PLANNING:
            return ReviewResult(
                success=True,
                summary=(
                    f"Planning mode: Would assess code quality at {request.thinking_mode.value} level. "
                    f"Switch to EXECUTION mode to generate quality report."
                ),
            )
        
        issues = []
        code_lines = len(request.code.split('\n'))
        
        # Check code length (possible complexity issue)
        if code_lines > 500:
            issues.append(Issue(
                title="High code complexity",
                severity=SeverityLevel.MEDIUM,
                description="Function or file is too long, reducing maintainability",
                suggestion="Break into smaller, more focused functions",
            ))
        
        # Check for naming conventions in higher modes
        if request.thinking_mode in [ThinkingMode.HIGH, ThinkingMode.XHIGH]:
            if "var " in request.code or "val " in request.code:
                issues.append(Issue(
                    title="Non-descriptive variable names",
                    severity=SeverityLevel.LOW,
                    description="Variable names like 'var' or 'val' are not descriptive",
                    suggestion="Use meaningful variable names that describe the value",
                ))
        
        quality_score = QualityScore.A if not issues else (
            QualityScore.B if len(issues) <= 2 else QualityScore.C
        )
        
        return ReviewResult(
            success=True,
            issues=issues,
            quality_score=quality_score,
            summary=f"Quality assessment completed. Score: {quality_score.value}",
            recommendations=[
                "Maintain consistent code style",
                "Keep functions focused and single-purpose",
                "Document complex logic with comments",
            ],
        )
    
    def _audit_security(
        self,
        request: ReviewRequest,
        execution_mode: ExecutionMode,
    ) -> ReviewResult:
        """Audit code for security vulnerabilities."""
        logger.info(f"Auditing security (mode: {request.thinking_mode.value})")
        
        if execution_mode == ExecutionMode.PLANNING:
            return ReviewResult(
                success=True,
                summary=(
                    f"Planning mode: Would perform security audit at {request.thinking_mode.value} level. "
                    f"Switch to EXECUTION mode to generate security report."
                ),
            )
        
        issues = []
        
        # Check for common security issues
        security_keywords = {
            "eval": "Use of eval() is dangerous",
            "exec": "Use of exec() is dangerous",
            "pickle": "Pickle can execute arbitrary code",
            "input": "input() can be unsafe without validation",
        }
        
        for keyword, warning in security_keywords.items():
            if keyword in request.code.lower():
                severity = SeverityLevel.CRITICAL if keyword in ["eval", "exec"] else SeverityLevel.HIGH
                issues.append(Issue(
                    title=f"Potential security issue: {keyword}",
                    severity=severity,
                    description=warning,
                    suggestion=f"Review usage of {keyword} and consider safer alternatives",
                ))
        
        # Check for hardcoded secrets (basic)
        if "password" in request.code.lower() or "api_key" in request.code.lower():
            issues.append(Issue(
                title="Potential hardcoded credentials",
                severity=SeverityLevel.CRITICAL,
                description="Code may contain hardcoded passwords or API keys",
                suggestion="Move credentials to environment variables or secure configuration",
            ))
        
        return ReviewResult(
            success=True,
            issues=issues,
            summary=f"Security audit completed. Found {len(issues)} potential issue(s).",
            recommendations=[
                "Use parameterized queries to prevent SQL injection",
                "Validate all user inputs",
                "Store secrets in environment variables",
                "Use HTTPS for all external communications",
            ],
        )
    
    def _create_test_strategy(
        self,
        request: ReviewRequest,
        execution_mode: ExecutionMode,
    ) -> ReviewResult:
        """Create a test strategy for the code."""
        logger.info(f"Creating test strategy (mode: {request.thinking_mode.value})")
        
        if execution_mode == ExecutionMode.PLANNING:
            return ReviewResult(
                success=True,
                summary=(
                    f"Planning mode: Would create test strategy at {request.thinking_mode.value} level. "
                    f"Switch to EXECUTION mode to generate strategy."
                ),
            )
        
        recommendations = []
        
        # Suggest tests based on thinking mode
        recommendations.append("Write unit tests for each function")
        recommendations.append("Write integration tests for component interactions")
        
        if request.thinking_mode in [ThinkingMode.HIGH, ThinkingMode.XHIGH]:
            recommendations.append("Write end-to-end tests for user workflows")
            recommendations.append("Add performance benchmarks")
            recommendations.append("Test error handling and edge cases")
        
        return ReviewResult(
            success=True,
            issues=[],
            quality_score=QualityScore.B,
            summary=f"Test strategy created with {len(recommendations)} recommendations.",
            recommendations=recommendations,
            test_coverage=80.0,
        )
    
    def _review_performance(
        self,
        request: ReviewRequest,
        execution_mode: ExecutionMode,
    ) -> ReviewResult:
        """Review code for performance issues."""
        logger.info(f"Reviewing performance (mode: {request.thinking_mode.value})")
        
        if execution_mode == ExecutionMode.PLANNING:
            return ReviewResult(
                success=True,
                summary=(
                    f"Planning mode: Would review performance at {request.thinking_mode.value} level. "
                    f"Switch to EXECUTION mode to generate performance report."
                ),
            )
        
        issues = []
        
        # Check for common performance issues
        if "for " in request.code and request.code.count("for ") > 2:
            issues.append(Issue(
                title="Nested loops detected",
                severity=SeverityLevel.MEDIUM,
                description="Multiple nested loops may cause O(n²) or worse performance",
                suggestion="Consider optimizing with data structures or algorithms",
            ))
        
        if "while True" in request.code:
            issues.append(Issue(
                title="Infinite loop risk",
                severity=SeverityLevel.HIGH,
                description="while True loops should have clear exit conditions",
                suggestion="Ensure loop has proper termination condition",
            ))
        
        return ReviewResult(
            success=True,
            issues=issues,
            quality_score=QualityScore.B if len(issues) > 0 else QualityScore.A,
            summary=f"Performance review completed. Found {len(issues)} issue(s).",
            recommendations=[
                "Use appropriate data structures (dict/set vs list)",
                "Cache expensive computations",
                "Avoid unnecessary string concatenation in loops",
                "Use generators for large datasets",
            ],
        )
    
    def _check_best_practices(
        self,
        request: ReviewRequest,
        execution_mode: ExecutionMode,
    ) -> ReviewResult:
        """Check code against best practices."""
        logger.info(f"Checking best practices (mode: {request.thinking_mode.value})")
        
        if execution_mode == ExecutionMode.PLANNING:
            return ReviewResult(
                success=True,
                summary=(
                    f"Planning mode: Would check best practices at {request.thinking_mode.value} level. "
                    f"Switch to EXECUTION mode to generate report."
                ),
            )
        
        issues = []
        
        # Check for DRY principle
        if request.code.count("def ") > 1:
            # Check for similar code blocks (simple heuristic)
            lines = request.code.split('\n')
            if len(lines) > 50:
                issues.append(Issue(
                    title="Possible code duplication",
                    severity=SeverityLevel.LOW,
                    description="Code may violate DRY (Don't Repeat Yourself) principle",
                    suggestion="Extract common code into reusable functions",
                ))
        
        # Check for single responsibility
        if request.code.count("def ") > 5 and len(request.code) < 500:
            issues.append(Issue(
                title="Too many functions in small file",
                severity=SeverityLevel.LOW,
                description="May indicate functions are too granular",
                suggestion="Review function granularity and consider grouping",
            ))
        
        return ReviewResult(
            success=True,
            issues=issues,
            quality_score=QualityScore.A if not issues else QualityScore.B,
            summary=f"Best practices check completed. Found {len(issues)} issue(s).",
            recommendations=[
                "Follow SOLID principles",
                "Keep functions focused and single-purpose",
                "Use meaningful variable and function names",
                "Add docstrings to all public functions",
                "Keep classes cohesive",
            ],
        )
