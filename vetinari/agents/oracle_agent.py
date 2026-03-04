"""
Vetinari Oracle Agent

The Oracle agent is responsible for architectural decisions, risk assessment,
and high-level debugging strategies.
"""

from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult
)


class OracleAgent(BaseAgent):
    """Oracle agent - architectural decisions, risk assessment, debugging."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.ORACLE, config)
        self._max_risks = self._config.get("max_risks", 5)
        
    def get_system_prompt(self) -> str:
        return """You are Vetinari's Oracle. Your role is to provide architectural guidance, 
assess risks, and propose robust designs for complex tasks.

You must:
1. Review architectural options and propose the best approach
2. Identify potential risks with likelihood and impact
3. Provide mitigation strategies for each risk
4. Outline trade-offs clearly
5. Give recommended guidelines for implementation

Output format must include architecture_vision, risks array, and recommended_guidelines."""
    
    def get_capabilities(self) -> List[str]:
        return [
            "architecture_design",
            "risk_assessment",
            "tradeoff_analysis",
            "debugging_strategy",
            "technical_guidance"
        ]
    
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the oracle task.
        
        Args:
            task: The task containing the architectural question
            
        Returns:
            AgentResult containing the architectural guidance
        """
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )
        
        task = self.prepare_task(task)
        
        try:
            goal = task.prompt or task.description
            context = task.context or {}
            
            # Generate architectural guidance
            guidance = self._provide_guidance(goal, context)
            
            return AgentResult(
                success=True,
                output=guidance,
                metadata={
                    "risks_identified": len(guidance.get("risks", [])),
                    "guidelines_count": len(guidance.get("recommended_guidelines", []))
                }
            )
            
        except Exception as e:
            self._log("error", f"Oracle guidance failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)]
            )
    
    def verify(self, output: Any) -> VerificationResult:
        """Verify the architectural guidance meets quality standards.
        
        Args:
            output: The guidance to verify
            
        Returns:
            VerificationResult with pass/fail status
        """
        issues = []
        score = 1.0
        
        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            score -= 0.5
            return VerificationResult(passed=False, issues=issues, score=score)
        
        # Check for required fields
        if "architecture_vision" not in output:
            issues.append({"type": "missing_vision", "message": "No architecture vision provided"})
            score -= 0.3
        
        risks = output.get("risks", [])
        if not risks:
            issues.append({"type": "no_risks", "message": "No risks identified"})
            score -= 0.2
        
        guidelines = output.get("recommended_guidelines", [])
        if not guidelines:
            issues.append({"type": "no_guidelines", "message": "No guidelines provided"})
            score -= 0.2
        
        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0, score))
    
    def _provide_guidance(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide architectural guidance for the goal.
        
        Args:
            goal: The architectural question
            context: Additional context
            
        Returns:
            Dictionary containing guidance
        """
        goal_lower = goal.lower()
        
        # Analyze the goal to determine architecture type
        architecture_type = self._determine_architecture(goal_lower)
        
        # Generate vision
        vision = self._generate_vision(architecture_type, goal_lower)
        
        # Identify risks
        risks = self._identify_risks(architecture_type, goal_lower)
        
        # Generate guidelines
        guidelines = self._generate_guidelines(architecture_type, goal_lower)
        
        return {
            "architecture_vision": vision,
            "architecture_type": architecture_type,
            "risks": risks,
            "recommended_guidelines": guidelines,
            "constraints": context.get("constraints", {})
        }
    
    def _determine_architecture(self, goal: str) -> str:
        """Determine the type of architecture needed.
        
        Args:
            goal: The goal description
            
        Returns:
            Architecture type
        """
        if "api" in goal or "rest" in goal:
            return "api_service"
        elif "web" in goal or "frontend" in goal:
            return "web_application"
        elif "data" in goal or "database" in goal:
            return "data_platform"
        elif "microservice" in goal:
            return "microservice"
        else:
            return "general"
    
    def _generate_vision(self, architecture_type: str, goal: str) -> str:
        """Generate an architecture vision.
        
        Args:
            architecture_type: Type of architecture
            goal: The goal
            
        Returns:
            Architecture vision string
        """
        visions = {
            "api_service": "A clean RESTful API with proper separation of concerns, using a service layer pattern for business logic and a repository pattern for data access.",
            "web_application": "A modern single-page application with a separate backend API, using component-based architecture and state management.",
            "data_platform": "A scalable data platform with proper ETL pipelines, data warehousing, and analytics capabilities.",
            "microservice": "A distributed microservice architecture with API gateway, service discovery, and proper inter-service communication.",
            "general": "A modular, maintainable codebase with clear separation of concerns, proper testing coverage, and documentation."
        }
        return visions.get(architecture_type, visions["general"])
    
    def _identify_risks(self, architecture_type: str, goal: str) -> List[Dict[str, Any]]:
        """Identify potential risks.
        
        Args:
            architecture_type: Type of architecture
            goal: The goal
            
        Returns:
            List of risk dictionaries
        """
        common_risks = [
            {
                "risk": "Scope creep",
                "likelihood": 0.6,
                "impact": 0.7,
                "mitigation": "Define clear MVP requirements and prioritize features"
            },
            {
                "risk": "Technical debt accumulation",
                "likelihood": 0.5,
                "impact": 0.6,
                "mitigation": "Schedule regular refactoring sprints"
            },
            {
                "risk": "Integration challenges",
                "likelihood": 0.4,
                "impact": 0.7,
                "mitigation": "Define clear API contracts early and use contract testing"
            }
        ]
        
        # Add architecture-specific risks
        if architecture_type == "microservice":
            common_risks.extend([
                {
                    "risk": "Service communication overhead",
                    "likelihood": 0.5,
                    "impact": 0.5,
                    "mitigation": "Use asynchronous messaging where possible"
                },
                {
                    "risk": "Distributed system complexity",
                    "likelihood": 0.6,
                    "impact": 0.8,
                    "mitigation": "Implement proper observability and tracing"
                }
            ])
        
        return common_risks[:self._max_risks]
    
    def _generate_guidelines(self, architecture_type: str, goal: str) -> List[str]:
        """Generate recommended guidelines.
        
        Args:
            architecture_type: Type of architecture
            goal: The goal
            
        Returns:
            List of guideline strings
        """
        base_guidelines = [
            "Use version control from day one",
            "Write automated tests for all new code",
            "Use continuous integration and deployment",
            "Document architectural decisions",
            "Use code reviews for all changes"
        ]
        
        if architecture_type == "api_service":
            base_guidelines.extend([
                "Follow RESTful conventions",
                "Use OpenAPI/Swagger for API documentation",
                "Implement proper error handling and HTTP status codes"
            ])
        elif architecture_type == "web_application":
            base_guidelines.extend([
                "Use a component library for consistency",
                "Implement proper state management",
                "Optimize for performance"
            ])
        
        return base_guidelines


# Singleton instance
_oracle_agent: Optional[OracleAgent] = None


def get_oracle_agent(config: Optional[Dict[str, Any]] = None) -> OracleAgent:
    """Get the singleton Oracle agent instance."""
    global _oracle_agent
    if _oracle_agent is None:
        _oracle_agent = OracleAgent(config)
    return _oracle_agent
