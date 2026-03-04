"""
Vetinari Cost Planner Agent

The Cost Planner agent is responsible for cost accounting, compute usage optimization,
and model selection guidance.
"""

from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult
)


class CostPlannerAgent(BaseAgent):
    """Cost Planner agent - cost accounting, usage optimization, model selection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.COST_PLANNER, config)
        
    def get_system_prompt(self) -> str:
        return """You are Vetinari's Cost Planner. Track compute costs and model usage; advise 
cost-aware model selection and plan cost targets.

You must:
1. Calculate task execution costs
2. Track model usage patterns
3. Recommend cost-efficient models
4. Set budget constraints
5. Generate cost reports
6. Identify optimization opportunities

Output format must include cost_report, model_recommendations, budget_constraints, and optimizations."""
    
    def get_capabilities(self) -> List[str]:
        return [
            "cost_calculation",
            "budget_planning",
            "model_selection",
            "usage_tracking",
            "cost_reporting",
            "optimization_analysis"
        ]
    
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the cost planning task.
        
        Args:
            task: The task containing plan outputs and usage stats
            
        Returns:
            AgentResult containing cost analysis
        """
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )
        
        task = self.prepare_task(task)
        
        try:
            plan_outputs = task.context.get("plan_outputs", task.description)
            usage_stats = task.context.get("usage_stats", {})
            
            # Perform cost analysis (simulated - in production would use actual cost data)
            analysis = self._analyze_costs(plan_outputs, usage_stats)
            
            return AgentResult(
                success=True,
                output=analysis,
                metadata={
                    "total_cost": analysis.get("cost_report", {}).get("total_cost", 0),
                    "recommendations_count": len(analysis.get("model_recommendations", []))
                }
            )
            
        except Exception as e:
            self._log("error", f"Cost analysis failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)]
            )
    
    def verify(self, output: Any) -> VerificationResult:
        """Verify the cost analysis output meets quality standards.
        
        Args:
            output: The cost analysis to verify
            
        Returns:
            VerificationResult with pass/fail status
        """
        issues = []
        score = 1.0
        
        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            score -= 0.5
            return VerificationResult(passed=False, issues=issues, score=score)
        
        # Check for cost report
        if not output.get("cost_report"):
            issues.append({"type": "missing_report", "message": "Cost report missing"})
            score -= 0.25
        
        # Check for model recommendations
        if not output.get("model_recommendations"):
            issues.append({"type": "missing_recommendations", "message": "Model recommendations missing"})
            score -= 0.2
        
        # Check for budget constraints
        if not output.get("budget_constraints"):
            issues.append({"type": "missing_constraints", "message": "Budget constraints missing"})
            score -= 0.2
        
        # Check for optimizations
        if not output.get("optimizations"):
            issues.append({"type": "missing_optimizations", "message": "Optimization suggestions missing"})
            score -= 0.15
        
        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0, score))
    
    def _analyze_costs(self, plan_outputs: str, usage_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze costs for plan execution.
        
        Args:
            plan_outputs: Description of plan outputs
            usage_stats: Usage statistics
            
        Returns:
            Dictionary containing cost analysis
        """
        # This is a simplified implementation
        # In production, this would use actual cost data from providers
        
        # Simulate cost calculation
        cost_breakdown = {
            "planning_phase": 0.05,
            "exploration_phase": 0.15,
            "analysis_phase": 0.25,
            "implementation_phase": 0.45,
            "testing_phase": 0.10
        }
        
        total_cost = sum(cost_breakdown.values())
        
        cost_report = {
            "breakdown": cost_breakdown,
            "total_cost": round(total_cost, 2),
            "currency": "USD",
            "period": "per execution",
            "notes": "Estimated costs based on average usage"
        }
        
        model_recommendations = [
            {
                "task": "Planning",
                "current_model": "gpt-4",
                "recommended_model": "gpt-3.5-turbo",
                "cost_savings": "60%",
                "reasoning": "gpt-3.5-turbo sufficient for planning task"
            },
            {
                "task": "Code Generation",
                "current_model": "gpt-4",
                "recommended_model": "gpt-4",
                "cost_savings": "0%",
                "reasoning": "gpt-4 necessary for code generation quality"
            },
            {
                "task": "Documentation",
                "current_model": "gpt-4",
                "recommended_model": "gpt-3.5-turbo",
                "cost_savings": "60%",
                "reasoning": "gpt-3.5-turbo adequate for documentation"
            }
        ]
        
        budget_constraints = {
            "monthly_budget": 1000.0,
            "per_execution_budget": 5.0,
            "high_priority_allocation": 0.7,
            "experimentation_allocation": 0.3
        }
        
        optimizations = [
            {
                "optimization": "Model selection",
                "potential_savings": "$150/month",
                "effort": "Low",
                "description": "Use smaller models for non-critical tasks"
            },
            {
                "optimization": "Caching",
                "potential_savings": "$100/month",
                "effort": "Medium",
                "description": "Cache common exploration results"
            },
            {
                "optimization": "Batch processing",
                "potential_savings": "$80/month",
                "effort": "Medium",
                "description": "Batch similar analysis tasks"
            }
        ]
        
        return {
            "cost_report": cost_report,
            "model_recommendations": model_recommendations,
            "budget_constraints": budget_constraints,
            "optimizations": optimizations,
            "summary": f"Cost analysis complete. Total estimated cost: ${total_cost:.2f}. Potential savings: $330/month"
        }


# Singleton instance
_cost_planner_agent: Optional[CostPlannerAgent] = None


def get_cost_planner_agent(config: Optional[Dict[str, Any]] = None) -> CostPlannerAgent:
    """Get the singleton Cost Planner agent instance."""
    global _cost_planner_agent
    if _cost_planner_agent is None:
        _cost_planner_agent = CostPlannerAgent(config)
    return _cost_planner_agent
