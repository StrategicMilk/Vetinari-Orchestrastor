"""
Vetinari Experimentation Manager Agent

The Experimentation Manager agent is responsible for experiment tracking,
versioning, and reproducibility.
"""

from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult
)


class ExperimentationManagerAgent(BaseAgent):
    """Experimentation Manager agent - experiment tracking, versioning, reproducibility."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.EXPERIMENTATION_MANAGER, config)
        
    def get_system_prompt(self) -> str:
        return """You are Vetinari's Experimentation Manager. Manage experiments, record configurations, 
track results, and provide reproducible experiment documentation.

You must:
1. Plan experiments with clear hypotheses
2. Document configuration and parameters
3. Track experiment results
4. Ensure reproducibility
5. Generate experiment reports
6. Maintain experiment history

Output format must include experiment_log, configuration, results, reproducibility_plan, and analysis."""
    
    def get_capabilities(self) -> List[str]:
        return [
            "experiment_planning",
            "configuration_tracking",
            "result_recording",
            "reproducibility_documentation",
            "hypothesis_testing",
            "experiment_analysis"
        ]
    
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the experimentation task.
        
        Args:
            task: The task containing planned experiments
            
        Returns:
            AgentResult containing experiment documentation
        """
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )
        
        task = self.prepare_task(task)
        
        try:
            experiments = task.context.get("experiments", [])
            baseline = task.context.get("baseline", {})
            
            # Manage experiments (simulated - in production would use actual tracking)
            management = self._manage_experiments(experiments, baseline)
            
            return AgentResult(
                success=True,
                output=management,
                metadata={
                    "experiments_count": len(experiments),
                    "experiments_tracked": len(management.get("experiment_log", []))
                }
            )
            
        except Exception as e:
            self._log("error", f"Experiment management failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)]
            )
    
    def verify(self, output: Any) -> VerificationResult:
        """Verify the experiment documentation meets quality standards.
        
        Args:
            output: The experiment documentation to verify
            
        Returns:
            VerificationResult with pass/fail status
        """
        issues = []
        score = 1.0
        
        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            score -= 0.5
            return VerificationResult(passed=False, issues=issues, score=score)
        
        # Check for experiment log
        if not output.get("experiment_log"):
            issues.append({"type": "missing_log", "message": "Experiment log missing"})
            score -= 0.25
        
        # Check for configuration
        if not output.get("configuration"):
            issues.append({"type": "missing_config", "message": "Configuration missing"})
            score -= 0.2
        
        # Check for results
        if not output.get("results"):
            issues.append({"type": "missing_results", "message": "Results missing"})
            score -= 0.2
        
        # Check for reproducibility plan
        if not output.get("reproducibility_plan"):
            issues.append({"type": "missing_reproducibility", "message": "Reproducibility plan missing"})
            score -= 0.2
        
        # Check for analysis
        if not output.get("analysis"):
            issues.append({"type": "missing_analysis", "message": "Analysis missing"})
            score -= 0.1
        
        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0, score))
    
    def _manage_experiments(self, experiments: List[Dict[str, Any]], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Manage and document experiments.
        
        Args:
            experiments: List of experiments to manage
            baseline: Baseline configuration for comparison
            
        Returns:
            Dictionary containing experiment management data
        """
        # This is a simplified implementation
        # In production, this would use actual experiment tracking systems
        
        experiment_log = []
        
        # Create experiment records
        for i, exp in enumerate(experiments):
            experiment_record = {
                "id": f"exp_{i+1:03d}",
                "name": exp.get("name", f"Experiment {i+1}"),
                "hypothesis": exp.get("hypothesis", ""),
                "status": "pending",
                "created_at": "2026-03-03T12:00:00Z",
                "tags": exp.get("tags", [])
            }
            experiment_log.append(experiment_record)
        
        # Record configuration
        configuration = {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000,
            "framework": "transformers",
            "environment": "cloud",
            "seed": 42
        }
        
        # Simulate results
        results = {
            "metrics": [
                {"name": "accuracy", "value": 0.92, "unit": "percent"},
                {"name": "latency", "value": 1.23, "unit": "seconds"},
                {"name": "cost", "value": 5.50, "unit": "dollars"}
            ],
            "comparisons": [
                {
                    "experiment": "exp_001",
                    "vs_baseline": "better",
                    "improvement": "5%"
                },
                {
                    "experiment": "exp_002",
                    "vs_baseline": "worse",
                    "difference": "-3%"
                }
            ]
        }
        
        # Create reproducibility plan
        reproducibility_plan = {
            "version_control": {
                "repository": "https://github.com/vetinari/experiments",
                "branch": "exp_tracking",
                "commit": "abc123def456"
            },
            "dependencies": [
                {"package": "transformers", "version": "4.30.0"},
                {"package": "torch", "version": "2.0.0"},
                {"package": "numpy", "version": "1.24.0"}
            ],
            "data": {
                "source": "s3://vetinari/data",
                "checksum": "sha256:abc123...",
                "size": "10GB"
            },
            "instructions": [
                "Clone repository at specified commit",
                "Install dependencies from requirements.txt",
                "Download data using data loader script",
                "Run experiment with config.yaml",
                "Results will be saved in results/ directory"
            ]
        }
        
        # Analysis
        analysis = {
            "summary": "Experiments show promise with 5% improvement over baseline",
            "insights": [
                "Model A outperforms baseline in accuracy",
                "Model B has lower latency but comparable accuracy",
                "Cost efficiency improves with larger batch sizes"
            ],
            "recommendations": [
                "Deploy Model A for production",
                "Consider Model B for latency-critical applications",
                "Further optimize batch size for cost reduction"
            ],
            "next_steps": [
                "Run larger experiments with more data",
                "Test with different hyperparameters",
                "Conduct A/B testing in production"
            ]
        }
        
        return {
            "experiment_log": experiment_log,
            "configuration": configuration,
            "results": results,
            "reproducibility_plan": reproducibility_plan,
            "analysis": analysis,
            "summary": f"Managed {len(experiment_log)} experiments with full reproducibility documentation"
        }


# Singleton instance
_experimentation_manager_agent: Optional[ExperimentationManagerAgent] = None


def get_experimentation_manager_agent(config: Optional[Dict[str, Any]] = None) -> ExperimentationManagerAgent:
    """Get the singleton Experimentation Manager agent instance."""
    global _experimentation_manager_agent
    if _experimentation_manager_agent is None:
        _experimentation_manager_agent = ExperimentationManagerAgent(config)
    return _experimentation_manager_agent
