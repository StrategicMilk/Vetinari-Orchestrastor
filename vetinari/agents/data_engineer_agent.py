"""
Vetinari Data Engineer Agent

The Data Engineer agent is responsible for data pipelines, schemas, migrations,
and ETL processes.
"""

from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult
)


class DataEngineerAgent(BaseAgent):
    """Data Engineer agent - data pipelines, schemas, migrations, ETL."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.DATA_ENGINEER, config)
        self._db_type = self._config.get("db_type", "postgresql")
        
    def get_system_prompt(self) -> str:
        return """You are Vetinari's Data Engineer. Design robust data pipelines, schemas, and 
migrations to support the workflow. Output data model designs, migration steps, and validation checks.

You must:
1. Design normalized data schemas
2. Create migration scripts
3. Define data pipelines
4. Generate validation tests
5. Document ETL processes
6. Plan for scalability

Output format must include data_models, migration_plan, pipelines, validation_tests, and etl_documentation."""
    
    def get_capabilities(self) -> List[str]:
        return [
            "schema_design",
            "migration_planning",
            "pipeline_design",
            "etl_development",
            "data_validation",
            "performance_optimization"
        ]
    
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the data engineering task.
        
        Args:
            task: The task containing data requirements
            
        Returns:
            AgentResult containing the data design
        """
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"]
            )
        
        task = self.prepare_task(task)
        
        try:
            data_requirements = task.context.get("data_requirements", task.description)
            
            # Generate data design (simulated - in production would use actual data modeling)
            design = self._design_data_architecture(data_requirements)
            
            return AgentResult(
                success=True,
                output=design,
                metadata={
                    "db_type": self._db_type,
                    "tables_count": len(design.get("data_models", {}).get("tables", [])),
                    "migrations_count": len(design.get("migration_plan", []))
                }
            )
            
        except Exception as e:
            self._log("error", f"Data architecture design failed: {str(e)}")
            return AgentResult(
                success=False,
                output=None,
                errors=[str(e)]
            )
    
    def verify(self, output: Any) -> VerificationResult:
        """Verify the data design meets quality standards.
        
        Args:
            output: The data design to verify
            
        Returns:
            VerificationResult with pass/fail status
        """
        issues = []
        score = 1.0
        
        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            score -= 0.5
            return VerificationResult(passed=False, issues=issues, score=score)
        
        # Check for data models
        if not output.get("data_models"):
            issues.append({"type": "missing_models", "message": "Data models missing"})
            score -= 0.25
        
        # Check for migration plan
        if not output.get("migration_plan"):
            issues.append({"type": "missing_migrations", "message": "Migration plan missing"})
            score -= 0.2
        
        # Check for pipelines
        if not output.get("pipelines"):
            issues.append({"type": "missing_pipelines", "message": "Data pipelines missing"})
            score -= 0.2
        
        # Check for validation tests
        if not output.get("validation_tests"):
            issues.append({"type": "missing_tests", "message": "Validation tests missing"})
            score -= 0.15
        
        # Check for ETL documentation
        if not output.get("etl_documentation"):
            issues.append({"type": "missing_docs", "message": "ETL documentation missing"})
            score -= 0.1
        
        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0, score))
    
    def _design_data_architecture(self, data_requirements: str) -> Dict[str, Any]:
        """Design data architecture based on requirements.
        
        Args:
            data_requirements: The data requirements
            
        Returns:
            Dictionary containing data design
        """
        # This is a simplified implementation
        # In production, this would use actual data modeling techniques
        
        data_models = {
            "tables": [
                {
                    "name": "users",
                    "columns": [
                        {"name": "id", "type": "SERIAL PRIMARY KEY"},
                        {"name": "username", "type": "VARCHAR(255) UNIQUE"},
                        {"name": "email", "type": "VARCHAR(255) UNIQUE"},
                        {"name": "created_at", "type": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"}
                    ]
                },
                {
                    "name": "plans",
                    "columns": [
                        {"name": "id", "type": "SERIAL PRIMARY KEY"},
                        {"name": "user_id", "type": "INTEGER REFERENCES users(id)"},
                        {"name": "title", "type": "VARCHAR(255)"},
                        {"name": "created_at", "type": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"}
                    ]
                }
            ],
            "relationships": [
                {"from": "users", "to": "plans", "type": "one_to_many"}
            ]
        }
        
        migration_plan = [
            {
                "version": "001",
                "description": "Create users table",
                "sql": "CREATE TABLE users (id SERIAL PRIMARY KEY, username VARCHAR(255) UNIQUE, email VARCHAR(255) UNIQUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
            },
            {
                "version": "002",
                "description": "Create plans table",
                "sql": "CREATE TABLE plans (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id), title VARCHAR(255), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);"
            }
        ]
        
        pipelines = [
            {
                "name": "user_import",
                "source": "csv",
                "destination": "users table",
                "transformations": ["validate_email", "hash_password", "check_duplicates"]
            },
            {
                "name": "plan_export",
                "source": "plans table",
                "destination": "json files",
                "transformations": ["format_timestamps", "sanitize_data"]
            }
        ]
        
        validation_tests = [
            {"test": "check_referential_integrity", "description": "Verify foreign keys"},
            {"test": "check_unique_constraints", "description": "Verify unique constraints"},
            {"test": "check_not_null", "description": "Verify NOT NULL constraints"}
        ]
        
        return {
            "data_models": data_models,
            "migration_plan": migration_plan,
            "pipelines": pipelines,
            "validation_tests": validation_tests,
            "etl_documentation": {
                "overview": "Data architecture for plan storage and management",
                "data_flow": "User data -> Validation -> Storage -> Plan generation",
                "backup_strategy": "Daily backups with 30-day retention",
                "disaster_recovery": "Master-slave replication for HA"
            },
            "summary": "Data architecture designed and documented"
        }


# Singleton instance
_data_engineer_agent: Optional[DataEngineerAgent] = None


def get_data_engineer_agent(config: Optional[Dict[str, Any]] = None) -> DataEngineerAgent:
    """Get the singleton Data Engineer agent instance."""
    global _data_engineer_agent
    if _data_engineer_agent is None:
        _data_engineer_agent = DataEngineerAgent(config)
    return _data_engineer_agent
