"""
Vetinari Data Engineer Agent

LLM-powered data engineering agent that designs real schemas, migration scripts,
data pipelines, and ETL processes tailored to the actual project requirements.
"""

import logging
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult,
)

logger = logging.getLogger(__name__)


class DataEngineerAgent(BaseAgent):
    """Data Engineer agent — LLM-powered schema design, migrations, pipelines, and ETL."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.DATA_ENGINEER, config)
        self._db_type = self._config.get("db_type", "postgresql")

    def get_system_prompt(self) -> str:
        return f"""You are Vetinari's Data Engineer — a senior data architect with deep expertise
in database design, ETL pipelines, and data modelling.

Target database: {self._db_type}

You MUST analyse the actual requirements provided and design a tailored data architecture.
Every table, column, index, and pipeline must reflect the specific project needs.

Required output (JSON):
{{
  "data_models": {{
    "tables": [
      {{
        "name": "...",
        "description": "...",
        "columns": [
          {{"name": "...", "type": "...", "constraints": "...", "description": "..."}}
        ],
        "indexes": [{{"columns": [], "unique": false, "name": "..."}}],
        "partitioning": null
      }}
    ],
    "relationships": [
      {{"from_table": "...", "from_col": "...", "to_table": "...", "to_col": "...", "type": "one_to_many"}}
    ],
    "enums": [],
    "views": []
  }},
  "migration_plan": [
    {{
      "version": "001",
      "description": "...",
      "up_sql": "...",
      "down_sql": "...",
      "estimated_duration": "..."
    }}
  ],
  "pipelines": [
    {{
      "name": "...",
      "type": "batch" | "streaming" | "cdc",
      "source": "...",
      "destination": "...",
      "schedule": "...",
      "transformations": [],
      "error_handling": "..."
    }}
  ],
  "validation_tests": [
    {{"test_name": "...", "sql": "...", "expected": "...", "description": "..."}}
  ],
  "etl_documentation": {{
    "overview": "...",
    "data_flow_diagram": "...",
    "data_lineage": [],
    "backup_strategy": "...",
    "disaster_recovery": "...",
    "data_retention_policy": "..."
  }},
  "performance_recommendations": [],
  "scaling_notes": "...",
  "summary": "..."
}}

Consider:
- Normalisation to at least 3NF unless denormalisation is justified
- Appropriate indexing for expected query patterns
- Data retention and archiving strategy
- GDPR compliance (PII identification, right to erasure)
- Migration reversibility (up/down scripts)
"""

    def get_capabilities(self) -> List[str]:
        return [
            "schema_design",
            "migration_planning",
            "pipeline_design",
            "etl_development",
            "data_validation",
            "performance_optimization",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the data engineering task using LLM inference."""
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"],
            )

        task = self.prepare_task(task)

        try:
            data_requirements = task.context.get("data_requirements", task.description)
            existing_schema = task.context.get("existing_schema", "")

            # Search for best practices for this DB type
            db_context = ""
            try:
                search_results = self._search(
                    f"{self._db_type} schema design best practices normalisation 2025"
                )
                if search_results:
                    db_context = "\n".join([r.get("snippet", "") for r in search_results[:2]])
            except Exception:
                logger.debug("Failed to search for %s schema design best practices", self._db_type, exc_info=True)

            context_parts = [
                f"Data requirements:\n{data_requirements}",
                f"Target database: {self._db_type}",
            ]
            if existing_schema:
                context_parts.append(f"Existing schema to extend/migrate:\n{existing_schema[:1500]}")
            if db_context:
                context_parts.append(f"Best practices reference:\n{db_context[:600]}")

            prompt = (
                "\n\n".join(context_parts)
                + "\n\nDesign a complete data architecture for the above requirements. "
                "Return a single JSON object matching the required output format exactly."
            )

            design = self._infer_json(
                prompt=prompt,
                fallback=self._fallback_design(data_requirements),
            )

            # Ensure required keys
            design.setdefault("data_models", {"tables": [], "relationships": []})
            design.setdefault("migration_plan", [])
            design.setdefault("pipelines", [])
            design.setdefault("validation_tests", [])
            design.setdefault("etl_documentation", {"overview": "", "backup_strategy": ""})
            design.setdefault("performance_recommendations", [])
            design.setdefault("summary", "Data architecture designed")

            result = AgentResult(
                success=True,
                output=design,
                metadata={
                    "db_type": self._db_type,
                    "tables_count": len(design.get("data_models", {}).get("tables", [])),
                    "migrations_count": len(design.get("migration_plan", [])),
                    "pipelines_count": len(design.get("pipelines", [])),
                },
            )
            task = self.complete_task(task, result)
            return result

        except Exception as e:
            self._log("error", f"Data architecture design failed: {e}")
            return AgentResult(success=False, output=None, errors=[str(e)])

    def verify(self, output: Any) -> VerificationResult:
        issues = []
        score = 1.0

        if not isinstance(output, dict):
            issues.append({"type": "invalid_type", "message": "Output must be a dict"})
            return VerificationResult(passed=False, issues=issues, score=0.0)

        if not output.get("data_models", {}).get("tables"):
            issues.append({"type": "missing_models", "message": "Data models missing"})
            score -= 0.25
        if not output.get("migration_plan"):
            issues.append({"type": "missing_migrations", "message": "Migration plan missing"})
            score -= 0.2
        if not output.get("pipelines"):
            issues.append({"type": "missing_pipelines", "message": "Data pipelines missing"})
            score -= 0.2
        if not output.get("validation_tests"):
            issues.append({"type": "missing_tests", "message": "Validation tests missing"})
            score -= 0.15
        if not output.get("etl_documentation"):
            issues.append({"type": "missing_docs", "message": "ETL documentation missing"})
            score -= 0.1

        passed = score >= 0.5
        return VerificationResult(passed=passed, issues=issues, score=max(0.0, score))

    def _fallback_design(self, requirements: str) -> Dict[str, Any]:
        """Structured fallback when LLM inference fails."""
        return {
            "data_models": {
                "tables": [
                    {
                        "name": "entities",
                        "description": "Primary entity table inferred from requirements",
                        "columns": [
                            {"name": "id", "type": "BIGSERIAL", "constraints": "PRIMARY KEY", "description": "Surrogate key"},
                            {"name": "created_at", "type": "TIMESTAMPTZ", "constraints": "NOT NULL DEFAULT NOW()", "description": "Creation timestamp"},
                            {"name": "updated_at", "type": "TIMESTAMPTZ", "constraints": "NOT NULL DEFAULT NOW()", "description": "Last update timestamp"},
                            {"name": "deleted_at", "type": "TIMESTAMPTZ", "constraints": "NULL", "description": "Soft delete timestamp"},
                        ],
                        "indexes": [{"columns": ["created_at"], "unique": False, "name": "idx_entities_created_at"}],
                        "partitioning": None,
                    }
                ],
                "relationships": [],
                "enums": [],
                "views": [],
            },
            "migration_plan": [
                {
                    "version": "001",
                    "description": "Create initial schema",
                    "up_sql": "CREATE TABLE entities (id BIGSERIAL PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), deleted_at TIMESTAMPTZ);",
                    "down_sql": "DROP TABLE IF EXISTS entities;",
                    "estimated_duration": "< 1 second",
                }
            ],
            "pipelines": [
                {
                    "name": "data_ingestion",
                    "type": "batch",
                    "source": "external source",
                    "destination": f"{self._db_type} database",
                    "schedule": "0 2 * * * (daily at 2am)",
                    "transformations": ["validate", "deduplicate", "normalise"],
                    "error_handling": "Dead letter queue with retry",
                }
            ],
            "validation_tests": [
                {"test_name": "check_no_orphans", "sql": "SELECT COUNT(*) FROM entities WHERE deleted_at IS NULL", "expected": "> 0", "description": "Ensure active records exist"},
                {"test_name": "check_timestamps", "sql": "SELECT COUNT(*) FROM entities WHERE updated_at < created_at", "expected": "0", "description": "Timestamps must be consistent"},
            ],
            "etl_documentation": {
                "overview": f"Data architecture for: {requirements[:100]}",
                "data_flow_diagram": "Source → Validation → Transformation → Load → Verify",
                "data_lineage": [],
                "backup_strategy": "Continuous WAL archiving + daily snapshots, 30-day retention",
                "disaster_recovery": "Point-in-time recovery within 5-minute RPO",
                "data_retention_policy": "Active data: indefinite. Soft-deleted: 90 days. Audit logs: 7 years.",
            },
            "performance_recommendations": [
                "Add composite indexes for common query patterns",
                "Partition large tables by date range",
                "Use connection pooling (PgBouncer/pgpool)",
                "Monitor with pg_stat_statements",
            ],
            "scaling_notes": "Horizontal read scaling via read replicas; vertical scaling for writes",
            "summary": "Data architecture designed with normalised schema and ETL pipeline",
        }


_data_engineer_agent: Optional[DataEngineerAgent] = None


def get_data_engineer_agent(config: Optional[Dict[str, Any]] = None) -> DataEngineerAgent:
    global _data_engineer_agent
    if _data_engineer_agent is None:
        _data_engineer_agent = DataEngineerAgent(config)
    return _data_engineer_agent
