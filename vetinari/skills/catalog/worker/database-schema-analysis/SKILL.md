---
name: Database Schema Analysis
description: Analyze schemas, query patterns, migration strategies, index coverage, and normalization levels
mode: database
agent: worker
version: "1.0.0"
capabilities:
  - database_analysis
  - code_discovery
tags:
  - research
  - database
  - schema
  - migration
---

# Database Schema Analysis

## Purpose

Database Schema Analysis examines database schemas (SQLite, PostgreSQL, or ORM models) to assess their design quality, query performance characteristics, migration readiness, and alignment with the application's data access patterns. It identifies missing indexes, normalization issues, schema drift between code and database, and opportunities for optimization. For Vetinari, this primarily covers the SQLite-based memory system (memories.db) and any future persistence layers.

## When to Use

- Before adding new tables or columns to an existing database
- When query performance degrades and index coverage needs review
- When planning a schema migration that must preserve existing data
- When the application's data access patterns have changed and the schema may need restructuring
- When auditing the memory system's FTS5 configuration for search quality
- When evaluating whether to normalize or denormalize for a specific access pattern

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| task            | string          | Yes      | What to analyze and why                                            |
| database_path   | string          | No       | Path to the database file or connection string                     |
| schema_files    | list[string]    | No       | ORM model files or SQL schema files to analyze                     |
| query_patterns  | list[string]    | No       | Common queries to optimize for                                     |
| context         | dict            | No       | Application context (read/write ratio, data volume, growth rate)   |

## Process Steps

1. **Schema extraction** -- Read the current schema from the database or ORM model files. Document all tables, columns, types, constraints (NOT NULL, UNIQUE, CHECK), and relationships (foreign keys).

2. **Normalization assessment** -- Evaluate the normalization level of each table. Identify: redundant data (denormalization), update anomalies, insertion anomalies, and deletion anomalies. Recommend normalization changes only when anomalies cause real problems.

3. **Index coverage analysis** -- Map all existing indexes (primary key, unique, composite, FTS). Cross-reference against common query patterns to identify: missing indexes (full table scans), redundant indexes (covered by other indexes), and unused indexes (overhead with no benefit).

4. **Query pattern analysis** -- For each identified query pattern, analyze: execution plan (EXPLAIN), estimated cost, index usage, join strategy, and result set size. Flag queries that perform full table scans on large tables.

5. **FTS configuration review** (if applicable) -- For FTS5 tables, review: tokenizer configuration, column weighting, content sync strategy, and ranking function. Verify that search queries use the FTS index rather than LIKE patterns.

6. **Migration safety assessment** -- If schema changes are proposed, evaluate: data loss risk, backward compatibility, migration duration estimate, rollback strategy, and whether online migration is feasible.

7. **Data integrity audit** -- Check for: orphaned records (foreign key violations in databases without FK enforcement), NULL values in semantically required columns, and data type mismatches between schema and actual data.

8. **Growth projection** -- Based on current data volume and growth rate, project when performance problems will emerge. Identify tables approaching size thresholds where current query patterns will degrade.

9. **Recommendation synthesis** -- Compile findings into prioritized recommendations: critical (data integrity), high (performance), medium (maintenance), low (cosmetic).

## Output Format

The skill produces a schema analysis report:

```json
{
  "success": true,
  "output": {
    "database": ".vetinari/memory/memories.db",
    "tables": [
      {
        "name": "memories",
        "columns": 8,
        "rows_estimate": 1250,
        "indexes": ["PRIMARY KEY (id)", "FTS5 (title, content, tags)"],
        "issues": [
          {"severity": "medium", "issue": "No index on memory_type column, used in 60% of queries"},
          {"severity": "low", "issue": "created_at stored as TEXT, should be INTEGER (epoch) for range queries"}
        ]
      }
    ],
    "query_analysis": [
      {
        "query": "SELECT * FROM memories WHERE memory_type = ? ORDER BY created_at DESC",
        "plan": "SCAN TABLE memories",
        "issue": "Full table scan, no index on memory_type",
        "fix": "CREATE INDEX idx_memories_type_date ON memories(memory_type, created_at DESC)"
      }
    ],
    "recommendations": [
      {"priority": "high", "action": "Add composite index on (memory_type, created_at)"},
      {"priority": "medium", "action": "Migrate created_at from TEXT to INTEGER epoch"},
      {"priority": "low", "action": "Add CHECK constraint on memory_type column"}
    ]
  },
  "provenance": [
    {"source": ".vetinari/memory/memories.db", "method": "sqlite3 schema inspection"},
    {"source": "vetinari/memory/", "method": "ORM model analysis"}
  ]
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-001**: Research modes MUST cite sources -- file paths, URLs, or commit SHAs
- **STD-WRK-007**: Database schemas MUST include indexes, constraints, and migration strategy
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **CON-WRK-001**: Research modes are READ-ONLY -- MUST NOT modify production files

## Examples

### Example: Pre-migration analysis for adding a new column

**Input:**
```
task: "Analyze the memories table before adding an 'embedding' BLOB column for vector search"
database_path: ".vetinari/memory/memories.db"
query_patterns: ["SELECT id, embedding FROM memories WHERE memory_type = ?", "nearest-neighbor search on embedding column"]
```

**Output (abbreviated):**
```
schema: memories table has 8 columns, 1250 rows, FTS5 index on text fields

migration_assessment:
  risk: "low"
  strategy: "ALTER TABLE ADD COLUMN (SQLite supports this without table rebuild)"
  data_loss_risk: "none (additive change)"
  backward_compatible: true
  estimated_duration: "<1 second for 1250 rows"
  rollback: "ALTER TABLE DROP COLUMN (SQLite 3.35+)"

recommendations:
  1. [high] "Add embedding as BLOB with NULL default -- populate asynchronously after migration"
  2. [medium] "Do NOT add a standard index on embedding -- use a separate vector index table or external library"
  3. [low] "Consider a separate embeddings table with FK to memories for cleaner separation of concerns"
```
