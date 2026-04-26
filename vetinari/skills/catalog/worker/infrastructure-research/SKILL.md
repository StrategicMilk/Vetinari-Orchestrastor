---
name: Infrastructure Research
description: Research CI/CD configurations, Docker setups, deployment strategies, and infrastructure patterns
mode: devops
agent: worker
version: "1.0.0"
capabilities:
  - infrastructure_research
  - code_discovery
tags:
  - research
  - infrastructure
  - ci-cd
  - devops
  - deployment
---

# Infrastructure Research

## Purpose

Infrastructure Research investigates CI/CD pipelines, containerization configurations, deployment strategies, and infrastructure-as-code patterns to inform operational decisions. It examines existing infrastructure setups, evaluates alternatives, and produces recommendations that account for reliability, cost, and operational complexity. This skill ensures that infrastructure decisions are made with full understanding of trade-offs rather than defaulting to familiar but suboptimal patterns.

## When to Use

- Before setting up or modifying CI/CD pipelines (GitHub Actions, GitLab CI, etc.)
- When evaluating containerization strategies (Docker, docker-compose, Kubernetes)
- When designing deployment workflows (blue-green, canary, rolling updates)
- When investigating build failures or slow CI pipelines
- When planning infrastructure for a new service or microservice
- When evaluating hosting options and their trade-offs

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| task            | string          | Yes      | What infrastructure to research and why                            |
| files           | list[string]    | No       | Infrastructure files to analyze (Dockerfile, CI configs, etc.)     |
| platform        | string          | No       | Target platform (e.g., "github-actions", "docker", "kubernetes")   |
| constraints     | list[string]    | No       | Infrastructure constraints (budget, region, compliance)            |
| context         | dict            | No       | Current infrastructure state and requirements                     |

## Process Steps

1. **Current state assessment** -- Inventory existing infrastructure configurations: CI/CD pipelines, Dockerfiles, deployment scripts, monitoring configs. Document what exists, what works, and what is missing.

2. **Configuration analysis** -- For each infrastructure file, analyze: correctness (valid syntax, no deprecated features), efficiency (build caching, layer optimization, parallelism), security (no secrets in configs, minimal base images, least privilege).

3. **Pipeline flow mapping** -- Map the CI/CD pipeline from trigger to deployment: stages, jobs, dependencies, artifacts, caching strategy. Identify bottlenecks (slow steps), single points of failure (no fallback), and waste (steps that always pass).

4. **Build optimization analysis** -- Analyze build times and identify optimization opportunities: Docker layer caching, dependency caching, parallel job execution, conditional stage execution, and incremental builds.

5. **Security posture review** -- Check for: secrets exposed in configs, overly permissive IAM roles, unscanned base images, missing vulnerability scanning steps, and unsigned artifacts.

6. **Reliability assessment** -- Evaluate: retry mechanisms for flaky steps, health checks in deployment, rollback procedures, monitoring and alerting integration, and disaster recovery readiness.

7. **Cost analysis** -- Estimate infrastructure costs: compute minutes, storage, egress, and any paid services. Compare against alternatives that could reduce costs without sacrificing reliability.

8. **Best practice comparison** -- Compare the current setup against industry best practices for the platform. Note gaps and prioritize by impact.

9. **Recommendation synthesis** -- Compile findings into actionable recommendations with effort estimates and expected impact.

## Output Format

The skill produces an infrastructure analysis report:

```json
{
  "success": true,
  "output": {
    "current_state": {
      "ci_platform": "GitHub Actions",
      "pipelines": [
        {"name": "CI", "file": ".github/workflows/ci.yml", "avg_duration": "4m30s"}
      ],
      "containerization": "None (runs directly on OS)",
      "deployment": "Manual"
    },
    "findings": [
      {
        "category": "efficiency",
        "severity": "medium",
        "finding": "pip install runs from scratch on every CI run -- no dependency caching",
        "recommendation": "Add actions/cache for pip dependencies",
        "expected_improvement": "2-3 minute reduction in CI time"
      },
      {
        "category": "reliability",
        "severity": "high",
        "finding": "No health check after deployment",
        "recommendation": "Add post-deploy health check step that verifies /api/health returns 200"
      }
    ],
    "recommendations": [
      {"priority": "high", "action": "Add dependency caching to CI pipeline", "effort": "S"},
      {"priority": "high", "action": "Add health check to deployment workflow", "effort": "S"},
      {"priority": "medium", "action": "Add Dockerfile for consistent dev/prod environments", "effort": "M"}
    ]
  },
  "provenance": [
    {"source": ".github/workflows/ci.yml", "method": "YAML analysis"},
    {"source": "GitHub Actions documentation", "topic": "caching best practices"}
  ]
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-001**: Research modes MUST cite sources -- file paths, URLs, or commit SHAs
- **STD-WRK-009**: DevOps pipelines MUST include rollback procedure and health checks
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **STD-UNI-003**: All skill executions MUST log entry and exit at INFO level with timing
- **CON-WRK-001**: Research modes are READ-ONLY -- MUST NOT modify production files

## Examples

### Example: CI pipeline optimization

**Input:**
```
task: "Analyze why our GitHub Actions CI takes 8+ minutes and recommend optimizations"
files: [".github/workflows/ci.yml"]
platform: "github-actions"
```

**Output (abbreviated):**
```
current_state:
  pipeline: CI workflow with 3 jobs (lint, test, build)
  avg_duration: 8m15s
  bottleneck: test job (6m30s)

findings:
  1. [high] No pip caching -- installing 42 dependencies from scratch each run (+3m)
  2. [medium] Tests run sequentially -- pytest-xdist could parallelize across 4 workers (-40% test time)
  3. [low] Lint and test jobs run sequentially -- could run in parallel (-1m)

recommendations:
  1. Add pip caching: actions/cache with requirements hash key (effort: XS, saves: 3m)
  2. Parallelize tests: add pytest-xdist, run with -n auto (effort: S, saves: 2-3m)
  3. Parallelize CI jobs: run lint and test in parallel (effort: XS, saves: 1m)

projected_duration: 2m30s (down from 8m15s, 70% reduction)
```
