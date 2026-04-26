---
name: Dependency Analysis
description: Map dependency trees, check for CVEs using advisory databases, identify license conflicts, and flag outdated packages
mode: code_discovery
agent: worker
version: "1.0.0"
capabilities:
  - dependency_analysis
  - code_discovery
tags:
  - research
  - dependencies
  - security
  - licenses
---

# Dependency Analysis

## Purpose

Dependency Analysis examines the project's dependency tree to identify security vulnerabilities (CVEs), license conflicts, outdated packages, and unnecessary transitive dependencies. It produces a health report that enables informed decisions about upgrades, replacements, and risk acceptance. This skill prevents the silent accumulation of security debt and ensures the project's dependency posture is auditable and defensible.

## When to Use

- During periodic dependency health checks (recommended monthly)
- Before adding a new dependency to the project
- When a security advisory affects a dependency in the project's tree
- Before a major release to ensure no known vulnerabilities ship
- When investigating unexpected behavior that might stem from dependency conflicts
- When a dependency is abandoned or deprecated and needs replacement

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| task            | string          | Yes      | What to analyze (e.g., "Full dependency health check")             |
| scope           | string          | No       | "all", "production", "dev", or specific package name               |
| check_cves      | bool            | No       | Whether to query CVE databases (default: true)                     |
| check_licenses  | bool            | No       | Whether to verify license compatibility (default: true)            |
| check_updates   | bool            | No       | Whether to check for available updates (default: true)             |
| context         | dict            | No       | Project context (pyproject.toml contents, Python version)          |

## Process Steps

1. **Dependency tree extraction** -- Parse pyproject.toml to build the direct dependency list. Separate production dependencies from dev/test dependencies. Note version constraints (pinned, minimum, range).

2. **Transitive resolution** -- Resolve transitive dependencies to build the full dependency tree. Identify cases where multiple packages depend on the same transitive dependency at different versions (potential conflicts).

3. **Version currency check** -- For each dependency, compare installed version against latest available. Classify as: current (within 1 minor version), outdated (2+ minor versions behind), deprecated (maintainer has marked end-of-life), or abandoned (no commits in 12+ months).

4. **CVE scanning** -- Query vulnerability databases (PyPI advisory DB, OSV, GitHub Security Advisories) for known CVEs affecting installed versions. For each CVE, record: severity (CVSS score), affected versions, fixed version, and whether the project's usage pattern is actually vulnerable.

5. **License audit** -- Extract license information for each dependency. Check for compatibility with the project's license. Flag any copyleft licenses (GPL, AGPL) that could impose requirements on the project. Note any dependencies with unclear or missing license declarations.

6. **Dependency weight analysis** -- Calculate the "weight" of each dependency: number of transitive dependencies it brings, total size, and import time impact. Identify heavyweight dependencies that might have lighter alternatives.

7. **Unused dependency detection** -- Cross-reference the dependency list against actual imports in the codebase. Flag any dependencies that are declared but never imported (phantom dependencies) or imported but not declared (missing dependencies).

8. **Upgrade path analysis** -- For outdated or vulnerable dependencies, determine the upgrade path. Check changelogs for breaking changes between current and target versions. Identify any dependencies that would need coordinated upgrades.

9. **Risk scoring** -- Assign a risk score to the overall dependency posture based on: number of CVEs, severity of CVEs, number of outdated packages, license issues, and abandoned dependencies.

10. **Report generation** -- Compile the full dependency health report with: summary score, detailed per-package analysis, prioritized action items, and upgrade recommendations.

## Output Format

The skill produces a dependency health report:

```json
{
  "success": true,
  "output": {
    "summary": {
      "total_deps": 42,
      "direct": 15,
      "transitive": 27,
      "health_score": "B",
      "critical_issues": 1,
      "warnings": 4
    },
    "cves": [
      {
        "package": "cryptography",
        "installed": "41.0.1",
        "cve": "CVE-2024-XXXX",
        "severity": "high",
        "cvss": 8.1,
        "fixed_in": "41.0.7",
        "exploitable": true,
        "recommendation": "Upgrade to >=41.0.7"
      }
    ],
    "outdated": [
      {"package": "flask", "installed": "3.0.0", "latest": "3.1.2", "behind": "1 minor"}
    ],
    "license_issues": [],
    "unused": ["legacy-package"],
    "action_items": [
      {"priority": "critical", "action": "Upgrade cryptography to >=41.0.7 (CVE fix)"},
      {"priority": "low", "action": "Remove unused dependency: legacy-package"}
    ]
  },
  "provenance": [
    {"source": "pyproject.toml", "section": "dependencies"},
    {"source": "PyPI Advisory DB", "queried": "2025-01-15"}
  ]
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-001**: Research modes MUST cite sources -- file paths, URLs, or commit SHAs
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **STD-UNI-003**: All skill executions MUST log entry and exit at INFO level with timing
- **CON-WRK-001**: Research modes are READ-ONLY -- MUST NOT modify production files
- **GDL-WRK-001**: Use code_discovery before build to understand existing patterns

## Examples

### Example: Pre-release dependency audit

**Input:**
```
task: "Full dependency health check before v1.0 release"
scope: "production"
check_cves: true
check_licenses: true
```

**Output (abbreviated):**
```
summary:
  health_score: B
  total: 28 production deps (12 direct, 16 transitive)
  critical: 0 CVEs
  warnings: 2 outdated packages, 1 unused dependency

outdated:
  - pydantic 2.5.0 -> 2.7.1 (2 minor versions behind, no breaking changes)
  - flask 3.0.0 -> 3.1.2 (1 minor version, new features only)

unused:
  - python-dotenv: declared in pyproject.toml but not imported anywhere

action_items:
  1. [medium] Upgrade pydantic to 2.7.x for bug fixes
  2. [low] Upgrade flask to 3.1.x for performance improvements
  3. [low] Remove python-dotenv from dependencies
```
