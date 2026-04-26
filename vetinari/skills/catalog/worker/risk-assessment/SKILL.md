---
name: Risk Assessment
description: Build risk matrices with probability times impact scoring for technical, operational, and security risks
mode: risk_assessment
agent: worker
version: "1.0.0"
capabilities:
  - risk_assessment
  - architecture_review
tags:
  - architecture
  - risk
  - assessment
  - security
---

# Risk Assessment

## Purpose

Risk Assessment systematically identifies, categorizes, and scores risks associated with proposed changes, architectural decisions, or system states. It builds risk matrices using probability times impact scoring, producing prioritized risk registers with mitigation strategies. This skill ensures that the Foreman's plans account for what could go wrong, not just what should go right, and that high-stakes decisions receive proportionate scrutiny.

## When to Use

- Before approving any plan that involves destructive operations (file deletion, schema migration)
- When evaluating architectural changes that affect multiple subsystems
- When introducing new dependencies with security or reliability implications
- When planning changes to production-critical code paths
- As input to the Foreman's plan risk section for complex tasks
- When a contrarian review has surfaced potential failure modes that need scoring

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| task            | string          | Yes      | What to assess risks for                                           |
| scope           | dict            | No       | Scope analysis output (blast radius, affected files)               |
| design          | dict            | No       | Proposed design or plan to assess                                  |
| threat_model    | list[string]    | No       | Known threats or attack vectors to consider                        |
| context         | dict            | No       | System context (deployment env, user base, data sensitivity)       |
| risk_appetite   | string          | No       | "low" (conservative), "medium" (balanced), "high" (aggressive)     |

## Process Steps

1. **Risk identification** -- Enumerate all potential risks across categories:
   - **Technical**: code breakage, performance degradation, data corruption, integration failures
   - **Operational**: deployment failures, monitoring gaps, on-call burden, runbook gaps
   - **Security**: new attack surfaces, privilege escalation, data exposure, dependency vulnerabilities
   - **Project**: scope creep, schedule overrun, resource unavailability, knowledge loss

2. **Probability estimation** -- For each risk, estimate the probability of occurrence:
   - **Rare** (0.1): requires multiple unlikely conditions to align
   - **Unlikely** (0.3): possible but not expected in normal operations
   - **Possible** (0.5): could happen, has happened in similar contexts
   - **Likely** (0.7): more likely than not, partial indicators already present
   - **Almost certain** (0.9): strong indicators, would be surprised if it didn't happen

3. **Impact assessment** -- For each risk, estimate the impact if it occurs:
   - **Negligible** (1): no user impact, cosmetic only
   - **Minor** (2): minor inconvenience, easy workaround, quick fix
   - **Moderate** (3): noticeable user impact, requires investigation, hours to resolve
   - **Major** (4): significant feature degradation, data loss possible, days to resolve
   - **Critical** (5): system down, data breach, security incident, weeks to resolve

4. **Risk scoring** -- Calculate risk score = probability x impact for each risk. Classify as:
   - **Critical** (score >= 3.5): Must mitigate before proceeding
   - **High** (score 2.0-3.4): Should mitigate, accept with justification
   - **Medium** (score 1.0-1.9): Mitigate if cost-effective
   - **Low** (score < 1.0): Accept and monitor

5. **Mitigation strategy** -- For each risk above the risk appetite threshold, propose a mitigation:
   - **Avoid**: Change approach to eliminate the risk entirely
   - **Reduce**: Take action to lower probability or impact
   - **Transfer**: Shift risk to another party (monitoring, alerts, insurance)
   - **Accept**: Acknowledge and document the risk with rationale

6. **Residual risk calculation** -- After mitigations, recalculate risk scores to determine residual risk. Verify residual risks are within acceptable thresholds.

7. **Risk matrix visualization** -- Organize risks into a 5x5 probability-impact matrix for visual assessment. This makes it easy to identify clusters of risk and overall risk posture.

8. **Contingency planning** -- For critical and high risks, define contingency plans: what to do if the risk materializes despite mitigation. Include rollback procedures, communication plans, and escalation paths.

9. **Risk register assembly** -- Compile the full risk register with: risk ID, description, category, probability, impact, score, mitigation, residual score, and owner.

## Output Format

The skill produces a risk assessment report:

```json
{
  "success": true,
  "output": {
    "overall_risk_level": "medium",
    "risk_count": {"critical": 0, "high": 2, "medium": 3, "low": 4},
    "risk_register": [
      {
        "id": "RISK-001",
        "description": "Schema migration corrupts existing memory data",
        "category": "technical",
        "probability": 0.3,
        "impact": 4,
        "score": 1.2,
        "level": "medium",
        "mitigation": {
          "strategy": "reduce",
          "action": "Backup database before migration, test migration on copy first",
          "residual_score": 0.4
        },
        "contingency": "Restore from backup, roll back migration script"
      }
    ],
    "risk_matrix": "... (5x5 grid with risk IDs placed by probability/impact) ...",
    "recommendations": [
      "Proceed with mitigations in place",
      "Add automated backup step before any schema migration",
      "Monitor error rates for 24 hours post-deployment"
    ]
  },
  "metadata": {
    "adr_reference": "ADR-XXXX (risk assessment for schema migration)"
  }
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-003**: Architecture mode MUST produce or reference an ADR for every design decision
- **STD-WRK-004**: Architecture modes are READ-ONLY -- MUST NOT modify production files
- **STD-WRK-010**: Designs MUST list alternatives considered with trade-off analysis
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **CON-WRK-002**: Architecture modes are READ-ONLY -- produce ADRs, not code changes
- **CON-WRK-003**: High-stakes categories require 3+ alternatives evaluated
- **GDL-WRK-004**: Risk assessment should feed into Foreman's plan risk section

## Examples

### Example: Risk assessment for a dependency upgrade

**Input:**
```
task: "Assess risks of upgrading pydantic from 2.5 to 2.7"
scope: {affected_files: 15, affected_tests: 22}
context: {deployment: "single instance", users: "development team"}
```

**Output (abbreviated):**
```
overall_risk_level: medium
risk_register:
  RISK-001: "Breaking API change in pydantic 2.6 model_dump() signature"
    probability: 0.5, impact: 3, score: 1.5, level: medium
    mitigation: "Run full test suite against 2.7 before merging"

  RISK-002: "Validator behavior change causes silent data truncation"
    probability: 0.3, impact: 4, score: 1.2, level: medium
    mitigation: "Add property-based tests for all Pydantic models"

  RISK-003: "Transitive dependency conflict with other packages"
    probability: 0.3, impact: 2, score: 0.6, level: low
    mitigation: "Check pip dependency resolver output for conflicts"

recommendations:
  1. "Create a branch, upgrade, and run full test suite before merging"
  2. "Review pydantic 2.6 and 2.7 changelogs for breaking changes"
  3. "Test serialization/deserialization roundtrip for all models"
```
