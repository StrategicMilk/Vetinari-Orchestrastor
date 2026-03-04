# Drift Prevention and Alignment Strategy
## Keeping Code and Documentation in Sync

**Version:** 1.0  
**Status:** Active  
**Last Updated:** March 3, 2026

---

## Overview

This document outlines the strategy to prevent drift between code and documentation across all migration phases. It establishes mechanisms for maintaining alignment as Vetinari evolves through its hierarchical multi-agent orchestration rollout.

---

## Key Pillars

### 1. Central Migration Index

The `MIGRATION_INDEX.md` serves as the single source of truth for phase status, artifacts, owners, and acceptance criteria.

**Implementation:**
- All phase changes must update MIGRATION_INDEX.md
- Status changes require explicit acceptance criteria proof
- Owners must sign off before phase progression

### 2. Versioned Contracts

All data contracts (Plan, Task, AgentTask, AgentSpec) carry a version field.

**Contract Versioning Rules:**
- Major version bump: Breaking changes to schema
- Minor version bump: Backward-compatible additions
- Patch version bump: Documentation fixes

```json
{
  "plan_id": "plan_001",
  "version": "v0.1.0",
  "goal": "..."
}
```

**Enforcement:**
- Code must validate against current contract version
- Documentation must reference current contract version
- Version mismatches trigger CI failure

### 3. CI Doc Alignment Gates

Every PR must include documentation changes when code changes affect contracts or interfaces.

#### Gate Implementation

```yaml
# .github/workflows/doc_alignment.yml
name: Doc Alignment Check

on: [pull_request]

jobs:
  doc-alignment:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Check for contract changes
        id: contract_check
        run: |
          git diff --name-only ${{ github.event.pull_request.base.sha }} HEAD \
            | grep -E "(contracts|schema|interface)" \
            > /tmp/contract_files
          
          if [ -s /tmp/contract_files ]; then
            echo "has_contract_changes=true" >> $GITHUB_OUTPUT
          else
            echo "has_contract_changes=false" >> $GITHUB_OUTPUT
          fi
      
      - name: Check doc updates
        if: steps.contract_check.outputs.has_contract_changes == 'true'
        run: |
          git diff --name-only ${{ github.event.pull_request.base.sha }} HEAD \
            | grep -E "^docs/" > /tmp/doc_files
          
          if [ -s /tmp/doc_files ]; then
            echo "✅ Documentation updated for contract changes"
            exit 0
          else
            echo "❌ Contract changes require documentation updates"
            exit 1
          fi
```

#### PR Checklist

When contracts or interfaces change:

- [ ] Update relevant schema in `vetinari/interfaces/contracts.py`
- [ ] Bump contract version
- [ ] Update `SKILL_MIGRATION_GUIDE.md` schema examples
- [ ] Update `MIGRATION_INDEX.md` if phase artifacts affected
- [ ] Add migration notes if breaking changes
- [ ] Run contract validation tests

### 4. Phase Gating

Each migration phase requires formal acceptance before progressing.

#### Phase Transition Requirements

1. **Acceptance Criteria Met:** All criteria checked off in MIGRATION_INDEX.md
2. **Documentation Updated:** All relevant docs reflect current state
3. **Tests Passing checks:** All CI green
4. **Owner Sign-off:** Phase owner approves transition

#### Phase Review Process

```python
# Pseudocode for phase transition
def transition_phase(current_phase, next_phase):
    if not current_phase.acceptance_criteria_met():
        raise PhaseTransitionError("Acceptance criteria not met")
    
    if not current_phase.docs_updated():
        raise PhaseTransitionError("Documentation not updated")
    
    if not current_phase.tests_passing():
        raise PhaseTransitionError("Tests not passing")
    
    if not current_phase.owner_signoff():
        raise PhaseTransitionError("Owner sign-off required")
    
    current_phase.status = "Complete"
    next_phase.status = "In Progress"
    update_migration_index()
```

### 5. Drift Auditing

Runtime outputs are audited against documented contracts; mismatches trigger governance alerts.

#### Audit Mechanisms

1. **Contract Validation:** JSON schema validation on all Plan/Task payloads
2. **Output Auditing:** Agent outputs compared against documented capabilities
3. **Capability Tracking:** Registry of agent capabilities vs. documented capabilities

```python
from jsonschema import validate
import logging

logger = logging.getLogger(__name__)

# Contract schemas
PLAN_SCHEMA = {
    "type": "object",
    "required": ["plan_id", "version", "goal", "tasks"],
    "properties": {
        "plan_id": {"type": "string"},
        "version": {"type": "string", "pattern": "^v\\d+\\.\\d+\\.\\d+$"},
        "goal": {"type": "string"},
        "tasks": {"type": "array"}
    }
}

def validate_plan(plan: dict) -> bool:
    """Validate plan against contract schema."""
    try:
        validate(instance=plan, schema=PLAN_SCHEMA)
        logger.info(f"Plan {plan.get('plan_id')} validated successfully")
        return True
    except ValidationError as e:
        logger.error(f"Plan validation failed: {e}")
        return False
```

---

## Automated Checks

### 1. Contract Schema Validation

```python
# tests/test_contract_validation.py
import pytest
from vetinari.interfaces.contracts import Plan, Task, AgentTask

def test_plan_schema_validation():
    valid_plan = {
        "plan_id": "test_001",
        "version": "v0.1.0",
        "goal": "Test goal",
        "tasks": []
    }
    plan = Plan(**valid_plan)
    assert plan.version == "v0.1.0"

def test_invalid_version_raises():
    invalid_plan = {
        "plan_id": "test_001",
        "version": "invalid",
        "goal": "Test goal",
        "tasks": []
    }
    with pytest.raises(ValidationError):
        Plan(**invalid_plan)
```

### 2. Doc-Contract Alignment Check

```python
# scripts/check_doc_contract_alignment.py
import re
import json
import sys

def extract_version_from_doc(doc_path: str) -> str:
    """Extract version from documentation."""
    with open(doc_path) as f:
        content = f.read()
    match = re.search(r'version["\s:]+(v\d+\.\d+\.\d+)', content)
    return match.group(1) if match else None

def extract_version_from_code(code_path: str) -> str:
    """Extract version from code contracts."""
    # Parse code to find version definitions
    pass

def check_alignment():
    doc_version = extract_version_from_doc("docs/SKILL_MIGRATION_GUIDE.md")
    code_version = extract_version_from_code("vetinari/interfaces/contracts.py")
    
    if doc_version != code_version:
        print(f"❌ Version mismatch: Doc={doc_version}, Code={code_version}")
        return False
    print(f"✅ Versions aligned: {doc_version}")
    return True

if __name__ == "__main__":
    sys.exit(0 if check_alignment() else 1)
```

### 3. Agent Capability Audit

```python
# scripts/check_agent_capabilities.py
def audit_agent_capabilities():
    """Verify documented capabilities match actual implementations."""
    
    # Load documented capabilities
    with open("docs/SKILL_MIGRATION_GUIDE.md") as f:
        doc = f.read()
    
    # Extract agent capabilities from code
    for agent in get_all_agents():
        code_caps = agent.get_capabilities()
        doc_caps = extract_capabilities_from_doc(doc, agent.name)
        
        if set(code_caps) != set(doc_caps):
            print(f"⚠️  Capability drift in {agent.name}")
            print(f"   Code: {code_caps}")
            print(f"   Doc:  {doc_caps}")
            return False
    
    print("✅ All agent capabilities aligned")
    return True
```

---

## Governance Process

### PR Review Requirements

1. **Code Review:** Standard code review by maintainer
2. **Documentation Review:** Docs owner verifies doc changes
3. **Contract Review:** If contracts changed, verify version bump
4. **Phase Review:** If phase artifact, verify MIGRATION_INDEX.md update

### Conflict Resolution

| Scenario | Resolution |
|----------|------------|
| Version mismatch | Reject PR until versions aligned |
| Missing docs | Reject PR until docs added |
| Missing MIGRATION_INDEX update | Reject PR until updated |
| Capability drift | Discuss with agent owner; update doc or code |

### Escalation Path

1. **Minor Drift:** Fix in same PR
2. **Major Drift:** Create issue; schedule fix sprint
3. **Blocking Drift:** Halt release; emergency fix required

---

## Recommended CI Checks

### Full CI Pipeline

```yaml
# .github/workflows/ci.yml
name: Vetinari CI

on: [push, pull_request]

jobs:
  # ... existing jobs (lint, typecheck, tests)
  
  doc-alignment:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run doc alignment check
        run: python scripts/check_doc_contract_alignment.py
      
  contract-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run contract tests
        run: pytest tests/test_contract_validation.py
      
  migration-index-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Verify MIGRATION_INDEX status
        run: python scripts/check_migration_index.py
      
  capability-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Audit agent capabilities
        run: python scripts/check_agent_capabilities.py
```

---

## Rollback Procedures

### If Drift Detected in Production

1. **Immediate:** Roll back code to last known good version
2. **Analysis:** Identify root cause of drift
3. **Fix:** Update docs or code to restore alignment
4. **Prevention:** Add check to prevent recurrence
5. **Review:** Conduct post-mortem

### Rollback Command

```bash
# Rollback to previous release
git revert HEAD
git push origin main
```

---

## Maintenance Schedule

| Activity | Frequency | Owner |
|----------|------------|-------|
| Doc alignment check | Every PR | CI |
| Contract validation | Every PR | CI |
| MIGRATION_INDEX review | Monthly | Documentation Lead |
| Capability audit | Quarterly | All Agents Owners |
| Full drift audit | Bi-annually | Architecture Team |

---

## Related Documents

- `MIGRATION_INDEX.md` - Central phase tracking
- `SKILL_MIGRATION_GUIDE.md` - Migration process and agent prompts
- `DEVELOPER_GUIDE.md` - Developer onboarding
- `ARCHITECTURE.md` - System architecture
