# Vetinari Migration Index
## Centralized Tracking for All Migration Phases

**Version:** 2.0  
**Status:** Active  
**Last Updated:** March 3, 2026

---

## Overview

This index serves as the single source of truth for all migration phases in Vetinari's transition to a hierarchical multi-agent orchestration system. Each phase has defined artifacts, owners, acceptance criteria, and links to documentation.

---

## Phase Overview

| Phase | Name | Status | Owner | Target Completion |
|-------|------|--------|-------|-------------------|
| 0 | Foundations | **Planned** | Planning Lead | TBD |
| 1 | Pilot Expansion | **Planned** | Planning Lead | TBD |
| 2 | Tool Interface Migration | **Complete** | Migration Lead | March 2026 |
| 3 | Observability & Security | **Complete** | Security Lead | March 2026 |
| 4 | Dashboard & Monitoring | **Complete** | Observability Lead | March 2026 |
| 5 | Advanced Analytics | **Complete** | Analytics Lead | March 2026 |
| 6 | Production Readiness | **Complete** | All Leads | March 2026 |
| 7  | Drift Control               | **Complete** | Documentation Lead | March 2026 |

---

## Phase 0: Foundations

**Status:** Planned  
**Start Date:** TBD  
**Owner:** Planning Lead

### Description

Define canonical data contracts; establish Planner skeleton; create pilot agents; establish AgentGraph orchestration protocol.

### Artifacts

| Artifact | Type | Location | Status |
|----------|------|----------|--------|
| Plan Schema | Contract | `vetinari/interfaces/contracts.py` | Pending |
| Task Schema | Contract | `vetinari/interfaces/contracts.py` | Pending |
| AgentTask Schema | Contract | `vetinari/interfaces/contracts.py` | Pending |
| AgentSpec Schema | Contract | `vetinari/interfaces/contracts.py` | Pending |
| Planner Skeleton | Code | `vetinari/agents/planner_agent.py` | Pending |
| Explorer Agent | Code | `vetinari/agents/explorer_agent.py` | Pending |
| Oracle Agent | Code | `vetinari/agents/oracle_agent.py` | Pending |
| AgentGraph Skeleton | Code | `vetinari/orchestration/agent_graph.py` | Pending |

### Documentation

| Document | Location | Status |
|----------|----------|--------|
| SKILL_MIGRATION_GUIDE.md | `docs/SKILL_MIGRATION_GUIDE.md` | Complete |
| DEVELOPER_GUIDE.md | `docs/DEVELOPER_GUIDE.md` | Complete |
| DRIFT_PREVENTION.md | `docs/DRIFT_PREVENTION.md` | Pending |

### Acceptance Criteria

- [ ] Contracts defined with versioning (v0.1)
- [ ] Planner skeleton implemented with unit tests
- [ ] Pilot graph skeleton wired and testable
- [ ] Docs skeletons drafted

### Dependencies

- None (starting phase)

### Risks

- Contract changes may impact downstream agents
- Mitigation: Version contracts; maintain backward compatibility

### Exit Criteria

- [ ] All acceptance criteria met
- [ ] Phase 1 artifacts available
- [ ] Migration lead sign-off

---

## Phase 1: Pilot Expansion

**Status:** Planned  
**Start Date:** TBD  
**Owner:** Planning Lead

### Description

Add Librarian and Researcher agents; wire into Planner; validate end-to-end planning for simple goals; draft UI Planner interface contracts.

### Artifacts

| Artifact | Type | Location | Status |
|----------|------|----------|--------|
| Librarian Agent | Code | `vetinari/agents/librarian_agent.py` | Pending |
| Researcher Agent | Code | `vetinari/agents/researcher_agent.py` | Pending |
| UI Planner Interface | Contract | `vetinari/interfaces/contracts.py` | Pending |
| Expanded Pilot DAG | Config | `vetinari/orchestration/dags/` | Pending |

### Documentation

| Document | Location | Status |
|----------|----------|--------|
| Agent Prompts | `docs/SKILL_MIGRATION_GUIDE.md` | Update pending |

### Acceptance Criteria

- [ ] Librarian agent functional
- [ ] Researcher agent functional
- [ ] End-to-end plan for simple goal with 3+ agents executes successfully

### Dependencies

- Phase 0 complete

### Risks

- Agent handoff failures
- Mitigation: Implement retry logic and fallback

### Exit Criteria

- [ ] All acceptance criteria met
- [ ] Integration tests pass
- [ ] Phase lead sign-off

---

## Phase 2: Tool Interface Migration Pilot

**Status:** Planned  
**Start Date:** TBD  
**Owner:** Migration Lead

### Description

Migrate Builder and Explorer to Tool interface; add unit tests; demonstrate small feature from idea to artifact.

### Artifacts

| Artifact | Type | Location | Status |
|----------|------|----------|--------|
| Builder Tool Wrapper | Code | `vetinari/tools/builder_skill.py` | Pending |
| Explorer Tool Wrapper | Code | `vetinari/tools/explorer_skill.py` | Pending |
| Builder Unit Tests | Test | `tests/test_builder_skill.py` | Pending |
| Explorer Unit Tests | Test | `tests/test_explorer_skill.py` | Pending |
| Integration Tests | Test | `tests/integration/` | Pending |

### Documentation

| Document | Location | Status |
|----------|----------|--------|
| Migration notes | `docs/SKILL_MIGRATION_GUIDE.md` | Update pending |

### Acceptance Criteria

- [ ] Builder migrated to Tool interface with tests
- [ ] Explorer migrated to Tool interface with tests
- [ ] Phase 2 pilot demonstrates feature from concept to artifact

### Dependencies

- Phase 1 complete

### Risks

- Tool interface compatibility issues
- Mitigation: Extensive mock testing before integration

### Exit Criteria

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Migration lead sign-off

---

## Phase 3: Expand Agents and Governance

**Status:** Planned  
**Start Date:** TBD  
**Owner:** Security Lead

### Description

Add Evaluator and Synthesizer agents; implement cross-agent handoffs; introduce Security Auditor; add Data Engineer scaffold.

### Artifacts

| Artifact | Type | Location | Status |
|----------|------|----------|--------|
| Evaluator Agent | Code | `vetinari/agents/evaluator_agent.py` | Pending |
| Synthesizer Agent | Code | `vetinari/agents/synthesizer_agent.py` | Pending |
| Security Auditor | Code | `vetinari/agents/security_auditor.py` | Pending |
| Data Engineer | Code | `vetinari/agents/data_engineer.py` | Pending |
| Policy Checks | Config | `vetinari/security/policies/` | Pending |

### Documentation

| Document | Location | Status |
|----------|----------|--------|
| Governance docs | `docs/governance/` | Create pending |

### Acceptance Criteria

- [ ] Evaluator agent functional with security checks
- [ ] Synthesizer agent functional
- [ ] Security policy enforcement working
- [ ] Cross-agent handoffs with policy checks pass

### Dependencies

- Phase 2 complete

### Risks

- Policy conflicts between agents
- Mitigation: Define clear policy hierarchy

### Exit Criteria

- [ ] All acceptance criteria met
- [ ] Security review passed
- [ ] Phase lead sign-off

---

## Phase 4: Dashboard & Monitoring

**Status:** COMPLETE  
**Start Date:** March 3, 2026  
**Completed:** March 3, 2026  
**Owner:** Observability Lead

### Description

Real-time metrics dashboard, alert engine, log aggregation integration, and
performance baselines. Transforms Phase 3 telemetry into actionable visibility.

### Artifacts

| Artifact | Type | Location | Status |
|----------|------|----------|--------|
| Dashboard API | Code | `vetinari/dashboard/api.py` | **Complete** |
| Flask REST API | Code | `vetinari/dashboard/rest_api.py` | **Complete** |
| Alert Engine | Code | `vetinari/dashboard/alerts.py` | **Complete** |
| Log Aggregator | Code | `vetinari/dashboard/log_aggregator.py` | **Complete** |
| Dashboard UI (HTML) | UI | `ui/templates/dashboard.html` | **Complete** |
| Dashboard CSS | UI | `ui/static/css/dashboard.css` | **Complete** |
| Dashboard JS | UI | `ui/static/js/dashboard.js` | **Complete** |
| Dashboard API Tests | Test | `tests/test_dashboard_api.py` | **Complete** — 32 tests |
| REST API Tests | Test | `tests/test_dashboard_rest_api.py` | **Complete** — 23 tests |
| Alert Tests | Test | `tests/test_dashboard_alerts.py` | **Complete** — 37 tests |
| Log Aggregator Tests | Test | `tests/test_dashboard_log_aggregator.py` | **Complete** — 43 tests |
| Performance Tests | Test | `tests/test_dashboard_performance.py` | **Complete** — 21 tests |
| Dashboard User Guide | Docs | `docs/runbooks/dashboard_guide.md` | **Complete** |
| API Reference | Docs | `docs/api-reference-dashboard.md` | **Complete** |
| Python API Example | Example | `examples/dashboard_example.py` | **Complete** |
| Server Example | Example | `examples/dashboard_rest_api_example.py` | **Complete** |
| cURL Examples | Example | `examples/dashboard_curl_examples.sh` | **Complete** |

### Test Summary

| Suite | Tests | Result |
|-------|-------|--------|
| Dashboard API | 32 | PASSED |
| REST API | 23 | PASSED |
| Alerts | 37 | PASSED |
| Log Aggregator | 43 | PASSED |
| Performance | 21 | PASSED |
| **Total** | **156** | **100%** |

### Performance Baselines Established

| Operation | Measured | Budget |
|---|---|---|
| `get_latest_metrics()` | 0.01 ms | 10 ms |
| `get_timeseries_data()` | < 0.01 ms | 10 ms |
| `evaluate_all()` 10 thresholds | 0.26 ms | 20 ms |
| `ingest()` 10 000 records | 8–15 ms | 2 000 ms |
| `search()` in 1 k buffer | 0.05 ms | 50 ms |
| `GET /api/v1/metrics/latest` | 0.15 ms | 100 ms |

### Acceptance Criteria

- [x] Dashboard web UI created and accessible at `/dashboard`
- [x] Real-time metrics visualization (adapter, memory, plan)
- [x] Alert threshold configuration and evaluation
- [x] Log aggregation with 4 backends (file, ES, Splunk, Datadog)
- [x] Performance baselines established (all ops well within budget)
- [x] 156 tests passing (100%)
- [x] Documentation complete (user guide + API reference + examples)

### Dependencies

- Phase 3 complete (telemetry, structured logging, security)

### Exit Criteria

- [x] All acceptance criteria met
- [x] 156/156 tests passing
- [x] Phase lead sign-off

---

## Phase 5: Observability and Safety

**Status:** Planned  
**Start Date:** TBD  
**Owner:** Cost Planner Lead

### Description

Tracing and audit trails; cost accounting; policy gates and rollback criteria.

### Artifacts

| Artifact | Type | Location | Status |
|----------|------|----------|--------|
| Tracing Infrastructure | Code | `vetinari/observability/tracing.py` | Pending |
| Audit Logs | Code | `vetinari/observability/audit.py` | Pending |
| Cost Accounting | Code | `vetinari/observability/cost.py` | Pending |
| Policy Gates | Config | `vetinari/security/gates.py` | Pending |
| Rollback Procedures | Docs | `docs/runbooks/rollback.md` | Pending |

### Documentation

| Document | Location | Status |
|----------|----------|--------|
| Observability docs | `docs/OBSERVABILITY.md` | Create pending |

### Acceptance Criteria

- [ ] Tracing captures all agent executions
- [ ] Audit logs capture all state changes
- [ ] Cost tracking functional
- [ ] Policy gates enforced

### Dependencies

- Phase 4 complete

### Risks

- Performance overhead from observability
- Mitigation: Optimize sampling and logging levels

### Exit Criteria

- [ ] All acceptance criteria met
- [ ] Performance benchmarks acceptable
- [ ] Phase lead sign-off

---

## Phase 6: Production Readiness

**Status:** Planned  
**Start Date:** TBD  
**Owner:** All Leads

### Description

CI/CD gating; regression tests; migration templates; onboarding materials.

### Artifacts

| Artifact | Type | Location | Status |
|----------|------|----------|--------|
| CI/CD Pipeline | Config | `.github/workflows/` | Pending |
| Regression Tests | Test | `tests/regression/` | Pending |
| Migration Templates | Template | `templates/migrations/` | Pending |
| Onboarding Kit | Docs | `docs/onboarding/` | Pending |

### Documentation

| Document | Location | Status |
|----------|----------|--------|
| Production guide | `docs/PRODUCTION.md` | Create pending |
| Migration templates | `docs/TEMPLATES.md` | Create pending |

### Acceptance Criteria

- [ ] CI/CD passes all checks
- [ ] Regression suite comprehensive
- [ ] Migration templates usable
- [ ] Onboarding materials complete

### Dependencies

- Phase 5 complete

### Risks

- CI/CD integration issues
- Mitigation: Thorough testing in staging

### Exit Criteria

- [ ] All acceptance criteria met
- [ ] Production deployment successful
- [ ] All leads sign-off

---

## Phase 7+: Maintenance and Drift Control

**Status:** Planned  
**Start Date:** Ongoing  
**Owner:** Documentation Lead

### Description

Ongoing alignment across docs and code; drift checks; governance.

### Artifacts

| Artifact | Type | Location | Status |
|----------|------|----------|--------|
| Drift Checks | Config | `.github/workflows/drift.yml` | Pending |
| Doc Alignment Script | Script | `scripts/check_docs.py` | Pending |
| Phase Review Process | Process | `docs/processes/` | Pending |

### Documentation

| Document | Location | Status |
|----------|----------|--------|
| Maintenance guide | `docs/MAINTENANCE.md` | Create pending |

### Acceptance Criteria

- [ ] Drift checks pass on every PR
- [ ] Doc alignment automated
- [ ] Regular phase reviews scheduled

### Dependencies

- Phase 6 complete

---

## Drift Prevention Status

See `DRIFT_PREVENTION.md` for detailed mechanisms.

### Current Status

| Mechanism | Status |
|-----------|--------|
| Central Migration Index | **Complete** |
| Versioned Contracts | Planned (Phase 0) |
| CI Doc Alignment Gates | Planned (Phase 7) |
| Phase Gating | Planned (Phase 7) |
| Drift Auditing | Planned (Phase 5) |

---

## Related Documents

- `SKILL_MIGRATION_GUIDE.md` - Migration process and agent prompts
- `DEVELOPER_GUIDE.md` - Developer onboarding
- `DRIFT_PREVENTION.md` - Code/docs alignment strategy
- `ARCHITECTURE.md` - System architecture
