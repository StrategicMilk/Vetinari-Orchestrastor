# Vetinari — New Developer Onboarding Guide

**Audience**: Engineers joining the project for the first time  
**Time estimate**: 2–4 hours to complete

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Tour](#repository-tour)
3. [Development Environment Setup](#development-environment-setup)
4. [Architecture Primer](#architecture-primer)
5. [Running Tests](#running-tests)
6. [Making Your First Change](#making-your-first-change)
7. [Key Concepts](#key-concepts)
8. [Debugging Tips](#debugging-tips)
9. [Getting Help](#getting-help)

---

## Project Overview

Vetinari is a **hierarchical multi-agent orchestration system** for local LLMs.
It lets you compose specialized agents (Builder, Explorer, Evaluator, etc.) into
pipelines that run on locally-hosted models via LM Studio.

**Phase milestones completed:**

| Phase | Name | What it adds |
|---|---|---|
| 2 | Tool Interface Migration | Standardized Tool/ToolResult protocol for all skills |
| 3 | Observability & Security | Telemetry, structured logging, secret scanning |
| 4 | Dashboard & Monitoring | REST API, real-time UI, alerts, log aggregation |
| 5 | Advanced Analytics | Anomaly detection, cost attribution, SLA tracking, forecasting |
| 6 | Production Readiness | CI/CD, regression tests, migration templates, this guide |

---

## Repository Tour

```
vetinari/
├── adapters/          Model adapter implementations
├── agents/            Agent base classes and orchestration
├── analytics/         Phase 5: anomaly, cost, sla, forecasting
├── dashboard/         Phase 4: API, alerts, log_aggregator, rest_api
├── memory/            Dual-memory backend (OC + Mnemosyne)
├── orchestration/     Agent graph and multi-agent coordination
├── skills/            Skill implementations (Builder, Explorer, …)
├── tools/             Tool interface wrappers for each skill
├── telemetry.py       Phase 3: metrics collection
├── security.py        Phase 3: secret detection and sanitization
├── structured_logging.py  Phase 3: distributed tracing
└── execution_context.py   Permission and execution mode system

tests/
├── regression/        Cross-cutting regression tests (Phases 4 & 5)
├── test_dashboard_*.py     Phase 4 tests (156 total)
├── test_analytics_*.py     Phase 5 tests (88 total)
└── test_*.py               Phase 2–3 unit tests

docs/
├── onboarding/        This guide + QUICK_START.md
├── runbooks/          Operational runbooks
├── api-reference-*.md API references for each phase
└── MIGRATION_INDEX.md Master phase tracking document

examples/
├── dashboard_example.py           Phase 4 Python API walkthrough
├── dashboard_rest_api_example.py  Phase 4 Flask server example
├── analytics_example.py           Phase 5 analytics walkthrough
└── builder_skill_example.py       Phase 2 skill example

templates/migrations/
├── new_skill_template.py          Copy to create a new skill
├── new_skill_tests_template.py    Copy to test a new skill
└── migration_checklist.md         Step-by-step migration checklist
```

---

## Development Environment Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install development dependencies
pip install -r requirements.txt
pip install -e .
pip install pytest pytest-cov

# 3. Verify setup
pytest tests/ -q
# Expected: 983+ tests collected, all passing

# 4. Start LM Studio (separate terminal)
# Ensure it's listening on the configured LMSTUDIO_HOST

# 5. Run an example
python examples/analytics_example.py
```

---

## Architecture Primer

### Data flow

```
User / CLI
    ↓
Orchestrator (multi_agent_orchestrator.py)
    ↓
AgentGraph (orchestration/agent_graph.py)
    ↓ dispatches tasks to
Skills / Tools (skills/, tools/)
    ↓ call
LM Studio Adapter (adapters/)
    ↓ records to
TelemetryCollector (telemetry.py)
    ↓ read by
DashboardAPI (dashboard/api.py)
    ↓ served by
Flask REST API (dashboard/rest_api.py)
    ↓
Browser (ui/templates/dashboard.html)
```

### Singleton pattern

All major services use a thread-safe singleton:

```python
from vetinari.dashboard.api       import get_dashboard_api
from vetinari.dashboard.alerts    import get_alert_engine
from vetinari.analytics            import get_cost_tracker
from vetinari.telemetry            import get_telemetry_collector
```

In tests, always call the matching `reset_*()` function in `setUp`/`tearDown`
to get a clean instance.

### Tool interface

Every skill implements `Tool.execute(**kwargs) → ToolResult`:

```python
from vetinari.tool_interface import Tool, ToolResult

class MySkillTool(Tool):
    def execute(self, **kwargs) -> ToolResult:
        ...
        return ToolResult(success=True, data={...})
```

Use `templates/migrations/new_skill_template.py` as your starting point.

---

## Running Tests

```bash
# All tests
pytest tests/ -q

# Specific phase
pytest tests/test_dashboard_*.py -v
pytest tests/test_analytics_*.py -v

# Regression suite
pytest tests/regression/ -v

# With coverage
pytest tests/ --cov=vetinari --cov-report=term-missing
```

Test count baseline (Phase 6): **983+ tests, 100% pass rate**.

---

## Making Your First Change

1. **Branch**: `git checkout -b feat/your-feature-name`
2. **Copy template**: `cp templates/migrations/new_skill_template.py vetinari/skills/your_skill.py`
3. **Implement**: follow the `# TODO` markers
4. **Test**: `pytest tests/test_your_skill.py -v`
5. **Regression check**: `pytest tests/regression/ -q`
6. **Full suite**: `pytest tests/ -q`
7. **PR**: follow `GIT_COMMIT_GUIDE.md` for commit message format

---

## Key Concepts

### ExecutionMode

Controls what a skill is allowed to do:

| Mode | Description |
|---|---|
| `READ_ONLY` | No side effects; safe for analysis |
| `SUGGEST` | May propose changes; human approval required |
| `AUTO_EDIT` | Can make changes without approval |
| `PLAN` | Plan-gate must approve before execution |

### Telemetry

Always record adapter calls so the dashboard has data:

```python
telemetry = get_telemetry_collector()
telemetry.record_adapter_latency("openai", "gpt-4", latency_ms, success=True, tokens_used=n)
```

### Secret scanning

Never log raw user inputs or API responses without sanitizing:

```python
from vetinari.security import get_secret_scanner
safe_text = get_secret_scanner().sanitize(raw_text)
logger.info("Response: %s", safe_text)
```

### Distributed tracing

Wrap operations in a `CorrelationContext` to get a trace ID in all logs:

```python
from vetinari.structured_logging import CorrelationContext
with CorrelationContext() as ctx:
    ctx.set_span_id("my-span")
    logger.info("This log will have trace_id and span_id")
```

---

## Debugging Tips

| Problem | Solution |
|---|---|
| `ImportError: vetinari` | Run `pip install -e .` from the project root |
| Dashboard shows no data | Feed telemetry first: see `examples/dashboard_example.py` |
| Alert never fires | Call `engine.evaluate_all()` manually; check `engine.get_stats()` |
| Test singleton interference | Call `reset_X()` in both `setUp` and `tearDown` |
| LM Studio connection refused | Check `LMSTUDIO_HOST` in `.env`; ensure LM Studio is running |
| Flask template 404 | Confirm `ui/templates/dashboard.html` exists; check `rest_api.py` template_folder |

---

## Getting Help

- Internal docs: `docs/DEVELOPER_GUIDE.md`, `docs/ARCHITECTURE.md`
- Runbooks: `docs/runbooks/`
- Open an issue: follow the issue template in `.github/`
- Check `QUICK_REFERENCE.md` for common commands
