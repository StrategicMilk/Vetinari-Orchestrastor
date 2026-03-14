# Vetinari Migration & Improvement Index
## Centralized Tracking for All Migration and Master Plan Phases

**Version:** 3.4.2
**Status:** Active
**Last Updated:** March 7, 2026
**Test Suite:** 1,968 tests passing (0 failures, 0 warnings)

---

## Overview

This index serves as the single source of truth for all migration and improvement work in Vetinari's evolution as a hierarchical multi-agent orchestration system (~52K lines, 27 agents: 21 legacy + 6 consolidated). It tracks both the original migration phases (0-7) and the comprehensive 13-phase master plan executed across multiple sessions.

**Master plan reference:** `~/.claude/plans/unified-cuddling-porcupine.md`
**Execution plan reference:** `~/.claude/plans/transient-noodling-steele.md`

---

## Original Migration Phases

| Phase | Name | Status | Completed |
|-------|------|--------|-----------|
| 0 | Foundations | **Planned** | — |
| 1 | Pilot Expansion | **Planned** | — |
| 2 | Tool Interface Migration | **Complete** | March 2026 |
| 3 | Observability & Security | **Complete** | March 2026 |
| 4 | Dashboard & Monitoring | **Complete** | March 2026 |
| 5 | Advanced Analytics | **Complete** | March 2026 |
| 6 | Production Readiness | **Complete** | March 2026 |
| 7 | Drift Control | **Complete** | March 2026 |

---

## Master Plan Phases

| # | Phase | Status | Key Deliverables |
|---|-------|--------|------------------|
| 1 | Foundation Cleanup | ✅ **Complete** | Enum consolidation (9+ dupes → types.py), deprecating facades, dead code cleanup, all `__init__.py` present, version single-source, config populated |
| 2 | Cross-Cutting Refactoring | ✅ **Complete** | `vetinari/exceptions.py` hierarchy, `SingletonMeta` metaclass, `constants.py`, dual-memory error handling, 0 bare excepts remaining |
| 2b | Code Quality & Security | ✅ **Complete** | 673 bare excepts fixed, 12 runtime bugs fixed, C1–C4+H6 critical security items, goal verifier fail-closed |
| 3 | Agent Consolidation (22→8) | ✅ **Complete** | MultiModeAgent base, 6 consolidated agents in `agents/consolidated/`, AgentGraph updated (27 agents) |
| 4 | Replace LLM with Algorithms | ✅ **Complete** | Heuristic vagueness (4.1), quality calibration interval LLM/10 (4.3), 15-archetype decomposition (4.4), AST code extraction (4.5) |
| 4b | ML/Algorithm Quality | ✅ **Complete** | Thompson Sampling Beta(2,2) prior, division-by-zero fix, per-content-type token ratios, `ml_config.yaml`, EMA/t-test/bootstrap/bias fixes |
| 5 | Analytics Pipeline Wiring | ✅ **Complete** | All 23 steps — CostTracker fix, SLA/forecaster/anomaly wiring, REST API (7 endpoints), dashboard UI, integration tests |
| 6 | model_relay.py Deprecation | ✅ **Complete** | `model_relay.py` converted to deprecating shim |
| 7 | Agent Wiring Completeness | ✅ **Complete** | SkillSpec, skill registry, GoalCategory routing, FailureType, SharedExecutionContext, blackboard Event+routing, all deferred batches 1–4 |
| 8 | Constraints Architecture | ✅ **Complete** | ArchitectureConstraint, ResourceConstraint, QualityGate, StyleConstraint, ConstraintRegistry, BaseAgent+AgentGraph wiring, rules.yaml agents section |
| 9 | Standardization | ✅ **Complete** | `pyproject.toml` (9.1), `.pre-commit-config.yaml` ruff+mypy+bandit (9.2), `conftest.py`, `py.typed` |
| 10 | Modularize God Modules | ✅ **Complete** | `two_layer_orchestration.py` split into 4 modules (~1,210 lines → focused units) |
| 11 | Complete Features | ✅ **Complete** | Per-task model config, NeMo guardrails, file tool, git tool, sandbox enforcement |
| 12 | Documentation Overhaul | ✅ **Complete** | MIGRATION_INDEX.md comprehensive rewrite |
| 13 | Final Polish | ✅ **Complete** | Kaizen verification, security hardening, learning chain tests (35 new), constraint wiring for all 6 consolidated agents |

### Additional Completed Batches

| Batch | Scope | Status |
|-------|-------|--------|
| Security (Batch 2) | Hardcoded API token removal, sandbox auth, shell=True removal, path traversal | ✅ **Complete** |
| Foundation Cleanup (Batch 3) | Duplicate enums, planning modules, orchestrator assessment | ✅ **Complete** |
| ML Quality (Batch 4/4b) | EMA double-counting, wrong t-test, broken bootstrap, scoring bias, exploration penalty | ✅ **Complete** |
| LM Studio Adapter Consolidation | Legacy adapter → provider adapter shim | ✅ **Complete** |
| Memory System Consolidation | SharedMemory singleton pattern | ✅ **Complete** |
| Deprecation Facades | `enhanced_memory.py`, `planning.py`, `model_relay.py`, `tool_registry_integration.py` — all shimmed with `DeprecationWarning` | ✅ **Complete** |
| Analytics Pipeline (Steps 1–23) | Full execution plan — all 23 steps wired and tested | ✅ **Complete** |
| Phase 2b Security Hardening | C1–C4+H6 critical items, goal verifier fail-closed | ✅ **Complete** |
| Batch 6 | Foundation Cleanup Remaining: `enhanced_memory.py` + `model_relay.py` deprecation facades | ✅ **Complete** |
| Batch 7 | Phase 4 LLM→Algorithm: Quality calibration interval, AST code compression, 15 decomposition templates | ✅ **Complete** |
| Batch 8 | Security + Permissions: `enforce_permission` in base_agent, AST security scanning, docs update | ✅ **Complete** |
| Batch 9 | Test Infrastructure: `conftest.py` fixtures, `provider_models.yaml`, assertion strengthening | ✅ **Complete** |
| Batch 10 | Model Search Merge: `model_discovery.py` (1052→749 lines), BaseAgent template methods | ✅ **Complete** |
| Batch 11 | Config + Helpers: `VetinariConfig` dataclass, RFC 9457 `error_response()`, agent affinity completeness | ✅ **Complete** |
| Batch 12 | Adapter + Security: Inflated baseline fix (0.5→0.0), cloud adapter telemetry, Flask `secret_key`, system prompt alignment | ✅ **Complete** |
| Batch 13 | Quality Polish: Ponder policy_penalty scale fix, model_pool resource_load dynamic, CWE ID examples in security prompts | ✅ **Complete** |
| Batch 14 | Final Polish: Builder `tests_added` placeholder→dynamic, `duckduckgo_search`→`ddgs` migration (41 warnings eliminated), real webhook/email alert dispatchers | ✅ **Complete** |
| Batch 15 | Exception Logging: Debug/warning logging added to 9 silent `except Exception: pass` patterns; ddgs install eliminates all test warnings (1817 passed, 0 warnings) | ✅ **Complete** |
| Batch 16 | Test Coverage: 151 new tests for 3 critical untested modules — `dynamic_model_router` (87), `vram_manager` (33), `agent_affinity` (31) | ✅ **Complete** |

---

## Master Plan Phase Details

### MP-1: Foundation Cleanup — COMPLETE

**Completed:** March 5–7, 2026

| Item | Description | Status |
|------|-------------|--------|
| 1.1 | Consolidate duplicate enums to `vetinari/types.py` (CodingTaskType, CodingTaskStatus, SeverityLevel, QualityGrade + prior enums = 9+ total) | ✅ **Complete** |
| 1.2 | LM Studio adapter consolidated — `lmstudio_adapter.py` is a thin shim over `vetinari/adapters/lmstudio_adapter.py` | ✅ **Complete** |
| 1.3 | `enhanced_memory.py` converted to deprecating facade (warnings on import) | ✅ **Complete** |
| 1.4 | `planning.py` deprecated with `warnings.warn()` directing callers to `planning_engine.py` | ✅ **Complete** |
| 1.5 | Legacy orchestrators shimmed with `DeprecationWarning` on instantiation | ✅ **Complete** |
| 1.6 | Blueprint extraction from `web_ui.py` | ✅ **Complete** |
| 1.7 | Bare except cleanup (673 instances → proper exception types) | ✅ **Complete** |
| 1.8 | Dead code deprecated — `MemoryStore` facade + `tool_registry_integration.py` shim | ✅ **Complete** |
| 1.9 | Dead files cleaned up; superseded docs archived under `docs/archive/` | ✅ **Complete** |
| 1.10 | Version single source of truth in `vetinari/__init__.py` (`__version__`) | ✅ **Complete** |
| 1.11 | All `__init__.py` files present across package tree | ✅ **Complete** |
| 1.12 | Config fields populated (no blank/placeholder required fields remain) | ✅ **Complete** |

### MP-2: Cross-Cutting Refactoring — COMPLETE

**Completed:** March 6, 2026

| Artifact | Type | Location | Status |
|----------|------|----------|--------|
| Exception hierarchy | Code | `vetinari/exceptions.py` | ✅ **Complete** |
| SingletonMeta metaclass | Code | `vetinari/utils.py` | ✅ **Complete** |
| Shared constants | Code | `vetinari/constants.py` | ✅ **Complete** (pre-existing, verified) |
| Agent boilerplate | Code | `vetinari/agents/base_agent.py` | ✅ **Complete** (verified well-factored) |

**Sub-items:**

| Item | Description | Status |
|------|-------------|--------|
| 2.0 | `constants.py` and `exceptions.py` created — canonical home for shared constants and exception hierarchy | ✅ **Complete** |
| 2.0d | Dual memory error handling verified well-designed — graceful fallback path between backends | ✅ **Complete** |
| 2.1 | All bare except blocks fixed — 0 remaining bare `except:` clauses across codebase | ✅ **Complete** |

**Exception hierarchy:** `VetinariError` base with rich `**context` kwargs → `ConfigurationError`, `StorageError`, `InferenceError` (→ `ModelNotFoundError`, `TimeoutError`), `AdapterError`, `AgentError` (→ `PlanningError`, `ExecutionError`, `VerificationError`, `CircularDependencyError`), `SecurityError` (→ `SandboxError`, `GuardrailError`), `DriftError`, `SLABreachError`

### MP-2b: Code Quality & Security Hardening — COMPLETE

**Completed:** March 5–7, 2026

| Item | Count | Status |
|------|-------|--------|
| Bare excepts fixed | 673 | ✅ **Complete** |
| Runtime bugs fixed | 12 | ✅ **Complete** |
| Security findings addressed | 28 (4 CRITICAL) | ✅ **Complete** |
| CostTracker.record() bug | 1 | ✅ **Complete** |

**Critical security fixes (C1–C4 + H6):**
- ✅ C1: Removed hardcoded API token from `lmstudio_adapter.py`
- ✅ C2: Added auth to `/api/sandbox/execute` (remote code execution endpoint)
- ✅ C3: Removed `shell=True` from `execute_shell()`
- ✅ C4: Added path traversal protection
- ✅ H6: Bind to `127.0.0.1` by default
- ✅ Goal verifier fail-closed on security check failure (secure default)

### MP-4: Replace LLM with Algorithms — COMPLETE

**Completed:** March 5–7, 2026

| Item | Replacement | From | To | Status |
|------|-------------|------|----|--------|
| 4.1 | Vagueness check | LLM call | Heuristic scorer (word count, specificity signals) | ✅ **Complete** |
| 4.2 | Security audit | LLM call | Static analysis (25 regex heuristic patterns) | ✅ **Complete** |
| 4.3 | Quality check | LLM call | Hybrid (heuristic + LLM every 10th task — calibration interval) | ✅ **Complete** |
| 4.4 | Decomposition | LLM call | Template-based patterns (expanded 3 → 15 archetypes) | ✅ **Complete** |
| 4.5 | Code signature extraction / compression | LLM call | AST-based extraction with regex fallback | ✅ **Complete** |

### MP-4b: ML/Algorithm Quality — COMPLETE

**Completed:** March 5–7, 2026

Fixes for 69 statistical/scoring/cold-start/race-condition issues including:

| Fix | Description | Status |
|-----|-------------|--------|
| Thompson Sampling prior | Beta(2,2) skeptical prior (was Beta(1,1) uniform) | ✅ **Complete** |
| Quality scorer division-by-zero | Guard added — no more ZeroDivisionError on empty score lists | ✅ **Complete** |
| Token optimizer chars-per-token | Per-content-type chars-per-token ratios (code vs prose vs JSON) | ✅ **Complete** |
| ML hyperparameter externalization | `ml_config.yaml` — all tunable values moved out of source code | ✅ **Complete** |
| EMA double-counting | Fixed exponential moving average accumulation bug | ✅ **Complete** |
| Wrong t-test | Corrected statistical test selection | ✅ **Complete** |
| Broken bootstrap | Repaired bootstrap resampling implementation | ✅ **Complete** |
| Scoring bias | Removed exploration penalty and cost penalty from quality score | ✅ **Complete** |
| Cascade routing | Implemented cost-aware cascade model routing | ✅ **Complete** |
| Contextual bandits | Improved contextual bandit exploration strategy | ✅ **Complete** |

### MP-5: Analytics Pipeline Wiring — COMPLETE

**Completed:** March 4, 2026

Wired existing analytics modules (`cost.py`, `sla.py`, `anomaly.py`, `forecasting.py`) into the live pipeline. All 23 steps from the execution plan complete.

| Step | Description | File(s) | Status |
|------|-------------|---------|--------|
| 1 | Fix CostTracker.record() bug | `adapters/base.py` | ✅ **Complete** |
| 2 | Wire SLATracker into telemetry | `adapters/base.py` | ✅ **Complete** |
| 3 | Wire Forecaster into telemetry | `adapters/base.py` | ✅ **Complete** |
| 4 | Wire AnomalyDetector into telemetry | `adapters/base.py` | ✅ **Complete** |
| 5 | Register default SLO targets | `orchestrator.py` | ✅ **Complete** |
| 6 | Apply AutoTuner config | `orchestrator.py` | ✅ **Complete** |
| 7 | CostOptimizer in ModelPool scoring | `model_pool.py` | ✅ **Complete** |
| 8 | WorkflowLearner in plan generation | `plan_mode.py` | ✅ **Complete** |
| 9 | Analytics REST API (7 endpoints) | `dashboard/rest_api.py` | ✅ **Complete** |
| 10 | Dashboard UI (Cost, SLA, Anomaly, Forecast panels) | `ui/templates/`, `ui/static/` | ✅ **Complete** |
| 11 | Integration tests | `tests/test_phase5_integration.py` | ✅ **Complete** |
| 12 | REST API tests | `tests/test_analytics_rest_api.py` | ✅ **Complete** |
| 13–23 | Additional pipeline wiring, edge cases, and observability hardening | various | ✅ **Complete** |

### MP-6: model_relay.py Deprecation — COMPLETE

**Completed:** March 7, 2026

| Item | Description | Status |
|------|-------------|--------|
| 6.1 | `model_relay.py` converted to deprecating shim — all public symbols re-exported from `model_pool.py`; `DeprecationWarning` emitted on import | ✅ **Complete** |

---

### MP-9: Standardization — COMPLETE

**Completed:** March 6–7, 2026

| Item | Artifact | Type | Location | Status |
|------|----------|------|----------|--------|
| 9.1 | Modern Python packaging | Config | `pyproject.toml` | ✅ **Complete** |
| 9.2 | Pre-commit hooks | Config | `.pre-commit-config.yaml` | ✅ **Complete** |
| 9.3 | Shared test fixtures | Code | `conftest.py` | ✅ **Complete** |
| 9.4 | PEP 561 type marker | Marker | `vetinari/py.typed` | ✅ **Complete** |

**pyproject.toml** includes: ruff (lint + format), mypy (strict on adapters/analytics), pytest config, coverage config, bandit security scanning. Replaces `setup.py`.

**pre-commit hooks** include: ruff lint + format, mypy type checking, bandit security scanning.

**conftest.py** provides: `mock_adapter`, `mock_adapter_manager`, `mock_lmstudio_response`, `fresh_shared_memory`, `flask_app`, `flask_client`, temp directory fixtures, logging suppression (autouse), custom marker registration.

### MP-10: Modularize God Modules — COMPLETE

**Completed:** March 6, 2026

Split `two_layer_orchestration.py` (1,399 lines) into 4 focused modules:

| Module | Contents | Lines |
|--------|----------|-------|
| `vetinari/orchestration/execution_graph.py` | `TaskNode`, `ExecutionGraph` | ~220 |
| `vetinari/orchestration/plan_generator.py` | `PlanGenerator` | ~230 |
| `vetinari/orchestration/durable_execution.py` | `DurableExecutionEngine`, `ExecutionEvent`, `Checkpoint` | ~380 |
| `vetinari/orchestration/two_layer.py` | `TwoLayerOrchestrator`, factory functions | ~380 |

Original file converted to thin backward-compatible shim (~55 lines) re-exporting all public names. `vetinari/orchestration/__init__.py` updated with `ExecutionTaskNode` alias to disambiguate from `agent_graph.TaskNode`.

### MP-11: Complete Features — COMPLETE

**Completed:** March 6, 2026

| Feature | Status | Files |
|---------|--------|-------|
| Per-task model config (Steps 14-18) | **Complete** | `config/task_inference_profiles.json`, `vetinari/config/inference_config.py` |
| NeMo Guardrails (Steps 19-23) | **Complete** | `vetinari/safety/guardrails.py`, `config/guardrails/` |
| File tool | **Complete** | `vetinari/tools/file_tool.py` — `FileOperations`, `FileOperationsTool`, `FileInfo`, `_safe_resolve()` path traversal protection |
| Git tool | **Complete** | `vetinari/tools/git_tool.py` — `GitOperations`, `GitOperationsTool`, `GitResult` via subprocess with timeout |
| Sandbox enforcement | **Complete** | `vetinari/code_sandbox.py` — `_restricted_import()` module blocking, network blocking, `_WRAPPER_NEEDS` exemption set |

**Tests:** 45 new tests in `tests/test_phase11_features.py` covering file ops (20), git ops (11), sandbox enforcement (9), dataclasses (5).

### MP-3: Agent Consolidation (22→8) — COMPLETE

**Completed:** March 6, 2026

Consolidated 22 single-purpose agents into 8 multi-mode agents using `MultiModeAgent` base class. Legacy agents preserved for backward compatibility; both coexist in AgentGraph (27 total slots).

| Consolidated Agent | Replaces | Modes | Key File |
|-------------------|----------|-------|----------|
| PLANNER | (1:1, unchanged) | — | `vetinari/agents/planner_agent.py` |
| ORCHESTRATOR | USER_INTERACTION + CONTEXT_MANAGER | clarify, consolidate, summarise, prune, extract, monitor | `vetinari/agents/consolidated/orchestrator_agent.py` |
| CONSOLIDATED_RESEARCHER | EXPLORER + RESEARCHER + LIBRARIAN | code_discovery, domain_research, api_lookup, lateral_thinking | `vetinari/agents/consolidated/researcher_agent.py` |
| CONSOLIDATED_ORACLE | ORACLE + PONDER | architecture, risk_assessment, ontological_analysis, contrarian_review | `vetinari/agents/consolidated/oracle_agent.py` |
| BUILDER | (1:1, unchanged) | — | `vetinari/agents/builder_agent.py` |
| ARCHITECT | UI_PLANNER + DATA_ENGINEER + DEVOPS + VERSION_CONTROL | ui_design, database, devops, git_workflow | `vetinari/agents/consolidated/architect_agent.py` |
| QUALITY | EVALUATOR + SECURITY_AUDITOR + TEST_AUTOMATION | code_review, security_audit, test_generation, simplification | `vetinari/agents/consolidated/quality_agent.py` |
| OPERATIONS | SYNTHESIZER + DOCUMENTATION_AGENT + COST_PLANNER + EXPERIMENTATION_MANAGER + IMPROVEMENT + ERROR_RECOVERY + IMAGE_GENERATOR | documentation, creative_writing, cost_analysis, experiment, error_recovery, synthesis, image_generation, improvement | `vetinari/agents/consolidated/operations_agent.py` |

**Infrastructure created:**

| Artifact | Description | File |
|----------|-------------|------|
| MultiModeAgent | Base class with mode routing, keyword inference, legacy type mapping | `vetinari/agents/multi_mode_agent.py` |
| Consolidated package | Package init with all exports | `vetinari/agents/consolidated/__init__.py` |
| AgentType entries | 6 new enum values (ORCHESTRATOR, CONSOLIDATED_RESEARCHER, etc.) | `vetinari/types.py` |
| AGENT_REGISTRY specs | 6 new AgentSpec entries with model/thinking config | `vetinari/agents/contracts.py` |
| AgentGraph registration | 21 legacy + 6 consolidated = 27 agents | `vetinari/orchestration/agent_graph.py` |

**Preserved specialized logic:**
- 25 security heuristic regex patterns in QualityAgent (from SecurityAuditorAgent)
- 10 error pattern categories with regex and quick-fix suggestions in OperationsAgent (from ErrorRecoveryAgent)
- Model pricing dictionary (7 models) in OperationsAgent (from CostPlannerAgent)
- Ambiguity detection heuristics in OrchestratorAgent (from UserInteractionAgent)
- Risk matrix scoring (likelihood x impact) in ConsolidatedOracleAgent

**Mode resolution order:** 1) Explicit `task.context["mode"]`, 2) Legacy agent type mapping via `LEGACY_TYPE_TO_MODE`, 3) Keyword matching against task description, 4) `DEFAULT_MODE` fallback.

### MP-7: Agent Wiring Completeness — COMPLETE

**Completed:** March 6, 2026

| Item | Description | Status |
|------|-------------|--------|
| 7.3b | `SkillSpec` dataclass (`vetinari/skills/skill_spec.py`) | **Complete** |
| 7.3b | Programmatic skill registry with 8 consolidated skills (`vetinari/skills/skill_registry.py`) | **Complete** |
| 7.3b | Legacy agent type → consolidated skill mapping (22 → 8) | **Complete** |
| 7.5 | 6 new `TaskType` entries in `DynamicModelRouter` (CREATIVE_WRITING, SECURITY_AUDIT, DEVOPS, IMAGE_GENERATION, COST_ANALYSIS, SPECIFICATION) | **Complete** |
| 7.5 | Enhanced `infer_task_type()` with priority ordering for specific categories | **Complete** |
| 7.6 | `GoalCategory` enum (9 categories) in `vetinari/types.py` | **Complete** |
| 7.6 | `classify_goal()` and `get_goal_routing()` functions with keyword-based 9-category routing table | **Complete** |
| 7.9C | `FailureType` enum (TRANSIENT, DECOMPOSITION, DELEGATION, UNSOLVABLE, POLICY_VIOLATION) in `vetinari/types.py` | **Complete** |
| 7.9E | `SharedExecutionContext` key-value store with provenance tracking in `vetinari/blackboard.py` | **Complete** |
| 7.9F | Replaced `time.sleep(0.1)` polling with `threading.Event` in blackboard `get_result()` | **Complete** |
| 7.9G | `REQUEST_TYPE_ROUTING` map (15 request types → agent lists) with `get_capable_agents()` API | **Complete** |

**Completed deferred items** (Phase 13.4):
- Blackboard `REQUEST_TYPE_ROUTING` updated — consolidated agents listed first with legacy fallbacks (15 request types)

**Completed deferred items** (Phase 7 batch 2):
- 7.1: SkillSpec entries validated — all 8 consolidated agents have complete specs with modes, capabilities, input/output schemas
- 7.4: Prompt assembler updated — 6 consolidated agent role defs added to `_ROLE_DEFS` in `vetinari/prompts/assembler.py`
- 7.9: Capability-based routing — `get_agent_by_capability()`, `get_skill_spec()`, `get_agents_for_task_type()` added to `AgentGraph`

**Completed deferred items** (Phase 7 batch 3):
- 7.2-7.3: Legacy `SkillRegistry` (`vetinari/registry.py`) updated — `get_skill()`, `get_skill_manifest()`, `get_skill_by_capability()`, `list_skills()`, `list_agents()`, `get_agent_skills()`, `search_skills()` all fall back to programmatic SkillSpec registry when disk manifests are missing
- 7.7-7.8: Planner prompt updated — consolidated agents listed as preferred, legacy agents kept for backward compat, affinity table (8 task-type → agent mappings), `_decompose_goal_llm` available_agents includes all 8 consolidated types, rule 9 directs LLM to prefer consolidated agents

**Completed deferred items** (Phase 7 batch 4):
- 7.9H: Permission enforcement wired into `AgentGraph._execute_task_node()` (enforce MODEL_INFERENCE before agent.execute) and `Blackboard.claim()` (check_permission gate)
- 7.9I: Dependency results incorporation — `BaseAgent._incorporate_prior_results()` extracts `context["dependency_results"]` injected by AgentGraph before execution
- 7.9J: Dynamic graph modification — `AgentGraph.inject_task()` inserts tasks mid-execution, re-wires DAG dependencies, rebuilds topological order
- 7.9K: Maker-checker pattern — QUALITY reviews BUILDER output; on rejection, BUILDER gets feedback and retries (max 3 iterations)
- 7.9A: AgentGraph wired into TwoLayerOrchestrator — `execute_with_agent_graph()` converts ExecutionGraph to contracts.Plan for AgentGraph execution, falls back to `generate_and_execute()` on failure
- 7.9B: Blackboard inter-agent delegation — `request_help()` (synchronous post+wait), `escalate_error()` (priority-1 error_recovery), `request_consensus()` (architecture_decision for multi-agent voting)

**All Phase 7 deferred items complete.** 30 new tests added (90 total in test_deferred_phases.py). Full suite: 1,662 tests passing.

### MP-8: Constraints Architecture — COMPLETE

Unified constraint architecture for agent delegation rules, resource limits, quality gates, and output validation.

| Item | Description | File(s) |
|------|-------------|---------|
| 8.1 Architecture Constraints | `ArchitectureConstraint` dataclass with delegation rules (can/cannot delegate, max depth, allowed modes/task types) for 18 agent types | `vetinari/constraints/architecture.py` |
| 8.3 Resource Constraints | `ResourceConstraint` dataclass with per-agent limits (max_tokens, max_retries, timeout, cost cap, parallelism) for 22+ agent types | `vetinari/constraints/resources.py` |
| 8.4 Quality Gates | `QualityGate` dataclass with verification score thresholds, mode-specific overrides, `check_quality_gate()` function | `vetinari/constraints/quality_gates.py` |
| 8.7 Unified Registry | `ConstraintRegistry` singleton aggregating all constraint types, violation tracking, unified query API | `vetinari/constraints/registry.py` |
| 8.8 Execution Path Wiring | Delegation validation in `AgentGraph.create_execution_plan()`, resource constraint enforcement (retry capping) in `_execute_task_node()` | `vetinari/orchestration/agent_graph.py` |
| 8.10 BaseAgent Integration | Resource constraints applied in `prepare_task()`, quality gate enforcement in `complete_task()` after quality scoring | `vetinari/agents/base_agent.py` |
| Package Init | Clean exports for all constraint types and registry accessor | `vetinari/constraints/__init__.py` |

**Completed deferred items** (Phase 13.4):
- Architecture constraints added for 5 consolidated agents (ORCHESTRATOR, CONSOLIDATED_RESEARCHER, CONSOLIDATED_ORACLE, ARCHITECT, OPERATIONS — QUALITY already existed)
- PLANNER `can_delegate_to` updated to include all 6 consolidated agent types
- Resource constraints added for all 6 consolidated agents
- Quality gates added for all 6 consolidated agents + 2 mode-specific gates (QUALITY:security_audit, QUALITY:code_review, OPERATIONS:creative_writing)

**Completed deferred items** (Phase 8 batch 2):
- 8.2: Per-agent output schema validation — `_validate_output_schema()` in `AgentGraph` checks required fields and types against SkillSpec.output_schema (non-blocking, logs deviations)
- 8.5: Document style constraints — `StyleConstraint` with `doc-no-placeholder`, `doc-consistent-headings` rules, `require_headings`/`require_sections` flags
- 8.6: Code style constraints — `StyleConstraint` with `code-no-todo`, `code-no-bare-except`, `code-no-hardcoded-secrets`, `code-no-print-debug` rules, forbidden phrases
- Style system: `vetinari/constraints/style.py` — `StyleRule`, `StyleConstraint`, `validate_output_style()`, agent-to-domain mapping, mode-specific overrides

**Completed deferred items** (Phase 8 batch 3):
- 8.9: `rules.yaml` expanded with `agents` section — all 8 consolidated agents (PLANNER, ORCHESTRATOR, RESEARCHER, ORACLE, BUILDER, ARCHITECT, QUALITY, OPERATIONS) with per-agent rules, verification scores, legacy replacement notes, and resource budgets

**Remaining deferred items**: None — all Phase 8 items complete.

### MP-13: Final Polish — COMPLETE

**Completed:** March 6, 2026

| Item | Description | Status |
|------|-------------|--------|
| 13.1 | Kaizen verification — traced learning chain in `base_agent.py:497-547` (QualityScorer → FeedbackLoop → ThompsonSampling → PromptEvolver), confirmed fully wired | **Complete** |
| 13.2a | Security hardening — `.gitignore` updated for `*.db`, `training_data.jsonl` | **Complete** |
| 13.2b | Security hardening — verified `web_ui.py` binds to `127.0.0.1` by default (line 3111) | **Complete** |
| 13.3 | Learning chain test coverage (was 0%) — `tests/test_learning_chain.py` with 35 tests covering QualityScorer (12), FeedbackLoop (5), ThompsonSampling (9), BetaArm (5), integration (2), QualityScore (2) | **Complete** |
| 13.4 | Constraint wiring for consolidated agents — architecture, resource, quality gate entries for all 6 consolidated agents + 2 mode-specific quality gates + blackboard routing updated | **Complete** |

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
| Dashboard User Guide | Docs | `docs/runbooks/dashboard-guide.md` | **Complete** |
| API Reference | Docs | `docs/api/dashboard.md` | **Complete** |
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

---

## Original Migration Phase Details

### Phase 5: Advanced Analytics — COMPLETE

**Completed:** March 3, 2026

Analytics modules (`vetinari/analytics/cost.py`, `sla.py`, `anomaly.py`, `forecasting.py`) created with unit tests. Subsequently wired into the live pipeline via Master Plan Phase 5 (analytics pipeline wiring).

### Phase 6: Production Readiness — COMPLETE

**Completed:** March 3, 2026

Core infrastructure stabilized with adapter manager, model registry, and provider abstraction layer.

### Phase 7: Drift Control — COMPLETE

**Completed:** March 3, 2026

Drift prevention mechanisms and documentation alignment strategy established.

---

## Test Suite Summary

**Total: 1,817 tests — 100% passing**

| Category | Tests | Status |
|----------|-------|--------|
| Dashboard (API, REST, alerts, logs, performance) | 156 | PASSED |
| Analytics (cost, SLA, anomaly, forecasting, autotuner) | ~120 | PASSED |
| Agents (22 legacy + 6 consolidated) | ~350 | PASSED |
| Orchestration (agent_graph, execution, planning, durable) | ~180 | PASSED |
| Adapters (LM Studio, cloud providers) | ~100 | PASSED |
| Security (sandbox, guardrails, policies) | ~80 | PASSED |
| Memory (shared memory, learning) | ~60 | PASSED |
| Learning chain (QualityScorer, FeedbackLoop, ThompsonSampling) | 35 | PASSED |
| Model config (inference profiles) | ~50 | PASSED |
| Integration tests (phase 5, REST API) | ~80 | PASSED |
| Other (tools, search, utils, web UI) | ~236 | PASSED |

---

## Key Architecture Files

| File | Purpose | Lines |
|------|---------|-------|
| `vetinari/orchestration/two_layer.py` | TwoLayerOrchestrator (assembly-line pipeline) | ~380 |
| `vetinari/orchestration/durable_execution.py` | Checkpoint-based durable execution | ~380 |
| `vetinari/orchestration/plan_generator.py` | Goal decomposition into execution graphs | ~230 |
| `vetinari/orchestration/execution_graph.py` | TaskNode & ExecutionGraph data structures | ~220 |
| `vetinari/orchestration/agent_graph.py` | AgentGraph DAG orchestration protocol | ~400 |
| `vetinari/agents/base_agent.py` | Base agent with `_infer`, `_infer_json`, `_search` | ~537 |
| `vetinari/agents/multi_mode_agent.py` | MultiModeAgent base class for consolidated agents | ~179 |
| `vetinari/agents/consolidated/` | 6 consolidated multi-mode agents (Phase 3) | ~1,800 |
| `vetinari/adapters/base.py` | Provider adapter base with telemetry | ~300 |
| `vetinari/exceptions.py` | Centralized exception hierarchy | ~118 |
| `vetinari/types.py` | Consolidated enums (TaskStatus, PlanStatus, etc.) | ~200 |
| `vetinari/utils.py` | SingletonMeta, shared utilities | ~250 |
| `vetinari/constraints/architecture.py` | Architecture delegation rules for 18 agent types | ~180 |
| `vetinari/constraints/resources.py` | Per-agent resource limits (tokens, retries, timeout) | ~170 |
| `vetinari/constraints/quality_gates.py` | Quality gate thresholds and enforcement | ~130 |
| `vetinari/constraints/registry.py` | Unified constraint registry with violation tracking | ~160 |
| `vetinari/config/inference_config.py` | Per-task model config loader | ~120 |
| `vetinari/safety/guardrails.py` | NeMo Guardrails wrapper | ~150 |
| `pyproject.toml` | Modern Python packaging & tool config | ~100 |
| `conftest.py` | Shared test fixtures | ~161 |

---

## Recommended Next Steps

All master plan phases and deferred items are complete as of March 7, 2026.

| Item | Status |
|------|--------|
| ~~Deferred Phase 7 items~~ | ✅ **ALL COMPLETE** (batches 1–4) |
| ~~Deferred Phase 8 items — rules.yaml expansion (8.9)~~ | ✅ **COMPLETE** |
| ~~MP-6 — model_relay.py shim~~ | ✅ **COMPLETE** |
| ~~Analytics Pipeline Steps 1–23~~ | ✅ **COMPLETE** |
| ~~Foundation deprecation facades (1.2–1.5, 1.8)~~ | ✅ **COMPLETE** |
| ~~MP-4b ML config externalization (ml_config.yaml)~~ | ✅ **COMPLETE** |

**Remaining stretch items (low priority):**
- MP-11: Coding bridge external bridge completion (task type stubs remain skeletal)
- MP-11: Image generation hardening (SD API integration is skeletal)

**Completed in recent sessions (Batches 6-16):**
- Deprecation facades for `enhanced_memory.py`, `model_relay.py`
- Quality scorer LLM calibration interval (every 10th task)
- AST-based code compression and security scanning
- 15 decomposition templates (was 3)
- Shared test fixtures (`conftest.py`)
- Provider model externalization (`config/provider_models.yaml`)
- Model search consolidation (`model_discovery.py`)
- VetinariConfig centralized configuration
- RFC 9457 error response helpers
- Agent affinity table completeness (29/29 AgentType entries)
- Cloud adapter telemetry (OpenAI, Cohere, Gemini)
- Flask security hardening (secret_key, debug mode)
- Ponder scoring scale fix (-100→-1)
- Dynamic resource load estimation in model_pool
- Builder `tests_added` dynamic computation (was hardcoded 5)
- `duckduckgo_search` → `ddgs` package migration (41 test warnings eliminated)
- Real webhook/email alert dispatchers (env-var configured, graceful no-op)
- Debug/warning logging added to 9 silent `except Exception: pass` patterns
- `ddgs` package installed — test suite now 0 warnings
- 151 new tests for `dynamic_model_router`, `vram_manager`, `agent_affinity` (1968 total)

**Completed in Batch 17 (test coverage expansion):**
- `test_token_optimizer.py` — 138 tests for TokenBudget, LocalPreprocessor, TokenOptimizer
- `test_cost_optimizer_learning.py` — 37 tests for CostEfficiency, CostOptimizer, budget forecast
- `test_inference_config.py` — 126 tests for InferenceProfile, model size classification, config loading
- Total: +301 tests (2243 cumulative, 0 failures)

**Completed in Batch 18 (test coverage expansion):**
- `test_plan_mode.py` — 195 tests for PlanModeEngine (domain templates, candidates, approval, coding execution)
- `test_blackboard.py` — 159 tests for Blackboard, SharedExecutionContext, thread safety, observer pattern
- `test_code_sandbox.py` — 103 tests for CodeSandbox, CodeExecutor, execution validation
- `test_decomposition.py` — 93 tests for DecompositionEngine, template filtering, keyword decomposition
- `test_goal_verifier.py` — 135 tests for GoalVerifier, compliance scoring, corrective tasks
- Total: +685 tests (2928 cumulative, 0 failures)

---

## Related Documents

- `archive/skill-migration-guide.md` — Legacy migration process and agent prompts (superseded by consolidated architecture)
- `getting-started/onboarding.md` — Developer onboarding
- `planning/drift-prevention.md` — Code/docs alignment strategy
- `../AGENTS.md` — Agent specifications
- `api/dashboard.md` — Dashboard REST API reference
- `api/analytics.md` — Analytics REST API reference
- `api/ponder.md` — Ponder API contracts
- `reference/config.md` — Configuration reference
- `reference/production.md` — Production deployment guide
- `runbooks/dashboard-guide.md` — Dashboard user guide
- `runbooks/end-to-end-workflow.md` — End-to-end orchestration workflow
- `runbooks/ponder.md` — Ponder operations runbook
