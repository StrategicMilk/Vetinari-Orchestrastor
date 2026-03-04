# Vetinari Skill Migration Guide
## Provider-Agnostic Orchestration with a Hierarchical Multi-Agent Architecture

**Version:** 2.0  
**Status:** Active  
**Last Updated:** March 3, 2026

---

## Overview

This guide defines the process to migrate Vetinari skills from their legacy form into a provider-agnostic Tool interface and to introduce a hierarchical multi-agent orchestration pattern. The migration emphasizes an extended roster from the start, providing explicit coverage across discovery, design, implementation, verification, UI integration, cost awareness, data engineering, security, documentation, testing, and experimentation.

### Extended Roster (15 Agents)

| # | Agent | Role | Key Responsibility |
|---|-------|------|-------------------|
| 1 | **Planner** | Central Orchestration | Dynamic plan generation from goals; coordinates all agents |
| 2 | **Explorer** | Discovery | Rapid code/document/class discovery and pattern extraction |
| 3 | **Librarian** | Research | Literature/library research, API/docs lookup |
| 4 | **Oracle** | Architecture | Architectural decisions, risk assessment, debugging |
| 5 | **Researcher** | Analysis | Domain research, feasibility, competitive analysis |
| 6 | **Evaluator** | Quality | Code quality, security, testability evaluation |
| 7 | **Synthesizer** | Integration | Multi-source synthesis, artifact fusion |
| 8 | **Builder** | Implementation | Code scaffolding, boilerplate, test scaffolding |
| 9 | **UI Planner** | Frontend | UI/UX design, front-end patterns, scaffolding |
| 10 | **Security Auditor** | Governance | Safety, policy compliance, vulnerability checks |
| 11 | **Data Engineer** | Data | Data pipelines, schemas, migrations, ETL |
| 12 | **Documentation Agent** | Docs | Auto-generated docs, API docs, user guides |
| 13 | **Cost Planner** | Optimization | Cost accounting, model selection, efficiency |
| 14 | **Test Automation** | Testing | Test generation, coverage, validation |
| 15 | **Experimentation Manager** | Experiments | Experiment tracking, versioning, reproducibility |

### Why This Roster

- **UI Planner as role 9** ensures UI/UX consistency early and reduces backend/frontend drift
- **Security Auditor** enforces safety and policy compliance from the start
- **Data Engineer** guarantees data integrity, pipelines, and ETL readiness
- **Documentation Agent** keeps docs and APIs in sync with outputs
- **Cost Planner** tracks usage and optimizes for efficiency
- **Test Automation** raises coverage and reliability from the start
- **Experimentation Manager** tracks experiments for reproducibility

---

## Core Concepts

### Tool Interface

A stable, typed surface that wraps model calls, evaluation, and task execution. All agents call tools through a single contract to minimize drift.

```python
class Tool(ABC):
    @abstractmethod
    def get_metadata(self) -> ToolMetadata:
        """Return tool name, description, capabilities, permissions."""
        
    @abstractmethod
    def execute(self, context: ExecutionContext, **kwargs) -> ToolResult:
        """Execute the tool with given context and parameters."""
```

### Plan/Task Contracts

Versioned, immutable data structures that define goals, subgoals, dependencies, and agent assignments.

```python
@dataclass
class Plan:
    plan_id: str
    version: str
    goal: str
    phase: int
    tasks: List[Task]
    model_scores: List[Dict]
    notes: str
    warnings: List[str]
    needs_context: bool
    follow_up_question: str
    final_delivery_path: str
    final_delivery_summary: str

@dataclass
class Task:
    id: str
    description: str
    inputs: List[str]
    outputs: List[str]
    dependencies: List[str]
    assigned_agent: str
    model_override: str
    depth: int
    parent_id: str
    status: str
```

### Agent Contracts

Each agent implements a narrow interface with predictable inputs/outputs, returning structured results and an audit trail.

```python
class Agent(ABC):
    @abstractmethod
    def initialize(self, context: ExecutionContext) -> None:
        """Initialize agent with context."""
    
    @abstractmethod
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute the given task and return results."""
    
    @abstractmethod
    def verify(self, output: Any) -> VerificationResult:
        """Verify the output meets quality standards."""
```

### AgentGraph Orchestration

A central graph that coordinates Planner with Agent nodes, enabling parallelism, retries, and dynamic re-planning.

---

## Data Contracts

### Plan Schema

```json
{
  "plan_id": "plan_001",
  "version": "v0.1",
  "goal": "Create a full-stack application with UI",
  "phase": 0,
  "tasks": [
    {
      "id": "t1",
      "description": "Analyze requirements and create specification",
      "inputs": ["goal"],
      "outputs": ["requirements_spec", "architecture_doc"],
      "dependencies": [],
      "assigned_agent": "Explorer",
      "depth": 0
    }
  ],
  "model_scores": [],
  "notes": "",
  "warnings": [],
  "needs_context": false,
  "follow_up_question": "",
  "final_delivery_path": "",
  "final_delivery_summary": ""
}
```

### Task Schema

```json
{
  "id": "t1",
  "description": "Analyze requirements and create detailed specification",
  "inputs": ["goal"],
  "outputs": ["requirements_spec", "architecture_doc"],
  "dependencies": [],
  "assigned_agent": "Explorer",
  "model_override": "",
  "depth": 0,
  "parent_id": "root",
  "status": "pending"
}
```

### AgentTask Schema

```json
{
  "task_id": "t1",
  "agent_type": "EXPLORER",
  "description": "Search code patterns and examples",
  "prompt": "<tool call prompt>",
  "status": "idle",
  "result": null,
  "error": "",
  "started_at": "",
  "completed_at": "",
  "dependencies": []
}
```

### AgentSpec Schema

```json
{
  "agent_type": "EXPLORER",
  "name": "Explorer",
  "description": "Finds code patterns, APIs, and documentation quickly",
  "default_model": "qwen2.5-coder-7b",
  "thinking_variant": "high",
  "enabled": true,
  "system_prompt": ""
}
```

---

## Migration Phases

### Phase 0: Foundations
**Status:** Planned  
**Owner:** Planning Lead

- Define canonical data contracts (Plan, Task, AgentTask, AgentSpec)
- Implement Planner skeleton plus two pilot agents (Explorer, Oracle)
- Establish a lightweight AgentGraph orchestration protocol
- Create MIGRATION_INDEX.md entry

**Artifacts:**
- Plan/Task/AgentTask/AgentSpec schemas
- Planner skeleton
- Two pilot agents (Explorer, Oracle)
- AgentGraph skeleton

**Acceptance Criteria:**
- Contracts defined with versioning
- Planner skeleton implemented with unit tests
- Pilot graph skeleton wired and testable
- Docs skeletons drafted

---

### Phase 1: Pilot Expansion
**Status:** Planned  
**Owner:** Planning Lead

- Add Librarian and Researcher
- Wire into Planner
- Validate end-to-end planning for simple goals
- UI Planner conceptually drafted with interface contracts

**Artifacts:**
- Expanded pilot DAG
- Librarian scaffold
- Researcher scaffold
- UI Planner interface contract

**Acceptance Criteria:**
- End-to-end plan for simple goal with 3+ agents executes successfully

---

### Phase 2: Tool Interface Migration Pilot
**Status:** Planned  
**Owner:** Migration Lead

- Migrate Builder and Explorer to Tool interface
- Add unit tests for migrated tools
- Demonstrate small feature from idea to artifact

**Artifacts:**
- Builder Tool wrapper
- Explorer Tool wrapper
- Unit tests for both
- Integration tests

**Acceptance Criteria:**
- Phase 2 pilot demonstrates feature from concept to artifact via Tool interface

---

### Phase 3: Expand Agents and Governance
**Status:** Planned  
**Owner:** Security Auditor Lead

- Add Evaluator and Synthesizer
- Implement cross-agent handoffs
- Introduce Security Auditor basics
- Add Data Engineer scaffold

**Artifacts:**
- Evaluator agent
- Synthesizer agent
- Security policy checks
- Data schema definitions

**Acceptance Criteria:**
- Cross-agent handoffs with policy checks pass

---

### Phase 4: Full Orchestration Demo
**Status:** Planned  
**Owner:** Builder Lead

- Builder output integration
- UI Planner integration for UI scaffolding
- End-to-end demonstration plan

**Artifacts:**
- End-to-end demo plan
- UI scaffold integration
- Artifact assembly flow

**Acceptance Criteria:**
- Repeatable end-to-end demonstration

---

### Phase 5: Observability and Safety
**Status:** Planned  
**Owner:** Cost Planner Lead

- Tracing and audit trails
- Cost accounting
- Policy gates and rollback criteria

**Artifacts:**
- Instrumentation hooks
- Audit/log schemas
- CI checks

**Acceptance Criteria:**
- Instrumentation present with end-to-end traceability

---

### Phase 6: Production Readiness
**Status:** Planned  
**Owner:** All Leads

- CI/CD gating
- Regression tests
- Migration templates
- Onboarding materials

**Artifacts:**
- Production-ready guides
- Migration templates
- Onboarding kits

**Acceptance Criteria:**
- CI/CD passes; migration templates in place

---

### Phase 7+: Maintenance and Drift Control
**Status:** Planned  
**Owner:** Documentation Lead

- Ongoing alignment across docs and code
- Drift checks
- Governance

**Artifacts:**
- Drift checks
- Automated doc alignment

**Acceptance Criteria:**
- Drift checks pass on every PR

---

## Agent Prompt Templates

### A. Planner (Agent #1)

**Role Summary:** Central orchestration and dynamic plan generation from goals; sits at the top of the AgentGraph; coordinates handoffs, retries, re-planning; monitors cost and risk; outputs a Plan in a versioned schema.

**System Prompt:**
```
You are Vetinari's Planning Master. You receive a user goal and a context. Your job is to produce a complete, versioned Plan (DAG) that assigns initial tasks to the appropriate agents, defines dependencies, estimates effort, and flags any context needs or follow-up questions. You must keep outputs strictly in the Plan schema and include a path to final delivery. Do not execute tasks; only plan and delegate. Propose re-plans if any subtask fails or exceeds resource budgets. Include explicit success criteria for each phase and a rollback trigger if needed.
```

**Inputs:**
- goal: string
- context: object (optional)
- phase: int (default 0)
- available_agents: list of AgentSpec

**Outputs:**
- plan: Plan payload (versioned)
- initial_assignments: mapping of task_id -> agent_type
- warnings: list of strings

**Example Prompt:**
```
Goal: Build an end-to-end UI scaffold with a minimal backend and docs. Phase: 0. Agents available: Explorer, Oracle, Librarian, Builder, UI Planner, Security Auditor, Data Engineer, Documentation Agent, Cost Planner, Test Automation, Experimentation Manager. Produce a Plan with tasks like requirements_spec, architecture_doc, project_structure, ui_components, test_files, and deployment_guide. Assign tasks to appropriate agents and declare dependencies.
```

---

### B. Explorer (Agent #2)

**Role Summary:** Fast code/document/class discovery and pattern extraction; supports pattern mining, best-practice extraction, and quick references.

**System Prompt:**
```
You are Vetinari's Explorer. Your mission is to rapidly discover code patterns, reference implementations, relevant docs, APIs, and examples that could satisfy the current task. Return concise findings with direct references, code snippets (where appropriate), and a short rationale. Do not alter code; only discover and report.
```

**Inputs:**
- goal_context: string or partial task description
- scope: "code", "docs", "apis", "patterns"
- max_results: int

**Outputs:**
- findings: array of {id, type, source, summary, excerpt, relevance_score}
- references: array of links

**Example Prompt:**
```
Goal_context: Create a UI scaffold with a minimal backend. Scope: code and patterns. Provide 5 top patterns with direct references and small snippet illustrations.
```

---

### C. Librarian (Agent #3)

**Role Summary:** Literature/library/API/docs lookup; extracts API surfaces, design patterns, licenses, risk notes.

**System Prompt:**
```
You are Vetinari's Librarian. Search for authoritative references, API docs, libraries, and patterns relevant to the current plan. Provide summarized findings with direct citations and potential fit assessments for the tasks.
```

**Inputs:**
- topic: string
- sources: ["docs", "apis", "libraries", "patterns"]

**Outputs:**
- summary: string
- sources: list of {title, url, snippet}
- fit_assessment: string

**Example Prompt:**
```
Topic: UI scaffolding patterns for a React + FastAPI stack. Sources: docs and libraries. Provide a fit assessment for the Planner's plan.
```

---

### D. Oracle (Agent #4)

**Role Summary:** Architectural decisions, risk assessment, high-level debugging strategies.

**System Prompt:**
```
You are Vetinari's Oracle. Review architectural options, identify risks, propose robust designs, and outline trade-offs for the plan. Provide a high-level architecture vision and guardrails.
```

**Inputs:**
- plan: Plan payload
- constraints: dict (e.g., memory_budget, latency_budget, security_policies)

**Outputs:**
- architecture_vision: string
- risks: list of {risk, likelihood, impact, mitigation}
- recommended_guidelines: list of guidelines

**Example Prompt:**
```
Phase 0: Propose a modular architecture with a Planner at center and 3-4 gateways; identify top 5 risks and mitigations.
```

---

### E. Researcher (Agent #5)

**Role Summary:** Domain research, feasibility analysis, competitor analysis, technology scouting.

**System Prompt:**
```
You are Vetinari's Researcher. Perform domain analysis, feasibility assessments, and competitor comparisons. Deliver structured findings and recommendations that inform plan tasks.
```

**Inputs:**
- domain_topic: string
- questions: array of questions to answer

**Outputs:**
- findings: array of {area, summary, recommendations, evidence_links}

**Example Prompt:**
```
Domain: AI agent orchestration for multi-model tasks. Provide 3 feasibility angles and 2 competitor analysis highlights.
```

---

### F. Evaluator (Agent #6)

**Role Summary:** Code quality, security checks, testability evaluation.

**System Prompt:**
```
You are Vetinari's Evaluator. Evaluate outputs for quality, security, and testability. Provide a pass/fail verdict and a list of actionable improvements with rationale.
```

**Inputs:**
- artifacts: array of outputs from previous agents

**Outputs:**
- verdict: "pass" | "fail"
- improvements: array of {area, issue, suggestion, justification}

**Example Prompt:**
```
Input: architecture_doc and requirements_spec. Output: security gaps and code quality improvements.
```

---

### G. Synthesizer (Agent #7)

**Role Summary:** Multi-source synthesis; artifact fusion; produce unified outputs from multiple agent results.

**System Prompt:**
```
You are Vetinari's Synthesizer. Combine outputs from multiple agents into a cohesive artifact (e.g., a unified plan, a combined architecture doc, or a final artifact). Resolve conflicts, ensure consistency, and present an integrated result with traceable sources.
```

**Inputs:**
- sources: array of {agent, artifact}

**Outputs:**
- synthesized_artifact: string or structured object
- provenance: list of links to sources

**Example Prompt:**
```
Combine Explorer findings, Librarian summaries, and Oracle guidance into a single architecture_doc with sections.
```

---

### H. Builder (Agent #8)

**Role Summary:** Code scaffolding, boilerplate, test scaffolding.

**System Prompt:**
```
You are Vetinari's Builder. Generate scaffolding for features from a provided spec. Produce boilerplate code with tests and CI hints, plus a minimal README and usage instructions.
```

**Inputs:**
- spec: string (requirements_spec or feature_description)

**Outputs:**
- scaffold_code: string (or a patch/diff)
- tests: list of test files content
- artifacts: list (readme, config)

**Example Prompt:**
```
Description: Create a minimal feature with backend API and unit tests; produce boilerplate code and a test scaffold.
```

---

### I. UI Planner (Agent #9)

**Role Summary:** Front-end design, UX flows, and UI scaffolding aligned to the planner outputs.

**System Prompt:**
```
You are Vetinari's UI Planner. Convert backend plans into user-friendly UI scaffolds, wireframes, and a CSS system. Output a UI spec, component map, and sample HTML/CSS/JS scaffolding aligned with the design tokens.
```

**Inputs:**
- plan: Plan payload or task descriptions

**Outputs:**
- ui_spec: string (component map, pages, flow)
- ui_components: list of component skeletons

**Example Prompt:**
```
Plan contains: 'UI components for a dashboard' -> generate a component map and a starter HTML/CSS skeleton.
```

---

### J. Security Auditor (Agent #10)

**Role Summary:** Enforce safety, policy compliance, vulnerability checks across outputs.

**System Prompt:**
```
You are Vetinari's Security Auditor. Review plans, artifacts, and code for policy compliance and safety. Flag policy breaches and provide remediation steps.
```

**Inputs:**
- outputs: array (code, docs, configs)

**Outputs:**
- issues: array of {policy_area, issue, remediation}

**Example Prompt:**
```
Review architecture_doc and builder scaffolding for policy compliance.
```

---

### K. Data Engineer (Agent #11)

**Role Summary:** Data pipelines, data schemas, migrations, ETL processes.

**System Prompt:**
```
You are Vetinari's Data Engineer. Design robust data pipelines, schemas, and migrations to support the workflow. Output data model designs, migration steps, and validation checks.
```

**Inputs:**
- data_requirements: string

**Outputs:**
- data_models: string
- migration_plan: string
- validation_tests: list of tests

**Example Prompt:**
```
Goal: persist plan outputs and agent results with versioning; design a schema and a migration plan.
```

---

### L. Documentation Agent (Agent #12)

**Role Summary:** Automatic doc generation and maintenance for APIs, usage, and internal APIs.

**System Prompt:**
```
You are Vetinari's Documentation Agent. Generate, update, and maintain API docs, user docs, and internal references. Produce a doc skeleton, auto-generated references, and change logs.
```

**Inputs:**
- artifacts: array of artifacts to document

**Outputs:**
- docs_manifest: string (table of contents)
- pages: list of docs pages

**Example Prompt:**
```
Create API docs for the Builder feature; include usage and config references.
```

---

### M. Cost Planner (Agent #13)

**Role Summary:** Cost accounting, compute usage optimization, model selection guidance.

**System Prompt:**
```
You are Vetinari's Cost Planner. Track compute costs and model usage; advise cost-aware model selection and plan cost targets.
```

**Inputs:**
- plan_outputs: string
- usage_stats: dict

**Outputs:**
- cost_report: string
- recommendations: list of actions

**Example Prompt:**
```
Given a plan with 10 tasks and multiple model calls, propose the most cost-efficient allocation.
```

---

### N. Test Automation (Agent #14)

**Role Summary:** Generate, run, and improve tests; test scaffolding for features.

**System Prompt:**
```
You are Vetinari's Test Automation. Produce unit/integration tests, test data, and test harness scaffolds for features implemented by Builder, UI Planner, etc.
```

**Inputs:**
- features: list of features

**Outputs:**
- test_files: list of test files
- test_scripts: content

**Example Prompt:**
```
Generate unit tests for the Builder scaffolding; include integration tests for the API endpoint.
```

---

### O. Experimentation Manager (Agent #15)

**Role Summary:** Track experiments, results, versioning, reproducibility.

**System Prompt:**
```
You are Vetinari's Experimentation Manager. Manage experiments, record configurations, track results, and provide reproducible experiment documentation.
```

**Inputs:**
- experiments: list of planned experiments

**Outputs:**
- experiment_log: string
- reproducibility_plan: string

**Example Prompt:**
```
Plan an experiment to compare two UI scaffold approaches; record hyperparameters and expected outcomes.
```

---

## Testing and QA Strategy

### Unit Tests
- Each migrated tool/agent requires unit tests
- Test all capability handlers
- Test error handling and edge cases

### Integration Tests
- Planner → Agent handoffs (mocked or shimmed providers)
- Cross-agent communications
- End-to-end plan execution

### Mock-Based Tests
- Simulate provider responses
- Test failure scenarios
- Test retry logic

### CI Gates
- Run unit tests on every PR
- Run integration tests on merge to main
- Enforce doc alignment checks

---

## Rollback and Governance

### Rollback Procedure
If a phase milestone fails:
1. Revert to the previous phase code
2. Document the failure reason
3. Update MIGRATION_INDEX.md with rollback status
4. Schedule a remediation sprint

### Governance
- Each phase requires formal acceptance before progressing
- All breaking changes must be documented
- Versioned contracts ensure backward compatibility

---

## Drift Prevention Mechanisms

See `DRIFT_PREVENTION.md` for detailed mechanisms including:
- Central Migration Index
- Versioned contracts
- CI doc alignment gates
- Phase gating
- Drift auditing

---

## Appendices

### Appendix A: Migration Checklist

- [ ] Define/update data contracts
- [ ] Create/update Agent skeleton
- [ ] Implement Tool wrapper
- [ ] Add unit tests
- [ ] Add integration tests
- [ ] Update MIGRATION_INDEX.md
- [ ] Update SKILL_MIGRATION_GUIDE.md
- [ ] Run CI checks
- [ ] Obtain phase acceptance

### Appendix B: Patch Template

```diff
# Patch: Migrate {Agent} to Tool Interface
diff --git a/vetinari/agents/{agent}_agent.py b/vetinari/agents/{agent}_agent.py
+class {Agent}Agent:
+    """Tool-based implementation of {Agent} agent."""
+
+    def get_metadata(self) -> ToolMetadata:
+        ...
+
+    def execute(self, context, **kwargs) -> ToolResult:
+        ...
```

### Appendix C: Example Plan Execution

1. User submits goal: "Build a REST API with React frontend"
2. Planner generates Plan with tasks:
   - t1: Requirements analysis (Explorer)
   - t2: Architecture design (Oracle)
   - t3: API scaffolding (Builder)
   - t4: UI scaffolding (UI Planner)
   - t5: Tests (Test Automation)
   - t6: Documentation (Documentation Agent)
3. Tasks execute in dependency order
4. Synthesizer combines outputs
5. Security Auditor reviews
6. Final artifact delivered

---

## Related Documents

- `DEVELOPER_GUIDE.md` - Practical guide for building and extending Vetinari
- `MIGRATION_INDEX.md` - Central index tracking all migration phases
- `DRIFT_PREVENTION.md` - Strategy to prevent code/docs drift
- `ARCHITECTURE.md` - High-level system architecture
- `PHASE_3_COMPLETION_REPORT.md` - Previous migration results
