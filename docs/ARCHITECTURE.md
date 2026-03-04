# Vetinari Architecture

Version: 0.3.0 (post-comprehensive-improvement-pass)

## Overview

Vetinari is a **multi-agent AI orchestration system** that accepts structured project
goals and delivers working outputs using local LLMs via LM Studio with cloud fallback.
It is designed for minimal human intervention while keeping the user informed and in
control.

---

## System Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│  Web UI (Flask)  │  CLI (vetinari.cli)  │  REST API             │
│  Project Intake Form  │  Chat  │  Task Todo List                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    ORCHESTRATION LAYER                           │
│                                                                  │
│  TwoLayerOrchestrator (two_layer_orchestration.py)               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Stage 1: Input Analysis (goal classification)          │    │
│  │  Stage 2-3: Plan Generation + Task Decomposition (DAG)  │    │
│  │  Stage 4: Model Assignment (DynamicModelRouter)         │    │
│  │  Stage 5: Parallel Execution (DurableExecutionEngine)   │    │
│  │  Stage 6: Output Review (EvaluatorAgent)                │    │
│  │  Stage 7: Final Assembly (SynthesizerAgent)             │    │
│  │  Stage 8: Goal Verification (GoalVerifier)              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Legacy: Orchestrator (orchestrator.py) — manifest-based tasks  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                       AGENT LAYER (22 agents)                    │
│                                                                  │
│  Planning:    PlannerAgent, UserInteractionAgent                 │
│  Research:    ExplorerAgent, OracleAgent, LibrarianAgent,        │
│               ResearcherAgent                                    │
│  Building:    BuilderAgent, UIPlannnerAgent, DataEngineerAgent   │
│               DevOpsAgent, VersionControlAgent                   │
│               ImageGeneratorAgent (NEW - Stable Diffusion/SVG)  │
│  Quality:     EvaluatorAgent, SecurityAuditorAgent               │
│               TestAutomationAgent                                │
│  Learning:    ImprovementAgent, ExperimentationManagerAgent      │
│  Meta:        SynthesizerAgent, DocumentationAgent               │
│               CostPlannerAgent, ContextManagerAgent              │
│               ErrorRecoveryAgent                                 │
│                                                                  │
│  All inherit from BaseAgent, which provides:                     │
│  - _infer() / _infer_json() via AdapterManager                   │
│  - Learning hooks (QualityScorer, FeedbackLoop, Thompson)        │
│  - Training data collection                                      │
│  - Rules injection via RulesManager                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                       MODEL LAYER                                │
│                                                                  │
│  Adapters: LMStudio (primary), OpenAI, Anthropic, Gemini, Cohere│
│  DynamicModelRouter: capability-weighted + Thompson Sampling     │
│  ModelSearchEngine: unified live+cached search (HF, Reddit, GH) │
│  VRAMManager: GPU memory tracking + eviction recommendations     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    LEARNING & SELF-IMPROVEMENT                   │
│                                                                  │
│  QualityScorer → FeedbackLoop → ThompsonSampling                 │
│  PromptEvolver (A/B testing) → WorkflowLearner                   │
│  CostOptimizer → AutoTuner → ImprovementAgent                    │
│  TrainingDataCollector → TrainingPipeline (QLoRA)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Modules

### Core Orchestration

| Module | Purpose |
|--------|---------|
| `two_layer_orchestration.py` | **Primary** 8-stage assembly-line pipeline |
| `orchestrator.py` | Legacy manifest-based orchestrator |
| `planning_engine.py` | Legacy keyword-based planning (deprecated path) |
| `agents/planner_agent.py` | LLM-powered goal decomposition (22 agent types) |

### New Modules (v0.3.0)

| Module | Purpose |
|--------|---------|
| `types.py` | **Canonical** enums: TaskStatus, PlanStatus, AgentType, ModelProvider |
| `constants.py` | All default values and tuning parameters |
| `rules_manager.py` | Hierarchical rules: global → project → model → combo |
| `goal_verifier.py` | Post-delivery compliance check against original spec |
| `decomposition.py` | Decomposition Lab engine (wraps PlannerAgent) |
| `decomposition_agent.py` | Decomposition agent interface |
| `assignment_pass.py` | Model/agent assignment for subtask tree |
| `multi_agent_orchestrator.py` | Agent status tracking for web UI |
| `agents/image_generator_agent.py` | Stable Diffusion + SVG fallback image generation |

### Configuration

| File | Purpose |
|------|---------|
| `vetinari.yaml` | Primary project manifest |
| `config/models.yaml` | Hardware-aware model catalog |
| `config/sandbox_policy.yaml` | Code execution security |
| `.env` | Environment variables (API keys, hosts) |
| `rules.yaml` | Auto-generated rules configuration (NEW) |

---

## Planning Pipeline (Detailed)

### Project Intake
```
User fills Project Intake Form:
  - Project Name
  - General Goal (required)
  - Detailed Description
  - Required Features (list)
  - Things to Avoid (list)
  - Target Platform (checkboxes: web, api, cli, desktop, mobile, library, data, ml)
  - Tech Stack (comma-separated)
  - Priority (quality/speed/cost)
  - Expected Outputs (checkboxes: code, tests, docs, ci, docker, assets)
  - Hardware Profile (auto-detected from /api/status)
```

### Plan Generation Flow
```
1. UserInteractionAgent.detect_ambiguity() — ask clarifying questions if needed
2. PlannerAgent._generate_plan() — LLM decomposes goal into Task DAG
   - Uses all 22 agent types in system prompt
   - Calculates real DAG depth (not dependency count)
   - Returns tasks with acceptance_criteria per task
   - Falls back to keyword decomposition if LLM fails or < 3 tasks
3. RulesManager.build_system_prompt_prefix() — inject rules into every agent
4. TwoLayerOrchestrator.generate_and_execute() — run enriched goal through pipeline
```

### Goal Verification Flow
```
After Stage 7 (Final Assembly):
8. GoalVerifier.verify():
   - Feature compliance check (heuristic keyword matching + LLM evaluation)
   - Security audit (SecurityAuditorAgent)
   - Test presence detection
   - Returns GoalVerificationReport with compliance_score (0.0-1.0)
   
If not fully_compliant:
   - Generate corrective tasks (GetCorrectiveTasks())
   - Re-enter execution pipeline
   - Present gaps to user with verification matrix UI
   - Accept user feedback and create follow-up tasks
```

---

## Rules System

Rules are injected into every agent's system prompt in hierarchical order:

```
GLOBAL RULES (all projects, all models)
  → PROJECT RULES (this project, all models)
    → MODEL RULES (all projects, this model)
      → PROJECT+MODEL RULES (this project, this model)
```

**Storage:** `rules.yaml` in project root  
**API:**
- `GET/POST /api/rules/global` — global rules list
- `GET/POST /api/rules/global-prompt` — global system prompt override  
- `GET/POST /api/rules/project/<id>` — project-specific rules  
- `GET/POST /api/rules/model/<id>` — model-specific rules  
**UI:** Settings → "Global Rules & System Prompts" section

---

## Image Generation

The `ImageGeneratorAgent` supports two backends:

1. **Stable Diffusion WebUI API** (primary)
   - Requires Automatic1111 running with `--api` flag
   - Configure: `SD_WEBUI_HOST=http://localhost:7860`
   - Enable: `SD_WEBUI_ENABLED=true`
   - API: `POST /api/generate-image`
   
2. **LLM-generated SVG** (fallback, always available)
   - Works without any extra server
   - Suitable for logos, icons, diagrams
   - Generates valid SVG code via the current LLM

**Trigger keywords:** "logo", "icon", "artwork", "visual", "diagram", "mockup"  
**Planner:** Automatically assigns IMAGE_GENERATOR tasks when these are detected

---

## Training Pipeline (Hierarchical)

Training data is collected from every agent execution and organized into tiers:

```
TIER 1 - GENERAL: All models, basic instruction following
TIER 2 - CATEGORY: Models by role (coding, research, review, planning)  
TIER 3 - INDIVIDUAL: Per-model fine-tuning
```

**Pipeline:** TrainingDataCollector → DataCurator → LocalTrainer (QLoRA/unsloth) →  
GGUFConverter → ModelDeployer → LM Studio  
**API:** `GET /api/training/stats`, `POST /api/training/export`, `POST /api/training/start`  
**UI:** Settings → "Model Training" section

---

## Data Flow

```
Project Intake Form
      │
      ▼
/api/new-project (POST)
      │ goal + required_features + things_to_avoid + tech_stack + ...
      ▼
TwoLayerOrchestrator.generate_and_execute()
      │ enriched_goal (with features, avoid list, tech stack)
      │ context._rules_prefix (from RulesManager)
      ▼
PlannerAgent → TaskDAG (ExecutionGraph)
      │
      ▼
DurableExecutionEngine (per-task parallel execution, max_concurrent respected)
      │ each task → BaseAgent → _infer() via AdapterManager
      │ post-task: QualityScorer → FeedbackLoop → ThompsonSampling
      ▼
EvaluatorAgent (Stage 6 review)
      │
      ▼
SynthesizerAgent (Stage 7 final assembly)
      │
      ▼
GoalVerifier (Stage 8 compliance check)
      │
      ├─ Compliant → final_delivery_panel (markdown rendered)
      └─ Gaps found → corrective_tasks → back to execution
```

---

## Bugs Fixed in v0.3.0

| Bug | Location | Fix |
|-----|---------|-----|
| Phantom imports (4 missing modules) | web_ui.py | Created stub implementations |
| Duplicate `/api/search` route | web_ui.py | Renamed code search to `/api/code-search` |
| `max_concurrent` never enforced | two_layer_orchestration.py:649 | Use `min(len(layer), max_concurrent)` |
| `assigned_model` read from wrong field | two_layer_orchestration.py:721 | Read from `task.input_data` |
| Transitive task cancellation | two_layer_orchestration.py:794 | Iterative expansion to all dependents |
| PlannerAgent verify always fails on warnings | planner_agent.py:149 | Separate score from issues |
| `min_tasks` gate discards valid simple plans | planner_agent.py:249 | Return all tasks (min=3) |
| Cache hash non-deterministic | model_search.py | Use `hashlib.md5()` instead of `hash()` |
| Cache deserialization crash | model_search.py | Deserialize provenance on load |
| Reddit `_parse_post` returns only first mention | live_model_search.py | Process best model name |
| Training script code injection | training/pipeline.py | Use `json.dumps()` for all params |
| `debug=True` in production | web_ui.py | Env var controlled |
| Hardcoded Tailscale IP in relay | model_relay.py | Use env var |
| Bare `except:` clauses (9) | Various | Changed to `except Exception:` |

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LM_STUDIO_HOST` | `http://localhost:1234` | LM Studio server |
| `LM_STUDIO_API_TOKEN` | empty | API authentication |
| `VETINARI_WEB_PORT` | `5000` | Web UI port |
| `VETINARI_WEB_HOST` | `0.0.0.0` | Web UI bind address |
| `FLASK_DEBUG` | `false` | Flask debug mode |
| `PLAN_MODE_ENABLE` | `true` | Enable plan-first execution |
| `PLAN_DEPTH_CAP` | `16` | Max task decomposition depth |
| `SD_WEBUI_HOST` | `http://localhost:7860` | Stable Diffusion WebUI |
| `SD_WEBUI_ENABLED` | `false` | Enable image generation |
| `CODING_BRIDGE_HOST` | `http://localhost:4096` | External coding agent |
| `CODING_BRIDGE_ENABLED` | `false` | Enable coding bridge |
| `ENABLE_EXTERNAL_DISCOVERY` | `true` | Enable external model search |
| `VETINARI_SEARCH_BACKEND` | `duckduckgo` | Web search backend |
| `CLAUDE_API_KEY` | empty | Anthropic Claude API key |
| `GEMINI_API_KEY` | empty | Google Gemini API key |
| `OPENAI_API_KEY` | empty | OpenAI API key |

---

## Testing

```
Total tests: 1,376 passing (non-integration)
Coverage: ~65%

Key test files:
  tests/test_vetinari.py           - Core system tests
  tests/test_base_agent.py         - BaseAgent tests  
  tests/test_agent_contracts.py    - Contract tests
  tests/test_new_modules.py        - All new v0.3.0 modules (75 tests)
  tests/test_builder_skill.py      - Builder agent
  tests/test_evaluator_skill.py    - Evaluator agent
  tests/test_dashboard_*.py        - Dashboard API (5 files)
  tests/test_analytics_*.py        - Analytics modules (4 files)
  tests/test_ponder.py             - Model ranking
  tests/test_security.py           - Security scanning
```
