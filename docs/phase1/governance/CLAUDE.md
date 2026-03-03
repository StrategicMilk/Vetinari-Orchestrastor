# Vetinari Development Guide

## Overview

This is the development guide for Vetinari. This file is auto-generated - see `/docs/governance/` for the governance process.

## Project Structure

```
Vetinari/
├── vetinari/              # Core application
│   ├── web_ui.py         # Flask web server
│   ├── planning_engine.py # Plan generation
│   ├── multi_agent_orchestrator.py # Agent orchestration
│   ├── shared_memory.py  # Hive mind memory
│   ├── model_pool.py     # Model management
│   ├── cocoindex_client.py # Code search
│   └── ...
├── skills/               # Agent skill definitions
│   ├── explorer/
│   ├── librarian/
│   └── ...
├── ui/                   # Web interface
│   ├── templates/
│   └── static/
├── projects/             # User projects
├── docs/                 # Documentation
│   ├── phase1/          # Phase 1 specs
│   └── governance/     # Governance docs
└── tests/               # Test suite
```

## Key Concepts

### Planning

Plans contain Waves, Waves contain Tasks:
- **Plan**: High-level goal (e.g., "Add authentication")
- **Wave**: Milestone (e.g., "Research", "Implementation")
- **Task**: Atomic work unit assigned to an agent

### Memory

All agent interactions are stored in SharedMemory:
- Memories tagged by type (intent, discovery, decision, etc.)
- Memories tagged by agent
- Searchable timeline

### Model Selection

Model Relay picks best available model:
1. Filter by capabilities
2. Score by policy (privacy, latency, cost)
3. Select highest score
4. Fallback to cloud if needed

### Sandbox

Two-layer execution safety:
- **Layer 1**: In-process sandbox for quick code
- **Layer 2**: External plugin sandbox with permissions

## Configuration

| File | Purpose |
|------|---------|
| `vetinari.yaml` | Main configuration |
| `models.yaml` | Model catalog |
| `sandbox_policy.yaml` | Security policies |

## API Endpoints

### Plans
- `POST /api/plans` - Create plan
- `GET /api/plans` - List plans
- `GET /api/plans/{id}` - Get plan
- `POST /api/plans/{id}/start` - Start execution
- `POST /api/plans/{id}/pause` - Pause
- `POST /api/plans/{id}/resume` - Resume

### Models
- `GET /api/models` - List models
- `POST /api/models/select` - Select model
- `GET /api/models/policy` - Get policy
- `PUT /api/models/policy` - Update policy

### Sandbox
- `POST /api/sandbox/execute` - Execute code
- `GET /api/sandbox/status` - Status
- `GET /api/sandbox/audit` - Audit log

### Search
- `GET /api/search` - Search code
- `POST /api/search/index` - Index project
- `GET /api/search/status` - Backend status

## Development Workflow

1. Create a plan from a prompt
2. Plan generates waves and tasks
3. Tasks assigned to agents
4. Agents execute and store results
5. Results combined by Synthesizer

## Auto-Update

This file is auto-generated. Manual changes may be overwritten.
To update: Edit `/docs/governance/templates/` and run the governance update script.

---

*Last Updated: 2026-03-02*
*Version: 1.0*
