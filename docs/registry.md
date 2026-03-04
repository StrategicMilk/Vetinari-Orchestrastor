# Vetinari Skill Registry

## Overview

The Skill Registry is a centralized system for managing and discovering all Vetinari skills. It provides a unified API for agents to discover available skills, their capabilities, permissions, and sample usage.

## Architecture

### Components

1. **Central Registry** (`vetinari/skills_registry.json`)
   - Master index of all skills
   - Contains basic metadata, versions, and capabilities
   - References per-skill manifests

2. **Per-Skill Manifests** (`vetinari/skills/{skill_id}/manifest.json`)
   - Machine-readable skill specification
   - Input/output schemas
   - Sample usage examples
   - Context references

3. **Agent Mappings** (`vetinari/config/agent_skill_map.json`)
   - Maps agents to their default skills
   - Defines predefined workflows
   - Environment-specific overrides

4. **Context Catalog** (`vetinari/context_registry.json`)
   - Sample data for demonstrations
   - Test fixtures
   - Reference materials

## Usage

### Python API

```python
from vetinari import registry

# List all skills
skills = registry.list_skills()
print(f"Available skills: {[s['id'] for s in skills]}")

# Get specific skill
skill = registry.get_skill("builder")
print(f"Builder capabilities: {skill['capabilities']}")

# Get skill manifest
manifest = registry.get_skill_manifest("evaluator")
print(f"Required permissions: {manifest['required_permissions']}")

# Get skills for an agent
agent_skills = registry.get_agent_skills("builder_agent")
print(f"Builder agent skills: {agent_skills}")

# Get sample context
context = registry.get_context("sample_code_snippet")
print(f"Context data: {context['data']}")

# Search skills
results = registry.search_skills("review")
print(f"Matching skills: {[s['id'] for s in results]}")

# Validate registry
validation = registry.validate_registry()
print(f"Errors: {validation['errors']}")
print(f"Warnings: {validation['warnings']}")
```

### CLI Usage

```bash
# List all skills
python -m vetinari.registry list-skills

# Get skill details
python -m vetinari.registry get-skill builder

# Validate registry
python -m vetinari.registry validate
```

## Adding a New Skill

1. Create manifest file: `vetinari/skills/{skill_id}/manifest.json`
2. Update central registry: Add entry to `vetinari/skills_registry.json`
3. Add sample contexts (optional): Update `vetinari/context_registry.json`
4. Map to agents (optional): Update `vetinari/config/agent_skill_map.json`
5. Run validation: `python -m vetinari.registry validate`

## Manifest Schema

```json
{
  "skill_id": "string",
  "name": "string",
  "version": "semver",
  "description": "string",
  "capabilities": ["string"],
  "thinking_modes": ["low", "medium", "high", "xhigh"],
  "triggers": ["string"],
  "required_permissions": ["FILE_READ", ...],
  "allowed_modes": ["EXECUTION", "PLANNING"],
  "sample_usage": {...},
  "inputs": {...},
  "outputs": {...},
  "contexts": ["context_id"],
  "external_endpoints": {
    "allowed": boolean,
    "endpoints": ["url"]
  }
}
```

## Workflows

Predefined skill workflows are available in `agent_skill_map.json`:

- `code_review_pipeline`: Explorer → Evaluator → Synthesizer
- `feature_implementation_pipeline`: Explorer → Librarian → Builder → Evaluator → Synthesizer
- `research_pipeline`: Researcher → Librarian → Oracle → Synthesizer

## Version Compatibility

The registry maintains a compatibility matrix between Vetinari core and skill versions:

```json
{
  "version_matrix": {
    "vetinari_core": {
      "min": "0.1.0",
      "recommended": "0.4.0",
      "compatibility": {
        "builder": ">=1.0.0",
        ...
      }
    }
  }
}
```

## Security

- All skills enforce permissions via `ToolMetadata.required_permissions`
- External network access is explicitly whitelisted per skill
- Registry validation runs in CI to catch misconfigurations
