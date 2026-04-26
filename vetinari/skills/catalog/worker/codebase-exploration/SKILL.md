---
name: Codebase Exploration
description: Progressive zoom navigation from directory to function level with call graph analysis using LSP
mode: code_discovery
agent: worker
version: "1.0.0"
capabilities:
  - code_discovery
  - dependency_analysis
tags:
  - research
  - exploration
  - lsp
  - call-graph
---

# Codebase Exploration

## Purpose

Codebase Exploration systematically navigates unfamiliar or partially known codebases using a progressive zoom pattern: directory structure first, then file contents, then function signatures, then line-level detail. It leverages LSP tools (goto-definition, find-references, document-symbols) for precise navigation rather than text-based grep, ensuring accuracy even in codebases with common identifier names. This skill is the foundation for all other Worker skills -- understanding existing patterns before making changes prevents inconsistency and duplication.

## When to Use

- Before any build task, to understand existing patterns and conventions
- When a task description references modules, functions, or concepts you have not yet inspected
- When investigating where a feature is implemented across multiple files
- When tracing data flow through the system (input to output)
- When looking for existing utilities to avoid duplicating functionality
- When the Foreman's scope analysis indicates affected files that need review
- Before refactoring, to understand all callers and usage patterns

## Inputs

| Parameter      | Type            | Required | Description                                                       |
|----------------|-----------------|----------|-------------------------------------------------------------------|
| task           | string          | Yes      | What to explore and why (e.g., "Find all rate limiting logic")    |
| entry_points   | list[string]    | No       | Starting files or symbols to explore from                         |
| files          | list[string]    | No       | Specific files to examine                                        |
| depth          | int (1-4)       | No       | Exploration depth: 1=directory, 2=file, 3=function, 4=line       |
| focus          | string          | No       | Specific aspect to focus on (e.g., "error handling", "imports")   |

## Process Steps

1. **Directory survey** -- Use Glob to map the top-level directory structure. Identify which directories are relevant to the task. Note naming conventions, package organization, and module boundaries.

2. **File identification** -- Within relevant directories, identify candidate files by name pattern matching and module docstring scanning. Read module-level docstrings to understand each file's purpose without reading full contents.

3. **Symbol enumeration** -- Use LSP document-symbols on candidate files to get a structured view of classes, functions, constants, and their visibility (public vs private). This provides a table of contents without reading implementation details.

4. **Signature analysis** -- For relevant symbols, read function signatures and docstrings. Note parameter types, return types, and documented behavior. Identify the public API surface versus internal implementation.

5. **Implementation deep-dive** -- For the most relevant functions, read the full implementation. Note patterns: error handling style, logging conventions, data structures used, import sources. These patterns must be matched by any new code.

6. **Call graph tracing** -- Use LSP find-references and goto-definition to trace call chains. Map who calls what, and what calls what. Identify entry points (nothing calls them) and leaf functions (they call nothing external).

7. **Pattern extraction** -- Synthesize discovered patterns into a summary: naming conventions, error handling approach, logging style, test patterns, configuration approach. This becomes the style guide for subsequent build tasks.

8. **Gap identification** -- Note missing elements: functions without tests, modules without docstrings, dead code (defined but never called), inconsistent patterns. These inform both the current task and future improvement opportunities.

## Output Format

The skill produces a structured exploration report:

```json
{
  "success": true,
  "output": {
    "summary": "Explored vetinari/web/ -- 6 route modules, Flask blueprint pattern, all routes use @login_required decorator",
    "structure": {
      "vetinari/web/": {
        "projects_api.py": "Project CRUD routes, 12 endpoints",
        "training_routes.py": "Training data management, 5 endpoints",
        "__init__.py": "Blueprint registration and CORS setup"
      }
    },
    "patterns": {
      "routing": "Flask blueprints with url_prefix, all routes return JSON",
      "auth": "@login_required decorator on all non-public routes",
      "errors": "Custom error handlers return {error: string, code: int}",
      "logging": "logger.info on entry, logger.exception on error"
    },
    "relevant_symbols": [
      {"name": "projects_bp", "type": "Blueprint", "file": "projects_api.py", "line": 15},
      {"name": "get_project", "type": "function", "file": "projects_api.py", "line": 42}
    ],
    "gaps": [
      "No rate limiting middleware found",
      "training_routes.py missing docstrings on 3/5 endpoints"
    ]
  },
  "provenance": [
    {"tool": "Glob", "query": "vetinari/web/**/*.py", "results": 6},
    {"tool": "LSP/document_symbols", "file": "projects_api.py", "symbols": 15}
  ]
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-001**: Research modes MUST cite sources -- file paths, URLs, or commit SHAs
- **STD-WRK-002**: Code discovery MUST use progressive zoom: directory, file, function, line
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **STD-UNI-003**: All skill executions MUST log entry and exit at INFO level with timing
- **CON-WRK-001**: Research modes are READ-ONLY -- MUST NOT modify production files
- **GDL-WRK-001**: Use code_discovery before build to understand existing patterns

## Examples

### Example: Exploring the agent system before adding a new mode

**Input:**
```
task: "Understand how agent modes are defined, registered, and dispatched to add a new 'synthesis' mode"
entry_points: ["vetinari/agents/consolidated/"]
depth: 3
```

**Output (abbreviated):**
```
summary: "Agents use a multi-mode pattern. Each consolidated agent defines modes as a list in its AgentSpec. Mode dispatch happens in base_agent.py via a mode-to-method mapping. Adding a new mode requires: (1) add to modes list in contracts.py, (2) add handler method in the agent class, (3) register in skill_registry.py."

patterns:
  mode_definition: "Modes listed in AgentSpec.modes and skill_registry SkillSpec.modes"
  mode_dispatch: "base_agent.execute() switches on mode parameter"
  mode_handler: "Each mode has a _handle_{mode}() private method"
  mode_registration: "skill_registry.py lists all valid modes per agent"

relevant_symbols:
  - AgentSpec.modes in contracts.py:45
  - BaseAgent.execute() in base_agent.py:78
  - WorkerAgent._handle_build() in worker_agent.py:112

gaps:
  - "No validation that registered modes match handler methods"
  - "Mode dispatch does not log which mode was selected"
```
