---
name: Code Generation
description: Generate boilerplate from templates including new agent modes, API endpoints, test scaffolds, and dataclass definitions
mode: build
agent: worker
version: "1.0.0"
capabilities:
  - code_generation
  - feature_implementation
tags:
  - build
  - generation
  - templates
  - scaffolding
---

# Code Generation

## Purpose

Code Generation produces well-structured boilerplate code from established templates when the output follows a predictable pattern. Rather than writing each new agent mode, API endpoint, or test scaffold from scratch, this skill applies templates that encode project conventions, ensuring consistency across the codebase. It handles the mechanical parts of code creation (file structure, imports, decorators, type annotations) so the developer can focus on the unique logic. All generated code is fully functional, not stub or placeholder code.

## When to Use

- When adding a new agent mode that follows the established mode handler pattern
- When creating a new API endpoint that follows the Flask blueprint pattern
- When scaffolding test files for new modules
- When creating new dataclass or Pydantic model definitions
- When adding new entries to the skill registry
- When the implementation follows a well-established pattern with predictable structure
- When creating new modules that need standard boilerplate (imports, logger, docstring)

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| task            | string          | Yes      | What to generate and its purpose                                   |
| template_type   | string          | No       | Template: "agent_mode", "endpoint", "test", "dataclass", "module"  |
| parameters      | dict            | No       | Template variables (name, fields, methods, etc.)                   |
| files           | list[string]    | No       | Existing files to use as pattern reference                         |
| context         | dict            | No       | Project context, conventions to follow                             |

## Process Steps

1. **Template selection** -- Identify the appropriate template based on the generation target:
   - **Module**: standard Python module with docstring, future annotations, imports, logger
   - **Agent mode**: handler method with mode registration, input validation, output schema
   - **API endpoint**: Flask route with blueprint, decorators, request parsing, response format
   - **Test file**: pytest file with imports, fixtures, test classes, test function skeletons
   - **Dataclass**: dataclass or Pydantic model with fields, validators, type annotations

2. **Pattern extraction** -- If an existing file of the same type exists, read it to extract the exact patterns used. Copy the style, not just the structure: import order, comment style, spacing, naming conventions.

3. **Variable substitution** -- Fill in template variables with task-specific values:
   - Module name and docstring
   - Class and function names
   - Field names, types, and defaults
   - Route paths and HTTP methods
   - Test case names and assertions

4. **Import resolution** -- Determine all required imports for the generated code:
   - Standard library imports
   - Third-party imports (verify they exist in pyproject.toml)
   - Local imports (use canonical sources)
   - Arrange in the project's import order convention

5. **Type annotation generation** -- Add complete type annotations to all generated signatures:
   - Function parameters and return types
   - Dataclass field types
   - Variable annotations where non-obvious
   - Generic types with proper parameters

6. **Docstring generation** -- Write meaningful docstrings (not restating the name):
   - Module-level docstring explaining purpose
   - Class docstring explaining responsibility
   - Method/function docstrings with Args, Returns, Raises sections
   - Minimum 10 characters, maximum usefulness

7. **Wiring code generation** -- Generate the registration/integration code:
   - Add to `__init__.py` for new modules
   - Register routes with Flask app for new endpoints
   - Add to skill registry for new agent modes
   - Import in parent module for new classes

8. **Test scaffold generation** -- For every new production code file, generate a corresponding test file:
   - Mirror the source file structure
   - Create test class per production class
   - Create test function per production function
   - Include happy path, edge case, and error case skeletons
   - Add pytest fixtures for common setup

9. **Validation** -- Verify generated code:
   - Syntax check (compile without errors)
   - Import check (all imports resolve)
   - Type check (annotations are valid)
   - Convention check (matches project style)

## Output Format

The skill produces generated code with a manifest:

```json
{
  "success": true,
  "output": "Generated new API endpoint with test scaffold and registration",
  "files_changed": [
    "vetinari/web/analytics_api.py (new - 3 endpoint functions)",
    "vetinari/web/__init__.py (updated - registered analytics blueprint)",
    "tests/test_analytics_api.py (new - 9 test functions)"
  ],
  "metadata": {
    "template_used": "endpoint",
    "functions_generated": 3,
    "tests_generated": 9,
    "lines_generated": 185,
    "pattern_source": "vetinari/web/projects_api.py"
  }
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-011**: Build mode is the SOLE writer of production files
- **STD-WRK-012**: All new code MUST have type annotations, Google-style docstrings, and tests
- **STD-WRK-013**: Imports MUST use canonical sources
- **STD-WRK-014**: No print() in production code
- **STD-WRK-016**: No TODO, FIXME, pass bodies, NotImplementedError, or placeholder strings
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **CON-WRK-004**: Build mode is the SOLE production file writer

## Examples

### Example: Generating a new agent mode handler

**Input:**
```
task: "Generate a synthesis mode handler for the Worker agent"
template_type: "agent_mode"
parameters: {mode_name: "synthesis", agent: "worker", purpose: "Combine outputs from multiple tasks into a unified summary"}
```

**Output (abbreviated generated code):**
```python
async def _handle_synthesis(
    self,
    task: Task,
    context: ExecutionContext,
) -> AgentResult:
    """Combine outputs from multiple prior tasks into a unified summary.

    Synthesis mode reads the outputs of completed tasks listed in
    task.dependencies and produces a coherent summary that preserves
    key findings, decisions, and action items from each input.

    Args:
        task: The synthesis task with dependency references.
        context: Execution context with access to completed task outputs.

    Returns:
        AgentResult with the synthesized summary as output.
    """
    logger.info("Synthesis mode: combining %d task outputs", len(task.dependencies))

    # Gather outputs from completed dependency tasks
    dependency_outputs = []
    for dep_id in task.dependencies:
        dep_result = context.get_task_result(dep_id)
        if dep_result is None:
            logger.warning("Dependency %s has no result, skipping", dep_id)
            continue
        dependency_outputs.append(dep_result)

    # ... synthesis logic ...
```
