# Skill-to-Tool Migration Guide

## Overview

This document outlines the pattern for migrating Vetinari skills to the standardized Tool interface. The builder skill serves as the reference implementation for migrating the remaining 7 skills.

## Migration Architecture

### Before: Skill Model
```
skills/
├── builder/
│   ├── SKILL.md          (markdown spec)
│   └── references/       (supporting docs)
└── [other skills]
```

### After: Tool Model
```
vetinari/
├── tools/
│   ├── __init__.py       (exports all tools)
│   ├── builder_skill.py  (Tool wrapper)
│   └── [other_skill_tool.py]
```

## Step-by-Step Migration Process

### Step 1: Understand the Skill

**Location**: `skills/{skill_name}/SKILL.md`

Examine the skill specification to understand:
- Purpose and description
- Capabilities (the main operations it performs)
- Triggers (when to activate the skill)
- Thinking modes (if applicable)
- Input/output format
- Workflow and use cases

**Example (Builder)**:
```yaml
name: builder
description: Code implementation, refactoring, and testing
capabilities:
  - feature_implementation
  - refactoring
  - test_writing
  - error_handling
  - code_generation
  - debugging
thinking_modes:
  low: Quick implementation
  medium: Full feature with tests
  high: Complete with error handling
  xhigh: Production-ready with full coverage
```

### Step 2: Define Tool Structure

Create a new file: `vetinari/tools/{skill_name}_skill.py`

Structure:
```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from vetinari.tool_interface import (
    Tool,
    ToolMetadata,
    ToolResult,
    ToolParameter,
    ToolCategory,
)
from vetinari.execution_context import ToolPermission, ExecutionMode

# 1. Define capability enums
class {Skill}Capability(str, Enum):
    """Capabilities of the {skill} skill."""
    CAPABILITY_ONE = "capability_one"
    CAPABILITY_TWO = "capability_two"

# 2. Define thinking modes (if applicable)
class ThinkingMode(str, Enum):
    """Thinking modes for implementation approach."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"

# 3. Define request/result dataclasses
@dataclass
class {Skill}Request:
    """Request structure for {skill} operations."""
    capability: {Skill}Capability
    description: str
    context: Optional[str] = None
    # ... other fields

# 4. Define Tool class
class {Skill}SkillTool(Tool):
    """Tool wrapper for the {skill} skill."""
    
    def __init__(self):
        metadata = ToolMetadata(
            name="{skill}",
            description="...",
            category=ToolCategory.{CATEGORY},
            parameters=[...],
            required_permissions=[...],
            allowed_modes=[...],
        )
        super().__init__(metadata)
    
    def execute(self, **kwargs) -> ToolResult:
        # Validate and execute
        pass
```

### Step 3: Map Capabilities to Tool

**Key Concept**: Each skill capability becomes a separate execution method.

**Pattern**:
```python
def execute(self, **kwargs) -> ToolResult:
    """Main execution method."""
    try:
        # 1. Extract and validate parameters
        capability = kwargs.get("capability")
        
        # 2. Route to capability handler
        result = self._execute_capability(capability, **kwargs)
        
        # 3. Return structured ToolResult
        return ToolResult(
            success=result.success,
            output=result.to_dict(),
            error=None if result.success else "Failed",
        )
    except Exception as e:
        # 4. Handle errors gracefully
        return ToolResult(success=False, output=None, error=str(e))

def _execute_capability(self, capability, **kwargs):
    """Route to specific capability handler."""
    if capability == Capability.ONE:
        return self._handle_capability_one(kwargs)
    elif capability == Capability.TWO:
        return self._handle_capability_two(kwargs)
```

### Step 4: Define Permissions

Map skill requirements to ToolPermissions.

**Common patterns**:
- Read/analyze code → `FILE_READ`, `MODEL_INFERENCE`
- Generate/modify code → `FILE_READ`, `FILE_WRITE`, `MODEL_INFERENCE`
- Execute code → `BASH_EXECUTE`, `PYTHON_EXECUTE`
- Write to storage → `DATABASE_WRITE`, `MEMORY_WRITE`

**Example**:
```python
required_permissions=[
    ToolPermission.FILE_READ,      # Read existing code
    ToolPermission.FILE_WRITE,     # Create/modify files
    ToolPermission.MODEL_INFERENCE, # Use LLM for generation
]
```

### Step 5: Define Allowed Execution Modes

Determine which execution modes support the tool.

**Pattern**:
- `ExecutionMode.EXECUTION`: Full capabilities, all permissions available
- `ExecutionMode.PLANNING`: Analysis-only, limited permissions
- `ExecutionMode.SANDBOX`: Restricted for untrusted code

**Example**:
```python
allowed_modes=[
    ExecutionMode.EXECUTION,  # Full implementation
    ExecutionMode.PLANNING,   # Analysis only
]
```

### Step 6: Create Unit Tests

**Location**: `tests/test_{skill}_skill.py`

**Test Classes to Implement**:

1. **Metadata Tests**
   ```python
   class Test{Skill}ToolMetadata:
       def test_initialization(self): ...
       def test_allowed_execution_modes(self): ...
       def test_required_permissions(self): ...
       def test_parameters_defined(self): ...
   ```

2. **Execution Tests**
   ```python
   class Test{Skill}ToolExecution:
       def test_capability_execution_success(self): ...
       def test_invalid_capability(self): ...
       def test_invalid_parameters(self): ...
       def test_permission_enforcement(self): ...
       def test_execution_mode_handling(self): ...
   ```

3. **Parameter Validation Tests**
   ```python
   class Test{Skill}ToolParameterValidation:
       def test_required_parameters(self): ...
       def test_invalid_values(self): ...
       def test_optional_parameters(self): ...
   ```

4. **Edge Cases Tests**
   ```python
   class Test{Skill}ToolEdgeCases:
       def test_empty_input(self): ...
       def test_large_input(self): ...
       def test_unicode_characters(self): ...
       def test_special_characters(self): ...
   ```

**Minimum Test Coverage**: 30+ tests per skill

### Step 7: Create Example Scripts

**Location**: `examples/{skill}_skill_example.py`

**Structure**:
```python
#!/usr/bin/env python
"""Example: Using the {Skill} Skill Tool"""

from vetinari.tools.{skill}_skill import {Skill}SkillTool
from vetinari.execution_context import ExecutionMode, get_context_manager

def example_capability_one():
    """Demonstrate capability 1."""
    tool = {Skill}SkillTool()
    result = tool.run(
        capability="capability_one",
        description="Do something",
        # ... other params
    )
    print(f"Success: {result.success}")
    if result.output:
        print(f"Output: {result.output}")

def example_execution_modes():
    """Show different execution modes."""
    context_manager = get_context_manager()
    
    # EXECUTION mode
    context_manager.switch_mode(ExecutionMode.EXECUTION)
    # ... run tool
    
    # PLANNING mode
    context_manager.switch_mode(ExecutionMode.PLANNING)
    # ... run tool (limited permissions)

def main():
    """Run all examples."""
    example_capability_one()
    example_execution_modes()
    # ... more examples

if __name__ == "__main__":
    main()
```

**Examples to Include**:
- Each major capability
- Different parameter combinations
- Execution mode differences
- Error handling
- Metadata inspection

### Step 8: Update Package Exports

**File**: `vetinari/tools/__init__.py`

```python
from vetinari.tools.builder_skill import BuilderSkillTool
from vetinari.tools.new_skill_tool import NewSkillTool

__all__ = [
    "BuilderSkillTool",
    "NewSkillTool",
]
```

### Step 9: Documentation

Create/update documentation in `docs/` for:

1. **Skill Overview** - What the skill does, capabilities, use cases
2. **API Reference** - Parameters, return types, examples
3. **Execution Modes** - How the skill behaves in different modes
4. **Error Handling** - Common errors and solutions
5. **Examples** - Real-world usage scenarios

## Reference Implementation: Builder Skill

### Files Created

```
vetinari/
└── tools/
    ├── __init__.py                    (44 lines)
    └── builder_skill.py               (537 lines)

examples/
└── builder_skill_example.py           (317 lines)

tests/
└── test_builder_skill.py              (711 lines)
```

### Key Statistics

- **Total Lines of Code**: ~1,600
- **Unit Tests**: 42 (100% passing)
- **Test Coverage**: Metadata, execution, validation, edge cases
- **Capabilities**: 6 (feature_implementation, refactoring, test_writing, error_handling, code_generation, debugging)
- **Thinking Modes**: 4 (low, medium, high, xhigh)

### Important Implementation Details

1. **Capability Routing**
   - Use enums for capabilities
   - Route via `_execute_capability()` method
   - Each capability has its own handler

2. **Execution Mode Handling**
   - Access via `self._context_manager.current_context.mode`
   - Use ExecutionMode.PLANNING for analysis-only operations
   - Use ExecutionMode.EXECUTION for full capabilities
   - Include mode information in result metadata

3. **Parameter Validation**
   - Inherit from Tool base class
   - `validate_inputs()` method checks parameters
   - Return (bool, error_msg) tuple
   - Test both required and optional parameters

4. **Permission Checking**
   - Define in ToolMetadata.required_permissions
   - Base Tool class handles enforcement via `run()` method
   - Test permission failures in unit tests
   - Gracefully fail when permissions not available

5. **Error Handling**
   - Try-catch in execute() method
   - Return ToolResult with success=False and error message
   - Log detailed errors for debugging
   - Include metadata about what failed

## Remaining Skills to Migrate

1. **evaluator** - Code review and evaluation
2. **explorer** - Code exploration and analysis
3. **librarian** - Package and dependency management
4. **oracle** - Architecture and design consultation
5. **researcher** - Research and information gathering
6. **synthesizer** - Combining solutions and insights
7. **ui-planner** - UI/UX planning and design

### Migration Priority Recommendation

1. **Tier 1** (Similar to builder):
   - explorer (code analysis)
   - evaluator (code review)

2. **Tier 2** (Different patterns):
   - librarian (dependency management)
   - oracle (guidance/suggestions)

3. **Tier 3** (Complex):
   - researcher (information gathering)
   - synthesizer (solution combining)
   - ui-planner (design planning)

## Testing Strategy

### Unit Test Pattern
```python
def setup_method(self):
    """Set up test fixtures."""
    self.tool = {Skill}SkillTool()
    self.mock_ctx_manager = Mock()
    self.mock_context = Mock()
    self.tool._context_manager = self.mock_ctx_manager

def test_capability_execution(self):
    """Test a specific capability."""
    self.mock_context.mode = ExecutionMode.EXECUTION
    
    result = self.tool.execute(
        capability="capability_name",
        # ... other params
    )
    
    assert result.success is True
    assert "expected_text" in result.output["explanation"]
```

### Running Tests
```bash
# Run all tests for a skill
python -m pytest tests/test_{skill}_skill.py -v

# Run specific test class
python -m pytest tests/test_{skill}_skill.py::TestClassName -v

# Run with coverage
python -m pytest tests/test_{skill}_skill.py --cov=vetinari.tools.{skill}_skill
```

## Quality Checklist

Before marking a skill migration as complete:

- [ ] Tool class inherits from Tool base class
- [ ] ToolMetadata properly defined with all fields
- [ ] All capabilities mapped to execution handlers
- [ ] Parameters validated with type checking
- [ ] Permissions defined and enforced
- [ ] Execution modes supported and tested
- [ ] Error handling comprehensive
- [ ] 30+ unit tests written
- [ ] 100% test pass rate achieved
- [ ] Example script demonstrating usage
- [ ] Documentation updated
- [ ] Tool exported in `__init__.py`
- [ ] Code follows project style and conventions

## Key Design Patterns

### 1. Enumeration Pattern
Use enums for predefined values:
```python
class Capability(str, Enum):
    ONE = "one"
    TWO = "two"
```

### 2. Request/Result Pattern
Encapsulate inputs and outputs:
```python
@dataclass
class Request:
    capability: Capability
    param1: str
    param2: Optional[str] = None

@dataclass
class Result:
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
```

### 3. Capability Routing Pattern
Switch on capability:
```python
def _execute_capability(self, capability):
    if capability == Capability.ONE:
        return self._handle_one()
    # ...
```

### 4. Context-Aware Execution
Use execution mode to determine behavior:
```python
if execution_mode == ExecutionMode.PLANNING:
    # Analysis only
    return ToolResult(success=True, output=planning_result)
else:
    # Full execution
    return ToolResult(success=True, output=full_result)
```

## Common Pitfalls to Avoid

1. **Forgetting ExecutionContext.mode attribute**
   - Use `ctx.mode` not `ctx.execution_mode`
   - Get context via `self._context_manager.current_context`

2. **Hardcoding Permissions**
   - Define in ToolMetadata.required_permissions
   - Let base Tool class enforce
   - Don't manually check permissions in execute()

3. **Missing Error Handling**
   - Always wrap execute() in try-catch
   - Return ToolResult with error message
   - Log exceptions for debugging

4. **Incomplete Test Coverage**
   - Test each capability
   - Test parameter validation
   - Test execution mode differences
   - Test error cases
   - Test edge cases

5. **Not Exporting Tool**
   - Add to vetinari/tools/__init__.py
   - Include in __all__ list
   - Required for external imports

## Summary

Skill-to-Tool migration is a systematic process:

1. Understand the skill → Define capabilities/parameters
2. Create Tool class → Implement execute() and handlers
3. Define permissions → Map to required ToolPermissions
4. Write tests → 30+ test methods covering all paths
5. Create examples → Demonstrate each capability
6. Update exports → Make tool available for import
7. Document → Explain usage and behavior

Total migration time per skill: 4-6 hours (depending on complexity)

The builder skill migration serves as the reference implementation showing this entire process end-to-end.
