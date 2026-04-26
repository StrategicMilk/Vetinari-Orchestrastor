---
name: Error Handling Hardening
description: Replace bare excepts with specific exception types, add proper chaining, and ensure no silent failures
mode: build
agent: worker
version: "1.0.0"
capabilities:
  - error_handling_hardening
  - feature_implementation
tags:
  - build
  - error-handling
  - robustness
  - exceptions
---

# Error Handling Hardening

## Purpose

Error Handling Hardening systematically improves exception handling throughout the codebase by replacing bare except clauses with specific exception types, adding proper exception chaining (`from`), ensuring no exceptions are silently swallowed, and making error messages actionable. Poor error handling is the leading cause of "silent failures" where the system appears to work but produces wrong results or loses data. This skill transforms error handling from a liability into a diagnostic asset.

## When to Use

- When the Inspector identifies bare except clauses or swallowed exceptions
- When error messages are unhelpful ("an error occurred") and need improvement
- When exceptions are caught but not logged, re-raised, or reported
- When the codebase lacks consistent exception handling patterns
- When adding error handling to new code paths
- When investigating bugs caused by silently handled exceptions
- After a production incident traced to poor error handling

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| task            | string          | Yes      | What error handling to improve and why                             |
| files           | list[string]    | No       | Files to harden                                                    |
| patterns        | list[string]    | No       | Specific anti-patterns to fix (e.g., "bare_except", "silent_catch")|
| context         | dict            | No       | Exception hierarchy, custom exceptions in the project              |

## Process Steps

1. **Anti-pattern scan** -- Scan target files for error handling anti-patterns:
   - `except:` (bare except -- catches SystemExit, KeyboardInterrupt)
   - `except Exception:` with no body or only `pass`
   - `except Exception as e:` where `e` is never used
   - Missing `from` chaining: `raise NewError("msg")` instead of `raise NewError("msg") from exc`
   - Catch-and-return-None patterns that hide failures
   - Log-and-swallow: `logger.error(...)` followed by `pass` or `return None`

2. **Exception taxonomy** -- Catalog which exceptions can actually occur at each catch site:
   - File I/O: `FileNotFoundError`, `PermissionError`, `IsADirectoryError`
   - JSON/YAML parsing: `json.JSONDecodeError`, `yaml.YAMLError`
   - Network: `ConnectionError`, `TimeoutError`, `HTTPError`
   - Data validation: `ValueError`, `TypeError`, `KeyError`
   - Project-specific: exceptions from `vetinari/exceptions.py`

3. **Specificity improvement** -- Replace each broad `except` with the specific exceptions that can actually occur. If uncertain, check the called function's documentation and source code for raised exceptions.

4. **Chain enforcement** -- Add `from exc` to every `raise` inside an except block:
   ```python
   # BEFORE:
   except ValueError:
       raise ConfigError("Invalid config")
   # AFTER:
   except ValueError as exc:
       raise ConfigError("Invalid config") from exc
   ```

5. **Error message enrichment** -- Make every error message actionable with three components:
   - What happened: "Failed to parse configuration file"
   - What was expected: "Expected valid YAML with 'model' key"
   - What to do: "Check that config/settings.yaml exists and contains valid YAML"

6. **Recovery strategy implementation** -- For each catch site, determine the appropriate response:
   - **Re-raise**: the caller should handle this (propagate with context)
   - **Fallback**: provide a safe default and continue (log at WARNING)
   - **Retry**: transient error that may succeed on retry (with backoff)
   - **Transform**: convert to domain-specific exception (with chaining)

7. **Logging integration** -- Ensure every catch site has appropriate logging:
   - `logger.exception("msg")` for unexpected errors (includes traceback)
   - `logger.warning("msg")` for handled errors with fallback
   - `logger.debug("msg")` for expected errors (e.g., cache miss)

8. **Test update** -- For each hardened error handler, verify or add test coverage:
   - Test that specific exceptions are raised (not generic ones)
   - Test that error messages contain actionable information
   - Test that exception chains preserve the original cause
   - Test that fallback values are correct

9. **Consistency verification** -- After hardening, verify the entire file follows a consistent error handling pattern. No mixing of styles within a single module.

## Output Format

The skill produces a hardening report:

```json
{
  "success": true,
  "output": "Hardened error handling in 4 files, fixed 12 anti-patterns",
  "files_changed": [
    "vetinari/prompts/assembler.py (3 bare excepts -> specific exceptions)",
    "vetinari/memory/blackboard.py (2 silent catches -> log + re-raise)",
    "vetinari/orchestration/replan_engine.py (4 missing from-chains added)",
    "vetinari/web/projects_api.py (3 generic error messages enriched)"
  ],
  "warnings": [],
  "metadata": {
    "anti_patterns_fixed": {
      "bare_except": 3,
      "silent_catch": 2,
      "missing_chain": 4,
      "vague_message": 3
    },
    "tests_updated": 5,
    "tests_added": 3
  }
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-011**: Build mode is the SOLE writer of production files
- **STD-WRK-012**: All new code MUST have type annotations, Google-style docstrings, and tests
- **STD-WRK-014**: No print() in production code -- use logging module with %-style formatting
- **STD-WRK-016**: No TODO, FIXME, pass bodies, or placeholder strings
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **STD-UNI-002**: All skill executions MUST return a ToolResult; exceptions MUST be caught and reported
- **CON-WRK-004**: Build mode is the SOLE production file writer

## Examples

### Example: Hardening a data loading function

**Input:**
```
task: "Fix error handling in load_config() -- currently catches Exception and returns None"
files: ["vetinari/config/loader.py"]
patterns: ["silent_catch", "bare_except"]
```

**Before:**
```python
def load_config(path):
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None
```

**After:**
```python
def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration from the given path.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ConfigError: If the file contains invalid YAML.
    """
    try:
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise  # Let caller handle missing file
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: check syntax and indentation") from exc

    if config is None:
        raise ConfigError(f"Configuration file {path} is empty")

    return config
```
