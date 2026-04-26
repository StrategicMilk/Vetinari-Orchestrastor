---
name: Bug Diagnosis and Fix
description: Systematic bug resolution following Reproduce, Localize, Root-cause, Fix, Verify with regression test first
mode: build
agent: worker
version: "1.0.0"
capabilities:
  - bug_diagnosis
  - feature_implementation
  - test_writing
tags:
  - build
  - debugging
  - diagnosis
  - regression
---

# Bug Diagnosis and Fix

## Purpose

Bug Diagnosis and Fix provides a systematic methodology for resolving defects: Reproduce the bug, Localize the failure, Root-cause the defect, Fix the underlying issue, and Verify the fix with a regression test. The critical principle is to write the regression test BEFORE applying the fix, confirming that the test fails for the right reason, then fixing the code and observing the test pass. This prevents false fixes (changes that do not actually address the bug) and ensures the bug cannot silently return.

## When to Use

- When a test failure reveals unexpected behavior in production code
- When a user reports a bug with a reproducible scenario
- When the Inspector rejects code due to a correctness issue
- When a regression is detected after a recent change
- When error logs indicate unexpected behavior in a specific code path
- When a code path produces wrong results, not just errors

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| task            | string          | Yes      | Bug description with symptoms and reproduction steps               |
| files           | list[string]    | No       | Suspected files containing the bug                                 |
| error_message   | string          | No       | Error message or stack trace                                       |
| expected        | string          | No       | What should happen (correct behavior)                              |
| actual          | string          | No       | What actually happens (buggy behavior)                             |
| context         | dict            | No       | System state, recent changes, relevant logs                       |

## Process Steps

1. **Symptom cataloging** -- Document all observable symptoms: error messages, wrong outputs, unexpected state, timing anomalies. Distinguish between the symptom (what the user sees) and the defect (what the code does wrong).

2. **Reproduction** -- Create a minimal reproduction case that triggers the bug deterministically. This may be a test case, a script, or a sequence of API calls. If the bug is intermittent, identify the conditions that increase reproduction probability (timing, data size, concurrency).

3. **Regression test writing** -- Write a test that captures the bug BEFORE fixing it:
   - The test should fail with the current code (confirming it catches the bug)
   - The test should pass after the fix is applied
   - The test should describe the expected behavior, not the buggy behavior
   - Name the test descriptively: `test_parse_config_does_not_crash_on_empty_file`

4. **Localization** -- Narrow down the location of the defect:
   - Use the stack trace to identify the failing function
   - Add targeted assertions or logging to isolate the exact line
   - Use git blame to identify when the bug was introduced
   - Use git bisect if the bug is a regression and the introduction point is unknown

5. **Root cause analysis** -- Determine WHY the code is wrong, not just WHERE:
   - Off-by-one error? Logic inversion? Missing null check?
   - Is this a single-site bug or a pattern that exists in multiple places?
   - Was the original code ever correct, or was it always wrong?
   - What assumption did the original author make that turned out to be false?

6. **Fix implementation** -- Apply the minimal fix that corrects the root cause:
   - Fix the root cause, not the symptom
   - Do not refactor unrelated code in the same change (separate concerns)
   - If the bug exists in multiple places, fix all instances
   - Ensure the fix does not break any existing tests

7. **Fix verification** -- Run the regression test to confirm it now passes. Run the full test suite to confirm no other tests broke. If the bug was in a shared module, run ALL tests, not just related ones.

8. **Side effect check** -- Verify the fix does not introduce new issues:
   - Check callers of the fixed function for assumptions about old behavior
   - Verify error messages and log output are still correct
   - Check performance impact if the fix changes algorithmic complexity

9. **Documentation** -- Add a comment at the fix site explaining the bug and fix:
   ```python
   # Fix: handle empty input list (previously raised IndexError)
   # Regression test: test_process_items_with_empty_list
   ```

## Output Format

The skill produces a bug fix report:

```json
{
  "success": true,
  "output": "Fixed IndexError in process_items when called with empty list",
  "files_changed": [
    "vetinari/planning/decomposition.py (fixed empty list handling in process_items)",
    "tests/test_decomposition.py (added regression test)"
  ],
  "diagnosis": {
    "symptom": "IndexError when decomposing a goal with no sub-tasks",
    "root_cause": "process_items() accessed items[0] without checking if list is empty",
    "introduced_by": "commit abc1234 (2024-06-15)",
    "fix": "Added early return for empty list before accessing first element"
  },
  "tests_added": 1,
  "tests_passed": "all 245 tests pass",
  "metadata": {
    "regression_test": "test_process_items_with_empty_list",
    "related_fixes": "Same pattern found and fixed in process_tasks() on line 89"
  }
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-011**: Build mode is the SOLE writer of production files
- **STD-WRK-012**: All new code MUST have type annotations, Google-style docstrings, and tests
- **STD-WRK-016**: No TODO, FIXME, pass bodies, or placeholder strings
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **STD-UNI-002**: All skill executions MUST return a ToolResult; exceptions MUST be caught and reported
- **CON-WRK-004**: Build mode is the SOLE production file writer
- **GDL-WRK-005**: Write tests before implementation (TDD) for complex logic

## Examples

### Example: Fixing a race condition

**Input:**
```
task: "Fix intermittent KeyError in blackboard.read() when called concurrently from multiple agents"
error_message: "KeyError: 'plan_001' in blackboard.py line 45"
expected: "Concurrent reads should return consistent data or raise a descriptive error"
actual: "Random KeyError when two agents read the blackboard simultaneously"
```

**Output (abbreviated):**
```
diagnosis:
  symptom: "KeyError during concurrent blackboard reads"
  root_cause: "dict iteration in read() not thread-safe. Another thread's write() modifies the dict during iteration, removing a key between the existence check and the access."
  fix: "Add threading.Lock to protect read/write operations on the shared dict"

regression_test:
  name: "test_blackboard_concurrent_reads_do_not_raise"
  approach: "Use ThreadPoolExecutor to issue 100 concurrent reads while writes happen"
  before_fix: "FAILS with KeyError ~30% of the time"
  after_fix: "PASSES consistently"

files_changed:
  - vetinari/memory/blackboard.py (added threading.Lock, wrapped read/write in lock context)
  - tests/test_blackboard.py (added concurrent access regression test)
```
