---
name: Test Writing
description: Write comprehensive tests covering happy path, edge cases, and error paths using pytest with fixtures and parametrize
mode: build
agent: worker
version: "1.0.0"
capabilities:
  - test_writing
  - feature_implementation
tags:
  - build
  - testing
  - pytest
  - quality
---

# Test Writing

## Purpose

Test Writing creates comprehensive, well-structured test suites that verify code correctness across happy paths, edge cases, and error conditions. It uses pytest idioms (fixtures, parametrize, markers) for clean, maintainable tests and ensures proper isolation between test cases. Tests serve as both verification and documentation -- a well-written test suite tells a reader exactly what the code does, what it handles, and what it rejects.

## When to Use

- After implementing a new feature (every public function needs at least one test)
- Before fixing a bug (write a regression test that reproduces the bug first)
- When test coverage analysis identifies untested code paths
- When refactoring requires confidence that behavior is preserved
- When acceptance criteria need automated verification
- When the Inspector identifies missing test coverage

## Inputs

| Parameter       | Type            | Required | Description                                                        |
|-----------------|-----------------|----------|--------------------------------------------------------------------|
| task            | string          | Yes      | What to test and the testing objective                             |
| files           | list[string]    | No       | Source files to test                                               |
| functions       | list[string]    | No       | Specific functions or methods to test                              |
| acceptance      | list[string]    | No       | Acceptance criteria to convert to test cases                       |
| context         | dict            | No       | Existing test patterns, fixtures, mocking strategy                 |
| thinking_mode   | string          | No       | Thinking budget tier                                               |

## Process Steps

1. **Test subject analysis** -- Read the source code to understand: function signatures, expected behavior (from docstrings), edge cases (from type annotations and validation), error conditions (from exception handling), and dependencies (what needs mocking).

2. **Test case enumeration** -- For each function, enumerate test cases in this order:
   - **Happy path**: typical input produces expected output
   - **Boundary cases**: empty inputs, single-element inputs, maximum inputs
   - **Type edge cases**: None, zero, empty string, empty list, negative numbers
   - **Error cases**: invalid input, missing required fields, external failures
   - **Integration cases**: interaction between components (if applicable)

3. **Fixture design** -- Identify shared setup that multiple tests need. Create pytest fixtures for:
   - Test data (sample objects, configuration dicts)
   - Mock objects (external services, file system, database)
   - Temporary resources (temp directories, test databases)
   - Use `conftest.py` for fixtures shared across test files

4. **Test structure** -- Organize tests following project conventions:
   - File naming: `tests/test_{module_name}.py`
   - Group related tests in classes: `class TestMyFunction:`
   - Test naming: `test_{function}_{scenario}` (e.g., `test_parse_config_with_empty_file`)
   - Arrange-Act-Assert pattern within each test

5. **Parametrize identification** -- When multiple test cases have the same structure but different data, use `@pytest.mark.parametrize` to avoid duplication. This is especially useful for boundary value testing and input validation.

6. **Mock strategy** -- Mock external dependencies to ensure test isolation:
   - Use `unittest.mock.patch` for module-level mocks
   - Use `pytest.monkeypatch` for environment variables and simple attributes
   - Mock at the boundary (where your code meets external code)
   - Never mock the code under test -- only its dependencies

7. **Assertion design** -- Write assertions that are specific and informative:
   - Assert exact values for deterministic outputs
   - Assert type and structure for dynamic outputs
   - Assert exception type and message for error cases
   - Use `pytest.raises` for expected exceptions
   - Use `pytest.approx` for floating point comparisons

8. **Independence verification** -- Ensure each test is independent:
   - No shared mutable state between tests
   - No ordering dependencies (test B relies on test A running first)
   - Each test can run in isolation: `pytest tests/test_foo.py::test_specific_case`
   - Fixtures use function scope unless explicitly justified

9. **Coverage assessment** -- After writing tests, check coverage of the target code. Identify any untested branches, uncovered exception handlers, or missing edge cases. Add tests until all significant paths are covered.

10. **Test execution** -- Run all new tests and verify they pass. Run with `-v` to see individual test results. Run existing tests to verify no regressions.

## Output Format

The skill produces test code with a test summary:

```json
{
  "success": true,
  "output": "Created 12 tests for vetinari/web/middleware.py covering happy, edge, and error paths",
  "files_changed": [
    "tests/test_middleware.py (new - 12 test functions in 2 test classes)"
  ],
  "tests_added": 12,
  "tests_passed": 12,
  "coverage": {
    "vetinari/web/middleware.py": "95% (missing: line 87 -- unreachable error fallback)"
  },
  "metadata": {
    "happy_path_tests": 4,
    "edge_case_tests": 5,
    "error_case_tests": 3,
    "parametrized": 2,
    "fixtures_created": 3
  }
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-011**: Build mode is the SOLE writer of production files
- **STD-WRK-012**: All new code MUST have type annotations, Google-style docstrings, and tests
- **STD-WRK-016**: No TODO, FIXME, pass bodies, or placeholder strings in test code
- **STD-INS-004**: Test generation MUST cover happy path, edge cases, and error paths
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **CON-WRK-004**: Build mode is the SOLE production file writer
- **GDL-WRK-005**: Write tests before implementation (TDD) for complex logic

## Examples

### Example: Testing a rate limiter

**Input:**
```
task: "Write tests for the token bucket rate limiter in vetinari/web/middleware.py"
functions: ["TokenBucket.__init__", "TokenBucket.consume", "rate_limit_middleware"]
```

**Output (abbreviated test code):**
```python
class TestTokenBucket:
    def test_consume_within_limit_returns_true(self, bucket):
        """Happy path: consuming tokens within capacity succeeds."""
        assert bucket.consume() is True

    def test_consume_exceeding_limit_returns_false(self, bucket):
        """Error path: consuming beyond capacity is rejected."""
        for _ in range(10):  # exhaust capacity
            bucket.consume()
        assert bucket.consume() is False

    @pytest.mark.parametrize("capacity,rate", [(0, 1), (1, 0)])
    def test_init_with_zero_values_raises(self, capacity, rate):
        """Edge case: zero capacity or rate is invalid."""
        with pytest.raises(ValueError, match="must be positive"):
            TokenBucket(capacity=capacity, refill_rate=rate)

    def test_tokens_refill_after_time(self, bucket, monkeypatch):
        """Happy path: tokens regenerate over time."""
        for _ in range(10):
            bucket.consume()
        monkeypatch.setattr("time.monotonic", lambda: time.monotonic() + 2.0)
        assert bucket.consume() is True
```
