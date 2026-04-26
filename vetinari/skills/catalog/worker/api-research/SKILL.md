---
name: API Research
description: Research external APIs, verify version compatibility, check rate limits, and document integration patterns
mode: api_lookup
agent: worker
version: "1.0.0"
capabilities:
  - api_research
  - domain_research
tags:
  - research
  - api
  - integration
  - compatibility
---

# API Research

## Purpose

API Research investigates external APIs and libraries to determine their suitability for integration, verify version compatibility with the current project, and document usage patterns. It produces actionable integration guides rather than generic documentation, focusing on the specific subset of an API that the current task requires. This prevents the common failure mode of integrating an API based on outdated docs, deprecated endpoints, or incompatible versions.

## When to Use

- Before integrating a new third-party library or API
- When upgrading a dependency and need to verify backward compatibility
- When an existing integration is failing and the API may have changed
- When evaluating multiple API options and need a structured comparison
- When the task requires calling an external service with specific authentication or rate limits
- When documenting how the project uses an external API for future maintainers

## Inputs

| Parameter       | Type            | Required | Description                                                       |
|-----------------|-----------------|----------|-------------------------------------------------------------------|
| task            | string          | Yes      | What API to research and why                                      |
| api_name        | string          | No       | Name of the API or library to investigate                         |
| current_version | string          | No       | Currently installed version (from pyproject.toml)                 |
| target_version  | string          | No       | Version to evaluate upgrading to                                  |
| use_case        | string          | No       | Specific functionality needed from the API                        |
| context         | dict            | No       | Project context (Python version, existing dependencies)           |

## Process Steps

1. **Version inventory** -- Check the currently installed version in pyproject.toml, the latest available version, and any version constraints from other dependencies. Identify if the current version is deprecated, has known CVEs, or is end-of-life.

2. **API surface mapping** -- Identify the relevant subset of the API for the current use case. Map the key classes, functions, and endpoints. Note which are stable, beta, or deprecated.

3. **Authentication and authorization** -- Document the authentication mechanism (API keys, OAuth, tokens), required scopes or permissions, and how credentials should be stored (environment variables, config files, secrets manager).

4. **Rate limit analysis** -- Document rate limits, quotas, and throttling behavior. Calculate whether the project's expected usage fits within limits. Note retry-after headers or backoff requirements.

5. **Request/response schema** -- For each relevant endpoint or function, document the input parameters, response schema, error codes, and content types. Note any pagination, streaming, or webhook patterns.

6. **Version compatibility check** -- Compare the API's requirements against the project's Python version, OS platform, and existing dependencies. Check for conflicts with current dependency versions. Verify the API's own transitive dependencies don't conflict.

7. **Error handling patterns** -- Document what exceptions the API raises, what HTTP status codes it returns, and the recommended error handling approach. Map API-specific errors to Vetinari's exception hierarchy.

8. **Integration pattern recommendation** -- Based on the research, recommend the integration approach: direct import, wrapper class, adapter pattern, or client factory. Include code snippets that follow Vetinari's coding conventions.

9. **Testing strategy** -- Document how to mock or stub the API in tests. Identify whether the API provides a sandbox/test environment. Note any test fixtures or factories needed.

10. **Summary and risk assessment** -- Compile findings into a recommendation: integrate, defer, or reject. List risks (breaking changes, vendor lock-in, reliability concerns) and mitigations.

## Output Format

The skill produces a structured API research report:

```json
{
  "success": true,
  "output": {
    "api": "llama-cpp-python",
    "current_version": "0.2.56",
    "latest_version": "0.2.79",
    "recommendation": "upgrade",
    "compatibility": {
      "python": "3.10+ (matches project requirement)",
      "os": "Windows/Linux/macOS (all supported)",
      "conflicts": "none detected"
    },
    "relevant_features": [
      {"name": "Llama.__init__", "status": "stable", "breaking_changes": "none since 0.2.50"},
      {"name": "ChatCompletionRequestMessage", "status": "stable", "note": "schema expanded in 0.2.70"}
    ],
    "rate_limits": "N/A (local inference, no external rate limits)",
    "error_handling": {
      "exceptions": ["ValueError (invalid model path)", "RuntimeError (OOM)"],
      "recommendation": "Wrap in VetinariInferenceError with from-chaining"
    },
    "integration_pattern": "Wrapper class in vetinari/inference/llama_backend.py",
    "testing": "Mock Llama class; use small test model for integration tests",
    "risks": [
      "C++ build dependency complicates CI",
      "GPU memory management requires explicit cleanup"
    ]
  },
  "provenance": [
    {"source": "pyproject.toml", "field": "dependencies"},
    {"source": "https://github.com/abetlen/llama-cpp-python/releases", "accessed": "2025-01-15"}
  ]
}
```

## Quality Standards

This skill is governed by the following standards from the skill registry:

- **STD-WRK-001**: Research modes MUST cite sources -- file paths, URLs, or commit SHAs
- **STD-UNI-001**: All skill outputs MUST conform to the declared output_schema
- **STD-UNI-003**: All skill executions MUST log entry and exit at INFO level with timing
- **CON-WRK-001**: Research modes are READ-ONLY -- MUST NOT modify production files
- **GDL-WRK-001**: Use code_discovery before build to understand existing patterns

## Examples

### Example: Evaluating a new dependency

**Input:**
```
task: "Research the 'httpx' library as a replacement for 'requests' in the web scraping module"
api_name: "httpx"
current_version: null
use_case: "async HTTP client for parallel web fetching"
context: {python_version: "3.10+", existing_deps: ["requests==2.31.0", "aiohttp==3.9.1"]}
```

**Output (abbreviated):**
```
recommendation: "defer"
rationale: "httpx provides async support but aiohttp already covers async HTTP. Adding httpx would create a third HTTP library. Consolidate on aiohttp for async, keep requests for sync, rather than adding a third option."

compatibility: {python: "3.10+ OK", conflicts: "none"}
relevant_features:
  - AsyncClient: "async context manager, similar API to requests"
  - HTTP/2 support: "available but not needed for current use case"
risks:
  - "Adds a third HTTP library to the dependency tree"
  - "Team must learn another API surface"
  - "httpx and aiohttp have subtly different timeout semantics"
```
