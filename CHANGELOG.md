# Changelog

All notable changes to AM Workbench and the Vetinari engine are documented in
this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Ledger Citation Convention

Starting at **v0.7.0** (the version after the current latest release, v0.6.0),
every substantive claim line under a version heading **must** carry at least one
inline `[ledger:<id>]` tag referencing a record in
`outputs/release/<version>/ledger.jsonl`.

Example annotated entry:

```
- Inspector fails closed on missing citations. [ledger:inspector-fail-closed-01]
```

Entries in versions **prior to v0.7.0** (i.e., v0.6.0 and earlier) were written
before this convention existed and are **not** retroactively annotated.
`scripts/release/pre_release_gate.py` enforces the convention on the latest version
heading only; historical sections are not checked.

The public convention is intentionally simple: new release claims should point
to reproducible public evidence such as tests, build logs, or release-gate
output. Private maintainer evidence is not required to read this changelog.

---

## [Unreleased]

- No unreleased changes yet.

## [0.6.0] - 2026-04-22

### Added

- Final release signoff evidence was produced privately; the public release now
  carries only the source, package boundaries, and user-facing documentation
  needed for external review.
- Blocking CI release-proof coverage for package build/install smoke, route-auth proof, audit-prevention checks, and release-certifier wiring.
- Explicit package-boundary proof for shipped runtime assets, including `LICENSE`, `NOTICE`, bounded `vetinari/config/**` data, and clean install smoke from built artifacts.

### Changed

- Canonical release metadata now resolves to `v0.6.0` from the package version source and aligned release-bearing docs.
- The release narrative now reflects the verified current system: a three-agent factory pipeline, bounded learning/autonomy claims, no shipped browser UI surface, and blocking release-proof gates.
- Test-governance evidence now comes from the repaired full baseline, the strengthened test-quality scanner, and documented `noqa` rationales instead of historical blocker snapshots.

### Fixed

- The full pytest baseline is green again, including prior route/auth/protocol, runtime, dashboard, training, and shutdown regressions.
- `scripts/quality/check_test_quality.py` now correctly maps package stems and reports a clean suite instead of false blocker counts.
- Packaging/install proof now succeeds from built wheel and sdist artifacts, with the installed `vetinari` wrapper exporting a real `main()` entry point.
- Release artifacts are bounded to intended package inputs rather than leaking maintainer roots, audit trees, frontend dependency trees, or model payloads.
- Visible mojibake and non-ASCII drift in release-facing CLI/runtime files were cleaned so installed help and metadata read cleanly.

### Removed

- The broken `vetinari-asgi` console-script release surface is no longer published.
- Internal maintainer-only roots and accidental packaging residues are excluded from shipped artifacts.

### Security

- Release proof now treats route-auth, degraded-health, and prevention checks as blocking evidence instead of advisory narrative.
- `noqa` suppressions are now reviewed under the dedicated suppression policy so stale or convenience-only escapes are surfaced explicitly.

## [0.5.0] - 2026-03-11

### Added

- Analytics REST endpoints for cost, SLA, anomaly, forecast, model, agent, and summary reporting.
- Tiered cascade routing, batch processing, file-based agent governance, and enriched agent registry metadata.

### Changed

- Stale legacy agent references were migrated to the consolidated agent set across code, docs, tests, and benchmarks.
- The project converged on the post-consolidation governance model that later fed the three-agent factory pipeline.

### Security

- Constant-time token verification, trusted-proxy handling, rate limiting, and stricter input validation were enforced across sensitive web routes.
- Mutating routes for sandbox, planning, ADR, decomposition, ponder, rules, and training were hardened behind auth checks.

## [0.4.0] - 2026-02

### Added

- Consolidated six-agent architecture with typed output schemas, circuit breakers, token budgets, dynamic model routing, and SQLite-backed cost tracking.
- `TwoLayerOrchestrator` as the single execution engine replacing the prior assembly-line orchestrator.

### Changed

- Agent enums and dispatch tables were migrated to the consolidated agent family.

## [0.3.0] - 2026-01

### Added

- A 22-agent multi-stage system with DAG scheduling, blackboard memory, feedback loops, prompt evolution, and a Flask dashboard.
- Structured logging, OpenTelemetry tracing, and multi-source search support.

### Changed

- The web server replaced the CLI-only interface as the primary runtime surface.

## [0.2.0] - 2025-12

### Added

- Planning engine, dual memory tiers, shared blackboard, constraints, safety package, checkpoint recovery, cost tracking, and tracing foundations.

### Changed

- Provider and runtime configuration moved out of hardcoded values and into versioned config files.

## [0.1.0] - 2025-11

### Added

- Initial LM Studio adapter, execution-context system, tool registry, provider abstraction layer, verifier pipeline, and enhanced CLI.
- Core package scaffolding for exceptions, types, contracts, and agent interfaces.

---

[Unreleased]: https://github.com/StrategicMilk/AM-Workbench/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/StrategicMilk/AM-Workbench/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/StrategicMilk/AM-Workbench/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/StrategicMilk/AM-Workbench/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/StrategicMilk/AM-Workbench/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/StrategicMilk/AM-Workbench/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/StrategicMilk/AM-Workbench/releases/tag/v0.1.0
