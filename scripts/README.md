# Scripts

These scripts are public developer and release utilities. Active command paths
are intentionally kept stable because tests, docs, and pre-commit reference
them directly.

## Release And Publication

- `check_publication_boundary.py` - rejects private paths, model blobs, and
  oversized history blobs before publishing.
- `check_release_artifacts.py` - inspects wheel/sdist contents.
- `pre_release_gate.py` - checks changelog/release claim discipline.
- `release_certifier.py` - release evidence and package metadata checks.
- `release_doctor.py` - end-to-end release smoke workflow.
- `check_ci_release_proof.py` - validates expected CI release gates.

## Development Gates

- `run_tests.py` and `test_summary.py` - local test execution and readable
  summaries.
- `run_mypy_gate.py`, `check_syntax.py`, `semgrep_scan.py` - focused static
  analysis.
- `check_vetinari_rules.py`, `check_test_quality.py`,
  `check_noqa_suppressions.py` - project rule and test quality gates.
- `check_config_wiring.py`, `check_wiring_audit.py`,
  `check_doc_contract_alignment.py`, `verify_wiring.py` - wiring and contract
  checks.

## Maintenance And Inspection

- `generate_architecture.py`, `generate_config_docs.py`,
  `generate_route_to_test_matrix.py` - generated reference material.
- `impact_analysis.py`, `impact_map.py`, `file_context.py` - change impact and
  context helpers.
- `migrate_to_unified_db.py`, `validate_vetinari.py` - maintenance utilities.
- `run_benchmarks.py` - smoke benchmark helper.

## One-Off Repair Helpers

`fix_vet023.py`, `fix_vet120.py`, `fix_vet123.py`, and `_fuzz_failures.py`
are retained only because tests still document their behavior. They should be
removed or moved once those historical tests are retired.

## Runtime Launchers

Repo-root launchers such as `start.bat`, `start.sh`, `python.cmd`, and the WSL
backend PowerShell helpers are public convenience entry points. They remain in
the root while tests and docs use those paths.
