# Scripts

These scripts are public developer and release utilities. They are grouped by
purpose so the public tree does not read like a pile of one-off maintainer
tools.

## Release And Publication

- `release/check_publication_boundary.py` - rejects private paths, model blobs, and
  oversized history blobs before publishing.
- `release/check_release_artifacts.py` - inspects wheel/sdist contents.
- `release/pre_release_gate.py` - checks changelog/release claim discipline.
- `release/release_certifier.py` - release evidence and package metadata checks.
- `release/release_doctor.py` - end-to-end release smoke workflow.

## Development Gates

- `dev/run_tests.py` and `dev/test_summary.py` - local test execution and readable
  summaries.
- `quality/run_mypy_gate.py`, `quality/check_syntax.py`,
  `quality/semgrep_scan.py` - focused static
  analysis.
- `quality/check_vetinari_rules.py`, `quality/check_test_quality.py`,
  `quality/check_noqa_suppressions.py` - project rule and test quality gates.
- `maintenance/check_config_wiring.py`, `maintenance/check_wiring_audit.py`,
  `maintenance/check_doc_contract_alignment.py`, `maintenance/verify_wiring.py`
  - wiring and contract checks.

## Maintenance And Inspection

- `inspect/generate_architecture.py`, `inspect/generate_config_docs.py`,
  `inspect/generate_route_to_test_matrix.py` - generated reference material.
- `inspect/impact_analysis.py`, `dev/file_context.py` - change impact and
  context helpers.
- `maintenance/migrate_to_unified_db.py`, `maintenance/validate_vetinari.py` -
  maintenance utilities.
- `inspect/run_benchmarks.py` - smoke benchmark helper.

## One-Off Repair Helpers

Historical one-off repair helpers were removed from the public export. If a
repair workflow is still useful, promote it into one of the categories above
with public documentation and tests.

## Runtime Launchers

Repo-root launchers such as `start.bat`, `start.sh`, `python.cmd`, and the WSL
backend PowerShell helpers are public convenience entry points. They remain in
the root because users expect those commands to work from a fresh checkout.
