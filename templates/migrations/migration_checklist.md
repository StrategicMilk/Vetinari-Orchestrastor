# Vetinari Migration Checklist

Use this checklist when migrating a skill to the Tool interface or adding a
new Phase capability.

---

## Pre-migration

- [ ] Read `docs/SKILL_MIGRATION_GUIDE.md`
- [ ] Review `docs/DEVELOPER_GUIDE.md` for coding conventions
- [ ] Check `docs/MIGRATION_INDEX.md` for phase dependencies
- [ ] Create a feature branch: `git checkout -b migrate/your-skill-name`

---

## Implementation

- [ ] Copy `templates/migrations/new_skill_template.py`
      → `vetinari/skills/your_skill_name.py`
- [ ] Copy `templates/migrations/new_skill_tests_template.py`
      → `tests/test_your_skill_name.py`
- [ ] Replace all `MySkill` / `my_skill` / `my-skill` placeholders
- [ ] Implement all `# TODO` sections
- [ ] Add the skill to `vetinari/skills/__init__.py` exports
- [ ] Add the tool wrapper to `vetinari/tools/__init__.py` exports
- [ ] Update `vetinari/skills_registry.json` with the new skill entry

---

## Telemetry

- [ ] Wrap model calls with `telemetry.record_adapter_latency(...)`
- [ ] Wrap memory operations with `telemetry.record_memory_write/read/search(...)`
- [ ] Use `CorrelationContext` for distributed trace propagation

---

## Observability integration

- [ ] Register alert thresholds in `vetinari/dashboard/alerts.py` if needed
- [ ] Feed cost entries via `get_cost_tracker().record(...)` for paid providers
- [ ] Register SLO targets in `vetinari/analytics/sla.py` if SLA required

---

## Testing

- [ ] All unit tests pass: `pytest tests/test_your_skill_name.py -v`
- [ ] No regressions: `pytest tests/regression/ -v`
- [ ] Full suite passes: `pytest tests/ -q`
- [ ] Coverage >= 80%: `pytest --cov=vetinari/skills/your_skill_name`

---

## Documentation

- [ ] Module docstring written (purpose, usage example)
- [ ] All public methods have docstrings
- [ ] Add entry to `docs/MIGRATION_INDEX.md` artifact table
- [ ] Update phase status in `docs/MIGRATION_INDEX.md` if completing a phase

---

## Review & merge

- [ ] Self-review: run `git diff` and verify no debug code / secrets
- [ ] Security check: `from vetinari.security import get_secret_scanner`
      and verify no patterns in new code
- [ ] Create PR with description matching `GIT_COMMIT_GUIDE.md` conventions
- [ ] At least one reviewer approves
- [ ] CI pipeline green (all jobs pass)
- [ ] Merge to `develop`, then to `main` after integration test

---

## Post-merge

- [ ] Update `docs/MIGRATION_INDEX.md` status to **Complete**
- [ ] Add example to `examples/` if skill has interesting usage patterns
- [ ] Announce in team channel with link to PR
