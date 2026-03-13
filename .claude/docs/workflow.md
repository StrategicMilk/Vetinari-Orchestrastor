# Development Workflow

## Branch Naming

```
feature/short-description        # New features
fix/bug-description              # Bug fixes
refactor/what-is-changing        # Refactoring
docs/what-is-documented          # Documentation only
test/what-is-being-tested        # Test additions
chore/what-is-being-done         # Maintenance tasks
```

## Commit Conventions

Follow Conventional Commits format:

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `ci`

Examples:
```
feat(researcher): add git_workflow mode to ConsolidatedResearcherAgent
fix(quality): correct security pattern matching for f-string SQL injection
test(builder): add tests for image_generation mode
docs(agents): update AGENTS.md with Phase 3 delegation rules
```

## Before Committing

1. Run `python -m pytest tests/ -x -q` — all tests must pass.
2. Verify `python -c "import vetinari; print('OK')"` succeeds.
3. Check no new `TODO`/`FIXME`/`HACK` comments without issue references.
4. Confirm no hardcoded secrets or credentials.
5. Confirm all new public functions have type hints and docstrings.

## Pull Request Requirements

- All CI checks must pass.
- At least one test added for every new function.
- AGENTS.md updated if agent roles, modes, or file jurisdiction changed.
- CHANGELOG.md entry added under `[Unreleased]`.

---

*This file is a reference copy of the workflow section from the Vetinari Development Guide.*
