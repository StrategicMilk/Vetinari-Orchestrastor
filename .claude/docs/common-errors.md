# Common Errors and Recovery

## Import Errors

**`ModuleNotFoundError: No module named 'vetinari.X'`**
- Check if the module exists: `ls vetinari/X.py` or `ls vetinari/X/`
- If it's a new module, ensure `__init__.py` exists in the directory
- If it's a third-party dep, check it's in pyproject.toml and installed

**`ImportError: cannot import name 'X' from 'vetinari.Y'`**
- The symbol was likely moved or renamed. Search for it: `grep -r "class X" vetinari/`
- If it's an enum, it probably lives in `vetinari/types.py`

## Test Failures

**Tests pass individually but fail together**
- Shared mutable state. Check for module-level dicts/lists being mutated.
- Check for missing `pytest.fixture` cleanup.

**`TimeoutError` in tests**
- Test is hitting a real network endpoint. Mock it.
- Default timeout is 30s. If legitimately slow, mark with `@pytest.mark.slow`.

**Tests fail after modifying contracts.py or types.py**
- These are high-impact files. Run ALL tests: `python -m pytest tests/ -x -q`
- Check if you removed a field or enum value that existing code depends on.

## Build Errors

**`ruff` reports fixable errors**
- Run `python -m ruff check --fix vetinari/` to auto-fix.
- Then `python -m ruff format vetinari/` to reformat.

**Circular import**
- Move the shared type to `vetinari/types.py`
- Use `TYPE_CHECKING` guard: `if TYPE_CHECKING: from vetinari.X import Y`
- Use late/lazy import inside the function body

## Git Errors

**Pre-commit hook fails**
- Fix the reported issues, don't skip hooks with `--no-verify`
- Run `pre-commit run --all-files` to see all violations

---

*This file is a reference guide for AI agents recovering from common failure states.*
