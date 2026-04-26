# VET Rule Fixture Inventory

Catalogue of all VET rules enforced by `scripts/check_vetinari_rules.py`, with one bad/good fixture pair per rule.

Each bad fixture MUST trigger the rule when passed to `check_file()`.
Each good fixture MUST produce zero violations for that rule.

## Test Strategies

| Strategy | Rules | How |
|----------|-------|-----|
| `single_file` | Most rules | `check_file(path)` directly |
| `vetinari_scoped` | Rules gated on `is_in_vetinari()` | Monkeypatch `is_in_vetinari → True` before calling `check_file()` |
| `hot_path` | VET130 | Monkeypatch `_HOT_PATH_FILES` to include fixture path |
| `markdown` | VET100-102 | Direct file content check via `check_markdown_files()` equivalent |
| `cross_file` | VET109, VET120-124 | Integration test — not in parametrized suite; see `tests/integration/` |

## Rule Table

| Rule ID | Category | Description | Detection | Severity | Scope | Fixture Bad | Fixture Good |
|---------|----------|-------------|-----------|----------|-------|-------------|--------------|
| VET001 | imports | AgentType used as string literal (not imported from vetinari.types) | regex | error | any | bad_vet001.py | good_vet001.py |
| VET002 | imports | TaskStatus used as string literal (not imported from vetinari.types) | regex | error | any | bad_vet002.py | good_vet002.py |
| VET003 | imports | ExecutionMode used as string literal (not imported from vetinari.types) | regex | error | any | bad_vet003.py | good_vet003.py |
| VET004 | imports | PlanStatus used as string literal (not imported from vetinari.types) | regex | error | any | bad_vet004.py | good_vet004.py |
| VET005 | imports | Duplicate enum definition — already defined in canonical source | regex | error | any | bad_vet005.py | good_vet005.py |
| VET006 | imports | Wildcard import from vetinari.* | regex | error | any | bad_vet006.py | good_vet006.py |
| VET010 | style | Missing `from __future__ import annotations` | regex | warning | vetinari_scoped | bad_vet010.py | good_vet010.py |
| VET020 | error_handling | Bare `except:` without exception type | AST | error | any | bad_vet020.py | good_vet020.py |
| VET022 | error_handling | Empty except block — swallowed exception | AST | error | any | bad_vet022.py | good_vet022.py |
| VET023 | error_handling | Exception logged at DEBUG level for real failure | AST | warning | vetinari_scoped | bad_vet023.py | good_vet023.py |
| VET024 | error_handling | Success-masking except (sets result True in except) | AST | error | vetinari_scoped | bad_vet024.py | good_vet024.py |
| VET025 | error_handling | Broad except with early exit but no logging | AST | warning | vetinari_scoped | bad_vet025.py | good_vet025.py |
| VET030 | completeness | TODO/FIXME/HACK without issue reference | regex | error | any | bad_vet030.py | good_vet030.py |
| VET031 | completeness | `pass` as sole function body | AST | error | any | bad_vet031.py | good_vet031.py |
| VET032 | completeness | `...` (Ellipsis) as sole function body | AST | error | any | bad_vet032.py | good_vet032.py |
| VET033 | completeness | raise NotImplementedError outside @abstractmethod | AST | error | any | bad_vet033.py | good_vet033.py |
| VET034 | completeness | Placeholder string detected | regex | warning | any | bad_vet034.py | good_vet034.py |
| VET035 | completeness | print() in production code | AST | error | vetinari_scoped | bad_vet035.py | good_vet035.py |
| VET036 | completeness | Commented-out code block | regex | warning | any | bad_vet036.py | good_vet036.py |
| VET040 | security | Hardcoded credential pattern | regex | error | any | bad_vet040.py | good_vet040.py |
| VET041 | security | Hardcoded localhost URL | regex | warning | any | bad_vet041.py | good_vet041.py |
| VET050 | logging | Root logger usage (logging.info() not logger.info()) | regex | warning | vetinari_scoped | bad_vet050.py | good_vet050.py |
| VET051 | logging | f-string in logger call | regex | warning | vetinari_scoped | bad_vet051.py | good_vet051.py |
| VET060 | robustness | open() without encoding= parameter | regex | warning | vetinari_scoped | bad_vet060.py | good_vet060.py |
| VET061 | robustness | Debug code (breakpoint/pdb) | regex | error | vetinari_scoped | bad_vet061.py | good_vet061.py |
| VET062 | robustness | time.sleep() > 5 seconds | regex | warning | vetinari_scoped | bad_vet062.py | good_vet062.py |
| VET063 | robustness | os.path.join() — prefer pathlib.Path | regex | warning | vetinari_scoped | bad_vet063.py | good_vet063.py |
| VET070 | integration | Import not in pyproject.toml dependencies | regex | error | vetinari_scoped | bad_vet070.py | good_vet070.py |
| VET081 | organization | Python directory missing __init__.py | filesystem | error | vetinari_scoped | (integration) | (integration) |
| VET082 | organization | Filename not snake_case | filesystem | warning | any | bad_VET082.py | good_vet082.py |
| VET090 | documentation | Public class/function missing docstring | AST | warning | vetinari_scoped | bad_vet090.py | good_vet090.py |
| VET091 | documentation | Docstring too short (<10 chars) | AST | warning | vetinari_scoped | bad_vet091.py | good_vet091.py |
| VET092 | documentation | Docstring missing Args section (2+ params) | AST | warning | vetinari_scoped | bad_vet092.py | good_vet092.py |
| VET093 | documentation | Docstring missing Returns section | AST | warning | vetinari_scoped | bad_vet093.py | good_vet093.py |
| VET094 | documentation | Docstring missing Raises section | AST | warning | vetinari_scoped | bad_vet094.py | good_vet094.py |
| VET095 | documentation | Module missing module-level docstring | AST | warning | vetinari_scoped | bad_vet095.py | good_vet095.py |
| VET096 | documentation | Docstring restates function name verbatim | AST | warning | vetinari_scoped | bad_vet096.py | good_vet096.py |
| VET100 | markdown | Markdown file missing top-level # heading | text | error | markdown | bad_vet100.md | good_vet100.md |
| VET101 | markdown | Markdown empty section | text | warning | markdown | bad_vet101.md | good_vet101.md |
| VET102 | markdown | Markdown file very short content | text | warning | markdown | bad_vet102.md | good_vet102.md |
| VET103 | ai_antipatterns | datetime.now() without timezone / deprecated utcnow() | regex | warning | vetinari_scoped | bad_vet103.py | good_vet103.py |
| VET104 | ai_antipatterns | raise without `from exc` — exception chain lost | AST | warning | vetinari_scoped | bad_vet104.py | good_vet104.py |
| VET105 | ai_antipatterns | Manual to_dict() on dataclass | AST | warning | vetinari_scoped | bad_vet105.py | good_vet105.py |
| VET106 | ai_antipatterns | Zero-logic property (just returns self._x) | AST | warning | vetinari_scoped | bad_vet106.py | good_vet106.py |
| VET107 | ai_antipatterns | Entry/exit logging ("entering function X") | regex | warning | vetinari_scoped | bad_vet107.py | good_vet107.py |
| VET108 | ai_antipatterns | Redundant `return True if X else False` | AST | warning | vetinari_scoped | bad_vet108.py | good_vet108.py |
| VET109 | ai_antipatterns | ABC with only one concrete implementation | AST | warning | cross_file | (integration) | (integration) |
| VET110 | ai_antipatterns | Empty __init__.py without __all__ | AST | warning | vetinari_scoped | bad_vet110.py | good_vet110.py |
| VET111 | ai_antipatterns | Redundant intermediate variable | AST | warning | vetinari_scoped | bad_vet111.py | good_vet111.py |
| VET112 | ai_antipatterns | Defensive `or ""` on non-nullable field | AST | warning | vetinari_scoped | bad_vet112.py | good_vet112.py |
| VET113 | ai_antipatterns | Domain class missing __repr__ | AST | warning | vetinari_scoped | bad_vet113.py | good_vet113.py |
| VET114 | ai_antipatterns | Value-type @dataclass not using frozen=True | AST | error | vetinari_scoped | bad_vet114.py | good_vet114.py |
| VET115 | ai_antipatterns | yaml.safe_load/json.load inside route handler | AST | error | vetinari_scoped | bad_vet115.py | good_vet115.py |
| VET116 | ai_antipatterns | str(e) returned to API client | regex | warning | vetinari_scoped | bad_vet116.py | good_vet116.py |
| VET120 | wiring | Public function defined but never called | cross-file | error | cross_file | (integration) | (integration) |
| VET121 | wiring | Public class defined but never referenced | cross-file | error | cross_file | (integration) | (integration) |
| VET122 | wiring | Python module never imported | cross-file | warning | cross_file | (integration) | (integration) |
| VET123 | wiring | __init__.py re-export never imported externally | cross-file | warning | cross_file | (integration) | (integration) |
| VET124 | wiring | Test file has no corresponding source module | cross-file | warning | cross_file | (integration) | (integration) |
| VET130 | hot_path | Import statement inside hot-path function body | AST | error | hot_path | bad_vet130.py | good_vet130.py |
| VET141 | security | Filesystem write with no preceding `enforce_blocked_paths` in a sandbox-importing module | AST | error | vetinari_scoped | bad_vet141.py | good_vet141.py |
| VET142 | security | shutil.rmtree / os.remove in web/ or safety/ without @protected_mutation or lifecycle-fence comment | AST | error | vetinari_scoped | vet142_bad.py | vet142_good.py |
| VET210 | structural | Singleton without double-checked locking | AST | error | any | bad_vet210.py | good_vet210.py |
| VET220 | structural | Unbounded list[float] in analytics class | AST | warning | any | bad_vet220.py | good_vet220.py |
| VET230 | structural | Relative path to data/DB file | regex | warning | any | bad_vet230.py | good_vet230.py |

## Notes

- **VET081**: Requires a directory without `__init__.py` — filesystem-level check that cannot be meaningfully tested with a single Python file. Covered by integration tests.
- **VET082**: Bad fixture filename itself IS the violation — the file must be named `bad_VET082.py` (not snake_case) to trigger the rule.
- **VET100-102**: Check Markdown files, not Python. Fixtures are `.md` files tested via a separate helper.
- **VET109, VET120-124**: Cross-file rules requiring full codebase index. Not in the parametrized suite. See `tests/integration/test_vet_cross_file.py`.
- **VET130**: Requires the fixture path to appear in `_HOT_PATH_FILES`. Test monkeypatches that set to trigger the rule.
- **VET141**: Only fires for modules that import `vetinari.security.sandbox`. The bad fixture imports the module and writes without a preceding `enforce_blocked_paths` call; the good fixture guards every write. The rule is scope-aware — writes inside a function body are checked against guards in the same function; nested function bodies are scoped separately.
- **VET142**: Scoped to `vetinari/web/**` and `vetinari/safety/**`. The bad fixture calls `shutil.rmtree` without `@protected_mutation` or a `# VET142-excluded:` comment. The good fixture uses `RecycleStore.retire` with the exclusion comment. Excludes `protected_mutation.py` itself and `purge_expired` in `recycle.py`.
