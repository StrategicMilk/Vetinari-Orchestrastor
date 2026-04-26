from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_agent_routing.py"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "docs" / "internal" / "ai-workflows" / "agent-routing.yaml"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_agent_routing", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECKER = load_checker()


FRONTMATTER_LINES = [
    "routing_manifest: docs/internal/ai-workflows/agent-routing.yaml",
    "orchestrator_model: claude-opus-4-7",
    "orchestrator_effort: high",
    "default_worker_model: claude-sonnet-4-6",
    "default_worker_effort: high",
    "mechanical_worker_model: claude-haiku-4-5-20251001",
    "mechanical_worker_effort: medium",
    "reviewer_model: claude-sonnet-4-6",
    "reviewer_effort: high",
    "codex_default_model: gpt-5.4",
    "codex_mechanical_model: gpt-5.4-mini",
    "codex_terminal_model: gpt-5.3-codex",
    "codex_reviewer_model: gpt-5.4",
    "codex_architect_model: gpt-5.4",
]


def write_plan(path: Path, *, name: str, plan_type: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join([
        "---",
        f"name: {name}",
        f"plan_type: {plan_type}",
        *FRONTMATTER_LINES,
        "---",
        body.strip(),
        "",
    ])
    path.write_text(text, encoding="utf-8")


def manifest() -> dict:
    return CHECKER.load_yaml(MANIFEST_PATH)


def test_hierarchical_plan_pack_validates_recursively(tmp_path: Path) -> None:
    session_dir = tmp_path / "SESSION-01-demo"
    master = tmp_path / "MASTER-INDEX.md"
    parent = session_dir / "SESSION.md"
    shard = session_dir / "SHARD-01-demo.md"

    write_plan(
        master,
        name="demo-master",
        plan_type="master-index",
        body="""
# Demo Master

## Load Rule
Load this file first.

## Execution Waves
- Wave 1: `SESSION-01-demo/SESSION.md`

## Session Registry
| Session | Status | Parent Plan |
|---|---|---|
| SESSION-01 | ready | `SESSION-01-demo/SESSION.md` |

## Routing Map
| Work | Session |
|---|---|
| demo | SESSION-01 |
""",
    )
    write_plan(
        parent,
        name="SESSION-01-demo",
        plan_type="session-orchestrator",
        body="""
# Session 01 Demo

## Dependencies
None.

## Scope
Demo parent.

## Owned Write Scope
- `docs/demo.md`

## Agent Shards
| Shard | Agent | Model | Effort | Parallel Group | Depends On | Owned Scope | Plan |
|---|---|---|---|---|---|---|---|
| SHARD-01 | builder | claude-sonnet-4-6 | high | wave-a | none | docs demo | `SHARD-01-demo.md` |

## Session Verification
- `echo verify`
""",
    )
    write_plan(
        shard,
        name="SHARD-01-demo",
        plan_type="agent-shard",
        body="""
# Shard 01 Demo

## Owned Write Scope
- `docs/demo.md`

## Tasks

### Task 1.1: Demo task
- id: task-1-1-demo
- agent: builder
- model: claude-sonnet-4-6
- effort: high
- acceptance: docs/demo.md contains demo text.

Do the demo task.

## Verification
- `echo verify`
""",
    )

    assert CHECKER.validate_plan(master, manifest()) == []


def test_session_orchestrator_requires_existing_shards(tmp_path: Path) -> None:
    parent = tmp_path / "SESSION-01-demo" / "SESSION.md"
    write_plan(
        parent,
        name="SESSION-01-demo",
        plan_type="session-orchestrator",
        body="""
# Session 01 Demo

## Dependencies
None.

## Scope
Demo parent.

## Agent Shards
| Shard | Agent | Model | Effort | Parallel Group | Depends On | Owned Scope | Plan |
|---|---|---|---|---|---|---|---|
| SHARD-01 | builder | claude-sonnet-4-6 | high | wave-a | none | docs demo | `SHARD-01-missing.md` |

## Session Verification
- `echo verify`
""",
    )

    failures = CHECKER.validate_plan(parent, manifest())

    assert any("referenced child plan does not exist" in failure.message for failure in failures)


def test_low_tier_task_touching_destructive_ownership_rejected(tmp_path: Path) -> None:
    """Mechanical-builder task referencing a destructive_ownership path is flagged."""
    shard = tmp_path / "SHARD-01-destructive.md"
    write_plan(
        shard,
        name="SHARD-01-destructive",
        plan_type="agent-shard",
        body="""
# Shard 01 Destructive

## Owned Write Scope
- `vetinari/safety/recycle.py`

## Tasks

### Task 1.1: Mechanical tweak to recycle
- id: task-1-1-recycle
- agent: mechanical-builder
- model: claude-haiku-4-5-20251001
- effort: low
- acceptance: vetinari/safety/recycle.py updated.

Update a comment in vetinari/safety/recycle.py.

## Verification
- `echo verify`
""",
    )

    failures = CHECKER.validate_plan(shard, manifest())

    assert any("destructive-ownership" in failure.message for failure in failures), (
        f"Expected destructive-ownership failure, got: {[f.message for f in failures]}"
    )


def test_low_tier_task_outside_destructive_ownership_accepted(tmp_path: Path) -> None:
    """Mechanical-builder task whose owned scope is NOT in destructive_ownership is accepted."""
    shard = tmp_path / "SHARD-01-benign.md"
    write_plan(
        shard,
        name="SHARD-01-benign",
        plan_type="agent-shard",
        body="""
# Shard 01 Benign

## Owned Write Scope
- `vetinari/utils/string_helpers.py`

## Tasks

### Task 1.1: Mechanical tweak to string helpers
- id: task-1-1-string-helpers
- agent: mechanical-builder
- model: claude-haiku-4-5-20251001
- effort: low
- acceptance: vetinari/utils/string_helpers.py updated.

Update a comment in vetinari/utils/string_helpers.py.

## Verification
- `echo verify`
""",
    )

    failures = CHECKER.validate_plan(shard, manifest())

    destructive_failures = [f for f in failures if "destructive-ownership" in f.message]
    assert destructive_failures == [], (
        f"Low-tier task outside destructive_ownership should not be flagged, "
        f"got: {[f.message for f in destructive_failures]}"
    )


def test_full_tier_task_touching_destructive_ownership_accepted(tmp_path: Path) -> None:
    """Full-tier worker task referencing a destructive_ownership path is allowed."""
    shard = tmp_path / "SHARD-01-recycle-full.md"
    write_plan(
        shard,
        name="SHARD-01-recycle-full",
        plan_type="agent-shard",
        body="""
# Shard 01 Recycle Full Tier

## Owned Write Scope
- `vetinari/safety/recycle.py`

## Tasks

### Task 1.1: Full-tier update to recycle
- id: task-1-1-recycle-full
- agent: builder
- model: claude-sonnet-4-6
- effort: high
- acceptance: vetinari/safety/recycle.py updated with new purge logic.

Add a new purge policy method to vetinari/safety/recycle.py.

## Verification
- `echo verify`
""",
    )

    failures = CHECKER.validate_plan(shard, manifest())

    destructive_failures = [f for f in failures if "destructive-ownership" in f.message]
    assert destructive_failures == [], (
        f"Full-tier task should not trigger destructive-ownership check, got: {[f.message for f in destructive_failures]}"
    )


def test_explicit_plan_argument_skips_default_plan_scan(tmp_path: Path, monkeypatch) -> None:
    shard = tmp_path / "SHARD-01-demo.md"
    write_plan(
        shard,
        name="SHARD-01-demo",
        plan_type="agent-shard",
        body="""
# Shard 01 Demo

## Owned Write Scope
- `docs/demo.md`

## Tasks

### Task 1.1: Demo task
- id: task-1-1-demo
- agent: builder
- model: claude-sonnet-4-6
- effort: high
- acceptance: docs/demo.md contains demo text.

Do the demo task.

## Verification
- `echo verify`
""",
    )

    def fail_default_scan(*_args, **_kwargs):
        raise AssertionError("explicit --plan should not scan default plan roots")

    monkeypatch.setattr(CHECKER, "default_plan_files", fail_default_scan)

    assert CHECKER.main(["--manifest", str(MANIFEST_PATH), "--plan", str(shard)]) == 0
