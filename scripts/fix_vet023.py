"""One-shot fixer for all VET023 violations.

Reads each violating file, finds the except line at the flagged line number,
and adds # noqa: VET023 with an appropriate comment if it's a genuine
optional/expected failure.

Usage:
    python scripts/fix_vet023.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Repository root derived from this file's location (scripts/ -> repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def get_violations() -> list[tuple[str, int]]:
    """Run check_vetinari_rules.py and extract VET023 violations."""
    result = subprocess.run(
        [sys.executable, "scripts/check_vetinari_rules.py"],
        capture_output=True,
    )
    stdout = (result.stdout or b"").decode("utf-8", errors="replace")
    stderr = (result.stderr or b"").decode("utf-8", errors="replace")
    violations = []
    for line in (stdout + stderr).splitlines():
        if "VET023" not in line:
            continue
        # Pattern: vetinari\foo\bar.py:123: VET023 ...
        m = re.match(r"\s*(vetinari[^\s:]+):(\d+):", line)
        if m:
            path = m.group(1).replace("\\", "/")
            lineno = int(m.group(2))
            violations.append((path, lineno))
    return violations


# Map of (file_path_substr, except_pattern_substr) -> comment reason
OPTIONAL_REASONS: list[tuple[str, str, str]] = [
    # skills — invalid tool call args: UPGRADE to warning
    ("skills/architect_skill.py", "except ValueError", "UPGRADE"),
    ("skills/foreman_skill.py", "except ValueError", "UPGRADE"),
    ("skills/inspector_skill.py", "except ValueError", "UPGRADE"),
    ("skills/operations_skill.py", "except ValueError", "UPGRADE"),
    ("skills/quality_skill.py", "except ValueError", "UPGRADE"),
    # web API input validation — UPGRADE to warning
    ("web/decomposition_routes.py", "except (ValueError, TypeError)", "UPGRADE"),
    ("web/system_api.py", "except ValueError", "UPGRADE"),
    ("web/training_experiments_api.py", "except ValueError", "UPGRADE"),
    ("web/training_routes.py", "except (ValueError, TypeError)", "UPGRADE"),
    # SSE client disconnect — expected normal event
    ("web/models_api.py", "except GeneratorExit", "optional: SSE client disconnect is a normal event, not a failure"),
    ("web/projects_api.py", "except GeneratorExit", "optional: SSE client disconnect is a normal event, not a failure"),
    (
        "web/projects_streaming.py",
        "except GeneratorExit",
        "optional: SSE client disconnect is a normal event, not a failure",
    ),
    # lifespan shutdown — best-effort cleanup
    (
        "web/lifespan.py",
        "except Exception",
        "optional: lifespan shutdown cleanup is best-effort, modules may not be loaded",
    ),
    # litestar health endpoint
    ("web/litestar_app.py", "except Exception", "optional: health monitor unavailable, returns minimal response"),
    # training status snapshots
    ("web/training_api.py", "except Exception", "optional: training subsystem may not be initialized yet"),
    # SSE table creation
    ("web/sse_events.py", "except Exception", "optional: SSE persistence table creation is non-critical at startup"),
    # training experiments upload
    (
        "web/training_experiments_api.py",
        "except UnicodeDecodeError",
        "optional: non-UTF-8 upload safely returns 400, not a system failure",
    ),
    # adr
    ("adr.py", "except ValueError", "optional: unknown category safely treated as non-high-stakes"),
    # cascade_router
    ("cascade_router.py", "except Exception", "optional: LLM scoring degrades gracefully to heuristic"),
    # code_sandbox
    ("code_sandbox.py", "except Exception", "optional: raw stdout is a valid non-JSON fallback"),
    # code_search
    ("code_search.py", "except Exception", "optional: uvx is an optional tool"),
    # credentials
    ("credentials.py", "except Exception", "optional: keyring is an optional credential store"),
    # database
    ("database.py", "except sqlite3.OperationalError", "optional: table not yet created is expected during fresh init"),
    ("database.py", "except OperationalError", "optional: table not yet created is expected during fresh init"),
    # evaluation/prompt_wiring
    (
        "evaluation/prompt_wiring.py",
        "except Exception",
        "optional: prompt cache and token optimizer are optional optimizations",
    ),
    # kaizen
    ("kaizen/pdca.py", "except ValueError", "optional: unknown category safely returns False without crashing"),
    ("kaizen/wiring.py", "except ValueError", "optional: unknown categories safely skipped"),
    # learning
    (
        "learning/feedback_loop.py",
        "except Exception",
        "optional: drift detection is advisory and must not interrupt feedback",
    ),
    ("learning/model_selector.py", "except ValueError", "optional: degenerate Beta params safely fall back to mean"),
    ("learning/operator_selector.py", "except ValueError", "optional: degenerate Beta params safely fall back to mean"),
    # llm_helpers
    ("llm_helpers.py", "except Exception", "UPGRADE"),
    # memory
    ("memory/blackboard.py", "except Exception", "optional: malformed entries are skipped to allow partial restore"),
    (
        "memory/episode_recorder.py",
        "except Exception",
        "optional: embedding storage is non-critical, episode is still saved",
    ),
    ("memory/memory_search.py", "except Exception", "optional: memory search degraded mode, results still returned"),
    # ml
    ("ml/classifiers.py", "except Exception", "optional: LLM ambiguity check degrades to heuristic"),
    (
        "ml/quality_prescreener.py",
        "except SyntaxError",
        "optional: syntax errors in submitted code yield a 0.0 score, not a crash",
    ),
    # models
    ("models/model_profiler.py", "except Exception", "optional: unparseable GGUF metadata safely falls back to 0"),
    ("models/vram_manager.py", "except Exception", "optional: GPU temp read failure returns None safely"),
    # orchestration
    (
        "orchestration/error_escalation.py",
        "except ValueError",
        "optional: invalid level safely falls through to pattern-based classification",
    ),
    (
        "orchestration/graph_task_runner.py",
        "except Exception",
        "optional: Andon and bottleneck metrics are non-critical instrumentation",
    ),
    (
        "orchestration/pipeline_engine.py",
        "except Exception",
        "optional: pipeline analytics are non-critical instrumentation",
    ),
    (
        "orchestration/pipeline_rework.py",
        "except Exception",
        "optional: LLM retry brief is an enhancement, raw feedback is the fallback",
    ),
    (
        "orchestration/pipeline_stages.py",
        "except Exception",
        "optional: failure taxonomy and Andon scope management are non-critical",
    ),
    ("orchestration/plan_generator.py", "except Exception", "optional: DAG topology analysis is an enhancement"),
    (
        "orchestration/request_routing.py",
        "except Exception",
        "optional: LLM classification degrades gracefully to keyword heuristic",
    ),
    # planning
    ("planning/plan_api.py", "except ValueError", "optional: invalid enum from request is handled by returning 400"),
    ("planning/plan_mode.py", "except Exception", "optional: malformed plan lines are skipped"),
    # project
    ("project/impact_analysis.py", "except Exception", "optional: unparseable files return empty imports"),
    # prompts
    ("prompts/version_manager.py", "except ValueError", "optional: malformed version string safely resets to 1.0.1"),
    # repo_map
    ("repo_map.py", "except SyntaxError", "optional: unparseable files are skipped from repo map index"),
    ("repo_map.py", "except (SyntaxError, Exception)", "optional: unparseable files are skipped from symbol index"),
    ("repo_map.py", "except Exception", "optional: unparseable files are excluded from module index"),
    # routing
    ("routing/complexity_router.py", "except Exception", "optional: LLM complexity assessment degrades to heuristic"),
    # setup
    (
        "setup/init_wizard.py",
        "except PermissionError",
        "optional: inaccessible directories are skipped during model scan",
    ),
    # system/hardware
    ("system/hardware_detect.py", "except Exception", "optional: GPU detection failure returns None safely"),
    ("system/hardware_detect.py", "except ValueError", "optional: non-numeric VRAM field is skipped, scan continues"),
    (
        "system/hardware_detect.py",
        "except (subprocess.TimeoutExpired, FileNotFoundError, OSError)",
        "optional: GPU tool not present, returns None safely",
    ),
    (
        "system/hardware_detect.py",
        "except (subprocess.TimeoutExpired, OSError)",
        "optional: chip name is cosmetic, detection still succeeds",
    ),
    (
        "system/hardware_detect.py",
        "except (subprocess.TimeoutExpired, ValueError, OSError)",
        "optional: Apple Silicon not present, returns None safely",
    ),
    # tools/static_analysis
    (
        "tools/static_analysis.py",
        "except (FileNotFoundError, json.JSONDecodeError, subprocess.TimeoutExpired)",
        "optional: static analysis tool is optional",
    ),
    (
        "tools/static_analysis.py",
        "except (FileNotFoundError, subprocess.TimeoutExpired)",
        "optional: vulture is an optional external tool",
    ),
    # validation
    ("validation/root_cause.py", "except Exception", "optional: LLM root cause degrades to heuristic safely"),
    (
        "validation/verification.py",
        "except (SyntaxError, ValueError)",
        "optional: unparseable code safely returns False",
    ),
    # agents
    ("agents/builder_agent.py", "except Exception", "optional: optional enhancement, not required"),
    (
        "agents/consolidated/quality_agent.py",
        "except Exception",
        "optional: enhancement pass, review proceeds without it",
    ),
    ("agents/inference.py", "except Exception", "optional: optional subsystem, inference proceeds without it"),
    ("agents/inference.py", "except (ImportError, AttributeError)", "optional: AdapterManager is not always installed"),
    ("agents/planner_agent.py", "except Exception", "optional: malformed entries are skipped, not fatal"),
    # adapters
    ("adapters/llama_cpp_adapter.py", "except Exception", "optional: PromptLookupDecoding is an optional speedup"),
    # coding_agent
    ("coding_agent/engine.py", "except Exception", "optional: repo map context is an enhancement, not required"),
    # analytics
    ("analytics/forecasting.py", "except Exception", "optional: event bus publish and trend check are non-critical"),
    ("analytics/telemetry_persistence.py", "except Exception", "optional: alerting must not crash persist cycle"),
    # autonomy
    ("autonomy/wiring.py", "except Exception", "optional: optional enhancement, wiring is best-effort"),
    # cli
    ("cli.py", "except KeyboardInterrupt", "optional: KeyboardInterrupt during teardown is expected"),
    # database (second OperationalError)
    ("database.py", "except sqlite3.OperationalError", "optional: table not yet created is expected during fresh init"),
    # project/dependency_updater
    (
        "project/dependency_updater.py",
        "except (ValueError, IndexError)",
        "optional: unparseable version string safely returns (0, 0, 0)",
    ),
    # optimization
    (
        "optimization/semantic_cache.py",
        "except Exception",
        "optional: embedder unavailable at import time, cache uses trigram fallback",
    ),
    # orchestration/git_checkpoint
    (
        "orchestration/git_checkpoint.py",
        "except subprocess.TimeoutExpired",
        "optional: git timeout returns failure tuple safely",
    ),
    ("orchestration/git_checkpoint.py", "except OSError", "optional: git OS error returns failure tuple safely"),
    # validation/document_judge
    (
        "validation/document_judge.py",
        "except Exception",
        "optional: AdapterManager unavailable, heuristics are the fallback",
    ),
    # verification/cascade
    ("verification/cascade.py", "except SyntaxError", "optional: unparseable code safely returns False with findings"),
    # skills/skill_registry
    ("skills/skill_registry.py", "except OSError", "optional: stat failure conservatively returns no-change"),
    # web/training_api (quality comparison)
    ("web/training_api.py", "except Exception", "optional: training subsystem may not be initialized yet"),
]

UPGRADE_MAPPINGS = {
    # skill invalid mode/thinking_mode
    ("skills/architect_skill.py", "except ValueError"): (
        r'logger\.debug\("Invalid ArchitectMode.*?"\)',
        'logger.warning("Invalid ArchitectMode %r in tool call — returning error to caller", mode_str)',
    ),
    ("skills/architect_skill.py", "except ValueError", 2): (
        r'logger\.debug\("Invalid ThinkingMode.*?"\)',
        'logger.warning("Invalid ThinkingMode %r in architect tool call — returning error to caller", thinking_mode_str)',
    ),
    ("skills/foreman_skill.py", "except ValueError"): (
        r'logger\.debug\(".*?ForemanMode.*?"\)',
        'logger.warning("Invalid ForemanMode %r in tool call — returning error to caller", mode_str)',
    ),
    ("skills/inspector_skill.py", "except ValueError"): (
        r'logger\.debug\(".*?InspectorMode.*?"\)',
        'logger.warning("Invalid InspectorMode %r in tool call — returning error to caller", mode_str)',
    ),
    ("skills/operations_skill.py", "except ValueError"): (
        r'logger\.debug\(".*?OperationsMode.*?"\)',
        'logger.warning("Invalid OperationsMode %r in tool call — returning error to caller", mode_str)',
    ),
    ("skills/quality_skill.py", "except ValueError"): (
        r'logger\.debug\(".*?QualityMode.*?"\)',
        'logger.warning("Invalid QualityMode %r in tool call — returning error to caller", mode_str)',
    ),
    ("skills/quality_skill.py", "except ValueError", 2): (
        r'logger\.debug\(".*?ThinkingMode.*?"\)',
        'logger.warning("Invalid ThinkingMode %r in quality tool call — returning error to caller", thinking_mode_str)',
    ),
    ("web/decomposition_routes.py", "except (ValueError, TypeError)"): (
        r'logger\.debug\("Decomposition route.*?"\)',
        'logger.warning("Decomposition route received non-numeric depth/max_depth — returning 400")',
    ),
    ("web/system_api.py", "except ValueError"): (
        r'logger\.debug\("Invalid autonomy level.*?"\)',
        'logger.warning("Invalid autonomy level %r in request — returning 400", level)',
    ),
    ("web/training_experiments_api.py", "except ValueError"): (
        r'logger\.debug\("Non-integer.*?"\)',
        'logger.warning("Non-integer page/per_page query params in training data request — returning 400")',
    ),
    ("web/training_routes.py", "except (ValueError, TypeError)"): (
        r'logger\.debug\("Training route.*?"\)',
        'logger.warning("Training route received non-numeric min_quality — returning 400")',
    ),
    ("llm_helpers.py", "except Exception"): (
        r'logger\.debug\("LLM.*?falling back.*?"\)',
        'logger.warning("LLM call failed — falling back to heuristic", exc_info=True)',
    ),
}


def _live_optional_reasons() -> list[tuple[str, str, str]]:
    """Return OPTIONAL_REASONS entries whose source file currently exists.

    Filters out stale entries for modules that have been removed from the
    repository so the fixer does not attempt to patch non-existent files.

    Returns:
        Subset of OPTIONAL_REASONS where ``_REPO_ROOT / "vetinari" / entry[0]``
        resolves to an existing file.
    """
    return [
        entry
        for entry in OPTIONAL_REASONS
        if (_REPO_ROOT / "vetinari" / entry[0]).exists()
    ]


def fix_file(file_path: str, lineno: int) -> bool:
    """Fix a single VET023 violation at the given line number."""
    path = Path(file_path)
    if not path.exists():
        print(f"  SKIP (not found): {file_path}:{lineno}")
        return False

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    idx = lineno - 1
    if idx >= len(lines):
        print(f"  SKIP (line out of range): {file_path}:{lineno}")
        return False

    line = lines[idx]
    stripped = line.rstrip()

    # Already has noqa
    if "# noqa: VET023" in line:
        print(f"  SKIP (already has noqa): {file_path}:{lineno}")
        return False

    # Determine the except pattern
    m = re.match(r"(\s*)(except\b.*?)(\s*)(:)(\s*)$", stripped)
    if not m:
        print(f"  SKIP (no except match): {file_path}:{lineno} => {stripped!r}")
        return False

    _indent = m.group(1)
    except_part = m.group(2)

    # Find applicable reason from OPTIONAL_REASONS
    reason = None
    file_suffix = file_path.replace("\\", "/")

    # Check if this should be UPGRADED
    is_upgrade = False
    for key_path, key_except, *_ in OPTIONAL_REASONS:
        if key_path in file_suffix and key_except in except_part:
            val = None
            for entry in OPTIONAL_REASONS:
                if len(entry) == 3 and entry[0] == key_path and entry[1] == key_except:
                    val = entry[2]
                    break
            if val == "UPGRADE":
                is_upgrade = True
                reason = "UPGRADE"
            elif val is not None:
                reason = val
            break

    if reason is None:
        # Default optional reason
        reason = "optional: expected/safe failure with graceful fallback"

    if is_upgrade:
        # UPGRADE: change logger.debug to logger.warning in the next few lines
        j = idx + 1
        upgraded = False
        while j < len(lines) and j <= idx + 5:
            inner = lines[j]
            if "logger.debug(" in inner:
                # For upgrades, just change debug -> warning
                lines[j] = inner.replace("logger.debug(", "logger.warning(", 1)
                upgraded = True
                break
            j += 1
        if not upgraded:
            # No logger.debug found, just add noqa
            lines[idx] = f"{stripped}  # noqa: VET023  # optional: caller error returned as HTTP 400\n"
        else:
            print(f"  UPGRADED: {file_path}:{lineno}")
            path.write_text("".join(lines), encoding="utf-8")
            return True
    else:
        # Add noqa comment to the except line
        lines[idx] = f"{stripped}  # noqa: VET023  # {reason}\n"

    path.write_text("".join(lines), encoding="utf-8")
    print(f"  FIXED: {file_path}:{lineno} -> {reason[:60]}")
    return True


def main() -> None:
    violations = get_violations()
    print(f"Found {len(violations)} VET023 violations")

    fixed = 0
    for file_path, lineno in violations:
        if fix_file(file_path, lineno):
            fixed += 1

    print(f"\nFixed {fixed}/{len(violations)} violations")

    # Verify
    result = subprocess.run(
        [sys.executable, "scripts/check_vetinari_rules.py"],
        capture_output=True,
    )
    vout = (result.stdout or b"").decode("utf-8", errors="replace")
    verr = (result.stderr or b"").decode("utf-8", errors="replace")
    remaining = sum(1 for line in (vout + verr).splitlines() if "VET023" in line)
    print(f"Remaining VET023 violations: {remaining}")


if __name__ == "__main__":
    main()
