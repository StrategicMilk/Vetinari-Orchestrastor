import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / ".codex" / "skills" / "full-spectrum-audit" / "scripts" / "init_audit_run.py"
)
VALIDATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / ".codex"
    / "skills"
    / "full-spectrum-audit"
    / "scripts"
    / "validate_checkpoint_state.py"
)
LANE_EVIDENCE_VALIDATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / ".codex"
    / "skills"
    / "full-spectrum-audit"
    / "scripts"
    / "validate_lane_evidence.py"
)
RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / ".codex"
    / "skills"
    / "full-spectrum-audit"
    / "scripts"
    / "run_full_spectrum_audit.py"
)
RUN_VALIDATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / ".codex"
    / "skills"
    / "full-spectrum-audit"
    / "scripts"
    / "validate_run.py"
)
SELECT_LATEST_RUN_PATH = (
    Path(__file__).resolve().parents[1]
    / ".codex"
    / "skills"
    / "full-spectrum-audit"
    / "scripts"
    / "select_latest_run.py"
)
FINDING_REGISTRY_VALIDATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / ".codex"
    / "skills"
    / "full-spectrum-audit"
    / "scripts"
    / "validate_finding_registry.py"
)
PLAN_COVERAGE_VALIDATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / ".codex"
    / "skills"
    / "full-spectrum-audit"
    / "scripts"
    / "validate_plan_coverage.py"
)
FALSE_NEGATIVE_EVAL_PATH = (
    Path(__file__).resolve().parents[1]
    / ".codex"
    / "skills"
    / "full-spectrum-audit"
    / "scripts"
    / "run_false_negative_evals.py"
)
PROBE_IDS = {
    "deep": ["deep-repo-hotspot-map", "deep-cross-surface-trace"],
    "structural-critique": ["structural-boundary-map", "structural-decision-challenge"],
    "performance-capacity": ["performance-latency-throughput", "performance-resource-growth"],
    "governance-theater": [
        "governance-advertised-implementation-map",
        "governance-semantic-discriminator",
    ],
}


def load_initializer():
    spec = importlib.util.spec_from_file_location("full_spectrum_init_audit_run", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


INITIALIZER = load_initializer()


def run_initializer(monkeypatch, workspace: Path, *extra_args: str) -> Path:
    run_id = "test-full-spectrum-run"
    audit_root = workspace / ".ai-codex" / "audit" / "full-spectrum"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "init_audit_run.py",
            "--workspace-root",
            str(workspace),
            "--audit-root",
            str(audit_root),
            "--run-id",
            run_id,
            "--no-auto-conditional",
            *extra_args,
        ],
    )

    assert INITIALIZER.main() == 0
    return audit_root / "runs" / run_id


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def minimal_finding(
    finding_id: str = "FSA-0001",
    *,
    cluster_ids: list[str] | None = None,
    artifacts: list[str] | None = None,
    status: str = "open",
) -> dict:
    return {
        "id": finding_id,
        "canonical_key": f"canonical-{finding_id.lower()}",
        "title": "Release proof false-green",
        "status": status,
        "severity": "high",
        "confidence": "high",
        "evidence_tier": "tier-3-code-backed",
        "scope": "full-spectrum validation",
        "root_cause": "Validator accepted unresolved proof.",
        "root_cause_cluster_ids": cluster_ids or ["RCG-0001"],
        "impact": "Open findings can be presented as release-safe.",
        "lane_tags": ["deep"],
        "stable_code_anchors": [],
        "broken": "Validator accepted missing evidence.",
        "missing": "Cross-artifact reference checks.",
        "delete_targets": [],
        "merge_replace_targets": [],
        "bleeding_edge_standard": "Every release claim resolves to executable evidence.",
        "long_term_redesign": "Make run validation the blocking entry point.",
        "evidence": [{"kind": "doc", "summary": "Fixture evidence."}],
        "artifacts": artifacts or ["HANDOFF-BRIEF.md"],
        "updated_at": "2026-04-21T00:00:00+00:00",
    }


def write_registry(path: Path, findings: list[dict]) -> None:
    write_json(
        path,
        {
            "schema_version": "1.0",
            "suite_root": str(path.parent),
            "generated_at": "2026-04-21T00:00:00+00:00",
            "repo_fingerprint": {
                "cwd": str(path.parent),
                "head_commit": None,
                "dirty": False,
                "scope_note": "test",
            },
            "findings": findings,
        },
    )


def required_doc_text(title: str, sections: list[str]) -> str:
    return "\n".join([f"# {title}", "", *[f"## {section}\nFixture content." for section in sections]]) + "\n"


def write_required_run_docs(run_root: Path) -> None:
    docs = {
        "HANDOFF-BRIEF.md": [
            "Audit scope and repo fingerprint",
            "Suite convergence status",
            "Enabled lanes and why they were enabled",
            "Top root causes across the suite",
            "Severity counts and evidence-tier distribution",
            "Critical constraints, assumptions, and open questions",
            "The action artifacts to consume",
            "Recommended planning objective",
            "Suggested planning priorities",
            "Full coverage accounting",
            "Appendix",
        ],
        "PREVENTION-HANDOFF.md": [
            "Run identity and scope",
            "Recurrent root causes worth preventing, not just fixing",
            "Rule candidates",
            "Anti-pattern candidates",
            "Hook or automation candidates",
            "Test and verification check candidates",
            "Lint, schema, or policy gate opportunities",
            "Suggested ownership",
            "Traceability back to canonical finding IDs",
        ],
        "CONTRADICTION-REPORT.md": [
            "Run identity and scope",
            "Summary of contradictions found",
            "Lane disagreements",
            "Conflicting evidence",
            "Recommended resolution path",
            "Findings that remain open because of contradiction",
            "Traceability back to canonical finding IDs",
        ],
        "DELTA-REPORT.md": [
            "Run identity and baseline",
            "New findings",
            "Fixed findings",
            "Regressed findings",
            "Unresolved findings",
            "Contradicted findings",
            "Waivers affecting interpretation",
            "Planning implications",
        ],
        "PREVENTION-DEPLOYMENT.md": [
            "Run identity and scope",
            "Prevention priorities to deploy now",
            "Rule and policy updates",
            "Hook and automation deployments",
            "CI, lint, schema, or gate deployments",
            "Skill and prompt deployments",
            "Verification and rollback plan",
            "Ownership and rollout order",
            "Traceability back to prevention candidates and canonical finding IDs",
        ],
    }
    for name, sections in docs.items():
        (run_root / name).write_text(required_doc_text(name, sections), encoding="utf-8")


def write_minimal_run_artifacts(
    run_root: Path,
    *,
    finding: dict | None = None,
    include_evidence_debt: bool = True,
    artifact_path: str = "HANDOFF-BRIEF.md",
    closure_items: list[dict] | None = None,
    waivers: list[dict] | None = None,
) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    finding = finding or minimal_finding(artifacts=[artifact_path])
    write_registry(run_root / "finding-registry.json", [finding])
    write_json(
        run_root / "ROOT-CAUSE-GRAPH.json",
        {
            "schema_version": "1.0",
            "run_id": run_root.name,
            "run_root": str(run_root),
            "generated_at": "2026-04-21T00:00:00+00:00",
            "clusters": [
                {
                    "cluster_id": "RCG-0001",
                    "label": "False-green validation",
                    "summary": "The run needs cross-artifact proof.",
                    "confidence": "high",
                    "finding_ids": ["FSA-0001"],
                    "lanes": ["deep"],
                    "artifact_paths": [artifact_path],
                    "likely_causes": ["Missing validator cross-checks."],
                    "user_or_operator_impact": "Release proof can be overstated.",
                    "recommended_attack_order": 1,
                }
            ],
            "edges": [],
        },
    )
    write_json(
        run_root / "ARTIFACT-INDEX.json",
        {
            "schema_version": "1.0",
            "run_id": run_root.name,
            "run_root": str(run_root),
            "generated_at": "2026-04-21T00:00:00+00:00",
            "artifacts": [
                {
                    "path": str(run_root / artifact_path),
                    "lane": None,
                    "artifact_type": "handoff",
                    "summary": "Fixture handoff.",
                    "headings": ["Run identity and scope"],
                    "tags": ["handoff"],
                    "is_shard": False,
                    "finding_ids": ["FSA-0001"],
                    "related_artifacts": [],
                }
            ],
        },
    )
    write_json(
        run_root / "CLOSURE-STATUS.json",
        {
            "schema_version": "1.0",
            "run_id": run_root.name,
            "evaluated_at": "2026-04-21T00:00:00+00:00",
            "findings": closure_items
            if closure_items is not None
            else [
                {
                    "finding_id": "FSA-0001",
                    "closure_status": "still-open",
                    "verification_type": "review",
                    "evidence_refs": [],
                    "verified_at": "2026-04-21T00:00:00+00:00",
                }
            ],
        },
    )
    write_json(
        run_root / "WAIVERS.json",
        {
            "schema_version": "1.0",
            "generated_at": "2026-04-21T00:00:00+00:00",
            "waivers": waivers or [],
        },
    )
    (run_root / "HANDOFF-BRIEF.md").write_text("# Handoff\n\nFSA-0001 remains open.\n", encoding="utf-8")
    write_required_run_docs(run_root)
    if include_evidence_debt:
        (run_root / "EVIDENCE-DEBT.md").write_text("- FSA-0001: open\n", encoding="utf-8")


def write_lane_evidence(run_root: Path, lane: str, probe_ids: list[str] | None = None) -> None:
    probe_ids = probe_ids or PROBE_IDS.get(lane, PROBE_IDS["deep"])
    write_json(
        run_root / lane / "LANE-EVIDENCE.json",
        {
            "schema_version": "1.0.0",
            "run_id": "terminal-run",
            "lane": lane,
            "round": 1,
            "status": "converged",
            "generated_at": "2026-04-21T00:00:00+00:00",
            "inspected_files": [
                {
                    "path": "tests/test_example.py",
                    "reason": "Evidence anchor for lane proof.",
                    "evidence_tier": "tier-3-code-backed",
                }
            ],
            "commands_run": [
                {
                    "command_id": "cmd-1",
                    "command": "python -m pytest tests/test_example.py -q",
                    "cwd": ".",
                    "purpose": "Executable lane proof command.",
                    "status": "passed",
                    "exit_code": 0,
                    "evidence_refs": ["tests/test_example.py"],
                }
            ],
            "probes_attempted": [
                {
                    "probe_id": probe_id,
                    "description": f"Required probe {probe_id}.",
                    "status": "passed",
                    "evidence_tier": "tier-3-code-backed",
                    "evidence_refs": ["tests/test_example.py"],
                    "command_refs": ["cmd-1"],
                    "finding_ids": [],
                    "pass_summary": f"{probe_id} passed with concrete evidence.",
                }
                for probe_id in probe_ids
            ],
            "findings": [],
            "skipped_areas": [],
            "no_findings_proof": [
                {
                    "check_id": "coverage-map",
                    "claim": "The lane mapped its target surface.",
                    "result": "passed",
                    "evidence_tier": "tier-3-code-backed",
                    "evidence_refs": ["tests/test_example.py"],
                },
                {
                    "check_id": "hostile-probe",
                    "claim": "The lane attempted its hostile or purpose-specific probe.",
                    "result": "passed",
                    "evidence_tier": "tier-3-code-backed",
                    "evidence_refs": ["tests/test_example.py"],
                },
                {
                    "check_id": "finding-dedupe",
                    "claim": "No new canonical finding remained after dedupe.",
                    "result": "passed",
                    "evidence_tier": "tier-3-code-backed",
                    "evidence_refs": ["tests/test_example.py"],
                },
            ],
            "notes": "Substantive lane evidence for validator tests.",
        },
    )


def substantive_lane_summary() -> str:
    return "\n".join(
        [
            "# Lane Summary",
            "",
            "## What is broken",
            "Concrete broken behavior. Evidence tier: tier-3-code-backed.",
            "## What is missing",
            "Concrete missing capability. Evidence: tests/test_example.py.",
            "## What should be deleted",
            "Concrete deletion candidate.",
            "## What should be merged or replaced",
            "Concrete merge or replacement candidate.",
            "## Bleeding-edge standard",
            "Current high bar for repo-health, module workflow, architecture boundary, ADR tradeoff, and tension review.",
            "## Better long-term redesign",
            "Durable redesign path. Impact: user and operator. Root cause: fragmented proof.",
            "## Evidence Debt",
            "Remaining proof work. The lane records why the conclusion is not just a generic summary: it names the "
            "repo-health surface inspected, the architecture boundary challenged, the ADR decision tension, the "
            "test evidence anchor, and the follow-up verification needed before downstream planning can treat the "
            "finding as closed. This text intentionally exceeds the terminal-run proof floor so the validator can "
            "distinguish a substantive lane artifact from headings padded with a few isolated trigger words.",
            "",
        ]
    )


def universal_only_lane_summary() -> str:
    return "\n".join(
        [
            "# Lane Summary",
            "",
            "## What is broken",
            "Concrete broken behavior. Evidence tier: tier-3-code-backed.",
            "## What is missing",
            "Concrete missing capability. Evidence: tests/test_example.py.",
            "## What should be deleted",
            "Concrete deletion candidate.",
            "## What should be merged or replaced",
            "Concrete merge or replacement candidate.",
            "## Bleeding-edge standard",
            "Current high bar.",
            "## Better long-term redesign",
            "Durable redesign path. Impact: user and operator. Root cause: fragmented proof.",
            "## Evidence Debt",
            "Remaining proof work.",
            "",
        ]
    )


def governance_theater_lane_summary() -> str:
    return "\n".join(
        [
            "# Governance Theater Lane Summary",
            "",
            "## What is broken",
            "An advertised implementation claim can say DAPO while the runtime code path uses DPO. Evidence tier: tier-3-code-backed.",
            "## What is missing",
            "The lane needs a semantic discriminator and negative control that proves the named algorithm implements its defining behavior instead of a cheaper substitute.",
            "## What should be deleted",
            "Delete status laundering language that treats import-only training proof as governance evidence.",
            "## What should be merged or replaced",
            "Replace name-vs-reality DAPO wording with DPO unless the implementation adds the missing runtime semantics.",
            "## Bleeding-edge standard",
            "Governance, authority, policy, certifier, and named algorithm claims resolve to code path evidence, defining behavior, and a blocking discriminator.",
            "## Better long-term redesign",
            "Root cause: labels can be made internally consistent while executable implementation semantics remain wrong. Impact: operator and user trust shifts to theater. The redesign requires every advertised mechanism to carry label source, live code path, substitute analysis, and known-bad evidence.",
            "## Evidence Debt",
            "Remaining proof work includes re-running the advertised implementation map against ADR, training, audit validator, release, and policy surfaces. The lane must keep a separate inventory of MATCH, SUBSTITUTE, PARTIAL, LABEL-ONLY, OUT-OF-SCOPE, and EVIDENCE-DEBT classifications so governance theater is not hidden inside generic claim/reality review.",
            "",
        ]
    )


def write_terminal_run(
    workspace: Path,
    lane: str = "deep",
    *,
    include_manifest: bool = True,
    include_lane_evidence: bool = True,
    summary_text: str | None = None,
    include_adr: bool = False,
    include_adr_review: bool = False,
) -> Path:
    run_root = workspace / ".ai-codex" / "audit" / "full-spectrum" / "runs" / "terminal-run"
    lane_dir = run_root / lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    (run_root / "lane-status.md").write_text("# Lane Status\n", encoding="utf-8")
    summary_path = lane_dir / "LANE-SUMMARY.md"
    summary_path.write_text(summary_text or substantive_lane_summary(), encoding="utf-8")
    if include_lane_evidence:
        write_lane_evidence(run_root, lane)
    if include_adr:
        adr_dir = workspace / "adr"
        adr_dir.mkdir(parents=True, exist_ok=True)
        write_json(adr_dir / "ADR-0001.json", {"adr_id": "ADR-0001", "status": "accepted"})
    if include_adr_review:
        (run_root / "structural-critique").mkdir(parents=True, exist_ok=True)
        (run_root / "structural-critique" / "ADR-REVIEW.md").write_text(
            "# ADR Review\n\n- ADR-0001: challenged for current-code tension.\n",
            encoding="utf-8",
        )
    checkpoint = {
        "schema_version": "1.0.0",
        "run_id": "terminal-run",
        "run_root": str(run_root),
        "updated_at": "2026-04-21T00:00:00+00:00",
        "phase": "suite-converged",
        "baseline_run_id": None,
        "current_round": 1,
        "run_limits": {
            "max_rounds": 6,
            "max_lane_sweeps": 3,
            "max_lane_probes": 50,
            "max_runtime_minutes": 720,
            "max_concurrent_agents": 2,
            "allow_interactive_between_rounds": False,
            "on_limit": "finalize-partial-with-evidence-debt-and-session-plans",
            "requires_session_plans": True,
        },
        "rounds": [
            {
                "round": 1,
                "completed_at": "2026-04-21T00:00:00+00:00",
                "lanes_run": [lane],
                "new_findings_this_round": 0,
                "registry_restructures_this_round": 0,
                "finding_registry_snapshot_hash": "sha256:test",
            }
        ],
        "reusable_artifacts": [],
        "lane_states": [
            {
                "lane": lane,
                "status": "converged",
                "last_completed_step": "full scope complete",
                "last_updated": "2026-04-21T00:00:00+00:00",
                "artifacts": [str(summary_path)],
                "reusable": True,
                "reason": "test",
                "converged_at_round": 1,
                "last_run_round": 1,
                "implicated_by_round": None,
                "sweep_count": 1,
                "probe_count": 0,
                "resume_count": 0,
                "restart_count": 0,
                "frontier_count": 0,
                "pass_cursor": None,
            }
        ],
        "suite_convergence": {
            "round": 1,
            "new_global_findings": 0,
            "registry_restructures": 0,
            "ready_to_stop": True,
            "convergence_reason": "test",
        },
        "pending_reruns": [],
    }
    write_json(run_root / "CHECKPOINT-STATE.json", checkpoint)
    if include_manifest:
        write_json(
            run_root / "AGENT-LAUNCH-MANIFEST.json",
            {
                "schema_version": "1.0.0",
                "run_id": "terminal-run",
                "run_root": str(run_root),
                "generated_at": "2026-04-21T00:00:00+00:00",
                "status": "completed",
                "launch_policy": {
                    "max_concurrent_agents": 2,
                    "batching": "launch-at-most-cap-then-wait-for-completion-before-next-wave",
                    "scope": "All Codex lane and lane-local subagents in this run share this cap.",
                },
                "lanes": [
                    {
                        "lane": lane,
                        "slug": lane,
                        "artifact_dir": str(lane_dir),
                        "status": "completed",
                        "agent_id": f"agent-{lane}",
                        "agent_role": "audit-lane-agent",
                        "launched_at": "2026-04-21T00:00:00+00:00",
                        "completed_at": "2026-04-21T00:01:00+00:00",
                    }
                ],
            },
        )
    return run_root


def write_single_run_index(run_root: Path, *, status: str = "completed") -> Path:
    index = run_root.parent.parent / "RUN-INDEX.json"
    write_json(
        index,
        {
            "schema_version": "1.0",
            "generated_at": "2026-04-21T00:05:00+00:00",
            "latest_run_id": run_root.name,
            "latest_run_root": str(run_root),
            "runs": [
                {
                    "run_id": run_root.name,
                    "run_root": str(run_root),
                    "started_at": "2026-04-21T00:00:00+00:00",
                    "completed_at": "2026-04-21T00:05:00+00:00" if status == "completed" else None,
                    "scope_note": "terminal validation fixture",
                    "head_commit": None,
                    "status": status,
                    "archived": False,
                    "pinned": False,
                    "has_handoff_brief": True,
                    "has_prevention_handoff": True,
                    "has_contradiction_report": True,
                }
            ],
        },
    )
    return index


def run_checkpoint_validator(run_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed Python executable and repo-local script path.
        [sys.executable, str(VALIDATOR_PATH), str(run_root / "CHECKPOINT-STATE.json")],
        check=False,
        capture_output=True,
        text=True,
    )


def run_lane_evidence_validator(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed Python executable and repo-local script path.
        [sys.executable, str(LANE_EVIDENCE_VALIDATOR_PATH), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )


def run_full_spectrum_runner(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed Python executable and repo-local script path.
        [sys.executable, str(RUNNER_PATH), *(str(arg) for arg in args)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_initializer_records_unattended_run_limits(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    run_root = run_initializer(monkeypatch, workspace)

    checkpoint = read_json(run_root / "CHECKPOINT-STATE.json")
    assert checkpoint["run_limits"] == {
        "max_rounds": 6,
        "max_lane_sweeps": 3,
        "max_lane_probes": 50,
        "max_runtime_minutes": 720,
        "max_concurrent_agents": 2,
        "allow_interactive_between_rounds": False,
        "on_limit": "finalize-partial-with-evidence-debt-and-session-plans",
        "requires_session_plans": True,
    }
    assert checkpoint["suite_convergence"]["ready_to_stop"] is False
    assert "Rounds continue automatically" in checkpoint["notes"]


def test_initializer_overrides_limits_and_tracks_launch_visibility(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    run_root = run_initializer(
        monkeypatch,
        workspace,
        "--max-rounds",
        "2",
        "--max-lane-sweeps",
        "1",
        "--max-lane-probes",
        "7",
        "--max-runtime-minutes",
        "90",
        "--max-concurrent-agents",
        "3",
        "--allow-interactive-between-rounds",
    )

    checkpoint = read_json(run_root / "CHECKPOINT-STATE.json")
    assert checkpoint["run_limits"]["max_rounds"] == 2
    assert checkpoint["run_limits"]["max_lane_sweeps"] == 1
    assert checkpoint["run_limits"]["max_lane_probes"] == 7
    assert checkpoint["run_limits"]["max_runtime_minutes"] == 90
    assert checkpoint["run_limits"]["max_concurrent_agents"] == 3
    assert checkpoint["run_limits"]["allow_interactive_between_rounds"] is True

    launch_manifest = read_json(run_root / "AGENT-LAUNCH-MANIFEST.json")
    assert launch_manifest["status"] == "initialized"
    assert launch_manifest["launch_policy"]["max_concurrent_agents"] == 3
    assert launch_manifest["lanes"]
    assert all(lane["agent_id"] is None for lane in launch_manifest["lanes"])
    assert all(lane["status"] == "not-launched" for lane in launch_manifest["lanes"])


def test_initializer_creates_lane_evidence_placeholders(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    run_root = run_initializer(monkeypatch, workspace)

    evidence = read_json(run_root / "deep" / "LANE-EVIDENCE.json")
    assert evidence["lane"] == "deep"
    assert evidence["status"] == "not-started"
    assert evidence["probes_attempted"] == []
    artifact_index = read_json(run_root / "ARTIFACT-INDEX.json")
    assert any(entry["artifact_type"] == "lane-evidence" for entry in artifact_index["artifacts"])


def test_initializer_includes_governance_theater_lane(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    run_root = run_initializer(monkeypatch, workspace)

    checkpoint = read_json(run_root / "CHECKPOINT-STATE.json")
    lane_names = {state["lane"] for state in checkpoint["lane_states"]}
    assert "governance-theater-audit" in lane_names
    assert (run_root / "governance-theater" / "LANE-EVIDENCE.json").exists()


def test_checkpoint_validator_rejects_terminal_run_without_launch_manifest(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_root = write_terminal_run(workspace, include_manifest=False)

    result = run_checkpoint_validator(run_root)

    assert result.returncode == 1
    assert "Missing lane launch proof" in result.stderr


def test_checkpoint_validator_rejects_thin_lane_summary(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_root = write_terminal_run(workspace, summary_text="# Lane Summary\n\nOnly ids.\n")

    result = run_checkpoint_validator(run_root)

    assert result.returncode == 1
    assert "lane artifacts missing mandatory output sections" in result.stderr


def test_checkpoint_validator_rejects_lane_missing_purpose_specific_proof(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_root = write_terminal_run(
        workspace,
        lane="performance-capacity",
        summary_text=universal_only_lane_summary(),
        include_lane_evidence=False,
    )

    result = run_checkpoint_validator(run_root)

    assert result.returncode == 1
    assert "missing proof for performance dimension" in result.stderr


def test_lane_evidence_validator_rejects_missing_required_probe(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_root = write_terminal_run(workspace, lane="dead-feature")
    evidence_path = run_root / "dead-feature" / "LANE-EVIDENCE.json"
    evidence = read_json(evidence_path)
    evidence["probes_attempted"] = []
    write_json(evidence_path, evidence)

    result = run_lane_evidence_validator(evidence_path)

    assert result.returncode == 1
    assert "missing required probe" in result.stderr


def test_checkpoint_validator_requires_structural_adr_review_when_adrs_exist(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_root = write_terminal_run(workspace, lane="structural-critique", include_adr=True)

    result = run_checkpoint_validator(run_root)

    assert result.returncode == 1
    assert "missing ADR debate artifact" in result.stderr


def test_checkpoint_validator_accepts_substantive_terminal_run(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_root = write_terminal_run(
        workspace,
        lane="structural-critique",
        include_adr=True,
        include_adr_review=True,
    )

    result = run_checkpoint_validator(run_root)

    assert result.returncode == 0, result.stderr


def test_checkpoint_validator_accepts_governance_theater_lane(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_root = write_terminal_run(
        workspace,
        lane="governance-theater",
        summary_text=governance_theater_lane_summary(),
    )

    result = run_checkpoint_validator(run_root)

    assert result.returncode == 0, result.stderr


def test_run_index_rejects_completed_latest_without_terminal_validation(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_root = write_terminal_run(workspace, include_manifest=False)
    write_minimal_run_artifacts(run_root)
    index = write_single_run_index(run_root)

    result = subprocess.run(  # noqa: S603 - fixed Python executable and repo-local script path.
        [sys.executable, str(RUN_VALIDATOR_PATH), "--index", str(index), "--index-only"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "completed run failed terminal validation" in result.stderr
    assert "Missing lane launch proof" in result.stderr


def test_select_latest_rejects_completed_latest_without_terminal_validation(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_root = write_terminal_run(workspace, include_manifest=False)
    write_minimal_run_artifacts(run_root)
    index = write_single_run_index(run_root)

    result = subprocess.run(  # noqa: S603 - fixed Python executable and repo-local script path.
        [sys.executable, str(SELECT_LATEST_RUN_PATH), "--index", str(index)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "RUN-INDEX.json is not selectable" in result.stderr
    assert "Missing lane launch proof" in result.stderr


def test_select_latest_accepts_completed_latest_with_terminal_validation(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_root = write_terminal_run(workspace)
    write_minimal_run_artifacts(run_root)
    index = write_single_run_index(run_root)

    result = subprocess.run(  # noqa: S603 - fixed Python executable and repo-local script path.
        [sys.executable, str(SELECT_LATEST_RUN_PATH), "--index", str(index)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(run_root)


def test_runner_refuses_completed_finalize_without_convergence(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_root = run_initializer(monkeypatch, workspace)

    result = run_full_spectrum_runner("finalize", run_root, "--status", "completed")

    assert result.returncode == 1
    assert "ready_to_stop is not true" in result.stderr


def test_runner_enforces_codex_active_agent_cap(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_root = run_initializer(monkeypatch, workspace, "--max-concurrent-agents", "2")

    first = run_full_spectrum_runner("record-launch", run_root, "--lane", "deep", "--agent-id", "agent-1")
    second = run_full_spectrum_runner(
        "record-launch",
        run_root,
        "--lane",
        "bare-minimum",
        "--agent-id",
        "agent-2",
    )
    overflow = run_full_spectrum_runner(
        "record-launch",
        run_root,
        "--lane",
        "murphys-law",
        "--agent-id",
        "agent-3",
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert overflow.returncode == 2
    assert "Launch cap reached: 2/2 active agents" in overflow.stderr

    completed = run_full_spectrum_runner("record-completion", run_root, "--lane", "deep")
    retried = run_full_spectrum_runner(
        "record-launch",
        run_root,
        "--lane",
        "murphys-law",
        "--agent-id",
        "agent-3",
    )

    assert completed.returncode == 0, completed.stderr
    assert retried.returncode == 0, retried.stderr
    manifest = read_json(run_root / "AGENT-LAUNCH-MANIFEST.json")
    assert manifest["active_agent_count"] == 2
    assert manifest["launch_policy"]["max_concurrent_agents"] == 2


def test_runner_reserves_codex_launch_slot_before_agent_id(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_root = run_initializer(monkeypatch, workspace, "--max-concurrent-agents", "2")

    reserved = run_full_spectrum_runner(
        "reserve-launch",
        run_root,
        "--lane",
        "deep",
        "--agent-role",
        "round3-deep-audit-lane-agent",
    )
    launched = run_full_spectrum_runner("record-launch", run_root, "--lane", "deep", "--agent-id", "agent-1")
    second_reserved = run_full_spectrum_runner("reserve-launch", run_root, "--lane", "bare-minimum")
    overflow = run_full_spectrum_runner("reserve-launch", run_root, "--lane", "murphys-law")
    released_reservation = run_full_spectrum_runner(
        "record-completion",
        run_root,
        "--lane",
        "bare-minimum",
        "--status",
        "cancelled",
    )

    assert reserved.returncode == 0, reserved.stderr
    assert launched.returncode == 0, launched.stderr
    assert second_reserved.returncode == 0, second_reserved.stderr
    assert overflow.returncode == 2
    assert "Launch cap reached: 2/2 active agents" in overflow.stderr
    assert released_reservation.returncode == 0, released_reservation.stderr

    manifest = read_json(run_root / "AGENT-LAUNCH-MANIFEST.json")
    deep = next(entry for entry in manifest["lanes"] if entry["slug"] == "deep")
    bare_minimum = next(entry for entry in manifest["lanes"] if entry["slug"] == "bare-minimum")
    assert deep["agent_id"] == "agent-1"
    assert deep["status"] == "launched"
    assert bare_minimum["agent_id"] is None
    assert bare_minimum["status"] == "cancelled"
    assert manifest["active_agent_count"] == 1


def test_initializer_rejects_codex_agent_cap_above_crash_threshold(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    audit_root = workspace / ".ai-codex" / "audit" / "full-spectrum"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "init_audit_run.py",
            "--workspace-root",
            str(workspace),
            "--audit-root",
            str(audit_root),
            "--run-id",
            "bad-cap",
            "--no-auto-conditional",
            "--max-concurrent-agents",
            "7",
        ],
    )

    try:
        INITIALIZER.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("initializer accepted a crash-prone Codex agent cap")


def test_false_negative_eval_harness_materializes_fixture_repos(tmp_path):
    target = tmp_path / "fixtures"

    result = subprocess.run(  # noqa: S603 - fixed Python executable and repo-local script path.
        [sys.executable, str(FALSE_NEGATIVE_EVAL_PATH), "--materialize", str(target)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert len([path for path in target.iterdir() if path.is_dir()]) >= 24


def test_finding_registry_and_plan_coverage_reject_zero_proof(tmp_path):
    registry = tmp_path / "finding-registry.json"
    write_registry(registry, [])
    plan = tmp_path / "SESSION.md"
    plan.write_text("# Session\n\nNo canonical findings are named.\n", encoding="utf-8")

    registry_result = subprocess.run(  # noqa: S603 - fixed Python executable and repo-local script path.
        [sys.executable, str(FINDING_REGISTRY_VALIDATOR_PATH), str(registry)],
        check=False,
        capture_output=True,
        text=True,
    )
    coverage_result = subprocess.run(  # noqa: S603 - fixed Python executable and repo-local script path.
        [sys.executable, str(PLAN_COVERAGE_VALIDATOR_PATH), str(registry), str(plan)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert registry_result.returncode == 1
    assert "at least one canonical finding" in registry_result.stderr
    assert coverage_result.returncode == 1
    assert "Registry contains no canonical finding IDs" in coverage_result.stderr


def test_plan_coverage_rejects_missing_pass_anchors_and_coordination_parents(tmp_path):
    registry = tmp_path / "finding-registry.json"
    write_registry(registry, [minimal_finding()])
    plan = tmp_path / "SESSION-34B.md"
    plan.write_text(
        "\n".join(
            [
                "# Session 34B",
                "Load only full-spectrum/audit-validator passes named here.",
                "Route FSA-0001 to executable child work.",
                "Execute 34D as the implementation shard.",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(  # noqa: S603 - fixed Python executable and repo-local script path.
        [sys.executable, str(PLAN_COVERAGE_VALIDATOR_PATH), str(registry), str(plan)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "names no exact convergence pass" in result.stderr
    assert "coordination-only parent" in result.stderr


def test_plan_coverage_rejects_frozen_routes_and_release_report_without_caveat(tmp_path):
    registry = tmp_path / "finding-registry.json"
    write_registry(registry, [minimal_finding()])
    plan = tmp_path / "release-report.md"
    plan.write_text(
        "\n".join(
            [
                "# Release Report",
                "FSA-0001 is routed to SESSION-29G.",
                "Run 20260421T170658-full-spectrum reached suite convergence and is release-safe.",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(  # noqa: S603 - fixed Python executable and repo-local script path.
        [sys.executable, str(PLAN_COVERAGE_VALIDATOR_PATH), str(registry), str(plan)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "frozen 29-series" in result.stderr
    assert "without required open-finding caveat" in result.stderr


def test_run_validator_rejects_stale_latest_selection_and_missing_evidence_debt(tmp_path):
    audit_root = tmp_path / ".ai-codex" / "audit" / "full-spectrum"
    old_run = audit_root / "runs" / "old-run"
    new_run = audit_root / "runs" / "new-run"
    write_minimal_run_artifacts(old_run, include_evidence_debt=False)
    write_minimal_run_artifacts(new_run)
    index = audit_root / "RUN-INDEX.json"
    write_json(
        index,
        {
            "schema_version": "1.0",
            "generated_at": "2026-04-21T00:00:00+00:00",
            "latest_run_id": "old-run",
            "latest_run_root": str(old_run),
            "runs": [
                {
                    "run_id": "old-run",
                    "run_root": str(old_run),
                    "started_at": "2026-04-21T00:00:00+00:00",
                    "completed_at": "2026-04-21T00:01:00+00:00",
                    "scope_note": "old",
                    "head_commit": None,
                    "status": "completed",
                    "archived": False,
                    "pinned": False,
                    "has_handoff_brief": True,
                    "has_prevention_handoff": True,
                    "has_contradiction_report": True,
                },
                {
                    "run_id": "new-run",
                    "run_root": str(new_run),
                    "started_at": "2026-04-21T00:02:00+00:00",
                    "completed_at": "2026-04-21T00:03:00+00:00",
                    "scope_note": "new",
                    "head_commit": None,
                    "status": "completed",
                    "archived": False,
                    "pinned": False,
                    "has_handoff_brief": True,
                    "has_prevention_handoff": True,
                    "has_contradiction_report": True,
                },
            ],
        },
    )

    result = subprocess.run(  # noqa: S603 - fixed Python executable and repo-local script path.
        [
            sys.executable,
            str(RUN_VALIDATOR_PATH),
            "--index",
            str(index),
            "--skip-terminal-proof",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "stale latest-run selection" in result.stderr
    assert "Missing evidence debt artifact" in result.stderr


def test_run_validator_rejects_unresolved_cross_references(tmp_path):
    run_root = tmp_path / ".ai-codex" / "audit" / "full-spectrum" / "runs" / "bad-run"
    write_minimal_run_artifacts(
        run_root,
        finding=minimal_finding(cluster_ids=["RCG-9999"], artifacts=["missing-action.md"]),
        artifact_path="missing-action.md",
        closure_items=[
            {
                "finding_id": "FSA-9999",
                "closure_status": "resolved",
                "verification_type": "review",
                "evidence_refs": [],
                "verified_at": "2026-04-21T00:00:00+00:00",
            }
        ],
        waivers=[
            {
                "waiver_id": "FSA-WVR-0001",
                "finding_id": "FSA-9999",
                "status": "active",
                "rationale": "bad fixture",
                "owner": "tests",
                "created_at": "2026-04-21T00:00:00+00:00",
                "expires_at": "2026-04-22T00:00:00+00:00",
                "recheck_after": "2026-04-21T12:00:00+00:00",
            }
        ],
    )

    result = subprocess.run(  # noqa: S603 - fixed Python executable and repo-local script path.
        [
            sys.executable,
            str(RUN_VALIDATOR_PATH),
            str(run_root),
            "--skip-terminal-proof",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "unknown root-cause cluster" in result.stderr
    assert "action artifact path does not exist" in result.stderr
    assert "Closure status references unknown finding" in result.stderr
    assert "Waiver references unknown finding" in result.stderr
