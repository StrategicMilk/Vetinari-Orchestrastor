"""Tests for durable audit-store materialization and promotion tooling."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from scripts import audit_store


def _finding(fid: str, cluster: str, severity: str = "high") -> dict[str, object]:
    return {
        "id": fid,
        "canonical_key": f"{cluster}:{fid}",
        "title": f"Finding {fid}",
        "status": "open",
        "severity": severity,
        "confidence": "high",
        "evidence_tier": "tier-3-code-backed",
        "scope": "test scope",
        "root_cause": "shared root cause",
        "root_cause_cluster_ids": [cluster],
        "impact": "operator impact",
        "lane_tags": ["deep-audit"],
        "stable_code_anchors": [
            {
                "path": "vetinari/example.py",
                "anchor_type": "exact-search",
                "lookup_key": "example",
            }
        ],
        "broken": "broken",
        "missing": "missing",
        "delete_targets": [],
        "merge_replace_targets": [],
        "bleeding_edge_standard": "standard",
        "long_term_redesign": "redesign",
        "evidence": [
            {
                "kind": "file",
                "path": "vetinari/example.py",
                "summary": "code evidence",
                "anchor_lookup_key": "example",
            }
        ],
        "artifacts": ["deep/SUMMARY.md"],
        "updated_at": "2026-04-19T00:00:00Z",
    }


def _write_registry(path: Path) -> None:
    payload = {
        "schema_version": "test-registry.v1",
        "suite_root": str(path.parent),
        "generated_at": "2026-04-19T00:00:00Z",
        "repo_fingerprint": {
            "cwd": "C:/dev/Vetinari",
            "head_commit": None,
            "dirty": False,
            "scope_note": "test",
        },
        "findings": [
            _finding("FSA-0001", "RCG-0001", "critical"),
            _finding("FSA-0002", "RCG-0001", "high"),
            _finding("FSA-0003", "RCG-0002", "medium"),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_graph(path: Path) -> None:
    payload = {
        "schema_version": "test-graph.v1",
        "run_id": "run-test",
        "run_root": str(path.parent),
        "generated_at": "2026-04-19T00:00:00Z",
        "clusters": [
            {
                "cluster_id": "RCG-0001",
                "label": "Primary cluster",
                "summary": "Two findings share the same cause.",
                "confidence": "high",
                "finding_ids": ["FSA-0001", "FSA-0002"],
                "lanes": ["deep-audit"],
                "artifact_paths": ["deep/SUMMARY.md"],
                "likely_causes": ["missing durable index"],
                "user_or_operator_impact": "crashes sessions",
                "recommended_attack_order": 1,
            },
            {
                "cluster_id": "RCG-0002",
                "label": "Secondary cluster",
                "summary": "One finding has a separate cause.",
                "confidence": "medium",
                "finding_ids": ["FSA-0003"],
                "lanes": ["deep-audit"],
                "artifact_paths": [],
                "likely_causes": ["stale index"],
                "user_or_operator_impact": "wastes tokens",
                "recommended_attack_order": 2,
            },
        ],
        "edges": [
            {
                "source_type": "finding",
                "source_id": "FSA-0001",
                "relation": "belongs_to",
                "target_type": "cluster",
                "target_id": "RCG-0001",
                "confidence": "high",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_materialize_sharded_registry_and_ai_codex_views(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    registry = tmp_path / "finding-registry.json"
    graph = tmp_path / "ROOT-CAUSE-GRAPH.json"
    ai_codex = tmp_path / ".ai-codex"
    _write_registry(registry)
    _write_graph(graph)

    audit_store.init_store(run_root)
    assert audit_store.ingest_finding_registry(run_root, registry) == 3
    assert audit_store.ingest_root_cause_graph(run_root, graph) == (2, 1)
    result = audit_store.materialize_views(run_root, shard_size=1, ai_codex_dir=ai_codex)

    assert result == {"cluster_count": 2, "finding_count": 3, "shard_count": 3}
    index = json.loads((run_root / "finding-registry" / "index.json").read_text(encoding="utf-8"))
    assert index["finding_count"] == 3
    assert index["severity_counts"]["critical"] == 1
    assert (run_root / "finding-registry" / "by-cluster" / "RCG-0001-001.json").exists()
    assert (run_root / "finding-registry" / "by-severity" / "critical.json").exists()
    assert json.loads((ai_codex / "audit-findings-index.json").read_text(encoding="utf-8"))["cluster_count"] == 2
    assert "Primary cluster" in (ai_codex / "audit-root-causes.md").read_text(encoding="utf-8")
    assert "vetinari/example.py" in (ai_codex / "audit-hotspots.md").read_text(encoding="utf-8")


def test_index_convergence_archives_materializes_pass_index(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    archive_dir = tmp_path / "convergence"
    archive_dir.mkdir()
    load_index = tmp_path / "CONVERGENCE-LOAD-INDEX.md"
    load_index.write_text("# index\n", encoding="utf-8")
    (archive_dir / "0001-0099.md").write_text(
        "## Pass 1\n\n- first summary\n\n## Pass 2\n\n- second summary\n",
        encoding="utf-8",
    )

    audit_store.init_store(run_root)
    assert audit_store.index_convergence_archives(run_root, load_index, archive_dir) == 2
    audit_store.materialize_views(run_root)

    pass_index = json.loads((run_root / "convergence-pass-index.json").read_text(encoding="utf-8"))
    assert pass_index["pass_count"] == 2
    assert pass_index["latest_pass"] == 2
    assert pass_index["passes"][0]["summary"] == "- first summary"


def test_promote_to_knowledge_surfaces_writes_cluster_level_entries(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    registry = tmp_path / "finding-registry.json"
    graph = tmp_path / "ROOT-CAUSE-GRAPH.json"
    knowledge_db = tmp_path / "knowledge-graph.db"
    wiki_log = tmp_path / "wiki" / "log.md"
    _write_registry(registry)
    _write_graph(graph)

    audit_store.init_store(run_root)
    audit_store.ingest_finding_registry(run_root, registry)
    audit_store.ingest_root_cause_graph(run_root, graph)
    actions = audit_store.promote_to_knowledge_surfaces(
        run_root,
        knowledge_db=knowledge_db,
        wiki_log=wiki_log,
        apply=True,
    )

    assert any("KG upsert failure node audit-rcg-0001" in action for action in actions)
    with sqlite3.connect(str(knowledge_db)) as conn:
        title = conn.execute(
            "SELECT title FROM nodes WHERE id = 'audit-rcg-0001'"
        ).fetchone()[0]
    assert title == "Primary cluster"
    assert "Audit root-cause index refreshed" in wiki_log.read_text(encoding="utf-8")
