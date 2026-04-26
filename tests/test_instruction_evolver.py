"""Tests for vetinari/learning/instruction_evolver.py"""

from __future__ import annotations

import pytest

from vetinari.learning.instruction_evolver import (
    InstructionEvolver,
    InstructionRule,
    InstructionTier,
    get_instruction_evolver,
)


@pytest.fixture
def evolver(tmp_path, monkeypatch):
    """Fresh InstructionEvolver backed by a per-test SQLite DB in tmp_path."""
    import vetinari.database as db_module

    db_path = tmp_path / "test_evolver.db"
    monkeypatch.setenv("VETINARI_DB_PATH", str(db_path))
    # Close any cached thread-local connection so the new env var takes effect
    db_module.close_connection()
    yield InstructionEvolver()
    # Cleanup after test
    db_module.close_connection()


class TestInstructionRule:
    def test_compute_hash_deterministic(self):
        rule = InstructionRule(
            rule_id="r1",
            tier="constitutional",
            content="Never harm users.",
        )
        h1 = rule.compute_hash()
        h2 = rule.compute_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_compute_hash_different_content(self):
        r1 = InstructionRule(rule_id="r1", tier="tactical", content="A")
        r2 = InstructionRule(rule_id="r2", tier="tactical", content="B")
        assert r1.compute_hash() != r2.compute_hash()

    def test_repr_shows_key_fields(self):
        rule = InstructionRule(rule_id="r99", tier="behavioral", content="x", agent_type="worker")
        r = repr(rule)
        assert "r99" in r
        assert "behavioral" in r
        assert "worker" in r


class TestInstructionTier:
    def test_all_tiers_have_values(self):
        assert InstructionTier.CONSTITUTIONAL.value == "constitutional"
        assert InstructionTier.STRUCTURAL.value == "structural"
        assert InstructionTier.BEHAVIORAL.value == "behavioral"
        assert InstructionTier.TACTICAL.value == "tactical"


class TestInstructionEvolver:
    def test_add_rule_returns_id(self, evolver):
        rule_id = evolver.add_rule(
            content="Always be helpful.",
            tier=InstructionTier.TACTICAL,
            agent_type="worker",
        )
        assert rule_id.startswith("rule_")

    def test_add_and_get_rule(self, evolver):
        evolver.add_rule("Focus on quality.", InstructionTier.BEHAVIORAL, "foreman")
        rules = evolver.get_rules(agent_type="foreman")
        assert len(rules) >= 1
        contents = [r.content for r in rules]
        assert "Focus on quality." in contents

    def test_get_rules_includes_all_scoped_rules(self, evolver):
        evolver.add_rule("Universal rule.", InstructionTier.TACTICAL, "all")
        evolver.add_rule("Worker-only rule.", InstructionTier.BEHAVIORAL, "worker")
        rules = evolver.get_rules(agent_type="worker")
        contents = [r.content for r in rules]
        assert "Universal rule." in contents
        assert "Worker-only rule." in contents

    def test_get_rules_does_not_include_other_agents(self, evolver):
        evolver.add_rule("Inspector rule.", InstructionTier.BEHAVIORAL, "inspector")
        rules = evolver.get_rules(agent_type="worker")
        contents = [r.content for r in rules]
        assert "Inspector rule." not in contents

    def test_get_rules_tier_filter(self, evolver):
        evolver.add_rule("Tactical tip.", InstructionTier.TACTICAL, "all")
        evolver.add_rule("Behavioral rule.", InstructionTier.BEHAVIORAL, "all")
        tactical = evolver.get_rules(agent_type="all", tier=InstructionTier.TACTICAL)
        assert all(r.tier == "tactical" for r in tactical)

    def test_record_trigger_increments_evidence(self, evolver):
        rule_id = evolver.add_rule("Verify output.", InstructionTier.BEHAVIORAL, "worker")
        evolver.record_trigger(rule_id, effectiveness=0.8)
        rules = evolver.get_rules(agent_type="worker")
        rule = next(r for r in rules if r.rule_id == rule_id)
        assert rule.evidence_count == 1

    def test_record_trigger_updates_effectiveness(self, evolver):
        rule_id = evolver.add_rule("Check facts.", InstructionTier.BEHAVIORAL, "worker")
        # Initial score is 0.5
        evolver.record_trigger(rule_id, effectiveness=1.0)
        rules = evolver.get_rules(agent_type="worker")
        rule = next(r for r in rules if r.rule_id == rule_id)
        # EMA: 0.5 * 0.8 + 1.0 * 0.2 = 0.6
        assert rule.effectiveness_score == pytest.approx(0.6)

    def test_transfer_rule_creates_new_rule(self, evolver):
        source_id = evolver.add_rule("Good practice.", InstructionTier.TACTICAL, "foreman")
        new_id = evolver.transfer_rule(source_id, target_agent_type="worker")
        assert new_id is not None
        assert new_id != source_id
        worker_rules = evolver.get_rules(agent_type="worker")
        contents = [r.content for r in worker_rules]
        assert "Good practice." in contents

    def test_transfer_rule_missing_source_returns_none(self, evolver):
        result = evolver.transfer_rule("nonexistent_id", target_agent_type="worker")
        assert result is None

    def test_transfer_constitutional_rule_demotes_to_behavioral(self, evolver):
        source_id = evolver.add_rule("Safety rule.", InstructionTier.CONSTITUTIONAL, "all", allow_constitutional=True)
        new_id = evolver.transfer_rule(source_id, target_agent_type="worker")
        assert new_id is not None
        worker_rules = evolver.get_rules(agent_type="worker")
        transferred = next((r for r in worker_rules if r.rule_id == new_id), None)
        assert transferred is not None
        assert transferred.tier == "behavioral"  # Demoted from constitutional

    def test_evolve_behavioral_skips_high_quality(self, evolver):
        result = evolver.evolve_behavioral("worker", quality_score=0.9)
        assert result is None

    def test_evolve_behavioral_with_trace(self, evolver):
        trace = {"output": "", "error": "", "quality_score": 0.0}
        result = evolver.evolve_behavioral("worker", quality_score=0.3, trace=trace)
        # May produce a new rule or None depending on optimizer result
        assert result is None or result.startswith("rule_")

    def test_get_stale_rules_excludes_constitutional(self, evolver):
        evolver.add_rule("Safety.", InstructionTier.CONSTITUTIONAL, "all", allow_constitutional=True)
        stale = evolver.get_stale_rules(days=0)
        # Constitutional rules should not appear in stale list
        assert all(r.tier != "constitutional" for r in stale)

    def test_get_instruction_evolver_is_not_broken(self):
        # Singleton may be initialized from previous tests; just check it's callable
        evolver_instance = get_instruction_evolver()
        assert evolver_instance is not None
        assert isinstance(evolver_instance, InstructionEvolver)
