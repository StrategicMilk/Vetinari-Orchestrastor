"""Tests for vetinari.planning.decision_tree.

Verifies decision extraction from goals, auto-resolution behavior,
context enrichment, and the LLM/keyword fallback cascade.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from vetinari.planning.decision_tree import (
    DecisionNode,
    DecisionTreeResult,
    Option,
    auto_resolve_decisions,
    enrich_context_with_decisions,
    extract_decisions,
    get_resolved_context,
)

# -- Option / DecisionNode unit tests ----------------------------------------


class TestDataclasses:
    """Tests for Option and DecisionNode data structures."""

    def test_option_frozen(self) -> None:
        """Option must be immutable (frozen dataclass)."""
        opt = Option(name="React", description="UI lib", trade_offs="Large")
        with pytest.raises((AttributeError, TypeError)):
            opt.name = "Vue"  # type: ignore[misc]

    def test_option_default_recommended_false(self) -> None:
        """Option.recommended must default to False when not specified."""
        opt = Option(name="X", description="D", trade_offs="T")
        assert opt.recommended is False

    def test_decision_node_repr(self) -> None:
        """DecisionNode repr must show domain and resolution status."""
        node = DecisionNode(question="Which DB?", domain="database")
        assert "UNRESOLVED" in repr(node)
        assert "database" in repr(node)

    def test_decision_node_repr_resolved(self) -> None:
        """DecisionNode repr must show resolution when resolved."""
        node = DecisionNode(question="Which DB?", domain="database", resolution="SQLite")
        assert "SQLite" in repr(node)

    def test_decision_tree_result_frozen(self) -> None:
        """DecisionTreeResult must be immutable (frozen dataclass)."""
        result = DecisionTreeResult(
            decisions=(),
            all_resolved=True,
            resolved_context={},
            blocking_decisions=(),
        )
        with pytest.raises((AttributeError, TypeError)):
            result.all_resolved = False  # type: ignore[misc]


# -- Keyword extraction tests ------------------------------------------------


class TestKeywordExtraction:
    """Tests for the keyword-based fallback extraction path."""

    def test_goal_produces_decision_tree(self) -> None:
        """A goal mentioning 'web app' and 'database' must produce frontend + database decisions."""
        result = extract_decisions("Build a web app with a database for user management")

        assert isinstance(result, DecisionTreeResult)
        assert len(result.decisions) >= 2

        domains = {d.domain for d in result.decisions}
        assert "frontend" in domains
        assert "database" in domains

    def test_frontend_goal_produces_frontend_decision(self) -> None:
        """A goal mentioning 'dashboard' must produce a frontend decision."""
        result = extract_decisions("Create a dashboard for monitoring")
        domains = {d.domain for d in result.decisions}
        assert "frontend" in domains

    def test_auth_goal_produces_auth_decision(self) -> None:
        """A goal mentioning 'login' must produce an auth decision."""
        result = extract_decisions("Add a login page with user sessions")
        domains = {d.domain for d in result.decisions}
        assert "auth" in domains

    def test_deployment_goal_produces_deployment_decision(self) -> None:
        """A goal mentioning 'deploy' must produce a deployment decision."""
        result = extract_decisions("Deploy the application to the cloud")
        domains = {d.domain for d in result.decisions}
        assert "deployment" in domains

    def test_no_domain_keywords_produces_empty(self) -> None:
        """A goal with no domain keywords must produce an empty decision tree."""
        result = extract_decisions("Refactor the sorting algorithm")
        assert len(result.decisions) == 0
        assert result.all_resolved is True

    def test_decisions_have_options(self) -> None:
        """Keyword-extracted decisions for known domains must include pre-built options."""
        result = extract_decisions("Build a web app with a database")
        for decision in result.decisions:
            if decision.domain in ("frontend", "database"):
                assert len(decision.options) >= 2, f"Domain {decision.domain} has too few options"

    def test_options_have_recommended(self) -> None:
        """At least one option per known domain must be flagged as recommended."""
        result = extract_decisions("Build a web app")
        for decision in result.decisions:
            if decision.domain == "frontend":
                recommended = [o for o in decision.options if o.recommended]
                assert len(recommended) >= 1


# -- Auto-resolution tests ---------------------------------------------------


class TestAutoResolution:
    """Tests for auto_resolve_decisions()."""

    def test_auto_resolved_decisions_flow_to_dag(self) -> None:
        """High-confidence decisions with care_level='auto' must be resolved automatically."""
        decisions = [
            DecisionNode(
                question="Which DB?",
                domain="database",
                options=[
                    Option(name="SQLite", description="Embedded", trade_offs="Simple", recommended=True),
                    Option(name="PostgreSQL", description="Full", trade_offs="Complex"),
                ],
                confidence=0.9,
            ),
        ]

        resolved = auto_resolve_decisions(decisions, care_levels={})

        assert resolved[0].resolution == "SQLite"
        assert resolved[0].auto_resolved is True

    def test_low_confidence_not_auto_resolved(self) -> None:
        """Decisions with confidence below threshold must remain unresolved."""
        decisions = [
            DecisionNode(
                question="Which API style?",
                domain="api",
                options=[
                    Option(name="REST", description="RESTful", trade_offs="Standard", recommended=True),
                ],
                confidence=0.5,
            ),
        ]

        resolved = auto_resolve_decisions(decisions, care_levels={})

        assert resolved[0].resolution is None
        assert resolved[0].auto_resolved is False

    def test_unresolved_decisions_block_generation(self) -> None:
        """Decisions with care_level='review' must remain unresolved and appear as blocking."""
        result = extract_decisions(
            "Build a web app with a database",
            domain_care_levels={"frontend": "review", "database": "review"},
        )

        assert result.all_resolved is False
        assert len(result.blocking_decisions) >= 1
        blocking_domains = {d.domain for d in result.blocking_decisions}
        assert "frontend" in blocking_domains or "database" in blocking_domains

    def test_manual_care_level_blocks(self) -> None:
        """Decisions with care_level='manual' must also block decomposition."""
        decisions = [
            DecisionNode(
                question="Which framework?",
                domain="frontend",
                options=[
                    Option(name="React", description="UI lib", trade_offs="Large", recommended=True),
                ],
                confidence=0.95,
            ),
        ]

        resolved = auto_resolve_decisions(decisions, care_levels={"frontend": "manual"})

        assert resolved[0].resolution is None

    def test_already_resolved_not_overwritten(self) -> None:
        """A decision that already has a resolution must not be changed by auto_resolve."""
        decisions = [
            DecisionNode(
                question="Which DB?",
                domain="database",
                options=[
                    Option(name="SQLite", description="Embedded", trade_offs="Simple", recommended=True),
                ],
                confidence=0.95,
                resolution="PostgreSQL",
            ),
        ]

        resolved = auto_resolve_decisions(decisions, care_levels={})

        assert resolved[0].resolution == "PostgreSQL"  # Unchanged

    def test_no_recommended_option_skips_auto_resolve(self) -> None:
        """When no option is marked recommended, auto-resolve should not pick one."""
        decisions = [
            DecisionNode(
                question="Which testing framework?",
                domain="testing",
                options=[
                    Option(name="pytest", description="Test runner", trade_offs="Flexible"),
                    Option(name="unittest", description="Built-in", trade_offs="Verbose"),
                ],
                confidence=0.9,
            ),
        ]

        resolved = auto_resolve_decisions(decisions, care_levels={})

        assert resolved[0].resolution is None


# -- Context helpers ----------------------------------------------------------


class TestResolvedContext:
    """Tests for get_resolved_context()."""

    def test_resolved_context_maps_domain_to_resolution(self) -> None:
        """get_resolved_context must return {domain: resolution} for resolved decisions only."""
        decisions = [
            DecisionNode(question="Q1", domain="frontend", resolution="React"),
            DecisionNode(question="Q2", domain="database"),  # unresolved
            DecisionNode(question="Q3", domain="auth", resolution="JWT"),
        ]

        ctx = get_resolved_context(decisions)

        assert ctx == {"frontend": "React", "auth": "JWT"}

    def test_empty_decisions_returns_empty(self) -> None:
        """Empty decision list must produce empty context."""
        assert get_resolved_context([]) == {}


# -- Context enrichment -------------------------------------------------------


class TestContextEnrichment:
    """Tests for enrich_context_with_decisions()."""

    def test_enrichment_adds_decision_keys(self) -> None:
        """enrich_context_with_decisions must add _decisions and _resolved_decisions keys."""
        enriched = enrich_context_with_decisions(
            "Build a web app with a database",
            {"existing_key": "value"},
        )

        assert "_decisions" in enriched
        assert "_resolved_decisions" in enriched
        assert enriched["existing_key"] == "value"  # Original preserved

    def test_enrichment_does_not_modify_original(self) -> None:
        """The original context dict must not be modified in place."""
        original = {"key": "value"}
        enrich_context_with_decisions("Build a web app", original)
        assert "_decisions" not in original

    def test_blocking_decisions_added_when_present(self) -> None:
        """When decisions are blocking, _blocking_decisions must appear in enriched context."""
        enriched = enrich_context_with_decisions(
            "Build a web app with a database",
            {},
            domain_care_levels={"frontend": "manual", "database": "manual"},
        )

        assert "_blocking_decisions" in enriched
        assert len(enriched["_blocking_decisions"]) >= 1

    def test_no_blocking_when_all_auto_resolved(self) -> None:
        """When all decisions auto-resolve, _blocking_decisions must not appear."""
        # "Refactor sorting" produces no decisions -> no blocking
        enriched = enrich_context_with_decisions("Refactor the sorting algorithm", {})
        assert "_blocking_decisions" not in enriched


# -- LLM extraction path (mocked) --------------------------------------------


class TestLLMExtraction:
    """Tests for the LLM-based extraction path via cascade router."""

    def test_llm_extraction_used_when_available(self) -> None:
        """When the cascade router is available and returns valid JSON, use LLM results."""
        mock_router = MagicMock()
        mock_router.route.return_value = (
            '[{"question": "Which DB?", "domain": "database", '
            '"options": [{"name": "SQLite", "description": "Embedded", '
            '"trade_offs": "Simple", "recommended": true}], '
            '"confidence": 0.85}]'
        )

        # Inject a fake cascade_router module into sys.modules
        fake_module = ModuleType("vetinari.inference.cascade_router")
        fake_module.get_cascade_router = MagicMock(return_value=mock_router)  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"vetinari.inference.cascade_router": fake_module}):
            result = extract_decisions("Build an app with a database")

        assert len(result.decisions) >= 1
        db_decisions = [d for d in result.decisions if d.domain == "database"]
        assert len(db_decisions) == 1
        assert db_decisions[0].question == "Which DB?"

    def test_llm_unavailable_falls_back_to_keywords(self) -> None:
        """When cascade router import fails, keyword extraction must be used."""
        # The module doesn't exist, so the local import inside
        # _extract_decisions_llm will fail and trigger keyword fallback
        result = extract_decisions("Build a web app with a database")

        # Should still get results from keyword extraction
        assert len(result.decisions) >= 1

    def test_llm_returns_invalid_json_falls_back(self) -> None:
        """When LLM returns unparseable JSON, keyword extraction must be used as fallback."""
        mock_router = MagicMock()
        mock_router.route.return_value = "this is not valid json at all"

        fake_module = ModuleType("vetinari.inference.cascade_router")
        fake_module.get_cascade_router = MagicMock(return_value=mock_router)  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"vetinari.inference.cascade_router": fake_module}):
            result = extract_decisions("Build a web app with a database")

        # Keyword fallback should still produce results
        assert len(result.decisions) >= 1


# -- Module wiring tests -----------------------------------------------------


class TestModuleWiring:
    """Verify the module is importable and exports are correct."""

    def test_imports_from_planning_package(self) -> None:
        """Key types must be importable from vetinari.planning.decision_tree."""
        from vetinari.planning.decision_tree import (
            DecisionNode,
            DecisionTreeResult,
            Option,
            auto_resolve_decisions,
            enrich_context_with_decisions,
            extract_decisions,
            get_resolved_context,
        )

        assert DecisionNode is not None
        assert DecisionTreeResult is not None
        assert Option is not None
        assert auto_resolve_decisions is not None
        assert enrich_context_with_decisions is not None
        assert extract_decisions is not None
        assert get_resolved_context is not None

    def test_imports_from_planning_init(self) -> None:
        """Key types must be importable from vetinari.planning."""
        from vetinari.planning import (
            DecisionNode,
            DecisionTreeResult,
            Option,
            extract_decisions,
        )

        assert DecisionNode is not None
        assert DecisionTreeResult is not None
        assert Option is not None
        assert extract_decisions is not None
