"""Tests for CausalGraph causal root-cause analysis (item 14.5, session 4B).

Covers edge creation, graph traversal, cycle detection, and the convenience
factory functions that wrap the CausalGraph API.
"""

from __future__ import annotations

import pytest

from vetinari.validation.root_cause import (
    CausalEdge,
    CausalGraph,
    build_causal_graph,
    walk_graph_for_root_cause,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _simple_chain_graph() -> CausalGraph:
    """Return A -> B -> C (A causes B causes C)."""
    graph = CausalGraph()
    graph.add_edge("A", "B", strength=0.9)
    graph.add_edge("B", "C", strength=0.85)
    return graph


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_add_edge_creates_link() -> None:
    """add_edge stores the edge and populates the adjacency map correctly."""
    graph = CausalGraph()
    graph.add_edge("cause1", "effect1", strength=0.75, evidence="log line 42")

    assert len(graph._edges) == 1
    edge: CausalEdge = graph._edges[0]
    assert edge.cause == "cause1"
    assert edge.effect == "effect1"
    assert edge.strength == 0.75
    assert edge.evidence == "log line 42"

    # Adjacency maps effect -> [edge]
    assert "effect1" in graph._adjacency
    assert graph._adjacency["effect1"][0] is edge


def test_self_loop_rejected() -> None:
    """add_edge raises ValueError when cause and effect are the same node."""
    graph = CausalGraph()
    with pytest.raises(ValueError, match="Self-loop not allowed"):
        graph.add_edge("node_x", "node_x")


def test_build_from_failures() -> None:
    """build_from_failures converts failure dicts into graph edges."""
    failures = [
        {"id": "F1", "category": "bad_spec"},
        {"id": "F2", "category": "hallucination", "caused_by": "F1", "evidence": "spec was vague"},
        {"id": "F3", "category": "integration", "caused_by": "F2"},
    ]
    graph = CausalGraph()
    graph.build_from_failures(failures)

    assert len(graph._edges) == 2  # F1->F2 and F2->F3; F1 has no caused_by

    causes = {e.cause for e in graph._edges}
    effects = {e.effect for e in graph._edges}
    assert "F1" in causes
    assert "F2" in causes
    assert "F2" in effects
    assert "F3" in effects

    # Evidence should be propagated
    f1_to_f2 = next(e for e in graph._edges if e.cause == "F1" and e.effect == "F2")
    assert f1_to_f2.evidence == "spec was vague"


def test_walk_to_root_cause_simple_chain() -> None:
    """A -> B -> C: walking from C returns [C, B, A], root cause is A."""
    graph = _simple_chain_graph()
    path = graph.walk_to_root_cause("C")

    assert path[0] == "C", "Path must start with the symptom"
    assert path[-1] == "A", "Path must end at the root cause"
    assert path == ["C", "B", "A"]


def test_walk_to_root_cause_single_node() -> None:
    """Symptom with no known cause returns a path containing only itself."""
    graph = CausalGraph()
    graph.add_edge("ROOT", "LEAF")
    path = graph.walk_to_root_cause("ROOT")
    assert path == ["ROOT"]


def test_walk_to_root_cause_branching() -> None:
    """A and B both cause C: walk from C reaches one of them as root cause."""
    graph = CausalGraph()
    # A causes C with higher strength, B causes C with lower strength
    graph.add_edge("A", "C", strength=0.9)
    graph.add_edge("B", "C", strength=0.6)

    path = graph.walk_to_root_cause("C")
    # The highest-strength edge is followed: A -> C, so root is A
    assert path[0] == "C"
    assert path[-1] == "A"

    # get_all_paths should find both chains
    all_paths = graph.get_all_paths("C")
    roots_found = {p[-1] for p in all_paths}
    assert roots_found == {"A", "B"}


def test_get_root_causes() -> None:
    """get_root_causes returns nodes that cause things but are never caused."""
    graph = _simple_chain_graph()  # A -> B -> C
    roots = graph.get_root_causes()
    assert roots == ["A"], f"Expected ['A'], got {roots}"


def test_get_root_causes_multiple() -> None:
    """Two independent root causes are both returned."""
    graph = CausalGraph()
    graph.add_edge("ROOT1", "MID")
    graph.add_edge("ROOT2", "MID")
    graph.add_edge("MID", "LEAF")

    roots = graph.get_root_causes()
    assert set(roots) == {"ROOT1", "ROOT2"}


def test_walk_graph_for_root_cause_convenience() -> None:
    """walk_graph_for_root_cause returns the deepest root-cause node id."""
    graph = _simple_chain_graph()  # A -> B -> C
    result = walk_graph_for_root_cause(graph, "C")
    assert result == "A"


def test_walk_graph_for_root_cause_not_in_graph() -> None:
    """walk_graph_for_root_cause returns None when the symptom is unknown."""
    graph = _simple_chain_graph()
    result = walk_graph_for_root_cause(graph, "UNKNOWN_NODE")
    assert result is None


def test_cycle_detection() -> None:
    """walk_to_root_cause stops cleanly when a cycle is present instead of looping."""
    graph = CausalGraph()
    # Manually add a cycle by bypassing validation (to test runtime guard)
    from vetinari.validation.root_cause import CausalEdge

    # A -> B -> A (cycle)
    e1 = CausalEdge(cause="A", effect="B")
    e2 = CausalEdge(cause="B", effect="A")
    graph._edges.extend([e1, e2])
    graph._adjacency.setdefault("B", []).append(e1)
    graph._adjacency.setdefault("A", []).append(e2)

    # Must terminate, not loop forever
    path = graph.walk_to_root_cause("A")
    assert len(path) >= 1
    assert path[0] == "A"


def test_empty_graph() -> None:
    """Operations on an empty graph return safe empty results."""
    graph = CausalGraph()

    assert graph.get_root_causes() == []
    assert graph.walk_to_root_cause("anything") == ["anything"]
    assert graph.get_all_paths("anything") == [["anything"]]
    assert walk_graph_for_root_cause(graph, "anything") is None


def test_build_causal_graph_factory() -> None:
    """build_causal_graph factory returns a populated CausalGraph."""
    failures = [
        {"id": "X", "category": "bad_spec"},
        {"id": "Y", "category": "hallucination", "caused_by": "X"},
    ]
    graph = build_causal_graph(failures)
    assert isinstance(graph, CausalGraph)
    assert len(graph._edges) == 1
    assert graph._edges[0].cause == "X"
    assert graph._edges[0].effect == "Y"
