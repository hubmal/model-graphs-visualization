"""Shared assertions for tests that build or modify a Graph."""

from graph_viewer.graph import Graph


def assert_graph_consistent(graph: Graph) -> None:
    """Check that the per-node edge indexes agree with the edge set."""
    edges = list(graph.edges())
    node_ids = {node.node_id for node in graph.nodes()}

    assert len(edges) == graph.num_edges()
    assert len(node_ids) == graph.num_nodes()

    for edge in edges:
        assert edge.src in node_ids, f"edge {edge.edge_id} starts at missing node {edge.src}"
        assert edge.dst in node_ids, f"edge {edge.edge_id} ends at missing node {edge.dst}"
        assert edge in graph.out_edges(edge.src)
        assert edge in graph.in_edges(edge.dst)

    # Every indexed edge must still exist (no stale ids left behind after removals).
    assert sum(len(graph.in_edges(n)) for n in node_ids) == len(edges)
    assert sum(len(graph.out_edges(n)) for n in node_ids) == len(edges)
