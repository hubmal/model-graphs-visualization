"""Tests for the intermediate graph representation (graph_viewer.graph).

API these tests expect:

    Shape = tuple[int | None, ...] | None
        # None inside the tuple -> dynamic dimension, e.g. (None, 2048) for [?, 2048]
        # None as the whole shape -> unknown rank / no metadata
        # ()                      -> scalar

    Node: node_id: int, name: str, op_type: str,
          attributes: dict[str, Any], viz: dict[str, Any]   # both default to {}
    Edge: edge_id: int, src: int, src_port: int, dst: int, dst_port: int,
          dtype: str | None, shape: Shape

    Graph()
        add_node(name, op_type, attributes=None) -> Node   # assigns a new unique node_id
        add_edge(src, src_port, dst, dst_port, dtype=None, shape=None) -> Edge
        remove_edge(edge_id) -> None
        remove_node(node_id) -> None                       # also removes its edges
        get_node(node_id) -> Node                          # KeyError if missing
        get_edge(edge_id) -> Edge                          # KeyError if missing
        nodes() -> Iterable[Node]
        edges() -> Iterable[Edge]
        num_nodes() -> int
        num_edges() -> int
        in_edges(node_id) -> list[Edge]    # sorted by dst_port
        out_edges(node_id) -> list[Edge]   # sorted by src_port, then edge_id
"""

import subprocess
import sys

import pytest

from graph_checks import assert_graph_consistent
from graph_viewer.graph import Graph


@pytest.fixture
def graph() -> Graph:
    return Graph()


# --- nodes -----------------------------------------------------------------


def test_add_node_assigns_unique_ids(graph: Graph) -> None:
    a = graph.add_node("a", "Parameter")
    b = graph.add_node("b", "Relu")

    assert a.node_id != b.node_id
    assert graph.get_node(a.node_id) is a
    assert graph.num_nodes() == 2


def test_node_stores_name_type_and_attributes(graph: Graph) -> None:
    node = graph.add_node("mm", "MatMul", {"transpose_b": True})

    assert node.name == "mm"
    assert node.op_type == "MatMul"
    assert node.attributes == {"transpose_b": True}
    assert node.viz == {}


def test_duplicate_names_are_separate_nodes(graph: Graph) -> None:
    a = graph.add_node("same", "Add")
    b = graph.add_node("same", "Add")

    assert a.node_id != b.node_id
    assert graph.num_nodes() == 2


def test_default_dicts_are_not_shared_between_nodes(graph: Graph) -> None:
    a = graph.add_node("a", "Add")
    b = graph.add_node("b", "Add")

    a.attributes["x"] = 1
    a.viz["color"] = "red"

    assert b.attributes == {}
    assert b.viz == {}


def test_node_ids_are_not_reused_after_removal(graph: Graph) -> None:
    a = graph.add_node("a", "Add")
    graph.remove_node(a.node_id)
    b = graph.add_node("b", "Add")

    assert b.node_id != a.node_id


def test_get_missing_node_raises_key_error(graph: Graph) -> None:
    with pytest.raises(KeyError):
        graph.get_node(12345)


# --- edges -----------------------------------------------------------------


def test_edge_keeps_ports_dtype_and_shape(graph: Graph) -> None:
    src = graph.add_node("src", "Parameter")
    dst = graph.add_node("dst", "MatMul")

    edge = graph.add_edge(src.node_id, 0, dst.node_id, 1, dtype="f32", shape=(1, 768))

    assert (edge.src, edge.src_port, edge.dst, edge.dst_port) == (src.node_id, 0, dst.node_id, 1)
    assert edge.dtype == "f32"
    assert edge.shape == (1, 768)
    assert graph.get_edge(edge.edge_id) is edge


def test_edge_metadata_is_optional(graph: Graph) -> None:
    src = graph.add_node("src", "Parameter")
    dst = graph.add_node("dst", "Relu")

    edge = graph.add_edge(src.node_id, 0, dst.node_id, 0)

    assert edge.dtype is None
    assert edge.shape is None


@pytest.mark.parametrize(
    "shape",
    [
        (None, 2048),  # partially dynamic: [?, 2048]
        (None, None),  # all dimensions dynamic: [?, ?]
        None,          # unknown rank
        (),            # scalar, must stay different from None
    ],
)
def test_dynamic_and_special_shapes_are_preserved(graph: Graph, shape: tuple | None) -> None:
    src = graph.add_node("src", "Parameter")
    dst = graph.add_node("dst", "Relu")

    edge = graph.add_edge(src.node_id, 0, dst.node_id, 0, dtype="f32", shape=shape)

    assert edge.shape == shape


def test_in_and_out_edges(graph: Graph) -> None:
    x = graph.add_node("x", "Parameter")
    w = graph.add_node("w", "Constant")
    mm = graph.add_node("mm", "MatMul")
    out = graph.add_node("out", "Result")
    e_x = graph.add_edge(x.node_id, 0, mm.node_id, 0)
    e_w = graph.add_edge(w.node_id, 0, mm.node_id, 1)
    e_out = graph.add_edge(mm.node_id, 0, out.node_id, 0)

    assert graph.in_edges(mm.node_id) == [e_x, e_w]
    assert graph.out_edges(mm.node_id) == [e_out]
    assert graph.in_edges(x.node_id) == []
    assert graph.out_edges(out.node_id) == []
    assert_graph_consistent(graph)


def test_in_edges_are_sorted_by_destination_port(graph: Graph) -> None:
    a = graph.add_node("a", "Parameter")
    b = graph.add_node("b", "Parameter")
    concat = graph.add_node("concat", "Concat")
    # Added in reverse port order on purpose.
    e1 = graph.add_edge(b.node_id, 0, concat.node_id, 1)
    e0 = graph.add_edge(a.node_id, 0, concat.node_id, 0)

    assert graph.in_edges(concat.node_id) == [e0, e1]


def test_multiple_edges_between_same_pair_of_nodes(graph: Graph) -> None:
    """Split out 0 -> Add in 0 and Split out 1 -> Add in 1 are two distinct edges."""
    split = graph.add_node("split", "Split")
    add = graph.add_node("add", "Add")

    e0 = graph.add_edge(split.node_id, 0, add.node_id, 0)
    e1 = graph.add_edge(split.node_id, 1, add.node_id, 1)

    assert e0.edge_id != e1.edge_id
    assert graph.out_edges(split.node_id) == [e0, e1]
    assert graph.in_edges(add.node_id) == [e0, e1]
    assert graph.num_edges() == 2


def test_one_output_can_feed_several_consumers(graph: Graph) -> None:
    w = graph.add_node("w", "Constant")
    mm1 = graph.add_node("mm1", "MatMul")
    mm2 = graph.add_node("mm2", "MatMul")

    graph.add_edge(w.node_id, 0, mm1.node_id, 1)
    graph.add_edge(w.node_id, 0, mm2.node_id, 1)

    assert [e.dst for e in graph.out_edges(w.node_id)] == [mm1.node_id, mm2.node_id]


def test_input_port_accepts_only_one_producer(graph: Graph) -> None:
    """In a dataflow graph each input port has exactly one source."""
    a = graph.add_node("a", "Parameter")
    b = graph.add_node("b", "Parameter")
    relu = graph.add_node("relu", "Relu")
    graph.add_edge(a.node_id, 0, relu.node_id, 0)

    with pytest.raises(ValueError):
        graph.add_edge(b.node_id, 0, relu.node_id, 0)


def test_add_edge_to_missing_node_raises_key_error(graph: Graph) -> None:
    a = graph.add_node("a", "Parameter")

    with pytest.raises(KeyError):
        graph.add_edge(a.node_id, 0, 999, 0)
    with pytest.raises(KeyError):
        graph.add_edge(999, 0, a.node_id, 0)
    assert graph.num_edges() == 0


def test_negative_port_raises_value_error(graph: Graph) -> None:
    a = graph.add_node("a", "Parameter")
    b = graph.add_node("b", "Relu")

    with pytest.raises(ValueError):
        graph.add_edge(a.node_id, -1, b.node_id, 0)


# --- modification ----------------------------------------------------------


def test_remove_edge_updates_both_endpoints(graph: Graph) -> None:
    a = graph.add_node("a", "Parameter")
    b = graph.add_node("b", "Relu")
    edge = graph.add_edge(a.node_id, 0, b.node_id, 0)

    graph.remove_edge(edge.edge_id)

    assert graph.out_edges(a.node_id) == []
    assert graph.in_edges(b.node_id) == []
    assert graph.num_edges() == 0
    with pytest.raises(KeyError):
        graph.get_edge(edge.edge_id)


def test_removed_input_port_can_be_reconnected(graph: Graph) -> None:
    a = graph.add_node("a", "Parameter")
    b = graph.add_node("b", "Parameter")
    relu = graph.add_node("relu", "Relu")
    old = graph.add_edge(a.node_id, 0, relu.node_id, 0)

    graph.remove_edge(old.edge_id)
    new = graph.add_edge(b.node_id, 0, relu.node_id, 0)

    assert graph.in_edges(relu.node_id) == [new]


def test_remove_node_removes_its_edges_and_keeps_the_rest(graph: Graph) -> None:
    x = graph.add_node("x", "Parameter")
    w = graph.add_node("w", "Constant")
    mm = graph.add_node("mm", "MatMul")
    out = graph.add_node("out", "Result")
    graph.add_edge(x.node_id, 0, mm.node_id, 0)
    graph.add_edge(w.node_id, 0, mm.node_id, 1)
    graph.add_edge(mm.node_id, 0, out.node_id, 0)
    unrelated = graph.add_edge(x.node_id, 0, out.node_id, 1)

    graph.remove_node(mm.node_id)

    assert graph.num_nodes() == 3
    assert graph.num_edges() == 1
    assert graph.out_edges(x.node_id) == [unrelated]
    assert graph.out_edges(w.node_id) == []
    assert graph.in_edges(out.node_id) == [unrelated]
    with pytest.raises(KeyError):
        graph.get_node(mm.node_id)
    assert_graph_consistent(graph)


def test_remove_node_with_self_loop(graph: Graph) -> None:
    """The edge is both incoming and outgoing; it must be removed exactly once."""
    node = graph.add_node("loop", "Add")
    graph.add_edge(node.node_id, 0, node.node_id, 1)

    graph.remove_node(node.node_id)

    assert graph.num_nodes() == 0
    assert graph.num_edges() == 0


def test_edges_cannot_be_rewired_in_place(graph: Graph) -> None:
    """Changing edge.dst directly would desynchronize the graph's indexes."""
    a = graph.add_node("a", "Parameter")
    b = graph.add_node("b", "Relu")
    edge = graph.add_edge(a.node_id, 0, b.node_id, 0)

    with pytest.raises(AttributeError):
        edge.dst = a.node_id  # type: ignore[misc]


def test_remove_missing_node_or_edge_raises_key_error(graph: Graph) -> None:
    with pytest.raises(KeyError):
        graph.remove_node(999)
    with pytest.raises(KeyError):
        graph.remove_edge(999)


def test_large_graph_modifications_stay_consistent(graph: Graph) -> None:
    """20k-node chain. With O(N) work per removal this test takes far longer than a second."""
    ids = [graph.add_node(f"n{i}", "Relu").node_id for i in range(20_000)]
    for src, dst in zip(ids, ids[1:]):
        graph.add_edge(src, 0, dst, 0)

    for node_id in ids[::2]:
        graph.remove_node(node_id)

    assert graph.num_nodes() == 10_000
    assert graph.num_edges() == 0
    assert_graph_consistent(graph)


# --- architecture ----------------------------------------------------------


def test_graph_module_does_not_import_frameworks() -> None:
    """The IR must stay independent of OpenVINO and Graphviz."""
    code = (
        "import sys, graph_viewer.graph; "
        "bad = {'openvino', 'graphviz'} & set(sys.modules); "
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
