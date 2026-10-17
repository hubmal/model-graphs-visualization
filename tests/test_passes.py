"""Tests for transformation passes, the registry and the pipeline (synthetic graphs only)."""

import logging

import pytest

from graph_checks import assert_graph_consistent
from graph_viewer.graph import Graph, Node
from graph_viewer.passes import registry
from graph_viewer.passes import (
    FoldConstantConvertsPass,
    HideConstantsPass,
    HighlightPass,
    PassConfigError,
    Pipeline,
    TransformationPass,
    available_passes,
    create_pass,
    register_pass,
)


def linear_graph() -> tuple[Graph, dict[str, Node]]:
    """x -> MatMul(w) -> Add(b) -> out, with dtypes and shapes on every edge."""
    g = Graph()
    n = {
        "x": g.add_node("x", "Parameter"),
        "w": g.add_node("w", "Constant"),
        "mm": g.add_node("mm", "MatMul", {"transpose_b": True}),
        "b": g.add_node("b", "Constant"),
        "add": g.add_node("add", "Add"),
        "out": g.add_node("out", "Result"),
    }
    g.add_edge(n["x"].node_id, 0, n["mm"].node_id, 0, "f32", (None, 768))
    g.add_edge(n["w"].node_id, 0, n["mm"].node_id, 1, "f32", (1024, 768))
    g.add_edge(n["mm"].node_id, 0, n["add"].node_id, 0, "f32", (None, 1024))
    g.add_edge(n["b"].node_id, 0, n["add"].node_id, 1, "f32", (1024,))
    g.add_edge(n["add"].node_id, 0, n["out"].node_id, 0, "f32", (None, 1024))
    return g, n


# --- highlight -------------------------------------------------------------


def test_highlight_marks_only_matching_nodes() -> None:
    g, n = linear_graph()

    HighlightPass(["mm", "out"]).apply(g)

    assert n["mm"].viz == {"border_color": "red"}
    assert n["out"].viz == {"border_color": "red"}
    for name in ("x", "w", "b", "add"):
        assert n[name].viz == {}


def test_highlight_marks_every_node_with_a_duplicate_name() -> None:
    g = Graph()
    a = g.add_node("same", "Add")
    b = g.add_node("same", "Add")
    other = g.add_node("other", "Add")

    HighlightPass(["same"]).apply(g)

    assert a.viz["border_color"] == b.viz["border_color"] == "red"
    assert "border_color" not in other.viz


def test_highlight_does_not_change_structure_or_attributes() -> None:
    g, n = linear_graph()
    edges_before = g.edges()

    HighlightPass(["mm"]).apply(g)

    assert g.edges() == edges_before
    assert n["mm"].attributes == {"transpose_b": True}


def test_highlight_custom_color_and_missing_name_warning(caplog: pytest.LogCaptureFixture) -> None:
    g, n = linear_graph()

    with caplog.at_level(logging.WARNING):
        HighlightPass(["add", "does-not-exist"], color="blue").apply(g)

    assert n["add"].viz["border_color"] == "blue"
    assert "does-not-exist" in caplog.text


# --- hide constants --------------------------------------------------------


def test_hide_constants_single_consumer() -> None:
    g, n = linear_graph()

    HideConstantsPass().apply(g)

    assert n["mm"].attributes == {"transpose_b": True, "[1]": "f32[1024, 768]"}
    assert n["add"].attributes == {"[1]": "f32[1024]"}
    assert {node.op_type for node in g.nodes()} == {"Parameter", "MatMul", "Add", "Result"}
    assert_graph_consistent(g)


def test_hide_constants_preserves_unrelated_edges() -> None:
    g, n = linear_graph()

    HideConstantsPass().apply(g)

    connections = sorted((e.src, e.src_port, e.dst, e.dst_port) for e in g.edges())
    assert connections == sorted([
        (n["x"].node_id, 0, n["mm"].node_id, 0),
        (n["mm"].node_id, 0, n["add"].node_id, 0),
        (n["add"].node_id, 0, n["out"].node_id, 0),
    ])
    # Remaining edges keep their tensor metadata.
    assert g.in_edges(n["mm"].node_id)[0].shape == (None, 768)


def test_hide_constants_shared_constant_marks_every_consumer_with_its_id() -> None:
    g = Graph()
    w = g.add_node("w", "Constant")
    mm1 = g.add_node("mm1", "MatMul")
    mm2 = g.add_node("mm2", "MatMul")
    g.add_edge(w.node_id, 0, mm1.node_id, 1, "f16", (2048, 2048))
    g.add_edge(w.node_id, 0, mm2.node_id, 0, "f16", (2048, 2048))

    HideConstantsPass().apply(g)

    assert mm1.attributes == {"[1]": f"f16[2048, 2048] (shared: {w.node_id})"}
    assert mm2.attributes == {"[0]": f"f16[2048, 2048] (shared: {w.node_id})"}
    assert g.num_nodes() == 2
    assert g.num_edges() == 0


def test_hide_constants_attributes_are_sorted_by_port() -> None:
    g = Graph()
    c1 = g.add_node("c1", "Constant")  # visited first, but feeds port 2
    c0 = g.add_node("c0", "Constant")
    op = g.add_node("op", "Select")
    g.add_edge(c1.node_id, 0, op.node_id, 2, "f32", (1,))
    g.add_edge(c0.node_id, 0, op.node_id, 0, "boolean", (1,))

    HideConstantsPass().apply(g)

    assert list(op.attributes) == ["[0]", "[2]"]


@pytest.mark.parametrize(
    ("dtype", "shape", "expected"),
    [
        ("f32", (None, 4), "f32[?, 4]"),
        ("i64", (), "i64[]"),
        (None, None, "?[...]"),
    ],
)
def test_hide_constants_handles_dynamic_and_missing_metadata(
    dtype: str | None, shape: tuple | None, expected: str
) -> None:
    g = Graph()
    c = g.add_node("c", "Constant")
    op = g.add_node("op", "Reshape")
    g.add_edge(c.node_id, 0, op.node_id, 1, dtype, shape)

    HideConstantsPass().apply(g)

    assert op.attributes["[1]"] == expected


def test_hide_constants_keeps_constant_without_consumers() -> None:
    g = Graph()
    c = g.add_node("lonely", "Constant")

    HideConstantsPass().apply(g)

    assert g.get_node(c.node_id).name == "lonely"


def test_hide_constants_never_copies_tensor_values() -> None:
    g = Graph()
    c = g.add_node("c", "Constant", {"element_type": "f32", "shape": [2], "value": [1.5, 2.5]})
    op = g.add_node("op", "Add")
    g.add_edge(c.node_id, 0, op.node_id, 1, "f32", (2,))

    HideConstantsPass().apply(g)

    assert op.attributes == {"[1]": "f32[2]"}


# --- fold converts ---------------------------------------------------------


def test_fold_converts_connects_constant_directly_to_consumers() -> None:
    g = Graph()
    w = g.add_node("w", "Constant")
    cvt = g.add_node("cvt", "Convert")
    mm1 = g.add_node("mm1", "MatMul")
    mm2 = g.add_node("mm2", "MatMul")
    g.add_edge(w.node_id, 0, cvt.node_id, 0, "f16", (8, 8))
    g.add_edge(cvt.node_id, 0, mm1.node_id, 1, "f32", (8, 8))
    g.add_edge(cvt.node_id, 0, mm2.node_id, 1, "f32", (8, 8))

    FoldConstantConvertsPass().apply(g)

    with pytest.raises(KeyError):
        g.get_node(cvt.node_id)
    out = g.out_edges(w.node_id)
    assert [(e.dst, e.dst_port) for e in out] == [(mm1.node_id, 1), (mm2.node_id, 1)]
    assert all(e.dtype == "f16" and e.shape == (8, 8) for e in out)  # stored weight type
    assert_graph_consistent(g)


def test_fold_converts_keeps_converts_of_computed_values() -> None:
    g = Graph()
    x = g.add_node("x", "Parameter")
    cvt = g.add_node("cvt", "Convert")
    g.add_edge(x.node_id, 0, cvt.node_id, 0, "i64", (4,))

    FoldConstantConvertsPass().apply(g)

    assert g.get_node(cvt.node_id).op_type == "Convert"
    assert g.num_edges() == 1


def test_fold_converts_then_hide_constants_hides_compressed_weights() -> None:
    g = Graph()
    x = g.add_node("x", "Parameter")
    w = g.add_node("w", "Constant")
    cvt = g.add_node("cvt", "Convert")
    mm = g.add_node("mm", "MatMul")
    g.add_edge(x.node_id, 0, mm.node_id, 0, "f32", (1, 8))
    g.add_edge(w.node_id, 0, cvt.node_id, 0, "f16", (8, 8))
    g.add_edge(cvt.node_id, 0, mm.node_id, 1, "f32", (8, 8))

    Pipeline([FoldConstantConvertsPass(), HideConstantsPass()]).run(g)

    assert mm.attributes == {"[1]": "f16[8, 8]"}
    assert [n.op_type for n in g.nodes()] == ["Parameter", "MatMul"]


# --- pipeline --------------------------------------------------------------


class RecordingPass(TransformationPass):
    name = "recording"

    def __init__(self, label: str, log: list[str]) -> None:
        self.label = label
        self.log = log

    def apply(self, graph: Graph) -> Graph:
        self.log.append(self.label)
        return graph


def test_pipeline_runs_passes_in_given_order() -> None:
    log: list[str] = []

    Pipeline([RecordingPass("first", log), RecordingPass("second", log), RecordingPass("third", log)]).run(Graph())

    assert log == ["first", "second", "third"]


def test_pipeline_order_matters_and_result_stays_valid() -> None:
    def build() -> tuple[Graph, Node]:
        g = Graph()
        w = g.add_node("w", "Constant")
        cvt = g.add_node("cvt", "Convert")
        mm = g.add_node("mm", "MatMul")
        g.add_edge(w.node_id, 0, cvt.node_id, 0, "f16", (8, 8))
        g.add_edge(cvt.node_id, 0, mm.node_id, 1, "f32", (8, 8))
        return g, mm

    folded_first, mm_a = build()
    Pipeline([FoldConstantConvertsPass(), HideConstantsPass()]).run(folded_first)
    hidden_first, mm_b = build()
    Pipeline([HideConstantsPass(), FoldConstantConvertsPass()]).run(hidden_first)

    assert folded_first.num_nodes() == 1 and mm_a.attributes == {"[1]": "f16[8, 8]"}
    # Hiding first only removes the Constant -> Convert edge; the Convert is no longer foldable.
    assert hidden_first.num_nodes() == 2 and mm_b.attributes == {}
    assert_graph_consistent(folded_first)
    assert_graph_consistent(hidden_first)


def test_empty_pipeline_leaves_graph_unchanged() -> None:
    g, _ = linear_graph()

    result = Pipeline([]).run(g)

    assert result is g
    assert g.num_nodes() == 6 and g.num_edges() == 5


def test_pipeline_records_timing_per_pass() -> None:
    g, _ = linear_graph()
    pipeline = Pipeline([HighlightPass(["mm"]), HideConstantsPass()])

    pipeline.run(g)

    assert [name for name, _ in pipeline.timings] == ["highlight", "hide-constants"]
    assert all(seconds >= 0 for _, seconds in pipeline.timings)


# --- registry and options --------------------------------------------------


def test_builtin_passes_are_registered() -> None:
    assert {"highlight", "hide-constants", "fold-converts"} <= set(available_passes())


def test_create_pass_with_options() -> None:
    p = create_pass("highlight", {"names": "a, b", "color": "blue"})

    assert isinstance(p, HighlightPass)
    assert p.names == {"a", "b"}
    assert p.color == "blue"


@pytest.mark.parametrize(
    ("name", "options", "message"),
    [
        ("no-such-pass", {}, "unknown pass"),
        ("highlight", {}, "needs names"),
        ("highlight", {"names": "a", "colour": "red"}, "unknown options"),
        ("hide-constants", {"verbose": "1"}, "takes no options"),
    ],
)
def test_create_pass_rejects_bad_configuration(name: str, options: dict[str, str], message: str) -> None:
    with pytest.raises(PassConfigError, match=message):
        create_pass(name, options)


def test_new_pass_can_be_registered_without_touching_existing_code(monkeypatch: pytest.MonkeyPatch) -> None:
    # Work on a copy of the registry so the test pass doesn't leak into other tests.
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))

    @register_pass
    class UppercaseNamesPass(TransformationPass):
        name = "test-uppercase-names"

        def apply(self, graph: Graph) -> Graph:
            for node in graph.nodes():
                node.name = node.name.upper()
            return graph

    g, n = linear_graph()
    Pipeline([create_pass("test-uppercase-names")]).run(g)

    assert n["mm"].name == "MM"


def test_registering_a_duplicate_name_fails() -> None:
    class Duplicate(TransformationPass):
        name = "highlight"

        def apply(self, graph: Graph) -> Graph:
            return graph

    with pytest.raises(ValueError, match="already registered"):
        register_pass(Duplicate)


def test_passes_package_does_not_import_frameworks() -> None:
    import subprocess
    import sys

    code = (
        "import sys, graph_viewer.passes; "
        "bad = {'openvino', 'graphviz'} & set(sys.modules); "
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
