"""Tests for GraphvizRenderer (synthetic graphs; SVG tests need the `dot` binary)."""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from graph_viewer.graph import Graph
from graph_viewer.passes import HideConstantsPass, HighlightPass
from graph_viewer.renderers import GraphvizRenderer, Renderer, RenderError
from graph_viewer.renderers.graphviz_renderer import DEFAULT_FILL

needs_dot = pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz 'dot' not installed")
SVG_NS = "{http://www.w3.org/2000/svg}"


@pytest.fixture
def graph() -> Graph:
    """x -> MatMul(w) -> out, plus a Split feeding one Add twice."""
    g = Graph()
    x = g.add_node("x", "Parameter")
    w = g.add_node("w", "Constant")
    mm = g.add_node("mm", "MatMul", {"transpose_b": True})
    split = g.add_node("split", "Split")
    add = g.add_node("add", "Add")
    out = g.add_node("out", "Result")
    g.add_edge(x.node_id, 0, mm.node_id, 0, "f32", (None, 768))
    g.add_edge(w.node_id, 0, mm.node_id, 1, "f32", (1024, 768))
    g.add_edge(mm.node_id, 0, split.node_id, 0, "f32", (None, 1024))
    g.add_edge(split.node_id, 0, add.node_id, 0, "f32", (None, 512))
    g.add_edge(split.node_id, 1, add.node_id, 1, "f32", (None, 512))
    g.add_edge(add.node_id, 0, out.node_id, 0, "f32", (None, 512))
    return g


def node_ids_in_svg(svg_path: Path) -> set[str]:
    root = ET.parse(svg_path).getroot()
    return {g.get("id") for g in root.iter(f"{SVG_NS}g") if g.get("class") == "node"}


def test_renderer_implements_interface() -> None:
    assert isinstance(GraphvizRenderer(), Renderer)


# --- DOT source (no Graphviz binary needed) --------------------------------


def test_dot_contains_every_node_and_edge(graph: Graph) -> None:
    dot = GraphvizRenderer().to_dot(graph)

    for node in graph.nodes():
        assert f"n{node.node_id} [" in dot
    assert dot.count(" -> ") == graph.num_edges()


def test_dot_uses_top_to_bottom_layout(graph: Graph) -> None:
    assert "rankdir=TB" in GraphvizRenderer().to_dot(graph)


def test_node_label_shows_type_name_id_and_attributes(graph: Graph) -> None:
    dot = GraphvizRenderer().to_dot(graph)

    assert "<B>MatMul</B>" in dot
    assert "transpose_b: True" in dot
    assert "id: 2" in dot


def test_attributes_can_be_hidden(graph: Graph) -> None:
    dot = GraphvizRenderer(show_attributes=False).to_dot(graph)

    assert "transpose_b" not in dot


def test_edge_labels_show_ports_and_tensor_type(graph: Graph) -> None:
    dot = GraphvizRenderer().to_dot(graph)

    assert "out 0 → in 1" in dot      # Constant -> MatMul input 1
    assert "out 1 → in 1" in dot      # second Split output
    assert "f32[1024, 768]" in dot
    assert "f32[?, 768]" in dot       # dynamic dimension


def test_hidden_constants_appear_in_consumer_label(graph: Graph) -> None:
    HideConstantsPass().apply(graph)

    dot = GraphvizRenderer().to_dot(graph)

    assert "[1]: f32[1024, 768]" in dot
    assert "Constant" not in dot


def test_highlight_metadata_becomes_red_border(graph: Graph) -> None:
    HighlightPass(["mm"]).apply(graph)

    dot = GraphvizRenderer().to_dot(graph)

    mm_line = next(line for line in dot.splitlines() if line.strip().startswith("n2 ["))
    assert "color=red" in mm_line and "penwidth=3" in mm_line
    x_line = next(line for line in dot.splitlines() if line.strip().startswith("n0 ["))
    assert "color=black" in x_line


def test_unknown_op_type_gets_default_style() -> None:
    g = Graph()
    g.add_node("custom", "MyBrandNewOp")

    dot = GraphvizRenderer().to_dot(g)

    assert "MyBrandNewOp" in dot
    assert f'fillcolor="{DEFAULT_FILL}"' in dot


def test_special_characters_are_escaped() -> None:
    g = Graph()
    g.add_node("a<b>&c", "Weird<Op>", {"expr": "x < y"})

    dot = GraphvizRenderer().to_dot(g)

    assert "a&lt;b&gt;&amp;c" in dot
    assert "Weird&lt;Op&gt;" in dot
    assert "x &lt; y" in dot


def test_long_names_are_shortened_but_kept_in_tooltip() -> None:
    long_name = "__module.model.layers.0.self_attn.q_proj/ov_ext::linear/MatMul"
    g = Graph()
    g.add_node(long_name, "MatMul")

    dot = GraphvizRenderer(max_text_length=20).to_dot(g)

    assert "…_ext::linear/MatMul" in dot  # 20 characters including the ellipsis
    assert dot.count(long_name) == 1      # the full name appears only in the tooltip
    assert f'tooltip="{long_name}"' in dot


# --- SVG output ------------------------------------------------------------


@needs_dot
def test_render_writes_valid_svg(graph: Graph, tmp_path: Path) -> None:
    path = GraphvizRenderer().render(graph, tmp_path / "out" / "graph.svg")

    assert path.exists()
    root = ET.parse(path).getroot()  # raises if the XML is invalid
    assert root.tag == f"{SVG_NS}svg"
    assert node_ids_in_svg(path) == {f"node{n.node_id}" for n in graph.nodes()}


@needs_dot
def test_highlight_is_visible_in_svg(graph: Graph, tmp_path: Path) -> None:
    HighlightPass(["mm"]).apply(graph)

    path = GraphvizRenderer().render(graph, tmp_path / "graph.svg")

    root = ET.parse(path).getroot()
    mm_group = next(g for g in root.iter(f"{SVG_NS}g") if g.get("id") == "node2")
    assert any(shape.get("stroke") == "red" for shape in mm_group.iter() if shape.get("stroke"))


def test_missing_dot_binary_raises_render_error(
    graph: Graph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))  # no `dot` here

    with pytest.raises(RenderError, match="Graphviz"):
        GraphvizRenderer().render(graph, tmp_path / "graph.svg")
