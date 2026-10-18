"""Tests for OpenVINOLoader (graph_viewer.loaders.openvino_loader).

Expected API:
    OpenVINOLoader().load(path: str | Path) -> Graph
        raises ModelLoadError (graph_viewer.loaders.base) for missing,
        unsupported or invalid files.

Small models are built with the OpenVINO Python API and saved to a temporary
directory, so these tests do not need any downloaded model.
"""

from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import pytest

from conftest import save_model
from graph_checks import assert_graph_consistent
from graph_viewer.graph import Edge, Graph, Node
from graph_viewer.loaders.base import ModelLoadError, ModelLoader
from graph_viewer.loaders.openvino_loader import OpenVINOLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
TINYLLAMA_XML = REPO_ROOT / "ml_models" / "tinyllama_ov" / "openvino_model.xml"


# --- helpers ---------------------------------------------------------------


def nodes_named(graph: Graph, name: str) -> list[Node]:
    return [n for n in graph.nodes() if n.name == name]


def node_named(graph: Graph, name: str) -> Node:
    matches = nodes_named(graph, name)
    assert len(matches) == 1, f"expected one node named {name!r}, found {len(matches)}"
    return matches[0]


def edge_between(graph: Graph, src: Node, dst: Node) -> Edge:
    edges = [e for e in graph.out_edges(src.node_id) if e.dst == dst.node_id]
    assert len(edges) == 1, f"expected one edge {src.name} -> {dst.name}, found {len(edges)}"
    return edges[0]


# --- interface -------------------------------------------------------------


def test_loader_implements_model_loader_interface() -> None:
    assert isinstance(OpenVINOLoader(), ModelLoader)


# --- graph structure -------------------------------------------------------


def test_loads_all_operations_with_types_and_names(linear_model_path: Path) -> None:
    graph = OpenVINOLoader().load(linear_model_path)

    assert graph.num_nodes() == 6
    assert graph.num_edges() == 5
    types = {n.name: n.op_type for n in graph.nodes()}
    assert types["x"] == "Parameter"
    assert types["w"] == "Constant"
    assert types["mm"] == "MatMul"
    assert types["add"] == "Add"
    assert_graph_consistent(graph)


def test_model_inputs_and_outputs_are_nodes(linear_model_path: Path) -> None:
    graph = OpenVINOLoader().load(linear_model_path)

    op_types = [n.op_type for n in graph.nodes()]
    assert op_types.count("Parameter") == 1
    assert op_types.count("Result") == 1


def test_edges_keep_source_and_destination_ports(linear_model_path: Path) -> None:
    graph = OpenVINOLoader().load(linear_model_path)
    x, w, mm = node_named(graph, "x"), node_named(graph, "w"), node_named(graph, "mm")

    x_to_mm = edge_between(graph, x, mm)
    w_to_mm = edge_between(graph, w, mm)

    assert (x_to_mm.src_port, x_to_mm.dst_port) == (0, 0)
    assert (w_to_mm.src_port, w_to_mm.dst_port) == (0, 1)


def test_edges_carry_dtype_and_shape(linear_model_path: Path) -> None:
    graph = OpenVINOLoader().load(linear_model_path)
    x, w, mm, add = (node_named(graph, n) for n in ("x", "w", "mm", "add"))

    assert edge_between(graph, w, mm).dtype == "f32"
    assert edge_between(graph, w, mm).shape == (3, 4)
    assert edge_between(graph, x, mm).shape == (None, 4)    # dynamic batch
    assert edge_between(graph, mm, add).shape == (None, 3)  # inferred by OpenVINO


def test_operation_attributes_are_copied(linear_model_path: Path) -> None:
    graph = OpenVINOLoader().load(linear_model_path)

    attributes = node_named(graph, "mm").attributes
    assert attributes["transpose_a"] is False
    assert attributes["transpose_b"] is True


def test_graph_holds_no_openvino_objects_or_tensor_values(linear_model_path: Path) -> None:
    """Attributes must be plain Python data; Constant weights must not be copied."""
    graph = OpenVINOLoader().load(linear_model_path)

    def is_plain(value: object) -> bool:
        if isinstance(value, (list, tuple)):
            return all(is_plain(v) for v in value)
        if isinstance(value, dict):
            return all(isinstance(k, str) and is_plain(v) for k, v in value.items())
        return value is None or isinstance(value, (str, int, float, bool))

    for node in graph.nodes():
        assert is_plain(node.attributes), f"{node.name}: {node.attributes!r}"
    assert "value" not in node_named(graph, "w").attributes


def test_multiple_outputs_and_multiple_edges_between_same_nodes(tmp_path: Path) -> None:
    """Split out 0 -> Add in 0, Split out 1 -> Add in 1."""
    x = ops.parameter(ov.PartialShape([2, 6]), ov.Type.f16, name="x")
    split = ops.split(x, ops.constant(np.int64(1)), 2, name="split")
    add = ops.add(split.output(0), split.output(1), name="add")
    path = save_model(ov.Model([ops.result(add)], [x], "split"), tmp_path)

    graph = OpenVINOLoader().load(path)
    split_node, add_node = node_named(graph, "split"), node_named(graph, "add")

    edges = [e for e in graph.out_edges(split_node.node_id) if e.dst == add_node.node_id]
    assert sorted((e.src_port, e.dst_port) for e in edges) == [(0, 0), (1, 1)]
    assert all(e.dtype == "f16" and e.shape == (2, 3) for e in edges)
    assert_graph_consistent(graph)


def test_duplicate_node_names_become_separate_nodes(tmp_path: Path) -> None:
    a = ops.parameter(ov.PartialShape([2]), ov.Type.f32, name="dup")
    b = ops.parameter(ov.PartialShape([2]), ov.Type.f32, name="dup")
    add = ops.add(a, b, name="add")
    path = save_model(ov.Model([ops.result(add)], [a, b], "dup"), tmp_path)

    graph = OpenVINOLoader().load(path)

    dups = nodes_named(graph, "dup")
    assert len(dups) == 2
    assert dups[0].node_id != dups[1].node_id
    assert sorted(e.dst_port for e in graph.in_edges(node_named(graph, "add").node_id)) == [0, 1]


@pytest.mark.parametrize(
    ("partial_shape", "expected"),
    [
        (ov.PartialShape([-1, -1]), (None, None)),
        (ov.PartialShape([ov.Dimension(1, 10), 8]), (None, 8)),  # bounded range -> dynamic
        (ov.PartialShape.dynamic(), None),                       # unknown rank
        (ov.PartialShape([]), ()),                               # scalar
    ],
)
def test_dynamic_shapes(tmp_path: Path, partial_shape: ov.PartialShape, expected: tuple | None) -> None:
    x = ops.parameter(partial_shape, ov.Type.f32, name="x")
    relu = ops.relu(x, name="relu")
    path = save_model(ov.Model([ops.result(relu)], [x], "dyn"), tmp_path)

    graph = OpenVINOLoader().load(path)

    assert edge_between(graph, node_named(graph, "x"), node_named(graph, "relu")).shape == expected


def test_original_model_file_is_not_modified(linear_model_path: Path) -> None:
    xml_before = linear_model_path.read_bytes()
    bin_before = linear_model_path.with_suffix(".bin").read_bytes()

    OpenVINOLoader().load(linear_model_path)

    assert linear_model_path.read_bytes() == xml_before
    assert linear_model_path.with_suffix(".bin").read_bytes() == bin_before


def test_accepts_str_path(linear_model_path: Path) -> None:
    graph = OpenVINOLoader().load(str(linear_model_path))

    assert graph.num_nodes() == 6


# --- errors ----------------------------------------------------------------


def test_missing_file_raises_model_load_error(tmp_path: Path) -> None:
    missing = tmp_path / "nope.xml"

    with pytest.raises(ModelLoadError, match="nope.xml"):
        OpenVINOLoader().load(missing)


def test_unsupported_extension_raises_model_load_error(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"\x00")

    with pytest.raises(ModelLoadError, match=r"\.xml"):
        OpenVINOLoader().load(path)


def test_invalid_xml_raises_model_load_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.xml"
    path.write_text("this is not an OpenVINO IR file")

    with pytest.raises(ModelLoadError, match="broken.xml: not a valid OpenVINO IR model"):
        OpenVINOLoader().load(path)


def test_corrupted_ir_raises_model_load_error(linear_model_path: Path) -> None:
    """Looks like IR at the start, but the XML is cut off."""
    text = linear_model_path.read_text()
    linear_model_path.write_text(text[: len(text) // 2])

    with pytest.raises(ModelLoadError, match="not a valid OpenVINO IR model"):
        OpenVINOLoader().load(linear_model_path)


def test_missing_bin_file_raises_model_load_error(linear_model_path: Path) -> None:
    linear_model_path.with_suffix(".bin").unlink()

    with pytest.raises(ModelLoadError, match=r"\.bin"):
        OpenVINOLoader().load(linear_model_path)


# --- real model ------------------------------------------------------------


@pytest.mark.skipif(not TINYLLAMA_XML.exists(), reason="TinyLlama IR not exported (see README)")
def test_tinyllama_loads_every_operation() -> None:
    expected_ops = len(ov.Core().read_model(TINYLLAMA_XML).get_ordered_ops())

    graph = OpenVINOLoader().load(TINYLLAMA_XML)

    assert graph.num_nodes() == expected_ops
    assert_graph_consistent(graph)
