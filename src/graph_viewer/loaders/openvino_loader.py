"""Loader for OpenVINO IR models (.xml + .bin)."""

from pathlib import Path
from typing import Any

import openvino as ov

from graph_viewer.graph import Graph, Shape
from graph_viewer.loaders.base import ModelLoader, ModelLoadError


class OpenVINOLoader(ModelLoader):
    """Reads an OpenVINO IR model and copies its structure into a Graph.

    No OpenVINO object is stored in the returned Graph, and the model file is only read.
    Tensor values (Constant weights) are never copied, only their dtype and shape.
    """

    def load(self, path: str | Path) -> Graph:
        model = self._read_model(Path(path))
        return self._convert(model)

    def _read_model(self, path: Path) -> ov.Model:
        if path.suffix.lower() != ".xml":
            raise ModelLoadError(f"{path}: expected an OpenVINO IR .xml file, got '{path.suffix}'")
        if not path.is_file():
            raise ModelLoadError(f"{path}: file not found")
        # IR files start with '<?xml ...?><net ...>'. Checking this first gives a clear error
        # (OpenVINO would otherwise try other frontends, e.g. TensorFlow, and log noise).
        with path.open("rb") as file:
            if b"<net" not in file.read(1024):
                raise ModelLoadError(f"{path}: not a valid OpenVINO IR model (no <net> element)")

        try:
            return ov.Core().read_model(path)
        except RuntimeError as error:
            # OpenVINO's own messages are long C++ traces and don't name the missing file.
            bin_path = path.with_suffix(".bin")
            if not bin_path.is_file():
                raise ModelLoadError(f"{path}: weights file {bin_path.name} (.bin) not found") from error
            raise ModelLoadError(f"{path}: not a valid OpenVINO IR model") from error

    def _convert(self, model: ov.Model) -> Graph:
        graph = Graph()
        ops = model.get_ordered_ops()

        # OpenVINO node wrappers compare and hash by the underlying C++ node,
        # so they work as dict keys even when obtained through different calls.
        node_ids: dict[ov.Node, int] = {}
        for op in ops:
            node = graph.add_node(op.get_friendly_name(), op.get_type_name(), _to_plain(op.get_attributes()))
            node_ids[op] = node.node_id

        # Edges in a second loop: both endpoints must already exist in the graph.
        for op in ops:
            for target in op.inputs():
                source = target.get_source_output()
                graph.add_edge(
                    src=node_ids[source.get_node()],
                    src_port=source.get_index(),
                    dst=node_ids[op],
                    dst_port=target.get_index(),
                    dtype=_to_dtype(source.get_element_type()),
                    shape=_to_shape(source.get_partial_shape()),
                )
        return graph


def _to_dtype(element_type: ov.Type) -> str | None:
    if element_type.is_dynamic():
        return None
    return element_type.get_type_name()  # "f32", "f16", "i64", ...


def _to_shape(partial_shape: ov.PartialShape) -> Shape:
    if partial_shape.rank.is_dynamic:
        return None
    # Bounded dimensions such as 1..10 are not static either, so they become None too.
    return tuple(dim.get_length() if dim.is_static else None for dim in partial_shape)


def _to_plain(value: Any) -> Any:
    """Convert attribute values to plain Python data (no numpy or OpenVINO objects)."""
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):  # numpy arrays and scalars
        return _to_plain(value.tolist())
    return str(value)
