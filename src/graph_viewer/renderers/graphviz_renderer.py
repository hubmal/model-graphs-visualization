"""SVG renderer using Graphviz `dot` (hierarchical top-to-bottom layout)."""

import html
import subprocess
from pathlib import Path

import graphviz

from graph_viewer.graph import Edge, Graph, Node, format_tensor
from graph_viewer.renderers.base import Renderer, RenderError

# Fill colors for common operation types; any other type gets DEFAULT_FILL.
FILL_BY_OP_TYPE = {
    "Parameter": "#EDEADE",
    "Result": "#EDEADE",
    "Constant": "#FFD54F",
    "MatMul": "#90CAF9",
    "Convolution": "#90CAF9",
    "Add": "#B39DDB",
    "Multiply": "#B39DDB",
    "Softmax": "#A5D6A7",
    "Relu": "#A5D6A7",
    "Reshape": "#E0E0E0",
    "Transpose": "#E0E0E0",
    "Convert": "#E0E0E0",
}
DEFAULT_FILL = "#FFFFFF"

# Settings for large graphs (measurements in README.md, Performance):
# - nslimit/nslimit1/mclimit limit dot's network-simplex and crossing-minimization
#   iterations; without them dot didn't finish TinyLlama (1.5k-3k nodes) in 10 minutes.
# - splines=line draws straight edges; curved edges crashed dot 2.43 (SIGBUS) on TinyLlama.
FAST_LAYOUT_ATTRS = {"nslimit": "2", "nslimit1": "2", "mclimit": "0.3", "splines": "line"}


class GraphvizRenderer(Renderer):
    def __init__(
        self, show_attributes: bool = True, max_text_length: int = 48, fast_layout_threshold: int = 1000
    ) -> None:
        self.show_attributes = show_attributes
        # Graphs with more nodes than this get a cheaper (slightly less tidy) layout.
        self.fast_layout_threshold = fast_layout_threshold
        # Long names and attribute values are shortened in the label; full name is in the tooltip.
        self.max_text_length = max_text_length

    def render(self, graph: Graph, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        try:
            svg = self.to_digraph(graph).pipe(format="svg")
        except graphviz.ExecutableNotFound as error:
            raise RenderError("Graphviz 'dot' not found; install Graphviz (e.g. apt install graphviz)") from error
        except subprocess.CalledProcessError as error:
            # A negative return code means dot was killed by a signal (e.g. -7 = SIGBUS).
            details = (error.stderr or b"").decode(errors="replace").strip()
            raise RenderError(f"Graphviz 'dot' failed with exit code {error.returncode}: {details}") from error
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(svg)
        return output_path

    def to_dot(self, graph: Graph) -> str:
        """DOT source text; useful for tests and debugging without running Graphviz."""
        return self.to_digraph(graph).source

    def to_digraph(self, graph: Graph) -> graphviz.Digraph:
        dot = graphviz.Digraph()
        dot.attr(rankdir="TB", ranksep="0.3")
        if graph.num_nodes() > self.fast_layout_threshold:
            dot.attr(**FAST_LAYOUT_ATTRS)
        dot.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="10")
        dot.attr("edge", fontname="Helvetica", fontsize="8", arrowsize="0.6")

        for node in graph.nodes():
            border = node.viz.get("border_color")
            dot.node(
                f"n{node.node_id}",
                label=self._node_label(node),
                fillcolor=node.viz.get("fill_color", FILL_BY_OP_TYPE.get(node.op_type, DEFAULT_FILL)),
                color=border or "black",
                penwidth="3" if border else "1",
                tooltip=node.name,
                id=f"node{node.node_id}",
            )
        for edge in graph.edges():
            dot.edge(f"n{edge.src}", f"n{edge.dst}", label=self._edge_label(edge))
        return dot

    def _node_label(self, node: Node) -> str:
        # Graphviz HTML-like label: <...>. All text must be HTML-escaped.
        rows = [
            f"<B>{self._text(node.op_type)}</B>",
            self._text(node.name),
            f"id: {node.node_id}",
        ]
        if self.show_attributes:
            rows += [self._text(f"{key}: {value}") for key, value in node.attributes.items()]
        return "<" + "<BR/>".join(rows) + ">"

    def _edge_label(self, edge: Edge) -> str:
        return f"out {edge.src_port} → in {edge.dst_port}\n{format_tensor(edge.dtype, edge.shape)}"

    def _text(self, value: object) -> str:
        text = str(value)
        if len(text) > self.max_text_length:
            # Keep the end: OpenVINO names end with the most specific part (".../q_proj/MatMul").
            text = "…" + text[-(self.max_text_length - 1):]
        return html.escape(text)
